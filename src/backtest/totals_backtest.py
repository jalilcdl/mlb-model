"""
Grade the totals (over/under) model against REAL historical market lines --
the validation we could not do in the main backtest because no historical
closing totals ship with pybaseball or the MLB Stats API (see README).

This is the totals analogue of the moneyline backtest: walk-forward, no
lookahead, compared to naive baselines, with calibration and a paired
bootstrap significance test. The one thing it needs that the rest of the
project generates for free is the market line for each past game -- you must
supply that.

REQUIRED INPUT  (config.HISTORICAL_TOTALS_FILE, a CSV):

    date,away_team,home_team,total_line,over_odds,under_odds
    2024-04-01,NYM,MIL,8.5,-110,-110
    ...

  - date: YYYY-MM-DD (the game date).
  - away_team/home_team: team codes; common aliases are normalized (see
    odds_adapter._normalize_team). Use the *closing* line for a fair test.
  - total_line: the market's over/under number (e.g. 8.5, 9.0).
  - over_odds/under_odds: American odds. OPTIONAL -- if omitted, ROI is
    computed at an assumed -110/-110 and flagged as approximate; the
    pick-accuracy / calibration / significance results don't need them.

Where to get the file (all real, none fabricated -- do NOT let the model
invent lines):
  - Sportsbook Reviews Online season spreadsheets (free; ML/RL/totals).
  - Kaggle MLB odds datasets (free; often SBR mirrors).
  - The Odds API historical endpoint (paid).
  - Your own live-odds connector, if it exposes historical/closing lines.

If the file is absent this module does nothing but tell you so -- it will
never fabricate lines to produce a number.

WHAT IT MEASURES:
  - ATS record: how often the model's over/under pick beats the actual result
    (pushes excluded), vs. always-under and always-over baselines.
  - ROI at flat stakes, using real closing prices when supplied.
  - Calibration: model P(over) vs actual over-frequency, plus ECE.
  - Paired bootstrap CI on (model correct - baseline correct) per game -- the
    same "is this distinguishable from noise" test used for the pitcher
    adjustment.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from src import config
from src.models import monte_carlo
from src.models.run_model import TeamRunRatings
from src.odds.odds_adapter import american_to_decimal, _normalize_team


def load_market_totals(path=None):
    path = path or config.HISTORICAL_TOTALS_FILE
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_team"] = df["home_team"].map(_normalize_team)
    df["away_team"] = df["away_team"].map(_normalize_team)
    df = df.dropna(subset=["home_team", "away_team", "total_line"])
    for c in ("over_odds", "under_odds"):
        if c not in df.columns:
            df[c] = -110  # assumed standard juice; flagged in the summary
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-110)
    return df


def _pair_with_results(lines, games):
    """Join market lines to actual results on date + matchup, pairing
    doubleheaders in listed order (same approach as the pitcher backtest)."""
    key = ["date", "home_team", "away_team"]
    g = games.copy()
    l = lines.copy()
    g["_d"] = g.groupby(key).cumcount()
    l["_d"] = l.groupby(key).cumcount()
    return pd.merge(g, l, on=key + ["_d"], how="inner")


def run_totals_backtest(lines=None, games=None, warmup_seasons=None, n_sims=None):
    n_sims = n_sims or config.BACKTEST_MC_SIMS
    lines = load_market_totals() if lines is None else lines
    if lines is None or lines.empty:
        return None
    games = games if games is not None else pd.read_csv(config.GAMES_FILE, parse_dates=["date"])

    paired = _pair_with_results(lines, games).sort_values("date").reset_index(drop=True)
    if paired.empty:
        return pd.DataFrame()

    all_seasons = sorted(games["season"].unique())
    warmup = set(warmup_seasons or all_seasons[:1])

    rows = []
    for date, day in paired.groupby("date"):
        train = games[games["date"] < date]
        if train.empty or int(str(date)[:4]) in warmup:
            continue
        try:
            ratings = TeamRunRatings().fit(train)
        except ValueError:
            continue
        for r in day.itertuples(index=False):
            mu_h, mu_a = ratings.predict_mus(r.home_team, r.away_team)
            sim = monte_carlo.simulate_game(mu_h, mu_a, n_sims=n_sims, overdispersion=ratings.overdispersion)
            over_p, push_p, under_p = monte_carlo.total_outcome_probs(sim["home_runs"], sim["away_runs"], r.total_line)
            actual = r.home_score + r.away_score
            outcome = "over" if actual > r.total_line else ("under" if actual < r.total_line else "push")
            no_push = over_p + under_p
            over_cond = over_p / no_push if no_push else 0.5
            rows.append({
                "date": r.date, "away_team": r.away_team, "home_team": r.home_team,
                "total_line": r.total_line, "actual_total": actual, "outcome": outcome,
                "model_expected_total": sim["expected_total"],
                "model_p_over": over_p, "model_p_under": under_p, "model_p_over_cond": over_cond,
                "model_pick": "over" if over_cond > 0.5 else "under",
                "over_odds": r.over_odds, "under_odds": r.under_odds,
            })
    return pd.DataFrame(rows)


def _bootstrap_ci(diffs, n_boot=2000, seed=0):
    diffs = np.asarray(diffs, float)
    if len(diffs) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    means = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(n_boot)]
    return float(diffs.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _roi(picks, outcomes, over_odds, under_odds, stake=100.0):
    profit = turnover = 0.0
    for pick, out, oo, uo in zip(picks, outcomes, over_odds, under_odds):
        if out == "push":
            continue
        turnover += stake
        dec = american_to_decimal(oo if pick == "over" else uo)
        profit += stake * (dec - 1.0) if pick == out else -stake
    return (profit / turnover) if turnover else np.nan, profit, turnover


def summarize(df):
    decided = df[df["outcome"] != "push"].copy()
    n_all, n_dec, n_push = len(df), len(decided), int((df["outcome"] == "push").sum())
    if n_dec == 0:
        return {"n_games": n_all, "note": "no decided games (all pushes / empty)"}

    model_correct = (decided["model_pick"] == decided["outcome"]).astype(int)
    under_correct = (decided["outcome"] == "under").astype(int)   # always-under baseline
    over_correct = (decided["outcome"] == "over").astype(int)     # always-over baseline

    # Calibration of model P(over) (conditional, no-push) vs actual over-rate.
    p = decided["model_p_over_cond"].values
    y = (decided["outcome"] == "over").astype(int).values
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    cal, ece = [], 0.0
    for b in range(10):
        m = idx == b
        if not m.any():
            continue
        mp, ar = float(p[m].mean()), float(y[m].mean())
        cal.append({"bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}", "n": int(m.sum()),
                    "mean_p_over": mp, "actual_over_rate": ar})
        ece += (m.sum() / n_dec) * abs(mp - ar)

    md, lo, hi = _bootstrap_ci(model_correct.values - under_correct.values)
    roi, profit, turnover = _roi(df["model_pick"], df["outcome"], df["over_odds"], df["under_odds"])
    assumed = bool((df["over_odds"] == -110).all() and (df["under_odds"] == -110).all())

    return {
        "n_games": n_all, "n_decided": n_dec, "n_push": n_push,
        "model_ats_win_rate": float(model_correct.mean()),
        "baseline_always_under_win_rate": float(under_correct.mean()),
        "baseline_always_over_win_rate": float(over_correct.mean()),
        "model_vs_under_edge_mean": md,
        "model_vs_under_edge_95ci": [lo, hi],
        "model_beats_under_significantly": bool(lo is not None and (lo > 0 or hi < 0)),
        "mean_model_total": float(df["model_expected_total"].mean()),
        "mean_market_line": float(df["total_line"].mean()),
        "mean_actual_total": float(df["actual_total"].mean()),
        "over_calibration": cal,
        "over_ece": float(ece),
        "roi_flat_100": None if pd.isna(roi) else float(roi),
        "roi_profit": float(profit), "roi_turnover": float(turnover),
        "roi_prices_assumed_-110": assumed,
    }


def run_and_save():
    df = run_totals_backtest()
    if df is None:
        msg = (
            "No historical market totals found at "
            f"{config.HISTORICAL_TOTALS_FILE}.\n"
            "This backtest needs REAL closing over/under lines (see this module's "
            "docstring for the CSV format and free/paid sources). It will not run on "
            "fabricated lines. Drop the file in and re-run."
        )
        print(msg)
        return None, None
    if df.empty:
        print("Market totals file loaded, but none matched games in games.csv (check dates/team codes).")
        return df, None
    summary = summarize(df)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.TOTALS_BACKTEST_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    return df, summary


def main():
    ap = argparse.ArgumentParser(description="Backtest the totals model vs real historical market lines.")
    ap.add_argument("lines_csv", nargs="?", help="path to historical totals CSV (overrides config default)")
    args = ap.parse_args()
    if args.lines_csv:
        config.HISTORICAL_TOTALS_FILE = __import__("pathlib").Path(args.lines_csv)
    df, summary = run_and_save()
    if summary:
        print(json.dumps(summary, indent=2))
    elif df is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
