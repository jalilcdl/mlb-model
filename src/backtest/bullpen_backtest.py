"""
Does splitting the bullpen out of team run-prevention actually improve
predictions? Walk-forward, no lookahead, paired against the current model.

THE HYPOTHESIS BEING TESTED
The model projects a team's run prevention as:

    eff_def = starter_share * starter_factor + (1 - starter_share) * team_def

That second term is the problem. It uses the team's OVERALL run-prevention rate
to stand in for the innings the bullpen will actually throw -- but a team's
overall rate is dominated by its rotation. A club with a good rotation and a bad
pen gets credited with a good bullpen, and vice versa. Measured league-wide, the
rotation/bullpen FIP gap spans 2.13 runs, so this is not a rounding error.

The test swaps that term for the bullpen's own measured rate:

    eff_def = starter_share * starter_factor + (1 - starter_share) * bullpen_rate

and asks whether it predicts better. Nothing ships unless it does.

DATA, and why it is derived rather than fetched
Per-reliever game logs are not available in bulk (the sports-data MCP caps
queries at 100 rows, which cannot cover ~24k pitcher-games a season). So bullpen
performance is derived from data already on disk:

    bullpen_ER = team runs allowed - starting pitcher's earned runs
    bullpen_IP = 9 - starting pitcher's innings

with the starter's line coming from pitcher_game_logs.csv (98.4% of starts
matched) and the team's runs allowed from games.csv.

Two approximations are baked in and neither is hidden:
  1. Team runs allowed includes UNEARNED runs; the starter's line is EARNED
     runs. The difference lands in the bullpen bucket, slightly overstating
     bullpen ER. It applies identically to both model variants, so it cannot
     manufacture a win for the test arm.
  2. Defensive innings are assumed to be 9. A home team that wins pitches 8, and
     extra-inning games run longer. Again symmetric across variants.

Both would need fixing before these bullpen rates were used as a published
statistic. For a paired A/B on the same games they are acceptable.

    python -m src.backtest.bullpen_backtest
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import skellam

from src import config
from src.models.run_model import TeamRunRatings
from src.backtest.pitcher_backtest import _as_of_date_stats, _build_pitcher_index, _pair_doubleheaders

SHRINK_INNINGS = 120.0  # bullpen IP before its own rate is trusted over league average


def build_bullpen_history(games, starters, logs):
    """One row per team-game: bullpen ER and IP, derived as described above."""
    logs = logs.copy()
    logs["key"] = logs["pitcher_id"].astype("Int64").astype(str) + "_" + logs["date"].dt.strftime("%Y-%m-%d")
    line = logs.set_index("key")[["ip", "er"]]

    paired = _pair_doubleheaders(games, starters)
    rows = []
    for r in paired.itertuples(index=False):
        d = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        for side, team, runs_allowed in (
            ("home", r.home_team, r.away_score),   # home pitching allows away runs
            ("away", r.away_team, r.home_score),
        ):
            pid = getattr(r, f"{side}_pitcher_id")
            if pd.isna(pid):
                continue
            k = f"{int(pid)}_{d}"
            if k not in line.index:
                continue
            row = line.loc[k]
            s_ip = float(row["ip"].iloc[0] if isinstance(row["ip"], pd.Series) else row["ip"])
            s_er = float(row["er"].iloc[0] if isinstance(row["er"], pd.Series) else row["er"])
            rows.append({
                "date": r.date, "team": team,
                "bp_ip": max(0.0, 9.0 - s_ip),
                "bp_er": max(0.0, float(runs_allowed) - s_er),
            })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def prior_bullpen_rates(bp):
    """As-of-date bullpen ERA per team using ONLY prior games, plus the running
    league bullpen ERA to normalize against."""
    bp = bp.sort_values("date").copy()
    g = bp.groupby("team", sort=False)
    bp["cum_er"] = g["bp_er"].transform(lambda x: x.shift(1).expanding().sum())
    bp["cum_ip"] = g["bp_ip"].transform(lambda x: x.shift(1).expanding().sum())
    bp["lg_er"] = bp["bp_er"].shift(1).expanding().sum()
    bp["lg_ip"] = bp["bp_ip"].shift(1).expanding().sum()
    return bp


def run_validation(eval_seasons=(2025,), starter_share=None):
    starter_share = starter_share or config.STARTER_INNINGS_SHARE
    games = pd.read_csv(config.GAMES_FILE, parse_dates=["date"])
    games = games[games["home_score"] != games["away_score"]]
    starters = pd.read_csv(config.HISTORICAL_STARTERS_FILE, parse_dates=["date"]).dropna(
        subset=["home_pitcher_id", "away_pitcher_id"])
    logs = pd.read_csv(config.PITCHER_GAME_LOGS_FILE, parse_dates=["date"])
    pindex = _build_pitcher_index(logs)

    bp = prior_bullpen_rates(build_bullpen_history(games, starters, logs))
    bp_key = bp.set_index(["team", "date"])

    paired = _pair_doubleheaders(games, starters)
    paired = paired[paired["season"].isin(eval_seasons)].sort_values("date")

    rows = []
    for date, day in paired.groupby("date"):
        train = games[games["date"] < date]
        if train.empty:
            continue
        try:
            ratings = TeamRunRatings().fit(train)
        except ValueError:
            continue

        for r in day.itertuples(index=False):
            season = int(r.season)
            facs, bpens = {}, {}
            ok = True
            for side, team in (("home", r.home_team), ("away", r.away_team)):
                stats = _as_of_date_stats(pindex, getattr(r, f"{side}_pitcher_id"), season, date)
                facs[side] = ratings.pitcher_factor(stats, ratings.league_avg_era_proxy)
                try:
                    b = bp_key.loc[(team, date)]
                except KeyError:
                    ok = False
                    break
                if isinstance(b, pd.DataFrame):
                    b = b.iloc[0]
                ip, er = b["cum_ip"], b["cum_er"]
                lg_ip, lg_er = b["lg_ip"], b["lg_er"]
                if not (np.isfinite(ip) and ip > 20 and np.isfinite(lg_ip) and lg_ip > 0):
                    ok = False
                    break
                team_bp_era = 9.0 * er / ip
                lg_bp_era = 9.0 * lg_er / lg_ip
                raw = team_bp_era / lg_bp_era if lg_bp_era > 0 else 1.0
                w = ip / (ip + SHRINK_INNINGS)          # shrink thin samples to league average
                bpens[side] = w * raw + (1 - w) * 1.0
            if not ok or facs["home"] is None or facs["away"] is None:
                continue

            lg = ratings.league_avg_runs
            park = ratings.park_factor.get(r.home_team, 1.0)
            off_h = ratings.off_rating.get(r.home_team, 1.0)
            off_a = ratings.off_rating.get(r.away_team, 1.0)
            def_h = ratings.def_rating.get(r.home_team, 1.0)
            def_a = ratings.def_rating.get(r.away_team, 1.0)

            def probs(bp_h, bp_a):
                eff_h = starter_share * facs["home"] + (1 - starter_share) * bp_h
                eff_a = starter_share * facs["away"] + (1 - starter_share) * bp_a
                mu_h = max(lg * off_h * eff_a * park, 0.3)
                mu_a = max(lg * off_a * eff_h * park, 0.3)
                tie = skellam.pmf(0, mu_h, mu_a)
                return (1 - skellam.cdf(0, mu_h, mu_a)) + config.EXTRA_INNING_HOME_WIN_PROB * tie

            rows.append({
                "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
                "home_win": int(r.home_score > r.away_score),
                # BASELINE: team-wide run prevention fills the non-starter innings
                "p_baseline": probs(def_h, def_a),
                # TEST: the bullpen's own measured rate fills them
                "p_bullpen": probs(bpens["home"], bpens["away"]),
                "bp_home": bpens["home"], "bp_away": bpens["away"],
                "def_home": def_h, "def_away": def_a,
            })
    return pd.DataFrame(rows)


def summarize(df):
    y = df["home_win"].values
    out = {"n_games": int(len(df))}
    for name, col in (("baseline", "p_baseline"), ("bullpen_split", "p_bullpen")):
        p = np.clip(df[col].values, 1e-9, 1 - 1e-9)
        out[name] = {
            "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        }
    d = ((df["p_baseline"].values - y) ** 2) - ((df["p_bullpen"].values - y) ** 2)
    rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    out["brier_improvement"] = {
        "mean": float(d.mean()), "ci95": [float(lo), float(hi)],
        "significant": bool(lo > 0 or hi < 0),
    }
    out["how_different"] = {
        "mean_abs_prob_shift_pts": float((df["p_bullpen"] - df["p_baseline"]).abs().mean() * 100),
        "max_abs_prob_shift_pts": float((df["p_bullpen"] - df["p_baseline"]).abs().max() * 100),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Validate the rotation/bullpen split.")
    ap.add_argument("--seasons", default="2025")
    args = ap.parse_args()
    seasons = tuple(int(s) for s in args.seasons.split(","))
    df = run_validation(seasons)
    if df.empty:
        print("No evaluable games.")
        return
    s = summarize(df)
    df.to_csv(config.PROCESSED_DIR / "bullpen_backtest_results.csv", index=False)
    with open(config.PROCESSED_DIR / "bullpen_backtest_summary.json", "w") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
