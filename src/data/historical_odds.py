"""
Real historical MLB closing odds loader -- gives the totals backtest actual
market lines to grade against (the gap flagged in the README).

Source: the free, public `pwu97/bettingtools` dataset on GitHub (R `.rda`
files, MLB 2014-2019), which originates from Sportsbook Reviews Online. Each
row is one game with closing over/under line + odds, closing moneylines, the
run line, AND the final score -- so it doubles as both the odds source and the
game log the model fits on. No scraping, no API key, no paid tier.

WHY THIS ERA (an honest constraint, not a choice):
  Free historical-odds archives (SBR mirrors, this dataset, most Kaggle sets)
  cover roughly 2010-2021. Our live games.csv is 2023-2026. They do NOT
  overlap, so real odds can't be joined to our current games. The fix is to
  run the totals backtest entirely within the odds' own era: fit the model on
  2014-2019 games (reconstructed from these same files) and grade its totals
  picks against these real 2014-2019 closing lines. That makes it a genuine
  out-of-sample test on a period never touched during development -- arguably
  stronger than testing on the seasons the model was built around.

PRICING CAVEAT (surfaced, not hidden):
  The source carries a single `close_ou_odds` per game without a reliable
  over/under side label, so we do NOT trust it for exact ROI: total_line is
  real and is what the pick-accuracy / calibration / significance results
  depend on, but over/under prices are written as -110/-110 (flagged
  `roi_prices_assumed_-110` in the backtest) with the raw value preserved in a
  column for anyone who wants to dig. The line is real; the juice is nominal.
"""
import argparse
import tempfile

import pandas as pd

from src import config
from src.data import team_mapping

BASE = "https://raw.githubusercontent.com/pwu97/bettingtools/master/data/mlb_odds_{year}.rda"
YEARS = range(2014, 2020)
# Dedicated files for this one-time bulk historical set -- kept separate from
# config.HISTORICAL_TOTALS_FILE, which the live accumulator grows going forward.
HIST_GAMES_FILE = config.DATA_DIR / "raw" / "hist_odds_games.csv"
HIST_TOTALS_FILE = config.DATA_DIR / "raw" / "historical_totals_2014_2019.csv"


def _fetch_year(year):
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError(
            "pyreadr is required to read the .rda odds files ('pip install pyreadr')."
        ) from exc
    import requests

    r = requests.get(BASE.format(year=year), timeout=30)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".rda", delete=False) as f:
        f.write(r.content)
        path = f.name
    try:
        df = list(pyreadr.read_r(path).values())[0]
    finally:
        import os
        os.unlink(path)
    df["season"] = year
    return df


def build(years=YEARS, verbose=True):
    """Download the odds files and return (games_df, totals_df) with canonical
    team codes. games_df matches games.csv's shape; totals_df matches the
    format src/backtest/totals_backtest.py consumes."""
    raw = pd.concat([_fetch_year(y) for y in years], ignore_index=True)

    # Resolve teams by full NAME (robust) rather than the source's inconsistent
    # abbreviations (CUB/KAN/SDG/LOS/HOW...). code_from_name handles full names
    # and city/club keywords.
    raw["home_code"] = raw.apply(lambda r: team_mapping.code_from_name(r["home_name"], r["home_abbrev"]), axis=1)
    raw["away_code"] = raw.apply(lambda r: team_mapping.code_from_name(r["away_name"], r["away_abbrev"]), axis=1)

    unresolved = raw[raw["home_code"].isna() | raw["away_code"].isna()]
    if verbose and not unresolved.empty:
        names = pd.unique(unresolved[["home_name", "away_name"]].values.ravel())
        print(f"[!] {len(unresolved)} rows with unresolved team names (dropped): {list(names)[:10]}")
    raw = raw.dropna(subset=["home_code", "away_code", "home_score", "away_score", "close_ou_line"]).copy()
    raw["date"] = pd.to_datetime(raw["date"])

    games = raw[["date", "season", "home_code", "away_code", "home_score", "away_score"]].rename(
        columns={"home_code": "home_team", "away_code": "away_team"}
    ).reset_index(drop=True)

    totals = pd.DataFrame({
        "date": raw["date"].dt.strftime("%Y-%m-%d"),
        "away_team": raw["away_code"],
        "home_team": raw["home_code"],
        "total_line": raw["close_ou_line"],
        "over_odds": -110,   # nominal -- see PRICING CAVEAT in module docstring
        "under_odds": -110,
        "close_ou_odds_raw": raw["close_ou_odds"],  # preserved for transparency
        "source": "pwu97/bettingtools (SBR, closing)",
    }).reset_index(drop=True)

    return games, totals


def save(years=YEARS):
    games, totals = build(years)
    HIST_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(HIST_GAMES_FILE, index=False)
    totals.to_csv(HIST_TOTALS_FILE, index=False)
    print(f"Saved {len(games)} games -> {HIST_GAMES_FILE}")
    print(f"Saved {len(totals)} closing totals -> {HIST_TOTALS_FILE}")
    print(f"Seasons: {sorted(games['season'].unique())}")
    return games, totals


def run_backtest(n_sims=5000, warmup_seasons=(2014,)):
    """Grade the totals model against these real closing lines, walk-forward.
    Fits the model on games from this same era (strictly prior to each date)
    and evaluates its over/under picks vs the actual closing number."""
    import json
    from src.backtest import totals_backtest as tb

    if not HIST_TOTALS_FILE.exists() or not HIST_GAMES_FILE.exists():
        save()
    games = pd.read_csv(HIST_GAMES_FILE, parse_dates=["date"])
    lines = tb.load_market_totals(HIST_TOTALS_FILE)
    df = tb.run_totals_backtest(lines=lines, games=games, warmup_seasons=list(warmup_seasons), n_sims=n_sims)
    summary = tb.summarize(df)
    df.to_csv(config.PROCESSED_DIR / "totals_backtest_results.csv", index=False)
    with open(config.TOTALS_BACKTEST_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    return df, summary


def main():
    ap = argparse.ArgumentParser(description="Download real historical MLB closing odds (2014-2019, free) and/or backtest totals against them.")
    ap.add_argument("--start", type=int, default=2014)
    ap.add_argument("--end", type=int, default=2019)
    ap.add_argument("--run-backtest", action="store_true", help="after saving, grade the totals model against the real closing lines")
    ap.add_argument("--sims", type=int, default=5000)
    args = ap.parse_args()
    save(range(args.start, args.end + 1))
    if args.run_backtest:
        import json
        _, summary = run_backtest(n_sims=args.sims)
        print(json.dumps({k: v for k, v in summary.items() if k != "over_calibration"}, indent=2))


if __name__ == "__main__":
    main()
