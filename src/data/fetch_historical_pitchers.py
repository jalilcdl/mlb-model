"""
Historical starting pitchers + per-pitcher game logs, for isolating the
starting-pitcher adjustment factor in a proper walk-forward backtest
(src/backtest/pitcher_backtest.py).

Two datasets, both pulled from the free MLB Stats API:

  1. Probable starters for every historical game in the eval window
     (data/processed/historical_starters.csv) -- pulled from the same
     schedule endpoint live predictions use for today's games, so this is
     genuinely the pitcher information that would have been available
     before each game (not a box-score reconstruction after the fact).

  2. Per-pitcher, per-game pitching lines, date-stamped
     (data/processed/pitcher_game_logs.csv) -- lets the backtest compute
     each pitcher's stats *as of* any cutoff date by summing only games
     strictly before it, with no lookahead.

Slower than the team-level game log pull (one schedule call per date, one
game-log call per unique pitcher-season), so kept to a smaller, separately
configurable window (config.PITCHER_BACKTEST_SEASONS) rather than the full
HISTORICAL_SEASONS range.
"""
import time

import pandas as pd

from src import config
from src.data import statsapi_client


def fetch_historical_starters(dates, pause=0.3, verbose=True):
    """dates: iterable of 'YYYY-MM-DD' strings with games. Returns a
    DataFrame: date, home_team, away_team, home_pitcher_id, home_pitcher_name,
    away_pitcher_id, away_pitcher_name (regular-season games only)."""
    rows = []
    dates = list(dates)
    for i, date_str in enumerate(dates):
        try:
            games = statsapi_client.get_schedule(date_str)
        except Exception as exc:
            if verbose:
                print(f"  [warn] {date_str}: {exc}")
            time.sleep(pause)
            continue
        for g in games:
            if g.get("game_type") != "R":
                continue
            rows.append(
                {
                    "date": date_str,
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "home_pitcher_id": g["home_probable_pitcher_id"],
                    "home_pitcher_name": g["home_probable_pitcher_name"],
                    "away_pitcher_id": g["away_probable_pitcher_id"],
                    "away_pitcher_name": g["away_probable_pitcher_name"],
                }
            )
        if verbose and i % 20 == 0:
            print(f"  [{i + 1}/{len(dates)}] {date_str}: {len(games)} games")
        time.sleep(pause)
    return pd.DataFrame(rows)


def fetch_pitcher_game_logs(pitcher_season_pairs, pause=0.3, verbose=True):
    """pitcher_season_pairs: iterable of (pitcher_id, season). Returns a
    DataFrame: pitcher_id, season, date, ip, er, hr, bb, hbp, so (one row
    per pitcher-game)."""
    pairs = list(pitcher_season_pairs)
    frames = []
    for i, (pid, season) in enumerate(pairs):
        try:
            log = statsapi_client.get_pitcher_game_log(pid, season)
        except Exception as exc:
            if verbose:
                print(f"  [warn] pitcher {pid} {season}: {exc}")
            time.sleep(pause)
            continue
        if log:
            df = pd.DataFrame(log)
            df["pitcher_id"] = pid
            df["season"] = season
            frames.append(df)
        if verbose and i % 25 == 0:
            print(f"  [{i + 1}/{len(pairs)}] pitcher {pid} ({season}): {len(log)} games logged")
        time.sleep(pause)
    if not frames:
        return pd.DataFrame(columns=["pitcher_id", "season", "date", "ip", "er", "hr", "bb", "hbp", "so"])
    return pd.concat(frames, ignore_index=True)


def build_and_save(seasons=None, games=None):
    seasons = seasons or config.PITCHER_BACKTEST_SEASONS
    games = games if games is not None else pd.read_csv(config.GAMES_FILE, parse_dates=["date"])
    games = games[games["season"].isin(seasons)]
    dates = sorted(games["date"].dt.strftime("%Y-%m-%d").unique())

    print(f"Fetching probable starters for {len(dates)} dates across seasons {seasons}...")
    starters = fetch_historical_starters(dates)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    starters.to_csv(config.HISTORICAL_STARTERS_FILE, index=False)
    print(f"Saved {len(starters)} game-starter rows to {config.HISTORICAL_STARTERS_FILE}")

    clean = starters.dropna(subset=["home_pitcher_id", "away_pitcher_id"])
    pairs = set()
    for _, row in clean.iterrows():
        season = int(row["date"][:4])
        pairs.add((int(row["home_pitcher_id"]), season))
        pairs.add((int(row["away_pitcher_id"]), season))
    pairs = sorted(pairs)

    print(f"Fetching game logs for {len(pairs)} unique pitcher-seasons...")
    logs = fetch_pitcher_game_logs(pairs)
    logs.to_csv(config.PITCHER_GAME_LOGS_FILE, index=False)
    print(f"Saved {len(logs)} pitcher-game rows to {config.PITCHER_GAME_LOGS_FILE}")

    return starters, logs


if __name__ == "__main__":
    build_and_save()
