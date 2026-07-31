"""
Pull historical game-by-game results from Baseball-Reference (via pybaseball).

This is the slow, network-heavy part of the pipeline (one scrape per team per
season, rate-limited to be polite to Baseball-Reference). Run it via
`python -m src.data.fetch_historical` (or scripts/build_dataset.py) to
(re)populate data/processed/games.csv. Everything downstream reads that cached
CSV, so day-to-day predictions don't need to re-scrape.
"""
import re
import time

import pandas as pd
import pybaseball as pb
import requests

from src import config
from src.data import team_mapping

pb.cache.enable()

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _patch_requests_user_agent():
    """FanGraphs (used as a pitching-stats fallback) rejects the default
    python-requests User-Agent with a 403. Baseball-Reference doesn't care
    either way, so it's safe to patch globally."""
    if getattr(requests.get, "_mlb_model_patched", False):
        return
    _orig_get = requests.get

    def _patched_get(*args, **kwargs):
        headers = kwargs.pop("headers", {}) or {}
        headers.setdefault("User-Agent", _UA)
        kwargs["headers"] = headers
        return _orig_get(*args, **kwargs)

    _patched_get._mlb_model_patched = True
    requests.get = _patched_get


_patch_requests_user_agent()

_DATE_SUFFIX_RE = re.compile(r"\s*\((\d)\)\s*$")


def _parse_date_and_game_num(date_str, season):
    m = _DATE_SUFFIX_RE.search(date_str)
    game_num = int(m.group(1)) if m else 1
    cleaned = _DATE_SUFFIX_RE.sub("", date_str).strip()
    date = pd.to_datetime(f"{cleaned} {season}", format="%A, %b %d %Y")
    return date, game_num


def fetch_team_season(code, season):
    """One team's home-game rows for a season, standardized to canonical team codes."""
    bref_code = team_mapping.bref_abbr(code, season)
    df = pb.schedule_and_record(season, bref_code)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[df["Home_Away"] == "Home"].copy()
    if df.empty:
        return df
    df = df.dropna(subset=["R", "RA", "Opp", "W/L"])
    if df.empty:
        return pd.DataFrame()
    parsed = df["Date"].apply(lambda d: _parse_date_and_game_num(d, season))
    df["date"] = parsed.apply(lambda t: t[0])
    df["game_num"] = parsed.apply(lambda t: t[1])
    df["season"] = season
    df["home_team"] = code
    df["away_team"] = df["Opp"].apply(lambda o: team_mapping.code_from_bref(o, season))
    df["home_score"] = df["R"].astype(float)
    df["away_score"] = df["RA"].astype(float)
    df["home_win"] = df["W/L"].str.startswith("W").astype(int)
    out = df[
        ["date", "season", "game_num", "home_team", "away_team", "home_score", "away_score", "home_win"]
    ]
    return out.reset_index(drop=True)


def fetch_all(seasons, pause=1.0, verbose=True):
    frames = []
    for season in seasons:
        for code in team_mapping.all_codes():
            try:
                team_df = fetch_team_season(code, season)
            except Exception as exc:
                if verbose:
                    print(f"  [warn] {season} {code}: {exc}")
                time.sleep(pause)
                continue
            if verbose:
                print(f"  {season} {code}: {len(team_df)} home games")
            if not team_df.empty:
                frames.append(team_df)
            time.sleep(pause)
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    games = games.sort_values("date").drop_duplicates(
        subset=["date", "game_num", "home_team", "away_team"]
    )
    return games.reset_index(drop=True)


def build_and_save(seasons=None, out_path=None):
    seasons = seasons or (config.HISTORICAL_SEASONS + [config.CURRENT_SEASON])
    out_path = out_path or config.GAMES_FILE
    games = fetch_all(seasons)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(out_path, index=False)
    print(f"Saved {len(games)} games to {out_path}")
    return games


if __name__ == "__main__":
    build_and_save()
