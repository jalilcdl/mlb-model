"""
Real historical MLB closing moneyline loader and vig-removal utility.

Source: pwu97/bettingtools (GitHub), MLB 2014-2019. Each row has closing
moneylines, run lines, and total lines. This module focuses on the moneyline
side for training a classifier and backtesting against the market.

No API key, no auth, no scraping. Free public data.
"""
import tempfile
from pathlib import Path

import pandas as pd
import requests

from src import config
from src.data import team_mapping

BASE_URL = "https://raw.githubusercontent.com/pwu97/bettingtools/master/data/mlb_odds_{year}.rda"
YEARS = range(2014, 2020)

# Cached files (one-time bulk download)
HIST_GAMES_FILE = config.DATA_DIR / "raw" / "hist_odds_games.csv"
HIST_ML_FILE = config.DATA_DIR / "raw" / "historical_moneylines_2014_2019.csv"


def _fetch_year(year):
    """Download one year's .rda file and return the raw DataFrame."""
    try:
        import pyreadr
    except ImportError as exc:
        raise RuntimeError("pyreadr required: pip install pyreadr") from exc

    r = requests.get(BASE_URL.format(year=year), timeout=30)
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


def _resolve_teams(raw):
    """Map team names to canonical codes. Drop unresolvable rows."""
    raw["home_code"] = raw.apply(
        lambda r: team_mapping.code_from_name(r["home_name"], r["home_abbrev"]), axis=1
    )
    raw["away_code"] = raw.apply(
        lambda r: team_mapping.code_from_name(r["away_name"], r["away_abbrev"]), axis=1
    )
    unresolved = raw[raw["home_code"].isna() | raw["away_code"].isna()]
    if not unresolved.empty:
        names = pd.unique(unresolved[["home_name", "away_name"]].values.ravel())
        print(f"[!] {len(unresolved)} rows dropped (unresolved names): {list(names)[:10]}")
    return raw.dropna(subset=["home_code", "away_code", "home_score", "away_score"]).copy()


def _ml_to_prob(ml):
    """Convert American moneyline to implied probability (before vig removal)."""
    ml = pd.to_numeric(ml, errors="coerce")
    prob = pd.Series(index=ml.index, dtype=float)
    prob[ml < 0] = (-ml[ml < 0]) / (-ml[ml < 0] + 100)
    prob[ml > 0] = 100 / (ml[ml > 0] + 100)
    return prob


def _remove_vig(home_prob, away_prob):
    """Multiplicative vig removal: normalize so probs sum to 1."""
    total = home_prob + away_prob
    return home_prob / total, away_prob / total


def build(years=YEARS, verbose=True):
    """Download moneyline odds and return (games_df, ml_df).

    games_df: same shape as games.csv (date, season, home_team, away_team,
              home_score, away_score, home_win)
    ml_df:    date, home_team, away_team, home_close_ml, away_close_ml,
              home_impl_prob_raw, away_impl_prob_raw,
              home_impl_prob, away_impl_prob,  (vig-removed)
              market_favorite, market_favorite_prob
    """
    raw = pd.concat([_fetch_year(y) for y in years], ignore_index=True)
    raw = _resolve_teams(raw)
    raw["date"] = pd.to_datetime(raw["date"])

    # --- Game results ---
    games = raw[["date", "season", "home_code", "away_code", "home_score", "away_score"]].rename(
        columns={"home_code": "home_team", "away_code": "away_team"}
    ).copy()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)

    # --- Moneyline features ---
    ml = pd.DataFrame({
        "date": raw["date"],
        "home_team": raw["home_code"],
        "away_team": raw["away_code"],
        "home_close_ml": raw["home_close_ml"],
        "away_close_ml": raw["away_close_ml"],
    }).copy()

    # Convert to implied probabilities
    ml["home_impl_prob_raw"] = _ml_to_prob(ml["home_close_ml"])
    ml["away_impl_prob_raw"] = _ml_to_prob(ml["away_close_ml"])

    # Remove vig
    h_prob, a_prob = _remove_vig(ml["home_impl_prob_raw"], ml["away_impl_prob_raw"])
    ml["home_impl_prob"] = h_prob
    ml["away_impl_prob"] = a_prob

    # Market favorite (for naive baseline)
    ml["market_favorite"] = ml.apply(
        lambda r: "home" if r["home_impl_prob"] >= r["away_impl_prob"] else "away", axis=1
    )
    ml["market_favorite_prob"] = ml[["home_impl_prob", "away_impl_prob"]].max(axis=1)

    # Sanity checks
    bad = ml[(ml["home_impl_prob"] + ml["away_impl_prob"] - 1.0).abs() > 0.001]
    if not bad.empty and verbose:
        print(f"[!] {len(bad)} rows where vig-removed probs don't sum to 1 (should be 0)")

    return games.reset_index(drop=True), ml.reset_index(drop=True)


def save(years=YEARS):
    """Download and cache to disk."""
    games, ml = build(years)
    HIST_GAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(HIST_GAMES_FILE, index=False)
    ml.to_csv(HIST_ML_FILE, index=False)
    print(f"Saved {len(games)} games -> {HIST_GAMES_FILE}")
    print(f"Saved {len(ml)} moneylines -> {HIST_ML_FILE}")
    print(f"Seasons: {sorted(games['season'].unique())}")
    return games, ml


def load():
    """Load from cache, downloading if necessary."""
    if not HIST_GAMES_FILE.exists() or not HIST_ML_FILE.exists():
        save()
    games = pd.read_csv(HIST_GAMES_FILE, parse_dates=["date"])
    ml = pd.read_csv(HIST_ML_FILE, parse_dates=["date"])
    return games, ml


def implied_prob_to_ml(prob):
    """Reverse: fair probability -> American moneyline (for EV calc)."""
    prob = float(prob)
    if prob >= 0.5:
        return -round(100 * prob / (1 - prob))
    else:
        return round(100 * (1 - prob) / prob)


def ev_percent(model_prob, market_ml):
    """Expected value % if betting $100 at the given market moneyline.

    model_prob: our fair probability of the bet winning
    market_ml:  American moneyline of the bet
    """
    market_ml = float(market_ml)
    if market_ml < 0:
        profit = 100 * 100 / (-market_ml)  # e.g. -140 -> win $71.43
    else:
        profit = market_ml  # e.g. +120 -> win $120
    loss = 100.0
    return (model_prob * profit - (1 - model_prob) * loss) / 100.0


if __name__ == "__main__":
    games, ml = save()
    print(f"\nSample moneyline row:")
    print(ml.head(1).to_string())
    print(f"\nMarket favorite accuracy (naive baseline):")
    merged = games.merge(ml, on=["date", "home_team", "away_team"])
    naive_correct = (
        (merged["market_favorite"] == "home") & (merged["home_win"] == 1)
    ) | (
        (merged["market_favorite"] == "away") & (merged["home_win"] == 0)
    )
    print(f"  {naive_correct.mean():.3f} ({naive_correct.sum()}/{len(merged)})")
