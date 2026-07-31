"""
Team-level pitching and fielding stats, for grades that mean what they say.

Why this module exists: the offense/"pitching" grades used to be computed from
runs scored and runs allowed. Runs allowed is NOT pitching -- it bundles the
rotation, the bullpen, the defense behind them, and sequencing luck into one
number. This module sources the components separately so the three things can be
graded separately.

  Pitching  -> FIP from MLB Stats API team pitching totals.
               FIP is defense-INDEPENDENT by construction (only HR, BB, HBP, K
               -- outcomes no fielder touches), so it isolates pitching without
               having to subtract a fielding estimate.
  Fielding  -> Outs Above Average / fielding runs prevented from Baseball
               Savant (via pybaseball). A real defensive metric, not errors.
  Combined  -> park-adjusted runs allowed stays available as "Run Prevention",
               which is the honest label for what actually happened.

Both sources are free and need no auth or API key. Everything is fetched at
runtime and cached per (season, day) so a dashboard rerun does not re-hit the
network.

KNOWN LIMITS, surfaced in the UI rather than buried here:
  - Savant OAA covers only qualified fielders (~8 per club), so team totals
    UNDERCOUNT: bench and part-time defenders are missing, and players who
    changed teams sit in a "---" bucket that is dropped entirely.
  - OAA is season-to-date only. There is no rolling window, so unlike the run
    ratings it cannot be recency-weighted -- an early-season slump stays baked
    in all year.
  - FIP is park-influenced through home runs, so it is park-normalized here
    using each club's actual schedule exposure (see park_exposure).
"""
import datetime as dt

import numpy as np
import pandas as pd
import requests

from src import config
from src.data import team_mapping

STATSAPI_TEAM_STATS = "https://statsapi.mlb.com/api/v1/teams/stats"
_UA = {"User-Agent": "mlb-model/1.0 (+local research tool)"}
_TIMEOUT = 30
_cache = {}


def _parse_ip(ip_str):
    """MLB innings notation: '54.1' = 54 + 1/3, '54.2' = 54 + 2/3."""
    whole, _, frac = str(ip_str).partition(".")
    return (float(whole) if whole else 0.0) + {"": 0.0, "0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(frac, 0.0)


def fetch_team_pitching(season=None):
    """Team pitching components from the MLB Stats API, with FIP computed.

    Returns a DataFrame: team, ERA, FIP, WHIP, K9, BB9, HR9, IP.
    FIP is rescaled so the league mean matches league ERA, which is what the
    additive FIP constant does -- computing it from this season's own data
    rather than hardcoding 3.10.
    """
    season = season or config.CURRENT_SEASON
    key = ("pitching", season, dt.date.today().isoformat())
    if key in _cache:
        return _cache[key]

    r = requests.get(STATSAPI_TEAM_STATS,
                     params={"season": season, "sportIds": 1, "group": "pitching", "stats": "season"},
                     headers=_UA, timeout=_TIMEOUT)
    r.raise_for_status()
    rows = []
    for s in r.json().get("stats", [{}])[0].get("splits", []):
        st = s.get("stat", {})
        try:
            code = team_mapping.code_from_statsapi_id(s["team"]["id"])
        except (KeyError, TypeError):
            continue
        ip = _parse_ip(st.get("inningsPitched", "0"))
        if ip <= 0:
            continue
        hr, bb = st.get("homeRuns", 0), st.get("baseOnBalls", 0)
        hbp, k = st.get("hitBatsmen", 0), st.get("strikeOuts", 0)
        rows.append({
            "team": code,
            "ERA": float(st.get("era") or 0),
            "WHIP": float(st.get("whip") or 0),
            "IP": ip,
            "K9": 9.0 * k / ip,
            "BB9": 9.0 * bb / ip,
            "HR9": 9.0 * hr / ip,
            "_fip_raw": (13 * hr + 3 * (bb + hbp) - 2 * k) / ip,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # FIP constant: shift so league-mean FIP == league-mean ERA.
    df["FIP"] = df["_fip_raw"] + (df["ERA"].mean() - df["_fip_raw"].mean())
    df = df.drop(columns=["_fip_raw"])
    _cache[key] = df
    return df


def fetch_team_fielding(season=None):
    """Team fielding from Baseball Savant OAA (primary) plus MLB Stats API
    fielding percentage / errors (secondary, weaker but complete).

    Returns a DataFrame: team, OAA, runs_prevented, n_fielders, fielding_pct,
    errors. OAA columns are NaN if Savant is unreachable, so callers can fall
    back rather than crash.
    """
    season = season or config.CURRENT_SEASON
    key = ("fielding", season, dt.date.today().isoformat())
    if key in _cache:
        return _cache[key]

    # --- Savant OAA (primary) ---
    oaa = pd.DataFrame(columns=["team", "OAA", "runs_prevented", "n_fielders"])
    try:
        import pybaseball as pb
        raw = pb.statcast_outs_above_average(season, pos="all")
        raw = raw.copy()
        raw["team"] = raw["display_team_name"].map(team_mapping.code_from_name)
        raw = raw[raw["team"].notna()]  # drops the '---' multi-team bucket
        oaa = (raw.groupby("team")
                  .agg(OAA=("outs_above_average", "sum"),
                       runs_prevented=("fielding_runs_prevented", "sum"),
                       n_fielders=("player_id", "count"))
                  .reset_index())
    except Exception:
        pass  # Savant unavailable -> OAA stays empty, caller degrades to fielding_pct

    # --- statsapi fielding (secondary: complete but much weaker signal) ---
    trad = pd.DataFrame(columns=["team", "fielding_pct", "errors"])
    try:
        r = requests.get(STATSAPI_TEAM_STATS,
                         params={"season": season, "sportIds": 1, "group": "fielding", "stats": "season"},
                         headers=_UA, timeout=_TIMEOUT)
        r.raise_for_status()
        rows = []
        for s in r.json().get("stats", [{}])[0].get("splits", []):
            st = s.get("stat", {})
            try:
                code = team_mapping.code_from_statsapi_id(s["team"]["id"])
            except (KeyError, TypeError):
                continue
            rows.append({"team": code,
                         "fielding_pct": float(st.get("fielding") or 0),
                         "errors": int(st.get("errors") or 0)})
        trad = pd.DataFrame(rows)
    except Exception:
        pass

    df = trad.merge(oaa, on="team", how="outer") if not trad.empty else oaa
    _cache[key] = df
    return df


def park_exposure(games, season=None):
    """Each club's schedule-weighted park factor: the mean park factor of the
    venues it actually played in.

    Needed because FIP is park-influenced through home runs, and a club's
    exposure is not simply "half its own park" -- interleague and unbalanced
    divisional schedules skew it. Returns {team: mean_park_factor}.
    """
    from src.models.run_model import TeamRunRatings
    ratings = TeamRunRatings().fit(games)
    season = season or config.CURRENT_SEASON
    cur = games[games["season"] == season]
    exposure = {}
    for team in set(cur["home_team"]) | set(cur["away_team"]):
        played = cur[(cur["home_team"] == team) | (cur["away_team"] == team)]
        pf = played["home_team"].map(ratings.park_factor).astype(float)
        exposure[team] = float(pf.mean()) if len(pf) else 1.0
    return exposure
