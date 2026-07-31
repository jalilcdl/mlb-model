"""
Split a club's pitching into ROTATION vs BULLPEN.

Why this matters: the current pitching grade is a single team-wide FIP, so a
club with a strong rotation and an exhausted bullpen grades identically to the
reverse. Those are very different things to bet into -- especially for totals
and live/late-game markets -- and the market prices them separately.

STATUS: SCAFFOLDING. This module produces the split, but it is NOT wired into
any grade or projection yet, and must not be until the walk-forward backtest in
`validate_roles` shows it actually adds signal. The starting-pitcher adjustment
was shipped on plausibility and later measured as no better than nothing; that
is not repeating.

Data sources, and why two of them:
  - RUNTIME (dashboard): MLB Stats API `/stats?stats=season&group=pitching`
    returns every pitcher-season in ONE call (~750 rows) with team, games,
    games started, and the FIP components. Free, no auth, fast enough to cache.
  - BACKTEST (offline): the connected sports-data MCP has per-player, per-GAME
    pitching rows back to 2005, which is what a walk-forward validation needs.
    The MCP is not reachable from application code, so that path is an offline
    cache-to-CSV step rather than a runtime dependency.

Role classification: a pitcher is counted as a STARTER if he started at least
half his appearances, otherwise a RELIEVER. This is the conventional split and
is deliberately simple; swingmen and openers are genuinely ambiguous and get
assigned to whichever role dominates their usage. Season totals cannot be
decomposed by appearance type from this endpoint, so a swingman's relief innings
ride along with the rotation (and vice versa) -- a known, bounded imprecision
that `role_purity` reports per club so it is visible rather than hidden.
"""
import datetime as dt

import numpy as np
import pandas as pd
import requests

from src import config
from src.data import team_mapping

STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
_UA = {"User-Agent": "mlb-model/1.0 (+local research tool)"}
_TIMEOUT = 45
_cache = {}

STARTER_GS_SHARE = 0.5  # started >= half his appearances -> starter


def _parse_ip(ip_str):
    whole, _, frac = str(ip_str).partition(".")
    return (float(whole) if whole else 0.0) + {"": 0.0, "0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(frac, 0.0)


def fetch_pitcher_seasons(season=None):
    """Every pitcher-season in the league, one row per player, with role."""
    season = season or config.CURRENT_SEASON
    key = ("pitchers", season, dt.date.today().isoformat())
    if key in _cache:
        return _cache[key]

    r = requests.get(STATS_URL, params={
        "stats": "season", "group": "pitching", "season": season,
        "sportId": 1, "playerPool": "All", "limit": 2000,
    }, headers=_UA, timeout=_TIMEOUT)
    r.raise_for_status()

    rows = []
    for s in r.json().get("stats", [{}])[0].get("splits", []):
        st = s.get("stat", {})
        team_obj = s.get("team") or {}
        code = None
        if team_obj.get("id") is not None:
            try:
                code = team_mapping.code_from_statsapi_id(team_obj["id"])
            except KeyError:
                code = None
        ip = _parse_ip(st.get("inningsPitched", "0"))
        g, gs = st.get("gamesPitched", 0) or 0, st.get("gamesStarted", 0) or 0
        if ip <= 0 or g <= 0:
            continue
        rows.append({
            "player": (s.get("player") or {}).get("fullName"),
            "team": code,
            "G": g, "GS": gs, "IP": ip,
            "HR": st.get("homeRuns", 0) or 0,
            "BB": st.get("baseOnBalls", 0) or 0,
            "HBP": st.get("hitBatsmen", 0) or 0,
            "K": st.get("strikeOuts", 0) or 0,
            "ER": st.get("earnedRuns", 0) or 0,
            "role": "rotation" if (gs / g) >= STARTER_GS_SHARE else "bullpen",
        })
    df = pd.DataFrame(rows)
    _cache[key] = df
    return df


def team_role_splits(season=None, pitchers=None):
    """Per team per role: IP, FIP, ERA, K/9, BB/9, plus each role's IP share.

    FIP is league-normalized across BOTH roles together so rotation and bullpen
    numbers sit on the same scale and are directly comparable. (Relievers
    normally post better raw FIP than starters -- shorter outings, max effort --
    so a role-specific constant would hide exactly the gap we want to see.)
    """
    df = fetch_pitcher_seasons(season) if pitchers is None else pitchers
    df = df[df["team"].notna()]
    if df.empty:
        return pd.DataFrame()

    grp = (df.groupby(["team", "role"])
             .agg(IP=("IP", "sum"), HR=("HR", "sum"), BB=("BB", "sum"),
                  HBP=("HBP", "sum"), K=("K", "sum"), ER=("ER", "sum"),
                  pitchers=("player", "count"))
             .reset_index())
    grp = grp[grp["IP"] > 0]
    grp["ERA"] = 9.0 * grp["ER"] / grp["IP"]
    grp["K9"] = 9.0 * grp["K"] / grp["IP"]
    grp["BB9"] = 9.0 * grp["BB"] / grp["IP"]
    grp["_fip_raw"] = (13 * grp["HR"] + 3 * (grp["BB"] + grp["HBP"]) - 2 * grp["K"]) / grp["IP"]
    # Single league constant across both roles (see docstring).
    grp["FIP"] = grp["_fip_raw"] + (grp["ERA"].mean() - grp["_fip_raw"].mean())
    grp = grp.drop(columns=["_fip_raw"])

    tot = grp.groupby("team")["IP"].transform("sum")
    grp["ip_share"] = grp["IP"] / tot
    return grp


def role_purity(season=None, pitchers=None):
    """How cleanly each club splits into starters and relievers.

    Reports the share of innings thrown by pitchers whose usage is ambiguous
    (started between 20% and 80% of their appearances). A high number means that
    club's rotation/bullpen figures are blurred by swingmen and should be read
    with caution -- surfaced rather than silently absorbed.
    """
    df = fetch_pitcher_seasons(season) if pitchers is None else pitchers
    df = df[df["team"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    share = df["GS"] / df["G"]
    df["ambiguous"] = (share > 0.2) & (share < 0.8)
    out = (df.groupby("team")
             .apply(lambda x: pd.Series({
                 "ambiguous_ip_share": float(x.loc[x["ambiguous"], "IP"].sum() / x["IP"].sum()),
                 "total_IP": float(x["IP"].sum()),
             }), include_groups=False)
             .reset_index())
    return out


def summary(season=None):
    """Wide table: one row per club with rotation vs bullpen FIP and the gap."""
    splits = team_role_splits(season)
    if splits.empty:
        return splits
    wide = splits.pivot(index="team", columns="role",
                        values=["FIP", "ERA", "IP", "ip_share", "pitchers"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    if "FIP_rotation" in wide and "FIP_bullpen" in wide:
        # Positive gap => bullpen is worse than the rotation.
        wide["fip_gap"] = wide["FIP_bullpen"] - wide["FIP_rotation"]
    return wide.merge(role_purity(season), on="team", how="left")
