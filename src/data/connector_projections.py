"""
Connector model projections: normalize, blend, snapshot, and log.

The connector's `get_game_projections` returns an INDEPENDENT model's projected
score distribution per game -- percentiles p1..p99 plus a mean -- for total,
spread, homeScore, awayScore, and homeWinProbability. Critically, the
projection is FROZEN at game time (its `updated_at` sits ~seconds before first
pitch and does not move afterward), so a historical pull is a genuine pregame
number with no lookahead -- verified on 2026-05-01 and 2026-07-28 games.

This module does three things, none of which touch the network (the connector is
MCP-only and agent/orchestrator-pulled, exactly like the moneyline get_events and
the totals get_game_odds feeds):

  1. normalize() -- turn a get_game_projections response into a flat per-game row.
  2. blend_total() -- combine our model's projected total with the connector's,
     weighted by config.TOTALS_CONNECTOR_WEIGHT (0.5 = equal). This is the number
     the totals over/under EDGE is now computed from. The blend is a plain mean of
     two independent projections; it does not claim either is better -- that's what
     the compare log below is for.
  3. append_compare_log() -- append one row per game per night to
     projection_compare_log.csv: our total, connector total, blended total, market
     consensus, and the actual total once known. This is the ongoing, growing
     paired dataset to eventually TUNE the blend weight against real outcomes (a
     full historical backtest wasn't run -- the connector's projections are
     per-event MCP-only, so ~1k games can't be pulled by hand; the log grows the
     sample ~15 games/night instead).
"""
import datetime as dt
import json

import pandas as pd

from src import config
from src.odds.odds_adapter import _normalize_team

SNAPSHOT_DIR = config.ODDS_DIR / "snapshots"
COMPARE_LOG = config.PROCESSED_DIR / "projection_compare_log.csv"
_LOG_COLUMNS = [
    "date", "away_team", "home_team", "our_total", "connector_total",
    "blended_total", "connector_weight", "consensus_total", "actual_total",
    "connector_updated_at", "logged_at",
]


def normalize(resp):
    """One get_game_projections response (list with a single dict, or the dict) ->
    a flat dict, or None if it carries no usable total mean. Team codes are mapped
    to our canonical codes so snapshots/logs join cleanly to predictions."""
    if isinstance(resp, list):
        resp = resp[0] if resp else None
    if not resp:
        return None
    total_mean = None
    percentiles = {}
    for proj in resp.get("projections", []):
        if proj.get("projectionType") == "total":
            total_mean = proj.get("mean")
            percentiles = {k: proj[k] for k in
                           ("p10", "p25", "p50", "p75", "p90") if k in proj}
            break
    if total_mean is None:
        return None
    away = _normalize_team(resp.get("away_team"))
    home = _normalize_team(resp.get("home_team"))
    if not away or not home:
        return None
    return {
        "away_team": away,
        "home_team": home,
        "start_time_utc": resp.get("start_date"),
        "connector_total": float(total_mean),
        "connector_home_win_prob": resp.get("homeWinProbability"),
        "connector_updated_at": resp.get("updated_at"),
        "percentiles": percentiles,
    }


def blend_total(our_total, connector_total, weight=None):
    """Weighted mean of the two projected totals. `weight` is the weight on the
    CONNECTOR (default config.TOTALS_CONNECTOR_WEIGHT); our weight is 1 - weight.
    Returns our_total unchanged if the connector value is missing, so a game with
    no projection degrades to our-model-only rather than dropping out."""
    w = config.TOTALS_CONNECTOR_WEIGHT if weight is None else weight
    if connector_total is None or (isinstance(connector_total, float) and pd.isna(connector_total)):
        return float(our_total)
    return float((1.0 - w) * float(our_total) + w * float(connector_total))


# --- snapshot: {(away, home): normalized-dict}, so build_totals_rows can look up
#     the connector total for a game the same way it looks up its odds. ---
def snapshot_path(date_str):
    return SNAPSHOT_DIR / f"connector_projections_{date_str}.json"


def write_snapshot(date_str, rows):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_path(date_str)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    return path


def load_snapshot(date_str):
    """Return {(away, home): row} for the date, or {} if no snapshot exists."""
    path = snapshot_path(date_str)
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    return {(r["away_team"], r["home_team"]): r for r in rows}


def append_compare_log(records, path=None):
    """Append rows to the ongoing compare log, de-duping on (date, away, home) so
    re-running a night overwrites rather than double-counts. Each record needs at
    least date/away_team/home_team; missing optional fields are filled with NA."""
    path = path or COMPARE_LOG
    new = pd.DataFrame(records)
    if new.empty:
        return path
    now = dt.datetime.now().isoformat(timespec="seconds")
    new["logged_at"] = now
    for c in _LOG_COLUMNS:
        if c not in new.columns:
            new[c] = pd.NA
    new = new[_LOG_COLUMNS]

    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["date", "away_team", "home_team"], keep="last")
    else:
        combined = new
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return path
