"""MLB sport adapter -- wraps the vendored mlb_lib (copied from mlb-model's
src/live/*) behind the common SportAdapter interface (sports/base.py).

Odds: SharpAPI live moneylines (SHARPAPI_API_KEY env var; mock slate if unset).
State: MLB Stats API feed/live (free, no key).
Model: in-game Monte Carlo win probability vs de-vigged market -- identical
math to mlb-model's live_signal.py, just repackaged to emit unified rows
instead of writing its own JSONL file directly.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from mlb_lib.data import statsapi_client
from mlb_lib.live import game_state as gs
from mlb_lib.live import live_odds
from mlb_lib.live import signal as sig
from mlb_lib.live.win_expectancy import load_team_ratings

SPORT_KEY = "mlb"
SPORT_LABEL = "MLB"
SPORT_ICON = "⚾"

_LIVE_STATES = {"In Progress", "Manager challenge", "Replay Review"}
_DEVIG_METHOD = os.environ.get("LIVE_DEVIG_METHOD", "proportional")

_ratings = None  # lazy-loaded, cached for the life of the process


def _get_ratings():
    global _ratings
    if _ratings is None:
        _ratings = load_team_ratings()
    return _ratings


def _now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _odds_by_matchup(provider):
    if not hasattr(provider, "list_moneylines"):
        # MockOddsProvider (no SHARPAPI_API_KEY configured) only supports
        # per-game .moneyline() lookups, not a batch listing -- nothing to poll.
        return {}
    try:
        book = provider.list_moneylines(is_live=True)
    except Exception as e:
        print(f"[mlb] live odds fetch failed: {type(e).__name__}: {e}")
        return {}
    return {(ml.away_team, ml.home_team): ml for ml in book.values()}


def _to_unified_row(game, meta, state, ml, signal) -> dict:
    bases_on = [i for i, b in enumerate(state.bases, start=1) if b]
    state_desc = (f"{'Top' if state.half == 'top' else 'Bot'} {state.inning}, "
                  f"{state.outs} out" + (f", on {'/'.join(map(str, bases_on))}" if bases_on else ""))
    return {
        "mode": "OBSERVE_ONLY",
        "sport": SPORT_KEY,
        "logged_at_utc": _now_utc(),
        "game_id": str(game["game_pk"]),
        "matchup": f"{signal.away_team} @ {signal.home_team}",
        "home_team": signal.home_team, "away_team": signal.away_team,
        "home_score": state.home_score, "away_score": state.away_score,
        "state_desc": state_desc,
        "model_home_wp": signal.model_home_wp, "market_home_wp": signal.market_home_wp,
        "edge_home": signal.edge_home, "edge": signal.edge, "flagged": signal.flagged,
        "pick_team": signal.pick_team, "devig_method": signal.devig_method,
        "odds_source": ml.source,
        # MLB-specific extras, preserved for the dashboard's detail view
        "inning": state.inning, "half": state.half, "outs": state.outs,
        "bases": list(state.bases), "batter": meta.get("batter"), "pitcher": meta.get("pitcher"),
        "home_odds": ml.home_odds, "away_odds": ml.away_odds, "vig": signal.vig,
    }


def poll() -> list[dict]:
    """Unified rows for every MLB game currently in progress. Empty if none.

    Checks both the current UTC date and the previous one. MLB's schedule is
    keyed by the US game date, but this poller can run on a UTC-clock host
    (GitHub Actions runners are UTC) -- a 9pm Eastern game is already
    "tomorrow" in UTC, so date.today() alone silently queries the wrong day
    and reports 0 live games while real games are in progress. Checking
    yesterday-UTC too covers that gap without needing a timezone database.
    """
    today_utc = dt.datetime.now(dt.timezone.utc).date()
    candidate_dates = {today_utc.isoformat(), (today_utc - dt.timedelta(days=1)).isoformat()}

    live_games = []
    seen_pks = set()
    for date_str in candidate_dates:
        try:
            schedule = statsapi_client.get_schedule(date_str)
        except Exception as e:
            print(f"[mlb] schedule fetch failed for {date_str}: {type(e).__name__}: {e}")
            continue
        for g in schedule:
            if g.get("status") in _LIVE_STATES and g["game_pk"] not in seen_pks:
                seen_pks.add(g["game_pk"])
                live_games.append(g)
    if not live_games:
        return []

    provider = live_odds.get_provider()
    odds = _odds_by_matchup(provider)
    ratings = _get_ratings()

    rows = []
    for g in live_games:
        pk, a, h = g["game_pk"], g["away_team"], g["home_team"]
        try:
            meta = gs.live_game_state(pk)
        except Exception as e:
            print(f"[mlb] {a}@{h}: state fetch failed ({type(e).__name__}); skipping")
            continue
        state = meta["state"]
        if state is None:
            continue
        ml = odds.get((a, h))
        if ml is None:
            continue
        s = sig.evaluate(state, ml, ratings=ratings, n_sims=20000, seed=7,
                         devig_method=_DEVIG_METHOD)
        rows.append(_to_unified_row(g, meta, state, ml, s))
    return rows


def state_key(row: dict) -> tuple:
    """What counts as 'a new state worth logging' for this game."""
    return (row["game_id"], row["inning"], row["half"], row["outs"],
            tuple(row["bases"]), row["home_score"], row["away_score"])
