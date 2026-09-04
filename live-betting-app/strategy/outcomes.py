"""Resolve a logged game's FINAL outcome, for the backtest only.

The signal log has in-progress snapshots, never a final result -- there's no
ground truth to grade a bet against without this. Backtest-only: the poller
never imports this module, so a bug or API change here cannot affect live
signal logging or Telegram.

Both lookups use free-tier endpoints only:
  MLB: MLB Stats API feed/live (already vendored, same one game_state.py uses)
  CFB: CFBD's plain /games?id=X (NOT the Patreon-gated live endpoints --
       backtested games are already final by the time this runs, so the
       ordinary free /games lookup is exactly what's needed)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent.parent
for vendor in ("sports/mlb/vendor", "sports/cfb/vendor"):
    p = str(_ROOT / vendor)
    if p not in sys.path:
        sys.path.insert(0, p)

from mlb_lib.live.game_state import fetch_feed as mlb_fetch_feed, parse_game_state


def resolve_mlb_outcome(game_pk: str) -> dict | None:
    """{"completed": bool, "home_won": bool|None, "home_score": int, "away_score": int}
    or None if the fetch failed. home_won is None on a tie (extra-inning games
    don't tie in MLB, but defensive anyway)."""
    try:
        feed = mlb_fetch_feed(int(game_pk))
        parsed = parse_game_state(feed)
    except Exception as e:
        print(f"[outcomes] mlb {game_pk}: fetch failed: {type(e).__name__}: {e}")
        return None
    if parsed["abstract"] != "Final":
        return {"completed": False, "home_won": None,
                "home_score": parsed["home_score"], "away_score": parsed["away_score"]}
    hs, aws = parsed["home_score"], parsed["away_score"]
    home_won = hs > aws if hs != aws else None
    return {"completed": True, "home_won": home_won, "home_score": hs, "away_score": aws}


def resolve_cfb_outcome(game_id: str) -> dict | None:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        print("[outcomes] cfb: CFBD_API_KEY not set")
        return None
    try:
        r = requests.get("https://api.collegefootballdata.com/games",
                         headers={"Authorization": f"Bearer {key}"},
                         params={"id": int(game_id)}, timeout=15)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        print(f"[outcomes] cfb {game_id}: fetch failed: {type(e).__name__}: {e}")
        return None
    if not rows:
        return None
    g = rows[0]
    if not g.get("completed"):
        return {"completed": False, "home_won": None,
                "home_score": g.get("homePoints"), "away_score": g.get("awayPoints")}
    hs, aws = g.get("homePoints"), g.get("awayPoints")
    if hs is None or aws is None:
        return None
    home_won = hs > aws if hs != aws else None
    return {"completed": True, "home_won": home_won, "home_score": hs, "away_score": aws}


def resolve_outcome(sport: str, game_id: str) -> dict | None:
    if sport == "mlb":
        return resolve_mlb_outcome(game_id)
    if sport == "cfb":
        return resolve_cfb_outcome(game_id)
    raise ValueError(f"unknown sport: {sport}")
