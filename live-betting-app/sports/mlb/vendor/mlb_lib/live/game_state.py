"""
Live MLB game state from the free, public MLB Stats API (no key required).

SharpAPI's free tier is odds + schedule only; its /gamestate is a paid add-on
(a free key gets 403 tier_restricted). So we take the game STATE -- score,
inning, half, outs, baserunners -- from the MLB Stats API instead, and use
SharpAPI purely for odds. The two are joined by team codes + date elsewhere
(see live_signal.py).

Endpoint (public, free, no auth):
    https://statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live

We read liveData.linescore for the base/out/inning/score state and gameData for
status and team codes, and return the GameState the win-prob model consumes.
"""
from __future__ import annotations

import time

import requests

from mlb_lib.data import team_mapping
from mlb_lib.live.win_expectancy import GameState

FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
TIMESTAMPS_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live/timestamps"
_TIMEOUT = 10
_RETRIES = 3
_BACKOFF = 0.5
_UA = "mlb-model-live-prototype/0.1"


def fetch_feed(game_pk, timecode=None) -> dict:
    """GET the live feed for one gamePk, with retry/backoff on transient errors.
    `timecode` (YYYYMMDD_HHMMSS) returns the game's state AS OF that moment -- used
    to replay a real mid-game snapshot from a finished game for testing."""
    url = FEED_URL.format(game_pk=game_pk)
    params = {"timecode": timecode} if timecode else None
    last = None
    for attempt in range(_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT,
                                headers={"User-Agent": _UA})
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500:
                raise
            last = exc
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last = exc
        if attempt < _RETRIES - 1:
            time.sleep(_BACKOFF * (2 ** attempt))
    raise last


def list_timecodes(game_pk):
    """Ordered list of valid timecodes (YYYYMMDD_HHMMSS) for a game, for replaying
    a real mid-game snapshot. Empty list on error."""
    url = TIMESTAMPS_URL.format(game_pk=game_pk)
    try:
        resp = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA})
        resp.raise_for_status()
        return resp.json() or []
    except requests.exceptions.RequestException:
        return []


def _team_code(team_obj):
    """Canonical code for a gameData team object; fall back to its abbreviation."""
    tid = (team_obj or {}).get("id")
    if tid is not None:
        try:
            return team_mapping.code_from_statsapi_id(tid)
        except KeyError:
            pass
    return (team_obj or {}).get("abbreviation")


def parse_game_state(feed: dict) -> dict:
    """Turn a feed/live payload into a dict:
        {status, is_live, home_team, away_team, state (GameState|None),
         batter, pitcher, balls, strikes}
    `state` is None for a game that hasn't started (Preview) -- there is no
    base/out state yet -- and for a Final game (nothing left to simulate)."""
    game = feed.get("gameData", {})
    live = feed.get("liveData", {})
    status = game.get("status", {})
    abstract = status.get("abstractGameState")       # Preview / Live / Final
    detailed = status.get("detailedState")

    teams = game.get("teams", {})
    home_code = _team_code(teams.get("home"))
    away_code = _team_code(teams.get("away"))

    ls = live.get("linescore", {})
    ls_teams = ls.get("teams", {})
    home_runs = (ls_teams.get("home") or {}).get("runs")
    away_runs = (ls_teams.get("away") or {}).get("runs")
    offense = ls.get("offense", {})
    # Base keys ('first'/'second'/'third') are present only when occupied.
    bases = (int("first" in offense), int("second" in offense), int("third" in offense))
    is_top = ls.get("isTopInning")
    inning = ls.get("currentInning")
    outs = ls.get("outs")

    # Build a GameState whenever the linescore carries a valid in-progress state.
    # Not gated on abstractGameState so a timecode replay of a finished game (which
    # still reports abstract 'Final') yields its real mid-game state; the live
    # orchestrator gates on `is_live` separately, via the schedule status.
    state = None
    if inning and is_top is not None and outs is not None and int(outs) < 3:
        state = GameState(
            inning=int(inning),
            half="top" if is_top else "bottom",
            outs=int(outs),
            away_score=int(away_runs or 0),
            home_score=int(home_runs or 0),
            bases=bases,
        )

    # Current batter/pitcher, if the feed carries them (nice for context/logging).
    plays = live.get("plays", {}).get("currentPlay", {})
    matchup = plays.get("matchup", {})
    count = plays.get("count", {})
    return {
        "status": detailed or abstract,
        "abstract": abstract,
        "is_live": abstract == "Live",
        "home_team": home_code,
        "away_team": away_code,
        "home_score": int(home_runs or 0),
        "away_score": int(away_runs or 0),
        "state": state,
        "batter": (matchup.get("batter") or {}).get("fullName"),
        "pitcher": (matchup.get("pitcher") or {}).get("fullName"),
        "balls": count.get("balls"),
        "strikes": count.get("strikes"),
    }


def live_game_state(game_pk, timecode=None) -> dict:
    """Fetch + parse one game. Convenience wrapper around fetch_feed/parse.
    Pass `timecode` to replay a real mid-game snapshot of a finished game."""
    return parse_game_state(fetch_feed(game_pk, timecode=timecode))
