"""Live CFB game state from CollegeFootballData.com (CFBD), replacing ESPN's
public scoreboard as the state source.

Why: ESPN's scoreboard endpoint 403s specifically from GitHub Actions'
runner IPs -- confirmed by direct testing (the identical request, same
headers, succeeds from other networks and fails only from CI), which no
request header fixes. CFBD is a small, purpose-built developer API rather
than a major consumer site with the same anti-scraping posture, and this
project already holds a CFBD_API_KEY (used elsewhere in cfb-model's pregame
pipeline) that authorizes it -- provide it via the CFBD_API_KEY env var.

Two calls, mirroring the shape this replaces:
  GET /scoreboard          -> which games are in_progress, team names/ids/scores
  GET /live/plays?gameId=X -> structured down/distance/clock/yardsToGoal/possession

Produces the exact same GameState (see wp_model.py) the rest of the pipeline
already consumes, so nothing downstream (win_probability, the signal
comparison, logging, dashboard) needs to change.
"""
from __future__ import annotations

import os

import requests

from cfb_lib.live.wp_model import GameState

BASE = "https://api.collegefootballdata.com"
_TIMEOUT = 15


def _headers() -> dict:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise RuntimeError(
            "CFBD_API_KEY not set. Free key at https://collegefootballdata.com/key")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


def fetch_scoreboard(classification: str = "fbs") -> list[dict]:
    r = requests.get(f"{BASE}/scoreboard", headers=_headers(),
                     params={"classification": classification}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_live_plays(game_id: int) -> dict:
    r = requests.get(f"{BASE}/live/plays", headers=_headers(),
                     params={"gameId": game_id}, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def list_live_games(classification: str = "fbs") -> list[dict]:
    """Scoreboard entries currently in_progress."""
    games = fetch_scoreboard(classification)
    return [g for g in games if (g.get("status") or "") == "in_progress"]


def _clock_to_seconds(clock: str | None) -> int:
    if not clock:
        return 0
    parts = str(clock).split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(float(parts[0]))
    except ValueError:
        return 0


def build_game_state(scoreboard_game: dict) -> tuple[GameState, dict]:
    """Fetch /live/plays for one in-progress scoreboard game and build its
    GameState. Returns (state, meta); meta carries team names/ids for
    matching against odds, the same role espn_state.parse_live_event's
    second return value played."""
    game_id = scoreboard_game["id"]
    live = fetch_live_plays(game_id)

    home_team = scoreboard_game.get("homeTeam") or {}
    away_team = scoreboard_game.get("awayTeam") or {}
    home_id, home_name = home_team.get("id"), home_team.get("name")
    away_id, away_name = away_team.get("id"), away_team.get("name")

    # /live/plays carries the authoritative live score per team; fall back to
    # the scoreboard's own points if a team entry is momentarily missing.
    home_score, away_score = home_team.get("points", 0), away_team.get("points", 0)
    for t in live.get("teams", []) or []:
        if t.get("homeAway") == "home":
            home_score = t.get("points", home_score)
        elif t.get("homeAway") == "away":
            away_score = t.get("points", away_score)

    possession = live.get("possession")
    home_has_ball = None
    if possession is not None:
        # CFBD doesn't document whether `possession` is a team name or a team
        # id -- match against both defensively.
        home_has_ball = (str(possession) == str(home_name)
                          or str(possession) == str(home_id))

    state = GameState(
        home_score=int(home_score or 0),
        away_score=int(away_score or 0),
        period=int(live.get("period") or scoreboard_game.get("period") or 0),
        clock_seconds=_clock_to_seconds(live.get("clock") or scoreboard_game.get("clock")),
        down=live.get("down"),
        distance=live.get("distance"),
        yards_to_endzone=live.get("yardsToGoal"),
        home_has_ball=home_has_ball,
        state="in",
    )
    meta = {"home_id": home_id, "away_id": away_id,
            "home_name": home_name, "away_name": away_name}
    return state, meta
