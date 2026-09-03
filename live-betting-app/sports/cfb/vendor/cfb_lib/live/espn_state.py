"""Live CFB game state from ESPN's public site API. Free, no key, no account.

Two paths, because they carry the state in different places:

- LIVE  : `/scoreboard` -> `competitions[0].situation` (down/distance/field
          position/possession) plus `.status` (period/clock). Verified reachable
          and correctly shaped for scheduled and completed games; the
          `situation` block is only populated while a game is actually in
          progress, so see the caveat below.
- REPLAY: `/summary?event=ID` -> `drives.previous[].plays[]`, where every play
          carries `start.down`, `start.distance`, `start.yardsToEndzone`,
          `start.team.id`, `period.number`, `clock.displayValue` and the running
          `homeScore`/`awayScore`. That is a complete mid-game snapshot per
          play, so a finished game replays as a state sequence -- the same trick
          the MLB prototype used with Stats API timecodes.

HONEST CAVEAT: the live `situation` field names below are taken from ESPN's
published shape and could not be observed against an in-progress game at build
time (nothing was live). `parse_live_event` therefore reads every field
defensively and reports what it actually found via `situation_seen`, rather than
assuming. Confirm against a live game before trusting the live path.

The summary feed also carries ESPN's own `winprobability` array
(`homeWinPercentage` per play), which `espn_win_probability` returns for use as
a free reference curve when sanity-checking our model.
"""
from __future__ import annotations

import datetime as _dt
import json
import urllib.request
from typing import Iterator

from cfb_lib.live.wp_model import GameState

BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
# ESPN 403s on unusual User-Agent strings -- a custom one naming this
# project was rejected outright. A plain browser UA is required.
UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch_scoreboard(date: str | None = None) -> dict:
    """`date` as YYYYMMDD; defaults to TODAY.

    The date is always sent explicitly. Omitting it does NOT give today's
    slate -- ESPN returns a week-scoped view instead, which was observed to
    exclude an in-progress game that `?dates=` returned as state="in". Passing
    the date is the difference between seeing live games and seeing none.
    """
    if date is None:
        date = _dt.datetime.now().strftime("%Y%m%d")
    return _get(f"{BASE}/scoreboard?dates={date}")


def fetch_summary(event_id: str | int) -> dict:
    return _get(f"{BASE}/summary?event={event_id}")


def _clock_to_seconds(display: str | None) -> int:
    if not display:
        return 0
    parts = str(display).split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(float(parts[0]))
    except ValueError:
        return 0


def list_events(scoreboard: dict) -> list[dict]:
    """Flatten the scoreboard into simple per-event dicts."""
    out = []
    for e in scoreboard.get("events", []):
        comp = (e.get("competitions") or [{}])[0]
        teams = {}
        for c in comp.get("competitors", []):
            teams[c.get("homeAway")] = {
                "id": c.get("team", {}).get("id"),
                "abbr": c.get("team", {}).get("abbreviation"),
                "name": c.get("team", {}).get("displayName"),
                "score": int(c.get("score") or 0),
            }
        status = comp.get("status", {})
        out.append({
            "event_id": e.get("id"),
            "short_name": e.get("shortName"),
            "date": e.get("date"),
            "state": status.get("type", {}).get("state"),
            "period": status.get("period"),
            "clock": status.get("displayClock"),
            "home": teams.get("home", {}),
            "away": teams.get("away", {}),
            "_competition": comp,
        })
    return out


def parse_live_event(event: dict) -> tuple[GameState, list[str]]:
    """GameState for an in-progress (or pre/post) scoreboard event.

    Returns the state plus the list of `situation` keys actually present, so a
    caller can log what ESPN really sent instead of trusting this module's
    assumptions about the live shape.
    """
    comp = event.get("_competition", {})
    sit = comp.get("situation") or {}
    situation_seen = sorted(sit.keys())

    home_id = event.get("home", {}).get("id")
    possession = sit.get("possession")
    home_has_ball = None
    if possession is not None and home_id is not None:
        home_has_ball = str(possession) == str(home_id)

    # ESPN reports field position two ways depending on endpoint. Prefer an
    # explicit yards-to-endzone if present; otherwise derive it.
    ytg = sit.get("yardsToEndzone")
    if ytg is None and sit.get("yardLine") is not None:
        ytg = 100 - int(sit["yardLine"])

    state = GameState(
        home_score=event.get("home", {}).get("score", 0),
        away_score=event.get("away", {}).get("score", 0),
        period=int(event.get("period") or 0),
        clock_seconds=_clock_to_seconds(event.get("clock")),
        down=sit.get("down"),
        distance=sit.get("distance"),
        yards_to_endzone=int(ytg) if ytg is not None else None,
        home_has_ball=home_has_ball,
        state=event.get("state") or "pre",
    )
    return state, situation_seen


def replay_states(summary: dict) -> Iterator[tuple[GameState, dict]]:
    """Yield (GameState, play) for every play of a finished game.

    This is the offline test harness: a real game becomes a sequence of real
    mid-game states with the true outcome known.
    """
    header = summary.get("header", {})
    comps = (header.get("competitions") or [{}])[0]
    home_id = away_id = None
    for c in comps.get("competitors", []):
        if c.get("homeAway") == "home":
            home_id = str(c.get("id") or c.get("team", {}).get("id"))
        else:
            away_id = str(c.get("id") or c.get("team", {}).get("id"))

    for drive in (summary.get("drives", {}) or {}).get("previous", []):
        for play in drive.get("plays", []):
            start = play.get("start") or {}
            ytg = start.get("yardsToEndzone")
            pos_team = str((start.get("team") or {}).get("id") or "")
            home_has_ball = None
            if pos_team and home_id:
                home_has_ball = pos_team == home_id
            gs = GameState(
                home_score=int(play.get("homeScore") or 0),
                away_score=int(play.get("awayScore") or 0),
                period=int((play.get("period") or {}).get("number") or 0),
                clock_seconds=_clock_to_seconds(
                    (play.get("clock") or {}).get("displayValue")),
                down=start.get("down") or None,
                distance=start.get("distance"),
                yards_to_endzone=int(ytg) if ytg is not None else None,
                home_has_ball=home_has_ball,
                state="in",
            )
            yield gs, play


def espn_win_probability(summary: dict) -> list[float]:
    """ESPN's own in-game home win probability, one point per play.

    Used ONLY as an external reference curve to sanity-check our model. It is
    not an input to anything.
    """
    return [p.get("homeWinPercentage") for p in summary.get("winprobability", [])
            if p.get("homeWinPercentage") is not None]


if __name__ == "__main__":
    sb = fetch_scoreboard()
    evs = list_events(sb)
    print(f"{len(evs)} events on today's scoreboard")
    for e in evs[:8]:
        gs, seen = parse_live_event(e)
        print(f"  {e['event_id']} {e['state']:4} {e['short_name']:18} "
              f"{gs.home_score}-{gs.away_score} P{gs.period} "
              f"situation={seen or '[]'}")
