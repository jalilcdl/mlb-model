"""
Injured-list context for a club, from the MLB Stats API roster endpoint.

DISPLAY ONLY. This is deliberately NOT wired into any rating, grade, or
projection, and it should stay that way unless the validation problem below is
solved first.

Why it is not a model input: the API exposes only the CURRENT roster state.
There is no history -- you cannot ask what a club's injured list looked like on
some past date. Without that, an injury adjustment cannot be walk-forward
backtested, so shipping one would mean shipping an unvalidated weight that
silently moves projections. That is precisely how the starting-pitcher
adjustment ended up in the model unproven (the isolation backtest later found no
significant improvement in any market). One unvalidated adjustment is enough.

So: show the user who is hurt, let them apply judgement, and keep it out of the
math.
"""
import datetime as dt

import requests

from src.data import team_mapping

ROSTER_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
_UA = {"User-Agent": "mlb-model/1.0 (+local research tool)"}
_TIMEOUT = 25
_cache = {}

# Status codes that mean "unavailable due to injury". Deliberately excludes
# administrative absences (Reassigned to Minors, Restricted List, Not Yet
# Reported), which are not injuries and would inflate the count.
_INJURY_MARKERS = ("injured", "rehab", "disabled")


def fetch_injuries(team_code, season=None):
    """Injured players for one club. Returns a list of dicts:
    {name, position, status}. Empty list on any failure -- this is optional
    context and must never break a page."""
    season = season or dt.date.today().year
    key = (team_code, season, dt.date.today().isoformat())
    if key in _cache:
        return _cache[key]

    rec = team_mapping.TEAM_BY_CODE.get(team_code)
    if not rec:
        return []
    try:
        # 40Man, deliberately. 'fullRoster' returns the whole organization
        # (~272 players) and reports 38-48 "injured" per club, nearly all
        # minor-leaguers on full-season IL who are irrelevant to tonight's game
        # -- that would look authoritative and be badly misleading. 'active'
        # is the opposite error: it excludes injured players by definition, so
        # it always reports zero. The 40-man is the roster whose injuries
        # actually bear on a major-league game.
        r = requests.get(ROSTER_URL.format(team_id=rec["id"]),
                         params={"rosterType": "40Man", "season": season},
                         headers=_UA, timeout=_TIMEOUT)
        r.raise_for_status()
        roster = r.json().get("roster", [])
    except Exception:
        return []

    out = []
    for p in roster:
        status = (p.get("status") or {})
        desc = str(status.get("description") or "")
        if not any(m in desc.lower() for m in _INJURY_MARKERS):
            continue
        out.append({
            "name": (p.get("person") or {}).get("fullName", "?"),
            "position": (p.get("position") or {}).get("abbreviation", "?"),
            "status": desc,
        })
    out.sort(key=lambda x: (x["status"], x["name"]))
    _cache[key] = out
    return out


def summarize(injured):
    """One-line summary, e.g. '9 injured (4 P)'."""
    if not injured:
        return "No injured players listed"
    pitchers = sum(1 for i in injured if i["position"] == "P")
    return f"{len(injured)} injured" + (f" ({pitchers} P)" if pitchers else "")
