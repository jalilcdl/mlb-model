"""
Thin client for the free, public MLB Stats API (statsapi.mlb.com).

No API key required. Used for two things this project needs that
Baseball-Reference doesn't give us cleanly: (1) today's/upcoming schedule
with probable pitchers, and (2) a given pitcher's current-season stat line
(used for the starting-pitcher adjustment on live predictions).
"""
import time

import requests

from src.data import team_mapping

BASE_URL = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 15
_RETRIES = 3            # total attempts per request
_BACKOFF = 0.6         # seconds; grows 0.6, 1.2, 2.4 ...
_UA = "mlb-model/1.0 (+local research tool)"


def _get(path, params=None):
    """GET one endpoint with retry + backoff on transient failures.

    predict_date fires ~2N+1 of these per slate (schedule + two pitcher lines
    per game), so a single transient blip on any one call must not take down
    the whole prediction. Connection errors, timeouts, and 5xx responses are
    retried; a genuine 4xx (bad id, etc.) raises immediately since retrying
    won't help. After the last attempt the underlying error propagates so
    callers can decide whether it's fatal (schedule) or skippable (a pitcher).
    """
    last_exc = None
    for attempt in range(_RETRIES):
        try:
            resp = requests.get(
                f"{BASE_URL}{path}", params=params or {},
                timeout=_TIMEOUT, headers={"User-Agent": _UA},
            )
            if 500 <= resp.status_code < 600:
                resp.raise_for_status()  # retry server-side errors
            resp.raise_for_status()      # 4xx -> raise now (not retried below)
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500:
                raise  # client error -- retrying is pointless
            last_exc = exc
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
        if attempt < _RETRIES - 1:
            time.sleep(_BACKOFF * (2 ** attempt))
    raise last_exc


def _parse_innings_pitched(ip_str):
    """MLB box-score innings notation: '54.1' = 54 + 1/3, '54.2' = 54 + 2/3."""
    if ip_str is None:
        return 0.0
    whole, _, frac = str(ip_str).partition(".")
    whole = float(whole) if whole not in ("", "-") else 0.0
    frac_map = {"": 0.0, "0": 0.0, "1": 1 / 3, "2": 2 / 3}
    return whole + frac_map.get(frac, 0.0)


def get_schedule(date_str):
    """Games (any status) for a single date (YYYY-MM-DD), with probable pitchers."""
    data = _get(
        "/schedule",
        {"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team,venue"},
    )
    games = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            teams = g["teams"]
            home, away = teams["home"], teams["away"]
            try:
                home_code = team_mapping.code_from_statsapi_id(home["team"]["id"])
                away_code = team_mapping.code_from_statsapi_id(away["team"]["id"])
            except KeyError:
                # Spring training / all-star / non-MLB-club games — skip.
                continue
            games.append(
                {
                    "game_pk": g["gamePk"],
                    "game_date": g.get("officialDate", date_str),
                    "game_datetime_utc": g.get("gameDate"),
                    "status": g.get("status", {}).get("detailedState"),
                    "game_type": g.get("gameType"),
                    "home_team": home_code,
                    "away_team": away_code,
                    "home_team_name": home["team"]["name"],
                    "away_team_name": away["team"]["name"],
                    "venue_name": g.get("venue", {}).get("name"),
                    "home_probable_pitcher_id": (home.get("probablePitcher") or {}).get("id"),
                    "home_probable_pitcher_name": (home.get("probablePitcher") or {}).get("fullName"),
                    "away_probable_pitcher_id": (away.get("probablePitcher") or {}).get("id"),
                    "away_probable_pitcher_name": (away.get("probablePitcher") or {}).get("fullName"),
                }
            )
    return games


def get_pitcher_season_stats(pitcher_id, season):
    """Current-season pitching line for one pitcher, or None if they haven't pitched."""
    if pitcher_id is None:
        return None
    data = _get(
        f"/people/{pitcher_id}/stats",
        {"stats": "season", "group": "pitching", "season": season},
    )
    stats_blocks = data.get("stats", [])
    if not stats_blocks or not stats_blocks[0].get("splits"):
        return None
    s = stats_blocks[0]["splits"][0]["stat"]
    ip = _parse_innings_pitched(s.get("inningsPitched"))
    if ip <= 0:
        return None
    return {
        "innings_pitched": ip,
        "games_started": s.get("gamesStarted", 0),
        "era": float(s.get("era", "0") or 0),
        "whip": float(s.get("whip", "0") or 0),
        "strikeouts": s.get("strikeOuts", 0),
        "walks": s.get("baseOnBalls", 0),
        "hit_by_pitch": s.get("hitBatsmen", 0),
        "home_runs": s.get("homeRuns", 0),
        "earned_runs": s.get("earnedRuns", 0),
    }


def get_pitcher_game_log(pitcher_id, season):
    """Per-game, date-stamped pitching lines for one pitcher-season. Used to
    reconstruct as-of-date stats for any cutoff date (summing only games
    strictly before it) without lookahead -- see
    src/backtest/pitcher_backtest.py, which isolates whether the starting
    pitcher adjustment used in live predictions actually helps."""
    data = _get(
        f"/people/{pitcher_id}/stats",
        {"stats": "gameLog", "group": "pitching", "season": season},
    )
    stats_blocks = data.get("stats", [])
    if not stats_blocks:
        return []
    rows = []
    for s in stats_blocks[0].get("splits", []):
        stat = s.get("stat", {})
        ip = _parse_innings_pitched(stat.get("inningsPitched"))
        if ip <= 0:
            continue
        rows.append(
            {
                "date": s.get("date"),
                "ip": ip,
                "er": stat.get("earnedRuns", 0),
                "hr": stat.get("homeRuns", 0),
                "bb": stat.get("baseOnBalls", 0),
                "hbp": stat.get("hitBatsmen", 0),
                "so": stat.get("strikeOuts", 0),
            }
        )
    return rows


def fip_component(stats):
    """Unscaled FIP numerator/IP: (13*HR + 3*(BB+HBP) - 2*K) / IP.

    Deliberately omits the additive league constant — callers compare this
    value against the league-average of the same formula, so the constant
    cancels out. See run_model.py.
    """
    if not stats or stats["innings_pitched"] <= 0:
        return None
    hr, bb, hbp, k, ip = (
        stats["home_runs"],
        stats["walks"],
        stats["hit_by_pitch"],
        stats["strikeouts"],
        stats["innings_pitched"],
    )
    return (13 * hr + 3 * (bb + hbp) - 2 * k) / ip
