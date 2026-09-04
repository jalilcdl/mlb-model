"""CFB sport adapter -- wraps the vendored cfb_lib (copied from cfb-model's
src/live/live_odds.py, wp_model.py) behind the common SportAdapter interface
(sports/base.py).

Odds: SharpAPI live moneylines (SHARPAPI_API_KEY env var; mock slate if unset).
State: CollegeFootballData.com's /scoreboard + /live/plays (CFBD_API_KEY env
var) -- NOT ESPN. ESPN's scoreboard 403s specifically from GitHub Actions'
runner IPs (confirmed by direct testing: identical request succeeds from
other networks, fails only from CI); CFBD doesn't share that anti-scraping
posture. See cfb_lib/live/cfbd_state.py for the detailed why/how.
Prior: always the market-implied margin (inverse of the pregame win-prob
curve). cfb-model's dashboard also has an optional "v5 model" prior sourced
from its full pregame feature/regression pipeline (predict_upcoming.py) --
deliberately NOT vendored here: it needs ~5MB of historical CFBD data,
scikit-learn, and ~20s per call, and cfb-model's own code already treats it
as a soft, optional enhancement that falls back to exactly this
market-implied prior whenever it's unavailable. Skipping it keeps this
poller light enough to run every few minutes on GitHub Actions' free tier.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from statistics import NormalDist

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from cfb_lib.live import cfbd_state
from cfb_lib.live.live_odds import get_provider
from cfb_lib.live.wp_model import win_probability

SPORT_KEY = "cfb"
SPORT_LABEL = "College Football"
SPORT_ICON = "🏈"

_THRESHOLD = float(os.environ.get("CFB_LIVE_SIGNAL_THRESHOLD", "0.05"))
_DEVIG_METHOD = os.environ.get("LIVE_DEVIG_METHOD", "proportional")


def _now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _name_match(sharp: str, cfbd_name: str, cfbd_abbr: str = "") -> bool:
    """Same fuzzy team-name matching cfb-model's live_signal.py used against
    ESPN -- SharpAPI and CFBD don't necessarily agree on team naming either
    (CFBD gives full school names only, no abbreviation, so cfbd_abbr is
    usually empty here; kept as a parameter in case that ever changes)."""
    a, b, ab = _norm(sharp), _norm(cfbd_name), _norm(cfbd_abbr)
    if not a or not b:
        return False
    return a == b or b.startswith(a) or a.startswith(b) or a == ab or (b.startswith(ab) and a.startswith(ab))


def _market_implied_margin(p_home: float) -> float:
    p = min(max(p_home, 1e-4), 1 - 1e-4)
    return NormalDist().inv_cdf(p) * 16.14


def poll() -> list[dict]:
    """Unified rows for every CFB game currently in progress. Empty if none."""
    try:
        live_games = cfbd_state.list_live_games()
    except Exception as e:
        print(f"[cfb] scoreboard fetch failed: {type(e).__name__}: {e}")
        return []
    if not live_games:
        return []

    provider = get_provider()
    try:
        odds = provider.list_moneylines(is_live=True)
    except Exception as e:
        print(f"[cfb] live odds fetch failed: {type(e).__name__}: {e}")
        odds = []

    rows = []
    for g in live_games:
        try:
            gs, meta = cfbd_state.build_game_state(g)
        except Exception as e:
            print(f"[cfb] live/plays fetch failed for game {g.get('id')}: {type(e).__name__}: {e}")
            continue
        home_name, away_name = meta["home_name"], meta["away_name"]
        cands = [o for o in odds
                 if _name_match(o.home_abbr, home_name)
                 and _name_match(o.away_abbr, away_name)]
        if not cands:
            continue
        pair = cands[0]

        mkt_home, _ = pair.implied(_DEVIG_METHOD)
        prior = _market_implied_margin(mkt_home)
        model_home = win_probability(gs, prior)
        gap = model_home - mkt_home
        flagged = abs(gap) >= _THRESHOLD
        pick_team = None
        if flagged:
            pick_team = home_name if gap > 0 else away_name

        state_desc = f"Q{gs.period} {gs.clock_seconds // 60}:{gs.clock_seconds % 60:02d}"
        if gs.down:
            state_desc += f", {gs.down}&{gs.distance}"

        rows.append({
            "mode": "OBSERVE_ONLY",
            "sport": SPORT_KEY,
            "logged_at_utc": _now_utc(),
            "game_id": str(g["id"]),
            "matchup": f"{away_name} @ {home_name}",
            "home_team": home_name, "away_team": away_name,
            "home_score": gs.home_score, "away_score": gs.away_score,
            "state_desc": state_desc,
            "model_home_wp": round(model_home, 4), "market_home_wp": round(mkt_home, 4),
            "edge_home": round(gap, 4), "edge": round(abs(gap), 4), "flagged": flagged,
            "pick_team": pick_team, "devig_method": _DEVIG_METHOD,
            "odds_source": pair.book or "sharpapi",
            # CFB-specific extras
            "period": gs.period, "clock": f"{gs.clock_seconds // 60}:{gs.clock_seconds % 60:02d}",
            "down": gs.down, "distance": gs.distance,
            "prior_source": "market-implied", "pregame_margin": round(prior, 2),
            "book": pair.book or "", "overround": round(pair.overround, 4),
        })
    return rows


def state_key(row: dict) -> tuple:
    """What counts as 'a new state worth logging' for this game."""
    return (row["game_id"], row["period"], row.get("clock"),
            row["home_score"], row["away_score"], row.get("down"), row.get("distance"))
