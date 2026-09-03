"""Unified log + notification-dedup state, shared by every sport.

Both files are plain JSON/JSONL committed back to this repo by the GitHub
Actions poller after each run (see .github/workflows/poll.yml) -- that's the
whole "database": no external service, and it's what lets the free Streamlit
Community Cloud dashboard see fresh data (a git push to the repo triggers an
auto-redeploy, but ONLY happens when something actually changed, i.e. during
live games -- not on every 5-minute cron tick).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_PATH = DATA_DIR / "live_signal_log.jsonl"
NOTIFIED_STATE_PATH = DATA_DIR / "notified_state.json"

# Re-notify about a still-flagged game (same side) after this long, so a
# persistent signal isn't announced exactly once and then never again --
# but never on every single ~5min poll cycle either.
RENOTIFY_COOLDOWN_SECONDS = 30 * 60


def load_log_tail_by_game(sport: str) -> dict[str, dict]:
    """{game_id: last logged row} for one sport, from the existing log.
    Used to detect whether a fresh poll represents a NEW state worth logging."""
    if not LOG_PATH.exists():
        return {}
    last_by_game: dict[str, dict] = {}
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("sport") == sport:
                last_by_game[row["game_id"]] = row
    return last_by_game


def append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_notified_state() -> dict:
    if not NOTIFIED_STATE_PATH.exists():
        return {}
    try:
        return json.loads(NOTIFIED_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_notified_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTIFIED_STATE_PATH.write_text(json.dumps(state, indent=2))


def should_notify(state: dict, sport: str, game_id: str, pick_team: str, now_ts: float) -> bool:
    """Dedup rule: notify on a NEW flagged game, on a flip to the other side, or
    after the cooldown has elapsed for a still-flagged same-side signal."""
    key = f"{sport}:{game_id}"
    prev = state.get(key)
    if prev is None:
        return True
    if prev.get("pick_team") != pick_team:
        return True
    return (now_ts - prev.get("last_notified_ts", 0)) >= RENOTIFY_COOLDOWN_SECONDS


def record_notified(state: dict, sport: str, game_id: str, pick_team: str, now_ts: float) -> None:
    key = f"{sport}:{game_id}"
    state[key] = {"pick_team": pick_team, "last_notified_ts": now_ts}
