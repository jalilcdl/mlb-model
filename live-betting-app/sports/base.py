"""Common interface every sport adapter implements.

Adding a new sport (e.g. NFL) means writing one module with:
  SPORT_KEY, SPORT_LABEL, SPORT_ICON
  poll() -> list[dict]        unified-schema rows for every currently-live game
  state_key(row) -> tuple     identifies "this exact game state", for dedup

...and registering it in sports/registry.py. Nothing else in this app --
poller, storage, dashboard, Telegram formatting -- needs to change.

Unified row schema (superset; sport-specific extra keys are fine and preserved,
the dashboard just renders what it finds):
    mode: "OBSERVE_ONLY"            (always -- see core/storage.py: enforced there too)
    sport: "mlb" | "cfb" | ...
    logged_at_utc: ISO8601 str
    game_id: str                    stable id for the game (game_pk / event_id)
    matchup: "AWAY @ HOME"
    home_team / away_team: str
    home_score / away_score: int
    state_desc: str                 human-readable state, e.g. "T5 2out" or "Q3 08:41 3rd&7"
    model_home_wp / market_home_wp: float
    edge_home: float                signed, home POV
    edge: float                     magnitude
    flagged: bool
    pick_team: str | None
    devig_method: str
    odds_source: str
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class SportAdapter(Protocol):
    SPORT_KEY: str
    SPORT_LABEL: str
    SPORT_ICON: str

    def poll(self) -> list[dict]: ...
    def state_key(self, row: dict) -> tuple: ...


REQUIRED_ROW_KEYS = (
    "mode", "sport", "logged_at_utc", "game_id", "matchup",
    "home_team", "away_team", "home_score", "away_score", "state_desc",
    "model_home_wp", "market_home_wp", "edge_home", "edge", "flagged",
    "pick_team", "devig_method", "odds_source",
)


def validate_row(row: dict) -> None:
    missing = [k for k in REQUIRED_ROW_KEYS if k not in row]
    if missing:
        raise ValueError(f"signal row missing required keys: {missing}")
    if row["mode"] != "OBSERVE_ONLY":
        raise ValueError("signal row must be mode=OBSERVE_ONLY -- refusing to log/notify")
