"""Live CFB moneyline odds. SharpAPI when a key is present, mock otherwise.

Contract taken from the verified MLB integration (see the mlb-live-ingame-signal
notes): base `https://api.sharpapi.io/api/v1`, header `X-API-Key`, endpoint
`GET /odds?league=<slug>&market=moneyline&is_live=true|false`. The response is a
FLAT list of one selection per row, so `pair_moneylines` regroups them by event
and pairs the two sides.

LEAGUE SLUG: the docs' Common Leagues table gives `ncaaf` for NCAA football, and
the docs also say `/leagues` is the authoritative list. That call needs a key, so
`ncaaf` is the default and `LEAGUE_CANDIDATES` holds the fallbacks to try if it
returns nothing. Confirm against `/leagues` once a key is available.

De-vigging defaults to the proportional method (the same
`remove_vig_two_way` the rest of this repo uses, so live and backtested numbers
stay comparable). If a response ever carries a provider-supplied no-vig price,
`devig="provider"` uses it instead.

OBSERVE-ONLY. This module reads prices. It contains no bet-placing code and
must never gain any.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

# Importing config loads .env via python-dotenv, so SHARPAPI_API_KEY is on
# os.environ even when this module is used standalone. The key lives ONLY
# in .env (gitignored) and is never written to source or logged.
import cfb_lib.config  # noqa: F401
from cfb_lib.backtest.stats import implied_prob_from_american, remove_vig_two_way

BASE_URL = "https://api.sharpapi.io/api/v1"
# CONFIRMED against a live /leagues call (1,227 leagues): the CFB slug is
# "ncaaf". Neighbours like ncaa_fcs / ncaaf_fcs / ncaa_men_football exist and
# are NOT the FBS feed.
LEAGUE = "ncaaf"
LEAGUE_CANDIDATES = ["ncaaf", "cfb", "college-football", "ncaa_football"]

# Books actually seen on the ncaaf feed. "novig" is a real book here and is a
# genuine exchange: its two-way prices sum to ~1.004 against DraftKings' ~1.036
# on the same game, i.e. it posts almost no vig by design. That matters for
# de-vigging -- see DEVIG note in MoneylinePair.implied.
BOOK_PREFERENCE = ["draftkings", "fanduel", "caesars", "novig"]


@dataclass(frozen=True)
class MoneylinePair:
    event_id: str
    home_abbr: str
    away_abbr: str
    home_american: int
    away_american: int
    book: str | None = None
    is_live: bool = False
    provider_home_novig: float | None = None
    provider_away_novig: float | None = None

    def implied(self, devig: str = "proportional") -> tuple[float, float]:
        """(home, away) fair probabilities.

        DEVIG NOTE: SharpAPI's `odds_probability` is the RAW vigged implied
        probability (the two sides sum to ~1.04, not 1.0), so it is not a
        provider no-vig figure and `devig="provider"` falls back to
        proportional unless a real no-vig field ever appears. Proportional
        de-vig on a `novig` book row is close to a no-op, which is correct --
        that book already posts ~1.004 -- rather than a double-removal.
        """
        if devig == "provider" and self.provider_home_novig is not None:
            return self.provider_home_novig, self.provider_away_novig
        return remove_vig_two_way(
            implied_prob_from_american(self.home_american),
            implied_prob_from_american(self.away_american))

    @property
    def overround(self) -> float:
        return (implied_prob_from_american(self.home_american)
                + implied_prob_from_american(self.away_american))


class OddsProvider:
    name = "base"

    def list_moneylines(self, is_live: bool = True) -> list[MoneylinePair]:
        raise NotImplementedError


class MockOddsProvider(OddsProvider):
    """Deterministic offline slate so the pipeline is runnable without a key."""
    name = "mock"

    def __init__(self, pairs: list[MoneylinePair] | None = None):
        self._pairs = pairs if pairs is not None else [
            MoneylinePair("mock-1", "USC", "SJSU", -2500, 1200, "MockBook", True),
            MoneylinePair("mock-2", "TCU", "UNC", -235, 195, "MockBook", True),
            MoneylinePair("mock-3", "UVA", "NCSU", -145, 125, "MockBook", True),
        ]

    def list_moneylines(self, is_live: bool = True) -> list[MoneylinePair]:
        return [p for p in self._pairs if p.is_live == is_live] or self._pairs


class SharpAPIProvider(OddsProvider):
    name = "sharpapi"

    def __init__(self, api_key: str, league: str = LEAGUE):
        self.api_key = api_key
        self.league = league

    def _get(self, path: str, params: dict) -> dict:
        r = requests.get(f"{BASE_URL}{path}",
                         headers={"X-API-Key": self.api_key},
                         params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def available_leagues(self) -> list[str]:
        """Authoritative slug list -- use this to confirm the CFB identifier."""
        data = self._get("/leagues", {})
        rows = data.get("data", data if isinstance(data, list) else [])
        out = []
        for row in rows:
            slug = row.get("id") or row.get("slug") or row.get("league")
            if slug:
                out.append(str(slug))
        return out

    def list_moneylines(self, is_live: bool = True) -> list[MoneylinePair]:
        data = self._get("/odds", {"league": self.league, "market": "moneyline",
                                   "is_live": str(is_live).lower()})
        return pair_moneylines(data.get("data", []))


def pair_moneylines(rows: list[dict]) -> list[MoneylinePair]:
    """Regroup SharpAPI's flat selection rows into two-sided pairs.

    One row per selection, so an event only becomes usable once both its home
    and away rows are present. Books are preferred in BOOK_PREFERENCE order;
    an event with only one side is dropped rather than half-priced.
    """
    by_event: dict[str, dict] = {}
    for r in rows:
        ev = str(r.get("event_id") or r.get("eventId") or "")
        if not ev:
            continue
        # The feed carries selection_type "other" rows (6 of 50 observed);
        # only the two real sides can form a two-way price.
        side = r.get("selection_type") or r.get("side")
        if side not in ("home", "away"):
            continue
        book = r.get("sportsbook") or r.get("book") or r.get("bookmaker")
        slot = by_event.setdefault(ev, {})
        slot.setdefault(book, {})[side] = r

    out = []
    for ev, books in by_event.items():
        chosen = None
        for pref in BOOK_PREFERENCE:
            for book, sides in books.items():
                if book and pref.lower() in str(book).lower() and len(sides) >= 2:
                    chosen = (book, sides)
                    break
            if chosen:
                break
        if chosen is None:
            for book, sides in books.items():
                if len(sides) >= 2:
                    chosen = (book, sides)
                    break
        if chosen is None:
            continue
        book, sides = chosen
        home, away = sides.get("home"), sides.get("away")
        if not home or not away:
            continue
        # The ncaaf feed identifies teams by FULL NAME (home_team/away_team),
        # not by abbreviation the way the MLB feed does. Verified live.
        out.append(MoneylinePair(
            event_id=ev,
            home_abbr=str(home.get("home_team") or ""),
            away_abbr=str(away.get("away_team") or ""),
            home_american=int(home.get("odds_american") or home.get("price")),
            away_american=int(away.get("odds_american") or away.get("price")),
            book=book,
            is_live=bool(home.get("is_live")),
            provider_home_novig=home.get("no_vig_probability"),
            provider_away_novig=away.get("no_vig_probability"),
        ))
    return out


def get_provider(league: str = LEAGUE) -> OddsProvider:
    """SharpAPI if SHARPAPI_API_KEY is set, else the mock slate."""
    key = os.environ.get("SHARPAPI_API_KEY")
    return SharpAPIProvider(key, league) if key else MockOddsProvider()
