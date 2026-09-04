"""
Live MLB odds fetching (SharpAPI, sharpapi.io).

Wired to SharpAPI's real REST contract (verified against docs.sharpapi.io):
  * Base URL   : https://api.sharpapi.io/api/v1   (config.SHARPAPI_BASE_URL)
  * Auth       : header  X-API-Key: sk_live_...   (config.SHARPAPI_API_KEY, from
                 the SHARPAPI_API_KEY env var -- never hardcoded or logged)
  * Endpoint   : GET /odds?league=mlb&market=moneyline[&is_live=true]
  * Free tier  : 12 req/min, sportsbooks DraftKings + FanDuel only.

RESPONSE SHAPE. /odds returns a FLAT `data[]` where each row is ONE selection for
ONE sportsbook: {event_id, sportsbook, home_team, away_team, market_type,
selection_type: 'home'|'away', odds_american, is_live, home:{abbreviation}, ...}.
So a game's two-way moneyline is two rows (home + away) per book; we group by
event_id and pair them, preferring DraftKings then FanDuel.

Design:
  * `LiveOddsProvider` interface: `.moneyline(game_id)` -> `LiveMoneyline`.
  * `MockOddsProvider` -- canned prices, no key/network, for offline testing.
  * `SharpAPIProvider` -- the real one, gated on a key. It imports `requests`
    lazily and raises a clear error if the key is missing; it never fabricates
    prices, so a mock number can't be mistaken for a live one.
  * `SharpAPIProvider.list_moneylines(...)` -- all MLB games at once (the useful
    call: one request returns the whole slate to scan for signals).

`get_provider()` returns SharpAPI when a key is configured, else the mock.
"""
from __future__ import annotations

from dataclasses import dataclass

from mlb_lib import config
from mlb_lib.odds.odds_adapter import _normalize_team


@dataclass
class LiveMoneyline:
    """A live moneyline two-way for one game. American odds. `source` records
    where it came from (e.g. 'sharpapi:draftkings') so a mock price is never
    confused for a real one. `is_live` marks in-play vs prematch.

    home_fair/away_fair carry a book- or provider-supplied NO-VIG probability when
    the feed provides one (SharpAPI's Pinnacle-referenced no-vig is a paid-tier
    field, so these are None on the free tier -> the signal layer falls back to its
    own de-vig). home_implied/away_implied are the raw vig-included implied probs
    if the feed reports them (SharpAPI `odds_probability`)."""
    game_id: str
    home_team: str
    away_team: str
    home_odds: int
    away_odds: int
    source: str
    last_update: str | None = None
    is_live: bool = False
    home_implied: float | None = None
    away_implied: float | None = None
    home_fair: float | None = None
    away_fair: float | None = None


@dataclass
class LiveSpread:
    """Live run-line two-way for one game. `home_line`/`away_line` are each
    side's OWN signed line (e.g. home -1.5 favorite, away +1.5 -- SharpAPI
    gives each selection its own already-signed number, verified directly).
    A team covers its line when its own margin exceeds -line."""
    game_id: str
    home_team: str
    away_team: str
    home_line: float
    away_line: float
    home_odds: int
    away_odds: int
    source: str
    last_update: str | None = None
    is_live: bool = False


@dataclass
class LiveTotal:
    """Live game-total two-way for one game. One `line` shared by both sides
    (verified directly: over/under main-line rows post the same number)."""
    game_id: str
    home_team: str
    away_team: str
    line: float
    over_odds: int
    under_odds: int
    source: str
    last_update: str | None = None
    is_live: bool = False


class LiveOddsProvider:
    """Interface. Implementations return a LiveMoneyline for a game id."""

    name = "base"

    def moneyline(self, game_id: str) -> LiveMoneyline:
        raise NotImplementedError


class MockOddsProvider(LiveOddsProvider):
    """Serves canned two-way prices for offline testing. Seed it with a dict of
    {game_id: LiveMoneyline} or via .add(); unknown ids get a neutral -110/-110
    so the pipeline always returns something to compare against."""

    name = "mock"

    def __init__(self, book: dict | None = None):
        self._book = book or {}

    def add(self, game_id, home_team, away_team, home_odds, away_odds):
        self._book[game_id] = LiveMoneyline(
            game_id, home_team, away_team, home_odds, away_odds,
            source="mock", last_update="mock")
        return self

    def moneyline(self, game_id: str) -> LiveMoneyline:
        if game_id in self._book:
            return self._book[game_id]
        return LiveMoneyline(game_id, "HOME", "AWAY", -110, -110,
                             source="mock", last_update="mock")


# Free-tier books, in preference order (DraftKings first to match the pregame
# charts' single-book de-vig convention; FanDuel as the fallback).
FREE_TIER_BOOKS = ("draftkings", "fanduel")


class SharpAPIProvider(LiveOddsProvider):
    """Live MLB odds from SharpAPI's /odds endpoint."""

    name = "sharpapi"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 10.0):
        self.api_key = api_key or config.SHARPAPI_API_KEY
        self.base_url = (base_url or config.SHARPAPI_BASE_URL).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                "No SharpAPI key. Sign up at https://sharpapi.io (free tier, no "
                "card), then set the SHARPAPI_API_KEY environment variable (or "
                "config.SHARPAPI_API_KEY). Until then use MockOddsProvider.")

    def _get_odds(self, **params):
        """GET /odds with the given filters. Returns (rows, pagination). Never
        includes the API key in any raised message or log."""
        import requests  # lazy: prototype has no hard requests dependency

        q = {k: v for k, v in params.items() if v is not None}
        if "is_live" in q and isinstance(q["is_live"], bool):
            q["is_live"] = "true" if q["is_live"] else "false"
        resp = requests.get(
            f"{self.base_url}/odds", params=q,
            headers={"X-API-Key": self.api_key, "Accept": "application/json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("data", []), payload.get("pagination", {})

    def list_moneylines(self, is_live=None, league="mlb", sport=None,
                        sportsbook_priority=FREE_TIER_BOOKS, limit=200,
                        event_id=None) -> dict:
        """{event_id: LiveMoneyline} for MLB moneylines. is_live=True for in-play
        only, False for prematch only, None for both. Pass league='mlb' (default)
        or, as a fallback, league=None with sport='baseball'."""
        rows, _pag = self._get_odds(
            league=league, sport=sport, market="moneyline", is_live=is_live,
            limit=limit, event_id=event_id)
        return _pair_moneylines(rows, sportsbook_priority)

    def moneyline(self, game_id: str) -> LiveMoneyline:
        book = self.list_moneylines(event_id=game_id)
        if game_id not in book:
            raise LookupError(f"No moneyline two-way returned for event {game_id}")
        return book[game_id]

    def list_spreads(self, is_live=None, league="mlb", sport=None,
                     sportsbook_priority=FREE_TIER_BOOKS, limit=200,
                     event_id=None) -> dict:
        """{event_id: LiveSpread}. MLB's run-line market is queried as
        market="spread" but comes back tagged market_type "run_line" --
        verified directly against the live API, not assumed."""
        rows, _pag = self._get_odds(
            league=league, sport=sport, market="spread", is_live=is_live,
            limit=limit, event_id=event_id)
        return _pair_spreads(rows, sportsbook_priority)

    def list_totals(self, is_live=None, league="mlb", sport=None,
                    sportsbook_priority=FREE_TIER_BOOKS, limit=200,
                    event_id=None) -> dict:
        """{event_id: LiveTotal}. The full-game total is market="total_runs" --
        market="total" silently returns an unrelated 1st-3-innings team-total
        prop instead; verified directly, not assumed."""
        rows, _pag = self._get_odds(
            league=league, sport=sport, market="total_runs", is_live=is_live,
            limit=limit, event_id=event_id)
        return _pair_totals(rows, sportsbook_priority)


# Candidate keys a SharpAPI (or other) odds row might use for a NO-VIG / fair
# probability. None appear on the free tier today; listed so the signal layer can
# use a real Pinnacle-referenced no-vig automatically if a paid tier provides one.
_FAIR_PROB_KEYS = ("no_vig_probability", "novig_probability", "fair_probability",
                   "devig_probability", "probability_no_vig", "fair_prob")


def _extract_probs(row):
    """(implied, fair) probabilities from one selection row, or (None, None).
    `implied` = raw vig-included (SharpAPI `odds_probability`); `fair` = provider
    no-vig if any recognized key is present."""
    implied = row.get("odds_probability")
    fair = None
    for k in _FAIR_PROB_KEYS:
        if row.get(k) is not None:
            fair = row[k]
            break
    try:
        implied = float(implied) if implied is not None else None
    except (TypeError, ValueError):
        implied = None
    try:
        fair = float(fair) if fair is not None else None
    except (TypeError, ValueError):
        fair = None
    return implied, fair


def _norm(code):
    """Best-effort map a SharpAPI abbreviation to our canonical team code, so the
    win-prob prior can look the team up in the run ratings; fall back to the raw
    upper-cased abbreviation (run_rates then degrades that team to league average)."""
    if not code:
        return code
    return _normalize_team(code) or str(code).upper()


def _pair_moneylines(rows, sportsbook_priority=FREE_TIER_BOOKS) -> dict:
    """Group flat /odds selection rows by event and pair home+away into a single
    LiveMoneyline, choosing the first sportsbook (by priority) that posts both
    sides. Games without a complete two-way at any book are dropped, not faked."""
    events = {}
    for r in rows:
        if r.get("market_type") != "moneyline":
            continue
        events.setdefault(r["event_id"], []).append(r)

    out = {}
    for eid, ers in events.items():
        by_book = {}
        for r in ers:
            by_book.setdefault(r.get("sportsbook"), {})[r.get("selection_type")] = r
        order = list(sportsbook_priority) + [b for b in by_book
                                             if b not in sportsbook_priority]
        chosen = next((b for b in order
                       if "home" in by_book.get(b, {}) and "away" in by_book.get(b, {})),
                      None)
        if chosen is None:
            continue
        home_row, away_row = by_book[chosen]["home"], by_book[chosen]["away"]
        ref = ers[0]
        home_abbr = (ref.get("home") or {}).get("abbreviation") or ref.get("home_team")
        away_abbr = (ref.get("away") or {}).get("abbreviation") or ref.get("away_team")
        h_impl, h_fair = _extract_probs(home_row)
        a_impl, a_fair = _extract_probs(away_row)
        out[eid] = LiveMoneyline(
            game_id=eid,
            home_team=_norm(home_abbr),
            away_team=_norm(away_abbr),
            home_odds=int(home_row["odds_american"]),
            away_odds=int(away_row["odds_american"]),
            source=f"sharpapi:{chosen}",
            last_update=ref.get("timestamp"),
            is_live=bool(ref.get("is_live")),
            home_implied=h_impl, away_implied=a_impl,
            home_fair=h_fair, away_fair=a_fair,
        )
    return out


def _pair_spreads(rows, sportsbook_priority=FREE_TIER_BOOKS) -> dict:
    """Same event/book grouping as _pair_moneylines, but spread/total markets
    carry many ALTERNATE lines per event (a full ladder, e.g. every half-run
    total from 4.5 to 12.5 was observed on one real game) -- only the row(s)
    flagged is_main_line=True are the actual current market number; every
    other row is a different, non-current line and must be excluded, not
    averaged or picked arbitrarily."""
    events = {}
    for r in rows:
        if r.get("market_type") != "run_line" or not r.get("is_main_line"):
            continue
        events.setdefault(r["event_id"], []).append(r)

    out = {}
    for eid, ers in events.items():
        by_book = {}
        for r in ers:
            by_book.setdefault(r.get("sportsbook"), {})[r.get("selection_type")] = r
        order = list(sportsbook_priority) + [b for b in by_book
                                             if b not in sportsbook_priority]
        chosen = next((b for b in order
                       if "home" in by_book.get(b, {}) and "away" in by_book.get(b, {})),
                      None)
        if chosen is None:
            continue
        home_row, away_row = by_book[chosen]["home"], by_book[chosen]["away"]
        ref = ers[0]
        home_abbr = (ref.get("home") or {}).get("abbreviation") or ref.get("home_team")
        away_abbr = (ref.get("away") or {}).get("abbreviation") or ref.get("away_team")
        out[eid] = LiveSpread(
            game_id=eid, home_team=_norm(home_abbr), away_team=_norm(away_abbr),
            home_line=float(home_row["line"]), away_line=float(away_row["line"]),
            home_odds=int(home_row["odds_american"]), away_odds=int(away_row["odds_american"]),
            source=f"sharpapi:{chosen}", last_update=ref.get("timestamp"),
            is_live=bool(ref.get("is_live")),
        )
    return out


def _pair_totals(rows, sportsbook_priority=FREE_TIER_BOOKS) -> dict:
    """Same is_main_line filtering as _pair_spreads (see its docstring) --
    verified directly: over/under main-line rows post the SAME line number,
    so pairing them is safe once alternates are excluded."""
    events = {}
    for r in rows:
        if r.get("market_type") != "total_runs" or not r.get("is_main_line"):
            continue
        events.setdefault(r["event_id"], []).append(r)

    out = {}
    for eid, ers in events.items():
        by_book = {}
        for r in ers:
            by_book.setdefault(r.get("sportsbook"), {})[r.get("selection_type")] = r
        order = list(sportsbook_priority) + [b for b in by_book
                                             if b not in sportsbook_priority]
        chosen = next((b for b in order
                       if "over" in by_book.get(b, {}) and "under" in by_book.get(b, {})),
                      None)
        if chosen is None:
            continue
        over_row, under_row = by_book[chosen]["over"], by_book[chosen]["under"]
        ref = ers[0]
        home_abbr = (ref.get("home") or {}).get("abbreviation") or ref.get("home_team")
        away_abbr = (ref.get("away") or {}).get("abbreviation") or ref.get("away_team")
        out[eid] = LiveTotal(
            game_id=eid, home_team=_norm(home_abbr), away_team=_norm(away_abbr),
            line=float(over_row["line"]),
            over_odds=int(over_row["odds_american"]), under_odds=int(under_row["odds_american"]),
            source=f"sharpapi:{chosen}", last_update=ref.get("timestamp"),
            is_live=bool(ref.get("is_live")),
        )
    return out


def get_provider(prefer_live: bool = True) -> LiveOddsProvider:
    """Return the live provider when a SharpAPI key is configured, else the mock.
    Set prefer_live=False to force the mock even when a key exists (e.g. tests)."""
    if prefer_live and config.SHARPAPI_API_KEY:
        return SharpAPIProvider()
    return MockOddsProvider()
