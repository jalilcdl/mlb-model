"""
Totals (over/under) odds from the connector's get_game_odds response.

WHY THIS EXISTS
The totals PDF used to source over/under lines from ESPN's public scoreboard,
which rate-limits (HTTP 403) unpredictably and only ever carried a single book
(DraftKings). The same connector already integrated for moneylines (get_events)
also serves full per-sportsbook totals via get_game_odds(offer_type="total"),
so this normalizes that richer feed into the `totals_items` shape that
play_chart.build_totals_rows already consumes -- making the connector the
PRIMARY totals source and ESPN merely a fallback.

HOW IT'S WIRED (important)
The connector is an MCP tool -- agent/orchestrator-callable, NOT importable from
Python -- exactly like get_events is for moneylines. So the data flow is:

    orchestrator/agent  ->  get_game_odds per game (MCP)
                        ->  connector_odds.totals_items_from_game_odds(responses)
                        ->  write_snapshot()  (odds/snapshots/connector_totals_{date}.json)
    run_totals(date)    ->  reads that snapshot  ->  build_totals_rows -> PDF

This module is pure data-shaping with no network dependency, so it is fully
unit-testable offline and never itself hits a rate limit.

REFERENCE BOOK
To stay consistent with the prior ESPN days (which de-vigged a single
DraftKings two-way at the consensus line), the reference two-way price is taken
from ONE book at the consensus line -- DraftKings by default, else the first
book that posts both sides at exactly that line. The connector carries ~20 books.

NOVIG GROUNDWORK
Novig is a peer-to-peer betting EXCHANGE, so its price is a peer-matched fair
line rather than a book's shaded line -- which is exactly why comparing it to
the rest of the market is interesting. Its two-way at the consensus line is
captured per game (novig_over_odds / novig_under_odds) so a later
"model vs Novig" / "Novig vs consensus" comparison can be built without
re-pulling. Nothing here renders it yet; it just carries the data through.
"""
import datetime as dt
import json

from src import config

# Preferred reference books, in order. DraftKings first to match the prior
# ESPN/DraftKings single-book methodology; the rest are just fallbacks so a game
# where DK hasn't posted the exact consensus line still gets a real two-way.
REFERENCE_BOOKS = ["DraftKings", "FanDuel", "Fanduel", "BetMGM", "Circa",
                   "BetRivers", "Fanatics", "Caesars"]

_SNAPSHOT_NAME = "connector_totals_{date}.json"


def _two_way_at_line(entries, line, tol=1e-9):
    """From one book's list of {side,line,odds}, return (over_odds, under_odds)
    at exactly `line`, or (None, None) if either side is missing."""
    over = under = None
    for e in entries or []:
        try:
            if abs(float(e.get("line")) - line) > tol:
                continue
        except (TypeError, ValueError):
            continue
        if e.get("side") == "over" and over is None:
            over = int(e["odds"])
        elif e.get("side") == "under" and under is None:
            under = int(e["odds"])
    return over, under


def total_item_from_game_odds(resp):
    """Convert one connector get_game_odds(offer_type='total') response into a
    single totals_item in the ESPN-compatible shape build_totals_rows consumes,
    plus Novig comparison fields. Returns None if no book posts a complete
    two-way at the consensus line (so the game is dropped, not faked)."""
    if isinstance(resp, list):
        resp = resp[0] if resp else {}
    if not isinstance(resp, dict):
        return None
    line = resp.get("consensus_line")
    books = resp.get("odds") or {}
    away, home = resp.get("away_team"), resp.get("home_team")
    if line is None or not books or not away or not home:
        return None
    line = float(line)

    # Reference two-way: preferred books first, then any remaining book.
    order = REFERENCE_BOOKS + [b for b in books if b not in REFERENCE_BOOKS]
    ref_book = ref_over = ref_under = None
    for b in order:
        if b not in books:
            continue
        o, u = _two_way_at_line(books[b], line)
        if o is not None and u is not None:
            ref_book, ref_over, ref_under = b, o, u
            break
    if ref_book is None:
        return None

    item = {
        "away_team": away,
        "home_team": home,
        "start_time_utc": resp.get("start_date"),
        "offer_type": "total",
        "consensus_line": line,
        "source": f"Connector ({ref_book})",
        "odds": {ref_book: [
            {"side": "over", "line": line, "odds": ref_over},
            {"side": "under", "line": line, "odds": ref_under},
        ]},
    }
    # Novig groundwork: capture its two-way at the consensus line if present.
    n_over, n_under = _two_way_at_line(books.get("Novig"), line)
    if n_over is not None and n_under is not None:
        item["novig_over_odds"] = n_over
        item["novig_under_odds"] = n_under
    return item


def totals_items_from_game_odds(responses):
    """Normalize a list of connector get_game_odds responses -> totals_items.
    Silently drops games with no complete two-way at the consensus line."""
    out = []
    for r in responses:
        item = total_item_from_game_odds(r)
        if item is not None:
            out.append(item)
    return out


def snapshot_path(date_str):
    return config.ODDS_DIR / "snapshots" / _SNAPSHOT_NAME.format(date=date_str)


def write_snapshot(date_str, items):
    """Persist normalized connector totals so run_totals can read them without
    calling the (MCP-only) connector itself. Returns the path written."""
    path = snapshot_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f, indent=2)
    return path


def load_snapshot(date_str):
    """Return the connector totals_items for a date, or None if no snapshot."""
    path = snapshot_path(date_str)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data if data else None
    except (json.JSONDecodeError, OSError):
        return None
