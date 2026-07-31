"""
Highlightly free-tier live-odds source (https://highlightly.net).

Purpose (Problem A only): a reliable, free replacement for the manual
ESPN-scrape workaround for *daily live* odds, while the Optimal Bet/Tangiers
connector is down. Free "Basic" tier: 100 requests/day, no card, 70+ books,
moneyline / run line / totals, pre-game and live.

This does NOT solve Problem B (historical totals for the backtest). A live-odds
API gives today's numbers forward, not years of closing lines. What it *does*
enable is organically accumulating our own real totals history from today on
(see accumulate_totals), which becomes backtest-worthy only after many months.
See README "Totals vs. the real market".

--- API contract (from https://highlightly.net/documentation/baseball/) ---
  Base:   config.HIGHLIGHTLY_BASE_URL  (https://baseball.highlightly.net)
  Auth:   header  x-rapidapi-key: <key>
  Games:  GET /matches?league=MLB&date=YYYY-MM-DD&timezone=Etc/UTC
            -> data[].{id, date(ISO UTC), homeTeam.{name,displayName}, awayTeam.{...}}
  Odds:   GET /odds?leagueName=MLB&date=YYYY-MM-DD&oddsType=prematch
            -> data[].{matchId, odds[].{bookmakerName, market, values[].{odd, value}}}
          markets: "Home/Away" (Home/Away), "Over/Under 8.5" (Over/Under),
                   "Asian Handicap -1.5/+1.5" (run line); LINE IS IN THE market
                   STRING, and `odd` is DECIMAL (converted to American here).

The exact response shapes are taken from the published docs but have NOT been
run against a live key in this build. Everything version-specific is isolated
in the parse_* helpers and flagged, and `probe()` dumps raw JSON so the mapping
can be confirmed/fixed in one place the first time a real key is used.
"""
import datetime as dt
import json
import re
import time

import requests

from src import config
from src.data import team_mapping
from src.odds.odds_adapter import american_to_decimal, decimal_to_american

_TIMEOUT = 20
_RETRIES = 3
_BACKOFF = 0.6


class HighlightlyError(RuntimeError):
    pass


def _key():
    key = config.HIGHLIGHTLY_API_KEY
    if not key:
        raise HighlightlyError(
            "No Highlightly API key configured. Set the HIGHLIGHTLY_API_KEY "
            "environment variable (or config.HIGHLIGHTLY_API_KEY). Free key: "
            "https://highlightly.net"
        )
    return key


def _get(path, params):
    """GET with auth header + retry/backoff on transient failures (same
    resilience policy as the statsapi client)."""
    headers = {"x-rapidapi-key": _key()}
    last = None
    for attempt in range(_RETRIES):
        try:
            r = requests.get(f"{config.HIGHLIGHTLY_BASE_URL}{path}", params=params,
                             headers=headers, timeout=_TIMEOUT)
            if 500 <= r.status_code < 600:
                r.raise_for_status()
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status and 400 <= status < 500:
                raise HighlightlyError(f"Highlightly {status} on {path}: {exc}") from exc
            last = exc
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last = exc
        if attempt < _RETRIES - 1:
            time.sleep(_BACKOFF * (2 ** attempt))
    raise HighlightlyError(f"Highlightly request failed after {_RETRIES} attempts: {last}")


# --------------------------------------------------------------------------
# Raw fetches
# --------------------------------------------------------------------------
def fetch_matches(date_str):
    return _get("/matches", {"league": config.HIGHLIGHTLY_LEAGUE, "date": date_str, "timezone": "Etc/UTC"})


def fetch_odds(date_str, odds_type="prematch"):
    return _get("/odds", {"leagueName": config.HIGHLIGHTLY_LEAGUE, "date": date_str, "oddsType": odds_type})


# --------------------------------------------------------------------------
# Parsing (version-specific -- isolated so a docs/reality mismatch is a 1-spot fix)
# --------------------------------------------------------------------------
def _line_from_market(market):
    """Pull the numeric line out of a market label, e.g. 'Over/Under 8.5' -> 8.5,
    'Asian Handicap -1.5/+1.5' -> 1.5. Returns None if none present."""
    m = re.search(r"[-+]?\d+\.?\d*", str(market))
    return abs(float(m.group())) if m else None


def _match_index(matches_json):
    """{matchId: {'away','home','start_utc'}} from /matches, resolving team codes
    from Highlightly's free-text names. Unresolvable matches are skipped and
    reported by the caller rather than silently miskeyed."""
    index, unresolved = {}, []
    for g in (matches_json or {}).get("data", []):
        home = g.get("homeTeam", {}) or {}
        away = g.get("awayTeam", {}) or {}
        hc = team_mapping.code_from_name(home.get("displayName"), home.get("name"))
        ac = team_mapping.code_from_name(away.get("displayName"), away.get("name"))
        if not hc or not ac:
            unresolved.append((away.get("displayName") or away.get("name"),
                               home.get("displayName") or home.get("name")))
            continue
        index[g.get("id")] = {"away": ac, "home": hc, "start_utc": g.get("date")}
    return index, unresolved


def _collect(odds_for_match, market_predicate, want_sides):
    """For one match's odds list, build {bookmaker: [{side,line,odds}]} for the
    markets matching `market_predicate`. `want_sides` maps Highlightly value
    labels (e.g. 'Home','Over') to our side labels."""
    by_book = {}
    for entry in odds_for_match:
        market = entry.get("market", "")
        if not market_predicate(market):
            continue
        line = _line_from_market(market)
        book = entry.get("bookmakerName")
        for v in entry.get("values", []):
            side = want_sides.get(str(v.get("value")).strip().lower())
            if side is None or v.get("odd") in (None, 0):
                continue
            american = decimal_to_american(v["odd"])
            by_book.setdefault(book, []).append(
                {"side": side, "line": (0 if line is None else line), "odds": american}
            )
    return by_book


def normalize_slate(date_str, odds_type="prematch"):
    """Fetch matches + odds for a date and return snapshot items in the exact
    shape src/odds/odds_adapter.load_snapshots() consumes (moneyline / total /
    spread, odds keyed by book), plus a list of any unresolved team names."""
    matches = fetch_matches(date_str)
    odds = fetch_odds(date_str, odds_type=odds_type)
    idx, unresolved = _match_index(matches)

    items = []
    for m in (odds or {}).get("data", []):
        info = idx.get(m.get("matchId"))
        if not info:
            continue  # odds for a game we couldn't map to team codes
        base = {
            "event_id": f"highlightly-{m['matchId']}",
            "start_time_utc": info["start_utc"],
            "away_team": info["away"], "home_team": info["home"],
            "source": "Highlightly (free tier)",
        }
        book_odds = m.get("odds", [])

        ml = _collect(book_odds, lambda mk: mk.strip().lower() in ("home/away", "moneyline", "money line"),
                      {"home": "home", "away": "away"})
        if ml:
            items.append({**base, "offer_type": "moneyline", "odds": ml})

        totals = _collect(book_odds, lambda mk: mk.strip().lower().startswith("over/under"),
                          {"over": "over", "under": "under"})
        if totals:
            items.append({**base, "offer_type": "total", "consensus_line": _total_line(totals), "odds": totals})

        spread = _collect(book_odds, lambda mk: "handicap" in mk.strip().lower(),
                          {"home": "home", "away": "away"})
        if spread:
            home_line = _sign_run_line(spread, ml)  # -1.5 if home favoured, else +1.5
            for offers in spread.values():
                for o in offers:
                    o["line"] = home_line if o["side"] == "home" else -home_line
            items.append({**base, "offer_type": "spread", "consensus_line": home_line, "odds": spread})

    return items, unresolved


def _total_line(by_book):
    """Most common (consensus) total line across books."""
    lines = [o["line"] for offers in by_book.values() for o in offers if o.get("line")]
    return max(set(lines), key=lines.count) if lines else None


def _sign_run_line(spread_by_book, ml_by_book):
    """Return the HOME run-line number (-1.5 if home is the favourite, else
    +1.5). MLB run lines are always exactly +/-1.5; only the sign is in doubt
    because Highlightly's handicap label is unsigned. Infer the favourite from
    the moneyline (shorter price = favourite); if no moneyline is available,
    fall back to the run-line prices themselves (the -1.5 favourite pays the
    longer price). Defaults to home favourite if genuinely undeterminable."""
    def _avg(by_book, side):
        vals = [american_to_decimal(o["odds"]) for offers in by_book.values()
                for o in offers if o["side"] == side]
        return sum(vals) / len(vals) if vals else None

    home_ml, away_ml = _avg(ml_by_book, "home"), _avg(ml_by_book, "away")
    if home_ml is not None and away_ml is not None:
        return -config.RUN_LINE if home_ml <= away_ml else config.RUN_LINE
    # Fallback: on the run line the favourite (-1.5) is the longer price.
    home_sp, away_sp = _avg(spread_by_book, "home"), _avg(spread_by_book, "away")
    if home_sp is not None and away_sp is not None:
        return -config.RUN_LINE if home_sp >= away_sp else config.RUN_LINE
    return -config.RUN_LINE


# --------------------------------------------------------------------------
# Snapshot writing + historical accumulation
# --------------------------------------------------------------------------
def write_snapshot(date_str, items=None, folder=None):
    """Write a normalized slate to odds/snapshots/highlightly_<date>.json so the
    existing pipeline (de-vig, push-fix, doubleheader matching, edge finder)
    consumes it unchanged."""
    if items is None:
        items, _ = normalize_slate(date_str)
    folder = folder or (config.ODDS_DIR / "snapshots")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"highlightly_{date_str}.json"
    with open(path, "w") as f:
        json.dump(items, f, indent=2)
    return path, len(items)


def accumulate_totals(date_str, items=None, path=None):
    """Append each game's totals line to the running historical file
    (config.HISTORICAL_TOTALS_FILE) so we build our own real over/under history
    from today forward. Deduplicated on (date, away, home, line): re-pulling a
    slate won't create duplicate rows, and the last write wins if a line moved.

    Only the LINE is logged here; game RESULTS are supplied at backtest time by
    joining to games.csv (src/backtest/totals_backtest.py), so nothing has to
    be settled by hand. Requires no model run -- pure odds capture.
    """
    import csv
    if items is None:
        items, _ = normalize_slate(date_str)
    path = path or config.HISTORICAL_TOTALS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                existing[(row["date"], row["away_team"], row["home_team"], row["total_line"])] = row

    added = 0
    for it in items:
        if it.get("offer_type") != "total" or it.get("consensus_line") is None:
            continue
        line = it["consensus_line"]
        over = _best_price(it["odds"], "over")
        under = _best_price(it["odds"], "under")
        key = (date_str, it["away_team"], it["home_team"], f"{line:g}")
        existing[key] = {
            "date": date_str, "away_team": it["away_team"], "home_team": it["home_team"],
            "total_line": f"{line:g}", "over_odds": over, "under_odds": under,
            "source": it.get("source", "Highlightly"),
        }
        added += 1

    fields = ["date", "away_team", "home_team", "total_line", "over_odds", "under_odds", "source"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(existing.values(), key=lambda r: (r["date"], r["away_team"], r["home_team"])):
            w.writerow(row)
    return path, added


def _best_price(by_book, side):
    prices = [o["odds"] for offers in by_book.values() for o in offers if o["side"] == side]
    if not prices:
        return ""
    return max(prices, key=american_to_decimal)  # best (highest-paying) price


def refresh(date_str=None, odds_type="prematch"):
    """One call for daily use: fetch, write the snapshot for edge-finding, and
    log totals to the historical accumulator. Returns a small status dict."""
    date_str = date_str or dt.date.today().isoformat()
    items, unresolved = normalize_slate(date_str, odds_type=odds_type)
    snap_path, n_items = write_snapshot(date_str, items=items)
    hist_path, n_totals = accumulate_totals(date_str, items=items)
    return {
        "date": date_str, "snapshot": str(snap_path), "offer_items": n_items,
        "totals_logged": n_totals, "historical_file": str(hist_path),
        "unresolved_teams": unresolved,
    }


def probe(date_str=None):
    """Dump raw /matches and /odds JSON (first records) so the real response
    shape can be confirmed against the parser the first time a key is used."""
    date_str = date_str or dt.date.today().isoformat()
    matches = fetch_matches(date_str)
    odds = fetch_odds(date_str)
    md = (matches or {}).get("data", [])
    od = (odds or {}).get("data", [])
    print(f"/matches: {len(md)} games. First record:")
    print(json.dumps(md[0], indent=2)[:1500] if md else "  (none)")
    print(f"\n/odds: {len(od)} games. First record:")
    print(json.dumps(od[0], indent=2)[:2500] if od else "  (none)")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Highlightly free-tier live odds: refresh snapshot + log totals history.")
    ap.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--probe", action="store_true", help="dump raw API JSON to verify schema (use once with a new key)")
    ap.add_argument("--live", action="store_true", help="use live in-game odds instead of prematch")
    args = ap.parse_args()
    if args.probe:
        probe(args.date)
        return
    status = refresh(args.date, odds_type="live" if args.live else "prematch")
    print(json.dumps(status, indent=2))
    if status["unresolved_teams"]:
        print("\n[!] Some team names did not resolve to codes (odds for these games were skipped):")
        for a, h in status["unresolved_teams"]:
            print(f"    {a} @ {h}")
        print("    Add them to team_mapping._NAME_KEYWORDS and re-run.")


if __name__ == "__main__":
    main()
