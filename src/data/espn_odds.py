"""
ESPN public scoreboard odds (free, no API key, no scraping of rendered HTML).

ESPN's public scoreboard JSON carries a full DraftKings line for most games:
moneyline, run line, and total, with prices. That makes it a reliable automated
replacement for hand-pasting ESPN's web page -- and unlike Highlightly's free
tier (odds paywalled) it actually returns odds.

  GET https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=YYYYMMDD
    events[].competitions[0].odds[0]:
      provider.name                 -> book (DraftKings)
      moneyline.{home,away}.close.odds
      pointSpread.{home,away}.close.{line,odds}     ("Runline")
      total.{over,under}.close.{line,odds}          (line like "o8.5")

Single-book by nature: everything downstream should say "vs DraftKings", not
"consensus of N books" -- the de-vig of one book's two-way price is that book's
fair number, not a market consensus. n_books will be 1.

Normalizes into the same snapshot shape as odds/snapshots/, so the existing
de-vig / push-correction / impossible-line validator / doubleheader-by-start-time
machinery applies unchanged.
"""
import datetime as dt
import json
import re

import requests

from src import config
from src.data import team_mapping

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
_TIMEOUT = 25
_UA = "mlb-model/1.0 (+local research tool)"


def fetch_scoreboard(date_str):
    """date_str: YYYY-MM-DD."""
    r = requests.get(
        SCOREBOARD_URL,
        params={"dates": date_str.replace("-", "")},
        headers={"User-Agent": _UA},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def _num(x):
    """ESPN prices come as strings like '-149', '+139', 'EVEN', 'o8.5'."""
    if x is None:
        return None
    s = str(x).strip().replace("+", "")
    if s.upper() in ("EVEN", "EV"):
        return 100
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


def _team_code(team_obj):
    if not team_obj:
        return None
    return team_mapping.code_from_name(
        team_obj.get("displayName"), team_obj.get("name"), team_obj.get("abbreviation")
    )


def normalize_slate(date_str, scoreboard=None):
    """Return (snapshot_items, skipped) for a date. `skipped` lists games we
    couldn't use, with the reason -- surfaced rather than silently dropped."""
    sb = scoreboard if scoreboard is not None else fetch_scoreboard(date_str)
    items, skipped = [], []

    for event in sb.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        odds_list = comp.get("odds") or []

        home_t = away_t = None
        for c in comp.get("competitors", []):
            if c.get("homeAway") == "home":
                home_t = c.get("team")
            elif c.get("homeAway") == "away":
                away_t = c.get("team")
        home, away = _team_code(home_t), _team_code(away_t)
        if not home or not away:
            skipped.append((event.get("name"), "unmapped team name"))
            continue
        if not odds_list:
            skipped.append((f"{away} @ {home}", "no odds posted"))
            continue

        o = odds_list[0]
        book = (o.get("provider") or {}).get("name") or "ESPN"
        start = event.get("date")
        base = {
            "event_id": f"espn-{event.get('id')}",
            "start_time_utc": start,
            "away_team": away, "home_team": home,
            "source": f"ESPN public scoreboard ({book})",
        }

        ml = o.get("moneyline") or {}
        h_ml = _num(((ml.get("home") or {}).get("close") or {}).get("odds"))
        a_ml = _num(((ml.get("away") or {}).get("close") or {}).get("odds"))
        if h_ml and a_ml:
            items.append({**base, "offer_type": "moneyline", "odds": {book: [
                {"side": "home", "line": 0, "odds": int(h_ml)},
                {"side": "away", "line": 0, "odds": int(a_ml)},
            ]}})

        tot = o.get("total") or {}
        ov, un = (tot.get("over") or {}).get("close") or {}, (tot.get("under") or {}).get("close") or {}
        o_line, u_line = _num(ov.get("line")), _num(un.get("line"))
        o_odds, u_odds = _num(ov.get("odds")), _num(un.get("odds"))
        if o_odds and u_odds and o_line is not None:
            # Only pair sides posted at the SAME number (a mismatch means one is
            # an alternate line -- a different bet, not a counterpart).
            if u_line is not None and abs(o_line - u_line) > 1e-9:
                skipped.append((f"{away} @ {home}", f"total lines mismatch (o{o_line} / u{u_line}) - skipped"))
            else:
                items.append({**base, "offer_type": "total", "consensus_line": o_line, "odds": {book: [
                    {"side": "over", "line": o_line, "odds": int(o_odds)},
                    {"side": "under", "line": o_line, "odds": int(u_odds)},
                ]}})

        ps = o.get("pointSpread") or {}
        ph, pa = (ps.get("home") or {}).get("close") or {}, (ps.get("away") or {}).get("close") or {}
        h_line, h_odds = _num(ph.get("line")), _num(ph.get("odds"))
        a_line, a_odds = _num(pa.get("line")), _num(pa.get("odds"))
        if h_odds and a_odds and h_line is not None and a_line is not None:
            items.append({**base, "offer_type": "spread", "consensus_line": h_line, "odds": {book: [
                {"side": "home", "line": h_line, "odds": int(h_odds)},
                {"side": "away", "line": a_line, "odds": int(a_odds)},
            ]}})

    return items, skipped


def write_snapshot(date_str, items=None, folder=None):
    if items is None:
        items, _ = normalize_slate(date_str)
    folder = folder or (config.ODDS_DIR / "snapshots")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"espn_{date_str}.json"
    with open(path, "w") as f:
        json.dump(items, f, indent=2)
    return path, len(items)


def refresh(date_str=None, clear_same_day=True):
    """Fetch, write the snapshot, and log totals to the running historical file.
    By default clears other snapshots for the same date first so a game isn't
    double-listed across sources."""
    from src.data.highlightly import accumulate_totals  # shared accumulator
    date_str = date_str or dt.date.today().isoformat()
    items, skipped = normalize_slate(date_str)
    folder = config.ODDS_DIR / "snapshots"
    if clear_same_day and folder.exists():
        for p in folder.glob(f"*{date_str}*.json"):
            p.unlink()
    snap, n = write_snapshot(date_str, items=items, folder=folder)
    hist, n_tot = accumulate_totals(date_str, items=items)
    return {"date": date_str, "snapshot": str(snap), "offer_items": n,
            "totals_logged": n_tot, "historical_file": str(hist), "skipped": skipped}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Pull free MLB odds from ESPN's public scoreboard JSON.")
    ap.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    status = refresh(args.date)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
