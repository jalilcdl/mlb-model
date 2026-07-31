"""
Live odds ingestion + edge-finding.

We don't have direct live odds API access in this environment. Instead, real
odds snapshots get handed to us periodically (by whoever is orchestrating
this project) as JSON files dropped into odds/snapshots/. No polling needed
on our end -- just read whatever's in that folder at prediction time.

Snapshot shape (one event+market per file, or a list of them in one file):

    {
      "event_id": "fef71589-...",
      "offer_type": "moneyline",      # or "spread" (run line) or "total"
      "away_team": "nym", "home_team": "phi",
      "odds": {
        "DraftKings": [{"side": "away", "line": 0, "odds": 115}, {"side": "home", "line": 0, "odds": -139}],
        "FanDuel":    [{"side": "away", "line": 0, "odds": 110}, {"side": "home", "line": 0, "odds": -130}],
        "Pinnacle":   [{"side": "away", "line": 0, "odds": 117}, {"side": "home", "line": 0, "odds": -127}],
        ...   # ~28 sportsbooks total, same shape
      }
    }

"spread" is MLB's run line: side is home/away, line is the real signed spread
(e.g. -1.5 / +1.5 -- not always exactly that, so we don't assume it). "total"
uses side over/under with a real line (e.g. 8.5). For each event+market we
compute, per side: the single best price across all books (what a bettor
actually shopping lines would get) and a de-vigged *consensus* probability
averaged across every book's own no-vig number (a "wisdom of the market"
fair-value estimate, not just one book's price). Edges are reported both
ways: edge vs. consensus (does the model disagree with the market's
collective fair value?) and EV vs. the single best price (what you'd actually
earn betting the best line available).

A simpler legacy format (odds/odds.json, one book, one game per entry) is
still supported for quick manual entry -- see odds_example.json. Both formats
are normalized into the same internal shape before edge computation, so
nothing downstream needs to know which one was used.

To add a real self-serve live feed later (e.g. The Odds API), implement
fetch_live_odds() below and have it return snapshot-shaped dicts; everything
else (parsing, de-vigging, edge calc) already works from that shape.
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.data import team_mapping
from src.models import monte_carlo

# Aliases for team codes that don't match our canonical MLB Stats API
# abbreviations directly (see src/data/team_mapping.py for the canonical
# list). Extend this if a real snapshot uses a code we don't recognize --
# load_snapshots() silently skips unmapped teams, so check its return value
# or the diagnostics helper below if games seem to be going missing.
_TEAM_ALIASES = {
    "AZ": "ARI",
    "OAK": "ATH",
    "WAS": "WSH",
    "SFO": "SF",
    "TAM": "TB",
    "NYA": "NYY",
    "NYN": "NYM",
    "CHA": "CWS",
    "CHN": "CHC",
    "SDN": "SD",
    "SLN": "STL",
    "KCA": "KC",
}


def _normalize_team(code):
    if not code:
        return None
    code = str(code).strip().upper()
    code = _TEAM_ALIASES.get(code, code)
    return code if code in team_mapping.all_codes() else None


# --------------------------------------------------------------------------
# Odds math
# --------------------------------------------------------------------------
def american_to_prob(odds):
    """American odds -> raw implied probability (includes the book's vig)."""
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def american_to_decimal(odds):
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / -odds


def prob_to_american(prob):
    """Probability -> the FAIR (break-even, zero-vig) American price for it.

    This is the number to line-shop against: at exactly this price the bet is
    EV-neutral *if the model is right*, so only a better price is +EV. E.g.
    p=0.60 -> -150: you need better than -150 (say -140, or +110) to profit."""
    p = float(prob)
    if p <= 0 or p >= 1:
        return None
    return decimal_to_american(1.0 / p)


def required_price(prob, edge_pts=0.0):
    """The American price you must BEAT for the model to show at least
    `edge_pts` percentage points of edge over the break-even bar.

    Guards against acting on hairline differences: with a model whose own
    error is non-trivial, "fair price exactly" is not a bet -- you want margin.
    edge_pts=2 means: only bet if the price implies a probability at least 2
    points below the model's."""
    p = float(prob) - (float(edge_pts) / 100.0)
    if p <= 0 or p >= 1:
        return None
    return decimal_to_american(1.0 / p)


def decimal_to_american(decimal_odds):
    """Decimal odds (e.g. 2.8) -> American (e.g. +180). Feeds don't all use the
    same convention; Highlightly returns decimal, our snapshot format is American."""
    d = float(decimal_odds)
    if d <= 1.0:
        return 0
    if d >= 2.0:
        return int(round((d - 1.0) * 100.0))
    return int(round(-100.0 / (d - 1.0)))


def remove_vig_two_way(prob_a, prob_b):
    """Proportional de-vig: scale two implied probabilities so they sum to 1."""
    total = prob_a + prob_b
    if total <= 0:
        return prob_a, prob_b
    return prob_a / total, prob_b / total


def expected_value_pct(model_prob, american_odds, push_prob=0.0):
    """Expected value per $1 staked at the given American odds, using our
    model's win probability as the true probability. E.g. 0.05 = +5% EV.

    push_prob is the chance the bet pushes and the stake comes back (whole-
    number totals). A push is neither a win nor a loss, so it must be excluded
    from the losing mass rather than counted against the bet."""
    decimal_odds = american_to_decimal(american_odds)
    lose_prob = max(0.0, 1.0 - model_prob - push_prob)
    return model_prob * (decimal_odds - 1.0) - lose_prob


def _better_price(odds_a, odds_b):
    """True if American odds `odds_a` pays more than `odds_b` (same side)."""
    return american_to_decimal(odds_a) > american_to_decimal(odds_b)


# --------------------------------------------------------------------------
# Multi-sportsbook snapshot parsing
# --------------------------------------------------------------------------
def _extract_side_odds(offers, side, required_line=None):
    """Find the offer for `side`. Books commonly list several alternate
    lines per side (e.g. -1.5, -2.5, -0.5 run lines); when `required_line` is
    given, only an exact match counts -- an alternate line is a different bet
    with a different fair price, not a substitute for the consensus line."""
    matches = [o for o in offers if o.get("side") == side]
    if required_line is not None:
        matches = [o for o in matches if o.get("line") == required_line]
    return matches[0] if matches else None


def _consensus_and_best(odds_by_book, side_a, side_b, line_a=None, line_b=None):
    """Given {book: [offers]} for a two-way market, return per-side best
    price/book and a de-vigged consensus probability (averaged across books).

    If line_a/line_b are given (spread and total markets, where books list
    multiple alternate lines per side), only offers at exactly that line are
    considered -- an alternate line is a different bet with a different fair
    price, not a substitute for the consensus line.

    Best-price search and consensus de-vigging are deliberately decoupled:
    a book's best price for one side counts even if that same book didn't
    post the consensus line for the *other* side (real feeds do this --
    e.g. a book offering the primary home line but only an alternate away
    line). De-vigging still requires both sides from the *same* book,
    since pairing one book's home price with a different book's away price
    would mix two different vig structures into a meaningless number.
    """
    best = {side_a: (None, None), side_b: (None, None)}  # (odds, book)
    novig_probs = {side_a: [], side_b: []}
    line_value = {side_a: line_a, side_b: line_b}

    for book, offers in (odds_by_book or {}).items():
        oa = _extract_side_odds(offers, side_a, line_a)
        ob = _extract_side_odds(offers, side_b, line_b)

        for side, entry in ((side_a, oa), (side_b, ob)):
            if not entry or entry.get("odds") is None:
                continue
            if line_value[side] is None:
                line_value[side] = entry.get("line")
            cur_odds, _ = best[side]
            if cur_odds is None or _better_price(entry["odds"], cur_odds):
                best[side] = (entry["odds"], book)

        if oa and ob and oa.get("odds") is not None and ob.get("odds") is not None:
            pa, pb = american_to_prob(oa["odds"]), american_to_prob(ob["odds"])
            na, nb = remove_vig_two_way(pa, pb)
            novig_probs[side_a].append(na)
            novig_probs[side_b].append(nb)

    result = {}
    for side in (side_a, side_b):
        result[side] = {
            "line": line_value[side],
            "best_odds": best[side][0],
            "best_book": best[side][1],
            "consensus_prob": float(np.mean(novig_probs[side])) if novig_probs[side] else None,
            "n_books": len(novig_probs[side]),
        }
    return result


def load_snapshots(folder=None):
    """Load every JSON file in odds/snapshots/ (each a single event-offer
    object, or a list of them) and merge same-event moneyline/spread/total
    records into one entry per game, keyed by (away_team, home_team).

    Returns {(away_team, home_team): [event, ...]} -- a LIST per matchup,
    because a doubleheader legitimately has two different games between the
    same two teams on the same day. If a snapshot carries `start_time_utc`,
    attach_edges uses it to pick the right one (see _select_event); without
    it, an ambiguous matchup is left unmatched rather than guessed at.
    """
    folder = folder or (config.ODDS_DIR / "snapshots")
    if not folder.exists():
        return {}

    by_event_id = {}
    for path in sorted(folder.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            _ingest_snapshot_item(item, by_event_id)

    events = {}
    for ev in by_event_id.values():
        _drop_impossible_run_line(ev)
        events.setdefault((ev["away_team"], ev["home_team"]), []).append(ev)
    return events


def _drop_impossible_run_line(ev):
    """Discard run-line prices that contradict the same event's moneyline.

    A team cannot win by 2+ runs more often than it wins at all, so
    P(cover -1.5) > P(win) is arithmetically impossible and means the feed is
    corrupt for that market (sides swapped, a mis-scrape, a stale line). This
    is worth checking rather than trusting: a broken run line doesn't look
    broken downstream, it looks like an enormous edge -- the model happily
    reports the market as wildly mispriced on precisely the market whose data
    is wrong, and that lands at the top of any EV-sorted list. Dropping the
    market costs one bet; believing it costs real money.
    """
    ml, rl = ev.get("moneyline"), ev.get("run_line")
    if not ml or not rl:
        return
    for side in ("home", "away"):
        m, r = ml.get(side), rl.get(side)
        if not m or not r or r.get("line") is None or r["line"] >= 0:
            continue  # only the favourite's -1.5 side can violate this
        p_win, p_cover = m.get("consensus_prob"), r.get("consensus_prob")
        if p_win is None or p_cover is None:
            continue
        if p_cover > p_win + 1e-9:
            ev["run_line_warning"] = (
                f"run line dropped: implied P({side} covers {r['line']:+g})={p_cover:.3f} "
                f"exceeds P({side} wins)={p_win:.3f}, which is impossible"
            )
            ev.pop("run_line", None)
            return


def _ingest_snapshot_item(item, by_event_id):
    home = _normalize_team(item.get("home_team"))
    away = _normalize_team(item.get("away_team"))
    if not home or not away:
        return  # unmapped team code -- see _TEAM_ALIASES

    # Key by event_id when present; otherwise fall back to matchup+start time so
    # that two games of a doubleheader stay distinct rather than merging.
    key = item.get("event_id") or f"{away}@{home}@{item.get('start_time_utc') or ''}"
    ev = by_event_id.setdefault(
        key,
        {
            "home_team": home,
            "away_team": away,
            "event_id": item.get("event_id"),
            "start_time_utc": item.get("start_time_utc"),
            "source": item.get("source") or "snapshot",
        },
    )

    offer_type = item.get("offer_type")
    odds_by_book = item.get("odds", {})
    consensus_line = item.get("consensus_line")  # present on spread/total; None for moneyline

    if offer_type == "moneyline":
        ev["moneyline"] = _consensus_and_best(odds_by_book, "away", "home")
    elif offer_type == "total":
        # Total lines are unsigned -- over and under share the same number.
        sides = _consensus_and_best(odds_by_book, "over", "under", consensus_line, consensus_line)
        ev["total"] = {"line": sides["over"]["line"] or sides["under"]["line"], **sides}
    elif offer_type == "spread":
        # Spread lines are signed and symmetric: consensus_line is the home
        # line (e.g. -1.5); the away line is its negation (+1.5). Books that
        # only posted an alternate line for one side (seen in real feeds)
        # contribute nothing for that side rather than being mismatched.
        away_line = -consensus_line if consensus_line is not None else None
        ev["run_line"] = _consensus_and_best(odds_by_book, "away", "home", away_line, consensus_line)


# --------------------------------------------------------------------------
# Legacy single-book format (odds/odds.json) -- normalized to the same shape
# --------------------------------------------------------------------------
def _legacy_to_normalized(g):
    home, away = g["home_team"], g["away_team"]
    ev = {"home_team": home, "away_team": away, "event_id": None, "source": "manual odds.json"}

    ml = g.get("moneyline")
    if ml and "home" in ml and "away" in ml:
        ph, pa = american_to_prob(ml["home"]), american_to_prob(ml["away"])
        nh, na = remove_vig_two_way(ph, pa)
        ev["moneyline"] = {
            "home": {"line": None, "best_odds": ml["home"], "best_book": "manual", "consensus_prob": nh, "n_books": 1},
            "away": {"line": None, "best_odds": ml["away"], "best_book": "manual", "consensus_prob": na, "n_books": 1},
        }

    total = g.get("total")
    if total and "over" in total and "under" in total and "line" in total:
        po, pu = american_to_prob(total["over"]), american_to_prob(total["under"])
        no, nu = remove_vig_two_way(po, pu)
        ev["total"] = {
            "line": total["line"],
            "over": {"line": total["line"], "best_odds": total["over"], "best_book": "manual", "consensus_prob": no, "n_books": 1},
            "under": {"line": total["line"], "best_odds": total["under"], "best_book": "manual", "consensus_prob": nu, "n_books": 1},
        }

    rl = g.get("run_line")
    if rl and "home" in rl and "away" in rl:
        h, a = rl["home"], rl["away"]
        ph, pa = american_to_prob(h["odds"]), american_to_prob(a["odds"])
        nh, na = remove_vig_two_way(ph, pa)
        ev["run_line"] = {
            "home": {"line": h["line"], "best_odds": h["odds"], "best_book": "manual", "consensus_prob": nh, "n_books": 1},
            "away": {"line": a["line"], "best_odds": a["odds"], "best_book": "manual", "consensus_prob": na, "n_books": 1},
        }
    return ev


def load_odds_legacy(path=None):
    path = path or (config.ODDS_DIR / "odds.json")
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    events = {}
    for g in raw.get("games", []):
        home = _normalize_team(g.get("home_team"))
        away = _normalize_team(g.get("away_team"))
        if not home or not away:
            continue
        events.setdefault((away, home), []).append(
            _legacy_to_normalized({**g, "home_team": home, "away_team": away})
        )
    return events


def load_market_odds(snapshot_folder=None, legacy_path=None):
    """Load odds from whichever source is available, preferring the richer
    multi-sportsbook snapshot folder over the legacy single-book odds.json."""
    snapshots = load_snapshots(snapshot_folder)
    if snapshots:
        return snapshots
    return load_odds_legacy(legacy_path)


def fetch_live_odds(date_str, api_key=None):
    """A real self-serve live-odds source IS implemented: src/data/highlightly.py
    (Highlightly free tier). It fetches a slate and writes a snapshot into
    odds/snapshots/, which load_snapshots() then consumes unchanged -- so
    there's nothing to wire in here. Call highlightly.refresh(date_str), or use
    the dashboard's "Fetch live odds" button / `python -m src.data.highlightly`.

    (This lives in a separate module rather than here to avoid a circular
    import: highlightly imports this module's odds-math helpers.)
    """
    from src.data import highlightly  # local import: avoids circular dependency
    return highlightly.refresh(date_str)


# --------------------------------------------------------------------------
# Edge computation
# --------------------------------------------------------------------------
_EDGE_COLUMNS = [
    "event_id", "odds_source",
    "ml_home_consensus_prob", "ml_home_best_odds", "ml_home_best_book", "ml_home_n_books",
    "edge_home_ml", "ev_home_ml_pct",
    "ml_away_consensus_prob", "ml_away_best_odds", "ml_away_best_book", "ml_away_n_books",
    "edge_away_ml", "ev_away_ml_pct",
    "total_line", "total_push_prob", "model_over_prob", "model_under_prob",
    "total_over_consensus_prob", "total_over_best_odds", "total_over_best_book", "total_over_n_books",
    "edge_over", "ev_over_pct",
    "total_under_consensus_prob", "total_under_best_odds", "total_under_best_book", "total_under_n_books",
    "edge_under", "ev_under_pct",
    "rl_home_line", "rl_home_consensus_prob", "rl_home_best_odds", "rl_home_best_book", "rl_home_n_books",
    "edge_home_rl", "ev_home_rl_pct",
    "rl_away_line", "rl_away_consensus_prob", "rl_away_best_odds", "rl_away_best_book", "rl_away_n_books",
    "edge_away_rl", "ev_away_rl_pct",
    "odds_available",
]


DOUBLEHEADER_MATCH_TOLERANCE_SECONDS = 90 * 60


def _select_event(candidates, row, require_time=False):
    """Pick which odds event goes with this prediction row.

    One candidate -> use it (the normal case; no start time needed). More than
    one means a doubleheader: the same two teams play twice in a day, so the
    matchup alone is ambiguous and we disambiguate on scheduled start time.
    If we can't confidently tell them apart, return None -- silently attaching
    the nightcap's odds to the afternoon game would produce confidently wrong
    edges, which is worse than reporting no odds.

    `require_time` closes a hole in that reasoning. The single-candidate
    shortcut assumes one candidate means one game, but on a doubleheader a feed
    routinely lists only ONE of the two games -- books pull the opener once it
    starts. The shortcut then hands the nightcap's line to the completed opener
    with no time check at all, which is exactly the confidently-wrong match the
    rest of this function exists to prevent. Callers that know the matchup is a
    doubleheader pass require_time=True to force the time check even when only
    one candidate survives.
    """
    if not candidates:
        return None
    if len(candidates) == 1 and not require_time:
        return candidates[0]

    game_ts = pd.to_datetime(row.get("game_datetime_utc"), utc=True, errors="coerce")
    if pd.isna(game_ts):
        return None

    best, best_delta = None, None
    for ev in candidates:
        ts = pd.to_datetime(ev.get("start_time_utc"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        delta = abs((ts - game_ts).total_seconds())
        if best_delta is None or delta < best_delta:
            best, best_delta = ev, delta

    if best is None or best_delta > DOUBLEHEADER_MATCH_TOLERANCE_SECONDS:
        return None
    return best


def attach_edges(predictions_df, events, n_sims=None):
    """Enrich a predictions DataFrame with market price/de-vigged-probability
    columns and model edges wherever a matching event exists. Events are keyed
    by (away_team, home_team) and may hold more than one game (doubleheaders),
    resolved by scheduled start time -- see _select_event.

    Total and run-line probabilities are recomputed with a fresh Monte Carlo
    simulation at each market's *actual* line (which varies by book/game and
    won't generally match our default expected-value output), reusing one
    simulation per game for both markets. No-op (NaN columns) for games with no
    odds available.

    Note on `*_consensus_prob`: it's the vig-removed market probability
    averaged across whatever books the snapshot carried. With a single-book
    feed that's just that one book's fair price, not a market consensus --
    check the accompanying `*_n_books` before describing it as consensus.
    """
    df = predictions_df.copy()
    for c in _EDGE_COLUMNS:
        df[c] = pd.NA

    # A matchup appearing twice in the predictions is a doubleheader, so odds
    # for it must be time-matched even when the feed lists only one of the two
    # games -- see require_time in _select_event.
    dh_matchups = {
        k for k, n in df.groupby(["away_team", "home_team"]).size().items() if n > 1
    }

    for idx, row in df.iterrows():
        matchup = (row["away_team"], row["home_team"])
        candidates = events.get(matchup) or []
        if not isinstance(candidates, list):  # tolerate the older single-event shape
            candidates = [candidates]
        ev = _select_event(candidates, row, require_time=matchup in dh_matchups)
        if not ev:
            df.at[idx, "odds_available"] = False
            continue
        df.at[idx, "odds_available"] = True
        df.at[idx, "event_id"] = ev.get("event_id")
        df.at[idx, "odds_source"] = ev.get("source")

        sim = monte_carlo.simulate_game(
            row["expected_home_runs"],
            row["expected_away_runs"],
            n_sims=n_sims,
            overdispersion=row.get("overdispersion"),
        )

        ml = ev.get("moneyline")
        if ml:
            _fill_side(df, idx, "ml_home", ml.get("home"), row["home_win_prob"])
            _fill_side(df, idx, "ml_away", ml.get("away"), row["away_win_prob"])

        total = ev.get("total")
        if total and total.get("line") is not None:
            line = total["line"]
            over_p, push_p, under_p = monte_carlo.total_outcome_probs(
                sim["home_runs"], sim["away_runs"], line
            )
            df.at[idx, "total_line"] = line
            df.at[idx, "total_push_prob"] = push_p
            # Empirical simulated probabilities as first-class columns. These are
            # raw counts off the simulation (P(over)+P(push)+P(under) == 1), not a
            # distributional approximation, and not something a consumer should
            # have to reconstruct as consensus+edge.
            df.at[idx, "model_over_prob"] = over_p
            df.at[idx, "model_under_prob"] = under_p
            # A book's de-vigged two-way price is implicitly conditional on the
            # bet not pushing (a push refunds everyone), so the model prob we
            # compare against it must be conditional too -- otherwise the
            # push mass shows up as a phantom edge on whole-number lines. EV,
            # by contrast, uses the raw probabilities plus push_prob, since a
            # push really does return the stake.
            no_push = over_p + under_p
            over_cond = over_p / no_push if no_push > 0 else 0.5
            _fill_side(df, idx, "total_over", total.get("over"), over_cond,
                       ev_prob=over_p, push_prob=push_p)
            _fill_side(df, idx, "total_under", total.get("under"), 1 - over_cond,
                       ev_prob=under_p, push_prob=push_p)

        rl = ev.get("run_line")
        if rl:
            home_r, away_r = rl.get("home"), rl.get("away")
            if home_r and home_r.get("line") is not None:
                home_cover = monte_carlo.cover_probability(sim["home_runs"], sim["away_runs"], home_r["line"])
                df.at[idx, "rl_home_line"] = home_r["line"]
                _fill_side(df, idx, "rl_home", home_r, home_cover)
            if away_r and away_r.get("line") is not None:
                away_cover = monte_carlo.cover_probability(sim["away_runs"], sim["home_runs"], away_r["line"])
                df.at[idx, "rl_away_line"] = away_r["line"]
                _fill_side(df, idx, "rl_away", away_r, away_cover)

    return df


def _fill_side(df, idx, prefix, market_side, model_prob, ev_prob=None, push_prob=0.0):
    """model_prob is compared against the de-vigged market price to get the
    edge (both conditional on no push). ev_prob defaults to model_prob and is
    the raw win probability used for EV, where push_prob is stake returned.
    They differ only for whole-number totals; for moneyline and the ±1.5 run
    line no push is possible and the two coincide."""
    if not market_side or market_side.get("best_odds") is None:
        return
    if ev_prob is None:
        ev_prob = model_prob
    consensus = market_side.get("consensus_prob")
    df.at[idx, f"{prefix}_consensus_prob"] = consensus
    df.at[idx, f"{prefix}_best_odds"] = market_side["best_odds"]
    df.at[idx, f"{prefix}_best_book"] = market_side.get("best_book")
    if f"{prefix}_n_books" in df.columns:
        df.at[idx, f"{prefix}_n_books"] = market_side.get("n_books")
    edge_col = _edge_col_name(prefix)
    ev_col = _ev_col_name(prefix)
    if consensus is not None:
        df.at[idx, edge_col] = model_prob - consensus
    df.at[idx, ev_col] = expected_value_pct(ev_prob, market_side["best_odds"], push_prob=push_prob)


def _edge_col_name(prefix):
    mapping = {
        "ml_home": "edge_home_ml", "ml_away": "edge_away_ml",
        "total_over": "edge_over", "total_under": "edge_under",
        "rl_home": "edge_home_rl", "rl_away": "edge_away_rl",
    }
    return mapping[prefix]


def _ev_col_name(prefix):
    mapping = {
        "ml_home": "ev_home_ml_pct", "ml_away": "ev_away_ml_pct",
        "total_over": "ev_over_pct", "total_under": "ev_under_pct",
        "rl_home": "ev_home_rl_pct", "rl_away": "ev_away_rl_pct",
    }
    return mapping[prefix]
