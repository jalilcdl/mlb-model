"""
Single-page PDF chart: model probability vs de-vigged market implied
probability for every moneyline play on a slate, so the edge is visible at a
glance.

Scope note: moneyline only, deliberately. Totals and the run line are excluded
because neither has ever shown an edge in backtesting -- the totals model was
tested against 11,706 real closing lines (50.25% ATS vs a 52.38% break-even)
and again after recalibration, and the run line matched a naive baseline.
Charting them alongside moneyline would give unvalidated markets the same
visual weight as the one validated market.

Each model bar carries its win probability as an explicit numeric label, so a
row can be read precisely without eyeballing bar length against the axis.

Colour encodes settlement status (green = settled win, red = settled loss,
blue = pending/unsettled), matched against data/processed/bet_log.csv where a
row for that play exists.

    python -m src.reports.play_chart              # today
    python -m src.reports.play_chart 2026-07-21
"""
import argparse
import datetime as dt
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.models import monte_carlo
# _select_event is private but is THE doubleheader disambiguator (matches a
# prediction to its odds event by scheduled start time). Reusing it keeps the
# chart and attach_edges from ever disagreeing about which game is which.
from src.odds.odds_adapter import (
    _normalize_team, _select_event, american_to_prob, remove_vig_two_way,
)

ET = ZoneInfo("America/New_York")

SETTLED_WIN = "#2e7d32"
SETTLED_LOSS = "#c62828"
PENDING = "#1565c0"
MARKET = "#9e9e9e"


def _bet_log_status(date_str):
    """{(matchup, pick_team): result} from the bet log, for colour-coding."""
    path = config.BET_LOG_FILE
    if not path.exists():
        return {}
    log = pd.read_csv(path)
    out = {}
    for r in log.itertuples():
        if str(r.date) != date_str or str(r.market).strip().lower() != "moneyline":
            continue
        team = str(r.pick).replace("ML", "").strip()
        res = str(r.result).strip().lower()
        out[team] = res if res in ("win", "loss", "push") else "pending"
    return out


def _et_clock(ts):
    """'1:35pm' in Eastern, or None. Hand-formatted because Windows strftime
    has no %-I and would render '01:35pm'."""
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    t = t.tz_convert(ET)
    return f"{t.hour % 12 or 12}:{t.minute:02d}{'am' if t.hour < 12 else 'pm'}"


def game_suffixes(preds):
    """{game_pk: label suffix}. Empty for ordinary games; for a doubleheader
    EVERY game gets an explicit '(G1, 1:35pm ET)' tag.

    This is the fix for the bug that mattered: the chart used to key on
    (away_team, home_team) and keep only the first match, so a doubleheader
    silently rendered as ONE row. A reader could not tell there were two games,
    which of them the row described, or that the other existed at all -- and
    the two games have different starters and therefore genuinely different
    probabilities, not merely different labels. Both games now always render.
    """
    out = {}
    for _, grp in preds.groupby(["away_team", "home_team"], sort=False):
        if len(grp) == 1:
            out[grp.iloc[0]["game_pk"]] = ""
            continue
        for i, (_, r) in enumerate(grp.sort_values("game_datetime_utc").iterrows(), 1):
            clock = _et_clock(r["game_datetime_utc"])
            out[r["game_pk"]] = f" (G{i}, {clock} ET)" if clock else f" (G{i})"
    return out


def _game_flags(g):
    """Trailing markers that must never be silently dropped:
      *  a probable starter is not yet announced -> that side is modelled at
         league average, so the projection is materially softer than it looks.
      [FINAL] / [LIVE]  the game is not bettable at the price shown.
    """
    flags = []
    if pd.isna(g.get("home_probable_pitcher")) or pd.isna(g.get("away_probable_pitcher")):
        flags.append("*")
    st = str(g.get("status") or "")
    if st == "Final":
        flags.append("[FINAL]")
    elif st not in ("Scheduled", "Pre-Game", "Warmup", ""):
        flags.append("[LIVE]")
    return "".join(f" {f}" for f in flags)


def _doubleheader_matchups(preds):
    """Matchups played twice today. Odds for these must be time-matched even
    when the feed lists only one of the two games (books pull the opener once
    it starts), or the nightcap's line gets pinned onto the completed opener."""
    return {k for k, n in preds.groupby(["away_team", "home_team"]).size().items() if n > 1}


def _events_by_matchup(events):
    """Group odds events by canonical matchup, keeping doubleheaders together
    so _select_event can pick the right one by start time."""
    by, unmapped = {}, []
    for ev in events:
        # Normalize through the canonical mapping: feeds use their own codes
        # (e.g. the connector says "oak" where we say "ATH"), and a raw .upper()
        # silently drops those games from the chart instead of failing loudly.
        a = _normalize_team(ev["away_team"])
        h = _normalize_team(ev["home_team"])
        if not a or not h:
            unmapped.append(f"{ev['away_team']}@{ev['home_team']} (unmappable code)")
            continue
        by.setdefault((a, h), []).append(ev)
    return by, unmapped


def build_rows(preds, events, date_str):
    """One row per side PER GAME: model prob, de-vigged market prob, edge.

    Iterates predictions (one row per game_pk) rather than odds events, so a
    doubleheader yields four rows, not two.
    """
    status = _bet_log_status(date_str)
    by_matchup, unmatched = _events_by_matchup(events)
    suffix = game_suffixes(preds)
    dh = _doubleheader_matchups(preds)

    rows = []
    for _, g in preds.iterrows():
        a, h = g["away_team"], g["home_team"]
        sfx = suffix.get(g["game_pk"], "") + _game_flags(g)
        ev = _select_event(by_matchup.get((a, h), []), g, require_time=(a, h) in dh)
        if ev is None:
            unmatched.append(f"{a}@{h}{sfx} (no odds event matched by start time)")
            continue
        fair_h, fair_a = remove_vig_two_way(
            american_to_prob(float(ev["consensus_home_ml"])),
            american_to_prob(float(ev["consensus_away_ml"])),
        )
        for team, model_p, mkt_p, best in (
            (a, g["away_win_prob"], fair_a, ev["best_away_ml"]),
            (h, g["home_win_prob"], fair_h, ev["best_home_ml"]),
        ):
            rows.append({
                "label": f"{a}@{h}{sfx} — {team} ML",
                "team": team,
                "model": float(model_p),
                "market": float(mkt_p),
                "edge": float(model_p - mkt_p),
                "best": int(best),
                # Bet-log rows are keyed by team only, so on a doubleheader this
                # colour is ambiguous between the two games. Deliberately left
                # as-is: fixing it needs game_pk in the bet log, which is a
                # separate change.
                "status": status.get(team, "pending"),
            })
    return pd.DataFrame(rows), unmatched


def _bet_log_status_totals(date_str):
    """{'over 7.5': result} from the bet log, for colour-coding totals rows."""
    path = config.BET_LOG_FILE
    if not path.exists():
        return {}
    log = pd.read_csv(path)
    out = {}
    for r in log.itertuples():
        if str(r.date) != date_str or str(r.market).strip().lower() != "total":
            continue
        res = str(r.result).strip().lower()
        out[str(r.pick).strip().lower()] = res if res in ("win", "loss", "push") else "pending"
    return out


def build_totals_rows(preds, totals_items, date_str, n_sims=20000):
    """Rows for the totals chart: model P(over)/P(under) at each game's REAL
    posted line, against that book's de-vigged two-way price.

    Both are conditional on the bet not pushing. A book's two-way price is
    implicitly no-push (a push refunds everyone), so the model probability has
    to be normalized the same way or whole-number lines would compare a
    push-inclusive number against a push-exclusive one.
    """
    status = _bet_log_status_totals(date_str)
    by_matchup, unmatched = _events_by_matchup(totals_items)
    suffix = game_suffixes(preds)
    dh = _doubleheader_matchups(preds)

    rows = []
    for _, g in preds.iterrows():
        a, h = g["away_team"], g["home_team"]
        sfx = suffix.get(g["game_pk"], "") + _game_flags(g)
        it = _select_event(by_matchup.get((a, h), []), g, require_time=(a, h) in dh)
        if it is None:
            unmatched.append(f"{a}@{h}{sfx} (no totals event matched by start time)")
            continue
        line = float(it["consensus_line"])
        book = next(iter(it["odds"]))
        offers = {o["side"]: int(o["odds"]) for o in it["odds"][book]}
        if "over" not in offers or "under" not in offers:
            unmatched.append(f"{a}@{h}{sfx} (incomplete two-way total)")
            continue

        fair_o, fair_u = remove_vig_two_way(
            american_to_prob(offers["over"]), american_to_prob(offers["under"])
        )
        sim = monte_carlo.simulate_game(
            g["expected_home_runs"], g["expected_away_runs"],
            n_sims=n_sims, overdispersion=g.get("overdispersion"),
        )
        over_p, push_p, under_p = monte_carlo.total_outcome_probs(
            sim["home_runs"], sim["away_runs"], line
        )
        no_push = over_p + under_p
        over_c = over_p / no_push if no_push else 0.5

        for side, model_p, mkt_p in (
            ("Over", over_c, fair_o),
            ("Under", 1 - over_c, fair_u),
        ):
            pick = f"{side} {line:g}"
            rows.append({
                "label": f"{a}@{h}{sfx} — {pick}",
                "team": pick,
                "model": float(model_p),
                "market": float(mkt_p),
                "edge": float(model_p - mkt_p),
                "best": offers[side.lower()],
                "push_prob": float(push_p),
                "status": status.get(pick.lower(), "pending"),
            })
    return pd.DataFrame(rows), unmatched


def run_totals(date_str=None, totals_items=None, n_sims=20000):
    date_str = date_str or dt.date.today().isoformat()
    preds = _load_or_build_predictions(date_str)
    if totals_items is None:
        from src.data import espn_odds
        items, _ = espn_odds.normalize_slate(date_str)
        totals_items = [i for i in items if i["offer_type"] == "total"]
    rows, unmatched = build_totals_rows(preds, totals_items, date_str, n_sims=n_sims)
    if unmatched:
        print("[!] games excluded from chart:")
        for u in unmatched:
            print("   ", u)
    out = config.PROCESSED_DIR / "play_probabilities_totals.pdf"
    n_push = int((rows["push_prob"] > 0).sum() // 2) if not rows.empty else 0
    render(
        rows, date_str, out, excluded=unmatched,
        title=(f"MLB model vs market — TOTALS (over/under), {date_str}\n"
               "Sorted by edge. Probabilities are conditional on no push. "
               "Reference only — this market has no validated edge."),
        footnote=("NOT A VALIDATED MARKET. The totals model was tested against 11,706 real closing "
                  "lines (2015-19): 50.25% ATS vs a 52.38% break-even, and recalibration did not fix it "
                  "(AUC 0.503 = no ability to rank games). Apparent edges here are expected to be noise. "
                  f"{n_push} game(s) on whole-number lines can push."),
    )
    return out, rows


def _excluded_note(excluded):
    """Name games that did not make the chart, ON the chart. A game silently
    absent is indistinguishable from a game that does not exist -- the same
    failure mode as the deduped doubleheader."""
    if not excluded:
        return ""
    return "  NOT SHOWN (" + str(len(excluded)) + "): " + "; ".join(excluded) + "."


def _marker_note(rows):
    """Explain the G1/G2, * and [FINAL] markers -- but only the ones actually
    present, so the footnote never describes something the reader can't see."""
    if rows.empty:
        return ""
    labels = " ".join(rows["label"])
    parts = []
    if "(G1" in labels or "(G2" in labels:
        parts.append("Doubleheaders are shown as SEPARATE rows per game (G1/G2 with start time); "
                     "each is its own simulation off that game's own probable starter")
    if "*" in labels:
        parts.append("* = probable starter not yet announced, so that side is modelled at league "
                     "average and the projection is softer than it appears")
    if "[FINAL]" in labels:
        parts.append("[FINAL] = game already complete; the price shown is not available")
    if "[LIVE]" in labels:
        parts.append("[LIVE] = game in progress; pregame price shown is stale")
    return ("  " + ". ".join(parts) + "." if parts else "")


def render(rows, date_str, out_path, title=None, footnote=None, excluded=None):
    # Fail soft when there is nothing to chart. An all-Final or line-less slate
    # leaves every game unmatched, so build_*_rows returns an empty, column-less
    # frame -- and sort_values("edge") on that used to raise KeyError and kill
    # the whole send. Emit a valid one-page "no lines available" PDF instead,
    # still naming the games that were dropped so the reader knows why.
    if rows is None or rows.empty or "edge" not in rows.columns:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.axis("off")
        ax.set_title(title or f"MLB model vs market, {date_str}", fontsize=11, loc="left")
        ax.text(0.5, 0.62, "No market lines available for this slate.",
                ha="center", va="center", fontsize=13, color="#616161", transform=ax.transAxes)
        ax.text(0.5, 0.46,
                "Every game was excluded — already final, in progress, or no line posted — "
                "so there is nothing to compare against a market number.",
                ha="center", va="center", fontsize=8.5, color="#9e9e9e",
                transform=ax.transAxes, wrap=True)
        fig.text(0.01, 0.005, (footnote or "") + _excluded_note(excluded),
                 fontsize=7, color="#616161", wrap=True)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, format="pdf")
        plt.close(fig)
        return out_path

    rows = rows.sort_values("edge").reset_index(drop=True)
    n = len(rows)
    fig_h = max(6.0, 0.34 * n + 2.6)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y = np.arange(n)
    bar_h = 0.36
    colour = {"win": SETTLED_WIN, "loss": SETTLED_LOSS}
    model_colours = [colour.get(s, PENDING) for s in rows["status"]]

    ax.barh(y + bar_h / 2, rows["model"] * 100, height=bar_h,
            color=model_colours, label="Model probability", zorder=3)
    ax.barh(y - bar_h / 2, rows["market"] * 100, height=bar_h,
            color=MARKET, label="Market implied (de-vigged)", zorder=3)

    for i, r in rows.iterrows():
        # Model win probability as an explicit number, drawn INSIDE its own bar
        # (white on the coloured fill). Inside rather than after the bar so it
        # never collides with the edge label sitting to the right of whichever
        # bar is longer. Very short bars can't hold the text, so those fall back
        # to dark text just outside.
        mp = r["model"] * 100
        if mp >= 14:
            ax.text(mp - 1.2, i + bar_h / 2, f"{mp:.1f}%", va="center", ha="right",
                    fontsize=7.5, color="white", fontweight="bold", zorder=4)
        else:
            ax.text(mp + 1.2, i + bar_h / 2, f"{mp:.1f}%", va="center", ha="left",
                    fontsize=7.5, color="#212121", fontweight="bold", zorder=4)

        x = max(r["model"], r["market"]) * 100
        sign = "+" if r["edge"] >= 0 else ""
        ax.text(x + 1.0, i, f"{sign}{r['edge']*100:.1f} pts  ({r['best']:+d})",
                va="center", fontsize=8,
                color="#212121" if r["edge"] >= 0 else "#757575")

    ax.axvline(50, color="#bdbdbd", ls="--", lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rows["label"], fontsize=8)
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-1, n)
    ax.set_title(
        title or (
            f"MLB model vs market — moneyline, {date_str}\n"
            "Sorted by edge. Label shows model-minus-market in points, and best available price."
        ),
        fontsize=11, loc="left",
    )
    ax.grid(axis="x", alpha=0.25, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        mpatches.Patch(color=PENDING, label="Model prob — pending"),
        mpatches.Patch(color=SETTLED_WIN, label="Model prob — settled WIN"),
        mpatches.Patch(color=SETTLED_LOSS, label="Model prob — settled LOSS"),
        mpatches.Patch(color=MARKET, label="Market implied (de-vigged consensus)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95)

    fig.text(0.01, 0.005,
             (footnote or (
                 "Moneyline only — the sole market validated out-of-sample (55.5% vs 53.0% naive). "
                 "Totals and run line are excluded: neither beat its baseline in backtesting."
             )) + _marker_note(rows) + _excluded_note(excluded),
             fontsize=7, color="#616161", wrap=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    return out_path


def _load_or_build_predictions(date_str):
    """Load predictions_{date}.csv, or generate it if missing.

    The scheduled 9am send assumes a predictions file already exists on disk.
    If an upstream step didn't run (or ran in an environment that didn't persist
    it), reading it would raise FileNotFoundError and kill the whole send. This
    makes the pipeline self-healing: a missing file is rebuilt from the same
    predict_date path the manual runs use, rather than being a hard failure.

    Lazy import of pipeline so this module stays cheap to import and to avoid any
    chance of an import cycle."""
    path = config.PROCESSED_DIR / f"predictions_{date_str}.csv"
    if path.exists():
        return pd.read_csv(path)
    print(f"[i] predictions_{date_str}.csv missing; generating it now.")
    from src import pipeline
    preds = pipeline.predict_date(date_str)
    if preds is None or preds.empty:
        raise ValueError(
            f"No predictions could be generated for {date_str} -- the schedule "
            "returned no MLB games. Cannot build the chart."
        )
    pipeline.save_predictions(preds, path)
    return preds


def _espn_fallback_events(date_str):
    """Moneyline events from ESPN's free public scoreboard, reshaped into the
    connector format build_rows expects (flat consensus_/best_ keys plus
    start_time_utc for doubleheader time-matching).

    ESPN is single-book (DraftKings), so consensus == best == that one price;
    downstream de-vig of a single two-way price is that book's fair number, not
    a market consensus, which is already how the ESPN totals path is labelled.

    Returns [] when ESPN has nothing usable for the date (e.g. every game has
    already started, so books have pulled the lines). Never raises -- a fallback
    that throws is worse than the failure it's meant to cover."""
    try:
        from src.data import espn_odds
        items, _ = espn_odds.normalize_slate(date_str)
    except Exception as e:
        print(f"[!] ESPN fallback fetch failed: {type(e).__name__}: {e}")
        return []
    events = []
    for it in items:
        if it.get("offer_type") != "moneyline":
            continue
        book = next(iter(it.get("odds") or {}), None)
        if not book:
            continue
        sides = {o["side"]: int(o["odds"]) for o in it["odds"][book]}
        if "home" not in sides or "away" not in sides:
            continue
        events.append({
            "away_team": it["away_team"], "home_team": it["home_team"],
            "start_time_utc": it.get("start_time_utc"),
            "consensus_home_ml": sides["home"], "consensus_away_ml": sides["away"],
            "best_home_ml": sides["home"], "best_away_ml": sides["away"],
            "source": it.get("source", "ESPN public scoreboard"),
        })
    return events


def run(date_str=None, events=None):
    date_str = date_str or dt.date.today().isoformat()
    preds = _load_or_build_predictions(date_str)
    if not events:
        # No connector events supplied, or the connector returned nothing (a
        # paywalled/failed MCP shows up here as an empty list). Fall back to
        # ESPN's free scoreboard so the scheduled send still produces a chart
        # instead of hard-failing -- the same free feed run_totals already uses.
        events = _espn_fallback_events(date_str)
        if events:
            print(f"[i] no connector events supplied; using ESPN free scoreboard fallback "
                  f"({len(events)} games).")
    if not events:
        raise ValueError(
            "No odds events available from the connector OR the ESPN fallback for "
            f"{date_str} -- cannot build the moneyline chart. (ESPN carries lines only "
            "for games that have not started; check the date and that a slate exists.)"
        )
    rows, unmatched = build_rows(preds, events, date_str)
    if unmatched:
        print('[!] games excluded from chart:')
        for u in unmatched:
            print('   ', u)
    out = config.PROCESSED_DIR / "play_probabilities.pdf"
    render(rows, date_str, out, excluded=unmatched)
    return out, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render the model-vs-market play chart to PDF.")
    ap.add_argument("date", nargs="?", default=None)
    ap.parse_args()
    print("Import and call run(date_str, events) with connector events.")
