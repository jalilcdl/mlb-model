"""
"Dispatch" -- a fast, plain-text rundown of the model's simulation output for a
day's games, meant to be read at a glance (in a terminal, or piped anywhere).

`build_rundown_text(preds, ...)` turns a predictions DataFrame into a compact
text block and is deliberately delivery-agnostic: it returns a plain string, so
the same rundown can be printed here, texted via SMS, emailed, or posted to a
scheduled job later without duplicating any formatting. `dispatch()` is the
convenience path that runs the pipeline for a date and hands back that text.

Run it:
    python -m src.dispatch            # today
    python -m src.dispatch 2026-07-16 # a specific date
    python -m src.dispatch --sims 3000
"""
import argparse
import datetime as dt

import pandas as pd

from src import config, pipeline


def _pct(x):
    return f"{x * 100:.0f}%" if pd.notna(x) else "n/a"


def _fmt_american(odds):
    if pd.isna(odds):
        return "n/a"
    odds = int(odds)
    return f"+{odds}" if odds > 0 else str(odds)


def _best_edge_line(g):
    """Return (label, text, ev) for the single most +EV bet in this game, or
    None if no odds are attached. Keeps the rundown focused on the one thing
    worth acting on rather than dumping every market."""
    if not g.get("odds_available"):
        return None
    candidates = [
        (f"{g['home_team']} ML", "edge_home_ml", "ev_home_ml_pct", "ml_home_best_odds", "ml_home_best_book"),
        (f"{g['away_team']} ML", "edge_away_ml", "ev_away_ml_pct", "ml_away_best_odds", "ml_away_best_book"),
        (f"Over {g.get('total_line')}", "edge_over", "ev_over_pct", "total_over_best_odds", "total_over_best_book"),
        (f"Under {g.get('total_line')}", "edge_under", "ev_under_pct", "total_under_best_odds", "total_under_best_book"),
        (f"{g['home_team']} {g.get('rl_home_line')}", "edge_home_rl", "ev_home_rl_pct", "rl_home_best_odds", "rl_home_best_book"),
        (f"{g['away_team']} +{g.get('run_line')}", "edge_away_rl", "ev_away_rl_pct", "rl_away_best_odds", "rl_away_best_book"),
    ]
    best = None
    for label, edge_col, ev_col, odds_col, book_col in candidates:
        ev = g.get(ev_col)
        edge = g.get(edge_col)
        if pd.isna(ev) or pd.isna(edge):
            continue
        if best is None or ev > best[2]:
            text = (
                f"{label} @ {_fmt_american(g.get(odds_col))} ({g.get(book_col)}): "
                f"{edge * 100:+.1f}% edge, {ev * 100:+.1f}% EV"
            )
            best = (label, text, ev)
    return best


def build_rundown_text(preds, date_str, n_sims=None):
    if preds is None or preds.empty:
        return f"MLB model rundown — {date_str}\nNo games scheduled."

    n_sims = int(preds["n_sims"].iloc[0]) if "n_sims" in preds.columns else (n_sims or config.MC_DEFAULT_SIMS)
    fetched = preds["pitchers_fetched_at"].dropna()
    fetched_note = ""
    if not fetched.empty:
        fetched_note = f" · pitchers pulled {pd.to_datetime(fetched).max():%H:%M}"

    lines = [
        f"MLB model rundown — {date_str}",
        f"{len(preds)} game(s) · {n_sims:,} sims/game{fetched_note}",
        "",
    ]

    for _, g in preds.iterrows():
        lines.append(f"{g['away_team']} @ {g['home_team']}  ({g.get('venue_name') or 'venue TBD'})")
        lines.append(
            f"  SP: {g.get('away_probable_pitcher') or 'TBD'} (away) vs "
            f"{g.get('home_probable_pitcher') or 'TBD'} (home)"
        )
        # Pitcher-change warnings -- the whole point of checking close to game time.
        for side, who in (("away", g["away_team"]), ("home", g["home_team"])):
            if g.get(f"{side}_pitcher_changed"):
                lines.append(
                    f"  ⚠ {who} SP changed since last check: was "
                    f"{g.get(f'{side}_pitcher_previous')}, now {g.get(f'{side}_probable_pitcher')}"
                )
        lines.append(
            f"  Moneyline: {g['away_team']} {_pct(g['away_win_prob'])} / "
            f"{g['home_team']} {_pct(g['home_win_prob'])}  → pick {g['moneyline_pick']}"
        )
        total_line_note = ""
        if g.get("odds_available") and pd.notna(g.get("total_line")):
            total_line_note = f"  (mkt O/U {g['total_line']:g})"
        lines.append(
            f"  Total: {g['expected_total']:.1f} runs (±{g['total_std']:.1f}){total_line_note}"
        )
        lines.append(f"  Run line: pick {g['run_line_pick']} ({_pct(g['run_line_pick_prob'])})")

        best = _best_edge_line(g)
        if best is not None:
            flag = "  ★ best edge: " if best[2] > 0 else "  best available: "
            lines.append(f"{flag}{best[1]}")
        elif not g.get("odds_available"):
            lines.append("  (no odds loaded — model numbers only)")
        lines.append("")

    lines.append(
        "Reminder: re-check probable pitchers close to first pitch; edges are a "
        "hypothesis to sanity-check, not a guarantee."
    )
    return "\n".join(lines)


def dispatch(date_str=None, n_sims=None, apply_odds=True):
    """Run the pipeline for a date and return the plain-text rundown string."""
    date_str = date_str or dt.date.today().isoformat()
    preds = pipeline.predict_date(date_str, apply_odds=apply_odds, n_sims=n_sims)
    return build_rundown_text(preds, date_str, n_sims=n_sims)


def main():
    parser = argparse.ArgumentParser(description="Print a plain-text rundown of the model's sim output.")
    parser.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--sims", type=int, default=None, help=f"simulations per game (default {config.MC_DEFAULT_SIMS})")
    parser.add_argument("--no-odds", action="store_true", help="skip odds/edge lookup")
    args = parser.parse_args()
    print(dispatch(date_str=args.date, n_sims=args.sims, apply_odds=not args.no_odds))


if __name__ == "__main__":
    main()
