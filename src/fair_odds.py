"""
Fair-odds ("true price") sheet for line shopping.

Instead of ingesting a book's odds and computing an edge, this inverts the
problem: the model publishes the fair, zero-vig price for every side, and you
shop for anything better. No odds feed, no API key, no scraping -- and it works
at whatever line/price any book happens to be offering.

How to read it: the fair price is EV-neutral *if the model is right*. Beat it
and you're +EV by the model's reckoning; take it exactly and you're breaking
even before vig. Because the model has real error, the sheet also prints a
"need better than" price carrying a margin (config default 2 points), which is
the number actually worth acting on.

Totals are printed as a LADDER across candidate lines rather than a single
number, because books differ on the line itself -- so you can look up the fair
price at whichever total you're being offered (e.g. 8.5 at one book, 9 at
another) instead of only the one line we happened to model.

Confidence differs sharply by market -- see the warnings printed with the
sheet and the README's Backtest section. Moneyline is the only market with a
validated out-of-sample edge; run-line "skill" was indistinguishable from a
naive baseline; totals have never been graded against real market lines.

    python -m src.fair_odds                 # today
    python -m src.fair_odds 2026-07-18
    python -m src.fair_odds --edge 3        # require 3 pts of margin
"""
import argparse
import datetime as dt

import numpy as np
import pandas as pd

from src import config, pipeline
from src.models import monte_carlo
from src.odds.odds_adapter import prob_to_american, required_price

DEFAULT_EDGE_PTS = 2.0
TOTALS_LADDER_SPAN = 1.5   # +/- runs around the model's expected total
TOTALS_LADDER_STEP = 0.5


def _fmt(odds):
    if odds is None:
        return "n/a"
    return f"+{int(odds)}" if odds > 0 else str(int(odds))


def _row(label, prob, edge_pts):
    return {
        "bet": label,
        "model_prob": prob,
        "fair_price": prob_to_american(prob),
        "need_better_than": required_price(prob, edge_pts),
    }


def totals_ladder(sim, expected_total, edge_pts=DEFAULT_EDGE_PTS,
                  span=TOTALS_LADDER_SPAN, step=TOTALS_LADDER_STEP):
    """Fair over/under prices across a range of candidate lines, from one
    simulation. Push probability is reported because on whole-number lines a
    push refunds the stake -- the fair price shown is conditional on the bet
    not pushing (which is what a book's two-way price is too)."""
    lo = np.floor((expected_total - span) * 2) / 2
    hi = np.ceil((expected_total + span) * 2) / 2
    rows = []
    line = lo
    while line <= hi + 1e-9:
        over_p, push_p, under_p = monte_carlo.total_outcome_probs(
            sim["home_runs"], sim["away_runs"], line
        )
        no_push = over_p + under_p
        if no_push > 0:
            oc, uc = over_p / no_push, under_p / no_push
            rows.append({
                "line": line, "push_prob": push_p,
                "over_prob": oc, "over_fair": prob_to_american(oc),
                "over_need": required_price(oc, edge_pts),
                "under_prob": uc, "under_fair": prob_to_american(uc),
                "under_need": required_price(uc, edge_pts),
            })
        line += step
    return rows


def build_sheet(date_str=None, n_sims=None, edge_pts=DEFAULT_EDGE_PTS, preds=None):
    date_str = date_str or dt.date.today().isoformat()
    if preds is None:
        preds = pipeline.predict_date(date_str, n_sims=n_sims, apply_odds=False)
    games = []
    for _, g in preds.iterrows():
        sim = monte_carlo.simulate_game(
            g["expected_home_runs"], g["expected_away_runs"],
            n_sims=int(g.get("n_sims") or config.MC_DEFAULT_SIMS),
            overdispersion=g.get("overdispersion"),
        )
        games.append({
            "matchup": f"{g['away_team']} @ {g['home_team']}",
            "start_utc": g.get("game_datetime_utc"),
            "pitchers": f"{g.get('away_probable_pitcher') or 'TBD'} vs {g.get('home_probable_pitcher') or 'TBD'}",
            "expected_total": g["expected_total"],
            "moneyline": [
                _row(f"{g['away_team']} ML", g["away_win_prob"], edge_pts),
                _row(f"{g['home_team']} ML", g["home_win_prob"], edge_pts),
            ],
            "run_line": [
                _row(f"{g['away_team']} +{g['run_line']:g}", g["away_covers_prob"], edge_pts),
                _row(f"{g['home_team']} -{g['run_line']:g}", g["home_covers_prob"], edge_pts),
            ],
            "totals": totals_ladder(sim, g["expected_total"], edge_pts=edge_pts),
        })
    return {"date": date_str, "edge_pts": edge_pts, "games": games}


def format_sheet(sheet):
    out = []
    out.append("=" * 72)
    out.append(f"FAIR ODDS / LINE-SHOPPING SHEET - {sheet['date']}")
    out.append(f"Fair = break-even if the model is right. 'Need' carries a "
               f"{sheet['edge_pts']:g}pt margin -- only bet if you can BEAT it.")
    out.append("=" * 72)
    if not sheet["games"]:
        out.append("No games scheduled.")
        return "\n".join(out)

    for g in sheet["games"]:
        out.append("")
        out.append(f"{g['matchup']}   ({g['pitchers']})")
        out.append(f"  model expected total: {g['expected_total']:.2f} runs")
        out.append(f"  {'bet':<18} {'model':>7} {'fair':>8} {'need better than':>18}")
        for r in g["moneyline"] + g["run_line"]:
            out.append(f"  {r['bet']:<18} {r['model_prob']*100:>6.1f}% {_fmt(r['fair_price']):>8} {_fmt(r['need_better_than']):>18}")
        out.append(f"  totals ladder (fair prices at each line):")
        out.append(f"    {'line':>5} {'push':>6} | {'over':>6} {'fair':>7} {'need':>7} | {'under':>6} {'fair':>7} {'need':>7}")
        for t in g["totals"]:
            out.append(
                f"    {t['line']:>5g} {t['push_prob']*100:>5.1f}% | "
                f"{t['over_prob']*100:>5.1f}% {_fmt(t['over_fair']):>7} {_fmt(t['over_need']):>7} | "
                f"{t['under_prob']*100:>5.1f}% {_fmt(t['under_fair']):>7} {_fmt(t['under_need']):>7}"
            )

    out.append("")
    out.append("-" * 72)
    out.append("CONFIDENCE VARIES BY MARKET (from the backtest -- read before acting):")
    out.append("  moneyline : the ONLY market with validated out-of-sample edge, and it's")
    out.append("              modest (~55.7% vs 53.0% naive). Fair prices here mean the most.")
    out.append("  run line  : model showed NO skill vs an 'always take the underdog +1.5'")
    out.append("              baseline (64.5% vs 64.4%). Treat these as reference only.")
    out.append("  totals    : NEVER graded against real market lines (no historical odds).")
    out.append("              Internally well-calibrated, but unproven vs the market.")
    out.append("  A big gap between the fair price and what books offer is usually the")
    out.append("  market knowing something the model doesn't -- not free money.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Print fair (true) odds for line shopping.")
    ap.add_argument("date", nargs="?", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--sims", type=int, default=None, help="simulations per game")
    ap.add_argument("--edge", type=float, default=DEFAULT_EDGE_PTS,
                    help=f"required edge margin in percentage points (default {DEFAULT_EDGE_PTS:g})")
    args = ap.parse_args()
    sheet = build_sheet(args.date, n_sims=args.sims, edge_pts=args.edge)
    print(format_sheet(sheet))


if __name__ == "__main__":
    main()
