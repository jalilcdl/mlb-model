"""Backtest harness: turn logged signal-detection history into "would this
strategy have made money" evidence, before it's trusted as a live
recommendation feed.

Run: python -m strategy.backtest   (from live-betting-app/)

This is a read-only analysis script. It reads data/live_signal_log.jsonl and
calls each sport's free outcome-lookup API (see outcomes.py); it never
writes to the log, never calls the poller, and the poller never imports
anything from this package -- a bug here cannot affect live signal logging
or Telegram pushes.

Two different questions, kept separate on purpose:
  1. CALIBRATION -- is the model's win-probability trustworthy at all? Over
     every flagged, resolved bet-opportunity, regardless of whether it was
     ever sized. (Not "CLV" in the pregame-closing-line sense -- there's no
     clean closing-line concept mid-game; this is the closest honest analog:
     bucket by predicted probability, compare to realized win rate.)
  2. STRATEGY PERFORMANCE -- would fractional-Kelly sizing (strategy/sizing.py)
     have made money? Only over bet-opportunities that actually got sized
     ("action": "bet"), walked in chronological order against a simulated
     bankroll.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import storage
from strategy import config, outcomes, sizing

LOG_PATH = ROOT / "data" / "live_signal_log.jsonl"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def load_rows() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    rows = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["logged_at_utc"])
    return rows


def _legacy_pick_odds(row: dict) -> int | None:
    """Rows logged before pick_odds_american existed. MLB always logged raw
    home_odds/away_odds, so it's recoverable; older CFB rows have no raw
    price at all and stay unpriced."""
    if row.get("pick_odds_american") is not None:
        return row["pick_odds_american"]
    if row["sport"] == "mlb" and "home_odds" in row and "away_odds" in row:
        return row["home_odds"] if row["pick_team"] == row["home_team"] else row["away_odds"]
    return None


def find_bet_opportunities(rows: list[dict]) -> list[dict]:
    """One entry per (game_id, pick_side) bet opportunity, using the EXACT
    same dedup/cooldown rule as the live Telegram notifier
    (core.storage.should_notify/record_notified) -- so the backtest's bet
    count matches what Jalil would actually have been notified about live,
    not an inflated per-snapshot count."""
    notified_state: dict = {}
    opportunities = []
    for row in rows:
        if not row.get("flagged") or not row.get("pick_team"):
            continue
        now_ts = dt.datetime.fromisoformat(row["logged_at_utc"]).timestamp()
        sport, game_id, pick_team = row["sport"], row["game_id"], row["pick_team"]
        if storage.should_notify(notified_state, sport, game_id, pick_team, now_ts):
            opportunities.append(row)
            storage.record_notified(notified_state, sport, game_id, pick_team, now_ts)
    return opportunities


def grade_opportunity(row: dict, outcome: dict) -> bool | None:
    """Did the picked side actually win? None if unresolved/push."""
    if not outcome or not outcome["completed"] or outcome["home_won"] is None:
        return None
    picked_home = row["pick_team"] == row["home_team"]
    return outcome["home_won"] if picked_home else not outcome["home_won"]


def run_backtest(bankroll_start: float = config.DEFAULT_BANKROLL) -> dict:
    rows = load_rows()
    opportunities = find_bet_opportunities(rows)

    outcome_cache: dict[tuple, dict | None] = {}
    graded = []
    for row in opportunities:
        key = (row["sport"], row["game_id"])
        if key not in outcome_cache:
            outcome_cache[key] = outcomes.resolve_outcome(*key)
        outcome = outcome_cache[key]
        hit = grade_opportunity(row, outcome)
        if hit is None:
            continue  # not final yet, or push -- excluded, not counted as a loss
        p = sizing.pick_side_probability(row)
        odds = _legacy_pick_odds(row)
        decision = sizing.size_bet(row["edge"], p, odds, bankroll_start)
        graded.append({**row, "hit": hit, "p": p, "pick_odds_used": odds, **decision})

    return {"rows_total": len(rows), "opportunities_total": len(opportunities),
            "graded": graded}


def _calibration_table(graded: list[dict]) -> list[dict]:
    bands = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
             (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    table = []
    for lo, hi in bands:
        in_band = [g for g in graded if g["p"] is not None and lo <= g["p"] < hi]
        if not in_band:
            continue
        realized = sum(1 for g in in_band if g["hit"]) / len(in_band)
        table.append({"band": f"{lo:.0%}-{hi:.0%}", "n": len(in_band),
                     "predicted_avg": sum(g["p"] for g in in_band) / len(in_band),
                     "realized_win_rate": realized})
    return table


def _bankroll_curve(graded: list[dict], bankroll_start: float) -> list[float]:
    bankroll = bankroll_start
    curve = [bankroll]
    for g in graded:
        if g["action"] != "bet":
            continue
        stake = g["stake"]
        if g["hit"]:
            d = sizing.implied_decimal_odds(g["pick_odds_used"])
            bankroll += stake * (d - 1)
        else:
            bankroll -= stake
        curve.append(bankroll)
    return curve


def print_report(result: dict, bankroll_start: float = config.DEFAULT_BANKROLL) -> None:
    graded = result["graded"]
    n = len(graded)
    print("=" * 70)
    print(f"BACKTEST REPORT -- {result['rows_total']} log rows, "
          f"{result['opportunities_total']} bet opportunities, {n} resolved")
    print("=" * 70)

    if n < config.MIN_BETS_FOR_CONFIDENCE:
        print(f"\n*** Only {n} resolved bet opportunit{'y' if n == 1 else 'ies'} so far "
              f"(need {config.MIN_BETS_FOR_CONFIDENCE}+ to say anything meaningful). ***")
        print("*** Numbers below are a preview of the mechanics, NOT a validated result. ***\n")

    if n == 0:
        print("Nothing resolved yet -- keep the poller running and re-run this later.")
        return

    hits = sum(1 for g in graded if g["hit"])
    print(f"\nOverall hit rate (all flagged opportunities, any sizing action): "
          f"{hits}/{n} = {hits/n:.1%}")
    for sport in sorted(set(g["sport"] for g in graded)):
        sg = [g for g in graded if g["sport"] == sport]
        sh = sum(1 for g in sg if g["hit"])
        print(f"  {sport}: {sh}/{len(sg)} = {sh/len(sg):.1%}" if sg else f"  {sport}: n/a")

    print("\nCalibration (predicted win% band vs realized win rate):")
    for row in _calibration_table(graded):
        print(f"  {row['band']:>10}  n={row['n']:<4} predicted avg {row['predicted_avg']:.1%}  "
              f"realized {row['realized_win_rate']:.1%}")

    sized = [g for g in graded if g["action"] == "bet"]
    reviewed = [g for g in graded if g["action"] == "review"]
    skipped = [g for g in graded if g["action"] == "no_bet"]
    print(f"\nSizing actions: {len(sized)} bet, {len(skipped)} no_bet (below threshold/"
          f"no edge at real price), {len(reviewed)} review (suspiciously large edge)")

    if sized:
        curve = _bankroll_curve(graded, bankroll_start)
        pnl = curve[-1] - bankroll_start
        wins = sum(1 for g in sized if g["hit"])
        print(f"\nSimulated strategy performance ({len(sized)} actually-sized bets, "
              f"bankroll start ${bankroll_start:,.0f}):")
        print(f"  Sized-bet hit rate: {wins}/{len(sized)} = {wins/len(sized):.1%}")
        print(f"  Ending bankroll: ${curve[-1]:,.2f}  (P&L: {'+' if pnl >= 0 else ''}{pnl:,.2f})")

        REPORTS_DIR.mkdir(exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(curve, marker="o")
            plt.axhline(bankroll_start, color="gray", linestyle="--", linewidth=1)
            plt.title("Simulated bankroll (fractional Kelly)")
            plt.xlabel("bet #")
            plt.ylabel("bankroll ($)")
            out = REPORTS_DIR / "bankroll_curve.png"
            plt.tight_layout()
            plt.savefig(out)
            print(f"  Bankroll curve saved -> {out}")
        except ImportError:
            print("  (matplotlib not installed -- skipping chart; "
                  "pip install -r requirements-strategy.txt)")
    else:
        print("\nNo bets actually got sized yet (all below threshold, unpriced, or "
              "no edge at the real price) -- nothing to simulate a bankroll curve from.")


if __name__ == "__main__":
    result = run_backtest()
    print_report(result)
