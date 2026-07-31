"""
Forward-test bet tracker: a dead-simple record of the picks you actually acted
on, so the model earns (or loses) a real track record over time instead of
being judged on vibes or on the in-sample backtest.

The log is a plain CSV (config.BET_LOG_FILE) you edit by hand -- add a row when
you place a bet, fill in `result` when the game ends. Columns:

    date, matchup, market, pick, odds, model_prob, implied_prob,
    edge_pct, stake, result, notes

Single source of truth: `implied_prob` and `edge_pct` are DERIVED from `odds`
and `model_prob`. This module recomputes them on load, so you never have to do
the arithmetic -- if you change the odds you took, just leave those two columns
alone (or blank) and the summary stays correct. The columns are kept in the CSV
only so the file is readable on its own.

Conventions (see README "Bet tracker" section):
  - odds: American (e.g. -116, +150).
  - model_prob: the model's probability for the side you bet, at bet time.
    For moneyline that's the blended Elo+MC number; for totals/run line it's
    the pure Monte Carlo probability (those markets aren't blended). You record
    it; nothing recomputes it, since the model's inputs move day to day.
  - implied_prob: RAW break-even probability from the odds (includes vig). This
    is the honest bar the bet has to clear to profit, and what ROI is measured
    against -- not the de-vigged consensus (that's a market-disagreement metric,
    which is a different question than "did this bet make money").
  - edge_pct = (model_prob - implied_prob) * 100, in percentage points.
  - result: win / loss / push, or blank for a bet not yet settled (pending).
  - stake: in UNITS (1u flat), defaulting to 1.0 if left blank. Deliberately
    unit-normalized rather than dollar-denominated -- what matters for judging
    the model is return per unit risked, not how much was on it that day.
    Change a row's stake only if you genuinely sized it differently.
  - cost / payout: OPTIONAL real dollar amounts actually paid and received.

REALIZED vs QUOTED accounting (the `basis` column, derived):
  If a row has both `cost` and `payout`, profit is computed from the REAL money
  -- (payout - cost) / cost -- so exchange/book fees, partial fills, and an
  average price that differed from the quoted one are all captured. This is
  what actually landed in the account.

  If those are blank, profit falls back to the QUOTED price in `odds`, which
  silently assumes zero fees and a perfect fill. On Kalshi that has measured
  ~6-7% of the profit on a winning bet, so a log full of quoted wins reads
  optimistically. Rows are flagged so the two are never confused, and the
  summary reports how much of the record rests on each.

  Note losses are identical under both methods (-1 unit either way); the
  distinction only moves winners.

Venues: entries may come from traditional sportsbooks or from Kalshi-style
binary contracts. Kalshi prices ARE probabilities (a 56c contract = 56%
implied), so they're converted to the American-odds equivalent on entry and
the original price is kept in `notes`; the economics are the same bet.

CLOSING LINE VALUE (CLV) -- the `closing_odds` column:
  CLV compares the price you got to the CLOSING price on the same side (the
  final line right before first pitch). Consistently getting a better number
  than the close is the single most reliable predictor of long-term
  profitability -- more so than short-run win/loss, because the closing line
  is the sharpest estimate the market produces. You can be a long-term winner
  and still lose this week; positive CLV says you're beating the number even
  when variance hides it.

  Fill in `closing_odds` (American, same side you bet) once a game finishes.
  Derived: clv_pct = decimal(your odds)/decimal(closing odds) - 1. Positive
  means you got a better price than the close (you "beat the close"). Rows
  without a closing price show CLV as N/A rather than a guessed value.
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.odds.odds_adapter import american_to_decimal

_SETTLED = {"win", "loss", "push"}
COLUMNS = [
    "date", "matchup", "market", "pick", "odds", "model_prob",
    "implied_prob", "edge_pct", "stake", "result", "cost", "payout",
    "closing_odds", "notes",
]


def implied_prob(american_odds):
    """Raw break-even probability implied by American odds (vig included)."""
    return 1.0 / american_to_decimal(american_odds)


def profit_on_settled(odds, result, stake):
    """Profit for one settled bet at flat `stake`. Win pays the odds, loss
    loses the stake, push returns it (0). None for anything not yet settled."""
    r = str(result).strip().lower()
    if r == "win":
        return stake * (american_to_decimal(odds) - 1.0)
    if r == "loss":
        return -stake
    if r == "push":
        return 0.0
    return None


def load_log(path=None):
    """Load the bet log, normalize types, and recompute the derived columns
    (implied_prob, edge_pct) so they're always consistent with odds/model_prob."""
    path = path or config.BET_LOG_FILE
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["model_prob"] = pd.to_numeric(df["model_prob"], errors="coerce")
    df["stake"] = pd.to_numeric(df["stake"], errors="coerce").fillna(1.0)
    df["result"] = df["result"].astype("string").str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA})

    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["payout"] = pd.to_numeric(df["payout"], errors="coerce")
    df["closing_odds"] = pd.to_numeric(df["closing_odds"], errors="coerce")

    # Derived -- always recomputed, never trusted from the file.
    df["implied_prob"] = df["odds"].apply(lambda o: implied_prob(o) if pd.notna(o) else np.nan)
    df["edge_pct"] = (df["model_prob"] - df["implied_prob"]) * 100.0
    df["basis"] = np.where(
        df["cost"].notna() & df["payout"].notna() & (df["cost"] > 0), "realized", "quoted"
    )
    df["profit"] = df.apply(_row_profit, axis=1)
    df["settled"] = df["result"].isin(_SETTLED)
    # Closing Line Value: how your price compares to the closing price (same side).
    df["clv_pct"] = df.apply(_row_clv, axis=1)
    df["beat_close"] = df["clv_pct"].apply(lambda x: bool(x > 0) if pd.notna(x) else pd.NA)
    return df


def _row_clv(r):
    """CLV% = decimal(your odds) / decimal(closing odds) - 1, on the side you
    bet. Positive means you got a better number than the close. N/A without a
    closing price -- never guessed."""
    if pd.isna(r.get("odds")) or pd.isna(r.get("closing_odds")):
        return np.nan
    return (american_to_decimal(r["odds"]) / american_to_decimal(r["closing_odds"]) - 1.0) * 100.0


def _row_profit(r):
    """Realized profit from real dollars when available (captures fees and
    fill quality), else the quoted-price calculation."""
    if str(r["result"]).strip().lower() not in _SETTLED:
        return None
    if r["basis"] == "realized":
        return (r["payout"] - r["cost"]) / r["cost"] * r["stake"]
    return profit_on_settled(r["odds"], r["result"], r["stake"])


def _metrics(df):
    """Record / win-rate / ROI for a set of rows. Pushes count as settled but
    are excluded from win rate and from staking turnover (stake returned, no
    action). Pending bets are excluded from everything but the pending count."""
    settled = df[df["settled"]]
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())
    pushes = int((settled["result"] == "push").sum())
    pending = int(len(df) - len(settled))

    decided = wins + losses
    win_rate = wins / decided if decided else np.nan

    turnover = settled.loc[settled["result"] != "push", "stake"].sum()
    profit = settled["profit"].sum()
    roi = profit / turnover if turnover else np.nan

    graded = settled[settled["result"] != "push"]
    avg_model = graded["model_prob"].mean() if len(graded) else np.nan
    avg_edge = settled["edge_pct"].mean() if len(settled) else np.nan

    realized = settled[settled["basis"] == "realized"] if "basis" in settled.columns else settled.iloc[0:0]
    quoted_wins = int(((settled["basis"] == "quoted") & (settled["result"] == "win")).sum()) if "basis" in settled.columns else 0

    # CLV is outcome-independent: computed over every bet that has a closing
    # price, won/lost/pending alike.
    clv = df["clv_pct"].dropna() if "clv_pct" in df.columns else pd.Series(dtype=float)
    n_clv = int(len(clv))
    avg_clv = float(clv.mean()) if n_clv else np.nan
    beat_rate = float((clv > 0).mean()) if n_clv else np.nan

    return {
        "bets_total": int(len(df)),
        "wins": wins, "losses": losses, "pushes": pushes, "pending": pending,
        "win_rate": win_rate,
        "turnover": float(turnover),
        "profit": float(profit),
        "roi": roi,
        "avg_model_prob": avg_model,   # vs win_rate = a crude calibration check
        "avg_edge_pct": avg_edge,
        "n_realized": int(len(realized)),
        "n_quoted": int(len(settled) - len(realized)),
        "quoted_wins": quoted_wins,    # the rows whose profit is fee-blind
        "n_with_clv": n_clv,
        "n_without_clv": int(len(df) - n_clv),
        "avg_clv_pct": avg_clv,
        "beat_close_rate": beat_rate,
    }


def summarize(df=None):
    df = load_log() if df is None else df
    summary = {"overall": _metrics(df), "by_market": {}}
    for market, grp in df.groupby("market"):
        summary["by_market"][str(market)] = _metrics(grp)
    return summary


def _fmt_pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def _fmt_units(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:+.2f}u"


def format_summary(df=None):
    df = load_log() if df is None else df
    if df.empty:
        return "No bets logged yet. Add rows to " + str(config.BET_LOG_FILE)

    s = summarize(df)
    o = s["overall"]
    lines = []
    lines.append("=" * 60)
    lines.append("BET TRACKER - forward-test record (1u flat stakes)")
    lines.append("=" * 60)
    lines.append(
        f"Record: {o['wins']}-{o['losses']}"
        + (f"-{o['pushes']} (W-L-P)" if o['pushes'] else " (W-L)")
        + f"   |   {o['pending']} pending"
    )
    lines.append(f"Win rate (decided): {_fmt_pct(o['win_rate'])}  ({o['wins']}/{o['wins'] + o['losses']})")
    lines.append(f"Staked (turnover):  {o['turnover']:.2f}u")
    lines.append(f"Net profit:         {_fmt_units(o['profit'])}")
    lines.append(f"ROI:                {_fmt_pct(o['roi'])}")
    lines.append(f"Avg model prob vs actual win rate: {_fmt_pct(o['avg_model_prob'])} vs {_fmt_pct(o['win_rate'])}")
    lines.append(
        f"Accounting basis:   {o['n_realized']} realized (real $, fees included) / "
        f"{o['n_quoted']} quoted (fee-blind)"
    )
    if o["quoted_wins"]:
        lines.append(
            f"                    ^ {o['quoted_wins']} WIN(S) still priced off quoted odds -- "
            "their profit is optimistic by roughly the fee drag."
        )
    lines.append("")
    lines.append("Closing Line Value (best long-run predictor of profitability):")
    if o["n_with_clv"]:
        lines.append(
            f"  avg CLV: {o['avg_clv_pct']:+.2f}%  |  beat the close: {_fmt_pct(o['beat_close_rate'])} "
            f"of bets  ({o['n_with_clv']} with a closing price, {o['n_without_clv']} without)"
        )
        lines.append(
            "  (positive avg CLV = you're consistently getting a better number than the market's "
            "final line -- the signal that survives even when short-run W/L doesn't)"
        )
    else:
        lines.append(
            f"  no closing prices logged yet ({o['n_without_clv']} bets). Fill the `closing_odds` "
            "column (same side, price at first pitch) to start tracking CLV."
        )
    lines.append("")
    lines.append("By market:")
    header = f"  {'market':<12} {'W-L-P':>9} {'win%':>7} {'profit':>11} {'ROI':>8}  pend"
    lines.append(header)
    for market, m in sorted(s["by_market"].items()):
        wlp = f"{m['wins']}-{m['losses']}-{m['pushes']}"
        lines.append(
            f"  {market:<12} {wlp:>9} {_fmt_pct(m['win_rate']):>7} "
            f"{_fmt_units(m['profit']):>11} {_fmt_pct(m['roi']):>8}  {m['pending']}"
        )
    lines.append("")
    n_settled = o["wins"] + o["losses"] + o["pushes"]
    if n_settled < 30:
        lines.append(
            f"[!] Only {n_settled} settled bet(s). This is FAR too small to conclude anything -- "
            "a positive ROI here is noise, not proven edge. Meaningful read needs ~50-100+ bets."
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Show the bet-tracking forward-test summary.")
    parser.add_argument("--table", action="store_true", help="also print the full bet log")
    args = parser.parse_args()

    df = load_log()
    if args.table and not df.empty:
        show = df[["date", "matchup", "market", "pick", "odds", "model_prob", "implied_prob",
                   "edge_pct", "result", "profit", "closing_odds", "clv_pct"]].copy()
        show["model_prob"] = (show["model_prob"] * 100).round(1)
        show["implied_prob"] = (show["implied_prob"] * 100).round(1)
        show["edge_pct"] = show["edge_pct"].round(1)
        show["clv_pct"] = show["clv_pct"].round(2)
        print(show.to_string(index=False))
        print()
    print(format_summary(df))


if __name__ == "__main__":
    main()
