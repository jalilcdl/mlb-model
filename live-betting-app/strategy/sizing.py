"""Fractional-Kelly bet sizing. Pure, stateless functions -- no I/O, no log
reads, no bet placement. This computes a NUMBER; it never sends anything
anywhere. See strategy/config.py for the tunable thresholds this applies.

Kelly needs the REAL bettable price, not the de-vigged fair probability the
signal layer uses to detect an edge in the first place. The de-vigged number
is what makes "model vs market" an apples-to-apples comparison; the vig is
real money the book keeps on top of that, and Kelly must be sized against
what you're actually paid if you win. A positive de-vigged edge can still
correspond to a non-positive Kelly fraction at the real price -- the vig can
eat the whole edge. That is a real, expected outcome here, not a bug to
paper over: size_bet() reports it as "no_bet" with a reason, rather than
silently clamping a negative f* to zero.
"""
from __future__ import annotations

from strategy import config as default_config


def implied_decimal_odds(american_odds: int) -> float:
    """American -> decimal odds. -110 -> 1.909..., +150 -> 2.5."""
    if american_odds > 0:
        return 1.0 + (american_odds / 100.0)
    return 1.0 + (100.0 / abs(american_odds))


def kelly_fraction(p: float, american_odds: int) -> float:
    """f* = (p*d - 1) / (d - 1), d = decimal odds, for a single binary bet.

    p is the model's probability the PICKED side wins (already the
    pick-side-adjusted number, not necessarily model_home_wp -- if the pick
    is away, the caller passes 1 - model_home_wp).
    """
    d = implied_decimal_odds(american_odds)
    b = d - 1.0
    return (p * d - 1.0) / b


def size_bet(edge: float, p: float, american_odds: int, bankroll: float,
             cfg=default_config) -> dict:
    """One sizing decision for one flagged, priced signal.

    edge: the de-vigged |model - market| gap already computed by the signal
          layer (row["edge"]).
    p: model's win probability for the PICKED side (pick-side-adjusted).
    american_odds: the real quoted price for the picked side (row
          ["pick_odds_american"]).
    bankroll: current bankroll to size against.

    Returns {action, stake, reason, kelly_raw, fraction_applied} where
    action in {"bet", "no_bet", "review"}. "review" means the edge is large
    enough to be suspicious (stale/illiquid line) rather than confidently
    actionable -- shown to Jalil, not auto-sized.
    """
    if american_odds is None:
        return {"action": "no_bet", "stake": 0.0, "reason": "no priced side",
                "kelly_raw": None, "fraction_applied": 0.0}

    if abs(edge) > cfg.EDGE_SANITY_CEILING:
        return {"action": "review", "stake": 0.0,
                "reason": f"edge {edge:.1%} exceeds sanity ceiling "
                          f"{cfg.EDGE_SANITY_CEILING:.0%} -- likely stale/illiquid line",
                "kelly_raw": None, "fraction_applied": 0.0}

    if abs(edge) < cfg.MIN_EDGE_TO_SIZE:
        return {"action": "no_bet", "stake": 0.0,
                "reason": f"edge {edge:.1%} below sizing threshold {cfg.MIN_EDGE_TO_SIZE:.0%}",
                "kelly_raw": None, "fraction_applied": 0.0}

    f_star = kelly_fraction(p, american_odds)
    if f_star <= 0:
        return {"action": "no_bet", "stake": 0.0,
                "reason": "no edge at the real (vig-included) price, despite de-vigged edge",
                "kelly_raw": round(f_star, 4), "fraction_applied": 0.0}

    f_applied = min(cfg.KELLY_FRACTION * f_star, cfg.MAX_BET_PCT_OF_BANKROLL)
    return {
        "action": "bet",
        "stake": round(f_applied * bankroll, 2),
        "reason": f"{cfg.KELLY_FRACTION:.0%} Kelly"
                  + (" (capped)" if cfg.KELLY_FRACTION * f_star > cfg.MAX_BET_PCT_OF_BANKROLL else ""),
        "kelly_raw": round(f_star, 4),
        "fraction_applied": round(f_applied, 4),
    }


def pick_side_probability(row: dict) -> float | None:
    """model_home_wp, adjusted to the picked side's own win probability.
    None if nothing's flagged/picked on this row."""
    if not row.get("flagged") or row.get("pick_team") is None:
        return None
    is_home_pick = row["pick_team"] == row["home_team"]
    return row["model_home_wp"] if is_home_pick else 1.0 - row["model_home_wp"]
