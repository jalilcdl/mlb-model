"""Combine the Elo win-probability signal with the Poisson run-model's
implied win probability into a single moneyline number.

Two independent views of the same game tend to be better calibrated blended
than either alone: Elo captures whole-season team strength trends (including
things the run model doesn't see directly, like bullpen quality trends and
recent form beyond the run-rate window), while the Poisson model is grounded
in this specific matchup's offense/defense/park/pitcher numbers. There's no
principled "correct" weight without a proper backtest-tuned blend, so we
default to a simple 50/50 average (config.ELO_BLEND_WEIGHT) and report both
components individually in the dashboard so the split is never hidden.
"""
from src import config


def blend_win_prob(elo_prob, poisson_prob, weight=None):
    weight = weight if weight is not None else config.ELO_BLEND_WEIGHT
    blended = weight * elo_prob + (1 - weight) * poisson_prob
    return min(max(blended, 0.01), 0.99)
