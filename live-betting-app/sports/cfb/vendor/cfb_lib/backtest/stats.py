"""Shared bootstrap utilities for the market backtests."""
import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap on the mean of `values`. Returns (mean, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def american_odds_profit(odds: float, won: bool) -> float:
    """Profit per 1 unit staked at American odds `odds`, given the bet won or lost."""
    if not won:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def implied_prob_from_american(odds: float) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def remove_vig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    total = prob_a + prob_b
    return prob_a / total, prob_b / total
