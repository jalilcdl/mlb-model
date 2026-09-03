"""
Monte Carlo simulation engine -- the primary prediction method.

For each game, sample each team's runs from a distribution fit to their
current scoring rate (mu, from TeamRunRatings.predict_mus) -- Poisson by
default, or a negative binomial whose extra variance is estimated directly
from real historical team-game data (TeamRunRatings.overdispersion in
run_model.py) to match MLB's real, meaningfully overdispersed run
distributions (empirically ~2.2x the Poisson variance in our training data --
see README). Moneyline win probability, the total-runs distribution, and
run-line cover probability are then read directly off the simulated outcomes
(empirical frequencies) rather than computed from a closed-form formula.

We only have game-level scoring rates (Baseball-Reference doesn't give us
inning-by-inning box scores without a much heavier data pull), so games are
simulated at the whole-game level, not inning by inning -- see README.

src/models/run_model.py's closed-form Poisson/Skellam math is kept and run
alongside every simulation as a fast, independent cross-check: it assumes
pure Poisson (no overdispersion), so a persistent, large gap between the two
methods' total-runs/interval numbers is *expected* (that gap is exactly the
overdispersion correction), but a gap in moneyline win probability beyond
what sampling noise at the configured sim count would explain is worth
investigating.
"""
import numpy as np

from mlb_lib import config


def _sample_runs(mu, n_sims, overdispersion, distribution, rng):
    mu = max(mu, 0.05)
    if distribution == "poisson" or not overdispersion or overdispersion <= 1.0 + 1e-9:
        return rng.poisson(mu, n_sims)
    # Negative binomial matched to (mean=mu, variance=mu*overdispersion) via
    # numpy's (n, p) parameterization: mean = n(1-p)/p, variance = mean/p.
    p = 1.0 / overdispersion
    n_param = mu / (overdispersion - 1.0)
    return rng.negative_binomial(n_param, p, n_sims)


def total_over_prob(home_runs, away_runs, line):
    """Empirical P(total > line) from simulated run arrays. Works for any real
    line (e.g. from an odds snapshot), not just a fixed default.

    NB: P(under) is NOT 1 - this. On a whole-number line the total can land
    exactly on it and push -- use total_outcome_probs() when you need both
    sides."""
    return float(np.mean((home_runs + away_runs) > line))


def total_outcome_probs(home_runs, away_runs, line):
    """(P(over), P(push), P(under)) for a total line -- they sum to 1.

    Pushes are real money: on a whole-number line (8, 9, 10...) a total landing
    exactly on the number returns the stake, winning for neither side, and in
    MLB that's ~8-9% of the time. Treating P(under) as 1 - P(over) silently
    hands those pushes to the under, which inflates every whole-number under by
    that same 8-9 points and invents edges that aren't there. Half-point lines
    (8.5) can never push, so the two definitions agree there -- which is exactly
    why this is easy to miss."""
    total = home_runs + away_runs
    return (
        float(np.mean(total > line)),
        float(np.mean(total == line)),
        float(np.mean(total < line)),
    )


def cover_probability(team_a_runs, team_b_runs, line):
    """Empirical P(team_a covers `line` against team_b) from simulated run
    arrays. `line` is signed from team_a's perspective, same convention as
    run_model.cover_probability (-1.5 = team_a must win by 2+, +1.5 = team_a
    covers unless it loses by 2+)."""
    return float(np.mean((team_a_runs - team_b_runs) + line > 0))


def simulate_game(mu_home, mu_away, n_sims=None, overdispersion=None, distribution=None, run_line=None, seed=None):
    """Simulate one game n_sims times and return empirical probabilities plus
    the raw simulated run arrays (so callers -- e.g. the odds adapter matching
    an arbitrary sportsbook line, or the dashboard's distribution chart -- can
    derive further empirical probabilities without re-simulating)."""
    n_sims = n_sims or config.MC_DEFAULT_SIMS
    n_sims = int(np.clip(n_sims, config.MC_MIN_SIMS, config.MC_MAX_SIMS))
    distribution = distribution or config.MC_DISTRIBUTION
    run_line = run_line if run_line is not None else config.RUN_LINE
    rng = np.random.default_rng(seed)

    home_runs = _sample_runs(mu_home, n_sims, overdispersion, distribution, rng)
    away_runs = _sample_runs(mu_away, n_sims, overdispersion, distribution, rng)

    diff = home_runs - away_runs
    home_win = diff > 0
    tie_mask = diff == 0
    n_ties = int(tie_mask.sum())
    if n_ties:
        home_win = home_win.copy()
        home_win[tie_mask] = rng.random(n_ties) < config.EXTRA_INNING_HOME_WIN_PROB

    total = home_runs + away_runs
    home_win_prob = float(home_win.mean())

    return {
        "n_sims": n_sims,
        "distribution": distribution,
        "overdispersion": overdispersion,
        "mu_home": mu_home,
        "mu_away": mu_away,
        "home_runs": home_runs,
        "away_runs": away_runs,
        "home_win_prob": home_win_prob,
        "away_win_prob": 1.0 - home_win_prob,
        "tie_fraction": n_ties / n_sims,
        "expected_total": float(total.mean()),
        "median_total": float(np.median(total)),
        "total_std": float(total.std()),
        "home_covers_prob": cover_probability(home_runs, away_runs, -run_line),
        "away_covers_prob": cover_probability(away_runs, home_runs, run_line),
        "run_line": run_line,
    }
