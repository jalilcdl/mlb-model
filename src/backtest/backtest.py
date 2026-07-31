"""
Walk-forward backtest of the core Elo + Monte Carlo run-scoring model against
real historical results.

Scope, stated plainly (see README for the full limitations section):

  - We backtest the TEAM-LEVEL model only: Elo win probability + the Monte
    Carlo simulation built on team offense/defense/park ratings. The
    starting-pitcher adjustment used for live/upcoming predictions is
    isolated separately in src/backtest/pitcher_backtest.py, which compares
    it against this same team-level baseline on a like-for-like walk-forward
    basis -- see that module for whether the adjustment actually helps.

  - We do not have historical sportsbook closing lines in this build, so we
    cannot grade "would this have beaten the market" for the total
    (over/under) market -- that requires a real historical line to compare
    against, and we don't have one. What we CAN grade honestly without a
    market line:
      * Moneyline: pick accuracy, Brier score, log loss, calibration bins,
        and Expected Calibration Error (only need the actual winner, which
        we have).
      * Run line: pick accuracy and calibration against the *fixed*
        -1.5/+1.5 structure MLB uses (a rule of the sport, not a market
        number, so no odds data is needed to grade it).
      * Total: MAE/RMSE, predictive-interval coverage, and a proper
        randomized probability-integral-transform (PIT) calibration check
        -- the correct way to check whether a full predicted *distribution*
        (not just its mean) is calibrated for a discrete/count outcome. If
        the model's distribution is well calibrated, PIT values should be
        uniformly distributed on [0, 1]; the PIT calibration bins and the
        summary deviation-from-uniform number tell you if they aren't.

Every game is run through both the Monte Carlo engine (primary) and the
closed-form Poisson/Skellam math (cross-check); both are kept in the results
so a persistent disagreement between them is visible rather than discarded.
The Monte Carlo engine here uses fewer simulations than the live dashboard
default (config.BACKTEST_MC_SIMS vs config.MC_DEFAULT_SIMS) purely for
runtime -- thousands of games each need their own simulation.

Everything here is walk-forward: the run-model ratings used to predict a
given date are re-fit using only games strictly before that date, and Elo
ratings are inherently sequential (a game's prediction only uses state
built from earlier games). Results are also broken out by season
(summarize_by_season) so a real effect can be told apart from a one-season
fluke.
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.models import ensemble, monte_carlo
from src.models.elo import EloModel
from src.models.run_model import TeamRunRatings, game_probabilities


def _randomized_pit(sim_values, actual, rng):
    """Randomized probability-integral-transform value for a discrete
    predictive distribution: F(actual-1) + U*(F(actual)-F(actual-1)), U~Unif(0,1).
    Uniform on [0,1] under correct calibration; the non-randomized version
    (just F(actual)) is biased/lumpy for discrete outcomes and would
    understate miscalibration, so this is the statistically correct check."""
    n = len(sim_values)
    lo = float(np.sum(sim_values < actual)) / n
    hi = float(np.sum(sim_values <= actual)) / n
    return lo + rng.random() * (hi - lo)


def _walk_forward_run_model_predictions(games, eval_mask, n_sims=None, seed=None):
    n_sims = n_sims or config.BACKTEST_MC_SIMS
    rng = np.random.default_rng(seed)
    eval_games = games[eval_mask].copy()
    results = []
    for date, day_games in eval_games.groupby("date"):
        train = games[games["date"] < date]
        if train.empty:
            continue
        try:
            ratings = TeamRunRatings().fit(train)
        except ValueError:
            continue
        for row in day_games.itertuples(index=False):
            mu_home, mu_away = ratings.predict_mus(row.home_team, row.away_team)
            sim = monte_carlo.simulate_game(mu_home, mu_away, n_sims=n_sims, overdispersion=ratings.overdispersion)
            cf = game_probabilities(mu_home, mu_away)

            actual_total = row.home_score + row.away_score
            total_sim = sim["home_runs"] + sim["away_runs"]
            coverage = {}
            for interval in (0.5, 0.8):
                lo_q, hi_q = (1 - interval) / 2, 1 - (1 - interval) / 2
                lo, hi = np.quantile(total_sim, [lo_q, hi_q])
                coverage[f"covered_{int(interval * 100)}pct"] = bool(lo <= actual_total <= hi)
            pit = _randomized_pit(total_sim, actual_total, rng)

            results.append(
                {
                    "date": row.date,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "home_score": row.home_score,
                    "away_score": row.away_score,
                    "home_win": row.home_win,
                    "mc_home_win_prob": sim["home_win_prob"],
                    "cf_home_win_prob": cf["home_win_prob"],
                    "mc_expected_total": sim["expected_total"],
                    "cf_expected_total": cf["expected_total"],
                    "mc_home_covers_prob": sim["home_covers_prob"],
                    "cf_home_covers_prob": cf["home_covers_prob"],
                    "total_pit": pit,
                    **coverage,
                }
            )
    return pd.DataFrame(results)


def run_backtest(games=None, warmup_seasons=None, eval_seasons=None, n_sims=None, seed=None):
    games = games if games is not None else pd.read_csv(config.GAMES_FILE, parse_dates=["date"])
    games = games[games["home_score"] != games["away_score"]].copy()
    all_seasons = sorted(games["season"].unique())

    # Use as much data as available: one season of warm-up (needed for Elo
    # and the run ratings to stabilize before being evaluated), everything
    # else -- including the current, still-in-progress season -- evaluated.
    # Walk-forward fitting means including partial-season games in eval is
    # safe (each prediction still only uses strictly-prior games); it's not
    # safe to *train on* an incomplete season as if it were representative,
    # which is why it's fine here but is excluded from HISTORICAL_SEASONS
    # re-scrapes elsewhere.
    warmup_seasons = warmup_seasons or all_seasons[:1]
    eval_seasons = eval_seasons or [s for s in all_seasons if s not in warmup_seasons]

    elo = EloModel().fit(games)
    elo_eval = elo.history[elo.history["season"].isin(eval_seasons)].copy()

    eval_mask = games["season"].isin(eval_seasons)
    mc_eval = _walk_forward_run_model_predictions(games, eval_mask, n_sims=n_sims, seed=seed)
    if mc_eval.empty:
        raise ValueError("Not enough historical data to backtest the given eval_seasons.")

    merged = pd.merge(
        elo_eval,
        mc_eval,
        on=["date", "home_team", "away_team", "home_score", "away_score", "home_win"],
        how="inner",
    )
    merged["blended_home_win_prob"] = merged.apply(
        lambda r: ensemble.blend_win_prob(r["elo_win_prob_home"], r["mc_home_win_prob"]), axis=1
    )
    merged["actual_diff"] = merged["home_score"] - merged["away_score"]
    threshold = int(np.floor(config.RUN_LINE))
    merged["home_covered"] = (merged["actual_diff"] >= threshold + 1).astype(int)
    merged["actual_total"] = merged["home_score"] + merged["away_score"]

    return merged, eval_seasons, warmup_seasons


def _brier(probs, outcomes):
    return float(np.mean((probs - outcomes) ** 2))


def _log_loss(probs, outcomes, eps=1e-9):
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def _calibration_bins(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "n": int(mask.sum()),
                "mean_predicted": float(np.mean(probs[mask])),
                "actual_rate": float(np.mean(outcomes[mask])),
            }
        )
    return pd.DataFrame(rows)


def _expected_calibration_error(cal_df, total_n):
    """Weighted-average |mean_predicted - actual_rate| across bins -- a
    single number summarizing the calibration bins/curve (lower is better;
    0 = perfect calibration). Standard ECE definition."""
    if cal_df.empty or total_n == 0:
        return None
    return float(sum((row["n"] / total_n) * abs(row["mean_predicted"] - row["actual_rate"]) for _, row in cal_df.iterrows()))


def _pit_calibration_bins(pit_values, n_bins=10):
    counts, edges = np.histogram(pit_values, bins=n_bins, range=(0, 1))
    n = len(pit_values)
    rows = []
    for i in range(n_bins):
        rows.append(
            {
                "bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
                "n": int(counts[i]),
                "pct": float(counts[i] / n) if n else None,
                "expected_pct": 1.0 / n_bins,
            }
        )
    mean_abs_dev = float(np.mean([abs(r["pct"] - r["expected_pct"]) for r in rows])) if n else None
    return rows, mean_abs_dev


def summarize(merged):
    summary = {}
    n = len(merged)

    # Naive baselines -- essential context. A market-aware "accuracy" number is
    # easy to mistake for skill when it's really just base-rate imbalance (e.g.
    # MLB run-line underdogs cover ~65% of the time by the sport's own math, so
    # a model that mostly picks the underdog will look "accurate" without much
    # real signal). Always compare pick accuracy to these before trusting it.
    summary["baselines"] = {
        "always_pick_home_moneyline_accuracy": float(merged["home_win"].mean()),
        "always_pick_away_run_line_accuracy": float((1 - merged["home_covered"]).mean()),
    }

    for label, col in [
        ("elo", "elo_win_prob_home"),
        ("monte_carlo", "mc_home_win_prob"),
        ("closed_form", "cf_home_win_prob"),
        ("blended", "blended_home_win_prob"),
    ]:
        probs = merged[col].values
        outcomes = merged["home_win"].values
        summary[f"moneyline_{label}"] = {
            "accuracy": float(((probs >= 0.5).astype(int) == outcomes).mean()),
            "brier_score": _brier(probs, outcomes),
            "log_loss": _log_loss(probs, outcomes),
            "n_games": int(n),
        }
    ml_cal = _calibration_bins(merged["blended_home_win_prob"].values, merged["home_win"].values)
    summary["moneyline_calibration_blended"] = ml_cal.to_dict(orient="records")
    summary["moneyline_ece_blended"] = _expected_calibration_error(ml_cal, n)
    summary["mc_vs_closed_form_mean_abs_diff"] = float(
        (merged["mc_home_win_prob"] - merged["cf_home_win_prob"]).abs().mean()
    )

    rl_probs = merged["mc_home_covers_prob"].values
    rl_outcomes = merged["home_covered"].values
    summary["run_line"] = {
        "accuracy": float(((rl_probs >= 0.5).astype(int) == rl_outcomes).mean()),
        "brier_score": _brier(rl_probs, rl_outcomes),
        "log_loss": _log_loss(rl_probs, rl_outcomes),
        "n_games": int(n),
    }
    rl_cal = _calibration_bins(rl_probs, rl_outcomes)
    summary["run_line_calibration"] = rl_cal.to_dict(orient="records")
    summary["run_line_ece"] = _expected_calibration_error(rl_cal, n)

    for label, col in [("monte_carlo", "mc_expected_total"), ("closed_form", "cf_expected_total")]:
        err = merged[col] - merged["actual_total"]
        summary[f"total_runs_{label}"] = {
            "mae": float(err.abs().mean()),
            "rmse": float(np.sqrt((err**2).mean())),
            "mean_predicted_total": float(merged[col].mean()),
            "mean_actual_total": float(merged["actual_total"].mean()),
            "n_games": int(n),
        }
    # Interval coverage + PIT calibration use the Monte Carlo engine's own
    # empirical simulated distribution per game, not a normal/Poisson
    # approximation -- this is where negative-binomial overdispersion should
    # show up as coverage/PIT closer to nominal than the pure-Poisson
    # closed-form model would give.
    summary["total_runs_monte_carlo"]["coverage_50pct_interval"] = float(merged["covered_50pct"].mean())
    summary["total_runs_monte_carlo"]["coverage_80pct_interval"] = float(merged["covered_80pct"].mean())
    pit_bins, pit_dev = _pit_calibration_bins(merged["total_pit"].values)
    summary["total_runs_monte_carlo"]["pit_calibration_bins"] = pit_bins
    summary["total_runs_monte_carlo"]["pit_mean_abs_deviation_from_uniform"] = pit_dev

    return summary


def summarize_by_season(merged):
    return {str(int(season)): summarize(grp) for season, grp in merged.groupby("season")}


def run_and_save(games=None):
    merged, eval_seasons, warmup_seasons = run_backtest(games)
    config.BACKTEST_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(config.BACKTEST_RESULTS_FILE, index=False)
    summary = summarize(merged)
    summary["by_season"] = summarize_by_season(merged)
    summary["_meta"] = {
        "eval_seasons": [int(s) for s in eval_seasons],
        "warmup_seasons": [int(s) for s in warmup_seasons],
        "n_games": int(len(merged)),
        "mc_sims_per_game": config.BACKTEST_MC_SIMS,
    }
    with open(config.BACKTEST_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    return merged, summary


if __name__ == "__main__":
    _, summary = run_and_save()
    print(json.dumps({k: v for k, v in summary.items() if k != "by_season"}, indent=2))
