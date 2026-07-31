"""
Isolates whether the starting-pitcher adjustment used in live predictions
(TeamRunRatings.pitcher_factor, blended into mu via predict_mus) actually
improves predictions versus the team-level-only baseline -- on a proper,
walk-forward, no-lookahead holdout.

This directly reuses the production pitcher_factor() method (same ERA/FIP
blend, same 15-IP minimum, same clipping) rather than reimplementing it, so
"isolate the pitcher adjustment" means exactly the mechanism that ships in
src/models/run_model.py and src/pipeline.py -- not an idealized version of it.

Data and no-lookahead discipline:
  - Starters come from data/processed/historical_starters.csv: the MLB Stats
    API's probable-pitcher field for each historical game, pulled from the
    same endpoint live predictions use for today's games. This is genuinely
    the pitcher information available before each game (not a box-score
    reconstruction), so there's no "the model knew who actually started and
    pitched well" leakage.
  - Each starter's stats are recomputed *as of* the day before their game,
    from data/processed/pitcher_game_logs.csv (summing only that pitcher's
    games strictly before the target date, within the same season -- matching
    exactly what get_pitcher_season_stats() would have returned live on that
    day). A pitcher's first few starts of a season have no current-season
    log yet and fall back to no adjustment (home_pitcher_factor=None), same
    as live behavior for a rookie/new-to-team pitcher.
  - Team ratings (offense/defense/park/overdispersion) come from the same
    walk-forward TeamRunRatings fit used in backtest.py, trained only on
    games strictly before the target date.
  - The pitcher_factor() formula itself is a fixed blend of ERA/FIP ratios
    with hand-set weights (config.py) -- it is not fit/learned from data, so
    there's no separate "train the adjustment, then test it" split needed
    the way there would be for a fitted parameter; the walk-forward,
    prior-games-only stat computation is the relevant safeguard, and it's
    applied throughout.

Both engines get the SAME random draws per game (a shared seed for the
baseline and pitcher-adjusted Monte Carlo runs) so any measured difference
reflects the change in mu from the pitcher adjustment, not incidental
simulation noise -- a standard variance-reduction technique for paired
comparisons.
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.models import monte_carlo
from src.models.run_model import TeamRunRatings

def _load_inputs():
    games = pd.read_csv(config.GAMES_FILE, parse_dates=["date"])
    games = games[games["home_score"] != games["away_score"]].copy()
    starters = pd.read_csv(config.HISTORICAL_STARTERS_FILE, parse_dates=["date"])
    starters = starters.dropna(subset=["home_pitcher_id", "away_pitcher_id"]).copy()
    starters["home_pitcher_id"] = starters["home_pitcher_id"].astype(int)
    starters["away_pitcher_id"] = starters["away_pitcher_id"].astype(int)
    logs = pd.read_csv(config.PITCHER_GAME_LOGS_FILE, parse_dates=["date"])
    return games, starters, logs


def _pair_doubleheaders(games, starters):
    """Both games.csv and historical_starters.csv can have >1 row for the
    same (date, home, away) on doubleheader days. Pair them in the order
    each source lists them (both ultimately follow MLB's official schedule
    order for that date) rather than dropping them or letting a naive merge
    produce a spurious cross-product."""
    key = ["date", "home_team", "away_team"]
    g = games.copy()
    s = starters.copy()
    g["_dupe_idx"] = g.groupby(key).cumcount()
    s["_dupe_idx"] = s.groupby(key).cumcount()
    return pd.merge(g, s, on=key + ["_dupe_idx"], how="inner")


def _build_pitcher_index(logs):
    """{(pitcher_id, season): DataFrame sorted by date} for fast as-of-date slicing."""
    index = {}
    for (pid, season), grp in logs.groupby(["pitcher_id", "season"]):
        index[(int(pid), int(season))] = grp.sort_values("date").reset_index(drop=True)
    return index


def _as_of_date_stats(pitcher_index, pitcher_id, season, cutoff_date):
    """Aggregate a pitcher's game log entries strictly before cutoff_date into
    the same stats-dict shape TeamRunRatings.pitcher_factor() expects. None if
    no prior-games data exists for that pitcher-season (mirrors live: a
    pitcher with no fetched log this season gets no adjustment)."""
    df = pitcher_index.get((pitcher_id, season))
    if df is None:
        return None
    prior = df[df["date"] < cutoff_date]
    if prior.empty:
        return None
    ip = float(prior["ip"].sum())
    if ip <= 0:
        return None
    er = float(prior["er"].sum())
    return {
        "innings_pitched": ip,
        "era": 9.0 * er / ip,
        "home_runs": float(prior["hr"].sum()),
        "walks": float(prior["bb"].sum()),
        "hit_by_pitch": float(prior["hbp"].sum()),
        "strikeouts": float(prior["so"].sum()),
    }


def run_pitcher_backtest(n_sims=None, seasons=None):
    seasons = seasons or config.PITCHER_BACKTEST_SEASONS
    n_sims = n_sims or config.BACKTEST_MC_SIMS

    games, starters, logs = _load_inputs()
    paired = _pair_doubleheaders(games, starters)
    paired = paired[paired["season"].isin(seasons)].sort_values("date").reset_index(drop=True)
    pitcher_index = _build_pitcher_index(logs)

    results = []
    game_counter = 0
    for date, day_games in paired.groupby("date"):
        train = games[games["date"] < date]
        if train.empty:
            continue
        try:
            ratings = TeamRunRatings().fit(train)
        except ValueError:
            continue

        for row in day_games.itertuples(index=False):
            season = int(row.season)
            home_stats = _as_of_date_stats(pitcher_index, row.home_pitcher_id, season, date)
            away_stats = _as_of_date_stats(pitcher_index, row.away_pitcher_id, season, date)
            home_pf = ratings.pitcher_factor(home_stats, ratings.league_avg_era_proxy)
            away_pf = ratings.pitcher_factor(away_stats, ratings.league_avg_era_proxy)

            mu_home_base, mu_away_base = ratings.predict_mus(row.home_team, row.away_team)
            mu_home_adj, mu_away_adj = ratings.predict_mus(
                row.home_team, row.away_team, home_pitcher_factor=home_pf, away_pitcher_factor=away_pf
            )

            seed = game_counter
            game_counter += 1
            sim_base = monte_carlo.simulate_game(
                mu_home_base, mu_away_base, n_sims=n_sims, overdispersion=ratings.overdispersion, seed=seed
            )
            sim_adj = monte_carlo.simulate_game(
                mu_home_adj, mu_away_adj, n_sims=n_sims, overdispersion=ratings.overdispersion, seed=seed
            )

            actual_total = row.home_score + row.away_score
            actual_diff = row.home_score - row.away_score
            threshold = int(np.floor(config.RUN_LINE))
            home_covered = int(actual_diff >= threshold + 1)

            results.append(
                {
                    "date": row.date,
                    "season": season,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "home_score": row.home_score,
                    "away_score": row.away_score,
                    "home_win": row.home_win,
                    "home_covered": home_covered,
                    "actual_total": actual_total,
                    "home_pitcher_ip_prior": home_stats["innings_pitched"] if home_stats else 0.0,
                    "away_pitcher_ip_prior": away_stats["innings_pitched"] if away_stats else 0.0,
                    "home_has_pf": home_pf is not None,
                    "away_has_pf": away_pf is not None,
                    "any_pf_active": (home_pf is not None) or (away_pf is not None),
                    "home_pitcher_factor": home_pf,
                    "away_pitcher_factor": away_pf,
                    "base_home_win_prob": sim_base["home_win_prob"],
                    "adj_home_win_prob": sim_adj["home_win_prob"],
                    "base_expected_total": sim_base["expected_total"],
                    "adj_expected_total": sim_adj["expected_total"],
                    "base_home_covers_prob": sim_base["home_covers_prob"],
                    "adj_home_covers_prob": sim_adj["home_covers_prob"],
                }
            )

    return pd.DataFrame(results)


def _brier(probs, outcomes):
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def _log_loss(probs, outcomes, eps=1e-9):
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    o = np.asarray(outcomes, dtype=float)
    return float(-np.mean(o * np.log(p) + (1 - o) * np.log(1 - p)))


def _bootstrap_ci_mean(values, n_boot=2000, ci=0.95, seed=0):
    """Bootstrap CI on the mean of `values` (e.g. per-game baseline-minus-
    adjusted Brier score). If the CI excludes 0, the difference is
    statistically distinguishable from noise at this holdout's sample size;
    if it includes 0, it isn't -- report that honestly either way."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        means[i] = sample.mean()
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    lo, hi = np.quantile(means, [lo_q, hi_q])
    return float(values.mean()), float(lo), float(hi)


def _compare(df, label):
    """Baseline vs pitcher-adjusted comparison on one slice of games (Brier/
    log-loss for moneyline and run line, MAE/RMSE for totals), plus a
    bootstrap significance check on the per-game Brier score difference."""
    n = len(df)
    if n == 0:
        return {"n_games": 0}

    out = {"n_games": int(n)}

    for market, base_col, adj_col, outcome_col in [
        ("moneyline", "base_home_win_prob", "adj_home_win_prob", "home_win"),
        ("run_line", "base_home_covers_prob", "adj_home_covers_prob", "home_covered"),
    ]:
        base_p, adj_p, y = df[base_col].values, df[adj_col].values, df[outcome_col].values
        base_brier_per_game = (base_p - y) ** 2
        adj_brier_per_game = (adj_p - y) ** 2
        mean_diff, ci_lo, ci_hi = _bootstrap_ci_mean(base_brier_per_game - adj_brier_per_game)
        out[market] = {
            "baseline_brier": _brier(base_p, y),
            "adjusted_brier": _brier(adj_p, y),
            "baseline_log_loss": _log_loss(base_p, y),
            "adjusted_log_loss": _log_loss(adj_p, y),
            "baseline_accuracy": float(((base_p >= 0.5).astype(int) == y).mean()),
            "adjusted_accuracy": float(((adj_p >= 0.5).astype(int) == y).mean()),
            "brier_improvement_mean": mean_diff,  # positive = adjusted better (lower Brier)
            "brier_improvement_95ci": [ci_lo, ci_hi],
            "statistically_significant": bool(ci_lo is not None and (ci_lo > 0 or ci_hi < 0)),
        }

    base_err = df["base_expected_total"] - df["actual_total"]
    adj_err = df["adj_expected_total"] - df["actual_total"]
    mean_mae_diff, mae_ci_lo, mae_ci_hi = _bootstrap_ci_mean(base_err.abs().values - adj_err.abs().values)
    out["total_runs"] = {
        "baseline_mae": float(base_err.abs().mean()),
        "adjusted_mae": float(adj_err.abs().mean()),
        "baseline_rmse": float(np.sqrt((base_err**2).mean())),
        "adjusted_rmse": float(np.sqrt((adj_err**2).mean())),
        "mae_improvement_mean": mean_mae_diff,  # positive = adjusted better (lower MAE)
        "mae_improvement_95ci": [mae_ci_lo, mae_ci_hi],
        "statistically_significant": bool(mae_ci_lo is not None and (mae_ci_lo > 0 or mae_ci_hi < 0)),
    }

    return out


def summarize(df):
    summary = {"overall": _compare(df, "overall")}
    summary["active_only"] = _compare(df[df["any_pf_active"]], "active_only")
    summary["inactive_only"] = _compare(df[~df["any_pf_active"]], "inactive_only")
    summary["by_season"] = {
        str(int(season)): _compare(grp, str(season)) for season, grp in df.groupby("season")
    }
    summary["by_season_active_only"] = {
        str(int(season)): _compare(grp[grp["any_pf_active"]], str(season))
        for season, grp in df.groupby("season")
    }
    summary["pitcher_factor_stats"] = {
        "mean_home_factor": float(df["home_pitcher_factor"].dropna().mean()) if df["home_pitcher_factor"].notna().any() else None,
        "mean_away_factor": float(df["away_pitcher_factor"].dropna().mean()) if df["away_pitcher_factor"].notna().any() else None,
        "pct_games_with_any_adjustment": float(df["any_pf_active"].mean()),
        "pct_games_with_both_adjusted": float((df["home_has_pf"] & df["away_has_pf"]).mean()),
    }
    return summary


def run_and_save(n_sims=None, seasons=None):
    df = run_pitcher_backtest(n_sims=n_sims, seasons=seasons)
    if df.empty:
        raise ValueError(
            "No games with identified starters found. Run "
            "`python -m src.data.fetch_historical_pitchers` first."
        )
    config.PITCHER_BACKTEST_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PITCHER_BACKTEST_RESULTS_FILE, index=False)
    summary = summarize(df)
    summary["_meta"] = {
        "seasons": seasons or config.PITCHER_BACKTEST_SEASONS,
        "n_sims": n_sims or config.BACKTEST_MC_SIMS,
        "min_ip_for_adjustment": config.PITCHER_MIN_IP_FOR_ADJUSTMENT,
        "n_games_total": int(len(df)),
    }
    with open(config.PITCHER_BACKTEST_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    return df, summary


if __name__ == "__main__":
    _, summary = run_and_save()
    print(json.dumps(summary, indent=2))
