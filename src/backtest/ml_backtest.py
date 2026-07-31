"""
Walk-forward backtest for the independent moneyline models.

Trains on all data before each date, predicts that date's games,
grades against real closing moneylines.

Reports BOTH:
  - Predictive accuracy (AUC, Brier, log-loss)
  - Market-beating value (EV%, ROI, vs naive baseline)

Bootstrap confidence intervals on all metrics.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.models.gbm_independent import FeatureEngineer, LogisticBaseline, XGBoostModel
from src.data import moneyline_odds

warnings.filterwarnings("ignore")


def _naive_baseline(home_impl_prob, away_impl_prob, home_win):
    """Always pick the market favorite."""
    pred = (home_impl_prob >= away_impl_prob).astype(int)
    return (pred == home_win).mean()


def _ev_for_bet(model_prob, market_ml):
    """Expected value % for betting $100 at market moneyline."""
    market_ml = pd.to_numeric(market_ml, errors="coerce")
    profit = np.where(
        market_ml < 0,
        100 * 100 / (-market_ml),
        market_ml
    )
    loss = 100.0
    return (model_prob * profit - (1 - model_prob) * loss) / 100.0


def _expected_value_results(df):
    """Compute EV% for bets where model disagrees with market."""
    # Home side EV
    home_ev = _ev_for_bet(df["pred_home_prob"], df["home_close_ml"])
    # Away side EV (model's away prob = 1 - home prob)
    away_ev = _ev_for_bet(1 - df["pred_home_prob"], df["away_close_ml"])

    # Bet home if model says home prob > market home prob
    bet_home = df["pred_home_prob"] > df["home_impl_prob"]
    # Bet away if model says away prob > market away prob
    bet_away = (1 - df["pred_home_prob"]) > df["away_impl_prob"]

    results = []

    # Home bets
    home_bets = df[bet_home].copy()
    if len(home_bets) > 0:
        home_wins = home_bets["home_win"].sum()
        home_ev_values = _ev_for_bet(home_bets["pred_home_prob"], home_bets["home_close_ml"])
        results.append({
            "side": "home",
            "n_bets": len(home_bets),
            "n_wins": int(home_wins),
            "win_rate": float(home_wins / len(home_bets)),
            "avg_ev_pct": float(home_ev_values.mean()),
            "total_ev_pct": float(home_ev_values.sum()),
        })

    # Away bets
    away_bets = df[bet_away].copy()
    if len(away_bets) > 0:
        away_wins = (1 - away_bets["home_win"]).sum()
        away_ev_values = _ev_for_bet(1 - away_bets["pred_home_prob"], away_bets["away_close_ml"])
        results.append({
            "side": "away",
            "n_bets": len(away_bets),
            "n_wins": int(away_wins),
            "win_rate": float(away_wins / len(away_bets)),
            "avg_ev_pct": float(away_ev_values.mean()),
            "total_ev_pct": float(away_ev_values.sum()),
        })

    return pd.DataFrame(results)


def _bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    """Bootstrap percentile CI for a 1D array of values."""
    values = np.array(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": np.nan, "low": np.nan, "high": np.nan}

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(sample.mean())

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    return {
        "mean": float(values.mean()),
        "low": float(np.quantile(boot_means, alpha)),
        "high": float(np.quantile(boot_means, 1 - alpha)),
    }


def _calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(bin_acc - bin_conf)
    return ece


class WalkForwardBacktest:
    """Walk-forward backtest engine.

    For efficiency, features are pre-built for ALL games once,
    then the walk-forward loop just slices by date.
    """

    def __init__(self, games_df, ml_df, model_class, model_kwargs=None,
                 min_train_games=1000, retrain_every="1D",
                 include_pitcher=False):
        """
        games_df: DataFrame with game results
        ml_df:    DataFrame with market moneylines
        model_class: LogisticBaseline or XGBoostModel
        model_kwargs: dict of kwargs for model constructor
        min_train_games: minimum training samples before first prediction
        retrain_every: "1D" = daily, "7D" = weekly
        include_pitcher: whether to include pitcher features
        """
        self.games = games_df.copy().sort_values("date").reset_index(drop=True)
        self.ml = ml_df.copy()
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.min_train_games = min_train_games
        self.retrain_every = retrain_every
        self.include_pitcher = include_pitcher

        # Pre-build features for all games
        print("Building features for all games...")
        self.fe = FeatureEngineer(self.games)
        self.X, self.y, self.meta = self.fe.build_training_data(
            min_games=20, include_pitcher=include_pitcher
        )
        if self.X.empty:
            raise ValueError("No training data generated")
        print(f"  Features built: {len(self.X)} games, {len(self.X.columns)} features")

        # Merge with market lines
        self.meta["date"] = pd.to_datetime(self.meta["date"])
        self.ml["date"] = pd.to_datetime(self.ml["date"])
        self.merged = self.meta.merge(
            self.ml,
            on=["date", "home_team", "away_team"],
            how="inner"
        )
        if len(self.merged) == 0:
            raise ValueError("No overlap between game results and market lines")
        print(f"  Joined with market lines: {len(self.merged)} games")

    def run(self, verbose=True):
        """Run the walk-forward backtest.

        Returns DataFrame with one row per predicted game.
        """
        dates = sorted(self.merged["date"].unique())
        results = []
        model = None
        last_retrain = None

        for i, date in enumerate(dates):
            # Split: train on all games before date, predict games ON date
            train_mask = self.merged["date"] < date
            test_mask = self.merged["date"] == date

            if train_mask.sum() < self.min_train_games:
                continue
            if test_mask.sum() == 0:
                continue

            # Retrain model if needed
            should_retrain = (
                model is None
                or last_retrain is None
                or (date - last_retrain).days >= pd.Timedelta(self.retrain_every).days
            )

            if should_retrain:
                if verbose and i % 50 == 0:
                    print(f"  [{date.date()}] Retraining on {train_mask.sum()} games...")

                train_idx = self.merged[train_mask].index
                X_train = self.X.loc[train_idx]
                y_train = self.y.loc[train_idx]

                model = self.model_class(**self.model_kwargs)
                model.fit(X_train, y_train)
                last_retrain = date

            # Predict
            test_idx = self.merged[test_mask].index
            X_test = self.X.loc[test_idx]
            y_test = self.y.loc[test_idx]
            meta_test = self.merged.loc[test_idx].copy()

            probs = model.predict_proba(X_test)
            meta_test["pred_home_prob"] = probs
            meta_test["pred_away_prob"] = 1 - probs
            meta_test["pred_correct"] = (
                (probs > 0.5).astype(int) == y_test.values
            ).astype(int)

            results.append(meta_test)

        if not results:
            raise ValueError("No predictions generated")

        return pd.concat(results, ignore_index=True)


def summarize(df, n_bootstrap=1000):
    """Compute all metrics with bootstrap CIs."""
    y_true = df["home_win"].values
    y_prob = df["pred_home_prob"].values

    # ── Predictive metrics ──
    accuracy = (y_prob > 0.5).astype(int) == y_true
    naive_acc = _naive_baseline(df["home_impl_prob"], df["away_impl_prob"], df["home_win"])

    summary = {
        "n_games": len(df),
        "model_accuracy": _bootstrap_ci(accuracy.astype(float), n_bootstrap),
        "naive_baseline_accuracy": {"mean": float(naive_acc), "low": np.nan, "high": np.nan},
        "accuracy_vs_naive": _bootstrap_ci(
            (accuracy.astype(float) - naive_acc), n_bootstrap
        ),
        "auc": {"mean": float(roc_auc_score(y_true, y_prob)), "low": np.nan, "high": np.nan},
        "brier": {"mean": float(brier_score_loss(y_true, y_prob)), "low": np.nan, "high": np.nan},
        "logloss": {"mean": float(log_loss(y_true, y_prob)), "low": np.nan, "high": np.nan},
        "ece": {"mean": float(_calibration_error(y_true, y_prob)), "low": np.nan, "high": np.nan},
    }

    # ── Market-beating metrics ──
    ev_results = _expected_value_results(df)
    if not ev_results.empty:
        summary["ev_summary"] = ev_results.to_dict("records")

    # Overall: any bet where model disagrees with market
    model_home = y_prob > df["home_impl_prob"].values
    model_away = (1 - y_prob) > df["away_impl_prob"].values

    # Home bets EV
    home_bets = df[model_home]
    if len(home_bets) > 0:
        home_evs = _ev_for_bet(home_bets["pred_home_prob"], home_bets["home_close_ml"])
        summary["home_bets"] = {
            "n": len(home_bets),
            "win_rate": float(home_bets["home_win"].mean()),
            "avg_ev_pct": _bootstrap_ci(home_evs, n_bootstrap),
            "total_ev_pct": float(home_evs.sum()),
        }

    # Away bets EV
    away_bets = df[model_away]
    if len(away_bets) > 0:
        away_evs = _ev_for_bet(1 - away_bets["pred_home_prob"], away_bets["away_close_ml"])
        summary["away_bets"] = {
            "n": len(away_bets),
            "win_rate": float((1 - away_bets["home_win"]).mean()),
            "avg_ev_pct": _bootstrap_ci(away_evs, n_bootstrap),
            "total_ev_pct": float(away_evs.sum()),
        }

    # Combined
    all_bets_mask = model_home | model_away
    all_bets = df[all_bets_mask]
    if len(all_bets) > 0:
        bet_evs = []
        for _, row in all_bets.iterrows():
            if row["pred_home_prob"] > row["home_impl_prob"]:
                ev = _ev_for_bet(row["pred_home_prob"], row["home_close_ml"])
            else:
                ev = _ev_for_bet(1 - row["pred_home_prob"], row["away_close_ml"])
            bet_evs.append(ev)
        bet_evs = np.array(bet_evs)
        summary["all_bets"] = {
            "n": len(all_bets),
            "avg_ev_pct": _bootstrap_ci(bet_evs, n_bootstrap),
            "total_ev_pct": float(bet_evs.sum()),
        }

    return summary


def run_backtest(model_class, model_kwargs=None, include_pitcher=False,
                 retrain_every="7D", verbose=True, save_path=None):
    """End-to-end backtest: load data, run, summarize, save."""
    print("=" * 60)
    print(f"Backtest: {model_class.__name__}")
    print(f"  Pitcher features: {include_pitcher}")
    print(f"  Retrain every: {retrain_every}")
    print("=" * 60)

    games, ml = moneyline_odds.load()

    bt = WalkForwardBacktest(
        games_df=games,
        ml_df=ml,
        model_class=model_class,
        model_kwargs=model_kwargs,
        retrain_every=retrain_every,
        include_pitcher=include_pitcher,
    )

    results = bt.run(verbose=verbose)
    summary = summarize(results)

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Games predicted: {summary['n_games']}")
    print(f"Model accuracy:  {summary['model_accuracy']['mean']:.3f} "
          f"(CI: {summary['model_accuracy']['low']:.3f} - {summary['model_accuracy']['high']:.3f})")
    print(f"Naive baseline:  {summary['naive_baseline_accuracy']['mean']:.3f}")
    print(f"AUC:             {summary['auc']['mean']:.3f}")
    print(f"Brier:           {summary['brier']['mean']:.4f}")
    print(f"ECE:             {summary['ece']['mean']:.4f}")

    if "all_bets" in summary:
        ab = summary["all_bets"]
        print(f"\nMarket-beating:")
        print(f"  Bets placed: {ab['n']}")
        print(f"  Avg EV%:     {ab['avg_ev_pct']['mean']:.3f} "
              f"(CI: {ab['avg_ev_pct']['low']:.3f} - {ab['avg_ev_pct']['high']:.3f})")
        print(f"  Total EV%:   {ab['total_ev_pct']:.1f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(save_path, index=False)
        json_path = str(save_path).replace(".csv", "_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSaved results -> {save_path}")
        print(f"Saved summary -> {json_path}")

    return results, summary


if __name__ == "__main__":
    # Quick test: logistic regression baseline, team-only
    run_backtest(
        model_class=LogisticBaseline,
        model_kwargs={"C": 1.0},
        include_pitcher=False,
        retrain_every="7D",
        save_path=config.PROCESSED_DIR / "ml_backtest_logistic_teamonly.csv"
    )
