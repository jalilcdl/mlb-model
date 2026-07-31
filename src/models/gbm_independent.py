"""
Independent MLB moneyline prediction models.

Two architectures, both genuinely independent from the existing Poisson/Elo
ensemble:

  1. LogisticRegressionBaseline — fast, interpretable, catches leaks early.
  2. XGBoostModel — gradient-boosted classifier, the production model.

Both use the SAME feature engineering pipeline (no model-specific leakage).
The feature engineer is temporal: every feature is computed from games
STRICTLY before the prediction date.

Two tracks:
  - team_only:   validated on 2014-2019 real closing moneylines
  - with_pitcher:  adds starting-pitcher features for live predictions
                   (HONESTLY labeled as unvalidated against closing lines)
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost not installed; XGBoostModel will not be available.")


# ───────────────────────────────────────────────
# Feature Engineering (temporal, no leakage)
# ───────────────────────────────────────────────

class FeatureEngineer:
    """Transform historical game logs into ML features with strict temporal
    discipline. Every feature is computed from games BEFORE the prediction date.

    Optimized: pre-builds per-team time series, uses vectorized rolling.
    """

    def __init__(self, games_df, pitcher_df=None):
        """
        games_df:   DataFrame with columns date, home_team, away_team,
                    home_score, away_score, home_win (sorted or not)
        pitcher_df: Optional. DataFrame with date, home_team, away_team,
                    home_pitcher_id, away_pitcher_id, plus stat columns.
                    Only used for the with_pitcher track.
        """
        self.games = games_df.copy().sort_values("date").reset_index(drop=True)
        self.pitcher_df = pitcher_df.copy() if pitcher_df is not None else None

        # Pre-build per-team time series for fast rolling lookups
        self._team_series = self._build_team_series()

    def _build_team_series(self):
        """Build a dict: team -> DataFrame of that team's games in chronological
        order, with standardized columns (date, runs_scored, runs_allowed,
        win, is_home).
        """
        series = {}
        all_teams = pd.unique(self.games[["home_team", "away_team"]].values.ravel())

        for team in all_teams:
            home_games = self.games[self.games["home_team"] == team].copy()
            home_games["runs_scored"] = home_games["home_score"]
            home_games["runs_allowed"] = home_games["away_score"]
            home_games["win"] = (home_games["home_score"] > home_games["away_score"]).astype(int)
            home_games["is_home"] = 1

            away_games = self.games[self.games["away_team"] == team].copy()
            away_games["runs_scored"] = away_games["away_score"]
            away_games["runs_allowed"] = away_games["home_score"]
            away_games["win"] = (away_games["away_score"] > away_games["home_score"]).astype(int)
            away_games["is_home"] = 0

            team_df = pd.concat([
                home_games[["date", "runs_scored", "runs_allowed", "win", "is_home"]],
                away_games[["date", "runs_scored", "runs_allowed", "win", "is_home"]],
            ], ignore_index=True).sort_values("date").reset_index(drop=True)

            series[team] = team_df

        return series

    def _rolling_features(self, team, date, windows=(5, 10, 20, 45)):
        """Fast rolling features using pre-built per-team series."""
        ts = self._team_series.get(team)
        if ts is None or len(ts) == 0:
            feats = {f"{k}_{w}": 0.0 for k in [
                "wins", "runs_scored", "runs_allowed", "run_diff",
                "avg_runs_scored", "avg_runs_allowed", "win_pct",
                "home_win_pct", "away_win_pct"
            ] for w in windows}
            feats["games_played"] = 0
            return feats

        prior = ts[ts["date"] < date]
        if len(prior) == 0:
            feats = {f"{k}_{w}": 0.0 for k in [
                "wins", "runs_scored", "runs_allowed", "run_diff",
                "avg_runs_scored", "avg_runs_allowed", "win_pct",
                "home_win_pct", "away_win_pct"
            ] for w in windows}
            feats["games_played"] = 0
            return feats

        feats = {"games_played": len(prior)}

        for w in windows:
            recent = prior.tail(w)
            wins = recent["win"].values
            rs = recent["runs_scored"].values
            ra = recent["runs_allowed"].values
            is_home = recent["is_home"].values

            feats[f"wins_{w}"] = float(wins.sum())
            feats[f"runs_scored_{w}"] = float(rs.sum())
            feats[f"runs_allowed_{w}"] = float(ra.sum())
            feats[f"run_diff_{w}"] = float(rs.sum() - ra.sum())
            feats[f"avg_runs_scored_{w}"] = float(rs.mean())
            feats[f"avg_runs_allowed_{w}"] = float(ra.mean())
            feats[f"win_pct_{w}"] = float(wins.mean())

            home_mask = is_home == 1
            away_mask = is_home == 0
            feats[f"home_win_pct_{w}"] = float(wins[home_mask].mean()) if home_mask.any() else 0.5
            feats[f"away_win_pct_{w}"] = float(wins[away_mask].mean()) if away_mask.any() else 0.5

        return feats

    def _head_to_head(self, home, away, date, n=10):
        """Recent head-to-head record."""
        matchups = self.games[
            (
                ((self.games["home_team"] == home) & (self.games["away_team"] == away))
                | ((self.games["home_team"] == away) & (self.games["away_team"] == home))
            )
            & (self.games["date"] < date)
        ].tail(n)

        if len(matchups) == 0:
            return {
                "h2h_total": 0,
                "h2h_home_wins": 0,
                "h2h_home_win_pct": 0.5,
            }

        home_wins = (matchups["home_score"] > matchups["away_score"]).sum()
        return {
            "h2h_total": len(matchups),
            "h2h_home_wins": int(home_wins),
            "h2h_home_win_pct": float(home_wins / len(matchups)),
        }

    def _rest_days(self, team, date):
        """Days since last game."""
        ts = self._team_series.get(team)
        if ts is None or len(ts) == 0:
            return {"rest_days": 3}
        prior = ts[ts["date"] < date]
        if len(prior) == 0:
            return {"rest_days": 3}
        last = prior["date"].max()
        return {"rest_days": max((pd.Timestamp(date) - pd.Timestamp(last)).days, 1)}

    def _season_context(self, home, away, date):
        """Season-level context (games played so far, month, etc)."""
        ts = pd.Timestamp(date)
        season_start = ts.replace(month=1, day=1)
        season_games = self.games[
            (self.games["date"] < date)
            & (self.games["date"] >= season_start)
        ]
        home_games = season_games[
            (season_games["home_team"] == home) | (season_games["away_team"] == home)
        ]
        away_games = season_games[
            (season_games["home_team"] == away) | (season_games["away_team"] == away)
        ]
        return {
            "home_season_games": len(home_games),
            "away_season_games": len(away_games),
            "month": ts.month,
            "day_of_week": ts.dayofweek,
        }

    def _pitcher_features(self, home, away, date):
        """Starting pitcher features (ONLY for with_pitcher track).
        Returns empty dict if no pitcher data available.
        """
        if self.pitcher_df is None:
            return {}

        mask = (
            (self.pitcher_df["home_team"] == home)
            & (self.pitcher_df["away_team"] == away)
            & (self.pitcher_df["date"] <= date)
        )
        row = self.pitcher_df[mask].sort_values("date").tail(1)
        if row.empty:
            return {}

        r = row.iloc[0]
        feats = {}
        for side in ("home", "away"):
            for stat in ("era", "whip", "k9", "bb9", "hr9", "fip", "ip"):
                col = f"{side}_pitcher_{stat}"
                if col in r:
                    feats[col] = float(r[col])
        return feats

    # ── Public interface ──

    def build_features_for_game(self, home_team, away_team, date,
                                include_pitcher=False):
        """Build feature dict for a single game."""
        features = {}

        # Rolling form
        for prefix, team in (("home", home_team), ("away", away_team)):
            team_feats = self._rolling_features(team, date)
            for k, v in team_feats.items():
                features[f"{prefix}_{k}"] = v

        # Head-to-head
        features.update(self._head_to_head(home_team, away_team, date))

        # Rest
        features.update({f"home_{k}": v for k, v in self._rest_days(home_team, date).items()})
        features.update({f"away_{k}": v for k, v in self._rest_days(away_team, date).items()})

        # Season context
        features.update(self._season_context(home_team, away_team, date))

        # Derived: matchup-quality features
        for w in (5, 10, 20, 45):
            features[f"runs_scored_diff_{w}"] = (
                features[f"home_avg_runs_scored_{w}"] - features[f"away_avg_runs_allowed_{w}"]
            )
            features[f"runs_allowed_diff_{w}"] = (
                features[f"home_avg_runs_allowed_{w}"] - features[f"away_avg_runs_scored_{w}"]
            )
            features[f"momentum_{w}"] = (
                features[f"home_win_pct_{w}"] - features[f"away_win_pct_{w}"]
            )

        # Pitcher features (only if requested and available)
        if include_pitcher:
            pitcher_feats = self._pitcher_features(home_team, away_team, date)
            features.update(pitcher_feats)

        return features

    def build_training_data(self, min_games=20, include_pitcher=False,
                            verbose=True):
        """Build full (X, y) matrices from historical games.

        Only includes games where BOTH teams have at least min_games of
        history before that date.
        """
        records = []
        labels = []
        meta = []

        n_total = len(self.games)
        for i, (_, row) in enumerate(self.games.iterrows()):
            if verbose and i % 2000 == 0:
                print(f"  Building features: {i}/{n_total} games...")

            date = row["date"]
            home = row["home_team"]
            away = row["away_team"]

            # Skip if insufficient history
            home_ts = self._team_series.get(home)
            away_ts = self._team_series.get(away)
            home_n = len(home_ts[home_ts["date"] < date]) if home_ts is not None else 0
            away_n = len(away_ts[away_ts["date"] < date]) if away_ts is not None else 0
            if home_n < min_games or away_n < min_games:
                continue

            feats = self.build_features_for_game(home, away, date, include_pitcher)
            records.append(feats)
            labels.append(int(row["home_score"] > row["away_score"]))
            meta.append({"date": date, "home_team": home, "away_team": away})

        if not records:
            return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()

        X = pd.DataFrame(records)
        y = pd.Series(labels, name="home_win")
        meta_df = pd.DataFrame(meta)
        return X, y, meta_df


# ───────────────────────────────────────────────
# Logistic Regression Baseline
# ───────────────────────────────────────────────

class LogisticBaseline:
    """Interpretable baseline. Catches feature leaks before we burn compute
    on the black-box model."""

    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state,
            solver="lbfgs"
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X, y, val_X=None, val_y=None):
        """Train on X, y. Optional validation set for reporting."""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Report training metrics
        train_pred = self.model.predict_proba(X_scaled)[:, 1]
        print(f"  Train Brier: {brier_score_loss(y, train_pred):.4f}")
        print(f"  Train LogLoss: {log_loss(y, train_pred):.4f}")
        print(f"  Train AUC: {roc_auc_score(y, train_pred):.4f}")

        if val_X is not None and val_y is not None:
            val_pred = self.predict_proba(val_X)
            print(f"  Val Brier:   {brier_score_loss(val_y, val_pred):.4f}")
            print(f"  Val LogLoss: {log_loss(val_y, val_pred):.4f}")
            print(f"  Val AUC:     {roc_auc_score(val_y, val_pred):.4f}")

        # Feature importance (coefficients)
        coeffs = pd.Series(self.model.coef_[0], index=self.feature_names)
        print(f"  Top positive features:")
        print(coeffs.nlargest(5).to_string())
        print(f"  Top negative features:")
        print(coeffs.nsmallest(5).to_string())

        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self.is_fitted = True
        return self


# ───────────────────────────────────────────────
# XGBoost Model
# ───────────────────────────────────────────────

class XGBoostModel:
    """Gradient-boosted classifier. The production model."""

    def __init__(self,
                 n_estimators=500,
                 max_depth=5,
                 learning_rate=0.05,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 early_stopping_rounds=50,
                 random_state=42):
        if not HAS_XGB:
            raise ImportError("xgboost is required for XGBoostModel")
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "early_stopping_rounds": early_stopping_rounds,
            "random_state": random_state,
        }
        self.model = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X, y, val_X=None, val_y=None):
        self.feature_names = list(X.columns)

        eval_set = []
        if val_X is not None and val_y is not None:
            eval_set = [(val_X[self.feature_names], val_y)]

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **self.params
        )
        self.model.fit(
            X[self.feature_names], y,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True

        # Report metrics
        train_pred = self.model.predict_proba(X[self.feature_names])[:, 1]
        print(f"  Train Brier: {brier_score_loss(y, train_pred):.4f}")
        print(f"  Train LogLoss: {log_loss(y, train_pred):.4f}")
        print(f"  Train AUC: {roc_auc_score(y, train_pred):.4f}")

        if val_X is not None and val_y is not None:
            val_pred = self.predict_proba(val_X)
            print(f"  Val Brier:   {brier_score_loss(val_y, val_pred):.4f}")
            print(f"  Val LogLoss: {log_loss(val_y, val_pred):.4f}")
            print(f"  Val AUC:     {roc_auc_score(val_y, val_pred):.4f}")

        # Feature importance
        importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        )
        print(f"  Top features by gain:")
        print(importance.nlargest(10).to_string())

        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "params": self.params,
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.params = data["params"]
        self.is_fitted = True
        return self
