# Challenger Model — XGBoost MLB Predictor
# Built to compete against the existing Poisson/ELO/Monte Carlo ensemble
# Uses gradient boosting on box-score features + rolling form

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

# Reuse existing project structure
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config
from src.data import statsapi_client
from src.models.run_model import TeamRunRatings
from src.models.monte_carlo import simulate_game


# ───────────────────────────────────────────────
# Feature Engineering
# ───────────────────────────────────────────────

class FeatureEngineer:
    """Transform raw game logs into ML features."""

    def __init__(self, games_df):
        self.games = games_df.copy()
        self.games = self.games.sort_values("date").reset_index(drop=True)
        self.ratings = None

    def _rolling_features(self, team, date, n=(5, 10, 20, 45)):
        """Rolling form features for a team as of a given date."""
        team_games = self.games[
            ((self.games["home_team"] == team) | (self.games["away_team"] == team)) &
            (self.games["date"] < date)
        ].sort_values("date")

        if len(team_games) == 0:
            return {f"{k}_{w}": 0.0 for k in [
                "wins", "runs_scored", "runs_allowed", "run_diff",
                "avg_runs_scored", "avg_runs_allowed", "win_pct",
                "home_win_pct", "away_win_pct", "day_win_pct"
            ] for w in n}

        features = {}
        for window in n:
            recent = team_games.tail(window)
            is_home = recent["home_team"] == team
            runs_scored = np.where(is_home, recent["home_score"], recent["away_score"])
            runs_allowed = np.where(is_home, recent["away_score"], recent["home_score"])
            wins = np.where(
                is_home,
                recent["home_score"] > recent["away_score"],
                recent["away_score"] > recent["home_score"]
            )

            features[f"wins_{window}"] = float(wins.sum())
            features[f"runs_scored_{window}"] = float(runs_scored.sum())
            features[f"runs_allowed_{window}"] = float(runs_allowed.sum())
            features[f"run_diff_{window}"] = float(runs_scored.sum() - runs_allowed.sum())
            features[f"avg_runs_scored_{window}"] = float(runs_scored.mean())
            features[f"avg_runs_allowed_{window}"] = float(runs_allowed.mean())
            features[f"win_pct_{window}"] = float(wins.mean())

            home_games = recent[recent["home_team"] == team]
            away_games = recent[recent["away_team"] == team]
            features[f"home_win_pct_{window}"] = float(
                (home_games["home_score"] > home_games["away_score"]).mean()
            ) if len(home_games) > 0 else 0.5
            features[f"away_win_pct_{window}"] = float(
                (away_games["away_score"] > away_games["home_score"]).mean()
            ) if len(away_games) > 0 else 0.5

            # Day/night split (if we had that data)
            features[f"day_win_pct_{window}"] = features[f"win_pct_{window}"]

        return features

    def _head_to_head(self, home, away, date, n=10):
        """Recent head-to-head record."""
        matchups = self.games[
            ((self.games["home_team"] == home) & (self.games["away_team"] == away)) |
            ((self.games["home_team"] == away) & (self.games["away_team"] == home))
        ]
        matchups = matchups[matchups["date"] < date].tail(n)

        if len(matchups) == 0:
            return {"h2h_home_wins": 0, "h2h_total": 0, "h2h_home_win_pct": 0.5}

        home_wins = (matchups["home_score"] > matchups["away_score"]).sum()
        return {
            "h2h_home_wins": int(home_wins),
            "h2h_total": len(matchups),
            "h2h_home_win_pct": float(home_wins / len(matchups)),
        }

    def _rest_days(self, team, date):
        """Days since last game."""
        team_games = self.games[
            ((self.games["home_team"] == team) | (self.games["away_team"] == team)) &
            (self.games["date"] < date)
        ]
        if len(team_games) == 0:
            return {"rest_days": 3}  # default
        last_game = team_games["date"].max()
        rest = (pd.Timestamp(date) - pd.Timestamp(last_game)).days
        return {"rest_days": max(rest, 1)}

    def build_features_for_game(self, home_team, away_team, date, ratings=None):
        """Build a feature dict for a single game."""
        features = {}

        # Rolling form (home team)
        home_features = self._rolling_features(home_team, date)
        for k, v in home_features.items():
            features[f"home_{k}"] = v

        # Rolling form (away team)
        away_features = self._rolling_features(away_team, date)
        for k, v in away_features.items():
            features[f"away_{k}"] = v

        # Head-to-head
        h2h = self._head_to_head(home_team, away_team, date)
        features.update(h2h)

        # Rest days
        features.update({f"home_{k}": v for k, v in self._rest_days(home_team, date).items()})
        features.update({f"away_{k}": v for k, v in self._rest_days(away_team, date).items()})

        # Poisson model predictions as features (steal from the original model!)
        if ratings is not None:
            mu_home, mu_away = ratings.predict_mus(home_team, away_team)
            features["poisson_mu_home"] = mu_home
            features["poisson_mu_away"] = mu_away
            features["poisson_expected_total"] = mu_home + mu_away
            features["poisson_run_diff"] = mu_home - mu_away
        else:
            features["poisson_mu_home"] = 4.5
            features["poisson_mu_away"] = 4.5
            features["poisson_expected_total"] = 9.0
            features["poisson_run_diff"] = 0.0

        # Derived features
        for window in (5, 10, 20, 45):
            features[f"runs_scored_diff_{window}"] = (
                features[f"home_avg_runs_scored_{window}"] - features[f"away_avg_runs_allowed_{window}"]
            )
            features[f"runs_allowed_diff_{window}"] = (
                features[f"home_avg_runs_allowed_{window}"] - features[f"away_avg_runs_scored_{window}"]
            )
            features[f"momentum_{window}"] = (
                features[f"home_win_pct_{window}"] - features[f"away_win_pct_{window}"]
            )

        return features

    def build_training_data(self, ratings=None, min_games=20):
        """Build full training set from historical games."""
        records = []
        labels = []
        game_ids = []

        # Sort by date to maintain temporal order
        games = self.games.copy().sort_values("date").reset_index(drop=True)

        for idx, row in games.iterrows():
            date = row["date"]
            home = row["home_team"]
            away = row["away_team"]

            # Skip if either team doesn't have enough history
            home_history = games[
                ((games["home_team"] == home) | (games["away_team"] == home)) &
                (games["date"] < date)
            ]
            away_history = games[
                ((games["home_team"] == away) | (games["away_team"] == away)) &
                (games["date"] < date)
            ]
            if len(home_history) < min_games or len(away_history) < min_games:
                continue

            features = self.build_features_for_game(home, away, date, ratings=ratings)
            records.append(features)
            labels.append({
                "home_win": int(row["home_score"] > row["away_score"]),
                "home_margin": int(row["home_score"] - row["away_score"]),
                "total_runs": int(row["home_score"] + row["away_score"]),
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
            })
            game_ids.append(idx)

        if not records:
            return pd.DataFrame(), pd.DataFrame()

        X = pd.DataFrame(records)
        y = pd.DataFrame(labels)
        return X, y


# ───────────────────────────────────────────────
# XGBoost Challenger Model
# ───────────────────────────────────────────────

class XGBChallenger:
    """Gradient-boosted challenger to the Poisson/ELO ensemble."""

    def __init__(self):
        self.moneyline_model = None
        self.total_model = None
        self.runline_model = None
        self.feature_engineer = None
        self.feature_names = None

    def fit(self, games_df, ratings=None, val_size=0.1):
        """Train on historical games."""
        print("Building features...")
        self.feature_engineer = FeatureEngineer(games_df)
        X, y = self.feature_engineer.build_training_data(ratings=ratings)

        if X.empty:
            raise ValueError("No training data generated — check game history")

        self.feature_names = list(X.columns)
        n_total = len(X)
        n_train = int(n_total * (1 - val_size))

        # Temporal split — crucial for sports
        X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]

        print(f"Training on {len(X_train)} games, validating on {len(X_val)}")

        # ── Moneyline model ──
        print("Training moneyline model...")
        self.moneyline_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=42,
        )
        self.moneyline_model.fit(
            X_train, y_train["home_win"],
            eval_set=[(X_val, y_val["home_win"])],
            verbose=False,
        )
        ml_pred = self.moneyline_model.predict_proba(X_val)[:, 1]
        print(f"  Val Brier: {brier_score_loss(y_val['home_win'], ml_pred):.4f}")
        print(f"  Val LogLoss: {log_loss(y_val['home_win'], ml_pred):.4f}")
        print(f"  Val Accuracy: {((ml_pred > 0.5) == y_val['home_win'].values).mean():.3f}")

        # ── Total runs model ──
        print("Training total runs model...")
        self.total_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            early_stopping_rounds=50,
            random_state=42,
        )
        self.total_model.fit(
            X_train, y_train["total_runs"],
            eval_set=[(X_val, y_val["total_runs"])],
            verbose=False,
        )
        total_pred = self.total_model.predict(X_val)
        print(f"  Val MAE: {mean_absolute_error(y_val['total_runs'], total_pred):.2f}")
        print(f"  Val RMSE: {np.sqrt(mean_squared_error(y_val['total_runs'], total_pred)):.2f}")

        # ── Run line model ──
        print("Training run line model...")
        # Create binary target: home covers -1.5
        y_train_rl = y_train.copy()
        y_val_rl = y_val.copy()
        y_train_rl["home_covers"] = (y_train_rl["home_margin"] > 1).astype(int)
        y_val_rl["home_covers"] = (y_val_rl["home_margin"] > 1).astype(int)

        self.runline_model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=42,
        )
        self.runline_model.fit(
            X_train, y_train_rl["home_covers"],
            eval_set=[(X_val, y_val_rl["home_covers"])],
            verbose=False,
        )
        rl_pred = self.runline_model.predict_proba(X_val)[:, 1]
        print(f"  Val Brier: {brier_score_loss(y_val_rl['home_covers'], rl_pred):.4f}")
        print(f"  Val Accuracy: {((rl_pred > 0.5) == y_val_rl['home_covers'].values).mean():.3f}")

        return self

    def predict_game(self, home_team, away_team, date, ratings=None):
        """Predict a single game."""
        features = self.feature_engineer.build_features_for_game(
            home_team, away_team, date, ratings=ratings
        )
        X = pd.DataFrame([features])

        # Ensure column order matches training
        X = X[self.feature_names]

        home_win_prob = float(self.moneyline_model.predict_proba(X)[0, 1])
        expected_total = float(self.total_model.predict(X)[0])
        home_covers_prob = float(self.runline_model.predict_proba(X)[0, 1])

        return {
            "model": "xgboost_challenger",
            "home_win_prob": home_win_prob,
            "away_win_prob": 1.0 - home_win_prob,
            "expected_total": expected_total,
            "home_covers_prob": home_covers_prob,
            "away_covers_prob": 1.0 - home_covers_prob,
            "run_line": 1.5,
        }

    def predict_games(self, games_list, ratings=None):
        """Predict multiple games. games_list: [(home, away, date), ...]"""
        return [self.predict_game(h, a, d, ratings=ratings) for h, a, d in games_list]

    def save(self, path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "moneyline_model": self.moneyline_model,
                "total_model": self.total_model,
                "runline_model": self.runline_model,
                "feature_names": self.feature_names,
            }, f)
        print(f"Model saved to {path}")

    def load(self, path, games_df=None):
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.moneyline_model = data["moneyline_model"]
        self.total_model = data["total_model"]
        self.runline_model = data["runline_model"]
        self.feature_names = data["feature_names"]
        if games_df is not None:
            self.feature_engineer = FeatureEngineer(games_df)
        return self


# ───────────────────────────────────────────────
# Comparison / Head-to-Head
# ───────────────────────────────────────────────

def compare_models(games_df, ratings, n_splits=5):
    """Time-series cross-validation comparing original vs challenger."""
    fe = FeatureEngineer(games_df)
    X, y = fe.build_training_data(ratings=ratings)

    if X.empty:
        raise ValueError("No data for comparison")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train challenger on this fold
        challenger = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        challenger.fit(X_train, y_train["home_win"])
        challenger_pred = challenger.predict_proba(X_test)[:, 1]

        # Original model predictions (from the Poisson model features in X)
        original_pred = X_test["poisson_mu_home"] / (
            X_test["poisson_mu_home"] + X_test["poisson_mu_away"]
        )

        results.append({
            "fold": fold + 1,
            "n_test": len(y_test),
            "challenger_brier": brier_score_loss(y_test["home_win"], challenger_pred),
            "original_brier": brier_score_loss(y_test["home_win"], original_pred),
            "challenger_logloss": log_loss(y_test["home_win"], challenger_pred),
            "original_logloss": log_loss(y_test["home_win"], original_pred),
            "challenger_acc": ((challenger_pred > 0.5) == y_test["home_win"].values).mean(),
            "original_acc": ((original_pred > 0.5) == y_test["home_win"].values).mean(),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Quick test
    from src.pipeline import load_games, fit_models
    games = load_games()
    elo, ratings = fit_models(games)

    print("\n=== Training Challenger Model ===")
    challenger = XGBChallenger()
    challenger.fit(games, ratings=ratings)

    # Save
    challenger.save(config.PROCESSED_DIR / "challenger_model.pkl")

    # Test prediction on today's games
    print("\n=== Sample Prediction ===")
    sample = challenger.predict_game("ATL", "NYM", "2026-07-23", ratings=ratings)
    print(json.dumps(sample, indent=2))
