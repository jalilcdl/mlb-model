"""
Backtest comparison: Original (Poisson/ELO/MC) vs Challenger (XGBoost)
on Moneyline and Totals.

Time-series walk-forward: for each game, both models only see data
from before that game. No lookahead.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.models.challenger import FeatureEngineer, XGBChallenger
from src.models.elo import EloModel
from src.models.monte_carlo import simulate_game
from src.models.run_model import TeamRunRatings, game_probabilities


def backtest_models(games_df, min_history=30, step=200):
    """
    Walk-forward backtest. step controls how often we retrain
    the challenger (expensive). Original model refits on every
    game (cheap enough).
    """
    games = games_df.sort_values("date").reset_index(drop=True)
    results = []

    # Pre-build feature engineer (uses all games for history lookup,
    # but we filter by date at query time)
    fe = FeatureEngineer(games)

    # We'll train challenger once on first N games, then periodically
    challenger = None
    challenger_trained_through = 0

    for idx in range(min_history, len(games)):
        row = games.iloc[idx]
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]
        actual_home_win = int(row["home_score"] > row["away_score"])
        actual_total = row["home_score"] + row["away_score"]

        # Train data = everything before this game
        train_games = games.iloc[:idx]

        # ── Original Model ──
        # Fit ratings + ELO on training data
        try:
            elo = EloModel().fit(train_games)
            ratings = TeamRunRatings().fit(train_games, as_of_date=date)

            mu_home, mu_away = ratings.predict_mus(home, away)
            orig_ml = mu_home / (mu_home + mu_away)
            orig_total = mu_home + mu_away

            # Ensemble with ELO
            elo_prob = elo.win_probability(home, away)
            orig_ml = 0.5 * elo_prob + 0.5 * orig_ml

        except Exception:
            continue

        # ── Challenger Model ──
        # Retrain periodically (expensive)
        if challenger is None or idx >= challenger_trained_through + step:
            print(f"  Retraining challenger at game {idx}/{len(games)}...")
            try:
                challenger = XGBChallenger()
                challenger.fit(games_df=train_games, ratings=ratings, val_size=0.1)
                challenger_trained_through = idx
            except Exception as e:
                print(f"    Train failed: {e}")
                challenger = None

        if challenger:
            try:
                pred = challenger.predict_game(home, away, date, ratings=ratings)
                chall_ml = pred["home_win_prob"]
                chall_total = pred["expected_total"]
            except Exception:
                chall_ml = orig_ml
                chall_total = orig_total
        else:
            chall_ml = orig_ml
            chall_total = orig_total

        results.append({
            "date": date,
            "home": home,
            "away": away,
            "actual_home_win": actual_home_win,
            "actual_total": actual_total,
            "orig_ml": orig_ml,
            "chall_ml": chall_ml,
            "orig_total": orig_total,
            "chall_total": chall_total,
        })

    return pd.DataFrame(results)


def evaluate_backtest(df):
    """Compute and print all metrics."""
    print("=" * 60)
    print("MONEYLINE RESULTS")
    print("=" * 60)

    # Accuracy
    orig_acc = accuracy_score(df["actual_home_win"], df["orig_ml"] > 0.5)
    chall_acc = accuracy_score(df["actual_home_win"], df["chall_ml"] > 0.5)
    print(f"Accuracy:")
    print(f"  Original:  {orig_acc:.3f}")
    print(f"  Challenger: {chall_acc:.3f}")
    print(f"  Winner: {'Challenger' if chall_acc > orig_acc else 'Original'} (+{abs(chall_acc - orig_acc):.3f})")

    # Brier Score
    orig_brier = brier_score_loss(df["actual_home_win"], df["orig_ml"])
    chall_brier = brier_score_loss(df["actual_home_win"], df["chall_ml"])
    print(f"\nBrier Score (lower is better):")
    print(f"  Original:  {orig_brier:.4f}")
    print(f"  Challenger: {chall_brier:.4f}")
    print(f"  Winner: {'Challenger' if chall_brier < orig_brier else 'Original'} ({abs(chall_brier - orig_brier):.4f})")

    # Log Loss
    orig_ll = log_loss(df["actual_home_win"], np.clip(df["orig_ml"], 0.01, 0.99))
    chall_ll = log_loss(df["actual_home_win"], np.clip(df["chall_ml"], 0.01, 0.99))
    print(f"\nLog Loss (lower is better):")
    print(f"  Original:  {orig_ll:.4f}")
    print(f"  Challenger: {chall_ll:.4f}")
    print(f"  Winner: {'Challenger' if chall_ll < orig_ll else 'Original'} ({abs(chall_ll - orig_ll):.4f})")

    # Calibration bins
    print(f"\nCalibration (predicted vs actual win rate):")
    for model, col in [("Original", "orig_ml"), ("Challenger", "chall_ml")]:
        print(f"\n  {model}:")
        df["bin"] = pd.cut(df[col], bins=[0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0])
        cal = df.groupby("bin", observed=False).agg(
            n=("actual_home_win", "count"),
            pred=(col, "mean"),
            actual=("actual_home_win", "mean")
        )
        for _, r in cal.iterrows():
            if r["n"] > 10:
                bias = r["actual"] - r["pred"]
                print(f"    {str(r.name):>12}: n={int(r['n']):>4}, pred={r['pred']:.3f}, actual={r['actual']:.3f}, bias={bias:+.3f}")

    print("\n" + "=" * 60)
    print("TOTALS RESULTS")
    print("=" * 60)

    # MAE / RMSE
    orig_mae = mean_absolute_error(df["actual_total"], df["orig_total"])
    chall_mae = mean_absolute_error(df["actual_total"], df["chall_total"])
    orig_rmse = np.sqrt(mean_squared_error(df["actual_total"], df["orig_total"]))
    chall_rmse = np.sqrt(mean_squared_error(df["actual_total"], df["chall_total"]))

    print(f"MAE (lower is better):")
    print(f"  Original:  {orig_mae:.2f} runs")
    print(f"  Challenger: {chall_mae:.2f} runs")
    print(f"  Winner: {'Challenger' if chall_mae < orig_mae else 'Original'} ({abs(chall_mae - orig_mae):.2f})")

    print(f"\nRMSE (lower is better):")
    print(f"  Original:  {orig_rmse:.2f} runs")
    print(f"  Challenger: {chall_rmse:.2f} runs")
    print(f"  Winner: {'Challenger' if chall_rmse < orig_rmse else 'Original'} ({abs(chall_rmse - orig_rmse):.2f})")

    # Over/under accuracy (using mean as naive line)
    line = df["actual_total"].mean()
    orig_over = df["orig_total"] > line
    actual_over = df["actual_total"] > line
    chall_over = df["chall_total"] > line

    orig_ou_acc = accuracy_score(actual_over, orig_over)
    chall_ou_acc = accuracy_score(actual_over, chall_over)
    print(f"\nOver/Under Accuracy (line={line:.1f}):")
    print(f"  Original:  {orig_ou_acc:.3f}")
    print(f"  Challenger: {chall_ou_acc:.3f}")

    # Per-team bias
    print(f"\nPer-Team Total Bias (predicted - actual):")
    df["orig_bias"] = df["orig_total"] - df["actual_total"]
    df["chall_bias"] = df["chall_total"] - df["actual_total"]
    print(f"  Original:  {df['orig_bias'].mean():+.2f} runs (std: {df['orig_bias'].std():.2f})")
    print(f"  Challenger: {df['chall_bias'].mean():+.2f} runs (std: {df['chall_bias'].std():.2f})")

    # Games where challenger was right and original wrong (and vice versa)
    print(f"\n" + "=" * 60)
    print("HEAD-TO-HEAD BREAKDOWN")
    print("=" * 60)

    orig_correct = (df["orig_ml"] > 0.5) == df["actual_home_win"]
    chall_correct = (df["chall_ml"] > 0.5) == df["actual_home_win"]

    both_right = orig_correct & chall_correct
    both_wrong = ~orig_correct & ~chall_correct
    orig_only = orig_correct & ~chall_correct
    chall_only = ~orig_correct & chall_correct

    print(f"Both correct:   {both_right.sum():>4} games ({both_right.mean():.1%})")
    print(f"Both wrong:     {both_wrong.sum():>4} games ({both_wrong.mean():.1%})")
    print(f"Original only:  {orig_only.sum():>4} games ({orig_only.mean():.1%})")
    print(f"Challenger only: {chall_only.sum():>4} games ({chall_only.mean():.1%})")

    if chall_only.sum() > 0:
        print(f"\nGames Challenger got right that Original missed:")
        sample = df[chall_only][["date", "away", "home", "orig_ml", "chall_ml", "actual_home_win"]].head(10)
        print(sample.to_string(index=False))

    return {
        "moneyline": {
            "orig_acc": orig_acc, "chall_acc": chall_acc,
            "orig_brier": orig_brier, "chall_brier": chall_brier,
            "orig_ll": orig_ll, "chall_ll": chall_ll,
        },
        "totals": {
            "orig_mae": orig_mae, "chall_mae": chall_mae,
            "orig_rmse": orig_rmse, "chall_rmse": chall_rmse,
        },
        "breakdown": {
            "both_right": int(both_right.sum()),
            "both_wrong": int(both_wrong.sum()),
            "orig_only": int(orig_only.sum()),
            "chall_only": int(chall_only.sum()),
            "total": len(df),
        }
    }


if __name__ == "__main__":
    from src.pipeline import load_games

    print("Loading games...")
    games = load_games()
    print(f"Total games: {len(games)}")

    # For speed, test on 2025-2026 season only
    test_games = games[games["date"] >= "2025-01-01"].copy()
    print(f"Testing on {len(test_games)} games (2025-2026)")

    print("\nRunning walk-forward backtest (this will take a few minutes)...")
    results = backtest_models(test_games, min_history=100, step=300)

    print(f"\nBacktested {len(results)} games.")
    print()
    metrics = evaluate_backtest(results)
