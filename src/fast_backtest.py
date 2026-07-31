"""Fast backtest: single time-based split comparing Original vs Challenger."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.challenger import XGBChallenger
from src.models.elo import EloModel
from src.models.run_model import TeamRunRatings
from src.pipeline import load_games


def main():
    games = load_games()
    print(f"Total games: {len(games)}")

    # Time-based split
    train = games[games["date"] < "2026-06-01"].copy()
    test = games[games["date"] >= "2026-06-01"].copy()
    print(f"Train: {len(train)} games, Test: {len(test)} games")

    # Fit original model
    elo = EloModel().fit(train)
    ratings = TeamRunRatings().fit(train)

    # Train challenger
    print("\nTraining challenger...")
    challenger = XGBChallenger()
    challenger.fit(train, ratings=ratings, val_size=0.1)

    # Evaluate
    print("\n=== TESTING ON JUNE-JULY 2026 ===")
    results = []
    for _, row in test.iterrows():
        home, away = row["home_team"], row["away_team"]
        date = row["date"]
        actual_home_win = int(row["home_score"] > row["away_score"])
        actual_total = row["home_score"] + row["away_score"]

        mu_home, mu_away = ratings.predict_mus(home, away)
        elo_prob = elo.win_probability(home, away)
        orig_prob = 0.5 * elo_prob + 0.5 * (mu_home / (mu_home + mu_away))
        orig_total = mu_home + mu_away

        pred = challenger.predict_game(home, away, date, ratings=ratings)

        results.append({
            "actual_home_win": actual_home_win,
            "actual_total": actual_total,
            "orig_prob": orig_prob,
            "chall_prob": pred["home_win_prob"],
            "orig_total_pred": orig_total,
            "chall_total_pred": pred["expected_total"],
        })

    df = pd.DataFrame(results)
    n = len(df)

    # ── Moneyline ──
    print("\n" + "=" * 50)
    print("MONEYLINE")
    print("=" * 50)

    orig_acc = accuracy_score(df["actual_home_win"], df["orig_prob"] > 0.5)
    chall_acc = accuracy_score(df["actual_home_win"], df["chall_prob"] > 0.5)
    print(f"Accuracy:    Original {orig_acc:.3f}  |  Challenger {chall_acc:.3f}")

    orig_brier = brier_score_loss(df["actual_home_win"], df["orig_prob"])
    chall_brier = brier_score_loss(df["actual_home_win"], df["chall_prob"])
    print(f"Brier:       Original {orig_brier:.4f}  |  Challenger {chall_brier:.4f}")

    orig_ll = log_loss(df["actual_home_win"], np.clip(df["orig_prob"], 0.01, 0.99))
    chall_ll = log_loss(df["actual_home_win"], np.clip(df["chall_prob"], 0.01, 0.99))
    print(f"LogLoss:     Original {orig_ll:.4f}  |  Challenger {chall_ll:.4f}")

    winner = "Challenger" if chall_brier < orig_brier else "Original"
    margin = abs(chall_brier - orig_brier)
    print(f"\n>>> Winner: {winner} (by {margin:.4f} Brier)")

    # ── Totals ──
    print("\n" + "=" * 50)
    print("TOTALS")
    print("=" * 50)

    orig_mae = mean_absolute_error(df["actual_total"], df["orig_total_pred"])
    chall_mae = mean_absolute_error(df["actual_total"], df["chall_total_pred"])
    print(f"MAE:   Original {orig_mae:.2f}  |  Challenger {chall_mae:.2f}")

    orig_rmse = np.sqrt(mean_squared_error(df["actual_total"], df["orig_total_pred"]))
    chall_rmse = np.sqrt(mean_squared_error(df["actual_total"], df["chall_total_pred"]))
    print(f"RMSE:  Original {orig_rmse:.2f}  |  Challenger {chall_rmse:.2f}")

    winner = "Challenger" if chall_mae < orig_mae else "Original"
    margin = abs(chall_mae - orig_mae)
    print(f"\n>>> Winner: {winner} (by {margin:.2f} MAE)")

    # ── Head to head ──
    print("\n" + "=" * 50)
    print("HEAD-TO-HEAD (ML correct calls)")
    print("=" * 50)

    orig_correct = (df["orig_prob"] > 0.5) == df["actual_home_win"]
    chall_correct = (df["chall_prob"] > 0.5) == df["actual_home_win"]

    both = (orig_correct & chall_correct).sum()
    neither = (~orig_correct & ~chall_correct).sum()
    orig_only = (orig_correct & ~chall_correct).sum()
    chall_only = (~orig_correct & chall_correct).sum()

    print(f"Both right:      {both:>3} ({both/n:.1%})")
    print(f"Both wrong:      {neither:>3} ({neither/n:.1%})")
    print(f"Original only:   {orig_only:>3} ({orig_only/n:.1%})")
    print(f"Challenger only: {chall_only:>3} ({chall_only/n:.1%})")

    print("\n" + "=" * 50)
    print("FINAL VERDICT")
    print("=" * 50)
    if chall_brier < orig_brier and chall_mae < orig_mae:
        print("Challenger wins on BOTH moneyline and totals!")
    elif chall_brier < orig_brier:
        print("Challenger wins moneyline, Original wins totals.")
    elif chall_mae < orig_mae:
        print("Original wins moneyline, Challenger wins totals.")
    else:
        print("Original wins on both metrics.")


if __name__ == "__main__":
    main()
