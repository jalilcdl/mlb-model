"""Compare original model vs challenger on today's games."""
import datetime as dt
import pandas as pd

from src import config
from src.data import statsapi_client
from src.models.challenger import XGBChallenger
from src.pipeline import load_games, fit_models


def main():
    print("Loading games and ratings...")
    games = load_games()
    elo, ratings = fit_models(games)

    print("Loading challenger model...")
    challenger = XGBChallenger()
    challenger.load(config.PROCESSED_DIR / "challenger_model.pkl", games_df=games)

    print("\n=== Generating Predictions for Today ===")
    schedule = statsapi_client.get_schedule(dt.date.today().isoformat())
    results = []
    for g in schedule:
        home, away = g["home_team"], g["away_team"]

        # Original model prediction
        mu_home, mu_away = ratings.predict_mus(home, away)
        orig_home_win = mu_home / (mu_home + mu_away)
        orig_total = mu_home + mu_away

        # Challenger prediction
        pred = challenger.predict_game(home, away, dt.date.today().isoformat(), ratings=ratings)

        results.append({
            "game": f"{away} @ {home}",
            "orig_home_win": orig_home_win,
            "chall_home_win": pred["home_win_prob"],
            "orig_total": orig_total,
            "chall_total": pred["expected_total"],
            "diff_ml": pred["home_win_prob"] - orig_home_win,
            "diff_total": pred["expected_total"] - orig_total,
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    print("\n=== Head-to-Head Comparison ===")
    avg_ml_diff = df["diff_ml"].abs().mean()
    max_ml_diff = df["diff_ml"].abs().max()
    avg_total_diff = df["diff_total"].abs().mean()
    print(f"Average moneyline diff: {avg_ml_diff:.3f}")
    print(f"Max moneyline diff: {max_ml_diff:.3f}")
    print(f"Average total diff: {avg_total_diff:.2f} runs")

    # Show where models disagree most
    print("\n=== Biggest Disagreements ===")
    df_sorted = df.reindex(df["diff_ml"].abs().sort_values(ascending=False).index)
    print(df_sorted[["game", "orig_home_win", "chall_home_win", "diff_ml"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
