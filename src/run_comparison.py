"""Run both models with live odds and compare results."""
import datetime as dt
import os
import sys

import pandas as pd

sys.path.insert(0, str(os.path.dirname(__file__)))

from src import config
from src.data import statsapi_client
from src.models.challenger import XGBChallenger
from src.pipeline import load_games, fit_models, predict_date

# The Odds API key comes from the environment -- never hard-code a real key in
# committed source. Set THE_ODDS_API_KEY in your shell (or .env) before running.
if not os.environ.get("THE_ODDS_API_KEY"):
    sys.exit("Set the THE_ODDS_API_KEY environment variable before running this comparison.")

# Load data
games = load_games()
elo, ratings = fit_models(games)

# Load challenger
challenger = XGBChallenger()
challenger.load(config.PROCESSED_DIR / "challenger_model.pkl", games_df=games)

today = dt.date.today().isoformat()

# Original model with live odds
print("=" * 60)
print("ORIGINAL MODEL (with live odds)")
print("=" * 60)
orig_preds = predict_date(today, games=games, elo=elo, ratings=ratings, apply_odds=True, n_sims=10000)

cols = [
    "away_team", "home_team",
    "home_win_prob", "moneyline_pick", "moneyline_pick_prob",
    "expected_total", "run_line_pick"
]
print(orig_preds[cols].to_string(index=False))

# Show edges if available
if "ev_home_ml_pct" in orig_preds.columns:
    print("\n--- BEST EDGES ---")
    for _, row in orig_preds.iterrows():
        best_edge = None
        best_ev = -999
        for label, ev_col in [
            (f"{row['home_team']} ML", "ev_home_ml_pct"),
            (f"{row['away_team']} ML", "ev_away_ml_pct"),
            (f"Over {row.get('total_line', 'N/A')}", "ev_over_pct"),
            (f"Under {row.get('total_line', 'N/A')}", "ev_under_pct"),
        ]:
            ev = row.get(ev_col)
            if pd.notna(ev) and ev > best_ev:
                best_ev = ev
                best_edge = (label, ev)
        if best_edge and best_edge[1] > 0:
            print(f"{row['away_team']} @ {row['home_team']}: {best_edge[0]} (+{best_edge[1]*100:.1f}% EV)")

# Challenger model
print("\n" + "=" * 60)
print("CHALLENGER MODEL (XGBoost)")
print("=" * 60)

schedule = statsapi_client.get_schedule(today)
for g in schedule:
    home, away = g["home_team"], g["away_team"]
    pred = challenger.predict_game(home, away, today, ratings=ratings)
    pick = home if pred["home_win_prob"] >= 0.5 else away
    prob = max(pred["home_win_prob"], pred["away_win_prob"])
    print(f"{away} @ {home}: {pick} {prob:.1%} (total: {pred['expected_total']:.1f})")

# Disagreements
print("\n" + "=" * 60)
print("WHERE MODELS DISAGREE")
print("=" * 60)
for _, row in orig_preds.iterrows():
    home, away = row["home_team"], row["away_team"]
    orig_pick = row["moneyline_pick"]
    orig_prob = row["moneyline_pick_prob"]

    pred = challenger.predict_game(home, away, today, ratings=ratings)
    chall_pick = home if pred["home_win_prob"] >= 0.5 else away
    chall_prob = max(pred["home_win_prob"], pred["away_win_prob"])

    if orig_pick != chall_pick:
        print(f"DISAGREE: {away} @ {home}")
        print(f"  Original: {orig_pick} {orig_prob:.1%}")
        print(f"  Challenger: {chall_pick} {chall_prob:.1%}")
