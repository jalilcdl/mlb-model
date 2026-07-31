"""Show MLB model predictions for July 24, 2026."""
import datetime as dt
import pandas as pd
from src.pipeline import predict_date, load_games, fit_models

games = load_games()
elo, ratings = fit_models(games)

date_str = "2026-07-24"
preds = predict_date(date_str, games=games, elo=elo, ratings=ratings, apply_odds=True, n_sims=10000)

if preds.empty:
    print("No games scheduled for", date_str)
else:
    print("=" * 55)
    print("MLB MODEL — JULY 24, 2026")
    print("=" * 55)
    
    cols = ["away_team", "home_team", "home_win_prob", "moneyline_pick", "moneyline_pick_prob", "expected_total", "run_line_pick"]
    print(preds[cols].to_string(index=False))
    
    print()
    print("=" * 55)
    print("BEST EDGES")
    print("=" * 55)
    for _, row in preds.iterrows():
        best_edge = None
        best_ev = -999
        candidates = [
            (f"{row['home_team']} ML", "ev_home_ml_pct"),
            (f"{row['away_team']} ML", "ev_away_ml_pct"),
            (f"Over {row.get('total_line', 'N/A')}", "ev_over_pct"),
            (f"Under {row.get('total_line', 'N/A')}", "ev_under_pct"),
        ]
        for label, ev_col in candidates:
            ev = row.get(ev_col)
            if pd.notna(ev) and ev > best_ev:
                best_ev = ev
                best_edge = (label, ev)
        if best_edge and best_edge[1] > 0:
            print(f"{row['away_team']} @ {row['home_team']}: {best_edge[0]} (+{best_edge[1] * 100:.1f}% EV)")
    
    # Show pitcher changes if any
    print()
    print("=" * 55)
    print("PITCHER STATUS")
    print("=" * 55)
    for _, row in preds.iterrows():
        changed = False
        for side in ("away", "home"):
            if row.get(f"{side}_pitcher_changed"):
                changed = True
                print(f"⚠ {row[side + '_team']}: SP changed to {row.get(side + '_probable_pitcher')}")
        if not changed:
            pass
    if not any(preds.get(f"{s}_pitcher_changed", pd.Series([False]*len(preds))).any() for s in ("away", "home")):
        print("No pitcher changes detected.")
