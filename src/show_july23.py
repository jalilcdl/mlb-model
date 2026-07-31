"""Show July 23 games with live odds and edges."""
import datetime as dt
import pandas as pd
from src.pipeline import predict_date, load_games, fit_models

games = load_games()
elo, ratings = fit_models(games)

date_str = "2026-07-23"
preds = predict_date(date_str, games=games, elo=elo, ratings=ratings, apply_odds=True, n_sims=10000)

if preds.empty:
    print("No games found for", date_str)
else:
    print("=" * 55)
    print("JULY 23, 2026 — ALL GAMES")
    print("=" * 55)
    
    for _, row in preds.iterrows():
        print()
        print(f"{row['away_team']} @ {row['home_team']}")
        print(f"  Status: {row.get('status', 'Unknown')}")
        print(f"  Model Pick: {row['moneyline_pick']} {row['moneyline_pick_prob']:.1%}")
        print(f"  Expected Total: {row['expected_total']:.1f}")
        
        edges = []
        for label, ev_col in [
            (f"{row['home_team']} ML", "ev_home_ml_pct"),
            (f"{row['away_team']} ML", "ev_away_ml_pct"),
            (f"Over {row.get('total_line', 'N/A')}", "ev_over_pct"),
            (f"Under {row.get('total_line', 'N/A')}", "ev_under_pct"),
        ]:
            ev = row.get(ev_col)
            if pd.notna(ev) and ev > 0:
                edges.append((label, ev))
        
        if edges:
            edges.sort(key=lambda x: x[1], reverse=True)
            print("  Edges found:")
            for label, ev in edges:
                print(f"    - {label}: +{ev*100:.1f}% EV")
        else:
            print("  No +EV edges currently")
