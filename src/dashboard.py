"""
MLB Model Dashboard — Side-by-side comparison of Original vs Challenger models.

Run:
    cd ~/mlb-model
    venv\Scripts\streamlit run src\dashboard.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data import statsapi_client
from src.models.challenger import XGBChallenger
from src.models.monte_carlo import simulate_game
from src.models.run_model import TeamRunRatings, game_probabilities
from src.models.elo import EloModel
from src.odds import odds_adapter
from src.pipeline import load_games, fit_models

# ───────────────────────────────────────────────
# Page Config
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Model Arena",
    page_icon="⚾",
    layout="wide",
)

st.title("⚾ MLB Model Arena: Original vs Challenger")
st.caption("Compare the physics-based ensemble against the XGBoost challenger")


# ───────────────────────────────────────────────
# Load Data & Models (cached)
# ───────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    games = load_games()
    elo, ratings = fit_models(games)
    return games, elo, ratings


@st.cache_resource(ttl=300)
def load_challenger(games):
    challenger = XGBChallenger()
    model_path = config.PROCESSED_DIR / "challenger_model.pkl"
    if model_path.exists():
        challenger.load(model_path, games_df=games)
        return challenger
    return None


@st.cache_data(ttl=300)
def get_schedule(date_str):
    return statsapi_client.get_schedule(date_str)


@st.cache_data(ttl=300)
def load_backtest_summary():
    path = config.BACKTEST_SUMMARY_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


games, elo, ratings = load_data()
challenger = load_challenger(games)
backtest = load_backtest_summary()

# ───────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────
st.sidebar.header("Controls")
date_input = st.sidebar.date_input("Game Date", value=pd.Timestamp.today())
date_str = date_input.strftime("%Y-%m-%d")
n_sims = st.sidebar.slider("Monte Carlo Sims", 1000, 50000, 10000, 1000)
show_odds = st.sidebar.checkbox("Show Live Odds", value=True)

# ───────────────────────────────────────────────
# Tabs
# ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Today's Games", "⚔️ Model Comparison", "📈 Historical Performance", "🔧 Settings"])

# ═══════════════════════════════════════════════
# TAB 1: Today's Games
# ═══════════════════════════════════════════════
with tab1:
    st.header(f"Predictions for {date_str}")

    schedule = get_schedule(date_str)
    if not schedule:
        st.warning("No games scheduled for this date (or season may be over).")
    else:
        # Build predictions for both models
        rows = []
        for g in schedule:
            home, away = g["home_team"], g["away_team"]

            # Original model
            mu_home, mu_away = ratings.predict_mus(home, away)
            orig_ml = mu_home / (mu_home + mu_away)
            orig_total = mu_home + mu_away
            sim = simulate_game(mu_home, mu_away, n_sims=n_sims, overdispersion=ratings.overdispersion)
            orig_rl = sim["home_covers_prob"]

            # Challenger model
            if challenger:
                chall = challenger.predict_game(home, away, date_str, ratings=ratings)
                chall_ml = chall["home_win_prob"]
                chall_total = chall["expected_total"]
                chall_rl = chall["home_covers_prob"]
            else:
                chall_ml = orig_ml
                chall_total = orig_total
                chall_rl = orig_rl

            rows.append({
                "game": f"{away} @ {home}",
                "home": home,
                "away": away,
                "orig_home_win": orig_ml,
                "chall_home_win": chall_ml,
                "orig_total": orig_total,
                "chall_total": chall_total,
                "orig_home_rl": orig_rl,
                "chall_home_rl": chall_rl,
                "venue": g.get("venue_name", "TBD"),
                "home_pitcher": g.get("home_probable_pitcher_name", "TBD"),
                "away_pitcher": g.get("away_probable_pitcher_name", "TBD"),
            })

        preds_df = pd.DataFrame(rows)

        # Display each game as a card
        for _, row in preds_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.subheader(row["game"])
                    st.caption(f"📍 {row['venue']}")
                    st.write(f"🧢 {row['away_pitcher']} vs {row['home_pitcher']}")

                with col2:
                    # Moneyline comparison
                    ml_diff = abs(row["chall_home_win"] - row["orig_home_win"])
                    if ml_diff > 0.05:
                        st.warning(f"Models disagree by {ml_diff:.1%}!")

                    st.write("**Moneyline (Home Win %)**")
                    st.write(f"Original: {row['orig_home_win']:.1%}")
                    st.write(f"Challenger: {row['chall_home_win']:.1%}")

                    st.write("**Total Runs**")
                    st.write(f"Original: {row['orig_total']:.1f}")
                    st.write(f"Challenger: {row['chall_total']:.1f}")

                with col3:
                    # Run line
                    st.write("**Run Line (Home -1.5)**")
                    st.write(f"Original: {row['orig_home_rl']:.1%}")
                    st.write(f"Challenger: {row['chall_home_rl']:.1%}")

                    # Pick indicator
                    orig_pick = "HOME" if row["orig_home_win"] > 0.5 else "AWAY"
                    chall_pick = "HOME" if row["chall_home_win"] > 0.5 else "AWAY"
                    if orig_pick == chall_pick:
                        st.success(f"Both pick: {orig_pick}")
                    else:
                        st.error(f"DISAGREE: Orig={orig_pick}, Chall={chall_pick}")

                st.divider()

        # Summary chart
        st.subheader("Visual Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=preds_df["game"], y=preds_df["orig_home_win"],
            mode='lines+markers', name='Original',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=preds_df["game"], y=preds_df["chall_home_win"],
            mode='lines+markers', name='Challenger',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title="Home Win Probability Comparison",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 2: Model Comparison
# ═══════════════════════════════════════════════
with tab2:
    st.header("Head-to-Head Analysis")

    if challenger:
        st.success("Challenger model loaded and ready")
    else:
        st.error("Challenger model not found. Run training first.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Model (Poisson/ELO/MC)")
        st.markdown("""
        - **Type:** Physics-based ensemble
        - **Components:** ELO ratings + Poisson run model + Monte Carlo simulation
        - **Features:** Park factors, pitcher adjustments, rolling form
        - **Distribution:** Negative binomial (empirically calibrated overdispersion)
        - **Strengths:** Interpretable, well-calibrated, handles uncertainty explicitly
        """)

    with col2:
        st.subheader("Challenger Model (XGBoost)")
        st.markdown("""
        - **Type:** Gradient boosted trees
        - **Components:** 3 separate models (moneyline, total, run line)
        - **Features:** Rolling windows (5/10/20/45 games), head-to-head, rest days, Poisson predictions as meta-features
        - **Algorithm:** XGBoost with early stopping
        - **Strengths:** Finds non-linear patterns, optimizes raw accuracy, fast inference
        """)

    if backtest:
        st.subheader("Backtest Results (Original Model)")

        bt = backtest
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Moneyline Accuracy", f"{bt['moneyline_blended']['accuracy']:.1%}")
            st.metric("Brier Score", f"{bt['moneyline_blended']['brier_score']:.3f}")
        with col2:
            st.metric("Run Line Accuracy", f"{bt['run_line']['accuracy']:.1%}")
            st.metric("Run Line Brier", f"{bt['run_line']['brier_score']:.3f}")
        with col3:
            st.metric("Total MAE", f"{bt['total_runs_monte_carlo']['mae']:.2f} runs")
            st.metric("Total RMSE", f"{bt['total_runs_monte_carlo']['rmse']:.2f} runs")

        # Calibration chart
        st.subheader("Moneyline Calibration")
        cal = pd.DataFrame(bt["moneyline_calibration_blended"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cal["mean_predicted"], y=cal["actual_rate"],
            mode='lines+markers', name='Calibration',
            marker=dict(size=cal["n"]/10)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines',
            name='Perfect Calibration', line=dict(dash='dash', color='gray')
        ))
        fig.update_layout(
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Win Rate",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3: Historical Performance
# ═══════════════════════════════════════════════
with tab3:
    st.header("Historical Performance by Season")

    if backtest and "by_season" in backtest:
        season_data = []
        for season, data in backtest["by_season"].items():
            season_data.append({
                "Season": season,
                "ML Accuracy": data["moneyline_blended"]["accuracy"],
                "ML Brier": data["moneyline_blended"]["brier_score"],
                "RL Accuracy": data["run_line"]["accuracy"],
                "Total MAE": data["total_runs_monte_carlo"]["mae"],
                "Games": data["moneyline_blended"]["n_games"],
            })

        season_df = pd.DataFrame(season_data)
        st.dataframe(season_df.style.format({
            "ML Accuracy": "{:.1%}",
            "ML Brier": "{:.3f}",
            "RL Accuracy": "{:.1%}",
            "Total MAE": "{:.2f}",
        }), use_container_width=True)

        # Season trends
        fig = px.bar(season_df, x="Season", y="ML Accuracy",
                     title="Moneyline Accuracy by Season",
                     labels={"ML Accuracy": "Accuracy"},
                     text=season_df["ML Accuracy"].apply(lambda x: f"{x:.1%}"))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No historical backtest data available.")

# ═══════════════════════════════════════════════
# TAB 4: Settings
# ═══════════════════════════════════════════════
with tab4:
    st.header("Model Configuration")

    st.subheader("Monte Carlo Settings")
    st.write(f"Default Sims: {config.MC_DEFAULT_SIMS:,}")
    st.write(f"Distribution: {config.MC_DISTRIBUTION}")
    st.write(f"Overdispersion (estimated): {ratings.overdispersion:.2f}")

    st.subheader("ELO Settings")
    st.write(f"K-Factor: {config.ELO_K}")
    st.write(f"Home Advantage: {config.ELO_HOME_ADVANTAGE} points")
    st.write(f"Season Regression: {config.ELO_SEASON_REGRESSION:.0%}")

    st.subheader("Run Model Settings")
    st.write(f"Recent Games Window: {config.RUN_MODEL_RECENT_GAMES}")
    st.write(f"Recent Weight: {config.RUN_MODEL_RECENT_WEIGHT}")
    st.write(f"Starter Innings Share: {config.STARTER_INNINGS_SHARE:.0%}")

    if st.button("Retrain Challenger Model"):
        with st.spinner("Training... This may take a few minutes."):
            from src.models.challenger import XGBChallenger
            new_challenger = XGBChallenger()
            new_challenger.fit(games, ratings=ratings)
            new_challenger.save(config.PROCESSED_DIR / "challenger_model.pkl")
        st.success("Challenger model retrained and saved!")
        st.rerun()

# ───────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("MLB Model Arena v1.0")
st.sidebar.caption("Original: Poisson/ELO/MC Ensemble")
st.sidebar.caption("Challenger: XGBoost Gradient Boosting")
