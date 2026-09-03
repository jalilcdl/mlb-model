"""Unified live betting-signal dashboard -- MLB + CFB (more sports to come).

OBSERVE-ONLY. Shows in-game model win% vs the de-vigged live market, as logged
by core/poller.py (run on a schedule by .github/workflows/poll.yml). There is
no bet-placement code anywhere in this app, matching the hard rule in the two
source repos (mlb-model, cfb-model) this app's underlying logic was vendored
from.

Data freshness: the poller commits new rows back to this repo only when
something changes (a new game state, a newly flagged game) -- which is what
triggers Streamlit Community Cloud to redeploy with fresh data. During
non-live hours nothing changes, so nothing redeploys; that's expected, not a
bug.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sports.registry import SPORTS

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "data" / "live_signal_log.jsonl"

st.set_page_config(page_title="Live Betting Signals", layout="wide", page_icon="🚦")


@st.cache_data(ttl=10)
def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["logged_at"] = pd.to_datetime(df["logged_at_utc"], utc=True, errors="coerce")
    return df


st.title("🚦 Live Betting Signals")
st.caption(
    "In-game model win% vs the de-vigged live market, across every tracked sport. "
    "**OBSERVE-ONLY** -- this detects and logs disagreements between the model and "
    "the market. It does not size stakes, place bets, or recommend action. "
    "No bet-placement code exists anywhere in this app.")

ctrl1, ctrl2, _ = st.columns([1, 1, 4])
with ctrl1:
    auto = st.checkbox("Auto-refresh", value=True)
with ctrl2:
    interval = st.select_slider("Every (s)", options=[10, 15, 30, 60], value=15)

sport_tabs = st.tabs([f"{a.SPORT_ICON} {a.SPORT_LABEL}" for a in SPORTS.values()])

for tab, (sport_key, adapter) in zip(sport_tabs, SPORTS.items()):
    with tab:
        @st.fragment(run_every=interval if auto else None)
        def _render(sport_key=sport_key, adapter=adapter):
            df_all = load_log()
            now_utc = datetime.now(timezone.utc)
            df = df_all[df_all["sport"] == sport_key] if not df_all.empty else df_all

            if df.empty:
                st.info(
                    f"No {adapter.SPORT_LABEL} live-signal rows logged yet. Once a game "
                    "goes in progress, the scheduled poller appends rows here (checks every "
                    "few minutes -- see .github/workflows/poll.yml).")
                st.caption(f"Checked {now_utc:%H:%M:%S} UTC")
                return

            last = df["logged_at"].max()
            latest = df.groupby("game_id").tail(1)
            n_flagged = int(latest["flagged"].sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Games tracked", df["game_id"].nunique())
            c2.metric("Signal rows", len(df))
            c3.metric("Currently flagged", n_flagged)
            c4.metric("Last row (UTC)", f"{last:%H:%M:%S}")
            st.caption(f"Refreshed {now_utc:%H:%M:%S} UTC"
                       + (f" · auto every {interval}s" if auto else " · auto-refresh off"))
            st.divider()

            for game_id, g in df.groupby("game_id"):
                cur = g.iloc[-1]
                flag = "🚩 **FLAGGED**" if cur["flagged"] else ""
                st.subheader(f"{cur['matchup']} — {cur['state_desc']}  {flag}")
                st.caption(
                    f"{cur['away_team']} {int(cur['away_score'])}–{int(cur['home_score'])} "
                    f"{cur['home_team']}")

                m1, m2, m3 = st.columns(3)
                m1.metric(f"Model P({cur['home_team']} win)", f"{cur['model_home_wp']:.1%}")
                m2.metric(f"Market P({cur['home_team']} win)", f"{cur['market_home_wp']:.1%}")
                m3.metric("Edge (model − market)", f"{cur['edge_home']:+.1%}",
                          delta=(f"model favors {cur['pick_team']}" if cur["flagged"] else "no flag"),
                          delta_color=("normal" if cur["flagged"] else "off"))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=g["logged_at"], y=g["model_home_wp"],
                                         name="Model", mode="lines+markers",
                                         line=dict(color="#2E86DE", width=2)))
                fig.add_trace(go.Scatter(x=g["logged_at"], y=g["market_home_wp"],
                                         name="Market", mode="lines+markers",
                                         line=dict(color="#E67E22", width=2)))
                fig.update_layout(
                    height=260, yaxis_tickformat=".0%",
                    yaxis_title=f"P({cur['home_team']} win)", xaxis_title="",
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
                st.plotly_chart(fig, width="stretch", key=f"chart_{sport_key}_{game_id}")
                st.caption(
                    f"Odds source: {cur['odds_source']} · de-vig {cur['devig_method']} · "
                    f"{len(g)} state snapshot(s)")
                st.divider()

            fl = df[df["flagged"]].sort_values("logged_at", ascending=False)
            if not fl.empty:
                st.subheader(f"Flagged snapshots ({len(fl)})")
                show = fl.assign(
                    time=fl["logged_at"].dt.strftime("%H:%M:%S"),
                    model=(fl["model_home_wp"] * 100).round(1),
                    market=(fl["market_home_wp"] * 100).round(1),
                    edge=(fl["edge"] * 100).round(1),
                )[["time", "matchup", "state_desc", "model", "market", "edge", "pick_team"]]
                show.columns = ["Time UTC", "Game", "State", "Model% (home)",
                                "Market% (home)", "Edge pts", "Model favors"]
                st.dataframe(show, width="stretch", hide_index=True)

        _render()

st.sidebar.markdown("---")
st.sidebar.caption("Live Betting Signals · unified dashboard")
st.sidebar.caption("OBSERVE-ONLY -- no bet-placement code exists in this app.")
st.sidebar.caption(f"Sports tracked: {', '.join(a.SPORT_LABEL for a in SPORTS.values())}")
