"""Unified live betting-signal dashboard -- MLB + CFB (more sports to come).

OBSERVE-ONLY. Shows in-game model win% vs the de-vigged live market -- for
moneyline, spread, and (MLB only) totals -- as logged by core/poller.py (run
on a schedule by .github/workflows/poll-live-signals.yml). There is no
bet-placement code anywhere in this app, matching the hard rule in the two
source repos (mlb-model, cfb-model) this app's underlying logic was vendored
from.

CFB totals are deliberately not shown as a market tab with data -- CFB's
win-probability model (cfb_lib/live/wp_model.py) only models margin, never
total points, so there is no total_* data to plot. The tab is still there,
labeled "not modeled yet", so that's a stated fact on screen, not a silent
gap someone has to notice on their own.

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

# One entry per market: which columns to read, how to label it, and whether
# it's a real gap in the model (not_modeled) vs. just "no line right now".
# Adding a market anywhere (a new sport, NFL totals once modeled, etc.) means
# adding one entry here -- nothing else in the render loop needs to change.
MARKETS = [
    {"key": "moneyline", "label": "Moneyline", "model_col": "model_home_wp",
     "market_col": "market_home_wp", "edge_col": "edge_home", "flagged_col": "flagged",
     "pick_col": "pick_team", "y_label": "P(home wins)", "not_modeled": False},
    {"key": "spread", "label": "Spread", "model_col": "spread_model_prob",
     "market_col": "spread_market_prob", "edge_col": "spread_edge",
     "flagged_col": "spread_flagged", "pick_col": "spread_pick",
     "y_label": "P(home covers)", "not_modeled": False},
    {"key": "total", "label": "Total", "model_col": "total_model_prob",
     "market_col": "total_market_prob", "edge_col": "total_edge",
     "flagged_col": "total_flagged", "pick_col": "total_pick",
     "y_label": "P(over)", "not_modeled": False},
]


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


def _markets_for(sport_key: str) -> list[dict]:
    """CFB gets totals labeled not-modeled instead of the generic market
    entry -- a fact stated on screen, not a column that's just always empty
    with no explanation."""
    markets = [dict(m) for m in MARKETS]
    if sport_key == "cfb":
        for m in markets:
            if m["key"] == "total":
                m["not_modeled"] = True
    return markets


def _render_market(g: pd.DataFrame, cur: pd.Series, market: dict, chart_key: str) -> None:
    model_col, market_col, edge_col = market["model_col"], market["market_col"], market["edge_col"]
    flagged_col, pick_col = market["flagged_col"], market["pick_col"]

    if market["not_modeled"]:
        st.info(f"{market['label']} isn't modeled for this sport yet -- "
                "the win-probability model only estimates margin, with no "
                "total-points distribution to compare against a total line. "
                "Shown here rather than hidden so that's a stated fact, not "
                "a silent gap.")
        return

    if model_col not in cur or pd.isna(cur[model_col]):
        st.info(f"No {market['label'].lower()} line currently quoted for this game.")
        return

    flag = "🚩 **FLAGGED**" if cur[flagged_col] else ""
    m1, m2, m3 = st.columns(3)
    label = "covers" if market["key"] == "spread" else ("hits over" if market["key"] == "total" else "win")
    subject = cur["home_team"] if market["key"] != "total" else "total"
    m1.metric(f"Model P({subject} {label})" if market["key"] != "moneyline"
              else f"Model P({cur['home_team']} win)", f"{cur[model_col]:.1%}")
    m2.metric(f"Market P({subject} {label})" if market["key"] != "moneyline"
              else f"Market P({cur['home_team']} win)", f"{cur[market_col]:.1%}")
    m3.metric("Edge (model − market)", f"{cur[edge_col]:+.1%}  {flag}",
              delta=(f"model favors {cur[pick_col]}" if cur[flagged_col] else "no flag"),
              delta_color=("normal" if cur[flagged_col] else "off"))
    if market["key"] == "spread" and pd.notna(cur.get("spread_line")):
        st.caption(f"Line: home {cur['spread_line']:+g}")
    if market["key"] == "total" and pd.notna(cur.get("total_line")):
        st.caption(f"Line: {cur['total_line']:g}")

    plot_df = g.dropna(subset=[model_col, market_col])
    if len(plot_df) < 1:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["logged_at"], y=plot_df[model_col],
                             name="Model", mode="lines+markers",
                             line=dict(color="#2E86DE", width=2)))
    fig.add_trace(go.Scatter(x=plot_df["logged_at"], y=plot_df[market_col],
                             name="Market", mode="lines+markers",
                             line=dict(color="#E67E22", width=2)))
    fig.update_layout(
        height=240, yaxis_tickformat=".0%",
        yaxis_title=market["y_label"], xaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _flagged_snapshots(df: pd.DataFrame, markets: list[dict]) -> pd.DataFrame:
    """One row per (snapshot, market) that was flagged, across all markets --
    a spread flag and a moneyline flag on the same snapshot both show up as
    their own rows, tagged by which market flagged."""
    parts = []
    for market in markets:
        if market["not_modeled"] or market["flagged_col"] not in df.columns:
            continue
        sub = df[df[market["flagged_col"]] == True]  # noqa: E712 (pandas bool column)
        if sub.empty:
            continue
        parts.append(pd.DataFrame({
            "Time UTC": sub["logged_at"].dt.strftime("%H:%M:%S"),
            "Game": sub["matchup"], "State": sub["state_desc"],
            "Market": market["label"],
            "Model%": (sub[market["model_col"]] * 100).round(1),
            "Book%": (sub[market["market_col"]] * 100).round(1),
            "Edge pts": (sub[market["edge_col"]].abs() * 100).round(1),
            "Model favors": sub[market["pick_col"]],
            "_sort": sub["logged_at"],
        }))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True).sort_values("_sort", ascending=False)
    return out.drop(columns="_sort")


st.title("🚦 Live Betting Signals")
st.caption(
    "In-game model win% vs the de-vigged live market -- moneyline, spread, and "
    "(MLB only, see the Total tab) totals -- across every tracked sport. "
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
            markets = _markets_for(sport_key)

            if df.empty:
                st.info(
                    f"No {adapter.SPORT_LABEL} live-signal rows logged yet. Once a game "
                    "goes in progress, the scheduled poller appends rows here (checks every "
                    "few minutes -- see .github/workflows/poll-live-signals.yml).")
                st.caption(f"Checked {now_utc:%H:%M:%S} UTC")
                return

            last = df["logged_at"].max()
            latest = df.groupby("game_id").tail(1)
            any_flag_cols = [m["flagged_col"] for m in markets
                             if not m["not_modeled"] and m["flagged_col"] in latest.columns]
            n_flagged = int(latest[any_flag_cols].any(axis=1).sum()) if any_flag_cols else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Games tracked", df["game_id"].nunique())
            c2.metric("Signal rows", len(df))
            c3.metric("Currently flagged (any market)", n_flagged)
            c4.metric("Last row (UTC)", f"{last:%H:%M:%S}")
            st.caption(f"Refreshed {now_utc:%H:%M:%S} UTC"
                       + (f" · auto every {interval}s" if auto else " · auto-refresh off"))
            st.divider()

            for game_id, g in df.groupby("game_id"):
                cur = g.iloc[-1]
                st.subheader(f"{cur['matchup']} — {cur['state_desc']}")
                st.caption(
                    f"{cur['away_team']} {int(cur['away_score'])}–{int(cur['home_score'])} "
                    f"{cur['home_team']}")

                market_tabs = st.tabs([m["label"] for m in markets])
                for mtab, market in zip(market_tabs, markets):
                    with mtab:
                        _render_market(g, cur, market, f"chart_{sport_key}_{game_id}_{market['key']}")

                st.caption(
                    f"Odds source: {cur['odds_source']} · de-vig {cur['devig_method']} · "
                    f"{len(g)} state snapshot(s)")
                st.divider()

            fl = _flagged_snapshots(df, markets)
            if not fl.empty:
                st.subheader(f"Flagged snapshots ({len(fl)})")
                st.dataframe(fl, width="stretch", hide_index=True)

        _render()

st.sidebar.markdown("---")
st.sidebar.caption("Live Betting Signals · unified dashboard")
st.sidebar.caption("OBSERVE-ONLY -- no bet-placement code exists in this app.")
st.sidebar.caption(f"Sports tracked: {', '.join(a.SPORT_LABEL for a in SPORTS.values())}")
