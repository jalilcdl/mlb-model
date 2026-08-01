"""
Streamlit dashboard for the MLB prediction model.

Run with:  streamlit run dashboard/app.py
(or use the run_dashboard script in the project root)

Navigation is a session-state "website" flow, not tabs: a Scoreboard homepage of
clickable game cards -> a per-game detail view -> back to the scoreboard, plus
top-nav sections for the Bet log, Team ratings, Model performance, and About.
The click-through is driven by st.session_state["view"] / ["game_pk"], set from
on_click callbacks (which run before the rerun, so no explicit st.rerun needed).
"""
import datetime as dt
import hmac
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, pipeline
from src import bet_tracker
from src.data import fetch_historical, highlightly, injuries
from src.models import monte_carlo
from src.backtest import backtest as backtest_mod
from src.reports import game_detail

st.set_page_config(page_title="MLB Prediction Model", layout="wide", page_icon=":material/sports_baseball:")


# ---------------------------------------------------------------------------
# Optional password gate (opt-in via the MLB_DASHBOARD_PASSWORD env var)
# ---------------------------------------------------------------------------
# Off by default: with no env var set, this is a no-op and the dashboard opens
# straight to the app -- which is the right default when the only way to reach
# it is a private Tailscale mesh of your own devices (Tailscale itself is the
# access control). Set MLB_DASHBOARD_PASSWORD (see run_dashboard_remote.bat /
# REMOTE_ACCESS.md) to require a password too -- worth doing if you ever expose
# it more broadly than a private tailnet (e.g. a public tunnel). hmac.compare_digest
# avoids leaking the answer via response-time differences.
def _check_password():
    expected = os.environ.get("MLB_DASHBOARD_PASSWORD")
    if not expected:
        return True  # no password configured -> no gate
    if st.session_state.get("_password_ok"):
        return True

    entered = st.text_input("Password", type="password", key="_password_input")
    if entered == "":
        st.caption("This dashboard is password-protected. Enter the password to continue.")
        st.stop()
    if hmac.compare_digest(entered, expected):
        st.session_state["_password_ok"] = True
        st.rerun()
    else:
        st.error("Incorrect password.")
        st.stop()


_check_password()


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
def _games_mtime():
    return config.GAMES_FILE.stat().st_mtime if config.GAMES_FILE.exists() else None


@st.cache_resource(show_spinner=False)
def _fit_models_cached(games_mtime):
    # games_mtime is part of the cache key (not underscore-prefixed) so this
    # automatically refits if games.csv changes on disk, without needing an
    # explicit cache-clear button press.
    games = pipeline.load_games()
    elo, ratings = pipeline.fit_models(games)
    return games, elo, ratings


@st.cache_data(show_spinner=True, ttl=900)
def get_predictions(date_str, use_pitcher_adjustment, n_sims, games_mtime):
    games, elo, ratings = _fit_models_cached(games_mtime)
    return pipeline.predict_date(
        date_str,
        games=games,
        elo=elo,
        ratings=ratings,
        use_pitcher_adjustment=use_pitcher_adjustment,
        n_sims=n_sims,
    )


@st.cache_data(show_spinner=False, ttl=3600)
def _team_stats_cached(games_mtime):
    """Team pitching (FIP) + fielding (Savant OAA) + park exposure for the grades.

    Network-backed, so cached for an hour: these are season-to-date totals that
    move slowly, and a dashboard rerun should not re-hit the APIs. Degrades to
    empty frames rather than crashing if a source is unreachable -- the grades
    then show N/A instead of the page failing.
    """
    from src.data import team_stats
    try:
        pitching = team_stats.fetch_team_pitching()
    except Exception:
        pitching = pd.DataFrame()
    try:
        fielding = team_stats.fetch_team_fielding()
    except Exception:
        fielding = pd.DataFrame()
    try:
        exposure = team_stats.park_exposure(pipeline.load_games())
    except Exception:
        exposure = {}
    return pitching, fielding, exposure


@st.cache_data(show_spinner=False)
def _load_backtest_summary(summary_mtime):
    if not config.BACKTEST_SUMMARY_FILE.exists():
        return None
    with open(config.BACKTEST_SUMMARY_FILE) as f:
        return json.load(f)


def confidence_label(win_prob):
    dist = abs(win_prob - 0.5)
    if dist < 0.05:
        return "Toss-up"
    if dist < 0.15:
        return "Lean"
    if dist < 0.25:
        return "Confident"
    return "Strong"


def edge_label(edge):
    if edge is None or pd.isna(edge):
        return None
    ae = abs(edge)
    if ae < 0.02:
        return "No real edge"
    if ae < 0.05:
        return "Small edge"
    if ae < 0.08:
        return "Medium edge"
    return "Large edge"


def pct(x, digits=1):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "-"
    return f"{x * 100:.{digits}f}%"


_EDGE_COL_MAP = {
    "ml_home": ("edge_home_ml", "ev_home_ml_pct"),
    "ml_away": ("edge_away_ml", "ev_away_ml_pct"),
    "total_over": ("edge_over", "ev_over_pct"),
    "total_under": ("edge_under", "ev_under_pct"),
    "rl_home": ("edge_home_rl", "ev_home_rl_pct"),
    "rl_away": ("edge_away_rl", "ev_away_rl_pct"),
}


def _render_side_edge(g, prefix, side_label):
    """One line summarizing the best available price, the de-vigged consensus
    across every book we have odds for, and the model's edge/EV against it."""
    best_odds = g.get(f"{prefix}_best_odds")
    if pd.isna(best_odds):
        return
    consensus = g.get(f"{prefix}_consensus_prob")
    best_book = g.get(f"{prefix}_best_book")
    n_books = g.get(f"{prefix}_n_books")
    edge_col, ev_col = _EDGE_COL_MAP[prefix]
    edge = g.get(edge_col)
    ev = g.get(ev_col)

    odds_str = f"{'+' if best_odds > 0 else ''}{int(best_odds)}"
    books_str = f" ({int(n_books)} books)" if pd.notna(n_books) else ""
    line = f"Best: {side_label} {odds_str} @ {best_book}{books_str}"
    if pd.notna(consensus):
        line += f" — consensus {pct(consensus)}"
    if pd.notna(edge):
        line += f" — edge {pct(edge)} ({edge_label(edge)})"
    if pd.notna(ev):
        line += f" — EV {pct(ev)}"
    st.caption(line)


def _render_total_histogram(g):
    """Live re-simulation (using the same mu's and overdispersion behind the
    displayed probabilities) so the total-runs distribution is visible, not
    just its mean -- the whole point of Monte Carlo over a single point
    estimate."""
    sim = monte_carlo.simulate_game(
        g["expected_home_runs"], g["expected_away_runs"], n_sims=n_sims, overdispersion=g.get("overdispersion")
    )
    total = sim["home_runs"] + sim["away_runs"]
    fig = go.Figure(go.Histogram(x=total, histnorm="probability"))
    line_val = g.get("total_line")
    if pd.notna(line_val):
        fig.add_vline(x=line_val, line_dash="dash", line_color="red", annotation_text=f"Line {line_val:.1f}")
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Simulated total runs",
        yaxis_title="Probability",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def _logo_chip(team, size=24):
    """Team logo on a small white rounded chip. Several MLB logos are dark navy
    or black (NYY, CWS, SD, TB...) and nearly vanish directly on the dark canvas,
    so every logo sits on a consistent light chip -- the same treatment ESPN /
    theScore use on their dark scoreboards. Returns an HTML string for st.markdown
    with unsafe_allow_html=True."""
    return (
        f'<span style="background:#ffffff;border-radius:8px;padding:3px;'
        f'display:inline-flex;align-items:center;justify-content:center;">'
        f'<img src="{game_detail.logo_url(team)}" width="{size}" height="{size}" '
        f'style="display:block;object-fit:contain;"></span>'
    )


def _win_prob_bar(g):
    """A single stacked horizontal bar showing the model's away/home win split --
    the core validated Monte Carlo + Elo output, shown visually rather than as
    two bare numbers."""
    away, home = g["away_team"], g["home_team"]
    ap, hp = float(g["away_win_prob"]), float(g["home_win_prob"])
    fig = go.Figure()
    # Fills chosen to read clearly on the dark canvas; white inside-text is pinned
    # explicitly so the label contrast doesn't depend on Plotly's auto choice.
    fig.add_trace(go.Bar(x=[ap * 100], y=["win %"], orientation="h", name=away,
                         marker_color="#2f80ed", text=f"{away} {pct(ap)}", textposition="inside",
                         insidetextanchor="middle", textfont=dict(color="white", size=13)))
    fig.add_trace(go.Bar(x=[hp * 100], y=["win %"], orientation="h", name=home,
                         marker_color="#eb5757", text=f"{home} {pct(hp)}", textposition="inside",
                         insidetextanchor="middle", textfont=dict(color="white", size=13)))
    fig.update_layout(
        barmode="stack", height=90, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 100], title=None, showticklabels=False),
        yaxis=dict(showticklabels=False), showlegend=False,
    )
    fig.add_vline(x=50, line_dash="dot", line_color="#9e9e9e")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Navigation state + helpers
# ---------------------------------------------------------------------------
st.session_state.setdefault("view", "home")      # home | game | bets | ratings | performance | about
st.session_state.setdefault("game_pk", None)     # which game the detail view shows
st.session_state.setdefault("board_date", dt.date.today())


def _go(view):
    """Top-nav navigation. Leaving a game clears the selected game."""
    st.session_state.view = view
    if view != "game":
        st.session_state.game_pk = None


def _open_game(pk):
    st.session_state.view = "game"
    st.session_state.game_pk = pk


def _shift_date(days):
    st.session_state.board_date = st.session_state.board_date + dt.timedelta(days=days)


# ---------------------------------------------------------------------------
# Sidebar (global settings — separate from the top-nav navigation)
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")

if not config.GAMES_FILE.exists():
    st.sidebar.error("No historical data found yet.")
    st.sidebar.write("Run this once from the project root:")
    st.sidebar.code("python -m src.data.fetch_historical", language="bash")
    st.stop()

data_dt = dt.datetime.fromtimestamp(_games_mtime())
st.sidebar.caption(f"Historical data last updated: {data_dt:%Y-%m-%d %H:%M}")

use_pitcher_adj = st.sidebar.checkbox("Apply starting-pitcher adjustment", value=True)

n_sims = st.sidebar.number_input(
    "Simulations per game",
    min_value=config.MC_MIN_SIMS,
    max_value=config.MC_MAX_SIMS,
    value=config.MC_DEFAULT_SIMS,
    step=1000,
    help="Monte Carlo simulation is the primary prediction method — more simulations "
    "reduce sampling noise but take longer to run.",
)

snapshots_dir = config.ODDS_DIR / "snapshots"
snapshot_files = list(snapshots_dir.glob("*.json")) if snapshots_dir.exists() else []
legacy_odds_path = config.ODDS_DIR / "odds.json"
if snapshot_files:
    st.sidebar.success(
        f"{len(snapshot_files)} odds snapshot file(s) found (odds/snapshots/) — "
        "multi-book edges will be shown."
    )
elif legacy_odds_path.exists():
    st.sidebar.success("Manual odds file found (odds/odds.json) — edges will be shown.")
else:
    st.sidebar.info(
        "No odds found. Drop real multi-sportsbook snapshot files into odds/snapshots/ "
        "(preferred), or copy odds/odds_example.json to odds/odds.json for a quick "
        "single-book manual entry, to see model-vs-market edges."
    )

if config.HIGHLIGHTLY_API_KEY:
    if st.sidebar.button("Fetch live odds (Highlightly)", icon=":material/download:"):
        try:
            status = highlightly.refresh(st.session_state.board_date.isoformat())
            get_predictions.clear()
            msg = (f"Pulled {status['offer_items']} odds markets · logged "
                   f"{status['totals_logged']} totals to history.")
            if status["unresolved_teams"]:
                msg += f" ⚠️ {len(status['unresolved_teams'])} game(s) had unmapped team names (skipped)."
            st.sidebar.success(msg)
        except highlightly.HighlightlyError as exc:
            st.sidebar.error(f"Highlightly fetch failed: {exc}")
else:
    st.sidebar.caption(
        "Set HIGHLIGHTLY_API_KEY (free tier) to pull live odds in-app — see README. "
        "Meanwhile, drop snapshot files into odds/snapshots/."
    )

if st.sidebar.button("Refresh probable pitchers & re-run", icon=":material/refresh:"):
    get_predictions.clear()
    st.sidebar.caption("Refreshed — re-pulling from the MLB Stats API now.")

st.sidebar.caption(
    "Re-pulls just today's schedule, probable pitchers, and pitcher stats from the live "
    "MLB Stats API and re-runs projections (a few seconds) — separate from, and much "
    "faster than, the historical rebuild below. Predictions also auto-refresh every "
    "15 minutes on their own; use this button to check right now instead of waiting."
)

with st.sidebar.expander("Rebuild historical dataset (slow)"):
    st.write(
        "Re-scrapes Baseball-Reference for all 30 teams across "
        f"{config.HISTORICAL_SEASONS + [config.CURRENT_SEASON]}. Takes several minutes "
        "and is rate-limited to be polite to the source site."
    )
    if st.button("Rebuild now"):
        with st.spinner("Pulling historical game logs... this can take 5-10 minutes."):
            fetch_historical.build_and_save()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Historical dataset rebuilt.")
        st.rerun()


# ---------------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------------
st.title("MLB Prediction Model")

_NAV = [
    ("home", "Scoreboard", ":material/scoreboard:"),
    ("bets", "Bet log", ":material/receipt_long:"),
    ("ratings", "Team ratings", ":material/leaderboard:"),
    ("performance", "Model performance", ":material/verified:"),
    ("about", "About & limitations", ":material/info:"),
]
_active = st.session_state.view
with st.container(horizontal=True):
    for key, label, icon in _NAV:
        # "game" is a sub-view of the scoreboard, so Scoreboard stays highlighted there.
        is_active = (key == _active) or (key == "home" and _active == "game")
        st.button(
            label, icon=icon, key=f"nav_{key}",
            type="primary" if is_active else "secondary",
            on_click=_go, args=(key,), width="stretch",
        )
st.divider()


# ---------------------------------------------------------------------------
# View: Scoreboard (homepage) — clickable game cards
# ---------------------------------------------------------------------------
def render_scoreboard():
    # Date bar with prev/next arrows, like a real scoreboard page.
    left, mid, right, spacer = st.columns([1, 3, 1, 6])
    left.button("Prev", icon=":material/chevron_left:", on_click=_shift_date, args=(-1,),
                width="stretch", key="date_prev")
    board_date = mid.date_input("Date", key="board_date", label_visibility="collapsed")
    right.button("Next", icon=":material/chevron_right:", on_click=_shift_date, args=(1,),
                 width="stretch", key="date_next")
    date_str = board_date.isoformat()

    try:
        preds = get_predictions(date_str, use_pitcher_adj, n_sims, _games_mtime())
    except Exception as exc:
        st.error(f"Couldn't fetch/predict games for {date_str}: {exc}")
        return

    if preds.empty:
        st.info(f"No MLB games found for {date_str}.")
        return

    st.caption(f"{len(preds)} game(s) · {date_str}")
    if "pitchers_fetched_at" in preds.columns and preds["pitchers_fetched_at"].notna().any():
        fetched_times = pd.to_datetime(preds["pitchers_fetched_at"])
        st.caption(
            f"Probable pitchers last pulled at {fetched_times.max():%H:%M:%S} · auto-refreshes "
            "every 15 min (Settings sidebar to force it). Click a game for the full breakdown."
        )
    changed_mask = preds.get("home_pitcher_changed", pd.Series(dtype=bool)).fillna(False) | preds.get(
        "away_pitcher_changed", pd.Series(dtype=bool)).fillna(False)
    if changed_mask.any():
        chg = preds.loc[changed_mask, ["away_team", "home_team"]]
        matchups = ", ".join(f"{r.away_team} @ {r.home_team}" for r in chg.itertuples())
        st.warning(f"⚠️ Probable pitcher changed since the last check for: {matchups} — open the game for details.")

    # Card grid, 3 across.
    rows = list(preds.iterrows())
    for i in range(0, len(rows), 3):
        cols = st.columns(3)
        for col, (_, g) in zip(cols, rows[i:i + 3]):
            with col:
                _game_card(g)


def _game_card(g):
    away, home = g["away_team"], g["home_team"]
    pk = g["game_pk"]
    when = str(g.get("game_datetime_utc") or "")
    tstr = f"{when[11:16]}Z" if len(when) >= 16 else ""
    status = g.get("status") or ""

    # Tiny totals badge for the top-right corner: the model's over/under lean vs a
    # loaded market line if one is present, else the model's own projected total.
    # Deliberately small and unobtrusive -- totals is REFERENCE-ONLY (not a
    # validated market), which the hover tooltip states outright.
    et = g.get("expected_total")
    tline = g.get("total_line")
    mover = g.get("model_over_prob")
    if pd.notna(tline) and pd.notna(mover):
        lean = "Over" if mover >= 0.5 else "Under"
        ou_txt = f"O/U {tline:g} · {lean[0]}"
        ou_tip = (f"Model leans {lean} the {tline:g} line ({pct(mover)} over). "
                  "Totals is reference-only — not a validated market.")
    elif pd.notna(et):
        ou_txt = f"O/U {float(et):.1f}"
        ou_tip = ("Model's projected total runs (no market line loaded). "
                  "Reference-only — totals is not a validated market.")
    else:
        ou_txt = ou_tip = ""
    badge_html = (
        f'<span title="{ou_tip}" style="font-size:0.72rem;padding:1px 8px;'
        f'border-radius:10px;background:rgba(255,255,255,0.08);color:#aeb4bf;'
        f'white-space:nowrap;">{ou_txt}</span>' if ou_txt else ""
    )
    status_txt = f" · {status}" if status and status != "Scheduled" else ""

    with st.container(border=True):
        st.markdown(
            f'<div style="display:flex;align-items:center;justify-content:space-between;'
            f'gap:8px;margin-bottom:4px;">'
            f'<span style="font-size:0.8rem;color:#8b929e;">&#128337; {tstr}{status_txt}</span>'
            f'{badge_html}</div>',
            unsafe_allow_html=True,
        )

        # Matchup with logos. `sp` is that side's probable starter -- the SAME
        # field driving the PDF's "*" caveat -- so an "SP TBD" marker appears
        # whenever a starter isn't announced yet and clears itself on the next
        # refresh once the schedule confirms one (get_predictions re-pulls live).
        for side, team, prob, sp in (
            ("away", away, g["away_win_prob"], g.get("away_probable_pitcher")),
            ("home", home, g["home_win_prob"], g.get("home_probable_pitcher")),
        ):
            # "@ " marks the home side. Logo sits on a light chip (see _logo_chip)
            # so dark team logos stay legible on the dark canvas.
            prefix = "@ " if side == "home" else ""
            tbd = pd.isna(sp) or str(sp).strip().lower() in ("", "tbd", "nan", "none")
            sp_marker = (
                '<span title="Probable starter not announced yet — this team is modelled at '
                'league average until one is confirmed (updates automatically on refresh)." '
                'style="font-size:0.6rem;font-weight:600;padding:0 5px;border-radius:8px;'
                'background:rgba(255,193,7,0.18);color:#ffca7a;white-space:nowrap;'
                'margin-left:6px;vertical-align:middle;">SP TBD</span>'
            ) if tbd else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                f'{_logo_chip(team, 24)}'
                f'<span style="font-size:0.95rem;">{prefix}<b>{team}</b> · {pct(prob)}{sp_marker}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Quick-glance stat: the model's pick + best available ML edge if odds loaded.
        pick = g.get("moneyline_pick")
        conf = confidence_label(g["moneyline_pick_prob"]) if pd.notna(g.get("moneyline_pick_prob")) else ""
        line = f"Model pick: **{pick} ML** ({conf})"
        if g.get("odds_available"):
            edges = {"ml_away": g.get("edge_away_ml"), "ml_home": g.get("edge_home_ml")}
            best = max((v for v in edges.values() if pd.notna(v)), key=abs, default=None)
            if best is not None:
                side = "ml_home" if edges["ml_home"] is best else "ml_away"
                team = home if side == "ml_home" else away
                line += f" · {team} edge **{best * 100:+.1f} pts** ({edge_label(best)})"
        st.caption(line)

        st.button("View game", icon=":material/arrow_forward:", key=f"card_{pk}",
                  on_click=_open_game, args=(pk,), width="stretch")


# ---------------------------------------------------------------------------
# View: single-game detail (reached by clicking a card)
# ---------------------------------------------------------------------------
def render_game_detail():
    st.button("Back to scoreboard", icon=":material/arrow_back:", on_click=_go, args=("home",), key="back_top")

    date_str = st.session_state.board_date.isoformat()
    try:
        preds_d = get_predictions(date_str, use_pitcher_adj, n_sims, _games_mtime())
    except Exception as exc:
        st.error(f"Couldn't load games for {date_str}: {exc}")
        return
    pk = st.session_state.game_pk
    match = preds_d[preds_d["game_pk"] == pk] if not preds_d.empty else preds_d
    if match.empty:
        st.info("That game isn't on the selected date anymore. Head back to the scoreboard.")
        return
    g = match.iloc[0]
    away, home = g["away_team"], g["home_team"]

    games_d, elo_d, ratings_d = _fit_models_cached(_games_mtime())
    records = game_detail.team_records(games_d)
    diag = game_detail.model_diagnostics(g, ratings_d)

    # ---- Header ----
    h1, h2, h3 = st.columns([3, 2, 3])
    with h1:
        lc, tc = st.columns([1, 3])
        with lc:
            st.markdown(_logo_chip(away, 56), unsafe_allow_html=True)
        with tc:
            st.markdown(f"### {away}")
        st.caption(f"{g.get('away_team_name','')} · {game_detail.format_record(records, away)}")
        st.metric("Projected runs", f"{g['expected_away_runs']:.2f}")
    with h2:
        st.markdown("### @")
        when = str(g.get("game_datetime_utc") or "")
        st.caption(f"{when[:10]} {when[11:16]}Z" if when else "")
        st.caption(g.get("venue_name") or "")
        st.caption(f"Status: {g.get('status','')}")
    with h3:
        lc, tc = st.columns([1, 3])
        with lc:
            st.markdown(_logo_chip(home, 56), unsafe_allow_html=True)
        with tc:
            st.markdown(f"### {home}")
        st.caption(f"{g.get('home_team_name','')} · {game_detail.format_record(records, home)}")
        st.metric("Projected runs", f"{g['expected_home_runs']:.2f}")

    # ---- Monte Carlo headline: win-prob split + honest framing ----
    st.subheader("Monte Carlo win probability")
    _win_prob_bar(g)
    st.caption(
        f"Model pick **{g['moneyline_pick']}** ({confidence_label(g['moneyline_pick_prob'])}). "
        f"Blend of Elo ({pct(g['elo_home_win_prob'])} home) and Monte Carlo "
        f"({int(g['n_sims']):,} sims: {pct(g['mc_home_win_prob'])} home; closed-form cross-check "
        f"{pct(g['cf_home_win_prob'])}). Moneyline is the one market validated out-of-sample."
    )
    st.caption(
        f"Projected final: **{away} {g['expected_away_runs']:.1f} — "
        f"{home} {g['expected_home_runs']:.1f}**  (total {g['expected_total']:.2f}). "
        "Simulated means, not integer score predictions."
    )
    # Starting pitchers. Same fields as the scoreboard's "SP TBD" marker; a NaN
    # here must read as "not announced" rather than a literal "nan" (note NaN is
    # truthy in Python, so the old `x or 'TBD'` fallback silently printed "nan").
    def _tbd(sp):
        return pd.isna(sp) or str(sp).strip().lower() in ("", "tbd", "nan", "none")
    a_sp, h_sp = g.get("away_probable_pitcher"), g.get("home_probable_pitcher")
    st.caption(
        f"Starting pitchers — **{away}:** {'TBD — not announced yet' if _tbd(a_sp) else a_sp}"
        f"  ·  **{home}:** {'TBD — not announced yet' if _tbd(h_sp) else h_sp}"
    )
    if _tbd(a_sp) or _tbd(h_sp):
        st.caption(
            "A pitcher shown as **TBD** isn't confirmed yet, so that side is modelled at league "
            "average until the starter is announced — this updates automatically on refresh."
        )
    if g.get("away_pitcher_changed"):
        st.warning(
            f"⚠️ Away pitcher changed since last check: was **{g.get('away_pitcher_previous')}**, "
            f"now **{g.get('away_probable_pitcher')}**. Projections already reflect the new pitcher.")
    if g.get("home_pitcher_changed"):
        st.warning(
            f"⚠️ Home pitcher changed since last check: was **{g.get('home_pitcher_previous')}**, "
            f"now **{g.get('home_probable_pitcher')}**. Projections already reflect the new pitcher.")
    st.divider()

    # ---- Model vs market, per market (validated / reference-only framing intact) ----
    st.subheader("Model vs market")
    m_ml, m_tot, m_rl = st.tabs(["Moneyline", "Total (O/U)", "Run Line"])

    def _market_block(market_key, sides):
        status, evidence = game_detail.MARKET_VALIDATION[market_key]
        (st.success if status == "VALIDATED" else st.warning)(f"**{status}** — {evidence}")
        for label, model_p, mkt_p, best, book in sides:
            if pd.isna(model_p):
                continue
            st.markdown(f"#### {label}")
            if pd.isna(mkt_p):
                st.info(f"Model: **{pct(model_p)}** · no market price loaded for this side.")
                continue
            edge_pts = (model_p - mkt_p) * 100
            stars, note = game_detail.star_rating(edge_pts, market_key, diag)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", pct(model_p))
            c2.metric("Market (de-vig)", pct(mkt_p))
            c3.metric("Edge", f"{edge_pts:+.1f} pts")
            c4.metric("Best price", f"{int(best):+d}" if pd.notna(best) else "n/a")
            st.markdown(f"**{'★' * stars}{'☆' * (5 - stars)}**  ({stars}/5) — {note}")
            lead, verdict = game_detail.recommendation_text(
                label, model_p, mkt_p, best if pd.notna(best) else None, book, market_key)
            st.markdown(lead)
            st.markdown(f"> {verdict}")
            st.divider()

    with m_ml:
        _market_block("moneyline", [
            (away, g["away_win_prob"], g.get("ml_away_consensus_prob"),
             g.get("ml_away_best_odds"), g.get("ml_away_best_book")),
            (home, g["home_win_prob"], g.get("ml_home_consensus_prob"),
             g.get("ml_home_best_odds"), g.get("ml_home_best_book")),
        ])
    with m_tot:
        line = g.get("total_line")
        over_c = g.get("total_over_consensus_prob")
        model_over = (over_c + g.get("edge_over")) if pd.notna(over_c) and pd.notna(g.get("edge_over")) else np.nan
        _market_block("total", [
            (f"Over {line:g}" if pd.notna(line) else "Over", model_over, over_c,
             g.get("total_over_best_odds"), g.get("total_over_best_book")),
            (f"Under {line:g}" if pd.notna(line) else "Under",
             (1 - model_over) if pd.notna(model_over) else np.nan,
             g.get("total_under_consensus_prob"),
             g.get("total_under_best_odds"), g.get("total_under_best_book")),
        ])
        if pd.notna(g.get("total_push_prob")) and g["total_push_prob"] > 0:
            st.caption(f"Whole-number line: **{pct(g['total_push_prob'])}** chance of a push (stake returned).")
        st.markdown("**Simulated total-runs distribution**")
        _render_total_histogram(g)
    with m_rl:
        _market_block("run_line", [
            (f"{away} +{g['run_line']:g}", g["away_covers_prob"], g.get("rl_away_consensus_prob"),
             g.get("rl_away_best_odds"), g.get("rl_away_best_book")),
            (f"{home} -{g['run_line']:g}", g["home_covers_prob"], g.get("rl_home_consensus_prob"),
             g.get("rl_home_best_odds"), g.get("rl_home_best_book")),
        ])
    st.caption("First-5-innings markets are not offered here: the simulation models whole-game "
               "run totals and has no inning structure, so any first-half number would be invented.")

    # ---- Team comparison ----
    st.subheader("Team comparison")
    pitch_df, field_df, exposure = _team_stats_cached(_games_mtime())
    rows_cmp = []
    for side, team in (("away", away), ("home", home)):
        tg = game_detail.team_grades(team, elo_d, ratings_d, pitching_df=pitch_df,
                                     fielding_df=field_df, exposure=exposure)
        p_letter, p_caveats = game_detail.pitcher_grade(g.get(f"{side}_pitcher_factor"))
        rows_cmp.append({
            "": team,
            "Record": game_detail.format_record(records, team),
            "Proj runs": round(float(g[f"expected_{side}_runs"]), 2),
            "Proj win %": pct(g[f"{side}_win_prob"]),
            "Home field": "✓ (+24 Elo)" if side == "home" else "—",
            "Team": tg["team"]["grade"],
            "Offense": tg["offense"]["grade"],
            "Run prev.": tg["run_prevention"]["grade"],
            "Pitching": tg["pitching"]["grade"],
            "Fielding": tg["fielding"]["grade"],
            "Starter": p_letter,
        })
    st.dataframe(pd.DataFrame(rows_cmp), hide_index=True, width="stretch")
    st.caption(
        "⚠️ **Fielding** uses Savant OAA, which counts only qualified fielders (~8 per club vs "
        "13+ who actually field) and is season-to-date with no rolling window — so it undercounts "
        "and can't be recency-weighted. **Pitching** is FIP (defense-independent), so it will "
        "disagree with runs allowed when a club's defense or luck is unusual — that disagreement "
        "is the point."
    )
    with st.expander("How these grades are derived (and what they don't mean)"):
        st.markdown(
            "- **Run prev. / Pitching / Fielding are three different things.** Run prevention is "
            "runs actually allowed (pitching + defense + luck combined). Pitching is **FIP** — "
            "home runs, walks, HBP and strikeouts only, which no fielder touches. Fielding is "
            "**Savant runs prevented**. A club can grade A in run prevention while its pitching "
            "and fielding grades diverge sharply; that split is real information the old combined "
            "grade hid.\n"
            "- **Team / Offense** — quintile rank across all 30 clubs (6 teams per "
            "letter): Elo for Team, park-adjusted runs scored for Offense. These are *relative* "
            "grades, not absolute quality.\n"
            "- **Offense and Run prev. are park-adjusted** — each game's runs are divided by that "
            "venue's park factor before averaging. Without this, Coors manufactured grade spread "
            "out of thin air.\n"
            "- **Starter** — banded directly off the pitcher factor (1.00 = league average, lower "
            "better). ⚠️ This component has **no validated predictive lift** — the isolation "
            "backtest found no significant improvement in any market. Treat the letter as "
            "descriptive, not predictive.\n"
            "- **Home field** — the actual +24 Elo points the model applies, not a generic icon."
        )
        for side, team in (("away", away), ("home", home)):
            _, cav = game_detail.pitcher_grade(g.get(f"{side}_pitcher_factor"))
            if cav:
                st.markdown(f"**{team} starter caveats:**")
                for c in cav:
                    st.markdown(f"- {c}")

    # ---- Injured list (display-only context, never a model input) ----
    st.subheader("Injured list")
    st.caption(
        "📋 **Context only — this does NOT feed the model.** The API exposes only the "
        "*current* roster state with no history, so an injury adjustment could never be "
        "walk-forward backtested. Rather than ship a second unvalidated weight (the "
        "starting-pitcher adjustment is already one), the model ignores this entirely and "
        "leaves the judgement to you. 40-man roster only."
    )
    inj_cols = st.columns(2)
    for col, team in zip(inj_cols, (away, home)):
        with col:
            try:
                hurt = injuries.fetch_injuries(team)
            except Exception:
                hurt = []
            st.markdown(f"**{team}** — {injuries.summarize(hurt)}")
            if hurt:
                st.dataframe(pd.DataFrame(hurt).rename(
                    columns={"name": "Player", "position": "Pos", "status": "Status"}),
                    hide_index=True, width="stretch")

    # ---- Diagnostics (replaces BetQL's licensed money-flow panel) ----
    st.subheader("Model diagnostics")
    st.caption(
        "⚠️ This is **our own internal-agreement diagnostic — NOT sportsbook money-flow data.** "
        "BetQL's 'Sharp Bettor Report' (% of money vs % of tickets) uses licensed betting-flow "
        "data that none of our free sources provide, so it is deliberately not reproduced here "
        "rather than faked."
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Elo (home)", pct(diag["elo_home_prob"]))
    d2.metric("Monte Carlo (home)", pct(diag["mc_home_prob"]))
    d3.metric("Components agree", f"{diag['elo_mc_gap_pts']:.1f} pts apart",
              help="Elo ignores pitchers entirely; the sim uses them. A large gap means the "
                   "unvalidated pitcher adjustment is driving the number.")
    flags = []
    if diag["agreement"] == "weak":
        flags.append(f"Elo and the simulation disagree by {diag['elo_mc_gap_pts']:.0f} points — "
                     "the pitcher adjustment is doing most of the work here.")
    if diag["pitcher_clamped"]:
        flags.append("A starting-pitcher factor is **clamped at its guardrail** — the model is "
                     "extrapolating past where it's designed to work.")
    if diag["park_clipped"]:
        flags.append(f"Park factor is **at the clip limit** ({diag['park_factor']:.3f}) — this "
                     "park's true run environment is more extreme than the model can represent.")
    if flags:
        for f in flags:
            st.warning(f)
    else:
        st.success("No red flags: components agree, no clamped pitcher factor, park within range.")
    st.caption(
        f"Closed-form cross-check: {pct(diag['cf_home_prob'])} home · "
        f"overdispersion {diag['overdispersion']:.2f}x Poisson · "
        f"{int(diag['n_sims']):,} simulations"
    )
    st.divider()
    st.button("Back to scoreboard", icon=":material/arrow_back:", on_click=_go, args=("home",), key="back_bottom")


# ---------------------------------------------------------------------------
# View: Team ratings
# ---------------------------------------------------------------------------
def render_ratings():
    games, elo, ratings = _fit_models_cached(_games_mtime())

    st.subheader("Elo Power Rankings")
    elo_df = elo.ratings_table()
    fig = go.Figure(go.Bar(x=elo_df["elo"], y=elo_df["team"], orientation="h"))
    fig.update_layout(
        height=800, yaxis={"categoryorder": "total ascending"}, margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Offense / Defense / Park Ratings")
    st.caption(
        "Ratios relative to league average (1.00 = average). Offense/defense are "
        f"blended {config.RUN_MODEL_RECENT_WEIGHT:.0%} recent-form "
        f"(last {config.RUN_MODEL_RECENT_GAMES} team games) / "
        f"{1 - config.RUN_MODEL_RECENT_WEIGHT:.0%} full-sample, shrunk toward league "
        "average for small samples. Park factor uses each team's home games only."
    )
    st.dataframe(ratings.to_frame(), width="stretch", hide_index=True)

    st.caption(
        f"Empirical run-scoring overdispersion (variance/mean of runs scored per team-game, "
        f"estimated from the training data): **{ratings.overdispersion:.2f}x** a pure Poisson. "
        "This is what the Monte Carlo engine's negative-binomial sampling uses instead of "
        "assuming a textbook correction — see the Scoreboard and About views."
    )


# ---------------------------------------------------------------------------
# View: Model performance
# ---------------------------------------------------------------------------
def render_performance():
    summary = _load_backtest_summary(
        config.BACKTEST_SUMMARY_FILE.stat().st_mtime if config.BACKTEST_SUMMARY_FILE.exists() else None
    )
    if summary is None:
        st.info("No backtest has been run yet.")
        st.caption(
            "This walks the model forward through history with no lookahead and compares "
            "predictions to actual results. Takes roughly a minute."
        )
        if st.button("Run backtest now"):
            with st.spinner("Backtesting..."):
                _, summary = backtest_mod.run_and_save()
            st.rerun()
        return

    meta = summary.get("_meta", {})
    st.caption(
        f"Evaluated on seasons {meta.get('eval_seasons')} "
        f"(warm-up: {meta.get('warmup_seasons')}). Team-level model only — "
        "no starting-pitcher adjustment (see About view)."
    )
    if st.button("Re-run backtest"):
        with st.spinner("Backtesting..."):
            _, summary = backtest_mod.run_and_save()
        st.rerun()

    baselines = summary.get("baselines", {})

    st.subheader("Moneyline")
    ml_labels = [("elo", "Elo"), ("monte_carlo", "Monte Carlo"), ("closed_form", "Closed-form"), ("blended", "Blended")]
    ml_cols = st.columns(1 + len(ml_labels))
    with ml_cols[0]:
        st.metric(
            "Naive baseline (always pick home)",
            pct(baselines.get("always_pick_home_moneyline_accuracy")),
            help="Home teams just win more often (home-field advantage) — any real model needs to beat this.",
        )
    for col, (label, display) in zip(ml_cols[1:], ml_labels):
        m = summary[f"moneyline_{label}"]
        with col:
            st.metric(
                f"{display} accuracy",
                pct(m["accuracy"]),
                help=f"Brier {m['brier_score']:.4f} · Log loss {m['log_loss']:.4f} · n={m['n_games']}",
            )
    beat_baseline = summary["moneyline_blended"]["accuracy"] - baselines.get(
        "always_pick_home_moneyline_accuracy", 0
    )
    st.caption(
        f"Blended model beats the naive home-favorite baseline by {beat_baseline*100:+.1f} points — "
        "a real, if modest, signal. Brier score and log loss measure calibration, not just accuracy "
        "(lower is better). A coin-flip model scores ~0.25 Brier / ~0.69 log loss; MLB games are "
        "genuinely hard to predict game-to-game — even strong favorites lose ~35-40% of the time. "
        f"Monte Carlo and closed-form disagree on win probability by "
        f"{pct(summary.get('mc_vs_closed_form_mean_abs_diff'))} on average across the backtest — "
        "that gap is mostly the negative-binomial overdispersion correction the closed-form math "
        "doesn't have (see Total Runs below), not a sign of a bug."
    )

    cal = pd.DataFrame(summary["moneyline_calibration_blended"])
    if not cal.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=cal["mean_predicted"], y=cal["actual_rate"], mode="markers+lines", name="Model")
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash"))
        )
        fig.update_layout(
            title="Moneyline calibration (predicted vs actual home win rate)",
            xaxis_title="Mean predicted probability",
            yaxis_title="Actual win rate",
            height=450,
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Run Line (-1.5 / +1.5)")
    rl = summary["run_line"]
    rl_cols = st.columns(2)
    with rl_cols[0]:
        st.metric(
            "Naive baseline (always pick underdog +1.5)",
            pct(baselines.get("always_pick_away_run_line_accuracy")),
            help="Underdogs cover +1.5 whenever they don't lose by 2+ runs — that's most games, by MLB's own math.",
        )
    with rl_cols[1]:
        st.metric(
            "Model accuracy", pct(rl["accuracy"]),
            help=f"Brier {rl['brier_score']:.4f} · Log loss {rl['log_loss']:.4f} · n={rl['n_games']}",
        )
    rl_beat = rl["accuracy"] - baselines.get("always_pick_away_run_line_accuracy", 0)
    st.warning(
        f"The model beats the naive underdog-baseline by only {rl_beat*100:+.1f} points — "
        "run-line 'accuracy' on its own is close to meaningless here because the underdog side "
        "covers most games by construction. Treat run-line picks as informative mainly through "
        "the *probability* they give you (for edge-finding against real odds), not as a "
        "standalone win/loss record."
    )

    st.subheader("Total Runs")
    tr_mc = summary["total_runs_monte_carlo"]
    tr_cf = summary["total_runs_closed_form"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Monte Carlo MAE / RMSE", f"{tr_mc['mae']:.2f} / {tr_mc['rmse']:.2f} runs")
    c2.metric("Closed-form MAE / RMSE", f"{tr_cf['mae']:.2f} / {tr_cf['rmse']:.2f} runs")
    c3.metric(
        "Predicted vs actual mean",
        f"{tr_mc['mean_predicted_total']:.2f} / {tr_mc['mean_actual_total']:.2f}",
    )
    st.caption(
        f"Monte Carlo 50% predictive interval coverage: {pct(tr_mc['coverage_50pct_interval'])} "
        f"(target ~50%) · 80% interval coverage: {pct(tr_mc['coverage_80pct_interval'])} "
        "(target ~80%) — computed from each game's actual simulated distribution, not a formula. "
        "This is the payoff of using Monte Carlo with an empirically-fit overdispersion: the "
        "closed-form (pure-Poisson) model's equivalent coverage runs meaningfully low (its "
        "intervals are too narrow because it doesn't know MLB run totals are more spread out than "
        "a true Poisson) — MAE/RMSE barely move between the two because they're about the *mean*, "
        "which overdispersion doesn't change, only the *spread* around it. No historical "
        "sportsbook total lines were available to grade over/under picks directly against a real "
        "market number — see About view."
    )


# ---------------------------------------------------------------------------
# View: Bet log
# ---------------------------------------------------------------------------
def render_bets():
    st.subheader("Forward-test bet log")
    st.caption(
        "The picks you actually acted on, and how they're doing — a real, out-of-sample "
        "track record that grows over time. Edit "
        f"`{config.BET_LOG_FILE.name}` to add bets or fill in results; this refreshes on reload."
    )
    bet_df = bet_tracker.load_log()
    if bet_df.empty:
        st.info(f"No bets logged yet. Add rows to {config.BET_LOG_FILE}.")
        return

    s = bet_tracker.summarize(bet_df)
    o = s["overall"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Record (W-L-P)", f"{o['wins']}-{o['losses']}-{o['pushes']}", help=f"{o['pending']} pending")
    c2.metric("Win rate (decided)", pct(o["win_rate"]) if pd.notna(o["win_rate"]) else "n/a")
    c3.metric("Net profit (1u flat)", f"{o['profit']:+.2f}u")
    c4.metric("ROI", pct(o["roi"]) if pd.notna(o["roi"]) else "n/a")
    clv_val = f"{o['avg_clv_pct']:+.2f}%" if pd.notna(o.get("avg_clv_pct")) else "n/a"
    c5.metric("Avg CLV", clv_val,
              help=("Closing Line Value — how your price compares to the closing line. "
                    "The best long-run predictor of profitability. "
                    f"{o['n_with_clv']} of {o['bets_total']} bets have a closing price logged."))

    n_settled = o["wins"] + o["losses"] + o["pushes"]
    if n_settled < 30:
        st.warning(
            f"⚠️ Only {n_settled} settled bet(s). Far too small to mean anything — a positive ROI "
            "here is noise, not proven edge. A real read needs ~50–100+ bets. This is the honest "
            "point of the log: let the sample grow before trusting it."
        )

    show = bet_df[[
        "date", "matchup", "market", "pick", "odds", "model_prob",
        "implied_prob", "edge_pct", "stake", "result", "profit", "closing_odds", "clv_pct", "notes",
    ]].copy()
    show["model_prob"] = (show["model_prob"] * 100).round(1)
    show["implied_prob"] = (show["implied_prob"] * 100).round(1)
    show["edge_pct"] = show["edge_pct"].round(1)
    show["clv_pct"] = show["clv_pct"].round(2)
    show["result"] = show["result"].fillna("pending")
    show = show.rename(columns={
        "model_prob": "model %", "implied_prob": "implied %", "edge_pct": "edge pts",
        "profit": "P/L", "closing_odds": "close", "clv_pct": "CLV %",
    })
    st.dataframe(show, width="stretch", hide_index=True)
    if pd.notna(o.get("avg_clv_pct")):
        st.caption(
            f"CLV: you beat the closing line on **{pct(o['beat_close_rate'])}** of priced bets, "
            f"avg **{o['avg_clv_pct']:+.2f}%**. Positive CLV is the earliest real sign the "
            "approach is beating the market — it shows up long before W/L does."
        )
    else:
        st.caption(
            "CLV not tracked yet — fill the `closing_odds` column (same side, price at first pitch) "
            "to measure whether you're beating the closing line. It's the best long-run signal here."
        )

    st.markdown("**By market**")
    rows = []
    for market, m in sorted(s["by_market"].items()):
        rows.append({
            "market": market,
            "W-L-P": f"{m['wins']}-{m['losses']}-{m['pushes']}",
            "pending": m["pending"],
            "win rate": pct(m["win_rate"]) if pd.notna(m["win_rate"]) else "n/a",
            "profit": f"${m['profit']:,.0f}",
            "ROI": pct(m["roi"]) if pd.notna(m["roi"]) else "n/a",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "Convention: `implied %` is the raw break-even from the odds (vig included); "
        "`edge pts` = model % − implied %. ROI is net profit over money actually at risk "
        "(pushes and pending bets excluded from turnover). See the README's Bet tracker section."
    )


# ---------------------------------------------------------------------------
# View: About & limitations
# ---------------------------------------------------------------------------
def render_about():
    st.subheader("How this works")
    st.markdown(
        """
**Three components:**
1. **Elo rating system** — every team starts at 1500 and is updated after each game based on
   the result, a home-field-advantage adjustment, and a margin-of-victory multiplier (blowouts
   move ratings more, but the effect is dampened when a big favorite wins big and amplified when
   an underdog blows out a favorite). Ratings regress partway toward the mean at the start of
   each season.
2. **Team offense/defense/park ratings** feed expected runs (mu_home, mu_away) for each side of
   a matchup — offense/defense rate relative to league average (recent-form blended with
   full-sample, shrunk for small samples), a park factor for the home stadium, and (for
   live/upcoming games only) the actual probable starting pitcher's season ERA and a FIP-style
   component, blended with the team's overall pitching rate.
3. **Monte Carlo simulation is the primary prediction method.** Each game is simulated
   (10,000 times by default, configurable in the sidebar): both teams' runs are sampled from a
   negative binomial distribution built from mu_home/mu_away and an *empirically estimated*
   overdispersion factor (measured directly from real historical team-game data — real MLB
   run totals turn out to be substantially more spread out than a naive Poisson would predict,
   about 2x the variance in our training data; see the Team ratings view). Moneyline win
   probability, the total-runs distribution, and run-line cover probability are then read off
   directly as empirical frequencies from the simulated outcomes — not computed from a formula.
   The closed-form Poisson/Skellam math (src/models/run_model.py) still runs alongside every
   simulation as a fast, independent cross-check, shown next to the Monte Carlo numbers
   throughout the dashboard; it assumes pure Poisson (no overdispersion), so it's *expected* to
   disagree with Monte Carlo somewhat on spread-sensitive numbers (like total-runs interval
   width) while roughly agreeing on means.

Elo's win probability and the Monte Carlo engine's win probability are averaged (50/50 by
default, `config.ELO_BLEND_WEIGHT`) into the final moneyline number shown on the Scoreboard.
        """
    )

    st.subheader("Data sources")
    st.markdown(
        f"""
- **Historical training data**: Baseball-Reference game logs via `pybaseball`, seasons
  {config.HISTORICAL_SEASONS + [config.CURRENT_SEASON]}.
- **Today's schedule & probable pitchers**: the free public MLB Stats API (statsapi.mlb.com),
  no API key required.
- **Live odds**: this project doesn't have its own live odds API connection. Real multi-sportsbook
  odds snapshots (moneyline/spread/total, ~28 books per game) are instead handed to it
  periodically as JSON files dropped into `odds/snapshots/` — see that folder's example files and
  `src/odds/odds_adapter.py`'s module docstring for the exact format. A simpler single-book
  manual format (`odds/odds_example.json` → `odds/odds.json`) also still works for quick testing.
  To wire in a real self-serve API later (e.g. The Odds API), implement
  `odds_adapter.fetch_live_odds()` — the docstring spells out the shape to return.
        """
    )

    st.subheader("Limitations — read before betting anything on this")
    st.markdown(
        """
- **Run-line "accuracy" is mostly base rate, not skill**: MLB underdogs cover +1.5 in roughly
  two-thirds of games just because the sport's own math says so (you only fail to cover if you lose
  by 2+ runs). Our backtested run-line pick accuracy beats a naive "always take the underdog"
  baseline by only a couple of points — see the Model performance view. The run-line probabilities
  are still useful for comparing against real sportsbook prices (where the underdog side is priced
  accordingly), but don't read a headline accuracy number here as proof of skill.
- **Backtest scope**: the Model performance view backtests the *team-level* Elo + Monte Carlo engine
  only, and uses fewer simulations per game than the live default (2,000 vs 10,000) purely for
  runtime across thousands of games.
- **The starting-pitcher adjustment was isolated and tested separately**
  (`src/backtest/pitcher_backtest.py`, real historical probable-pitcher data, genuine as-of-date
  stats, no lookahead, bootstrapped significance testing) — and **does not demonstrate a
  measurable benefit** over the team-level-only baseline. A real calibration bug was found and
  fixed along the way (the FIP blend used the wrong additive constant, which had been making the
  adjustment *significantly worse* for total runs specifically); after the fix that harm
  disappears, but no significant improvement takes its place anywhere, and one season/market slice
  (2024 moneyline) still shows significant harm. It's still used for live predictions here since
  it's a reasonable model on its own terms, but don't treat live picks as "sharper" because of it
  — see the README's Backtest section for the full numbers.
- **No historical odds data**: this build has no historical sportsbook closing lines, so the
  total (over/under) market couldn't be backtested against real historical lines — only against
  the model's own predicted mean (MAE/RMSE) and its own predictive interval, evaluated from each
  game's actual simulated distribution. Moneyline and run-line backtests don't have this problem
  since they only need the actual game outcome.
- **Odds snapshot matching has no date field**: the real snapshot format we receive doesn't
  include a game date, so odds are matched to predictions by team matchup only. Two games between
  the same two teams on different dates in a short series can't be told apart — fine for the
  common case of same-day snapshots, but worth knowing.
- **Small, recent sample**: training data covers only a few seasons. Rule changes, roster
  turnover, and rare events (extreme weather, injuries) are not modeled.
- **Whole-game simulation, not inning-by-inning**: Monte Carlo samples each team's *full-game*
  run total from a distribution, rather than simulating inning by inning — Baseball-Reference
  doesn't give us inning-by-inning box scores without a much heavier data pull, and the
  scoring-rate inputs (mu_home/mu_away) are themselves game-level rates, so inning-level
  simulation wouldn't add real information here.
- **Overdispersion is a single global estimate**, not fit per team — some teams' game-to-game
  variance may genuinely differ (e.g. a volatile offense vs. a consistent one), which this
  doesn't capture.
- **Extra innings**: modeled as a simple fixed-probability tiebreak (52% home team) in both the
  Monte Carlo and closed-form engines, not simulated inning by inning.
- **Elo constants** (home-field advantage, K-factor, margin-of-victory scaling) are reasonable,
  commonly-used starting values, not fitted by optimizing against a validation set.
- **Park factors** are estimated from a limited number of home games per park in the training
  window and aren't split by batter/pitcher handedness or weather.
- **This is not licensed betting or financial advice.** It's a statistical model built for
  analysis; treat any "edge" it finds as a hypothesis to sanity-check, not a guarantee.
        """
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
_VIEW = st.session_state.view
if _VIEW == "game":
    render_game_detail()
elif _VIEW == "bets":
    render_bets()
elif _VIEW == "ratings":
    render_ratings()
elif _VIEW == "performance":
    render_performance()
elif _VIEW == "about":
    render_about()
else:
    render_scoreboard()
