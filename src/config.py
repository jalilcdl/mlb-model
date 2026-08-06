"""Central configuration for the MLB model."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
ODDS_DIR = ROOT_DIR / "odds"

GAMES_FILE = PROCESSED_DIR / "games.csv"
ELO_RATINGS_FILE = PROCESSED_DIR / "elo_ratings_latest.csv"
TEAM_RATINGS_FILE = PROCESSED_DIR / "team_run_ratings_latest.csv"
PREDICTIONS_FILE = PROCESSED_DIR / "predictions_latest.csv"
BACKTEST_RESULTS_FILE = PROCESSED_DIR / "backtest_results.csv"
BACKTEST_SUMMARY_FILE = PROCESSED_DIR / "backtest_summary.json"
HISTORICAL_STARTERS_FILE = PROCESSED_DIR / "historical_starters.csv"
PITCHER_GAME_LOGS_FILE = PROCESSED_DIR / "pitcher_game_logs.csv"
PITCHER_WATCH_FILE = PROCESSED_DIR / "pitcher_watch.json"  # last-seen probable pitcher per game_pk,
                                                             # for last-minute-change detection
BET_LOG_FILE = PROCESSED_DIR / "bet_log.csv"  # manually-maintained forward-test record of acted-on picks

# Real historical closing over/under lines for validating the totals model against
# the market (NOT available from pybaseball/statsapi -- must be supplied; see
# src/backtest/totals_backtest.py for the required CSV format and sources).
HISTORICAL_TOTALS_FILE = DATA_DIR / "raw" / "historical_totals.csv"
TOTALS_BACKTEST_SUMMARY_FILE = PROCESSED_DIR / "totals_backtest_summary.json"

# ---- Highlightly free-tier live odds (fallback/replacement for manual snapshots) ----
# Get a free "Basic" key at https://highlightly.net (100 req/day, no card).
# Provide it via the HIGHLIGHTLY_API_KEY environment variable (preferred), or
# paste it here as a fallback. Leave as None to disable the Highlightly source.
import os as _os
HIGHLIGHTLY_API_KEY = _os.environ.get("HIGHLIGHTLY_API_KEY") or None
HIGHLIGHTLY_BASE_URL = "https://baseball.highlightly.net"
HIGHLIGHTLY_LEAGUE = "MLB"
PITCHER_BACKTEST_RESULTS_FILE = PROCESSED_DIR / "pitcher_backtest_results.csv"
PITCHER_BACKTEST_SUMMARY_FILE = PROCESSED_DIR / "pitcher_backtest_summary.json"

# Seasons used to isolate the starting-pitcher adjustment (needs per-game
# historical starter data, which is more expensive to fetch than team-level
# results, so kept smaller than HISTORICAL_SEASONS by default).
PITCHER_BACKTEST_SEASONS = [2024, 2025]
PITCHER_MIN_IP_FOR_ADJUSTMENT = 15  # must match run_model.TeamRunRatings.pitcher_factor()

# Seasons pulled for training. Update HISTORICAL_SEASONS as years pass.
# CURRENT_SEASON is pulled separately (season-to-date) since it's incomplete.
HISTORICAL_SEASONS = [2023, 2024, 2025]
CURRENT_SEASON = 2026

# ---- Elo model ----
ELO_START = 1500.0
ELO_K = 4.0                # base K-factor per game (MLB: low, since 162 games/season)
ELO_HOME_ADVANTAGE = 24.0  # Elo points added to home team before win-prob calc
ELO_MOV_MULT_BASE = 0.62   # scales the margin-of-victory multiplier (ln-based, 538-style)
ELO_SEASON_REGRESSION = 0.25  # fraction each team's rating regresses toward 1500 at season start

# ---- Run scoring (Poisson) model ----
RUN_MODEL_RECENT_GAMES = 45      # rolling window size (team games) for "recent form"
RUN_MODEL_RECENT_WEIGHT = 0.45   # weight given to recent-form rate vs full-sample rate
RUN_MODEL_MIN_GAMES = 10         # minimum games before trusting a team rate over league avg
STARTER_INNINGS_SHARE = 0.56     # approx fraction of a game's defensive innings thrown by the starter
LEAGUE_AVG_FIP_CONST = 3.10      # standard FIP constant (approx, recalculated dynamically when possible)

# ---- Monte Carlo simulation engine (primary prediction method) ----
MC_DEFAULT_SIMS = 10000
MC_MIN_SIMS = 1000
MC_MAX_SIMS = 100000
MC_DISTRIBUTION = "negative_binomial"  # "negative_binomial" (uses empirically estimated
                                        # overdispersion, see TeamRunRatings.overdispersion)
                                        # or "poisson" (no extra variance)
EXTRA_INNING_HOME_WIN_PROB = 0.52  # mild home-field edge in a hypothetical extra-innings tiebreak
                                    # (shared by the closed-form and Monte Carlo engines)

# ---- Ensemble ----
ELO_BLEND_WEIGHT = 0.5   # weight on Elo win prob vs Monte Carlo win prob (rest = 1 - this)

# Totals ensemble: the over/under edge blends our model's projected total with the
# connector's independent model projection (get_game_projections mean). This is the
# WEIGHT ON THE CONNECTOR; our weight is (1 - this). 0.5 = equal weight -- the safe
# default (averaging two decent independent models tends to cut error even without
# knowing which is better), pending real paired data from the nightly compare log to
# tune it. Set to 0.0 to fall back to our-model-only totals.
TOTALS_CONNECTOR_WEIGHT = 0.5

# ---- Run line ----
RUN_LINE = 1.5  # standard MLB run line (favorite -1.5 / underdog +1.5)

# ---- Backtest ----
BACKTEST_HOLDOUT_SEASON = 2025  # season(s) evaluated out-of-sample; earlier seasons used for warm-up
BACKTEST_MC_SIMS = 2000  # fewer sims than MC_DEFAULT_SIMS purely for runtime across thousands of games
