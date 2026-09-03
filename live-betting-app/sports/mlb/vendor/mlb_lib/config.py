"""Trimmed config for the vendored MLB live-signal path only.

Vendored from mlb-model/src/config.py -- kept to just the constants
src.live.* actually reads (see the dependency trace that produced this vendor
copy). The full model config (Elo/Poisson/backtest tuning, other data files)
lives in the source repo and is irrelevant here.
"""
from pathlib import Path
import os as _os

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data_files"
PROCESSED_DIR = DATA_DIR  # vendored CSVs live directly under data_files/

TEAM_RATINGS_FILE = PROCESSED_DIR / "team_run_ratings_latest.csv"
ODDS_DIR = ROOT_DIR / "_unused_odds_dir"  # odds_adapter.py references this in code paths never called here
RUN_LINE = 1.5  # monte_carlo.py default; never exercised by the live win-prob path but kept for safety

# ---- SharpAPI live odds ----
SHARPAPI_API_KEY = _os.environ.get("SHARPAPI_API_KEY") or None
SHARPAPI_BASE_URL = "https://api.sharpapi.io/api/v1"
LIVE_SIGNAL_THRESHOLD = 0.04
LIVE_DEVIG_METHOD = "proportional"

# ---- Monte Carlo simulation engine (used by win_expectancy's in-game sim) ----
MC_DEFAULT_SIMS = 10000
MC_MIN_SIMS = 1000
MC_MAX_SIMS = 100000
MC_DISTRIBUTION = "negative_binomial"
EXTRA_INNING_HOME_WIN_PROB = 0.52
