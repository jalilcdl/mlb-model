"""Tunable knobs for the sizing engine and backtest. All defaults are
starting points Jalil can tune once there's enough backtested history to
justify changing them -- nothing here is a claim that these are "correct".
"""

KELLY_FRACTION = 0.25            # 25% Kelly -- full Kelly is too volatile for
                                  # a live, still-thin-sample model.
MAX_BET_PCT_OF_BANKROLL = 0.02   # Hard cap regardless of what Kelly says.
MIN_EDGE_TO_SIZE = 0.04          # Below this de-vigged edge, no stake at all --
                                  # a noise filter, separate from (and can be
                                  # stricter than) each sport's own logging
                                  # threshold (LIVE_SIGNAL_THRESHOLD / 0.04 MLB,
                                  # CFB_LIVE_SIGNAL_THRESHOLD / 0.05 CFB).
EDGE_SANITY_CEILING = 0.20       # Above this de-vigged edge, flag for manual
                                  # review instead of auto-sizing -- more
                                  # likely a stale/illiquid line than free
                                  # money.
DEFAULT_BANKROLL = 1000.0        # Placeholder. Plug in the real number when
                                  # actually using this for a live stake.

# Small-sample guard for the backtest report -- below this many resolved
# bets, treat any hit-rate/calibration number as not-yet-meaningful.
MIN_BETS_FOR_CONFIDENCE = 30

# How long a game+side stays "the same bet opportunity" before a persistent
# flag would count as a new one -- mirrors core/storage.py's
# RENOTIFY_COOLDOWN_SECONDS exactly, so the backtest's bet count matches what
# Jalil would actually have been notified about live.
RENOTIFY_COOLDOWN_SECONDS = 30 * 60
