"""
In-game MLB win-probability model (PROTOTYPE).

APPROACH -- "good enough for a prototype", deliberately not overbuilt:
We simulate the REMAINDER of the game from the current base/out/inning/score
state and count how often the home team ends up ahead. Two ingredients:

  1. Base-out run expectancy (RE24). A standard, well-established table giving the
     mean runs scored from the current base/out state to the end of the half-
     inning. Used to seed runs for the CURRENT (partial) half-inning.

  2. Per-inning team run rates as the PRIOR. Full future innings each start empty
     with nobody out, so their run distribution is driven purely by team strength.
     We read this repo's team run ratings (offense/defense multipliers, the same
     numbers the pregame model trains) and turn them into runs-per-inning:
         rate = LEAGUE_RUNS_PER_INNING * offense(batter) * defense(opponent)
     This is exactly where "reuse the existing team-strength ratings as a prior"
     enters: an even game between two average teams reduces to a ~50/50 pregame
     coin flip, and a strong offense vs a weak staff tilts every future inning.

Runs per half-inning are drawn from a negative binomial calibrated so a fresh
(empty, 0-out) inning has mean ~0.48 and P(0 runs) ~0.73 -- the empirical MLB
shape. That over-dispersion (variance/mean ~2) is what keeps the model from being
over-confident that a lead is safe. The dispersion is tuned for the empty/0-out
case that dominates future innings; the single current partial inning it is
slightly off for is a documented prototype simplification.

WHAT THIS PROTOTYPE INTENTIONALLY OMITS (candidates if the signal proves out):
  - a real empirical win-expectancy table (Retrosheet/Statcast) as ground truth
  - the current batter/pitcher, platoon splits, bullpen state, park, weather
  - the extra-innings ghost-runner rule (resolved by a flat home-field constant)
  - leverage-aware pinch-hit / bullpen decisions
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

from mlb_lib import config

# League-average runs per team per inning (~4.3 R/G / 9). Anchors the RE table
# and the run-rate conversion so an average matchup is internally consistent.
LEAGUE_RUNS_PER_INNING = 0.48

# Negative-binomial "size" (number of successes) for half-inning run counts.
# Chosen so an empty/0-out inning (mean 0.48) has P(0)~0.72 and variance/mean~2,
# matching the empirical over-dispersion of MLB half-inning scoring.
HALF_INNING_NB_SIZE = 0.45

# Base-out run expectancy: mean runs from (bases, outs) to end of half-inning.
# bases = (on_first, on_second, on_third) as 0/1. Standard ~2010s league values.
RUN_EXPECTANCY = {
    (0, 0, 0): (0.48, 0.25, 0.10),
    (1, 0, 0): (0.85, 0.51, 0.22),
    (0, 1, 0): (1.10, 0.66, 0.32),
    (0, 0, 1): (1.35, 0.95, 0.38),
    (1, 1, 0): (1.44, 0.88, 0.43),
    (1, 0, 1): (1.75, 1.14, 0.48),
    (0, 1, 1): (1.96, 1.38, 0.56),
    (1, 1, 1): (2.29, 1.54, 0.75),
}


def parse_bases(bases) -> tuple:
    """Accept ('1','_','3'), '1_3', (True,False,True), or a set like {1,3} and
    return a canonical (first, second, third) 0/1 tuple."""
    if isinstance(bases, str):
        s = bases.replace(" ", "")
        return (int("1" in s), int("2" in s), int("3" in s))
    if isinstance(bases, (set, frozenset)):
        return (int(1 in bases), int(2 in bases), int(3 in bases))
    return tuple(int(bool(x)) for x in bases)


@dataclass
class GameState:
    """A snapshot of one live game. `half` is 'top' (away batting) or 'bottom'
    (home batting). `bases` is anything parse_bases accepts."""
    inning: int
    half: str
    outs: int
    away_score: int
    home_score: int
    bases: tuple = (0, 0, 0)

    def __post_init__(self):
        self.half = self.half.lower()
        if self.half not in ("top", "bottom"):
            raise ValueError(f"half must be 'top' or 'bottom', got {self.half!r}")
        if not 0 <= self.outs <= 2:
            raise ValueError(f"outs must be 0-2, got {self.outs}")
        self.bases = parse_bases(self.bases)


# --------------------------------------------------------------------------- #
# Team run rates (the prior)                                                   #
# --------------------------------------------------------------------------- #
def load_team_ratings(path=None) -> dict:
    """Return {TEAM: {'offense': float, 'defense': float}} from the run-ratings
    CSV the pregame model already produces. Empty dict if the file is missing."""
    path = path or config.TEAM_RATINGS_FILE
    out = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[row["team"].upper()] = {
                    "offense": float(row["offense_rating"]),
                    "defense": float(row["defense_rating"]),
                }
    except (FileNotFoundError, KeyError, ValueError):
        return out
    return out


def run_rates(home_team, away_team, ratings=None):
    """Expected runs/inning for (home, away). Uses each team's offense against the
    other's defense; falls back to league average for any team not in the ratings
    (so the prototype still runs on a matchup the ratings file doesn't cover)."""
    ratings = ratings if ratings is not None else load_team_ratings()
    h = ratings.get((home_team or "").upper(), {"offense": 1.0, "defense": 1.0})
    a = ratings.get((away_team or "").upper(), {"offense": 1.0, "defense": 1.0})
    home_rate = LEAGUE_RUNS_PER_INNING * h["offense"] * a["defense"]
    away_rate = LEAGUE_RUNS_PER_INNING * a["offense"] * h["defense"]
    return home_rate, away_rate


# --------------------------------------------------------------------------- #
# Simulation                                                                   #
# --------------------------------------------------------------------------- #
def _draw_runs(mean, n, rng):
    """Vectorized negative-binomial half-inning run counts with the given mean."""
    mean = max(float(mean), 1e-6)
    p = HALF_INNING_NB_SIZE / (HALF_INNING_NB_SIZE + mean)
    return rng.negative_binomial(HALF_INNING_NB_SIZE, p, size=n)


def _future_halves(inning, half):
    """(inning, half) half-innings to play AFTER the current one is finished,
    through the bottom of the 9th. Extras (a tie after that) are resolved by a
    flat home-field constant, so this list never needs to go past inning 9."""
    seq = [(inning, "bottom")] if half == "top" else []
    for inn in range(inning + 1, 10):
        seq += [(inn, "top"), (inn, "bottom")]
    return seq


def is_game_over(state: GameState):
    """Return the decided home-win probability (1.0 / 0.0) if the state is already
    final, else None. Only the unambiguous cases are caught -- a home lead once the
    home team no longer bats (during/after the bottom of the 9th or later)."""
    if state.inning >= 9 and state.half == "bottom" and state.home_score > state.away_score:
        return 1.0
    return None


# Safety cap on simulated extra innings (i.e. up to inning 9+MAX_EXTRA_INNINGS).
# Real MLB games essentially never approach this -- the longest in recorded
# history was 26 innings -- so this bound almost never binds; it exists only
# to guarantee the simulation loop terminates. Any simulation still tied after
# the cap falls back to the flat EXTRA_INNING_HOME_WIN_PROB coin, exactly the
# shortcut the rest of this function no longer needs.
MAX_EXTRA_INNINGS = 20


def simulate_remainder(state: GameState, home_team=None, away_team=None,
                       ratings=None, n_sims=20000, seed=None):
    """Simulate the rest of the game from `state`, returning (h, a): each
    array is one simulated FINAL score per trial (n_sims of them). This is
    the shared core win_probability/cover_probability/over_probability all
    build on -- margin (h-a) and total (h+a) are real per-trial simulated
    quantities, not a derived approximation.

    Extra innings are genuinely simulated (same run-rate machinery as
    regulation, repeated a half-inning at a time only for the subset of
    trials still tied), not resolved by a flat coin-flip -- that shortcut
    would silently understate total-runs for any trial that ties after 9,
    which matters for over/under even though it barely matters for margin
    (a modeled tie contributes margin 0, which doesn't cover either side of a
    real line anyway)."""
    rng = np.random.default_rng(seed)
    home_rate, away_rate = run_rates(home_team, away_team, ratings)

    h = np.full(n_sims, state.home_score, dtype=np.int64)
    a = np.full(n_sims, state.away_score, dtype=np.int64)

    # 1) Finish the current (partial) half-inning from its base/out state. Scale
    #    the RE mean by the batting team's strength relative to league average.
    re_mean = RUN_EXPECTANCY[state.bases][state.outs]
    if state.half == "top":
        a += _draw_runs(re_mean * (away_rate / LEAGUE_RUNS_PER_INNING), n_sims, rng)
    else:
        # Home batting; only add if the home team isn't already ahead in the 9th+
        # (they wouldn't keep batting -- covers a bottom-9th walk-off-pending state).
        runs = _draw_runs(re_mean * (home_rate / LEAGUE_RUNS_PER_INNING), n_sims, rng)
        if state.inning >= 9:
            runs = np.where(h > a, 0, runs)
        h += runs

    # 2) Play the remaining full half-innings through the bottom of the 9th.
    for inn, half in _future_halves(state.inning, state.half):
        if half == "top":
            a += _draw_runs(away_rate, n_sims, rng)
        else:
            runs = _draw_runs(home_rate, n_sims, rng)
            if inn >= 9:  # home skips batting when already ahead (walk-off logic)
                runs = np.where(h > a, 0, runs)
            h += runs

    # 3) Real extra innings for whatever trials are still tied -- only the
    #    tied subset is touched each round, so a trial decided in the 10th
    #    doesn't keep accumulating runs in later rounds.
    for _ in range(MAX_EXTRA_INNINGS):
        tied = h == a
        if not np.any(tied):
            break
        away_runs = _draw_runs(away_rate, n_sims, rng)
        a[tied] += away_runs[tied]
        home_leads = h > a  # after the top half just played
        home_runs = _draw_runs(home_rate, n_sims, rng)
        home_runs = np.where(home_leads, 0, home_runs)  # walk-off: skip if already ahead
        h[tied] += home_runs[tied]

    # 4) Whatever's still tied after the cap (should essentially never happen
    #    in practice -- see MAX_EXTRA_INNINGS) falls back to the flat constant,
    #    same number the old shortcut used for every tie, now used for none.
    still_tied = h == a
    if np.any(still_tied):
        n_left = int(still_tied.sum())
        home_wins_tiebreak = rng.random(n_left) < config.EXTRA_INNING_HOME_WIN_PROB
        # Encode the coin-flip as a nominal 1-0 margin so h>a / totals stay
        # well-defined for this vanishingly rare leftover, rather than a
        # permanent tie that would corrupt margin/total math downstream.
        h[still_tied] += home_wins_tiebreak.astype(np.int64)
        a[still_tied] += (~home_wins_tiebreak).astype(np.int64)

    return h, a


def win_probability(state: GameState, home_team=None, away_team=None,
                    ratings=None, n_sims=20000, seed=None):
    """P(home team wins) from the current state. Team codes are optional; without
    them (or for teams missing from the ratings) both sides use league-average
    run rates, i.e. a pure base-out-score-inning estimate with no team prior."""
    decided = is_game_over(state)
    if decided is not None:
        return decided
    h, a = simulate_remainder(state, home_team, away_team, ratings, n_sims, seed)
    return float((h > a).mean())


def cover_probability(state: GameState, home_line: float, home_team=None,
                      away_team=None, ratings=None, n_sims=20000, seed=None):
    """P(home team covers `home_line`), home team's OWN signed line (e.g. -1.5
    favorite, +1.5 underdog -- each side's odds row already carries its own
    correctly-signed line; pass that straight through, don't re-derive a sign).
    Home covers when its own margin exceeds -home_line."""
    decided = is_game_over(state)
    if decided is not None:
        margin = state.home_score - state.away_score
        return 1.0 if margin > -home_line else 0.0
    h, a = simulate_remainder(state, home_team, away_team, ratings, n_sims, seed)
    margin = h - a
    return float((margin > -home_line).mean())


def over_probability(state: GameState, total_line: float, home_team=None,
                     away_team=None, ratings=None, n_sims=20000, seed=None):
    """P(total runs > total_line)."""
    decided = is_game_over(state)
    if decided is not None:
        total = state.home_score + state.away_score
        return 1.0 if total > total_line else 0.0
    h, a = simulate_remainder(state, home_team, away_team, ratings, n_sims, seed)
    total = h + a
    return float((total > total_line).mean())
