"""In-game win probability for college football.

PROTOTYPE. Not validated as a betting signal, and deliberately simple.

Approach: the "Brownian motion" model of a football game (Stern 1994 for the
NFL; the same shape underlies most public in-game WP models, ESPN's included).
Treat the remaining margin as a random walk. Then

    P(home wins) = Phi( mu / sigma )

    mu    = current margin
            + pregame margin rate * fraction of game remaining
            + expected points from the possession in progress
    sigma = full-game margin sd * sqrt(fraction remaining)

Why this shape for THIS repo rather than a fitted play-by-play model:

- It consumes the pregame prior we already have. `mu` takes v5's predicted
  margin (the anchor the deployed Monte Carlo uses) and decays it as the clock
  runs, so the in-game model inherits the team-strength work already done
  instead of re-deriving it.
- Its one dispersion parameter is a number this project MEASURED rather than
  guessed. The accuracy round found the realised margin sd is 16.14 with the
  simulator's own sd ratio at 1.00, so `SIGMA_FULL_GAME` is that figure, not a
  literature value.
- Fitting a real down/distance model would need play-by-play. CFBD's `/plays`
  is ~2,600 calls against a 1,000/month free tier, which is why the earlier
  play-level simulation was dropped. ESPN's summary feed gives plays for free,
  so a fitted upgrade is possible later; it is not needed to answer "does a
  gap between model and market exist at all".

The possession term is an approximation of the standard expected-points curve
and is the crudest piece here -- see EP_AT_OWN_GOAL / EP_PER_YARD.

Only margin (moneyline/spread) is modeled here -- there is deliberately no
total-points function. Margin and total are different linear combinations of
each team's own score (home-away vs home+away); knowing the margin
distribution says nothing about the total distribution, and this file has
never derived one. A totals model would need its own empirically-measured
variance (the way SIGMA_FULL_GAME was measured for margin, not guessed) --
real modeling work, not an extension of what's here. Bolting a number on
would be worse than not having one.

Everything is observe-only. Nothing in this module places or recommends a bet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Realised full-game margin sd measured on 5,870 walk-forward games in the
# accuracy round (data/processed/sim_accuracy_round.json). Using the project's
# own number keeps the in-game model consistent with the deployed simulator.
SIGMA_FULL_GAME = 16.14

REGULATION_SECONDS = 60 * 60  # four 15-minute quarters

# Expected points for a 1st-and-10, as a straight line in yards-to-endzone.
# Real EP curves are gently convex; over the 1-99 range a line is within ~0.4
# points of published curves, which is well inside the noise this prototype is
# trying to detect. Anchors: ~6.0 at the goal line, ~-0.4 backed up at own 1.
EP_AT_OWN_GOAL = 6.0
EP_PER_YARD = 0.065
# Each down past 1st costs roughly this much expected value, and long-yardage
# situations cost more. Both are round numbers, not fitted.
EP_PER_DOWN = 0.45
EP_PER_YARD_TO_GO = 0.06
EP_BASE_DISTANCE = 10

# Below this much game left, sigma stops shrinking. Without a floor the model
# becomes a step function in the last seconds and reports 0.000/1.000, which is
# both false (a single play can still flip a one-score game) and useless as a
# comparison against a market that is still quoting a price.
MIN_SIGMA = 1.6


@dataclass(frozen=True)
class GameState:
    """A pregame-or-in-progress snapshot, from the home team's perspective."""
    home_score: int
    away_score: int
    period: int
    clock_seconds: int          # seconds left in the current period
    down: int | None = None
    distance: int | None = None
    yards_to_endzone: int | None = None   # for the team with the ball
    home_has_ball: bool | None = None
    state: str = "in"           # "pre" | "in" | "post"

    @property
    def margin(self) -> int:
        return self.home_score - self.away_score

    def seconds_remaining(self) -> int:
        """Regulation seconds left. Overtime reports 0 -- see win_probability."""
        if self.period >= 5:
            return 0
        completed_after = max(0, 4 - self.period)
        return int(self.clock_seconds + completed_after * 15 * 60)


def expected_points(down: int | None, distance: int | None,
                    yards_to_endzone: int | None) -> float:
    """Approximate expected points for the team currently in possession."""
    if yards_to_endzone is None:
        return 0.0
    ep = EP_AT_OWN_GOAL - EP_PER_YARD * float(yards_to_endzone)
    if down:
        ep -= EP_PER_DOWN * (down - 1)
    if distance is not None:
        ep -= EP_PER_YARD_TO_GO * max(0, distance - EP_BASE_DISTANCE)
    # A 4th-and-long deep in own territory is worth roughly a punt, not a
    # negative fortune; and no field position is worth more than a touchdown.
    return max(-1.0, min(6.9, ep))


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _margin_distribution(state: GameState, pregame_margin: float,
                         sigma_full: float) -> tuple[float, float] | None:
    """(mu, sigma) of the in-progress margin ~ Normal(mu, sigma^2) model, or
    None if the game's already decided (state.state == "post") -- callers
    handle that case directly off state.margin instead. Shared by
    win_probability/cover_probability so the two questions ("does home win"
    vs "does home cover") always come from the exact same distribution, just
    a different threshold -- not two models that could quietly drift apart.
    """
    if state.state == "post":
        return None

    remaining = state.seconds_remaining()
    frac = 1.0 if state.state == "pre" else remaining / REGULATION_SECONDS

    mu = state.margin + pregame_margin * frac
    if state.home_has_ball is not None:
        ep = expected_points(state.down, state.distance, state.yards_to_endzone)
        mu += ep if state.home_has_ball else -ep

    sigma = max(MIN_SIGMA, sigma_full * math.sqrt(max(frac, 0.0)))
    return mu, sigma


def win_probability(state: GameState, pregame_margin: float,
                    sigma_full: float = SIGMA_FULL_GAME) -> float:
    """P(home team wins), in [0, 1].

    `pregame_margin` is the model's predicted home margin BEFORE kickoff
    (positive = home favoured) -- v5's number in this repo.
    """
    if state.state == "post":
        if state.margin > 0:
            return 1.0
        if state.margin < 0:
            return 0.0
        return 0.5

    # Overtime: possession rules dominate and the Brownian model does not
    # describe them. Report the honest coin-flip-with-a-lead rather than
    # pretending to model it.
    if state.period >= 5:
        if state.margin > 0:
            return 0.85
        if state.margin < 0:
            return 0.15
        return 0.5

    mu, sigma = _margin_distribution(state, pregame_margin, sigma_full)
    return _phi(mu / sigma)


def cover_probability(state: GameState, pregame_margin: float, home_line: float,
                      sigma_full: float = SIGMA_FULL_GAME) -> float:
    """P(home team covers `home_line`, home's own signed line -- e.g. -7 for a
    7-point favorite). Built on the EXACT same margin ~ Normal(mu, sigma^2)
    distribution win_probability() uses (see _margin_distribution) -- just a
    different threshold (-home_line instead of 0), not a separate model.
    Home covers when its margin exceeds -home_line, so
        P(cover) = Phi((mu - (-home_line)) / sigma) = Phi((mu + home_line) / sigma).
    """
    if state.state == "post":
        if state.margin > -home_line:
            return 1.0
        if state.margin < -home_line:
            return 0.0
        return 0.5  # exact push

    if state.period >= 5:
        # Same coarse OT simplification win_probability() uses for the win
        # question -- applied to the cover threshold instead of zero, for
        # consistency, not new precision the model doesn't actually have.
        if state.margin > -home_line:
            return 0.85
        if state.margin < -home_line:
            return 0.15
        return 0.5

    mu, sigma = _margin_distribution(state, pregame_margin, sigma_full)
    return _phi((mu + home_line) / sigma)
