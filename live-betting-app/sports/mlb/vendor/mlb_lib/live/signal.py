"""
Live signal layer (PROTOTYPE): model win-prob vs the live market's de-vigged price.

The chain, per game:
  1. Live two-way moneyline (home/away american odds) from a LiveOddsProvider.
  2. De-vig it: raw implied probs include the book's margin, so we strip it
     proportionally (the same remove_vig_two_way the pregame charts use) to get a
     fair market probability that sums to 1 across the two sides.
  3. Compare to our in-game model win-probability for the same side.
  4. Flag when |model - market| exceeds a threshold, and label which side the
     model likes (the +EV side is the one the model rates higher than the market).

This is a SIGNAL detector, not a bet recommender: it says "the model and the live
market disagree by X here," which is the thing we're trying to establish exists.
It does not size stakes, model in-play juice/limits, or account for the latency
between seeing a price and being able to take it -- all real concerns before any
of this could be acted on.
"""
from __future__ import annotations

from dataclasses import dataclass

from mlb_lib import config
from mlb_lib.odds.odds_adapter import american_to_prob, remove_vig_two_way
from mlb_lib.live.live_odds import LiveMoneyline
from mlb_lib.live.win_expectancy import GameState, win_probability

# Default gap (in win-probability points) above which a disagreement is flagged.
DEFAULT_THRESHOLD = config.LIVE_SIGNAL_THRESHOLD

# De-vig method:
#   'proportional'  -- compute our own de-vig from the two-way american prices
#                      (proportional scaling to sum to 1). Works on any tier.
#   'sharp_novig'   -- use the provider's supplied NO-VIG probability (SharpAPI's
#                      Pinnacle-referenced no-vig) when the line carries one, else
#                      fall back to proportional. On SharpAPI's FREE tier the
#                      no-vig field is absent (it's a Pro-tier feature), so this
#                      transparently behaves like 'proportional' until upgraded.
DEFAULT_DEVIG_METHOD = getattr(config, "LIVE_DEVIG_METHOD", "proportional")


@dataclass
class Signal:
    game_id: str
    home_team: str
    away_team: str
    model_home_wp: float      # our in-game P(home win)
    market_home_wp: float     # de-vigged market P(home win)
    edge_home: float          # model_home_wp - market_home_wp (signed, home POV)
    vig: float                # book margin stripped out (overround - 1)
    flagged: bool
    pick_side: str | None     # 'home' / 'away' / None -- side the model favors vs market
    pick_team: str | None
    edge: float               # magnitude of the flagged edge (abs)
    source: str
    devig_method: str = "proportional"  # method actually used for market_home_wp

    def describe(self) -> str:
        tag = "FLAG" if self.flagged else "  - "
        line = (f"[{tag}] {self.away_team}@{self.home_team} "
                f"model(home)={self.model_home_wp:.3f} "
                f"market(home)={self.market_home_wp:.3f} "
                f"edge={self.edge_home:+.3f}")
        if self.flagged:
            line += f"  -> {self.pick_team} +{self.edge:.3f} (vig {self.vig*100:.1f}%)"
        return line


def market_home_win_prob(ml: LiveMoneyline, method=DEFAULT_DEVIG_METHOD):
    """De-vigged P(home win), the book's margin, and the method actually used.

    'sharp_novig' uses the provider's no-vig probabilities when the line carries
    them (normalized defensively), else falls back to 'proportional'. 'proportional'
    strips the vig from the two american prices ourselves."""
    raw_home = american_to_prob(ml.home_odds)
    raw_away = american_to_prob(ml.away_odds)
    vig = (raw_home + raw_away) - 1.0

    if method == "sharp_novig" and ml.home_fair is not None and ml.away_fair is not None:
        total = ml.home_fair + ml.away_fair
        if total > 0:
            return ml.home_fair / total, vig, "sharp_novig"

    fair_home, _ = remove_vig_two_way(raw_home, raw_away)
    return fair_home, vig, "proportional"


def evaluate(state: GameState, ml: LiveMoneyline, *, home_team=None, away_team=None,
             ratings=None, n_sims=20000, threshold=DEFAULT_THRESHOLD, seed=None,
             devig_method=DEFAULT_DEVIG_METHOD) -> Signal:
    """Score one game: run the in-game model, de-vig the live line, compare, flag.
    Team codes for the model prior default to the ones on the live line.
    `devig_method` selects our own de-vig ('proportional') or the provider's no-vig
    ('sharp_novig', auto-falling back on the free tier)."""
    home_team = home_team or ml.home_team
    away_team = away_team or ml.away_team

    model_home = win_probability(state, home_team=home_team, away_team=away_team,
                                 ratings=ratings, n_sims=n_sims, seed=seed)
    market_home, vig, method_used = market_home_win_prob(ml, method=devig_method)
    edge_home = model_home - market_home

    flagged = abs(edge_home) >= threshold
    pick_side = pick_team = None
    if flagged:
        if edge_home > 0:            # model rates home higher than the market -> back home
            pick_side, pick_team = "home", home_team
        else:                        # model rates away higher -> back away
            pick_side, pick_team = "away", away_team

    return Signal(
        game_id=ml.game_id, home_team=home_team, away_team=away_team,
        model_home_wp=round(model_home, 4), market_home_wp=round(market_home, 4),
        edge_home=round(edge_home, 4), vig=round(vig, 4),
        flagged=flagged, pick_side=pick_side, pick_team=pick_team,
        edge=round(abs(edge_home), 4), source=ml.source, devig_method=method_used)
