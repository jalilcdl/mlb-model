"""
Team offense/defense/park ratings, and the closed-form Poisson/Skellam math
used as a fast cross-check against the Monte Carlo engine (src/models/monte_carlo.py),
which is the primary prediction method -- see that module's docstring.

Independent premise: each team's runs in a game are Poisson-distributed with
a mean (mu) driven by that team's offense, the opponent's defense/pitching,
and the park. This is a standard, well-established simplification in
sabermetrics (real MLB run distributions are slightly overdispersed vs a
true Poisson -- TeamRunRatings estimates that overdispersion empirically and
hands it to the Monte Carlo engine, which can sample from a matched negative
binomial instead; this module's closed-form math stays pure-Poisson, which is
exactly why it's useful as an independent cross-check rather than the primary
number).

Given mu_home and mu_away for a specific game:
  - home_runs ~ Poisson(mu_home), away_runs ~ Poisson(mu_away), independent
  - run differential (home - away) ~ Skellam(mu_home, mu_away)
  - total runs (home + away) ~ Poisson(mu_home + mu_away)

...which gives closed-form probabilities for the moneyline, the total
(over/under), and the run line (MLB's fixed -1.5/+1.5 spread), without
needing simulation.
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import skellam

from src import config

PARK_FACTOR_CLIP = (0.85, 1.15)
MIN_GAMES_FOR_DISPERSION = 20  # per-team minimum sample before trusting its variance estimate


def _shrink(team_rate, league_rate, n_games, min_games):
    """Empirical-Bayes-style shrinkage toward the league average for small samples."""
    weight = n_games / (n_games + min_games)
    return weight * team_rate + (1 - weight) * league_rate


class TeamRunRatings:
    """Offense/defense/park ratings for every team, fit from a game log.

    All ratings are expressed as ratios to the league average (1.0 = exactly
    average). Fit once from data/processed/games.csv; cheap enough to refit
    on every dashboard load.
    """

    def __init__(self):
        self.league_avg_runs = None       # avg runs scored by one team in one game
        self.league_avg_total_runs = None  # avg combined runs (both teams) in one game
        self.off_rating = {}       # raw, park-contaminated -- used by predict_mus
        self.def_rating = {}       # raw, park-contaminated -- used by predict_mus
        self.off_rating_pa = {}    # park-adjusted -- used for descriptive grades only
        self.def_rating_pa = {}    # park-adjusted -- used for descriptive grades only
        self.park_factor = {}
        self.league_avg_era_proxy = None
        self.league_avg_fip_component = None
        self.games_played = {}
        self.overdispersion = 1.0  # index of dispersion (variance/mean) for runs scored; 1.0 = pure Poisson

    def fit(self, games, as_of_date=None):
        games = games[games["home_score"] != games["away_score"]].copy()
        if as_of_date is not None:
            games = games[games["date"] < pd.Timestamp(as_of_date)]
        games = games.sort_values("date").reset_index(drop=True)
        if games.empty:
            raise ValueError("No games available to fit TeamRunRatings")

        self.league_avg_total_runs = float((games["home_score"] + games["away_score"]).mean())
        self.league_avg_runs = self.league_avg_total_runs / 2.0
        # Runs are ~92% earned historically league-wide; used only to convert our
        # runs-based league baseline into an ERA-scale number for pitcher normalization.
        self.league_avg_era_proxy = self.league_avg_runs * 0.92 * 9.0 / 9.0

        recent_n = config.RUN_MODEL_RECENT_GAMES
        recent_w = config.RUN_MODEL_RECENT_WEIGHT
        min_games = config.RUN_MODEL_MIN_GAMES

        # Park factor: scoring environment at each team's home park vs league
        # average. Computed BEFORE the team loop because the park-adjusted
        # ratings below need it per game.
        for team, grp in games.groupby("home_team"):
            n = len(grp)
            park_total = (grp["home_score"] + grp["away_score"]).mean()
            raw_factor = park_total / self.league_avg_total_runs
            shrunk_factor = _shrink(raw_factor, 1.0, n, min_games)
            lo, hi = PARK_FACTOR_CLIP
            self.park_factor[team] = float(np.clip(shrunk_factor, lo, hi))

        # Build a long team-game table: one row per team per game they played,
        # with runs_scored / runs_allowed regardless of home/away. `park` is the
        # venue that game was played in (always the home team's park), which is
        # what lets us neutralize the run environment per game rather than
        # assuming every team's road schedule averages out to league-neutral.
        home_rows = games[["date", "home_team", "home_score", "away_score"]].rename(
            columns={"home_team": "team", "home_score": "runs_scored", "away_score": "runs_allowed"}
        )
        home_rows["park"] = games["home_team"].values
        away_rows = games[["date", "away_team", "away_score", "home_score"]].rename(
            columns={"away_team": "team", "away_score": "runs_scored", "home_score": "runs_allowed"}
        )
        away_rows["park"] = games["home_team"].values
        team_games = pd.concat([home_rows, away_rows], ignore_index=True).sort_values("date")

        # Park-neutral runs: divide each game's runs by that venue's factor, so a
        # run at Coors counts for less than a run at T-Mobile. Used ONLY for the
        # descriptive offense/pitching grades, NOT for predict_mus -- the
        # prediction path applies the park factor separately at projection time,
        # and changing its inputs would invalidate the backtested moneyline
        # numbers. See TeamRunRatings.off_rating_pa / def_rating_pa.
        pf = team_games["park"].map(self.park_factor).fillna(1.0).replace(0, 1.0)
        team_games["runs_scored_pa"] = team_games["runs_scored"] / pf
        team_games["runs_allowed_pa"] = team_games["runs_allowed"] / pf

        for team, grp in team_games.groupby("team"):
            grp = grp.sort_values("date")
            n = len(grp)
            recent = grp.tail(recent_n)

            def _blend(col):
                blended = recent_w * recent[col].mean() + (1 - recent_w) * grp[col].mean()
                return _shrink(blended, self.league_avg_runs, n, min_games) / self.league_avg_runs

            # Raw (park-contaminated) -- unchanged, still what predict_mus uses.
            self.off_rating[team] = _blend("runs_scored")
            self.def_rating[team] = _blend("runs_allowed")
            # Park-adjusted -- what the grades use.
            self.off_rating_pa[team] = _blend("runs_scored_pa")
            self.def_rating_pa[team] = _blend("runs_allowed_pa")
            self.games_played[team] = n

        # Empirical overdispersion (index of dispersion = variance/mean) for the
        # Monte Carlo engine's negative-binomial sampling. Real MLB team-game run
        # totals are typically a bit more spread out than a pure Poisson would
        # predict; this estimates how much directly from the training data rather
        # than assuming a textbook value. Computed per team (weighted by games
        # played) then averaged, since a single team's sample is noisy on its own.
        per_team = team_games.groupby("team")["runs_scored"].agg(["mean", "var", "count"])
        per_team = per_team[per_team["count"] >= MIN_GAMES_FOR_DISPERSION]
        per_team = per_team[per_team["mean"] > 0]
        if not per_team.empty:
            ratios = per_team["var"] / per_team["mean"]
            self.overdispersion = float(np.average(ratios, weights=per_team["count"]))
        self.overdispersion = max(self.overdispersion, 1.0)  # never model underdispersion

        # League-average FIP-style component (unscaled), used to normalize pitcher quality.
        # Approximate league total from team-average rates weighted by innings isn't
        # available without play-by-play, so we fall back to a fixed, well-documented
        # constant for the *component* scale and normalize pitchers against their own
        # season rather than needing a perfectly matched league figure. See run_model
        # pitcher_factor() docstring.
        self.league_avg_fip_component = config.LEAGUE_AVG_FIP_CONST - 3.10  # ~0, centers the component scale

        return self

    def _rating_or_default(self, d, team):
        return d.get(team, 1.0)

    def to_frame(self):
        teams = sorted(self.off_rating.keys())
        return pd.DataFrame(
            {
                "team": teams,
                "offense_rating": [self.off_rating[t] for t in teams],
                "defense_rating": [self.def_rating[t] for t in teams],
                "park_factor": [self.park_factor.get(t, 1.0) for t in teams],
                "games_played": [self.games_played.get(t, 0) for t in teams],
            }
        ).sort_values("offense_rating", ascending=False).reset_index(drop=True)

    def pitcher_factor(self, pitcher_stats, league_avg_era):
        """Blend ERA and a FIP-style component into a single run-prevention ratio,
        relative to league average (ratio < 1 => better than average pitcher).
        Returns None if the pitcher doesn't have enough innings to trust.
        """
        if not pitcher_stats or pitcher_stats["innings_pitched"] < config.PITCHER_MIN_IP_FOR_ADJUSTMENT:
            return None
        from src.data.statsapi_client import fip_component

        era_ratio = pitcher_stats["era"] / league_avg_era if league_avg_era else 1.0
        fip_comp = fip_component(pitcher_stats)
        # fip_comp is unscaled -- add the real FIP constant (not league_avg_era)
        # so it lands on the same run scale as ERA. This constant exists
        # specifically because fip_component's own league average isn't zero
        # (empirically ~1.0-1.1 in our data); using anything else as the
        # additive constant reintroduces exactly that offset as bias. Using
        # league_avg_era here instead of LEAGUE_AVG_FIP_CONST was a bug found
        # via src/backtest/pitcher_backtest.py: it inflated the average
        # pitcher factor to ~1.08 (should center near 1.0) and made the
        # adjustment measurably hurt total-runs predictions.
        fip_scaled = fip_comp + config.LEAGUE_AVG_FIP_CONST
        fip_ratio = fip_scaled / league_avg_era if league_avg_era else 1.0
        blended = 0.5 * era_ratio + 0.5 * fip_ratio
        return float(np.clip(blended, 0.5, 1.8))

    def predict_mus(self, home_team, away_team, home_pitcher_factor=None, away_pitcher_factor=None):
        off_home = self._rating_or_default(self.off_rating, home_team)
        off_away = self._rating_or_default(self.off_rating, away_team)
        def_home = self._rating_or_default(self.def_rating, home_team)
        def_away = self._rating_or_default(self.def_rating, away_team)
        park = self.park_factor.get(home_team, 1.0)

        starter_share = config.STARTER_INNINGS_SHARE
        eff_def_home = (
            starter_share * home_pitcher_factor + (1 - starter_share) * def_home
            if home_pitcher_factor is not None
            else def_home
        )
        eff_def_away = (
            starter_share * away_pitcher_factor + (1 - starter_share) * def_away
            if away_pitcher_factor is not None
            else def_away
        )

        mu_home = self.league_avg_runs * off_home * eff_def_away * park
        mu_away = self.league_avg_runs * off_away * eff_def_home * park
        return max(mu_home, 0.3), max(mu_away, 0.3)


# NOTE: a closed-form Poisson total_probabilities() used to live here and was
# deleted deliberately. It was dead code (never called) AND it carried the push
# bug -- it computed under as 1 - over, which silently awards pushes to the
# under and inflates every whole-number under by ~8-9 points. Totals
# probabilities come from the simulation itself, empirically:
#     src/models/monte_carlo.py :: total_outcome_probs(home_runs, away_runs, line)
# which counts simulated outcomes (> / == / < the line) and returns
# (P(over), P(push), P(under)). Do not reintroduce a closed-form totals
# approximation -- the empirical version is both more accurate (it inherits the
# negative-binomial overdispersion) and push-correct.


def cover_probability(mu_a, mu_b, line):
    """Closed-form P(team_a covers `line` against team_b) via the Skellam
    distribution. `line` is signed from team_a's perspective: -1.5 means team_a
    (as favorite) must win by 2+ to cover; +1.5 means team_a covers unless it
    loses by 2+. General on purpose -- real sportsbook lines aren't always
    exactly +/-1.5, and this is also how src/odds/odds_adapter.py checks the
    model against whatever line a given book actually posted."""
    threshold = math.floor(-line)
    return float(1 - skellam.cdf(threshold, mu_a, mu_b))


def game_probabilities(mu_home, mu_away, run_line=None):
    """Given expected runs for each side, derive moneyline and run-line (spread)
    probabilities in closed form from the Poisson/Skellam distributions.

    Does NOT compute an over/under pick -- MLB total lines vary per game, and
    totals probabilities are read empirically off the simulation instead (see
    monte_carlo.total_outcome_probs, wired in via src/odds/odds_adapter.py).
    This function always reports expected_total (the model's point prediction
    for total runs) so it's useful with or without a market line.
    """
    run_line = run_line if run_line is not None else config.RUN_LINE

    tie_prob = skellam.pmf(0, mu_home, mu_away)
    home_win_reg = 1 - skellam.cdf(0, mu_home, mu_away)
    away_win_reg = skellam.cdf(-1, mu_home, mu_away)
    home_win_prob = home_win_reg + config.EXTRA_INNING_HOME_WIN_PROB * tie_prob
    away_win_prob = away_win_reg + (1 - config.EXTRA_INNING_HOME_WIN_PROB) * tie_prob

    mu_total = mu_home + mu_away

    home_covers_prob = cover_probability(mu_home, mu_away, -run_line)
    away_covers_prob = cover_probability(mu_away, mu_home, run_line)

    return {
        "mu_home": mu_home,
        "mu_away": mu_away,
        "expected_total": mu_total,
        "home_win_prob": float(home_win_prob),
        "away_win_prob": float(away_win_prob),
        "home_covers_prob": float(home_covers_prob),
        "away_covers_prob": float(away_covers_prob),
        "run_line": run_line,
    }
