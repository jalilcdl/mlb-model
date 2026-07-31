"""
Elo team-strength rating system for MLB, updated game by game.

Standard logistic Elo (as used for chess, and adapted by outlets like
FiveThirtyEight for team sports) with two MLB-specific adjustments:

  1. A fixed home-field advantage added to the home team's rating before
     computing win probability (typically ~55-56% of MLB home teams win,
     so a modest boost is warranted).
  2. A margin-of-victory multiplier so blowouts move ratings more than
     one-run games, but the multiplier is dampened when a big favorite
     wins big (already expected) and amplified when an underdog blows out
     a favorite (surprising). This is the same general idea FiveThirtyEight
     uses in its NFL/NBA Elo models, adapted here with our own constant
     rather than a claim of exact fidelity to any published formula.

K is intentionally small (default 4) because a 162-game MLB season has far
more games than the NFL's Elo models were tuned for -- a large K would let
a handful of games swing ratings too aggressively.
"""
import math

import pandas as pd

from src import config


class EloModel:
    def __init__(
        self,
        start=None,
        k=None,
        home_advantage=None,
        mov_mult_base=None,
        season_regression=None,
    ):
        self.start = start if start is not None else config.ELO_START
        self.k = k if k is not None else config.ELO_K
        self.home_advantage = home_advantage if home_advantage is not None else config.ELO_HOME_ADVANTAGE
        self.mov_mult_base = mov_mult_base if mov_mult_base is not None else config.ELO_MOV_MULT_BASE
        self.season_regression = (
            season_regression if season_regression is not None else config.ELO_SEASON_REGRESSION
        )
        self.ratings = {}
        self.history = None

    def _get(self, team):
        return self.ratings.get(team, self.start)

    @staticmethod
    def expected_win_prob(elo_diff):
        return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

    def _regress_to_mean(self):
        for team in list(self.ratings.keys()):
            self.ratings[team] = self.start + (self.ratings[team] - self.start) * (1 - self.season_regression)

    def _mov_multiplier(self, run_diff, winner_elo, loser_elo):
        if run_diff <= 0:
            return 0.5  # defensive fallback; ties are filtered out before fit() runs
        winner_margin_elo_diff = max(winner_elo - loser_elo, 0)
        return math.log(run_diff + 1) * (
            self.mov_mult_base / (0.001 * winner_margin_elo_diff + self.mov_mult_base)
        )

    def fit(self, games):
        """games: DataFrame with date, season, game_num, home_team, away_team,
        home_score, away_score, home_win, sorted or not (we sort here).
        Processes games chronologically, updating self.ratings in place, and
        records a pre-game snapshot of every game for calibration/backtesting.
        """
        games = games[games["home_score"] != games["away_score"]].copy()
        games = games.sort_values(["date", "game_num"]).reset_index(drop=True)

        records = []
        current_season = None
        for row in games.itertuples(index=False):
            if current_season is not None and row.season != current_season:
                self._regress_to_mean()
            current_season = row.season

            home_elo = self._get(row.home_team)
            away_elo = self._get(row.away_team)
            elo_diff_pregame = (home_elo + self.home_advantage) - away_elo
            exp_home = self.expected_win_prob(elo_diff_pregame)

            actual_home = 1.0 if row.home_win else 0.0
            run_diff = abs(row.home_score - row.away_score)
            if row.home_win:
                winner_elo, loser_elo = home_elo + self.home_advantage, away_elo
            else:
                winner_elo, loser_elo = away_elo, home_elo + self.home_advantage
            mov_mult = self._mov_multiplier(run_diff, winner_elo, loser_elo)
            delta = self.k * mov_mult * (actual_home - exp_home)

            records.append(
                {
                    "date": row.date,
                    "season": row.season,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "home_elo_pre": home_elo,
                    "away_elo_pre": away_elo,
                    "elo_win_prob_home": exp_home,
                    "home_win": row.home_win,
                    "home_score": row.home_score,
                    "away_score": row.away_score,
                }
            )

            self.ratings[row.home_team] = home_elo + delta
            self.ratings[row.away_team] = away_elo - delta

        self.history = pd.DataFrame(records)
        return self

    def rating(self, team):
        return self._get(team)

    def win_probability(self, home_team, away_team):
        home_elo = self._get(home_team)
        away_elo = self._get(away_team)
        diff = (home_elo + self.home_advantage) - away_elo
        return self.expected_win_prob(diff)

    def ratings_table(self):
        return pd.DataFrame(
            sorted(self.ratings.items(), key=lambda kv: -kv[1]), columns=["team", "elo"]
        )

    def save(self, path):
        self.ratings_table().to_csv(path, index=False)

    def load_ratings(self, path):
        df = pd.read_csv(path)
        self.ratings = dict(zip(df["team"], df["elo"]))
        return self
