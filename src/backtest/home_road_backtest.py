"""
Do home/road splits improve predictions? Walk-forward, no lookahead, paired
against the current model on identical games.

THE PROBLEM WITH "JUST ADD HOME/ROAD SPLITS"
It is two different hypotheses wearing one name, and they must be tested apart
or the weaker one free-rides on the stronger.

  A. TEAM-SPECIFIC split skill. Does THIS club genuinely hit better at home
     than the average club does, beyond what the park factor already captures?
     This is the interesting claim and the shaky one.

  B. LEAGUE-WIDE home-field advantage inside the run model. predict_mus has
     NO home-field term at all -- HFA currently lives only in Elo
     (ELO_HOME_ADVANTAGE = 24), and the final probability is a 50/50 blend of
     Elo and the Monte Carlo run model. So half the ensemble is blind to HFA.
     This is a plain structural gap, unrelated to any team's split skill.

Naive home/road ratings bundle B into A: home rates are higher than road rates
for nearly every club simply because home teams score more league-wide, so an
"A" arm built on raw splits would win on B's merits and get credited to the
wrong hypothesis -- and would double-count HFA against Elo's existing term.

So arm A normalizes each side against the LEAGUE's home and road baselines
separately. A team that outscores its road self by exactly the league-average
margin comes out at 1.0 in both splits -- correctly reporting "no team-specific
split skill." Arm B is tested on its own as a single league-wide multiplier.

WHY A IS EXPECTED TO FAIL, AND WHY IT IS TESTED ANYWAY
Splitting halves the sample behind every rating, so noise rises ~40% while the
true team-specific signal (if any) is small. The sabermetric consensus is that
team home/road splits are mostly park plus league HFA, both already modelled.
Stated up front so a null result reads as confirmation rather than an excuse.
Arm A shrinks each split toward the team's own POOLED rating (not the league
average) -- the best-faith version of the hypothesis, since it keeps the split
from thrashing on thin samples.

    python -m src.backtest.home_road_backtest --seasons 2024,2025
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import skellam

from src import config
from src.models.run_model import TeamRunRatings
from src.backtest.pitcher_backtest import _as_of_date_stats, _build_pitcher_index, _pair_doubleheaders

# Games behind a home-or-road split before its own rate outweighs the team's
# pooled rate. Half a season of one-way games; deliberately heavy, because a
# split rating carries half the sample of the pooled one it competes with.
SPLIT_SHRINK_GAMES = 40.0


def _blend_recency(grp, col, league_rate, recent_n, recent_w, min_games):
    """Same recency blend + league shrinkage the shipped model uses, so the arms
    differ ONLY in home/road handling and not in smoothing."""
    n = len(grp)
    if n == 0:
        return 1.0
    recent = grp.tail(recent_n)
    blended = recent_w * recent[col].mean() + (1 - recent_w) * grp[col].mean()
    weight = n / (n + min_games)
    shrunk = weight * blended + (1 - weight) * league_rate
    return shrunk / league_rate if league_rate else 1.0


def split_ratings(train_games, park_factor=None):
    """Per-team offense/defense ratings computed separately for home and road
    games, each normalized against that side's OWN league baseline.

    Returns (ratings_dict, league_hfa) where
        ratings_dict[(team, 'home'|'away')] = {'off': r, 'def': r}
        league_hfa = league mean home runs / league mean road runs
    The normalization is what isolates arm A from arm B: dividing home rates by
    the league HOME average removes the league-wide advantage entirely, leaving
    only each club's deviation from it.
    """
    recent_n = config.RUN_MODEL_RECENT_GAMES
    recent_w = config.RUN_MODEL_RECENT_WEIGHT
    min_games = config.RUN_MODEL_MIN_GAMES

    home_rows = train_games[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "runs_scored", "away_score": "runs_allowed"})
    home_rows["side"] = "home"
    home_rows["park"] = train_games["home_team"].values
    away_rows = train_games[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "runs_scored", "home_score": "runs_allowed"})
    away_rows["side"] = "away"
    away_rows["park"] = train_games["home_team"].values
    tg = pd.concat([home_rows, away_rows], ignore_index=True).sort_values("date")

    # Optional park neutralization. This matters more here than anywhere else in
    # the model: a team's HOME split is, by construction, every game it played in
    # its own park, so the park effect lands entirely on one side of the split.
    # Colorado's home offense rating comes out +0.27 above its road rating almost
    # entirely because of Coors. predict_mus ALREADY multiplies by the park
    # factor, so feeding it park-contaminated splits double-counts the park and
    # would make arm A look bad for a reason that has nothing to do with whether
    # team-specific split skill exists. Running the arm both ways is what makes
    # a null result interpretable instead of ambiguous.
    if park_factor is not None:
        pf = tg["park"].map(park_factor).fillna(1.0).replace(0, 1.0)
        tg["runs_scored"] = tg["runs_scored"] / pf
        tg["runs_allowed"] = tg["runs_allowed"] / pf

    # League baselines PER SIDE -- the core of the A/B separation.
    lg = tg.groupby("side")["runs_scored"].mean()
    lg_home = float(lg.get("home", np.nan))
    lg_away = float(lg.get("away", np.nan))
    if not (np.isfinite(lg_home) and np.isfinite(lg_away) and lg_away > 0):
        return {}, 1.0
    league_hfa = lg_home / lg_away

    # Pooled per-team ratings, used as the shrinkage target for the splits.
    pooled_league = float(tg["runs_scored"].mean())
    pooled = {}
    for team, grp in tg.groupby("team"):
        pooled[team] = {
            "off": _blend_recency(grp.sort_values("date"), "runs_scored", pooled_league,
                                  recent_n, recent_w, min_games),
            "def": _blend_recency(grp.sort_values("date"), "runs_allowed", pooled_league,
                                  recent_n, recent_w, min_games),
        }

    out = {}
    for (team, side), grp in tg.groupby(["team", "side"]):
        grp = grp.sort_values("date")
        n = len(grp)
        lg_scored = lg_home if side == "home" else lg_away
        lg_allowed = lg_away if side == "home" else lg_home  # home pitching faces road bats
        raw_off = _blend_recency(grp, "runs_scored", lg_scored, recent_n, recent_w, min_games)
        raw_def = _blend_recency(grp, "runs_allowed", lg_allowed, recent_n, recent_w, min_games)
        # Shrink toward the team's own pooled rating, not the league's.
        w = n / (n + SPLIT_SHRINK_GAMES)
        p = pooled.get(team, {"off": 1.0, "def": 1.0})
        out[(team, side)] = {
            "off": w * raw_off + (1 - w) * p["off"],
            "def": w * raw_def + (1 - w) * p["def"],
        }
    return out, league_hfa


def run_validation(eval_seasons=(2024, 2025), starter_share=None):
    starter_share = starter_share or config.STARTER_INNINGS_SHARE
    games = pd.read_csv(config.GAMES_FILE, parse_dates=["date"])
    games = games[games["home_score"] != games["away_score"]]
    starters = pd.read_csv(config.HISTORICAL_STARTERS_FILE, parse_dates=["date"]).dropna(
        subset=["home_pitcher_id", "away_pitcher_id"])
    logs = pd.read_csv(config.PITCHER_GAME_LOGS_FILE, parse_dates=["date"])
    pindex = _build_pitcher_index(logs)

    paired = _pair_doubleheaders(games, starters)
    paired = paired[paired["season"].isin(eval_seasons)].sort_values("date")

    rows = []
    for date, day in paired.groupby("date"):
        train = games[games["date"] < date]
        if train.empty:
            continue
        try:
            ratings = TeamRunRatings().fit(train)
        except ValueError:
            continue
        splits, league_hfa = split_ratings(train)
        splits_pa, _ = split_ratings(train, park_factor=ratings.park_factor)
        if not splits or not splits_pa:
            continue

        for r in day.itertuples(index=False):
            season = int(r.season)
            facs = {}
            for side, _team in (("home", r.home_team), ("away", r.away_team)):
                stats = _as_of_date_stats(pindex, getattr(r, f"{side}_pitcher_id"), season, date)
                facs[side] = ratings.pitcher_factor(stats, ratings.league_avg_era_proxy)
            if facs["home"] is None or facs["away"] is None:
                continue

            lg = ratings.league_avg_runs
            park = ratings.park_factor.get(r.home_team, 1.0)

            def probs(off_h, def_h, off_a, def_a, hfa=1.0):
                eff_h = starter_share * facs["home"] + (1 - starter_share) * def_h
                eff_a = starter_share * facs["away"] + (1 - starter_share) * def_a
                # HFA applied as a multiplier on the home side's scoring only;
                # sqrt split so the league total run environment is preserved
                # rather than inflated by the adjustment.
                mu_h = max(lg * off_h * eff_a * park * np.sqrt(hfa), 0.3)
                mu_a = max(lg * off_a * eff_h * park / np.sqrt(hfa), 0.3)
                tie = skellam.pmf(0, mu_h, mu_a)
                return (1 - skellam.cdf(0, mu_h, mu_a)) + config.EXTRA_INNING_HOME_WIN_PROB * tie

            oh = ratings.off_rating.get(r.home_team, 1.0)
            oa = ratings.off_rating.get(r.away_team, 1.0)
            dh = ratings.def_rating.get(r.home_team, 1.0)
            da = ratings.def_rating.get(r.away_team, 1.0)

            sh = splits.get((r.home_team, "home"))
            sa = splits.get((r.away_team, "away"))
            ph = splits_pa.get((r.home_team, "home"))
            pa_ = splits_pa.get((r.away_team, "away"))
            if sh is None or sa is None or ph is None or pa_ is None:
                continue

            rows.append({
                "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
                "home_win": int(r.home_score > r.away_score),
                # BASELINE: pooled ratings, no HFA in the run model (shipped model)
                "p_baseline": probs(oh, dh, oa, da),
                # ARM A: team-specific splits, league HFA normalized OUT
                "p_split": probs(sh["off"], sh["def"], sa["off"], sa["def"]),
                # ARM B: pooled ratings + a single league-wide HFA multiplier
                "p_hfa": probs(oh, dh, oa, da, hfa=league_hfa),
                # ARM A-pa: same as A but park-neutralized, so the park is not
                # counted twice (see split_ratings). This is the arm that
                # actually isolates team-specific split skill.
                "p_split_pa": probs(ph["off"], ph["def"], pa_["off"], pa_["def"]),
                # A+B together, for completeness
                "p_split_hfa": probs(sh["off"], sh["def"], sa["off"], sa["def"], hfa=league_hfa),
                "league_hfa": league_hfa,
            })
    return pd.DataFrame(rows)


ARMS = [
    ("baseline", "p_baseline"),
    ("A_team_splits", "p_split"),
    ("A_team_splits_park_adj", "p_split_pa"),
    ("B_league_hfa", "p_hfa"),
    ("AB_both", "p_split_hfa"),
]


def summarize(df):
    y = df["home_win"].values
    out = {"n_games": int(len(df)), "league_hfa_mean": float(df["league_hfa"].mean())}
    for name, col in ARMS:
        p = np.clip(df[col].values, 1e-9, 1 - 1e-9)
        out[name] = {
            "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
            "brier": float(np.mean((p - y) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        }

    base = df["p_baseline"].values
    rng = np.random.default_rng(0)
    for name, col in ARMS[1:]:
        d = ((base - y) ** 2) - ((df[col].values - y) ** 2)  # positive => arm beats baseline
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        out[name]["brier_improvement"] = {
            "mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0),
        }
        shift = (df[col] - df["p_baseline"]).abs()
        out[name]["mean_abs_prob_shift_pts"] = float(shift.mean() * 100)
        out[name]["max_abs_prob_shift_pts"] = float(shift.max() * 100)
    return out


def main():
    ap = argparse.ArgumentParser(description="Validate home/road splits and run-model HFA.")
    ap.add_argument("--seasons", default="2024,2025")
    args = ap.parse_args()
    seasons = tuple(int(s) for s in args.seasons.split(","))
    df = run_validation(seasons)
    if df.empty:
        print("No evaluable games.")
        return
    s = summarize(df)
    df.to_csv(config.PROCESSED_DIR / "home_road_backtest_results.csv", index=False)
    with open(config.PROCESSED_DIR / "home_road_backtest_summary.json", "w") as f:
        json.dump(s, f, indent=2)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
