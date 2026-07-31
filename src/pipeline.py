"""
End-to-end orchestration: fit ratings from cached historical data, pull a
day's schedule + probable pitchers from the live MLB Stats API, and produce
a full prediction table (moneyline / total / run line), optionally enriched
with manual odds edges.

Fast path: fitting Elo + the run model over ~3 seasons of cached game logs
takes well under a second, so this can be re-run on every dashboard load
without re-scraping anything. Only src.data.fetch_historical hits the slow,
rate-limited network scrape.
"""
import datetime as dt
import json

import pandas as pd
import requests

from src import config
from src.data import statsapi_client
from src.models import ensemble, monte_carlo
from src.models.elo import EloModel
from src.models.run_model import TeamRunRatings, game_probabilities
from src.odds import odds_adapter


def _check_pitcher_changes(rows, watch_file=None):
    """Compare each game's freshly-fetched probable pitchers against the
    last-seen value for that game_pk (a small on-disk JSON file, keyed by
    MLB's own game ID -- no history, no database, just "did it change since
    the last time this was checked"). Flags a change in-place on each row
    and updates the file to the new value either way (last-write-wins).

    Probable pitchers can change last-minute (bullpen games, rain delays,
    injuries), and predictions are only as good as this input -- this is the
    mechanism behind the dashboard's "pitcher changed" warning. Deliberately
    simple: one flat file, no retention policy, no background process. It
    only ever reflects state from calls that actually hit the live API (see
    predict_date's pitchers_fetched_at), not from anything cached.
    """
    watch_file = watch_file or config.PITCHER_WATCH_FILE
    if watch_file.exists():
        try:
            with open(watch_file) as f:
                watch = json.load(f)
        except (json.JSONDecodeError, OSError):
            watch = {}
    else:
        watch = {}

    checked_at = dt.datetime.now().isoformat(timespec="seconds")
    for row in rows:
        key = str(row["game_pk"])
        prev = watch.get(key)
        for side in ("home", "away"):
            pitcher_key = f"{side}_probable_pitcher"
            changed_key = f"{side}_pitcher_changed"
            previous_key = f"{side}_pitcher_previous"
            current = row.get(pitcher_key)
            previous = prev.get(pitcher_key) if prev else None
            changed = bool(prev) and previous is not None and current is not None and previous != current
            row[changed_key] = changed
            row[previous_key] = previous if changed else None
        watch[key] = {
            "home_probable_pitcher": row.get("home_probable_pitcher"),
            "away_probable_pitcher": row.get("away_probable_pitcher"),
            "checked_at": checked_at,
        }

    watch_file.parent.mkdir(parents=True, exist_ok=True)
    with open(watch_file, "w") as f:
        json.dump(watch, f, indent=2)
    return rows


def load_games(path=None):
    path = path or config.GAMES_FILE
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def fit_models(games, as_of_date=None):
    train = games if as_of_date is None else games[games["date"] < pd.Timestamp(as_of_date)]
    elo = EloModel().fit(train)
    ratings = TeamRunRatings().fit(games, as_of_date=as_of_date)
    return elo, ratings


def _pick_moneyline(row):
    return row["home_team"] if row["home_win_prob"] >= 0.5 else row["away_team"]


def _pick_run_line(row):
    if row["home_covers_prob"] >= 0.5:
        return f"{row['home_team']} -{row['run_line']:g}"
    return f"{row['away_team']} +{row['run_line']:g}"


def predict_date(
    date_str,
    games=None,
    elo=None,
    ratings=None,
    use_pitcher_adjustment=True,
    apply_odds=True,
    n_sims=None,
):
    """Monte Carlo simulation is the primary prediction method here (see
    src/models/monte_carlo.py): each game is simulated n_sims times (default
    config.MC_DEFAULT_SIMS) and probabilities are read off the simulated
    outcomes. The closed-form Poisson/Skellam math in run_model.py is run
    alongside every game as a fast, independent cross-check -- both are kept
    in the output so a persistent disagreement between them is visible rather
    than silently discarded.
    """
    games = games if games is not None else load_games()
    if elo is None or ratings is None:
        elo, ratings = fit_models(games)
    n_sims = n_sims or config.MC_DEFAULT_SIMS

    schedule = statsapi_client.get_schedule(date_str)
    pitchers_fetched_at = dt.datetime.now().isoformat(timespec="seconds")
    season = int(date_str[:4])
    rows = []
    for g in schedule:
        home, away = g["home_team"], g["away_team"]
        home_pf = away_pf = None
        home_pitcher_stats = away_pitcher_stats = None
        if use_pitcher_adjustment:
            # A pitcher-stat lookup failing (transient network blip, missing id)
            # must not sink the whole slate -- fall back to no adjustment for
            # that starter, exactly as if the pitcher were unknown/TBD. The
            # schedule call above is the only truly required network call.
            try:
                home_pitcher_stats = statsapi_client.get_pitcher_season_stats(
                    g["home_probable_pitcher_id"], season
                )
            except requests.exceptions.RequestException:
                home_pitcher_stats = None
            try:
                away_pitcher_stats = statsapi_client.get_pitcher_season_stats(
                    g["away_probable_pitcher_id"], season
                )
            except requests.exceptions.RequestException:
                away_pitcher_stats = None
            home_pf = ratings.pitcher_factor(home_pitcher_stats, ratings.league_avg_era_proxy)
            away_pf = ratings.pitcher_factor(away_pitcher_stats, ratings.league_avg_era_proxy)

        mu_home, mu_away = ratings.predict_mus(
            home, away, home_pitcher_factor=home_pf, away_pitcher_factor=away_pf
        )

        sim = monte_carlo.simulate_game(mu_home, mu_away, n_sims=n_sims, overdispersion=ratings.overdispersion)
        cf = game_probabilities(mu_home, mu_away)  # closed-form cross-check, pure Poisson

        elo_prob = elo.win_probability(home, away)
        final_home_win_prob = ensemble.blend_win_prob(elo_prob, sim["home_win_prob"])
        final_away_win_prob = 1 - final_home_win_prob

        rows.append(
            {
                "game_date": g["game_date"],
                "game_datetime_utc": g.get("game_datetime_utc"),
                "game_pk": g["game_pk"],
                "status": g["status"],
                "home_team": home,
                "away_team": away,
                "home_team_name": g["home_team_name"],
                "away_team_name": g["away_team_name"],
                "venue_name": g["venue_name"],
                "home_probable_pitcher": g["home_probable_pitcher_name"],
                "away_probable_pitcher": g["away_probable_pitcher_name"],
                "pitchers_fetched_at": pitchers_fetched_at,
                "home_pitcher_factor": home_pf,
                "away_pitcher_factor": away_pf,
                "elo_home_win_prob": elo_prob,
                "mc_home_win_prob": sim["home_win_prob"],
                "cf_home_win_prob": cf["home_win_prob"],
                "home_win_prob": final_home_win_prob,
                "away_win_prob": final_away_win_prob,
                "expected_home_runs": mu_home,
                "expected_away_runs": mu_away,
                "expected_total": sim["expected_total"],
                "cf_expected_total": cf["expected_total"],
                "total_std": sim["total_std"],
                "run_line": sim["run_line"],
                "home_covers_prob": sim["home_covers_prob"],
                "away_covers_prob": sim["away_covers_prob"],
                "cf_home_covers_prob": cf["home_covers_prob"],
                "cf_away_covers_prob": cf["away_covers_prob"],
                "n_sims": sim["n_sims"],
                "overdispersion": ratings.overdispersion,
                "home_elo": elo.rating(home),
                "away_elo": elo.rating(away),
            }
        )

    rows = _check_pitcher_changes(rows)

    preds = pd.DataFrame(rows)
    if preds.empty:
        return preds

    preds["moneyline_pick"] = preds.apply(_pick_moneyline, axis=1)
    preds["moneyline_pick_prob"] = preds[["home_win_prob", "away_win_prob"]].max(axis=1)
    preds["run_line_pick"] = preds.apply(_pick_run_line, axis=1)
    preds["run_line_pick_prob"] = preds[["home_covers_prob", "away_covers_prob"]].max(axis=1)

    if apply_odds:
        events = odds_adapter.load_market_odds()
        if events:
            preds = odds_adapter.attach_edges(preds, events, n_sims=n_sims)

    return preds


def save_predictions(preds, path=None):
    path = path or config.PREDICTIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(path, index=False)
    return path


def run_for_date(date_str, apply_odds=True, save=True, n_sims=None):
    games = load_games()
    elo, ratings = fit_models(games)
    preds = predict_date(date_str, games=games, elo=elo, ratings=ratings, apply_odds=apply_odds, n_sims=n_sims)
    if save:
        save_predictions(preds)
        elo.save(config.ELO_RATINGS_FILE)
        ratings.to_frame().to_csv(config.TEAM_RATINGS_FILE, index=False)
    return preds, elo, ratings


def run_for_today(apply_odds=True, save=True, n_sims=None):
    return run_for_date(dt.date.today().isoformat(), apply_odds=apply_odds, save=save, n_sims=n_sims)


if __name__ == "__main__":
    preds, _, _ = run_for_today()
    if preds.empty:
        print("No games found for today.")
    else:
        cols = [
            "away_team", "home_team", "away_win_prob", "home_win_prob",
            "expected_total", "run_line_pick",
        ]
        print(preds[cols].to_string(index=False))
