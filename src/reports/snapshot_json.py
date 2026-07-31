"""
Export today's slate as a single clean JSON document for external viewers
(mobile widget, etc.) -- the same numbers the dashboard's Game Detail tab shows.

Everything is computed through src/reports/game_detail.py, the module the
dashboard itself uses, so the JSON and the UI cannot drift apart.

Honesty constraints carried through into the payload:
  - `odds_comparable` is false once a game is underway. The simulation is a
    PRE-GAME model with no knowledge of the score, so its probability is not
    comparable to a live in-game price; any edge computed then is meaningless.
    Consumers should hide or grey out edges when this is false.
  - `market_validation` states, per market, whether we have out-of-sample
    evidence of an edge, with the real backing numbers. Moneyline is the only
    validated one.
  - There is NO sharp-money / ticket-percentage data. `disclaimers.sharp_money`
    says so explicitly; it is not silently omitted.
  - Grades are derived from real ratings (methodology in game_detail.py). The
    starter grade carries its own caveat list because that component has no
    validated predictive lift.

    python -m src.reports.snapshot_json                 # today
    python -m src.reports.snapshot_json 2026-07-21
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.reports import game_detail as gd
from src.odds.odds_adapter import (_normalize_team, american_to_prob,
                                   prob_to_american, remove_vig_two_way)


def _f(x, nd=None):
    """JSON-safe float (NaN/NA -> None)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return None
    v = float(x)
    return round(v, nd) if nd is not None else v


def _side_payload(team, model_p, market_p, best_odds, best_book, market, diag):
    edge_pts = (model_p - market_p) * 100 if market_p is not None else None
    payload = {
        "team": team,
        "model_prob": _f(model_p, 4),
        "model_fair_price": prob_to_american(model_p) if model_p else None,
        "market_implied_prob": _f(market_p, 4),
        "edge_pts": _f(edge_pts, 2),
        "best_price": int(best_odds) if best_odds is not None and pd.notna(best_odds) else None,
        "best_book": best_book,
    }
    if edge_pts is None:
        payload.update({"stars": None, "label": None, "rating_note": None,
                        "recommendation": None, "verdict": None})
        return payload
    stars, note = gd.star_rating(edge_pts, market, diag)
    lead, verdict = gd.recommendation_text(
        team, model_p, market_p, best_odds, best_book, market)
    payload.update({
        "stars": stars,
        "label": gd.edge_label(edge_pts),
        "rating_note": note,
        "recommendation": lead,
        "verdict": verdict,
    })
    return payload


def build(date_str, events, preds=None):
    preds = preds if preds is not None else pd.read_csv(
        config.PROCESSED_DIR / f"predictions_{date_str}.csv")
    games_df, elo, ratings = _load_models()
    records = gd.team_records(games_df)

    # Grade inputs. All optional: if a network source is down the grades degrade
    # to N/A rather than the export failing.
    try:
        from src.data import team_stats
        pitching_df = team_stats.fetch_team_pitching()
        fielding_df = team_stats.fetch_team_fielding()
        exposure = team_stats.park_exposure(games_df)
    except Exception:
        pitching_df = fielding_df = None
        exposure = {}

    injury_map = {}
    try:
        from src.data import injuries
        for t in set(preds["away_team"]) | set(preds["home_team"]):
            injury_map[t] = injuries.fetch_injuries(t)
    except Exception:
        pass

    by_key = {}
    for ev in events:
        a, h = _normalize_team(ev["away_team"]), _normalize_team(ev["home_team"])
        if a and h:
            by_key[(a, h)] = ev

    out_games, skipped = [], []
    for g in preds.itertuples(index=False):
        a, h = g.away_team, g.home_team
        ev = by_key.get((a, h))
        status = (ev or {}).get("status") or getattr(g, "status", None)
        comparable, note = gd.odds_comparability(status)
        diag = gd.model_diagnostics(g._asdict(), ratings)

        ml = {"away": None, "home": None}
        if ev:
            try:
                fair_h, fair_a = remove_vig_two_way(
                    american_to_prob(float(ev["consensus_home_ml"])),
                    american_to_prob(float(ev["consensus_away_ml"])))
            except (KeyError, TypeError, ValueError):
                fair_h = fair_a = None
            # The connector's payload gives a best price but not which book
            # offers it, so best_book is null rather than guessed.
            ml["away"] = _side_payload(a, float(g.away_win_prob), fair_a,
                                       ev.get("best_away_ml"), None, "moneyline", diag)
            ml["home"] = _side_payload(h, float(g.home_win_prob), fair_h,
                                       ev.get("best_home_ml"), None, "moneyline", diag)
        else:
            skipped.append(f"{a}@{h} (no odds event matched)")
            ml["away"] = _side_payload(a, float(g.away_win_prob), None, None, None, "moneyline", diag)
            ml["home"] = _side_payload(h, float(g.home_win_prob), None, None, None, "moneyline", diag)

        grades = {}
        for side, team in (("away", a), ("home", h)):
            tg = gd.team_grades(team, elo, ratings, pitching_df=pitching_df,
                                fielding_df=fielding_df, exposure=exposure)
            letter, caveats = gd.pitcher_grade(getattr(g, f"{side}_pitcher_factor"))
            grades[side] = {
                "team_grade": tg["team"]["grade"],
                "offense_grade": tg["offense"]["grade"],
                # Three distinct things -- see disclaimers.grades.
                "run_prevention_grade": tg["run_prevention"]["grade"],
                "pitching_grade": tg["pitching"]["grade"],
                "pitching_fip_park_adj": _f(tg["pitching"].get("fip_park_adj"), 3),
                "pitching_caveats": tg["pitching"].get("caveats", []),
                "fielding_grade": tg["fielding"]["grade"],
                "fielding_runs_prevented": _f(tg["fielding"].get("runs_prevented"), 1),
                "fielding_caveats": tg["fielding"].get("caveats", []),
                "injured_list": injury_map.get(team, []),
                "starter_grade": letter,
                "starter_name": getattr(g, f"{side}_probable_pitcher", None) or None,
                "starter_factor": _f(getattr(g, f"{side}_pitcher_factor"), 3),
                "starter_caveats": caveats,
                "elo": _f(tg["team"]["elo"], 1),
            }

        out_games.append({
            "game_pk": int(g.game_pk) if pd.notna(g.game_pk) else None,
            "matchup": f"{a} @ {h}",
            "start_time_utc": getattr(g, "game_datetime_utc", None),
            "venue": getattr(g, "venue_name", None),
            "status": status,
            "odds_comparable": comparable,
            "odds_note": note,
            "away": {
                "code": a, "name": getattr(g, "away_team_name", None),
                "logo": gd.logo_url(a), "record": gd.format_record(records, a),
            },
            "home": {
                "code": h, "name": getattr(g, "home_team_name", None),
                "logo": gd.logo_url(h), "record": gd.format_record(records, h),
            },
            "projection": {
                "away_runs": _f(g.expected_away_runs, 2),
                "home_runs": _f(g.expected_home_runs, 2),
                "total_runs": _f(g.expected_total, 2),
                "n_sims": int(g.n_sims) if pd.notna(g.n_sims) else None,
                "overdispersion": _f(g.overdispersion, 3),
                "note": "Simulated means, not integer score predictions.",
            },
            "moneyline": ml,
            "grades": {**grades, "home_field_elo_bonus": config.ELO_HOME_ADVANTAGE},
            "diagnostics": {
                "elo_home_prob": _f(diag["elo_home_prob"], 4),
                "monte_carlo_home_prob": _f(diag["mc_home_prob"], 4),
                "closed_form_home_prob": _f(diag["cf_home_prob"], 4),
                "elo_vs_mc_gap_pts": _f(diag["elo_mc_gap_pts"], 1),
                "agreement": diag["agreement"],
                "pitcher_factor_clamped": bool(diag["pitcher_clamped"]),
                "park_factor": _f(diag["park_factor"], 3),
                "park_factor_clipped": bool(diag["park_clipped"]),
                "red_flag": bool(diag["pitcher_clamped"] or diag["park_clipped"]
                                 or diag["agreement"] == "weak"),
            },
        })

    return {
        "generated_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "slate_date": date_str,
        "game_count": len(out_games),
        "games_without_odds": skipped,
        "market_validation": {
            k: {"status": v[0], "evidence": v[1]} for k, v in gd.MARKET_VALIDATION.items()
        },
        "model": {
            "demonstrated_moneyline_edge_pts": gd.DEMONSTRATED_EDGE_PTS,
            "star_rating_note": ("Not a raw edge ranking. Peaks in the credible band near our "
                                 "demonstrated skill and is DOWNGRADED for implausibly large edges "
                                 "and red-flag diagnostics."),
        },
        "disclaimers": {
            "sharp_money": ("No sharp-money or ticket-percentage data is included. That is licensed "
                            "sportsbook betting-flow data which none of our free sources provide, so "
                            "it is deliberately absent rather than fabricated. The `diagnostics` block "
                            "is our OWN internal-agreement metric, not market money flow."),
            "first_five": ("No first-5-innings or first-half projections. The simulation models "
                           "whole-game run totals and has no inning structure."),
            "live_odds": ("`odds_comparable: false` means the model's PRE-GAME probability cannot be "
                          "compared to the currently-posted price (game underway, postponed, or "
                          "finished). Hide the edge when this is false."),
            "best_book": ("`best_book` is null: the odds feed supplies a best price without naming "
                          "the sportsbook offering it."),
            "grades": ("run_prevention / pitching / fielding are THREE DIFFERENT THINGS. "
                       "run_prevention = park-adjusted runs actually allowed (pitching + defense + "
                       "luck). pitching = FIP (HR/BB/HBP/K only), defense-independent by "
                       "construction. fielding = Savant runs prevented. A club can be A in run "
                       "prevention while pitching and fielding diverge sharply -- that split is real "
                       "information a combined grade hides. Offense and run_prevention are "
                       "park-adjusted per game."),
            "injuries": ("`injured_list` is DISPLAY-ONLY CONTEXT and feeds nothing. The source "
                         "exposes only current roster state with no history, so an injury "
                         "adjustment cannot be walk-forward backtested; shipping one would repeat "
                         "the unvalidated-pitcher-factor mistake. 40-man roster only."),
        },
        "games": out_games,
    }


def _load_models():
    from src import pipeline
    games = pipeline.load_games()
    elo, ratings = pipeline.fit_models(games)
    return games, elo, ratings


def run(date_str=None, events=None, out_path=None):
    date_str = date_str or dt.date.today().isoformat()
    if events is None:
        from src.data import espn_odds
        items, _ = espn_odds.normalize_slate(date_str)
        events = _events_from_espn(items)
    payload = build(date_str, events)
    out_path = Path(out_path or (config.PROCESSED_DIR / "snapshot_today.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path, payload


def _events_from_espn(items):
    """Fallback path: reshape ESPN moneyline snapshots into the event shape."""
    out = {}
    for it in items:
        if it["offer_type"] != "moneyline":
            continue
        book = next(iter(it["odds"]))
        offers = {o["side"]: o["odds"] for o in it["odds"][book]}
        out[(it["away_team"], it["home_team"])] = {
            "away_team": it["away_team"], "home_team": it["home_team"],
            "consensus_home_ml": offers.get("home"), "consensus_away_ml": offers.get("away"),
            "best_home_ml": offers.get("home"), "best_away_ml": offers.get("away"),
            "status": None,
        }
    return list(out.values())


def main():
    ap = argparse.ArgumentParser(description="Export today's slate as JSON for external viewers.")
    ap.add_argument("date", nargs="?", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    path, payload = run(args.date, out_path=args.out)
    print(f"wrote {path}")
    print(f"games: {payload['game_count']} | "
          f"odds-comparable: {sum(1 for g in payload['games'] if g['odds_comparable'])}")


if __name__ == "__main__":
    main()
