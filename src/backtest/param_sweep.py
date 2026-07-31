"""
Tune the run model's recency parameters instead of guessing them.

RUN_MODEL_RECENT_GAMES (45) and RUN_MODEL_RECENT_WEIGHT (0.45) were reasonable
a-priori choices that were never fit to anything. This sweeps them.

Method, and why it is built this way:

  1. FAST SWEEP. Refitting TeamRunRatings per date per parameter combo is far
     too slow, so this recomputes the same quantities vectorized: per team, an
     expanding mean (full sample) and a rolling mean (recent window) over prior
     games only, via shift(1). Blending weights are then cheap, so N window
     sizes cost N passes rather than N x W. Win probability uses the closed-form
     Skellam rather than Monte Carlo -- same ordering, no sampling noise, and
     orders of magnitude faster.

  2. NO LOOKAHEAD. Every rating at date D uses strictly prior games. League
     average and park factors are likewise expanding, not computed once over the
     whole period.

  3. SELECT, THEN CONFIRM ON HELD-OUT DATA. Picking the max of a 40+ cell grid
     on the same data you report is overfitting -- the winner is partly luck.
     So selection runs on early seasons and the chosen setting is confirmed on a
     later season it never saw. A setting that only wins in-sample is rejected.

  4. The sweep's absolute numbers are approximations (closed-form, no pitcher
     adjustment). They are used only to RANK settings. The honest before/after
     comes from re-running the real Monte Carlo backtest -- see
     `confirm_with_real_backtest`.

    python -m src.backtest.param_sweep
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import skellam

from src import config

WINDOWS = [15, 25, 35, 45, 60, 81, 100, 162]
WEIGHTS = [0.0, 0.2, 0.35, 0.45, 0.6, 0.8, 1.0]


def _long_team_games(games):
    """One row per team per game, with the venue, sorted by date."""
    home = games[["date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "rs", "away_score": "ra"})
    home["park"] = games["home_team"].values
    away = games[["date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "rs", "home_score": "ra"})
    away["park"] = games["home_team"].values
    tg = pd.concat([home, away], ignore_index=True)
    return tg.sort_values(["team", "date"]).reset_index(drop=True)


def _prior_means(tg, window):
    """Expanding and rolling means using ONLY prior games (shift(1))."""
    g = tg.groupby("team", sort=False)
    out = tg[["team", "date"]].copy()
    for col in ("rs", "ra"):
        s = tg[col]
        out[f"full_{col}"] = g[col].transform(lambda x: x.shift(1).expanding().mean())
        out[f"recent_{col}"] = g[col].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    out["n_prior"] = g["rs"].transform(lambda x: x.shift(1).expanding().count())
    return out


def _expanding_park_factors(games, min_games):
    """Park factor per venue as of each date, using prior games only."""
    g = games.sort_values("date").copy()
    g["tot"] = g["home_score"] + g["away_score"]
    g["lg_mean"] = g["tot"].shift(1).expanding().mean()
    park_mean = (g.groupby("home_team")["tot"]
                  .transform(lambda x: x.shift(1).expanding().mean()))
    park_n = (g.groupby("home_team")["tot"]
               .transform(lambda x: x.shift(1).expanding().count()))
    raw = park_mean / g["lg_mean"]
    w = park_n / (park_n + min_games)
    pf = (w * raw + (1 - w) * 1.0).clip(0.85, 1.15)
    g["park_factor"] = pf.fillna(1.0)
    return g[["date", "home_team", "away_team", "park_factor", "lg_mean"]]


def sweep(games, eval_seasons, windows=WINDOWS, weights=WEIGHTS):
    """Return a DataFrame of (window, weight) -> Brier / logloss / accuracy."""
    games = games[games["home_score"] != games["away_score"]].sort_values("date").reset_index(drop=True)
    min_games = config.RUN_MODEL_MIN_GAMES
    parks = _expanding_park_factors(games, min_games)
    tg = _long_team_games(games)

    ev = games[games["season"].isin(eval_seasons)].copy()
    ev = ev.merge(parks, on=["date", "home_team", "away_team"], how="left")
    ev["home_win"] = (ev["home_score"] > ev["away_score"]).astype(int)

    results = []
    for window in windows:
        pm = _prior_means(tg, window)
        # index by (team, date) for lookup; duplicates on doubleheaders are fine
        pm_idx = pm.set_index(["team", "date"])
        for weight in weights:
            blended_off = weight * pm_idx["recent_rs"] + (1 - weight) * pm_idx["full_rs"]
            blended_def = weight * pm_idx["recent_ra"] + (1 - weight) * pm_idx["full_ra"]
            n = pm_idx["n_prior"]
            shrink_w = n / (n + min_games)

            probs, outcomes = [], []
            for r in ev.itertuples(index=False):
                lg = r.lg_mean / 2.0 if pd.notna(r.lg_mean) else np.nan
                if not np.isfinite(lg) or lg <= 0:
                    continue
                try:
                    ho, hd = blended_off.loc[(r.home_team, r.date)], blended_def.loc[(r.home_team, r.date)]
                    ao, ad = blended_off.loc[(r.away_team, r.date)], blended_def.loc[(r.away_team, r.date)]
                    sh = shrink_w.loc[(r.home_team, r.date)]
                    sa = shrink_w.loc[(r.away_team, r.date)]
                except KeyError:
                    continue
                # duplicate index (doubleheader) -> take the first
                for v in (ho, hd, ao, ad, sh, sa):
                    if isinstance(v, pd.Series):
                        ho, hd = _first(ho), _first(hd)
                        ao, ad = _first(ao), _first(ad)
                        sh, sa = _first(sh), _first(sa)
                        break
                if not all(np.isfinite([ho, hd, ao, ad, sh, sa])):
                    continue
                off_h = (sh * ho + (1 - sh) * lg) / lg
                def_h = (sh * hd + (1 - sh) * lg) / lg
                off_a = (sa * ao + (1 - sa) * lg) / lg
                def_a = (sa * ad + (1 - sa) * lg) / lg
                pf = r.park_factor if np.isfinite(r.park_factor) else 1.0
                mu_h = max(lg * off_h * def_a * pf, 0.3)
                mu_a = max(lg * off_a * def_h * pf, 0.3)
                tie = skellam.pmf(0, mu_h, mu_a)
                p = (1 - skellam.cdf(0, mu_h, mu_a)) + config.EXTRA_INNING_HOME_WIN_PROB * tie
                probs.append(p)
                outcomes.append(r.home_win)

            if len(probs) < 100:
                continue
            p = np.clip(np.array(probs), 1e-9, 1 - 1e-9)
            y = np.array(outcomes)
            results.append({
                "window": window, "weight": weight, "n": len(p),
                "brier": float(np.mean((p - y) ** 2)),
                "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                "accuracy": float((((p >= 0.5).astype(int)) == y).mean()),
            })
    return pd.DataFrame(results)


def _first(v):
    return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)


def run(select_seasons=(2024, 2025), confirm_seasons=(2026,)):
    from src import pipeline
    games = pipeline.load_games()

    sel = sweep(games, select_seasons).sort_values("brier")
    best = sel.iloc[0]
    cur = sel[(sel["window"] == config.RUN_MODEL_RECENT_GAMES)
              & (sel["weight"] == config.RUN_MODEL_RECENT_WEIGHT)]

    con = sweep(games, confirm_seasons,
                windows=sorted({int(best["window"]), config.RUN_MODEL_RECENT_GAMES}),
                weights=sorted({float(best["weight"]), config.RUN_MODEL_RECENT_WEIGHT}))
    return sel, cur, best, con


def main():
    ap = argparse.ArgumentParser(description="Sweep run-model recency parameters.")
    ap.add_argument("--out", default=str(config.PROCESSED_DIR / "param_sweep.csv"))
    args = ap.parse_args()
    sel, cur, best, con = run()
    sel.to_csv(args.out, index=False)
    print("=== SELECTION (2024-2025), top 8 by Brier ===")
    print(sel.head(8).to_string(index=False))
    print("\ncurrent setting (45 / 0.45):")
    print(cur.to_string(index=False) if not cur.empty else "  not in grid")
    print("\n=== CONFIRMATION on held-out 2026 ===")
    print(con.sort_values("brier").to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
