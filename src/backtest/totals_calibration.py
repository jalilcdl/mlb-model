"""
Recalibrate the totals model's P(over) against real historical outcomes, and
test honestly whether calibration converts into betting edge.

Motivation: the totals backtest (src/backtest/totals_backtest.py, 11,706 real
closing lines 2015-2019) showed 50.25% ATS against a 52.38% break-even, with a
7.4% ECE and *inverted* confidence at the extremes -- when the model said 80%
over, overs hit 56%. That is a calibration failure, so recalibration is the
natural fix to try.

The critical caveat, stated up front because it determines what this can
possibly achieve: **calibration is a monotone transform.** It can fix
reliability (are stated probabilities honest?) but cannot create resolution
(can the model rank games at all?). If the model's AUC is ~0.50 it has no
ordering to exploit, and no recalibration will produce ATS edge -- the picks
barely change and the ones that do change are arbitrary. This module reports
AUC and the Brier decomposition alongside the calibrated results so that
distinction is always visible rather than hidden behind an improved ECE.

No lookahead: calibrators are fit walk-forward, on strictly prior seasons only,
then applied to the held-out season. Fitting on the same games you evaluate
would inflate every number here.

    python -m src.backtest.totals_calibration
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import config

BREAK_EVEN_110 = 110.0 / 210.0  # 0.5238 -- ATS win rate needed to profit at -110


# --------------------------------------------------------------------------
# Calibrators (hand-rolled to avoid a heavy sklearn dependency)
# --------------------------------------------------------------------------
def _pava(y, w=None):
    """Pool-Adjacent-Violators: the monotone non-decreasing fit minimizing
    weighted squared error. Standard stack implementation."""
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    vals, wts, cnts = [], [], []
    for i in range(len(y)):
        v, ww, c = y[i], w[i], 1
        while vals and vals[-1] >= v:
            pv, pw, pc = vals.pop(), wts.pop(), cnts.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww += pw
            c += pc
        vals.append(v); wts.append(ww); cnts.append(c)
    out = np.empty(len(y))
    pos = 0
    for v, c in zip(vals, cnts):
        out[pos:pos + c] = v
        pos += c
    return out


class IsotonicCalibrator:
    """Non-parametric monotone calibration. Flexible enough to correct the
    inverted-confidence pattern, at the cost of needing more data."""

    name = "isotonic"

    def fit(self, p, y):
        p = np.asarray(p, float); y = np.asarray(y, float)
        order = np.argsort(p, kind="mergesort")
        self.x_ = p[order]
        self.y_ = _pava(y[order])
        return self

    def predict(self, p):
        p = np.asarray(p, float)
        return np.clip(np.interp(p, self.x_, self.y_), 1e-6, 1 - 1e-6)


class PlattCalibrator:
    """Logistic (sigmoid) calibration on the log-odds: 1/(1+exp(-(a*logit(p)+b))).
    Only two parameters, so it is far more stable on small samples than
    isotonic, but it can only rescale/shift confidence -- it cannot invert a
    non-monotone reliability curve."""

    name = "platt"

    def fit(self, p, y):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, float)
        z = np.log(p / (1 - p))

        def nll(theta):
            a, b = theta
            t = a * z + b
            # numerically stable log-loss
            return float(np.mean(np.logaddexp(0, t) - y * t))

        res = minimize(nll, x0=np.array([1.0, 0.0]), method="BFGS")
        self.a_, self.b_ = res.x
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        return np.clip(1.0 / (1.0 + np.exp(-(self.a_ * z + self.b_))), 1e-6, 1 - 1e-6)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def auc(p, y):
    """Rank-based AUC. 0.50 = no ability to order games at all."""
    p = np.asarray(p, float); y = np.asarray(y, int)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p, y, n_bins=10):
    """Expected Calibration Error: weighted |stated - actual| across bins."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        tot += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(tot)


def brier_decomposition(p, y, n_bins=20):
    p = np.asarray(p, float); y = np.asarray(y, float)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    base = y.mean(); rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        w = m.sum() / len(p)
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return {"brier": float(np.mean((p - y) ** 2)), "reliability": float(rel),
            "resolution": float(res), "uncertainty": float(base * (1 - base))}


def ats_roi(pick, outcome, price=-110):
    """ATS win rate and flat-stake ROI for a set of graded picks."""
    pick = np.asarray(pick); outcome = np.asarray(outcome)
    n = len(pick)
    if n == 0:
        return {"n": 0, "ats": float("nan"), "roi": float("nan")}
    win = (pick == outcome)
    wr = float(win.mean())
    payout = 100.0 / abs(price) if price < 0 else price / 100.0
    roi = wr * payout - (1 - wr)
    return {"n": n, "ats": wr, "roi": float(roi)}


# --------------------------------------------------------------------------
# Walk-forward calibration
# --------------------------------------------------------------------------
def walk_forward(df, calibrator_cls, min_train=1500):
    """For each season, fit the calibrator on all STRICTLY EARLIER seasons and
    apply it to that season. Seasons without enough prior data are dropped
    rather than fit on themselves."""
    df = df.sort_values("date").copy()
    df["season"] = df["date"].dt.year
    out = []
    for season in sorted(df["season"].unique()):
        train = df[df["season"] < season]
        test = df[df["season"] == season].copy()
        if len(train) < min_train or test.empty:
            continue
        cal = calibrator_cls().fit(train["p_over"].values, train["y"].values)
        test["p_cal"] = cal.predict(test["p_over"].values)
        out.append(test)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def evaluate(df, p_col):
    """Grade two decision rules:
       naive  -- bet the side with prob > 0.5 (every game gets a bet)
       EV     -- bet only when prob clears the -110 break-even (selective)"""
    p = df[p_col].values
    y = df["y"].values
    res = {"n_games": int(len(df)), "auc": auc(p, y), "ece": ece(p, y),
           **brier_decomposition(p, y)}

    naive_pick = np.where(p > 0.5, "over", "under")
    res["naive"] = ats_roi(naive_pick, df["outcome"].values)

    bet = (p > BREAK_EVEN_110) | (p < 1 - BREAK_EVEN_110)
    ev_pick = np.where(p[bet] > 0.5, "over", "under")
    res["ev_rule"] = ats_roi(ev_pick, df["outcome"].values[bet])
    res["ev_rule"]["pct_of_slate_bet"] = float(bet.mean())
    return res


def run():
    src = config.PROCESSED_DIR / "totals_backtest_results.csv"
    raw = pd.read_csv(src, parse_dates=["date"])
    d = raw[raw["outcome"] != "push"].copy()
    d["y"] = (d["outcome"] == "over").astype(int)
    d["p_over"] = d["model_p_over_cond"]

    report = {"n_decided": int(len(d)), "break_even_at_-110": BREAK_EVEN_110,
              "baseline_uncalibrated": evaluate(d.assign(p_cal=d["p_over"]), "p_over")}

    for cls in (IsotonicCalibrator, PlattCalibrator):
        cal_df = walk_forward(d, cls)
        if cal_df.empty:
            continue
        r = evaluate(cal_df, "p_cal")
        r["seasons_evaluated"] = sorted(int(s) for s in cal_df["season"].unique())
        # Did calibration actually change any picks?
        flips = int((np.where(cal_df["p_cal"] > 0.5, "over", "under")
                     != np.where(cal_df["p_over"] > 0.5, "over", "under")).sum())
        r["picks_changed_vs_uncalibrated"] = flips
        report[cls.name] = r

    out = config.PROCESSED_DIR / "totals_calibration_summary.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    rep = run()
    print(json.dumps(rep, indent=2))
