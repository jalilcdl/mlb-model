"""
Run the independent moneyline model backtest.

Usage:
    python -m src.run_gbm_backtest --model logistic --track team_only
    python -m src.run_gbm_backtest --model xgboost --track team_only
    python -m src.run_gbm_backtest --model xgboost --track pitcher_augmented

This runs the walk-forward backtest against real 2014-2019 closing moneylines
from pwu97/bettingtools. The team_only track is fully validated; the
pitcher_augmented track adds pitcher features but is HONESTLY labeled as
unvalidated against closing lines.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.models.gbm_independent import LogisticBaseline, XGBoostModel
from src.backtest.ml_backtest import run_backtest


def main():
    ap = argparse.ArgumentParser(description="Independent MLB moneyline model backtest")
    ap.add_argument("--model", choices=["logistic", "xgboost"], default="logistic",
                    help="Model architecture (logistic baseline first, then xgboost)")
    ap.add_argument("--track", choices=["team_only", "pitcher_augmented"], default="team_only",
                    help="Feature track: team_only (validated) or pitcher_augmented (unvalidated)")
    ap.add_argument("--retrain", default="7D",
                    help="Retraining frequency: 1D (daily), 7D (weekly), etc.")
    ap.add_argument("--C", type=float, default=1.0,
                    help="Logistic regression regularization strength")
    ap.add_argument("--n-estimators", type=int, default=500,
                    help="XGBoost n_estimators")
    ap.add_argument("--max-depth", type=int, default=5,
                    help="XGBoost max_depth")
    ap.add_argument("--learning-rate", type=float, default=0.05,
                    help="XGBoost learning_rate")
    ap.add_argument("--output", default=None,
                    help="Output CSV path (default: auto-generated)")
    args = ap.parse_args()

    # Select model
    if args.model == "logistic":
        model_class = LogisticBaseline
        model_kwargs = {"C": args.C}
    else:
        model_class = XGBoostModel
        model_kwargs = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        }

    include_pitcher = args.track == "pitcher_augmented"

    # Auto-generate output path
    if args.output is None:
        suffix = f"{args.model}_{args.track}"
        args.output = config.PROCESSED_DIR / f"ml_backtest_{suffix}.csv"

    # Run
    results, summary = run_backtest(
        model_class=model_class,
        model_kwargs=model_kwargs,
        include_pitcher=include_pitcher,
        retrain_every=args.retrain,
        save_path=args.output,
    )

    return 0 if (summary.get("all_bets", {}).get("avg_ev_pct", {}).get("mean", -1) > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
