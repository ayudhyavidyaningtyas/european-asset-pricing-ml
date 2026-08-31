"""Run ML peer-implied fundamental mispricing signals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_depth import load_eur_short_rate  # noqa: E402
from fundamental_mispricing import (  # noqa: E402
    FundamentalMispricingConfig,
    SUPPORTED_FAIR_VALUE_MODELS,
    build_fundamental_mispricing_outputs,
)


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_ANNUAL = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "compustat_exports"
    / "compustat_global_fundamentals_annual.csv.gz"
)
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "fundamental_mispricing"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--annual", type=Path, default=DEFAULT_ANNUAL)
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(SUPPORTED_FAIR_VALUE_MODELS),
        default=["linear", "rf", "hist_gbm", "ensemble"],
    )
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--training-window-months", type=int, default=48)
    parser.add_argument("--min-training-rows", type=int, default=10_000)
    parser.add_argument("--min-training-months", type=int, default=24)
    parser.add_argument("--min-monthly-stocks", type=int, default=100)
    parser.add_argument("--min-accounting-features", type=int, default=8)
    parser.add_argument("--max-training-rows", type=int, default=150_000)
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-grid-bps", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--exclude-financials", action="store_true")
    parser.add_argument(
        "--fair-value-target",
        choices=["market_share", "log_market_share"],
        default="market_share",
        help="Paper-style deflated market value target, or the log variant.",
    )
    parser.add_argument("--linear-alpha", type=float, default=1.0)
    parser.add_argument("--rf-estimators", type=int, default=120)
    parser.add_argument("--rf-max-depth", type=int, default=10)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=50)
    parser.add_argument("--hist-learning-rate", type=float, default=0.05)
    parser.add_argument("--hist-max-iter", type=int, default=150)
    parser.add_argument("--hist-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--hist-min-samples-leaf", type=int, default=50)
    parser.add_argument(
        "--no-momentum-baseline",
        action="store_true",
        help="Do not add 12-2 momentum on the same scoreable universe.",
    )
    args = parser.parse_args()

    config = FundamentalMispricingConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        training_window_months=args.training_window_months,
        min_training_rows=args.min_training_rows,
        min_training_months=args.min_training_months,
        min_monthly_stocks=args.min_monthly_stocks,
        min_accounting_features=args.min_accounting_features,
        max_training_rows=args.max_training_rows,
        portfolio_quantile=args.portfolio_quantile,
        cost_grid_bps=tuple(args.cost_grid_bps),
        random_state=args.random_state,
        exclude_financials=args.exclude_financials,
        fair_value_target=args.fair_value_target,
        linear_alpha=args.linear_alpha,
        rf_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        hist_learning_rate=args.hist_learning_rate,
        hist_max_iter=args.hist_max_iter,
        hist_max_leaf_nodes=args.hist_max_leaf_nodes,
        hist_min_samples_leaf=args.hist_min_samples_leaf,
    )
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_fundamental_mispricing_outputs(
        args.panel,
        args.annual,
        args.output_dir,
        args.models,
        config,
        risk_free=risk_free,
        include_momentum=not args.no_momentum_baseline,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(
        "causality violations: "
        f"{manifest['causality_check']['train_signal_after_cutoff']}",
        flush=True,
    )
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
