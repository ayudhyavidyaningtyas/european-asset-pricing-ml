"""Run annual expanding-window ML asset-pricing benchmarks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_ml import (  # noqa: E402
    FEATURE_SETS,
    RESIDUAL_CONTROL_SETS,
    SUPPORTED_MODELS,
    WalkForwardConfig,
    build_ml_outputs,
)
from asset_pricing_depth import load_eur_short_rate  # noqa: E402


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_DELISTING_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "delisting_return_audit.csv"
)
DEFAULT_EUR_RATE = (
    PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="baseline",
        help="Keep the published 18-feature baseline separate from liquidity extensions.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["momentum", "ridge", "elastic_net", "hist_gbm", "mlp"],
        choices=sorted(SUPPORTED_MODELS),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["rank", "return", "residual_rank", "residual_return"],
        default=["rank", "return"],
    )
    parser.add_argument(
        "--residual-controls",
        choices=sorted(RESIDUAL_CONTROL_SETS),
        default="full",
        help=(
            "Neutralization design for residual targets. 'full' also controls "
            "for momentum, which invalidates the momentum baseline as a "
            "comparator; use 'country_sector' for a clean country/sector-neutral "
            "read that keeps momentum valid."
        ),
    )
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--min-training-rows", type=int, default=10_000)
    parser.add_argument(
        "--max-training-rows",
        type=int,
        help="Optional deterministic monthly-stratified cap for faster trial runs.",
    )
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-grid-bps", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--mlp-batch-size", type=int, default=8192)
    parser.add_argument("--dre-layers", type=int, default=2)
    parser.add_argument("--dre-features-per-block", type=int, default=64)
    parser.add_argument(
        "--dre-gammas",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0, 2.0],
    )
    parser.add_argument(
        "--dre-alphas",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--dre-final-alpha", type=float, default=10.0)
    parser.add_argument(
        "--dre-tune-final-alpha",
        action="store_true",
        help="Select the DRE final ridge alpha on the trailing validation window.",
    )
    parser.add_argument(
        "--dre-final-alphas",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0, 100.0],
    )
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--no-tuning", action="store_true")
    parser.add_argument("--placebo-repetitions", type=int, default=20)
    parser.add_argument(
        "--delisting-audit",
        type=Path,
        default=DEFAULT_DELISTING_AUDIT,
    )
    parser.add_argument(
        "--risk-free-rate",
        type=Path,
        default=DEFAULT_EUR_RATE,
        help=(
            "Monthly short-rate CSV used for long-only excess returns. The default "
            "is the existing EUR file for the European benchmark."
        ),
    )
    parser.add_argument(
        "--eur-rate",
        type=Path,
        dest="risk_free_rate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-risk-free",
        action="store_true",
        help="Evaluate long-only returns without subtracting a cash rate.",
    )
    parser.add_argument(
        "--skip-delisting-scenarios",
        action="store_true",
    )
    parser.add_argument(
        "--skip-importance",
        action="store_true",
        help="Skip fixed-model OOS ablation diagnostics for faster model screens.",
    )
    parser.add_argument(
        "--sample-start-date",
        help="Optional first signal month to keep in the model sample.",
    )
    parser.add_argument(
        "--sample-end-date",
        help="Optional last signal month to keep in the model sample.",
    )
    parser.add_argument(
        "--require-estimates-feature",
        action="store_true",
        help="Keep only rows with at least one non-null raw estimates feature.",
    )
    parser.add_argument(
        "--require-revision-signal",
        action="store_true",
        help="Keep only rows with at least one non-null raw analyst revision feature.",
    )
    parser.add_argument(
        "--require-estimate-signal-lag-months",
        type=int,
        help=(
            "Fail if any row with non-null estimates features has an "
            "est_signal_lag_months value below this threshold."
        ),
    )
    args = parser.parse_args()

    config = WalkForwardConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_training_rows=args.min_training_rows,
        max_training_rows=args.max_training_rows,
        portfolio_quantile=args.portfolio_quantile,
        cost_grid_bps=tuple(args.cost_grid_bps),
        random_state=args.random_state,
        validation_months=args.validation_months,
        tune_hyperparameters=not args.no_tuning,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        dre_layers=args.dre_layers,
        dre_features_per_block=args.dre_features_per_block,
        dre_gammas=tuple(args.dre_gammas),
        dre_alphas=tuple(args.dre_alphas),
        dre_final_alpha=args.dre_final_alpha,
        dre_tune_final_alpha=args.dre_tune_final_alpha,
        dre_final_alphas=tuple(args.dre_final_alphas),
    )
    risk_free = (
        load_eur_short_rate(args.risk_free_rate)
        if not args.no_risk_free and args.risk_free_rate.exists()
        else None
    )
    delisting_audit = (
        None if args.skip_delisting_scenarios else args.delisting_audit
    )
    manifest = build_ml_outputs(
        args.panel,
        args.output_dir,
        args.models,
        config,
        target_modes=tuple(args.targets),
        placebo_repetitions=args.placebo_repetitions,
        delisting_audit_path=delisting_audit,
        risk_free=risk_free,
        feature_set=args.feature_set,
        collect_importance=not args.skip_importance,
        sample_start_date=args.sample_start_date,
        sample_end_date=args.sample_end_date,
        require_estimates_feature=args.require_estimates_feature,
        require_revision_signal=args.require_revision_signal,
        require_estimate_signal_lag_months=args.require_estimate_signal_lag_months,
        residual_control_set=args.residual_controls,
    )
    print(f"predictions: {manifest['rows']['predictions']:,}")
    print(f"causality violations: {manifest['causality_check']['train_target_after_cutoff']}")
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
