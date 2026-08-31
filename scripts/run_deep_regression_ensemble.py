"""Run the ex-ante Deep Regression Ensemble asset-pricing experiment.

This runner is intentionally prespecified before the full estimates-panel run.
It is the dissertation's deep asset-pricing scaffold, not a tuning loop.

Frozen design:
- Architecture: Didisheim-Kelly-Malamud-style Deep Regression Ensemble using
  random ReLU feature blocks, closed-form ridge grids inside each layer, stacked
  layer predictions, and one validation-selected final ridge alpha.
- Data: full Compustat + Refinitiv estimates monthly panel
  (data/processed/asset_pricing/monthly_feature_panel_estimates.parquet).
- Sample: signal months from 2005-01-31 onward, annual expanding walk-forward
  splits, 2015-2026 out-of-sample test years.
- Features: estimates_enriched feature set, without restricting to estimates-
  covered stocks.
- Target: rank target for investable long-short and long-only portfolio scores.
- Baselines: momentum, ridge, HistGradientBoosting, and MLP on the identical
  sample/splits.
- DRE hyperparameters: 2 layers, 64 random ReLU features per gamma block,
  gamma grid (0.25, 0.5, 1.0, 2.0), per-layer ridge alpha grid
  (0.1, 1.0, 10.0, 100.0), final alpha grid (0.1, 1.0, 10.0, 100.0)
  selected on the trailing validation window and then refit on the full
  available training sample for that test year.

The output contract is exactly the existing ML pipeline contract:
predictions.parquet, prediction_metrics.csv, monthly_portfolios.csv,
model_summary.csv, predictive accuracy tests, and the manifest. This means the
portfolio, cost, ladder, and paired-test scripts can consume the run unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_depth import load_eur_short_rate  # noqa: E402
from asset_pricing_ml import (  # noqa: E402
    FEATURE_SETS,
    SUPPORTED_MODELS,
    WalkForwardConfig,
    build_ml_outputs,
)


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_estimates.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "dre_estimates_enriched_post2005_ex_ante"
)
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
DEFAULT_MODELS = ("momentum", "ridge", "hist_gbm", "mlp", "dre")
DEFAULT_TARGETS = ("rank",)
DEFAULT_SAMPLE_START_DATE = "2005-01-31"
DEFAULT_DRE_GAMMAS = (0.25, 0.5, 1.0, 2.0)
DEFAULT_DRE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
DEFAULT_DRE_FINAL_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the prespecified DRE experiment on the full estimates-enriched "
            "European asset-pricing panel."
        )
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="estimates_enriched",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=sorted(SUPPORTED_MODELS),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["rank", "return", "residual_rank", "residual_return"],
        default=list(DEFAULT_TARGETS),
    )
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--sample-start-date", default=DEFAULT_SAMPLE_START_DATE)
    parser.add_argument("--sample-end-date")
    parser.add_argument("--min-training-rows", type=int, default=10_000)
    parser.add_argument(
        "--max-training-rows",
        type=int,
        help="Optional deterministic monthly-stratified cap for smoke runs.",
    )
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-grid-bps", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--mlp-batch-size", type=int, default=8192)
    parser.add_argument("--dre-layers", type=int, default=2)
    parser.add_argument("--dre-features-per-block", type=int, default=64)
    parser.add_argument(
        "--dre-gammas",
        nargs="+",
        type=float,
        default=list(DEFAULT_DRE_GAMMAS),
    )
    parser.add_argument(
        "--dre-alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_DRE_ALPHAS),
    )
    parser.add_argument(
        "--dre-final-alphas",
        nargs="+",
        type=float,
        default=list(DEFAULT_DRE_FINAL_ALPHAS),
    )
    parser.add_argument(
        "--placebo-repetitions",
        type=int,
        default=0,
        help="Keep zero for the ex-ante DRE screen; run placebo separately.",
    )
    parser.add_argument(
        "--delisting-audit",
        type=Path,
        default=DEFAULT_DELISTING_AUDIT,
    )
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument(
        "--skip-delisting-scenarios",
        action="store_true",
    )
    parser.add_argument(
        "--collect-importance",
        action="store_true",
        help="Optional, expensive ablation diagnostics; off in the ex-ante screen.",
    )
    parser.add_argument(
        "--require-estimates-feature",
        action="store_true",
        help="Optional covered-stock-only diagnostic; off in the frozen design.",
    )
    parser.add_argument(
        "--require-estimate-signal-lag-months",
        type=int,
        help=(
            "Optional leakage guard for panels that include est_signal_lag_months."
        ),
    )
    return parser


def build_config(args: argparse.Namespace) -> WalkForwardConfig:
    return WalkForwardConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_training_rows=args.min_training_rows,
        max_training_rows=args.max_training_rows,
        portfolio_quantile=args.portfolio_quantile,
        cost_grid_bps=tuple(args.cost_grid_bps),
        random_state=args.random_state,
        validation_months=args.validation_months,
        tune_hyperparameters=True,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        dre_layers=args.dre_layers,
        dre_features_per_block=args.dre_features_per_block,
        dre_gammas=tuple(args.dre_gammas),
        dre_alphas=tuple(args.dre_alphas),
        dre_final_alpha=float(args.dre_final_alphas[0]),
        dre_tune_final_alpha=True,
        dre_final_alphas=tuple(args.dre_final_alphas),
    )


def ex_ante_manifest(args: argparse.Namespace, config: WalkForwardConfig) -> dict:
    return {
        "purpose": "prespecified_deep_regression_ensemble_screen",
        "panel": str(args.panel),
        "output_dir": str(args.output_dir),
        "feature_set": args.feature_set,
        "models": list(args.models),
        "targets": list(args.targets),
        "sample_start_date": args.sample_start_date,
        "sample_end_date": args.sample_end_date,
        "require_estimates_feature": args.require_estimates_feature,
        "require_estimate_signal_lag_months": (
            args.require_estimate_signal_lag_months
        ),
        "collect_importance": args.collect_importance,
        "placebo_repetitions": args.placebo_repetitions,
        "config": asdict(config),
        "frozen_config_note": (
            "DRE architecture, feature set, sample window, validation window, "
            "and alpha grids are fixed ex ante; post-result changes should be "
            "reported as separate diagnostics."
        ),
    }


def write_ex_ante_config(args: argparse.Namespace, config: WalkForwardConfig) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "dre_ex_ante_config.json"
    path.write_text(json.dumps(ex_ante_manifest(args, config), indent=2))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    config_path = write_ex_ante_config(args, config)
    risk_free = (
        load_eur_short_rate(args.eur_rate)
        if args.eur_rate.exists()
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
        collect_importance=args.collect_importance,
        sample_start_date=args.sample_start_date,
        sample_end_date=args.sample_end_date,
        require_estimates_feature=args.require_estimates_feature,
        require_estimate_signal_lag_months=args.require_estimate_signal_lag_months,
    )
    print(f"ex-ante config: {config_path}")
    print(f"predictions: {manifest['rows']['predictions']:,}")
    print(f"causality violations: {manifest['causality_check']['train_target_after_cutoff']}")
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
