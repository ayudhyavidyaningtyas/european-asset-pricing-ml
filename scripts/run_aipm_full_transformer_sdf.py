"""Run the full AIPM nonlinear transformer SDF adaptation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aipm_full_transformer_sdf import (  # noqa: E402
    AIPMFullTransformerConfig,
    build_aipm_full_transformer_outputs,
)
from asset_pricing_depth import load_eur_short_rate  # noqa: E402
from asset_pricing_ml import FEATURE_SETS  # noqa: E402


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "asset_pricing_ml" / "aipm_full_transformer"
DEFAULT_MODELS = [
    "bsv",
    "linear_attention",
    "dkkm_random_features",
    "own_asset_mlp",
    "nonlinear_transformer",
]


def parse_optional_positive_int(value: str) -> int | None:
    parsed = int(value)
    return None if parsed == 0 else parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="compustat_enriched",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--min-monthly-stocks", type=int, default=100)
    parser.add_argument("--min-training-months", type=int, default=60)
    parser.add_argument("--validation-months", type=int, default=12)
    parser.add_argument(
        "--training-window-months",
        type=parse_optional_positive_int,
        default=72,
        help="Rolling training window in months. Use 0 for expanding.",
    )
    parser.add_argument(
        "--refit-frequency",
        choices=["annual", "monthly"],
        default="annual",
        help="The paper uses monthly; annual is the tractable dissertation default.",
    )
    parser.add_argument("--minimum-size-percentile", type=float, default=0.05)
    parser.add_argument(
        "--max-monthly-stocks",
        type=parse_optional_positive_int,
        default=750,
        help="Top market-cap stocks per month for dense attention. Use 0 for all.",
    )
    parser.add_argument("--gross-leverage", type=float, default=1.0)
    parser.add_argument("--training-return-clip", type=float, default=1.0)
    parser.add_argument(
        "--linear-attention-features",
        type=parse_optional_positive_int,
        default=None,
        help="Use first N features for linear attention. Use 0 for all features.",
    )
    parser.add_argument("--random-feature-count", type=int, default=256)
    parser.add_argument("--transformer-blocks", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=1)
    parser.add_argument("--feedforward-width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--attention-temperature",
        type=float,
        help="Softmax attention temperature. Omit for sqrt(D).",
    )
    parser.add_argument("--implementability-aware-training", action="store_true")
    parser.add_argument(
        "--neural-position-limit",
        type=float,
        default=0.0,
        help="Absolute stock-weight cap inside neural training. Use 0 to disable.",
    )
    parser.add_argument("--neural-net-exposure-penalty", type=float, default=0.0)
    parser.add_argument("--neural-turnover-penalty", type=float, default=0.0)
    parser.add_argument("--neural-spread-cost-penalty", type=float, default=0.0)
    parser.add_argument("--neural-impact-cost-penalty", type=float, default=0.0)
    parser.add_argument("--neural-training-aum-eur", type=float, default=100_000_000.0)
    parser.add_argument("--execution-fallback-half-spread-bps", type=float, default=25.0)
    parser.add_argument("--execution-impact-coefficient", type=float, default=0.10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, mps, cuda, or auto. Dense attention can be memory-heavy.",
    )
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--pricing-error-ridge", type=float, default=1e-4)
    args = parser.parse_args()

    valid_models = set(DEFAULT_MODELS)
    unknown = sorted(set(args.models).difference(valid_models))
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}; choose from {sorted(valid_models)}")

    config = AIPMFullTransformerConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_monthly_stocks=args.min_monthly_stocks,
        min_training_months=args.min_training_months,
        validation_months=args.validation_months,
        training_window_months=args.training_window_months,
        refit_frequency=args.refit_frequency,
        minimum_size_percentile=args.minimum_size_percentile,
        max_monthly_stocks=args.max_monthly_stocks,
        gross_leverage=args.gross_leverage,
        training_return_clip=args.training_return_clip,
        linear_attention_features=args.linear_attention_features,
        random_feature_count=args.random_feature_count,
        transformer_blocks=args.transformer_blocks,
        attention_heads=args.attention_heads,
        feedforward_width=args.feedforward_width,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        attention_temperature=args.attention_temperature,
        implementability_aware_training=args.implementability_aware_training,
        neural_position_limit=args.neural_position_limit,
        neural_net_exposure_penalty=args.neural_net_exposure_penalty,
        neural_turnover_penalty=args.neural_turnover_penalty,
        neural_spread_cost_penalty=args.neural_spread_cost_penalty,
        neural_impact_cost_penalty=args.neural_impact_cost_penalty,
        neural_training_aum_eur=args.neural_training_aum_eur,
        execution_fallback_half_spread_bps=args.execution_fallback_half_spread_bps,
        execution_impact_coefficient=args.execution_impact_coefficient,
        seeds=tuple(args.seeds),
        random_state=args.random_state,
        device=args.device,
        hac_lags=args.hac_lags,
        pricing_error_ridge=args.pricing_error_ridge,
    )
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_aipm_full_transformer_outputs(
        args.panel,
        args.output_dir,
        config,
        risk_free=risk_free,
        feature_set=args.feature_set,
        models=tuple(args.models),
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(
        "causality violations: "
        f"{manifest['causality_check']['train_target_after_cutoff']}",
        flush=True,
    )
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
