"""Run the AIPM linear transformer SDF adaptation for European equities."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aipm_linear_transformer_sdf import (  # noqa: E402
    AIPMLinearTransformerConfig,
    build_aipm_linear_transformer_outputs,
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
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "aipm_linear_transformer_sdf"
)


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
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--min-monthly-stocks", type=int, default=100)
    parser.add_argument("--min-training-months", type=int, default=72)
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument(
        "--training-window-months",
        type=int,
        help="Use a rolling training window. Omit for expanding training.",
    )
    parser.add_argument("--minimum-size-percentile", type=float, default=0.05)
    parser.add_argument(
        "--max-attention-features",
        type=int,
        default=32,
        help="Use the first N ranked features for the attention basis. Use 0 for all.",
    )
    parser.add_argument("--gross-leverage", type=float, default=1.0)
    parser.add_argument("--training-return-clip", type=float, default=1.0)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    max_attention_features = (
        None if args.max_attention_features == 0 else args.max_attention_features
    )
    config = AIPMLinearTransformerConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_monthly_stocks=args.min_monthly_stocks,
        min_training_months=args.min_training_months,
        validation_months=args.validation_months,
        training_window_months=args.training_window_months,
        minimum_size_percentile=args.minimum_size_percentile,
        max_attention_features=max_attention_features,
        gross_leverage=args.gross_leverage,
        training_return_clip=args.training_return_clip,
        hac_lags=args.hac_lags,
    )
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_aipm_linear_transformer_outputs(
        args.panel,
        args.output_dir,
        config,
        risk_free=risk_free,
        feature_set=args.feature_set,
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
