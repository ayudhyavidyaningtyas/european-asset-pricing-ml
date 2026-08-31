"""Run the Gu-Kelly-Xiu conditional autoencoder adaptation."""
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
from asset_pricing_ml import FEATURE_SETS  # noqa: E402
from autoencoder_asset_pricing import (  # noqa: E402
    AutoencoderAssetPricingConfig,
    build_autoencoder_outputs,
)


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "asset_pricing_ml" / "autoencoder_asset_pricing"


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
    parser.add_argument("--n-factors", type=int, default=5)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[16])
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--factor-ridge", type=float, default=1e-4)
    parser.add_argument("--minimum-size-percentile", type=float, default=0.05)
    parser.add_argument("--training-return-clip", type=float, default=1.0)
    parser.add_argument(
        "--max-monthly-stocks",
        type=int,
        help="Optional deterministic monthly stock cap for faster architecture tests.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, mps, cuda, or auto. CPU is the reproducible default.",
    )
    args = parser.parse_args()

    config = AutoencoderAssetPricingConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_monthly_stocks=args.min_monthly_stocks,
        min_training_months=args.min_training_months,
        validation_months=args.validation_months,
        training_window_months=args.training_window_months,
        n_factors=args.n_factors,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        factor_ridge=args.factor_ridge,
        minimum_size_percentile=args.minimum_size_percentile,
        training_return_clip=args.training_return_clip,
        max_monthly_stocks=args.max_monthly_stocks,
        random_state=args.random_state,
        device=args.device,
    )
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_autoencoder_outputs(
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
