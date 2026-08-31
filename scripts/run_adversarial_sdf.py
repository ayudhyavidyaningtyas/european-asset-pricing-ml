"""Run the CPZ-style adversarial LSTM/GAN SDF adaptation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adversarial_sdf import (  # noqa: E402
    AdversarialSDFConfig,
    build_adversarial_sdf_outputs,
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
DEFAULT_MARKET_STATES = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "depth_analysis" / "market_states.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "adversarial_sdf_compustat"
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
    parser.add_argument("--market-states", type=Path, default=DEFAULT_MARKET_STATES)
    parser.add_argument(
        "--additional-state-features",
        type=Path,
        help="Optional CSV with date/signal_date and numeric state features.",
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
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--state-hidden-size", type=int, default=8)
    parser.add_argument("--sdf-hidden-sizes", nargs="+", type=int, default=[32, 16])
    parser.add_argument(
        "--adversary-hidden-sizes",
        nargs="+",
        type=int,
        default=[32, 16],
    )
    parser.add_argument("--test-assets", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--adversary-steps", type=int, default=1)
    parser.add_argument("--sdf-steps", type=int, default=1)
    parser.add_argument("--learning-rate-sdf", type=float, default=0.001)
    parser.add_argument("--learning-rate-adversary", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--sdf-gross-leverage", type=float, default=1.0)
    parser.add_argument("--adversary-gross-leverage", type=float, default=1.0)
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
        default="auto",
        help="auto, cpu, mps, or cuda. auto prefers MPS/CUDA when available.",
    )
    args = parser.parse_args()

    config = AdversarialSDFConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_monthly_stocks=args.min_monthly_stocks,
        min_training_months=args.min_training_months,
        validation_months=args.validation_months,
        training_window_months=args.training_window_months,
        sequence_length=args.sequence_length,
        state_hidden_size=args.state_hidden_size,
        sdf_hidden_sizes=tuple(args.sdf_hidden_sizes),
        adversary_hidden_sizes=tuple(args.adversary_hidden_sizes),
        test_assets=args.test_assets,
        epochs=args.epochs,
        patience=args.patience,
        adversary_steps=args.adversary_steps,
        sdf_steps=args.sdf_steps,
        learning_rate_sdf=args.learning_rate_sdf,
        learning_rate_adversary=args.learning_rate_adversary,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        sdf_gross_leverage=args.sdf_gross_leverage,
        adversary_gross_leverage=args.adversary_gross_leverage,
        minimum_size_percentile=args.minimum_size_percentile,
        training_return_clip=args.training_return_clip,
        max_monthly_stocks=args.max_monthly_stocks,
        random_state=args.random_state,
        device=args.device,
    )
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_adversarial_sdf_outputs(
        args.panel,
        args.output_dir,
        config,
        risk_free=risk_free,
        feature_set=args.feature_set,
        market_state_path=args.market_states,
        additional_state_path=args.additional_state_features,
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
