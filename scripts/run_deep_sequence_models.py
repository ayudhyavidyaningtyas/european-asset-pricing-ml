"""Run deep sequence modelling benchmarks for European stock selection."""
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
from deep_sequence_models import (  # noqa: E402
    SUPPORTED_SEQUENCE_MODELS,
    DeepSequenceConfig,
    build_deep_sequence_outputs,
)


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_DELISTING_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "delisting_return_audit.csv"
)
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "deep_sequence_compustat"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="compustat_enriched",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(SUPPORTED_SEQUENCE_MODELS),
        default=["last_mlp", "lstm", "gru", "attention_lstm"],
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["rank", "return"],
        default=["rank"],
    )
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--min-training-rows", type=int, default=10_000)
    parser.add_argument("--min-training-months", type=int, default=72)
    parser.add_argument(
        "--training-window-months",
        type=int,
        help="Use a rolling training window. Omit for expanding training.",
    )
    parser.add_argument(
        "--max-training-rows",
        type=int,
        default=150_000,
        help="Deterministic monthly-stratified cap for each annual refit.",
    )
    parser.add_argument(
        "--max-validation-rows",
        type=int,
        default=60_000,
        help="Deterministic cap for trailing validation months.",
    )
    parser.add_argument("--validation-months", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--min-history-observations", type=int, default=6)
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-grid-bps", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--recurrent-hidden-size", type=int, default=32)
    parser.add_argument("--recurrent-layers", type=int, default=1)
    parser.add_argument("--head-hidden-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--prediction-batch-size", type=int, default=32768)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--delisting-audit",
        type=Path,
        default=DEFAULT_DELISTING_AUDIT,
    )
    parser.add_argument("--skip-delisting-scenarios", action="store_true")
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--significance-bootstraps", type=int, default=2000)
    parser.add_argument(
        "--significance-blocks",
        nargs="+",
        type=int,
        default=[3, 6, 12],
    )
    args = parser.parse_args()

    config = DeepSequenceConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_training_rows=args.min_training_rows,
        min_training_months=args.min_training_months,
        training_window_months=args.training_window_months,
        max_training_rows=args.max_training_rows,
        max_validation_rows=args.max_validation_rows,
        validation_months=args.validation_months,
        sequence_length=args.sequence_length,
        min_history_observations=args.min_history_observations,
        portfolio_quantile=args.portfolio_quantile,
        cost_grid_bps=tuple(args.cost_grid_bps),
        recurrent_hidden_size=args.recurrent_hidden_size,
        recurrent_layers=args.recurrent_layers,
        head_hidden_sizes=tuple(args.head_hidden_sizes),
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        prediction_batch_size=args.prediction_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        random_state=args.random_state,
        device=args.device,
    )
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_deep_sequence_outputs(
        args.panel,
        args.output_dir,
        args.models,
        config,
        target_modes=tuple(args.targets),
        delisting_audit_path=(
            None if args.skip_delisting_scenarios else args.delisting_audit
        ),
        risk_free=risk_free,
        feature_set=args.feature_set,
        significance_n_boot=args.significance_bootstraps,
        significance_blocks=tuple(args.significance_blocks),
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
