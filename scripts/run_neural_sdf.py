"""Run the CPZ-inspired neural SDF adaptation for European equities."""
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
from neural_sdf import NeuralSDFConfig, build_neural_sdf_outputs  # noqa: E402


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
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "neural_sdf_compustat"
)
DEFAULT_LINEAR_ATTENTION = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "linear_attention_sdf"
    / "linear_attention_sdf_monthly.csv"
)
DEFAULT_ML_PORTFOLIOS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "compustat_enriched_full_layer1_p96"
    / "monthly_portfolios.csv"
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
    parser.add_argument(
        "--baseline-monthly",
        type=Path,
        action="append",
        default=None,
        help="CSV with signal_date, model, sdf_return for HAC comparisons.",
    )
    parser.add_argument(
        "--ml-monthly-portfolios",
        type=Path,
        default=DEFAULT_ML_PORTFOLIOS,
        help="Existing ML monthly_portfolios.csv used for paired Sharpe tests.",
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
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--risk-aversion", type=float, default=3.0)
    parser.add_argument("--utility-weight", type=float, default=1.0)
    parser.add_argument("--moment-penalty", type=float, default=100.0)
    parser.add_argument("--gross-leverage", type=float, default=2.0)
    parser.add_argument("--minimum-size-percentile", type=float, default=0.05)
    parser.add_argument("--training-return-clip", type=float, default=1.0)
    parser.add_argument("--cost-grid-bps", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--significance-bootstraps", type=int, default=5000)
    parser.add_argument(
        "--significance-blocks",
        nargs="+",
        type=int,
        default=[3, 6, 12],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    config = NeuralSDFConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_monthly_stocks=args.min_monthly_stocks,
        min_training_months=args.min_training_months,
        validation_months=args.validation_months,
        training_window_months=args.training_window_months,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        risk_aversion=args.risk_aversion,
        utility_weight=args.utility_weight,
        moment_penalty=args.moment_penalty,
        gross_leverage=args.gross_leverage,
        minimum_size_percentile=args.minimum_size_percentile,
        training_return_clip=args.training_return_clip,
        cost_grid_bps=tuple(args.cost_grid_bps),
        random_state=args.random_state,
        hac_lags=args.hac_lags,
    )
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_neural_sdf_outputs(
        args.panel,
        args.output_dir,
        config,
        risk_free=risk_free,
        feature_set=args.feature_set,
        market_state_path=args.market_states,
        additional_state_path=args.additional_state_features,
        baseline_monthly_paths=args.baseline_monthly or [DEFAULT_LINEAR_ATTENTION],
        ml_portfolio_path=args.ml_monthly_portfolios,
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
