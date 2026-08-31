"""Run the Kelly-Pruitt-Su instrumented PCA benchmark."""
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
from ipca_asset_pricing import IPCAConfig, build_ipca_outputs  # noqa: E402


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "asset_pricing_ml" / "ipca_asset_pricing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-set", choices=sorted(FEATURE_SETS), default="compustat_enriched"
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
    parser.add_argument(
        "--no-constant",
        action="store_true",
        help="Drop the constant instrument from the characteristic vector.",
    )
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--factor-ridge", type=float, default=1e-8)
    parser.add_argument("--gamma-ridge", type=float, default=1e-8)
    parser.add_argument("--minimum-size-percentile", type=float, default=0.05)
    parser.add_argument("--training-return-clip", type=float, default=1.0)
    parser.add_argument("--max-monthly-stocks", type=int)
    parser.add_argument(
        "--universe-selection",
        choices=["random", "top_size"],
        default="random",
        help="How --max-monthly-stocks picks the universe. top_size is the "
        "large-cap liquidity screen; random is a speed/robustness subsample.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = IPCAConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_monthly_stocks=args.min_monthly_stocks,
        min_training_months=args.min_training_months,
        validation_months=args.validation_months,
        training_window_months=args.training_window_months,
        n_factors=args.n_factors,
        include_constant=not args.no_constant,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        factor_ridge=args.factor_ridge,
        gamma_ridge=args.gamma_ridge,
        minimum_size_percentile=args.minimum_size_percentile,
        training_return_clip=args.training_return_clip,
        max_monthly_stocks=args.max_monthly_stocks,
        universe_selection=args.universe_selection,
        random_state=args.random_state,
    )
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_ipca_outputs(
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
