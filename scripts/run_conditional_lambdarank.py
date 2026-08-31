"""Run conditional multi-horizon LambdaRank from 2008."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conditional_lambdarank import (  # noqa: E402
    LambdaRankConfig,
    combine_horizons,
    economic_theme_importance,
    prediction_metrics,
    prepare_ranking_panel,
    run_walk_forward_rankers,
    write_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-test-year", type=int, default=2008)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/asset_pricing_ml/conditional_lambdarank_2008",
    )
    args = parser.parse_args()
    config = LambdaRankConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
    )
    print("preparing residual targets, causal states and execution costs", flush=True)
    panel = prepare_ranking_panel(
        PROJECT_ROOT / "data/processed/asset_pricing/monthly_feature_panel.parquet",
        PROJECT_ROOT
        / "results/asset_pricing_ml/depth_analysis/rolling_risk_estimates.parquet",
        PROJECT_ROOT / "results/asset_pricing_ml/depth_analysis/market_states.csv",
        PROJECT_ROOT / "results/asset_pricing_ml/depth_analysis/eur_market_return.csv",
        PROJECT_ROOT / "data/raw/fred_IR3TIB01EZM156N.csv",
        PROJECT_ROOT
        / "data/raw/asset_pricing/refinitiv_exports/supplemental/liquidity_monthly_full_period",
        config,
    )
    print("training annual unconditional and conditional LambdaRank models", flush=True)
    predictions, fit_log, _, theme_predictions = run_walk_forward_rankers(
        panel, config
    )
    metrics = prediction_metrics(predictions)
    print("computing utility-based economic theme importance", flush=True)
    theme_importance = economic_theme_importance(theme_predictions, config)
    print("selecting multi-horizon blends using trailing OOS utility", flush=True)
    combined, blend_choices = combine_horizons(predictions, config)
    manifest = write_outputs(
        args.output_dir,
        config,
        predictions,
        combined,
        fit_log,
        metrics,
        blend_choices,
        theme_importance,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
