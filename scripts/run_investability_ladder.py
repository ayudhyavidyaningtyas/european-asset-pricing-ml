"""Run the frozen-prediction investability ladder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from investability_ladder import (  # noqa: E402
    LadderConfig,
    load_ladder_panel,
    paired_ladder_inference,
    predictive_metrics_by_rung,
    simulate_investability_ladder,
    summarize_investability_ladder,
    write_ladder_outputs,
)


DEFAULT_PANEL = PROJECT_ROOT / "data/processed/asset_pricing/monthly_feature_panel.parquet"
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results/asset_pricing_ml/revised_full_eur_delisting/predictions.parquet"
)
DEFAULT_LIQUIDITY = (
    PROJECT_ROOT
    / "data/raw/asset_pricing/refinitiv_exports/supplemental/liquidity_monthly_full_period"
)
DEFAULT_RISK = (
    PROJECT_ROOT
    / "results/asset_pricing_ml/depth_analysis/rolling_risk_estimates.parquet"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/asset_pricing_ml/investability_ladder"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-assets", type=int, default=500)
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--baseline-model", default="momentum_rank")
    parser.add_argument("--ce-risk-aversion", type=float, default=3.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    for path in [args.panel, args.predictions]:
        if not path.exists():
            raise SystemExit(f"Missing required input: {path}")
    config = LadderConfig(
        maximum_assets=args.maximum_assets,
        portfolio_quantile=args.portfolio_quantile,
        baseline_model=args.baseline_model,
        ce_risk_aversion=args.ce_risk_aversion,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_blocks=tuple(args.bootstrap_blocks),
        hac_lags=args.hac_lags,
        random_state=args.random_state,
    )
    liquidity = args.liquidity if args.liquidity.exists() else None
    risk = args.risk if args.risk.exists() else None
    panel = load_ladder_panel(
        args.panel,
        args.predictions,
        liquidity,
        risk,
        config,
    )
    monthly = simulate_investability_ladder(panel, config)
    summary = summarize_investability_ladder(monthly, config)
    predictive = predictive_metrics_by_rung(panel, config)
    inference = (
        None if args.skip_inference else paired_ladder_inference(monthly, config)
    )
    manifest = write_ladder_outputs(
        args.output_dir,
        config,
        monthly,
        summary,
        predictive,
        inference,
        {
            "panel": args.panel,
            "predictions": args.predictions,
            "liquidity": liquidity,
            "risk": risk,
        },
    )
    print(json.dumps(manifest["rows"], indent=2))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
