"""Run the cost-aware implementable-efficient-frontier analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from implementable_frontier import (  # noqa: E402
    FrontierConfig,
    attach_execution_inputs,
    build_market_volatility,
    causal_strategy_selection,
    frontier_dominance,
    load_eur_risk_free,
    load_frontier_panel,
    load_monthly_liquidity,
    selected_sharpe_inference,
    simulate_selected_frontiers,
    simulate_frontiers,
    summarize_frontiers,
    summarize_selected,
    write_frontier_outputs,
)


DEFAULT_PANEL = PROJECT_ROOT / "data/processed/asset_pricing/monthly_feature_panel.parquet"
DEFAULT_BASELINE = PROJECT_ROOT / "results/asset_pricing_ml/revised_full_eur_delisting"
DEFAULT_RISK = PROJECT_ROOT / "results/asset_pricing_ml/depth_analysis/rolling_risk_estimates.parquet"
DEFAULT_MARKET = PROJECT_ROOT / "results/asset_pricing_ml/depth_analysis/eur_market_return.csv"
DEFAULT_RF = PROJECT_ROOT / "data/raw/fred_IR3TIB01EZM156N.csv"
DEFAULT_LIQUIDITY = (
    PROJECT_ROOT
    / "data/raw/asset_pricing/refinitiv_exports/supplemental/liquidity_monthly_full_period"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results/asset_pricing_ml/implementable_frontier"


def comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_RF)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--risk-aversions", default="5,20,80")
    parser.add_argument("--adjustment-speeds", default="1,0.5,0.25")
    parser.add_argument("--maximum-assets", type=int, default=500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()

    inputs = [
        args.panel,
        args.baseline_dir / "predictions.parquet",
        args.risk,
        args.market,
        args.eur_rate,
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}")

    config = FrontierConfig(
        maximum_assets=args.maximum_assets,
        risk_aversions=comma_floats(args.risk_aversions),
        adjustment_speeds=comma_floats(args.adjustment_speeds),
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    print("loading frozen signals, causal risk estimates and outcomes", flush=True)
    panel = load_frontier_panel(
        args.panel,
        args.baseline_dir / "predictions.parquet",
        args.risk,
    )
    liquidity_path = args.liquidity if args.liquidity.exists() else None
    liquidity = load_monthly_liquidity(liquidity_path)
    panel = attach_execution_inputs(panel, liquidity, config)
    market_volatility = build_market_volatility(args.market)
    risk_free = load_eur_risk_free(args.eur_rate)
    print(
        f"optimizing frontiers; rows={len(panel):,}; "
        f"observed spreads={int(panel['spread_observed'].sum()):,}",
        flush=True,
    )
    monthly = simulate_frontiers(panel, market_volatility, risk_free, config)
    print("summarizing net frontiers and causal strategy selection", flush=True)
    summary = summarize_frontiers(monthly, config)
    dominance = frontier_dominance(summary)
    _, selection_log = causal_strategy_selection(monthly, config)
    print("replaying causal choices with continuous holdings", flush=True)
    selected = simulate_selected_frontiers(
        panel,
        market_volatility,
        risk_free,
        selection_log,
        config,
    )
    selected_summary = summarize_selected(selected, config)
    inference = selected_sharpe_inference(selected, config)
    manifest = write_frontier_outputs(
        args.output_dir,
        config,
        monthly,
        summary,
        dominance,
        selected,
        selection_log,
        selected_summary,
        inference,
        {
            "panel": args.panel,
            "predictions": args.baseline_dir / "predictions.parquet",
            "risk": args.risk,
            "market": args.market,
            "eur_rate": args.eur_rate,
            "liquidity": liquidity_path,
        },
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
