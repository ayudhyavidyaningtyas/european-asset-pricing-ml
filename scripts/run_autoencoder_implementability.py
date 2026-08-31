"""Cost the conditional-autoencoder stock-level SDF weights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aipm_post_analysis import (  # noqa: E402
    AIPMPostAnalysisConfig,
    build_execution_input_panel,
    simulate_weight_implementability,
    summarize_implementability,
)


DEFAULT_AUTOENCODER_DIR = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "autoencoder_compustat_k5_main"
)
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_RISK = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "depth_analysis"
    / "rolling_risk_estimates.parquet"
)
DEFAULT_LIQUIDITY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "supplemental"
    / "liquidity_monthly_full_period"
)


def parse_comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder-dir", type=Path, default=DEFAULT_AUTOENCODER_DIR)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--aum-eur", default="10000000,100000000,500000000")
    parser.add_argument("--fallback-half-spread-bps", type=float, default=25.0)
    parser.add_argument("--impact-coefficient", type=float, default=0.10)
    args = parser.parse_args()

    weight_path = args.autoencoder_dir / "autoencoder_weights.parquet"
    if not weight_path.exists():
        raise SystemExit(f"Missing autoencoder weights: {weight_path}")
    output_dir = args.output_dir or args.autoencoder_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AIPMPostAnalysisConfig(
        aum_eur=parse_comma_floats(args.aum_eur),
        fallback_half_spread_bps=args.fallback_half_spread_bps,
        impact_coefficient=args.impact_coefficient,
    )
    execution_inputs = build_execution_input_panel(
        args.panel,
        args.risk if args.risk.exists() else None,
        args.liquidity if args.liquidity.exists() else None,
        config,
    )
    import pandas as pd

    weights = pd.read_parquet(weight_path)
    monthly = simulate_weight_implementability(weights, execution_inputs, config)
    summary = summarize_implementability(monthly)
    monthly.to_csv(output_dir / "autoencoder_implementability_monthly.csv", index=False)
    summary.to_csv(output_dir / "autoencoder_implementability_summary.csv", index=False)
    manifest = {
        "autoencoder_dir": str(args.autoencoder_dir),
        "panel": str(args.panel),
        "risk": str(args.risk if args.risk.exists() else None),
        "liquidity": str(args.liquidity if args.liquidity.exists() else None),
        "config": {
            "aum_eur": list(config.aum_eur),
            "fallback_half_spread_bps": config.fallback_half_spread_bps,
            "impact_coefficient": config.impact_coefficient,
        },
        "rows": {
            "weights": int(len(weights)),
            "monthly": int(len(monthly)),
            "summary": int(len(summary)),
        },
    }
    (output_dir / "autoencoder_implementability_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
