"""Run AIPM scaling, attention, implementability and principal-portfolio tests."""
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
    write_post_analysis_outputs,
)
from asset_pricing_depth import load_eur_short_rate  # noqa: E402


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "aipm_full_transformer_compustat_cap500_seed3"
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
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "aipm_post_analysis"
)


def parse_comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def default_aipm_dirs() -> list[Path]:
    return sorted(
        path
        for path in (PROJECT_ROOT / "results" / "asset_pricing_ml").glob(
            "aipm_full_transformer*"
        )
        if (path / "aipm_full_summary.csv").exists()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aipm-dirs", nargs="*", type=Path)
    parser.add_argument("--aum-eur", default="10000000,100000000,500000000")
    parser.add_argument("--fallback-half-spread-bps", type=float, default=25.0)
    parser.add_argument("--impact-coefficient", type=float, default=0.10)
    parser.add_argument("--null-attention-draws", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--principal-components", default="1,3,5")
    parser.add_argument("--skip-principal-portfolios", action="store_true")
    args = parser.parse_args()

    required = [args.run_dir, args.panel]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}")

    config = AIPMPostAnalysisConfig(
        aum_eur=parse_comma_floats(args.aum_eur),
        fallback_half_spread_bps=args.fallback_half_spread_bps,
        impact_coefficient=args.impact_coefficient,
        null_attention_draws=args.null_attention_draws,
        random_state=args.random_state,
        hac_lags=args.hac_lags,
    )
    aipm_dirs = args.aipm_dirs if args.aipm_dirs else default_aipm_dirs()
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = write_post_analysis_outputs(
        args.output_dir,
        run_dir=args.run_dir,
        aipm_dirs=aipm_dirs,
        panel_path=args.panel,
        risk_path=args.risk if args.risk.exists() else None,
        liquidity_path=args.liquidity if args.liquidity.exists() else None,
        risk_free=risk_free,
        principal_components=parse_comma_ints(args.principal_components),
        config=config,
        run_principal_portfolios=not args.skip_principal_portfolios,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
