"""Run currency-aligned external European factor regressions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_external_factors import (  # noqa: E402
    external_factor_spanning,
    load_external_europe_factors,
    load_monthly_eurusd_return,
)


DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "revised_full_eur_delisting"
)
DEFAULT_FRENCH_DIR = (
    PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "french"
)
DEFAULT_FX = PROJECT_ROOT / "data" / "raw" / "fred_DEXUSEU.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--french-dir", type=Path, default=DEFAULT_FRENCH_DIR)
    parser.add_argument("--fx-file", type=Path, default=DEFAULT_FX)
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    monthly_path = args.baseline_dir / "monthly_portfolios.csv"
    required = [
        monthly_path,
        args.french_dir / "Europe_5_Factors.csv",
        args.french_dir / "Europe_MOM_Factor.csv",
        args.fx_file,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}")

    monthly = pd.read_csv(
        monthly_path,
        parse_dates=["signal_date", "return_date"],
    )
    factors = load_external_europe_factors(
        args.french_dir / "Europe_5_Factors.csv",
        args.french_dir / "Europe_MOM_Factor.csv",
    )
    fx = load_monthly_eurusd_return(args.fx_file)
    results = external_factor_spanning(
        monthly,
        factors,
        fx,
        cost_bps=args.cost_bps,
        hac_lags=args.hac_lags,
    )
    output_path = args.baseline_dir / "external_factor_spanning_usd.csv"
    results.to_csv(output_path, index=False)
    manifest = {
        "monthly_portfolios": str(monthly_path),
        "five_factors": str(args.french_dir / "Europe_5_Factors.csv"),
        "momentum_factor": str(
            args.french_dir / "Europe_MOM_Factor.csv"
        ),
        "fx": str(args.fx_file),
        "fx_convention": "USD per EUR; monthly last observation",
        "portfolio_currency": "USD after conversion from EUR",
        "factor_currency": "USD",
        "cost_bps": args.cost_bps,
        "hac_lags": args.hac_lags,
        "rows": int(len(results)),
    }
    (args.baseline_dir / "external_factor_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"rows: {len(results)}")
    print(f"outputs -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
