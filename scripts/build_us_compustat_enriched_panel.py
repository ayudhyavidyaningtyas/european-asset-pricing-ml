"""Build the US Refinitiv + WRDS Compustat-enriched monthly feature panel."""
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

from asset_pricing import PanelConfig  # noqa: E402
from compustat_features import (  # noqa: E402
    build_compustat_enriched_panel,
    compustat_feature_dictionary_frame,
)
from us_market import DEFAULT_WRDS_US_DIR, load_wrds_compustat_us_exports  # noqa: E402


DEFAULT_BASE_PANEL = (
    PROJECT_ROOT / "data" / "processed" / "asset_pricing" / "monthly_feature_panel_us.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "asset_pricing"


def write_us_compustat_outputs(
    output_dir: Path,
    annual_features: pd.DataFrame,
    monthly_features: pd.DataFrame,
    panel: pd.DataFrame,
    audit: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annual_features.to_parquet(
        output_dir / "wrds_compustat_us_annual_features_lagged.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    monthly_features.to_parquet(
        output_dir / "wrds_compustat_us_monthly_features.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    panel.to_parquet(
        output_dir / "monthly_feature_panel_us_compustat.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    compustat_feature_dictionary_frame().to_csv(
        output_dir / "wrds_compustat_us_feature_dictionary.csv",
        index=False,
    )
    (output_dir / "wrds_compustat_us_enrichment_audit.json").write_text(
        json.dumps(audit, indent=2)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-panel", type=Path, default=DEFAULT_BASE_PANEL)
    parser.add_argument("--compustat-dir", type=Path, default=DEFAULT_WRDS_US_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--as-of", default="2026-07-08")
    parser.add_argument("--accounting-lag-months", type=int, default=6)
    args = parser.parse_args()

    config = PanelConfig(
        as_of=args.as_of,
        accounting_lag_months=args.accounting_lag_months,
    )
    base_panel = pd.read_parquet(args.base_panel)
    annual, monthly = load_wrds_compustat_us_exports(args.compustat_dir)
    annual_features, monthly_features, panel, audit = build_compustat_enriched_panel(
        base_panel,
        annual,
        monthly,
        config,
    )
    annual_features["market_region"] = "US"
    monthly_features["market_region"] = "US"
    panel["market_region"] = "US"
    audit["market_region"] = "US"
    audit["source"] = {
        "base_panel": str(args.base_panel),
        "wrds_compustat_dir": str(args.compustat_dir),
    }
    write_us_compustat_outputs(args.output_dir, annual_features, monthly_features, panel, audit)

    print(f"rows: {audit['panel']['rows']:,}")
    print(f"unique RICs: {audit['panel']['unique_rics']:,}")
    print(f"WRDS Compustat annual rows: {audit['panel']['rows_with_compustat_annual']:,}")
    print(f"WRDS Compustat monthly rows: {audit['panel']['rows_with_compustat_monthly']:,}")
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
