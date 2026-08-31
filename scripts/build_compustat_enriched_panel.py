"""Build the Compustat-enriched European monthly feature panel."""
from __future__ import annotations

import argparse
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
    load_compustat_exports,
    write_compustat_outputs,
)


DEFAULT_BASE_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel.parquet"
)
DEFAULT_COMPUSTAT_DIR = (
    PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "compustat_exports"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "asset_pricing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-panel", type=Path, default=DEFAULT_BASE_PANEL)
    parser.add_argument("--compustat-dir", type=Path, default=DEFAULT_COMPUSTAT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--as-of", default="2026-07-08")
    parser.add_argument("--accounting-lag-months", type=int, default=6)
    args = parser.parse_args()

    config = PanelConfig(
        as_of=args.as_of,
        accounting_lag_months=args.accounting_lag_months,
    )
    base_panel = pd.read_parquet(args.base_panel)
    annual, monthly = load_compustat_exports(args.compustat_dir)
    annual_features, monthly_features, panel, audit = build_compustat_enriched_panel(
        base_panel,
        annual,
        monthly,
        config,
    )
    write_compustat_outputs(
        args.output_dir,
        annual_features,
        monthly_features,
        panel,
        audit,
    )

    print(f"rows: {audit['panel']['rows']:,}")
    print(f"unique RICs: {audit['panel']['unique_rics']:,}")
    print(
        "Compustat annual rows: "
        f"{audit['panel']['rows_with_compustat_annual']:,}"
    )
    print(
        "Compustat monthly rows: "
        f"{audit['panel']['rows_with_compustat_monthly']:,}"
    )
    print(
        "mean deep feature count: "
        f"{audit['panel']['mean_deep_feature_count']:.1f}"
    )
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
