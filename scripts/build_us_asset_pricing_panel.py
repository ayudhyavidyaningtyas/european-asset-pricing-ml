"""Build the cleaned US monthly feature panel from Refinitiv/LSEG exports."""
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

from asset_pricing import (  # noqa: E402
    MONTHLY_COLUMNS,
    PanelConfig,
    build_feature_panel,
    feature_dictionary_frame,
    load_optional_reference_supplement,
)


DEFAULT_EXPORT_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_us_exports"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "asset_pricing"


def write_us_panel_outputs(
    output_dir: Path,
    clean_universe: pd.DataFrame,
    fundamental_features: pd.DataFrame,
    panel: pd.DataFrame,
    audit: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_universe.to_csv(output_dir / "clean_universe_us.csv", index=False)
    fundamental_features.to_parquet(
        output_dir / "fundamental_features_us_lagged.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    panel.to_parquet(
        output_dir / "monthly_feature_panel_us.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    feature_dictionary_frame().to_csv(output_dir / "feature_dictionary_us.csv", index=False)
    (output_dir / "cleaning_audit_us.json").write_text(json.dumps(audit, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--universe-file", type=Path)
    parser.add_argument("--monthly-file", type=Path)
    parser.add_argument("--fundamentals-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--as-of", default="2026-07-08")
    parser.add_argument("--accounting-lag-months", type=int, default=6)
    parser.add_argument("--min-returns", type=int, default=24)
    parser.add_argument("--min-market-cap-observations", type=int, default=12)
    args = parser.parse_args()

    export_dir = args.export_dir
    universe_file = args.universe_file or export_dir / "us_equity_universe.csv"
    monthly_file = args.monthly_file or export_dir / "refinitiv_monthly_panel_tidy.csv"
    fundamentals_file = args.fundamentals_file or export_dir / "refinitiv_fundamentals_annual.csv"

    config = PanelConfig(
        as_of=args.as_of,
        accounting_lag_months=args.accounting_lag_months,
        min_return_observations=args.min_returns,
        min_market_cap_observations=args.min_market_cap_observations,
    )
    universe = pd.read_csv(universe_file, low_memory=False)
    monthly = pd.read_csv(
        monthly_file,
        usecols=MONTHLY_COLUMNS,
        parse_dates=["date"],
        low_memory=False,
    )
    fundamentals = pd.read_csv(fundamentals_file, low_memory=False)
    clean_universe, fundamental_features, panel, audit = build_feature_panel(
        universe,
        monthly,
        fundamentals,
        config,
        load_optional_reference_supplement(export_dir),
    )
    clean_universe["market_region"] = "US"
    fundamental_features["market_region"] = "US"
    panel["market_region"] = "US"
    audit["market_region"] = "US"
    audit["source"] = {
        "refinitiv_export_dir": str(export_dir),
        "universe_file": str(universe_file),
        "monthly_file": str(monthly_file),
        "fundamentals_file": str(fundamentals_file),
    }
    write_us_panel_outputs(args.output_dir, clean_universe, fundamental_features, panel, audit)

    print(f"eligible securities: {audit['universe']['eligible_rics']:,}")
    print(f"model rows: {audit['panel']['model_eligible_rows']:,}")
    print(f"unique model securities: {audit['panel']['unique_model_securities']:,}")
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
