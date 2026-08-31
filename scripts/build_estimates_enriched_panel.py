"""Build a Refinitiv analyst-estimates-enriched monthly feature panel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from estimates_features import (  # noqa: E402
    EstimatesPanelConfig,
    build_estimates_enriched_panel,
    load_estimates_export,
    write_estimates_outputs,
)


DEFAULT_BASE_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_ESTIMATES_EXPORT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "refinitiv_analyst_estimates_monthly.csv.gz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "asset_pricing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-panel", type=Path, default=DEFAULT_BASE_PANEL)
    parser.add_argument("--estimates-export", type=Path, default=DEFAULT_ESTIMATES_EXPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--panel-filename",
        default="monthly_feature_panel_estimates.parquet",
        help="Name for the enriched output panel inside --output-dir.",
    )
    parser.add_argument(
        "--snapshot-filename",
        default="refinitiv_estimates_snapshot_features.parquet",
        help="Name for collapsed snapshot features inside --output-dir.",
    )
    parser.add_argument(
        "--dictionary-filename",
        default="refinitiv_estimates_feature_dictionary.csv",
        help="Name for the estimates feature dictionary inside --output-dir.",
    )
    parser.add_argument(
        "--audit-filename",
        default="refinitiv_estimates_enrichment_audit.json",
        help="Name for the estimates enrichment audit JSON inside --output-dir.",
    )
    parser.add_argument(
        "--strict-identifier-match",
        action="store_true",
        help="Reject estimate rows whose ISIN disagrees with the panel ISIN.",
    )
    parser.add_argument(
        "--estimate-signal-lag-months",
        type=int,
        default=0,
        help="Shift derived analyst-estimate features by this many months before ranking.",
    )
    parser.add_argument(
        "--filter-extreme-estimates",
        action="store_true",
        help="Null economically extreme estimate levels, revisions and dispersions before ranking.",
    )
    parser.add_argument("--max-abs-eps-yield", type=float, default=1.0)
    parser.add_argument("--min-price-target-upside", type=float, default=-0.95)
    parser.add_argument("--max-price-target-upside", type=float, default=5.0)
    parser.add_argument("--max-abs-revision", type=float, default=5.0)
    parser.add_argument("--max-dispersion", type=float, default=5.0)
    args = parser.parse_args()

    base_panel = pd.read_parquet(args.base_panel)
    estimates = load_estimates_export(args.estimates_export)
    config = EstimatesPanelConfig(
        strict_identifier_match=args.strict_identifier_match,
        signal_lag_months=args.estimate_signal_lag_months,
        filter_extreme_estimates=args.filter_extreme_estimates,
        max_abs_eps_yield=args.max_abs_eps_yield,
        min_price_target_upside=args.min_price_target_upside,
        max_price_target_upside=args.max_price_target_upside,
        max_abs_revision=args.max_abs_revision,
        max_dispersion=args.max_dispersion,
    )
    snapshot_features, panel, audit = build_estimates_enriched_panel(
        base_panel,
        estimates,
        config=config,
    )
    write_estimates_outputs(
        args.output_dir,
        snapshot_features,
        panel,
        audit,
        panel_filename=args.panel_filename,
        snapshot_filename=args.snapshot_filename,
        dictionary_filename=args.dictionary_filename,
        audit_filename=args.audit_filename,
    )

    print(f"rows: {audit['panel']['rows']:,}")
    print(f"unique RICs: {audit['panel']['unique_rics']:,}")
    print(f"rows with estimates: {audit['panel']['rows_with_estimates']:,}")
    print(
        "eligible rows with estimates: "
        f"{audit['panel']['eligible_rows_with_estimates']:,}"
    )
    print(
        "unique RICs with estimates: "
        f"{audit['panel']['unique_rics_with_estimates']:,}"
    )
    print(
        "mean estimates feature count, full panel: "
        f"{audit['panel']['mean_estimates_feature_count']:.1f}"
    )
    print(
        "mean estimates feature count, matched rows: "
        f"{audit['panel']['mean_estimates_feature_count_with_estimates']:.1f}"
    )
    print(
        "rows with any estimates feature: "
        f"{audit['panel']['rows_with_any_estimates_feature']:,}"
    )
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
