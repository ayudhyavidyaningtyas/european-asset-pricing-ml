"""Build the cleaned compact-GKX European monthly feature panel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing import (  # noqa: E402
    PanelConfig,
    build_feature_panel,
    load_optional_reference_supplement,
    load_source_data,
    write_panel_outputs,
)


DEFAULT_EXPORT_DIR = (
    PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_exports"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "asset_pricing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--as-of", default="2026-07-08")
    parser.add_argument("--accounting-lag-months", type=int, default=6)
    parser.add_argument("--min-returns", type=int, default=24)
    parser.add_argument("--min-market-cap-observations", type=int, default=12)
    parser.add_argument("--min-features", type=int, default=8)
    parser.add_argument("--microcap-quantile", type=float, default=0.05)
    parser.add_argument(
        "--ignore-reference-supplement",
        action="store_true",
        help="Use only the original primary-equity screener metadata.",
    )
    args = parser.parse_args()

    config = PanelConfig(
        as_of=args.as_of,
        accounting_lag_months=args.accounting_lag_months,
        min_return_observations=args.min_returns,
        min_market_cap_observations=args.min_market_cap_observations,
        min_features=args.min_features,
        microcap_quantile=args.microcap_quantile,
    )
    universe, monthly, fundamentals = load_source_data(args.export_dir)
    reference = (
        None
        if args.ignore_reference_supplement
        else load_optional_reference_supplement(args.export_dir)
    )
    clean_universe, fundamental_features, panel, audit = build_feature_panel(
        universe,
        monthly,
        fundamentals,
        config,
        reference,
    )
    write_panel_outputs(
        args.output_dir,
        clean_universe,
        fundamental_features,
        panel,
        audit,
    )

    print(f"eligible securities: {audit['universe']['eligible_rics']:,}")
    print(f"model rows: {audit['panel']['model_eligible_rows']:,}")
    print(f"model period: {audit['panel']['first_model_month']} to {audit['panel']['last_model_month']}")
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
