"""Compare European and US asset-pricing ML output folders."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from market_comparison import (  # noqa: E402
    DEFAULT_COMPARISON_OUTPUT,
    DEFAULT_EUROPE_OUTPUT,
    DEFAULT_US_OUTPUT,
    PRIMARY_FILTER,
    write_market_comparison_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--europe-output-dir", type=Path, default=DEFAULT_EUROPE_OUTPUT)
    parser.add_argument("--us-output-dir", type=Path, default=DEFAULT_US_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COMPARISON_OUTPUT)
    parser.add_argument("--weighting", default=PRIMARY_FILTER["weighting"])
    parser.add_argument("--universe-variant", default=PRIMARY_FILTER["universe_variant"])
    parser.add_argument("--portfolio", default=PRIMARY_FILTER["portfolio"])
    parser.add_argument("--cost-bps", type=int, default=PRIMARY_FILTER["cost_bps"])
    args = parser.parse_args()

    manifest = write_market_comparison_outputs(
        {
            "Europe": args.europe_output_dir,
            "US": args.us_output_dir,
        },
        args.output_dir,
        baseline_market="Europe",
        comparison_market="US",
        filters={
            "weighting": args.weighting,
            "universe_variant": args.universe_variant,
            "portfolio": args.portfolio,
            "cost_bps": args.cost_bps,
        },
    )
    print(f"side-by-side rows: {manifest['rows']['side_by_side_model_summary']:,}")
    print(f"common-month correlations: {manifest['rows']['monthly_return_correlations']:,}")
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
