"""Download US Compustat North America exports from WRDS."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from us_market import DEFAULT_WRDS_US_DIR, download_wrds_compustat_us  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_WRDS_US_DIR)
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2026-07-08")
    parser.add_argument("--schema", default="comp")
    parser.add_argument("--username", help="Optional WRDS username.")
    parser.add_argument("--skip-annual", action="store_true")
    parser.add_argument("--skip-monthly", action="store_true")
    parser.add_argument(
        "--all-exchanges",
        action="store_true",
        help="Do not restrict monthly security rows to NYSE/AMEX/Nasdaq Compustat exchange codes.",
    )
    args = parser.parse_args()

    manifest = download_wrds_compustat_us(
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        schema=args.schema,
        username=args.username,
        skip_annual=args.skip_annual,
        skip_monthly=args.skip_monthly,
        primary_exchange_only=not args.all_exchanges,
    )
    for label, output in manifest["outputs"].items():
        print(f"{label}: {output['rows']:,} rows -> {output['path']}")
    print(f"manifest -> {args.output_dir / 'wrds_compustat_us_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
