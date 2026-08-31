"""Build the US RIC universe used for analyst-estimates downloads."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_us_compustat.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_us_exports"
    / "us_estimates_model_universe_rics_only.csv"
)


def build_universe(panel_path: Path, output_path: Path) -> dict[str, object]:
    columns = ["ric"]
    available = pd.read_parquet(panel_path, columns=["ric", "target_return_1m"])
    eligible = available[available["target_return_1m"].notna()]
    universe = (
        eligible[columns]
        .dropna()
        .assign(ric=lambda frame: frame["ric"].astype(str).str.strip())
        .loc[lambda frame: frame["ric"].ne("")]
        .drop_duplicates()
        .sort_values("ric")
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_path, index=False)
    return {
        "panel": str(panel_path),
        "output": str(output_path),
        "panel_rows": int(len(available)),
        "eligible_rows": int(len(eligible)),
        "unique_rics": int(len(universe)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    audit = build_universe(args.panel, args.output)
    print(f"panel rows: {audit['panel_rows']:,}")
    print(f"eligible rows: {audit['eligible_rows']:,}")
    print(f"unique RICs: {audit['unique_rics']:,}")
    print(f"output -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
