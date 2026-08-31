"""Missingness negative control for the analyst-estimates data-depth effect.

The estimates layer could in principle be proxied by the bare fact of coverage:
a stock-month with many populated analyst fields differs from one with few, and
a model may be reading that composition signal rather than the analyst values.
This control interposes a third cell between the Test B pair:

    compustat_only          53 Compustat/baseline features
    coverage_only           the same + the estimates_feature_count rank
                            (how many analyst fields are populated -- no values)
    compustat_plus_estimates the same + the 11 actual analyst features

on identical stock-months, and splits the data-depth effect into

    missingness_increment   = IC(coverage_only) - IC(compustat_only)
    analyst_value_increment = IC(estimates)     - IC(coverage_only)

If the analyst layer is real information, the missingness increment should be
approximately zero and the value increment should carry essentially the whole
data-depth effect. If the missingness increment is large, the "analyst" effect
is a coverage-composition artefact.
"""
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

from estimates_identification import hac_mean, holm_within, monthly_ic  # noqa: E402

RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_COMPUSTAT_DIR = RESULTS_ROOT / "test_b_datadepth_compustat_enriched_refresh_20260816"
DEFAULT_COVERAGE_DIR = RESULTS_ROOT / "estimates_negcontrol_coverage_only_20260816"
DEFAULT_ESTIMATES_DIR = RESULTS_ROOT / "test_b_datadepth_estimates_enriched_refresh_20260816"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "estimates_missingness_control_20260816"

PREDICTION_COLUMNS = ["date", "ric", "base_model", "prediction", "target_return_1m"]
QUANTITIES = [
    ("missingness_increment", "coverage_only", "compustat_only"),
    ("analyst_value_increment", "compustat_plus_estimates", "coverage_only"),
    ("data_depth_effect", "compustat_plus_estimates", "compustat_only"),
]


def load_predictions(directory: Path, models: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(directory / "predictions.parquet", columns=PREDICTION_COLUMNS)
    frame = frame[frame["base_model"].isin(models)]
    return frame.dropna(subset=["prediction", "target_return_1m"])


def check_identical_cells(cells: dict[str, pd.DataFrame]) -> dict[str, object]:
    keys = {
        name: pd.MultiIndex.from_frame(frame[["ric", "date"]].drop_duplicates())
        for name, frame in cells.items()
    }
    counts = {name: int(len(index)) for name, index in keys.items()}
    shared = None
    for index in keys.values():
        shared = index if shared is None else shared.intersection(index)
    check = {
        "stock_months": counts,
        "shared_stock_months": int(len(shared)),
        "identical": len(set(counts.values())) == 1
        and len(shared) == next(iter(counts.values())),
    }
    if not check["identical"]:
        raise SystemExit(
            f"Control cells are not sample-matched: {check}. All three cells "
            "must be run on identical stock-months."
        )
    return check


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compustat-dir", type=Path, default=DEFAULT_COMPUSTAT_DIR)
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--estimates-dir", type=Path, default=DEFAULT_ESTIMATES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=["ridge", "hist_gbm", "mlp"])
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells = {
        "compustat_only": load_predictions(args.compustat_dir, args.models),
        "coverage_only": load_predictions(args.coverage_dir, args.models),
        "compustat_plus_estimates": load_predictions(args.estimates_dir, args.models),
    }
    sample_check = check_identical_cells(cells)

    ics = pd.concat(
        [monthly_ic(frame).assign(cell=name) for name, frame in cells.items()],
        ignore_index=True,
    )
    wide = ics.pivot_table(index="date", columns=["cell", "base_model"], values="ic")

    records = []
    for model in args.models:
        for label, minuend, subtrahend in QUANTITIES:
            if (minuend, model) not in wide.columns:
                continue
            if (subtrahend, model) not in wide.columns:
                continue
            difference = wide[(minuend, model)] - wide[(subtrahend, model)]
            records.append(
                {
                    "model": model,
                    "minuend_cell": minuend,
                    "subtrahend_cell": subtrahend,
                    "mean_ic_minuend": float(wide[(minuend, model)].mean()),
                    "mean_ic_subtrahend": float(wide[(subtrahend, model)].mean()),
                    **hac_mean(difference, args.hac_lags, label),
                }
            )
    table = holm_within(pd.DataFrame(records), ["quantity"])

    ics.to_csv(args.output_dir / "missingness_control_monthly_ics.csv", index=False)
    table.to_csv(args.output_dir / "missingness_control_tests.csv", index=False)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "cells": {
            "compustat_only": str(args.compustat_dir),
            "coverage_only": str(args.coverage_dir),
            "compustat_plus_estimates": str(args.estimates_dir),
        },
        "sample_check": sample_check,
        "hac_lags": args.hac_lags,
        "interpretation": (
            "Negative control: the missingness increment (coverage counts with "
            "no analyst values) should be ~0 and the analyst-value increment "
            "should carry essentially the whole data-depth effect. A large "
            "missingness increment would mean the estimates effect is a "
            "coverage-composition artefact."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(sample_check, indent=2))
    print(
        table[
            [
                "quantity",
                "model",
                "estimate",
                "t_stat",
                "p_value",
                "p_value_holm",
                "minimum_detectable_effect",
            ]
        ].to_string(index=False)
    )
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
