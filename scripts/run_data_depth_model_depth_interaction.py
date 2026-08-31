"""Test B: does data depth substitute for model depth?

The 2x2 is {ridge, deeper model} x {Compustat only, Compustat + analyst layer},
estimated on identical stock-months. Three monthly series are formed:

    data_depth_t(m)   = IC_estimates,t(m)  - IC_compustat,t(m)
    depth_premium_t   = IC_cell,t(m)       - IC_cell,t(ridge)
    interaction_t(m)  = depth_premium_t(estimates, m) - depth_premium_t(compustat, m)

A negative interaction means richer data substitutes for model depth -- the
advantage of the flexible model shrinks once the linear model is given better
inputs. A positive interaction means they are complements.

The script refuses to run unless both cells cover exactly the same stock-months,
because the whole test is only interpretable on a common sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_SHALLOW_DIR = RESULTS_ROOT / "test_b_datadepth_compustat_enriched"
DEFAULT_DEEP_DIR = RESULTS_ROOT / "test_b_datadepth_estimates_enriched"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "data_depth_model_depth_interaction"

BASELINE = "ridge"
DEEPER_MODELS = ["hist_gbm", "mlp", "dre"]
MDE_MULTIPLIER = 1.959963985 + 0.841621234


def load_predictions(directory: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        directory / "predictions.parquet",
        columns=["date", "ric", "base_model", "prediction", "target_return_1m"],
    )
    return frame.dropna(subset=["prediction", "target_return_1m"])


def monthly_ic(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["base_model", "date"]
    grouped = frame.groupby(keys, observed=True)
    working = frame.assign(
        x=grouped["prediction"].rank(), y=grouped["target_return_1m"].rank()
    )
    working["xy"] = working["x"] * working["y"]
    aggregated = working.groupby(keys, observed=True).agg(
        names=("x", "size"),
        mean_x=("x", "mean"),
        mean_y=("y", "mean"),
        mean_xy=("xy", "mean"),
        var_x=("x", "var"),
        var_y=("y", "var"),
    )
    scale = (aggregated["names"] - 1.0) / aggregated["names"]
    covariance = aggregated["mean_xy"] - aggregated["mean_x"] * aggregated["mean_y"]
    denominator = np.sqrt(aggregated["var_x"] * scale * aggregated["var_y"] * scale)
    aggregated["ic"] = covariance / denominator.replace(0.0, np.nan)
    return aggregated.reset_index()[["base_model", "date", "names", "ic"]]


def _hac_mean(series: pd.Series, hac_lags: int, label: str) -> dict[str, object]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 24:
        return {"quantity": label, "months": int(len(clean))}
    fit = sm.OLS(clean, np.ones(len(clean))).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    estimate = float(fit.params.iloc[0])
    standard_error = float(fit.bse.iloc[0])
    return {
        "quantity": label,
        "months": int(len(clean)),
        "estimate": estimate,
        "standard_error": standard_error,
        "t_stat": float(fit.tvalues.iloc[0]),
        "p_value": float(fit.pvalues.iloc[0]),
        "ci_low": estimate - 1.959963985 * standard_error,
        "ci_high": estimate + 1.959963985 * standard_error,
        "minimum_detectable_effect": MDE_MULTIPLIER * standard_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compustat-dir", type=Path, default=DEFAULT_SHALLOW_DIR)
    parser.add_argument("--estimates-dir", type=Path, default=DEFAULT_DEEP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument(
        "--allow-sample-mismatch",
        action="store_true",
        help="Intersect stock-months instead of failing. Records the drop.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compustat = load_predictions(args.compustat_dir)
    estimates = load_predictions(args.estimates_dir)

    keys_compustat = set(map(tuple, compustat[["ric", "date"]].drop_duplicates().values))
    keys_estimates = set(map(tuple, estimates[["ric", "date"]].drop_duplicates().values))
    shared = keys_compustat & keys_estimates
    mismatch = {
        "compustat_stock_months": len(keys_compustat),
        "estimates_stock_months": len(keys_estimates),
        "shared_stock_months": len(shared),
        "identical": keys_compustat == keys_estimates,
    }
    if not mismatch["identical"]:
        if not args.allow_sample_mismatch:
            raise SystemExit(
                "Cells do not share identical stock-months "
                f"({mismatch}); rerun with --allow-sample-mismatch to intersect."
            )
        index = pd.MultiIndex.from_tuples(sorted(shared), names=["ric", "date"])
        compustat = compustat.set_index(["ric", "date"]).loc[index].reset_index()
        estimates = estimates.set_index(["ric", "date"]).loc[index].reset_index()

    ic_compustat = monthly_ic(compustat).assign(cell="compustat_only")
    ic_estimates = monthly_ic(estimates).assign(cell="compustat_plus_estimates")
    ics = pd.concat([ic_compustat, ic_estimates], ignore_index=True)
    ics.to_csv(args.output_dir / "monthly_ics_by_cell.csv", index=False)

    wide = ics.pivot_table(index="date", columns=["cell", "base_model"], values="ic")
    records: list[dict[str, object]] = []

    for model in [BASELINE, *DEEPER_MODELS]:
        if ("compustat_only", model) not in wide.columns:
            continue
        data_depth = (
            wide[("compustat_plus_estimates", model)] - wide[("compustat_only", model)]
        )
        records.append(
            {"model": model, **_hac_mean(data_depth, args.hac_lags, "data_depth_effect")}
        )

    for model in DEEPER_MODELS:
        if ("compustat_only", model) not in wide.columns:
            continue
        premium_compustat = (
            wide[("compustat_only", model)] - wide[("compustat_only", BASELINE)]
        )
        premium_estimates = (
            wide[("compustat_plus_estimates", model)]
            - wide[("compustat_plus_estimates", BASELINE)]
        )
        records.append(
            {
                "model": model,
                **_hac_mean(
                    premium_compustat, args.hac_lags, "depth_premium_compustat_only"
                ),
            }
        )
        records.append(
            {
                "model": model,
                **_hac_mean(
                    premium_estimates, args.hac_lags, "depth_premium_with_estimates"
                ),
            }
        )
        records.append(
            {
                "model": model,
                **_hac_mean(
                    premium_estimates - premium_compustat, args.hac_lags, "interaction"
                ),
            }
        )

    table = pd.DataFrame(records)
    testable = table["p_value"].notna()
    table.loc[testable, "p_value_holm"] = (
        table[testable]
        .groupby("quantity")["p_value"]
        .transform(lambda values: multipletests(values, method="holm")[1])
    )
    results_path = args.output_dir / "data_depth_model_depth_interaction.csv"
    table.to_csv(results_path, index=False)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "cells": {
            "compustat_only": str(args.compustat_dir),
            "compustat_plus_estimates": str(args.estimates_dir),
        },
        "sample_check": mismatch,
        "baseline_model": BASELINE,
        "deeper_models": DEEPER_MODELS,
        "hac_lags": args.hac_lags,
        "interpretation": (
            "interaction < 0 => data depth substitutes for model depth; "
            "interaction > 0 => complements"
        ),
        "outputs": {"results": str(results_path)},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(mismatch, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
