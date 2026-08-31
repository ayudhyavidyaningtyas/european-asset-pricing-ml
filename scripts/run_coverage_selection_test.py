"""Coverage-selection test for the analyst-estimates data-depth effect.

Analyst coverage is not assigned at random: large, liquid, widely traded names
in particular countries and sectors are covered far more often than the rest of
the universe. The Test B data-depth effect is estimated on covered stock-months
only, so a sceptic can read it as "covered stocks are easier to predict" rather
than "analyst data carries information".

The diagnostic here separates the two. A monthly cross-sectional logit models
the probability that a stock-month is covered, given size, liquidity, turnover,
volatility, book-to-market, momentum, country and sector. Covered rows are then
reweighted by the inverse of that probability so the covered sample matches the
characteristic distribution of the full eligible universe, and the data-depth
effect is re-estimated on the reweighted sample. A companion cut reports the
effect within propensity strata, and the effect is re-estimated across a grid
of propensity floors so extreme-weight sensitivity is visible.

This is a selection-robustness diagnostic, not causal identification: the
propensity is estimated on observables and then treated as fixed in the HAC
test, so unobservable selection and first-stage estimation error remain outside
the inference. What the diagnostic rules out is the compositional reading --
that the measured effect only reflects which kinds of stocks analysts choose to
cover. Both cells always share identical stock-months, so this asks whether the
covered sample is representative, not whether the two cells are matched, which
they are by construction.

The universe is built without delisting candidates, matching the Test B refresh
cells, which were run with --skip-delisting-scenarios; the script verifies that
the covered universe equals the cells' stock-months exactly.
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

from asset_pricing_ml import COMPUSTAT_FEATURE_COLUMNS, load_model_panel  # noqa: E402
from estimates_identification import (  # noqa: E402
    COVERAGE_PROPENSITY_CATEGORICAL,
    COVERAGE_PROPENSITY_CONTINUOUS,
    categorical_balance,
    coverage_weights,
    fit_monthly_coverage_propensity,
    hac_mean,
    holm_within,
    monthly_ic,
    standardized_mean_differences,
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "strict_estimates_lag1"
    / "monthly_feature_panel_estimates_strict_lag1.parquet"
)
DEFAULT_COMPUSTAT_DIR = RESULTS_ROOT / "test_b_datadepth_compustat_enriched_refresh_20260816"
DEFAULT_ESTIMATES_DIR = RESULTS_ROOT / "test_b_datadepth_estimates_enriched_refresh_20260816"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "estimates_coverage_selection_20260816"

MODELS = ["ridge", "hist_gbm", "mlp", "dre"]
PREDICTION_COLUMNS = ["date", "ric", "base_model", "prediction", "target_return_1m"]


def load_predictions(directory: Path, models: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(directory / "predictions.parquet", columns=PREDICTION_COLUMNS)
    frame = frame[frame["base_model"].isin(models)]
    return frame.dropna(subset=["prediction", "target_return_1m"])


def check_cells_are_matched(
    compustat: pd.DataFrame, estimates: pd.DataFrame
) -> dict[str, object]:
    """Fail unless both cells cover exactly the same stock-months.

    This test asks whether the covered sample represents the universe, which is
    only a meaningful question once the two cells are matched to each other.
    """
    keys_compustat = pd.MultiIndex.from_frame(compustat[["ric", "date"]].drop_duplicates())
    keys_estimates = pd.MultiIndex.from_frame(estimates[["ric", "date"]].drop_duplicates())
    check = {
        "compustat_stock_months": int(len(keys_compustat)),
        "estimates_stock_months": int(len(keys_estimates)),
        "shared_stock_months": int(len(keys_compustat.intersection(keys_estimates))),
    }
    check["identical_cells"] = (
        check["compustat_stock_months"]
        == check["estimates_stock_months"]
        == check["shared_stock_months"]
    )
    if not check["identical_cells"]:
        raise SystemExit(
            f"Test B cells are not coverage-matched: {check}. "
            "Rerun the cells on identical stock-months before this test."
        )
    return check


def data_depth_table(
    compustat: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    weight_column: str | None,
    hac_lags: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HAC-tested data-depth effect per model, plus the monthly ICs behind it."""
    ic_compustat = monthly_ic(compustat, weight_column=weight_column).assign(
        cell="compustat_only"
    )
    ic_estimates = monthly_ic(estimates, weight_column=weight_column).assign(
        cell="compustat_plus_estimates"
    )
    ics = pd.concat([ic_compustat, ic_estimates], ignore_index=True)
    wide = ics.pivot_table(index="date", columns=["cell", "base_model"], values="ic")
    records = []
    for model in sorted({*compustat["base_model"].unique()}):
        if ("compustat_only", model) not in wide.columns:
            continue
        if ("compustat_plus_estimates", model) not in wide.columns:
            continue
        difference = (
            wide[("compustat_plus_estimates", model)] - wide[("compustat_only", model)]
        )
        records.append(
            {
                "weighting": label,
                "model": model,
                "mean_ic_compustat_only": float(
                    wide[("compustat_only", model)].mean()
                ),
                "mean_ic_with_estimates": float(
                    wide[("compustat_plus_estimates", model)].mean()
                ),
                **hac_mean(difference, hac_lags, "data_depth_effect"),
            }
        )
    return pd.DataFrame(records), ics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--compustat-dir", type=Path, default=DEFAULT_COMPUSTAT_DIR)
    parser.add_argument("--estimates-dir", type=Path, default=DEFAULT_ESTIMATES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--propensity-strata", type=int, default=5)
    parser.add_argument(
        "--min-propensity",
        type=float,
        default=0.01,
        help="Floor on the fitted coverage probability before inverting it.",
    )
    parser.add_argument(
        "--min-propensity-sensitivity",
        nargs="+",
        type=float,
        default=[0.02, 0.05],
        help="Additional propensity floors re-estimating the weighted effect.",
    )
    parser.add_argument(
        "--estimate-signal-lag-months",
        type=int,
        default=1,
        help="Lag guard applied when loading the universe, matching the Test B cells.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    compustat = load_predictions(args.compustat_dir, args.models)
    estimates = load_predictions(args.estimates_dir, args.models)
    sample_check = check_cells_are_matched(compustat, estimates)

    # The universe is every stock-month that could have been covered: the same
    # panel, filters and eligibility rules as the Test B cells, minus the
    # coverage requirement itself. Delisting candidates are excluded because the
    # cells were run with --skip-delisting-scenarios; loading the audit here
    # would add unlabelled delisting rows the cells never scored. Weights are
    # only used where the effect is measured, so the universe is restricted to
    # the out-of-sample months the predictions span; residual targets and
    # coverage are both computed within a month, so trimming the training years
    # changes nothing but the runtime.
    test_months = pd.Index(sorted(set(compustat["date"].unique())))
    universe = load_model_panel(
        args.panel,
        delisting_audit_path=None,
        feature_columns=COMPUSTAT_FEATURE_COLUMNS,
        sample_start_date=test_months.min(),
        sample_end_date=test_months.max(),
        require_estimates_feature=False,
        require_estimate_signal_lag_months=args.estimate_signal_lag_months,
        extra_columns=["estimates_feature_count"],
    )
    universe = universe[universe["date"].isin(test_months)].copy()
    universe["is_covered"] = pd.to_numeric(
        universe["estimates_feature_count"], errors="coerce"
    ).gt(0)

    # The covered universe must be exactly the stock-months the cells scored;
    # any gap means the propensity target and the effect sample have drifted.
    cell_keys = pd.MultiIndex.from_frame(compustat[["ric", "date"]].drop_duplicates())
    covered_keys = pd.MultiIndex.from_frame(
        universe.loc[universe["is_covered"], ["ric", "date"]]
    )
    universe_check = {
        "cell_stock_months": int(len(cell_keys)),
        "covered_universe_stock_months": int(len(covered_keys)),
        "shared": int(len(cell_keys.intersection(covered_keys))),
    }
    universe_check["identical"] = (
        universe_check["cell_stock_months"]
        == universe_check["covered_universe_stock_months"]
        == universe_check["shared"]
    )
    if not universe_check["identical"]:
        raise SystemExit(
            "Covered universe does not equal the Test B cells' stock-months "
            f"({universe_check}); the propensity target is inconsistent with "
            "the sample the effect is estimated on."
        )

    propensity, propensity_diagnostics = fit_monthly_coverage_propensity(
        universe,
        continuous_columns=COVERAGE_PROPENSITY_CONTINUOUS,
        categorical_columns=COVERAGE_PROPENSITY_CATEGORICAL,
    )
    universe["coverage_propensity"] = propensity
    universe["coverage_weight"] = coverage_weights(
        universe, min_propensity=args.min_propensity
    )

    covered = universe[universe["is_covered"]].copy()
    covered["propensity_stratum"] = (
        covered.groupby("date", observed=True)["coverage_propensity"]
        .transform(
            lambda values: pd.qcut(
                values.rank(method="first"),
                args.propensity_strata,
                labels=False,
                duplicates="drop",
            )
        )
        .astype("Int64")
    )

    weights = covered[
        ["date", "ric", "coverage_propensity", "coverage_weight", "propensity_stratum"]
    ]
    compustat = compustat.merge(weights, on=["date", "ric"], how="left", validate="many_to_one")
    estimates = estimates.merge(weights, on=["date", "ric"], how="left", validate="many_to_one")
    weight_coverage = {
        "prediction_rows": int(len(compustat)),
        "prediction_rows_with_weight": int(compustat["coverage_weight"].notna().sum()),
        "universe_rows": int(len(universe)),
        "universe_covered_rows": int(universe["is_covered"].sum()),
    }
    weight_coverage["unmatched_prediction_share"] = float(
        1.0
        - weight_coverage["prediction_rows_with_weight"]
        / max(weight_coverage["prediction_rows"], 1)
    )

    balance = pd.concat(
        [
            standardized_mean_differences(
                universe, COVERAGE_PROPENSITY_CONTINUOUS, weight_column=None
            ),
            standardized_mean_differences(
                universe, COVERAGE_PROPENSITY_CONTINUOUS, weight_column="coverage_weight"
            ),
        ],
        ignore_index=True,
    )
    categorical = pd.concat(
        [
            categorical_balance(
                universe, COVERAGE_PROPENSITY_CATEGORICAL, weight_column=None
            ),
            categorical_balance(
                universe, COVERAGE_PROPENSITY_CATEGORICAL, weight_column="coverage_weight"
            ),
        ],
        ignore_index=True,
    )

    weight_diagnostics = (
        covered.groupby("date", observed=True)
        .agg(
            covered_rows=("coverage_weight", "size"),
            mean_weight=("coverage_weight", "mean"),
            max_weight=("coverage_weight", "max"),
            weight_p99=("coverage_weight", lambda values: float(values.quantile(0.99))),
            effective_rows=(
                "coverage_weight",
                lambda values: float(values.sum() ** 2 / (values**2).sum()),
            ),
            min_propensity=("coverage_propensity", "min"),
            median_propensity=("coverage_propensity", "median"),
        )
        .reset_index()
    )
    weight_diagnostics["effective_sample_share"] = (
        weight_diagnostics["effective_rows"] / weight_diagnostics["covered_rows"]
    )

    unweighted, unweighted_ics = data_depth_table(
        compustat, estimates, weight_column=None, hac_lags=args.hac_lags, label="unweighted"
    )
    weighted, weighted_ics = data_depth_table(
        compustat.dropna(subset=["coverage_weight"]),
        estimates.dropna(subset=["coverage_weight"]),
        weight_column="coverage_weight",
        hac_lags=args.hac_lags,
        label="inverse_propensity",
    )
    # Extreme-weight sensitivity: re-derive the weights under stronger floors
    # and re-estimate the weighted effect. A result that only holds at the
    # loosest floor is being carried by a few near-uncovered stock-months.
    sensitivity_tables = []
    for floor in args.min_propensity_sensitivity:
        if floor == args.min_propensity:
            continue
        universe[f"weight_floor_{floor}"] = coverage_weights(
            universe, min_propensity=floor
        )
        floored = universe.loc[
            universe["is_covered"], ["date", "ric", f"weight_floor_{floor}"]
        ].rename(columns={f"weight_floor_{floor}": "sensitivity_weight"})
        compustat_floor = compustat.drop(
            columns=["sensitivity_weight"], errors="ignore"
        ).merge(floored, on=["date", "ric"], how="inner", validate="many_to_one")
        estimates_floor = estimates.drop(
            columns=["sensitivity_weight"], errors="ignore"
        ).merge(floored, on=["date", "ric"], how="inner", validate="many_to_one")
        table, _ = data_depth_table(
            compustat_floor,
            estimates_floor,
            weight_column="sensitivity_weight",
            hac_lags=args.hac_lags,
            label=f"inverse_propensity_floor_{floor}",
        )
        table["min_propensity_floor"] = floor
        sensitivity_tables.append(table)

    tests = pd.concat([unweighted, weighted], ignore_index=True)
    tests["min_propensity_floor"] = tests["weighting"].map(
        {"inverse_propensity": args.min_propensity}
    )
    tests = pd.concat([tests, *sensitivity_tables], ignore_index=True)
    tests = holm_within(tests, ["weighting"])

    stratum_records = []
    for stratum, group in compustat.dropna(subset=["propensity_stratum"]).groupby(
        "propensity_stratum", observed=True
    ):
        stratum_estimates = estimates[estimates["propensity_stratum"].eq(stratum)]
        table, _ = data_depth_table(
            group,
            stratum_estimates,
            weight_column=None,
            hac_lags=args.hac_lags,
            label=f"stratum_{int(stratum)}",
        )
        table.insert(0, "propensity_stratum", int(stratum))
        table["mean_propensity"] = float(group["coverage_propensity"].mean())
        stratum_records.append(table)
    strata = (
        holm_within(pd.concat(stratum_records, ignore_index=True), ["model"])
        if stratum_records
        else pd.DataFrame()
    )

    propensity_diagnostics.to_csv(
        args.output_dir / "coverage_propensity_monthly_diagnostics.csv", index=False
    )
    balance.to_csv(args.output_dir / "coverage_covariate_balance.csv", index=False)
    categorical.to_csv(
        args.output_dir / "coverage_categorical_balance.csv", index=False
    )
    weight_diagnostics.to_csv(
        args.output_dir / "coverage_weight_diagnostics.csv", index=False
    )
    pd.concat([unweighted_ics, weighted_ics], ignore_index=True).to_csv(
        args.output_dir / "coverage_monthly_ics.csv", index=False
    )
    tests.to_csv(args.output_dir / "coverage_selection_data_depth_tests.csv", index=False)
    strata.to_csv(
        args.output_dir / "coverage_propensity_stratum_data_depth.csv", index=False
    )

    fitted_months = int(propensity_diagnostics["model_fitted"].sum())
    manifest = {
        "script": str(Path(__file__).resolve()),
        "panel": str(args.panel),
        "cells": {
            "compustat_only": str(args.compustat_dir),
            "compustat_plus_estimates": str(args.estimates_dir),
        },
        "sample_check": sample_check,
        "universe_check": universe_check,
        "universe_construction": (
            "same panel, filters and eligibility as the Test B cells, without "
            "delisting candidates (the cells were run with "
            "--skip-delisting-scenarios), restricted to the out-of-sample months"
        ),
        "weight_coverage": weight_coverage,
        "propensity_model": {
            "continuous_covariates": COVERAGE_PROPENSITY_CONTINUOUS,
            "categorical_covariates": COVERAGE_PROPENSITY_CATEGORICAL,
            "specification": "monthly cross-sectional L2 logistic regression",
            "months": int(len(propensity_diagnostics)),
            "months_fitted": fitted_months,
            "mean_auc": float(propensity_diagnostics["auc"].mean()),
            "median_auc": float(propensity_diagnostics["auc"].median()),
            "mean_coverage_rate": float(propensity_diagnostics["coverage_rate"].mean()),
        },
        "weights": {
            "estimand": (
                "weighted monthly rank IC within the covered sample, weighted so "
                "the covered sample matches the characteristic mix of the full "
                "eligible universe. Ranks are still taken within the covered "
                "cross-section, which is the cross-section the models score."
            ),
            "min_propensity": args.min_propensity,
            "min_propensity_sensitivity": args.min_propensity_sensitivity,
            "normalisation": "mean weight one within each month",
            "mean_effective_sample_share": float(
                weight_diagnostics["effective_sample_share"].mean()
            ),
            "min_monthly_effective_sample_share": float(
                weight_diagnostics["effective_sample_share"].min()
            ),
            "max_weight": float(weight_diagnostics["max_weight"].max()),
            "mean_monthly_weight_p99": float(weight_diagnostics["weight_p99"].mean()),
        },
        "hac_lags": args.hac_lags,
        "propensity_strata": args.propensity_strata,
        "interpretation": (
            "Selection-robustness diagnostic, not causal identification: the "
            "propensity is estimated on observables and treated as fixed in the "
            "HAC test, so unobservable selection and first-stage estimation "
            "error remain outside the inference. A data-depth effect that keeps "
            "its sign and significance under inverse-propensity weighting -- "
            "and across the propensity-floor sensitivity grid -- is not "
            "explained by the observable characteristics of the stocks analysts "
            "choose to cover."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(json.dumps(manifest["propensity_model"], indent=2))
    print(tests.to_string(index=False))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
