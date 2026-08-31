"""Paired HAC/Holm tests and summary table for the estimates-decomposition ablation.

Compares every ablation feature-set run against the coverage-matched
Compustat-only baseline (marginal value of adding one estimates group), and the
full estimates run against each leave-one-family-out run (marginal contribution
of that family on top of everything else). Reuses the repo's `_hac_mean_test`
convention (OLS on a constant, HAC standard errors) with Holm correction within
each test family, mirroring estimates_coverage_matched_paired_ic_tests.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_ml import _hac_mean_test  # noqa: E402
from estimates_features import (  # noqa: E402
    ESTIMATES_FEATURE_FAMILIES,
    ESTIMATES_INFORMATION_TYPES,
)
from stats import (  # noqa: E402
    bootstrap_sharpe_diff,
    clark_west_estimator_eligible,
    jobson_korkie_memmel,
    clark_west_test,
    sharpe_ratio,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_BASELINE = RESULTS_DIR / "estimates_covered_only_compustat_features"
DEFAULT_FULL = RESULTS_DIR / "estimates_covered_only_estimates_features"
DEFAULT_OUTPUT = RESULTS_DIR / "estimates_family_ablation"

# The analyst layer is decomposed two ways -- by source family (EPS, revenue,
# price target) and by information type (levels, revisions, dispersion). Each
# grouping partitions the modelled features exactly, so "X_only" and "ex_X"
# together attribute the whole layer under either reading. The two designs
# answer different questions and are not interchangeable, because the analyst
# signals are correlated across groups: "X_only" versus the Compustat baseline
# measures the standalone content of group X, while full versus "ex_X" measures
# the marginal contribution of X conditional on every other analyst group. A
# group can be standalone-informative yet marginally redundant.
ABLATION_GROUPS = [*ESTIMATES_FEATURE_FAMILIES, *ESTIMATES_INFORMATION_TYPES]
ABLATION_VARIANTS = [
    *(f"{group}_only" for group in ABLATION_GROUPS),
    *(f"ex_{group}" for group in ABLATION_GROUPS),
]


def monthly_ic_series(run_dir: Path, model: str) -> pd.Series:
    predictions = pd.read_parquet(
        run_dir / "predictions.parquet",
        columns=["date", "prediction", "target_return_rank", "model", "target_mode"],
    )
    subset = predictions[
        predictions["model"].eq(model) & predictions["target_mode"].eq("rank")
    ].dropna(subset=["prediction", "target_return_rank"])
    return subset.groupby("date").apply(
        lambda month: month["prediction"].corr(
            month["target_return_rank"], method="spearman"
        ),
        include_groups=False,
    )


def monthly_prediction_panel(run_dir: Path, model: str) -> pd.DataFrame:
    predictions = pd.read_parquet(
        run_dir / "predictions.parquet",
        columns=[
            "date",
            "ric",
            "prediction",
            "target_return_rank",
            "model",
            "target_mode",
        ],
    )
    subset = predictions[
        predictions["model"].eq(model) & predictions["target_mode"].eq("rank")
    ].dropna(subset=["prediction", "target_return_rank"])
    return subset[["date", "ric", "prediction", "target_return_rank"]].copy()


def monthly_net_long_short(
    run_dir: Path,
    model: str,
    weighting: str,
    universe_variant: str,
    cost_bps: int,
) -> pd.Series:
    monthly = pd.read_csv(
        run_dir / "monthly_portfolios.csv", parse_dates=["return_date"]
    )
    subset = monthly[
        monthly["model"].eq(model)
        & monthly["weighting"].eq(weighting)
        & monthly["universe_variant"].eq(universe_variant)
    ].sort_values("return_date")
    net = subset["gross_long_short_return"].sub(
        subset["long_short_turnover"].mul(cost_bps / 10_000.0)
    )
    return pd.Series(net.to_numpy(dtype=float), index=pd.DatetimeIndex(subset["return_date"]))


def run_feature_columns(run_dir: Path) -> set[str]:
    manifest_path = run_dir / "ml_manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = json.loads(manifest_path.read_text())
    return set(manifest.get("feature_columns", []))


def stock_month_match(variant: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Whether two runs scored exactly the same stock-months.

    A feature-set ablation is only a clean marginal-value read if the compared
    runs share a sample. Sample drift is easy to introduce by accident -- a
    different delisting-scenario switch is enough -- and the paired tests would
    otherwise quietly inner-join it away.
    """
    keys_variant = pd.MultiIndex.from_frame(variant[["date", "ric"]].drop_duplicates())
    keys_reference = pd.MultiIndex.from_frame(
        reference[["date", "ric"]].drop_duplicates()
    )
    shared = keys_variant.intersection(keys_reference)
    return {
        "variant_stock_months": int(len(keys_variant)),
        "reference_stock_months": int(len(keys_reference)),
        "shared_stock_months": int(len(shared)),
        "samples_identical": bool(
            len(keys_variant) == len(keys_reference) == len(shared)
        ),
    }


def nesting_metadata(
    variant_name: str,
    variant_dir: Path,
    reference_name: str,
    reference_dir: Path,
) -> dict:
    variant_features = run_feature_columns(variant_dir)
    reference_features = run_feature_columns(reference_dir)
    if not variant_features or not reference_features:
        return {
            "clark_west_is_nested": False,
            "clark_west_restricted_feature_set": reference_name,
            "clark_west_unrestricted_feature_set": variant_name,
            "clark_west_nesting_note": "missing_manifest_feature_columns",
        }
    extra_features = variant_features - reference_features
    missing_restricted_features = reference_features - variant_features
    is_nested = not missing_restricted_features and bool(extra_features)
    return {
        "clark_west_is_nested": is_nested,
        "clark_west_restricted_feature_set": reference_name,
        "clark_west_unrestricted_feature_set": variant_name,
        "clark_west_nesting_note": (
            f"restricted_features={len(reference_features)}; "
            f"unrestricted_features={len(variant_features)}; "
            f"extra_features={len(extra_features)}; "
            f"missing_restricted_features={len(missing_restricted_features)}"
        ),
    }


def paired_delta(series_a: pd.Series, series_b: pd.Series, lags: int) -> dict:
    aligned = pd.concat({"a": series_a, "b": series_b}, axis=1, join="inner").dropna()
    delta = aligned["a"].sub(aligned["b"])
    test = _hac_mean_test(delta, lags)
    return {
        "months": test["months"],
        "mean_a": float(aligned["a"].mean()),
        "mean_b": float(aligned["b"].mean()),
        "delta_mean": test["mean_difference"],
        "hac_t_stat": test["t_stat"],
        "hac_p_two_sided": test["p_value"],
    }


def paired_sharpe_delta(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    expected_block: float,
    n_boot: int,
    seed: int,
) -> dict:
    aligned = pd.concat({"a": series_a, "b": series_b}, axis=1, join="inner").dropna()
    if aligned.empty:
        return {
            "sharpe_variant": np.nan,
            "sharpe_reference": np.nan,
            "delta_sharpe": np.nan,
            "sharpe_bootstrap_ci_low": np.nan,
            "sharpe_bootstrap_ci_high": np.nan,
            "sharpe_bootstrap_p_two_sided": np.nan,
            "sharpe_bootstrap_ci_includes_zero": np.nan,
            "sharpe_bootstrap_expected_block": expected_block,
            "sharpe_bootstrap_n": n_boot,
            "sharpe_jkm_delta_monthly": np.nan,
            "sharpe_jkm_delta_annualized": np.nan,
            "sharpe_jkm_z": np.nan,
            "sharpe_jkm_p_two_sided": np.nan,
        }
    a = aligned["a"].to_numpy(dtype=float)
    b = aligned["b"].to_numpy(dtype=float)
    risk_free = np.zeros(len(aligned), dtype=float)
    bootstrap = bootstrap_sharpe_diff(
        a,
        b,
        risk_free,
        expected_block=expected_block,
        n_boot=n_boot,
        seed=seed,
    )
    memmel = jobson_korkie_memmel(a, b, risk_free)
    return {
        "sharpe_variant": sharpe_ratio(a, risk_free),
        "sharpe_reference": sharpe_ratio(b, risk_free),
        "delta_sharpe": bootstrap["delta_sharpe"],
        "sharpe_bootstrap_ci_low": bootstrap["ci_low"],
        "sharpe_bootstrap_ci_high": bootstrap["ci_high"],
        "sharpe_bootstrap_p_two_sided": bootstrap["p_two_sided"],
        "sharpe_bootstrap_ci_includes_zero": bool(
            bootstrap["ci_low"] <= 0.0 <= bootstrap["ci_high"]
        ),
        "sharpe_bootstrap_expected_block": bootstrap["expected_block"],
        "sharpe_bootstrap_n": bootstrap["n_boot"],
        "sharpe_jkm_delta_monthly": memmel["delta_sharpe_monthly"],
        "sharpe_jkm_delta_annualized": float(
            memmel["delta_sharpe_monthly"] * np.sqrt(12.0)
        ),
        "sharpe_jkm_z": memmel["z"],
        "sharpe_jkm_p_two_sided": memmel["p_two_sided"],
    }


def paired_loss_tests(
    variant: pd.DataFrame,
    reference: pd.DataFrame,
    lags: int,
    *,
    clark_west_is_nested: bool,
    clark_west_estimator_ok: bool = True,
) -> dict:
    empty_clark_west = {
        "clark_west_adjusted_mean_difference": np.nan,
        "clark_west_hac_t_stat": np.nan,
        "clark_west_p_one_sided": np.nan,
    }
    clark_west_applicable = clark_west_is_nested and clark_west_estimator_ok
    common = variant.rename(columns={"prediction": "prediction_variant"}).merge(
        reference.rename(
            columns={
                "prediction": "prediction_reference",
                "target_return_rank": "target_return_rank_reference",
            }
        ),
        on=["date", "ric"],
        how="inner",
        validate="one_to_one",
    )
    if common.empty:
        return {
            "months": 0,
            "mean_loss_variant": np.nan,
            "mean_loss_reference": np.nan,
            "loss_mean_difference": np.nan,
            "loss_hac_t_stat": np.nan,
            "loss_hac_p_two_sided": np.nan,
            **empty_clark_west,
        }
    target = common["target_return_rank"]
    error_variant = target - common["prediction_variant"]
    error_reference = target - common["prediction_reference"]
    loss_variant = error_variant.pow(2)
    loss_reference = error_reference.pow(2)
    loss_difference = loss_variant - loss_reference
    monthly = pd.DataFrame(
        {
            "date": common["date"].to_numpy(),
            "loss_variant": loss_variant.to_numpy(),
            "loss_reference": loss_reference.to_numpy(),
            "loss_difference": loss_difference.to_numpy(),
        }
    ).groupby("date").mean()
    loss_test = _hac_mean_test(monthly["loss_difference"], lags)
    if clark_west_applicable:
        clark_west = clark_west_test(
            common["target_return_rank"],
            common["prediction_reference"],
            common["prediction_variant"],
            dates=common["date"],
            maxlags=lags,
        )
        clark_west_result = {
            "clark_west_adjusted_mean_difference": clark_west["mean_difference"],
            "clark_west_hac_t_stat": clark_west["t_stat"],
            "clark_west_p_one_sided": clark_west["p_one_sided"],
        }
    else:
        clark_west_result = empty_clark_west
    return {
        "months": loss_test["months"],
        "mean_loss_variant": float(monthly["loss_variant"].mean()),
        "mean_loss_reference": float(monthly["loss_reference"].mean()),
        "loss_mean_difference": loss_test["mean_difference"],
        "loss_hac_t_stat": loss_test["t_stat"],
        "loss_hac_p_two_sided": loss_test["p_value"],
        **clark_west_result,
    }


def holm_column_within_test(
    frame: pd.DataFrame,
    p_column: str,
    output_column: str,
) -> pd.DataFrame:
    frame = frame.copy()
    frame[output_column] = np.nan
    if frame.empty or p_column not in frame:
        return frame
    for _, index in frame.groupby("test").groups.items():
        pvalues = frame.loc[index, p_column]
        valid = pvalues.notna()
        if valid.sum() == 0:
            continue
        adjusted = multipletests(pvalues[valid], method="holm")[1]
        frame.loc[pvalues[valid].index, output_column] = adjusted
    return frame


def holm_within_test(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["hac_p_holm"] = np.nan
    for _, index in frame.groupby("test").groups.items():
        pvalues = frame.loc[index, "hac_p_two_sided"]
        valid = pvalues.notna()
        if valid.sum() == 0:
            continue
        adjusted = multipletests(pvalues[valid], method="holm")[1]
        frame.loc[pvalues[valid].index, "hac_p_holm"] = adjusted
    return frame


def summary_rows(run_name: str, run_dir: Path, weighting: str, universe_variant: str, cost_bps: int) -> pd.DataFrame:
    summary = pd.read_csv(run_dir / "model_summary.csv")
    subset = summary[
        summary["target_mode"].eq("rank")
        & summary["weighting"].eq(weighting)
        & summary["universe_variant"].eq(universe_variant)
        & summary["cost_bps"].eq(cost_bps)
        & summary["portfolio"].isin(["long_short", "long_only_top_decile"])
    ].copy()
    subset.insert(0, "feature_set", run_name)
    keep = [
        "feature_set",
        "model",
        "portfolio",
        "annualized_mean_return",
        "annualized_excess_return",
        "sharpe",
        "average_monthly_turnover",
        "mean_monthly_spearman_ic",
        "ic_information_ratio",
        "observations",
        "months",
    ]
    return subset[keep]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--full-dir", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--variant-prefix",
        default="estimates_covered_ablation_",
        help="Run-directory prefix under results/asset_pricing_ml for ablation variants.",
    )
    parser.add_argument("--models", nargs="+", default=["ridge_rank", "hist_gbm_rank", "mlp_rank"])
    parser.add_argument("--weighting", default="equal")
    parser.add_argument("--universe-variant", default="ex_bottom_20pct")
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--sharpe-expected-block", type=float, default=6.0)
    parser.add_argument("--sharpe-bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--require-matched-samples",
        action="store_true",
        help="Fail instead of warning when a compared pair of runs scored different stock-months.",
    )
    args = parser.parse_args()

    runs: dict[str, Path] = {}
    if args.full_dir.exists():
        runs["estimates_enriched"] = args.full_dir
    for variant in ABLATION_VARIANTS:
        run_dir = RESULTS_DIR / f"{args.variant_prefix}{variant}"
        if run_dir.exists() and (run_dir / "predictions.parquet").exists():
            runs[f"estimates_{variant}"] = run_dir
        else:
            print(f"skipping {variant}: no completed run at {run_dir}")
    if not args.baseline_dir.exists():
        raise FileNotFoundError(f"Missing baseline run: {args.baseline_dir}")

    ic_cache: dict[tuple[str, str], pd.Series] = {}
    loss_cache: dict[tuple[str, str], pd.DataFrame] = {}
    net_cache: dict[tuple[str, str], pd.Series] = {}

    def cached_ic(name: str, run_dir: Path, model: str) -> pd.Series:
        key = (name, model)
        if key not in ic_cache:
            ic_cache[key] = monthly_ic_series(run_dir, model)
        return ic_cache[key]

    def cached_net(name: str, run_dir: Path, model: str) -> pd.Series:
        key = (name, model)
        if key not in net_cache:
            net_cache[key] = monthly_net_long_short(
                run_dir, model, args.weighting, args.universe_variant, args.cost_bps
            )
        return net_cache[key]

    def cached_loss(name: str, run_dir: Path, model: str) -> pd.DataFrame:
        key = (name, model)
        if key not in loss_cache:
            loss_cache[key] = monthly_prediction_panel(run_dir, model)
        return loss_cache[key]

    comparisons: list[tuple[str, str, Path, str, Path]] = []
    for name, run_dir in runs.items():
        comparisons.append(
            ("variant_minus_compustat", name, run_dir, "compustat_enriched", args.baseline_dir)
        )
    if "estimates_enriched" in runs:
        for variant in ABLATION_VARIANTS:
            name = f"estimates_{variant}"
            if variant.startswith("ex_") and name in runs:
                comparisons.append(
                    ("full_minus_leave_one_out", "estimates_enriched", runs["estimates_enriched"], name, runs[name])
                )

    ic_records = []
    loss_records = []
    portfolio_records = []
    sample_checks = []
    for test, name_a, dir_a, name_b, dir_b in comparisons:
        nesting = nesting_metadata(name_a, dir_a, name_b, dir_b)
        reference_model = args.models[0]
        match = stock_month_match(
            cached_loss(name_a, dir_a, reference_model),
            cached_loss(name_b, dir_b, reference_model),
        )
        sample_checks.append(
            {"test": test, "variant": name_a, "reference": name_b, **match}
        )
        if not match["samples_identical"]:
            message = (
                f"{name_a} vs {name_b}: scored stock-months differ "
                f"({match['variant_stock_months']:,} vs "
                f"{match['reference_stock_months']:,}, "
                f"{match['shared_stock_months']:,} shared)"
            )
            if args.require_matched_samples:
                raise SystemExit(f"Unmatched ablation samples -- {message}")
            print(f"WARNING: {message}")
        for model in args.models:
            ic = paired_delta(
                cached_ic(name_a, dir_a, model),
                cached_ic(name_b, dir_b, model),
                args.hac_lags,
            )
            ic_records.append(
                {
                    "test": f"monthly_ic_{test}",
                    "variant": name_a,
                    "reference": name_b,
                    "model": model,
                    "months": ic["months"],
                    "mean_ic_variant": ic["mean_a"],
                    "mean_ic_reference": ic["mean_b"],
                    "delta_mean_ic": ic["delta_mean"],
                    "hac_t_stat": ic["hac_t_stat"],
                    "hac_p_two_sided": ic["hac_p_two_sided"],
                }
            )
            estimator_ok, estimator_note = clark_west_estimator_eligible(model)
            loss = paired_loss_tests(
                cached_loss(name_a, dir_a, model),
                cached_loss(name_b, dir_b, model),
                args.hac_lags,
                clark_west_is_nested=nesting["clark_west_is_nested"],
                clark_west_estimator_ok=estimator_ok,
            )
            loss_records.append(
                {
                    "test": f"rank_loss_{test}",
                    "variant": name_a,
                    "reference": name_b,
                    "model": model,
                    "interpretation": (
                        "loss_mean_difference negative favors variant; "
                        "clark_west positive favors variant only when "
                        "clark_west_is_nested and "
                        "clark_west_estimator_eligible are both true"
                    ),
                    **nesting,
                    "clark_west_estimator_eligible": estimator_ok,
                    "clark_west_estimator_note": estimator_note,
                    **loss,
                }
            )
            net = paired_delta(
                cached_net(name_a, dir_a, model),
                cached_net(name_b, dir_b, model),
                args.hac_lags,
            )
            sharpe = paired_sharpe_delta(
                cached_net(name_a, dir_a, model),
                cached_net(name_b, dir_b, model),
                expected_block=args.sharpe_expected_block,
                n_boot=args.sharpe_bootstrap_repetitions,
                seed=args.random_state + len(portfolio_records),
            )
            portfolio_records.append(
                {
                    "test": f"net_long_short_{test}",
                    "variant": name_a,
                    "reference": name_b,
                    "model": model,
                    "weighting": args.weighting,
                    "universe_variant": args.universe_variant,
                    "cost_bps": args.cost_bps,
                    "months": net["months"],
                    "annualized_net_variant": net["mean_a"] * 12.0,
                    "annualized_net_reference": net["mean_b"] * 12.0,
                    "delta_annualized_net": net["delta_mean"] * 12.0,
                    "delta_monthly_net": net["delta_mean"],
                    "hac_t_stat": net["hac_t_stat"],
                    "hac_p_two_sided": net["hac_p_two_sided"],
                    **sharpe,
                }
            )

    ic_tests = holm_within_test(pd.DataFrame(ic_records))
    loss_tests = holm_column_within_test(
        pd.DataFrame(loss_records),
        "loss_hac_p_two_sided",
        "loss_hac_p_holm",
    )
    loss_tests = holm_column_within_test(
        loss_tests,
        "clark_west_p_one_sided",
        "clark_west_p_one_sided_holm",
    )
    portfolio_tests = holm_within_test(pd.DataFrame(portfolio_records))
    portfolio_tests = holm_column_within_test(
        portfolio_tests,
        "sharpe_bootstrap_p_two_sided",
        "sharpe_bootstrap_p_two_sided_holm",
    )
    if not portfolio_tests.empty:
        portfolio_tests["sharpe_bootstrap_raw_significant_5pct"] = (
            portfolio_tests["sharpe_bootstrap_p_two_sided"].le(0.05)
        )
        portfolio_tests["sharpe_bootstrap_holm_significant_5pct"] = (
            portfolio_tests["sharpe_bootstrap_p_two_sided_holm"].le(0.05)
        )
        portfolio_tests["sharpe_bootstrap_ci_excludes_zero"] = ~portfolio_tests[
            "sharpe_bootstrap_ci_includes_zero"
        ].astype(bool)
        portfolio_tests[
            "sharpe_bootstrap_ci_excludes_zero_but_holm_not_significant"
        ] = (
            portfolio_tests["sharpe_bootstrap_ci_excludes_zero"]
            & ~portfolio_tests["sharpe_bootstrap_holm_significant_5pct"]
        )
        portfolio_tests["sharpe_inference_interpretation"] = np.select(
            [
                portfolio_tests["sharpe_bootstrap_holm_significant_5pct"],
                portfolio_tests[
                    "sharpe_bootstrap_ci_excludes_zero_but_holm_not_significant"
                ],
            ],
            [
                "sharpe_difference_survives_holm",
                "raw_ci_excludes_zero_but_holm_not_significant",
            ],
            default="not_statistically_resolved",
        )

    summaries = [
        summary_rows("compustat_enriched", args.baseline_dir, args.weighting, args.universe_variant, args.cost_bps)
    ]
    for name, run_dir in runs.items():
        summaries.append(
            summary_rows(name, run_dir, args.weighting, args.universe_variant, args.cost_bps)
        )
    summary = pd.concat(summaries, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ic_tests.to_csv(args.output_dir / "ablation_paired_ic_tests.csv", index=False)
    loss_tests.to_csv(args.output_dir / "ablation_paired_loss_tests.csv", index=False)
    portfolio_tests.to_csv(
        args.output_dir / "ablation_paired_portfolio_tests.csv", index=False
    )
    summary.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(sample_checks).to_csv(
        args.output_dir / "ablation_sample_checks.csv", index=False
    )
    print(f"runs compared: {sorted(runs)}")
    unmatched = [check for check in sample_checks if not check["samples_identical"]]
    print(f"sample checks: {len(sample_checks)}, unmatched: {len(unmatched)}")
    print(
        f"ic tests: {len(ic_tests)}, loss tests: {len(loss_tests)}, "
        f"portfolio tests: {len(portfolio_tests)}"
    )
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
