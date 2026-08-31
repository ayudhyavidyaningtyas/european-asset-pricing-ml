"""Build final econometric evidence tables from saved monthly artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stats as project_stats  # noqa: E402


PPY = 12
RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "econometric_evidence_tables"
DEFAULT_REVISION_DIR = (
    RESULTS_ROOT / "estimates_revisions_pure_strict_lag1_revision_signal_ridge"
)
DEFAULT_VALIDATION_DIR = (
    RESULTS_ROOT / "estimates_revisions_validation_selected_implementable_strategy"
)
DEFAULT_LAG_TEMPLATE = "lag_sensitivity_pure_revisions_lag{lag}_ridge"
DEFAULT_FMB = (
    RESULTS_ROOT
    / "depth_estimates_revisions_pure_strict_lag1_revision_signal_ridge"
    / "fama_macbeth_summary.csv"
)
DEFAULT_CLARK_WEST = (
    RESULTS_ROOT
    / "estimates_family_ablation"
    / "ablation_paired_loss_tests.csv"
)
DEFAULT_IC_ABLATION = (
    RESULTS_ROOT
    / "estimates_family_ablation"
    / "ablation_paired_ic_tests.csv"
)
DEFAULT_REVISION_SPANNING = (
    RESULTS_ROOT
    / "revision_strategy_final_exhibits"
    / "revision_external_factor_spanning.csv"
)
DEFAULT_REVISION_PREDICTION = (
    RESULTS_ROOT
    / "estimates_revisions_pure_strict_lag1_revision_signal_ridge"
    / "prediction_metrics.csv"
)
DEFAULT_PLACEBO = (
    RESULTS_ROOT
    / "strict_estimates_data_controls"
    / "revision_feature_placebo_summary.csv"
)


@dataclass(frozen=True)
class ConstrainedRun:
    label: str
    path: Path
    strategies: tuple[str, ...]
    constraint: str


DEFAULT_CONSTRAINED_RUNS = (
    ConstrainedRun(
        "constrained_estimates_long_only",
        RESULTS_ROOT / "constrained_estimates_long_only",
        (
            "validation_selected_estimates_long_only",
            "fixed_smooth75_ridge_top500_observed",
        ),
        "name5_country40_sector40_turnover",
    ),
    ConstrainedRun(
        "constrained_estimates_revisions_long_only",
        RESULTS_ROOT / "constrained_estimates_revisions_long_only",
        (
            "validation_selected_estimates_long_only",
            "fixed_smooth75_ridge_top500_observed",
        ),
        "name5_country40_sector40_turnover",
    ),
    ConstrainedRun(
        "constrained_pure_revisions_long_only",
        RESULTS_ROOT / "constrained_estimates_revisions_pure_strict_lag1_revision_signal_fixed",
        ("fixed_pure_revision_signal_smooth75_ridge_top500_observed",),
        "name5_country40_sector40_turnover",
    ),
)


def sharpe_ratio(values: pd.Series | np.ndarray) -> float:
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return np.nan
    volatility = clean.std(ddof=1)
    if volatility <= 0:
        return np.nan
    return float(clean.mean() / volatility * np.sqrt(PPY))


def metric_point(values: np.ndarray, metric: str) -> float:
    return project_stats.metric_point(values, metric, ppy=PPY)


def stationary_bootstrap_metric_ci(
    values: pd.Series | np.ndarray,
    *,
    metric: str,
    expected_block: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    return project_stats.stationary_bootstrap_metric_ci(
        values,
        metric=metric,
        expected_block=expected_block,
        n_boot=n_boot,
        seed=seed,
        ppy=PPY,
    )


def _append_metric_ci(
    records: list[dict[str, Any]],
    *,
    monthly_returns: pd.Series,
    source_run: str,
    source_file: Path,
    portfolio_object: str,
    object_class: str,
    strategy: str,
    constraint: str,
    portfolio: str,
    aum_label: str,
    metric: str,
    bootstrap_block: int,
    bootstrap_repetitions: int,
    seed: int,
    ci_scope: str,
) -> None:
    bootstrap_metric = metric
    if metric.startswith("annualized_"):
        bootstrap_metric = "annualized_mean"
    elif metric == "net_sharpe":
        bootstrap_metric = "sharpe"
    records.append(
        {
            "source_run": source_run,
            "source_file": str(source_file),
            "portfolio_object": portfolio_object,
            "object_class": object_class,
            "strategy": strategy,
            "constraint": constraint,
            "portfolio": portfolio,
            "aum_label": aum_label,
            "metric": metric,
            "bootstrap_block": bootstrap_block,
            "bootstrap_repetitions": bootstrap_repetitions,
            "ci_scope": ci_scope,
            **stationary_bootstrap_metric_ci(
                monthly_returns,
                metric=bootstrap_metric,
                expected_block=bootstrap_block,
                n_boot=bootstrap_repetitions,
                seed=seed,
            ),
        }
    )


def _unconstrained_net_return(
    monthly: pd.DataFrame,
    portfolio: str,
    cost_bps: int,
) -> pd.Series:
    cost_rate = cost_bps / 10_000.0
    if portfolio == "long_short":
        return (
            pd.to_numeric(monthly["gross_long_short_return"], errors="coerce")
            - pd.to_numeric(monthly["long_short_turnover"], errors="coerce") * cost_rate
        )
    if portfolio == "long_only_top_decile":
        return (
            pd.to_numeric(monthly["long_return"], errors="coerce")
            - pd.to_numeric(monthly["long_only_turnover"], errors="coerce") * cost_rate
        )
    raise ValueError(f"Unknown portfolio: {portfolio}")


def build_portfolio_level_bootstrap_cis(
    *,
    revision_dir: Path,
    constrained_runs: tuple[ConstrainedRun, ...],
    validation_dir: Path,
    cost_bps: int,
    bootstrap_block: int,
    bootstrap_repetitions: int,
    random_state: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if (revision_dir / "monthly_portfolios.csv").exists():
        monthly = pd.read_csv(
            revision_dir / "monthly_portfolios.csv",
            parse_dates=["signal_date", "return_date"],
        )
        selected = monthly[
            monthly["model"].eq("ridge_rank")
            & monthly["target_mode"].eq("rank")
            & monthly["weighting"].eq("equal")
            & monthly["universe_variant"].eq("ex_bottom_20pct")
        ].copy()
        for portfolio in ["long_short", "long_only_top_decile"]:
            returns = _unconstrained_net_return(
                selected.sort_values("return_date"),
                portfolio,
                cost_bps,
            )
            for metric in ["annualized_net_return", "sharpe"]:
                _append_metric_ci(
                    records,
                    monthly_returns=returns,
                    source_run=revision_dir.name,
                    source_file=revision_dir / "monthly_portfolios.csv",
                    portfolio_object=f"unconstrained_equal_ex_bottom_20pct_{portfolio}",
                    object_class="unconstrained_decile",
                    strategy="ridge_rank",
                    constraint="",
                    portfolio=portfolio,
                    aum_label="",
                    metric="net_sharpe" if metric == "sharpe" else metric,
                    bootstrap_block=bootstrap_block,
                    bootstrap_repetitions=bootstrap_repetitions,
                    seed=random_state + len(records),
                    ci_scope="unconditional_saved_monthly_returns",
                )

    for run in constrained_runs:
        monthly_path = run.path / "benchmark_relative_monthly.csv"
        if not monthly_path.exists():
            continue
        monthly = pd.read_csv(monthly_path, parse_dates=["date", "target_date"])
        for strategy in run.strategies:
            subset = monthly[
                monthly["strategy"].eq(strategy)
                & monthly["constraint"].eq(run.constraint)
            ].sort_values("target_date")
            if subset.empty:
                continue
            for aum_label in ["10m", "100m", "500m"]:
                for column, metric, portfolio in [
                    (f"net_return_{aum_label}", "annualized_net_return", "long_only"),
                    (f"net_return_{aum_label}", "net_sharpe", "long_only"),
                    (
                        f"active_return_{aum_label}",
                        "annualized_active_return",
                        "active_long_only",
                    ),
                    (
                        f"active_return_{aum_label}",
                        "information_ratio",
                        "active_long_only",
                    ),
                ]:
                    if column not in subset:
                        continue
                    _append_metric_ci(
                        records,
                        monthly_returns=pd.to_numeric(subset[column], errors="coerce"),
                        source_run=run.label,
                        source_file=monthly_path,
                        portfolio_object=f"{strategy}_{run.constraint}_{aum_label}_{portfolio}",
                        object_class="constrained_long_only",
                        strategy=strategy,
                        constraint=run.constraint,
                        portfolio=portfolio,
                        aum_label=aum_label,
                        metric=metric,
                        bootstrap_block=bootstrap_block,
                        bootstrap_repetitions=bootstrap_repetitions,
                        seed=random_state + len(records),
                        ci_scope="unconditional_saved_monthly_returns",
                    )

    validation_path = validation_dir / "validation_selected_monthly.csv"
    if validation_path.exists():
        validation = pd.read_csv(validation_path, parse_dates=["date", "target_date"])
        for (strategy, portfolio), group in validation.groupby(
            ["strategy", "portfolio"],
            sort=True,
        ):
            returns = pd.to_numeric(group.sort_values("target_date")["net_return_100m"], errors="coerce")
            for metric in ["annualized_net_return", "net_sharpe"]:
                _append_metric_ci(
                    records,
                    monthly_returns=returns,
                    source_run=validation_dir.name,
                    source_file=validation_path,
                    portfolio_object=f"{strategy}_{portfolio}_100m",
                    object_class="validation_selected_strategy",
                    strategy=strategy,
                    constraint="rolling_validation_selection",
                    portfolio=portfolio,
                    aum_label="100m",
                    metric=metric,
                    bootstrap_block=bootstrap_block,
                    bootstrap_repetitions=bootstrap_repetitions,
                    seed=random_state + len(records),
                    ci_scope="conditional_on_saved_selection_path",
                )

    return pd.DataFrame(records)


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
            month["target_return_rank"],
            method="spearman",
        ),
        include_groups=False,
    )


def monthly_portfolio_net_return(
    run_dir: Path,
    *,
    model: str,
    weighting: str,
    universe_variant: str,
    portfolio: str,
    cost_bps: int,
) -> pd.Series:
    monthly = pd.read_csv(
        run_dir / "monthly_portfolios.csv",
        parse_dates=["signal_date", "return_date"],
    )
    subset = monthly[
        monthly["model"].eq(model)
        & monthly["target_mode"].eq("rank")
        & monthly["weighting"].eq(weighting)
        & monthly["universe_variant"].eq(universe_variant)
    ].sort_values("return_date")
    returns = _unconstrained_net_return(subset, portfolio, cost_bps)
    return pd.Series(returns.to_numpy(dtype=float), index=pd.DatetimeIndex(subset["return_date"]))


def paired_hac_delta(left: pd.Series, right: pd.Series, hac_lags: int) -> dict[str, float]:
    aligned = pd.concat({"left": left, "right": right}, axis=1, join="inner").dropna()
    delta = aligned["left"] - aligned["right"]
    test = project_stats.hac_mean_diff_test(delta, maxlags=hac_lags)
    return {
        "months": int(test["n"]),
        "left_mean": float(aligned["left"].mean()) if len(aligned) else np.nan,
        "right_mean": float(aligned["right"].mean()) if len(aligned) else np.nan,
        "mean_difference": float(test["mean"]),
        "annualized_mean_difference": float(test["mean"] * PPY),
        "hac_standard_error": float(test["se"]),
        "t_stat": float(test["t"]),
        "p_two_sided": float(test["p_two_sided"]),
    }


def build_lag_sensitivity_paired_tests(
    *,
    results_root: Path,
    run_template: str,
    lags: tuple[int, ...],
    model: str,
    weighting: str,
    universe_variant: str,
    cost_bps: int,
    hac_lags: int,
    cells: tuple[tuple[str, str], ...] | None = None,
) -> pd.DataFrame:
    run_dirs = {
        lag: results_root / run_template.format(lag=lag)
        for lag in lags
    }
    ic = {lag: monthly_ic_series(run_dirs[lag], model) for lag in lags}
    portfolio_cells = cells or ((weighting, universe_variant),)
    records: list[dict[str, Any]] = []
    for left_lag_index, left_lag in enumerate(lags):
        for right_lag in lags[left_lag_index + 1 :]:
            result = paired_hac_delta(
                ic[left_lag],
                ic[right_lag],
                hac_lags,
            )
            records.append(
                {
                    "test_family": "monthly_ic",
                    "comparison": f"lag{left_lag}_minus_lag{right_lag}",
                    "left_lag_months": left_lag,
                    "right_lag_months": right_lag,
                    "model": model,
                    "weighting": "",
                    "universe_variant": "",
                    "portfolio": "",
                    "portfolio_cell": "",
                    "cost_bps": np.nan,
                    "scale": "monthly_ic",
                    "interpretation": "positive_mean_difference_favors_shorter_lag",
                    **result,
                }
            )
            for cell_weighting, cell_universe in portfolio_cells:
                long_short = {
                    lag: monthly_portfolio_net_return(
                        run_dirs[lag],
                        model=model,
                        weighting=cell_weighting,
                        universe_variant=cell_universe,
                        portfolio="long_short",
                        cost_bps=cost_bps,
                    )
                    for lag in lags
                }
                long_only = {
                    lag: monthly_portfolio_net_return(
                        run_dirs[lag],
                        model=model,
                        weighting=cell_weighting,
                        universe_variant=cell_universe,
                        portfolio="long_only_top_decile",
                        cost_bps=cost_bps,
                    )
                    for lag in lags
                }
                for family, series_map, portfolio in [
                    ("net_return_long_short", long_short, "long_short"),
                    ("net_return_long_only", long_only, "long_only_top_decile"),
                ]:
                    result = paired_hac_delta(
                        series_map[left_lag],
                        series_map[right_lag],
                        hac_lags,
                    )
                    records.append(
                        {
                            "test_family": family,
                            "comparison": f"lag{left_lag}_minus_lag{right_lag}",
                            "left_lag_months": left_lag,
                            "right_lag_months": right_lag,
                            "model": model,
                            "weighting": cell_weighting,
                            "universe_variant": cell_universe,
                            "portfolio": portfolio,
                            "portfolio_cell": f"{cell_weighting}_{cell_universe}",
                            "cost_bps": cost_bps,
                            "scale": "monthly_return",
                            "interpretation": "positive_mean_difference_favors_shorter_lag",
                            **result,
                        }
                    )
    tests = pd.DataFrame(records)
    if tests.empty:
        return tests
    tests["p_two_sided_holm"] = tests.groupby("test_family")[
        "p_two_sided"
    ].transform(lambda values: multipletests(values.fillna(1.0), method="holm")[1])
    return tests


def _add_summary_row(
    rows: list[dict[str, Any]],
    *,
    section: str,
    evidence: str,
    specification: str,
    metric: str,
    estimate: float,
    ci_low: float = np.nan,
    ci_high: float = np.nan,
    statistic: float = np.nan,
    p_value: float = np.nan,
    p_value_holm: float = np.nan,
    inference_family: str,
    source_file: Path,
    notes: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "evidence": evidence,
            "specification": specification,
            "metric": metric,
            "estimate": estimate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "statistic": statistic,
            "p_value": p_value,
            "p_value_holm": p_value_holm,
            "inference_family": inference_family,
            "source_file": str(source_file),
            "notes": notes,
        }
    )


def _factor_spanning_row(
    spanning: pd.DataFrame,
    *,
    weighting: str,
    universe_variant: str,
    portfolio: str,
) -> pd.Series | None:
    subset = spanning[
        spanning["comparison"].eq("absolute")
        & spanning["model"].eq("ridge_rank")
        & spanning["weighting"].eq(weighting)
        & spanning["universe_variant"].eq(universe_variant)
        & spanning["portfolio"].eq(portfolio)
        & spanning["cost_bps"].eq(25)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def append_revision_strategy_rows(
    rows: list[dict[str, Any]],
    *,
    revision_spanning_path: Path | None,
    pure_revision_prediction_path: Path | None,
    ic_ablation_path: Path | None,
) -> None:
    if revision_spanning_path is not None and revision_spanning_path.exists():
        spanning = pd.read_csv(revision_spanning_path)
        equal_alpha = _factor_spanning_row(
            spanning,
            weighting="equal",
            universe_variant="standard_ex_bottom_5pct",
            portfolio="long_short",
        )
        if equal_alpha is not None:
            _add_summary_row(
                rows,
                section="revision_strategy",
                evidence="Equal-weight revision FF5+WML alpha",
                specification="ridge revision signal, standard ex-bottom-5pct long-short, 25 bps",
                metric="alpha_annualized",
                estimate=float(equal_alpha["alpha_annualized"]),
                statistic=float(equal_alpha["alpha_t"]),
                p_value=float(equal_alpha["alpha_p"]),
                p_value_holm=float(equal_alpha["alpha_p_holm"]),
                inference_family="external_factor_spanning_alpha",
                source_file=revision_spanning_path,
                notes=(
                    "six-factor-plus-momentum spanning; "
                    f"WML beta={equal_alpha['beta_WML']}"
                ),
            )
        value_alpha = _factor_spanning_row(
            spanning,
            weighting="value",
            universe_variant="standard_ex_bottom_5pct",
            portfolio="long_short",
        )
        if value_alpha is not None:
            _add_summary_row(
                rows,
                section="revision_strategy",
                evidence="Value-weight revision momentum spanning",
                specification="ridge revision signal, standard ex-bottom-5pct value-weight long-short, 25 bps",
                metric="alpha_annualized",
                estimate=float(value_alpha["alpha_annualized"]),
                statistic=float(value_alpha["alpha_t"]),
                p_value=float(value_alpha["alpha_p"]),
                p_value_holm=float(value_alpha["alpha_p_holm"]),
                inference_family="external_factor_spanning_alpha",
                source_file=revision_spanning_path,
                notes=(
                    "large-cap weighted revision returns are spanned; "
                    f"WML beta={value_alpha['beta_WML']}"
                ),
            )

    if pure_revision_prediction_path is not None and pure_revision_prediction_path.exists():
        metrics = pd.read_csv(pure_revision_prediction_path)
        subset = metrics[
            metrics["model"].eq("ridge_rank")
            & metrics["target_mode"].eq("rank")
        ]
        if not subset.empty:
            row = subset.iloc[0]
            _add_summary_row(
                rows,
                section="revision_strategy",
                evidence="Pure revision standalone IC",
                specification="six strict-lag revision features only",
                metric="mean_monthly_spearman_ic",
                estimate=float(row["mean_monthly_spearman_ic"]),
                statistic=float(row["ic_information_ratio"]),
                inference_family="saved_monthly_prediction_metrics",
                source_file=pure_revision_prediction_path,
                notes=f"observations={row['observations']}",
            )

    if ic_ablation_path is not None and ic_ablation_path.exists():
        ic = pd.read_csv(ic_ablation_path)
        subset = ic[
            ic["test"].eq("monthly_ic_variant_minus_compustat")
            & ic["variant"].eq("estimates_revisions_only")
            & ic["reference"].eq("compustat_enriched")
            & ic["model"].eq("ridge_rank")
        ]
        if not subset.empty:
            row = subset.iloc[0]
            _add_summary_row(
                rows,
                section="revision_strategy",
                evidence="Revision IC increment over Compustat controls",
                specification="ridge rank: estimates revisions-only feature set minus Compustat enriched reference",
                metric="delta_mean_monthly_spearman_ic",
                estimate=float(row["delta_mean_ic"]),
                statistic=float(row["hac_t_stat"]),
                p_value=float(row["hac_p_two_sided"]),
                p_value_holm=float(row["hac_p_holm"]),
                inference_family="monthly_ic_variant_minus_compustat",
                source_file=ic_ablation_path,
                notes=(
                    "small marginal IC beside larger standalone revision IC; "
                    "manifest-guarded nested comparison adds revision features to Compustat controls"
                ),
            )


def build_econometric_evidence_summary(
    *,
    fmb_path: Path,
    clark_west_path: Path,
    placebo_path: Path,
    portfolio_cis: pd.DataFrame,
    lag_tests: pd.DataFrame,
    revision_spanning_path: Path | None = None,
    pure_revision_prediction_path: Path | None = None,
    ic_ablation_path: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if fmb_path.exists():
        fmb = pd.read_csv(fmb_path)
        spec = fmb[
            fmb["specification"].eq("characteristics_risk_country_sector")
        ]
        if not spec.empty:
            row = spec.iloc[0]
            _add_summary_row(
                rows,
                section="predictability",
                evidence="Fama-MacBeth revision-score slope",
                specification="controls: characteristics, beta/idiosyncratic risk, country and sector FE",
                metric="annualized_score_slope",
                estimate=float(row["annualized_score_slope"]),
                ci_low=float(row["ci_low"] * PPY),
                ci_high=float(row["ci_high"] * PPY),
                statistic=float(row["t_stat"]),
                p_value=float(row["p_value"]),
                p_value_holm=float(row["p_value_holm"]),
                inference_family="Fama-MacBeth specifications",
                source_file=fmb_path,
            )

    append_revision_strategy_rows(
        rows,
        revision_spanning_path=revision_spanning_path,
        pure_revision_prediction_path=pure_revision_prediction_path,
        ic_ablation_path=ic_ablation_path,
    )

    if clark_west_path.exists():
        cw = pd.read_csv(clark_west_path)
        subset = cw[
            cw["test"].eq("rank_loss_variant_minus_compustat")
            & cw["variant"].eq("estimates_revisions_only")
            & cw["reference"].eq("compustat_enriched")
        ]
        for _, row in subset.sort_values("model").iterrows():
            _add_summary_row(
                rows,
                section="secondary_predictive_accuracy",
                evidence="Clark-West nested rank-loss check",
                specification=f"{row['model']}: revisions-only plus Compustat reference",
                metric="clark_west_adjusted_monthly_loss_difference",
                estimate=float(row["clark_west_adjusted_mean_difference"]),
                statistic=float(row["clark_west_hac_t_stat"]),
                p_value=float(row["clark_west_p_one_sided"]),
                p_value_holm=float(row["clark_west_p_one_sided_holm"]),
                inference_family="rank_loss_variant_minus_compustat",
                source_file=clark_west_path,
                notes=(
                    f"clark_west_is_nested={row.get('clark_west_is_nested', '')}; "
                    f"{row.get('clark_west_nesting_note', '')}"
                ),
            )

    selected_ci_specs = [
        (
            "implementability",
            "Constrained pure revisions level CI",
            "fixed_pure_revision_signal_smooth75_ridge_top500_observed_name5_country40_sector40_turnover_100m_long_only",
            "annualized_net_return",
        ),
        (
            "implementability",
            "Constrained pure revisions Sharpe CI",
            "fixed_pure_revision_signal_smooth75_ridge_top500_observed_name5_country40_sector40_turnover_100m_long_only",
            "net_sharpe",
        ),
        (
            "benchmark_relative",
            "Constrained pure revisions active-return CI",
            "fixed_pure_revision_signal_smooth75_ridge_top500_observed_name5_country40_sector40_turnover_100m_active_long_only",
            "annualized_active_return",
        ),
        (
            "benchmark_relative",
            "Constrained pure revisions information-ratio CI",
            "fixed_pure_revision_signal_smooth75_ridge_top500_observed_name5_country40_sector40_turnover_100m_active_long_only",
            "information_ratio",
        ),
        (
            "selection",
            "Validation-selected long-only conditional CI",
            "validation_selected_long_only_long_only_100m",
            "net_sharpe",
        ),
    ]
    if {"portfolio_object", "metric"}.issubset(portfolio_cis.columns):
        for section, evidence, object_name, metric in selected_ci_specs:
            subset = portfolio_cis[
                portfolio_cis["portfolio_object"].eq(object_name)
                & portfolio_cis["metric"].eq(metric)
            ]
            if subset.empty:
                continue
            row = subset.iloc[0]
            _add_summary_row(
                rows,
                section=section,
                evidence=evidence,
                specification=object_name,
                metric=metric,
                estimate=float(row["point"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                p_value=float(row["p_two_sided_zero"]),
                inference_family="stationary_block_bootstrap_level_ci",
                source_file=Path(row["source_file"]),
                notes=str(row["ci_scope"]),
            )

    if not lag_tests.empty:
        subset = lag_tests[lag_tests["comparison"].eq("lag1_minus_lag3")].copy()
        sort_columns = [
            column
            for column in [
                "test_family",
                "weighting",
                "universe_variant",
                "portfolio",
            ]
            if column in subset.columns
        ]
        for _, row in subset.sort_values(sort_columns).iterrows():
            family = row["test_family"]
            if family == "monthly_ic":
                specification = "monthly_ic"
                notes = "monthly cross-sectional rank IC; resampling/test unit is months"
            else:
                specification = (
                    f"{family}: {row['weighting']}, "
                    f"{row['universe_variant']}, {row['portfolio']}"
                )
                notes = "net return derived from gross return minus turnover times 25 bps cost"
                if family == "net_return_long_only":
                    notes += (
                        "; long-only net return can reflect turnover-cost effects "
                        "as well as information decay"
                    )
            _add_summary_row(
                rows,
                section="lag_decay",
                evidence="Lag 1 minus lag 3 paired HAC test",
                specification=specification,
                metric=(
                    "monthly_mean_difference"
                    if family == "monthly_ic"
                    else "annualized_mean_difference"
                ),
                estimate=(
                    float(row["mean_difference"])
                    if family == "monthly_ic"
                    else float(row["annualized_mean_difference"])
                ),
                statistic=float(row["t_stat"]),
                p_value=float(row["p_two_sided"]),
                p_value_holm=float(row["p_two_sided_holm"]),
                inference_family=family,
                source_file=Path("revision_lag_sensitivity_paired_tests.csv"),
                notes=notes,
            )

    if placebo_path.exists():
        placebo = pd.read_csv(placebo_path)
        subset = placebo[placebo["run"].eq("revision_feature_placebo_pure_rank")]
        if not subset.empty:
            row = subset.iloc[0]
            _add_summary_row(
                rows,
                section="placebo",
                evidence="Within-month shuffled revision features",
                specification="pure revisions ridge rank",
                metric="actual_mean_monthly_ic",
                estimate=float(row["actual_mean_monthly_ic"]),
                p_value=float(row["p_placebo_ic_ge_actual"]),
                inference_family="within_month_revision_feature_placebo",
                source_file=placebo_path,
                notes=(
                    f"placebo mean IC={row['placebo_mean_monthly_ic_mean']}; "
                    f"repetitions={row['repetitions']}"
                ),
            )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--revision-dir", type=Path, default=DEFAULT_REVISION_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lag-run-template", default=DEFAULT_LAG_TEMPLATE)
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--model", default="ridge_rank")
    parser.add_argument(
        "--weighting",
        choices=["equal", "value"],
        help="Optional single-weighting override; otherwise all default weightings are tested.",
    )
    parser.add_argument(
        "--universe-variant",
        choices=["standard_ex_bottom_5pct", "ex_bottom_20pct"],
        help="Optional single-universe override; otherwise all default universes are tested.",
    )
    parser.add_argument(
        "--weightings",
        nargs="+",
        default=["equal", "value"],
        choices=["equal", "value"],
    )
    parser.add_argument(
        "--universe-variants",
        nargs="+",
        default=["standard_ex_bottom_5pct", "ex_bottom_20pct"],
        choices=["standard_ex_bottom_5pct", "ex_bottom_20pct"],
    )
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--bootstrap-block", type=int, default=6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--fmb", type=Path, default=DEFAULT_FMB)
    parser.add_argument("--clark-west", type=Path, default=DEFAULT_CLARK_WEST)
    parser.add_argument("--ic-ablation", type=Path, default=DEFAULT_IC_ABLATION)
    parser.add_argument(
        "--revision-spanning",
        type=Path,
        default=DEFAULT_REVISION_SPANNING,
    )
    parser.add_argument(
        "--pure-revision-prediction",
        type=Path,
        default=DEFAULT_REVISION_PREDICTION,
    )
    parser.add_argument("--placebo", type=Path, default=DEFAULT_PLACEBO)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weightings = tuple([args.weighting] if args.weighting else args.weightings)
    universe_variants = tuple(
        [args.universe_variant]
        if args.universe_variant
        else args.universe_variants
    )
    lag_cells = tuple(
        (cell_weighting, cell_universe)
        for cell_weighting in weightings
        for cell_universe in universe_variants
    )
    portfolio_cis = build_portfolio_level_bootstrap_cis(
        revision_dir=args.revision_dir,
        constrained_runs=DEFAULT_CONSTRAINED_RUNS,
        validation_dir=args.validation_dir,
        cost_bps=args.cost_bps,
        bootstrap_block=args.bootstrap_block,
        bootstrap_repetitions=args.bootstrap_repetitions,
        random_state=args.random_state,
    )
    lag_tests = build_lag_sensitivity_paired_tests(
        results_root=args.results_root,
        run_template=args.lag_run_template,
        lags=tuple(args.lags),
        model=args.model,
        weighting=weightings[0],
        universe_variant=universe_variants[0],
        cost_bps=args.cost_bps,
        hac_lags=args.hac_lags,
        cells=lag_cells,
    )
    summary = build_econometric_evidence_summary(
        fmb_path=args.fmb,
        clark_west_path=args.clark_west,
        placebo_path=args.placebo,
        portfolio_cis=portfolio_cis,
        lag_tests=lag_tests,
        revision_spanning_path=args.revision_spanning,
        pure_revision_prediction_path=args.pure_revision_prediction,
        ic_ablation_path=args.ic_ablation,
    )

    portfolio_path = args.output_dir / "portfolio_level_bootstrap_cis.csv"
    lag_path = args.output_dir / "revision_lag_sensitivity_paired_tests.csv"
    summary_path = args.output_dir / "econometric_evidence_summary.csv"
    manifest_path = args.output_dir / "manifest.json"
    portfolio_cis.to_csv(portfolio_path, index=False)
    lag_tests.to_csv(lag_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "inputs": {
            "revision_dir": str(args.revision_dir),
            "validation_dir": str(args.validation_dir),
            "fmb": str(args.fmb),
            "clark_west": str(args.clark_west),
            "ic_ablation": str(args.ic_ablation),
            "revision_spanning": str(args.revision_spanning),
            "pure_revision_prediction": str(args.pure_revision_prediction),
            "placebo": str(args.placebo),
            "constrained_runs": [
                {"label": run.label, "path": str(run.path)}
                for run in DEFAULT_CONSTRAINED_RUNS
            ],
        },
        "config": {
            "cost_bps": args.cost_bps,
            "hac_lags": args.hac_lags,
            "bootstrap_block": args.bootstrap_block,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "random_state": args.random_state,
            "lag_run_template": args.lag_run_template,
            "lags": args.lags,
            "lag_test_weightings": list(weightings),
            "lag_test_universe_variants": list(universe_variants),
        },
        "rows": {
            "portfolio_level_bootstrap_cis": int(len(portfolio_cis)),
            "revision_lag_sensitivity_paired_tests": int(len(lag_tests)),
            "econometric_evidence_summary": int(len(summary)),
        },
        "outputs": {
            "portfolio_level_bootstrap_cis": str(portfolio_path),
            "revision_lag_sensitivity_paired_tests": str(lag_path),
            "econometric_evidence_summary": str(summary_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest["rows"], indent=2))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
