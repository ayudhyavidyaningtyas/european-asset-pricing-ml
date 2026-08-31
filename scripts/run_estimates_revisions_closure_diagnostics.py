"""Closure diagnostics for the estimates revisions-only implementability result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [SRC_DIR, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stats as project_stats  # noqa: E402
from investability_ladder import LadderConfig, load_ladder_panel  # noqa: E402
from run_constrained_deep_hybrid_long_only import (  # noqa: E402
    choose_universe,
    max_drawdown,
    parse_constraint_specs,
    solve_constrained_long_only,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_estimates.parquet"
)
DEFAULT_LIQUIDITY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "supplemental"
    / "liquidity_monthly_full_period_top2000"
)
DEFAULT_RISK = RESULTS_DIR / "depth_analysis" / "rolling_risk_estimates.parquet"
DEFAULT_OUTPUT = RESULTS_DIR / "estimates_revisions_closure_diagnostics"

RUN_DIRS = {
    "full_estimates": RESULTS_DIR / "constrained_estimates_long_only",
    "revisions_only": RESULTS_DIR / "constrained_estimates_revisions_long_only",
}
SELECTED_DIRS = {
    "full_estimates": RESULTS_DIR / "estimates_validation_selected_implementable_strategy",
    "revisions_only": RESULTS_DIR / "estimates_revisions_validation_selected_implementable_strategy",
}
ATTRIBUTION_PAIRS = [
    ("validation_selected_estimates_long_only", "name5_country40_sector40"),
    ("validation_selected_estimates_long_only", "name3_country25_sector25_turnover"),
    ("fixed_smooth75_ridge_top500_observed", "name5_country40_sector40_turnover"),
]


def _annualized_sharpe(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    volatility = clean.std(ddof=1)
    if len(clean) < 2 or volatility <= 0:
        return np.nan
    return float(clean.mean() / volatility * np.sqrt(12.0))


def _summary_stats(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "months": int(len(clean)),
        "annualized_return": float(clean.mean() * 12.0),
        "annualized_volatility": float(clean.std(ddof=1) * np.sqrt(12.0)),
        "sharpe": _annualized_sharpe(clean),
        "max_drawdown": max_drawdown(clean),
    }


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    clean = pd.DataFrame({"value": values, "weight": weights}).dropna()
    total = clean["weight"].sum()
    if total <= 0:
        return np.nan
    return float(clean["value"].mul(clean["weight"]).sum() / total)


def write_capacity_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for run_label, run_dir in RUN_DIRS.items():
        frame = pd.read_csv(run_dir / "constrained_summary.csv")
        frame.insert(0, "run_label", run_label)
        frames.append(frame)
    capacity = pd.concat(frames, ignore_index=True)
    capacity = capacity[capacity["subperiod"].eq("full")].copy()
    key_pairs = pd.MultiIndex.from_tuples(ATTRIBUTION_PAIRS)
    pair_index = pd.MultiIndex.from_frame(capacity[["strategy", "constraint"]])
    key_capacity = capacity[pair_index.isin(key_pairs)].copy()
    keep = [
        "run_label",
        "strategy",
        "constraint",
        "aum_label",
        "annualized_net_return",
        "net_sharpe",
        "average_monthly_turnover",
        "annualized_spread_cost",
        "annualized_impact_cost",
        "average_effective_n",
        "average_max_country_weight",
        "average_max_sector_weight",
    ]
    capacity[keep].to_csv(output_dir / "capacity_full_period_all.csv", index=False)
    key_capacity[keep].to_csv(output_dir / "capacity_key_cells.csv", index=False)
    return capacity, key_capacity


def write_benchmark_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for run_label, run_dir in RUN_DIRS.items():
        frame = pd.read_csv(run_dir / "benchmark_relative_summary.csv")
        frame.insert(0, "run_label", run_label)
        frames.append(frame)
    benchmark = pd.concat(frames, ignore_index=True)
    benchmark = benchmark[benchmark["subperiod"].eq("full")].copy()
    key_pairs = pd.MultiIndex.from_tuples(ATTRIBUTION_PAIRS)
    pair_index = pd.MultiIndex.from_frame(benchmark[["strategy", "constraint"]])
    key_benchmark = benchmark[pair_index.isin(key_pairs)].copy()
    keep = [
        "run_label",
        "strategy",
        "constraint",
        "aum_label",
        "annualized_net_return",
        "annualized_benchmark_return",
        "annualized_active_return",
        "tracking_error",
        "information_ratio",
        "alpha_annualized",
        "alpha_t_stat",
        "alpha_p_two_sided",
        "alpha_p_holm",
        "benchmark_beta",
    ]
    benchmark[keep].to_csv(output_dir / "benchmark_relative_full_period_all.csv", index=False)
    key_benchmark[keep].to_csv(output_dir / "benchmark_relative_key_cells.csv", index=False)
    return benchmark, key_benchmark


def load_ladder_panel_with_beta(
    panel_path: Path,
    predictions_path: Path,
    liquidity_path: Path | None,
    risk_path: Path,
    config: LadderConfig,
) -> pd.DataFrame:
    panel = load_ladder_panel(
        panel_path,
        predictions_path,
        liquidity_path,
        risk_path,
        config,
    )
    risk = pd.read_parquet(risk_path, columns=["date", "ric", "beta_36m"])
    risk["date"] = pd.to_datetime(risk["date"])
    panel = panel.drop(columns=["beta_36m"], errors="ignore")
    return panel.merge(risk, on=["date", "ric"], how="left", validate="many_to_one")


def reconstruct_holdings(
    *,
    panel_path: Path,
    liquidity_path: Path | None,
    risk_path: Path,
    output_dir: Path,
    maximum_assets: int,
    fallback_half_spread_bps: float,
    impact_coefficient: float,
) -> pd.DataFrame:
    run_label = "revisions_only"
    selected_dir = SELECTED_DIRS[run_label]
    run_dir = RUN_DIRS[run_label]
    choices = pd.read_csv(run_dir / "strategy_choices.csv", parse_dates=["date", "target_date"])
    wanted = pd.DataFrame(ATTRIBUTION_PAIRS, columns=["strategy", "constraint"])
    choices = choices.merge(wanted[["strategy"]].drop_duplicates(), on="strategy")
    specs = {spec.name: spec for spec in parse_constraint_specs(None)}
    constraints_by_strategy = wanted.groupby("strategy")["constraint"].apply(list).to_dict()
    config = LadderConfig(
        maximum_assets=maximum_assets,
        fallback_half_spread_bps=fallback_half_spread_bps,
        impact_coefficient=impact_coefficient,
    )
    panel = load_ladder_panel_with_beta(
        panel_path,
        selected_dir / "candidate_predictions.parquet",
        liquidity_path,
        risk_path,
        config,
    )

    rows: list[dict[str, Any]] = []
    previous_weights: dict[tuple[str, str], dict[str, float]] = {}
    choices = choices.sort_values(["strategy", "date"])
    for choice in choices.itertuples(index=False):
        universe = choose_universe(
            panel,
            model=str(choice.model),
            date=pd.Timestamp(choice.date),
            rung=str(choice.rung),
            maximum_assets=maximum_assets,
        )
        if universe.empty:
            continue
        for constraint in constraints_by_strategy.get(str(choice.strategy), []):
            spec = specs[constraint]
            key = (str(choice.strategy), spec.name)
            weights, status = solve_constrained_long_only(
                universe,
                previous_weights.get(key, {}),
                spec,
            )
            if status != "ok":
                continue
            previous_weights[key] = weights
            weight_series = pd.Series(weights, name="weight")
            weight_series.index.name = "ric"
            holdings = universe.rename(
                columns={"TR.TRBCECONOMICSECTOR": "sector"}
            ).set_index("ric").reindex(weight_series.index).copy()
            holdings["weight"] = weight_series
            for row in holdings.reset_index().to_dict("records"):
                rows.append(
                    {
                        "run_label": run_label,
                        "strategy": choice.strategy,
                        "constraint": constraint,
                        "date": choice.date,
                        "target_date": choice.target_date,
                        "model": choice.model,
                        "rung": choice.rung,
                        "ric": row["ric"],
                        "weight": float(row["weight"]),
                        "target_return_1m": float(row["target_return_1m"]),
                        "company_market_cap": float(row["company_market_cap"]),
                        "market_cap_percentile": float(row["market_cap_percentile"]),
                        "screen_country": row["screen_country"],
                        "sector": row["sector"],
                        "beta_36m": float(row["beta_36m"])
                        if pd.notna(row["beta_36m"])
                        else np.nan,
                        "idio_vol_36m": float(row["idio_vol_36m"])
                        if pd.notna(row["idio_vol_36m"])
                        else np.nan,
                    }
                )
    holdings = pd.DataFrame(rows)
    holdings.to_parquet(output_dir / "reconstructed_revisions_key_holdings.parquet", index=False)
    holdings.to_csv(output_dir / "reconstructed_revisions_key_holdings.csv", index=False)
    return holdings


def load_benchmark_constituents(
    panel_path: Path,
    risk_path: Path,
    dates: pd.Series,
) -> pd.DataFrame:
    columns = [
        "date",
        "ric",
        "target_return_1m",
        "company_market_cap",
        "market_cap_percentile",
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
    ]
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"])
    wanted_dates = pd.to_datetime(pd.Series(dates).drop_duplicates())
    panel = panel[panel["date"].isin(set(wanted_dates))].copy()
    risk = pd.read_parquet(
        risk_path,
        columns=["date", "ric", "beta_36m", "idio_vol_36m"],
    )
    risk["date"] = pd.to_datetime(risk["date"])
    panel = panel.merge(risk, on=["date", "ric"], how="left", validate="many_to_one")
    panel["company_market_cap"] = pd.to_numeric(panel["company_market_cap"], errors="coerce")
    panel = panel[
        panel["company_market_cap"].gt(0)
        & pd.to_numeric(panel["target_return_1m"], errors="coerce").notna()
    ].copy()
    panel["screen_country"] = panel["screen_country"].fillna("UNKNOWN")
    panel["sector"] = panel["TR.TRBCECONOMICSECTOR"].fillna("UNKNOWN")
    totals = panel.groupby("date")["company_market_cap"].transform("sum")
    panel["benchmark_weight"] = panel["company_market_cap"] / totals
    return panel.drop(columns=["TR.TRBCECONOMICSECTOR"])


def summarize_monthly_exposures(
    holdings: pd.DataFrame,
    benchmark: pd.DataFrame,
    constrained_monthly: pd.DataFrame,
) -> pd.DataFrame:
    benchmark = benchmark.copy()
    benchmark["weighted_return"] = benchmark["benchmark_weight"] * benchmark["target_return_1m"]
    monthly_benchmark = (
        benchmark.groupby("date")
        .apply(
            lambda group: pd.Series(
                {
                    "benchmark_return": group["weighted_return"].sum(),
                    "benchmark_beta": _weighted_average(
                        group["beta_36m"],
                        group["benchmark_weight"],
                    ),
                    "benchmark_idio_vol": _weighted_average(
                        group["idio_vol_36m"],
                        group["benchmark_weight"],
                    ),
                    "benchmark_market_cap_percentile": _weighted_average(
                        group["market_cap_percentile"],
                        group["benchmark_weight"],
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    h = holdings.copy()
    h["weighted_return"] = h["weight"] * h["target_return_1m"]
    portfolio = (
        h.groupby(["run_label", "strategy", "constraint", "date", "target_date"])
        .apply(
            lambda group: pd.Series(
                {
                    "portfolio_return_reconstructed": group["weighted_return"].sum(),
                    "portfolio_beta": _weighted_average(group["beta_36m"], group["weight"]),
                    "portfolio_idio_vol": _weighted_average(
                        group["idio_vol_36m"],
                        group["weight"],
                    ),
                    "portfolio_market_cap_percentile": _weighted_average(
                        group["market_cap_percentile"],
                        group["weight"],
                    ),
                    "portfolio_holding_n": int(group["weight"].gt(1e-8).sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    constrained = constrained_monthly[
        [
            "strategy",
            "constraint",
            "date",
            "gross_return",
            "net_return_100m",
            "turnover_100m",
            "spread_cost_100m",
            "impact_cost_100m",
        ]
    ].copy()
    constrained["date"] = pd.to_datetime(constrained["date"])
    result = portfolio.merge(monthly_benchmark, on="date", how="left", validate="many_to_one")
    result = result.merge(
        constrained,
        on=["strategy", "constraint", "date"],
        how="left",
        validate="one_to_one",
    )
    result["gross_active_return"] = result["gross_return"] - result["benchmark_return"]
    result["net_active_return_100m"] = result["net_return_100m"] - result["benchmark_return"]
    result["transaction_cost_100m"] = result["gross_return"] - result["net_return_100m"]
    result["active_beta"] = result["portfolio_beta"] - result["benchmark_beta"]
    result["active_idio_vol"] = result["portfolio_idio_vol"] - result["benchmark_idio_vol"]
    result["active_market_cap_percentile"] = (
        result["portfolio_market_cap_percentile"]
        - result["benchmark_market_cap_percentile"]
    )
    result["gross_reconstruction_error"] = (
        result["portfolio_return_reconstructed"] - result["gross_return"]
    )
    return result


def _group_frame(frame: pd.DataFrame, group_columns: list[str], weight_column: str) -> pd.DataFrame:
    data = frame.copy()
    data["contribution"] = data[weight_column] * data["target_return_1m"]
    grouped = (
        data.groupby(["date", *group_columns], dropna=False)
        .agg(weight=(weight_column, "sum"), contribution=("contribution", "sum"))
        .reset_index()
    )
    grouped["return"] = grouped["contribution"] / grouped["weight"]
    return grouped


def group_attribution(
    holdings: pd.DataFrame,
    benchmark: pd.DataFrame,
    dimension: str,
    group_columns: list[str],
) -> pd.DataFrame:
    frames = []
    benchmark_groups = _group_frame(benchmark, group_columns, "benchmark_weight")
    benchmark_groups = benchmark_groups.rename(
        columns={
            "weight": "benchmark_weight",
            "contribution": "benchmark_contribution",
            "return": "benchmark_group_return",
        }
    )
    for keys, group in holdings.groupby(["run_label", "strategy", "constraint"], sort=True):
        portfolio_groups = _group_frame(group, group_columns, "weight").rename(
            columns={
                "weight": "portfolio_weight",
                "contribution": "portfolio_contribution",
                "return": "portfolio_group_return",
            }
        )
        merged = portfolio_groups.merge(
            benchmark_groups,
            on=["date", *group_columns],
            how="outer",
        )
        merged = merged[merged["date"].isin(group["date"].unique())].copy()
        for column in [
            "portfolio_weight",
            "portfolio_contribution",
            "benchmark_weight",
            "benchmark_contribution",
        ]:
            merged[column] = merged[column].fillna(0.0)
        merged["portfolio_group_return"] = merged["portfolio_group_return"].fillna(0.0)
        merged["benchmark_group_return"] = merged["benchmark_group_return"].fillna(0.0)
        merged["active_weight"] = merged["portfolio_weight"] - merged["benchmark_weight"]
        merged["allocation_effect"] = (
            merged["active_weight"] * merged["benchmark_group_return"]
        )
        merged["selection_effect"] = merged["portfolio_weight"] * (
            merged["portfolio_group_return"] - merged["benchmark_group_return"]
        )
        merged["active_contribution"] = (
            merged["portfolio_contribution"] - merged["benchmark_contribution"]
        )
        merged["run_label"], merged["strategy"], merged["constraint"] = keys
        merged["dimension"] = dimension
        merged["group_label"] = merged[group_columns].astype(str).agg(" | ".join, axis=1)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def summarize_attribution(
    monthly_exposures: pd.DataFrame,
    attribution: pd.DataFrame,
    hac_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = (
        attribution.groupby(["run_label", "strategy", "constraint", "dimension", "date"])
        .agg(
            active_contribution=("active_contribution", "sum"),
            allocation_effect=("allocation_effect", "sum"),
            selection_effect=("selection_effect", "sum"),
        )
        .reset_index()
    )
    exposure = monthly_exposures[
        [
            "run_label",
            "strategy",
            "constraint",
            "date",
            "gross_active_return",
            "net_active_return_100m",
            "transaction_cost_100m",
            "active_beta",
            "active_idio_vol",
            "active_market_cap_percentile",
            "gross_reconstruction_error",
        ]
    ]
    monthly = monthly.merge(
        exposure,
        on=["run_label", "strategy", "constraint", "date"],
        how="left",
        validate="many_to_one",
    )
    records = []
    for keys, group in monthly.groupby(["run_label", "strategy", "constraint", "dimension"]):
        selection_test = project_stats.hac_mean_diff_test(
            group["selection_effect"],
            maxlags=hac_lags,
        )
        records.append(
            {
                "run_label": keys[0],
                "strategy": keys[1],
                "constraint": keys[2],
                "dimension": keys[3],
                "months": int(len(group)),
                "annualized_gross_active_return": float(
                    group["gross_active_return"].mean() * 12.0
                ),
                "annualized_group_active_contribution": float(
                    group["active_contribution"].mean() * 12.0
                ),
                "annualized_allocation_effect": float(
                    group["allocation_effect"].mean() * 12.0
                ),
                "annualized_selection_effect": float(
                    group["selection_effect"].mean() * 12.0
                ),
                "selection_hac_t_stat": float(selection_test["t"]),
                "selection_hac_p_two_sided": float(selection_test["p_two_sided"]),
                "annualized_transaction_cost_100m": float(
                    group["transaction_cost_100m"].mean() * 12.0
                ),
                "annualized_net_active_return_100m": float(
                    group["net_active_return_100m"].mean() * 12.0
                ),
                "average_active_beta": float(group["active_beta"].mean()),
                "average_active_idio_vol": float(group["active_idio_vol"].mean()),
                "average_active_market_cap_percentile": float(
                    group["active_market_cap_percentile"].mean()
                ),
                "max_abs_gross_reconstruction_error": float(
                    group["gross_reconstruction_error"].abs().max()
                ),
            }
        )
    component_summary = pd.DataFrame(records)
    component_summary["selection_hac_p_holm_all"] = multipletests(
        component_summary["selection_hac_p_two_sided"],
        method="holm",
    )[1]
    component_summary["selection_hac_p_holm_by_dimension"] = component_summary.groupby(
        "dimension"
    )["selection_hac_p_two_sided"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )

    group_summary = (
        attribution.groupby(["run_label", "strategy", "constraint", "dimension", "group_label"])
        .agg(
            average_portfolio_weight=("portfolio_weight", "mean"),
            average_benchmark_weight=("benchmark_weight", "mean"),
            average_active_weight=("active_weight", "mean"),
            annualized_active_contribution=("active_contribution", lambda x: x.mean() * 12.0),
            annualized_allocation_effect=("allocation_effect", lambda x: x.mean() * 12.0),
            annualized_selection_effect=("selection_effect", lambda x: x.mean() * 12.0),
        )
        .reset_index()
    )
    return component_summary, group_summary


def write_attribution_outputs(
    output_dir: Path,
    panel_path: Path,
    liquidity_path: Path | None,
    risk_path: Path,
    maximum_assets: int,
    fallback_half_spread_bps: float,
    impact_coefficient: float,
    hac_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings = reconstruct_holdings(
        panel_path=panel_path,
        liquidity_path=liquidity_path,
        risk_path=risk_path,
        output_dir=output_dir,
        maximum_assets=maximum_assets,
        fallback_half_spread_bps=fallback_half_spread_bps,
        impact_coefficient=impact_coefficient,
    )
    benchmark = load_benchmark_constituents(panel_path, risk_path, holdings["date"])
    constrained = pd.read_csv(
        RUN_DIRS["revisions_only"] / "constrained_monthly.csv",
        parse_dates=["date", "target_date"],
    )
    monthly_exposures = summarize_monthly_exposures(holdings, benchmark, constrained)
    monthly_exposures.to_csv(
        output_dir / "revisions_key_monthly_exposures.csv",
        index=False,
    )

    dimensions = {
        "country": ["screen_country"],
        "sector": ["sector"],
        "country_sector": ["screen_country", "sector"],
    }
    attribution = pd.concat(
        [
            group_attribution(holdings, benchmark, dimension, columns)
            for dimension, columns in dimensions.items()
        ],
        ignore_index=True,
    )
    attribution.to_csv(output_dir / "revisions_country_sector_attribution_monthly.csv", index=False)
    component_summary, group_summary = summarize_attribution(
        monthly_exposures,
        attribution,
        hac_lags,
    )
    component_summary.to_csv(
        output_dir / "revisions_country_sector_attribution_summary.csv",
        index=False,
    )
    group_summary.to_csv(
        output_dir / "revisions_country_sector_group_contributions.csv",
        index=False,
    )
    return monthly_exposures, component_summary, group_summary


def write_selector_placebo(
    output_dir: Path,
    *,
    repetitions: int,
    random_state: int,
    hac_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_dir = SELECTED_DIRS["revisions_only"]
    selected = pd.read_csv(
        selected_dir / "validation_selected_monthly.csv",
        parse_dates=["date", "target_date"],
    )
    selected = selected[
        selected["strategy"].eq("validation_selected_long_only")
        & selected["portfolio"].eq("long_only")
    ].sort_values("date")
    actual_returns = selected.set_index("date")["net_return_100m"].astype(float)
    actual_stats = _summary_stats(actual_returns)
    actual_turnover = float(selected["turnover_100m"].mean())

    manifest = json.loads((selected_dir / "manifest.json").read_text())
    candidate = pd.read_parquet(selected_dir / "candidate_ladder_monthly.parquet")
    candidate["date"] = pd.to_datetime(candidate["date"])
    candidate = candidate[
        candidate["date"].isin(actual_returns.index)
        & candidate["portfolio"].eq("long_only")
        & candidate["weighting"].eq("value")
        & candidate["rung"].isin(manifest["rungs"])
        & candidate["model"].isin(manifest["candidate_models"])
    ].copy()
    values_by_date = [
        group["net_return_100m"].to_numpy(dtype=float)
        for _, group in candidate.groupby("date", sort=True)
    ]
    if len(values_by_date) != len(actual_returns):
        raise RuntimeError("Candidate placebo dates do not match selected dates")

    rng = np.random.default_rng(random_state)
    placebo_records = []
    for repetition in range(repetitions):
        returns = pd.Series(
            [rng.choice(values) for values in values_by_date],
            index=actual_returns.index,
        )
        stats = _summary_stats(returns)
        placebo_records.append(
            {
                "repetition": repetition,
                **stats,
            }
        )
    placebo = pd.DataFrame(placebo_records)
    placebo.to_csv(output_dir / "selector_random_placebo_distribution.csv", index=False)

    random_summary = pd.DataFrame(
        [
            {
                "strategy": "validation_selected_long_only",
                "comparison": "random_monthly_model_rung_selector",
                "months": actual_stats["months"],
                "actual_annualized_return": actual_stats["annualized_return"],
                "actual_sharpe": actual_stats["sharpe"],
                "actual_average_turnover": actual_turnover,
                "placebo_repetitions": repetitions,
                "placebo_mean_annualized_return": float(
                    placebo["annualized_return"].mean()
                ),
                "placebo_p95_annualized_return": float(
                    placebo["annualized_return"].quantile(0.95)
                ),
                "placebo_mean_sharpe": float(placebo["sharpe"].mean()),
                "placebo_p95_sharpe": float(placebo["sharpe"].quantile(0.95)),
                "p_placebo_return_ge_actual": float(
                    placebo["annualized_return"].ge(actual_stats["annualized_return"]).mean()
                ),
                "p_placebo_sharpe_ge_actual": float(
                    placebo["sharpe"].ge(actual_stats["sharpe"]).mean()
                ),
            }
        ]
    )

    baselines = pd.read_csv(
        selected_dir / "validation_baseline_monthly.csv",
        parse_dates=["date", "target_date"],
    )
    baselines = baselines[
        baselines["date"].isin(actual_returns.index)
        & baselines["portfolio"].eq("long_only")
    ].copy()
    fixed_records = []
    for strategy, group in baselines.groupby("strategy", sort=True):
        returns = group.sort_values("date").set_index("date")["net_return_100m"].astype(float)
        aligned = pd.concat({"actual": actual_returns, "baseline": returns}, axis=1).dropna()
        diff_test = project_stats.hac_mean_diff_test(
            aligned["actual"] - aligned["baseline"],
            maxlags=hac_lags,
        )
        stats = _summary_stats(aligned["baseline"])
        fixed_records.append(
            {
                "strategy": "validation_selected_long_only",
                "comparison": strategy,
                "months": int(len(aligned)),
                "actual_annualized_return": actual_stats["annualized_return"],
                "actual_sharpe": actual_stats["sharpe"],
                "actual_average_turnover": actual_turnover,
                "baseline_annualized_return": stats["annualized_return"],
                "baseline_sharpe": stats["sharpe"],
                "baseline_average_turnover": float(group["turnover_100m"].mean()),
                "delta_annualized_return": float(diff_test["mean"] * 12.0),
                "hac_t_stat": float(diff_test["t"]),
                "hac_p_two_sided": float(diff_test["p_two_sided"]),
            }
        )
    fixed_summary = pd.DataFrame(fixed_records)
    selector_summary = pd.concat([random_summary, fixed_summary], ignore_index=True)
    selector_summary.to_csv(output_dir / "selector_placebo_and_fixed_rule_summary.csv", index=False)
    return placebo, selector_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-assets", type=int, default=500)
    parser.add_argument("--fallback-half-spread-bps", type=float, default=25.0)
    parser.add_argument("--impact-coefficient", type=float, default=0.10)
    parser.add_argument("--selector-placebo-repetitions", type=int, default=5_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    liquidity = args.liquidity if args.liquidity.exists() else None
    capacity, key_capacity = write_capacity_tables(args.output_dir)
    benchmark, key_benchmark = write_benchmark_tables(args.output_dir)
    monthly_exposures, attribution_summary, group_summary = write_attribution_outputs(
        args.output_dir,
        args.panel,
        liquidity,
        args.risk,
        args.maximum_assets,
        args.fallback_half_spread_bps,
        args.impact_coefficient,
        args.hac_lags,
    )
    placebo, selector_summary = write_selector_placebo(
        args.output_dir,
        repetitions=args.selector_placebo_repetitions,
        random_state=args.random_state,
        hac_lags=args.hac_lags,
    )

    manifest = {
        "inputs": {
            "panel": str(args.panel),
            "liquidity": str(liquidity) if liquidity is not None else None,
            "risk": str(args.risk),
            "run_dirs": {key: str(value) for key, value in RUN_DIRS.items()},
            "selected_dirs": {key: str(value) for key, value in SELECTED_DIRS.items()},
        },
        "attribution_pairs": [
            {"strategy": strategy, "constraint": constraint}
            for strategy, constraint in ATTRIBUTION_PAIRS
        ],
        "selector_placebo_repetitions": args.selector_placebo_repetitions,
        "random_state": args.random_state,
        "hac_lags": args.hac_lags,
        "rows": {
            "capacity_full_period": int(len(capacity)),
            "capacity_key_cells": int(len(key_capacity)),
            "benchmark_full_period": int(len(benchmark)),
            "benchmark_key_cells": int(len(key_benchmark)),
            "monthly_exposures": int(len(monthly_exposures)),
            "attribution_summary": int(len(attribution_summary)),
            "group_contributions": int(len(group_summary)),
            "selector_placebo": int(len(placebo)),
            "selector_summary": int(len(selector_summary)),
        },
        "outputs": {
            "capacity_key_cells": str(args.output_dir / "capacity_key_cells.csv"),
            "benchmark_relative_key_cells": str(
                args.output_dir / "benchmark_relative_key_cells.csv"
            ),
            "monthly_exposures": str(
                args.output_dir / "revisions_key_monthly_exposures.csv"
            ),
            "attribution_summary": str(
                args.output_dir / "revisions_country_sector_attribution_summary.csv"
            ),
            "group_contributions": str(
                args.output_dir / "revisions_country_sector_group_contributions.csv"
            ),
            "selector_summary": str(
                args.output_dir / "selector_placebo_and_fixed_rule_summary.csv"
            ),
        },
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
