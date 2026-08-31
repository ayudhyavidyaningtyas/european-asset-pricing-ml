"""Constrained long-only construction for the frozen deep/hybrid signal.

This experiment keeps the deep/hybrid predictions and monthly model/rung
choices frozen, then changes only the portfolio construction layer.  It tests
whether the current best deep-learning candidate survives realistic
single-name, country, sector and turnover controls.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stats as project_stats  # noqa: E402
from implementable_frontier import execution_cost  # noqa: E402
from investability_ladder import (  # noqa: E402
    LadderConfig,
    investability_rungs,
    load_ladder_panel,
)


DEFAULT_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "validation_selected_deep_hybrid_liquid"
    / "validation_selected_monthly.csv"
)
DEFAULT_CANDIDATE_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "validation_selected_implementable_strategy"
    / "candidate_predictions.parquet"
)
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_LIQUIDITY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "supplemental"
    / "liquidity_monthly_full_period"
)
DEFAULT_RISK = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "depth_analysis"
    / "rolling_risk_estimates.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "constrained_deep_hybrid_long_only"
)
SUBPERIODS = [
    ("full", None, None),
    ("pre_covid_2017_2019", "2017-01-01", "2019-12-31"),
    ("covid_recovery_2020_2022", "2020-01-01", "2022-12-31"),
    ("recent_2023_2026", "2023-01-01", "2026-12-31"),
]


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    max_name_weight: float
    max_country_weight: float
    max_sector_weight: float
    turnover_penalty: float
    spread_penalty_multiplier: float = 0.0


DEFAULT_CONSTRAINTS = [
    ConstraintSpec("name5_country40_sector40", 0.05, 0.40, 0.40, 0.0),
    ConstraintSpec("name5_country40_sector40_turnover", 0.05, 0.40, 0.40, 0.005),
    ConstraintSpec("name3_country30_sector30", 0.03, 0.30, 0.30, 0.0),
    ConstraintSpec("name3_country30_sector30_turnover", 0.03, 0.30, 0.30, 0.005),
    ConstraintSpec("name3_country25_sector25_turnover", 0.03, 0.25, 0.25, 0.005),
]


def aum_label(aum: float) -> str:
    return f"{int(round(aum / 1_000_000.0))}m"


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return float(wealth.div(peak).sub(1.0).min())


def annualized_sharpe(returns: pd.Series) -> float:
    standard_deviation = returns.std(ddof=1)
    if standard_deviation <= 0 or pd.isna(standard_deviation):
        return np.nan
    return float(returns.mean() / standard_deviation * np.sqrt(12.0))


def load_frozen_choices(path: Path) -> pd.DataFrame:
    selected = pd.read_csv(path, parse_dates=["date", "target_date"])
    selected = selected[
        selected["strategy"].eq("validation_selected_long_only")
        & selected["selected_portfolio"].eq("long_only")
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No frozen long-only selected rows in {path}")
    return selected[
        ["date", "target_date", "model", "rung"]
    ].drop_duplicates(["date"])


def build_choice_panel(frozen_choices: pd.DataFrame) -> pd.DataFrame:
    frames = [
        frozen_choices.assign(strategy="frozen_deep_hybrid_selector"),
        frozen_choices[["date", "target_date"]].assign(
            model="momentum_rank",
            rung="top_500_observed_spread",
            strategy="fixed_momentum_top500_observed",
        ),
        frozen_choices[["date", "target_date"]].assign(
            model="blend90_gbm_attn_seq24_rank",
            rung="large_low_spread",
            strategy="fixed_blend90_gbm_attn24_large_low",
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def choose_universe(
    panel: pd.DataFrame,
    *,
    model: str,
    date: pd.Timestamp,
    rung: str,
    maximum_assets: int,
) -> pd.DataFrame:
    month = panel[
        panel["model"].eq(model)
        & pd.to_datetime(panel["date"]).eq(pd.Timestamp(date))
    ].copy()
    if month.empty:
        return month
    if rung not in investability_rungs(month, maximum_assets):
        raise KeyError(f"Unknown investability rung: {rung}")
    universe = investability_rungs(month, maximum_assets)[rung].copy()
    universe = universe.dropna(subset=["prediction", "target_return_1m"])
    universe["screen_country"] = universe["screen_country"].fillna("UNKNOWN")
    universe["TR.TRBCECONOMICSECTOR"] = universe[
        "TR.TRBCECONOMICSECTOR"
    ].fillna("UNKNOWN")
    return universe.sort_values("ric").reset_index(drop=True)


def _precheck_feasible(universe: pd.DataFrame, spec: ConstraintSpec) -> str | None:
    if len(universe) * spec.max_name_weight < 1.0 - 1e-10:
        return "not enough securities for name cap"
    if universe["screen_country"].nunique(dropna=False) * spec.max_country_weight < 1.0 - 1e-10:
        return "not enough country groups for country cap"
    if (
        universe["TR.TRBCECONOMICSECTOR"].nunique(dropna=False)
        * spec.max_sector_weight
        < 1.0 - 1e-10
    ):
        return "not enough sector groups for sector cap"
    return None


def solve_constrained_long_only(
    universe: pd.DataFrame,
    previous_weights: dict[str, float],
    spec: ConstraintSpec,
) -> tuple[dict[str, float], str]:
    reason = _precheck_feasible(universe, spec)
    if reason is not None:
        return {}, reason

    n_assets = len(universe)
    rics = universe["ric"].astype(str).tolist()
    previous = np.array([previous_weights.get(ric, 0.0) for ric in rics], dtype=float)
    scores = universe["prediction"].rank(method="first", pct=True).to_numpy(dtype=float)
    scores = scores - scores.mean()
    spread_fraction = (
        pd.to_numeric(universe["half_spread_bps"], errors="coerce")
        .fillna(25.0)
        .to_numpy(dtype=float)
        / 10_000.0
    )

    weights = cp.Variable(n_assets)
    trades = cp.Variable(n_assets, nonneg=True)
    constraints = [
        weights >= 0,
        weights <= float(spec.max_name_weight),
        cp.sum(weights) == 1.0,
        trades >= weights - previous,
        trades >= previous - weights,
    ]
    for _, indices in universe.groupby("screen_country", sort=False).groups.items():
        constraints.append(cp.sum(weights[list(indices)]) <= float(spec.max_country_weight))
    for _, indices in universe.groupby("TR.TRBCECONOMICSECTOR", sort=False).groups.items():
        constraints.append(cp.sum(weights[list(indices)]) <= float(spec.max_sector_weight))

    objective = cp.Maximize(
        scores @ weights
        - float(spec.turnover_penalty) * cp.sum(trades)
        - float(spec.spread_penalty_multiplier) * (spread_fraction @ trades)
    )
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(
            solver=cp.CLARABEL,
            max_iter=500,
            tol_gap_abs=1e-8,
            tol_feas=1e-8,
            verbose=False,
        )
    except cp.error.SolverError:
        try:
            problem.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, verbose=False)
        except cp.error.SolverError:
            problem.solve(solver=cp.SCS, max_iters=10_000, eps=1e-6, verbose=False)

    if weights.value is None or problem.status not in {"optimal", "optimal_inaccurate"}:
        return {}, f"optimizer status: {problem.status}"
    values = np.asarray(weights.value, dtype=float)
    values[np.abs(values) < 1e-9] = 0.0
    values = np.clip(values, 0.0, float(spec.max_name_weight) + 1e-8)
    total = values.sum()
    if total <= 0:
        return {}, "optimizer returned zero portfolio"
    values = values / total
    return {
        ric: float(weight)
        for ric, weight in zip(rics, values, strict=True)
        if weight > 1e-8
    }, "ok"


def transition_costs(
    current_weights: dict[str, float],
    previous_weights: dict[str, float],
    current_inputs: dict[str, tuple[float, float, float]],
    previous_inputs: dict[str, tuple[float, float, float]],
    aum: float,
    impact_coefficient: float,
) -> tuple[float, float, float, float]:
    names = sorted(set(current_weights) | set(previous_weights))
    if not names:
        return 0.0, 0.0, 0.0, 0.0
    delta = np.array(
        [current_weights.get(name, 0.0) - previous_weights.get(name, 0.0) for name in names],
        dtype=float,
    )
    fallback = (25.0, 10_000.0, 0.20)
    inputs = [
        current_inputs.get(name, previous_inputs.get(name, fallback))
        for name in names
    ]
    half_spread = np.array([value[0] for value in inputs], dtype=float)
    adv = np.array([value[1] for value in inputs], dtype=float)
    idio = np.array([value[2] for value in inputs], dtype=float)
    spread, impact, total = execution_cost(
        delta,
        half_spread,
        adv,
        idio,
        aum,
        impact_coefficient,
    )
    return float(np.abs(delta).sum()), spread, impact, total


def concentration_record(universe: pd.DataFrame, weights: dict[str, float]) -> dict[str, Any]:
    weight_series = pd.Series(weights, dtype=float)
    holdings = universe.set_index("ric").reindex(weight_series.index).copy()
    holdings["weight"] = weight_series
    country = holdings.groupby("screen_country", dropna=False)["weight"].sum()
    sector = holdings.groupby("TR.TRBCECONOMICSECTOR", dropna=False)["weight"].sum()
    return {
        "holding_n": int(weight_series.gt(1e-8).sum()),
        "effective_n": float(1.0 / np.square(weight_series).sum()),
        "max_single_name_weight": float(weight_series.max()),
        "top_5_name_weight": float(weight_series.sort_values(ascending=False).head(5).sum()),
        "max_country": str(country.idxmax()) if not country.empty else "",
        "max_country_weight": float(country.max()) if not country.empty else np.nan,
        "country_hhi": float(np.square(country).sum()) if not country.empty else np.nan,
        "max_sector": str(sector.idxmax()) if not sector.empty else "",
        "max_sector_weight": float(sector.max()) if not sector.empty else np.nan,
        "sector_hhi": float(np.square(sector).sum()) if not sector.empty else np.nan,
    }


def group_exposure_records(
    universe: pd.DataFrame,
    weights: dict[str, float],
    *,
    strategy: str,
    constraint: str,
    date: Any,
    target_date: Any,
    group_columns: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Per-group portfolio weight and realised return for one rebalance.

    Returns one row per (group_kind, group). The group return is the
    weight-weighted mean of ``target_return_1m`` inside the group, which is the
    quantity Brinson attribution needs.
    """
    weight_series = pd.Series(weights, dtype=float)
    holdings = universe.set_index("ric").reindex(weight_series.index).copy()
    holdings["weight"] = weight_series
    holdings = holdings[holdings["weight"].notna()]

    rows: list[dict[str, Any]] = []
    for group_column in group_columns:
        if group_column not in holdings:
            continue
        frame = holdings.copy()
        frame[group_column] = frame[group_column].fillna("UNKNOWN").astype(str)
        frame["contribution"] = frame["weight"] * frame["target_return_1m"]
        grouped = frame.groupby(group_column, sort=True).agg(
            portfolio_weight=("weight", "sum"),
            contribution=("contribution", "sum"),
            portfolio_n=("weight", "size"),
        )
        for group, record in grouped.iterrows():
            weight = float(record["portfolio_weight"])
            rows.append(
                {
                    "strategy": strategy,
                    "constraint": constraint,
                    "date": date,
                    "target_date": target_date,
                    "group_kind": group_column,
                    "group": str(group),
                    "portfolio_weight": weight,
                    "portfolio_return": (
                        float(record["contribution"]) / weight
                        if abs(weight) > 1e-12
                        else np.nan
                    ),
                    "portfolio_n": int(record["portfolio_n"]),
                }
            )
    return rows


def simulate_constrained(
    panel: pd.DataFrame,
    choices: pd.DataFrame,
    specs: list[ConstraintSpec],
    *,
    maximum_assets: int,
    aum_values: tuple[float, ...],
    impact_coefficient: float,
    exposure_sink: list[dict[str, Any]] | None = None,
    exposure_group_columns: tuple[str, ...] = (
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate constrained long-only portfolios.

    ``concentration_record`` keeps only maxima and HHIs, which cannot support a
    Brinson attribution. Passing ``exposure_sink`` collects the full per-group
    weight and return vectors as a side channel, leaving the return signature
    and every existing caller unchanged.
    """
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    previous_weights: dict[tuple[str, str], dict[str, float]] = {}
    previous_inputs: dict[tuple[str, str], dict[str, tuple[float, float, float]]] = {}

    choices = choices.sort_values(["strategy", "date"]).copy()
    for choice in choices.itertuples(index=False):
        universe = choose_universe(
            panel,
            model=str(choice.model),
            date=pd.Timestamp(choice.date),
            rung=str(choice.rung),
            maximum_assets=maximum_assets,
        )
        if universe.empty:
            failures.append(
                {
                    "strategy": choice.strategy,
                    "date": choice.date,
                    "model": choice.model,
                    "rung": choice.rung,
                    "constraint": "all",
                    "reason": "empty universe",
                }
            )
            continue
        returns = universe.set_index("ric")["target_return_1m"]
        current_inputs = {
            row.ric: (
                float(row.half_spread_bps),
                float(row.adv_eur),
                float(row.idio_vol_36m),
            )
            for row in universe.itertuples()
        }
        spread_observed = universe.set_index("ric")["spread_observed"].astype(bool)
        for spec in specs:
            key = (str(choice.strategy), spec.name)
            prior = previous_weights.get(key, {})
            weights, status = solve_constrained_long_only(universe, prior, spec)
            if status != "ok":
                failures.append(
                    {
                        "strategy": choice.strategy,
                        "date": choice.date,
                        "model": choice.model,
                        "rung": choice.rung,
                        "constraint": spec.name,
                        "reason": status,
                    }
                )
                continue
            weight_series = pd.Series(weights, dtype=float)
            gross_return = float(weight_series.mul(returns.reindex(weight_series.index)).sum())
            observed_weight = float(
                weight_series[
                    spread_observed.reindex(weight_series.index).fillna(False)
                ].sum()
            )
            row: dict[str, Any] = {
                "strategy": choice.strategy,
                "constraint": spec.name,
                "date": choice.date,
                "target_date": choice.target_date,
                "model": choice.model,
                "rung": choice.rung,
                "universe_n": int(len(universe)),
                "gross_return": gross_return,
                "observed_spread_weight": observed_weight,
                "observed_spread_fraction": float(universe["spread_observed"].mean()),
                "median_half_spread_bps": float(universe["half_spread_bps"].median()),
                **asdict(spec),
                **concentration_record(universe, weights),
            }
            old_inputs = previous_inputs.get(key, {})
            for aum in aum_values:
                label = aum_label(aum)
                turnover, spread, impact, total = transition_costs(
                    weights,
                    prior,
                    current_inputs,
                    old_inputs,
                    aum,
                    impact_coefficient,
                )
                row[f"turnover_{label}"] = turnover
                row[f"spread_cost_{label}"] = spread
                row[f"impact_cost_{label}"] = impact
                row[f"net_return_{label}"] = gross_return - total
            records.append(row)
            if exposure_sink is not None:
                exposure_sink.extend(
                    group_exposure_records(
                        universe,
                        weights,
                        strategy=str(choice.strategy),
                        constraint=spec.name,
                        date=choice.date,
                        target_date=choice.target_date,
                        group_columns=exposure_group_columns,
                    )
                )
            previous_weights[key] = weights
            previous_inputs[key] = current_inputs
    return pd.DataFrame(records), pd.DataFrame(failures)


def subperiod_frame(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    work = frame.copy()
    target_date = pd.to_datetime(work["target_date"])
    if start is not None:
        work = work[target_date.ge(pd.Timestamp(start))]
        target_date = pd.to_datetime(work["target_date"])
    if end is not None:
        work = work[target_date.le(pd.Timestamp(end))]
    return work


def summarize_constrained(monthly: pd.DataFrame, aum_values: tuple[float, ...]) -> pd.DataFrame:
    records = []
    for (strategy, constraint), group in monthly.groupby(["strategy", "constraint"], sort=True):
        for subperiod, start, end in SUBPERIODS:
            part = subperiod_frame(group, start, end)
            if len(part) < 6:
                continue
            for aum in aum_values:
                label = aum_label(aum)
                gross = part["gross_return"].astype(float)
                returns = part[f"net_return_{label}"].astype(float)
                gross_vol = float(gross.std(ddof=1) * np.sqrt(12.0))
                vol = float(returns.std(ddof=1) * np.sqrt(12.0))
                records.append(
                    {
                        "strategy": strategy,
                        "constraint": constraint,
                        "subperiod": subperiod,
                        "aum_eur": float(aum),
                        "aum_label": label,
                        "months": int(len(part)),
                        "annualized_gross_return": float(gross.mean() * 12.0),
                        "annualized_gross_volatility": gross_vol,
                        "gross_sharpe": annualized_sharpe(gross),
                        "annualized_net_return": float(returns.mean() * 12.0),
                        "annualized_net_volatility": vol,
                        "net_sharpe": annualized_sharpe(returns),
                        "max_drawdown": max_drawdown(returns),
                        "average_monthly_turnover": float(part[f"turnover_{label}"].mean()),
                        "annualized_spread_cost": float(part[f"spread_cost_{label}"].mean() * 12.0),
                        "annualized_impact_cost": float(part[f"impact_cost_{label}"].mean() * 12.0),
                        "average_effective_n": float(part["effective_n"].mean()),
                        "average_max_single_name_weight": float(part["max_single_name_weight"].mean()),
                        "average_top_5_name_weight": float(part["top_5_name_weight"].mean()),
                        "average_max_country_weight": float(part["max_country_weight"].mean()),
                        "average_max_sector_weight": float(part["max_sector_weight"].mean()),
                        "minimum_observed_spread_weight": float(part["observed_spread_weight"].min()),
                    }
                )
    return pd.DataFrame(records)


def summarize_concentration(monthly: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "holding_n",
        "effective_n",
        "max_single_name_weight",
        "top_5_name_weight",
        "max_country_weight",
        "country_hhi",
        "max_sector_weight",
        "sector_hhi",
    ]
    records = []
    for (strategy, constraint), group in monthly.groupby(["strategy", "constraint"], sort=True):
        for metric in metrics:
            values = group[metric].dropna()
            records.append(
                {
                    "strategy": strategy,
                    "constraint": constraint,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p95": float(values.quantile(0.95)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(records)


def infer_vs_constrained_momentum(
    monthly: pd.DataFrame,
    *,
    aum_values: tuple[float, ...],
    blocks: tuple[int, ...],
    n_boot: int,
    seed: int,
    hac_lags: int,
) -> pd.DataFrame:
    records = []
    for constraint, group in monthly.groupby("constraint", sort=True):
        model = group[group["strategy"].eq("frozen_deep_hybrid_selector")]
        baseline = group[group["strategy"].eq("fixed_momentum_top500_observed")]
        if model.empty or baseline.empty:
            continue
        for aum in aum_values:
            label = aum_label(aum)
            column = f"net_return_{label}"
            left = model.set_index("target_date")[column].astype(float)
            right = baseline.set_index("target_date")[column].astype(float)
            dates = left.index.intersection(right.index)
            if len(dates) < 24:
                continue
            left = left.reindex(dates)
            right = right.reindex(dates)
            mean_test = project_stats.hac_mean_diff_test(left - right, maxlags=hac_lags)
            for block in blocks:
                sharpe = project_stats.bootstrap_sharpe_diff(
                    left,
                    right,
                    np.zeros(len(dates)),
                    expected_block=block,
                    n_boot=n_boot,
                    seed=seed,
                )
                records.append(
                    {
                        "model": "frozen_deep_hybrid_selector",
                        "baseline": "fixed_momentum_top500_observed",
                        "constraint": constraint,
                        "aum_eur": float(aum),
                        "aum_label": label,
                        "months": int(len(dates)),
                        "model_annualized_net_return": float(left.mean() * 12.0),
                        "baseline_annualized_net_return": float(right.mean() * 12.0),
                        "delta_annualized_net_return": float(mean_test["mean"] * 12.0),
                        "hac_t_stat": float(mean_test["t"]),
                        "hac_p_two_sided": float(mean_test["p_two_sided"]),
                        **sharpe,
                    }
                )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    family = ["aum_label", "expected_block"]
    result["p_two_sided_holm"] = result.groupby(family)["p_two_sided"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    result["hac_p_two_sided_holm"] = result.groupby("aum_label")[
        "hac_p_two_sided"
    ].transform(lambda values: multipletests(values, method="holm")[1])
    return result


def parse_constraint_specs(values: list[str] | None) -> list[ConstraintSpec]:
    if not values:
        return DEFAULT_CONSTRAINTS
    specs = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 5:
            raise ValueError(
                "Constraint spec must be name:max_name:max_country:max_sector:turnover_penalty"
            )
        specs.append(
            ConstraintSpec(
                name=parts[0],
                max_name_weight=float(parts[1]),
                max_country_weight=float(parts[2]),
                max_sector_weight=float(parts[3]),
                turnover_penalty=float(parts[4]),
            )
        )
    return specs


def run_experiment(
    selected_path: Path,
    predictions_path: Path,
    panel_path: Path,
    liquidity_path: Path | None,
    risk_path: Path | None,
    output_dir: Path,
    specs: list[ConstraintSpec],
    aum_values: tuple[float, ...],
    maximum_assets: int,
    fallback_half_spread_bps: float,
    impact_coefficient: float,
    bootstrap_repetitions: int,
    bootstrap_blocks: tuple[int, ...],
    random_state: int,
    hac_lags: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_choices = load_frozen_choices(selected_path)
    choices = build_choice_panel(frozen_choices)
    choices.to_csv(output_dir / "strategy_choices.csv", index=False)

    ladder_config = LadderConfig(
        maximum_assets=maximum_assets,
        fallback_half_spread_bps=fallback_half_spread_bps,
        impact_coefficient=impact_coefficient,
        aum_eur=aum_values,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_blocks=bootstrap_blocks,
        random_state=random_state,
        hac_lags=hac_lags,
    )
    panel = load_ladder_panel(
        panel_path,
        predictions_path,
        liquidity_path,
        risk_path,
        ladder_config,
    )
    monthly, failures = simulate_constrained(
        panel,
        choices,
        specs,
        maximum_assets=maximum_assets,
        aum_values=aum_values,
        impact_coefficient=impact_coefficient,
    )
    if failures.empty:
        failures = pd.DataFrame(
            columns=["strategy", "date", "model", "rung", "constraint", "reason"]
        )
    summary = summarize_constrained(monthly, aum_values)
    concentration = summarize_concentration(monthly)
    inference = infer_vs_constrained_momentum(
        monthly,
        aum_values=aum_values,
        blocks=bootstrap_blocks,
        n_boot=bootstrap_repetitions,
        seed=random_state,
        hac_lags=hac_lags,
    )

    monthly.to_parquet(output_dir / "constrained_monthly.parquet", index=False, compression="zstd")
    monthly.to_csv(output_dir / "constrained_monthly.csv", index=False)
    summary.to_csv(output_dir / "constrained_summary.csv", index=False)
    concentration.to_csv(output_dir / "concentration_summary.csv", index=False)
    inference.to_csv(output_dir / "constrained_inference.csv", index=False)
    failures.to_csv(output_dir / "constraint_failures.csv", index=False)

    manifest = {
        "inputs": {
            "selected": str(selected_path),
            "candidate_predictions": str(predictions_path),
            "panel": str(panel_path),
            "liquidity": str(liquidity_path) if liquidity_path is not None else None,
            "risk": str(risk_path) if risk_path is not None else None,
        },
        "constraints": [asdict(spec) for spec in specs],
        "aum_values": aum_values,
        "maximum_assets": maximum_assets,
        "fallback_half_spread_bps": fallback_half_spread_bps,
        "impact_coefficient": impact_coefficient,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_blocks": bootstrap_blocks,
        "rows": {
            "panel": int(len(panel)),
            "choices": int(len(choices)),
            "monthly": int(len(monthly)),
            "summary": int(len(summary)),
            "concentration": int(len(concentration)),
            "inference": int(len(inference)),
            "failures": int(len(failures)),
        },
        "outputs": {
            "strategy_choices": str(output_dir / "strategy_choices.csv"),
            "constrained_monthly": str(output_dir / "constrained_monthly.csv"),
            "constrained_summary": str(output_dir / "constrained_summary.csv"),
            "concentration_summary": str(output_dir / "concentration_summary.csv"),
            "constrained_inference": str(output_dir / "constrained_inference.csv"),
            "constraint_failures": str(output_dir / "constraint_failures.csv"),
        },
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--candidate-predictions", type=Path, default=DEFAULT_CANDIDATE_PREDICTIONS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-assets", type=int, default=500)
    parser.add_argument("--aum-eur", nargs="+", type=float, default=[10_000_000.0, 100_000_000.0, 500_000_000.0])
    parser.add_argument("--fallback-half-spread-bps", type=float, default=25.0)
    parser.add_argument("--impact-coefficient", type=float, default=0.10)
    parser.add_argument("--constraint", action="append", default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--bootstrap-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    liquidity = args.liquidity if args.liquidity.exists() else None
    risk = args.risk if args.risk.exists() else None
    manifest = run_experiment(
        selected_path=args.selected,
        predictions_path=args.candidate_predictions,
        panel_path=args.panel,
        liquidity_path=liquidity,
        risk_path=risk,
        output_dir=args.output_dir,
        specs=parse_constraint_specs(args.constraint),
        aum_values=tuple(args.aum_eur),
        maximum_assets=args.maximum_assets,
        fallback_half_spread_bps=args.fallback_half_spread_bps,
        impact_coefficient=args.impact_coefficient,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_blocks=tuple(args.bootstrap_blocks),
        random_state=args.random_state,
        hac_lags=args.hac_lags,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
