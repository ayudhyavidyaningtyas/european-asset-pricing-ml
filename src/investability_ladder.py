"""Frozen-signal investability ladder with security-level execution costs."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import stats as project_stats
from asset_pricing_ml import _portfolio_weights
from implementable_frontier import (
    FrontierConfig,
    attach_execution_inputs,
    execution_cost,
    load_monthly_liquidity,
)


RUNGS = (
    "standard_ex_bottom_5pct",
    "top_70pct_by_market_cap",
    "top_500",
    "top_500_observed_spread",
    "large_low_spread",
)


@dataclass(frozen=True)
class LadderConfig:
    portfolio_quantile: float = 0.10
    maximum_assets: int = 500
    fallback_half_spread_bps: float = 25.0
    impact_coefficient: float = 0.10
    aum_eur: tuple[float, ...] = (
        10_000_000.0,
        100_000_000.0,
        500_000_000.0,
    )
    baseline_model: str = "momentum_rank"
    ce_risk_aversion: float = 3.0
    bootstrap_repetitions: int = 5_000
    bootstrap_blocks: tuple[int, ...] = (3, 6, 12)
    hac_lags: int = 6
    random_state: int = 42


def load_ladder_panel(
    panel_path: Path,
    predictions_path: Path,
    liquidity_path: Path | None,
    risk_path: Path | None = None,
    config: LadderConfig = LadderConfig(),
) -> pd.DataFrame:
    predictions = pd.read_parquet(predictions_path)
    predictions = predictions.dropna(subset=["prediction"]).copy()
    panel = pd.read_parquet(
        panel_path,
        columns=[
            "date",
            "ric",
            "company_market_cap",
            "market_cap_percentile",
            "turnover_12m",
            "volatility_12m",
        ],
    )
    merged = predictions.merge(
        panel,
        on=["date", "ric"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_panel"),
    )
    for column in ["company_market_cap", "market_cap_percentile"]:
        panel_column = f"{column}_panel"
        if panel_column in merged:
            merged[column] = merged[column].fillna(merged[panel_column])
            merged = merged.drop(columns=panel_column)

    merged["idio_vol_36m"] = (
        pd.to_numeric(merged["volatility_12m"], errors="coerce")
        * np.sqrt(12.0)
    ).clip(lower=0.02, upper=0.75)
    if risk_path is not None and risk_path.exists():
        risk = pd.read_parquet(
            risk_path,
            columns=["date", "ric", "idio_vol_36m"],
        ).rename(columns={"idio_vol_36m": "risk_idio_vol_36m"})
        merged = merged.merge(
            risk,
            on=["date", "ric"],
            how="left",
            validate="many_to_one",
        )
        merged["idio_vol_36m"] = merged[
            "risk_idio_vol_36m"
        ].fillna(merged["idio_vol_36m"])
        merged = merged.drop(columns="risk_idio_vol_36m")
    merged["idio_vol_36m"] = merged["idio_vol_36m"].fillna(0.20)

    liquidity = load_monthly_liquidity(liquidity_path)
    execution_config = FrontierConfig(
        fallback_half_spread_bps=config.fallback_half_spread_bps,
        impact_coefficient=config.impact_coefficient,
        aum_eur=config.aum_eur,
    )
    merged["beta_36m"] = np.nan
    return attach_execution_inputs(merged, liquidity, execution_config)


def investability_rungs(
    month: pd.DataFrame,
    maximum_assets: int,
) -> dict[str, pd.DataFrame]:
    standard = month[
        month["market_cap_percentile"].ge(0.05)
        & month["company_market_cap"].gt(0)
    ].copy()
    top_70 = standard[
        standard["market_cap_percentile"].ge(0.30)
    ].copy()
    top_500 = top_70.nlargest(maximum_assets, "company_market_cap").copy()
    observed = top_500[top_500["spread_observed"]].copy()
    if observed.empty:
        low_spread = observed.copy()
    else:
        cutoff = observed["half_spread_bps"].median()
        low_spread = observed[observed["half_spread_bps"].le(cutoff)].copy()
    return {
        "standard_ex_bottom_5pct": standard,
        "top_70pct_by_market_cap": top_70,
        "top_500": top_500,
        "top_500_observed_spread": observed,
        "large_low_spread": low_spread,
    }


def _cost_for_transition(
    prior_weights: dict[str, float],
    current_weights: dict[str, float],
    current_inputs: dict[str, tuple[float, float, float]],
    prior_inputs: dict[str, tuple[float, float, float]],
    aum_eur: float,
    config: LadderConfig,
) -> tuple[float, float, float, float]:
    names = sorted(set(prior_weights) | set(current_weights))
    delta = np.array(
        [
            current_weights.get(name, 0.0) - prior_weights.get(name, 0.0)
            for name in names
        ],
        dtype=float,
    )
    inputs = [
        current_inputs.get(name, prior_inputs.get(name))
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
        aum_eur,
        config.impact_coefficient,
    )
    return float(np.abs(delta).sum()), spread, impact, total


def simulate_investability_ladder(
    panel: pd.DataFrame,
    config: LadderConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    prior_weights: dict[tuple[str, str, str], dict[str, float]] = {}
    prior_inputs: dict[tuple[str, str, str], dict[str, tuple[float, float, float]]] = {}
    grouped = panel.sort_values(["model", "date", "ric"]).groupby(
        ["model", "date"], sort=True
    )
    for (model, date), month in grouped:
        target_mode = str(month["target_mode"].iloc[0])
        for rung, universe in investability_rungs(
            month,
            config.maximum_assets,
        ).items():
            investable = universe.dropna(
                subset=["prediction", "target_return_1m"]
            )
            if len(investable) < 20:
                continue
            current_inputs = {
                row.ric: (
                    float(row.half_spread_bps),
                    float(row.adv_eur),
                    float(row.idio_vol_36m),
                )
                for row in investable.itertuples()
            }
            returns = investable.set_index("ric")["target_return_1m"]
            for weighting in ["equal", "value"]:
                long_short, long_n, short_n = _portfolio_weights(
                    investable,
                    config.portfolio_quantile,
                    weighting,
                )
                if not long_short:
                    continue
                long_only = {
                    ric: weight
                    for ric, weight in long_short.items()
                    if weight > 0
                }
                for portfolio, weights in [
                    ("long_short", long_short),
                    ("long_only", long_only),
                ]:
                    weight_series = pd.Series(weights, dtype=float)
                    gross_return = float(
                        (
                            weight_series
                            * returns.reindex(weight_series.index)
                        ).sum()
                    )
                    key = (model, rung, f"{weighting}_{portfolio}")
                    previous = prior_weights.get(key, {})
                    previous_inputs = prior_inputs.get(key, {})
                    row: dict[str, Any] = {
                        "model": model,
                        "target_mode": target_mode,
                        "date": date,
                        "target_date": investable["target_date"].iloc[0],
                        "rung": rung,
                        "weighting": weighting,
                        "portfolio": portfolio,
                        "universe_n": int(len(investable)),
                        "long_n": long_n,
                        "short_n": short_n if portfolio == "long_short" else 0,
                        "gross_return": gross_return,
                        "observed_spread_fraction": float(
                            investable["spread_observed"].mean()
                        ),
                        "median_half_spread_bps": float(
                            investable["half_spread_bps"].median()
                        ),
                    }
                    for aum in config.aum_eur:
                        label = f"{int(round(aum / 1_000_000.0))}m"
                        turnover, spread, impact, total = _cost_for_transition(
                            previous,
                            weights,
                            current_inputs,
                            previous_inputs,
                            aum,
                            config,
                        )
                        row[f"turnover_{label}"] = turnover
                        row[f"spread_cost_{label}"] = spread
                        row[f"impact_cost_{label}"] = impact
                        row[f"net_return_{label}"] = gross_return - total
                    records.append(row)
                    prior_weights[key] = weights
                    prior_inputs[key] = current_inputs
    return pd.DataFrame(records)


def summarize_investability_ladder(
    monthly: pd.DataFrame,
    config: LadderConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = [
        "model",
        "target_mode",
        "rung",
        "weighting",
        "portfolio",
    ]
    for keys, group in monthly.groupby(group_columns, sort=True):
        base = dict(zip(group_columns, keys, strict=True))
        gross_vol = float(group["gross_return"].std(ddof=1) * np.sqrt(12))
        common = {
            **base,
            "months": int(len(group)),
            "average_universe_n": float(group["universe_n"].mean()),
            "gross_annualized_return": float(group["gross_return"].mean() * 12),
            "gross_annualized_volatility": gross_vol,
            "gross_sharpe": (
                float(group["gross_return"].mean() / group["gross_return"].std(ddof=1) * np.sqrt(12))
                if group["gross_return"].std(ddof=1) > 0
                else np.nan
            ),
            "observed_spread_fraction": float(
                group["observed_spread_fraction"].mean()
            ),
            "median_half_spread_bps": float(
                group["median_half_spread_bps"].median()
            ),
        }
        for aum in config.aum_eur:
            label = f"{int(round(aum / 1_000_000.0))}m"
            net = group[f"net_return_{label}"]
            records.append(
                {
                    **common,
                    "aum_eur": aum,
                    "aum_label": label,
                    "annualized_net_return": float(net.mean() * 12),
                    "annualized_net_volatility": float(
                        net.std(ddof=1) * np.sqrt(12)
                    ),
                    "net_sharpe": (
                        float(net.mean() / net.std(ddof=1) * np.sqrt(12))
                        if net.std(ddof=1) > 0
                        else np.nan
                    ),
                    "average_monthly_turnover": float(
                        group[f"turnover_{label}"].mean()
                    ),
                    "annualized_spread_cost": float(
                        group[f"spread_cost_{label}"].mean() * 12
                    ),
                    "annualized_impact_cost": float(
                        group[f"impact_cost_{label}"].mean() * 12
                    ),
                }
            )
    return pd.DataFrame(records)


def _certainty_equivalent(
    returns: np.ndarray,
    risk_aversion: float,
) -> float:
    if len(returns) < 2:
        return np.nan
    annual_mean = float(np.mean(returns) * 12.0)
    annual_vol = float(np.std(returns, ddof=1) * np.sqrt(12.0))
    return annual_mean - 0.5 * risk_aversion * annual_vol**2


def _bootstrap_ce_diff(
    model_returns: pd.Series,
    baseline_returns: pd.Series,
    risk_aversion: float,
    expected_block: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    model_values = model_returns.to_numpy(dtype=float)
    baseline_values = baseline_returns.to_numpy(dtype=float)
    n = len(model_values)
    point = _certainty_equivalent(
        model_values,
        risk_aversion,
    ) - _certainty_equivalent(baseline_values, risk_aversion)
    indices = project_stats.stationary_bootstrap_indices(
        n,
        expected_block,
        n_boot,
        np.random.default_rng(seed),
    )
    model_sample = model_values[indices]
    baseline_sample = baseline_values[indices]
    model_mean = model_sample.mean(axis=1) * 12.0
    baseline_mean = baseline_sample.mean(axis=1) * 12.0
    model_vol = model_sample.std(axis=1, ddof=1) * np.sqrt(12.0)
    baseline_vol = baseline_sample.std(axis=1, ddof=1) * np.sqrt(12.0)
    differences = (
        model_mean
        - 0.5 * risk_aversion * model_vol**2
        - baseline_mean
        + 0.5 * risk_aversion * baseline_vol**2
    )
    ci_low, ci_high = np.quantile(differences, [0.025, 0.975])
    p_two = min(
        1.0,
        2.0
        * min(
            float((differences >= 0).mean()),
            float((differences <= 0).mean()),
        ),
    )
    return {
        "delta_ce": float(point),
        "ce_ci_low": float(ci_low),
        "ce_ci_high": float(ci_high),
        "ce_p_two_sided": float(p_two),
    }


def paired_ladder_inference(
    monthly: pd.DataFrame,
    config: LadderConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    date_column = "target_date" if "target_date" in monthly else "date"
    setting_columns = ["rung", "weighting", "portfolio"]
    comparison_models = [
        model
        for model in sorted(monthly["model"].dropna().unique())
        if model != config.baseline_model
    ]
    if not comparison_models:
        return pd.DataFrame()

    for setting, setting_frame in monthly.groupby(setting_columns, sort=True):
        baseline = setting_frame[
            setting_frame["model"].eq(config.baseline_model)
        ].set_index(date_column)
        if baseline.empty:
            continue
        for aum in config.aum_eur:
            aum_label = f"{int(round(aum / 1_000_000.0))}m"
            return_column = f"net_return_{aum_label}"
            if return_column not in setting_frame:
                continue
            baseline_returns = baseline[return_column].dropna()
            for model_name in comparison_models:
                model = setting_frame[
                    setting_frame["model"].eq(model_name)
                ].set_index(date_column)
                dates = baseline_returns.index.intersection(
                    model[return_column].dropna().index
                )
                if len(dates) < 24:
                    continue
                model_returns = model.loc[dates, return_column].astype(float)
                comparison_returns = baseline_returns.reindex(dates).astype(float)
                mean_test = project_stats.hac_mean_diff_test(
                    model_returns - comparison_returns,
                    maxlags=config.hac_lags,
                )
                model_ce = _certainty_equivalent(
                    model_returns.to_numpy(dtype=float),
                    config.ce_risk_aversion,
                )
                baseline_ce = _certainty_equivalent(
                    comparison_returns.to_numpy(dtype=float),
                    config.ce_risk_aversion,
                )
                for block in config.bootstrap_blocks:
                    sharpe = project_stats.bootstrap_sharpe_diff(
                        model_returns,
                        comparison_returns,
                        np.zeros(len(dates)),
                        expected_block=block,
                        n_boot=config.bootstrap_repetitions,
                        seed=config.random_state,
                    )
                    ce = _bootstrap_ce_diff(
                        model_returns,
                        comparison_returns,
                        config.ce_risk_aversion,
                        block,
                        config.bootstrap_repetitions,
                        config.random_state,
                    )
                    records.append(
                        {
                            "model": model_name,
                            "baseline": config.baseline_model,
                            "target_mode": str(model["target_mode"].iloc[0]),
                            "baseline_target_mode": str(
                                baseline["target_mode"].iloc[0]
                            ),
                            **dict(zip(setting_columns, setting, strict=True)),
                            "aum_eur": float(aum),
                            "aum_label": aum_label,
                            "return_column": return_column,
                            "months": int(len(dates)),
                            "model_annualized_net_return": float(
                                model_returns.mean() * 12.0
                            ),
                            "baseline_annualized_net_return": float(
                                comparison_returns.mean() * 12.0
                            ),
                            "delta_annualized_net_return": float(
                                mean_test["mean"] * 12.0
                            ),
                            "hac_standard_error_annualized": float(
                                mean_test["se"] * 12.0
                            ),
                            "hac_t_stat": float(mean_test["t"]),
                            "hac_p_two_sided": float(
                                mean_test["p_two_sided"]
                            ),
                            "model_certainty_equivalent": float(model_ce),
                            "baseline_certainty_equivalent": float(
                                baseline_ce
                            ),
                            "ce_risk_aversion": float(
                                config.ce_risk_aversion
                            ),
                            **sharpe,
                            **ce,
                        }
                    )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    family = [
        "rung",
        "weighting",
        "portfolio",
        "aum_label",
        "expected_block",
    ]
    result["sharpe_p_two_sided_holm"] = result.groupby(family)[
        "p_two_sided"
    ].transform(lambda values: multipletests(values, method="holm")[1])
    result["ce_p_two_sided_holm"] = result.groupby(family)[
        "ce_p_two_sided"
    ].transform(lambda values: multipletests(values, method="holm")[1])
    result["hac_p_two_sided_holm"] = result.groupby(family)[
        "hac_p_two_sided"
    ].transform(lambda values: multipletests(values, method="holm")[1])
    result["primary_family"] = (
        result["weighting"].eq("value")
        & result["portfolio"].eq("long_only")
        & result["aum_label"].eq("100m")
        & result["expected_block"].eq(6)
    )
    return result


def predictive_metrics_by_rung(
    panel: pd.DataFrame,
    config: LadderConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model, model_panel in panel.groupby("model", sort=True):
        target_mode = str(model_panel["target_mode"].iloc[0])
        rung_frames = []
        for _, month in model_panel.groupby("date", sort=True):
            for rung, universe in investability_rungs(
                month,
                config.maximum_assets,
            ).items():
                if not universe.empty:
                    rung_frames.append(universe.assign(rung=rung))
        combined = pd.concat(rung_frames, ignore_index=True)
        for rung, group in combined.groupby("rung", sort=True):
            group = group.dropna(
                subset=["prediction", "target_return_1m", "target_return_rank"]
            )
            target = (
                group["target_return_1m"]
                if target_mode == "return"
                else group["target_return_rank"]
            )
            denominator = float(np.square(target).sum())
            r2 = (
                1.0
                - float(np.square(target - group["prediction"]).sum())
                / denominator
                if denominator > 0
                else np.nan
            )
            monthly_ic = group.groupby("date").apply(
                lambda month: month["prediction"].corr(
                    month["target_return_rank"], method="spearman"
                ),
                include_groups=False,
            )
            records.append(
                {
                    "model": model,
                    "target_mode": target_mode,
                    "rung": rung,
                    "observations": int(len(group)),
                    "months": int(group["date"].nunique()),
                    "target_r2_zero": r2,
                    "mean_monthly_spearman_ic": float(monthly_ic.mean()),
                }
            )
    return pd.DataFrame(records)


def write_ladder_outputs(
    output_dir: Path,
    config: LadderConfig,
    monthly: pd.DataFrame,
    summary: pd.DataFrame,
    predictive: pd.DataFrame,
    inference: pd.DataFrame | None,
    inputs: dict[str, Path | None],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(
        output_dir / "investability_ladder_monthly.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_csv(output_dir / "investability_ladder_summary.csv", index=False)
    predictive.to_csv(
        output_dir / "investability_ladder_predictive_metrics.csv",
        index=False,
    )
    if inference is not None:
        inference.to_csv(
            output_dir / "investability_ladder_inference.csv",
            index=False,
        )
    manifest = {
        "config": asdict(config),
        "rungs": list(RUNGS),
        "design": {
            "primary": "freeze full-universe predictions and resimulate each rung",
            "costs": "same security-level spread and square-root-impact function at every rung",
            "inference": "paired HAC net-return tests plus stationary-bootstrap Sharpe and CE differences",
            "causal_claim": False,
        },
        "inputs": {
            key: str(value) if value is not None else None
            for key, value in inputs.items()
        },
        "rows": {
            "monthly": int(len(monthly)),
            "summary": int(len(summary)),
            "predictive": int(len(predictive)),
            "inference": int(len(inference)) if inference is not None else 0,
        },
    }
    (output_dir / "investability_ladder_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
