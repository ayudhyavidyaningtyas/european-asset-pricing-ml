"""Robustness checks for the frozen deep/hybrid liquid long-only selector."""
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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stats as project_stats  # noqa: E402
from asset_pricing_ml import _portfolio_weights  # noqa: E402
from investability_ladder import (  # noqa: E402
    LadderConfig,
    investability_rungs,
    load_ladder_panel,
    simulate_investability_ladder,
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
    / "frozen_deep_hybrid_long_only_robustness"
)
DEFAULT_BASELINES = [
    ("fixed_momentum_top500_observed", "momentum_rank", "top_500_observed_spread"),
    ("fixed_ridge_top500_observed", "ridge_rank", "top_500_observed_spread"),
    ("fixed_smooth75_ridge_top500_observed", "smooth75_ridge_rank", "top_500_observed_spread"),
    ("fixed_blend90_gbm_attn24_large_low", "blend90_gbm_attn_seq24_rank", "large_low_spread"),
]
SUBPERIODS = [
    ("full", None, None),
    ("pre_covid_2017_2019", "2017-01-01", "2019-12-31"),
    ("covid_recovery_2020_2022", "2020-01-01", "2022-12-31"),
    ("recent_2023_2026", "2023-01-01", "2026-12-31"),
]


def aum_label(aum: float) -> str:
    return f"{int(round(aum / 1_000_000.0))}m"


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return float(wealth.div(peak).sub(1.0).min())


def summarize_returns(
    monthly: pd.DataFrame,
    *,
    strategy: str,
    portfolio: str,
    aum: float,
    subperiod: str,
) -> dict[str, Any]:
    label = aum_label(aum)
    return_column = f"net_return_{label}"
    turnover_column = f"turnover_{label}"
    spread_column = f"spread_cost_{label}"
    impact_column = f"impact_cost_{label}"
    returns = monthly[return_column].astype(float)
    volatility = float(returns.std(ddof=1) * np.sqrt(12.0))
    return {
        "strategy": strategy,
        "portfolio": portfolio,
        "subperiod": subperiod,
        "aum_eur": float(aum),
        "aum_label": label,
        "months": int(len(monthly)),
        "annualized_net_return": float(returns.mean() * 12.0),
        "annualized_net_volatility": volatility,
        "net_sharpe": (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(12.0))
            if returns.std(ddof=1) > 0
            else np.nan
        ),
        "max_drawdown": max_drawdown(returns),
        "average_monthly_turnover": float(monthly[turnover_column].mean()),
        "max_monthly_turnover": float(monthly[turnover_column].max()),
        "annualized_spread_cost": float(monthly[spread_column].mean() * 12.0),
        "annualized_impact_cost": float(monthly[impact_column].mean() * 12.0),
        "average_observed_spread_fraction": float(
            monthly["observed_spread_fraction"].mean()
        ),
        "minimum_observed_spread_fraction": float(
            monthly["observed_spread_fraction"].min()
        ),
        "average_universe_n": float(monthly["universe_n"].mean()),
    }


def subperiod_frame(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    work = frame.copy()
    target_date = pd.to_datetime(work["target_date"])
    if start is not None:
        work = work[target_date.ge(pd.Timestamp(start))]
        target_date = pd.to_datetime(work["target_date"])
    if end is not None:
        work = work[target_date.le(pd.Timestamp(end))]
    return work


def load_selected_choices(path: Path, strategy: str, portfolio: str) -> pd.DataFrame:
    selected = pd.read_csv(path, parse_dates=["date", "target_date", "selection_signal_date"])
    selected = selected[
        selected["strategy"].eq(strategy)
        & selected["selected_portfolio"].eq(portfolio)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No selected rows for {strategy}/{portfolio}")
    return selected[
        ["date", "target_date", "model", "rung", "selected_portfolio", "strategy"]
    ].drop_duplicates(["date"])


def selected_from_ladder(
    ladder: pd.DataFrame,
    choices: pd.DataFrame,
    *,
    portfolio: str,
) -> pd.DataFrame:
    work = ladder[
        ladder["weighting"].eq("value")
        & ladder["portfolio"].eq(portfolio)
    ].copy()
    merged = choices.merge(
        work,
        on=["date", "target_date", "model", "rung"],
        how="left",
        validate="one_to_one",
    )
    missing = merged["gross_return"].isna().sum()
    if missing:
        raise RuntimeError(f"Selected choices missing ladder rows: {missing}")
    merged["robustness_strategy"] = "frozen_deep_hybrid_selector"
    return merged


def fixed_baseline_from_ladder(
    ladder: pd.DataFrame,
    *,
    name: str,
    model: str,
    rung: str,
    portfolio: str,
    selected_dates: pd.Series,
) -> pd.DataFrame:
    baseline = ladder[
        ladder["model"].eq(model)
        & ladder["rung"].eq(rung)
        & ladder["weighting"].eq("value")
        & ladder["portfolio"].eq(portfolio)
        & ladder["date"].isin(set(pd.to_datetime(selected_dates)))
    ].copy()
    baseline["robustness_strategy"] = name
    return baseline


def infer_vs_baselines(
    selected: pd.DataFrame,
    baselines: pd.DataFrame,
    *,
    aum_values: tuple[float, ...],
    blocks: tuple[int, ...],
    n_boot: int,
    seed: int,
    hac_lags: int,
) -> pd.DataFrame:
    records = []
    for baseline_name, baseline in baselines.groupby("robustness_strategy", sort=True):
        for aum in aum_values:
            label = aum_label(aum)
            column = f"net_return_{label}"
            left = selected.set_index("target_date")[column].astype(float)
            right = baseline.set_index("target_date")[column].astype(float)
            dates = left.index.intersection(right.index)
            if len(dates) < 24:
                continue
            left = left.reindex(dates)
            right = right.reindex(dates)
            mean_test = project_stats.hac_mean_diff_test(
                left - right,
                maxlags=hac_lags,
            )
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
                        "baseline": baseline_name,
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


def concentration_for_choices(
    panel: pd.DataFrame,
    choices: pd.DataFrame,
    *,
    maximum_assets: int,
    portfolio_quantile: float,
) -> pd.DataFrame:
    keys = choices[["date", "model", "rung", "target_date"]].drop_duplicates()
    work = panel.merge(keys[["date", "model"]].drop_duplicates(), on=["date", "model"], how="inner")
    rung_lookup = {
        (row.model, row.date): row.rung
        for row in keys.itertuples()
    }
    target_lookup = {
        (row.model, row.date): row.target_date
        for row in keys.itertuples()
    }
    records = []
    for (model, date), month in work.groupby(["model", "date"], sort=True):
        rung = rung_lookup[(model, date)]
        universe = investability_rungs(month, maximum_assets)[rung]
        universe = universe.dropna(subset=["prediction", "target_return_1m"])
        weights, long_n, _ = _portfolio_weights(universe, portfolio_quantile, "value")
        if not weights:
            continue
        weight_series = pd.Series(weights, dtype=float)
        long_weights = weight_series[weight_series.gt(0)]
        holdings = universe.set_index("ric").reindex(long_weights.index).copy()
        holdings["weight"] = long_weights
        country = holdings.groupby("screen_country", dropna=False)["weight"].sum()
        sector = holdings.groupby("TR.TRBCECONOMICSECTOR", dropna=False)["weight"].sum()
        records.append(
            {
                "date": date,
                "target_date": target_lookup[(model, date)],
                "model": model,
                "rung": rung,
                "long_n": int(long_n),
                "weight_sum": float(long_weights.sum()),
                "effective_n": float(1.0 / np.square(long_weights).sum()),
                "max_single_name_weight": float(long_weights.max()),
                "top_5_name_weight": float(long_weights.sort_values(ascending=False).head(5).sum()),
                "max_country": str(country.idxmax()) if not country.empty else "",
                "max_country_weight": float(country.max()) if not country.empty else np.nan,
                "country_hhi": float(np.square(country).sum()) if not country.empty else np.nan,
                "max_sector": str(sector.idxmax()) if not sector.empty else "",
                "max_sector_weight": float(sector.max()) if not sector.empty else np.nan,
                "sector_hhi": float(np.square(sector).sum()) if not sector.empty else np.nan,
                "missing_country_weight": float(
                    holdings.loc[holdings["screen_country"].isna(), "weight"].sum()
                ),
                "missing_sector_weight": float(
                    holdings.loc[
                        holdings["TR.TRBCECONOMICSECTOR"].isna(),
                        "weight",
                    ].sum()
                ),
            }
        )
    return pd.DataFrame(records)


def summarize_concentration(monthly: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "effective_n",
        "max_single_name_weight",
        "top_5_name_weight",
        "max_country_weight",
        "country_hhi",
        "max_sector_weight",
        "sector_hhi",
        "missing_country_weight",
        "missing_sector_weight",
    ]
    records = []
    for metric in metrics:
        values = monthly[metric].dropna()
        records.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(records)


def fallback_sensitivity(selected: pd.DataFrame, fallback_grid: tuple[float, ...], aum_values: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    min_coverage = float(selected["observed_spread_fraction"].min())
    all_observed = bool(min_coverage >= 1.0)
    for fallback in fallback_grid:
        for aum in aum_values:
            base = summarize_returns(
                selected,
                strategy="frozen_deep_hybrid_selector",
                portfolio="long_only",
                aum=aum,
                subperiod="full",
            )
            rows.append(
                {
                    **base,
                    "fallback_half_spread_bps": float(fallback),
                    "fallback_used": not all_observed,
                    "reason": (
                        "all selected rows use observed spreads"
                        if all_observed
                        else "selected rows include fallback spreads"
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_robustness(
    selected_path: Path,
    predictions_path: Path,
    panel_path: Path,
    liquidity_path: Path | None,
    risk_path: Path | None,
    output_dir: Path,
    maximum_assets: int,
    aum_values: tuple[float, ...],
    fallback_grid: tuple[float, ...],
    bootstrap_repetitions: int,
    bootstrap_blocks: tuple[int, ...],
    random_state: int,
    hac_lags: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    choices = load_selected_choices(
        selected_path,
        strategy="validation_selected_long_only",
        portfolio="long_only",
    )
    config = LadderConfig(
        maximum_assets=maximum_assets,
        portfolio_quantile=0.10,
        aum_eur=aum_values,
        fallback_half_spread_bps=25.0,
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
        config,
    )
    ladder = simulate_investability_ladder(panel, config)
    ladder.to_parquet(output_dir / "robustness_ladder_monthly.parquet", index=False, compression="zstd")

    selected = selected_from_ladder(ladder, choices, portfolio="long_only")
    selected.to_csv(output_dir / "frozen_selected_monthly.csv", index=False)
    baseline_frames = [
        fixed_baseline_from_ladder(
            ladder,
            name=name,
            model=model,
            rung=rung,
            portfolio="long_only",
            selected_dates=choices["date"],
        )
        for name, model, rung in DEFAULT_BASELINES
    ]
    baselines = pd.concat(baseline_frames, ignore_index=True)
    baselines.to_csv(output_dir / "frozen_baseline_monthly.csv", index=False)

    summary_records = []
    frames = [selected, *[frame for _, frame in baselines.groupby("robustness_strategy", sort=True)]]
    for frame in frames:
        strategy = str(frame["robustness_strategy"].iloc[0])
        for aum in aum_values:
            for subperiod, start, end in SUBPERIODS:
                part = subperiod_frame(frame, start, end)
                if len(part) < 6:
                    continue
                summary_records.append(
                    summarize_returns(
                        part,
                        strategy=strategy,
                        portfolio="long_only",
                        aum=aum,
                        subperiod=subperiod,
                    )
                )
    summary = pd.DataFrame(summary_records)
    summary.to_csv(output_dir / "frozen_robustness_summary.csv", index=False)

    inference = infer_vs_baselines(
        selected,
        baselines,
        aum_values=aum_values,
        blocks=bootstrap_blocks,
        n_boot=bootstrap_repetitions,
        seed=random_state,
        hac_lags=hac_lags,
    )
    inference.to_csv(output_dir / "frozen_robustness_inference.csv", index=False)

    concentration = concentration_for_choices(
        panel,
        choices,
        maximum_assets=config.maximum_assets,
        portfolio_quantile=config.portfolio_quantile,
    )
    concentration.to_csv(output_dir / "frozen_selected_concentration_monthly.csv", index=False)
    concentration_summary = summarize_concentration(concentration)
    concentration_summary.to_csv(output_dir / "frozen_selected_concentration_summary.csv", index=False)

    fallback = fallback_sensitivity(selected, fallback_grid, aum_values)
    fallback.to_csv(output_dir / "fallback_spread_sensitivity.csv", index=False)

    turnover_spikes = selected.sort_values("turnover_100m", ascending=False).head(15)
    turnover_spikes.to_csv(output_dir / "turnover_spike_months.csv", index=False)

    manifest = {
        "inputs": {
            "selected": str(selected_path),
            "candidate_predictions": str(predictions_path),
            "panel": str(panel_path),
            "liquidity": str(liquidity_path) if liquidity_path is not None else None,
            "risk": str(risk_path) if risk_path is not None else None,
        },
        "aum_values": aum_values,
        "maximum_assets": maximum_assets,
        "fallback_grid_bps": fallback_grid,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_blocks": bootstrap_blocks,
        "random_state": random_state,
        "rows": {
            "panel": int(len(panel)),
            "ladder_monthly": int(len(ladder)),
            "selected_monthly": int(len(selected)),
            "baseline_monthly": int(len(baselines)),
            "summary": int(len(summary)),
            "inference": int(len(inference)),
            "concentration_monthly": int(len(concentration)),
        },
        "outputs": {
            "robustness_ladder_monthly": str(output_dir / "robustness_ladder_monthly.parquet"),
            "frozen_selected_monthly": str(output_dir / "frozen_selected_monthly.csv"),
            "frozen_baseline_monthly": str(output_dir / "frozen_baseline_monthly.csv"),
            "frozen_robustness_summary": str(output_dir / "frozen_robustness_summary.csv"),
            "frozen_robustness_inference": str(output_dir / "frozen_robustness_inference.csv"),
            "frozen_selected_concentration_monthly": str(
                output_dir / "frozen_selected_concentration_monthly.csv"
            ),
            "frozen_selected_concentration_summary": str(
                output_dir / "frozen_selected_concentration_summary.csv"
            ),
            "fallback_spread_sensitivity": str(output_dir / "fallback_spread_sensitivity.csv"),
            "turnover_spike_months": str(output_dir / "turnover_spike_months.csv"),
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
    parser.add_argument(
        "--aum-eur",
        nargs="+",
        type=float,
        default=[10_000_000.0, 100_000_000.0, 500_000_000.0],
    )
    parser.add_argument(
        "--fallback-half-spread-bps",
        nargs="+",
        type=float,
        default=[10.0, 25.0, 50.0],
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    liquidity = args.liquidity if args.liquidity.exists() else None
    risk = args.risk if args.risk.exists() else None
    manifest = run_robustness(
        selected_path=args.selected,
        predictions_path=args.candidate_predictions,
        panel_path=args.panel,
        liquidity_path=liquidity,
        risk_path=risk,
        output_dir=args.output_dir,
        maximum_assets=args.maximum_assets,
        aum_values=tuple(args.aum_eur),
        fallback_grid=tuple(args.fallback_half_spread_bps),
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
