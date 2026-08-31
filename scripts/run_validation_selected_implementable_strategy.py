"""Rolling validation-selected implementable deep/hybrid strategy.

The experiment keeps predictions frozen. It builds a costed investability
ladder for a pre-specified set of parent, sequence, blend and smoothed signals,
then chooses a model/rung cell each month using only prior completed returns.
"""
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
from investability_ladder import (  # noqa: E402
    LadderConfig,
    load_ladder_panel,
    simulate_investability_ladder,
    summarize_investability_ladder,
)


DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "turnover_aware_signal_smoothing"
    / "smoothed_with_parents_predictions.parquet"
)
DEFAULT_SMOOTHING_SPECS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "turnover_aware_signal_smoothing"
    / "signal_smoothing_specifications.csv"
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
    / "validation_selected_implementable_strategy"
)
DEFAULT_CORE_MODELS = [
    "momentum_rank",
    "ridge_rank",
    "dre_rank",
    "hist_gbm_rank",
    "mlp_rank",
    "attention_lstm_seq12_rank",
    "attention_lstm_seq24_rank",
    "gru_seq12_rank",
    "gru_seq24_rank",
    "blend90_mlp_attn_seq12_rank",
    "blend90_mlp_attn_seq24_rank",
    "blend90_gbm_attn_seq24_rank",
    "blend50_mom_gru_seq12_rank",
]
DEFAULT_RUNG_SET = [
    "top_500",
    "top_500_observed_spread",
    "large_low_spread",
]
DEFAULT_PORTFOLIOS = ["long_short", "long_only"]


@dataclass(frozen=True)
class SelectorConfig:
    validation_months: int = 36
    minimum_validation_months: int = 24
    risk_aversion: float = 3.0
    objective: str = "certainty_equivalent"
    weighting: str = "value"
    aum_label: str = "100m"
    hac_lags: int = 6
    bootstrap_repetitions: int = 2_000
    bootstrap_blocks: tuple[int, ...] = (3, 6, 12)
    random_state: int = 42


def _slug(model: str) -> str:
    return model.replace("attention_lstm", "attn").replace("_rank", "")


def build_candidate_model_list(
    core_models: list[str],
    smoothing_specs_path: Path,
    inertia_weights: tuple[float, ...],
) -> list[str]:
    candidates = list(core_models)
    if smoothing_specs_path.exists() and inertia_weights:
        specs = pd.read_csv(smoothing_specs_path)
        smooth = specs[
            specs["parent_model"].isin(core_models)
            & specs["inertia_weight"].isin(inertia_weights)
        ]["model"].tolist()
        candidates.extend(smooth)
    return sorted(dict.fromkeys(candidates))


def read_prediction_subset(path: Path, models: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        table = dataset.to_table(filter=ds.field("model").isin(models))
        predictions = table.to_pandas()
    except Exception:
        predictions = pd.read_parquet(path)
        predictions = predictions[predictions["model"].isin(models)].copy()
    found = set(predictions["model"].dropna().unique())
    missing = sorted(set(models) - found)
    if missing:
        raise RuntimeError(f"Missing candidate models in predictions: {missing}")
    duplicate_rows = int(predictions.duplicated(["model", "date", "ric"]).sum())
    if duplicate_rows:
        raise RuntimeError(f"Duplicate candidate model/date/ric predictions: {duplicate_rows}")
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    return predictions.sort_values(["model", "date", "ric"]).reset_index(drop=True)


def build_candidate_ladder(
    predictions_path: Path,
    smoothing_specs_path: Path,
    panel_path: Path,
    liquidity_path: Path | None,
    risk_path: Path | None,
    output_dir: Path,
    core_models: list[str],
    inertia_weights: tuple[float, ...],
    ladder_config: LadderConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], Path]:
    candidate_models = build_candidate_model_list(
        core_models,
        smoothing_specs_path,
        inertia_weights,
    )
    predictions = read_prediction_subset(predictions_path, candidate_models)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_predictions_path = output_dir / "candidate_predictions.parquet"
    predictions.to_parquet(
        candidate_predictions_path,
        index=False,
        compression="zstd",
    )
    panel = load_ladder_panel(
        panel_path,
        candidate_predictions_path,
        liquidity_path,
        risk_path,
        ladder_config,
    )
    monthly = simulate_investability_ladder(panel, ladder_config)
    summary = summarize_investability_ladder(monthly, ladder_config)
    monthly.to_parquet(
        output_dir / "candidate_ladder_monthly.parquet",
        index=False,
        compression="zstd",
    )
    summary.to_csv(output_dir / "candidate_ladder_summary.csv", index=False)
    return predictions, monthly, summary, candidate_models, candidate_predictions_path


def _certainty_equivalent(returns: pd.Series, risk_aversion: float) -> float:
    if len(returns) < 2:
        return np.nan
    annual_mean = float(returns.mean() * 12.0)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(12.0))
    return annual_mean - 0.5 * risk_aversion * annual_vol**2


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_max = wealth.cummax()
    return float(wealth.div(running_max).sub(1.0).min())


def validation_scores(
    validation_rows: pd.DataFrame,
    config: SelectorConfig,
    return_column: str,
    turnover_column: str,
) -> pd.DataFrame:
    records = []
    for keys, group in validation_rows.groupby(["model", "rung"], sort=True):
        returns = group[return_column].dropna()
        if len(returns) < config.minimum_validation_months:
            continue
        annual_mean = float(returns.mean() * 12.0)
        annual_vol = float(returns.std(ddof=1) * np.sqrt(12.0))
        sharpe = (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(12.0))
            if returns.std(ddof=1) > 0
            else np.nan
        )
        ce = _certainty_equivalent(returns, config.risk_aversion)
        objective_value = ce if config.objective == "certainty_equivalent" else sharpe
        records.append(
            {
                "model": keys[0],
                "rung": keys[1],
                "validation_months": int(len(returns)),
                "validation_annualized_return": annual_mean,
                "validation_annualized_volatility": annual_vol,
                "validation_sharpe": sharpe,
                "validation_certainty_equivalent": ce,
                "validation_objective": float(objective_value),
                "validation_average_turnover": float(group[turnover_column].mean()),
                "validation_observed_spread_fraction": float(
                    group["observed_spread_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def select_strategy_monthly(
    monthly: pd.DataFrame,
    *,
    portfolio: str,
    rungs: list[str],
    candidate_models: list[str],
    config: SelectorConfig,
) -> pd.DataFrame:
    return_column = f"net_return_{config.aum_label}"
    turnover_column = f"turnover_{config.aum_label}"
    if return_column not in monthly:
        raise RuntimeError(f"Missing return column: {return_column}")
    eligible = monthly[
        monthly["model"].isin(candidate_models)
        & monthly["rung"].isin(rungs)
        & monthly["weighting"].eq(config.weighting)
        & monthly["portfolio"].eq(portfolio)
    ].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    eligible["target_date"] = pd.to_datetime(eligible["target_date"])
    records = []
    for signal_date, current in eligible.groupby("date", sort=True):
        validation_start = signal_date - pd.DateOffset(months=config.validation_months)
        validation = eligible[
            eligible["target_date"].le(signal_date)
            & eligible["target_date"].gt(validation_start)
        ]
        scores = validation_scores(validation, config, return_column, turnover_column)
        if scores.empty:
            continue
        scores = scores.sort_values(
            [
                "validation_objective",
                "validation_months",
                "validation_observed_spread_fraction",
                "validation_average_turnover",
            ],
            ascending=[False, False, False, True],
        )
        selected = scores.iloc[0]
        row = current[
            current["model"].eq(selected["model"])
            & current["rung"].eq(selected["rung"])
        ]
        if row.empty:
            continue
        realised = row.iloc[0].to_dict()
        for key, value in selected.items():
            realised[key] = value
        realised["selection_signal_date"] = signal_date
        realised["selected_portfolio"] = portfolio
        realised["selection_rule"] = (
            f"{config.objective}_{config.validation_months}m_min"
            f"{config.minimum_validation_months}"
        )
        records.append(realised)
    return pd.DataFrame(records)


def summarize_strategy(
    monthly: pd.DataFrame,
    strategy: str,
    config: SelectorConfig,
) -> dict[str, Any]:
    return_column = f"net_return_{config.aum_label}"
    turnover_column = f"turnover_{config.aum_label}"
    returns = monthly[return_column].astype(float)
    volatility = float(returns.std(ddof=1) * np.sqrt(12.0))
    return {
        "strategy": strategy,
        "portfolio": str(monthly["selected_portfolio"].iloc[0]),
        "months": int(len(monthly)),
        "annualized_net_return": float(returns.mean() * 12.0),
        "annualized_net_volatility": volatility,
        "net_sharpe": (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(12.0))
            if returns.std(ddof=1) > 0
            else np.nan
        ),
        "max_drawdown": _max_drawdown(returns),
        "average_monthly_turnover": float(monthly[turnover_column].mean()),
        "average_observed_spread_fraction": float(
            monthly["observed_spread_fraction"].mean()
        ),
        "average_universe_n": float(monthly["universe_n"].mean()),
        "aum_label": config.aum_label,
        "objective": config.objective,
        "validation_months": config.validation_months,
        "minimum_validation_months": config.minimum_validation_months,
    }


def fixed_strategy(
    monthly: pd.DataFrame,
    *,
    model: str,
    rung: str,
    portfolio: str,
    config: SelectorConfig,
) -> pd.DataFrame:
    fixed = monthly[
        monthly["model"].eq(model)
        & monthly["rung"].eq(rung)
        & monthly["weighting"].eq(config.weighting)
        & monthly["portfolio"].eq(portfolio)
    ].copy()
    fixed["selected_portfolio"] = portfolio
    fixed["selection_rule"] = "fixed"
    return fixed


def compare_return_series(
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    model_name: str,
    baseline_name: str,
    config: SelectorConfig,
) -> pd.DataFrame:
    return_column = f"net_return_{config.aum_label}"
    left = model.set_index("target_date")[return_column].astype(float)
    right = baseline.set_index("target_date")[return_column].astype(float)
    dates = left.index.intersection(right.index)
    if len(dates) < 24:
        return pd.DataFrame()
    left = left.reindex(dates)
    right = right.reindex(dates)
    mean_test = project_stats.hac_mean_diff_test(
        left - right,
        maxlags=config.hac_lags,
    )
    records = []
    for block in config.bootstrap_blocks:
        sharpe = project_stats.bootstrap_sharpe_diff(
            left,
            right,
            np.zeros(len(dates)),
            expected_block=block,
            n_boot=config.bootstrap_repetitions,
            seed=config.random_state,
        )
        records.append(
            {
                "model": model_name,
                "baseline": baseline_name,
                "portfolio": str(model["selected_portfolio"].iloc[0]),
                "aum_label": config.aum_label,
                "months": int(len(dates)),
                "model_annualized_net_return": float(left.mean() * 12.0),
                "baseline_annualized_net_return": float(right.mean() * 12.0),
                "delta_annualized_net_return": float(mean_test["mean"] * 12.0),
                "hac_t_stat": float(mean_test["t"]),
                "hac_p_two_sided": float(mean_test["p_two_sided"]),
                **sharpe,
            }
        )
    return pd.DataFrame(records)


def run_validation_selection(
    monthly: pd.DataFrame,
    portfolios: list[str],
    rungs: list[str],
    candidate_models: list[str],
    config: SelectorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_frames = []
    baseline_frames = []
    inference_frames = []
    for portfolio in portfolios:
        selected = select_strategy_monthly(
            monthly,
            portfolio=portfolio,
            rungs=rungs,
            candidate_models=candidate_models,
            config=config,
        )
        if selected.empty:
            continue
        selected["strategy"] = f"validation_selected_{portfolio}"
        selected_frames.append(selected)

        momentum_selected = select_strategy_monthly(
            monthly,
            portfolio=portfolio,
            rungs=rungs,
            candidate_models=["momentum_rank"],
            config=config,
        )
        if not momentum_selected.empty:
            momentum_selected["strategy"] = f"momentum_validation_selected_rung_{portfolio}"
            baseline_frames.append(momentum_selected)
            inference_frames.append(
                compare_return_series(
                    selected,
                    momentum_selected,
                    model_name=f"validation_selected_{portfolio}",
                    baseline_name=f"momentum_validation_selected_rung_{portfolio}",
                    config=config,
                )
            )

        for baseline_model, rung in [
            ("momentum_rank", "top_500_observed_spread"),
            ("ridge_rank", "top_500_observed_spread"),
            ("smooth75_ridge_rank", "top_500_observed_spread"),
        ]:
            baseline = fixed_strategy(
                monthly,
                model=baseline_model,
                rung=rung,
                portfolio=portfolio,
                config=config,
            )
            if baseline.empty:
                continue
            baseline["strategy"] = f"fixed_{baseline_model}_{rung}_{portfolio}"
            baseline_frames.append(baseline)
            inference_frames.append(
                compare_return_series(
                    selected,
                    baseline,
                    model_name=f"validation_selected_{portfolio}",
                    baseline_name=f"fixed_{baseline_model}_{rung}_{portfolio}",
                    config=config,
                )
            )

    selected_monthly = (
        pd.concat(selected_frames, ignore_index=True)
        if selected_frames
        else pd.DataFrame()
    )
    baselines = (
        pd.concat(baseline_frames, ignore_index=True)
        if baseline_frames
        else pd.DataFrame()
    )
    inference = (
        pd.concat([frame for frame in inference_frames if not frame.empty], ignore_index=True)
        if inference_frames
        else pd.DataFrame()
    )
    if not inference.empty:
        family = ["portfolio", "aum_label", "expected_block"]
        inference["p_two_sided_holm"] = inference.groupby(family)[
            "p_two_sided"
        ].transform(lambda values: multipletests(values, method="holm")[1])
        inference["hac_p_two_sided_holm"] = inference.groupby(family)[
            "hac_p_two_sided"
        ].transform(lambda values: multipletests(values, method="holm")[1])
    return selected_monthly, baselines, inference


def write_selector_outputs(
    output_dir: Path,
    selected: pd.DataFrame,
    baselines: pd.DataFrame,
    inference: pd.DataFrame,
    config: SelectorConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    selected.to_csv(output_dir / "validation_selected_monthly.csv", index=False)
    baselines.to_csv(output_dir / "validation_baseline_monthly.csv", index=False)
    inference.to_csv(output_dir / "validation_selected_inference.csv", index=False)

    summaries = []
    for strategy, group in selected.groupby("strategy", sort=True):
        summaries.append(summarize_strategy(group, strategy, config))
    for strategy, group in baselines.groupby("strategy", sort=True):
        summaries.append(summarize_strategy(group, strategy, config))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "validation_selected_summary.csv", index=False)

    count_columns = ["strategy", "selected_portfolio", "model", "rung"]
    selection_counts = (
        selected.groupby(count_columns, sort=True)
        .size()
        .rename("selected_months")
        .reset_index()
    )
    selection_counts.to_csv(output_dir / "validation_selection_counts.csv", index=False)

    manifest = {
        **metadata,
        "selector_config": config.__dict__,
        "rows": {
            **metadata.get("rows", {}),
            "selected_monthly": int(len(selected)),
            "baseline_monthly": int(len(baselines)),
            "inference": int(len(inference)),
            "summary": int(len(summary)),
            "selection_counts": int(len(selection_counts)),
        },
        "outputs": {
            "candidate_predictions": metadata.get(
                "candidate_predictions_path",
                str(output_dir / "candidate_predictions.parquet"),
            ),
            "candidate_ladder_monthly": metadata.get(
                "candidate_ladder_monthly_path",
                str(output_dir / "candidate_ladder_monthly.parquet"),
            ),
            "candidate_ladder_summary": metadata.get(
                "candidate_ladder_summary_path",
                str(output_dir / "candidate_ladder_summary.csv"),
            ),
            "validation_selected_monthly": str(
                output_dir / "validation_selected_monthly.csv"
            ),
            "validation_baseline_monthly": str(
                output_dir / "validation_baseline_monthly.csv"
            ),
            "validation_selected_summary": str(
                output_dir / "validation_selected_summary.csv"
            ),
            "validation_selection_counts": str(
                output_dir / "validation_selection_counts.csv"
            ),
            "validation_selected_inference": str(
                output_dir / "validation_selected_inference.csv"
            ),
        },
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--smoothing-specs", type=Path, default=DEFAULT_SMOOTHING_SPECS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--candidate-ladder-monthly",
        type=Path,
        default=None,
        help="Reuse a previously simulated candidate_ladder_monthly.parquet.",
    )
    parser.add_argument(
        "--candidate-ladder-summary",
        type=Path,
        default=None,
        help="Optional summary matching --candidate-ladder-monthly.",
    )
    parser.add_argument(
        "--candidate-models",
        nargs="+",
        default=None,
        help="Optional candidate model list when reusing a ladder file.",
    )
    parser.add_argument("--core-models", nargs="+", default=DEFAULT_CORE_MODELS)
    parser.add_argument("--smoothing-inertia", nargs="+", type=float, default=[0.50, 0.75])
    parser.add_argument("--rungs", nargs="+", default=DEFAULT_RUNG_SET)
    parser.add_argument("--portfolios", nargs="+", default=DEFAULT_PORTFOLIOS)
    parser.add_argument("--validation-months", type=int, default=36)
    parser.add_argument("--minimum-validation-months", type=int, default=24)
    parser.add_argument("--risk-aversion", type=float, default=3.0)
    parser.add_argument(
        "--objective",
        choices=["certainty_equivalent", "sharpe"],
        default="certainty_equivalent",
    )
    parser.add_argument("--aum-eur", type=float, default=100_000_000.0)
    parser.add_argument("--maximum-assets", type=int, default=500)
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    aum_label = f"{int(round(args.aum_eur / 1_000_000.0))}m"
    ladder_config = LadderConfig(
        portfolio_quantile=args.portfolio_quantile,
        maximum_assets=args.maximum_assets,
        aum_eur=(args.aum_eur,),
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_blocks=tuple(args.bootstrap_blocks),
        hac_lags=args.hac_lags,
        random_state=args.random_state,
    )
    selector_config = SelectorConfig(
        validation_months=args.validation_months,
        minimum_validation_months=args.minimum_validation_months,
        risk_aversion=args.risk_aversion,
        objective=args.objective,
        aum_label=aum_label,
        hac_lags=args.hac_lags,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_blocks=tuple(args.bootstrap_blocks),
        random_state=args.random_state,
    )

    liquidity = args.liquidity if args.liquidity.exists() else None
    risk = args.risk if args.risk.exists() else None
    candidate_ladder_monthly_path = output_dir / "candidate_ladder_monthly.parquet"
    candidate_ladder_summary_path = output_dir / "candidate_ladder_summary.csv"
    candidate_predictions_path: Path | str = output_dir / "candidate_predictions.parquet"
    predictions_rows: int | None = None
    if args.candidate_ladder_monthly is not None:
        if not args.candidate_ladder_monthly.exists():
            raise FileNotFoundError(args.candidate_ladder_monthly)
        monthly = pd.read_parquet(args.candidate_ladder_monthly)
        if args.candidate_ladder_summary is not None and args.candidate_ladder_summary.exists():
            summary = pd.read_csv(args.candidate_ladder_summary)
        else:
            summary = summarize_investability_ladder(monthly, ladder_config)
        candidate_models = (
            sorted(args.candidate_models)
            if args.candidate_models is not None
            else sorted(monthly["model"].dropna().unique().tolist())
        )
        candidate_ladder_monthly_path = args.candidate_ladder_monthly
        candidate_ladder_summary_path = (
            args.candidate_ladder_summary
            if args.candidate_ladder_summary is not None
            else candidate_ladder_summary_path
        )
        candidate_predictions_path = "reused_ladder_no_prediction_subset_written"
    else:
        predictions, monthly, summary, candidate_models, candidate_predictions_path = (
            build_candidate_ladder(
                args.predictions,
                args.smoothing_specs,
                args.panel,
                liquidity,
                risk,
                output_dir,
                args.core_models,
                tuple(args.smoothing_inertia),
                ladder_config,
            )
        )
        predictions_rows = int(len(predictions))
    selected, baselines, inference = run_validation_selection(
        monthly,
        args.portfolios,
        args.rungs,
        candidate_models,
        selector_config,
    )
    manifest = write_selector_outputs(
        output_dir,
        selected,
        baselines,
        inference,
        selector_config,
        {
            "inputs": {
                "predictions": str(args.predictions),
                "smoothing_specs": str(args.smoothing_specs),
                "panel": str(args.panel),
                "liquidity": str(liquidity) if liquidity is not None else None,
                "risk": str(risk) if risk is not None else None,
            },
            "candidate_models": candidate_models,
            "core_models": args.core_models,
            "smoothing_inertia": args.smoothing_inertia,
            "rungs": args.rungs,
            "portfolios": args.portfolios,
            "candidate_predictions_path": str(candidate_predictions_path),
            "candidate_ladder_monthly_path": str(candidate_ladder_monthly_path),
            "candidate_ladder_summary_path": str(candidate_ladder_summary_path),
            "rows": {
                "candidate_predictions": predictions_rows,
                "candidate_models": int(len(candidate_models)),
                "candidate_ladder_monthly": int(len(monthly)),
                "candidate_ladder_summary": int(len(summary)),
            },
        },
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
