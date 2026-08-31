"""Constrained long-only construction for analyst-estimates ML signals."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [SRC_DIR, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stats as project_stats  # noqa: E402
from asset_pricing_depth import build_internal_market  # noqa: E402
from investability_ladder import LadderConfig, load_ladder_panel  # noqa: E402
from run_constrained_deep_hybrid_long_only import (  # noqa: E402
    ConstraintSpec,
    SUBPERIODS,
    aum_label,
    parse_constraint_specs,
    simulate_constrained,
    subperiod_frame,
    summarize_concentration,
    summarize_constrained,
)


DEFAULT_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "estimates_validation_selected_implementable_strategy"
    / "validation_selected_monthly.csv"
)
DEFAULT_CANDIDATE_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "estimates_validation_selected_implementable_strategy"
    / "candidate_predictions.parquet"
)
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
DEFAULT_RISK = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "depth_analysis"
    / "rolling_risk_estimates.parquet"
)
DEFAULT_MARKET = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "depth_analysis"
    / "eur_market_return.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "constrained_estimates_long_only"
)
DEFAULT_FIXED_CHOICES = [
    "fixed_smooth25_mlp_top500_observed:smooth25_mlp_rank:top_500_observed_spread",
    "fixed_smooth25_hist_gbm_large_low:smooth25_hist_gbm_rank:large_low_spread",
    "fixed_smooth75_mlp_top500_observed:smooth75_mlp_rank:top_500_observed_spread",
    "fixed_smooth75_ridge_top500_observed:smooth75_ridge_rank:top_500_observed_spread",
]


def parse_fixed_choice(value: str) -> dict[str, str]:
    parts = value.split(":")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError("Fixed choice must be strategy:model:rung")
    return {"strategy": parts[0], "model": parts[1], "rung": parts[2]}


def load_selected_long_only(
    path: Path,
    *,
    strategy_name: str,
) -> pd.DataFrame:
    selected = pd.read_csv(path, parse_dates=["date", "target_date"])
    selected = selected[
        selected["strategy"].eq("validation_selected_long_only")
        & selected["selected_portfolio"].eq("long_only")
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No validation-selected long-only rows in {path}")
    selected = selected[["date", "target_date", "model", "rung"]].drop_duplicates(
        ["date"]
    )
    selected["strategy"] = strategy_name
    return selected[["strategy", "date", "target_date", "model", "rung"]]


def build_choice_panel(
    selected: pd.DataFrame,
    fixed_choices: list[dict[str, str]],
) -> pd.DataFrame:
    calendar = selected[["date", "target_date"]].drop_duplicates("date")
    fixed = build_fixed_choice_panel(calendar, fixed_choices)
    frames = [selected.copy(), fixed]
    return pd.concat(frames, ignore_index=True).sort_values(["strategy", "date"])


def load_choice_calendar_from_predictions(
    path: Path,
    fixed_choices: list[dict[str, str]],
) -> pd.DataFrame:
    predictions = pd.read_parquet(path, columns=["date", "target_date", "model"])
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    available_models = set(predictions["model"].dropna().unique())
    missing = sorted({choice["model"] for choice in fixed_choices} - available_models)
    if missing:
        raise RuntimeError(f"Missing fixed-choice models in predictions: {missing}")
    return (
        predictions[["date", "target_date"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def build_fixed_choice_panel(
    calendar: pd.DataFrame,
    fixed_choices: list[dict[str, str]],
) -> pd.DataFrame:
    frames = []
    for fixed in fixed_choices:
        frames.append(
            calendar.assign(
                strategy=fixed["strategy"],
                model=fixed["model"],
                rung=fixed["rung"],
            )
        )
    if not frames:
        return pd.DataFrame(columns=["strategy", "date", "target_date", "model", "rung"])
    return pd.concat(frames, ignore_index=True).sort_values(["strategy", "date"])


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return float(wealth.div(peak).sub(1.0).min())


def load_benchmark_market(market_path: Path | None, panel_path: Path) -> pd.DataFrame:
    """Load or build the internal EUR value-weighted benchmark return."""
    if market_path is not None and market_path.exists():
        market = pd.read_csv(market_path, parse_dates=["date"])
    else:
        panel = pd.read_parquet(
            panel_path,
            columns=["date", "ric", "return_1m", "company_market_cap"],
        )
        market = build_internal_market(panel)
    required = {"date", "market_return_eur"}
    missing = required - set(market)
    if missing:
        raise ValueError(f"Benchmark market missing columns: {sorted(missing)}")
    market = market[["date", "market_return_eur"]].copy()
    market["date"] = pd.to_datetime(market["date"]).dt.to_period("M").dt.to_timestamp("M")
    return market.drop_duplicates("date", keep="last").sort_values("date")


def add_benchmark_relative_returns(
    monthly: pd.DataFrame,
    market: pd.DataFrame,
    aum_values: tuple[float, ...],
) -> pd.DataFrame:
    """Attach benchmark and active monthly returns to constrained output."""
    merged = monthly.copy()
    merged["target_date"] = pd.to_datetime(merged["target_date"])
    benchmark = market.rename(
        columns={
            "date": "target_date",
            "market_return_eur": "benchmark_return_eur",
        }
    )
    benchmark["target_date"] = pd.to_datetime(benchmark["target_date"])
    merged = merged.merge(
        benchmark[["target_date", "benchmark_return_eur"]],
        on="target_date",
        how="left",
        validate="many_to_one",
    )
    for aum in aum_values:
        label = aum_label(aum)
        merged[f"active_return_{label}"] = (
            pd.to_numeric(merged[f"net_return_{label}"], errors="coerce")
            - pd.to_numeric(merged["benchmark_return_eur"], errors="coerce")
        )
    return merged


def estimate_benchmark_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    hac_lags: int,
) -> dict[str, float]:
    clean = pd.DataFrame(
        {
            "portfolio_return": pd.to_numeric(returns, errors="coerce"),
            "benchmark_return": pd.to_numeric(benchmark, errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < max(24, hac_lags + 3):
        return {
            "alpha_observations": int(len(clean)),
            "alpha_monthly": np.nan,
            "alpha_annualized": np.nan,
            "alpha_t_stat": np.nan,
            "alpha_p_two_sided": np.nan,
            "benchmark_beta": np.nan,
            "benchmark_beta_p": np.nan,
            "benchmark_adjusted_r2": np.nan,
        }
    x = sm.add_constant(clean["benchmark_return"], has_constant="add")
    fit = sm.OLS(clean["portfolio_return"], x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )
    return {
        "alpha_observations": int(fit.nobs),
        "alpha_monthly": float(fit.params["const"]),
        "alpha_annualized": float(fit.params["const"] * 12.0),
        "alpha_t_stat": float(fit.tvalues["const"]),
        "alpha_p_two_sided": float(fit.pvalues["const"]),
        "benchmark_beta": float(fit.params["benchmark_return"]),
        "benchmark_beta_p": float(fit.pvalues["benchmark_return"]),
        "benchmark_adjusted_r2": float(fit.rsquared_adj),
    }


def summarize_benchmark_relative(
    monthly: pd.DataFrame,
    aum_values: tuple[float, ...],
    *,
    hac_lags: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (strategy, constraint), group in monthly.groupby(
        ["strategy", "constraint"],
        sort=True,
    ):
        for subperiod, start, end in SUBPERIODS:
            part = subperiod_frame(group, start, end)
            if len(part) < 6:
                continue
            for aum in aum_values:
                label = aum_label(aum)
                clean = pd.DataFrame(
                    {
                        "portfolio_return": pd.to_numeric(
                            part[f"net_return_{label}"],
                            errors="coerce",
                        ),
                        "benchmark_return": pd.to_numeric(
                            part["benchmark_return_eur"],
                            errors="coerce",
                        ),
                        "active_return": pd.to_numeric(
                            part[f"active_return_{label}"],
                            errors="coerce",
                        ),
                    }
                ).replace([np.inf, -np.inf], np.nan).dropna()
                if len(clean) < 6:
                    continue
                active_std = clean["active_return"].std(ddof=1)
                tracking_error = float(active_std * np.sqrt(12.0))
                mean_test = project_stats.hac_mean_diff_test(
                    clean["active_return"],
                    maxlags=hac_lags,
                )
                alpha = estimate_benchmark_alpha(
                    clean["portfolio_return"],
                    clean["benchmark_return"],
                    hac_lags=hac_lags,
                )
                records.append(
                    {
                        "strategy": strategy,
                        "constraint": constraint,
                        "subperiod": subperiod,
                        "aum_eur": float(aum),
                        "aum_label": label,
                        "benchmark": "internal_eur_value_weighted_market",
                        "months": int(len(clean)),
                        "annualized_net_return": float(
                            clean["portfolio_return"].mean() * 12.0
                        ),
                        "annualized_benchmark_return": float(
                            clean["benchmark_return"].mean() * 12.0
                        ),
                        "annualized_active_return": float(
                            clean["active_return"].mean() * 12.0
                        ),
                        "tracking_error": tracking_error,
                        "information_ratio": (
                            float(
                                clean["active_return"].mean()
                                / active_std
                                * np.sqrt(12.0)
                            )
                            if active_std > 0
                            else np.nan
                        ),
                        "active_max_drawdown": max_drawdown(clean["active_return"]),
                        "active_hac_t_stat": float(mean_test["t"]),
                        "active_hac_p_two_sided": float(mean_test["p_two_sided"]),
                        **alpha,
                    }
                )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    family = ["subperiod", "aum_label"]
    result["active_hac_p_holm"] = result.groupby(family)[
        "active_hac_p_two_sided"
    ].transform(lambda values: multipletests(values, method="holm")[1])
    result["alpha_p_holm"] = result.groupby(family)["alpha_p_two_sided"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    return result


def infer_vs_baselines(
    monthly: pd.DataFrame,
    *,
    selected_strategy: str,
    baseline_strategies: list[str],
    aum_values: tuple[float, ...],
    blocks: tuple[int, ...],
    n_boot: int,
    seed: int,
    hac_lags: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for constraint, group in monthly.groupby("constraint", sort=True):
        model = group[group["strategy"].eq(selected_strategy)]
        if model.empty:
            continue
        for baseline_strategy in baseline_strategies:
            baseline = group[group["strategy"].eq(baseline_strategy)]
            if baseline.empty:
                continue
            for aum in aum_values:
                label = f"{int(round(aum / 1_000_000.0))}m"
                column = f"net_return_{label}"
                left = model.set_index("target_date")[column].astype(float)
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
                            "model": selected_strategy,
                            "baseline": baseline_strategy,
                            "constraint": constraint,
                            "aum_eur": float(aum),
                            "aum_label": label,
                            "months": int(len(dates)),
                            "model_annualized_net_return": float(left.mean() * 12.0),
                            "baseline_annualized_net_return": float(
                                right.mean() * 12.0
                            ),
                            "delta_annualized_net_return": float(
                                mean_test["mean"] * 12.0
                            ),
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


def run_experiment(
    selected_path: Path | None,
    predictions_path: Path,
    panel_path: Path,
    liquidity_path: Path | None,
    risk_path: Path | None,
    market_path: Path | None,
    output_dir: Path,
    specs: list[ConstraintSpec],
    fixed_choices: list[dict[str, str]],
    selected_strategy: str,
    aum_values: tuple[float, ...],
    maximum_assets: int,
    fallback_half_spread_bps: float,
    impact_coefficient: float,
    bootstrap_repetitions: int,
    bootstrap_blocks: tuple[int, ...],
    random_state: int,
    hac_lags: int,
    fixed_only: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if fixed_only:
        calendar = load_choice_calendar_from_predictions(predictions_path, fixed_choices)
        choices = build_fixed_choice_panel(calendar, fixed_choices)
    else:
        if selected_path is None:
            raise ValueError("selected_path is required unless fixed_only=True")
        selected = load_selected_long_only(
            selected_path,
            strategy_name=selected_strategy,
        )
        choices = build_choice_panel(selected, fixed_choices)
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
    benchmark_market = load_benchmark_market(market_path, panel_path)
    benchmark_relative_monthly = add_benchmark_relative_returns(
        monthly,
        benchmark_market,
        aum_values,
    )
    benchmark_relative = summarize_benchmark_relative(
        benchmark_relative_monthly,
        aum_values,
        hac_lags=hac_lags,
    )
    if fixed_only:
        inference = pd.DataFrame()
    else:
        inference = infer_vs_baselines(
            monthly,
            selected_strategy=selected_strategy,
            baseline_strategies=[choice["strategy"] for choice in fixed_choices],
            aum_values=aum_values,
            blocks=bootstrap_blocks,
            n_boot=bootstrap_repetitions,
            seed=random_state,
            hac_lags=hac_lags,
        )

    monthly.to_parquet(
        output_dir / "constrained_monthly.parquet",
        index=False,
        compression="zstd",
    )
    monthly.to_csv(output_dir / "constrained_monthly.csv", index=False)
    summary.to_csv(output_dir / "constrained_summary.csv", index=False)
    concentration.to_csv(output_dir / "concentration_summary.csv", index=False)
    benchmark_relative_monthly.to_csv(
        output_dir / "benchmark_relative_monthly.csv",
        index=False,
    )
    benchmark_relative.to_csv(
        output_dir / "benchmark_relative_summary.csv",
        index=False,
    )
    inference.to_csv(output_dir / "constrained_inference.csv", index=False)
    failures.to_csv(output_dir / "constraint_failures.csv", index=False)

    manifest = {
        "inputs": {
            "selected": (
                None
                if fixed_only or selected_path is None
                else str(selected_path)
            ),
            "candidate_predictions": str(predictions_path),
            "panel": str(panel_path),
            "liquidity": str(liquidity_path) if liquidity_path is not None else None,
            "risk": str(risk_path) if risk_path is not None else None,
            "market": str(market_path) if market_path is not None else None,
        },
        "fixed_only": fixed_only,
        "benchmark": {
            "name": "internal_eur_value_weighted_market",
            "return_column": "market_return_eur",
            "alpha_model": (
                "net_return = alpha + beta * internal EUR value-weighted market return"
            ),
        },
        "selected_strategy": selected_strategy,
        "fixed_choices": fixed_choices,
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
            "benchmark_market": int(len(benchmark_market)),
            "benchmark_relative_monthly": int(len(benchmark_relative_monthly)),
            "benchmark_relative": int(len(benchmark_relative)),
            "inference": int(len(inference)),
            "failures": int(len(failures)),
        },
        "outputs": {
            "strategy_choices": str(output_dir / "strategy_choices.csv"),
            "constrained_monthly": str(output_dir / "constrained_monthly.csv"),
            "constrained_summary": str(output_dir / "constrained_summary.csv"),
            "concentration_summary": str(output_dir / "concentration_summary.csv"),
            "benchmark_relative_monthly": str(
                output_dir / "benchmark_relative_monthly.csv"
            ),
            "benchmark_relative_summary": str(
                output_dir / "benchmark_relative_summary.csv"
            ),
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
    parser.add_argument(
        "--candidate-predictions",
        type=Path,
        default=DEFAULT_CANDIDATE_PREDICTIONS,
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selected-strategy",
        default="validation_selected_estimates_long_only",
    )
    parser.add_argument(
        "--fixed-choice",
        action="append",
        default=None,
        help="Fixed strategy as strategy:model:rung.",
    )
    parser.add_argument("--maximum-assets", type=int, default=500)
    parser.add_argument(
        "--aum-eur",
        nargs="+",
        type=float,
        default=[10_000_000.0, 100_000_000.0, 500_000_000.0],
    )
    parser.add_argument("--fallback-half-spread-bps", type=float, default=25.0)
    parser.add_argument("--impact-coefficient", type=float, default=0.10)
    parser.add_argument("--constraint", action="append", default=None)
    parser.add_argument(
        "--fixed-only",
        action="store_true",
        help="Run only fixed strategy choices, using the prediction calendar.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--bootstrap-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    liquidity = args.liquidity if args.liquidity.exists() else None
    risk = args.risk if args.risk.exists() else None
    market = args.market if args.market.exists() else None
    fixed_choices = [
        parse_fixed_choice(value)
        for value in (args.fixed_choice or DEFAULT_FIXED_CHOICES)
    ]
    manifest = run_experiment(
        selected_path=None if args.fixed_only else args.selected,
        predictions_path=args.candidate_predictions,
        panel_path=args.panel,
        liquidity_path=liquidity,
        risk_path=risk,
        market_path=market,
        output_dir=args.output_dir,
        specs=parse_constraint_specs(args.constraint),
        fixed_choices=fixed_choices,
        selected_strategy=args.selected_strategy,
        aum_values=tuple(args.aum_eur),
        maximum_assets=args.maximum_assets,
        fallback_half_spread_bps=args.fallback_half_spread_bps,
        impact_coefficient=args.impact_coefficient,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_blocks=tuple(args.bootstrap_blocks),
        random_state=args.random_state,
        hac_lags=args.hac_lags,
        fixed_only=args.fixed_only,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
