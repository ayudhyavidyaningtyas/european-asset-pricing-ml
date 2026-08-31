"""Validation-only selection over constrained deep/hybrid portfolio rules.

This closure experiment keeps the fitted return signals and constrained
portfolio weights fixed.  It only asks whether the universe cap and constraint
rule can be selected with prior validation returns rather than full-sample
performance.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
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


DEFAULT_RUNS = {
    500: PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "constrained_deep_hybrid_long_only_refinitiv_refresh",
    1000: PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "constrained_deep_hybrid_long_only_top1000",
    2000: PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "constrained_deep_hybrid_long_only_top2000",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "validation_selected_constrained_deep_hybrid"
)
DEFAULT_STRATEGIES = [
    "frozen_deep_hybrid_selector",
    "fixed_momentum_top500_observed",
]
DEFAULT_PRIMARY_MODEL = "frozen_deep_hybrid_selector"
DEFAULT_PRIMARY_BASELINE = "fixed_momentum_top500_observed"


@dataclass(frozen=True)
class ConstrainedSelectorConfig:
    validation_months: int = 36
    minimum_validation_months: int = 24
    risk_aversion: float = 3.0
    objective: str = "certainty_equivalent"
    aum_label: str = "100m"
    hac_lags: int = 6
    bootstrap_repetitions: int = 2_000
    bootstrap_blocks: tuple[int, ...] = (3, 6, 12)
    random_state: int = 42


def _certainty_equivalent(returns: pd.Series, risk_aversion: float) -> float:
    returns = returns.dropna().astype(float)
    if len(returns) < 2:
        return np.nan
    annual_mean = float(returns.mean() * 12.0)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(12.0))
    return annual_mean - 0.5 * risk_aversion * annual_vol**2


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0).astype(float)).cumprod()
    peak = wealth.cummax()
    return float(wealth.div(peak).sub(1.0).min())


def parse_cap_run(value: str) -> tuple[int, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Run must be CAP:PATH, e.g. 1000:results/...")
    cap, path = value.split(":", 1)
    try:
        maximum_assets = int(cap)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid cap: {cap}") from exc
    return maximum_assets, Path(path)


def _infer_cap(path: Path) -> int:
    match = re.search(r"top(\d+)", str(path))
    if match:
        return int(match.group(1))
    return 500


def read_candidate_monthly(runs: list[tuple[int, Path]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for maximum_assets, run_dir in runs:
        monthly_path = run_dir / "constrained_monthly.csv"
        if not monthly_path.exists():
            raise FileNotFoundError(monthly_path)
        frame = pd.read_csv(monthly_path, parse_dates=["date", "target_date"])
        frame["maximum_assets"] = int(maximum_assets)
        frame["candidate_cell"] = (
            "top"
            + frame["maximum_assets"].astype(str)
            + "_"
            + frame["constraint"].astype(str)
        )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    duplicate_keys = [
        "strategy",
        "maximum_assets",
        "constraint",
        "date",
    ]
    duplicates = int(combined.duplicated(duplicate_keys).sum())
    if duplicates:
        raise RuntimeError(f"Duplicate constrained candidate rows: {duplicates}")
    return combined.sort_values(["strategy", "date", "maximum_assets", "constraint"])


def validation_scores(
    validation: pd.DataFrame,
    config: ConstrainedSelectorConfig,
) -> pd.DataFrame:
    return_column = f"net_return_{config.aum_label}"
    turnover_column = f"turnover_{config.aum_label}"
    records: list[dict[str, Any]] = []
    for (maximum_assets, constraint), group in validation.groupby(
        ["maximum_assets", "constraint"], sort=True
    ):
        returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
        if len(returns) < config.minimum_validation_months:
            continue
        annual_return = float(returns.mean() * 12.0)
        annual_vol = float(returns.std(ddof=1) * np.sqrt(12.0))
        sharpe = (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(12.0))
            if returns.std(ddof=1) > 0
            else np.nan
        )
        ce = _certainty_equivalent(returns, config.risk_aversion)
        objective = ce if config.objective == "certainty_equivalent" else sharpe
        records.append(
            {
                "maximum_assets": int(maximum_assets),
                "constraint": str(constraint),
                "candidate_cell": f"top{int(maximum_assets)}_{constraint}",
                "validation_months": int(len(returns)),
                "validation_annualized_return": annual_return,
                "validation_annualized_volatility": annual_vol,
                "validation_sharpe": sharpe,
                "validation_certainty_equivalent": ce,
                "validation_objective": float(objective),
                "validation_average_turnover": float(group[turnover_column].mean()),
                "validation_average_spread_cost": float(
                    group[f"spread_cost_{config.aum_label}"].mean()
                ),
                "validation_average_impact_cost": float(
                    group[f"impact_cost_{config.aum_label}"].mean()
                ),
                "validation_observed_spread_weight": float(
                    group["observed_spread_weight"].mean()
                ),
                "validation_average_effective_n": float(group["effective_n"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def select_strategy_monthly(
    candidates: pd.DataFrame,
    *,
    strategy: str,
    config: ConstrainedSelectorConfig,
) -> pd.DataFrame:
    return_column = f"net_return_{config.aum_label}"
    if return_column not in candidates:
        raise RuntimeError(f"Missing return column: {return_column}")
    eligible = candidates[candidates["strategy"].eq(strategy)].copy()
    records: list[dict[str, Any]] = []
    for signal_date, current in eligible.groupby("date", sort=True):
        validation_start = signal_date - pd.DateOffset(months=config.validation_months)
        validation = eligible[
            eligible["target_date"].le(signal_date)
            & eligible["target_date"].gt(validation_start)
        ]
        scores = validation_scores(validation, config)
        if scores.empty:
            continue
        scores = scores.sort_values(
            [
                "validation_objective",
                "validation_months",
                "validation_observed_spread_weight",
                "validation_average_turnover",
                "maximum_assets",
                "constraint",
            ],
            ascending=[False, False, False, True, True, True],
        )
        selected = scores.iloc[0]
        realised = current[
            current["maximum_assets"].eq(int(selected["maximum_assets"]))
            & current["constraint"].eq(str(selected["constraint"]))
        ]
        if realised.empty:
            continue
        row = realised.iloc[0].to_dict()
        for key, value in selected.items():
            row[key] = value
        row["selection_signal_date"] = signal_date
        row["selection_rule"] = (
            f"{config.objective}_{config.validation_months}m_min"
            f"{config.minimum_validation_months}"
        )
        row["selected_strategy"] = f"validation_selected_{strategy}"
        records.append(row)
    return pd.DataFrame.from_records(records)


def summarize_selected(monthly: pd.DataFrame, config: ConstrainedSelectorConfig) -> pd.DataFrame:
    return_column = f"net_return_{config.aum_label}"
    turnover_column = f"turnover_{config.aum_label}"
    records: list[dict[str, Any]] = []
    for strategy, group in monthly.groupby("selected_strategy", sort=True):
        returns = pd.to_numeric(group[return_column], errors="coerce")
        vol = float(returns.std(ddof=1) * np.sqrt(12.0))
        records.append(
            {
                "selected_strategy": strategy,
                "months": int(len(group)),
                "annualized_net_return": float(returns.mean() * 12.0),
                "annualized_net_volatility": vol,
                "net_sharpe": (
                    float(returns.mean() / returns.std(ddof=1) * np.sqrt(12.0))
                    if returns.std(ddof=1) > 0
                    else np.nan
                ),
                "max_drawdown": _max_drawdown(returns),
                "average_monthly_turnover": float(group[turnover_column].mean()),
                "annualized_spread_cost": float(
                    group[f"spread_cost_{config.aum_label}"].mean() * 12.0
                ),
                "annualized_impact_cost": float(
                    group[f"impact_cost_{config.aum_label}"].mean() * 12.0
                ),
                "average_effective_n": float(group["effective_n"].mean()),
                "average_top_5_name_weight": float(group["top_5_name_weight"].mean()),
                "average_max_country_weight": float(group["max_country_weight"].mean())
                if "max_country_weight" in group
                else np.nan,
                "average_max_sector_weight": float(group["max_sector_weight"].mean())
                if "max_sector_weight" in group
                else np.nan,
                "average_selected_maximum_assets": float(group["maximum_assets"].mean()),
                "aum_label": config.aum_label,
                "validation_months": config.validation_months,
                "minimum_validation_months": config.minimum_validation_months,
                "objective": config.objective,
            }
        )
    return pd.DataFrame.from_records(records)


def fixed_candidate(
    candidates: pd.DataFrame,
    *,
    strategy: str,
    maximum_assets: int,
    constraint: str,
) -> pd.DataFrame:
    frame = candidates[
        candidates["strategy"].eq(strategy)
        & candidates["maximum_assets"].eq(maximum_assets)
        & candidates["constraint"].eq(constraint)
    ].copy()
    frame["selected_strategy"] = (
        f"fixed_{strategy}_top{maximum_assets}_{constraint}"
    )
    return frame


def compare_return_series(
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    model_name: str,
    baseline_name: str,
    config: ConstrainedSelectorConfig,
) -> pd.DataFrame:
    return_column = f"net_return_{config.aum_label}"
    left = model.set_index("target_date")[return_column].astype(float)
    right = baseline.set_index("target_date")[return_column].astype(float)
    dates = left.index.intersection(right.index)
    if len(dates) < 24:
        return pd.DataFrame()
    left = left.reindex(dates)
    right = right.reindex(dates)
    mean_test = project_stats.hac_mean_diff_test(left - right, maxlags=config.hac_lags)
    records: list[dict[str, Any]] = []
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
    return pd.DataFrame.from_records(records)


def build_inference(
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    primary_model: str,
    primary_baseline: str,
    fixed_constraint: str,
    fixed_maximum_assets: int,
    config: ConstrainedSelectorConfig,
) -> pd.DataFrame:
    inference_frames: list[pd.DataFrame] = []
    model_selected = selected[
        selected["selected_strategy"].eq(f"validation_selected_{primary_model}")
    ].copy()
    baseline_selected = selected[
        selected["selected_strategy"].eq(f"validation_selected_{primary_baseline}")
    ].copy()
    if not model_selected.empty and not baseline_selected.empty:
        inference_frames.append(
            compare_return_series(
                model_selected,
                baseline_selected,
                model_name=f"validation_selected_{primary_model}",
                baseline_name=f"validation_selected_{primary_baseline}",
                config=config,
            )
        )
    fixed_model = fixed_candidate(
        candidates,
        strategy=primary_model,
        maximum_assets=fixed_maximum_assets,
        constraint=fixed_constraint,
    )
    fixed_baseline = fixed_candidate(
        candidates,
        strategy=primary_baseline,
        maximum_assets=fixed_maximum_assets,
        constraint=fixed_constraint,
    )
    if not model_selected.empty and not fixed_model.empty:
        inference_frames.append(
            compare_return_series(
                model_selected,
                fixed_model,
                model_name=f"validation_selected_{primary_model}",
                baseline_name=(
                    f"fixed_{primary_model}_top{fixed_maximum_assets}_{fixed_constraint}"
                ),
                config=config,
            )
        )
    if not model_selected.empty and not fixed_baseline.empty:
        inference_frames.append(
            compare_return_series(
                model_selected,
                fixed_baseline,
                model_name=f"validation_selected_{primary_model}",
                baseline_name=(
                    f"fixed_{primary_baseline}_top{fixed_maximum_assets}_{fixed_constraint}"
                ),
                config=config,
            )
        )
    result = (
        pd.concat([frame for frame in inference_frames if not frame.empty], ignore_index=True)
        if inference_frames
        else pd.DataFrame()
    )
    if not result.empty:
        result["p_two_sided_holm"] = result.groupby(["aum_label", "expected_block"])[
            "p_two_sided"
        ].transform(lambda values: multipletests(values, method="holm")[1])
        result["hac_p_two_sided_holm"] = result.groupby("aum_label")[
            "hac_p_two_sided"
        ].transform(lambda values: multipletests(values, method="holm")[1])
    return result


def run_selector(
    runs: list[tuple[int, Path]],
    output_dir: Path,
    strategies: list[str],
    config: ConstrainedSelectorConfig,
    primary_model: str,
    primary_baseline: str,
    fixed_constraint: str,
    fixed_maximum_assets: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = read_candidate_monthly(runs)
    candidates = candidates[candidates["strategy"].isin(strategies)].copy()
    if candidates.empty:
        raise RuntimeError("No candidate rows after strategy filtering")
    selected_frames = [
        select_strategy_monthly(candidates, strategy=strategy, config=config)
        for strategy in strategies
    ]
    selected = (
        pd.concat([frame for frame in selected_frames if not frame.empty], ignore_index=True)
        if selected_frames
        else pd.DataFrame()
    )
    summary = summarize_selected(selected, config) if not selected.empty else pd.DataFrame()
    count_columns = ["selected_strategy", "maximum_assets", "constraint", "candidate_cell"]
    selection_counts = (
        selected.groupby(count_columns, sort=True)
        .size()
        .rename("selected_months")
        .reset_index()
        if not selected.empty
        else pd.DataFrame(columns=count_columns + ["selected_months"])
    )
    inference = build_inference(
        candidates,
        selected,
        primary_model=primary_model,
        primary_baseline=primary_baseline,
        fixed_constraint=fixed_constraint,
        fixed_maximum_assets=fixed_maximum_assets,
        config=config,
    )

    candidates.to_parquet(
        output_dir / "candidate_constrained_monthly.parquet",
        index=False,
        compression="zstd",
    )
    candidates.to_csv(output_dir / "candidate_constrained_monthly.csv", index=False)
    selected.to_csv(output_dir / "validation_selected_monthly.csv", index=False)
    summary.to_csv(output_dir / "validation_selected_summary.csv", index=False)
    selection_counts.to_csv(output_dir / "validation_selection_counts.csv", index=False)
    inference.to_csv(output_dir / "validation_selected_inference.csv", index=False)

    manifest = {
        "inputs": {
            str(maximum_assets): str(path) for maximum_assets, path in runs
        },
        "strategies": strategies,
        "primary_model": primary_model,
        "primary_baseline": primary_baseline,
        "fixed_constraint": fixed_constraint,
        "fixed_maximum_assets": fixed_maximum_assets,
        "selector_config": asdict(config),
        "rows": {
            "candidate_constrained_monthly": int(len(candidates)),
            "selected_monthly": int(len(selected)),
            "summary": int(len(summary)),
            "selection_counts": int(len(selection_counts)),
            "inference": int(len(inference)),
        },
        "outputs": {
            "candidate_constrained_monthly": str(
                output_dir / "candidate_constrained_monthly.csv"
            ),
            "validation_selected_monthly": str(
                output_dir / "validation_selected_monthly.csv"
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
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=parse_cap_run,
        help="Candidate constrained run as CAP:PATH. Defaults to top500/top1000/top2000.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--primary-model", default=DEFAULT_PRIMARY_MODEL)
    parser.add_argument("--primary-baseline", default=DEFAULT_PRIMARY_BASELINE)
    parser.add_argument("--fixed-constraint", default="name5_country40_sector40_turnover")
    parser.add_argument("--fixed-maximum-assets", type=int, default=500)
    parser.add_argument("--validation-months", type=int, default=36)
    parser.add_argument("--minimum-validation-months", type=int, default=24)
    parser.add_argument("--risk-aversion", type=float, default=3.0)
    parser.add_argument(
        "--objective",
        choices=["certainty_equivalent", "sharpe"],
        default="certainty_equivalent",
    )
    parser.add_argument("--aum-eur", type=float, default=100_000_000.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--bootstrap-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    aum_label = f"{int(round(args.aum_eur / 1_000_000.0))}m"
    runs = args.run or list(DEFAULT_RUNS.items())
    config = ConstrainedSelectorConfig(
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
    manifest = run_selector(
        runs=[(int(cap), Path(path)) for cap, path in runs],
        output_dir=args.output_dir,
        strategies=args.strategies,
        config=config,
        primary_model=args.primary_model,
        primary_baseline=args.primary_baseline,
        fixed_constraint=args.fixed_constraint,
        fixed_maximum_assets=args.fixed_maximum_assets,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
