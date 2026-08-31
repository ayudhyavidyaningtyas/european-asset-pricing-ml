"""Apply ex-ante turnover-aware smoothing to frozen ML prediction scores."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_depth import load_eur_short_rate  # noqa: E402
from asset_pricing_ml import (  # noqa: E402
    construct_monthly_portfolios,
    paired_sharpe_significance,
    portfolio_summary,
    prediction_metrics,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_blend_experiment"
    / "blend_ladder_subset_predictions.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "turnover_aware_signal_smoothing"
)
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_INERTIA_WEIGHTS = [0.25, 0.50, 0.75]


def _slug(model: str) -> str:
    return model.replace("attention_lstm", "attn").replace("_rank", "")


def _smooth_values(values: pd.Series, inertia_weight: float) -> pd.Series:
    smoothed = np.empty(len(values), dtype="float64")
    prior = np.nan
    current_weight = 1.0 - inertia_weight
    for index, value in enumerate(values.to_numpy(dtype="float64")):
        if np.isnan(prior):
            prior = value
        else:
            prior = current_weight * value + inertia_weight * prior
        smoothed[index] = prior
    return pd.Series(smoothed, index=values.index)


def monthly_rank_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = predictions[["model", "date", "ric", "prediction"]].copy()
    scores["rank_score"] = (
        scores.groupby(["model", "date"], sort=False)["prediction"]
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )
    return scores.drop(columns=["prediction"])


def build_smoothed_predictions(
    predictions: pd.DataFrame,
    inertia_weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = monthly_rank_scores(predictions)
    metadata_columns = [
        column
        for column in predictions.columns
        if column not in {"model", "prediction"}
    ]
    metadata = predictions[metadata_columns].drop_duplicates(["date", "ric"])
    metadata = metadata.set_index(["date", "ric"])
    scores = scores.sort_values(["model", "ric", "date"]).reset_index(drop=True)

    frames = []
    specs = []
    for model, model_scores in scores.groupby("model", sort=True):
        model_scores = model_scores.copy()
        for inertia in inertia_weights:
            smooth_name = f"smooth{int(round(inertia * 100)):02d}_{_slug(model)}_rank"
            smoothed = (
                model_scores.groupby("ric", sort=False)["rank_score"]
                .transform(lambda values: _smooth_values(values, inertia))
            )
            frame = model_scores[["date", "ric"]].copy()
            frame["prediction"] = smoothed.to_numpy(dtype="float64")
            frame = frame.set_index(["date", "ric"]).join(metadata, how="left")
            frame["model"] = smooth_name
            frame["target_mode"] = "rank"
            frames.append(frame.reset_index())
            specs.append(
                {
                    "model": smooth_name,
                    "parent_model": model,
                    "inertia_weight": float(inertia),
                    "current_score_weight": float(1.0 - inertia),
                }
            )
    smoothed_predictions = pd.concat(frames, ignore_index=True)
    return smoothed_predictions, pd.DataFrame(specs)


def pair_smoothed_against_parents(
    monthly: pd.DataFrame,
    specs: pd.DataFrame,
    *,
    cost_bps: int,
    blocks: tuple[int, ...],
    n_boot: int,
    seed: int,
    risk_free: pd.Series | None,
) -> pd.DataFrame:
    frames = []
    for parent, group in specs.groupby("parent_model", sort=True):
        subset = monthly[monthly["model"].isin([parent, *group["model"].tolist()])]
        test = paired_sharpe_significance(
            subset,
            baseline_model=parent,
            cost_bps=cost_bps,
            blocks=blocks,
            n_boot=n_boot,
            seed=seed,
            risk_free=risk_free,
        )
        if test.empty:
            continue
        test = test[test["model"].isin(set(group["model"]))].copy()
        frames.append(test)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).merge(specs, on="model", how="left")


def run_signal_smoothing(
    input_path: Path,
    output_dir: Path,
    inertia_weights: list[float],
    portfolio_quantile: float,
    cost_grid_bps: tuple[int, ...],
    risk_free: pd.Series | None,
    significance_cost_bps: int,
    significance_blocks: tuple[int, ...],
    significance_n_boot: int,
    random_state: int,
    benchmark_model: str,
    benchmark_top_smoothed: int,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_predictions = pd.read_parquet(input_path)
    parent_predictions["date"] = pd.to_datetime(parent_predictions["date"])
    parent_predictions["target_date"] = pd.to_datetime(parent_predictions["target_date"])
    parent_predictions = parent_predictions.dropna(subset=["prediction"]).copy()
    duplicate_parents = int(
        parent_predictions.duplicated(["model", "date", "ric"]).sum()
    )
    if duplicate_parents:
        raise RuntimeError(f"Duplicate parent model/date/ric predictions: {duplicate_parents}")

    smoothed_predictions, specs = build_smoothed_predictions(
        parent_predictions,
        inertia_weights,
    )
    all_predictions = pd.concat(
        [parent_predictions, smoothed_predictions],
        ignore_index=True,
    ).sort_values(["model", "date", "ric"])
    duplicate_predictions = int(
        all_predictions.duplicated(["model", "date", "ric"]).sum()
    )
    if duplicate_predictions:
        raise RuntimeError(f"Duplicate smoothed model/date/ric predictions: {duplicate_predictions}")

    metrics = prediction_metrics(all_predictions)
    monthly = construct_monthly_portfolios(all_predictions, portfolio_quantile)
    summary = portfolio_summary(monthly, metrics, cost_grid_bps, risk_free=risk_free)
    parent_tests = pair_smoothed_against_parents(
        monthly,
        specs,
        cost_bps=significance_cost_bps,
        blocks=significance_blocks,
        n_boot=significance_n_boot,
        seed=random_state,
        risk_free=risk_free,
    )

    primary = summary[
        summary["portfolio"].eq("long_short")
        & summary["weighting"].eq("value")
        & summary["universe_variant"].eq("standard_ex_bottom_5pct")
        & summary["cost_bps"].eq(significance_cost_bps)
    ].copy()
    smoothed_set = set(specs["model"])
    benchmark_tests = pd.DataFrame()
    if benchmark_model in set(monthly["model"]):
        top_smoothed = (
            primary[primary["model"].isin(smoothed_set)]
            .sort_values("sharpe", ascending=False)["model"]
            .head(benchmark_top_smoothed)
            .tolist()
        )
        benchmark_subset = monthly[
            monthly["model"].isin([benchmark_model, *top_smoothed])
        ].copy()
        benchmark_tests = paired_sharpe_significance(
            benchmark_subset,
            baseline_model=benchmark_model,
            cost_bps=significance_cost_bps,
            blocks=significance_blocks,
            n_boot=significance_n_boot,
            seed=random_state,
            risk_free=risk_free,
        )

    smoothed_predictions.to_parquet(
        output_dir / "smoothed_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    all_predictions.to_parquet(
        output_dir / "smoothed_with_parents_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    specs.to_csv(output_dir / "signal_smoothing_specifications.csv", index=False)
    metrics.to_csv(output_dir / "signal_smoothing_prediction_metrics.csv", index=False)
    monthly.to_csv(output_dir / "signal_smoothing_monthly_portfolios.csv", index=False)
    summary.to_csv(output_dir / "signal_smoothing_model_summary.csv", index=False)
    parent_tests.to_csv(output_dir / "smoothing_vs_parent_sharpe_tests.csv", index=False)
    benchmark_test_path = (
        output_dir
        / f"smoothing_top{benchmark_top_smoothed}_vs_"
        f"{_slug(benchmark_model)}_sharpe_tests.csv"
    )
    benchmark_tests.to_csv(
        benchmark_test_path,
        index=False,
    )
    manifest = {
        "input_path": str(input_path),
        "inertia_weights": inertia_weights,
        "portfolio_quantile": portfolio_quantile,
        "cost_grid_bps": cost_grid_bps,
        "significance_cost_bps": significance_cost_bps,
        "significance_blocks": significance_blocks,
        "significance_n_boot": significance_n_boot,
        "benchmark_model": benchmark_model,
        "benchmark_top_smoothed": benchmark_top_smoothed,
        "rows": {
            "parent_predictions": int(len(parent_predictions)),
            "smoothed_predictions": int(len(smoothed_predictions)),
            "all_predictions": int(len(all_predictions)),
            "monthly_portfolios": int(len(monthly)),
            "smoothing_specs": int(len(specs)),
            "parent_sharpe_tests": int(len(parent_tests)),
            "benchmark_sharpe_tests": int(len(benchmark_tests)),
        },
        "causality_check": {
            "duplicate_model_security_month_predictions": duplicate_predictions,
            "prediction_min": float(smoothed_predictions["prediction"].min()),
            "prediction_max": float(smoothed_predictions["prediction"].max()),
        },
        "primary_top_models": (
            primary.sort_values("sharpe", ascending=False)
            .head(20)
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),
        "outputs": {
            "smoothed_predictions": str(output_dir / "smoothed_predictions.parquet"),
            "smoothed_with_parents_predictions": str(
                output_dir / "smoothed_with_parents_predictions.parquet"
            ),
            "signal_smoothing_model_summary": str(
                output_dir / "signal_smoothing_model_summary.csv"
            ),
            "smoothing_vs_parent_sharpe_tests": str(
                output_dir / "smoothing_vs_parent_sharpe_tests.csv"
            ),
            "smoothing_top_vs_benchmark_sharpe_tests": str(
                benchmark_test_path
            ),
        },
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--inertia-weights",
        nargs="+",
        type=float,
        default=DEFAULT_INERTIA_WEIGHTS,
    )
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-grid-bps", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--significance-cost-bps", type=int, default=25)
    parser.add_argument("--significance-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--significance-bootstraps", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--benchmark-model", default="momentum_rank")
    parser.add_argument("--benchmark-top-smoothed", type=int, default=15)
    args = parser.parse_args()

    if any(weight < 0 or weight > 1 for weight in args.inertia_weights):
        raise SystemExit("--inertia-weights must be in [0, 1]")
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = run_signal_smoothing(
        input_path=args.input,
        output_dir=args.output_dir,
        inertia_weights=args.inertia_weights,
        portfolio_quantile=args.portfolio_quantile,
        cost_grid_bps=tuple(args.cost_grid_bps),
        risk_free=risk_free,
        significance_cost_bps=args.significance_cost_bps,
        significance_blocks=tuple(args.significance_blocks),
        significance_n_boot=args.significance_bootstraps,
        random_state=args.random_state,
        benchmark_model=args.benchmark_model,
        benchmark_top_smoothed=args.benchmark_top_smoothed,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(
        "duplicate smoothed predictions: "
        f"{manifest['causality_check']['duplicate_model_security_month_predictions']}",
        flush=True,
    )
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
