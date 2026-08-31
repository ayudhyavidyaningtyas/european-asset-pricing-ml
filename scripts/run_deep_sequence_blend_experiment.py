"""Test fixed rank-score blends between sequence models and ML benchmarks."""
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
    predictive_accuracy_tests,
)


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_common_benchmark"
    / "common_predictions.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_blend_experiment"
)
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_BASE_MODELS = [
    "momentum_rank",
    "ridge_rank",
    "dre_rank",
    "hist_gbm_rank",
    "mlp_rank",
]
DEFAULT_SEQUENCE_MODELS = [
    "lstm_seq12_rank",
    "gru_seq12_rank",
    "attention_lstm_seq12_rank",
    "lstm_seq24_rank",
    "gru_seq24_rank",
    "attention_lstm_seq24_rank",
]
DEFAULT_BLEND_WEIGHTS = [0.9, 0.75, 0.5]


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")


def _slug(model: str) -> str:
    replacements = {
        "attention_lstm": "attn",
        "momentum": "mom",
        "hist_gbm": "gbm",
        "_rank": "",
    }
    slug = model
    for old, new in replacements.items():
        slug = slug.replace(old, new)
    return slug


def load_common_predictions(path: Path, models: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    predictions = pd.read_parquet(path)
    _require_columns(
        predictions,
        [
            "date",
            "target_date",
            "ric",
            "model",
            "prediction",
            "target_return_1m",
            "target_return_rank",
            "company_market_cap",
            "market_cap_percentile",
            "target_mode",
        ],
    )
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    predictions = predictions[predictions["model"].isin(models)].copy()
    found = set(predictions["model"].unique())
    missing = sorted(set(models) - found)
    if missing:
        raise RuntimeError(f"Input is missing requested models: {missing}")
    duplicates = int(predictions.duplicated(["model", "date", "ric"]).sum())
    if duplicates:
        raise RuntimeError(f"Duplicate model/date/ric predictions: {duplicates}")
    counts = (
        predictions[["date", "ric", "model"]]
        .drop_duplicates()
        .groupby(["date", "ric"], sort=False)["model"]
        .nunique()
    )
    incomplete = int(counts.lt(len(models)).sum())
    if incomplete:
        raise RuntimeError(
            "Input is not a strict common model/security/month sample; "
            f"{incomplete} keys are incomplete."
        )
    return predictions.sort_values(["model", "date", "ric"]).reset_index(drop=True)


def monthly_rank_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    scores = predictions[["model", "date", "ric", "prediction"]].copy()
    scores["rank_score"] = (
        scores.groupby(["model", "date"], sort=False)["prediction"]
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )
    return scores.drop(columns=["prediction"])


def metadata_by_stock_month(predictions: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in predictions.columns
        if column not in {"model", "prediction"}
    ]
    metadata = predictions[metadata_columns].drop_duplicates(["date", "ric"])
    duplicate_keys = int(metadata.duplicated(["date", "ric"]).sum())
    if duplicate_keys:
        raise RuntimeError(f"Duplicate metadata rows after de-duplication: {duplicate_keys}")
    return metadata


def build_fixed_blends(
    predictions: pd.DataFrame,
    base_models: list[str],
    sequence_models: list[str],
    base_weights: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = monthly_rank_scores(predictions)
    wide_scores = scores.pivot(
        index=["date", "ric"],
        columns="model",
        values="rank_score",
    )
    metadata = metadata_by_stock_month(predictions).set_index(["date", "ric"])
    records: list[pd.DataFrame] = []
    spec_records = []
    for base_model in base_models:
        for sequence_model in sequence_models:
            for base_weight in base_weights:
                sequence_weight = 1.0 - base_weight
                blend_model = (
                    f"blend{int(round(base_weight * 100)):02d}_"
                    f"{_slug(base_model)}_{_slug(sequence_model)}_rank"
                )
                blend_score = (
                    base_weight * wide_scores[base_model]
                    + sequence_weight * wide_scores[sequence_model]
                )
                frame = metadata.copy()
                frame["model"] = blend_model
                frame["prediction"] = blend_score
                frame["target_mode"] = "rank"
                records.append(frame.reset_index())
                spec_records.append(
                    {
                        "model": blend_model,
                        "base_model": base_model,
                        "sequence_model": sequence_model,
                        "base_weight": float(base_weight),
                        "sequence_weight": float(sequence_weight),
                    }
                )
    blends = pd.concat(records, ignore_index=True)
    specs = pd.DataFrame(spec_records)
    return blends, specs


def pair_blends_against_parents(
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
    for base_model, group in specs.groupby("base_model", sort=True):
        candidate_models = [base_model, *sorted(group["model"].unique())]
        subset = monthly[monthly["model"].isin(candidate_models)]
        test = paired_sharpe_significance(
            subset,
            baseline_model=base_model,
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
    tests = pd.concat(frames, ignore_index=True)
    return tests.merge(specs, on="model", how="left")


def select_ladder_models(
    summary: pd.DataFrame,
    specs: pd.DataFrame,
    base_models: list[str],
    sequence_models: list[str],
    top_blends: int,
) -> list[str]:
    primary = summary[
        summary["portfolio"].eq("long_short")
        & summary["weighting"].eq("value")
        & summary["universe_variant"].eq("standard_ex_bottom_5pct")
        & summary["cost_bps"].eq(25)
    ].copy()
    blend_set = set(specs["model"])
    selected_blends = (
        primary[primary["model"].isin(blend_set)]
        .sort_values("sharpe", ascending=False)["model"]
        .head(top_blends)
        .tolist()
    )
    return sorted(set([*base_models, *sequence_models, *selected_blends]))


def build_blend_outputs(
    input_path: Path,
    output_dir: Path,
    base_models: list[str],
    sequence_models: list[str],
    base_weights: list[float],
    portfolio_quantile: float,
    cost_grid_bps: tuple[int, ...],
    risk_free: pd.Series | None,
    significance_cost_bps: int,
    significance_blocks: tuple[int, ...],
    significance_n_boot: int,
    random_state: int,
    ladder_top_blends: int,
    run_predictive_tests: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_models = [*base_models, *sequence_models]
    parent_predictions = load_common_predictions(input_path, requested_models)
    blends, specs = build_fixed_blends(
        parent_predictions,
        base_models,
        sequence_models,
        base_weights,
    )
    all_predictions = pd.concat([parent_predictions, blends], ignore_index=True)
    all_predictions = all_predictions.sort_values(["model", "date", "ric"]).reset_index(
        drop=True
    )
    duplicate_predictions = int(
        all_predictions.duplicated(["model", "date", "ric"]).sum()
    )
    if duplicate_predictions:
        raise RuntimeError(f"Duplicate blended model/date/ric predictions: {duplicate_predictions}")

    metrics = prediction_metrics(all_predictions)
    monthly = construct_monthly_portfolios(all_predictions, portfolio_quantile)
    summary = portfolio_summary(monthly, metrics, cost_grid_bps, risk_free=risk_free)
    if run_predictive_tests:
        loss_tests, ic_tests = predictive_accuracy_tests(all_predictions)
    else:
        loss_tests = pd.DataFrame()
        ic_tests = pd.DataFrame()
    parent_tests = pair_blends_against_parents(
        monthly,
        specs,
        cost_bps=significance_cost_bps,
        blocks=significance_blocks,
        n_boot=significance_n_boot,
        seed=random_state,
        risk_free=risk_free,
    )
    ladder_models = select_ladder_models(
        summary,
        specs,
        base_models,
        sequence_models,
        ladder_top_blends,
    )
    ladder_predictions = all_predictions[
        all_predictions["model"].isin(ladder_models)
    ].copy()

    blends.to_parquet(
        output_dir / "blend_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    all_predictions.to_parquet(
        output_dir / "blend_with_parents_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    ladder_predictions.to_parquet(
        output_dir / "blend_ladder_subset_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    specs.to_csv(output_dir / "blend_specifications.csv", index=False)
    metrics.to_csv(output_dir / "blend_prediction_metrics.csv", index=False)
    monthly.to_csv(output_dir / "blend_monthly_portfolios.csv", index=False)
    summary.to_csv(output_dir / "blend_model_summary.csv", index=False)
    loss_tests.to_csv(output_dir / "blend_predictive_accuracy_loss_tests.csv", index=False)
    ic_tests.to_csv(output_dir / "blend_predictive_accuracy_ic_tests.csv", index=False)
    parent_tests.to_csv(output_dir / "blend_vs_parent_sharpe_tests.csv", index=False)

    primary = summary[
        summary["portfolio"].eq("long_short")
        & summary["weighting"].eq("value")
        & summary["universe_variant"].eq("standard_ex_bottom_5pct")
        & summary["cost_bps"].eq(significance_cost_bps)
    ].copy()
    primary["is_blend"] = primary["model"].isin(set(specs["model"]))
    manifest = {
        "input_path": str(input_path),
        "base_models": base_models,
        "sequence_models": sequence_models,
        "base_weights": base_weights,
        "portfolio_quantile": portfolio_quantile,
        "cost_grid_bps": cost_grid_bps,
        "significance_cost_bps": significance_cost_bps,
        "significance_blocks": significance_blocks,
        "significance_n_boot": significance_n_boot,
        "ladder_top_blends": ladder_top_blends,
        "run_predictive_tests": run_predictive_tests,
        "ladder_models": ladder_models,
        "rows": {
            "parent_predictions": int(len(parent_predictions)),
            "blend_predictions": int(len(blends)),
            "all_predictions": int(len(all_predictions)),
            "ladder_subset_predictions": int(len(ladder_predictions)),
            "monthly_portfolios": int(len(monthly)),
            "blend_specs": int(len(specs)),
            "parent_sharpe_tests": int(len(parent_tests)),
        },
        "causality_check": {
            "duplicate_model_security_month_predictions": duplicate_predictions,
            "blend_prediction_max_abs_score": float(blends["prediction"].abs().max()),
            "blend_prediction_min": float(blends["prediction"].min()),
            "blend_prediction_max": float(blends["prediction"].max()),
        },
        "primary_top_models": (
            primary.sort_values("sharpe", ascending=False)
            .head(20)
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),
        "outputs": {
            "blend_predictions": str(output_dir / "blend_predictions.parquet"),
            "blend_with_parents_predictions": str(
                output_dir / "blend_with_parents_predictions.parquet"
            ),
            "blend_ladder_subset_predictions": str(
                output_dir / "blend_ladder_subset_predictions.parquet"
            ),
            "blend_specifications": str(output_dir / "blend_specifications.csv"),
            "blend_model_summary": str(output_dir / "blend_model_summary.csv"),
            "blend_vs_parent_sharpe_tests": str(
                output_dir / "blend_vs_parent_sharpe_tests.csv"
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
    parser.add_argument("--base-models", nargs="+", default=DEFAULT_BASE_MODELS)
    parser.add_argument("--sequence-models", nargs="+", default=DEFAULT_SEQUENCE_MODELS)
    parser.add_argument(
        "--base-weights",
        nargs="+",
        type=float,
        default=DEFAULT_BLEND_WEIGHTS,
        help="Fixed parent-model weights; sequence weight is 1 minus this value.",
    )
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-grid-bps", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--significance-cost-bps", type=int, default=25)
    parser.add_argument("--significance-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--significance-bootstraps", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ladder-top-blends", type=int, default=12)
    parser.add_argument(
        "--run-predictive-tests",
        action="store_true",
        help=(
            "Run all-pairs Diebold-Mariano/IC tests across every blend. "
            "This is slow for broad blend grids and is disabled by default."
        ),
    )
    args = parser.parse_args()

    if any(weight < 0 or weight > 1 for weight in args.base_weights):
        raise SystemExit("--base-weights must be in [0, 1]")
    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_blend_outputs(
        input_path=args.input,
        output_dir=args.output_dir,
        base_models=args.base_models,
        sequence_models=args.sequence_models,
        base_weights=args.base_weights,
        portfolio_quantile=args.portfolio_quantile,
        cost_grid_bps=tuple(args.cost_grid_bps),
        risk_free=risk_free,
        significance_cost_bps=args.significance_cost_bps,
        significance_blocks=tuple(args.significance_blocks),
        significance_n_boot=args.significance_bootstraps,
        random_state=args.random_state,
        ladder_top_blends=args.ladder_top_blends,
        run_predictive_tests=args.run_predictive_tests,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(
        "duplicate blended predictions: "
        f"{manifest['causality_check']['duplicate_model_security_month_predictions']}",
        flush=True,
    )
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
