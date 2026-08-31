"""Compare deep sequence models against standard Compustat ML benchmarks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_common_benchmark"
)
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_INPUTS = [
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "compustat_enriched_full_layer1_p96"
    / "predictions.parquet",
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "compustat_enriched_nonlinear_rank"
    / "predictions.parquet",
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_compustat_full_seq12"
    / "predictions.parquet",
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_compustat_full_seq24"
    / "predictions.parquet",
]
DEFAULT_MODELS = [
    "momentum_rank",
    "ridge_rank",
    "dre_rank",
    "hist_gbm_rank",
    "mlp_rank",
    "last_mlp_seq12_rank",
    "lstm_seq12_rank",
    "gru_seq12_rank",
    "attention_lstm_seq12_rank",
    "lstm_seq24_rank",
    "gru_seq24_rank",
    "attention_lstm_seq24_rank",
]
DEFAULT_BASELINES = [
    "momentum_rank",
    "ridge_rank",
    "dre_rank",
    "hist_gbm_rank",
    "mlp_rank",
]
DEFAULT_SEQUENCE_MODELS = [
    "last_mlp_seq12_rank",
    "lstm_seq12_rank",
    "gru_seq12_rank",
    "attention_lstm_seq12_rank",
    "lstm_seq24_rank",
    "gru_seq24_rank",
    "attention_lstm_seq24_rank",
]


def load_prediction_inputs(paths: list[Path], models: list[str]) -> pd.DataFrame:
    frames = []
    required = set(models)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        frame = frame[frame["model"].isin(required)].copy()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No requested models were found in the input predictions")
    predictions = pd.concat(frames, ignore_index=True)
    found = set(predictions["model"].unique())
    missing = required - found
    if missing:
        raise RuntimeError(f"Missing requested models: {sorted(missing)}")
    duplicates = predictions.duplicated(["model", "date", "ric"]).sum()
    if duplicates:
        raise RuntimeError(f"Duplicate model/date/ric predictions: {duplicates}")
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    return predictions


def restrict_to_common_stock_months(
    predictions: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    keys = (
        predictions[["date", "ric", "model"]]
        .drop_duplicates()
        .groupby(["date", "ric"], sort=False)["model"]
        .nunique()
    )
    common_keys = keys[keys.eq(len(models))].index
    common = predictions.set_index(["date", "ric"]).loc[common_keys].reset_index()
    return common.sort_values(["model", "date", "ric"]).reset_index(drop=True)


def pair_sequence_against_baselines(
    monthly: pd.DataFrame,
    sequence_models: list[str],
    baselines: list[str],
    *,
    cost_bps: int,
    blocks: tuple[int, ...],
    n_boot: int,
    seed: int,
    risk_free: pd.Series | None,
) -> pd.DataFrame:
    frames = []
    for baseline in baselines:
        subset = monthly[monthly["model"].isin([baseline, *sequence_models])]
        if baseline not in set(subset["model"]):
            continue
        test = paired_sharpe_significance(
            subset,
            baseline_model=baseline,
            cost_bps=cost_bps,
            blocks=blocks,
            n_boot=n_boot,
            seed=seed,
            risk_free=risk_free,
        )
        if test.empty:
            continue
        test = test[test["model"].isin(sequence_models)].copy()
        frames.append(test)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_common_benchmark_outputs(
    input_paths: list[Path],
    output_dir: Path,
    models: list[str],
    sequence_models: list[str],
    baselines: list[str],
    portfolio_quantile: float,
    cost_grid_bps: tuple[int, ...],
    risk_free: pd.Series | None,
    significance_cost_bps: int,
    significance_blocks: tuple[int, ...],
    significance_n_boot: int,
    random_state: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_prediction_inputs(input_paths, models)
    common = restrict_to_common_stock_months(predictions, models)
    if common.empty:
        raise RuntimeError("Common model/security/month sample is empty")

    metrics = prediction_metrics(common)
    monthly = construct_monthly_portfolios(common, portfolio_quantile)
    summary = portfolio_summary(monthly, metrics, cost_grid_bps, risk_free=risk_free)
    loss_tests, ic_tests = predictive_accuracy_tests(common)
    sequence_vs_baselines = pair_sequence_against_baselines(
        monthly,
        sequence_models,
        baselines,
        cost_bps=significance_cost_bps,
        blocks=significance_blocks,
        n_boot=significance_n_boot,
        seed=random_state,
        risk_free=risk_free,
    )

    predictions.to_parquet(
        output_dir / "all_requested_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    common.to_parquet(
        output_dir / "common_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    metrics.to_csv(output_dir / "common_prediction_metrics.csv", index=False)
    monthly.to_csv(output_dir / "common_monthly_portfolios.csv", index=False)
    summary.to_csv(output_dir / "common_model_summary.csv", index=False)
    loss_tests.to_csv(output_dir / "common_predictive_accuracy_loss_tests.csv", index=False)
    ic_tests.to_csv(output_dir / "common_predictive_accuracy_ic_tests.csv", index=False)
    sequence_vs_baselines.to_csv(
        output_dir / "sequence_vs_baseline_sharpe_tests.csv",
        index=False,
    )

    model_counts = (
        common.groupby("model", sort=True)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )
    manifest = {
        "input_paths": [str(path) for path in input_paths],
        "models": models,
        "sequence_models": sequence_models,
        "baselines": baselines,
        "portfolio_quantile": portfolio_quantile,
        "cost_grid_bps": cost_grid_bps,
        "significance_cost_bps": significance_cost_bps,
        "significance_blocks": significance_blocks,
        "significance_n_boot": significance_n_boot,
        "rows": {
            "loaded_predictions": int(len(predictions)),
            "common_predictions": int(len(common)),
            "common_stock_months": int(common[["date", "ric"]].drop_duplicates().shape[0]),
            "monthly_portfolios": int(len(monthly)),
            "sharpe_tests": int(len(sequence_vs_baselines)),
        },
        "model_counts": model_counts,
        "causality_check": {
            "duplicate_model_security_month_predictions": int(
                common.duplicated(["model", "date", "ric"]).sum()
            )
        },
        "outputs": {
            "common_predictions": str(output_dir / "common_predictions.parquet"),
            "common_prediction_metrics": str(output_dir / "common_prediction_metrics.csv"),
            "common_monthly_portfolios": str(output_dir / "common_monthly_portfolios.csv"),
            "common_model_summary": str(output_dir / "common_model_summary.csv"),
            "sequence_vs_baseline_sharpe_tests": str(
                output_dir / "sequence_vs_baseline_sharpe_tests.csv"
            ),
        },
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sequence-models", nargs="+", default=DEFAULT_SEQUENCE_MODELS)
    parser.add_argument("--baselines", nargs="+", default=DEFAULT_BASELINES)
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-grid-bps", nargs="+", type=int, default=[0, 10, 25, 50])
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--significance-cost-bps", type=int, default=25)
    parser.add_argument("--significance-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--significance-bootstraps", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    risk_free = load_eur_short_rate(args.eur_rate) if args.eur_rate.exists() else None
    manifest = build_common_benchmark_outputs(
        input_paths=args.input or DEFAULT_INPUTS,
        output_dir=args.output_dir,
        models=args.models,
        sequence_models=args.sequence_models,
        baselines=args.baselines,
        portfolio_quantile=args.portfolio_quantile,
        cost_grid_bps=tuple(args.cost_grid_bps),
        risk_free=risk_free,
        significance_cost_bps=args.significance_cost_bps,
        significance_blocks=tuple(args.significance_blocks),
        significance_n_boot=args.significance_bootstraps,
        random_state=args.random_state,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(
        "duplicate common predictions: "
        f"{manifest['causality_check']['duplicate_model_security_month_predictions']}",
        flush=True,
    )
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
