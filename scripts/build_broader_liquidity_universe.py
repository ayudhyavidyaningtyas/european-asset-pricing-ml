"""Build broader Refinitiv RIC universes for constrained portfolio tests.

The existing liquidity pull covers the top-500 implementable frontier.  This
script constructs the union of RICs that can enter the frozen deep/hybrid and
benchmark portfolios when the market-cap cap is raised to top-1000 or top-2000.
"""
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


DEFAULT_SELECTED = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "validation_selected_deep_hybrid_liquid"
    / "validation_selected_monthly.csv"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "validation_selected_implementable_strategy"
    / "candidate_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_exports"
)
DEFAULT_BASELINE_MODELS = [
    "momentum_rank",
    "ridge_rank",
    "smooth75_ridge_rank",
    "blend90_gbm_attn_seq24_rank",
]


def load_required_model_dates(
    selected_path: Path,
    baseline_models: list[str],
) -> pd.DataFrame:
    selected = pd.read_csv(selected_path, parse_dates=["date", "target_date"])
    selected = selected[
        selected["strategy"].eq("validation_selected_long_only")
        & selected["selected_portfolio"].eq("long_only")
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No frozen long-only choices found in {selected_path}")
    deep = selected[["date", "target_date", "model"]].copy()
    baseline_frames = [
        selected[["date", "target_date"]].assign(model=model)
        for model in baseline_models
    ]
    choices = pd.concat([deep, *baseline_frames], ignore_index=True)
    choices["date"] = pd.to_datetime(choices["date"])
    return choices.drop_duplicates(["date", "model"]).sort_values(["date", "model"])


def read_prediction_subset(predictions_path: Path, choices: pd.DataFrame) -> pd.DataFrame:
    models = sorted(choices["model"].dropna().unique().tolist())
    dates = sorted(pd.to_datetime(choices["date"]).unique().tolist())
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(predictions_path, format="parquet")
        table = dataset.to_table(
            filter=ds.field("model").isin(models) & ds.field("date").isin(dates),
            columns=[
                "date",
                "ric",
                "model",
                "prediction",
                "target_return_1m",
                "company_market_cap",
                "market_cap_percentile",
            ],
        )
        predictions = table.to_pandas()
    except Exception:
        predictions = pd.read_parquet(
            predictions_path,
            columns=[
                "date",
                "ric",
                "model",
                "prediction",
                "target_return_1m",
                "company_market_cap",
                "market_cap_percentile",
            ],
        )
        predictions = predictions[
            predictions["model"].isin(models)
            & pd.to_datetime(predictions["date"]).isin(dates)
        ].copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    return predictions.dropna(subset=["prediction", "target_return_1m"])


def top_n_universe(
    predictions: pd.DataFrame,
    choices: pd.DataFrame,
    maximum_assets: int,
) -> pd.DataFrame:
    required = choices[["date", "model"]].drop_duplicates()
    work = predictions.merge(required, on=["date", "model"], how="inner")
    work = work[
        work["market_cap_percentile"].ge(0.30)
        & pd.to_numeric(work["company_market_cap"], errors="coerce").gt(0)
    ].copy()
    selected = []
    for _, group in work.groupby(["date", "model"], sort=True):
        selected.append(group.nlargest(maximum_assets, "company_market_cap"))
    if not selected:
        return pd.DataFrame(columns=["ric"])
    universe = pd.concat(selected, ignore_index=True)
    return (
        universe[["ric"]]
        .dropna()
        .assign(ric=lambda frame: frame["ric"].astype(str).str.strip())
        .query("ric != ''")
        .drop_duplicates()
        .sort_values("ric")
        .reset_index(drop=True)
    )


def run(
    selected_path: Path,
    predictions_path: Path,
    output_dir: Path,
    maximum_assets: list[int],
    baseline_models: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    choices = load_required_model_dates(selected_path, baseline_models)
    predictions = read_prediction_subset(predictions_path, choices)
    outputs = {}
    for n_assets in maximum_assets:
        universe = top_n_universe(predictions, choices, n_assets)
        path = output_dir / f"implementable_frontier_universe_top{n_assets}.csv"
        universe.to_csv(path, index=False)
        outputs[str(n_assets)] = {
            "path": str(path),
            "rics": int(universe["ric"].nunique()),
        }
    manifest = {
        "inputs": {
            "selected": str(selected_path),
            "predictions": str(predictions_path),
        },
        "baseline_models": baseline_models,
        "choice_rows": int(len(choices)),
        "prediction_rows": int(len(predictions)),
        "outputs": outputs,
    }
    with (output_dir / "broader_liquidity_universe_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maximum-assets", nargs="+", type=int, default=[1000, 2000])
    parser.add_argument("--baseline-models", nargs="+", default=DEFAULT_BASELINE_MODELS)
    args = parser.parse_args()

    manifest = run(
        selected_path=args.selected,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        maximum_assets=args.maximum_assets,
        baseline_models=args.baseline_models,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
