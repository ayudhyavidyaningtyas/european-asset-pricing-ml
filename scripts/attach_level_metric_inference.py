"""Attach level-metric bootstrap inference to saved ML output directories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_depth import load_eur_short_rate  # noqa: E402
from asset_pricing_ml import portfolio_summary, prediction_metrics  # noqa: E402


RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_EUR_RATE = (
    PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
)
DEFAULT_RUN_DIRS = (
    RESULTS_ROOT / "estimates_revisions_pure_strict_lag1_revision_signal_ridge",
    RESULTS_ROOT / "dre_estimates_enriched_strict_lag1_ex_ante",
)


def infer_cost_grid(summary_path: Path) -> tuple[int, ...]:
    if not summary_path.exists():
        return (0, 10, 25, 50)
    summary = pd.read_csv(summary_path, usecols=lambda column: column == "cost_bps")
    if "cost_bps" not in summary:
        return (0, 10, 25, 50)
    costs = sorted(pd.to_numeric(summary["cost_bps"], errors="coerce").dropna().unique())
    return tuple(int(cost) for cost in costs) or (0, 10, 25, 50)


def load_monthly_portfolios(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        parse_dates=["signal_date", "return_date"],
    )


def update_delisting_summary(
    run_dir: Path,
    metrics: pd.DataFrame,
    cost_grid: tuple[int, ...],
    risk_free: pd.Series | None,
) -> int:
    monthly_path = run_dir / "delisting_scenario_monthly_portfolios.csv"
    if not monthly_path.exists():
        return 0
    monthly = load_monthly_portfolios(monthly_path)
    frames = []
    group_columns = ["scenario", "missing_delisting_penalty"]
    for keys, group in monthly.groupby(group_columns, dropna=False, sort=True):
        scenario, missing_penalty = keys
        summary = portfolio_summary(
            group,
            metrics,
            cost_grid,
            risk_free=risk_free,
        )
        summary["scenario"] = scenario
        summary["missing_delisting_penalty"] = missing_penalty
        frames.append(summary)
    if not frames:
        return 0
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(run_dir / "delisting_scenario_summary.csv", index=False)
    return int(len(result))


def update_run_dir(
    run_dir: Path,
    *,
    risk_free: pd.Series | None,
) -> dict[str, int | str]:
    prediction_path = run_dir / "predictions.parquet"
    monthly_path = run_dir / "monthly_portfolios.csv"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {prediction_path}")
    if not monthly_path.exists():
        raise FileNotFoundError(f"Missing monthly portfolios file: {monthly_path}")

    predictions = pd.read_parquet(prediction_path)
    metrics = prediction_metrics(predictions)
    cost_grid = infer_cost_grid(run_dir / "model_summary.csv")
    monthly = load_monthly_portfolios(monthly_path)
    summary = portfolio_summary(
        monthly,
        metrics,
        cost_grid,
        risk_free=risk_free,
    )
    metrics.to_csv(run_dir / "prediction_metrics.csv", index=False)
    summary.to_csv(run_dir / "model_summary.csv", index=False)
    delisting_rows = update_delisting_summary(
        run_dir,
        metrics,
        cost_grid,
        risk_free,
    )
    return {
        "run_dir": str(run_dir),
        "prediction_metric_rows": int(len(metrics)),
        "model_summary_rows": int(len(summary)),
        "delisting_summary_rows": delisting_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dirs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_RUN_DIRS),
    )
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--no-risk-free", action="store_true")
    args = parser.parse_args()

    risk_free = None
    if not args.no_risk_free and args.eur_rate.exists():
        risk_free = load_eur_short_rate(args.eur_rate)
    records = [
        update_run_dir(run_dir, risk_free=risk_free)
        for run_dir in args.run_dirs
    ]
    print(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
