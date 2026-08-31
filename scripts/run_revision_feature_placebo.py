"""Monthly within-coverage placebo for analyst revision features."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_ml import (  # noqa: E402
    FEATURE_SETS,
    WalkForwardConfig,
    construct_monthly_portfolios,
    load_model_panel,
    portfolio_summary,
    prediction_metrics,
    run_walk_forward,
)
from estimates_features import ESTIMATES_INFORMATION_TYPES  # noqa: E402


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "strict_estimates_lag1"
    / "monthly_feature_panel_estimates_strict_lag1.parquet"
)
DEFAULT_DELISTING_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "delisting_return_audit.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "revision_feature_placebo"
)


def _target_column(target_mode: str) -> str:
    if target_mode == "rank":
        return "target_return_rank"
    if target_mode == "residual_rank":
        return "target_residual_rank"
    raise ValueError("Revision feature placebo supports rank or residual_rank targets")


def _portfolio_row(
    summary: pd.DataFrame,
    model: str,
    *,
    cost_bps: int,
) -> dict[str, Any]:
    row = summary[
        summary["model"].eq(model)
        & summary["cost_bps"].eq(cost_bps)
        & summary["weighting"].eq("equal")
        & summary["universe_variant"].eq("standard_ex_bottom_5pct")
        & summary["portfolio"].eq("long_short")
    ].iloc[0]
    keep = [
        "annualized_gross_mean_return",
        "gross_sharpe",
        "annualized_net_mean_return",
        "net_sharpe",
        "average_monthly_turnover",
        "max_drawdown",
    ]
    return {column: float(row[column]) for column in keep}


def _shuffle_revision_features(
    panel: pd.DataFrame,
    revision_columns: list[str],
    *,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = panel.copy()
    for _, index in out.groupby("date", sort=False).groups.items():
        if len(index) < 2:
            continue
        values = out.loc[index, revision_columns].to_numpy(copy=True)
        permutation = rng.permutation(len(index))
        out.loc[index, revision_columns] = values[permutation]
    return out


def _run_model(
    panel: pd.DataFrame,
    config: WalkForwardConfig,
    *,
    target_mode: str,
    feature_columns: list[str],
    cost_bps: int,
) -> tuple[pd.Series, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    predictions, fit_log, _, _ = run_walk_forward(
        panel,
        ["ridge"],
        config,
        target_column=_target_column(target_mode),
        target_mode=target_mode,
        feature_columns=feature_columns,
        collect_importance=False,
    )
    metrics = prediction_metrics(predictions)
    monthly = construct_monthly_portfolios(predictions, config.portfolio_quantile)
    summary = portfolio_summary(monthly, metrics, (cost_bps,))
    model = f"ridge_{target_mode}"
    metric = metrics[metrics["model"].eq(model)].iloc[0]
    row = {
        **metric.to_dict(),
        **_portfolio_row(summary, model, cost_bps=cost_bps),
        "causality_violations": int(
            (
                pd.to_datetime(fit_log["train_target_end"])
                > pd.to_datetime(fit_log["train_label_cutoff"])
            ).sum()
        ),
    }
    return predictions["prediction"], row, metrics, summary


def run_revision_feature_placebo(
    panel_path: Path,
    output_dir: Path,
    feature_set: str,
    target_mode: str,
    repetitions: int,
    sample_start_date: str | None,
    require_revision_signal: bool,
    require_estimate_signal_lag_months: int | None,
    delisting_audit_path: Path | None,
    config: WalkForwardConfig,
    cost_bps: int,
) -> dict[str, Any]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = FEATURE_SETS[feature_set]
    revision_columns = [
        f"{feature}_rank"
        for feature in ESTIMATES_INFORMATION_TYPES["revisions"]
        if f"{feature}_rank" in feature_columns
    ]
    if not revision_columns:
        raise ValueError(f"Feature set {feature_set!r} has no revision columns")

    panel = load_model_panel(
        panel_path,
        delisting_audit_path=delisting_audit_path,
        feature_columns=feature_columns,
        sample_start_date=sample_start_date,
        require_revision_signal=require_revision_signal,
        require_estimate_signal_lag_months=require_estimate_signal_lag_months,
    )
    sample_filter_audit = dict(panel.attrs.get("sample_filter_audit", {}))
    panel.attrs = {}

    _, actual, actual_metrics, actual_summary = _run_model(
        panel,
        config,
        target_mode=target_mode,
        feature_columns=feature_columns,
        cost_bps=cost_bps,
    )
    actual_metrics.to_csv(output_dir / "actual_prediction_metrics.csv", index=False)
    actual_summary.to_csv(output_dir / "actual_model_summary.csv", index=False)

    records = []
    for repetition in range(repetitions):
        placebo_panel = _shuffle_revision_features(
            panel,
            revision_columns,
            seed=config.random_state + repetition,
        )
        _, row, _, _ = _run_model(
            placebo_panel,
            config,
            target_mode=target_mode,
            feature_columns=feature_columns,
            cost_bps=cost_bps,
        )
        records.append({"repetition": repetition, **row})
    placebo = pd.DataFrame(records)
    placebo.to_csv(output_dir / "revision_feature_placebo_metrics.csv", index=False)

    placebo_ic = placebo["mean_monthly_spearman_ic"]
    placebo_sharpe = placebo["net_sharpe"]
    placebo_return = placebo["annualized_net_mean_return"]
    summary = pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "target_mode": target_mode,
                "revision_columns": json.dumps(revision_columns),
                "repetitions": repetitions,
                "actual_mean_monthly_ic": actual["mean_monthly_spearman_ic"],
                "placebo_mean_monthly_ic_mean": float(placebo_ic.mean()),
                "placebo_mean_monthly_ic_p95": float(placebo_ic.quantile(0.95)),
                "p_placebo_ic_ge_actual": float(
                    placebo_ic.ge(actual["mean_monthly_spearman_ic"]).mean()
                ),
                "actual_net_sharpe": actual["net_sharpe"],
                "placebo_net_sharpe_mean": float(placebo_sharpe.mean()),
                "placebo_net_sharpe_p95": float(placebo_sharpe.quantile(0.95)),
                "p_placebo_sharpe_ge_actual": float(
                    placebo_sharpe.ge(actual["net_sharpe"]).mean()
                ),
                "actual_annualized_net_return": actual[
                    "annualized_net_mean_return"
                ],
                "placebo_annualized_net_return_mean": float(
                    placebo_return.mean()
                ),
                "placebo_annualized_net_return_p95": float(
                    placebo_return.quantile(0.95)
                ),
                "p_placebo_return_ge_actual": float(
                    placebo_return.ge(actual["annualized_net_mean_return"]).mean()
                ),
            }
        ]
    )
    summary.to_csv(output_dir / "revision_feature_placebo_summary.csv", index=False)

    manifest = {
        "panel_path": str(panel_path),
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "target_mode": target_mode,
        "revision_columns_shuffled": revision_columns,
        "sample_filter_audit": sample_filter_audit,
        "config": asdict(config),
        "cost_bps": cost_bps,
        "rows": {
            "panel": int(len(panel)),
            "repetitions": repetitions,
            "placebo_metrics": int(len(placebo)),
            "causality_violations": int(placebo["causality_violations"].sum()),
        },
        "outputs": {
            "actual_prediction_metrics": str(
                output_dir / "actual_prediction_metrics.csv"
            ),
            "actual_model_summary": str(output_dir / "actual_model_summary.csv"),
            "revision_feature_placebo_metrics": str(
                output_dir / "revision_feature_placebo_metrics.csv"
            ),
            "revision_feature_placebo_summary": str(
                output_dir / "revision_feature_placebo_summary.csv"
            ),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="estimates_revisions_pure",
    )
    parser.add_argument(
        "--target-mode",
        choices=["rank", "residual_rank"],
        default="rank",
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--sample-start-date")
    parser.add_argument("--require-revision-signal", action="store_true")
    parser.add_argument("--require-estimate-signal-lag-months", type=int)
    parser.add_argument("--delisting-audit", type=Path, default=DEFAULT_DELISTING_AUDIT)
    parser.add_argument("--skip-delisting-scenarios", action="store_true")
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--min-training-rows", type=int, default=10_000)
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = WalkForwardConfig(
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        min_training_rows=args.min_training_rows,
        portfolio_quantile=args.portfolio_quantile,
        cost_grid_bps=(args.cost_bps,),
        random_state=args.random_state,
        tune_hyperparameters=False,
    )
    delisting_audit = (
        None if args.skip_delisting_scenarios else args.delisting_audit
    )
    manifest = run_revision_feature_placebo(
        panel_path=args.panel,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        target_mode=args.target_mode,
        repetitions=args.repetitions,
        sample_start_date=args.sample_start_date,
        require_revision_signal=args.require_revision_signal,
        require_estimate_signal_lag_months=args.require_estimate_signal_lag_months,
        delisting_audit_path=delisting_audit,
        config=config,
        cost_bps=args.cost_bps,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
