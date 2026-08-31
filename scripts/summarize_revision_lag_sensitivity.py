"""Summarize pure analyst-revision lag-sensitivity model runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_OUTPUT = (
    DEFAULT_RESULTS_ROOT
    / "strict_estimates_data_controls"
    / "revision_lag_sensitivity_summary.csv"
)


def read_run(
    lag_months: int,
    run_dir: Path,
    model: str,
    weighting: str,
    cost_bps: int,
) -> pd.DataFrame:
    manifest_path = run_dir / "ml_manifest.json"
    summary_path = run_dir / "model_summary.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing model summary: {summary_path}")

    manifest = json.loads(manifest_path.read_text())
    summary = pd.read_csv(summary_path)
    subset = summary[
        summary["model"].eq(model)
        & summary["target_mode"].eq("rank")
        & summary["weighting"].eq(weighting)
        & summary["cost_bps"].eq(cost_bps)
        & summary["portfolio"].isin(["long_short", "long_only_top_decile"])
    ].copy()
    subset.insert(0, "lag_months", lag_months)
    subset.insert(1, "run_dir", str(run_dir))
    sample_audit = manifest.get("sample_filter_audit", {})
    causality = manifest.get("causality_check", {})
    subset["sample_revision_signal_rows"] = sample_audit.get(
        "after_require_revision_signal",
    )
    subset["model_rows"] = sample_audit.get("model_rows")
    subset["predictions"] = manifest.get("rows", {}).get("predictions")
    subset["lag_guard"] = manifest.get("sample_filter", {}).get(
        "require_estimate_signal_lag_months",
    )
    subset["lag_violations"] = sample_audit.get("estimate_signal_lag_violations")
    subset["causality_violations"] = causality.get("train_target_after_cutoff")
    return subset


def build_summary(
    results_root: Path,
    lags: list[int],
    run_template: str,
    model: str,
    weighting: str,
    cost_bps: int,
) -> pd.DataFrame:
    frames = [
        read_run(
            lag_months=lag,
            run_dir=results_root / run_template.format(lag=lag),
            model=model,
            weighting=weighting,
            cost_bps=cost_bps,
        )
        for lag in lags
    ]
    summary = pd.concat(frames, ignore_index=True)
    keep = [
        "lag_months",
        "run_dir",
        "model",
        "portfolio",
        "weighting",
        "universe_variant",
        "cost_bps",
        "months",
        "observations",
        "sample_revision_signal_rows",
        "model_rows",
        "predictions",
        "annualized_net_mean_return",
        "net_sharpe",
        "annualized_gross_mean_return",
        "gross_sharpe",
        "average_monthly_turnover",
        "mean_monthly_spearman_ic",
        "ic_information_ratio",
        "positive_ic_month_fraction",
        "rank_r2_zero",
        "lag_guard",
        "lag_violations",
        "causality_violations",
    ]
    return summary[keep].sort_values(
        ["universe_variant", "portfolio", "lag_months"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--run-template",
        default="lag_sensitivity_pure_revisions_lag{lag}_ridge",
    )
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--model", default="ridge_rank")
    parser.add_argument("--weighting", default="equal")
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(
        results_root=args.results_root,
        lags=args.lags,
        run_template=args.run_template,
        model=args.model,
        weighting=args.weighting,
        cost_bps=args.cost_bps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"summary -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
