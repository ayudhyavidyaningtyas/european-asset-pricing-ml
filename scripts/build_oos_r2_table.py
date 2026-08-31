"""Out-of-sample predictive R-squared for comparability with the US literature.

The dissertation's primary accuracy metric is the monthly rank IC, which does
not map onto the out-of-sample R-squared that Gu-Kelly-Xiu (2020) and the
literature that follows report. This script computes the Campbell-Thompson /
GKX statistic from the stored return-target predictions,

    R2_oos = 1 - sum (r_it - rhat_it)^2 / sum r_it^2,

where the benchmark forecast is zero (the GKX convention: a zero forecast, not
the historical mean, because the cross-sectional mean is close to zero and the
historical-mean benchmark flatters R2). Reported per model, pooled across all
stock-months and as the mean of monthly R2 values, together with a
market-cap-bucket split so European magnitudes can be lined up against the
GKX-style small/large decomposition.

Rank-target models are excluded: an R2 on a rank target is not comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_RUNS = {
    "europe": RESULTS_ROOT / "europe_compustat_benchmark",
    "us": RESULTS_ROOT / "us_compustat_benchmark",
}
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "oos_r2_comparability"

PREDICTION_COLUMNS = [
    "date",
    "ric",
    "model",
    "target_mode",
    "prediction",
    "target_return_1m",
    "company_market_cap",
]


def r2_out_of_sample(actual: pd.Series, predicted: pd.Series) -> float:
    """GKX out-of-sample R2 against a zero forecast."""
    numerator = float(((actual - predicted) ** 2).sum())
    denominator = float((actual**2).sum())
    if denominator <= 0:
        return np.nan
    return 1.0 - numerator / denominator


def model_r2_rows(
    frame: pd.DataFrame, market: str, top_n: int
) -> list[dict[str, object]]:
    records = []
    ranks = frame.groupby("date", observed=True)["company_market_cap"].rank(
        method="first", ascending=False
    )
    frame = frame.assign(_top=ranks.le(top_n))
    for model, subset in frame.groupby("model", observed=True):
        monthly = subset.groupby("date", observed=True).apply(
            lambda month: r2_out_of_sample(
                month["target_return_1m"], month["prediction"]
            ),
            include_groups=False,
        )
        top = subset[subset["_top"]]
        records.append(
            {
                "market": market,
                "model": model,
                "observations": int(len(subset)),
                "months": int(subset["date"].nunique()),
                "r2_oos_pooled": r2_out_of_sample(
                    subset["target_return_1m"], subset["prediction"]
                ),
                "r2_oos_monthly_mean": float(monthly.mean()),
                "r2_oos_monthly_median": float(monthly.median()),
                "r2_oos_monthly_share_positive": float(monthly.gt(0).mean()),
                f"r2_oos_top_{top_n}": r2_out_of_sample(
                    top["target_return_1m"], top["prediction"]
                ),
                f"r2_oos_ex_top_{top_n}": r2_out_of_sample(
                    subset.loc[~subset["_top"], "target_return_1m"],
                    subset.loc[~subset["_top"], "prediction"],
                ),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("MARKET", "RUN_DIR"),
        help="Repeatable market/run-directory pair; defaults to the EU and US benchmarks.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = (
        {market: Path(run_dir) for market, run_dir in args.run}
        if args.run
        else DEFAULT_RUNS
    )

    records: list[dict[str, object]] = []
    for market, run_dir in runs.items():
        predictions = pd.read_parquet(
            run_dir / "predictions.parquet", columns=PREDICTION_COLUMNS
        )
        returns = predictions[
            predictions["target_mode"].eq("return")
        ].dropna(subset=["prediction", "target_return_1m"])
        if returns.empty:
            print(f"skipping {market}: no return-target predictions in {run_dir}")
            continue
        records.extend(model_r2_rows(returns, market, args.top_n))

    table = pd.DataFrame(records)
    table.to_csv(args.output_dir / "oos_r2_table.csv", index=False)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "runs": {market: str(run_dir) for market, run_dir in runs.items()},
        "statistic": (
            "Campbell-Thompson / GKX out-of-sample R2 against a zero forecast, "
            "computed on return-target predictions only"
        ),
        "top_n": args.top_n,
        "notes": (
            "Rank-target models are excluded because an R2 on a rank target is "
            "not comparable to the literature. Monthly-mean and pooled variants "
            "both reported; the pooled statistic is the GKX headline convention."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(table.round(5).to_string(index=False))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
