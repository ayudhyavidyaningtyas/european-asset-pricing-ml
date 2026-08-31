"""Build a size decomposition exhibit for the deep asset-pricing screen.

The exhibit explains how high cross-sectional IC can coexist with weak
implementable Sharpe: prediction strength may concentrate in smaller names,
where equal-weight paper spreads are harder to scale and turnover is costly.
It works from saved predictions only; no model re-estimation is involved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stats import hac_mean_diff_test  # noqa: E402


DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "dre_estimates_enriched_strict_lag1_ex_ante"
    / "predictions.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_size_decomposition_strict_lag1"
)
DEFAULT_MODELS = ("momentum_rank", "ridge_rank", "hist_gbm_rank", "mlp_rank", "dre_rank")
SIZE_BUCKETS = (
    ("small", 0.05, 1.0 / 3.0),
    ("middle", 1.0 / 3.0, 2.0 / 3.0),
    ("large", 2.0 / 3.0, 1.0),
)


def load_predictions(path: Path, models: Iterable[str]) -> pd.DataFrame:
    columns = [
        "date",
        "ric",
        "model",
        "prediction",
        "target_return_rank",
        "target_return_1m",
        "company_market_cap",
        "market_cap_percentile",
    ]
    predictions = pd.read_parquet(path, columns=columns)
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions = predictions[predictions["model"].isin(set(models))].copy()
    predictions = predictions.dropna(
        subset=[
            "prediction",
            "target_return_rank",
            "target_return_1m",
            "market_cap_percentile",
            "company_market_cap",
        ]
    )
    if predictions.empty:
        raise RuntimeError("No requested model predictions found")
    duplicates = predictions.duplicated(["model", "date", "ric"]).sum()
    if duplicates:
        raise RuntimeError(f"Duplicate model/date/ric predictions: {duplicates}")
    return predictions


def size_bucket_masks(month: pd.DataFrame, top_n: int) -> list[tuple[str, pd.Series]]:
    percentile = pd.to_numeric(month["market_cap_percentile"], errors="coerce")
    masks = [
        (
            name,
            percentile.gt(lower) & percentile.le(upper),
        )
        for name, lower, upper in SIZE_BUCKETS
    ]
    masks.insert(0, ("all_standard_ex_bottom_5pct", percentile.ge(0.05)))
    top_index = month.nlargest(min(top_n, len(month)), "company_market_cap").index
    masks.append(("top_500_by_market_cap", month.index.isin(top_index)))
    return masks


def _safe_spearman(month: pd.DataFrame, minimum_cross_section: int) -> float:
    if len(month) < minimum_cross_section:
        return np.nan
    return float(month["prediction"].corr(month["target_return_rank"], method="spearman"))


def _decile_weights(month: pd.DataFrame, quantile: float) -> tuple[dict[str, float], int, int]:
    if month["prediction"].nunique() < 2:
        return {}, 0, 0
    rank = month["prediction"].rank(method="first", pct=True)
    long = month.loc[rank.gt(1.0 - quantile)]
    short = month.loc[rank.le(quantile)]
    if long.empty or short.empty:
        return {}, 0, 0
    weights = {
        **dict(zip(long["ric"], np.repeat(1.0 / len(long), len(long)), strict=True)),
        **dict(zip(short["ric"], np.repeat(-1.0 / len(short), len(short)), strict=True)),
    }
    return weights, len(long), len(short)


def build_monthly_size_decomposition(
    predictions: pd.DataFrame,
    *,
    quantile: float = 0.10,
    cost_bps: int = 25,
    top_n: int = 500,
    minimum_cross_section: int = 30,
) -> pd.DataFrame:
    records = []
    previous_weights: dict[tuple[str, str], dict[str, float]] = {}
    for (model, date), month in predictions.sort_values(
        ["model", "date", "ric"]
    ).groupby(["model", "date"], sort=True):
        for bucket, mask in size_bucket_masks(month, top_n):
            cell = month.loc[mask].copy()
            if cell.empty:
                continue
            ic = _safe_spearman(cell, minimum_cross_section)
            weights, long_n, short_n = _decile_weights(cell, quantile)
            if weights:
                returns = cell.set_index("ric")["target_return_1m"]
                weight_series = pd.Series(weights, dtype=float)
                gross_return = float(
                    (weight_series * returns.reindex(weight_series.index)).sum()
                )
                previous = previous_weights.get((model, bucket), {})
                names = set(previous) | set(weights)
                turnover = 0.5 * sum(
                    abs(weights.get(name, 0.0) - previous.get(name, 0.0))
                    for name in names
                )
                previous_weights[(model, bucket)] = weights
                leg_market_cap = cell.set_index("ric")["company_market_cap"].reindex(
                    weight_series.index
                )
                long_market_cap = leg_market_cap[weight_series.gt(0)]
                short_market_cap = leg_market_cap[weight_series.lt(0)]
            else:
                gross_return = np.nan
                turnover = np.nan
                long_market_cap = pd.Series(dtype=float)
                short_market_cap = pd.Series(dtype=float)
            net_return = (
                gross_return - turnover * cost_bps / 10_000.0
                if np.isfinite(gross_return) and np.isfinite(turnover)
                else np.nan
            )
            records.append(
                {
                    "model": model,
                    "date": date,
                    "size_bucket": bucket,
                    "observations": int(len(cell)),
                    "average_market_cap": float(cell["company_market_cap"].mean()),
                    "median_market_cap": float(cell["company_market_cap"].median()),
                    "spearman_ic": ic,
                    "gross_long_short_return": gross_return,
                    "net_long_short_return": net_return,
                    "long_short_turnover": turnover,
                    "long_n": long_n,
                    "short_n": short_n,
                    "long_average_market_cap": (
                        float(long_market_cap.mean()) if not long_market_cap.empty else np.nan
                    ),
                    "short_average_market_cap": (
                        float(short_market_cap.mean()) if not short_market_cap.empty else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _annualized_sharpe(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    standard_deviation = clean.std(ddof=1)
    if len(clean) < 2 or standard_deviation <= 0:
        return np.nan
    return float(clean.mean() / standard_deviation * np.sqrt(12))


def summarize_size_decomposition(monthly: pd.DataFrame, hac_lags: int) -> pd.DataFrame:
    records = []
    for (model, bucket), group in monthly.groupby(["model", "size_bucket"], sort=True):
        ic_test = hac_mean_diff_test(group["spearman_ic"], maxlags=hac_lags)
        net_test = hac_mean_diff_test(group["net_long_short_return"], maxlags=hac_lags)
        records.append(
            {
                "model": model,
                "size_bucket": bucket,
                "months": int(group["date"].nunique()),
                "average_cross_section": float(group["observations"].mean()),
                "average_market_cap": float(group["average_market_cap"].mean()),
                "median_market_cap": float(group["median_market_cap"].median()),
                "mean_spearman_ic": ic_test["mean"],
                "ic_hac_t_stat": ic_test["t"],
                "ic_p_two_sided": ic_test["p_two_sided"],
                "annualized_net_long_short_return": net_test["mean"] * 12.0,
                "net_return_hac_t_stat": net_test["t"],
                "net_return_p_two_sided": net_test["p_two_sided"],
                "net_long_short_sharpe": _annualized_sharpe(
                    group["net_long_short_return"]
                ),
                "average_long_short_turnover": float(
                    group["long_short_turnover"].mean()
                ),
                "average_long_n": float(group["long_n"].mean()),
                "average_short_n": float(group["short_n"].mean()),
                "average_long_market_cap": float(
                    group["long_average_market_cap"].mean()
                ),
                "average_short_market_cap": float(
                    group["short_average_market_cap"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def write_markdown_exhibit(summary: pd.DataFrame, output_path: Path) -> None:
    display = summary[
        summary["size_bucket"].isin(["small", "middle", "large", "top_500_by_market_cap"])
    ].copy()
    display["annualized_net_return_pct"] = (
        display["annualized_net_long_short_return"] * 100.0
    )
    display["turnover_pct"] = display["average_long_short_turnover"] * 100.0
    display["average_long_market_cap_eur_m"] = (
        display["average_long_market_cap"] / 1_000_000.0
    )
    display = display.sort_values(["model", "size_bucket"])
    display = display[
        [
            "model",
            "size_bucket",
            "mean_spearman_ic",
            "annualized_net_return_pct",
            "net_long_short_sharpe",
            "turnover_pct",
            "average_cross_section",
            "average_long_market_cap_eur_m",
        ]
    ]
    display = display.rename(
        columns={
            "model": "model",
            "size_bucket": "size bucket",
            "mean_spearman_ic": "IC",
            "annualized_net_return_pct": "net return %",
            "net_long_short_sharpe": "Sharpe",
            "turnover_pct": "turnover %",
            "average_cross_section": "avg stocks/month",
            "average_long_market_cap_eur_m": "avg long cap EURm",
        }
    )
    output_path.write_text(
        "# Deep Size Decomposition Exhibit\n\n"
        "Rows are monthly size-bucket averages from saved OOS predictions. "
        "Long-short returns are equal-weight top-minus-bottom decile within "
        "each size bucket and net of the configured transaction cost.\n\n"
        + display.to_markdown(index=False, floatfmt=".4f")
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--portfolio-quantile", type=float, default=0.10)
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--minimum-cross-section", type=int, default=30)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    predictions = load_predictions(args.predictions, args.models)
    monthly = build_monthly_size_decomposition(
        predictions,
        quantile=args.portfolio_quantile,
        cost_bps=args.cost_bps,
        top_n=args.top_n,
        minimum_cross_section=args.minimum_cross_section,
    )
    summary = summarize_size_decomposition(monthly, args.hac_lags)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.output_dir / "deep_size_decomposition_monthly.csv", index=False)
    summary.to_csv(args.output_dir / "deep_size_decomposition_summary.csv", index=False)
    write_markdown_exhibit(
        summary,
        args.output_dir / "deep_size_decomposition_exhibit.md",
    )
    manifest = {
        "predictions": str(args.predictions),
        "models": args.models,
        "portfolio_quantile": args.portfolio_quantile,
        "cost_bps": args.cost_bps,
        "top_n": args.top_n,
        "minimum_cross_section": args.minimum_cross_section,
        "hac_lags": args.hac_lags,
        "rows": {
            "predictions": int(len(predictions)),
            "monthly": int(len(monthly)),
            "summary": int(len(summary)),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"monthly rows: {len(monthly):,}")
    print(f"summary rows: {len(summary):,}")
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
