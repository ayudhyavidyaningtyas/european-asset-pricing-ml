"""Capacity-gradient tests: does the value of model complexity fall as the
universe becomes more tradable?

The estimand is a difference-in-differences on paired monthly information
coefficients. For model m, benchmark b and tradability bucket k,

    d_t(m, b, k) = IC_t(m, k) - IC_t(b, k)

and the capacity gradient is

    g_t(m, b) = d_t(m, b, least tradable) - d_t(m, b, most tradable)

Pairing within month and bucket removes the mechanical advantage that any
signal enjoys where cross-sectional return dispersion is higher, so a non-zero
gradient cannot be explained by small stocks simply being noisier.

Two independent tradability dimensions are used: market capitalisation and
EUR trading value (an ADV proxy). Capacity constraints bind on trading value
rather than on size as such, so the trading-value dimension is the one that
speaks directly to the capacity barrier; agreement across both guards against
size acting as a proxy for something else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_common_benchmark"
    / "common_predictions.parquet"
)
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "capacity_gradient_tests"
)

MOMENTUM = "momentum_rank"
RIDGE = "ridge_rank"
SEQUENCE_MODELS = {
    "lstm_seq12_rank",
    "lstm_seq24_rank",
    "gru_seq12_rank",
    "gru_seq24_rank",
    "attention_lstm_seq12_rank",
    "attention_lstm_seq24_rank",
    "last_mlp_seq12_rank",
}

# (dimension, sort column, least tradable bucket, most tradable bucket)
DIMENSIONS = [
    ("market_cap", "company_market_cap", "low_cap", "top_500_cap"),
    ("trading_value", "log_trading_value_eur", "low_adv", "top_500_adv"),
]

MDE_MULTIPLIER = 1.959963985 + 0.841621234


def assign_buckets(
    frame: pd.DataFrame, sort_column: str, dimension: str, top_n: int
) -> pd.DataFrame:
    """Label each stock-month by within-month tradability bucket."""
    prefix = "cap" if dimension == "market_cap" else "adv"
    values = pd.to_numeric(frame[sort_column], errors="coerce")
    frame = frame.assign(_sort_value=values).dropna(subset=["_sort_value"])

    grouped = frame.groupby("date", observed=True)["_sort_value"]
    tercile = grouped.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False, duplicates="drop")
        if s.notna().sum() >= 30
        else pd.Series(np.nan, index=s.index)
    )
    labels = {0: f"low_{prefix}", 1: f"mid_{prefix}", 2: f"high_{prefix}"}
    frame = frame.assign(bucket=tercile.map(labels))

    rank_desc = grouped.rank(method="first", ascending=False)
    top = frame.assign(bucket=f"top_{top_n}_{prefix}")[rank_desc.le(top_n)]
    everything = frame.assign(bucket="all")
    return pd.concat(
        [frame.dropna(subset=["bucket"]), top, everything], ignore_index=True
    )


def monthly_information_coefficients(frame: pd.DataFrame, min_names: int) -> pd.DataFrame:
    """Spearman IC per model, month and bucket, computed as Pearson on ranks."""
    keys = ["model", "dimension", "bucket", "date"]
    grouped = frame.groupby(keys, observed=True)
    working = frame.assign(
        x=grouped["prediction"].rank(),
        y=grouped["target_return_1m"].rank(),
    )
    working["xy"] = working["x"] * working["y"]
    aggregated = working.groupby(keys, observed=True).agg(
        names=("x", "size"),
        mean_x=("x", "mean"),
        mean_y=("y", "mean"),
        mean_xy=("xy", "mean"),
        var_x=("x", "var"),
        var_y=("y", "var"),
    )
    aggregated = aggregated[aggregated["names"] >= min_names].copy()
    scale = (aggregated["names"] - 1.0) / aggregated["names"]
    covariance = aggregated["mean_xy"] - aggregated["mean_x"] * aggregated["mean_y"]
    denominator = np.sqrt(
        aggregated["var_x"] * scale * aggregated["var_y"] * scale
    )
    aggregated["ic"] = covariance / denominator.replace(0.0, np.nan)
    return aggregated.reset_index()[[*keys, "names", "ic"]]


def _hac_mean(series: pd.Series, hac_lags: int) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 24:
        return {"months": int(len(clean))}
    fit = sm.OLS(clean, np.ones(len(clean))).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    estimate = float(fit.params[0])
    standard_error = float(fit.bse[0])
    return {
        "months": int(len(clean)),
        "estimate": estimate,
        "standard_error": standard_error,
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "ci_low": estimate - 1.959963985 * standard_error,
        "ci_high": estimate + 1.959963985 * standard_error,
        "minimum_detectable_effect": MDE_MULTIPLIER * standard_error,
    }


def paired_premia(
    ics: pd.DataFrame, benchmark: str, label: str, hac_lags: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bucket-level paired premia and the capacity gradient across buckets."""
    wide = ics.pivot_table(
        index=["dimension", "bucket", "date"], columns="model", values="ic"
    )
    if benchmark not in wide.columns:
        return pd.DataFrame(), pd.DataFrame()

    bucket_records: list[dict[str, object]] = []
    gradient_records: list[dict[str, object]] = []
    models = [c for c in wide.columns if c != benchmark]

    for model in models:
        differences = (wide[model] - wide[benchmark]).dropna()
        for (dimension, bucket), series in differences.groupby(
            level=["dimension", "bucket"], observed=True
        ):
            bucket_records.append(
                {
                    "premium": label,
                    "model": model,
                    "benchmark": benchmark,
                    "dimension": dimension,
                    "bucket": bucket,
                    "sequence_budget_confounded": model in SEQUENCE_MODELS,
                    **_hac_mean(series.droplevel(["dimension", "bucket"]), hac_lags),
                }
            )

        for dimension, least, most in [
            (d, lo, hi) for d, _, lo, hi in DIMENSIONS
        ]:
            frame = differences.reset_index()
            frame = frame[frame["dimension"].eq(dimension)]
            pivot = frame.pivot_table(index="date", columns="bucket", values=0)
            if least not in pivot.columns or most not in pivot.columns:
                continue
            gradient = (pivot[least] - pivot[most]).dropna()
            gradient_records.append(
                {
                    "premium": label,
                    "model": model,
                    "benchmark": benchmark,
                    "dimension": dimension,
                    "least_tradable": least,
                    "most_tradable": most,
                    "sequence_budget_confounded": model in SEQUENCE_MODELS,
                    **_hac_mean(gradient, hac_lags),
                }
            )

    return pd.DataFrame(bucket_records), pd.DataFrame(gradient_records)


def _holm(frame: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    if frame.empty or "p_value" not in frame.columns:
        return frame
    testable = frame["p_value"].notna() & ~frame["sequence_budget_confounded"]
    frame.loc[testable, "p_value_holm"] = (
        frame[testable]
        .groupby(family)["p_value"]
        .transform(lambda values: multipletests(values, method="holm")[1])
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument(
        "--min-market-cap-percentile",
        type=float,
        default=0.05,
        help="Match the ex-bottom-5% screen used elsewhere in the project.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(
        args.predictions,
        columns=[
            "date",
            "ric",
            "target_return_1m",
            "company_market_cap",
            "market_cap_percentile",
            "prediction",
            "model",
        ],
    )
    predictions = predictions[
        predictions["market_cap_percentile"].gt(args.min_market_cap_percentile)
    ]
    predictions = predictions.dropna(subset=["prediction", "target_return_1m"])

    panel = pd.read_parquet(
        args.panel, columns=["ric", "date", "log_trading_value_eur"]
    )
    predictions = predictions.merge(
        panel, on=["ric", "date"], how="left", validate="many_to_one"
    )

    bucketed = []
    for dimension, sort_column, _, _ in DIMENSIONS:
        labelled = assign_buckets(predictions, sort_column, dimension, args.top_n)
        labelled["dimension"] = dimension
        bucketed.append(
            labelled[
                [
                    "model",
                    "dimension",
                    "bucket",
                    "date",
                    "prediction",
                    "target_return_1m",
                ]
            ]
        )
    stacked = pd.concat(bucketed, ignore_index=True)

    ics = monthly_information_coefficients(stacked, args.min_names)
    ics.to_csv(args.output_dir / "monthly_bucket_ics.csv", index=False)

    bucket_frames = []
    gradient_frames = []
    for benchmark, label in [
        (MOMENTUM, "flexibility_premium"),
        (RIDGE, "depth_premium"),
    ]:
        buckets, gradients = paired_premia(ics, benchmark, label, args.hac_lags)
        bucket_frames.append(buckets)
        gradient_frames.append(gradients)

    bucket_table = _holm(
        pd.concat(bucket_frames, ignore_index=True),
        ["premium", "dimension", "bucket"],
    )
    gradient_table = _holm(
        pd.concat(gradient_frames, ignore_index=True),
        ["premium", "dimension"],
    )

    bucket_path = args.output_dir / "paired_premium_by_bucket.csv"
    gradient_path = args.output_dir / "capacity_gradient_tests.csv"
    bucket_table.to_csv(bucket_path, index=False)
    gradient_table.to_csv(gradient_path, index=False)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "predictions": str(args.predictions),
        "panel": str(args.panel),
        "estimand": (
            "g_t = [IC_model - IC_benchmark]_least_tradable "
            "- [IC_model - IC_benchmark]_most_tradable"
        ),
        "dimensions": [
            {"dimension": d, "least_tradable": lo, "most_tradable": hi}
            for d, _, lo, hi in DIMENSIONS
        ],
        "benchmarks": {
            "flexibility_premium": MOMENTUM,
            "depth_premium": RIDGE,
        },
        "hac_lags": args.hac_lags,
        "min_names_per_bucket_month": args.min_names,
        "min_market_cap_percentile": args.min_market_cap_percentile,
        "sequence_budget_confound_note": (
            "Sequence models trained under a 150,000-row cap; excluded from Holm "
            "families and flagged in output."
        ),
        "outputs": {
            "monthly_bucket_ics": str(args.output_dir / "monthly_bucket_ics.csv"),
            "paired_premium_by_bucket": str(bucket_path),
            "capacity_gradient_tests": str(gradient_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
