"""Regime-robustness diagnostics from frozen asset-pricing outputs.

The script intentionally avoids refitting models. It reuses the saved
capacity-gradient monthly ICs and the common deep-sequence prediction file to
produce two small tables:

1. split-half capacity-gradient tests, using the same paired IC estimand as the
   main capacity-gradient script; and
2. annual IC trend diagnostics, used to scope sequence-model conclusions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPACITY_ICS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "capacity_gradient_tests"
    / "monthly_bucket_ics.csv"
)
DEFAULT_COMMON_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_common_benchmark"
    / "common_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "regime_robustness_diagnostics"
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
DIMENSIONS = [
    ("market_cap", "low_cap", "top_500_cap"),
    ("trading_value", "low_adv", "top_500_adv"),
]


def _hac_mean(series: pd.Series, hac_lags: int) -> dict[str, float | int]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 24:
        return {
            "months": int(len(clean)),
            "estimate": np.nan,
            "standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    fit = sm.OLS(clean.to_numpy(dtype=float), np.ones(len(clean))).fit(
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
    }


def _period_label(months: list[pd.Timestamp]) -> str:
    start = months[0].strftime("%Y-%m")
    end = months[-1].strftime("%Y-%m")
    return f"{start}_to_{end}"


def split_half_capacity_gradients(
    ics: pd.DataFrame, hac_lags: int
) -> pd.DataFrame:
    ics = ics.copy()
    ics["date"] = pd.to_datetime(ics["date"])
    months = list(pd.Series(ics["date"].dropna().unique()).sort_values())
    midpoint = len(months) // 2
    periods = [
        (_period_label(months[:midpoint]), months[:midpoint]),
        (_period_label(months[midpoint:]), months[midpoint:]),
    ]

    wide = ics.pivot_table(
        index=["dimension", "bucket", "date"], columns="model", values="ic"
    )
    records: list[dict[str, object]] = []
    for period, period_months in periods:
        period_wide = wide[wide.index.get_level_values("date").isin(period_months)]
        tests = [
            ("flexibility_premium", MOMENTUM),
            ("depth_premium", RIDGE),
        ]
        for premium, benchmark in tests:
            if benchmark not in period_wide.columns:
                continue
            for model in sorted(c for c in period_wide.columns if c != benchmark):
                differences = (period_wide[model] - period_wide[benchmark]).dropna()
                frame = differences.reset_index(name="paired_ic_difference")
                for dimension, least, most in DIMENSIONS:
                    subset = frame[frame["dimension"].eq(dimension)]
                    pivot = subset.pivot_table(
                        index="date", columns="bucket", values="paired_ic_difference"
                    )
                    if least not in pivot.columns or most not in pivot.columns:
                        continue
                    gradient = pivot[least] - pivot[most]
                    records.append(
                        {
                            "period": period,
                            "premium": premium,
                            "model": model,
                            "benchmark": benchmark,
                            "dimension": dimension,
                            "least_tradable": least,
                            "most_tradable": most,
                            "sequence_budget_confounded": model in SEQUENCE_MODELS,
                            **_hac_mean(gradient, hac_lags),
                        }
                    )

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["p_value_holm"] = np.nan
    testable = result["p_value"].notna() & ~result["sequence_budget_confounded"]
    families = result[testable].groupby(["period", "premium", "dimension"]).groups
    for _, index in families.items():
        result.loc[list(index), "p_value_holm"] = multipletests(
            result.loc[list(index), "p_value"], method="holm"
        )[1]
    return result


def _monthly_spearman(frame: pd.DataFrame) -> float:
    prediction = frame["prediction"].rank(method="average")
    target = frame["target_return_rank"].rank(method="average")
    return float(prediction.corr(target))


def annual_ic_trends(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.dropna(
        subset=["date", "model", "prediction", "target_return_rank"]
    ).copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    monthly = (
        predictions.groupby(["model", "date"], observed=True)
        .apply(_monthly_spearman, include_groups=False)
        .rename("ic")
        .reset_index()
    )
    monthly["year"] = monthly["date"].dt.year
    annual = (
        monthly.groupby(["model", "year"], observed=True)
        .agg(mean_ic=("ic", "mean"), months=("ic", "size"))
        .reset_index()
    )

    records: list[dict[str, object]] = []
    for model, group in annual.groupby("model", observed=True):
        group = group[group["months"].ge(5)].sort_values("year")
        if len(group) < 5:
            continue
        fold_index = np.arange(len(group), dtype=float)
        fit = sm.OLS(
            group["mean_ic"].to_numpy(dtype=float),
            sm.add_constant(fold_index),
        ).fit()
        records.append(
            {
                "model": model,
                "years": int(len(group)),
                "first_year": int(group["year"].iloc[0]),
                "last_year": int(group["year"].iloc[-1]),
                "sequence_model": model in SEQUENCE_MODELS,
                "mean_annual_ic": float(group["mean_ic"].mean()),
                "first_three_year_mean_ic": float(group["mean_ic"].head(3).mean()),
                "last_three_year_mean_ic": float(group["mean_ic"].tail(3).mean()),
                "slope_per_year": float(fit.params[1]),
                "slope_t_stat": float(fit.tvalues[1]),
                "slope_p_value": float(fit.pvalues[1]),
            }
        )
    return pd.DataFrame(records).sort_values(["sequence_model", "model"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-ics", type=Path, default=DEFAULT_CAPACITY_ICS)
    parser.add_argument("--common-predictions", type=Path, default=DEFAULT_COMMON_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    missing = [path for path in [args.capacity_ics, args.common_predictions] if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required input(s): {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ics = pd.read_csv(args.capacity_ics)
    split = split_half_capacity_gradients(ics, args.hac_lags)
    split_path = args.output_dir / "split_half_capacity_gradient_tests.csv"
    split.to_csv(split_path, index=False)

    predictions = pd.read_parquet(
        args.common_predictions,
        columns=["date", "model", "prediction", "target_return_rank"],
    )
    trends = annual_ic_trends(predictions)
    trends_path = args.output_dir / "annual_ic_trend_diagnostics.csv"
    trends.to_csv(trends_path, index=False)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "capacity_ics": str(args.capacity_ics),
            "common_predictions": str(args.common_predictions),
        },
        "outputs": {
            "split_half_capacity_gradient_tests": str(split_path),
            "annual_ic_trend_diagnostics": str(trends_path),
        },
        "rows": {
            "split_half_capacity_gradient_tests": int(len(split)),
            "annual_ic_trend_diagnostics": int(len(trends)),
        },
        "hac_lags": args.hac_lags,
        "notes": [
            "Split-half tests reuse the main paired IC capacity-gradient estimand.",
            "Sequence models are flagged because the main script treats their training budget as confounded.",
            "Annual IC trends are descriptive diagnostics, not a replacement for the main paired HAC tests.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
