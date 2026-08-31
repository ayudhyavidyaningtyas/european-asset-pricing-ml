"""Subperiod stability of the capacity gradient.

The capacity gradient is the dissertation's headline estimand, and the pooled
137-month estimate says nothing about whether it is a stable feature of the
cross-section or an artefact of one part of the sample. This script rebuilds
the monthly gradient series

    g_t(m, b, d) = [IC_t(m) - IC_t(b)]_least tradable
                 - [IC_t(m) - IC_t(b)]_most tradable

from the stored monthly bucket ICs (no model re-runs), splits the evaluation
window into two halves, and reports

  * the HAC gradient estimate within each half, and
  * an interaction test: OLS of g_t on a constant and a second-half dummy with
    HAC standard errors, so the difference between halves carries its own
    t-statistic and Holm-adjusted p-value.

Sequence models are carried through but flagged budget-confounded, exactly as
in the pooled test, and are excluded from the Holm families.
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

RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_MONTHLY_ICS = RESULTS_ROOT / "capacity_gradient_tests" / "monthly_bucket_ics.csv"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "capacity_gradient_subperiods"

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
DIMENSION_BUCKETS = {
    "market_cap": ("low_cap", "top_500_cap"),
    "trading_value": ("low_adv", "top_500_adv"),
}
MDE_MULTIPLIER = 1.959963985 + 0.841621234


def _hac_mean(series: pd.Series, hac_lags: int, min_months: int) -> dict[str, float]:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_months:
        return {"months": int(len(clean))}
    fit = sm.OLS(clean, np.ones(len(clean))).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    estimate = float(fit.params.iloc[0])
    standard_error = float(fit.bse.iloc[0])
    return {
        "months": int(len(clean)),
        "estimate": estimate,
        "standard_error": standard_error,
        "t_stat": float(fit.tvalues.iloc[0]),
        "p_value": float(fit.pvalues.iloc[0]),
        "ci_low": estimate - 1.959963985 * standard_error,
        "ci_high": estimate + 1.959963985 * standard_error,
        "minimum_detectable_effect": MDE_MULTIPLIER * standard_error,
    }


def _hac_shift(
    series: pd.Series,
    second_half: pd.Series,
    hac_lags: int,
    min_months: int,
) -> dict[str, float]:
    """OLS of g_t on [1, second-half dummy] with HAC errors.

    The dummy coefficient is the change in the gradient between halves; its
    HAC t-statistic is the interaction test.
    """
    frame = pd.DataFrame({"g": series, "d": second_half.astype(float)}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < min_months or frame["d"].nunique() < 2:
        return {"months": int(len(frame))}
    design = sm.add_constant(frame["d"])
    fit = sm.OLS(frame["g"], design).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    estimate = float(fit.params["d"])
    standard_error = float(fit.bse["d"])
    return {
        "months": int(len(frame)),
        "estimate": estimate,
        "standard_error": standard_error,
        "t_stat": float(fit.tvalues["d"]),
        "p_value": float(fit.pvalues["d"]),
        "ci_low": estimate - 1.959963985 * standard_error,
        "ci_high": estimate + 1.959963985 * standard_error,
        "minimum_detectable_effect": MDE_MULTIPLIER * standard_error,
    }


def monthly_gradients(ics: pd.DataFrame, benchmark: str, label: str) -> pd.DataFrame:
    """Monthly gradient series per model and dimension against one benchmark."""
    wide = ics.pivot_table(
        index=["dimension", "bucket", "date"], columns="model", values="ic"
    )
    if benchmark not in wide.columns:
        return pd.DataFrame()
    records = []
    for model in [column for column in wide.columns if column != benchmark]:
        differences = (wide[model] - wide[benchmark]).dropna().reset_index(name="d")
        for dimension, (least, most) in DIMENSION_BUCKETS.items():
            pivot = (
                differences[differences["dimension"].eq(dimension)]
                .pivot_table(index="date", columns="bucket", values="d")
            )
            if least not in pivot.columns or most not in pivot.columns:
                continue
            gradient = (pivot[least] - pivot[most]).dropna().rename("gradient")
            frame = gradient.reset_index()
            frame["premium"] = label
            frame["model"] = model
            frame["benchmark"] = benchmark
            frame["dimension"] = dimension
            records.append(frame)
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly-ics", type=Path, default=DEFAULT_MONTHLY_ICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-date",
        default="2021-01-01",
        help="First month of the second subperiod.",
    )
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument(
        "--min-months",
        type=int,
        default=48,
        help="Minimum months per subperiod for a HAC estimate.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ics = pd.read_csv(args.monthly_ics, parse_dates=["date"])
    # The stored file carries a separate row set per dimension only for the
    # bucketed rows; the "all" bucket is duplicated across dimensions upstream.
    series = pd.concat(
        [
            monthly_gradients(ics, MOMENTUM, "flexibility_premium"),
            monthly_gradients(ics, RIDGE, "depth_premium"),
        ],
        ignore_index=True,
    )
    series = series[series["model"].ne(MOMENTUM) | series["premium"].ne("depth_premium")]
    split = pd.Timestamp(args.split_date)
    series["subperiod"] = np.where(
        series["date"] < split, "first_half", "second_half"
    )

    subperiod_records = []
    shift_records = []
    for (premium, model, benchmark, dimension), group in series.groupby(
        ["premium", "model", "benchmark", "dimension"], observed=True
    ):
        group = group.sort_values("date")
        confounded = model in SEQUENCE_MODELS
        base = {
            "premium": premium,
            "model": model,
            "benchmark": benchmark,
            "dimension": dimension,
            "sequence_budget_confounded": confounded,
        }
        for subperiod, window in group.groupby("subperiod", observed=True):
            subperiod_records.append(
                {
                    **base,
                    "subperiod": subperiod,
                    "window_start": str(window["date"].min().date()),
                    "window_end": str(window["date"].max().date()),
                    **_hac_mean(
                        window.set_index("date")["gradient"],
                        args.hac_lags,
                        args.min_months,
                    ),
                }
            )
        shift_records.append(
            {
                **base,
                "quantity": "second_half_minus_first_half",
                **_hac_shift(
                    group.set_index("date")["gradient"],
                    group.set_index("date")["subperiod"].eq("second_half"),
                    args.hac_lags,
                    2 * args.min_months,
                ),
            }
        )

    subperiods = pd.DataFrame(subperiod_records)
    shifts = pd.DataFrame(shift_records)
    for frame in (subperiods, shifts):
        frame["p_value_holm"] = np.nan
        testable = frame["p_value"].notna() & ~frame["sequence_budget_confounded"]
        if testable.any():
            group_keys = ["premium", "dimension"] + (
                ["subperiod"] if "subperiod" in frame else []
            )
            frame.loc[testable, "p_value_holm"] = (
                frame[testable]
                .groupby(group_keys, observed=True)["p_value"]
                .transform(lambda values: multipletests(values, method="holm")[1])
            )

    series.to_csv(args.output_dir / "monthly_gradient_series.csv", index=False)
    subperiods.to_csv(
        args.output_dir / "capacity_gradient_subperiods.csv", index=False
    )
    shifts.to_csv(args.output_dir / "capacity_gradient_shift_tests.csv", index=False)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "monthly_ics": str(args.monthly_ics),
        "split_date": args.split_date,
        "hac_lags": args.hac_lags,
        "estimand": (
            "gradient g_t = paired premium (least tradable) minus (most "
            "tradable), per model, benchmark and dimension; subperiod HAC "
            "means plus an OLS-on-dummy HAC interaction test for the "
            "between-half change"
        ),
        "holm_families": {
            "subperiods": "premium x dimension x subperiod",
            "shifts": "premium x dimension",
        },
        "sequence_note": (
            "sequence models flagged budget-confounded and excluded from "
            "Holm families, matching the pooled test"
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    core = ~subperiods["sequence_budget_confounded"]
    print(
        subperiods[core][
            [
                "premium",
                "model",
                "dimension",
                "subperiod",
                "months",
                "estimate",
                "t_stat",
                "p_value_holm",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        shifts[~shifts["sequence_budget_confounded"]][
            [
                "premium",
                "model",
                "dimension",
                "months",
                "estimate",
                "t_stat",
                "p_value",
                "p_value_holm",
            ]
        ].to_string(index=False)
    )
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
