"""Two-panel identification figure for the analyst-estimates section.

Panel (a): the data-depth effect by signal lag on the stock-months common to
every lag, with 95% HAC confidence intervals -- the decay-and-boundary
evidence. Panel (b): the effect by coverage-propensity stratum -- the
thin-coverage concentration evidence. Ridge and HistGBM only: they are the
models the dissertation claims are robust beneficiaries, and the unresolved
models would add clutter without argument.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_LADDER = RESULTS_ROOT / "estimates_lag_ladder_20260816" / "lag_ladder_data_depth.csv"
DEFAULT_STRATA = (
    RESULTS_ROOT
    / "estimates_coverage_selection_20260816"
    / "coverage_propensity_stratum_data_depth.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "manuscript"

MODELS = {"ridge": "Ridge", "hist_gbm": "HistGBM"}
COLORS = {"ridge": "#0072B2", "hist_gbm": "#D55E00"}
OFFSETS = {"ridge": -0.08, "hist_gbm": 0.08}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--strata", type=Path, default=DEFAULT_STRATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="identification_lag_stratum")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ladder = pd.read_csv(args.ladder)
    ladder = ladder[
        ladder["sample_scope"].eq("common_across_lags")
        & ladder["model"].isin(MODELS)
    ]
    strata = pd.read_csv(args.strata)
    strata = strata[strata["model"].isin(MODELS)]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )
    figure, (axis_lag, axis_stratum) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), constrained_layout=True
    )

    for model, label in MODELS.items():
        subset = ladder[ladder["model"].eq(model)].sort_values("lag_months")
        x = subset["lag_months"] + OFFSETS[model]
        axis_lag.errorbar(
            x,
            subset["estimate"],
            yerr=[
                subset["estimate"] - subset["ci_low"],
                subset["ci_high"] - subset["estimate"],
            ],
            fmt="o-",
            capsize=2.5,
            markersize=4,
            linewidth=1.2,
            color=COLORS[model],
            label=label,
        )
    axis_lag.axhline(0.0, color="black", linewidth=0.8)
    axis_lag.set_xticks(sorted(ladder["lag_months"].unique()))
    axis_lag.set_xlabel("Analyst signal lag (months)")
    axis_lag.set_ylabel("Data-depth effect (monthly IC)")
    axis_lag.set_title("(a) Decay with signal staleness", fontsize=9, loc="left")
    axis_lag.legend(frameon=False, loc="upper right")

    for model, label in MODELS.items():
        subset = strata[strata["model"].eq(model)].sort_values("propensity_stratum")
        x = subset["propensity_stratum"] + OFFSETS[model]
        axis_stratum.errorbar(
            x,
            subset["estimate"],
            yerr=[
                subset["estimate"] - subset["ci_low"],
                subset["ci_high"] - subset["estimate"],
            ],
            fmt="o",
            capsize=2.5,
            markersize=4,
            color=COLORS[model],
            label=label,
        )
    axis_stratum.axhline(0.0, color="black", linewidth=0.8)
    strata_means = (
        strata.groupby("propensity_stratum")["mean_propensity"].mean().sort_index()
    )
    axis_stratum.set_xticks(strata_means.index)
    axis_stratum.set_xticklabels([f"{value:.2f}" for value in strata_means])
    axis_stratum.set_xlabel("Mean coverage propensity of stratum")
    axis_stratum.set_title(
        "(b) Concentration in thin-coverage strata", fontsize=9, loc="left"
    )

    for extension in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"{args.basename}.{extension}",
            dpi=300,
        )
    plt.close(figure)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "inputs": {"ladder": str(args.ladder), "strata": str(args.strata)},
        "panels": {
            "a": "data-depth effect by signal lag, common stock-months, 95% HAC CIs",
            "b": "data-depth effect by coverage-propensity stratum, 95% HAC CIs",
        },
        "models": list(MODELS),
    }
    (args.output_dir / f"{args.basename}_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"outputs -> {args.output_dir}/{args.basename}.{{png,pdf}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
