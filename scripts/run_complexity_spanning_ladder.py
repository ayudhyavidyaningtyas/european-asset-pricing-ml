"""Sequential spanning ladder for the model-complexity premium.

Each rung asks whether one additional increment of model complexity earns return
information that everything below it does not already provide:

    rung 1  momentum      ~ 6F + WML
    rung 2  ridge         ~ 6F + WML + momentum
    rung 3  shallow ML    ~ 6F + WML + momentum + ridge
    rung 4  sequence      ~ 6F + WML + momentum + ridge + best shallow

The alpha at each rung is the marginal return to that increment. Alongside the
usual HAC inference we report the 95% interval and the minimum detectable alpha
at 80% power, so that a failure to reject zero can be read as a bounded estimate
rather than as absence of evidence.
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

from asset_pricing_external_factors import (  # noqa: E402
    EXTERNAL_FACTORS,
    currency_align_portfolios,
    load_external_europe_factors,
    load_monthly_eurusd_return,
)

DEFAULT_BENCHMARK_DIR = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_common_benchmark"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "complexity_spanning_ladder"
)
DEFAULT_FRENCH_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "french"
DEFAULT_FX = PROJECT_ROOT / "data" / "raw" / "fred_DEXUSEU.csv"

MOMENTUM = "momentum_rank"
RIDGE = "ridge_rank"
SHALLOW = ["hist_gbm_rank", "mlp_rank", "dre_rank"]
BEST_SHALLOW = "hist_gbm_rank"
SEQUENCE = [
    "lstm_seq12_rank",
    "lstm_seq24_rank",
    "gru_seq12_rank",
    "gru_seq24_rank",
    "attention_lstm_seq12_rank",
    "attention_lstm_seq24_rank",
    "last_mlp_seq12_rank",
]

# Sequence models were fitted under a 150k training-row cap while the linear,
# tree and DRE models used the full panel. Their rung is reported but flagged:
# a null here is not evidence that the architecture fails.
BUDGET_CONFOUNDED = set(SEQUENCE)

PORTFOLIOS = [
    ("long_short", "net_long_short_usd", False),
    ("long_only_top_decile", "net_long_only_usd", True),
]

# two-sided 5% test, 80% power
MDE_MULTIPLIER = 1.959963985 + 0.841621234


def build_ladder(best_shallow: str) -> list[dict[str, object]]:
    """Return the rung definitions, innermost benchmark first."""
    rungs: list[dict[str, object]] = [
        {
            "rung": 1,
            "increment": "momentum_over_factors",
            "model": MOMENTUM,
            "benchmarks": [],
        },
        {
            "rung": 2,
            "increment": "ridge_over_momentum",
            "model": RIDGE,
            "benchmarks": [MOMENTUM],
        },
    ]
    rungs += [
        {
            "rung": 3,
            "increment": "shallow_ml_over_ridge",
            "model": model,
            "benchmarks": [MOMENTUM, RIDGE],
        }
        for model in SHALLOW
    ]
    rungs += [
        {
            "rung": 4,
            "increment": "sequence_over_shallow",
            "model": model,
            "benchmarks": [MOMENTUM, RIDGE, best_shallow],
        }
        for model in SEQUENCE
    ]
    return rungs


def _spanning_regression(
    frame: pd.DataFrame,
    dependent: str,
    benchmarks: list[str],
    hac_lags: int,
) -> dict[str, float]:
    regressors = [*EXTERNAL_FACTORS, *benchmarks]
    clean = (
        frame[[dependent, *regressors]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(clean) < 24:
        return {"observations": int(len(clean))}
    x = sm.add_constant(clean[regressors], has_constant="add")
    fit = sm.OLS(clean[dependent], x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )
    alpha = float(fit.params["const"])
    alpha_se = float(fit.bse["const"])
    result = {
        "observations": int(fit.nobs),
        "alpha_monthly": alpha,
        "alpha_annualized": alpha * 12.0,
        "alpha_se_annualized": alpha_se * 12.0,
        "alpha_t": float(fit.tvalues["const"]),
        "alpha_p": float(fit.pvalues["const"]),
        "alpha_ci_low_annualized": (alpha - 1.959963985 * alpha_se) * 12.0,
        "alpha_ci_high_annualized": (alpha + 1.959963985 * alpha_se) * 12.0,
        "minimum_detectable_alpha_annualized": MDE_MULTIPLIER * alpha_se * 12.0,
        "adjusted_r2": float(fit.rsquared_adj),
    }
    for benchmark in benchmarks:
        result[f"beta_{benchmark}"] = float(fit.params[benchmark])
        result[f"t_{benchmark}"] = float(fit.tvalues[benchmark])
    for factor in EXTERNAL_FACTORS:
        result[f"beta_{factor}"] = float(fit.params[factor])
    return result


def run_ladder(
    monthly_portfolios: pd.DataFrame,
    external_factors: pd.DataFrame,
    eurusd_return: pd.Series,
    cost_bps: int,
    hac_lags: int,
    best_shallow: str,
) -> pd.DataFrame:
    aligned = currency_align_portfolios(
        monthly_portfolios,
        eurusd_return,
        cost_bps,
    ).merge(
        external_factors,
        on="return_date",
        how="left",
        validate="many_to_one",
    )

    records: list[dict[str, object]] = []
    for (weighting, universe_variant), subset in aligned.groupby(
        ["weighting", "universe_variant"], sort=True
    ):
        for portfolio, dependent, subtract_rf in PORTFOLIOS:
            panel = subset[subset["portfolio"].eq(portfolio)] if (
                "portfolio" in subset.columns
            ) else subset
            wide = panel.pivot_table(
                index="return_date",
                columns="model",
                values=dependent,
                aggfunc="first",
            )
            if wide.empty:
                continue
            factor_block = (
                panel.drop_duplicates("return_date")
                .set_index("return_date")[[*EXTERNAL_FACTORS, "RF"]]
            )
            wide = wide.join(factor_block, how="inner")
            if subtract_rf:
                for column in wide.columns:
                    if column in EXTERNAL_FACTORS or column == "RF":
                        continue
                    wide[column] = wide[column] - wide["RF"]

            for rung in build_ladder(best_shallow):
                model = str(rung["model"])
                benchmarks = [
                    b for b in rung["benchmarks"] if b in wide.columns
                ]
                if model not in wide.columns:
                    continue
                if len(benchmarks) != len(rung["benchmarks"]):
                    continue
                records.append(
                    {
                        "rung": rung["rung"],
                        "increment": rung["increment"],
                        "model": model,
                        "benchmarks": "+".join(benchmarks) or "factors_only",
                        "weighting": weighting,
                        "universe_variant": universe_variant,
                        "portfolio": portfolio,
                        "cost_bps": cost_bps,
                        "budget_confounded": model in BUDGET_CONFOUNDED,
                        **_spanning_regression(
                            wide, model, benchmarks, hac_lags
                        ),
                    }
                )

    result = pd.DataFrame(records)
    if result.empty or "alpha_p" not in result.columns:
        return result
    family = ["weighting", "universe_variant", "portfolio", "cost_bps", "rung"]
    testable = result["alpha_p"].notna()
    result.loc[testable, "alpha_p_holm"] = (
        result[testable]
        .groupby(family)["alpha_p"]
        .transform(lambda values: multipletests(values, method="holm")[1])
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--french-dir", type=Path, default=DEFAULT_FRENCH_DIR)
    parser.add_argument("--fx-file", type=Path, default=DEFAULT_FX)
    parser.add_argument("--cost-bps", nargs="+", type=int, default=[0, 25])
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--best-shallow", type=str, default=BEST_SHALLOW)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_csv(
        args.benchmark_dir / "common_monthly_portfolios.csv",
        parse_dates=["signal_date", "return_date"],
    )
    if "portfolio" not in monthly.columns:
        monthly["portfolio"] = "long_short"

    factors = load_external_europe_factors(
        args.french_dir / "Europe_5_Factors.csv",
        args.french_dir / "Europe_MOM_Factor.csv",
    )
    fx = load_monthly_eurusd_return(args.fx_file)

    frames = [
        run_ladder(
            monthly,
            factors,
            fx,
            cost_bps=cost,
            hac_lags=args.hac_lags,
            best_shallow=args.best_shallow,
        )
        for cost in args.cost_bps
    ]
    ladder = pd.concat(frames, ignore_index=True)
    ladder_path = args.output_dir / "complexity_spanning_ladder.csv"
    ladder.to_csv(ladder_path, index=False)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "benchmark_dir": str(args.benchmark_dir),
        "hac_lags": args.hac_lags,
        "cost_bps": args.cost_bps,
        "best_shallow": args.best_shallow,
        "external_factors": EXTERNAL_FACTORS,
        "budget_confounded_models": sorted(BUDGET_CONFOUNDED),
        "budget_confound_note": (
            "Sequence models were fitted under a 150,000 training-row cap while "
            "momentum, ridge, HistGBM, MLP and DRE used the full panel. Rung-4 "
            "results are not architecture-comparable."
        ),
        "minimum_detectable_alpha": (
            "two-sided 5% test at 80% power; equals 2.8016 x annualized alpha SE"
        ),
        "outputs": {"ladder": str(ladder_path)},
        "rows": int(len(ladder)),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
