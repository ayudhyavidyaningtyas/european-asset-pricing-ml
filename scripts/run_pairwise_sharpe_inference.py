"""Pairwise Sharpe-difference inference for saved ML portfolio runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stats as project_stats  # noqa: E402


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "pairwise_sharpe_inference"
)


def monthly_net_returns(
    run_dir: Path,
    model: str,
    *,
    weighting: str,
    universe_variant: str,
    portfolio: str,
    cost_bps: int,
) -> pd.Series:
    monthly_path = run_dir / "monthly_portfolios.csv"
    if not monthly_path.exists():
        raise FileNotFoundError(f"Missing monthly portfolio output: {monthly_path}")
    monthly = pd.read_csv(monthly_path, parse_dates=["return_date"], low_memory=False)
    subset = monthly[
        monthly["model"].eq(model)
        & monthly["weighting"].eq(weighting)
        & monthly["universe_variant"].eq(universe_variant)
    ].copy()
    if subset.empty:
        raise ValueError(
            f"No rows for model={model}, weighting={weighting}, "
            f"universe_variant={universe_variant} in {run_dir}"
        )
    if portfolio == "long_short":
        return_column = "gross_long_short_return"
        turnover_column = "long_short_turnover"
    elif portfolio == "long_only_top_decile":
        return_column = "long_return"
        turnover_column = "long_only_turnover"
    else:
        raise ValueError(f"Unsupported portfolio: {portfolio}")
    missing = {return_column, turnover_column, "return_date"} - set(subset)
    if missing:
        raise ValueError(f"Monthly portfolio output missing columns: {sorted(missing)}")
    net = pd.to_numeric(subset[return_column], errors="coerce").sub(
        pd.to_numeric(subset[turnover_column], errors="coerce") * cost_bps / 10_000.0
    )
    return pd.Series(
        net.to_numpy(dtype=float),
        index=pd.DatetimeIndex(subset["return_date"]),
        name=model,
    ).sort_index()


def infer_pair(
    label: str,
    run_a: Path,
    model_a: str,
    run_b: Path,
    model_b: str,
    *,
    weighting: str,
    universe_variant: str,
    portfolio: str,
    cost_bps: int,
    expected_block: float,
    n_boot: int,
    seed: int,
    min_months: int,
) -> dict[str, object]:
    a = monthly_net_returns(
        run_a,
        model_a,
        weighting=weighting,
        universe_variant=universe_variant,
        portfolio=portfolio,
        cost_bps=cost_bps,
    )
    b = monthly_net_returns(
        run_b,
        model_b,
        weighting=weighting,
        universe_variant=universe_variant,
        portfolio=portfolio,
        cost_bps=cost_bps,
    )
    aligned = pd.concat({"a": a, "b": b}, axis=1, join="inner").dropna()
    if len(aligned) < min_months:
        raise ValueError(
            f"{label} has only {len(aligned)} common months; "
            f"minimum is {min_months}"
        )
    values_a = aligned["a"].to_numpy(dtype=float)
    values_b = aligned["b"].to_numpy(dtype=float)
    risk_free = np.zeros(len(aligned), dtype=float)
    bootstrap = project_stats.bootstrap_sharpe_diff(
        values_a,
        values_b,
        risk_free,
        expected_block=expected_block,
        n_boot=n_boot,
        seed=seed,
    )
    memmel = project_stats.jobson_korkie_memmel(values_a, values_b, risk_free)
    ledoit_wolf = project_stats.ledoit_wolf_sharpe_test(values_a, values_b, risk_free)
    return {
        "comparison": label,
        "run_a": str(run_a),
        "model_a": model_a,
        "run_b": str(run_b),
        "model_b": model_b,
        "weighting": weighting,
        "universe_variant": universe_variant,
        "portfolio": portfolio,
        "cost_bps": cost_bps,
        "months": int(len(aligned)),
        "first_month": str(aligned.index.min().date()),
        "last_month": str(aligned.index.max().date()),
        "annualized_return_a": float(values_a.mean() * 12.0),
        "annualized_return_b": float(values_b.mean() * 12.0),
        "delta_annualized_return_a_minus_b": float(
            (values_a.mean() - values_b.mean()) * 12.0
        ),
        "sharpe_a": project_stats.sharpe_ratio(values_a, risk_free),
        "sharpe_b": project_stats.sharpe_ratio(values_b, risk_free),
        "delta_sharpe_a_minus_b": bootstrap["delta_sharpe"],
        "bootstrap_ci_low": bootstrap["ci_low"],
        "bootstrap_ci_high": bootstrap["ci_high"],
        "bootstrap_p_two_sided": bootstrap["p_two_sided"],
        "bootstrap_ci_includes_zero": bool(
            bootstrap["ci_low"] <= 0.0 <= bootstrap["ci_high"]
        ),
        "bootstrap_expected_block": bootstrap["expected_block"],
        "bootstrap_n": bootstrap["n_boot"],
        "jkm_delta_sharpe_monthly": memmel["delta_sharpe_monthly"],
        "jkm_delta_sharpe_annualized": float(
            memmel["delta_sharpe_monthly"] * np.sqrt(12.0)
        ),
        "jkm_z": memmel["z"],
        "jkm_p_two_sided": memmel["p_two_sided"],
        "ledoit_wolf_delta_sharpe_annualized": ledoit_wolf["delta_sharpe_annualized"],
        "ledoit_wolf_z": ledoit_wolf["z"],
        "ledoit_wolf_p_two_sided": ledoit_wolf["p_two_sided"],
        "ledoit_wolf_hac_lags": ledoit_wolf["maxlags"],
    }


def parse_comparison(values: list[str]) -> tuple[str, Path, str, Path, str]:
    label, run_a, model_a, run_b, model_b = values
    return label, Path(run_a), model_a, Path(run_b), model_b


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        nargs=5,
        action="append",
        metavar=("LABEL", "RUN_A", "MODEL_A", "RUN_B", "MODEL_B"),
        required=True,
        help="Repeatable pair: label run_a model_a run_b model_b. Delta is A minus B.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-filename", default="pairwise_sharpe_tests.csv")
    parser.add_argument("--weighting", default="value")
    parser.add_argument("--universe-variant", default="standard_ex_bottom_5pct")
    parser.add_argument("--portfolio", default="long_short")
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--expected-block", type=float, default=6.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-months", type=int, default=24)
    args = parser.parse_args()

    records = []
    for index, raw in enumerate(args.comparison):
        label, run_a, model_a, run_b, model_b = parse_comparison(raw)
        records.append(
            infer_pair(
                label,
                run_a,
                model_a,
                run_b,
                model_b,
                weighting=args.weighting,
                universe_variant=args.universe_variant,
                portfolio=args.portfolio,
                cost_bps=args.cost_bps,
                expected_block=args.expected_block,
                n_boot=args.bootstrap_repetitions,
                seed=args.seed + index,
                min_months=args.min_months,
            )
        )
    result = pd.DataFrame(records)
    result["bootstrap_p_two_sided_holm"] = np.nan
    valid = result["bootstrap_p_two_sided"].notna()
    if valid.any():
        result.loc[valid, "bootstrap_p_two_sided_holm"] = multipletests(
            result.loc[valid, "bootstrap_p_two_sided"],
            method="holm",
        )[1]
    result["ledoit_wolf_p_two_sided_holm"] = np.nan
    ledoit_wolf_valid = result["ledoit_wolf_p_two_sided"].notna()
    if ledoit_wolf_valid.any():
        result.loc[ledoit_wolf_valid, "ledoit_wolf_p_two_sided_holm"] = multipletests(
            result.loc[ledoit_wolf_valid, "ledoit_wolf_p_two_sided"],
            method="holm",
        )[1]
    result["bootstrap_raw_significant_5pct"] = result[
        "bootstrap_p_two_sided"
    ].le(0.05)
    result["bootstrap_holm_significant_5pct"] = result[
        "bootstrap_p_two_sided_holm"
    ].le(0.05)
    result["bootstrap_ci_excludes_zero"] = ~result[
        "bootstrap_ci_includes_zero"
    ].astype(bool)
    result["bootstrap_ci_excludes_zero_but_holm_not_significant"] = (
        result["bootstrap_ci_excludes_zero"]
        & ~result["bootstrap_holm_significant_5pct"]
    )
    result["sharpe_inference_interpretation"] = np.select(
        [
            result["bootstrap_holm_significant_5pct"],
            result["bootstrap_ci_excludes_zero_but_holm_not_significant"],
        ],
        [
            "sharpe_difference_survives_holm",
            "raw_ci_excludes_zero_but_holm_not_significant",
        ],
        default="not_statistically_resolved",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / args.output_filename
    result.to_csv(output, index=False)
    print(f"comparisons: {len(result)}")
    print(f"outputs -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
