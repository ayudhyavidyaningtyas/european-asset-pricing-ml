"""Selection-aware bootstrap for the validation-selected implementable strategy.

The published interval for the validation-selected strategy conditions on the
realised selection path: the monthly model/rung choices are frozen and only
the return series is resampled. That answers "how uncertain is the mean of
this particular path", not "how uncertain is the strategy including the
selection rule", and an examiner can object that the selection process is part
of the estimator.

This script treats the selector as part of the estimator. Each stationary-
bootstrap draw resamples MONTHS of the full candidate panel jointly (all
model/rung cells move together, preserving the cross-candidate dependence),
and the trailing certainty-equivalent selection rule is re-run through the
resampled pseudo-time from scratch. The distribution of the resulting
post-selection Sharpe ratios therefore carries both return risk and selection
risk. The conditional (frozen-path) bootstrap is reported beside it on the
same draws grid, so the widening attributable to selection is visible.

The chronological reimplementation is validated against the stored
validation_selected_monthly.csv before any bootstrap runs; the script aborts
on mismatch.
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

from stats import stationary_bootstrap_indices  # noqa: E402

RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_INPUT_DIR = (
    RESULTS_ROOT / "estimates_revisions_validation_selected_implementable_strategy"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "estimates_revisions_selection_aware_bootstrap"

VALIDATION_MONTHS = 36
MINIMUM_VALIDATION_MONTHS = 24
RISK_AVERSION = 3.0


def load_candidate_matrices(
    input_dir: Path,
    portfolio: str,
    *,
    weighting: str,
    aum_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Per-cell monthly matrices: net return, turnover, observed-spread share."""
    monthly = pd.read_parquet(input_dir / "candidate_ladder_monthly.parquet")
    manifest = json.loads((input_dir / "manifest.json").read_text())
    eligible = monthly[
        monthly["model"].isin(manifest["candidate_models"])
        & monthly["rung"].isin(manifest["rungs"])
        & monthly["weighting"].eq(weighting)
        & monthly["portfolio"].eq(portfolio)
    ].copy()
    eligible["date"] = pd.to_datetime(eligible["date"])
    eligible["cell"] = eligible["model"] + "|" + eligible["rung"]
    returns = eligible.pivot_table(
        index="date", columns="cell", values=f"net_return_{aum_label}"
    ).sort_index()
    turnover = eligible.pivot_table(
        index="date", columns="cell", values=f"turnover_{aum_label}"
    ).reindex(returns.index)[returns.columns]
    spread = eligible.pivot_table(
        index="date", columns="cell", values="observed_spread_fraction"
    ).reindex(returns.index)[returns.columns]
    return returns, turnover, spread, returns.index


def run_selection(
    returns: np.ndarray,
    turnover: np.ndarray,
    spread: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trailing-CE selection over (pseudo-)time.

    At position p the validation window is the previous VALIDATION_MONTHS rows
    (the calendar rule "target_date in (signal - 36m, signal]" reduces to the
    trailing 36 rows on a consecutive monthly grid). Returns the realised
    return series and the selected column index per position (-1 = none).
    """
    n_months, n_cells = returns.shape
    realised = np.full(n_months, np.nan)
    chosen = np.full(n_months, -1, dtype=int)
    for position in range(n_months):
        start = max(0, position - VALIDATION_MONTHS)
        window = returns[start:position]
        if window.shape[0] == 0:
            continue
        counts = np.sum(~np.isnan(window), axis=0)
        valid = counts >= MINIMUM_VALIDATION_MONTHS
        if not valid.any():
            continue
        mean = np.nanmean(window, axis=0)
        # ddof=1 sample variance to mirror the selector's pandas std.
        variance = np.nanvar(window, axis=0, ddof=1)
        objective = 12.0 * mean - 0.5 * RISK_AVERSION * (12.0 * variance)
        window_turnover = np.nanmean(turnover[start:position], axis=0)
        window_spread = np.nanmean(spread[start:position], axis=0)
        objective = np.where(valid, objective, -np.inf)
        # Lexicographic tie-break identical to the selector: objective desc,
        # window months desc, observed-spread share desc, turnover asc.
        order = np.lexsort(
            (
                window_turnover,
                -window_spread,
                -counts.astype(float),
                -objective,
            )
        )
        best = order[0]
        if not valid[best] or not np.isfinite(returns[position, best]):
            # Fall back to the best valid cell with a finite realised return.
            best = -1
            for candidate in order:
                if valid[candidate] and np.isfinite(returns[position, candidate]):
                    best = candidate
                    break
            if best == -1:
                continue
        realised[position] = returns[position, best]
        chosen[position] = best
    return realised, chosen


def annualized_summary(series: np.ndarray) -> dict[str, float]:
    clean = series[np.isfinite(series)]
    if len(clean) < 12:
        return {"months": int(len(clean))}
    mean = clean.mean() * 12.0
    volatility = clean.std(ddof=1) * np.sqrt(12.0)
    return {
        "months": int(len(clean)),
        "annualized_net_return": float(mean),
        "annualized_net_volatility": float(volatility),
        "net_sharpe": float(mean / volatility) if volatility > 0 else np.nan,
    }


def percentile_interval(draws: np.ndarray, ci: float = 0.95) -> dict[str, float]:
    finite = draws[np.isfinite(draws)]
    if len(finite) == 0:
        return {"ci_low": np.nan, "ci_high": np.nan, "p_two_sided_zero": np.nan}
    alpha = 1.0 - ci
    return {
        "ci_low": float(np.quantile(finite, alpha / 2.0)),
        "ci_high": float(np.quantile(finite, 1.0 - alpha / 2.0)),
        "p_two_sided_zero": float(
            min(1.0, 2.0 * min((finite <= 0.0).mean(), (finite >= 0.0).mean()))
        ),
        "draws_used": int(len(finite)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--portfolios", nargs="+", default=["long_only", "long_short"])
    parser.add_argument("--weighting", default="value")
    parser.add_argument("--aum-label", default="100m")
    parser.add_argument("--expected-block", type=float, default=6.0)
    parser.add_argument("--repetitions", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reproduction-tolerance",
        type=float,
        default=1e-8,
        help="Maximum absolute deviation allowed between the reimplemented "
        "chronological series and the stored validation-selected series.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stored = pd.read_csv(
        args.input_dir / "validation_selected_monthly.csv",
        parse_dates=["target_date"],
    )
    return_column = f"net_return_{args.aum_label}"

    summary_rows = []
    draw_frames = []
    for portfolio in args.portfolios:
        returns, turnover, spread, dates = load_candidate_matrices(
            args.input_dir,
            portfolio,
            weighting=args.weighting,
            aum_label=args.aum_label,
        )
        matrix = returns.to_numpy(dtype=float)
        turnover_matrix = turnover.to_numpy(dtype=float)
        spread_matrix = spread.to_numpy(dtype=float)

        # 1. Chronological reproduction check against the stored series.
        realised, chosen = run_selection(matrix, turnover_matrix, spread_matrix)
        reference = stored[stored["selected_portfolio"].eq(portfolio)].sort_values(
            "target_date"
        )
        reproduced = pd.Series(realised, index=dates).dropna()
        aligned = pd.DataFrame(
            {
                "reimplemented": reproduced.to_numpy(),
                "stored": reference[return_column].to_numpy()[: len(reproduced)],
            }
        )
        if len(reproduced) != len(reference):
            raise SystemExit(
                f"{portfolio}: reimplementation yields {len(reproduced)} months, "
                f"stored series has {len(reference)}."
            )
        deviation = float(aligned.diff(axis=1)["stored"].abs().max())
        if deviation > args.reproduction_tolerance:
            raise SystemExit(
                f"{portfolio}: chronological reproduction deviates by {deviation:.2e} "
                f"(> {args.reproduction_tolerance}); selector reimplementation drifted."
            )
        point = annualized_summary(realised)

        # 2. Selection-aware bootstrap: months resampled jointly, selector re-run.
        rng = np.random.default_rng(args.seed)
        index_matrix = stationary_bootstrap_indices(
            matrix.shape[0], args.expected_block, args.repetitions, rng
        )
        aware_sharpe = np.full(args.repetitions, np.nan)
        aware_mean = np.full(args.repetitions, np.nan)
        switch_share = np.full(args.repetitions, np.nan)
        for draw in range(args.repetitions):
            order = index_matrix[draw]
            draw_realised, draw_chosen = run_selection(
                matrix[order], turnover_matrix[order], spread_matrix[order]
            )
            stats_row = annualized_summary(draw_realised)
            aware_sharpe[draw] = stats_row.get("net_sharpe", np.nan)
            aware_mean[draw] = stats_row.get("annualized_net_return", np.nan)
            picks = draw_chosen[draw_chosen >= 0]
            if len(picks) > 1:
                switch_share[draw] = float((np.diff(picks) != 0).mean())

        # 3. Conditional (frozen-path) bootstrap on the same draw grid.
        frozen = realised[np.isfinite(realised)]
        frozen_indices = stationary_bootstrap_indices(
            len(frozen),
            args.expected_block,
            args.repetitions,
            np.random.default_rng(args.seed),
        )
        frozen_draws = frozen[frozen_indices]
        frozen_mean = frozen_draws.mean(axis=1) * 12.0
        frozen_vol = frozen_draws.std(axis=1, ddof=1) * np.sqrt(12.0)
        frozen_sharpe = np.divide(
            frozen_mean,
            frozen_vol,
            out=np.full(args.repetitions, np.nan),
            where=frozen_vol > 0,
        )

        aware_sharpe_interval = percentile_interval(aware_sharpe)
        aware_mean_interval = percentile_interval(aware_mean)
        frozen_sharpe_interval = percentile_interval(frozen_sharpe)
        frozen_mean_interval = percentile_interval(frozen_mean)
        summary_rows.append(
            {
                "portfolio": portfolio,
                "aum_label": args.aum_label,
                "weighting": args.weighting,
                **{f"point_{k}": v for k, v in point.items()},
                "reproduction_max_abs_deviation": deviation,
                "aware_sharpe_ci_low": aware_sharpe_interval["ci_low"],
                "aware_sharpe_ci_high": aware_sharpe_interval["ci_high"],
                "aware_sharpe_p_two_sided_zero": aware_sharpe_interval[
                    "p_two_sided_zero"
                ],
                "aware_return_ci_low": aware_mean_interval["ci_low"],
                "aware_return_ci_high": aware_mean_interval["ci_high"],
                "aware_return_p_two_sided_zero": aware_mean_interval[
                    "p_two_sided_zero"
                ],
                "conditional_sharpe_ci_low": frozen_sharpe_interval["ci_low"],
                "conditional_sharpe_ci_high": frozen_sharpe_interval["ci_high"],
                "conditional_return_ci_low": frozen_mean_interval["ci_low"],
                "conditional_return_ci_high": frozen_mean_interval["ci_high"],
                "sharpe_ci_width_ratio_aware_over_conditional": (
                    (
                        aware_sharpe_interval["ci_high"]
                        - aware_sharpe_interval["ci_low"]
                    )
                    / (
                        frozen_sharpe_interval["ci_high"]
                        - frozen_sharpe_interval["ci_low"]
                    )
                    if frozen_sharpe_interval["ci_high"]
                    != frozen_sharpe_interval["ci_low"]
                    else np.nan
                ),
                "mean_within_draw_switch_share": float(np.nanmean(switch_share)),
                "expected_block": args.expected_block,
                "repetitions": args.repetitions,
                "seed": args.seed,
            }
        )
        draw_frames.append(
            pd.DataFrame(
                {
                    "portfolio": portfolio,
                    "draw": np.arange(args.repetitions),
                    "aware_net_sharpe": aware_sharpe,
                    "aware_annualized_net_return": aware_mean,
                    "conditional_net_sharpe": frozen_sharpe,
                    "within_draw_switch_share": switch_share,
                }
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "selection_aware_bootstrap_summary.csv", index=False)
    pd.concat(draw_frames, ignore_index=True).to_parquet(
        args.output_dir / "selection_aware_bootstrap_draws.parquet", index=False
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "input_dir": str(args.input_dir),
        "selector": {
            "objective": "certainty_equivalent",
            "risk_aversion": RISK_AVERSION,
            "validation_months": VALIDATION_MONTHS,
            "minimum_validation_months": MINIMUM_VALIDATION_MONTHS,
            "reproduction_check": "chronological series must match the stored "
            "validation_selected_monthly.csv before any bootstrap runs",
        },
        "bootstrap": {
            "kind": "stationary (Politis-Romano), months resampled jointly "
            "across all candidate cells; selector re-run per draw",
            "expected_block": args.expected_block,
            "repetitions": args.repetitions,
            "seed": args.seed,
        },
        "interpretation": (
            "The selection-aware interval treats the trailing-CE selector as "
            "part of the estimator, so it includes selection risk that the "
            "conditional (frozen-path) interval omits. Block joins in "
            "pseudo-time splice non-contiguous history into the validation "
            "window; this is inherent to block bootstraps of path-dependent "
            "estimators and is shared by both intervals' resampling."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.round(4).to_string(index=False))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
