"""Sensitivity of net performance to the half-spread assumed for uncovered names.

The Refinitiv liquidity export covers only 811 securities (the AIPM top-500
universe and its history). Any strategy that trades outside that set has its
execution cost imputed rather than measured, so a single net Sharpe is a point
on an assumption curve, not an estimate. This script traces the curve.

For a set of stock-level SDF weight files it recomputes net performance under

  * a grid of constant half-spreads applied to every uncovered name, and
  * a size-conditional imputation that fits log(half_spread) ~ log(market_cap)
    on the covered names each month and extrapolates to the uncovered tail,

and reports the break-even half-spread at which net Sharpe crosses zero, plus
the share of gross weight that is imputed at all. Models whose weight sits
mostly on covered names will show a flat curve; models that live in the
microcap tail will not.
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

from aipm_post_analysis import (  # noqa: E402
    AIPMPostAnalysisConfig,
    build_execution_input_panel,
    simulate_weight_implementability,
    summarize_implementability,
)


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DEFAULT_RISK = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "depth_analysis"
    / "rolling_risk_estimates.parquet"
)
DEFAULT_LIQUIDITY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "supplemental"
    / "liquidity_monthly_full_period"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "cost_assumption_sensitivity"
)

DEFAULT_SPREAD_GRID = (5.0, 10.0, 25.0, 36.0, 50.0, 75.0, 100.0, 150.0, 200.0)


def parse_comma_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def json_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def break_even_spread(curve: pd.DataFrame) -> float | None:
    """Linearly interpolate the half-spread where net Sharpe crosses zero."""

    ordered = curve.sort_values("assumed_half_spread_bps")
    spreads = ordered["assumed_half_spread_bps"].to_numpy(dtype=float)
    sharpes = ordered["net_sharpe"].to_numpy(dtype=float)
    for i in range(len(spreads) - 1):
        left, right = sharpes[i], sharpes[i + 1]
        if np.isnan(left) or np.isnan(right):
            continue
        if left > 0.0 >= right:
            span = left - right
            if span <= 0:
                continue
            return float(spreads[i] + (spreads[i + 1] - spreads[i]) * (left / span))
    return None


def config_for_scenario(
    scenario: dict,
    aum_eur: tuple[float, ...],
    impact_coefficient: float,
) -> AIPMPostAnalysisConfig:
    if scenario["scenario"] == "constant":
        return AIPMPostAnalysisConfig(
            aum_eur=aum_eur,
            impact_coefficient=impact_coefficient,
            fallback_half_spread_bps=scenario["assumed_half_spread_bps"],
            spread_imputation="constant",
        )
    if scenario["scenario"] == "size_conditional":
        return AIPMPostAnalysisConfig(
            aum_eur=aum_eur,
            impact_coefficient=impact_coefficient,
            spread_imputation="size_conditional",
        )
    raise ValueError(f"unknown scenario: {scenario['scenario']}")


def execution_input_diagnostics(
    execution_inputs: pd.DataFrame,
) -> dict[str, float | int | None]:
    observed = execution_inputs["spread_observed"].astype(bool)
    half_spread = pd.to_numeric(execution_inputs["half_spread_bps"], errors="coerce")
    return {
        "execution_input_rows": int(len(execution_inputs)),
        "execution_input_observed_rows": int(observed.sum()),
        "execution_input_uncovered_rows": int((~observed).sum()),
        "execution_input_observed_mean_half_spread_bps": json_float(
            half_spread[observed].mean()
        ),
        "execution_input_uncovered_mean_half_spread_bps": json_float(
            half_spread[~observed].mean()
        ),
    }


def build_scenario_runs(
    panel_path: Path,
    risk_path: Path | None,
    liquidity_path: Path | None,
    scenarios: list[dict],
    aum_eur: tuple[float, ...],
    impact_coefficient: float,
) -> list[dict]:
    """Build execution inputs separately for each spread assumption.

    `attach_execution_inputs` fills uncovered spreads immediately. Reusing a
    single execution panel would therefore pin every constant-spread scenario to
    the first fallback value used to build that panel.
    """

    runs = []
    for scenario in scenarios:
        config = config_for_scenario(scenario, aum_eur, impact_coefficient)
        execution_inputs = build_execution_input_panel(
            panel_path,
            risk_path if risk_path is not None and risk_path.exists() else None,
            liquidity_path
            if liquidity_path is not None and liquidity_path.exists()
            else None,
            config,
        )
        runs.append(
            {
                "scenario": scenario,
                "config": config,
                "execution_inputs": execution_inputs,
                "diagnostics": execution_input_diagnostics(execution_inputs),
            }
        )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        nargs="+",
        required=True,
        help="One or more stock-level SDF weight parquet files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels, one per weights file. Defaults to parent dir names.",
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aum-eur", default="10000000,100000000,500000000")
    parser.add_argument("--spread-grid", default=None)
    parser.add_argument("--impact-coefficient", type=float, default=0.10)
    args = parser.parse_args()

    labels = args.labels or [path.parent.name for path in args.weights]
    if len(labels) != len(args.weights):
        raise SystemExit("--labels must have one entry per --weights file")

    spread_grid = (
        parse_comma_floats(args.spread_grid) if args.spread_grid else DEFAULT_SPREAD_GRID
    )
    aum_eur = parse_comma_floats(args.aum_eur)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios: list[dict] = [
        {"scenario": "constant", "assumed_half_spread_bps": value}
        for value in spread_grid
    ]
    scenarios.append({"scenario": "size_conditional", "assumed_half_spread_bps": np.nan})
    scenario_runs = build_scenario_runs(
        args.panel,
        args.risk,
        args.liquidity,
        scenarios,
        aum_eur,
        args.impact_coefficient,
    )

    records: list[dict] = []
    for label, weight_path in zip(labels, args.weights, strict=True):
        if not weight_path.exists():
            print(f"skipping missing weights: {weight_path}", flush=True)
            continue
        weights = pd.read_parquet(weight_path)
        for run in scenario_runs:
            scenario = run["scenario"]
            config = run["config"]
            execution_inputs = run["execution_inputs"]
            diagnostics = run["diagnostics"]
            monthly = simulate_weight_implementability(weights, execution_inputs, config)
            summary = summarize_implementability(monthly)
            for _, row in summary.iterrows():
                records.append(
                    {
                        "label": label,
                        "model": row["model"],
                        "scenario": scenario["scenario"],
                        "assumed_half_spread_bps": scenario["assumed_half_spread_bps"],
                        "aum_label": row["aum_label"],
                        "aum_eur": float(row["aum_eur"]),
                        "months": int(row["months"]),
                        "annualized_gross_return": float(row["annualized_gross_return"]),
                        "annualized_net_return": float(row["annualized_net_return"]),
                        "net_sharpe": float(row["net_sharpe"]),
                        "annualized_spread_cost": float(row["annualized_spread_cost"]),
                        "annualized_impact_cost": float(row["annualized_impact_cost"]),
                        "average_monthly_turnover": float(row["average_monthly_turnover"]),
                        "spread_observed_weight": float(row["spread_observed_weight"]),
                        "adv_floored_weight": float(row["adv_floored_weight"]),
                        "realized_mean_half_spread_bps": float(row["mean_half_spread_bps"]),
                        **diagnostics,
                    }
                )
            print(
                f"{label} / {scenario['scenario']} "
                f"{scenario['assumed_half_spread_bps']} done",
                flush=True,
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        print("no scenarios produced results", flush=True)
        return 1

    sensitivity_path = args.output_dir / "cost_assumption_sensitivity.csv"
    frame.to_csv(sensitivity_path, index=False)

    break_even_records = []
    constant_only = frame[frame["scenario"] == "constant"]
    for (label, model, aum_label), group in constant_only.groupby(
        ["label", "model", "aum_label"], sort=True
    ):
        break_even_records.append(
            {
                "label": label,
                "model": model,
                "aum_label": aum_label,
                "spread_observed_weight": float(group["spread_observed_weight"].iloc[0]),
                "adv_floored_weight": float(group["adv_floored_weight"].iloc[0]),
                "net_sharpe_at_25bps": float(
                    group.loc[
                        group["assumed_half_spread_bps"].eq(25.0), "net_sharpe"
                    ].mean()
                ),
                "break_even_half_spread_bps": break_even_spread(group),
            }
        )
    break_even = pd.DataFrame(break_even_records)
    break_even_path = args.output_dir / "cost_assumption_break_even.csv"
    break_even.to_csv(break_even_path, index=False)

    manifest = {
        "weights": [str(path) for path in args.weights],
        "labels": labels,
        "spread_grid_bps": list(spread_grid),
        "aum_eur": list(aum_eur),
        "impact_coefficient": args.impact_coefficient,
        "scenarios": ["constant grid", "size_conditional"],
        "execution_input_diagnostics": [
            {
                "scenario": run["scenario"]["scenario"],
                "assumed_half_spread_bps": json_float(
                    run["scenario"]["assumed_half_spread_bps"]
                ),
                **run["diagnostics"],
            }
            for run in scenario_runs
        ],
        "note": (
            "Uncovered names have no Refinitiv quote, so their half-spread is "
            "assumed rather than measured. Net Sharpe should be read as a band "
            "across this grid, and the break-even column states how wide the "
            "assumed spread has to be before the strategy stops paying."
        ),
        "rows": {"sensitivity": int(len(frame)), "break_even": int(len(break_even))},
    }
    (args.output_dir / "cost_assumption_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(f"\nsensitivity -> {sensitivity_path}", flush=True)
    print(f"break-even  -> {break_even_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
