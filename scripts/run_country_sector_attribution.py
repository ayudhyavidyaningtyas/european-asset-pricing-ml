"""Country/sector Brinson attribution for the constrained long-only strategies.

Addresses a dissertation robustness check: the constrained estimates long-only
strategy shows a positive active return against the
internal EUR value-weighted market, but nothing so far establishes whether that
comes from *country/sector allocation* or from *within-group stock selection*.
A broad regional or sector tilt is cheap to replicate and would materially
weaken the claim that the ML signal picks stocks.

The script re-runs the constrained simulation with the group-exposure side
channel enabled, rebuilds the benchmark at country and sector granularity from
the same panel that produces the internal market, and decomposes gross active
return into allocation, selection and interaction with HAC tests on each.

Attribution is on GROSS returns (the benchmark is gross). The cost drag is
reported separately so gross active - costs reconciles to the net active return
in ``benchmark_relative_summary.csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [SRC_DIR, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import group_attribution as ga  # noqa: E402
from asset_pricing_depth import build_internal_market  # noqa: E402
from investability_ladder import LadderConfig, load_ladder_panel  # noqa: E402
from run_constrained_deep_hybrid_long_only import (  # noqa: E402
    ConstraintSpec,
    aum_label,
    parse_constraint_specs,
    simulate_constrained,
)
from run_constrained_estimates_long_only import (  # noqa: E402
    DEFAULT_CANDIDATE_PREDICTIONS,
    DEFAULT_FIXED_CHOICES,
    DEFAULT_LIQUIDITY,
    DEFAULT_PANEL,
    DEFAULT_RISK,
    DEFAULT_SELECTED,
    build_choice_panel,
    load_selected_long_only,
    parse_fixed_choice,
)


GROUP_KINDS = {
    "screen_country": "country",
    "TR.TRBCECONOMICSECTOR": "sector",
}
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "constrained_estimates_long_only_attribution"
)


def load_benchmark_groups(
    panel_path: Path,
    group_columns: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build country/sector benchmark weights and returns from the raw panel."""
    columns = [
        "date",
        "ric",
        "return_1m",
        "company_market_cap",
        *group_columns,
    ]
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"])

    market = build_internal_market(panel).set_index("date")["market_return_eur"]
    groups: dict[str, pd.DataFrame] = {}
    audit: dict[str, Any] = {"market_months": int(len(market))}
    for group_column in group_columns:
        frame = ga.build_benchmark_group_panel(panel, group_column)
        reconstructed = (
            frame.assign(
                weighted=frame["benchmark_weight"] * frame["benchmark_return"]
            )
            .groupby("date")["weighted"]
            .sum()
        )
        common = reconstructed.index.intersection(market.index)
        deviation = float(
            np.abs(
                reconstructed.loc[common].to_numpy() - market.loc[common].to_numpy()
            ).max()
        ) if len(common) else float("nan")
        audit[f"{GROUP_KINDS[group_column]}_reconciliation_max_abs_deviation"] = deviation
        audit[f"{GROUP_KINDS[group_column]}_groups"] = int(frame["group"].nunique())
        groups[group_column] = frame
    return groups, audit


def attribute_strategy(
    exposures: pd.DataFrame,
    benchmark_groups: dict[str, pd.DataFrame],
    *,
    hac_lags: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attribution for every (strategy, constraint, group_kind) combination."""
    attribution_rows = []
    monthly_rows = []
    summary_rows = []

    for (strategy, constraint, group_column), part in exposures.groupby(
        ["strategy", "constraint", "group_kind"], sort=True
    ):
        benchmark = benchmark_groups.get(group_column)
        if benchmark is None or benchmark.empty:
            continue
        # Portfolio exposures are keyed by formation date; the realised return
        # they carry belongs to target_date, which is how the benchmark series
        # is aligned in benchmark_relative_monthly.csv.
        portfolio = part[
            [
                "target_date",
                "group",
                "portfolio_weight",
                "portfolio_return",
                "portfolio_n",
            ]
        ].rename(columns={"target_date": "date"}).copy()
        portfolio["date"] = pd.to_datetime(portfolio["date"])
        benchmark = benchmark.copy()
        benchmark["date"] = pd.to_datetime(benchmark["date"])
        benchmark = benchmark[benchmark["date"].isin(set(portfolio["date"]))]
        if benchmark.empty:
            continue

        attribution = ga.brinson_attribution(portfolio, benchmark)
        monthly = ga.monthly_effect_series(attribution)
        summary = ga.summarize_effects(monthly, maxlags=hac_lags)

        label = {
            "strategy": strategy,
            "constraint": constraint,
            "group_kind": GROUP_KINDS.get(group_column, group_column),
        }
        attribution_rows.append(attribution.assign(**label))
        monthly_rows.append(monthly.assign(**label))
        summary_rows.append(summary.assign(**label))

        top = ga.top_group_contributions(attribution, top_n=8)
        if not top.empty:
            summary_rows.append(
                top.assign(**label, effect="top_group_contribution")
            )

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )

    return _concat(attribution_rows), _concat(monthly_rows), _concat(summary_rows)


def reconcile_costs(
    monthly_attribution: pd.DataFrame,
    constrained_monthly: pd.DataFrame,
    aum_values: tuple[float, ...],
) -> pd.DataFrame:
    """Gross active return minus costs, per AUM level, for the net bridge."""
    if monthly_attribution.empty:
        return pd.DataFrame()
    costs = constrained_monthly.copy()
    costs["date"] = pd.to_datetime(costs["target_date"])
    keep = ["strategy", "constraint", "date", "gross_return"]
    keep.extend(f"net_return_{aum_label(aum)}" for aum in aum_values)
    keep = [column for column in keep if column in costs]
    costs = costs[keep]

    merged = monthly_attribution.merge(
        costs, on=["strategy", "constraint", "date"], how="left"
    )
    records = []
    for (strategy, constraint, group_kind), part in merged.groupby(
        ["strategy", "constraint", "group_kind"], sort=True
    ):
        row: dict[str, Any] = {
            "strategy": strategy,
            "constraint": constraint,
            "group_kind": group_kind,
            "months": int(len(part)),
            "annualized_gross_active": float(
                part["gross_active_return"].mean() * ga.PPY
            ),
            "annualized_allocation": float(part["allocation"].mean() * ga.PPY),
            "annualized_selection": float(part["selection"].mean() * ga.PPY),
            "annualized_interaction": float(part["interaction"].mean() * ga.PPY),
        }
        for aum in aum_values:
            label = aum_label(aum)
            column = f"net_return_{label}"
            if column not in part or "gross_return" not in part:
                continue
            drag = pd.to_numeric(part["gross_return"], errors="coerce") - pd.to_numeric(
                part[column], errors="coerce"
            )
            row[f"annualized_cost_drag_{label}"] = float(drag.mean() * ga.PPY)
            row[f"annualized_net_active_{label}"] = float(
                (part["gross_active_return"] - drag).mean() * ga.PPY
            )
        records.append(row)
    return pd.DataFrame(records)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = parse_constraint_specs(args.constraints)
    fixed_choices = [parse_fixed_choice(value) for value in args.fixed_choices]
    aum_values = tuple(float(value) for value in args.aum)
    group_columns = tuple(GROUP_KINDS)

    selected = load_selected_long_only(
        args.selected, strategy_name=args.selected_strategy
    )
    choices = build_choice_panel(selected, fixed_choices)

    ladder_config = LadderConfig(
        maximum_assets=args.maximum_assets,
        fallback_half_spread_bps=args.fallback_half_spread_bps,
        impact_coefficient=args.impact_coefficient,
        aum_eur=aum_values,
        bootstrap_repetitions=1,
        bootstrap_blocks=(6,),
        random_state=args.random_state,
        hac_lags=args.hac_lags,
    )
    panel = load_ladder_panel(
        args.panel,
        args.predictions,
        args.liquidity,
        args.risk,
        ladder_config,
    )

    exposure_sink: list[dict[str, Any]] = []
    monthly, failures = simulate_constrained(
        panel,
        choices,
        specs,
        maximum_assets=args.maximum_assets,
        aum_values=aum_values,
        impact_coefficient=args.impact_coefficient,
        exposure_sink=exposure_sink,
        exposure_group_columns=group_columns,
    )
    exposures = pd.DataFrame(exposure_sink)
    if exposures.empty:
        raise RuntimeError("No group exposures collected; check the choice panel.")

    benchmark_groups, benchmark_audit = load_benchmark_groups(
        args.panel, group_columns
    )
    attribution, monthly_attribution, summary = attribute_strategy(
        exposures, benchmark_groups, hac_lags=args.hac_lags
    )
    bridge = reconcile_costs(monthly_attribution, monthly, aum_values)

    exposures.to_parquet(
        output_dir / "portfolio_group_exposures.parquet",
        index=False,
        compression="zstd",
    )
    attribution.to_parquet(
        output_dir / "group_attribution.parquet", index=False, compression="zstd"
    )
    monthly_attribution.to_csv(
        output_dir / "attribution_monthly.csv", index=False
    )
    summary.to_csv(output_dir / "attribution_summary.csv", index=False)
    bridge.to_csv(output_dir / "attribution_cost_bridge.csv", index=False)
    if not failures.empty:
        failures.to_csv(output_dir / "constraint_failures.csv", index=False)

    manifest = {
        "panel": str(args.panel),
        "predictions": str(args.predictions),
        "selected": str(args.selected),
        "output_dir": str(output_dir),
        "constraints": [spec.name for spec in specs],
        "aum_eur": list(aum_values),
        "hac_lags": args.hac_lags,
        "benchmark": "internal_eur_value_weighted_market",
        "benchmark_audit": benchmark_audit,
        "rows": {
            "exposures": int(len(exposures)),
            "attribution": int(len(attribution)),
            "monthly": int(len(monthly_attribution)),
            "summary": int(len(summary)),
        },
        "note": (
            "Attribution is on gross returns; see attribution_cost_bridge.csv "
            "for the gross-to-net reconciliation."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument(
        "--predictions", type=Path, default=DEFAULT_CANDIDATE_PREDICTIONS
    )
    parser.add_argument("--selected", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--liquidity", type=Path, default=DEFAULT_LIQUIDITY)
    parser.add_argument("--risk", type=Path, default=DEFAULT_RISK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--selected-strategy", type=str, default="validation_selected"
    )
    parser.add_argument(
        "--constraints",
        nargs="+",
        default=["name5_country40_sector40:0.05:0.40:0.40:0.0"],
        help="Constraint specs as name:max_name:max_country:max_sector:turnover.",
    )
    parser.add_argument(
        "--fixed-choices", nargs="*", default=list(DEFAULT_FIXED_CHOICES)
    )
    parser.add_argument(
        "--aum", nargs="+", type=float, default=[1e7, 1e8, 5e8]
    )
    parser.add_argument("--maximum-assets", type=int, default=500)
    parser.add_argument("--fallback-half-spread-bps", type=float, default=25.0)
    parser.add_argument("--impact-coefficient", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hac-lags", type=int, default=6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
