"""One table for the complexity chapter.

Joins the two pieces of evidence about each increment of model complexity:

  * the *return* evidence (Test C): incremental alpha over everything below it
    on the sequential spanning ladder, with 95% interval and the minimum
    detectable alpha at 80% power;
  * the *capacity* evidence (Test A): whether that increment's information
    advantage varies with how tradable the universe is.

Reading the two together is the point. An increment can be large in return
space and still be worthless to an investor if it lives only in the least
tradable segment -- and the capacity gradient is what reveals that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_LADDER = RESULTS_ROOT / "complexity_spanning_ladder"
DEFAULT_GRADIENT = RESULTS_ROOT / "capacity_gradient_tests"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "complexity_evidence_table"

MODEL_LABELS = {
    "momentum_rank": "Momentum",
    "ridge_rank": "Ridge",
    "hist_gbm_rank": "HistGBM",
    "mlp_rank": "MLP",
    "dre_rank": "DRE",
}

# increment label -> (ladder model, ladder rung, gradient model, gradient premium)
INCREMENTS = [
    ("Momentum over factor model", "momentum_rank", 1, None, None),
    (
        "Ridge over momentum (breadth)",
        "ridge_rank",
        2,
        "ridge_rank",
        "flexibility_premium",
    ),
    (
        "HistGBM over ridge (depth)",
        "hist_gbm_rank",
        3,
        "hist_gbm_rank",
        "depth_premium",
    ),
    ("MLP over ridge (depth)", "mlp_rank", 3, "mlp_rank", "depth_premium"),
    ("DRE over ridge (depth)", "dre_rank", 3, "dre_rank", "depth_premium"),
]


def _fmt_pct(value: float | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value * 100:.{decimals}f}"


def _fmt(value: float | None, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.{decimals}f}"


def build_table(
    ladder: pd.DataFrame,
    gradients: pd.DataFrame,
    weighting: str,
    universe: str,
    cost_bps: int,
) -> pd.DataFrame:
    cell = ladder[
        ladder["weighting"].eq(weighting)
        & ladder["universe_variant"].eq(universe)
        & ladder["portfolio"].eq("long_short")
        & ladder["cost_bps"].eq(cost_bps)
    ]
    records = []
    for label, ladder_model, rung, gradient_model, premium in INCREMENTS:
        row = cell[cell["model"].eq(ladder_model) & cell["rung"].eq(rung)]
        record: dict[str, object] = {"increment": label}
        if not row.empty:
            first = row.iloc[0]
            record.update(
                {
                    "spanning_alpha_annualized": first["alpha_annualized"],
                    "spanning_t": first["alpha_t"],
                    "spanning_p_holm": first.get("alpha_p_holm"),
                    "spanning_ci_low": first["alpha_ci_low_annualized"],
                    "spanning_ci_high": first["alpha_ci_high_annualized"],
                    "spanning_min_detectable": first[
                        "minimum_detectable_alpha_annualized"
                    ],
                    "spanning_benchmarks": first["benchmarks"],
                }
            )
        if gradient_model is not None:
            for dimension, suffix in [
                ("market_cap", "cap"),
                ("trading_value", "adv"),
            ]:
                match = gradients[
                    gradients["model"].eq(gradient_model)
                    & gradients["premium"].eq(premium)
                    & gradients["dimension"].eq(dimension)
                ]
                if match.empty:
                    continue
                first = match.iloc[0]
                record[f"capacity_gradient_{suffix}"] = first["estimate"]
                record[f"capacity_gradient_{suffix}_t"] = first["t_stat"]
                record[f"capacity_gradient_{suffix}_p_holm"] = first.get("p_value_holm")
        records.append(record)
    return pd.DataFrame(records)


def render_markdown(table: pd.DataFrame, weighting: str, universe: str, cost: int) -> str:
    lines = [
        f"### Marginal value of each increment of model complexity",
        "",
        f"Equal/{weighting}-weighted long-short, {universe}, {cost} bps costs. "
        "Spanning alpha is annualised and incremental over every rung below it "
        "(HAC lag 6, Holm within rung). The capacity gradient is the paired "
        "monthly IC difference in the least tradable bucket minus the most "
        "tradable bucket; a large positive value means the increment's advantage "
        "exists only where capital cannot be deployed.",
        "",
        "| Increment | Spanning alpha (%/yr) | t | Holm p | 95% CI | Min. detectable | Capacity gradient (size) | Capacity gradient (ADV) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        ci = (
            f"[{_fmt_pct(row.get('spanning_ci_low'))}, "
            f"{_fmt_pct(row.get('spanning_ci_high'))}]"
        )
        gradient_cap = row.get("capacity_gradient_cap")
        gradient_adv = row.get("capacity_gradient_adv")
        cap_text = (
            "--"
            if pd.isna(gradient_cap)
            else f"{_fmt(gradient_cap)} (t={_fmt(row.get('capacity_gradient_cap_t'), 2)})"
        )
        adv_text = (
            "--"
            if pd.isna(gradient_adv)
            else f"{_fmt(gradient_adv)} (t={_fmt(row.get('capacity_gradient_adv_t'), 2)})"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["increment"]),
                    _fmt_pct(row.get("spanning_alpha_annualized")),
                    _fmt(row.get("spanning_t"), 2),
                    _fmt(row.get("spanning_p_holm"), 4),
                    ci,
                    _fmt_pct(row.get("spanning_min_detectable")),
                    cap_text,
                    adv_text,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-dir", type=Path, default=DEFAULT_LADDER)
    parser.add_argument("--gradient-dir", type=Path, default=DEFAULT_GRADIENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weighting", default="equal")
    parser.add_argument("--universe", default="standard_ex_bottom_5pct")
    parser.add_argument("--cost-bps", type=int, default=25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ladder = pd.read_csv(args.ladder_dir / "complexity_spanning_ladder.csv")
    gradients = pd.read_csv(args.gradient_dir / "capacity_gradient_tests.csv")

    table = build_table(
        ladder, gradients, args.weighting, args.universe, args.cost_bps
    )
    csv_path = args.output_dir / "complexity_evidence_table.csv"
    md_path = args.output_dir / "complexity_evidence_table.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(
        render_markdown(table, args.weighting, args.universe, args.cost_bps)
    )

    manifest = {
        "script": str(Path(__file__).resolve()),
        "sources": [
            str(args.ladder_dir / "complexity_spanning_ladder.csv"),
            str(args.gradient_dir / "capacity_gradient_tests.csv"),
        ],
        "cell": {
            "weighting": args.weighting,
            "universe_variant": args.universe,
            "portfolio": "long_short",
            "cost_bps": args.cost_bps,
        },
        "outputs": {"csv": str(csv_path), "markdown": str(md_path)},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(md_path.read_text())


if __name__ == "__main__":
    main()
