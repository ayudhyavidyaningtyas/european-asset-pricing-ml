"""Assemble the identification evidence for the analyst-estimates layer.

One exhibit, four panels, in the order the argument runs:

  A. Coverage selection -- the data-depth effect unweighted, under
     inverse-propensity reweighting to the full eligible universe, and across
     the propensity-floor sensitivity grid.
  B. Lag decay -- the data-depth effect by signal lag on the stock-months
     common to every lag (primary scope); the own-lag matched sample is kept
     as a robustness column.
  C. Standalone attribution -- each analyst group alone on top of the
     Compustat baseline ("X_only" versus Compustat).
  D. Marginal attribution -- the full analyst set versus leave-one-group-out
     ("full" versus "ex_X"), conditional on every other analyst group.

Panels C and D answer different questions and are kept separate on purpose:
correlated groups can be standalone-informative yet marginally redundant.

The script refuses to build the attribution panels if any compared pair in
ablation_sample_checks.csv was not scored on identical stock-months.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_COVERAGE_DIR = RESULTS_ROOT / "estimates_coverage_selection_20260816"
DEFAULT_LADDER_DIR = RESULTS_ROOT / "estimates_lag_ladder_20260816"
DEFAULT_ABLATION_DIR = RESULTS_ROOT / "estimates_family_ablation_refresh_20260816"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "estimates_identification_evidence_20260816"

STANDALONE_TEST = "monthly_ic_variant_minus_compustat"
MARGINAL_TEST = "monthly_ic_full_minus_leave_one_out"


def coverage_panel(coverage_dir: Path) -> pd.DataFrame:
    tests = pd.read_csv(coverage_dir / "coverage_selection_data_depth_tests.csv")
    keep = [
        "weighting",
        "min_propensity_floor",
        "model",
        "months",
        "estimate",
        "standard_error",
        "t_stat",
        "p_value",
        "p_value_holm",
    ]
    out = tests[keep].copy()
    out.insert(0, "panel", "A_coverage_selection")
    return out


def lag_panel(ladder_dir: Path) -> pd.DataFrame:
    effects = pd.read_csv(ladder_dir / "lag_ladder_data_depth.csv")
    primary = effects[effects["sample_scope"].eq("common_across_lags")].copy()
    robustness = effects[effects["sample_scope"].eq("own_matched_sample")][
        ["lag_months", "model", "estimate", "p_value_holm"]
    ].rename(
        columns={
            "estimate": "own_sample_estimate",
            "p_value_holm": "own_sample_p_value_holm",
        }
    )
    out = primary.merge(robustness, on=["lag_months", "model"], how="left")
    keep = [
        "lag_months",
        "model",
        "stock_months",
        "months",
        "estimate",
        "standard_error",
        "t_stat",
        "p_value",
        "p_value_holm",
        "share_of_shortest_lag",
        "own_sample_estimate",
        "own_sample_p_value_holm",
    ]
    out = out[keep].copy()
    out.insert(0, "panel", "B_lag_decay_common_sample")
    return out


def _check_ablation_samples(ablation_dir: Path) -> pd.DataFrame:
    checks_path = ablation_dir / "ablation_sample_checks.csv"
    if not checks_path.exists():
        raise SystemExit(
            f"Missing {checks_path}; rerun run_estimates_ablation_paired_tests.py "
            "(it writes the sample checks alongside the test tables)."
        )
    checks = pd.read_csv(checks_path)
    unmatched = checks[~checks["samples_identical"]]
    if not unmatched.empty:
        raise SystemExit(
            "Ablation pairs scored on non-identical stock-months:\n"
            f"{unmatched.to_string(index=False)}\n"
            "Fix the runs before building the attribution panels."
        )
    return checks


def attribution_panels(ablation_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = _check_ablation_samples(ablation_dir)
    ic_tests = pd.read_csv(ablation_dir / "ablation_paired_ic_tests.csv")
    keep = [
        "variant",
        "reference",
        "model",
        "months",
        "mean_ic_variant",
        "mean_ic_reference",
        "delta_mean_ic",
        "hac_t_stat",
        "hac_p_two_sided",
        "hac_p_holm",
    ]

    # Standalone content = the "X_only" cells plus the full 11-feature layer as
    # the reference row. The "ex_X" cells are also compared against Compustat in
    # the same Holm family (which makes the correction conservative -- the
    # family spans every variant), but they are not standalone readings and are
    # reported only through Panel D.
    versus_compustat = ic_tests[ic_tests["test"].eq(STANDALONE_TEST)][keep].copy()
    is_only = versus_compustat["variant"].str.endswith("_only")
    is_full = versus_compustat["variant"].eq("estimates_enriched")
    standalone = versus_compustat[is_only | is_full].copy()
    standalone["group"] = (
        standalone["variant"]
        .str.removeprefix("estimates_")
        .str.removesuffix("_only")
        .replace({"enriched": "all_11_features"})
    )
    standalone.insert(0, "panel", "C_standalone_attribution")

    marginal = ic_tests[ic_tests["test"].eq(MARGINAL_TEST)][keep].copy()
    marginal["group"] = marginal["reference"].str.removeprefix("estimates_ex_")
    # full-minus-ex_X: a positive delta is X's marginal contribution.
    marginal.insert(0, "panel", "D_marginal_attribution")

    for frame in (standalone, marginal):
        frame.attrs["sample_checks"] = int(len(checks))
    return standalone, marginal


def render_markdown(
    coverage: pd.DataFrame,
    lag: pd.DataFrame,
    standalone: pd.DataFrame,
    marginal: pd.DataFrame,
) -> str:
    def table(frame: pd.DataFrame, columns: dict[str, str], sort: list[str]) -> str:
        out = frame[list(columns)].rename(columns=columns).sort_values(
            [columns[c] for c in sort]
        )
        return out.to_markdown(index=False, floatfmt=".4f")

    lines = [
        "# Identification evidence: analyst-estimates layer",
        "",
        "## Panel A -- coverage selection (selection-robustness diagnostic)",
        "",
        "Data-depth effect (monthly IC, estimates cell minus Compustat cell) "
        "unweighted and reweighted to the full eligible universe.",
        "",
        table(
            coverage,
            {
                "weighting": "weighting",
                "model": "model",
                "estimate": "estimate",
                "t_stat": "t",
                "p_value_holm": "Holm p",
            },
            ["weighting", "model"],
        ),
        "",
        "## Panel B -- lag decay (common stock-months across lags)",
        "",
        table(
            lag,
            {
                "model": "model",
                "lag_months": "lag",
                "estimate": "estimate",
                "p_value_holm": "Holm p",
                "share_of_shortest_lag": "share of lag-1",
                "own_sample_estimate": "own-sample est.",
            },
            ["model", "lag_months"],
        ),
        "",
        "## Panel C -- standalone attribution (X_only vs Compustat)",
        "",
        table(
            standalone,
            {
                "group": "group",
                "model": "model",
                "delta_mean_ic": "delta IC",
                "hac_t_stat": "t",
                "hac_p_holm": "Holm p",
            },
            ["group", "model"],
        ),
        "",
        "## Panel D -- marginal attribution (full vs ex_X)",
        "",
        table(
            marginal,
            {
                "group": "group",
                "model": "model",
                "delta_mean_ic": "delta IC",
                "hac_t_stat": "t",
                "hac_p_holm": "Holm p",
            },
            ["group", "model"],
        ),
        "",
        "Panels C and D are not interchangeable: standalone content and "
        "marginal contribution differ when analyst groups are correlated.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    parser.add_argument("--ladder-dir", type=Path, default=DEFAULT_LADDER_DIR)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coverage = coverage_panel(args.coverage_dir)
    lag = lag_panel(args.ladder_dir)
    standalone, marginal = attribution_panels(args.ablation_dir)

    combined = pd.concat(
        [coverage, lag, standalone, marginal], ignore_index=True
    )
    combined.to_csv(
        args.output_dir / "identification_evidence_panels.csv", index=False
    )
    (args.output_dir / "identification_evidence.md").write_text(
        render_markdown(coverage, lag, standalone, marginal)
    )
    manifest = {
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "coverage": str(args.coverage_dir),
            "lag_ladder": str(args.ladder_dir),
            "ablation": str(args.ablation_dir),
        },
        "panels": {
            "A": "coverage selection (unweighted, IPW, floor sensitivity)",
            "B": "lag decay, common stock-months primary, own-sample robustness",
            "C": "standalone attribution: X_only vs Compustat baseline",
            "D": "marginal attribution: full vs ex_X",
        },
        "rows": {
            "A": int(len(coverage)),
            "B": int(len(lag)),
            "C": int(len(standalone)),
            "D": int(len(marginal)),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(manifest["rows"], indent=2))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
