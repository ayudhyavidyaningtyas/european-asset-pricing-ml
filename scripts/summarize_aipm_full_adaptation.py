"""Build a single evidence bundle for the full AIPM adaptation.

The expensive Kelly-Kuznetsov-Malamud-Xu AIPM runs are produced by
``run_aipm_full_transformer_sdf.py`` and ``run_aipm_post_analysis.py``.  This
script deliberately does not retrain models.  It collates the completed
headline, robustness, depth-scaling, monthly-refit and implementability runs
into one auditable adaptation package.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_OUTPUT = RESULTS_ROOT / "aipm_full_adaptation_bundle"


MODEL_ORDER = {
    "bsv": 0,
    "linear_attention": 1,
    "dkkm_random_features": 2,
    "own_asset_mlp": 3,
    "nonlinear_transformer": 4,
}


@dataclass(frozen=True)
class AIPMRunSpec:
    label: str
    run_dir: Path
    post_dir: Path | None = None
    role: str = "evidence"


DEFAULT_RUNS = [
    AIPMRunSpec(
        "headline_top500_three_seed",
        RESULTS_ROOT / "aipm_full_transformer_compustat_cap500_seed3",
        RESULTS_ROOT / "aipm_post_analysis",
        "headline full-sample adaptation",
    ),
    AIPMRunSpec(
        "top1000_robustness",
        RESULTS_ROOT / "aipm_full_transformer_compustat_cap1000_robustness",
        RESULTS_ROOT / "aipm_post_analysis_cap1000_robustness",
        "cross-sectional breadth robustness",
    ),
    AIPMRunSpec(
        "monthly_refit_2020_2026",
        RESULTS_ROOT / "aipm_full_transformer_compustat_cap500_monthly_2020_2026",
        RESULTS_ROOT / "aipm_post_analysis_cap500_monthly_2020_2026",
        "paper-style monthly-refit robustness",
    ),
    AIPMRunSpec(
        "monthly_implementability_standard_2020_2026",
        RESULTS_ROOT
        / "aipm_full_transformer_compustat_cap500_monthly_implementable_2020_2026",
        RESULTS_ROOT / "aipm_post_analysis_cap500_monthly_implementable_2020_2026",
        "implementation-penalized neural objective",
    ),
    AIPMRunSpec(
        "monthly_implementability_mild_2020_2026",
        RESULTS_ROOT
        / "aipm_full_transformer_compustat_cap500_monthly_implementable_mild_2020_2026",
        RESULTS_ROOT / "aipm_post_analysis_cap500_monthly_implementable_mild_2020_2026",
        "mild implementation-penalized neural objective",
    ),
    AIPMRunSpec(
        "depth1_top500_seed1",
        RESULTS_ROOT / "aipm_full_transformer_depth1_cap500_seed1",
        None,
        "nonlinear transformer depth scaling",
    ),
    AIPMRunSpec(
        "depth2_top500_seed1",
        RESULTS_ROOT / "aipm_full_transformer_depth2_cap500_seed1",
        None,
        "nonlinear transformer depth scaling",
    ),
    AIPMRunSpec(
        "depth4_top500_seed1",
        RESULTS_ROOT / "aipm_full_transformer_depth4_cap500_seed1",
        None,
        "nonlinear transformer depth scaling",
    ),
]


PAPER_COMPONENTS = [
    {
        "paper_component": "BSV no-attention linear SDF",
        "paper_role": "Own-asset linear characteristic SDF benchmark.",
        "local_implementation": "src/aipm_full_transformer_sdf.py::fit_closed_form_model(model='bsv')",
        "evidence_file": "aipm_full_summary.csv",
        "status": "implemented",
    },
    {
        "paper_component": "Linear portfolio transformer",
        "paper_role": "Interpretable cross-asset attention surrogate.",
        "local_implementation": "src/aipm_full_transformer_sdf.py::fit_closed_form_model(model='linear_attention')",
        "evidence_file": "aipm_full_summary.csv and aipm_linear_transformer_summary.csv",
        "status": "implemented",
    },
    {
        "paper_component": "MSRR objective",
        "paper_role": "Train SDF weights by minimizing squared pricing-kernel errors.",
        "local_implementation": "src/aipm_full_transformer_sdf.py::_ridge_msrr and fit_neural_model",
        "evidence_file": "aipm_full_fit_log.csv",
        "status": "implemented",
    },
    {
        "paper_component": "DKKM-style random-feature SDF",
        "paper_role": "Own-asset nonlinear high-complexity benchmark without attention.",
        "local_implementation": "src/aipm_full_transformer_sdf.py::random_feature_matrix",
        "evidence_file": "aipm_full_summary.csv",
        "status": "implemented",
    },
    {
        "paper_component": "Own-asset MLP ablation",
        "paper_role": "Same neural training route without cross-asset attention.",
        "local_implementation": "src/aipm_full_transformer_sdf.py::NonlinearPortfolioTransformer(use_attention=False)",
        "evidence_file": "aipm_full_summary.csv and aipm_full_comparisons.csv",
        "status": "implemented",
    },
    {
        "paper_component": "Nonlinear portfolio transformer",
        "paper_role": "Full softmax attention transformer embedded in the SDF.",
        "local_implementation": "src/aipm_full_transformer_sdf.py::NonlinearPortfolioTransformer(use_attention=True)",
        "evidence_file": "aipm_full_summary.csv, aipm_full_weights.parquet, aipm_full_attention_examples.csv",
        "status": "implemented",
    },
    {
        "paper_component": "Out-of-sample Sharpe and HJD pricing errors",
        "paper_role": "Main paper evaluation metrics.",
        "local_implementation": "src/aipm_full_transformer_sdf.py::summarize_full_aipm and pricing_error_summary",
        "evidence_file": "aipm_full_summary.csv and aipm_full_pricing_errors.csv",
        "status": "implemented",
    },
    {
        "paper_component": "Pairwise alpha comparisons",
        "paper_role": "Test whether attention improves on benchmark SDF returns.",
        "local_implementation": "src/aipm_full_transformer_sdf.py::_model_comparisons",
        "evidence_file": "aipm_full_comparisons.csv",
        "status": "implemented",
    },
    {
        "paper_component": "Transformer scaling",
        "paper_role": "Check whether performance improves with architecture depth.",
        "local_implementation": "depth1/depth2/depth4 top-500 controlled runs",
        "evidence_file": "depth_scaling_summary.csv",
        "status": "implemented as tractable European depth grid",
    },
    {
        "paper_component": "Attention mechanism diagnostics",
        "paper_role": "Interpret economic nature of learned cross-asset links.",
        "local_implementation": "src/aipm_post_analysis.py::build_attention_pair_diagnostics",
        "evidence_file": "attention_mechanism_summary.csv",
        "status": "implemented with European issuer metadata",
    },
    {
        "paper_component": "Monthly rolling refits",
        "paper_role": "Paper-style frequent refitting robustness.",
        "local_implementation": "run_aipm_full_transformer_sdf.py --refit-frequency monthly",
        "evidence_file": "model_hierarchy_summary.csv",
        "status": "implemented for 2020-2026 tractable subperiod",
    },
    {
        "paper_component": "Implementability extension",
        "paper_role": "Dissertation-specific cost and liquidity audit beyond the paper.",
        "local_implementation": "src/aipm_post_analysis.py::simulate_weight_implementability",
        "evidence_file": "implementability_summary_100m.csv",
        "status": "implemented as extension",
    },
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _model_sort_key(model: str) -> int:
    return MODEL_ORDER.get(model, 99)


def collect_model_summaries(specs: list[AIPMRunSpec]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for spec in specs:
        summary = _read_csv(spec.run_dir / "aipm_full_summary.csv")
        if summary.empty:
            continue
        manifest = _read_json(spec.run_dir / "aipm_full_manifest.json")
        fit_log = _read_csv(spec.run_dir / "aipm_full_fit_log.csv")
        config = manifest.get("config", {})
        rows = manifest.get("rows", {})
        causality = manifest.get("causality_check", {})
        fit_seconds = (
            fit_log.groupby("model")["fit_seconds"].sum()
            if not fit_log.empty and "fit_seconds" in fit_log
            else pd.Series(dtype=float)
        )
        refits = (
            fit_log.groupby("model")["refit_id"].nunique()
            if not fit_log.empty and "refit_id" in fit_log
            else pd.Series(dtype=float)
        )
        n_parameters = (
            fit_log.groupby("model")["n_parameters"].max()
            if not fit_log.empty and "n_parameters" in fit_log
            else pd.Series(dtype=float)
        )
        for _, row in summary.iterrows():
            model = str(row["model"])
            records.append(
                {
                    "run_label": spec.label,
                    "run_role": spec.role,
                    "run_dir": str(spec.run_dir),
                    "model": model,
                    "model_order": _model_sort_key(model),
                    "first_test_year": config.get("first_test_year"),
                    "last_test_year": config.get("last_test_year"),
                    "refit_frequency": config.get("refit_frequency"),
                    "training_window_months": config.get("training_window_months"),
                    "validation_months": config.get("validation_months"),
                    "max_monthly_stocks": config.get("max_monthly_stocks"),
                    "transformer_blocks": config.get("transformer_blocks"),
                    "attention_heads": config.get("attention_heads"),
                    "feedforward_width": config.get("feedforward_width"),
                    "epochs": config.get("epochs"),
                    "seeds": ",".join(str(seed) for seed in config.get("seeds", [])),
                    "n_parameters": float(n_parameters.get(model, np.nan)),
                    "refits": int(refits.get(model, 0)) if not refits.empty else np.nan,
                    "fit_seconds": float(fit_seconds.get(model, np.nan)),
                    "stored_weight_rows": rows.get("weights"),
                    "stored_attention_rows": rows.get("attention_examples"),
                    "train_target_after_cutoff": causality.get("train_target_after_cutoff"),
                    "validation_target_after_cutoff": causality.get(
                        "validation_target_after_cutoff"
                    ),
                    "duplicate_weight_security_months": causality.get(
                        "duplicate_weight_security_months"
                    ),
                    **{
                        column: row[column]
                        for column in summary.columns
                        if column != "model"
                    },
                }
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).sort_values(
        ["run_label", "model_order", "model"]
    )


def collect_pairwise_comparisons(specs: list[AIPMRunSpec]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in specs:
        comparisons = _read_csv(spec.run_dir / "aipm_full_comparisons.csv")
        if comparisons.empty:
            continue
        comparisons = comparisons.copy()
        comparisons.insert(0, "run_label", spec.label)
        comparisons.insert(1, "run_role", spec.role)
        comparisons.insert(2, "run_dir", str(spec.run_dir))
        frames.append(comparisons)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_implementability(specs: list[AIPMRunSpec], aum_label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in specs:
        if spec.post_dir is None:
            continue
        summary = _read_csv(spec.post_dir / "aipm_implementability_summary.csv")
        if summary.empty:
            continue
        summary = summary[summary["aum_label"].astype(str).eq(aum_label)].copy()
        if summary.empty:
            continue
        summary.insert(0, "run_label", spec.label)
        summary.insert(1, "run_role", spec.role)
        summary.insert(2, "post_dir", str(spec.post_dir))
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_attention_mechanism(post_dir: Path) -> pd.DataFrame:
    lift = _read_csv(post_dir / "aipm_attention_lift.csv")
    if lift.empty:
        return pd.DataFrame()
    aggregate = (
        lift.groupby("metric", as_index=False)
        .agg(
            months=("signal_date", "nunique"),
            observed_mean=("observed", "mean"),
            null_mean=("null", "mean"),
            lift_mean=("lift", "mean"),
            lift_median=("lift", "median"),
        )
        .sort_values("metric")
    )
    keep = [
        "same_screen_country_weighted_mean",
        "same_trbceconomicsector_weighted_mean",
        "same_trbcbusinesssector_weighted_mean",
        "same_trbcindustrygroup_weighted_mean",
        "same_trbcindustry_weighted_mean",
        "abs_diff_market_cap_percentile_weighted_mean",
        "abs_diff_log_size_rank_weighted_mean",
        "abs_diff_book_to_market_rank_weighted_mean",
        "abs_diff_momentum_12_2_rank_weighted_mean",
        "abs_diff_volatility_12m_rank_weighted_mean",
    ]
    focused = aggregate[aggregate["metric"].isin(keep)].copy()
    if not focused.empty:
        focused["interpretation"] = focused["metric"].map(
            {
                "same_screen_country_weighted_mean": "positive lift means country clustering",
                "same_trbceconomicsector_weighted_mean": "positive lift means economic-sector peer attention",
                "same_trbcbusinesssector_weighted_mean": "positive lift means business-sector peer attention",
                "same_trbcindustrygroup_weighted_mean": "positive lift means industry-group peer attention",
                "same_trbcindustry_weighted_mean": "positive lift means industry peer attention",
                "abs_diff_market_cap_percentile_weighted_mean": "negative lift means attention to similarly sized stocks",
                "abs_diff_log_size_rank_weighted_mean": "negative lift means attention to similarly sized stocks",
                "abs_diff_book_to_market_rank_weighted_mean": "negative lift means valuation similarity",
                "abs_diff_momentum_12_2_rank_weighted_mean": "negative lift means momentum similarity",
                "abs_diff_volatility_12m_rank_weighted_mean": "negative lift means risk similarity",
            }
        )
    return focused


def build_depth_scaling(model_summaries: pd.DataFrame) -> pd.DataFrame:
    if model_summaries.empty:
        return pd.DataFrame()
    depth = model_summaries[
        model_summaries["run_label"].str.startswith("depth", na=False)
        & model_summaries["model"].isin(["own_asset_mlp", "nonlinear_transformer"])
    ].copy()
    if depth.empty:
        return depth
    pivot = depth.pivot_table(
        index=["run_label", "transformer_blocks"],
        columns="model",
        values=["annualized_return", "sharpe", "normalized_hjd_pricing_error"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{model}" for metric, model in pivot.columns]
    pivot = pivot.reset_index()
    pivot["transformer_minus_mlp_return"] = (
        pivot.get("annualized_return_nonlinear_transformer")
        - pivot.get("annualized_return_own_asset_mlp")
    )
    pivot["transformer_minus_mlp_sharpe"] = (
        pivot.get("sharpe_nonlinear_transformer")
        - pivot.get("sharpe_own_asset_mlp")
    )
    return pivot.sort_values("transformer_blocks")


def _format_pct(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _format_float(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_brief(
    output_dir: Path,
    model_summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    implementability: pd.DataFrame,
    attention: pd.DataFrame,
    depth: pd.DataFrame,
) -> None:
    headline = model_summaries[
        model_summaries["run_label"].eq("headline_top500_three_seed")
    ].copy()
    headline = headline.sort_values(["model_order", "model"])
    comp = comparisons[
        comparisons["run_label"].eq("headline_top500_three_seed")
        & comparisons["model"].eq("nonlinear_transformer")
        & comparisons["baseline"].isin(["bsv", "own_asset_mlp"])
    ]
    impl = implementability[
        implementability["run_label"].eq("headline_top500_three_seed")
        & implementability["model"].isin(["bsv", "own_asset_mlp", "nonlinear_transformer"])
    ]
    lines = [
        "# Full AIPM Adaptation Bundle",
        "",
        "This bundle collates the European adaptation of Kelly, Kuznetsov, "
        "Malamud and Xu's *Artificial Intelligence Asset Pricing Models*.",
        "The implementation estimates SDF portfolios using the AIPM hierarchy: "
        "BSV, linear attention, DKKM-style random features, own-asset MLP and "
        "the nonlinear portfolio transformer.",
        "",
        "## Headline Top-500 Run",
        "",
        "| model | annual return | Sharpe | HJD error | turnover | parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in headline.iterrows():
        lines.append(
            "| {model} | {ret} | {sharpe} | {hjd} | {turnover} | {params} |".format(
                model=row["model"],
                ret=_format_pct(row["annualized_return"]),
                sharpe=_format_float(row["sharpe"]),
                hjd=_format_float(row["normalized_hjd_pricing_error"]),
                turnover=_format_float(row["average_monthly_turnover"]),
                params="n/a"
                if pd.isna(row.get("n_parameters"))
                else f"{int(row['n_parameters']):,}",
            )
        )
    lines.extend(["", "## Attention Increment", ""])
    if comp.empty:
        lines.append("No headline transformer comparison rows were available.")
    else:
        lines.extend(
            [
                "| comparison | annual difference | alpha t | alpha p | correlation |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in comp.iterrows():
            lines.append(
                "| transformer minus {baseline} | {diff} | {t} | {p} | {corr} |".format(
                    baseline=row["baseline"],
                    diff=_format_pct(row["annualized_mean_difference"]),
                    t=_format_float(row["alpha_hac_t"]),
                    p=_format_float(row["alpha_hac_p"]),
                    corr=_format_float(row["correlation"]),
                )
            )
    lines.extend(["", "## EUR 100m Implementability", ""])
    if impl.empty:
        lines.append("No EUR 100m implementability rows were available.")
    else:
        lines.extend(
            [
                "| model | net return | net Sharpe | annual cost | spread coverage |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in impl.sort_values("model").iterrows():
            lines.append(
                "| {model} | {ret} | {sharpe} | {cost} | {coverage} |".format(
                    model=row["model"],
                    ret=_format_pct(row["annualized_net_return"]),
                    sharpe=_format_float(row["net_sharpe"]),
                    cost=_format_pct(row["annualized_total_cost"]),
                    coverage=_format_pct(row["spread_observed_weight"]),
                )
            )
    lines.extend(["", "## Depth Scaling", ""])
    if depth.empty:
        lines.append("No controlled depth-scaling rows were available.")
    else:
        lines.extend(
            [
                "| blocks | transformer Sharpe | MLP Sharpe | transformer-minus-MLP Sharpe | transformer-minus-MLP return |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in depth.iterrows():
            lines.append(
                "| {blocks:.0f} | {tsharpe} | {msharpe} | {dsharpe} | {dret} |".format(
                    blocks=row["transformer_blocks"],
                    tsharpe=_format_float(row["sharpe_nonlinear_transformer"]),
                    msharpe=_format_float(row["sharpe_own_asset_mlp"]),
                    dsharpe=_format_float(row["transformer_minus_mlp_sharpe"]),
                    dret=_format_pct(row["transformer_minus_mlp_return"]),
                )
            )
    lines.extend(["", "## Attention Mechanism", ""])
    if attention.empty:
        lines.append("No attention mechanism summary rows were available.")
    else:
        lines.extend(
            [
                "| metric | observed | null | lift | interpretation |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for _, row in attention.iterrows():
            lines.append(
                "| {metric} | {obs} | {null} | {lift} | {interp} |".format(
                    metric=row["metric"],
                    obs=_format_float(row["observed_mean"]),
                    null=_format_float(row["null_mean"]),
                    lift=_format_float(row["lift_mean"]),
                    interp=row.get("interpretation", ""),
                )
            )
    lines.extend(
        [
            "",
            "## Dissertation Interpretation",
            "",
            "The full AIPM architecture is feasible in European equities and the "
            "attention layer learns economically interpretable peer structure. "
            "However, the nonlinear transformer does not reliably dominate the "
            "own-asset MLP ablation in the European panel, and the controlled "
            "depth grid does not show the monotone scaling pattern reported in "
            "the U.S. AIPM paper. This strengthens the dissertation's central "
            "predictability-implementability gap rather than weakening it.",
            "",
        ]
    )
    (output_dir / "FULL_AIPM_ADAPTATION_BRIEF.md").write_text("\n".join(lines))


def write_bundle(specs: list[AIPMRunSpec], output_dir: Path, aum_label: str) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    component_map = pd.DataFrame(PAPER_COMPONENTS)
    model_summaries = collect_model_summaries(specs)
    comparisons = collect_pairwise_comparisons(specs)
    implementability = collect_implementability(specs, aum_label)
    attention = summarize_attention_mechanism(
        RESULTS_ROOT / "aipm_post_analysis"
    )
    depth = build_depth_scaling(model_summaries)

    component_map.to_csv(output_dir / "paper_component_map.csv", index=False)
    model_summaries.to_csv(output_dir / "model_hierarchy_summary.csv", index=False)
    comparisons.to_csv(output_dir / "pairwise_attention_comparisons.csv", index=False)
    implementability.to_csv(output_dir / "implementability_summary_100m.csv", index=False)
    attention.to_csv(output_dir / "attention_mechanism_summary.csv", index=False)
    depth.to_csv(output_dir / "depth_scaling_summary.csv", index=False)
    write_brief(output_dir, model_summaries, comparisons, implementability, attention, depth)

    manifest = {
        "run_specs": [asdict(spec) for spec in specs],
        "aum_label": aum_label,
        "rows": {
            "paper_component_map": int(len(component_map)),
            "model_hierarchy_summary": int(len(model_summaries)),
            "pairwise_attention_comparisons": int(len(comparisons)),
            "implementability_summary_100m": int(len(implementability)),
            "attention_mechanism_summary": int(len(attention)),
            "depth_scaling_summary": int(len(depth)),
        },
        "outputs": {
            "paper_component_map": str(output_dir / "paper_component_map.csv"),
            "model_hierarchy_summary": str(output_dir / "model_hierarchy_summary.csv"),
            "pairwise_attention_comparisons": str(
                output_dir / "pairwise_attention_comparisons.csv"
            ),
            "implementability_summary_100m": str(
                output_dir / "implementability_summary_100m.csv"
            ),
            "attention_mechanism_summary": str(
                output_dir / "attention_mechanism_summary.csv"
            ),
            "depth_scaling_summary": str(output_dir / "depth_scaling_summary.csv"),
            "brief": str(output_dir / "FULL_AIPM_ADAPTATION_BRIEF.md"),
        },
    }
    (output_dir / "aipm_full_adaptation_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    return manifest


def parse_run_spec(value: str) -> AIPMRunSpec:
    parts = value.split(":", 3)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("Run spec must be LABEL:RUN_DIR[:POST_DIR[:ROLE]]")
    label = parts[0]
    run_dir = Path(parts[1])
    post_dir = Path(parts[2]) if len(parts) >= 3 and parts[2] else None
    role = parts[3] if len(parts) >= 4 and parts[3] else "custom evidence"
    return AIPMRunSpec(label, run_dir, post_dir, role)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aum-label", default="100m")
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        help="Optional LABEL:RUN_DIR[:POST_DIR[:ROLE]] override. Defaults to all known AIPM runs.",
    )
    args = parser.parse_args()

    specs = args.run if args.run else DEFAULT_RUNS
    missing = [str(spec.run_dir) for spec in specs if not spec.run_dir.exists()]
    if missing:
        raise SystemExit(f"Missing AIPM run directories: {missing}")
    manifest = write_bundle(specs, args.output_dir, args.aum_label)
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
