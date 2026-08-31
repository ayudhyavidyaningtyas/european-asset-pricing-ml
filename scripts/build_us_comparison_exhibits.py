"""Build manuscript exhibits for the matched US-Europe market comparison.

Reads the frozen market-comparison outputs (Refinitiv-only and
Compustat-enriched), the Compustat enrichment audits and the benchmark run
manifests, and emits:

- tables (CSV plus a combined markdown render) under
  ``results/asset_pricing_ml/us_comparison_exhibits/``;
- figures (PNG and PDF with underlying data CSVs) under
  ``figures/manuscript/``.

The US comparison uses the ``expanded_liquidity`` and ``compustat_enriched``
feature sets. It does not use the Europe-only analyst-estimates panel, and no
exhibit produced here should be described as an analyst-estimates result.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "asset_pricing"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "us_comparison_exhibits"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures" / "manuscript"
DEFAULT_COMPARISONS = {
    "compustat_enriched": RESULTS_ROOT / "market_comparison_compustat",
    "refinitiv_only": RESULTS_ROOT / "market_comparison_refinitiv_only",
}
DEFAULT_BENCHMARKS = {
    "Europe": RESULTS_ROOT / "europe_compustat_benchmark",
    "US": RESULTS_ROOT / "us_compustat_benchmark",
}
DEFAULT_AUDITS = {
    "Europe": DATA_ROOT / "compustat_enrichment_audit.json",
    "US": DATA_ROOT / "wrds_compustat_us_enrichment_audit.json",
}

MODEL_ORDER = [
    "momentum_rank",
    "ridge_rank",
    "elastic_net_rank",
    "hist_gbm_rank",
    "mlp_rank",
]
MODEL_LABELS = {
    "momentum_rank": "Momentum",
    "ridge_rank": "Ridge",
    "elastic_net_rank": "Elastic Net",
    "hist_gbm_rank": "HistGBM",
    "mlp_rank": "MLP",
}
RETURN_TARGET_BASES = ["ridge", "elastic_net", "hist_gbm", "mlp"]
MARKET_COLORS = {"Europe": "#1f77b4", "US": "#ff7f0e"}
FEATURE_SET_LABELS = {
    "compustat_enriched": "Compustat-enriched",
    "refinitiv_only": "Refinitiv-only",
}


@dataclass(frozen=True)
class FigureRecord:
    figure: str
    png: str
    pdf: str
    data_csv: str
    source_files: str
    description: str


def _set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
        }
    )


def _model_sort_key(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def _save_figure(
    fig: plt.Figure,
    data: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    source_files: Iterable[Path],
    description: str,
) -> FigureRecord:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    data.to_csv(data_path, index=False)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return FigureRecord(
        figure=stem,
        png=str(png_path),
        pdf=str(pdf_path),
        data_csv=str(data_path),
        source_files=";".join(str(path) for path in source_files),
        description=description,
    )


def panel_coverage_table(
    benchmark_dirs: dict[str, Path], side_by_side_path: Path
) -> pd.DataFrame:
    """Matched-sample coverage per market.

    Panel and model-row counts come from each benchmark run manifest; the
    out-of-sample month count and average names per month come from the
    comparison side-by-side summary (rank-model rows share a single sample).
    """
    side = pd.read_csv(side_by_side_path)
    reference = side[side["model"].isin(MODEL_ORDER)].iloc[0]
    rows = []
    for market, run_dir in benchmark_dirs.items():
        manifest = json.loads((run_dir / "ml_manifest.json").read_text())
        audit = manifest["sample_filter_audit"]
        suffix = market.lower().replace(" ", "_")
        months = int(reference[f"months_{suffix}"])
        observations = int(reference[f"observations_{suffix}"])
        predictions = pd.read_parquet(
            run_dir / "predictions.parquet", columns=["ric"]
        )
        rows.append(
            {
                "market": market,
                "panel_rows": int(audit["loaded_rows"]),
                "model_rows": int(audit["model_rows"]),
                "securities_with_oos_predictions": int(
                    predictions["ric"].nunique()
                ),
                "oos_months": months,
                "oos_stock_months": observations,
                "avg_oos_names_per_month": round(observations / months, 1),
            }
        )
    return pd.DataFrame(rows)


def compustat_match_quality_table(audit_paths: dict[str, Path]) -> pd.DataFrame:
    """Compustat match coverage per market, from the enrichment audit files."""
    rows = []
    for market, path in audit_paths.items():
        audit = json.loads(path.read_text())
        panel = audit["panel"]
        rows.append(
            {
                "market": market,
                "panel_rows": int(panel["rows"]),
                "rows_with_compustat_annual": int(
                    panel["rows_with_compustat_annual"]
                ),
                "rows_with_compustat_monthly": int(
                    panel["rows_with_compustat_monthly"]
                ),
                "securities_with_compustat_annual": int(
                    panel["unique_rics_with_compustat_annual"]
                ),
                "securities_with_compustat_monthly": int(
                    panel["unique_rics_with_compustat_monthly"]
                ),
                "mean_compustat_feature_count": round(
                    float(panel["mean_compustat_feature_count"]), 2
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_model_comparison_table(
    side_by_side_path: Path, feature_set: str
) -> pd.DataFrame:
    """Rank-model IC and net portfolio comparison for one feature set."""
    side = pd.read_csv(side_by_side_path)
    data = side[side["model"].isin(MODEL_ORDER)].copy()
    data = data.sort_values("model", key=lambda col: col.map(_model_sort_key))
    out = pd.DataFrame(
        {
            "feature_set": feature_set,
            "model": data["model"],
            "model_label": data["model"].map(MODEL_LABELS),
            "ic_europe": data["mean_monthly_spearman_ic_europe"].round(4),
            "ic_us": data["mean_monthly_spearman_ic_us"].round(4),
            "ic_us_minus_europe": data[
                "mean_monthly_spearman_ic_us_minus_europe"
            ].round(4),
            "net_return_europe": data["annualized_net_mean_return_europe"].round(4),
            "net_return_us": data["annualized_net_mean_return_us"].round(4),
            "net_volatility_europe": data[
                "annualized_net_volatility_europe"
            ].round(4),
            "net_volatility_us": data["annualized_net_volatility_us"].round(4),
            "net_sharpe_europe": data["net_sharpe_europe"].round(2),
            "net_sharpe_us": data["net_sharpe_us"].round(2),
            "turnover_europe": data["average_monthly_turnover_europe"].round(2),
            "turnover_us": data["average_monthly_turnover_us"].round(2),
        }
    )
    return out.reset_index(drop=True)


def return_target_instability_table(
    side_by_side_path: Path, feature_set: str
) -> pd.DataFrame:
    """Rank versus return targets for the multi-characteristic models."""
    side = pd.read_csv(side_by_side_path)
    data = side[side["base_model"].isin(RETURN_TARGET_BASES)].copy()
    data = data.sort_values(
        ["base_model", "target_mode"],
        key=lambda col: (
            col.map(RETURN_TARGET_BASES.index)
            if col.name == "base_model"
            else col
        ),
    )
    out = pd.DataFrame(
        {
            "feature_set": feature_set,
            "base_model": data["base_model"],
            "target_mode": data["target_mode"],
            "ic_europe": data["mean_monthly_spearman_ic_europe"].round(4),
            "ic_us": data["mean_monthly_spearman_ic_us"].round(4),
            "net_return_europe": data["annualized_net_mean_return_europe"].round(4),
            "net_return_us": data["annualized_net_mean_return_us"].round(4),
            "net_sharpe_europe": data["net_sharpe_europe"].round(2),
            "net_sharpe_us": data["net_sharpe_us"].round(2),
        }
    )
    return out.reset_index(drop=True)


def return_correlation_table(
    correlations_path: Path, feature_set: str
) -> pd.DataFrame:
    """Common-month Europe-US long-short return correlations, rank models."""
    corr = pd.read_csv(correlations_path)
    data = corr[corr["model"].isin(MODEL_ORDER)].copy()
    data = data.sort_values("model", key=lambda col: col.map(_model_sort_key))
    out = pd.DataFrame(
        {
            "feature_set": feature_set,
            "model": data["model"],
            "model_label": data["model"].map(MODEL_LABELS),
            "common_months": data["common_months"].astype(int),
            "first_common_month": data["first_common_month"],
            "last_common_month": data["last_common_month"],
            "return_correlation": data["return_correlation"].round(2),
        }
    )
    return out.reset_index(drop=True)


def ic_sharpe_figure_data(side_by_side_path: Path) -> pd.DataFrame:
    side = pd.read_csv(side_by_side_path)
    data = side[side["model"].isin(MODEL_ORDER)].copy()
    data = data.sort_values("model", key=lambda col: col.map(_model_sort_key))
    return pd.DataFrame(
        {
            "model": data["model"],
            "model_label": data["model"].map(MODEL_LABELS),
            "ic_europe": data["mean_monthly_spearman_ic_europe"],
            "ic_us": data["mean_monthly_spearman_ic_us"],
            "net_sharpe_europe": data["net_sharpe_europe"],
            "net_sharpe_us": data["net_sharpe_us"],
        }
    ).reset_index(drop=True)


def build_ic_sharpe_figure(
    data: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    source_files: Iterable[Path],
    feature_set_label: str,
) -> FigureRecord:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    positions = range(len(data))
    width = 0.38
    panels = [
        ("ic", "Mean monthly Spearman IC", axes[0]),
        ("net_sharpe", "Net Sharpe ratio", axes[1]),
    ]
    for metric, title, ax in panels:
        for offset, market in ((-width / 2, "Europe"), (width / 2, "US")):
            ax.bar(
                [p + offset for p in positions],
                data[f"{metric}_{market.lower()}"],
                width=width,
                color=MARKET_COLORS[market],
                edgecolor="black",
                linewidth=0.6,
            )
        ax.set_xticks(list(positions))
        ax.set_xticklabels(data["model_label"], rotation=0)
        ax.set_title(title)
        ax.axhline(0.0, color="black", linewidth=0.8)
        if metric == "ic":
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _pos: f"{value:.2f}")
            )
    handles = [
        Line2D(
            [0],
            [0],
            color="none",
            marker="s",
            markerfacecolor=MARKET_COLORS[market],
            markeredgecolor="black",
            label=market,
        )
        for market in ("Europe", "US")
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=False)
    fig.suptitle(
        "Rank-model predictability and net performance, Europe vs US "
        f"({feature_set_label})",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        stem,
        source_files=source_files,
        description=(
            "Mean monthly Spearman IC and net Sharpe (value-weighted long-short,"
            " ex bottom 5%, 25bp) for rank-target models in Europe and the US, "
            f"{feature_set_label} feature set."
        ),
    )


def _format_table_markdown(table: pd.DataFrame) -> str:
    return table.to_markdown(index=False)


def write_markdown_report(
    output_dir: Path, tables: dict[str, pd.DataFrame]
) -> Path:
    lines = [
        "# US-Europe Comparison Exhibits",
        "",
        "Produced by `scripts/build_us_comparison_exhibits.py` from frozen",
        "market-comparison runs. Value-weighted long-short portfolios,",
        "`standard_ex_bottom_5pct`, 25bp costs, OOS 2015-02 to 2026-06.",
        "",
        "The US comparison uses Refinitiv plus Compustat feature sets. It does",
        "not use the Europe-only analyst-estimates panel.",
        "",
    ]
    titles = {
        "panel_coverage": "Matched-sample panel coverage",
        "compustat_match_quality": "Compustat match quality",
        "rank_model_comparison": "Rank-model comparison",
        "return_target_instability": "Rank vs return targets",
        "return_correlations": "Common-month return correlations",
    }
    for name, table in tables.items():
        lines.append(f"## {titles.get(name, name)}")
        lines.append("")
        lines.append(_format_table_markdown(table))
        lines.append("")
    path = output_dir / "us_comparison_tables.md"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--compustat-comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISONS["compustat_enriched"],
    )
    parser.add_argument(
        "--refinitiv-comparison-dir",
        type=Path,
        default=DEFAULT_COMPARISONS["refinitiv_only"],
    )
    parser.add_argument(
        "--europe-benchmark-dir", type=Path, default=DEFAULT_BENCHMARKS["Europe"]
    )
    parser.add_argument(
        "--us-benchmark-dir", type=Path, default=DEFAULT_BENCHMARKS["US"]
    )
    parser.add_argument(
        "--europe-audit", type=Path, default=DEFAULT_AUDITS["Europe"]
    )
    parser.add_argument("--us-audit", type=Path, default=DEFAULT_AUDITS["US"])
    args = parser.parse_args()

    _set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    comparison_dirs = {
        "compustat_enriched": args.compustat_comparison_dir,
        "refinitiv_only": args.refinitiv_comparison_dir,
    }
    benchmark_dirs = {
        "Europe": args.europe_benchmark_dir,
        "US": args.us_benchmark_dir,
    }
    audit_paths = {"Europe": args.europe_audit, "US": args.us_audit}

    compustat_side = comparison_dirs["compustat_enriched"] / (
        "side_by_side_model_summary.csv"
    )

    tables: dict[str, pd.DataFrame] = {}
    tables["panel_coverage"] = panel_coverage_table(
        benchmark_dirs, compustat_side
    )
    tables["compustat_match_quality"] = compustat_match_quality_table(audit_paths)
    tables["rank_model_comparison"] = pd.concat(
        [
            rank_model_comparison_table(
                comparison_dirs[feature_set] / "side_by_side_model_summary.csv",
                feature_set,
            )
            for feature_set in ("compustat_enriched", "refinitiv_only")
        ],
        ignore_index=True,
    )
    tables["return_target_instability"] = pd.concat(
        [
            return_target_instability_table(
                comparison_dirs[feature_set] / "side_by_side_model_summary.csv",
                feature_set,
            )
            for feature_set in ("compustat_enriched", "refinitiv_only")
        ],
        ignore_index=True,
    )
    tables["return_correlations"] = return_correlation_table(
        comparison_dirs["compustat_enriched"] / "monthly_return_correlations.csv",
        "compustat_enriched",
    )

    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    report_path = write_markdown_report(args.output_dir, tables)

    figure_records = []
    for feature_set, stem in (
        ("compustat_enriched", "us_europe_rank_ic_sharpe"),
        ("refinitiv_only", "us_europe_rank_ic_sharpe_refinitiv_only"),
    ):
        side_path = comparison_dirs[feature_set] / "side_by_side_model_summary.csv"
        figure_records.append(
            build_ic_sharpe_figure(
                ic_sharpe_figure_data(side_path),
                args.figure_dir,
                stem,
                source_files=[side_path],
                feature_set_label=FEATURE_SET_LABELS[feature_set],
            )
        )

    manifest = {
        "inputs": {
            "comparison_dirs": {k: str(v) for k, v in comparison_dirs.items()},
            "benchmark_dirs": {k: str(v) for k, v in benchmark_dirs.items()},
            "audit_paths": {k: str(v) for k, v in audit_paths.items()},
        },
        "tables": {
            name: str(args.output_dir / f"{name}.csv") for name in tables
        },
        "markdown_report": str(report_path),
        "figures": [asdict(record) for record in figure_records],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    (args.figure_dir / "us_comparison_figure_manifest.json").write_text(
        json.dumps([asdict(record) for record in figure_records], indent=2)
    )

    print(f"Tables written to {args.output_dir}")
    print(f"Figures written to {args.figure_dir}")


if __name__ == "__main__":
    main()
