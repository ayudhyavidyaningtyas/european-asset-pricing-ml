"""Build the core dissertation figures requested for Chapters 3 and 5."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "asset_pricing"
RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"

DEFAULT_PANEL = DATA_ROOT / "monthly_feature_panel.parquet"
DEFAULT_ESTIMATES_COVERED_PANEL = DATA_ROOT / "monthly_feature_panel_estimates_covered.parquet"
DEFAULT_GRADIENT_DIR = RESULTS_ROOT / "capacity_gradient_tests"
DEFAULT_IMPLEMENTATION_DIR = (
    RESULTS_ROOT / "constrained_estimates_revisions_pure_strict_lag1_revision_signal_fixed"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "manuscript"
DEFAULT_COVERAGE_START = "2000-01-31"

BREADTH_MODEL_ORDER = ["ridge_rank", "hist_gbm_rank", "mlp_rank", "dre_rank"]
DEPTH_MODEL_ORDER = ["hist_gbm_rank", "mlp_rank", "dre_rank"]
MODEL_LABELS = {
    "ridge_rank": "Ridge",
    "hist_gbm_rank": "HistGBM",
    "mlp_rank": "MLP",
    "dre_rank": "DRE",
}
MODEL_COLORS = {
    "ridge_rank": "#1f77b4",
    "hist_gbm_rank": "#2ca02c",
    "mlp_rank": "#9467bd",
    "dre_rank": "#d62728",
}
MODEL_MARKERS = {
    "ridge_rank": "o",
    "hist_gbm_rank": "s",
    "mlp_rank": "^",
    "dre_rank": "D",
}
MARKET_CAP_BUCKETS = ["low_cap", "mid_cap", "high_cap", "top_500_cap"]
MARKET_CAP_BUCKET_LABELS = [
    "Smallest\ntercile",
    "Middle\ntercile",
    "Largest\ntercile",
    "Top 500",
]


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


def _percent_formatter(decimals: int = 0) -> FuncFormatter:
    def _format(value: float, _position: float | None = None) -> str:
        text = f"{value * 100:.{max(decimals, 1)}f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}%"

    return FuncFormatter(_format)


def _aum_display_label(aum_label: str) -> str:
    return f"€{aum_label}"


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
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    data_path = data_dir / f"{stem}.csv"
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


def capitalisation_concentration_data(panel_path: Path, top_n: int = 500) -> pd.DataFrame:
    panel = pd.read_parquet(
        panel_path,
        columns=["date", "ric", "company_market_cap"],
    ).dropna(subset=["company_market_cap"])
    panel = panel[pd.to_numeric(panel["company_market_cap"], errors="coerce").gt(0)].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "company_market_cap"], ascending=[True, False])
    panel["rank"] = panel.groupby("date", sort=False).cumcount().add(1)
    sample_start = panel["date"].min()
    sample_end = panel["date"].max()
    sample_stock_months = len(panel)

    by_rank = (
        panel.groupby("rank", as_index=False)
        .agg(
            rank_market_cap_eur=("company_market_cap", "sum"),
            rank_observations=("ric", "size"),
        )
        .sort_values("rank")
    )
    by_rank["cumulative_market_cap_share"] = (
        by_rank["rank_market_cap_eur"].cumsum() / by_rank["rank_market_cap_eur"].sum()
    )
    by_rank["cumulative_observation_share"] = (
        by_rank["rank_observations"].cumsum() / by_rank["rank_observations"].sum()
    )
    by_rank["top_n_marker"] = top_n
    marker = by_rank[by_rank["rank"].le(top_n)].iloc[-1]
    by_rank["sample_definition"] = "positive market-cap panel"
    by_rank["sample_date_start"] = sample_start.date().isoformat()
    by_rank["sample_date_end"] = sample_end.date().isoformat()
    by_rank["sample_stock_months"] = sample_stock_months
    by_rank["top_n_observation_share"] = marker["cumulative_observation_share"]
    by_rank["top_n_market_cap_share"] = marker["cumulative_market_cap_share"]
    return by_rank


def plot_capitalisation_concentration(
    panel_path: Path,
    output_dir: Path,
    *,
    top_n: int = 500,
) -> FigureRecord:
    data = capitalisation_concentration_data(panel_path, top_n=top_n)
    marker_row = data[data["rank"].le(top_n)].iloc[-1]
    obs_share = float(marker_row["cumulative_observation_share"])
    cap_share = float(marker_row["cumulative_market_cap_share"])

    fig, ax = plt.subplots(figsize=(9.8, 5.3))
    ax.plot(
        data["rank"],
        data["cumulative_market_cap_share"],
        color="#1f77b4",
        linewidth=2.2,
    )
    ax.axvline(top_n, color="#333333", linewidth=1.0, linestyle="--")
    ax.axhline(cap_share, color="#333333", linewidth=0.9, linestyle=":")
    ax.scatter([top_n], [cap_share], color="#d62728", s=42, zorder=4)
    ax.annotate(
        f"Top {top_n}: {obs_share:.1%} of usable stock-months\n"
        f"and {cap_share:.1%} of aggregate market cap",
        xy=(top_n, cap_share),
        xytext=(top_n + 260, max(0.62, cap_share - 0.18)),
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.0},
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    ticks = [tick for tick in [1, 500, 1000, 2000, 3000, 4000, 5000] if tick <= data["rank"].max()]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:,}" for tick in ticks])
    ax.set_xlim(1, int(data["rank"].max()))
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(_percent_formatter(0))
    ax.set_xlabel("Within-month firm rank by market capitalisation")
    ax.set_ylabel("Cumulative share of aggregate market capitalisation")
    ax.set_title("Capitalisation concentration in the European equity panel")
    ax.grid(True, linewidth=0.5, alpha=0.5)
    fig.text(
        0.5,
        -0.01,
        "Sample: positive market-cap stock-months, December 1996 to June 2026.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "capitalisation_concentration",
        source_files=[panel_path],
        description=(
            "Cumulative aggregate market-capitalisation share by within-month firm rank, "
            f"with the top {top_n} stocks flagged."
        ),
    )


def panel_coverage_data(panel_path: Path, estimates_covered_panel_path: Path) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path, columns=["date", "ric", "company_market_cap"])
    panel = panel[pd.to_numeric(panel["company_market_cap"], errors="coerce").gt(0)].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    eligible = (
        panel.groupby("date", as_index=False)["ric"]
        .nunique()
        .rename(columns={"ric": "eligible_firms"})
    )

    covered = pd.read_parquet(
        estimates_covered_panel_path,
        columns=["date", "ric", "company_market_cap"],
    )
    covered = covered[pd.to_numeric(covered["company_market_cap"], errors="coerce").gt(0)].copy()
    covered["date"] = pd.to_datetime(covered["date"])
    estimates = (
        covered.groupby("date", as_index=False)["ric"]
        .nunique()
        .rename(columns={"ric": "estimates_covered_firms"})
    )

    data = eligible.merge(estimates, on="date", how="left").sort_values("date")
    first_covered_date = estimates["date"].min()
    data.loc[data["date"].lt(first_covered_date), "estimates_covered_firms"] = np.nan
    return data


def plot_panel_coverage(
    panel_path: Path,
    estimates_covered_panel_path: Path,
    output_dir: Path,
    *,
    coverage_start: str = DEFAULT_COVERAGE_START,
) -> FigureRecord:
    data = panel_coverage_data(panel_path, estimates_covered_panel_path)
    start = pd.Timestamp(coverage_start)
    data = data[data["date"].ge(start)].copy()
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.plot(
        data["date"],
        data["eligible_firms"],
        color="#1f77b4",
        linewidth=1.9,
        label="Eligible firms with market cap",
    )
    ax.plot(
        data["date"],
        data["estimates_covered_firms"],
        color="#d62728",
        linewidth=1.9,
        label="Estimates-covered firms",
    )
    ax.set_xlim(data["date"].min(), data["date"].max())
    ax.set_ylabel("Unique RICs per month")
    ax.set_xlabel("")
    ax.set_title("Usable panel coverage from January 2000")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "panel_coverage_over_time",
        source_files=[panel_path, estimates_covered_panel_path],
        description=(
            "European panel coverage with observed market capitalisation from "
            f"{start.date().isoformat()} to June 2026, with the estimates-covered "
            "subset from January 2005."
        ),
    )


def capacity_gradient_premia_data(gradient_dir: Path) -> pd.DataFrame:
    buckets = pd.read_csv(gradient_dir / "paired_premium_by_bucket.csv")
    breadth = buckets[
        buckets["premium"].eq("flexibility_premium")
        & buckets["dimension"].eq("market_cap")
        & buckets["model"].isin(BREADTH_MODEL_ORDER)
        & buckets["bucket"].isin(MARKET_CAP_BUCKETS)
    ].copy()
    breadth["premium_label"] = "Breadth premium: IC(model) - IC(momentum)"

    depth = buckets[
        buckets["premium"].eq("depth_premium")
        & buckets["dimension"].eq("market_cap")
        & buckets["model"].isin(DEPTH_MODEL_ORDER)
        & buckets["bucket"].isin(MARKET_CAP_BUCKETS)
    ].copy()
    depth["premium_label"] = "Depth premium: IC(model) - IC(ridge)"

    data = pd.concat([breadth, depth], ignore_index=True)
    data["bucket_order"] = data["bucket"].map(MARKET_CAP_BUCKETS.index)
    data["bucket_label"] = data["bucket_order"].map(dict(enumerate(MARKET_CAP_BUCKET_LABELS)))
    data["model_label"] = data["model"].map(MODEL_LABELS)
    data["model_order"] = data["model"].map(
        {model: i for i, model in enumerate(BREADTH_MODEL_ORDER)}
    )
    return data.sort_values(["premium", "model_order", "bucket_order"]).reset_index(drop=True)


def plot_capacity_gradient_premia(gradient_dir: Path, output_dir: Path) -> FigureRecord:
    data = capacity_gradient_premia_data(gradient_dir)
    panels = [
        ("flexibility_premium", "Breadth premium\nmulti-characteristic model minus momentum"),
        ("depth_premium", "Depth premium\nflexible model minus ridge"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.0), sharey=True)
    positions = np.arange(len(MARKET_CAP_BUCKETS))

    for ax, (premium, title) in zip(axes, panels, strict=True):
        panel = data[data["premium"].eq(premium)]
        order = BREADTH_MODEL_ORDER if premium == "flexibility_premium" else DEPTH_MODEL_ORDER
        ax.axvspan(2.62, 3.38, color="#bdbdbd", alpha=0.2, zorder=0)
        for model in order:
            series = (
                panel[panel["model"].eq(model)]
                .set_index("bucket")
                .reindex(MARKET_CAP_BUCKETS)
            )
            if series.empty:
                continue
            ax.errorbar(
                positions,
                series["estimate"],
                yerr=[
                    series["estimate"] - series["ci_low"],
                    series["ci_high"] - series["estimate"],
                ],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markersize=6.0,
                linewidth=1.9,
                elinewidth=1.0,
                capsize=2.5,
                label=MODEL_LABELS[model],
            )
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_title(title)
        ax.set_xticks(positions)
        ax.set_xticklabels(MARKET_CAP_BUCKET_LABELS)
        ax.set_xlim(-0.35, len(MARKET_CAP_BUCKETS) - 0.65)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", frameon=True)

    axes[0].set_ylabel("Paired monthly IC premium")
    axes[0].set_ylim(-0.05, 0.095)
    fig.suptitle("Tradability gradient: breadth collapses, depth does not", y=0.99)
    fig.text(
        0.5,
        -0.01,
        "Market-capitalisation buckets; paired within month and bucket; "
        "95% HAC lag-6 intervals; top-500 band shaded.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "capacity_gradient_premia",
        source_files=[
            gradient_dir / "paired_premium_by_bucket.csv",
            gradient_dir / "capacity_gradient_tests.csv",
        ],
        description=(
            "Breadth and depth IC premia by market-capitalisation tradability bucket, "
            "using paired monthly HAC intervals."
        ),
    )


def implementation_cumulative_data(implementation_dir: Path, aum_label: str = "100m") -> pd.DataFrame:
    monthly = pd.read_csv(
        implementation_dir / "benchmark_relative_monthly.csv",
        parse_dates=["target_date"],
    ).sort_values("target_date")
    net_column = f"net_return_{aum_label}"
    if net_column not in monthly.columns:
        raise ValueError(f"Missing {net_column} in benchmark_relative_monthly.csv")
    display_label = _aum_display_label(aum_label)

    frames = [
        pd.DataFrame(
            {
                "date": monthly["target_date"],
                "series": f"Constrained revision portfolio (net, {display_label})",
                "monthly_return": monthly[net_column],
            }
        ),
        pd.DataFrame(
            {
                "date": monthly["target_date"],
                "series": "EUR value-weighted benchmark",
                "monthly_return": monthly["benchmark_return_eur"],
            }
        ),
    ]
    data = pd.concat(frames, ignore_index=True).dropna(subset=["monthly_return"])
    data = data.sort_values(["series", "date"])
    data["cumulative_growth"] = data.groupby("series")["monthly_return"].transform(
        lambda returns: (1.0 + returns).cumprod()
    )
    data["aum_label"] = aum_label
    return data.reset_index(drop=True)


def plot_implementation_cumulative(
    implementation_dir: Path,
    output_dir: Path,
    *,
    aum_label: str = "100m",
) -> FigureRecord:
    data = implementation_cumulative_data(implementation_dir, aum_label=aum_label)
    fig, ax = plt.subplots(figsize=(10.3, 5.2))
    display_label = _aum_display_label(aum_label)
    palette = {
        f"Constrained revision portfolio (net, {display_label})": "#1f77b4",
        "EUR value-weighted benchmark": "#d62728",
    }
    styles = {"EUR value-weighted benchmark": "--"}
    for series, group in data.groupby("series", sort=False):
        ax.plot(
            group["date"],
            group["cumulative_growth"],
            color=palette.get(series),
            linestyle=styles.get(series, "-"),
            linewidth=1.9,
            label=series,
        )
    ax.axhline(1.0, color="black", linewidth=0.9)
    ax.set_yscale("log")
    growth_formatter = FuncFormatter(lambda value, _: f"{value:.2g}x")
    ax.yaxis.set_major_formatter(growth_formatter)
    ax.yaxis.set_minor_formatter(growth_formatter)
    ax.set_ylabel("Cumulative growth of 1 EUR (log scale)")
    ax.set_xlabel("")
    ax.set_title("Cumulative net performance against the value-weighted benchmark")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "implementation_cumulative_net_performance",
        source_files=[implementation_dir / "benchmark_relative_monthly.csv"],
        description=(
            "Cumulative net growth of the constrained revision long-only portfolio "
            "against the internal EUR value-weighted benchmark."
        ),
    )


def implementation_capacity_curve_data(implementation_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(implementation_dir / "constrained_summary.csv")
    summary = summary[summary["subperiod"].eq("full")].copy()
    relative = pd.read_csv(implementation_dir / "benchmark_relative_summary.csv")
    relative = relative[relative["subperiod"].eq("full")].copy()
    relative = relative[
        [
            "strategy",
            "constraint",
            "aum_eur",
            "annualized_benchmark_return",
            "annualized_active_return",
            "information_ratio",
            "active_hac_p_two_sided",
            "alpha_annualized",
            "alpha_p_two_sided",
        ]
    ]
    data = summary.merge(
        relative,
        on=["strategy", "constraint", "aum_eur"],
        how="left",
        validate="one_to_one",
    ).sort_values("aum_eur")
    return data.reset_index(drop=True)


def plot_implementation_capacity_curve(
    implementation_dir: Path,
    output_dir: Path,
) -> FigureRecord:
    data = implementation_capacity_curve_data(implementation_dir)
    x = np.arange(len(data))
    labels = [_aum_display_label(label) for label in data["aum_label"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    axes[0].plot(
        x,
        data["annualized_net_return"],
        marker="o",
        color="#1f77b4",
        linewidth=2.0,
        label="Net return",
    )
    axes[0].plot(
        x,
        data["annualized_active_return"],
        marker="s",
        color="#525252",
        linestyle="--",
        linewidth=1.7,
        label="Active return",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].yaxis.set_major_formatter(_percent_formatter(0))
    axes[0].set_title("Net return decays with scale")
    axes[0].set_ylabel("Annualized return")
    axes[0].legend(loc="best", frameon=True)

    axes[1].plot(
        x,
        data["net_sharpe"],
        marker="o",
        color="#2ca02c",
        linewidth=2.0,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_title("Risk-adjusted performance also declines")
    axes[1].set_ylabel("Net Sharpe")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Assets under management")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)

    fig.suptitle("Capacity curve for the constrained implementation portfolio", y=0.98)
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "implementation_capacity_curve",
        source_files=[
            implementation_dir / "constrained_summary.csv",
            implementation_dir / "benchmark_relative_summary.csv",
        ],
        description=(
            "Full-period constrained implementation performance across €10m, "
            "€100m and €500m AUM."
        ),
    )


def build_figures(args: argparse.Namespace) -> pd.DataFrame:
    _set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        plot_capitalisation_concentration(args.panel, args.output_dir, top_n=args.top_n),
        plot_panel_coverage(
            args.panel,
            args.estimates_covered_panel,
            args.output_dir,
            coverage_start=args.coverage_start,
        ),
        plot_capacity_gradient_premia(args.gradient_dir, args.output_dir),
        plot_implementation_cumulative(
            args.implementation_dir,
            args.output_dir,
            aum_label=args.aum_label,
        ),
        plot_implementation_capacity_curve(args.implementation_dir, args.output_dir),
    ]
    manifest = pd.DataFrame([record.__dict__ for record in records])
    manifest_path = args.output_dir / "core_figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (args.output_dir / "core_figure_manifest.json").write_text(
        json.dumps(manifest.to_dict(orient="records"), indent=2)
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument(
        "--estimates-covered-panel",
        type=Path,
        default=DEFAULT_ESTIMATES_COVERED_PANEL,
    )
    parser.add_argument("--gradient-dir", type=Path, default=DEFAULT_GRADIENT_DIR)
    parser.add_argument("--implementation-dir", type=Path, default=DEFAULT_IMPLEMENTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--aum-label", default="100m")
    parser.add_argument("--coverage-start", default=DEFAULT_COVERAGE_START)
    args = parser.parse_args()
    manifest = build_figures(args)
    print(
        json.dumps(
            {
                "figures": int(len(manifest)),
                "output_dir": str(args.output_dir),
                "manifest": str(args.output_dir / "core_figure_manifest.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
