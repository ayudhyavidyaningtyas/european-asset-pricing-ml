"""Build manuscript-ready figures from frozen dissertation outputs."""
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "manuscript"
DEFAULT_DRE_RUN = RESULTS_ROOT / "dre_estimates_enriched_strict_lag1_ex_ante"
DEFAULT_REVISION_RUN = (
    RESULTS_ROOT / "estimates_revisions_pure_strict_lag1_revision_signal_ridge"
)
DEFAULT_SIZE_DECOMPOSITION = (
    RESULTS_ROOT
    / "deep_size_decomposition_strict_lag1"
    / "deep_size_decomposition_summary.csv"
)
DEFAULT_FACTOR_SPANNING = (
    RESULTS_ROOT
    / "revision_strategy_final_exhibits"
    / "revision_external_factor_spanning.csv"
)
DEFAULT_ECONOMETRIC_DIR = RESULTS_ROOT / "econometric_evidence_tables"
DEFAULT_FAMA_MACBETH = (
    RESULTS_ROOT
    / "depth_estimates_revisions_pure_strict_lag1_revision_signal_ridge"
    / "fama_macbeth_summary.csv"
)
DEFAULT_MARKET_RETURN = RESULTS_ROOT / "depth_analysis" / "eur_market_return.csv"
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_LAG_TEMPLATE = "lag_sensitivity_pure_revisions_lag{lag}_ridge"

MODEL_ORDER = [
    "momentum_rank",
    "ridge_rank",
    "hist_gbm_rank",
    "mlp_rank",
    "dre_rank",
]
MODEL_LABELS = {
    "momentum_rank": "Momentum",
    "ridge_rank": "Ridge",
    "hist_gbm_rank": "HistGBM",
    "mlp_rank": "MLP",
    "dre_rank": "DRE",
}
MODEL_COLORS = {
    "momentum_rank": "#525252",
    "ridge_rank": "#1f77b4",
    "hist_gbm_rank": "#2ca02c",
    "mlp_rank": "#9467bd",
    "dre_rank": "#d62728",
}
UNIVERSE_ORDER = ["standard_ex_bottom_5pct", "ex_bottom_20pct"]
UNIVERSE_LABELS = {
    "standard_ex_bottom_5pct": "Ex bottom 5%",
    "ex_bottom_20pct": "Ex bottom 20%",
}
WEIGHTING_LABELS = {"equal": "Equal-weight", "value": "Value-weight"}
SIZE_BUCKET_ORDER = [
    "all_standard_ex_bottom_5pct",
    "small",
    "middle",
    "large",
    "top_500_by_market_cap",
]
SIZE_BUCKET_LABELS = {
    "all_standard_ex_bottom_5pct": "All",
    "small": "Small",
    "middle": "Middle",
    "large": "Large",
    "top_500_by_market_cap": "Top 500",
}
LAG_CELLS = (
    ("equal", "standard_ex_bottom_5pct"),
    ("equal", "ex_bottom_20pct"),
    ("value", "standard_ex_bottom_5pct"),
    ("value", "ex_bottom_20pct"),
)


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
    """Percent tick labels that never round a half-point tick onto a whole number.

    Matplotlib often places ticks on 2.5pp intervals; formatting those with zero
    decimals produced evenly spaced gridlines carrying unevenly spaced labels
    (0%, 2%, 5%, 8%, 10%). Keep one decimal when the tick needs it, drop it when
    it does not.
    """

    def _format(value: float, _position: float | None = None) -> str:
        text = f"{value * 100:.{max(decimals, 1)}f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}%"

    return FuncFormatter(_format)


def _percent_point_formatter() -> FuncFormatter:
    """As :func:`_percent_formatter`, for values already expressed in percent."""

    def _format(value: float, _position: float | None = None) -> str:
        text = f"{value:.1f}"
        if text.endswith(".0"):
            text = text[:-2]
        return f"{text}%"

    return FuncFormatter(_format)


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


def predictability_gap_data(model_summary_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(model_summary_path)
    data = summary[
        summary["target_mode"].eq("rank")
        & summary["portfolio"].eq("long_short")
        & summary["cost_bps"].eq(25)
        & summary["weighting"].isin(WEIGHTING_LABELS)
        & summary["universe_variant"].isin(UNIVERSE_ORDER)
        & summary["model"].isin(MODEL_ORDER)
    ].copy()
    data["model_label"] = data["model"].map(MODEL_LABELS)
    data["weighting_label"] = data["weighting"].map(WEIGHTING_LABELS)
    data["universe_label"] = data["universe_variant"].map(UNIVERSE_LABELS)
    return data.sort_values(
        by=["weighting", "universe_variant", "model"],
        key=lambda column: column.map(_model_sort_key) if column.name == "model" else column,
    )


def plot_predictability_gap(
    model_summary_path: Path,
    output_dir: Path,
) -> FigureRecord:
    data = predictability_gap_data(model_summary_path)
    if data.empty:
        raise ValueError("No rows available for predictability gap figure")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    marker_by_universe = {
        "standard_ex_bottom_5pct": "o",
        "ex_bottom_20pct": "s",
    }
    for ax, weighting in zip(axes, ["equal", "value"], strict=True):
        subset = data[data["weighting"].eq(weighting)]
        for universe in UNIVERSE_ORDER:
            cell = subset[subset["universe_variant"].eq(universe)]
            for model in MODEL_ORDER:
                row = cell[cell["model"].eq(model)]
                if row.empty:
                    continue
                # Without the bootstrap interval this panel reads as clear model
                # separation, which sharpe_significance.csv rejects: no model beats
                # momentum once the comparison is Holm-corrected.
                has_ci = {"net_sharpe_ci_low", "net_sharpe_ci_high"} <= set(row.columns)
                if has_ci and row[["net_sharpe_ci_low", "net_sharpe_ci_high"]].notna().all(axis=None):
                    ax.errorbar(
                        row["mean_monthly_spearman_ic"],
                        row["net_sharpe"],
                        yerr=np.vstack(
                            [
                                row["net_sharpe"] - row["net_sharpe_ci_low"],
                                row["net_sharpe_ci_high"] - row["net_sharpe"],
                            ]
                        ),
                        fmt="none",
                        ecolor=MODEL_COLORS[model],
                        elinewidth=1.2,
                        capsize=3,
                        alpha=0.55,
                        zorder=1,
                    )
                ax.scatter(
                    row["mean_monthly_spearman_ic"],
                    row["net_sharpe"],
                    s=58,
                    marker=marker_by_universe[universe],
                    color=MODEL_COLORS[model],
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.9,
                    zorder=3,
                )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(WEIGHTING_LABELS[weighting])
        ax.set_xlabel("Mean monthly rank IC")
        ax.xaxis.set_major_formatter(_percent_formatter(0))
        ax.grid(True, linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel("Net Sharpe, long-short, 25 bps")
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="black",
            markeredgewidth=0.4,
            label=MODEL_LABELS[model],
            markersize=6,
        )
        for model in MODEL_ORDER
    ]
    universe_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_by_universe[universe],
            color="black",
            linestyle="none",
            label=UNIVERSE_LABELS[universe],
            markersize=6,
        )
        for universe in UNIVERSE_ORDER
    ]
    axes[1].legend(
        handles=[*model_handles, *universe_handles],
        loc="best",
        frameon=True,
    )
    fig.suptitle("Predictability and implementability do not move one-for-one")
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "predictability_implementability_gap",
        source_files=[model_summary_path],
        description=(
            "DRE-run model ICs plotted against net Sharpe across weighting and "
            "universe cells, with stationary-bootstrap 95% intervals on Sharpe. "
            "The intervals overlap heavily: no model separates from momentum once "
            "the paired comparison is Holm-corrected."
        ),
    )


def deep_size_data(size_summary_path: Path) -> pd.DataFrame:
    data = pd.read_csv(size_summary_path)
    data = data[
        data["model"].isin(MODEL_ORDER)
        & data["size_bucket"].isin(SIZE_BUCKET_ORDER)
    ].copy()
    data["model_label"] = data["model"].map(MODEL_LABELS)
    data["size_bucket_label"] = data["size_bucket"].map(SIZE_BUCKET_LABELS)
    data["size_bucket_order"] = data["size_bucket"].map(
        {bucket: index for index, bucket in enumerate(SIZE_BUCKET_ORDER)}
    )
    return data.sort_values(["size_bucket_order", "model"])


def plot_deep_size_decomposition(
    size_summary_path: Path,
    output_dir: Path,
) -> FigureRecord:
    data = deep_size_data(size_summary_path)
    if data.empty:
        raise ValueError("No rows available for deep size-decomposition figure")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    metrics = [
        ("mean_spearman_ic", "Mean monthly rank IC", True),
        ("net_long_short_sharpe", "Net long-short Sharpe", False),
    ]
    # Small/Middle/Large partition the universe, but "All" spans everything and
    # "Top 500" is a subset of "Large". A connecting line would imply an ordered
    # partition that does not exist, so these are grouped bars with the two
    # overlapping aggregates separated off to the right.
    buckets = [
        bucket for bucket in SIZE_BUCKET_ORDER if bucket in set(data["size_bucket"])
    ]
    partition = [b for b in buckets if b not in {"all_standard_ex_bottom_5pct"}]
    partition = [b for b in partition if b != "top_500_by_market_cap"]
    aggregates = [b for b in buckets if b not in partition]
    ordered = partition + aggregates
    positions = {bucket: index for index, bucket in enumerate(ordered)}
    models = [model for model in MODEL_ORDER if model in set(data["model"])]
    width = 0.8 / max(len(models), 1)
    for ax, (column, ylabel, is_percent) in zip(axes, metrics, strict=True):
        for offset, model in enumerate(models):
            subset = data[data["model"].eq(model)]
            centres = [
                positions[bucket] + (offset - (len(models) - 1) / 2) * width
                for bucket in subset["size_bucket"]
                if bucket in positions
            ]
            heights = [
                value
                for bucket, value in zip(subset["size_bucket"], subset[column], strict=True)
                if bucket in positions
            ]
            ax.bar(
                centres,
                heights,
                width=width,
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
                edgecolor="black",
                linewidth=0.3,
            )
        if aggregates:
            ax.axvline(len(partition) - 0.5, color="#888888", linewidth=1.0, linestyle=":")
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels([SIZE_BUCKET_LABELS[bucket] for bucket in ordered])
        ax.set_ylabel(ylabel)
        if is_percent:
            ax.yaxis.set_major_formatter(_percent_formatter(0))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)
        ax.grid(False, axis="x")
    axes[1].legend(loc="best", frameon=True, ncols=2)
    fig.suptitle(
        "Forecast strength concentrates in small and mid caps, not the scalable bucket\n"
        "Left of the dotted line partitions the universe; All and Top 500 overlap it"
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "deep_size_decomposition",
        source_files=[size_summary_path],
        description=(
            "Size-bucket IC and net Sharpe for momentum, ML baselines, and DRE. "
            "Small/Middle/Large partition the universe; All and Top 500 overlap it "
            "and are separated by the dotted rule. Net returns apply a flat cost in "
            "bps of turnover, which understates market impact in the smaller buckets."
        ),
    )


def revision_factor_alpha_data(spanning_path: Path) -> pd.DataFrame:
    spanning = pd.read_csv(spanning_path)
    data = spanning[
        spanning["comparison"].eq("absolute")
        & spanning["model"].eq("ridge_rank")
        & spanning["portfolio"].eq("long_short")
        & spanning["cost_bps"].eq(25)
        & spanning["weighting"].isin(WEIGHTING_LABELS)
        & spanning["universe_variant"].isin(UNIVERSE_ORDER)
    ].copy()
    data["spec_label"] = (
        data["weighting"].map({"equal": "EW", "value": "VW"})
        + "\n"
        + data["universe_variant"].map(UNIVERSE_LABELS)
    )
    data["alpha_pct"] = data["alpha_annualized"] * 100.0
    data["alpha_se_pct"] = np.where(
        data["alpha_t"].abs().gt(1e-12),
        (data["alpha_annualized"].abs() / data["alpha_t"].abs()) * 100.0,
        np.nan,
    )
    data["alpha_ci95_low_pct"] = data["alpha_pct"] - 1.96 * data["alpha_se_pct"]
    data["alpha_ci95_high_pct"] = data["alpha_pct"] + 1.96 * data["alpha_se_pct"]
    data["sort_weighting"] = data["weighting"].map({"equal": 0, "value": 1})
    data["sort_universe"] = data["universe_variant"].map(
        {name: index for index, name in enumerate(UNIVERSE_ORDER)}
    )
    return data.sort_values(["sort_weighting", "sort_universe"])


def plot_revision_factor_alpha(
    spanning_path: Path,
    output_dir: Path,
) -> FigureRecord:
    data = revision_factor_alpha_data(spanning_path)
    if data.empty:
        raise ValueError("No rows available for revision factor-alpha figure")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(data))
    colors = data["weighting"].map({"equal": "#1f77b4", "value": "#ff7f0e"})
    yerr = np.vstack(
        [
            data["alpha_pct"] - data["alpha_ci95_low_pct"],
            data["alpha_ci95_high_pct"] - data["alpha_pct"],
        ]
    )
    ax.bar(
        x,
        data["alpha_pct"],
        yerr=yerr,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.86,
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Annualized FF5+WML alpha")
    ax.yaxis.set_major_formatter(_percent_point_formatter())
    ax.set_xticks(x, data["spec_label"])
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)
    # The style's vertical gridlines run straight through the t-statistic labels.
    ax.grid(False, axis="x")
    # Offset the t-labels beyond the interval whisker rather than on top of it.
    for xpos, row in enumerate(data.itertuples()):
        above = row.alpha_pct >= 0
        anchor = row.alpha_ci95_high_pct if above else row.alpha_ci95_low_pct
        ax.annotate(
            f"t={row.alpha_t:.2f}",
            (xpos, anchor),
            textcoords="offset points",
            xytext=(0, 7 if above else -7),
            ha="center",
            va="bottom" if above else "top",
            fontsize=8,
        )
    # These are four separate portfolios, not a progression: a connecting line
    # across equal- and value-weighting would imply a trend that does not exist.
    ax2 = ax.twinx()
    ax2.scatter(
        x,
        data["beta_WML"],
        color="#525252",
        marker="D",
        s=46,
        zorder=4,
        label="WML beta",
    )
    for xpos, beta in zip(x, data["beta_WML"], strict=True):
        ax2.annotate(
            f"{beta:.2f}",
            (xpos, beta),
            textcoords="offset points",
            xytext=(11, 0),
            va="center",
            fontsize=7.5,
            color="#525252",
        )
    ax2.set_ylabel("WML beta")
    ax2.grid(False)
    handles = [
        Line2D([0], [0], color="#1f77b4", marker="s", linestyle="none", label="Equal-weight"),
        Line2D([0], [0], color="#ff7f0e", marker="s", linestyle="none", label="Value-weight"),
        Line2D(
            [0],
            [0],
            color="#525252",
            marker="D",
            linestyle="none",
            label="WML beta (right axis)",
        ),
    ]
    # Both upper right and lower left collide with WML beta markers, so the legend
    # goes below the axes where nothing competes with it.
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncols=3,
        frameon=False,
    )
    fig.suptitle("Revision alpha survives under equal-weighting but is spanned under value-weighting")
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "revision_factor_alpha",
        source_files=[spanning_path],
        description="Revision long-short FF5+WML alpha with approximate 95% t-statistic intervals and WML beta.",
    )


def _summary_row(
    summary: pd.DataFrame,
    *,
    weighting: str,
    universe_variant: str,
    portfolio: str,
) -> pd.Series:
    subset = summary[
        summary["model"].eq("ridge_rank")
        & summary["weighting"].eq(weighting)
        & summary["universe_variant"].eq(universe_variant)
        & summary["portfolio"].eq(portfolio)
        & summary["cost_bps"].eq(25)
    ]
    if subset.empty:
        raise ValueError(
            f"Missing revision summary row for {weighting}/{universe_variant}/{portfolio}"
        )
    return subset.iloc[0]


def _portfolio_ci_row(
    portfolio_cis: pd.DataFrame,
    *,
    aum_label: str,
    metric: str,
) -> pd.Series:
    subset = portfolio_cis[
        portfolio_cis["object_class"].eq("constrained_long_only")
        & portfolio_cis["strategy"].eq(
            "fixed_pure_revision_signal_smooth75_ridge_top500_observed"
        )
        & portfolio_cis["portfolio"].eq("long_only")
        & portfolio_cis["aum_label"].eq(aum_label)
        & portfolio_cis["metric"].eq(metric)
    ]
    if subset.empty:
        raise ValueError(f"Missing constrained CI row for {aum_label}/{metric}")
    return subset.iloc[0]


def implementability_ladder_data(
    revision_summary_path: Path,
    portfolio_cis_path: Path,
) -> pd.DataFrame:
    summary = pd.read_csv(revision_summary_path)
    portfolio_cis = pd.read_csv(portfolio_cis_path)
    records: list[dict[str, float | str]] = []
    unconstrained_specs = [
        (
            "EW long-short",
            "Unconstrained",
            "equal",
            "standard_ex_bottom_5pct",
            "long_short",
        ),
        (
            "VW long-short",
            "Value-weighted",
            "value",
            "standard_ex_bottom_5pct",
            "long_short",
        ),
    ]
    for label, stage, weighting, universe, portfolio in unconstrained_specs:
        row = _summary_row(
            summary,
            weighting=weighting,
            universe_variant=universe,
            portfolio=portfolio,
        )
        records.append(
            {
                "label": label,
                "stage": stage,
                "annualized_net_return": row["annualized_net_mean_return"],
                "annualized_net_return_ci_low": row["annualized_net_mean_return_ci_low"],
                "annualized_net_return_ci_high": row["annualized_net_mean_return_ci_high"],
                "net_sharpe": row["net_sharpe"],
                "net_sharpe_ci_low": row["net_sharpe_ci_low"],
                "net_sharpe_ci_high": row["net_sharpe_ci_high"],
            }
        )
    for aum_label in ["10m", "100m", "500m"]:
        ret = _portfolio_ci_row(
            portfolio_cis,
            aum_label=aum_label,
            metric="annualized_net_return",
        )
        sharpe = _portfolio_ci_row(
            portfolio_cis,
            aum_label=aum_label,
            metric="net_sharpe",
        )
        records.append(
            {
                "label": f"Constrained {aum_label}",
                "stage": "Constrained long-only",
                "annualized_net_return": ret["point"],
                "annualized_net_return_ci_low": ret["ci_low"],
                "annualized_net_return_ci_high": ret["ci_high"],
                "net_sharpe": sharpe["point"],
                "net_sharpe_ci_low": sharpe["ci_low"],
                "net_sharpe_ci_high": sharpe["ci_high"],
            }
        )
    data = pd.DataFrame(records)
    data["order"] = np.arange(len(data))
    return data


def plot_implementability_ladder(
    revision_summary_path: Path,
    portfolio_cis_path: Path,
    output_dir: Path,
) -> FigureRecord:
    data = implementability_ladder_data(revision_summary_path, portfolio_cis_path)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    x = np.arange(len(data))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#2ca02c", "#2ca02c"]
    metrics = [
        (
            "annualized_net_return",
            "annualized_net_return_ci_low",
            "annualized_net_return_ci_high",
            "Annualized net return",
            True,
        ),
        (
            "net_sharpe",
            "net_sharpe_ci_low",
            "net_sharpe_ci_high",
            "Net Sharpe",
            False,
        ),
    ]
    for ax, (value_col, low_col, high_col, ylabel, is_percent) in zip(
        axes,
        metrics,
        strict=True,
    ):
        values = data[value_col].astype(float)
        yerr = np.vstack([values - data[low_col], data[high_col] - values])
        ax.bar(
            x,
            values,
            yerr=yerr,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.86,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        if is_percent:
            ax.yaxis.set_major_formatter(_percent_formatter(0))
        ax.set_xticks(x, data["label"], rotation=25, ha="right")
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)
    # Net return does not narrow monotonically -- constrained 10m slightly exceeds
    # the unconstrained long-short. Sharpe is what degrades as constraints bind.
    fig.suptitle("Risk-adjusted performance, not raw return, degrades as constraints bind")
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "revision_implementability_ladder",
        source_files=[revision_summary_path, portfolio_cis_path],
        description="Revision strategy net-return and Sharpe ladder from unconstrained long-short to constrained long-only AUM levels.",
    )


def lag_decay_data(
    results_root: Path,
    run_template: str,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for lag in lags:
        run_dir = results_root / run_template.format(lag=lag)
        summary_path = run_dir / "model_summary.csv"
        summary = pd.read_csv(summary_path)
        base = summary[
            summary["model"].eq("ridge_rank")
            & summary["target_mode"].eq("rank")
            & summary["cost_bps"].eq(25)
        ].copy()
        if base.empty:
            raise ValueError(f"No lag-summary rows in {summary_path}")
        ic_value = float(base["mean_monthly_spearman_ic"].dropna().iloc[0])
        records.append(
            {
                "lag": lag,
                "metric": "monthly_ic",
                "weighting": "",
                "universe_variant": "",
                "portfolio": "",
                "cell_label": "IC",
                "value": ic_value,
            }
        )
        for weighting, universe in LAG_CELLS:
            for portfolio in ["long_short", "long_only_top_decile"]:
                subset = base[
                    base["weighting"].eq(weighting)
                    & base["universe_variant"].eq(universe)
                    & base["portfolio"].eq(portfolio)
                ]
                if subset.empty:
                    continue
                records.append(
                    {
                        "lag": lag,
                        "metric": (
                            "long_short_net_return"
                            if portfolio == "long_short"
                            else "long_only_net_return"
                        ),
                        "weighting": weighting,
                        "universe_variant": universe,
                        "portfolio": portfolio,
                        "cell_label": (
                            f"{weighting.upper()[:2]} "
                            f"{UNIVERSE_LABELS[universe].replace('Ex ', 'ex ')}"
                        ),
                        "value": float(subset.iloc[0]["annualized_net_mean_return"]),
                    }
                )
    return pd.DataFrame(records)


def lag_significance(paired_tests_path: Path) -> pd.DataFrame:
    """Holm-significant adjacent-lag contrasts, keyed to the figure's panels."""
    if not paired_tests_path.exists():
        return pd.DataFrame()
    tests = pd.read_csv(paired_tests_path)
    metric_by_family = {
        "monthly_ic": "monthly_ic",
        "net_return_long_short": "long_short_net_return",
        "net_return_long_only": "long_only_net_return",
    }
    tests = tests[tests["comparison"].eq("lag1_minus_lag2")].copy()
    tests["metric"] = tests["test_family"].map(metric_by_family)
    tests["significant"] = tests["p_two_sided_holm"].lt(0.05)
    return tests[
        [
            "metric",
            "comparison",
            "weighting",
            "universe_variant",
            "portfolio",
            "t_stat",
            "p_two_sided_holm",
            "significant",
        ]
    ]


def plot_lag_decay(
    results_root: Path,
    run_template: str,
    output_dir: Path,
    lags: tuple[int, ...] = (1, 2, 3),
    paired_tests_path: Path | None = None,
) -> FigureRecord:
    data = lag_decay_data(results_root, run_template, lags)
    significance = (
        lag_significance(paired_tests_path)
        if paired_tests_path is not None
        else pd.DataFrame()
    )
    significant_cells = set()
    if not significance.empty:
        significant_cells = {
            (row.metric, row.weighting, row.universe_variant)
            for row in significance[significance["significant"]].itertuples()
        }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    panels = [
        ("monthly_ic", "Rank IC", True),
        ("long_short_net_return", "Long-short annualized net return", True),
        ("long_only_net_return", "Long-only annualized net return", True),
    ]
    line_styles = {
        "EQ ex bottom 5%": ("#1f77b4", "o"),
        "EQ ex bottom 20%": ("#2ca02c", "s"),
        "VA ex bottom 5%": ("#ff7f0e", "D"),
        "VA ex bottom 20%": ("#9467bd", "^"),
    }
    for ax, (metric, title, is_percent) in zip(axes, panels, strict=True):
        subset = data[data["metric"].eq(metric)]
        if metric == "monthly_ic":
            ordered = subset.sort_values("lag")
            # The IC contrast carries no weighting/universe keys, so it is looked up
            # on its own. It is not Holm-significant, and must not be drawn solid
            # while the legend reserves solid for significant series.
            ic_significant = False
            if not significance.empty:
                ic_rows = significance[significance["metric"].eq("monthly_ic")]
                ic_significant = bool(ic_rows["significant"].any())
            ax.plot(
                ordered["lag"],
                ordered["value"],
                marker="o",
                linewidth=2.4 if ic_significant else 1.3,
                linestyle="-" if ic_significant else "--",
                alpha=1.0 if ic_significant else 0.65,
                color="#525252",
                label="Rank IC*" if ic_significant else "Rank IC",
            )
            ax.legend(loc="best", frameon=True)
        else:
            for label, group in subset.groupby("cell_label", sort=True):
                color, marker = line_styles.get(label, ("#525252", "o"))
                ordered = group.sort_values("lag")
                cell = ordered.iloc[0]
                is_significant = (
                    metric,
                    cell["weighting"],
                    cell["universe_variant"],
                ) in significant_cells
                ax.plot(
                    ordered["lag"],
                    ordered["value"],
                    marker=marker,
                    linewidth=2.4 if is_significant else 1.3,
                    linestyle="-" if is_significant else "--",
                    alpha=1.0 if is_significant else 0.65,
                    label=f"{label}*" if is_significant else label,
                    color=color,
                )
        ax.set_title(title)
        ax.set_xlabel("Revision signal lag, months")
        ax.set_xticks(list(lags))
        ax.axhline(0.0, color="black", linewidth=0.8)
        if is_percent:
            ax.yaxis.set_major_formatter(_percent_formatter(0))
        ax.grid(True, linewidth=0.5, alpha=0.5)
    axes[1].legend(loc="best", frameon=True)
    axes[2].legend(loc="best", frameon=True)
    # Only 2 of 27 paired contrasts survive Holm, both long-only lag1 vs lag2, so
    # the earlier "information weakens with lag" headline overstated the evidence.
    fig.suptitle(
        "Only the long-only lag-1 vs lag-2 decay is statistically significant\n"
        "Solid lines with * are Holm-significant at 5%; dashed contrasts are not"
    )
    fig.tight_layout()
    source_files = [
        results_root / run_template.format(lag=lag) / "model_summary.csv"
        for lag in lags
    ]
    if paired_tests_path is not None:
        source_files.append(paired_tests_path)
    return _save_figure(
        fig,
        data,
        output_dir,
        "revision_lag_decay",
        source_files=source_files,
        description=(
            "Lag-1/2/3 point estimates for revision IC and net portfolio returns "
            "across tested cells. Solid starred series are the only adjacent-lag "
            "contrasts significant at 5% after Holm correction (2 of 27)."
        ),
    )


def _net_return(
    gross: pd.Series,
    turnover: pd.Series,
    cost_bps: int,
) -> pd.Series:
    """Repo convention: net = gross - turnover * cost_bps / 10_000."""
    return gross.sub(turnover.mul(cost_bps / 10_000.0))


def cumulative_performance_data(
    revision_run: Path,
    dre_run: Path,
    market_path: Path,
    *,
    cost_bps: int = 25,
    weighting: str = "equal",
    universe_variant: str = "standard_ex_bottom_5pct",
) -> pd.DataFrame:
    """Monthly net returns for the revision strategy, momentum, and the EUR market."""
    revision = pd.read_csv(
        revision_run / "monthly_portfolios.csv",
        parse_dates=["return_date"],
    )
    cell = revision[
        revision["weighting"].eq(weighting)
        & revision["universe_variant"].eq(universe_variant)
    ].sort_values("return_date")
    frames = [
        pd.DataFrame(
            {
                "date": cell["return_date"].to_numpy(),
                "series": "Revision long-short (net)",
                "monthly_return": _net_return(
                    cell["gross_long_short_return"],
                    cell["long_short_turnover"],
                    cost_bps,
                ).to_numpy(),
            }
        ),
        pd.DataFrame(
            {
                "date": cell["return_date"].to_numpy(),
                "series": "Revision long-only decile (net)",
                "monthly_return": _net_return(
                    cell["long_return"],
                    cell["long_only_turnover"],
                    cost_bps,
                ).to_numpy(),
            }
        ),
    ]
    momentum_path = dre_run / "monthly_portfolios.csv"
    if momentum_path.exists():
        momentum = pd.read_csv(momentum_path, parse_dates=["return_date"])
        momentum = momentum[
            momentum["model"].eq("momentum_rank")
            & momentum["weighting"].eq(weighting)
            & momentum["universe_variant"].eq(universe_variant)
        ].sort_values("return_date")
        if not momentum.empty:
            frames.append(
                pd.DataFrame(
                    {
                        "date": momentum["return_date"].to_numpy(),
                        "series": "Momentum long-short (net, full panel)",
                        "monthly_return": _net_return(
                            momentum["gross_long_short_return"],
                            momentum["long_short_turnover"],
                            cost_bps,
                        ).to_numpy(),
                    }
                )
            )
    if market_path.exists():
        market = pd.read_csv(market_path, parse_dates=["date"])
        window = market["date"].between(
            cell["return_date"].min(),
            cell["return_date"].max(),
        )
        market = market[window].sort_values("date")
        frames.append(
            pd.DataFrame(
                {
                    "date": market["date"].to_numpy(),
                    "series": "EUR market (cap-weighted)",
                    "monthly_return": market["market_return_eur"].to_numpy(),
                }
            )
        )
    data = pd.concat(frames, ignore_index=True).dropna(subset=["monthly_return"])
    data = data.sort_values(["series", "date"])
    data["cumulative_growth"] = data.groupby("series")["monthly_return"].transform(
        lambda returns: (1.0 + returns).cumprod()
    )
    data["cost_bps"] = cost_bps
    data["weighting"] = weighting
    data["universe_variant"] = universe_variant
    return data.reset_index(drop=True)


def plot_cumulative_performance(
    revision_run: Path,
    dre_run: Path,
    market_path: Path,
    output_dir: Path,
    *,
    cost_bps: int = 25,
) -> FigureRecord:
    data = cumulative_performance_data(
        revision_run,
        dre_run,
        market_path,
        cost_bps=cost_bps,
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    palette = {
        "Revision long-short (net)": "#1f77b4",
        "Revision long-only decile (net)": "#2ca02c",
        "Momentum long-short (net, full panel)": "#7f7f7f",
        "EUR market (cap-weighted)": "#d62728",
    }
    styles = {
        "Momentum long-short (net, full panel)": "--",
        "EUR market (cap-weighted)": ":",
    }
    for series, group in data.groupby("series", sort=False):
        ax.plot(
            group["date"],
            group["cumulative_growth"],
            label=series,
            color=palette.get(series),
            linestyle=styles.get(series, "-"),
            linewidth=1.7,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative growth of 1 EUR (log scale)")
    ax.set_xlabel("")
    ax.axhline(1.0, color="black", linewidth=0.9)
    # A 1x-7x log range labels most of its ticks as minors; without an explicit
    # minor formatter matplotlib mixes "1x" with scientific notation like "6 x 10^0".
    growth_formatter = FuncFormatter(lambda value, _: f"{value:.3g}x")
    ax.yaxis.set_major_formatter(growth_formatter)
    ax.yaxis.set_minor_formatter(growth_formatter)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title(
        f"Cumulative net performance, equal-weight, {cost_bps} bps "
        "(momentum shown on the broader full-panel universe)"
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "cumulative_net_performance",
        source_files=[
            revision_run / "monthly_portfolios.csv",
            dre_run / "monthly_portfolios.csv",
            market_path,
        ],
        description=(
            "Cumulative net growth for the revision long-short and long-only decile "
            "against momentum and the EUR market. Momentum is estimated on the "
            "unfiltered panel and is therefore a broader-universe reference, not a "
            "like-for-like sample match."
        ),
    )


def fama_macbeth_forest_data(fama_macbeth_path: Path) -> pd.DataFrame:
    """Annualized Fama-MacBeth slopes with HAC intervals.

    ``ci_low``/``ci_high`` are stored on the monthly scale while
    ``annualized_score_slope`` is annualized, so the interval is rescaled by 12
    to keep the point estimate and its interval on one axis.
    """
    summary = pd.read_csv(fama_macbeth_path)
    order = ["univariate", "characteristics", "characteristics_risk_country_sector"]
    labels = {
        "univariate": "Univariate",
        "characteristics": "+ momentum, size, book-to-market",
        "characteristics_risk_country_sector": (
            "+ beta, idio. vol, country & sector FE"
        ),
    }
    data = summary[summary["specification"].isin(order)].copy()
    data["order"] = data["specification"].map(order.index)
    data = data.sort_values("order")
    data["label"] = data["specification"].map(labels)
    data["annualized_ci_low"] = data["ci_low"].mul(12.0)
    data["annualized_ci_high"] = data["ci_high"].mul(12.0)
    return data[
        [
            "model",
            "specification",
            "label",
            "annualized_score_slope",
            "annualized_ci_low",
            "annualized_ci_high",
            "t_stat",
            "p_value_holm",
            "months",
            "average_cross_section",
        ]
    ].reset_index(drop=True)


def plot_fama_macbeth_forest(
    fama_macbeth_path: Path,
    output_dir: Path,
) -> FigureRecord:
    data = fama_macbeth_forest_data(fama_macbeth_path)
    fig, ax = plt.subplots(figsize=(9.5, 0.85 * len(data) + 1.5))
    positions = np.arange(len(data))[::-1]
    errors = np.vstack(
        [
            data["annualized_score_slope"] - data["annualized_ci_low"],
            data["annualized_ci_high"] - data["annualized_score_slope"],
        ]
    )
    ax.errorbar(
        data["annualized_score_slope"],
        positions,
        xerr=errors,
        fmt="o",
        color="#1f77b4",
        ecolor="#444444",
        elinewidth=1.4,
        capsize=4,
        markersize=7,
    )
    for position, row in zip(positions, data.itertuples(), strict=True):
        ax.annotate(
            f"t={row.t_stat:.2f}",
            (row.annualized_ci_high, position),
            textcoords="offset points",
            xytext=(8, 0),
            va="center",
            fontsize=8.5,
        )
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(data["label"])
    ax.set_ylim(-0.6, len(data) - 0.4)
    ax.set_xlim(left=0.0)
    ax.xaxis.set_major_formatter(_percent_formatter(0))
    ax.set_xlabel("Annualized Fama-MacBeth slope on the revision score")
    ax.set_title(
        "Revision slope survives the momentum, size and risk controls"
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "fama_macbeth_forest",
        source_files=[fama_macbeth_path],
        description=(
            "Annualized monthly Fama-MacBeth slopes on the revision score across "
            "three control specifications, with HAC intervals rescaled from the "
            "stored monthly bounds."
        ),
    )


def _monthly_ic(predictions_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        predictions_path,
        columns=["date", "model", "prediction", "target_return_rank"],
    ).dropna(subset=["prediction", "target_return_rank"])
    monthly = (
        frame.groupby(["model", "date"])
        .apply(
            lambda month: month["prediction"].corr(
                month["target_return_rank"],
                method="spearman",
            ),
            include_groups=False,
        )
        .rename("monthly_ic")
        .reset_index()
    )
    return monthly


def rolling_ic_data(
    revision_run: Path,
    dre_run: Path,
    *,
    window: int = 36,
) -> pd.DataFrame:
    frames = []
    revision = _monthly_ic(revision_run / "predictions.parquet")
    revision["panel"] = "Revision signal (pure revisions, strict lag 1)"
    frames.append(revision)
    dre_predictions = dre_run / "predictions.parquet"
    if dre_predictions.exists():
        enriched = _monthly_ic(dre_predictions)
        enriched["panel"] = "Estimates-enriched panel (strict lag 1)"
        frames.append(enriched)
    data = pd.concat(frames, ignore_index=True).sort_values(["panel", "model", "date"])
    data["rolling_ic"] = data.groupby(["panel", "model"])["monthly_ic"].transform(
        lambda series: series.rolling(window, min_periods=window).mean()
    )
    data["window_months"] = window
    return data.reset_index(drop=True)


def plot_rolling_ic(
    revision_run: Path,
    dre_run: Path,
    output_dir: Path,
    *,
    window: int = 36,
) -> FigureRecord:
    data = rolling_ic_data(revision_run, dre_run, window=window)
    panels = list(dict.fromkeys(data["panel"]))
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6.0 * len(panels), 4.4),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, panel in zip(axes, panels, strict=False):
        subset = data[data["panel"].eq(panel)]
        models = sorted(subset["model"].unique(), key=_model_sort_key)
        for model in models:
            group = subset[subset["model"].eq(model)].dropna(subset=["rolling_ic"])
            if group.empty:
                continue
            ax.plot(
                group["date"],
                group["rolling_ic"],
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model),
                linewidth=1.6,
            )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.yaxis.set_major_formatter(_percent_formatter(0))
        ax.set_title(panel)
        ax.legend(loc="best", frameon=True)
    axes[0].set_ylabel(f"Rolling {window}-month mean rank IC")
    fig.suptitle(
        f"Rolling {window}-month rank IC: is the edge concentrated in a subperiod?"
    )
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "rolling_information_coefficient",
        source_files=[
            revision_run / "predictions.parquet",
            dre_run / "predictions.parquet",
        ],
        description=(
            f"Rolling {window}-month mean monthly Spearman rank IC by model, for the "
            "pure-revision run and the estimates-enriched panel."
        ),
    )


def _sharpe_from_monthly(returns: pd.Series) -> float:
    clean = returns.replace([np.inf, -np.inf], np.nan).dropna()
    volatility = clean.std(ddof=1)
    if len(clean) < 3 or volatility <= 0:
        return np.nan
    return float(clean.mean() / volatility * np.sqrt(12))


def _derived_cost_curve(
    run: Path,
    costs: Iterable[int],
    *,
    weighting: str,
    universe_variant: str,
    risk_free: pd.Series | None,
) -> pd.DataFrame:
    """Net Sharpe across a cost grid, recomputed from saved monthly portfolios.

    Runs are not guaranteed to have been executed over the whole cost grid, so the
    curve is rebuilt from gross returns and turnover. This reproduces the published
    ``net_sharpe`` exactly at the costs a run does store. Following the pipeline, the
    risk-free rate is netted off only for the long-only leg; the long-short spread is
    self-financing.
    """
    portfolios_path = run / "monthly_portfolios.csv"
    if not portfolios_path.exists():
        return pd.DataFrame()
    portfolios = pd.read_csv(portfolios_path, parse_dates=["return_date"])
    cell = portfolios[
        portfolios["weighting"].eq(weighting)
        & portfolios["universe_variant"].eq(universe_variant)
    ]
    records = []
    for model, group in cell.groupby("model", sort=False):
        group = group.sort_values("return_date").set_index("return_date")
        for cost_bps in costs:
            cost = cost_bps / 10_000.0
            long_short = _net_return(
                group["gross_long_short_return"],
                group["long_short_turnover"],
                cost_bps,
            )
            long_only = _net_return(
                group["long_return"],
                group["long_only_turnover"],
                cost_bps,
            )
            if risk_free is not None:
                long_only = long_only.sub(risk_free.reindex(long_only.index)).dropna()
            records.extend(
                [
                    {
                        "model": model,
                        "portfolio": "long_short",
                        "cost_bps": cost_bps,
                        "net_sharpe": _sharpe_from_monthly(long_short),
                        "annualized_net_mean_return": float(long_short.mean() * 12),
                        "months": int(long_short.notna().sum()),
                    },
                    {
                        "model": model,
                        "portfolio": "long_only_top_decile",
                        "cost_bps": cost_bps,
                        "net_sharpe": _sharpe_from_monthly(long_only),
                        "annualized_net_mean_return": float(long_only.mean() * 12),
                        "months": int(long_only.notna().sum()),
                    },
                ]
            )
            del cost
    frame = pd.DataFrame(records)
    frame["weighting"] = weighting
    frame["universe_variant"] = universe_variant
    return frame


def cost_sensitivity_data(
    dre_run: Path,
    revision_run: Path,
    *,
    weighting: str = "equal",
    universe_variant: str = "standard_ex_bottom_5pct",
    costs: tuple[int, ...] = (0, 10, 25, 50),
    eur_rate_path: Path | None = None,
) -> pd.DataFrame:
    risk_free = None
    if eur_rate_path is not None and eur_rate_path.exists():
        import sys

        source_dir = str(PROJECT_ROOT / "src")
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        from asset_pricing_depth import load_eur_short_rate

        risk_free = load_eur_short_rate(eur_rate_path)
    frames = []
    for run, panel in (
        (dre_run, "Estimates-enriched panel"),
        (revision_run, "Revision signal"),
    ):
        curve = _derived_cost_curve(
            run,
            costs,
            weighting=weighting,
            universe_variant=universe_variant,
            risk_free=risk_free,
        )
        if curve.empty:
            continue
        curve["panel"] = panel
        frames.append(curve)
    data = pd.concat(frames, ignore_index=True)
    return data[
        [
            "panel",
            "model",
            "portfolio",
            "weighting",
            "universe_variant",
            "cost_bps",
            "net_sharpe",
            "annualized_net_mean_return",
            "months",
        ]
    ].sort_values(["panel", "portfolio", "model", "cost_bps"])


def plot_cost_sensitivity(
    dre_run: Path,
    revision_run: Path,
    output_dir: Path,
    *,
    eur_rate_path: Path | None = None,
) -> FigureRecord:
    data = cost_sensitivity_data(
        dre_run,
        revision_run,
        eur_rate_path=eur_rate_path,
    )
    portfolios = ["long_short", "long_only_top_decile"]
    titles = {
        "long_short": "Long-short decile spread",
        "long_only_top_decile": "Long-only top decile",
    }
    fig, axes = plt.subplots(1, len(portfolios), figsize=(11.5, 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, portfolio in zip(axes, portfolios, strict=False):
        subset = data[data["portfolio"].eq(portfolio)]
        for (panel, model), group in subset.groupby(["panel", "model"], sort=False):
            group = group.sort_values("cost_bps")
            if group.empty:
                continue
            is_revision = panel == "Revision signal"
            ax.plot(
                group["cost_bps"],
                group["net_sharpe"],
                marker="o",
                markersize=4.5,
                linewidth=1.9 if is_revision else 1.4,
                linestyle="-" if is_revision else "--",
                color="#000000" if is_revision else MODEL_COLORS.get(model),
                label=(
                    "Revision ridge"
                    if is_revision
                    else MODEL_LABELS.get(model, model)
                ),
            )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(titles[portfolio])
        ax.set_xlabel("Transaction cost, bps of turnover")
        ax.set_xticks(sorted(data["cost_bps"].unique()))
    axes[0].set_ylabel("Net Sharpe")
    axes[-1].legend(loc="best", frameon=True)
    fig.suptitle("Net Sharpe degrades with trading cost")
    fig.tight_layout()
    return _save_figure(
        fig,
        data,
        output_dir,
        "cost_sensitivity",
        source_files=[
            dre_run / "model_summary.csv",
            revision_run / "model_summary.csv",
        ],
        description=(
            "Net Sharpe across the tested cost grid for the estimates-enriched "
            "models and the revision strategy, equal-weight standard universe."
        ),
    )


def build_figures(args: argparse.Namespace) -> pd.DataFrame:
    _set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = [
        plot_predictability_gap(
            args.dre_run / "model_summary.csv",
            args.output_dir,
        ),
        plot_deep_size_decomposition(
            args.size_decomposition,
            args.output_dir,
        ),
        plot_revision_factor_alpha(
            args.factor_spanning,
            args.output_dir,
        ),
        plot_implementability_ladder(
            args.revision_run / "model_summary.csv",
            args.econometric_dir / "portfolio_level_bootstrap_cis.csv",
            args.output_dir,
        ),
        plot_lag_decay(
            args.results_root,
            args.lag_run_template,
            args.output_dir,
            lags=tuple(args.lags),
            paired_tests_path=(
                args.econometric_dir / "revision_lag_sensitivity_paired_tests.csv"
            ),
        ),
        plot_cumulative_performance(
            args.revision_run,
            args.dre_run,
            args.market_return,
            args.output_dir,
            cost_bps=args.cost_bps,
        ),
        plot_fama_macbeth_forest(
            args.fama_macbeth,
            args.output_dir,
        ),
        plot_rolling_ic(
            args.revision_run,
            args.dre_run,
            args.output_dir,
            window=args.rolling_window,
        ),
        plot_cost_sensitivity(
            args.dre_run,
            args.revision_run,
            args.output_dir,
            eur_rate_path=args.eur_rate,
        ),
    ]
    manifest = pd.DataFrame([record.__dict__ for record in records])
    manifest_path = args.output_dir / "figure_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest.to_dict(orient="records"), indent=2)
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--dre-run", type=Path, default=DEFAULT_DRE_RUN)
    parser.add_argument("--revision-run", type=Path, default=DEFAULT_REVISION_RUN)
    parser.add_argument(
        "--size-decomposition",
        type=Path,
        default=DEFAULT_SIZE_DECOMPOSITION,
    )
    parser.add_argument("--factor-spanning", type=Path, default=DEFAULT_FACTOR_SPANNING)
    parser.add_argument("--econometric-dir", type=Path, default=DEFAULT_ECONOMETRIC_DIR)
    parser.add_argument("--lag-run-template", default=DEFAULT_LAG_TEMPLATE)
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--fama-macbeth", type=Path, default=DEFAULT_FAMA_MACBETH)
    parser.add_argument("--market-return", type=Path, default=DEFAULT_MARKET_RETURN)
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--rolling-window", type=int, default=36)
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    args = parser.parse_args()

    manifest = build_figures(args)
    print(json.dumps({"figures": int(len(manifest)), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
