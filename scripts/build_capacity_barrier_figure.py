"""Headline figure: the flexibility premium by tradability bucket.

Plots IC(model) - IC(momentum) across tradability buckets on two independent
dimensions (market capitalisation and EUR trading value). The premium is large
and precisely estimated in the least tradable segment and indistinguishable
from zero in the largest 500 stocks -- the capacity barrier.

Model identity is carried by marker shape as well as colour, so the figure
survives greyscale printing and colour-vision deficiency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRADIENT_DIR = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "capacity_gradient_tests"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "manuscript"

MODEL_ORDER = ["ridge_rank", "hist_gbm_rank", "mlp_rank", "dre_rank"]
MODEL_LABELS = {
    "ridge_rank": "Ridge",
    "hist_gbm_rank": "HistGBM",
    "mlp_rank": "MLP",
    "dre_rank": "DRE",
}
# Colours match the existing manuscript figures; markers carry identity too.
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

PANELS = [
    (
        "market_cap",
        "By market capitalisation",
        ["low_cap", "mid_cap", "high_cap", "top_500_cap"],
        ["Smallest\ntercile", "Middle\ntercile", "Largest\ntercile", "Top 500\nby size"],
    ),
    (
        "trading_value",
        "By EUR trading value (ADV proxy)",
        ["low_adv", "mid_adv", "high_adv", "top_500_adv"],
        ["Lowest\ntercile", "Middle\ntercile", "Highest\ntercile", "Top 500\nby ADV"],
    ),
]


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


def build_figure(
    buckets: pd.DataFrame, gradients: pd.DataFrame, output_dir: Path
) -> dict[str, str]:
    _set_style()
    premium = buckets[buckets["premium"].eq("flexibility_premium")]
    # Restrict to the four fully-trained models: the sequence models ran under a
    # 150k training-row cap and must not enter the reported gradient range.
    gradient = gradients[
        gradients["premium"].eq("flexibility_premium")
        & gradients["model"].isin(MODEL_ORDER)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), sharey=True)
    plotted: list[pd.DataFrame] = []

    for ax, (dimension, title, order, tick_labels) in zip(axes, PANELS):
        panel = premium[premium["dimension"].eq(dimension)]
        positions = range(len(order))

        # Shade the investable frontier so the collapse is located, not just seen.
        ax.axvspan(
            len(order) - 1.42,
            len(order) - 0.58,
            color="#bdbdbd",
            alpha=0.22,
            zorder=0,
        )

        for model in MODEL_ORDER:
            series = (
                panel[panel["model"].eq(model)]
                .set_index("bucket")
                .reindex(order)
            )
            if series["estimate"].isna().all():
                continue
            offset = (MODEL_ORDER.index(model) - 1.5) * 0.045
            xs = [p + offset for p in positions]
            ax.errorbar(
                xs,
                series["estimate"],
                yerr=[
                    series["estimate"] - series["ci_low"],
                    series["ci_high"] - series["estimate"],
                ],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.9,
                linewidth=2.0,
                elinewidth=1.0,
                capsize=2.5,
                alpha=0.95,
                label=MODEL_LABELS[model],
                zorder=3,
            )
            frame = series.reset_index()[["bucket", "estimate", "ci_low", "ci_high", "t_stat"]]
            frame.insert(0, "model", model)
            frame.insert(0, "dimension", dimension)
            plotted.append(frame)

        ax.axhline(0.0, color="black", linewidth=0.9, zorder=2)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-0.5, len(order) - 0.5)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.02f}"))
        ax.grid(axis="x", visible=False)

        row = gradient[gradient["dimension"].eq(dimension)]
        if not row.empty:
            low = row["estimate"].min()
            high = row["estimate"].max()
            t_low = row["t_stat"].min()
            t_high = row["t_stat"].max()
            ax.text(
                0.03,
                0.05,
                f"Capacity gradient {low:+.3f} to {high:+.3f}\n"
                f"(t = {t_low:.1f} to {t_high:.1f}, Holm p < 0.01)",
                transform=ax.transAxes,
                fontsize=8,
                color="#333333",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#cccccc",
                    "boxstyle": "round,pad=0.35",
                    "alpha": 0.9,
                },
            )

    axes[0].set_ylabel("Paired monthly IC difference\nIC(model) - IC(momentum)")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.92)

    axes[1].annotate(
        "premium indistinguishable\nfrom zero at the\ninvestable frontier",
        xy=(3.0, -0.005),
        xytext=(2.05, -0.045),
        fontsize=8,
        color="#333333",
        ha="center",
        arrowprops={
            "arrowstyle": "->",
            "color": "#555555",
            "linewidth": 1.0,
            "shrinkB": 6,
        },
    )

    fig.suptitle(
        "The capacity barrier taxes characteristic breadth, not model depth",
        fontsize=12.5,
        y=0.99,
    )
    fig.text(
        0.5,
        -0.02,
        "Paired within month and bucket; 95% HAC (lag 6) intervals; "
        "137 months, common sample, ex bottom 5% by market capitalisation.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    stem = "capacity_barrier_flexibility_premium"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    data_path = data_dir / f"{stem}.csv"
    pd.concat(plotted, ignore_index=True).to_csv(data_path, index=False)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path), "data": str(data_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gradient-dir", type=Path, default=DEFAULT_GRADIENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    buckets = pd.read_csv(args.gradient_dir / "paired_premium_by_bucket.csv")
    gradients = pd.read_csv(args.gradient_dir / "capacity_gradient_tests.csv")
    outputs = build_figure(buckets, gradients, args.output_dir)

    manifest = {
        "figure": "capacity_barrier_flexibility_premium",
        "source_files": [
            str(args.gradient_dir / "paired_premium_by_bucket.csv"),
            str(args.gradient_dir / "capacity_gradient_tests.csv"),
        ],
        "description": (
            "Flexibility premium IC(model) - IC(momentum) across tradability "
            "buckets on two independent dimensions, with 95% HAC intervals."
        ),
        "outputs": outputs,
    }
    (args.output_dir / "capacity_barrier_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
