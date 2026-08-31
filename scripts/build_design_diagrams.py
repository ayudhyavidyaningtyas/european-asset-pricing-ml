"""Build conceptual design diagrams for the dissertation manuscript."""
from __future__ import annotations

import argparse
import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "manuscript"

BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
RED = "#d62728"
GRAY = "#525252"
LIGHT_GRAY = "#f7f7f7"
GRID = "#d9d9d9"
TEXT = "#222222"


@dataclass(frozen=True)
class DiagramRecord:
    figure: str
    png: str
    pdf: str
    data_csv: str
    source_files: str
    description: str


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
        }
    )


def _wrapped(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    title: str,
    lines: Iterable[str],
    face: str,
    edge: str,
    title_color: str | None = None,
    body_width: int = 32,
    title_size: float = 10.0,
    body_size: float = 8.0,
    linewidth: float = 1.2,
    radius: float = 0.018,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.011,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.026,
        _wrapped(title, max(14, body_width)),
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=title_color or edge,
        linespacing=1.05,
        zorder=3,
    )
    body = "\n".join(_wrapped(line, body_width) for line in lines)
    ax.text(
        x + 0.018,
        y + h - 0.079,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=TEXT,
        linespacing=1.18,
        zorder=3,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    rad: float = 0.0,
    linestyle: str = "-",
    linewidth: float = 1.4,
    mutation_scale: float = 14.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arrow)


def _header(ax: plt.Axes, x: float, y: float, w: float, text: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            0.046,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=color,
            edgecolor=color,
            linewidth=0.0,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + 0.023,
        text,
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="white",
        zorder=3,
    )


def _save_diagram(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    rows: list[dict[str, str]],
    description: str,
    source_files: Iterable[Path],
) -> DiagramRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    data_path = data_dir / f"{stem}.csv"
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "element", "label", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return DiagramRecord(
        figure=stem,
        png=str(png_path),
        pdf=str(pdf_path),
        data_csv=str(data_path),
        source_files=";".join(str(path) for path in source_files),
        description=description,
    )


def plot_research_design_map(output_dir: Path) -> DiagramRecord:
    fig, ax = plt.subplots(figsize=(15.6, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="none"))

    ax.text(
        0.5,
        0.965,
        "Figure 1.1. Research design map",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.5,
        0.925,
        "Breadth, depth, data depth and implementability are varied separately, then evaluated on paired out-of-sample evidence.",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#555555",
    )

    columns = [
        (0.03, 0.825, 0.215, "Data setting", BLUE),
        (0.275, 0.825, 0.215, "Separated components", PURPLE),
        (0.52, 0.825, 0.215, "Evaluation protocol", GREEN),
        (0.765, 0.825, 0.205, "Interpretation", ORANGE),
    ]
    for x, y, w, text, color in columns:
        _header(ax, x, y, w, text, color)

    boxes = [
        (
            "data_europe",
            0.03,
            0.665,
            0.215,
            0.13,
            "European equity panel",
            ["Refinitiv/LSEG monthly equities", "active and inactive securities"],
            "#eef5fb",
            BLUE,
        ),
        (
            "data_layers",
            0.03,
            0.49,
            0.215,
            0.15,
            "Information layers",
            ["baseline characteristics", "Compustat and analyst layers", "liquidity inputs"],
            "#eef5fb",
            BLUE,
        ),
        (
            "data_us",
            0.03,
            0.335,
            0.215,
            0.12,
            "Matched benchmark",
            ["US Refinitiv/WRDS panel", "same rank design"],
            "#eef5fb",
            BLUE,
        ),
        (
            "breadth",
            0.275,
            0.665,
            0.215,
            0.115,
            "Characteristic breadth",
            ["momentum to ridge"],
            "#f4eff9",
            PURPLE,
        ),
        (
            "depth",
            0.275,
            0.535,
            0.215,
            0.115,
            "Model depth",
            ["ridge to flexible models"],
            "#f4eff9",
            PURPLE,
        ),
        (
            "data_depth",
            0.275,
            0.405,
            0.215,
            0.115,
            "Data depth",
            ["Compustat to analyst layer"],
            "#f4eff9",
            PURPLE,
        ),
        (
            "implementability",
            0.275,
            0.275,
            0.215,
            0.115,
            "Implementability",
            ["tradability and cost constraints"],
            "#f4eff9",
            PURPLE,
        ),
        (
            "walk_forward",
            0.52,
            0.64,
            0.215,
            0.13,
            "Walk-forward out-of-sample",
            ["annual refits", "trailing validation"],
            "#eef8ef",
            GREEN,
        ),
        (
            "paired",
            0.52,
            0.465,
            0.215,
            0.135,
            "Paired comparison units",
            ["common months and stock-months", "within-bucket differences"],
            "#eef8ef",
            GREEN,
        ),
        (
            "inference",
            0.52,
            0.295,
            0.215,
            0.13,
            "Inference discipline",
            ["HAC lag 6", "Holm families"],
            "#eef8ef",
            GREEN,
        ),
        (
            "question_earned",
            0.765,
            0.64,
            0.205,
            0.13,
            "Performance source",
            ["breadth vs depth", "data depth vs model depth"],
            "#fff3e8",
            ORANGE,
        ),
        (
            "question_survives",
            0.765,
            0.465,
            0.205,
            0.135,
            "Capacity frontier",
            ["small stocks vs top 500", "trading-value gradient"],
            "#fff3e8",
            ORANGE,
        ),
        (
            "question_means",
            0.765,
            0.295,
            0.205,
            0.13,
            "Economic content",
            ["forecast errors", "factor spanning", "implementation"],
            "#fff3e8",
            ORANGE,
        ),
    ]
    for element, x, y, w, h, title, lines, face, edge in boxes:
        _box(
            ax,
            x,
            y,
            w,
            h,
            title=title,
            lines=lines,
            face=face,
            edge=edge,
            body_width=30,
            body_size=7.4,
        )

    arrow_pairs = [
        ((0.245, 0.73), (0.275, 0.73)),
        ((0.245, 0.56), (0.275, 0.56)),
        ((0.245, 0.395), (0.275, 0.395)),
        ((0.49, 0.725), (0.52, 0.705)),
        ((0.49, 0.592), (0.52, 0.535)),
        ((0.49, 0.462), (0.52, 0.535)),
        ((0.49, 0.332), (0.52, 0.36)),
        ((0.735, 0.705), (0.765, 0.705)),
        ((0.735, 0.532), (0.765, 0.532)),
        ((0.735, 0.36), (0.765, 0.36)),
    ]
    for start, end in arrow_pairs:
        _arrow(ax, start, end, color="#6b6b6b", linewidth=1.25)

    ax.text(
        0.5,
        0.145,
        "Within each contrast, timing, sample construction and evaluation rules are held fixed where possible.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#444444",
        bbox={"facecolor": LIGHT_GRAY, "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )

    rows = [
        {"figure": "research_design_map", "element": element, "label": title, "detail": "; ".join(lines)}
        for element, _x, _y, _w, _h, title, lines, _face, _edge in boxes
    ]
    rows.append(
        {
            "figure": "research_design_map",
            "element": "design_principle",
            "label": "Vary one component at a time",
            "detail": "Hold timing, sample and evaluation rules constant within each contrast.",
        }
    )
    return _save_diagram(
        fig,
        output_dir,
        "research_design_map",
        rows=rows,
        source_files=[],
        description=(
            "Conceptual map linking the European data setting, separated design "
            "components, paired out-of-sample evaluation and dissertation questions."
        ),
    )


def plot_walk_forward_timing(output_dir: Path) -> DiagramRecord:
    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="none"))

    ax.text(
        0.5,
        0.965,
        "Figure 4.1. Walk-forward timing and leakage-control diagram",
        ha="center",
        va="center",
        fontsize=15.0,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.5,
        0.925,
        "Each annual fold selects hyperparameters only from past data, then predicts twelve excluded out-of-sample monthly cross-sections.",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#555555",
    )

    ax.text(0.055, 0.845, "Annual walk-forward fold", fontsize=11.5, fontweight="bold", color=TEXT)
    baseline_y = 0.64

    timeline_blocks = [
        (
            "train_core",
            0.065,
            0.67,
            0.40,
            0.13,
            "Expanding training window",
            ["past observations before validation", "features observable at each signal month"],
            "#eef5fb",
            BLUE,
        ),
        (
            "validation",
            0.48,
            0.67,
            0.18,
            0.13,
            "Trailing validation block",
            ["fixed-grid tuning", "no evaluation-year data"],
            "#fff3e8",
            ORANGE,
        ),
        (
            "cutoff",
            0.675,
            0.67,
            0.075,
            0.13,
            "Refit cutoff",
            ["end Y-1"],
            "#f7f7f7",
            GRAY,
        ),
        (
            "oos",
            0.765,
            0.67,
            0.20,
            0.13,
            "Out-of-sample year",
            ["Jan-Dec Y predictions", "excluded from fit and tuning"],
            "#eef8ef",
            GREEN,
        ),
    ]
    for element, x, y, w, h, title, lines, face, edge in timeline_blocks:
        _box(
            ax,
            x,
            y,
            w,
            h,
            title=title,
            lines=lines,
            face=face,
            edge=edge,
            body_width=38,
            title_size=9.8,
            body_size=7.4,
            radius=0.012,
        )
    for x, label in [
        (0.065, "earlier history"),
        (0.48, "validation start"),
        (0.675, "cutoff"),
        (0.765, "Jan Y"),
        (0.965, "Dec Y"),
    ]:
        ax.plot([x, x], [baseline_y - 0.012, baseline_y + 0.018], color="#666666", linewidth=1.0)
        ax.text(x, baseline_y - 0.028, label, ha="center", va="top", fontsize=7.6, color="#555555")
    ax.plot([0.065, 0.965], [baseline_y, baseline_y], color="#888888", linewidth=1.0, zorder=0)
    _arrow(ax, (0.75, 0.735), (0.765, 0.735), color=GREEN, linewidth=1.4)
    ax.text(
        0.765,
        0.825,
        "Predictions begin only after refit",
        ha="left",
        va="center",
        fontsize=8.0,
        color=GREEN,
    )

    ax.text(0.055, 0.58, "Information timing inside an out-of-sample month", fontsize=11.5, fontweight="bold", color=TEXT)
    row_y = 0.415
    controls = [
        (
            "accounting_lag",
            0.065,
            row_y,
            0.20,
            0.13,
            "Accounting lag",
            ["six-month accounting lag"],
            "#f4eff9",
            PURPLE,
        ),
        (
            "analyst_lag",
            0.295,
            row_y,
            0.20,
            0.13,
            "Analyst lag",
            ["one-month shift before ranking"],
            "#f4eff9",
            PURPLE,
        ),
        (
            "signal_t",
            0.535,
            row_y,
            0.18,
            0.13,
            "Signal at month t",
            ["ranked features at month-end"],
            "#eef5fb",
            BLUE,
        ),
        (
            "return_t1",
            0.765,
            row_y,
            0.20,
            0.13,
            "Target in month t+1",
            ["realised return never enters month t features"],
            "#eef8ef",
            GREEN,
        ),
    ]
    for element, x, y, w, h, title, lines, face, edge in controls:
        _box(
            ax,
            x,
            y,
            w,
            h,
            title=title,
            lines=lines,
            face=face,
            edge=edge,
            body_width=31,
            title_size=9.8,
            body_size=7.6,
            radius=0.012,
        )
    for start, end in [
        ((0.265, row_y + 0.065), (0.295, row_y + 0.065)),
        ((0.495, row_y + 0.065), (0.535, row_y + 0.065)),
        ((0.715, row_y + 0.065), (0.765, row_y + 0.065)),
    ]:
        _arrow(ax, start, end, color="#6b6b6b", linewidth=1.2)

    ax.text(0.055, 0.305, "Leakage-control gates before a prediction is admitted", fontsize=11.5, fontweight="bold", color=TEXT)
    gates = [
        (
            "common_inputs",
            0.065,
            0.17,
            0.20,
            0.105,
            "Common ranked inputs",
            ["same within-month characteristics"],
            "#f7f7f7",
            GRAY,
        ),
        (
            "sequence_history",
            0.295,
            0.17,
            0.20,
            0.105,
            "Sequence histories",
            ["trailing windows end at t"],
            "#f7f7f7",
            GRAY,
        ),
        (
            "vintage_audit",
            0.535,
            0.17,
            0.18,
            0.105,
            "Vintage checks",
            ["announcement and snapshot timing audited"],
            "#f7f7f7",
            GRAY,
        ),
        (
            "paired_sample",
            0.765,
            0.17,
            0.20,
            0.105,
            "Paired OOS sample",
            ["common months and stock-months where required"],
            "#f7f7f7",
            GRAY,
        ),
    ]
    for element, x, y, w, h, title, lines, face, edge in gates:
        _box(
            ax,
            x,
            y,
            w,
            h,
            title=title,
            lines=lines,
            face=face,
            edge=edge,
            body_width=31,
            title_size=9.3,
            body_size=7.2,
            radius=0.012,
        )
    for start, end in [
        ((0.265, 0.223), (0.295, 0.223)),
        ((0.495, 0.223), (0.535, 0.223)),
        ((0.715, 0.223), (0.765, 0.223)),
    ]:
        _arrow(ax, start, end, color="#777777", linewidth=1.1)

    ax.text(
        0.5,
        0.075,
        "Key restriction: evaluation-year observations are never used for training or tuning.\nMonth t signals use only information available by month-end t.",
        ha="center",
        va="center",
        fontsize=8.4,
        color="#444444",
        bbox={"facecolor": LIGHT_GRAY, "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )

    rows = [
        {"figure": "walk_forward_timing_leakage_controls", "element": element, "label": title, "detail": "; ".join(lines)}
        for element, _x, _y, _w, _h, title, lines, _face, _edge in timeline_blocks + controls + gates
    ]
    rows.append(
        {
            "figure": "walk_forward_timing_leakage_controls",
            "element": "key_restriction",
            "label": "No evaluation-year leakage",
            "detail": "No evaluation-year observation is used to estimate parameters, select hyperparameters or construct month t signals after month t information.",
        }
    )
    return _save_diagram(
        fig,
        output_dir,
        "walk_forward_timing_leakage_controls",
        rows=rows,
        source_files=[PROJECT_ROOT / "DATA.md"],
        description=(
            "Walk-forward estimation and leakage-control diagram covering annual "
            "refits, trailing validation, out-of-sample evaluation and month t to "
            "t+1 information timing."
        ),
    )


def build_diagrams(args: argparse.Namespace) -> pd.DataFrame:
    _set_style()
    records = [
        plot_research_design_map(args.output_dir),
        plot_walk_forward_timing(args.output_dir),
    ]
    manifest = pd.DataFrame([record.__dict__ for record in records])
    manifest_path = args.output_dir / "design_diagram_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (args.output_dir / "design_diagram_manifest.json").write_text(
        json.dumps(manifest.to_dict(orient="records"), indent=2)
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = build_diagrams(args)
    print(json.dumps(manifest.to_dict(orient="records"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
