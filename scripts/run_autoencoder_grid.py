"""Pre-specified conditional-autoencoder architecture grid with costed evaluation.

The grid below is declared ex ante and run in full. Model selection is on mean
validation reconstruction loss across refits -- never on test Sharpe. Every cell
is costed against Refinitiv spreads plus square-root impact so that gross and net
results are reported side by side for the whole grid, not only for the winner.

Design (full factorial, 4 x 3 x 2 x 3 = 72 cells):
    n_factors        3, 5, 8, 10
    hidden_sizes     (16,), (32,), (32, 16)
    training window  expanding, rolling 120 months
    universe cap     none, top-1000, top-500 by size

Everything else is held fixed at the values in FIXED_ARGS so that cells differ
only along the four grid axes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_autoencoder_asset_pricing.py"
COST_SCRIPT = PROJECT_ROOT / "scripts" / "run_autoencoder_implementability.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "asset_pricing_ml" / "autoencoder_grid"

# Grid axes -- pre-specified, not tuned on test results.
N_FACTORS = (3, 5, 8, 10)
HIDDEN_SIZES = ((16,), (32,), (32, 16))
TRAINING_WINDOWS = (None, 120)  # None == expanding
UNIVERSE_CAPS = (None, 1000, 500)

# Held fixed across every cell.
FIXED_ARGS = {
    "feature-set": "compustat_enriched",
    "first-test-year": 2015,
    "last-test-year": 2026,
    "min-monthly-stocks": 100,
    "min-training-months": 72,
    "validation-months": 24,
    "activation": "relu",
    "epochs": 15,
    "patience": 4,
    "learning-rate": 0.001,
    "weight-decay": 0.0001,
    "gradient-clip-norm": 5.0,
    "factor-ridge": 0.0001,
    "minimum-size-percentile": 0.05,
    "training-return-clip": 1.0,
    "random-state": 42,
    "device": "cpu",
}

# Large per-cell artefacts. Removed after costing unless --keep-parquet is set,
# because the full grid would otherwise write several GB of predictions.
PRUNABLE = ("autoencoder_predictions.parquet", "autoencoder_weights.parquet")


def cell_name(n_factors: int, hidden: tuple[int, ...], window: int | None, cap: int | None) -> str:
    hidden_tag = "x".join(str(h) for h in hidden)
    window_tag = "expanding" if window is None else f"roll{window}"
    cap_tag = "capall" if cap is None else f"cap{cap}"
    return f"k{n_factors}_h{hidden_tag}_{window_tag}_{cap_tag}"


def build_cells() -> list[dict]:
    cells = []
    for n_factors, hidden, window, cap in itertools.product(
        N_FACTORS, HIDDEN_SIZES, TRAINING_WINDOWS, UNIVERSE_CAPS
    ):
        cells.append(
            {
                "name": cell_name(n_factors, hidden, window, cap),
                "n_factors": n_factors,
                "hidden_sizes": list(hidden),
                "training_window_months": window,
                "max_monthly_stocks": cap,
            }
        )
    return cells


def cell_is_complete(cell_dir: Path) -> bool:
    """A cell counts as done once it has both the gross and the costed summary."""
    return (cell_dir / "autoencoder_summary.csv").exists() and (
        cell_dir / "autoencoder_implementability_summary.csv"
    ).exists()


def run_cell(cell: dict, output_root: Path, aum_eur: str, keep_parquet: bool) -> dict:
    """Fit one grid cell, cost its stock-level SDF weights, prune big artefacts."""
    cell_dir = output_root / cell["name"]
    started = time.time()

    if cell_is_complete(cell_dir):
        return {**cell, "status": "skipped_complete", "elapsed_seconds": 0.0}

    # One BLAS/torch thread per worker. These fits are small, so parallelism
    # belongs across cells; letting each worker grab four threads oversubscribes
    # the machine and slows the whole grid down.
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    cmd = [sys.executable, str(RUN_SCRIPT), "--output-dir", str(cell_dir)]
    for key, value in FIXED_ARGS.items():
        cmd += [f"--{key}", str(value)]
    cmd += ["--n-factors", str(cell["n_factors"])]
    cmd += ["--hidden-sizes"] + [str(h) for h in cell["hidden_sizes"]]
    if cell["training_window_months"] is not None:
        cmd += ["--training-window-months", str(cell["training_window_months"])]
    if cell["max_monthly_stocks"] is not None:
        cmd += ["--max-monthly-stocks", str(cell["max_monthly_stocks"])]

    fit = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if fit.returncode != 0:
        return {**cell, "status": "fit_failed", "error": fit.stderr[-2000:]}

    cost = subprocess.run(
        [
            sys.executable,
            str(COST_SCRIPT),
            "--autoencoder-dir",
            str(cell_dir),
            "--aum-eur",
            aum_eur,
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if cost.returncode != 0:
        return {**cell, "status": "cost_failed", "error": cost.stderr[-2000:]}

    if not keep_parquet:
        for artefact in PRUNABLE:
            target = cell_dir / artefact
            if target.exists():
                target.unlink()

    return {**cell, "status": "ok", "elapsed_seconds": time.time() - started}


def collate(cells: list[dict], output_root: Path) -> pd.DataFrame:
    """Join per-cell fit log, gross summary and costed summary into one table."""
    records = []
    for cell in cells:
        cell_dir = output_root / cell["name"]
        fit_log_path = cell_dir / "autoencoder_fit_log.csv"
        summary_path = cell_dir / "autoencoder_summary.csv"
        cost_path = cell_dir / "autoencoder_implementability_summary.csv"
        if not (fit_log_path.exists() and summary_path.exists()):
            continue

        fit_log = pd.read_csv(fit_log_path)
        summary = pd.read_csv(summary_path).iloc[0]

        record = {
            "cell": cell["name"],
            "n_factors": cell["n_factors"],
            "hidden_sizes": "x".join(str(h) for h in cell["hidden_sizes"]),
            "n_hidden_layers": len(cell["hidden_sizes"]),
            "training_window": (
                "expanding"
                if cell["training_window_months"] is None
                else f"rolling{cell['training_window_months']}"
            ),
            "universe_cap": cell["max_monthly_stocks"] or 0,
            "n_refits": int(len(fit_log)),
            # Selection metrics -- validation only, never test.
            "mean_validation_loss": float(fit_log["validation_loss"].mean()),
            "mean_training_loss": float(fit_log["training_loss"].mean()),
            "mean_validation_total_r2": float(fit_log["validation_total_r2"].mean()),
            "mean_sdf_validation_loss": float(fit_log["sdf_validation_loss"].mean()),
            "mean_best_epoch": float(fit_log["best_epoch"].mean()),
            "mean_fit_seconds": float(fit_log["fit_seconds"].mean()),
            # Test-period reporting.
            "test_months": int(summary["months"]),
            "total_r2": float(summary["total_r2"]),
            "predictive_r2": float(summary["predictive_r2"]),
            "gross_sdf_sharpe": float(summary["sdf_sharpe"]),
            "gross_sdf_return": float(summary["annualized_sdf_return"]),
            "gross_sdf_volatility": float(summary["annualized_sdf_volatility"]),
            "average_pricing_moment_l2": float(summary["average_pricing_moment_l2"]),
            "average_n_test_stocks": float(summary["average_n_test_stocks"]),
        }

        if cost_path.exists():
            cost = pd.read_csv(cost_path)
            for _, row in cost.iterrows():
                label = row["aum_label"]
                record[f"net_sharpe_{label}"] = float(row["net_sharpe"])
                record[f"net_return_{label}"] = float(row["annualized_net_return"])
            first = cost.iloc[0]
            record["average_monthly_turnover"] = float(first["average_monthly_turnover"])
            record["spread_observed_weight"] = float(first["spread_observed_weight"])
            record["mean_half_spread_bps"] = float(first["mean_half_spread_bps"])

        records.append(record)

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    # Validation loss is only comparable within a universe cap: capping to the
    # largest N stocks drops volatile microcaps, so reconstruction loss falls
    # mechanically rather than because the model fits better. Rank within cap
    # and treat the cap itself as a reported design axis, not a selected one.
    frame = frame.sort_values(["universe_cap", "mean_validation_loss"])
    frame["validation_rank_within_cap"] = (
        frame.groupby("universe_cap")["mean_validation_loss"].rank(method="min").astype(int)
    )
    return frame.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--aum-eur", default="10000000,100000000,500000000")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep per-cell prediction/weight parquets (several GB across the grid).",
    )
    parser.add_argument(
        "--collate-only",
        action="store_true",
        help="Skip fitting and rebuild the collated grid table from existing cells.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N cells. Smoke-testing the driver, not a real grid.",
    )
    args = parser.parse_args()

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    cells = build_cells()
    if args.limit:
        cells = cells[: args.limit]

    if not args.collate_only:
        print(f"running {len(cells)} grid cells with {args.workers} workers", flush=True)
        started = time.time()
        completed = 0
        statuses = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_cell, cell, output_root, args.aum_eur, args.keep_parquet): cell
                for cell in cells
            }
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                statuses.append(result)
                print(
                    f"[{completed}/{len(cells)}] {result['name']} -> {result['status']}",
                    flush=True,
                )
                if result["status"] != "ok":
                    print(f"    error: {result.get('error', '')[:500]}", flush=True)
        print(f"grid finished in {time.time() - started:.1f}s", flush=True)
        (output_root / "grid_run_status.json").write_text(
            json.dumps(
                [{k: v for k, v in s.items() if k != "error"} | ({"error": s["error"][:500]} if "error" in s else {}) for s in statuses],
                indent=2,
            )
        )

    frame = collate(cells, output_root)
    if frame.empty:
        print("no completed cells to collate", flush=True)
        return 1

    grid_path = output_root / "autoencoder_grid_summary.csv"
    frame.to_csv(grid_path, index=False)

    selected = {}
    winners = frame[frame["validation_rank_within_cap"] == 1]
    for _, row in winners.iterrows():
        cap_label = "all" if row["universe_cap"] == 0 else int(row["universe_cap"])
        selected[str(cap_label)] = {
            "cell": row["cell"],
            "mean_validation_loss": float(row["mean_validation_loss"]),
            "gross_sdf_sharpe": float(row["gross_sdf_sharpe"]),
            "total_r2": float(row["total_r2"]),
            "predictive_r2": float(row["predictive_r2"]),
            "net_sharpe_100m": float(row.get("net_sharpe_100m", float("nan"))),
        }

    manifest = {
        "grid_axes": {
            "n_factors": list(N_FACTORS),
            "hidden_sizes": [list(h) for h in HIDDEN_SIZES],
            "training_window_months": list(TRAINING_WINDOWS),
            "max_monthly_stocks": list(UNIVERSE_CAPS),
        },
        "fixed_args": FIXED_ARGS,
        "selection_rule": (
            "minimum mean validation reconstruction loss across refits, ranked "
            "within universe cap; the cap is a reported design axis because "
            "validation loss is not comparable across different stock universes"
        ),
        "n_cells_specified": len(cells),
        "n_cells_completed": int(len(frame)),
        "validation_selected_per_cap": selected,
    }
    (output_root / "autoencoder_grid_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nvalidation-selected cell per universe cap:", flush=True)
    for cap_label, info in selected.items():
        print(f"  cap={cap_label}: {info['cell']}", flush=True)
    print(f"collated {len(frame)} cells -> {grid_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
