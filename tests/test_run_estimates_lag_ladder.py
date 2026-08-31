from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_estimates_lag_ladder.py"
)
SPEC = importlib.util.spec_from_file_location("run_estimates_lag_ladder", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MODELS = ["ridge"]
MONTHS = 36
NAMES = 40


def _write_cell(
    directory: Path,
    *,
    signal_share: float,
    seed: int,
    drop_names: int = 0,
) -> None:
    """A cell whose predictions load ``signal_share`` of the realised return."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-31", periods=MONTHS, freq="ME")
    frames = []
    for date in dates:
        rics = [f"S{index:03d}" for index in range(NAMES - drop_names)]
        signal = rng.normal(size=len(rics))
        noise = rng.normal(size=len(rics))
        frames.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ric": rics,
                    "base_model": "ridge",
                    "prediction": signal,
                    "target_return_1m": signal_share * signal
                    + np.sqrt(1.0 - signal_share**2) * noise,
                }
            )
        )
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        directory / "predictions.parquet", index=False
    )


def _build_ladder(root: Path, shares: dict[int, float]) -> None:
    for lag, share in shares.items():
        _write_cell(root / f"compustat_lag{lag}", signal_share=0.10, seed=lag)
        _write_cell(root / f"estimates_lag{lag}", signal_share=share, seed=lag)


def _run(root: Path, output_dir: Path, lags: list[int], extra: list[str] | None = None):
    argv = [
        "run_estimates_lag_ladder.py",
        "--lags",
        *[str(lag) for lag in lags],
        "--compustat-template",
        str(root / "compustat_lag{lag}"),
        "--estimates-template",
        str(root / "estimates_lag{lag}"),
        "--output-dir",
        str(output_dir),
        "--models",
        *MODELS,
        *(extra or []),
    ]
    original = sys.argv
    sys.argv = argv
    try:
        assert MODULE.main() == 0
    finally:
        sys.argv = original
    return pd.read_csv(output_dir / "lag_ladder_data_depth.csv")


def test_ladder_recovers_a_decaying_analyst_effect(tmp_path):
    # Predictions in the estimates cell share the same random draw as the
    # Compustat cell, so a larger signal share is a larger data-depth effect.
    _build_ladder(tmp_path, {1: 0.30, 2: 0.20, 6: 0.10})

    effects = _run(tmp_path, tmp_path / "out", [1, 2, 6])

    own = effects[effects["sample_scope"].eq("own_matched_sample")].set_index("lag_months")
    assert own.loc[1, "estimate"] > own.loc[2, "estimate"] > own.loc[6, "estimate"]
    assert own.loc[1, "p_value"] < 0.05
    assert own.loc[6, "estimate"] == pytest.approx(0.0, abs=0.02)
    assert own.loc[1, "share_of_shortest_lag"] == pytest.approx(1.0)
    assert 0.0 < own.loc[2, "share_of_shortest_lag"] < 1.0


def test_ladder_reports_both_sample_scopes_and_the_common_stock_months(tmp_path):
    _build_ladder(tmp_path, {1: 0.30, 3: 0.20})
    # Lag 3 covers fewer names, as a longer lag does in the real panel.
    _write_cell(tmp_path / "compustat_lag3", signal_share=0.10, seed=3, drop_names=10)
    _write_cell(tmp_path / "estimates_lag3", signal_share=0.20, seed=3, drop_names=10)

    output_dir = tmp_path / "out"
    effects = _run(tmp_path, output_dir, [1, 3])

    scopes = set(effects["sample_scope"])
    assert scopes == {"own_matched_sample", "common_across_lags"}
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["common_stock_months"] == (NAMES - 10) * MONTHS
    own = effects[
        effects["sample_scope"].eq("own_matched_sample") & effects["lag_months"].eq(1)
    ]
    common = effects[
        effects["sample_scope"].eq("common_across_lags") & effects["lag_months"].eq(1)
    ]
    assert own["stock_months"].iloc[0] == NAMES * MONTHS
    assert common["stock_months"].iloc[0] == (NAMES - 10) * MONTHS


def test_ladder_refuses_unmatched_cells_unless_told_to_intersect(tmp_path):
    _build_ladder(tmp_path, {1: 0.30})
    _write_cell(tmp_path / "estimates_lag1", signal_share=0.30, seed=1, drop_names=5)

    with pytest.raises(SystemExit, match="identical stock-months"):
        _run(tmp_path, tmp_path / "out", [1])

    effects = _run(tmp_path, tmp_path / "out", [1], extra=["--allow-cell-mismatch"])
    assert effects["stock_months"].max() == (NAMES - 5) * MONTHS
