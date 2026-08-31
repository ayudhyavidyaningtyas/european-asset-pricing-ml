from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_coverage_missingness_control.py"
)
SPEC = importlib.util.spec_from_file_location("run_coverage_missingness_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MONTHS = 36
NAMES = 50


def _write_cell(directory: Path, *, signal_share: float, seed: int = 11) -> None:
    """Predictions sharing one random draw so signal_share orders the cells."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-31", periods=MONTHS, freq="ME")
    frames = []
    for date in dates:
        signal = rng.normal(size=NAMES)
        noise = rng.normal(size=NAMES)
        frames.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ric": [f"S{index:03d}" for index in range(NAMES)],
                    "base_model": "ridge",
                    "prediction": signal,
                    "target_return_1m": 0.1 * signal + noise,
                }
            ).assign(prediction=lambda f: f["prediction"] + signal_share * f["target_return_1m"])
        )
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(
        directory / "predictions.parquet", index=False
    )


def _run(root: Path) -> pd.DataFrame:
    output_dir = root / "out"
    argv = [
        "run_coverage_missingness_control.py",
        "--compustat-dir",
        str(root / "compustat"),
        "--coverage-dir",
        str(root / "coverage"),
        "--estimates-dir",
        str(root / "estimates"),
        "--output-dir",
        str(output_dir),
        "--models",
        "ridge",
    ]
    original = sys.argv
    sys.argv = argv
    try:
        assert MODULE.main() == 0
    finally:
        sys.argv = original
    return pd.read_csv(output_dir / "missingness_control_tests.csv")


def test_control_splits_the_effect_into_missingness_and_value_increments(tmp_path):
    # Coverage cell adds nothing over Compustat; estimates cell adds signal.
    _write_cell(tmp_path / "compustat", signal_share=0.0)
    _write_cell(tmp_path / "coverage", signal_share=0.0)
    _write_cell(tmp_path / "estimates", signal_share=0.4)

    table = _run(tmp_path).set_index("quantity")

    assert table.loc["missingness_increment", "estimate"] == pytest.approx(0.0, abs=1e-9)
    assert table.loc["analyst_value_increment", "estimate"] > 0
    assert table.loc["analyst_value_increment", "estimate"] == pytest.approx(
        table.loc["data_depth_effect", "estimate"], abs=1e-9
    )
    assert table.loc["data_depth_effect", "p_value"] < 0.05


def test_control_detects_a_coverage_composition_artefact(tmp_path):
    # Here the coverage indicator alone reproduces the whole lift.
    _write_cell(tmp_path / "compustat", signal_share=0.0)
    _write_cell(tmp_path / "coverage", signal_share=0.4)
    _write_cell(tmp_path / "estimates", signal_share=0.4)

    table = _run(tmp_path).set_index("quantity")

    assert table.loc["missingness_increment", "estimate"] > 0
    assert table.loc["analyst_value_increment", "estimate"] == pytest.approx(
        0.0, abs=1e-9
    )


def test_control_refuses_sample_mismatched_cells(tmp_path):
    _write_cell(tmp_path / "compustat", signal_share=0.0)
    _write_cell(tmp_path / "coverage", signal_share=0.0)
    _write_cell(tmp_path / "estimates", signal_share=0.4)
    predictions = pd.read_parquet(tmp_path / "estimates" / "predictions.parquet")
    predictions.iloc[5:].to_parquet(
        tmp_path / "estimates" / "predictions.parquet", index=False
    )

    with pytest.raises(SystemExit, match="not sample-matched"):
        _run(tmp_path)
