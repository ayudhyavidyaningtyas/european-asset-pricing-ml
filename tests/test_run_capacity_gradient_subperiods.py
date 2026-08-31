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
    / "run_capacity_gradient_subperiods.py"
)
SPEC = importlib.util.spec_from_file_location("run_capacity_gradient_subperiods", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MONTHS = 132  # 2015-01 .. 2025-12, split 2021-01


def _monthly_ics(
    *,
    first_half_gradient: float,
    second_half_gradient: float,
    seed: int = 0,
) -> pd.DataFrame:
    """Bucket ICs where the ridge-benchmark model's gradient shifts at 2021."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=MONTHS, freq="ME")
    records = []
    for date in dates:
        gradient = (
            first_half_gradient
            if date < pd.Timestamp("2021-01-01")
            else second_half_gradient
        )
        for dimension, (least, most) in MODULE.DIMENSION_BUCKETS.items():
            for bucket in (least, most, "all"):
                base = rng.normal(scale=0.005)
                records.append(
                    {
                        "model": "ridge_rank",
                        "dimension": dimension,
                        "bucket": bucket,
                        "date": date,
                        "names": 500,
                        "ic": 0.05 + base,
                    }
                )
                premium = gradient if bucket == least else 0.0
                records.append(
                    {
                        "model": "hist_gbm_rank",
                        "dimension": dimension,
                        "bucket": bucket,
                        "date": date,
                        "names": 500,
                        "ic": 0.05 + base + premium + rng.normal(scale=0.002),
                    }
                )
    return pd.DataFrame(records)


def _run(tmp_path: Path, ics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ics_path = tmp_path / "monthly_bucket_ics.csv"
    ics.to_csv(ics_path, index=False)
    output_dir = tmp_path / "out"
    argv = [
        "run_capacity_gradient_subperiods.py",
        "--monthly-ics",
        str(ics_path),
        "--output-dir",
        str(output_dir),
    ]
    original = sys.argv
    sys.argv = argv
    try:
        assert MODULE.main() == 0
    finally:
        sys.argv = original
    return (
        pd.read_csv(output_dir / "capacity_gradient_subperiods.csv"),
        pd.read_csv(output_dir / "capacity_gradient_shift_tests.csv"),
    )


def test_stable_gradient_shows_no_shift(tmp_path):
    ics = _monthly_ics(first_half_gradient=0.06, second_half_gradient=0.06)

    subperiods, shifts = _run(tmp_path, ics)

    depth = subperiods[
        subperiods["premium"].eq("depth_premium")
        & subperiods["model"].eq("hist_gbm_rank")
        & subperiods["dimension"].eq("market_cap")
    ].set_index("subperiod")
    assert depth.loc["first_half", "estimate"] == pytest.approx(0.06, abs=0.005)
    assert depth.loc["second_half", "estimate"] == pytest.approx(0.06, abs=0.005)
    shift = shifts[
        shifts["model"].eq("hist_gbm_rank") & shifts["dimension"].eq("market_cap")
    ].iloc[0]
    assert abs(shift["estimate"]) < 0.005
    assert shift["p_value"] > 0.10


def test_halved_gradient_is_detected_by_the_interaction_test(tmp_path):
    ics = _monthly_ics(first_half_gradient=0.08, second_half_gradient=0.02)

    subperiods, shifts = _run(tmp_path, ics)

    depth = subperiods[
        subperiods["premium"].eq("depth_premium")
        & subperiods["model"].eq("hist_gbm_rank")
        & subperiods["dimension"].eq("market_cap")
    ].set_index("subperiod")
    assert depth.loc["first_half", "estimate"] == pytest.approx(0.08, abs=0.005)
    assert depth.loc["second_half", "estimate"] == pytest.approx(0.02, abs=0.005)
    shift = shifts[
        shifts["model"].eq("hist_gbm_rank") & shifts["dimension"].eq("market_cap")
    ].iloc[0]
    assert shift["estimate"] == pytest.approx(-0.06, abs=0.01)
    assert shift["p_value"] < 0.01
    # Interaction estimate must equal the difference of the window means.
    assert shift["estimate"] == pytest.approx(
        depth.loc["second_half", "estimate"] - depth.loc["first_half", "estimate"],
        abs=1e-9,
    )


def test_window_boundaries_and_month_counts(tmp_path):
    ics = _monthly_ics(first_half_gradient=0.05, second_half_gradient=0.05)

    subperiods, _ = _run(tmp_path, ics)

    row = subperiods[
        subperiods["model"].eq("hist_gbm_rank")
        & subperiods["dimension"].eq("market_cap")
        & subperiods["premium"].eq("depth_premium")
    ].set_index("subperiod")
    assert row.loc["first_half", "months"] == 72
    assert row.loc["second_half", "months"] == 60
    assert row.loc["first_half", "window_end"] == "2020-12-31"
    assert row.loc["second_half", "window_start"] == "2021-01-31"
