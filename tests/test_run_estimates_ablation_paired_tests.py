from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_estimates_ablation_paired_tests.py"
)
SPEC = importlib.util.spec_from_file_location("run_estimates_ablation_paired_tests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _prediction_frame(prediction_scale: float) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    records = []
    for index, date in enumerate(dates):
        target = np.sin(index / 4.0)
        records.append(
            {
                "date": date,
                "ric": "S00",
                "prediction": prediction_scale * target,
                "target_return_rank": target,
            }
        )
    return pd.DataFrame(records)


def _write_manifest(run_dir: Path, features: list[str]) -> None:
    run_dir.mkdir()
    (run_dir / "ml_manifest.json").write_text(
        json.dumps({"feature_columns": features})
    )


def test_paired_loss_tests_suppress_clark_west_when_not_nested():
    variant = _prediction_frame(0.9)
    reference = _prediction_frame(0.0)

    result = MODULE.paired_loss_tests(
        variant,
        reference,
        lags=3,
        clark_west_is_nested=False,
    )

    assert result["loss_mean_difference"] < 0
    assert np.isnan(result["clark_west_adjusted_mean_difference"])
    assert np.isnan(result["clark_west_p_one_sided"])


def test_paired_loss_tests_compute_clark_west_when_nested():
    variant = _prediction_frame(0.9)
    reference = _prediction_frame(0.0)

    result = MODULE.paired_loss_tests(
        variant,
        reference,
        lags=3,
        clark_west_is_nested=True,
    )

    assert result["loss_mean_difference"] < 0
    assert result["clark_west_adjusted_mean_difference"] > 0
    assert 0.0 <= result["clark_west_p_one_sided"] <= 1.0


def test_nesting_metadata_uses_manifest_feature_superset(tmp_path: Path):
    variant = tmp_path / "variant"
    reference = tmp_path / "reference"
    _write_manifest(variant, ["size_rank", "momentum_rank", "revision_rank"])
    _write_manifest(reference, ["size_rank", "momentum_rank"])

    result = MODULE.nesting_metadata("variant", variant, "reference", reference)

    assert result["clark_west_is_nested"] is True
    assert "extra_features=1" in result["clark_west_nesting_note"]


def test_nesting_metadata_rejects_missing_restricted_features(tmp_path: Path):
    variant = tmp_path / "variant"
    reference = tmp_path / "reference"
    _write_manifest(variant, ["revision_rank"])
    _write_manifest(reference, ["size_rank", "momentum_rank"])

    result = MODULE.nesting_metadata("variant", variant, "reference", reference)

    assert result["clark_west_is_nested"] is False
    assert "missing_restricted_features=2" in result["clark_west_nesting_note"]


def _stock_month_frame(names: int, months: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=months, freq="ME")
    return pd.DataFrame(
        [
            {"date": date, "ric": f"S{index:03d}", "prediction": 0.0}
            for date in dates
            for index in range(names)
        ]
    )


def test_stock_month_match_accepts_identical_samples():
    frame = _stock_month_frame(10)

    match = MODULE.stock_month_match(frame, frame.copy())

    assert match["samples_identical"]
    assert match["shared_stock_months"] == 40


def test_stock_month_match_flags_a_run_scored_on_extra_rows():
    reference = _stock_month_frame(10)
    variant = _stock_month_frame(12)

    match = MODULE.stock_month_match(variant, reference)

    assert not match["samples_identical"]
    assert match["variant_stock_months"] == 48
    assert match["reference_stock_months"] == 40
    assert match["shared_stock_months"] == 40


def test_ablation_variants_cover_both_decompositions():
    assert set(MODULE.ABLATION_VARIANTS) == {
        "eps_only",
        "revenue_only",
        "price_target_only",
        "levels_only",
        "revisions_only",
        "dispersion_only",
        "ex_eps",
        "ex_revenue",
        "ex_price_target",
        "ex_levels",
        "ex_revisions",
        "ex_dispersion",
    }
