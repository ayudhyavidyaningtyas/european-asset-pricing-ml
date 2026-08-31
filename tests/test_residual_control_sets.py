"""Tests for residual-target control sets and the predictor/control guard."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import asset_pricing_ml as apm  # noqa: E402
import stats as project_stats  # noqa: E402


def test_full_control_set_contains_momentum_and_others_do_not():
    assert "momentum_12_2_rank" in apm.RESIDUAL_CONTROL_SETS["full"]
    assert "momentum_12_2_rank" not in apm.RESIDUAL_CONTROL_SETS[
        "styles_ex_momentum"
    ]
    assert apm.RESIDUAL_CONTROL_SETS["country_sector"] == []
    # styles_ex_momentum must otherwise match full.
    assert set(apm.RESIDUAL_CONTROL_SETS["styles_ex_momentum"]) == set(
        apm.RESIDUAL_CONTROL_SETS["full"]
    ) - {"momentum_12_2_rank"}


def test_momentum_is_ineligible_under_full_controls():
    eligible, note = project_stats.residual_target_model_eligible(
        apm.RESIDUAL_CONTROL_SETS["full"], "momentum_residual_rank", "ridge_residual_rank"
    )
    assert eligible is False
    assert "momentum_12_2_rank" in note


def test_momentum_is_eligible_under_country_sector_controls():
    for control_set in ("country_sector", "styles_ex_momentum"):
        eligible, note = project_stats.residual_target_model_eligible(
            apm.RESIDUAL_CONTROL_SETS[control_set],
            "momentum_residual_rank",
            "ridge_residual_rank",
        )
        assert eligible is True, control_set
        assert note == "no_predictor_control_conflict"


def test_fitted_models_are_always_eligible():
    eligible, _ = project_stats.residual_target_model_eligible(
        apm.RESIDUAL_CONTROL_SETS["full"],
        "hist_gbm_residual_rank",
        "ridge_residual_rank",
    )
    assert eligible is True


def test_add_residual_targets_rejects_unknown_control_set():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"]),
            "target_return_1m": [0.01],
            "target_return_rank": [0.5],
        }
    )
    with pytest.raises(ValueError, match="Unknown residual control set"):
        apm.add_residual_targets(panel, control_set="nonsense")


def _residual_panel(control_set: str) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n_per_month = 60
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    rows = []
    for date in dates:
        momentum = rng.uniform(-1, 1, n_per_month)
        rows.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ric": [f"R{i}" for i in range(n_per_month)],
                    "momentum_12_2_rank": momentum,
                    "log_size_rank": rng.uniform(-1, 1, n_per_month),
                    "book_to_market_rank": rng.uniform(-1, 1, n_per_month),
                    "volatility_12m_rank": rng.uniform(-1, 1, n_per_month),
                    "return_1m_rank": rng.uniform(-1, 1, n_per_month),
                    "screen_country": rng.choice(["DE", "FR"], n_per_month),
                    "TR.TRBCECONOMICSECTOR": rng.choice(["A", "B"], n_per_month),
                    # Target is driven by momentum plus noise.
                    "target_return_1m": 0.05 * momentum
                    + rng.normal(0, 0.01, n_per_month),
                }
            )
        )
    panel = pd.concat(rows, ignore_index=True)
    panel["target_return_rank"] = panel.groupby("date")[
        "target_return_1m"
    ].rank(pct=True).mul(2).sub(1)
    return apm.add_residual_targets(panel, control_set=control_set)


def test_momentum_signal_is_destroyed_by_full_controls_but_survives_country_sector():
    """The bug this guard exists for, demonstrated end to end."""

    def momentum_ic(panel: pd.DataFrame) -> float:
        return float(
            panel.groupby("date")
            .apply(
                lambda month: month["momentum_12_2_rank"].corr(
                    month["target_residual_rank"], method="spearman"
                ),
                include_groups=False,
            )
            .mean()
        )

    full = _residual_panel("full")
    country_sector = _residual_panel("country_sector")

    # Regressed against itself, momentum's residual IC collapses to ~0.
    assert abs(momentum_ic(full)) < 0.05
    # With only country/sector removed, the real momentum signal survives.
    assert momentum_ic(country_sector) > 0.5


def test_residual_control_set_recorded_on_panel_attrs():
    panel = _residual_panel("styles_ex_momentum")
    assert panel.attrs["residual_control_set"] == "styles_ex_momentum"
    assert "momentum_12_2_rank" not in panel.attrs["residual_control_columns"]


def test_ineligible_comparisons_are_excluded_from_holm_family():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2020-01-31", periods=40, freq="ME")
    frames = []
    for model, quality in [
        ("momentum_residual_rank", 0.0),
        ("ridge_residual_rank", 0.5),
        ("hist_gbm_residual_rank", 0.6),
    ]:
        for date in dates:
            n = 40
            target = rng.normal(0, 1, n)
            frames.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "ric": [f"R{i}" for i in range(n)],
                        "model": model,
                        "base_model": model.replace("_residual_rank", ""),
                        "target_mode": "residual_rank",
                        "target_return_1m": target * 0.01,
                        "target_return_rank": target,
                        "target_residual_rank": target,
                        "prediction": quality * target
                        + rng.normal(0, 1, n) * (1 - quality),
                    }
                )
            )
    predictions = pd.concat(frames, ignore_index=True)

    _, ic = apm.predictive_accuracy_tests(
        predictions,
        residual_control_columns=apm.RESIDUAL_CONTROL_SETS["full"],
    )
    involves_momentum = ic["model_a"].str.startswith("momentum") | ic[
        "model_b"
    ].str.startswith("momentum")

    # Momentum rows are retained for audit but carry no Holm-adjusted p-value.
    assert not ic[involves_momentum].empty
    assert ic.loc[involves_momentum, "residual_control_eligible"].eq(False).all()
    assert ic.loc[involves_momentum, "p_value_holm"].isna().all()
    # Valid comparisons still get one.
    assert ic.loc[~involves_momentum, "residual_control_eligible"].all()
    assert ic.loc[~involves_momentum, "p_value_holm"].notna().all()


def test_non_residual_targets_are_never_flagged():
    rng = np.random.default_rng(9)
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    frames = []
    for model in ["momentum_rank", "ridge_rank"]:
        for date in dates:
            n = 30
            target = rng.normal(0, 1, n)
            frames.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "ric": [f"R{i}" for i in range(n)],
                        "model": model,
                        "base_model": model.replace("_rank", ""),
                        "target_mode": "rank",
                        "target_return_1m": target * 0.01,
                        "target_return_rank": target,
                        "prediction": rng.normal(0, 1, n),
                    }
                )
            )
    predictions = pd.concat(frames, ignore_index=True)
    _, ic = apm.predictive_accuracy_tests(
        predictions,
        residual_control_columns=apm.RESIDUAL_CONTROL_SETS["full"],
    )
    assert ic["residual_control_eligible"].all()
    assert ic["residual_control_note"].eq("not_a_residual_target").all()
    assert ic["p_value_holm"].notna().all()
