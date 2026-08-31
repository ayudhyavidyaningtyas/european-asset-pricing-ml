from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stats import (  # noqa: E402
    base_estimator_name,
    clark_west_estimator_eligible,
    clark_west_test,
    stationary_bootstrap_metric_ci,
)


def _cw_rejection_rate(builder, replications=25, alpha=0.05, seed=0):
    """Share of replications where Clark-West rejects at ``alpha`` (one-sided)."""
    rng = np.random.default_rng(seed)
    months, stocks = 120, 400
    dates = pd.Series(np.repeat(np.arange(months), stocks))
    rejections = 0
    for _ in range(replications):
        actual, restricted, unrestricted = builder(rng, months * stocks)
        result = clark_west_test(
            actual,
            restricted,
            unrestricted,
            dates=dates,
            maxlags=6,
        )
        if result["p_one_sided"] < alpha:
            rejections += 1
    return rejections / replications


def test_base_estimator_name_strips_target_mode_suffix():
    assert base_estimator_name("ridge_rank") == "ridge"
    assert base_estimator_name("hist_gbm_rank") == "hist_gbm"
    assert base_estimator_name("elastic_net_return") == "elastic_net"
    assert base_estimator_name("mlp_residual_rank") == "mlp"
    assert base_estimator_name("momentum") == "momentum"


def test_clark_west_estimator_eligibility_excludes_stochastic_learners():
    """Clark-West assumes a deterministic fit; stochastic learners must be excluded.

    An MLP trained on 17 extra features also differs by random initialisation, SGD
    path and early stopping. That training variance enters the (f_r - f_u)^2
    adjustment with no matching MSPE inflation, which manufactured p-values down to
    1e-23 for models that had *higher* out-of-sample loss.
    """
    eligible, note = clark_west_estimator_eligible("ridge_rank")
    assert eligible is True
    assert "deterministic_estimator=ridge" in note

    for stochastic in ("mlp_rank", "hist_gbm_rank", "dre_rank"):
        eligible, note = clark_west_estimator_eligible(stochastic)
        assert eligible is False, f"{stochastic} must not be Clark-West eligible"
        assert note.startswith("stochastic_estimator=")

    # A pair is only eligible when *both* estimators are deterministic.
    assert clark_west_estimator_eligible("momentum_rank", "ridge_rank")[0] is True
    assert clark_west_estimator_eligible("momentum_rank", "mlp_rank")[0] is False

    # Unknown estimators default to ineligible rather than silently inheriting it.
    eligible, note = clark_west_estimator_eligible("some_new_model_rank")
    assert eligible is False
    assert note.startswith("unknown_estimator=")


def test_clark_west_is_correctly_sized_when_nesting_holds():
    """Unrestricted = restricted + pure noise: CW must stay near its nominal size."""

    def builder(rng, n):
        actual = rng.normal(0.0, 1.0, n)
        restricted = rng.normal(0.0, 0.05, n)
        unrestricted = restricted + rng.normal(0.0, 0.05, n)
        return actual, restricted, unrestricted

    assert _cw_rejection_rate(builder, seed=1) <= 0.20


def test_clark_west_oversized_when_forecasts_are_not_nested():
    """Guards the failure mode the estimator/nesting gates exist to prevent.

    Two disjoint signal families with identical true skill are not nested. The
    adjustment term is then pure model-specification difference, and Clark-West
    rejects essentially always. This test documents *why* callers must gate the
    test rather than asserting the statistic is trustworthy here.
    """

    def builder(rng, n):
        family_a = rng.normal(0.0, 1.0, n)
        family_b = rng.normal(0.0, 1.0, n)
        actual = 0.05 * family_a + 0.05 * family_b + rng.normal(0.0, 1.0, n)
        return actual, 0.05 * family_a, 0.05 * family_b

    assert _cw_rejection_rate(builder, seed=2) >= 0.80


def test_clark_west_test_positive_mean_favors_unrestricted_forecast():
    dates = pd.date_range("2018-01-31", periods=36, freq="ME")
    actual = pd.Series(np.sin(np.arange(36) / 3.0))
    restricted = pd.Series(np.zeros(36))
    unrestricted = actual.mul(0.9)

    result = clark_west_test(
        actual,
        restricted,
        unrestricted,
        dates=dates,
        maxlags=3,
    )

    assert result["months"] == 36
    assert result["mean_difference"] > 0
    assert result["t_stat"] > 0
    assert 0.0 <= result["p_one_sided"] <= 1.0


def test_stationary_bootstrap_metric_ci_reports_level_metric_interval():
    values = pd.Series([0.01, 0.02, -0.005, 0.015, 0.0, 0.012, -0.004, 0.018])

    result = stationary_bootstrap_metric_ci(
        values,
        metric="annualized_mean",
        expected_block=3,
        n_boot=200,
        seed=11,
    )

    assert result["observations"] == len(values)
    assert np.isfinite(result["point"])
    assert result["ci_low"] < result["ci_high"]
    assert 0.0 <= result["p_two_sided_zero"] <= 1.0


def test_ledoit_wolf_close_to_jkm_under_iid_returns():
    from stats import jobson_korkie_memmel, ledoit_wolf_sharpe_test

    rng = np.random.default_rng(4)
    a = rng.normal(0.01, 0.04, size=240)
    b = 0.6 * a + rng.normal(0.004, 0.03, size=240)
    rf = np.zeros(240)

    lw = ledoit_wolf_sharpe_test(a, b, rf, maxlags=0)
    jkm = jobson_korkie_memmel(a, b, rf)

    # LW uses population moments; JKM uses ddof=1 — they differ by sqrt(n/(n-1)).
    assert lw["delta_sharpe_monthly"] == pytest.approx(
        jkm["delta_sharpe_monthly"] * np.sqrt(240 / 239), abs=1e-12
    )
    # With iid data and zero lags the HAC delta-method z should sit near JKM's.
    assert lw["z"] == pytest.approx(jkm["z"], rel=0.15)
    assert lw["p_two_sided"] == pytest.approx(jkm["p_two_sided"], abs=0.05)


def test_ledoit_wolf_is_antisymmetric_and_zero_for_identical_series():
    from stats import ledoit_wolf_sharpe_test

    rng = np.random.default_rng(5)
    a = rng.normal(0.008, 0.05, size=180)
    b = rng.normal(0.002, 0.05, size=180)
    rf = np.zeros(180)

    forward = ledoit_wolf_sharpe_test(a, b, rf)
    backward = ledoit_wolf_sharpe_test(b, a, rf)
    self_test = ledoit_wolf_sharpe_test(a, a.copy(), rf)

    assert forward["z"] == pytest.approx(-backward["z"], abs=1e-10)
    assert forward["delta_sharpe_annualized"] == pytest.approx(
        forward["delta_sharpe_monthly"] * np.sqrt(12.0)
    )
    assert self_test["delta_sharpe_monthly"] == pytest.approx(0.0, abs=1e-15)


def test_ledoit_wolf_handles_short_and_degenerate_input():
    from stats import ledoit_wolf_sharpe_test

    short = ledoit_wolf_sharpe_test([0.01] * 6, [0.02] * 6, [0.0] * 6)
    constant = ledoit_wolf_sharpe_test([0.01] * 60, list(np.random.default_rng(6).normal(size=60)), [0.0] * 60)

    assert np.isnan(short["z"]) and short["n"] == 6
    assert np.isnan(constant["z"])  # zero-variance leg cannot be tested
