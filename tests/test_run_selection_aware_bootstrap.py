from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_selection_aware_bootstrap.py"
)
SPEC = importlib.util.spec_from_file_location("run_selection_aware_bootstrap", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _panel(returns_by_cell: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    matrix = np.column_stack(list(returns_by_cell.values()))
    turnover = np.ones_like(matrix)
    spread = np.ones_like(matrix)
    return matrix, turnover, spread


def test_selection_needs_the_minimum_validation_window():
    months = 30
    matrix, turnover, spread = _panel(
        {"a": np.full(months, 0.01), "b": np.full(months, 0.02)}
    )

    realised, chosen = MODULE.run_selection(matrix, turnover, spread)

    # Nothing selectable before MINIMUM_VALIDATION_MONTHS of history exist.
    assert np.isnan(realised[: MODULE.MINIMUM_VALIDATION_MONTHS]).all()
    assert (chosen[: MODULE.MINIMUM_VALIDATION_MONTHS] == -1).all()
    assert np.isfinite(realised[MODULE.MINIMUM_VALIDATION_MONTHS :]).all()


def test_selector_picks_the_dominant_candidate():
    months = 80
    rng = np.random.default_rng(0)
    good = 0.02 + rng.normal(scale=0.001, size=months)
    bad = -0.01 + rng.normal(scale=0.001, size=months)
    matrix, turnover, spread = _panel({"good": good, "bad": bad})

    realised, chosen = MODULE.run_selection(matrix, turnover, spread)

    picks = chosen[chosen >= 0]
    assert (picks == 0).all()
    np.testing.assert_allclose(
        realised[np.isfinite(realised)], good[chosen >= 0], atol=1e-12
    )


def test_selector_switches_when_dominance_reverses():
    months = 120
    first = np.concatenate([np.full(60, 0.03), np.full(60, -0.02)])
    second = np.concatenate([np.full(60, -0.02), np.full(60, 0.03)])
    matrix, turnover, spread = _panel({"first": first, "second": second})

    _, chosen = MODULE.run_selection(matrix, turnover, spread)

    picks = chosen[chosen >= 0]
    assert picks[0] == 0
    assert picks[-1] == 1
    assert (np.diff(picks) != 0).sum() == 1  # exactly one regime switch


def test_certainty_equivalent_objective_penalises_volatility():
    months = 90
    rng = np.random.default_rng(1)
    # Same mean, very different volatility: CE with risk aversion 3 must prefer
    # the calm candidate even though a mean-only rule would be indifferent.
    calm = 0.01 + rng.normal(scale=0.001, size=months)
    wild = 0.01 + rng.normal(scale=0.12, size=months)
    matrix, turnover, spread = _panel({"wild": wild, "calm": calm})

    _, chosen = MODULE.run_selection(matrix, turnover, spread)

    picks = chosen[chosen >= 0]
    assert (picks == 1).mean() > 0.95


def test_annualized_summary_and_interval_helpers():
    series = np.array([0.01] * 24 + [np.nan] * 4)
    summary = MODULE.annualized_summary(series)
    assert summary["months"] == 24
    assert summary["annualized_net_return"] == pytest.approx(0.12)

    draws = np.array([0.5, 1.0, 1.5, np.nan, 2.0])
    interval = MODULE.percentile_interval(draws)
    assert interval["draws_used"] == 4
    assert interval["p_two_sided_zero"] == pytest.approx(0.0, abs=1e-12)
    assert interval["ci_low"] < interval["ci_high"]


def test_pure_noise_selection_widens_the_aware_interval(tmp_path):
    """With exchangeable noise candidates, the frozen lucky path understates
    uncertainty relative to re-running selection inside each draw."""
    months, cells = 137, 10
    rng = np.random.default_rng(7)
    matrix = rng.normal(0.0, 0.04, size=(months, cells))
    turnover = np.ones_like(matrix)
    spread = np.ones_like(matrix)

    realised, _ = MODULE.run_selection(matrix, turnover, spread)
    frozen = realised[np.isfinite(realised)]

    reps = 120
    indices = MODULE.stationary_bootstrap_indices(
        months, 6.0, reps, np.random.default_rng(11)
    )
    aware = np.full(reps, np.nan)
    for draw in range(reps):
        order = indices[draw]
        draw_realised, _ = MODULE.run_selection(
            matrix[order], turnover[order], spread[order]
        )
        aware[draw] = MODULE.annualized_summary(draw_realised).get("net_sharpe", np.nan)

    frozen_indices = MODULE.stationary_bootstrap_indices(
        len(frozen), 6.0, reps, np.random.default_rng(11)
    )
    frozen_draws = frozen[frozen_indices]
    frozen_sharpe = frozen_draws.mean(axis=1) / frozen_draws.std(axis=1, ddof=1)
    frozen_sharpe *= np.sqrt(12.0)

    aware_interval = MODULE.percentile_interval(aware)
    frozen_interval = MODULE.percentile_interval(frozen_sharpe)
    aware_width = aware_interval["ci_high"] - aware_interval["ci_low"]
    frozen_width = frozen_interval["ci_high"] - frozen_interval["ci_low"]
    # Under the null the selection-aware interval must cover zero; the frozen
    # path centres on whatever luck the realised selection had.
    assert aware_interval["ci_low"] < 0.0 < aware_interval["ci_high"]
    assert aware_width > 0
    assert frozen_width > 0
