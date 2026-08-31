"""Tests for the group-exposure side channel and the attribution driver."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [SRC_DIR, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import group_attribution as ga  # noqa: E402
from run_constrained_deep_hybrid_long_only import (  # noqa: E402
    group_exposure_records,
)
import run_country_sector_attribution as driver  # noqa: E402


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ric": ["A.DE", "B.DE", "C.FR", "D.GB"],
            "screen_country": ["DE", "DE", "FR", "GB"],
            "TR.TRBCECONOMICSECTOR": [
                "Industrials",
                "Financials",
                "Industrials",
                "Technology",
            ],
            "target_return_1m": [0.02, 0.04, -0.01, 0.03],
        }
    )


def test_group_exposure_records_weights_and_returns():
    weights = {"A.DE": 0.3, "B.DE": 0.2, "C.FR": 0.4, "D.GB": 0.1}
    rows = group_exposure_records(
        _universe(),
        weights,
        strategy="s",
        constraint="c",
        date=pd.Timestamp("2020-01-31"),
        target_date=pd.Timestamp("2020-02-29"),
        group_columns=("screen_country", "TR.TRBCECONOMICSECTOR"),
    )
    frame = pd.DataFrame(rows)

    countries = frame[frame["group_kind"] == "screen_country"].set_index("group")
    assert countries.loc["DE", "portfolio_weight"] == pytest.approx(0.5)
    # (0.3*0.02 + 0.2*0.04) / 0.5
    assert countries.loc["DE", "portfolio_return"] == pytest.approx(0.028)
    assert countries.loc["DE", "portfolio_n"] == 2

    # Each group kind must partition the whole portfolio.
    for kind, part in frame.groupby("group_kind"):
        assert part["portfolio_weight"].sum() == pytest.approx(1.0), kind


def test_group_exposure_return_reconciles_to_portfolio_return():
    weights = {"A.DE": 0.3, "B.DE": 0.2, "C.FR": 0.4, "D.GB": 0.1}
    universe = _universe()
    expected = float(
        sum(
            weights[row.ric] * row.target_return_1m
            for row in universe.itertuples()
        )
    )
    rows = group_exposure_records(
        universe,
        weights,
        strategy="s",
        constraint="c",
        date=pd.Timestamp("2020-01-31"),
        target_date=pd.Timestamp("2020-02-29"),
        group_columns=("screen_country",),
    )
    frame = pd.DataFrame(rows)
    realized = float(
        (frame["portfolio_weight"] * frame["portfolio_return"]).sum()
    )
    assert realized == pytest.approx(expected, abs=1e-12)


def test_simulate_constrained_signature_is_backwards_compatible():
    """Existing callers must keep working without the sink."""
    import inspect

    from run_constrained_deep_hybrid_long_only import simulate_constrained

    signature = inspect.signature(simulate_constrained)
    assert signature.parameters["exposure_sink"].default is None
    assert signature.parameters["exposure_group_columns"].default == (
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
    )


def test_attribute_strategy_end_to_end():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    exposures = pd.DataFrame(
        {
            "strategy": ["s"] * 4,
            "constraint": ["c"] * 4,
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "target_date": [dates[0], dates[0], dates[1], dates[1]],
            "group_kind": ["screen_country"] * 4,
            "group": ["DE", "FR", "DE", "FR"],
            "portfolio_weight": [0.6, 0.4, 0.5, 0.5],
            "portfolio_return": [0.03, -0.01, 0.02, 0.01],
            "portfolio_n": [2, 2, 2, 2],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "group": ["DE", "FR", "DE", "FR"],
            "benchmark_weight": [0.5, 0.5, 0.5, 0.5],
            "benchmark_return": [0.02, -0.02, 0.01, 0.02],
        }
    )
    attribution, monthly, summary = driver.attribute_strategy(
        exposures, {"screen_country": benchmark}, hac_lags=1
    )
    assert not attribution.empty and len(monthly) == 2
    assert set(monthly["group_kind"]) == {"country"}

    for date in dates:
        p = exposures[exposures["target_date"] == date]
        b = benchmark[benchmark["date"] == date]
        expected = float(
            (p["portfolio_weight"] * p["portfolio_return"]).sum()
        ) - float((b["benchmark_weight"] * b["benchmark_return"]).sum())
        row = monthly[monthly["date"] == date].iloc[0]
        assert row["gross_active_return"] == pytest.approx(expected, abs=1e-12)


def test_reconcile_costs_bridges_gross_to_net():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29"])
    monthly_attribution = pd.DataFrame(
        {
            "strategy": ["s", "s"],
            "constraint": ["c", "c"],
            "group_kind": ["country", "country"],
            "date": dates,
            "allocation": [0.001, 0.002],
            "selection": [0.004, 0.003],
            "interaction": [0.000, 0.001],
            "gross_active_return": [0.005, 0.006],
        }
    )
    constrained_monthly = pd.DataFrame(
        {
            "strategy": ["s", "s"],
            "constraint": ["c", "c"],
            "target_date": dates,
            "gross_return": [0.010, 0.012],
            "net_return_100m": [0.009, 0.010],
        }
    )
    bridge = driver.reconcile_costs(
        monthly_attribution, constrained_monthly, (1e8,)
    )
    row = bridge.iloc[0]
    # Cost drag averages (0.001 + 0.002) / 2 = 0.0015 monthly.
    assert row["annualized_cost_drag_100m"] == pytest.approx(0.0015 * 12)
    assert row["annualized_net_active_100m"] == pytest.approx(
        (0.005 - 0.001 + 0.006 - 0.002) / 2 * 12
    )
    assert row["annualized_gross_active"] == pytest.approx(0.0055 * 12)


def test_group_kinds_cover_country_and_sector():
    assert driver.GROUP_KINDS["screen_country"] == "country"
    assert driver.GROUP_KINDS["TR.TRBCECONOMICSECTOR"] == "sector"
