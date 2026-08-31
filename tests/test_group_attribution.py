"""Tests for Brinson-Fachler country/sector attribution."""
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

import group_attribution as ga  # noqa: E402


def _benchmark_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31"] * 3 + ["2020-02-29"] * 3
            ),
            "group": ["DE", "FR", "GB"] * 2,
            "benchmark_weight": [0.5, 0.3, 0.2, 0.4, 0.4, 0.2],
            "benchmark_return": [0.02, -0.01, 0.03, 0.01, 0.02, -0.02],
        }
    )


def _portfolio_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31"] * 3 + ["2020-02-29"] * 3
            ),
            "group": ["DE", "FR", "GB"] * 2,
            "portfolio_weight": [0.7, 0.1, 0.2, 0.2, 0.5, 0.3],
            "portfolio_return": [0.03, -0.01, 0.05, 0.00, 0.04, -0.01],
        }
    )


def test_effects_sum_to_gross_active_return():
    attribution = ga.brinson_attribution(_portfolio_frame(), _benchmark_frame())
    monthly = ga.monthly_effect_series(attribution)

    portfolio = _portfolio_frame()
    benchmark = _benchmark_frame()
    expected = []
    for date in sorted(portfolio["date"].unique()):
        p = portfolio[portfolio["date"] == date]
        b = benchmark[benchmark["date"] == date]
        r_p = float((p["portfolio_weight"] * p["portfolio_return"]).sum())
        r_b = float((b["benchmark_weight"] * b["benchmark_return"]).sum())
        expected.append(r_p - r_b)

    np.testing.assert_allclose(
        monthly["gross_active_return"].to_numpy(), np.array(expected), atol=1e-12
    )


def test_identity_holds_when_portfolio_holds_off_benchmark_group():
    portfolio = _portfolio_frame()
    extra = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"]),
            "group": ["NO"],
            "portfolio_weight": [0.2],
            "portfolio_return": [0.06],
        }
    )
    # Rescale January so the portfolio still sums to one.
    january = portfolio["date"] == pd.Timestamp("2020-01-31")
    portfolio.loc[january, "portfolio_weight"] *= 0.8
    portfolio = pd.concat([portfolio, extra], ignore_index=True)

    attribution = ga.brinson_attribution(portfolio, _benchmark_frame())
    monthly = ga.monthly_effect_series(attribution)

    p = portfolio[portfolio["date"] == pd.Timestamp("2020-01-31")]
    b = _benchmark_frame()
    b = b[b["date"] == pd.Timestamp("2020-01-31")]
    expected = float((p["portfolio_weight"] * p["portfolio_return"]).sum()) - float(
        (b["benchmark_weight"] * b["benchmark_return"]).sum()
    )

    january_row = monthly[monthly["date"] == pd.Timestamp("2020-01-31")].iloc[0]
    assert january_row["gross_active_return"] == pytest.approx(expected, abs=1e-12)
    assert january_row["off_benchmark_weight"] == pytest.approx(0.2, abs=1e-12)


def test_pure_allocation_bet_has_zero_selection():
    """Matching every group return isolates the effect into allocation."""
    benchmark = _benchmark_frame()
    portfolio = benchmark.rename(
        columns={
            "benchmark_weight": "portfolio_weight",
            "benchmark_return": "portfolio_return",
        }
    ).copy()
    # Tilt weights but keep group returns identical to the benchmark.
    portfolio["portfolio_weight"] = [0.7, 0.1, 0.2, 0.2, 0.5, 0.3]

    attribution = ga.brinson_attribution(portfolio, benchmark)
    monthly = ga.monthly_effect_series(attribution)

    assert monthly["selection"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert monthly["interaction"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert monthly["allocation"].abs().max() > 0.0


def test_pure_selection_has_zero_allocation():
    """Matching benchmark weights isolates the effect into selection."""
    benchmark = _benchmark_frame()
    portfolio = benchmark.rename(
        columns={
            "benchmark_weight": "portfolio_weight",
            "benchmark_return": "portfolio_return",
        }
    ).copy()
    portfolio["portfolio_return"] = portfolio["portfolio_return"] + 0.01

    attribution = ga.brinson_attribution(portfolio, benchmark)
    monthly = ga.monthly_effect_series(attribution)

    assert monthly["allocation"].abs().max() == pytest.approx(0.0, abs=1e-15)
    assert monthly["interaction"].abs().max() == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(
        monthly["selection"].to_numpy(), np.array([0.01, 0.01]), atol=1e-12
    )


def test_benchmark_group_panel_aggregates_to_internal_market():
    """Cap-weighted group returns must reproduce build_internal_market."""
    sys.path.insert(0, str(SRC_DIR))
    from asset_pricing_depth import build_internal_market

    dates = pd.to_datetime(["2019-12-31", "2020-01-31", "2020-02-29"])
    rows = []
    rng = np.random.default_rng(7)
    for ric, country in [("A.DE", "DE"), ("B.FR", "FR"), ("C.GB", "GB"), ("D.DE", "DE")]:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "ric": ric,
                    "screen_country": country,
                    "return_1m": float(rng.normal(0.0, 0.05)),
                    "company_market_cap": float(rng.uniform(1e9, 5e9)),
                }
            )
    panel = pd.DataFrame(rows)

    groups = ga.build_benchmark_group_panel(panel, "screen_country")
    reconstructed = (
        groups.assign(
            weighted=groups["benchmark_weight"] * groups["benchmark_return"]
        )
        .groupby("date")["weighted"]
        .sum()
    )
    market = build_internal_market(panel).set_index("date")["market_return_eur"]

    common = reconstructed.index.intersection(market.index)
    assert len(common) >= 2
    np.testing.assert_allclose(
        reconstructed.loc[common].to_numpy(),
        market.loc[common].to_numpy(),
        atol=1e-12,
    )


def test_benchmark_group_weights_sum_to_one_each_month():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"] * 3),
            "ric": ["A", "A", "B", "B", "C", "C"],
            "screen_country": ["DE", "DE", "FR", "FR", "GB", "GB"],
            "return_1m": [0.01, 0.02, -0.01, 0.03, 0.00, 0.01],
            "company_market_cap": [1e9, 1.1e9, 2e9, 2.1e9, 3e9, 3.1e9],
        }
    )
    groups = ga.build_benchmark_group_panel(panel, "screen_country")
    totals = groups.groupby("date")["benchmark_weight"].sum()
    np.testing.assert_allclose(totals.to_numpy(), np.ones(len(totals)), atol=1e-12)


def test_portfolio_group_exposures_weights_returns():
    holdings = pd.DataFrame(
        {
            "screen_country": ["DE", "DE", "FR", None],
            "weight": [0.3, 0.2, 0.4, 0.1],
            "target_return_1m": [0.02, 0.04, -0.01, 0.05],
        }
    )
    exposures = ga.portfolio_group_exposures(holdings, "screen_country")
    by_group = exposures.set_index("group")

    assert by_group.loc["DE", "portfolio_weight"] == pytest.approx(0.5)
    # Weighted average: (0.3*0.02 + 0.2*0.04) / 0.5
    assert by_group.loc["DE", "portfolio_return"] == pytest.approx(0.028)
    assert by_group.loc["UNKNOWN", "portfolio_weight"] == pytest.approx(0.1)


def test_summarize_effects_reports_hac_tests():
    rng = np.random.default_rng(3)
    monthly = pd.DataFrame(
        {
            "date": pd.date_range("2015-01-31", periods=120, freq="ME"),
            "allocation": rng.normal(0.0, 0.002, 120),
            "selection": rng.normal(0.004, 0.002, 120),
            "interaction": rng.normal(0.0, 0.001, 120),
        }
    )
    monthly["gross_active_return"] = monthly[
        ["allocation", "selection", "interaction"]
    ].sum(axis=1)

    summary = ga.summarize_effects(monthly).set_index("effect")
    assert summary.loc["selection", "hac_p_value"] < 0.01
    assert summary.loc["allocation", "hac_p_value"] > 0.05
    assert summary.loc["selection", "annualized"] == pytest.approx(
        monthly["selection"].mean() * 12, rel=1e-9
    )
    shares = summary.loc[list(ga.EFFECT_COLUMNS), "share_of_active"]
    assert shares.sum() == pytest.approx(1.0, rel=1e-9)
