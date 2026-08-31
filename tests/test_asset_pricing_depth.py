from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing_depth import (  # noqa: E402
    DepthConfig,
    build_internal_eur_factors,
    build_internal_market,
    estimate_rolling_risk,
    factor_spanning_tests,
    fama_macbeth_tests,
)


def test_internal_market_uses_prior_month_market_cap():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ric": ["A"] * 3 + ["B"] * 3,
            "return_1m": [0.0, 0.10, 0.0, 0.0, 0.0, 0.20],
            "company_market_cap": [100.0, 1000.0, 1000.0, 300.0, 300.0, 300.0],
        }
    )

    market = build_internal_market(panel).set_index("date")

    # February weights use January caps: A=25%, B=75%.
    assert np.isclose(market.loc[dates[1], "market_return_eur"], 0.025)
    # March weights use February caps: A=1000/1300, B=300/1300.
    assert np.isclose(
        market.loc[dates[2], "market_return_eur"],
        0.20 * 300.0 / 1300.0,
    )


def test_rolling_beta_recovers_known_loading_and_is_causal():
    dates = pd.date_range("2018-01-31", periods=48, freq="ME")
    market_return = np.linspace(-0.08, 0.09, len(dates))
    panel = pd.DataFrame(
        {
            "date": dates,
            "ric": "A",
            "return_1m": 0.005 + 2.0 * market_return,
        }
    )
    market = pd.DataFrame(
        {"date": dates, "market_return_eur": market_return}
    )

    full = estimate_rolling_risk(panel, market, window=36, minimum=24)
    truncated = estimate_rolling_risk(
        panel.iloc[:36],
        market.iloc[:36],
        window=36,
        minimum=24,
    )

    assert np.isclose(full.dropna().iloc[-1]["beta_36m"], 2.0)
    pd.testing.assert_series_equal(
        full.set_index("date").loc[dates[:36], "beta_36m"],
        truncated.set_index("date")["beta_36m"],
        check_names=False,
    )


def _factor_panel() -> pd.DataFrame:
    records = []
    for date in pd.date_range("2018-01-31", periods=30, freq="ME"):
        for security in range(120):
            size = security + 1.0
            style = (security % 30) / 29.0
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security:03d}",
                    "target_return_1m": 0.01 * style,
                    "company_market_cap": size,
                    "market_cap_percentile": (security + 1) / 120,
                    "book_to_market": style,
                    "operating_profitability": style,
                    "asset_growth": 1.0 - style,
                    "momentum_12_2": style,
                }
            )
    return pd.DataFrame(records)


def test_internal_factors_use_next_month_returns_and_have_expected_signs():
    panel = _factor_panel()
    rf = pd.Series(
        0.001,
        index=pd.date_range("2018-02-28", periods=30, freq="ME"),
    )

    factors = build_internal_eur_factors(panel, rf)

    assert (factors["HML_EUR"] > 0).all()
    assert (factors["RMW_EUR"] > 0).all()
    assert (factors["CMA_EUR"] > 0).all()
    assert (factors["MOM_EUR"] > 0).all()
    assert (
        factors["return_date"]
        == factors["signal_date"] + pd.offsets.MonthEnd(1)
    ).all()


def test_fama_macbeth_detects_incremental_score_slope():
    records = []
    rng = np.random.default_rng(7)
    for date in pd.date_range("2018-01-31", periods=36, freq="ME"):
        for security in range(150):
            score = (security - 74.5) / 75.0
            records.append(
                {
                    "model": "ridge_rank",
                    "date": date,
                    "target_return_1m": 0.02 * score
                    + rng.normal(0, 0.002),
                    "prediction_rank": score,
                    "momentum_12_2_rank": rng.uniform(-1, 1),
                    "log_size_rank": rng.uniform(-1, 1),
                    "book_to_market_rank": rng.uniform(-1, 1),
                    "beta_rank": rng.uniform(-1, 1),
                    "idio_vol_rank": rng.uniform(-1, 1),
                    "screen_country": "GB" if security % 2 else "DE",
                    "TR.TRBCECONOMICSECTOR": (
                        "Industrials" if security % 3 else "Technology"
                    ),
                }
            )
    predictions = pd.DataFrame(records)

    _, summary = fama_macbeth_tests(
        predictions,
        DepthConfig(minimum_cross_section=100),
    )
    full = summary[
        summary["specification"].eq(
            "characteristics_risk_country_sector"
        )
    ].iloc[0]

    assert full["mean_monthly_score_slope"] > 0.015
    assert full["p_value"] < 0.01


def test_factor_spanning_recovers_strategy_alpha():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2015-01-31", periods=72, freq="ME")
    factors = pd.DataFrame(
        {
            "signal_date": dates,
            "return_date": dates + pd.offsets.MonthEnd(1),
            "RF_EUR": 0.001,
            "MKT_RF_EUR": rng.normal(0.005, 0.03, len(dates)),
            "SMB_EUR": rng.normal(0.0, 0.02, len(dates)),
            "HML_EUR": rng.normal(0.0, 0.02, len(dates)),
            "RMW_EUR": rng.normal(0.0, 0.02, len(dates)),
            "CMA_EUR": rng.normal(0.0, 0.02, len(dates)),
            "MOM_EUR": rng.normal(0.0, 0.02, len(dates)),
        }
    )
    base = 0.5 * factors["MKT_RF_EUR"].to_numpy()
    rows = []
    for model, alpha in [("momentum_rank", 0.0), ("mlp_return", 0.01)]:
        for i, date in enumerate(dates):
            strategy_return = alpha + base[i]
            rows.append(
                {
                    "model": model,
                    "target_mode": "rank",
                    "weighting": "value",
                    "universe_variant": "standard_ex_bottom_5pct",
                    "signal_date": date,
                    "return_date": date + pd.offsets.MonthEnd(1),
                    "long_return": strategy_return + 0.001,
                    "gross_long_short_return": strategy_return,
                    "long_short_turnover": 0.0,
                    "long_only_turnover": 0.0,
                }
            )

    result = factor_spanning_tests(
        pd.DataFrame(rows),
        factors,
        DepthConfig(),
    )
    mlp = result[
        result["comparison"].eq("absolute")
        & result["model"].eq("mlp_return")
        & result["portfolio"].eq("long_short")
    ].iloc[0]

    assert np.isclose(mlp["alpha_monthly"], 0.01)
    assert mlp["alpha_p"] < 0.01
