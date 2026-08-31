from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing import (  # noqa: E402
    PanelConfig,
    add_cross_sectional_ranks,
    build_clean_universe,
    clean_monthly_returns,
    compute_market_features,
    prepare_fundamental_features,
)


def test_clean_monthly_returns_converts_percent_and_removes_data_errors():
    values = pd.Series([-99.0, 5.0, 1000.0, 1001.0, np.nan])

    result = clean_monthly_returns(values, maximum_monthly_return=10.0)

    assert result.iloc[0] == -0.99
    assert result.iloc[1] == 0.05
    assert result.iloc[2] == 10.0
    assert pd.isna(result.iloc[3])
    assert pd.isna(result.iloc[4])


def test_accounting_values_become_available_six_months_after_period_end():
    frame = pd.DataFrame(
        {
            "Instrument": ["AAA"],
            "TR.F.TOTASSETS": [100.0],
            "TR.F.TOTASSETS.DATE": ["2023-12-31"],
            "TR.F.TOTLIAB": [40.0],
            "TR.F.TOTLIAB.DATE": ["2023-12-31"],
            "TR.F.TOTSHHOLDEQ": [60.0],
            "TR.F.TOTSHHOLDEQ.DATE": ["2023-12-31"],
            "TR.F.TOTREVENUE": [80.0],
            "TR.F.TOTREVENUE.DATE": ["2023-12-31"],
            "TR.F.OPPROFBEFNONRECURINCEXPN": [12.0],
            "TR.F.OPPROFBEFNONRECURINCEXPN.DATE": ["2023-12-31"],
            "TR.F.INCBEFDISCOPSEXORDITEMS": [8.0],
            "TR.F.INCBEFDISCOPSEXORDITEMS.DATE": ["2023-12-31"],
            "TR.F.NETCASHFLOWOP": [10.0],
            "TR.F.NETCASHFLOWOP.DATE": ["2023-12-31"],
            "TR.F.CAPEXTOT": [5.0],
            "TR.F.CAPEXTOT.DATE": ["2023-12-31"],
        }
    )

    result, audit = prepare_fundamental_features(frame, PanelConfig())

    assert result.loc[0, "available_date"] == pd.Timestamp("2024-06-30")
    assert result.loc[0, "profitability_roa"] == 0.08
    assert result.loc[0, "accruals"] == -0.02
    assert audit["rows_with_disagreeing_field_dates"] == 0


def test_market_target_does_not_jump_across_missing_calendar_month():
    monthly = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-03-31"]),
            "ric": ["AAA", "AAA"],
            "company_market_cap": [100.0, 110.0],
            "price_close": [10.0, 11.0],
            "total_return_1m": [1.0, 2.0],
            "shares_outstanding": [10.0, 10.0],
            "volume": [1.0, 1.0],
        }
    )

    result = compute_market_features(monthly, PanelConfig())

    assert pd.isna(result.loc[0, "target_return_1m"])


def test_cross_sectional_rank_imputes_missing_to_median():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "ric": ["AAA", "BBB"],
            "eligible": [True, True],
            "company_market_cap": [100.0, 200.0],
            "target_return_1m": [0.01, 0.02],
            "return_history_n": [24, 24],
            **{
                feature: [1.0, np.nan] if feature == "log_size" else [1.0, 2.0]
                for feature in [
                    "log_size",
                    "book_to_market",
                    "return_1m",
                    "momentum_6_2",
                    "momentum_12_2",
                    "volatility_12m",
                    "max_return_12m",
                    "market_cap_growth_12m",
                    "turnover_1m",
                    "turnover_12m",
                    "asset_growth",
                    "sales_growth",
                    "profitability_roa",
                    "operating_profitability",
                    "leverage",
                    "accruals",
                    "capex_to_assets",
                    "cashflow_to_assets",
                ]
            },
        }
    )

    result = add_cross_sectional_ranks(panel, PanelConfig(microcap_quantile=0.0))

    assert result.loc[result["ric"].eq("BBB"), "log_size_rank"].iloc[0] == 0.0
    assert result["model_eligible"].all()


def test_clean_universe_retains_inactive_equity_with_valid_history():
    universe = pd.DataFrame(
        {
            "ric": ["AAA^A24"],
            "TR.RIC": ["AAA^A24"],
            "TR.ISIN": ["GB0000000001"],
            "TR.TRBCECONOMICSECTOR": ["Industrials"],
            "screen_state": ["inactive"],
            "screen_country": ["GB"],
        }
    )
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    monthly = pd.DataFrame(
        {
            "ric": "AAA^A24",
            "date": dates,
            "total_return_1m": 1.0,
            "company_market_cap": 100.0,
        }
    )
    fundamentals = pd.DataFrame(
        {
            "Instrument": ["AAA^A24"],
            "TR.F.TOTASSETS": [100.0],
        }
    )
    supplement = pd.DataFrame(
        {
            "Instrument": ["AAA^A24"],
            "TR.INSTRUMENTTYPE": ["Ordinary Shares"],
            "TR.ISPRIMARYQUOTE": [0],
        }
    )

    result = build_clean_universe(
        universe,
        monthly,
        fundamentals,
        PanelConfig(),
        supplement,
    )

    assert result.loc[0, "eligible"]
    assert not result.loc[0, "non_primary_quote"]


def test_clean_universe_excludes_etf_even_with_price_and_accounting_coverage():
    universe = pd.DataFrame(
        {
            "ric": ["ETF.L"],
            "TR.RIC": ["ETF.L"],
            "TR.ISIN": ["GB0000000002"],
            "TR.TRBCECONOMICSECTOR": ["Financials"],
            "screen_state": ["active"],
            "screen_country": ["GB"],
        }
    )
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    monthly = pd.DataFrame(
        {
            "ric": "ETF.L",
            "date": dates,
            "total_return_1m": 1.0,
            "company_market_cap": 100.0,
        }
    )
    fundamentals = pd.DataFrame(
        {"Instrument": ["ETF.L"], "TR.F.TOTASSETS": [100.0]}
    )
    supplement = pd.DataFrame(
        {
            "Instrument": ["ETF.L"],
            "TR.INSTRUMENTTYPE": ["Equity ETFs"],
            "TR.ISPRIMARYQUOTE": [1],
        }
    )

    result = build_clean_universe(
        universe, monthly, fundamentals, PanelConfig(), supplement
    )

    assert not result.loc[0, "eligible"]
    assert result.loc[0, "exclusion_reason"] == "non_common_equity"


def test_market_characteristics_are_unchanged_when_future_months_are_appended():
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    monthly = pd.DataFrame(
        {
            "date": dates,
            "ric": "AAA",
            "company_market_cap": np.linspace(100.0, 130.0, len(dates)),
            "price_close": np.linspace(10.0, 13.0, len(dates)),
            "total_return_1m": np.linspace(-2.0, 3.0, len(dates)),
            "shares_outstanding": 10.0,
            "volume": 1.0,
        }
    )
    cutoff = pd.Timestamp("2021-12-31")

    full = compute_market_features(monthly, PanelConfig())
    truncated = compute_market_features(
        monthly[monthly["date"].le(cutoff)], PanelConfig()
    )
    feature_columns = [
        "date",
        "log_size",
        "return_1m",
        "momentum_6_2",
        "momentum_12_2",
        "volatility_12m",
        "max_return_12m",
        "market_cap_growth_12m",
        "turnover_1m",
        "turnover_12m",
    ]

    pd.testing.assert_frame_equal(
        full.loc[full["date"].le(cutoff), feature_columns].reset_index(drop=True),
        truncated[feature_columns].reset_index(drop=True),
    )


def test_monthly_liquidity_extensions_are_currency_comparable_and_causal():
    dates = pd.date_range("2020-01-31", periods=14, freq="ME")
    turnover = np.arange(1.0, 15.0) / 100.0
    shares = 100.0
    monthly = pd.DataFrame(
        {
            "date": dates,
            "ric": "AAA",
            "company_market_cap": 1_000_000.0,
            "price_close": 10.0,
            "total_return_1m": 1.0,
            "shares_outstanding": shares,
            "volume": turnover * shares,
        }
    )

    full = compute_market_features(monthly, PanelConfig())
    truncated = compute_market_features(monthly.iloc[:12], PanelConfig())

    assert np.isclose(
        full.loc[11, "log_trading_value_eur"],
        np.log(1_000_000.0 * turnover[:12].mean()),
    )
    assert np.isclose(
        full.loc[11, "turnover_volatility_12m"],
        np.std(turnover[:12], ddof=1),
    )
    pd.testing.assert_series_equal(
        full.loc[:11, "turnover_volatility_12m"].reset_index(drop=True),
        truncated["turnover_volatility_12m"].reset_index(drop=True),
    )
