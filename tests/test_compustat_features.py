from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing import ALL_RAW_FEATURES, PanelConfig  # noqa: E402
from asset_pricing_ml import (  # noqa: E402
    COMPUSTAT_FEATURE_COLUMNS,
    EXPANDED_FEATURE_COLUMNS,
)
from compustat_features import (  # noqa: E402
    COMPUSTAT_EXTENSION_FEATURES,
    build_compustat_enriched_panel,
    prepare_compustat_monthly_features,
)


def base_panel() -> pd.DataFrame:
    rows = []
    for ric, isin, market_cap in [
        ("AAA.L", "GB0000000001", 1000.0),
        ("BBB.L", "GB0000000002", 2000.0),
    ]:
        for date in pd.to_datetime(["2024-05-31", "2024-06-30", "2024-07-31"]):
            row = {
                "date": date,
                "ric": ric,
                "TR.ISIN": isin,
                "company_market_cap": market_cap,
                "eligible": True,
            }
            row.update({feature: 1.0 for feature in ALL_RAW_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def annual_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gvkey": "001",
                "isin": "GB0000000001",
                "datadate": "2022-12-31",
                "curcd": "GBP",
                "at": 100.0,
                "revt": 80.0,
                "sale": 80.0,
                "ceq": 50.0,
                "seq": 50.0,
                "lt": 40.0,
                "dltt": 10.0,
                "dlc": 5.0,
                "che": 20.0,
                "cogs": 50.0,
                "oiadp": 12.0,
                "nicon": 8.0,
                "oancf": 6.0,
                "capx": 4.0,
            },
            {
                "gvkey": "001",
                "isin": "GB0000000001",
                "datadate": "2023-12-31",
                "curcd": "GBP",
                "at": 120.0,
                "revt": 100.0,
                "sale": 100.0,
                "ceq": 60.0,
                "seq": 60.0,
                "lt": 50.0,
                "dltt": 12.0,
                "dlc": 6.0,
                "che": 24.0,
                "cogs": 60.0,
                "oiadp": 15.0,
                "nicon": 10.0,
                "oancf": 7.0,
                "capx": 5.0,
            },
            {
                "gvkey": "002",
                "isin": "GB0000000002",
                "datadate": "2023-12-31",
                "curcd": "GBP",
                "at": 200.0,
                "revt": 160.0,
                "sale": 160.0,
                "ceq": 120.0,
                "seq": 120.0,
                "lt": 70.0,
                "dltt": 20.0,
                "dlc": 5.0,
                "che": 30.0,
                "cogs": 100.0,
                "oiadp": 18.0,
                "nicon": 12.0,
                "oancf": 10.0,
                "capx": 6.0,
            },
        ]
    )


def monthly_rows() -> pd.DataFrame:
    dates = pd.date_range("2024-01-31", periods=7, freq="ME")
    records = []
    for idx, date in enumerate(dates, start=1):
        records.append(
            {
                "gvkey": "001",
                "isin": "GB0000000001",
                "iid": "01",
                "datadate": date,
                "prccm": 10.0 + idx,
                "ajexm": 1.0,
                "cshtrm": 1000.0 + idx,
            }
        )
        records.append(
            {
                "gvkey": "002",
                "isin": "GB0000000002",
                "iid": "01",
                "datadate": date,
                "prccm": 20.0 + idx,
                "ajexm": 1.0,
                "cshtrm": 2000.0 + idx,
            }
        )
    return pd.DataFrame(records)


def test_compustat_annual_features_respect_accounting_lag():
    _, _, panel, audit = build_compustat_enriched_panel(
        base_panel(),
        annual_rows(),
        monthly_rows(),
        PanelConfig(accounting_lag_months=6),
    )

    aaa_may = panel[panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-05-31"))].iloc[0]
    aaa_june = panel[panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-06-30"))].iloc[0]

    assert aaa_may["comp_period_end"] == pd.Timestamp("2022-12-31")
    assert np.isclose(aaa_june["comp_book_to_market"], 60.0 / 1000.0)
    assert aaa_june["comp_period_end"] == pd.Timestamp("2023-12-31")
    assert np.isclose(aaa_june["comp_asset_growth"], 0.20)
    assert panel["comp_book_to_market_rank"].notna().all()
    assert audit["panel"]["unique_rics_with_compustat_annual"] == 2


def test_compustat_monthly_features_do_not_jump_missing_calendar_month():
    monthly = pd.DataFrame(
        {
            "gvkey": ["001", "001", "001"],
            "isin": ["GB0000000001", "GB0000000001", "GB0000000001"],
            "iid": ["01", "01", "01"],
            "datadate": pd.to_datetime(["2024-01-31", "2024-02-29", "2024-04-30"]),
            "prccm": [10.0, 11.0, 12.0],
            "ajexm": [1.0, 1.0, 1.0],
            "cshtrm": [100.0, 110.0, 120.0],
        }
    )

    result, _ = prepare_compustat_monthly_features(monthly)

    april = result[result["date"].eq(pd.Timestamp("2024-04-30"))].iloc[0]
    assert pd.isna(april["comp_price_return_1m"])


def test_ml_feature_set_exposes_compustat_rank_columns():
    added = set(COMPUSTAT_FEATURE_COLUMNS) - set(EXPANDED_FEATURE_COLUMNS)

    assert added == {f"{feature}_rank" for feature in COMPUSTAT_EXTENSION_FEATURES}
