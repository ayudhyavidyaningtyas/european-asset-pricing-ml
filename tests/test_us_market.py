from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from us_market import (  # noqa: E402
    build_wrds_compustat_us_annual_sql,
    build_wrds_compustat_us_monthly_sql,
    cusip_to_us_isin,
    normalize_wrds_compustat_us_annual,
    normalize_wrds_compustat_us_monthly,
)


def test_cusip_to_us_isin_matches_known_large_caps():
    assert cusip_to_us_isin("037833100") == "US0378331005"
    assert cusip_to_us_isin("594918104") == "US5949181045"


def test_wrds_compustat_us_annual_normalization_derives_isin_and_filters_us_standard():
    annual = pd.DataFrame(
        [
            {
                "gvkey": "001690",
                "datadate": "2023-09-30",
                "cusip": "037833100",
                "indfmt": "INDL",
                "datafmt": "STD",
                "popsrc": "D",
                "consol": "C",
                "fic": "USA",
            },
            {
                "gvkey": "099999",
                "datadate": "2023-12-31",
                "cusip": "123456789",
                "indfmt": "FS",
                "datafmt": "STD",
                "popsrc": "D",
                "consol": "C",
                "fic": "USA",
            },
        ]
    )

    normalized = normalize_wrds_compustat_us_annual(annual)

    assert normalized["gvkey"].tolist() == ["001690"]
    assert normalized["isin"].tolist() == ["US0378331005"]


def test_wrds_compustat_us_monthly_normalization_keeps_usd_rows():
    monthly = pd.DataFrame(
        [
            {
                "gvkey": "001690",
                "iid": "01",
                "datadate": "2024-01-31",
                "cusip": "037833100",
                "curcddvm": "USD",
            },
            {
                "gvkey": "001690",
                "iid": "01",
                "datadate": "2024-02-29",
                "cusip": "037833100",
                "curcddvm": "EUR",
            },
        ]
    )

    normalized = normalize_wrds_compustat_us_monthly(monthly)

    assert normalized["datadate"].tolist() == ["2024-01-31"]
    assert normalized["isin"].tolist() == ["US0378331005"]


def test_wrds_compustat_sql_contains_expected_filters():
    annual_sql = build_wrds_compustat_us_annual_sql(
        schema="comp",
        start="2001-01-01",
        end="2001-12-31",
    )
    monthly_sql = build_wrds_compustat_us_monthly_sql(
        schema="comp",
        start="2001-01-01",
        end="2001-12-31",
    )

    assert "from comp.funda" in annual_sql
    assert "f.fic = 'USA'" in annual_sql
    assert "date '2001-01-01'" in annual_sql
    assert "from comp.secm" in monthly_sql
    assert "s.curcddvm = 'USD'" in monthly_sql
    assert "s.exchg in (11, 12, 14)" in monthly_sql
