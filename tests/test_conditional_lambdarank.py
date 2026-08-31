import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conditional_lambdarank import (  # noqa: E402
    _forward_sum,
    _within_month_relevance,
)


def test_forward_sum_requires_contiguous_months():
    dates = pd.Series(
        pd.to_datetime(["2020-01-31", "2020-02-29", "2020-04-30"])
    )
    values = pd.Series([0.01, 0.02, 0.03])

    result, end_dates = _forward_sum(values, dates, 2)

    assert np.isclose(result.iloc[0], 0.03)
    assert pd.isna(result.iloc[1])
    assert end_dates.iloc[0] == pd.Timestamp("2020-02-29")


def test_relevance_is_month_local_and_bounded():
    dates = pd.Series(
        pd.to_datetime(["2020-01-31"] * 10 + ["2020-02-29"] * 10)
    )
    values = pd.Series(list(range(10)) + list(range(100, 110)), dtype=float)

    relevance = _within_month_relevance(values, dates, 10)

    assert relevance.min() >= 0
    assert relevance.max() <= 9
    assert relevance.iloc[:10].tolist() == relevance.iloc[10:].tolist()


def test_forward_horizon_end_date_is_label_availability_date():
    dates = pd.Series(pd.date_range("2020-01-31", periods=6, freq="ME"))
    values = pd.Series(np.ones(6))

    result, end_dates = _forward_sum(values, dates, 3)

    assert result.notna().sum() == 4
    assert end_dates.iloc[0] == pd.Timestamp("2020-03-31")
    assert result.iloc[0] == 3.0
