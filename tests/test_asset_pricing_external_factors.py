from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing_external_factors import (  # noqa: E402
    currency_align_portfolios,
    external_factor_spanning,
    load_monthly_eurusd_return,
)


def test_monthly_fx_uses_last_daily_observation(tmp_path):
    path = tmp_path / "fx.csv"
    pd.DataFrame(
        {
            "date": [
                "2020-01-02",
                "2020-01-31",
                "2020-02-03",
                "2020-02-28",
            ],
            "value": [1.10, 1.20, 1.21, 1.32],
        }
    ).to_csv(path, index=False)

    result = load_monthly_eurusd_return(path)

    assert np.isclose(result.loc["2020-02-29"], 0.10)


def test_currency_conversion_distinguishes_long_only_and_long_short():
    date = pd.Timestamp("2020-02-29")
    portfolios = pd.DataFrame(
        {
            "model": ["m"],
            "signal_date": [pd.Timestamp("2020-01-31")],
            "return_date": [date],
            "gross_long_short_return": [0.10],
            "long_return": [0.10],
            "long_short_turnover": [0.0],
            "long_only_turnover": [0.0],
        }
    )
    fx = pd.Series([0.05], index=[date], name="EURUSD_return")

    result = currency_align_portfolios(portfolios, fx)

    assert np.isclose(result.loc[0, "net_long_short_usd"], 0.105)
    assert np.isclose(result.loc[0, "net_long_only_usd"], 0.155)


def test_external_factor_spanning_recovers_alpha():
    rng = np.random.default_rng(2)
    dates = pd.date_range("2015-02-28", periods=72, freq="ME")
    factors = pd.DataFrame(
        {
            "return_date": dates,
            "Mkt-RF": rng.normal(0, 0.03, len(dates)),
            "SMB": rng.normal(0, 0.02, len(dates)),
            "HML": rng.normal(0, 0.02, len(dates)),
            "RMW": rng.normal(0, 0.02, len(dates)),
            "CMA": rng.normal(0, 0.02, len(dates)),
            "WML": rng.normal(0, 0.02, len(dates)),
            "RF": 0.001,
        }
    )
    rows = []
    for model, alpha in [("momentum_rank", 0.0), ("mlp_return", 0.01)]:
        for i, return_date in enumerate(dates):
            strategy_return = alpha + 0.5 * factors.loc[i, "Mkt-RF"]
            rows.append(
                {
                    "model": model,
                    "weighting": "value",
                    "universe_variant": "standard_ex_bottom_5pct",
                    "signal_date": return_date - pd.offsets.MonthEnd(1),
                    "return_date": return_date,
                    "gross_long_short_return": strategy_return,
                    "long_return": strategy_return + 0.001,
                    "long_short_turnover": 0.0,
                    "long_only_turnover": 0.0,
                }
            )
    fx = pd.Series(0.0, index=dates)

    result = external_factor_spanning(
        pd.DataFrame(rows),
        factors,
        fx,
    )
    mlp = result[
        result["comparison"].eq("absolute")
        & result["model"].eq("mlp_return")
        & result["portfolio"].eq("long_short")
    ].iloc[0]

    assert np.isclose(mlp["alpha_monthly"], 0.01)
    assert mlp["alpha_p"] < 0.01
