import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from implementable_frontier import (
    FrontierConfig,
    causal_strategy_selection,
    execution_cost,
    frontier_dominance,
    solve_aim_weights,
    trade_toward_aim,
)


def test_execution_cost_increases_with_aum_through_market_impact():
    delta = np.array([0.10, -0.10])
    spread = np.array([10.0, 10.0])
    adv = np.array([1_000_000.0, 1_000_000.0])
    volatility = np.array([0.10, 0.10])

    small = execution_cost(delta, spread, adv, volatility, 1_000_000.0, 0.10)
    large = execution_cost(delta, spread, adv, volatility, 100_000_000.0, 0.10)

    assert small[0] == large[0]
    assert large[1] > small[1] > 0
    assert large[2] > small[2]


def test_long_short_aim_respects_gross_net_beta_and_position_constraints():
    config = FrontierConfig(
        long_short_position_limit=0.20,
        beta_tolerance=0.01,
    )
    alpha = np.linspace(-0.01, 0.01, 20)
    beta = np.linspace(0.5, 1.5, 20)
    volatility = np.full(20, 0.10)

    weights = solve_aim_weights(
        alpha,
        beta,
        volatility,
        market_vol=0.05,
        risk_aversion=20.0,
        portfolio="long_short",
        config=config,
    )

    assert abs(weights.sum()) < 1e-6
    assert np.abs(weights).sum() <= 1.0 + 1e-6
    assert abs(beta @ weights) <= 0.01 + 1e-6
    assert np.abs(weights).max() <= 0.20 + 1e-6


def test_long_only_aim_respects_budget_and_position_constraints():
    config = FrontierConfig(long_only_position_limit=0.15)
    alpha = np.linspace(0.0, 0.01, 20)
    weights = solve_aim_weights(
        alpha,
        np.ones(20),
        np.full(20, 0.10),
        market_vol=0.05,
        risk_aversion=20.0,
        portfolio="long_only",
        config=config,
    )

    assert weights.min() >= -1e-7
    assert weights.sum() <= 1.0 + 1e-6
    assert weights.max() <= 0.15 + 1e-6


def test_slower_adjustment_trades_less_toward_same_long_only_aim():
    config = FrontierConfig(long_only_position_limit=1.0)
    prior = np.array([0.5, 0.5, 0.0])
    aim = np.array([0.0, 0.5, 0.5])
    beta = np.ones(3)

    fast = trade_toward_aim(prior, aim, beta, 1.0, "long_only", config)
    slow = trade_toward_aim(prior, aim, beta, 0.25, "long_only", config)

    assert np.abs(fast - prior).sum() > np.abs(slow - prior).sum()
    np.testing.assert_allclose(fast, aim)


def test_causal_selection_uses_only_prior_year_history():
    config = FrontierConfig(
        risk_aversions=(5.0, 20.0),
        adjustment_speeds=(1.0,),
        aum_eur=(10_000_000.0,),
        selection_lookback_months=12,
        default_risk_aversion=20.0,
        default_adjustment_speed=1.0,
    )
    records = []
    for signal in ["ml_return", "momentum", "sparse3"]:
        for portfolio in ["long_short", "long_only"]:
            for date in pd.date_range("2018-01-31", "2020-12-31", freq="ME"):
                for gamma in [5.0, 20.0]:
                    records.append(
                        {
                            "return_date": date,
                            "signal": signal,
                            "portfolio": portfolio,
                            "risk_aversion": gamma,
                            "adjustment_speed": 1.0,
                            "net_return_10m": 0.02 if gamma == 5.0 else 0.00,
                            "rf_eur": 0.0,
                            "turnover": 0.1,
                        }
                    )
    monthly = pd.DataFrame(records)

    _, choices = causal_strategy_selection(monthly, config)
    choice_2019 = choices[
        choices["test_year"].eq(2019)
        & choices["signal"].eq("ml_return")
        & choices["portfolio"].eq("long_short")
    ].iloc[0]

    assert choice_2019["history_months"] == 12
    assert choice_2019["risk_aversion"] == 5.0
    assert choice_2019["selection_source"] == "trailing_validation"


def test_frontier_dominance_scales_momentum_from_cash_origin():
    summary = pd.DataFrame(
        [
            {
                "signal": "momentum",
                "portfolio": "long_short",
                "aum_label": "100m",
                "risk_aversion": 5.0,
                "adjustment_speed": 1.0,
                "annualized_volatility": 0.10,
                "annualized_excess_return": 0.08,
                "efficient": True,
            },
            {
                "signal": "ml_return",
                "portfolio": "long_short",
                "aum_label": "100m",
                "risk_aversion": 20.0,
                "adjustment_speed": 0.5,
                "annualized_volatility": 0.05,
                "annualized_excess_return": 0.03,
                "efficient": True,
            },
        ]
    )

    result = frontier_dominance(summary).iloc[0]

    assert np.isclose(result["momentum_return_at_same_risk"], 0.04)
    assert np.isclose(result["frontier_return_improvement"], -0.01)
