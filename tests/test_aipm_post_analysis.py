from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aipm_post_analysis import (  # noqa: E402
    AIPMPostAnalysisConfig,
    PrincipalPortfolioConfig,
    attention_lift_table,
    build_attention_pair_diagnostics,
    build_execution_input_panel,
    impute_half_spreads,
    run_principal_portfolio_walk_forward,
    simulate_weight_implementability,
    summarize_attention_diagnostics,
    summarize_implementability,
)


def synthetic_panel(periods: int = 36, stocks: int = 8) -> pd.DataFrame:
    records = []
    for month_index, date in enumerate(pd.date_range("2018-01-31", periods=periods, freq="ME")):
        state = np.sin(month_index / 6.0)
        for security in range(stocks):
            x = -1.0 + security * 2.0 / max(stocks - 1, 1)
            y = 1.0 if security % 2 == 0 else -1.0
            z = np.cos(security)
            ret = 0.01 + 0.015 * x - 0.004 * y + 0.006 * x * state
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security}",
                    "target_return_1m": ret,
                    "aipm_target_return": ret,
                    "model_eligible": True,
                    "company_market_cap": 100.0 + security,
                    "market_cap_percentile": 0.3 + security / (stocks + 3),
                    "x_rank": x,
                    "y_rank": y,
                    "z_rank": z,
                }
            )
    return pd.DataFrame.from_records(records)


def test_attention_diagnostics_compare_observed_links_to_null():
    date = pd.Timestamp("2021-01-31")
    metadata = pd.DataFrame(
        {
            "signal_date": [date] * 4,
            "ric": ["A", "B", "C", "D"],
            "TR.EXCHANGECOUNTRY": ["FR", "FR", "DE", "DE"],
            "TR.TRBCECONOMICSECTOR": ["Tech", "Tech", "Bank", "Bank"],
            "market_cap_percentile": [0.9, 0.8, 0.7, 0.6],
            "momentum_12_2_rank": [0.1, 0.2, 0.8, 0.9],
        }
    )
    attention = pd.DataFrame(
        {
            "signal_date": [date, date],
            "seed": [0, 0],
            "source_ric": ["A", "C"],
            "attended_ric": ["B", "D"],
            "attention_weight": [0.7, 0.6],
        }
    )

    pairs = build_attention_pair_diagnostics(
        attention,
        metadata,
        null_draws=3,
        random_state=1,
    )
    summary = summarize_attention_diagnostics(pairs)
    lift = attention_lift_table(summary)

    assert set(pairs["sample"]) == {"observed", "null"}
    assert "same_exchangecountry" in pairs
    assert "abs_diff_momentum_12_2_rank" in pairs
    assert not summary.empty
    assert not lift.empty


def test_weight_implementability_applies_spread_and_impact_costs(tmp_path):
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ric": ["A", "B", "A", "B"],
            "company_market_cap": [1_000_000.0, 2_000_000.0] * 2,
            "turnover_12m": [0.02, 0.03] * 2,
            "volatility_12m": [0.12, 0.18] * 2,
            "target_return_1m": [0.01, -0.01, 0.02, -0.02],
        }
    )
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    weights = pd.DataFrame(
        {
            "signal_date": [dates[0], dates[0], dates[1], dates[1]],
            "target_date": [dates[0] + pd.offsets.MonthEnd(1)] * 2
            + [dates[1] + pd.offsets.MonthEnd(1)] * 2,
            "ric": ["A", "B", "A", "B"],
            "model": ["bsv"] * 4,
            "sdf_weight": [0.5, -0.5, 0.25, -0.75],
            "target_return": [0.01, -0.01, 0.02, -0.02],
        }
    )
    config = AIPMPostAnalysisConfig(aum_eur=(1_000_000.0,))

    inputs = build_execution_input_panel(panel_path, None, None, config)
    monthly = simulate_weight_implementability(weights, inputs, config)
    summary = summarize_implementability(monthly)

    assert len(monthly) == 2
    assert (monthly["total_cost"] > 0).all()
    assert (monthly["net_return"] < monthly["gross_return"]).all()
    assert summary.loc[0, "annualized_total_cost"] > 0


def test_size_conditional_imputation_widens_spreads_for_small_uncovered_names():
    date = pd.Timestamp("2020-01-31")
    # Covered names span three decades of size with spreads that widen as size
    # falls, so the fitted elasticity is negative and well identified.
    covered_caps = np.logspace(9.0, 11.0, 40)
    covered_spreads = 1000.0 * covered_caps ** -0.25
    # Uncovered names sit an order of magnitude below the covered sample.
    uncovered_caps = np.array([1e7, 1e8])

    frame = pd.DataFrame(
        {
            "signal_date": [date] * (len(covered_caps) + len(uncovered_caps)),
            "ric": [f"C{i}" for i in range(len(covered_caps))]
            + [f"U{i}" for i in range(len(uncovered_caps))],
            "market_cap": np.concatenate([covered_caps, uncovered_caps]),
            "half_spread_bps": np.concatenate(
                [covered_spreads, np.full(len(uncovered_caps), 25.0)]
            ),
            "spread_observed": [True] * len(covered_caps)
            + [False] * len(uncovered_caps),
        }
    )
    config = AIPMPostAnalysisConfig(spread_imputation="size_conditional")

    result = impute_half_spreads(frame, config)

    covered = result[result["spread_observed"]]
    uncovered = result[~result["spread_observed"]]
    # Covered names keep their measured quotes.
    assert np.allclose(covered["half_spread_bps"].to_numpy(), covered_spreads)
    assert not covered["half_spread_imputed_size_conditional"].any()
    assert uncovered["half_spread_imputed_size_conditional"].all()
    # Every uncovered name is smaller than the whole covered sample, so each is
    # charged more than the widest measured spread.
    assert (uncovered["half_spread_bps"] > covered_spreads.max()).all()
    # And the smaller of the two uncovered names is charged more than the larger.
    assert (
        uncovered.iloc[0]["half_spread_bps"] > uncovered.iloc[1]["half_spread_bps"]
    )
    # The fitted elasticity is recovered: spreads scale as cap ** -0.25, so a
    # 10x size drop widens the spread by 10 ** 0.25.
    ratio = uncovered.iloc[0]["half_spread_bps"] / uncovered.iloc[1]["half_spread_bps"]
    assert np.isclose(ratio, 10.0**0.25, rtol=1e-3)
    assert (uncovered["half_spread_bps"] <= config.imputed_spread_cap_bps).all()


def test_size_conditional_imputation_falls_back_when_too_few_covered_names():
    date = pd.Timestamp("2020-01-31")
    frame = pd.DataFrame(
        {
            "signal_date": [date] * 4,
            "ric": ["A", "B", "C", "D"],
            "market_cap": [1e10, 1e9, 1e8, 1e7],
            "half_spread_bps": [5.0, 8.0, 25.0, 25.0],
            "spread_observed": [True, True, False, False],
        }
    )
    config = AIPMPostAnalysisConfig(
        spread_imputation="size_conditional", min_spread_regression_obs=30
    )

    result = impute_half_spreads(frame, config)

    # Two covered names is below the regression minimum, so the constant stands.
    assert not result["half_spread_imputed_size_conditional"].any()
    assert np.allclose(result["half_spread_bps"].to_numpy(), [5.0, 8.0, 25.0, 25.0])


def test_principal_portfolio_walk_forward_is_causal_and_gross_normalized():
    panel = synthetic_panel(periods=36, stocks=8)
    config = PrincipalPortfolioConfig(
        first_test_year=2020,
        last_test_year=2020,
        min_monthly_stocks=4,
        min_training_months=10,
        validation_months=3,
        training_window_months=18,
        max_monthly_stocks=None,
        components=(1, 2),
    )

    monthly, fit_log, weights, summary, comparisons = run_principal_portfolio_walk_forward(
        panel,
        ["x_rank", "y_rank", "z_rank"],
        config,
    )

    assert set(monthly["model"]) == {
        "principal_portfolio_h1",
        "principal_portfolio_h2",
    }
    assert not fit_log.empty
    assert not weights.empty
    gross = weights.groupby(["signal_date", "model"])["sdf_weight"].apply(
        lambda values: float(values.abs().sum())
    )
    assert np.allclose(gross.to_numpy(), 1.0)
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    assert not summary.empty
    assert comparisons.empty
