from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_revision_strategy_exhibits.py"
SPEC = importlib.util.spec_from_file_location("build_revision_strategy_exhibits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_annualized_return_summary_matches_repo_arithmetic_convention():
    returns = pd.Series([0.01, -0.01, 0.02, 0.00])

    summary = MODULE.annualized_return_summary(returns)

    assert summary["months"] == 4
    assert summary["annualized_net_return"] == pytest.approx(0.06)
    assert summary["annualized_net_volatility"] > 0


def test_monthly_information_coefficients_are_rank_correlations():
    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2021-01-31")] * 12,
            "model": ["ridge_rank"] * 12,
            "prediction": list(range(12)),
            "target_return_rank": list(range(12)),
            "target_residual_rank": list(reversed(range(12))),
        }
    )

    ic = MODULE.monthly_information_coefficients(predictions)

    assert ic["spearman_ic"].iloc[0] == pytest.approx(1.0)
    assert ic["residual_spearman_ic"].iloc[0] == pytest.approx(-1.0)


def test_breakeven_cost_bps_uses_mean_gross_over_turnover():
    gross = pd.Series([0.02, 0.01])
    turnover = pd.Series([1.0, 0.5])

    bps = MODULE.breakeven_cost_bps(gross, turnover)

    assert bps == pytest.approx(200.0)


def test_breakeven_capacity_scales_square_root_impact():
    gross = pd.Series([0.02, 0.02])
    spread = pd.Series([0.005, 0.005])
    impact = pd.Series([0.003, 0.003])

    capacity = MODULE.breakeven_capacity_aum(gross, spread, impact, 100_000_000.0)

    assert capacity == pytest.approx(2_500_000_000.0)


def test_stationary_bootstrap_metric_ci_reports_sharpe_interval():
    returns = pd.Series([0.02, 0.01, -0.005, 0.015, 0.03, -0.01, 0.012, 0.004])

    result = MODULE.stationary_bootstrap_metric_ci(
        returns,
        metric="sharpe",
        expected_block=3,
        n_boot=200,
        seed=7,
    )

    assert result["observations"] == len(returns)
    assert result["point"] == pytest.approx(MODULE.sharpe_ratio(returns))
    assert result["ci_low"] < result["ci_high"]
    assert 0.0 <= result["p_two_sided_zero"] <= 1.0


def test_portfolio_net_returns_subtracts_rf_only_for_long_only():
    group = pd.DataFrame(
        {
            "return_date": pd.to_datetime(["2021-01-31", "2021-02-28"]),
            "gross_long_short_return": [0.02, 0.01],
            "long_short_turnover": [1.0, 1.0],
            "long_return": [0.03, 0.02],
            "long_only_turnover": [0.5, 0.5],
        }
    )
    risk_free = pd.Series(
        [0.001, 0.002],
        index=pd.to_datetime(["2021-01-31", "2021-02-28"]),
    )

    long_short = MODULE._portfolio_net_returns(group, "long_short", 25, risk_free)
    long_only = MODULE._portfolio_net_returns(
        group,
        "long_only_top_decile",
        25,
        risk_free,
    )

    assert long_short.iloc[0] == pytest.approx(0.0175)
    assert long_only.iloc[0] == pytest.approx(0.02775)
    assert long_only.iloc[1] == pytest.approx(0.01675)


def test_portfolio_summary_drops_missing_rf_months_for_long_only_sharpe():
    returns = pd.Series([0.02, 0.01, 0.03])
    dates = pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"])
    risk_free = pd.Series(
        [0.001, 0.002],
        index=pd.to_datetime(["2021-01-31", "2021-02-28"]),
    )

    summary = MODULE.annualized_portfolio_return_summary(returns, dates, risk_free)

    assert summary["months"] == 2
    assert summary["annualized_net_return"] == pytest.approx(0.18)
    assert summary["net_sharpe"] == pytest.approx(
        MODULE.sharpe_ratio(pd.Series([0.019, 0.008]))
    )


def test_stationary_bootstrap_sharpe_difference_ci_is_paired():
    ridge = pd.Series([0.02, 0.01, 0.015, -0.005, 0.03, 0.0, 0.02, -0.01])
    dre = pd.Series([0.005, 0.0, 0.01, -0.015, 0.012, -0.005, 0.006, -0.02])

    result = MODULE.stationary_bootstrap_sharpe_difference_ci(
        ridge,
        dre,
        expected_block=3,
        n_boot=200,
        seed=11,
    )

    assert result["observations"] == len(ridge)
    assert result["point"] == pytest.approx(
        MODULE.sharpe_ratio(ridge) - MODULE.sharpe_ratio(dre)
    )
    assert result["ci_low"] < result["ci_high"]
    assert 0.0 <= result["p_two_sided_zero"] <= 1.0


def test_coefficient_stability_table_summarizes_signs(tmp_path: Path):
    revision_dir = tmp_path / "revision"
    output_dir = tmp_path / "output"
    revision_dir.mkdir()
    output_dir.mkdir()
    pd.DataFrame(
        {
            "model": ["ridge", "ridge", "ridge", "ridge"],
            "model_label": ["ridge_rank"] * 4,
            "target_mode": ["rank"] * 4,
            "test_year": [2021, 2022, 2021, 2022],
            "feature": [
                "est_eps_revision_1m_rank",
                "est_eps_revision_1m_rank",
                "est_revenue_revision_1m_rank",
                "est_revenue_revision_1m_rank",
            ],
            "coefficient": [0.1, 0.2, -0.1, 0.1],
        }
    ).to_csv(revision_dir / "linear_coefficients.csv", index=False)

    _, summary = MODULE.build_ridge_coefficient_stability(revision_dir, output_dir)

    eps = summary[summary["feature"].eq("est_eps_revision_1m_rank")].iloc[0]
    revenue = summary[summary["feature"].eq("est_revenue_revision_1m_rank")].iloc[0]
    assert eps["positive_year_fraction"] == pytest.approx(1.0)
    assert revenue["sign_changes"] == 1
    assert (output_dir / "revision_ridge_coefficient_stability.png").exists()


def test_subperiod_definitions_split_realized_oos_dates():
    dates = pd.date_range("2015-02-28", periods=10, freq="ME")

    definitions = MODULE._subperiod_definitions(pd.Series(dates))

    assert definitions[0][0] == "full_oos"
    assert definitions[1][1] == dates[0]
    assert definitions[1][2] == dates[4]
    assert definitions[2][1] == dates[5]
