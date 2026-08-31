from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_forecast_error_mechanism_tests.py"
SPEC = importlib.util.spec_from_file_location("run_forecast_error_mechanism_tests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _mechanism_frame(months: int = 3, firms: int = 6) -> pd.DataFrame:
    records = []
    for month_index in range(months):
        date = pd.Timestamp("2021-01-31") + pd.offsets.MonthEnd(month_index)
        announce = date + pd.offsets.MonthEnd(6)
        period = pd.Timestamp("2021-12-31") + pd.DateOffset(years=month_index)
        for firm_index in range(firms):
            signal = firm_index - (firms - 1) / 2
            eps_error = 0.01 + 0.02 * signal
            revenue_error = 0.03 + 0.01 * signal
            records.append(
                {
                    "date": date,
                    "ric": f"F{firm_index}",
                    "est_snapshot_date": date,
                    "eps_announce_date": announce,
                    "eps_period_end": period,
                    "epsfr_announce_date": announce,
                    "epsfr_period_end": period,
                    "revenue_announce_date": announce,
                    "revenue_period_end": period,
                    "eps_error_valid": True,
                    "epsfr_error_valid": True,
                    "revenue_error_valid": True,
                    "eps_error_to_price_winsorized": eps_error,
                    "epsfr_error_to_price_winsorized": eps_error,
                    "revenue_error_to_market_cap_winsorized": revenue_error,
                    "est_eps_revision_3m_rank": signal,
                    "est_eps_revision_1m_rank": signal,
                    "est_revenue_revision_3m_rank": signal * 0.5,
                    "est_revenue_revision_1m_rank": signal * 0.5,
                }
            )
    return pd.DataFrame(records)


def _joint_mechanism_frame(months: int = 3) -> pd.DataFrame:
    records = []
    eps_signals = [-3, -2, -1, 0, 1, 2, 3, 4]
    revenue_signals = [1, -1, -2, 2, -1, 1, 2, -2]
    for month_index in range(months):
        date = pd.Timestamp("2021-01-31") + pd.offsets.MonthEnd(month_index)
        announce = date + pd.offsets.MonthEnd(6)
        period = pd.Timestamp("2021-12-31") + pd.DateOffset(years=month_index)
        for firm_index, (eps_signal, revenue_signal) in enumerate(
            zip(eps_signals, revenue_signals, strict=True)
        ):
            records.append(
                {
                    "date": date,
                    "ric": f"J{firm_index}",
                    "est_snapshot_date": date,
                    "eps_announce_date": announce,
                    "eps_period_end": period,
                    "eps_error_valid": True,
                    "eps_error_to_price_winsorized": (
                        0.01 + 0.02 * eps_signal + 0.005 * revenue_signal
                    ),
                    "est_eps_revision_3m_rank": eps_signal,
                    "est_revenue_revision_3m_rank": revenue_signal,
                }
            )
    return pd.DataFrame(records)


def test_fama_macbeth_recovers_signal_slope():
    spec = MODULE.default_specs()[0]
    config = MODULE.MechanismConfig(
        min_cross_section=5,
        hac_lags=0,
        min_hac_periods=2,
        include_fixed_effects=False,
        include_controls=False,
    )

    _, summary = MODULE.run_fama_macbeth_for_spec(
        _mechanism_frame(),
        spec,
        "overlapping_monthly",
        config,
    )

    assert summary["mean_signal_slope"] == pytest.approx(0.02)
    assert summary["periods"] == 3


def test_joint_fama_macbeth_recovers_conditional_slopes():
    spec = MODULE.default_joint_specs()[0]
    config = MODULE.MechanismConfig(
        min_cross_section=5,
        hac_lags=0,
        min_hac_periods=2,
        include_fixed_effects=False,
        include_controls=False,
    )

    _, summary = MODULE.run_joint_fama_macbeth_for_spec(
        _joint_mechanism_frame(),
        spec,
        "overlapping_monthly",
        config,
    )

    assert summary["primary_mean_slope"] == pytest.approx(0.02)
    assert summary["competing_mean_slope"] == pytest.approx(0.005)
    assert summary["mean_slope_difference"] == pytest.approx(0.015)


def test_collapsed_sample_keeps_one_snapshot_closest_to_target_lead():
    spec = MODULE.default_specs()[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-31", "2021-02-28", "2021-03-31"]),
            "ric": ["AAA", "AAA", "AAA"],
            "est_snapshot_date": pd.to_datetime(["2020-11-30", "2021-01-31", "2021-02-28"]),
            "eps_announce_date": pd.to_datetime(["2021-07-15", "2021-07-15", "2021-07-15"]),
            "eps_period_end": pd.to_datetime(["2020-12-31", "2020-12-31", "2020-12-31"]),
            "eps_error_valid": [True, True, True],
            "eps_error_to_price_winsorized": [0.1, 0.2, 0.3],
            "est_eps_revision_3m_rank": [0.8, 0.6, 0.5],
        }
    )

    collapsed = MODULE.build_collapsed_sample(frame, spec, lead_months=6)

    assert len(collapsed) == 1
    assert collapsed["lead_months"].iloc[0] == 6
    assert collapsed["est_eps_revision_3m_rank"].iloc[0] == 0.6


def test_runner_writes_mechanism_outputs(tmp_path: Path):
    panel_path = tmp_path / "forecast_error_panel.parquet"
    output_dir = tmp_path / "mechanism"
    _mechanism_frame().to_parquet(panel_path, index=False)
    config = MODULE.MechanismConfig(
        min_cross_section=5,
        hac_lags=0,
        min_hac_periods=2,
        include_fixed_effects=False,
        include_controls=False,
    )

    manifest = MODULE.run_forecast_error_mechanism_tests(panel_path, output_dir, config)
    summary = pd.read_csv(output_dir / "mechanism_fama_macbeth_summary.csv")

    assert manifest["rows"]["summary_rows"] == 24
    assert manifest["rows"]["joint_summary_rows"] == 12
    assert manifest["rows"]["specificity_rows"] == 12
    assert "eps_error_on_eps_revision_3m" in set(summary["spec"])
    assert "revenue_error_on_eps_revision_3m_placebo" in set(summary["spec"])
    assert (output_dir / "mechanism_joint_signal_summary.csv").exists()
    assert (output_dir / "mechanism_joint_signal_coefficients.csv").exists()
    assert (output_dir / "mechanism_specificity_tests.csv").exists()
    assert (output_dir / "mechanism_quintile_spread_tests.csv").exists()
    assert (output_dir / "mechanism_sign_accuracy.csv").exists()
