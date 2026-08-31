from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_forecast_error_panel.py"
SPEC = importlib.util.spec_from_file_location("build_forecast_error_panel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _actuals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Instrument": ["AAA", "AAA", "AAA", "AAA"],
            "TR.RIC": ["AAA", "", "", ""],
            "TR.ISIN": ["ISINAAA", "", "", ""],
            "TR.EPSACTVALUE": [1.1, 1.2, 1.3, 2.0],
            "TR.EPSACTVALUE.DATE": [
                "2021-02-10",
                "2021-02-05",
                "2022-02-10",
                "2023-02-10",
            ],
            "TR.EPSACTVALUE.PERIODENDDATE": [
                "2020-12-31",
                "2020-12-31",
                "2021-12-31",
                "2022-09-30",
            ],
            "TR.EPSFRACTVALUE": [1.1, 1.2, 1.3, 2.0],
            "TR.EPSFRACTVALUE.DATE": [
                "2021-02-10",
                "2021-02-05",
                "2022-02-10",
                "2023-02-10",
            ],
            "TR.EPSFRACTVALUE.PERIODENDDATE": [
                "2020-12-31",
                "2020-12-31",
                "2021-12-31",
                "2022-09-30",
            ],
            "TR.REVENUEACTVALUE": [100.0, 100.0, 120.0, 130.0],
            "TR.REVENUEACTVALUE.DATE": [
                "2021-02-10",
                "2021-02-05",
                "2022-02-10",
                "2023-02-10",
            ],
            "TR.REVENUEACTVALUE.PERIODENDDATE": [
                "2020-12-31",
                "2020-12-31",
                "2021-12-31",
                "2022-09-30",
            ],
        },
    )


def _panel_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2021-01-31", "2021-03-31"])
    return pd.DataFrame(
        {
            "date": dates,
            "ric": ["AAA", "AAA"],
            "TR.ISIN": ["ISINAAA", "ISINAAA"],
            "screen_country": ["GB", "GB"],
            "TR.TRBCECONOMICSECTOR": ["Industrials", "Industrials"],
            "company_market_cap": [1000.0, 1000.0],
            "price_close": [10.0, 10.0],
            "log_size_rank": [0.1, 0.1],
            "book_to_market_rank": [0.2, 0.2],
            "momentum_12_2_rank": [0.3, 0.3],
            "volatility_12m_rank": [0.4, 0.4],
            "est_ric": ["AAA", "AAA"],
            "est_isin": ["ISINAAA", "ISINAAA"],
            "est_snapshot_date": dates,
            "est_signal_lag_months": [1.0, 1.0],
            "estimates_feature_count": [3, 3],
            "est_eps_mean": [1.0, 1.1],
            "est_revenue_mean": [95.0, 110.0],
            "est_eps_revision_1m": [0.01, 0.02],
            "est_eps_revision_3m": [0.03, 0.04],
            "est_revenue_revision_1m": [0.01, 0.02],
            "est_revenue_revision_3m": [0.03, 0.04],
            "est_price_target_revision_1m": [0.01, 0.02],
            "est_price_target_revision_3m": [0.03, 0.04],
            "est_eps_revision_1m_rank": [0.1, 0.2],
            "est_eps_revision_3m_rank": [0.3, 0.4],
            "est_revenue_revision_1m_rank": [0.1, 0.2],
            "est_revenue_revision_3m_rank": [0.3, 0.4],
            "est_price_target_revision_1m_rank": [0.1, 0.2],
            "est_price_target_revision_3m_rank": [0.3, 0.4],
            "est_eps_dispersion_rank": [0.0, 0.0],
            "est_revenue_dispersion_rank": [0.0, 0.0],
            "est_coverage_composite_rank": [0.0, 0.0],
        },
    )


def test_actual_table_keeps_earliest_announcement_for_duplicate_period():
    actuals = MODULE.clean_actuals(_actuals_frame())
    table = MODULE.actual_table(
        actuals,
        "eps_actual",
        "eps_announce_date",
        "eps_period_end",
        "eps",
    )

    first_period = table[table["eps_period_end"].eq(pd.Timestamp("2020-12-31"))]

    assert len(first_period) == 1
    assert first_period["eps_actual"].iloc[0] == 1.2
    assert first_period["eps_announce_date"].iloc[0] == pd.Timestamp("2021-02-05")
    assert table["eps_fye_month_changed"].all()


def test_match_actuals_uses_earliest_actual_announced_after_snapshot():
    config = MODULE.ForecastErrorConfig(drop_fiscal_year_end_changes=False)
    panel = _panel_frame().assign(panel_row_id=[0, 1])
    actuals = MODULE.actual_table(
        MODULE.clean_actuals(_actuals_frame()),
        "eps_actual",
        "eps_announce_date",
        "eps_period_end",
        "eps",
    )

    matched = MODULE.match_actuals(panel, actuals, "eps", config).sort_values(
        "panel_row_id",
    )

    assert matched["eps_period_end"].tolist() == [
        pd.Timestamp("2020-12-31"),
        pd.Timestamp("2021-12-31"),
    ]


def test_build_forecast_error_panel_applies_timing_and_unit_gates(tmp_path: Path):
    panel_path = tmp_path / "panel.parquet"
    actuals_path = tmp_path / "actuals.csv.gz"
    _panel_frame().to_parquet(panel_path, index=False)
    _actuals_frame().to_csv(actuals_path, index=False, compression="gzip")

    result, audit = MODULE.build_forecast_error_panel(
        panel_path,
        actuals_path,
        MODULE.ForecastErrorConfig(drop_fiscal_year_end_changes=False),
    )

    assert audit["timing"]["eps_announce_on_or_before_snapshot"] == 0
    assert result["eps_period_end"].tolist() == [
        pd.Timestamp("2020-12-31"),
        pd.Timestamp("2021-12-31"),
    ]
    assert result["eps_error_valid"].all()
    assert result["revenue_error_valid"].all()
    assert result["eps_error_to_price"].round(6).tolist() == [0.02, 0.02]
