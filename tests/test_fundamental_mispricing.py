from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fundamental_mispricing import (  # noqa: E402
    ACCOUNTING_RANK_FEATURES,
    FundamentalMispricingConfig,
    _prediction_frame,
    load_fundamental_mispricing_panel,
    prepare_accounting_values,
    run_fundamental_mispricing_walk_forward,
)


def test_accounting_values_respect_six_month_availability_lag():
    annual = pd.DataFrame(
        [
            {
                "isin": "GB0000000001",
                "datadate": "2020-12-31",
                "gvkey": "1001",
                "at": 100.0,
                "act": 50.0,
                "lct": 20.0,
                "ceq": 40.0,
                "revt": 80.0,
            }
        ]
    )
    values = prepare_accounting_values(annual, FundamentalMispricingConfig())

    assert values.loc[0, "available_date"] == pd.Timestamp("2021-06-30")
    assert values.loc[0, "fv_assets_total"] == 100.0
    assert values.loc[0, "fv_common_equity"] == 40.0


def test_panel_merge_does_not_use_unavailable_accounts(tmp_path: Path):
    panel = pd.DataFrame(
        [
            {
                "date": "2021-05-31",
                "target_date": "2021-06-30",
                "ric": "AAA",
                "target_return_1m": 0.01,
                "target_return_rank": 0.5,
                "company_market_cap": 100.0,
                "market_cap_percentile": 0.5,
                "screen_country": "GB",
                "TR.TRBCECONOMICSECTOR": "Industrials",
                "TR.ISIN": "GB0000000001",
                "eligible": True,
                "model_eligible": True,
                "return_history_n": 36,
                "momentum_12_2_rank": 0.2,
            },
            {
                "date": "2021-06-30",
                "target_date": "2021-07-31",
                "ric": "AAA",
                "target_return_1m": 0.02,
                "target_return_rank": 0.5,
                "company_market_cap": 110.0,
                "market_cap_percentile": 0.5,
                "screen_country": "GB",
                "TR.TRBCECONOMICSECTOR": "Industrials",
                "TR.ISIN": "GB0000000001",
                "eligible": True,
                "model_eligible": True,
                "return_history_n": 37,
                "momentum_12_2_rank": 0.3,
            },
        ]
    )
    annual = pd.DataFrame(
        [
            {
                "isin": "GB0000000001",
                "datadate": "2020-12-31",
                "gvkey": "1001",
                "at": 100.0,
                "act": 50.0,
                "lct": 20.0,
                "ceq": 40.0,
                "revt": 80.0,
                "cogs": 30.0,
                "lt": 60.0,
                "oancf": 10.0,
            }
        ]
    )
    panel_path = tmp_path / "panel.parquet"
    annual_path = tmp_path / "annual.csv.gz"
    panel.to_parquet(panel_path, index=False)
    annual.to_csv(annual_path, index=False, compression="gzip")

    loaded = load_fundamental_mispricing_panel(
        panel_path,
        annual_path,
        FundamentalMispricingConfig(min_accounting_features=1),
    )

    may = loaded[loaded["date"].eq(pd.Timestamp("2021-05-31"))].iloc[0]
    june = loaded[loaded["date"].eq(pd.Timestamp("2021-06-30"))].iloc[0]
    assert pd.isna(may["fv_assets_total"])
    assert june["fv_assets_total"] == 100.0


def synthetic_fair_value_panel() -> pd.DataFrame:
    records = []
    for date in pd.date_range("2018-01-31", "2021-12-31", freq="ME"):
        for security in range(12):
            feature = -1.0 + 2.0 * security / 11.0
            market_share = np.exp(feature)
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security:02d}",
                    "target_return_1m": 0.01 * feature,
                    "target_return_rank": feature,
                    "company_market_cap": 100.0 * market_share,
                    "market_cap_percentile": 0.2 + security / 20.0,
                    "screen_country": "GB",
                    "TR.TRBCECONOMICSECTOR": "Industrials",
                    "actual_market_share": market_share / 12.0,
                    "log_market_share": np.log(market_share / 12.0),
                    "fundamental_feature_count": len(ACCOUNTING_RANK_FEATURES),
                    "fair_value_model_eligible": True,
                    "momentum_12_2_rank": feature,
                }
            )
    panel = pd.DataFrame(records)
    for feature in ACCOUNTING_RANK_FEATURES:
        panel[feature] = np.tile(np.linspace(-1.0, 1.0, 12), panel["date"].nunique())
    return panel


def test_walk_forward_fundamental_mispricing_is_causal_and_unique():
    panel = synthetic_fair_value_panel()
    config = FundamentalMispricingConfig(
        first_test_year=2021,
        last_test_year=2021,
        training_window_months=24,
        min_training_rows=50,
        min_training_months=12,
        min_monthly_stocks=4,
        max_training_rows=None,
        rf_estimators=5,
    )

    predictions, fit_log, importance = run_fundamental_mispricing_walk_forward(
        panel,
        ["linear"],
        config,
        include_momentum=True,
    )

    assert set(predictions["model"]) == {"fv_linear_signal", "momentum_rank"}
    assert not predictions.duplicated(["model", "date", "ric"]).any()
    assert (
        pd.to_datetime(fit_log["train_signal_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    assert not importance.empty


def test_positive_fair_value_gap_means_undervalued_signal():
    test = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2021-01-31"),
                "target_date": pd.Timestamp("2021-02-28"),
                "ric": "AAA",
                "target_return_1m": 0.01,
                "target_return_rank": 0.5,
                "company_market_cap": 100.0,
                "market_cap_percentile": 0.5,
                "screen_country": "GB",
                "TR.TRBCECONOMICSECTOR": "Industrials",
                "actual_market_share": 0.01,
                "log_market_share": np.log(0.01),
                "fundamental_feature_count": 10,
            }
        ]
    )

    output = _prediction_frame(
        test,
        "fv_linear_signal",
        "linear",
        np.array([0.02]),
        2021,
        pd.Timestamp("2020-12-31"),
        "market_share",
    )

    assert output.loc[0, "prediction"] > 0
    assert np.isclose(output.loc[0, "fair_value_to_market"], 2.0)
