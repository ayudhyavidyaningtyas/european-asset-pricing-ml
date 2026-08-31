from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from linear_attention_sdf import (  # noqa: E402
    LinearAttentionSDFConfig,
    build_monthly_sdf_basis,
    compare_attention_to_bsv,
    run_walk_forward_sdf,
)


FEATURES = ["x_rank", "y_rank"]


def synthetic_panel() -> pd.DataFrame:
    records = []
    for date in pd.date_range("2020-01-31", periods=12, freq="ME"):
        for security in range(4):
            x = -1.0 + security * 2.0 / 3.0
            y = 1.0 if security % 2 == 0 else -1.0
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "sdf_target_return": 0.01 + 0.02 * x - 0.005 * y,
                    "model_eligible": True,
                    "x_rank": x,
                    "y_rank": y,
                }
            )
    return pd.DataFrame(records)


def test_monthly_basis_has_bsv_and_d_cubed_attention_terms():
    panel = synthetic_panel()
    config = LinearAttentionSDFConfig(min_monthly_stocks=2)

    basis = build_monthly_sdf_basis(panel, config, FEATURES)

    bsv_columns = [column for column in basis if column.startswith("bsv_")]
    attention_columns = [
        column for column in basis if column.startswith("linear_attention_")
    ]
    assert len(bsv_columns) == 2
    assert len(attention_columns) == 8

    first_month = panel[panel["date"].eq(panel["date"].min())]
    x = first_month[FEATURES].to_numpy(dtype=float)
    returns = first_month["sdf_target_return"].to_numpy(dtype=float)
    expected_bsv = x.T @ returns / len(first_month)
    expected_attention = np.kron(
        (x.T @ x / len(first_month)).reshape(-1, order="F"),
        expected_bsv,
    )
    assert np.allclose(basis.loc[0, bsv_columns].to_numpy(dtype=float), expected_bsv)
    assert np.allclose(
        basis.loc[0, attention_columns].to_numpy(dtype=float),
        expected_attention,
    )


def test_walk_forward_training_target_dates_are_known_by_signal_date():
    panel = synthetic_panel()
    config = LinearAttentionSDFConfig(
        first_test_year=2020,
        last_test_year=2020,
        min_monthly_stocks=2,
        min_training_months=4,
        validation_months=2,
    )
    basis = build_monthly_sdf_basis(panel, config, FEATURES)

    monthly, fit_log = run_walk_forward_sdf(basis, config)

    assert set(monthly["model"]) == {"bsv", "linear_attention"}
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["signal_date"])
    ).all()


def test_attention_comparison_uses_common_months():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    monthly = pd.concat(
        [
            pd.DataFrame(
                {
                    "signal_date": dates,
                    "model": "bsv",
                    "sdf_return": np.linspace(0.01, 0.02, len(dates)),
                }
            ),
            pd.DataFrame(
                {
                    "signal_date": dates,
                    "model": "linear_attention",
                    "sdf_return": np.linspace(0.012, 0.025, len(dates)),
                }
            ),
        ],
        ignore_index=True,
    )

    comparison = compare_attention_to_bsv(monthly, hac_lags=3)

    assert comparison.loc[0, "months"] == 24
    assert comparison.loc[0, "annualized_mean_difference"] > 0
