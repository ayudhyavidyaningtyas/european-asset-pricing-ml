from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aipm_linear_transformer_sdf import (  # noqa: E402
    AIPMLinearTransformerConfig,
    add_weight_turnover,
    build_months,
    compare_attention_to_bsv,
    evaluate_month,
    fit_model,
    monthly_basis_vector,
    run_walk_forward_aipm,
)


FEATURES = ["x_rank", "y_rank"]


def synthetic_panel(periods: int = 48) -> pd.DataFrame:
    records = []
    for date in pd.date_range("2018-01-31", periods=periods, freq="ME"):
        for security in range(5):
            x = -1.0 + security * 0.5
            y = 1.0 if security % 2 == 0 else -1.0
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security}",
                    "target_return_1m": 0.01 + 0.02 * x - 0.005 * y,
                    "sdf_target_return": 0.01 + 0.02 * x - 0.005 * y,
                    "model_eligible": True,
                    "company_market_cap": 100.0 + security,
                    "market_cap_percentile": 0.2 + security / 10.0,
                    "x_rank": x,
                    "y_rank": y,
                }
            )
    return pd.DataFrame.from_records(records)


def test_basis_vectors_match_bsv_and_linear_attention_formula():
    config = AIPMLinearTransformerConfig(min_monthly_stocks=2)
    month = build_months(synthetic_panel(periods=1), FEATURES, config)[0]

    bsv = monthly_basis_vector(month, "bsv", training_returns=False)
    attention = monthly_basis_vector(month, "linear_attention", training_returns=False)

    x = month.features
    returns = month.evaluation_returns
    expected_bsv = x.T @ returns / month.n_stocks
    expected_attention = np.kron(
        expected_bsv,
        (x.T @ x / month.n_stocks).reshape(-1, order="C"),
    )
    assert np.allclose(bsv, expected_bsv)
    assert np.allclose(attention, expected_attention)


def test_evaluate_month_outputs_gross_normalized_weights():
    config = AIPMLinearTransformerConfig(min_monthly_stocks=2, gross_leverage=1.0)
    months = build_months(synthetic_panel(periods=12), FEATURES, config)
    fitted = fit_model(
        months[:8],
        months[8:10],
        "linear_attention",
        config,
    )

    record, weights = evaluate_month(months[-1], fitted, config)

    assert record["model"] == "linear_attention"
    assert np.isclose(weights["sdf_weight"].abs().sum(), 1.0)
    assert np.isfinite(record["raw_sdf_return"])
    assert np.isfinite(record["sdf_return"])
    assert {"raw_weight", "sdf_weight"}.issubset(weights.columns)


def test_walk_forward_aipm_is_causal_and_outputs_both_models():
    config = AIPMLinearTransformerConfig(
        first_test_year=2021,
        last_test_year=2021,
        min_monthly_stocks=2,
        min_training_months=12,
        validation_months=4,
        max_attention_features=None,
    )
    months = build_months(synthetic_panel(periods=48), FEATURES, config)

    monthly, fit_log, weights = run_walk_forward_aipm(months, config)

    assert set(monthly["model"]) == {"bsv", "linear_attention"}
    assert set(weights["model"]) == {"bsv", "linear_attention"}
    monthly = add_weight_turnover(monthly, weights)
    assert "weight_turnover" in monthly
    assert monthly["weight_turnover"].notna().any()
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    assert not weights.duplicated(["signal_date", "ric", "model"]).any()


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
