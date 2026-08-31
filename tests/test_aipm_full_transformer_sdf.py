from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aipm_full_transformer_sdf import (  # noqa: E402
    AIPMFullTransformerConfig,
    NonlinearPortfolioTransformer,
    build_months,
    fit_closed_form_model,
    pricing_error_summary,
    random_feature_matrix,
    random_feature_spec,
    run_walk_forward_full_aipm,
)


FEATURES = ["x_rank", "y_rank", "z_rank"]


def synthetic_panel(periods: int = 48, stocks: int = 8) -> pd.DataFrame:
    records = []
    for month_index, date in enumerate(pd.date_range("2018-01-31", periods=periods, freq="ME")):
        state = np.sin(month_index / 5.0)
        for security in range(stocks):
            x = -1.0 + security * 2.0 / max(stocks - 1, 1)
            y = 1.0 if security % 2 == 0 else -1.0
            z = np.cos(security)
            ret = 0.01 + 0.02 * x - 0.006 * y + 0.004 * x * state
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security}",
                    "target_return_1m": ret,
                    "aipm_target_return": ret,
                    "model_eligible": True,
                    "company_market_cap": 100.0 + security,
                    "market_cap_percentile": 0.2 + security / (stocks + 2),
                    "x_rank": x,
                    "y_rank": y,
                    "z_rank": z,
                }
            )
    return pd.DataFrame.from_records(records)


def test_nonlinear_transformer_attention_rows_are_probabilities():
    network = NonlinearPortfolioTransformer(
        n_features=3,
        n_blocks=1,
        n_heads=1,
        feedforward_width=4,
        attention_temperature=None,
        use_attention=True,
    )
    features = torch.randn(6, 3)
    scores, attention = network(features, return_attention=True)

    assert scores.shape == (6,)
    assert attention is not None
    assert attention.shape == (6, 6)
    assert torch.allclose(attention.sum(dim=1), torch.ones(6), atol=1e-6)
    assert torch.all(attention >= 0)


def test_own_asset_network_does_not_emit_attention_matrix():
    network = NonlinearPortfolioTransformer(
        n_features=3,
        n_blocks=2,
        n_heads=1,
        feedforward_width=5,
        attention_temperature=None,
        use_attention=False,
    )
    scores, attention = network(torch.randn(7, 3), return_attention=True)

    assert scores.shape == (7,)
    assert attention is None


def test_closed_form_and_random_feature_benchmarks_fit():
    config = AIPMFullTransformerConfig(
        min_monthly_stocks=4,
        max_monthly_stocks=None,
        linear_attention_features=None,
        random_feature_count=5,
    )
    months = build_months(synthetic_panel(periods=20), FEATURES, config)
    fitted_bsv = fit_closed_form_model(months[:12], months[12:16], "bsv", config)
    spec = random_feature_spec(len(FEATURES), config.random_feature_count, seed=3)
    transformed = random_feature_matrix(months[0], spec)
    fitted_dkkm = fit_closed_form_model(
        months[:12],
        months[12:16],
        "dkkm_random_features",
        config,
        random_feature_spec=spec,
    )

    assert fitted_bsv.n_parameters == len(FEATURES)
    assert transformed.shape == (8, config.random_feature_count)
    assert fitted_dkkm.n_parameters == config.random_feature_count


def test_walk_forward_full_aipm_is_causal_and_outputs_all_models():
    config = AIPMFullTransformerConfig(
        first_test_year=2021,
        last_test_year=2021,
        min_monthly_stocks=4,
        min_training_months=12,
        validation_months=4,
        training_window_months=24,
        max_monthly_stocks=None,
        random_feature_count=6,
        transformer_blocks=1,
        feedforward_width=4,
        epochs=2,
        patience=2,
        seeds=(0,),
        random_state=7,
    )
    months = build_months(synthetic_panel(periods=48), FEATURES, config)
    monthly, fit_log, weights, attention, test_assets = run_walk_forward_full_aipm(
        months,
        config,
        models=(
            "bsv",
            "linear_attention",
            "dkkm_random_features",
            "own_asset_mlp",
            "nonlinear_transformer",
        ),
    )

    assert set(monthly["model"]) == {
        "bsv",
        "linear_attention",
        "dkkm_random_features",
        "own_asset_mlp",
        "nonlinear_transformer",
    }
    assert not weights.empty
    assert not test_assets.empty
    assert not attention.empty
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    assert (
        pd.to_datetime(fit_log["validation_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    assert not weights.duplicated(["signal_date", "ric", "model"]).any()


def test_pricing_error_summary_uses_common_months():
    dates = pd.date_range("2020-01-31", periods=18, freq="ME")
    monthly = pd.DataFrame(
        {
            "signal_date": dates,
            "model": "nonlinear_transformer",
            "sdf_return": np.linspace(0.01, 0.02, len(dates)),
            "raw_sdf_return": np.linspace(0.02, 0.04, len(dates)),
        }
    )
    test_assets = pd.DataFrame(
        {
            "signal_date": dates,
            "test_asset_0": np.linspace(-0.01, 0.03, len(dates)),
            "test_asset_1": np.linspace(0.02, -0.01, len(dates)),
        }
    )

    summary = pricing_error_summary(monthly, test_assets, ridge=1e-4)

    assert set(summary["return_scale"]) == {"normalized", "raw"}
    assert summary["hjd_pricing_error"].notna().all()

