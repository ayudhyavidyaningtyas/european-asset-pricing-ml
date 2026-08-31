from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neural_sdf import (  # noqa: E402
    NeuralSDFConfig,
    compare_neural_to_ml_portfolios,
    load_neural_sdf_panel,
    run_walk_forward_neural_sdf,
    self_financing_weights,
)


FEATURES = ["x_rank", "y_rank"]


def synthetic_panel(start: str = "2018-01-31", periods: int = 36) -> pd.DataFrame:
    records = []
    for date in pd.date_range(start, periods=periods, freq="ME"):
        for security in range(12):
            signal = (security - 5.5) / 6.0
            secondary = 1.0 if security % 2 == 0 else -1.0
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security:02d}",
                    "target_return_1m": 0.01 + 0.02 * signal - 0.004 * secondary,
                    "model_eligible": True,
                    "company_market_cap": 100.0 + security,
                    "market_cap_percentile": (security + 1) / 12,
                    "x_rank": signal,
                    "y_rank": secondary,
                }
            )
    return pd.DataFrame(records)


def test_self_financing_weights_are_centered_and_gross_scaled():
    weights = self_financing_weights(np.array([3.0, 2.0, 1.0]), gross_leverage=2.0)

    assert np.isclose(weights.sum(), 0.0)
    assert np.isclose(np.abs(weights).sum(), 2.0)
    assert weights[0] > 0
    assert weights[-1] < 0


def test_state_features_merge_on_signal_date_not_target_date(tmp_path: Path):
    panel_path = tmp_path / "panel.parquet"
    panel = synthetic_panel(periods=2)
    panel.to_parquet(panel_path, index=False)
    state_features = pd.DataFrame(
        {
            "date": panel["date"].drop_duplicates().to_list(),
            "state_marker": [10.0, 20.0],
        }
    )

    loaded, state_columns = load_neural_sdf_panel(
        panel_path,
        feature_columns=FEATURES,
        state_features=state_features,
    )

    assert state_columns == ["state_marker"]
    first_signal = loaded[loaded["date"].eq(panel["date"].min())]
    assert first_signal["state_marker"].eq(10.0).all()
    assert not first_signal["state_marker"].eq(20.0).any()


def test_walk_forward_neural_sdf_is_causal_and_outputs_weights():
    panel = synthetic_panel()
    panel["sdf_target_return"] = panel["target_return_1m"]
    config = NeuralSDFConfig(
        first_test_year=2020,
        last_test_year=2020,
        min_monthly_stocks=6,
        min_training_months=10,
        validation_months=4,
        hidden_sizes=(4,),
        epochs=4,
        patience=2,
        learning_rate=0.005,
        minimum_size_percentile=0.0,
        random_state=7,
    )

    monthly, fit_log, weights = run_walk_forward_neural_sdf(
        panel,
        config,
        feature_columns=FEATURES,
    )

    assert set(monthly["model"]) == {"neural_sdf"}
    assert not weights.duplicated(["signal_date", "ric", "model"]).any()
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    assert np.allclose(
        weights.groupby("signal_date")["weight"].sum().to_numpy(),
        0.0,
        atol=1e-8,
    )


def test_neural_to_ml_portfolio_comparison_uses_common_return_months():
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    neural = pd.DataFrame(
        {
            "target_date": dates,
            "model": "neural_sdf",
            "sdf_return": np.linspace(0.01, 0.02, len(dates)),
            "long_short_turnover": 0.5,
        }
    )
    ml = pd.DataFrame(
        {
            "return_date": dates,
            "model": "ridge_return",
            "weighting": "value",
            "universe_variant": "standard_ex_bottom_5pct",
            "gross_long_short_return": np.linspace(0.005, 0.015, len(dates)),
            "long_short_turnover": 0.6,
        }
    )

    comparison = compare_neural_to_ml_portfolios(
        neural,
        ml,
        cost_bps=25,
        blocks=(3,),
        n_boot=100,
        seed=1,
    )

    assert comparison.loc[0, "months"] == 30
    assert comparison.loc[0, "baseline"] == "ridge_return"
    assert comparison.loc[0, "delta_sharpe"] > 0
