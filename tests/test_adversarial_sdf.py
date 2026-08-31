from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adversarial_sdf import (  # noqa: E402
    AdversarialSDFConfig,
    AdversarialSDFModel,
    _sequence_lookup,
    run_walk_forward_adversarial_sdf,
)


FEATURES = ["x_rank", "y_rank"]
STATES = ["state_a", "state_b"]


def synthetic_panel(start: str = "2018-01-31", periods: int = 36) -> pd.DataFrame:
    records = []
    for month_index, date in enumerate(pd.date_range(start, periods=periods, freq="ME")):
        state_a = month_index / 10.0
        state_b = 1.0 if month_index % 2 else -1.0
        for security in range(16):
            signal = (security - 7.5) / 8.0
            secondary = 1.0 if security % 2 == 0 else -1.0
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security:02d}",
                    "sdf_target_return": 0.01
                    + 0.02 * signal
                    + 0.003 * state_a
                    - 0.004 * secondary,
                    "target_return_1m": 0.01 + 0.02 * signal - 0.004 * secondary,
                    "model_eligible": True,
                    "company_market_cap": 100.0 + security,
                    "market_cap_percentile": (security + 1) / 16,
                    "x_rank": signal,
                    "y_rank": secondary,
                    "state_a": state_a,
                    "state_b": state_b,
                }
            )
    return pd.DataFrame(records)


def test_sequence_lookup_pads_with_past_only():
    panel = synthetic_panel(periods=5)
    config = AdversarialSDFConfig(sequence_length=3)

    lookup, columns = _sequence_lookup(panel, STATES, panel, config)

    assert columns == STATES
    dates = sorted(lookup)
    assert lookup[dates[0]].shape == (3, 2)
    assert np.allclose(lookup[dates[0]][0], 0.0)
    assert not np.allclose(lookup[dates[-1]][-1], lookup[dates[0]][-1])


def test_model_has_separate_sdf_and_adversary_lstm_heads():
    model = AdversarialSDFModel(
        firm_features=2,
        state_features=2,
        config=AdversarialSDFConfig(
            state_hidden_size=3,
            sdf_hidden_sizes=(4,),
            adversary_hidden_sizes=(5,),
            test_assets=2,
        ),
    )
    features = torch.zeros(6, 2)
    sequence = torch.zeros(3, 2)

    sdf_scores = model.sdf_network(features, sequence)
    adversary_scores = model.adversary_network(features, sequence)

    assert model.sdf_network.lstm is not model.adversary_network.lstm
    assert sdf_scores.shape == (6, 1)
    assert adversary_scores.shape == (6, 2)


def test_walk_forward_adversarial_sdf_is_causal_and_outputs_test_assets():
    panel = synthetic_panel()
    config = AdversarialSDFConfig(
        first_test_year=2020,
        last_test_year=2020,
        min_monthly_stocks=8,
        min_training_months=10,
        validation_months=4,
        sequence_length=4,
        state_hidden_size=3,
        sdf_hidden_sizes=(4,),
        adversary_hidden_sizes=(4,),
        test_assets=2,
        epochs=2,
        patience=2,
        adversary_steps=1,
        sdf_steps=1,
        minimum_size_percentile=0.0,
        max_monthly_stocks=12,
        random_state=11,
        device="cpu",
    )

    monthly, fit_log, weights = run_walk_forward_adversarial_sdf(
        panel,
        config,
        feature_columns=FEATURES,
        state_columns=STATES,
    )

    assert set(monthly["model"]) == {"adversarial_sdf_lstm_gan"}
    assert "adversarial_test_asset_return_0" in monthly
    assert "adversarial_test_asset_weight_0" in weights
    assert not weights.duplicated(["signal_date", "ric", "model"]).any()
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
