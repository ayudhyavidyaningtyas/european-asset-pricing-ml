from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from autoencoder_asset_pricing import (  # noqa: E402
    AutoencoderAssetPricingConfig,
    ConditionalBetaNetwork,
    build_month_batches,
    reconstruction_tensors,
    run_walk_forward_autoencoder,
)


FEATURES = ["x_rank", "y_rank", "z_rank"]


def synthetic_panel(periods: int = 48) -> pd.DataFrame:
    records = []
    for month_index, date in enumerate(pd.date_range("2018-01-31", periods=periods, freq="ME")):
        factor_0 = 0.02 + 0.004 * np.sin(month_index / 3.0)
        factor_1 = -0.01 + 0.003 * np.cos(month_index / 4.0)
        for security in range(8):
            x = -1.0 + security * 2.0 / 7.0
            y = 1.0 if security % 2 == 0 else -1.0
            z = np.sin(security)
            beta_0 = x
            beta_1 = y
            ret = beta_0 * factor_0 + beta_1 * factor_1
            records.append(
                {
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "ric": f"S{security}",
                    "target_return_1m": ret,
                    "autoencoder_target_return": ret,
                    "model_eligible": True,
                    "company_market_cap": 100.0 + security,
                    "market_cap_percentile": 0.2 + security / 20.0,
                    "x_rank": x,
                    "y_rank": y,
                    "z_rank": z,
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


def test_reconstruction_tensors_recover_exact_linear_returns():
    panel = synthetic_panel(periods=1)
    config = AutoencoderAssetPricingConfig(
        min_monthly_stocks=4,
        n_factors=2,
        hidden_sizes=(),
        factor_ridge=1e-8,
    )
    batch = build_month_batches(
        panel,
        FEATURES,
        np.zeros(len(FEATURES)),
        np.ones(len(FEATURES)),
        config,
    )[0]
    model = ConditionalBetaNetwork(
        n_features=len(FEATURES),
        n_factors=2,
        hidden_sizes=(),
        activation="relu",
    )
    with torch.no_grad():
        linear = model.network[0]
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[0, 0] = 1.0
        linear.weight[1, 1] = 1.0

    returns, _, _, reconstructed = reconstruction_tensors(
        model,
        batch,
        config,
        torch.device("cpu"),
        training_returns=False,
    )

    assert torch.mean((returns - reconstructed).square()).item() < 1e-8


def test_walk_forward_autoencoder_is_causal_and_outputs_predictions():
    panel = synthetic_panel(periods=48)
    config = AutoencoderAssetPricingConfig(
        first_test_year=2021,
        last_test_year=2021,
        min_monthly_stocks=4,
        min_training_months=12,
        validation_months=4,
        n_factors=2,
        hidden_sizes=(4,),
        epochs=2,
        patience=2,
        factor_ridge=1e-4,
        random_state=7,
    )

    monthly, fit_log, predictions, factors, weights = run_walk_forward_autoencoder(
        panel,
        config,
        feature_columns=FEATURES,
    )

    assert set(monthly["model"]) == {"conditional_autoencoder"}
    assert not predictions.empty
    assert not factors.empty
    assert not weights.empty
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    assert not predictions.duplicated(["signal_date", "ric", "model"]).any()
    assert not weights.duplicated(["signal_date", "ric", "model"]).any()
    gross = weights.groupby(["signal_date", "model"])["sdf_weight"].apply(
        lambda values: float(values.abs().sum())
    )
    assert np.allclose(gross.to_numpy(), 1.0)
    assert {"predicted_return", "reconstructed_return"}.issubset(predictions.columns)


def test_top_size_universe_cap_keeps_largest_names_deterministically():
    panel = synthetic_panel(periods=1)
    config = AutoencoderAssetPricingConfig(
        min_monthly_stocks=2,
        n_factors=2,
        hidden_sizes=(),
        max_monthly_stocks=3,
        universe_selection="top_size",
    )

    batch = build_month_batches(
        panel,
        FEATURES,
        np.zeros(len(FEATURES)),
        np.ones(len(FEATURES)),
        config,
    )[0]

    # Market cap is 100 + security index, so the three largest are S5, S6, S7.
    assert batch.n_stocks == 3
    assert sorted(batch.rics.tolist()) == ["S5", "S6", "S7"]


def test_random_and_top_size_caps_select_different_universes():
    panel = synthetic_panel(periods=1)
    shared = {
        "min_monthly_stocks": 2,
        "n_factors": 2,
        "hidden_sizes": (),
        "max_monthly_stocks": 3,
    }
    random_config = AutoencoderAssetPricingConfig(**shared, universe_selection="random")
    size_config = AutoencoderAssetPricingConfig(**shared, universe_selection="top_size")

    def universe(config):
        batch = build_month_batches(
            panel,
            FEATURES,
            np.zeros(len(FEATURES)),
            np.ones(len(FEATURES)),
            config,
        )[0]
        return sorted(batch.rics.tolist())

    # The random draw is not a size screen, so it should not reproduce the
    # large-cap universe. This guards against silently reading one as the other.
    assert universe(size_config) == ["S5", "S6", "S7"]
    assert universe(random_config) != universe(size_config)
