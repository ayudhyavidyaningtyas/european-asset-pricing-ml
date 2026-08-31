from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing import RAW_FEATURES  # noqa: E402
from deep_sequence_models import (  # noqa: E402
    SUPPORTED_SEQUENCE_MODELS,
    DeepSequenceConfig,
    SequencePredictionNet,
    build_deep_sequence_outputs,
    build_sequence_index,
    materialize_sequences,
)


FEATURES = [f"{feature}_rank" for feature in RAW_FEATURES]


def synthetic_panel(months: int = 48, securities: int = 20) -> pd.DataFrame:
    records = []
    dates = pd.date_range("2017-01-31", periods=months, freq="ME")
    for month_number, date in enumerate(dates):
        for security in range(securities):
            cross_signal = (security - (securities - 1) / 2) / securities
            time_signal = np.sin(month_number / 6)
            signal = cross_signal + 0.1 * time_signal
            record = {
                "date": date,
                "target_date": date + pd.offsets.MonthEnd(1),
                "ric": f"S{security:03d}",
                "target_return_1m": signal * 0.02,
                "target_return_rank": signal,
                "company_market_cap": 100.0 + security,
                "market_cap_percentile": (security + 1) / securities,
                "screen_country": "GB",
                "TR.TRBCECONOMICSECTOR": "Industrials",
                "eligible": True,
                "model_eligible": True,
                "return_history_n": 24,
                "feature_count": len(FEATURES),
            }
            for index, feature in enumerate(FEATURES):
                record[feature] = signal + index * 0.001
            records.append(record)
    return pd.DataFrame(records)


def test_sequence_index_uses_only_same_security_past_and_current_rows():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-01-31",
                    "2020-03-31",
                ]
            ),
            "ric": ["A", "A", "B", "A"],
            "feature": [1.0, 2.0, 100.0, 3.0],
        }
    )

    sequence_index, history_counts = build_sequence_index(panel, sequence_length=3)

    assert history_counts.tolist() == [1, 2, 1, 3]
    assert sequence_index[3].tolist() == [0, 1, 3]
    assert 2 not in sequence_index[3]


def test_materialized_sequences_pad_missing_history_with_zero():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "ric": ["A", "A"],
            "feature": [1.0, 2.0],
        }
    )
    sequence_index, _ = build_sequence_index(panel, sequence_length=3)
    values = panel[["feature"]].to_numpy(dtype="float32")

    sequence, mask = materialize_sequences(
        values,
        sequence_index,
        np.array([0]),
        np.array([1.0], dtype="float32"),
        np.array([1.0], dtype="float32"),
    )

    assert sequence.shape == (1, 3, 1)
    assert mask.tolist() == [[False, False, True]]
    assert sequence[0, :, 0].tolist() == [0.0, 0.0, 0.0]


def test_all_sequence_model_variants_produce_one_score_per_stock():
    config = DeepSequenceConfig(
        sequence_length=4,
        recurrent_hidden_size=5,
        head_hidden_sizes=(3,),
        recurrent_layers=1,
        dropout=0.0,
    )
    x = torch.randn(7, 4, 6)
    mask = torch.ones(7, 4, dtype=torch.bool)

    for model_name in SUPPORTED_SEQUENCE_MODELS:
        model = SequencePredictionNet(model_name, 6, 4, config)
        prediction = model(x, mask)
        assert prediction.shape == (7,)
        assert torch.isfinite(prediction).all()


def test_deep_sequence_outputs_are_causal_and_writable(tmp_path: Path):
    panel = synthetic_panel(months=48, securities=20)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    output_dir = tmp_path / "sequence_outputs"
    config = DeepSequenceConfig(
        first_test_year=2020,
        last_test_year=2020,
        min_training_rows=10,
        min_training_months=12,
        max_training_rows=200,
        max_validation_rows=80,
        validation_months=6,
        sequence_length=3,
        min_history_observations=1,
        recurrent_hidden_size=4,
        head_hidden_sizes=(4,),
        epochs=2,
        patience=1,
        batch_size=64,
        prediction_batch_size=128,
        device="cpu",
    )

    manifest = build_deep_sequence_outputs(
        panel_path,
        output_dir,
        ["last_mlp"],
        config,
        target_modes=("rank",),
        delisting_audit_path=None,
        feature_set="baseline",
        significance_n_boot=20,
    )

    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    summary = pd.read_csv(output_dir / "model_summary.csv")

    assert manifest["causality_check"]["train_target_after_cutoff"] == 0
    assert manifest["causality_check"]["duplicate_model_security_month_predictions"] == 0
    assert not predictions.empty
    assert set(predictions["model"]) == {"last_mlp_seq3_rank"}
    assert not summary.empty
