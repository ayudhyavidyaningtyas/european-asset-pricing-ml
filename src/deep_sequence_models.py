"""Deep sequence return-prediction models for the European equity panel.

This module adapts the deep sequence modelling literature to the same
walk-forward prediction and portfolio evaluation setup used by
``asset_pricing_ml.py``.  Each stock-month is represented by the trailing
sequence of its characteristic ranks, ending at the signal month; the target is
the next-month return or return rank.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from asset_pricing_ml import (
    FEATURE_SETS,
    WalkForwardConfig,
    _limit_training_rows,
    construct_monthly_portfolios,
    load_model_panel,
    paired_sharpe_significance,
    portfolio_summary,
    prediction_metrics,
    predictive_accuracy_tests,
    walk_forward_slices,
)


SUPPORTED_SEQUENCE_MODELS = {
    "last_mlp",
    "sequence_mlp",
    "lstm",
    "gru",
    "attention_lstm",
}


@dataclass(frozen=True)
class DeepSequenceConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_training_rows: int = 10_000
    min_training_months: int = 72
    training_window_months: int | None = None
    max_training_rows: int | None = 150_000
    max_validation_rows: int | None = 60_000
    validation_months: int = 24
    sequence_length: int = 12
    min_history_observations: int = 6
    portfolio_quantile: float = 0.10
    cost_grid_bps: tuple[int, ...] = (0, 10, 25, 50)
    recurrent_hidden_size: int = 32
    recurrent_layers: int = 1
    head_hidden_sizes: tuple[int, ...] = (32,)
    dropout: float = 0.10
    epochs: int = 15
    patience: int = 4
    batch_size: int = 8192
    prediction_batch_size: int = 32768
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 5.0
    random_state: int = 42
    device: str = "auto"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def build_sequence_index(
    panel: pd.DataFrame,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every row to its trailing same-security row indices.

    The returned index is causal by construction: the final column is the row
    itself, and earlier columns contain only older rows for the same security.
    Leading missing history is marked with -1 and later materialized as zeros.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    sequence_index = np.full((len(panel), sequence_length), -1, dtype=np.int32)
    history_counts = np.zeros(len(panel), dtype=np.int16)
    dates = pd.to_datetime(panel["date"]).to_numpy()
    for _, raw_positions in panel.groupby("ric", sort=False).indices.items():
        positions = np.asarray(raw_positions, dtype=np.int32)
        order = np.argsort(dates[positions])
        positions = positions[order]
        for local_index, row_position in enumerate(positions):
            start = max(0, local_index - sequence_length + 1)
            history = positions[start : local_index + 1]
            sequence_index[row_position, -len(history) :] = history
            history_counts[row_position] = len(history)
    return sequence_index, history_counts


def fit_feature_standardizer(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = frame[feature_columns].to_numpy(dtype=float, copy=False)
    mean = np.nanmean(values, axis=0).astype("float32")
    scale = np.nanstd(values, axis=0).astype("float32")
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
    return mean, scale


def materialize_sequences(
    feature_values: np.ndarray,
    sequence_index: np.ndarray,
    row_positions: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build standardized sequence tensors for selected panel rows."""
    raw_index = sequence_index[row_positions]
    valid = raw_index >= 0
    safe_index = np.where(valid, raw_index, 0)
    sequences = feature_values[safe_index].astype("float32", copy=False)
    sequences = (sequences - feature_mean.reshape(1, 1, -1)) / feature_scale.reshape(
        1,
        1,
        -1,
    )
    sequences[~valid] = 0.0
    sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
    return sequences.astype("float32", copy=False), valid


def _make_head(
    input_size: int,
    hidden_sizes: tuple[int, ...],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_size
    for hidden_size in hidden_sizes:
        layers.extend([nn.Linear(current, hidden_size), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = hidden_size
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


class SequencePredictionNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_features: int,
        sequence_length: int,
        config: DeepSequenceConfig,
    ):
        super().__init__()
        if model_name not in SUPPORTED_SEQUENCE_MODELS:
            raise ValueError(f"Unknown sequence model: {model_name}")
        self.model_name = model_name
        self.sequence_length = int(sequence_length)
        if model_name == "last_mlp":
            encoded_size = n_features
            self.encoder = None
            self.attention = None
        elif model_name == "sequence_mlp":
            encoded_size = n_features * sequence_length
            self.encoder = None
            self.attention = None
        elif model_name in {"lstm", "attention_lstm"}:
            self.encoder = nn.LSTM(
                input_size=n_features,
                hidden_size=config.recurrent_hidden_size,
                num_layers=config.recurrent_layers,
                dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
                batch_first=True,
            )
            encoded_size = config.recurrent_hidden_size
            self.attention = (
                nn.Linear(config.recurrent_hidden_size, 1)
                if model_name == "attention_lstm"
                else None
            )
        elif model_name == "gru":
            self.encoder = nn.GRU(
                input_size=n_features,
                hidden_size=config.recurrent_hidden_size,
                num_layers=config.recurrent_layers,
                dropout=config.dropout if config.recurrent_layers > 1 else 0.0,
                batch_first=True,
            )
            encoded_size = config.recurrent_hidden_size
            self.attention = None
        self.head = _make_head(encoded_size, config.head_hidden_sizes, config.dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.model_name == "last_mlp":
            encoded = sequence[:, -1, :]
        elif self.model_name == "sequence_mlp":
            encoded = sequence.reshape(sequence.shape[0], -1)
        elif self.model_name == "attention_lstm":
            output, _ = self.encoder(sequence)
            attention_scores = self.attention(output).squeeze(-1)
            if valid_mask is not None:
                attention_scores = attention_scores.masked_fill(
                    ~valid_mask.bool(),
                    -1.0e9,
                )
            weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
            encoded = (output * weights).sum(dim=1)
        elif self.model_name == "lstm":
            _, (hidden, _) = self.encoder(sequence)
            encoded = hidden[-1]
        elif self.model_name == "gru":
            _, hidden = self.encoder(sequence)
            encoded = hidden[-1]
        else:
            raise RuntimeError(f"Unhandled model: {self.model_name}")
        return self.head(encoded).squeeze(-1)


class TorchSequenceRegressor:
    def __init__(
        self,
        model_name: str,
        config: DeepSequenceConfig,
        feature_columns: list[str],
    ):
        self.model_name = model_name
        self.config = config
        self.feature_columns = feature_columns
        self.model: SequencePredictionNet | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.device = resolve_device(config.device)
        self.best_epoch: int | None = None
        self.validation_loss: float | None = None

    def fit(
        self,
        feature_values: np.ndarray,
        sequence_index: np.ndarray,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        target_column: str,
    ) -> "TorchSequenceRegressor":
        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

        self.feature_mean, self.feature_scale = fit_feature_standardizer(
            train,
            self.feature_columns,
        )
        x_train, mask_train = materialize_sequences(
            feature_values,
            sequence_index,
            train.index.to_numpy(dtype=np.int64),
            self.feature_mean,
            self.feature_scale,
        )
        y_train = train[target_column].to_numpy(dtype="float32", copy=False)
        x_valid, mask_valid = materialize_sequences(
            feature_values,
            sequence_index,
            validation.index.to_numpy(dtype=np.int64),
            self.feature_mean,
            self.feature_scale,
        )
        y_valid = validation[target_column].to_numpy(dtype="float32", copy=False)

        train_dataset = TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(mask_train),
            torch.from_numpy(y_train),
        )
        generator = torch.Generator().manual_seed(self.config.random_state)
        loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        valid_sequence = torch.from_numpy(x_valid).to(self.device)
        valid_mask = torch.from_numpy(mask_valid).to(self.device)
        valid_target = torch.from_numpy(y_valid).to(self.device)

        model = SequencePredictionNet(
            self.model_name,
            n_features=len(self.feature_columns),
            sequence_length=self.config.sequence_length,
            config=self.config,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.MSELoss()
        best_state = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        stale_epochs = 0

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            for batch_sequence, batch_mask, batch_target in loader:
                batch_sequence = batch_sequence.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_target = batch_target.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch_sequence, batch_mask), batch_target)
                loss.backward()
                if self.config.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.gradient_clip_norm,
                    )
                optimizer.step()

            model.eval()
            validation_losses = []
            with torch.no_grad():
                for start in range(0, len(valid_sequence), self.config.batch_size):
                    prediction = model(
                        valid_sequence[start : start + self.config.batch_size],
                        valid_mask[start : start + self.config.batch_size],
                    )
                    validation_losses.append(
                        loss_function(
                            prediction,
                            valid_target[start : start + self.config.batch_size],
                        )
                        .detach()
                        .cpu()
                        .item()
                    )
            validation_loss = float(np.mean(validation_losses))
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                self.best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.validation_loss = best_loss
        return self

    def predict(
        self,
        feature_values: np.ndarray,
        sequence_index: np.ndarray,
        row_positions: np.ndarray,
    ) -> np.ndarray:
        if self.model is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Sequence regressor has not been fitted")
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(row_positions), self.config.prediction_batch_size):
                rows = row_positions[start : start + self.config.prediction_batch_size]
                sequence, mask = materialize_sequences(
                    feature_values,
                    sequence_index,
                    rows,
                    self.feature_mean,
                    self.feature_scale,
                )
                sequence_tensor = torch.from_numpy(sequence).to(self.device)
                mask_tensor = torch.from_numpy(mask).to(self.device)
                predictions.append(
                    self.model(sequence_tensor, mask_tensor).detach().cpu().numpy()
                )
        return np.concatenate(predictions).astype("float32", copy=False)


def _split_train_validation(
    train: pd.DataFrame,
    config: DeepSequenceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = np.sort(train["date"].dropna().unique())
    if config.training_window_months is not None:
        months = months[-config.training_window_months :]
        train = train[train["date"].isin(months)].copy()
    validation_months = min(config.validation_months, max(1, len(months) // 5))
    validation_start = pd.Timestamp(months[-validation_months])
    core = train[train["date"].lt(validation_start)]
    validation = train[train["date"].ge(validation_start)]
    if core.empty or validation.empty:
        split = max(1, int(len(train) * 0.8))
        core = train.iloc[:split]
        validation = train.iloc[split:]
    return core, validation


def _cap_frame(
    frame: pd.DataFrame,
    maximum: int | None,
    random_state: int,
) -> pd.DataFrame:
    capped = _limit_training_rows(frame, maximum, random_state)
    return capped.sort_values(["date", "ric"])


def run_sequence_walk_forward(
    panel: pd.DataFrame,
    sequence_index: np.ndarray,
    history_counts: np.ndarray,
    model_names: list[str],
    config: DeepSequenceConfig,
    target_column: str = "target_return_rank",
    target_mode: str = "rank",
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = feature_columns or FEATURE_SETS["baseline"]
    unknown = set(model_names) - SUPPORTED_SEQUENCE_MODELS
    if unknown:
        raise ValueError(f"Unknown sequence models: {sorted(unknown)}")
    if "is_delisting_candidate" not in panel:
        panel = panel.assign(is_delisting_candidate=False)
    if "retire_month" not in panel:
        panel = panel.assign(retire_month=pd.NaT)

    feature_values = panel[feature_columns].to_numpy(dtype="float32", copy=False)
    predictions: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    base_columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "target_return_rank",
        "company_market_cap",
        "market_cap_percentile",
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
        "is_delisting_candidate",
        "retire_month",
    ]
    history_ok = history_counts >= config.min_history_observations

    for year, cutoff, train_mask, test_mask in walk_forward_slices(
        panel,
        config.first_test_year,
        config.last_test_year,
    ):
        available_train = panel.loc[
            train_mask & history_ok & panel[target_column].notna()
        ]
        if (
            len(available_train) < config.min_training_rows
            or available_train["date"].nunique() < config.min_training_months
        ):
            continue
        test = panel.loc[test_mask & history_ok].copy()
        if test.empty:
            continue

        core, validation = _split_train_validation(available_train, config)
        train = _cap_frame(core, config.max_training_rows, config.random_state + year)
        validation = _cap_frame(
            validation,
            config.max_validation_rows,
            config.random_state + year + 10_000,
        )
        if train.empty or validation.empty:
            continue

        for model_name in model_names:
            started = time.perf_counter()
            model = TorchSequenceRegressor(
                model_name,
                config,
                feature_columns,
            ).fit(
                feature_values,
                sequence_index,
                train,
                validation,
                target_column,
            )
            scores = model.predict(
                feature_values,
                sequence_index,
                test.index.to_numpy(dtype=np.int64),
            )
            elapsed = time.perf_counter() - started

            label = f"{model_name}_seq{config.sequence_length}_{target_mode}"
            output = test[base_columns].copy()
            output["prediction"] = scores
            output["model"] = label
            output["base_model"] = model_name
            output["target_mode"] = target_mode
            output["test_year"] = year
            output["train_label_cutoff"] = cutoff
            predictions.append(output)
            fit_records.append(
                {
                    "model": label,
                    "base_model": model_name,
                    "target_mode": target_mode,
                    "test_year": year,
                    "sequence_length": config.sequence_length,
                    "min_history_observations": config.min_history_observations,
                    "train_rows_available": int(len(available_train)),
                    "train_rows_used": int(len(train)),
                    "validation_rows_used": int(len(validation)),
                    "test_rows": int(len(test)),
                    "train_signal_start": str(train["date"].min().date()),
                    "train_signal_end": str(train["date"].max().date()),
                    "validation_signal_start": str(validation["date"].min().date()),
                    "validation_signal_end": str(validation["date"].max().date()),
                    "train_target_end": str(train["target_date"].max().date()),
                    "train_label_cutoff": str(cutoff.date()),
                    "fit_seconds": elapsed,
                    "device": str(model.device),
                    "best_epoch": model.best_epoch,
                    "validation_loss": model.validation_loss,
                    "selected_parameters": json.dumps(
                        {
                            "recurrent_hidden_size": config.recurrent_hidden_size,
                            "recurrent_layers": config.recurrent_layers,
                            "head_hidden_sizes": config.head_hidden_sizes,
                            "dropout": config.dropout,
                            "learning_rate": config.learning_rate,
                            "weight_decay": config.weight_decay,
                        },
                        sort_keys=True,
                    ),
                }
            )

    prediction_frame = (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    )
    return prediction_frame, pd.DataFrame(fit_records)


def build_deep_sequence_outputs(
    panel_path: Path,
    output_dir: Path,
    model_names: list[str],
    config: DeepSequenceConfig,
    target_modes: tuple[str, ...] = ("rank",),
    delisting_audit_path: Path | None = None,
    risk_free: pd.Series | None = None,
    feature_set: str = "compustat_enriched",
    significance_n_boot: int = 2000,
    significance_blocks: tuple[int, ...] = (3, 6, 12),
) -> dict[str, Any]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    feature_columns = FEATURE_SETS[feature_set]
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_model_panel(
        panel_path,
        delisting_audit_path=delisting_audit_path,
        feature_columns=feature_columns,
    )
    sequence_index, history_counts = build_sequence_index(
        panel,
        config.sequence_length,
    )

    prediction_frames: list[pd.DataFrame] = []
    fit_frames: list[pd.DataFrame] = []
    for target_mode in target_modes:
        if target_mode == "rank":
            target_column = "target_return_rank"
        elif target_mode == "return":
            target_column = "target_return_1m"
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
        mode_predictions, mode_fit = run_sequence_walk_forward(
            panel,
            sequence_index,
            history_counts,
            model_names,
            config,
            target_column=target_column,
            target_mode=target_mode,
            feature_columns=feature_columns,
        )
        prediction_frames.append(mode_predictions)
        fit_frames.append(mode_fit)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fit_log = pd.concat(fit_frames, ignore_index=True)
    if predictions.empty:
        raise RuntimeError("Deep sequence run produced no predictions")

    metrics = prediction_metrics(predictions)
    monthly = construct_monthly_portfolios(predictions, config.portfolio_quantile)
    summary = portfolio_summary(
        monthly,
        metrics,
        config.cost_grid_bps,
        risk_free=risk_free,
    )
    significance = paired_sharpe_significance(
        monthly,
        baseline_model=f"{model_names[0]}_seq{config.sequence_length}_rank",
        n_boot=significance_n_boot,
        blocks=significance_blocks,
        risk_free=risk_free,
    )
    loss_tests, ic_tests = predictive_accuracy_tests(predictions)

    predictions.to_parquet(
        output_dir / "predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    fit_log.to_csv(output_dir / "fit_log.csv", index=False)
    metrics.to_csv(output_dir / "prediction_metrics.csv", index=False)
    loss_tests.to_csv(output_dir / "predictive_accuracy_loss_tests.csv", index=False)
    ic_tests.to_csv(output_dir / "predictive_accuracy_ic_tests.csv", index=False)
    monthly.to_csv(output_dir / "monthly_portfolios.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    significance.to_csv(output_dir / "sharpe_significance.csv", index=False)

    manifest = {
        "panel_path": str(panel_path),
        "models": model_names,
        "target_modes": target_modes,
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "method": (
            "Deep sequence modelling with trailing stock-characteristic "
            "sequences; last_mlp is the static current-month benchmark, "
            "sequence_mlp tests flattened history, and recurrent models test "
            "ordered temporal dependence."
        ),
        "config": asdict(config),
        "rows": {
            "input_model_rows": int(len(panel)),
            "predictions": int(len(predictions)),
            "fit_log": int(len(fit_log)),
            "portfolio_months": int(len(monthly)),
            "history_eligible_rows": int(
                (history_counts >= config.min_history_observations).sum()
            ),
        },
        "causality_check": {
            "train_target_after_cutoff": int(
                (
                    pd.to_datetime(fit_log["train_target_end"])
                    > pd.to_datetime(fit_log["train_label_cutoff"])
                ).sum()
            ),
            "duplicate_model_security_month_predictions": int(
                predictions.duplicated(["model", "date", "ric"]).sum()
            ),
        },
        "outputs": {
            "predictions": str(output_dir / "predictions.parquet"),
            "fit_log": str(output_dir / "fit_log.csv"),
            "prediction_metrics": str(output_dir / "prediction_metrics.csv"),
            "monthly_portfolios": str(output_dir / "monthly_portfolios.csv"),
            "model_summary": str(output_dir / "model_summary.csv"),
            "sharpe_significance": str(output_dir / "sharpe_significance.csv"),
        },
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def to_walk_forward_config(config: DeepSequenceConfig) -> WalkForwardConfig:
    """Expose shared portfolio settings for small downstream utilities/tests."""
    return WalkForwardConfig(
        first_test_year=config.first_test_year,
        last_test_year=config.last_test_year,
        min_training_rows=config.min_training_rows,
        max_training_rows=config.max_training_rows,
        portfolio_quantile=config.portfolio_quantile,
        cost_grid_bps=config.cost_grid_bps,
        random_state=config.random_state,
        validation_months=config.validation_months,
    )
