"""Adversarial no-arbitrage SDF with LSTM state encoders.

This module implements a compact Chen-Pelger-Zhu-style adaptation for the
European equity panel.  It is deliberately separate from ``neural_sdf.py``:
that file learns a direct long-short utility portfolio, while this module
trains an SDF network against an adversarial test-asset network through the
minimax no-arbitrage moment

    min_omega max_g || E[(1 - omega_t' R_{t+1}) R_{i,t+1} g(I_t, I_{i,t})] ||^2.

Both the SDF and adversary have their own LSTM state encoder and feed-forward
firm-characteristic network.
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

from asset_pricing_ml import COMPUSTAT_FEATURE_COLUMNS, FEATURE_SETS
from neural_sdf import load_neural_sdf_panel, load_state_features


@dataclass(frozen=True)
class AdversarialSDFConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_monthly_stocks: int = 100
    min_training_months: int = 72
    validation_months: int = 24
    training_window_months: int | None = None
    sequence_length: int = 12
    state_hidden_size: int = 8
    sdf_hidden_sizes: tuple[int, ...] = (32, 16)
    adversary_hidden_sizes: tuple[int, ...] = (32, 16)
    test_assets: int = 4
    epochs: int = 20
    patience: int = 5
    adversary_steps: int = 1
    sdf_steps: int = 1
    learning_rate_sdf: float = 0.001
    learning_rate_adversary: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 5.0
    sdf_gross_leverage: float = 1.0
    adversary_gross_leverage: float = 1.0
    minimum_size_percentile: float = 0.05
    training_return_clip: float = 1.0
    max_monthly_stocks: int | None = None
    random_state: int = 42
    device: str = "auto"


@dataclass
class AdversarialMonthBatch:
    signal_date: pd.Timestamp
    target_date: pd.Timestamp
    rics: np.ndarray
    firm_features: np.ndarray
    state_sequence: np.ndarray
    training_returns: np.ndarray
    evaluation_returns: np.ndarray
    market_caps: np.ndarray
    market_cap_percentiles: np.ndarray

    @property
    def n_stocks(self) -> int:
        return int(len(self.rics))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _standardizer(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame[columns].to_numpy(dtype=float, copy=False)
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
    return mean, scale


def _state_by_date(panel: pd.DataFrame, state_columns: list[str]) -> pd.DataFrame:
    if not state_columns:
        dates = pd.DataFrame({"date": sorted(panel["date"].dropna().unique())})
        dates["state_constant"] = 0.0
        return dates
    state = (
        panel[["date", *state_columns]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    state[state_columns] = state[state_columns].apply(pd.to_numeric, errors="coerce")
    return state


def _sequence_lookup(
    panel: pd.DataFrame,
    state_columns: list[str],
    train: pd.DataFrame,
    config: AdversarialSDFConfig,
) -> tuple[dict[pd.Timestamp, np.ndarray], list[str]]:
    state = _state_by_date(panel, state_columns)
    effective_columns = [column for column in state.columns if column != "date"]
    train_dates = set(pd.to_datetime(train["date"]).dropna().unique())
    train_state = state[state["date"].isin(train_dates)]
    mean, scale = _standardizer(train_state, effective_columns)
    values = state[effective_columns].to_numpy(dtype=float, copy=False)
    values = np.nan_to_num((values - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    dates = pd.to_datetime(state["date"]).tolist()
    lookup: dict[pd.Timestamp, np.ndarray] = {}
    zero = np.zeros(values.shape[1], dtype=np.float32)
    for index, date in enumerate(dates):
        start = max(0, index - config.sequence_length + 1)
        sequence = values[start : index + 1]
        if len(sequence) < config.sequence_length:
            padding = np.tile(zero, (config.sequence_length - len(sequence), 1))
            sequence = np.vstack([padding, sequence])
        lookup[pd.Timestamp(date)] = sequence.astype("float32", copy=False)
    return lookup, effective_columns


def _sample_month(
    month: pd.DataFrame,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    if maximum is None or len(month) <= maximum:
        return month
    return month.sample(n=maximum, random_state=seed).sort_values("ric")


def build_month_batches(
    frame: pd.DataFrame,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    sequence_lookup: dict[pd.Timestamp, np.ndarray],
    config: AdversarialSDFConfig,
    seed_offset: int = 0,
) -> list[AdversarialMonthBatch]:
    eligible = frame[
        frame["market_cap_percentile"].ge(config.minimum_size_percentile)
    ].copy()
    batches: list[AdversarialMonthBatch] = []
    for month_number, (signal_date, month) in enumerate(
        eligible.groupby("date", sort=True)
    ):
        month = _sample_month(
            month,
            config.max_monthly_stocks,
            config.random_state + seed_offset + month_number,
        )
        if len(month) < config.min_monthly_stocks:
            continue
        date = pd.Timestamp(signal_date)
        if date not in sequence_lookup:
            continue
        x = month[feature_columns].to_numpy(dtype=float, copy=False)
        x = np.nan_to_num(
            (x - feature_mean) / feature_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        evaluation_returns = month["sdf_target_return"].to_numpy(dtype=float)
        training_returns = evaluation_returns.copy()
        if config.training_return_clip > 0:
            training_returns = np.clip(
                training_returns,
                -config.training_return_clip,
                config.training_return_clip,
            )
        batches.append(
            AdversarialMonthBatch(
                signal_date=date,
                target_date=pd.Timestamp(month["target_date"].max()),
                rics=month["ric"].astype(str).to_numpy(),
                firm_features=x.astype("float32", copy=False),
                state_sequence=sequence_lookup[date],
                training_returns=training_returns.astype("float32", copy=False),
                evaluation_returns=evaluation_returns.astype("float32", copy=False),
                market_caps=month["company_market_cap"].to_numpy(dtype=float),
                market_cap_percentiles=month["market_cap_percentile"].to_numpy(
                    dtype=float
                ),
            )
        )
    return batches


def _split_training_months(
    train: pd.DataFrame,
    config: AdversarialSDFConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = np.sort(train["date"].dropna().unique())
    if config.training_window_months is not None:
        months = months[-config.training_window_months :]
        train = train[train["date"].isin(months)].copy()
    validation_months = min(config.validation_months, max(1, len(months) // 5))
    validation_start = pd.Timestamp(months[-validation_months])
    core = train[train["date"].lt(validation_start)]
    validation = train[train["date"].ge(validation_start)]
    return core, validation


class LSTMConditioningNetwork(nn.Module):
    def __init__(
        self,
        firm_features: int,
        state_features: int,
        state_hidden_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_features,
            hidden_size=state_hidden_size,
            batch_first=True,
        )
        layers: list[nn.Module] = []
        input_size = firm_features + state_hidden_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU()])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, output_size))
        self.head = nn.Sequential(*layers)

    def forward(self, firm_features: torch.Tensor, state_sequence: torch.Tensor) -> torch.Tensor:
        if state_sequence.ndim == 2:
            state_sequence = state_sequence.unsqueeze(0)
        _, (hidden, _) = self.lstm(state_sequence)
        state = hidden[-1].expand(firm_features.shape[0], -1)
        return self.head(torch.cat([firm_features, state], dim=1))


class AdversarialSDFModel(nn.Module):
    def __init__(
        self,
        firm_features: int,
        state_features: int,
        config: AdversarialSDFConfig,
    ):
        super().__init__()
        self.config = config
        self.sdf_network = LSTMConditioningNetwork(
            firm_features=firm_features,
            state_features=state_features,
            state_hidden_size=config.state_hidden_size,
            hidden_sizes=config.sdf_hidden_sizes,
            output_size=1,
        )
        self.adversary_network = LSTMConditioningNetwork(
            firm_features=firm_features,
            state_features=state_features,
            state_hidden_size=config.state_hidden_size,
            hidden_sizes=config.adversary_hidden_sizes,
            output_size=config.test_assets,
        )

    @staticmethod
    def _gross_normalize(
        scores: torch.Tensor,
        gross_leverage: float,
        center: bool,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if center:
            scores = scores - scores.mean(dim=0, keepdim=True)
        gross = scores.abs().sum(dim=0, keepdim=True).clamp_min(eps)
        return scores / gross * gross_leverage

    def month_tensors(
        self,
        batch: AdversarialMonthBatch,
        device: torch.device,
        training_returns: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.as_tensor(batch.firm_features, dtype=torch.float32, device=device)
        sequence = torch.as_tensor(batch.state_sequence, dtype=torch.float32, device=device)
        returns = (
            batch.training_returns if training_returns else batch.evaluation_returns
        )
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
        return features, sequence, returns_tensor

    def moment_vector(
        self,
        batch: AdversarialMonthBatch,
        device: torch.device,
        training_returns: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, sequence, returns = self.month_tensors(batch, device, training_returns)
        omega_scores = self.sdf_network(features, sequence).squeeze(-1)
        omega = self._gross_normalize(
            omega_scores.unsqueeze(1),
            self.config.sdf_gross_leverage,
            center=False,
        ).squeeze(1)
        g_scores = self.adversary_network(features, sequence)
        g_weights = self._gross_normalize(
            g_scores,
            self.config.adversary_gross_leverage,
            center=True,
        )
        sdf_return = torch.sum(omega * returns)
        sdf_kernel = 1.0 - sdf_return
        moments = torch.sum(sdf_kernel * returns.unsqueeze(1) * g_weights, dim=0)
        return moments, sdf_return, omega, g_weights

    def moment_loss(
        self,
        batches: list[AdversarialMonthBatch],
        device: torch.device,
        training_returns: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not batches:
            raise ValueError("At least one monthly batch is required")
        moments = []
        sdf_returns = []
        for batch in batches:
            moment, sdf_return, _, _ = self.moment_vector(
                batch,
                device,
                training_returns=training_returns,
            )
            moments.append(moment)
            sdf_returns.append(sdf_return)
        moment_matrix = torch.stack(moments)
        average_moment = moment_matrix.mean(dim=0)
        loss = torch.mean(average_moment.square())
        sdf_return_tensor = torch.stack(sdf_returns)
        diagnostics = {
            "moment_loss": float(loss.detach().cpu().item()),
            "moment_l2": float(torch.linalg.vector_norm(average_moment).detach().cpu().item()),
            "max_abs_moment": float(average_moment.abs().max().detach().cpu().item()),
            "mean_sdf_return": float(sdf_return_tensor.mean().detach().cpu().item()),
            "sdf_return_volatility": float(
                sdf_return_tensor.std(unbiased=True).detach().cpu().item()
            )
            if len(sdf_returns) > 1
            else 0.0,
        }
        return loss, diagnostics


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


def _train_adversarial_model(
    model: AdversarialSDFModel,
    training_batches: list[AdversarialMonthBatch],
    validation_batches: list[AdversarialMonthBatch],
    config: AdversarialSDFConfig,
    device: torch.device,
) -> tuple[AdversarialSDFModel, dict[str, Any]]:
    model.to(device)
    sdf_optimizer = torch.optim.AdamW(
        model.sdf_network.parameters(),
        lr=config.learning_rate_sdf,
        weight_decay=config.weight_decay,
    )
    adversary_optimizer = torch.optim.AdamW(
        model.adversary_network.parameters(),
        lr=config.learning_rate_adversary,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    final_diagnostics: dict[str, float] = {}
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        _set_trainable(model.sdf_network, False)
        _set_trainable(model.adversary_network, True)
        for _ in range(config.adversary_steps):
            adversary_optimizer.zero_grad(set_to_none=True)
            loss, _ = model.moment_loss(training_batches, device, training_returns=True)
            (-loss).backward()
            torch.nn.utils.clip_grad_norm_(
                model.adversary_network.parameters(),
                config.gradient_clip_norm,
            )
            adversary_optimizer.step()

        _set_trainable(model.sdf_network, True)
        _set_trainable(model.adversary_network, False)
        for _ in range(config.sdf_steps):
            sdf_optimizer.zero_grad(set_to_none=True)
            loss, final_diagnostics = model.moment_loss(
                training_batches,
                device,
                training_returns=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.sdf_network.parameters(),
                config.gradient_clip_norm,
            )
            sdf_optimizer.step()
        _set_trainable(model.sdf_network, True)
        _set_trainable(model.adversary_network, True)

        model.eval()
        with torch.no_grad():
            validation_loss, validation_diagnostics = model.moment_loss(
                validation_batches or training_batches,
                device,
                training_returns=True,
            )
            validation_value = float(validation_loss.detach().cpu().item())
        if validation_value < best_validation_loss - 1e-10:
            best_validation_loss = validation_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_loss, train_diagnostics = model.moment_loss(
            training_batches,
            device,
            training_returns=True,
        )
        validation_loss, validation_diagnostics = model.moment_loss(
            validation_batches or training_batches,
            device,
            training_returns=True,
        )
    metadata = {
        "best_epoch": int(best_epoch),
        "fit_seconds": float(time.perf_counter() - started),
        "training_moment_loss": float(train_loss.detach().cpu().item()),
        "validation_moment_loss": float(validation_loss.detach().cpu().item()),
        **{f"training_{key}": value for key, value in train_diagnostics.items()},
        **{f"validation_{key}": value for key, value in validation_diagnostics.items()},
        **{f"last_epoch_{key}": value for key, value in final_diagnostics.items()},
    }
    return model, metadata


def _weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    return {
        "gross_weight": float(np.abs(weights).sum()),
        "net_weight": float(weights.sum()),
        "long_weight": float(weights[weights > 0].sum()),
        "short_weight": float(weights[weights < 0].sum()),
        "weight_hhi": float(np.square(weights).sum()),
    }


def _evaluate_batches(
    model: AdversarialSDFModel,
    batches: list[AdversarialMonthBatch],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_records: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            moments, sdf_return, omega, g_weights = model.moment_vector(
                batch,
                device,
                training_returns=False,
            )
            omega_np = omega.detach().cpu().numpy()
            g_np = g_weights.detach().cpu().numpy()
            moment_np = moments.detach().cpu().numpy()
            test_asset_returns = g_np.T @ batch.evaluation_returns
            record: dict[str, Any] = {
                "signal_date": batch.signal_date,
                "target_date": batch.target_date,
                "model": "adversarial_sdf_lstm_gan",
                "sdf_return": float(sdf_return.detach().cpu().item()),
                "n_test_stocks": batch.n_stocks,
                "pricing_moment_l2": float(np.linalg.norm(moment_np)),
                "max_abs_pricing_moment": float(np.max(np.abs(moment_np))),
                **_weight_diagnostics(omega_np),
            }
            for index, value in enumerate(moment_np):
                record[f"pricing_moment_{index}"] = float(value)
            for index, value in enumerate(test_asset_returns):
                record[f"adversarial_test_asset_return_{index}"] = float(value)
            monthly_records.append(record)
            weight_data: dict[str, Any] = {
                "signal_date": batch.signal_date,
                "target_date": batch.target_date,
                "ric": batch.rics,
                "model": "adversarial_sdf_lstm_gan",
                "sdf_weight": omega_np,
                "target_return": batch.evaluation_returns,
                "market_cap": batch.market_caps,
                "market_cap_percentile": batch.market_cap_percentiles,
            }
            for index in range(g_np.shape[1]):
                weight_data[f"adversarial_test_asset_weight_{index}"] = g_np[:, index]
            weight_frames.append(pd.DataFrame(weight_data))
    monthly = pd.DataFrame.from_records(monthly_records)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    return monthly, weights


def run_walk_forward_adversarial_sdf(
    panel: pd.DataFrame,
    config: AdversarialSDFConfig,
    feature_columns: list[str] | None = None,
    state_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    feature_columns = feature_columns or COMPUSTAT_FEATURE_COLUMNS
    state_columns = state_columns or []
    device = resolve_device(config.device)
    panel = panel.sort_values(["date", "ric"]).reset_index(drop=True)
    monthly_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []

    for year in range(config.first_test_year, config.last_test_year + 1):
        cutoff = pd.Timestamp(year=year - 1, month=12, day=31)
        train = panel[panel["target_date"].le(cutoff)].copy()
        test = panel[panel["date"].dt.year.eq(year)].copy()
        if test.empty:
            continue
        train = train[train["market_cap_percentile"].ge(config.minimum_size_percentile)]
        months_available = int(train["date"].nunique())
        if months_available < config.min_training_months + max(1, config.validation_months):
            continue
        core, validation = _split_training_months(train, config)
        if core["date"].nunique() < config.min_training_months or validation.empty:
            continue
        feature_mean, feature_scale = _standardizer(train, feature_columns)
        sequence_lookup, effective_state_columns = _sequence_lookup(
            panel,
            state_columns,
            train,
            config,
        )
        core_batches = build_month_batches(
            core,
            feature_columns,
            feature_mean,
            feature_scale,
            sequence_lookup,
            config,
            seed_offset=year * 10,
        )
        validation_batches = build_month_batches(
            validation,
            feature_columns,
            feature_mean,
            feature_scale,
            sequence_lookup,
            config,
            seed_offset=year * 20,
        )
        test_batches = build_month_batches(
            test,
            feature_columns,
            feature_mean,
            feature_scale,
            sequence_lookup,
            config,
            seed_offset=year * 30,
        )
        if len(core_batches) < config.min_training_months or not validation_batches:
            continue
        model = AdversarialSDFModel(
            firm_features=len(feature_columns),
            state_features=len(effective_state_columns),
            config=config,
        )
        model, metadata = _train_adversarial_model(
            model,
            core_batches,
            validation_batches,
            config,
            device,
        )
        monthly, weights = _evaluate_batches(model, test_batches, device)
        monthly_frames.append(monthly)
        weight_frames.append(weights)
        fit_records.append(
            {
                "model": "adversarial_sdf_lstm_gan",
                "test_year": year,
                "train_signal_start": train["date"].min(),
                "train_signal_end": train["date"].max(),
                "train_target_end": train["target_date"].max(),
                "train_label_cutoff": cutoff,
                "core_months": int(len(core_batches)),
                "validation_months": int(len(validation_batches)),
                "test_months": int(len(test_batches)),
                "train_rows_available": int(len(train)),
                "core_rows": int(sum(batch.n_stocks for batch in core_batches)),
                "validation_rows": int(
                    sum(batch.n_stocks for batch in validation_batches)
                ),
                "test_rows": int(sum(batch.n_stocks for batch in test_batches)),
                "state_columns": json.dumps(effective_state_columns),
                "device": str(device),
                **metadata,
            }
        )

    monthly = (
        pd.concat(monthly_frames, ignore_index=True)
        if monthly_frames
        else pd.DataFrame()
    )
    fit_log = pd.DataFrame.from_records(fit_records)
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else pd.DataFrame()
    )
    return monthly, fit_log, weights


def summarize_adversarial_sdf(monthly: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, group in monthly.groupby("model", sort=True):
        returns = group["sdf_return"].astype(float)
        annualized_return = float(returns.mean() * 12.0)
        annualized_volatility = float(returns.std(ddof=1) * math.sqrt(12.0))
        records.append(
            {
                "model": model,
                "months": int(len(group)),
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe": annualized_return / annualized_volatility
                if annualized_volatility > 0
                else np.nan,
                "average_pricing_moment_l2": float(group["pricing_moment_l2"].mean()),
                "max_abs_pricing_moment": float(group["max_abs_pricing_moment"].max()),
                "average_n_test_stocks": float(group["n_test_stocks"].mean()),
                "average_gross_weight": float(group["gross_weight"].mean()),
                "average_net_weight": float(group["net_weight"].mean()),
                "average_weight_hhi": float(group["weight_hhi"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def build_adversarial_sdf_outputs(
    panel_path: Path,
    output_dir: Path,
    config: AdversarialSDFConfig,
    risk_free: pd.Series | None = None,
    feature_set: str = "compustat_enriched",
    market_state_path: Path | None = None,
    additional_state_path: Path | None = None,
) -> dict[str, object]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    feature_columns = FEATURE_SETS[feature_set]
    state_features, _ = load_state_features(
        market_state_path,
        additional_state_path,
    )
    panel, state_columns = load_neural_sdf_panel(
        panel_path,
        risk_free=risk_free,
        feature_columns=feature_columns,
        state_features=state_features,
    )
    monthly, fit_log, weights = run_walk_forward_adversarial_sdf(
        panel,
        config,
        feature_columns=feature_columns,
        state_columns=state_columns,
    )
    if monthly.empty:
        raise RuntimeError("Adversarial SDF walk-forward produced no monthly returns")
    summary = summarize_adversarial_sdf(monthly)

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "adversarial_sdf_monthly.csv", index=False)
    fit_log.to_csv(output_dir / "adversarial_sdf_fit_log.csv", index=False)
    summary.to_csv(output_dir / "adversarial_sdf_summary.csv", index=False)
    weights.to_parquet(
        output_dir / "adversarial_sdf_weights.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    manifest: dict[str, object] = {
        "config": asdict(config),
        "panel_path": str(panel_path),
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "state_columns": state_columns,
        "rows": {
            "input_rows": int(len(panel)),
            "monthly": int(len(monthly)),
            "fit_log": int(len(fit_log)),
            "weights": int(len(weights)),
            "summary": int(len(summary)),
        },
        "return_definition": "excess_return" if risk_free is not None else "raw_return",
        "objective": (
            "min_omega max_g squared no-arbitrage pricing moments with "
            "separate SDF and adversarial LSTM state encoders"
        ),
        "source_paper": "Chen, Pelger and Zhu, Deep Learning in Asset Pricing",
        "causality_check": {
            "train_target_after_cutoff": int(
                (
                    pd.to_datetime(fit_log["train_target_end"])
                    > pd.to_datetime(fit_log["train_label_cutoff"])
                ).sum()
            )
            if not fit_log.empty
            else 0,
            "duplicate_weight_security_months": int(
                weights.duplicated(["signal_date", "ric", "model"]).sum()
            )
            if not weights.empty
            else 0,
        },
    }
    (output_dir / "adversarial_sdf_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
