"""Conditional autoencoder asset-pricing model for European equities.

This module implements a compact Gu-Kelly-Xiu-style conditional autoencoder.
Firm characteristics map into nonlinear conditional factor loadings, and each
month's latent factor realization is recovered from contemporaneous returns as
a factor-mimicking portfolio:

    r_{t+1} = beta(X_t) f_{t+1} + u_{t+1}.

The zero-intercept reconstruction is the asset-pricing/no-arbitrage restriction.
Outputs include total R2 from contemporaneous latent factors, predictive R2
from training-sample factor premia, characteristic-managed pricing moments, and
an SDF/MSRR portfolio formed from the learned factor returns.
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


@dataclass(frozen=True)
class AutoencoderAssetPricingConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_monthly_stocks: int = 100
    min_training_months: int = 72
    validation_months: int = 24
    training_window_months: int | None = None
    n_factors: int = 5
    hidden_sizes: tuple[int, ...] = (16,)
    activation: str = "relu"
    epochs: int = 15
    patience: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 5.0
    factor_ridge: float = 1e-4
    sdf_ridge_grid: tuple[float, ...] = (
        1e-6,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
    )
    minimum_size_percentile: float = 0.05
    training_return_clip: float = 1.0
    max_monthly_stocks: int | None = None
    # How max_monthly_stocks picks the monthly universe. "random" draws a random
    # subsample and is the historical default kept for reproducing earlier runs;
    # it is a speed/robustness device, not a liquidity screen. "top_size" keeps
    # the largest names by market cap and is the one that answers "does a
    # tradeable large-cap universe improve implementability".
    universe_selection: str = "random"
    random_state: int = 42
    device: str = "cpu"


@dataclass
class AutoencoderMonthBatch:
    signal_date: pd.Timestamp
    target_date: pd.Timestamp
    rics: np.ndarray
    features: np.ndarray
    training_returns: np.ndarray
    evaluation_returns: np.ndarray
    market_caps: np.ndarray
    market_cap_percentiles: np.ndarray

    @property
    def n_stocks(self) -> int:
        return int(len(self.rics))


class ConditionalBetaNetwork(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_factors: int,
        hidden_sizes: tuple[int, ...],
        activation: str,
    ):
        super().__init__()
        activation_layer: type[nn.Module]
        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "tanh":
            activation_layer = nn.Tanh
        else:
            raise ValueError("activation must be 'relu' or 'tanh'")

        layers: list[nn.Module] = []
        input_size = n_features
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_layer())
            input_size = hidden_size
        layers.append(nn.Linear(input_size, n_factors))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.network(features)
        raw = raw - raw.mean(dim=0, keepdim=True)
        scale = raw.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
        return raw / scale


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def load_autoencoder_panel(
    panel_path: Path,
    risk_free: pd.Series | None,
    feature_columns: list[str],
) -> pd.DataFrame:
    columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "model_eligible",
        "company_market_cap",
        "market_cap_percentile",
        *feature_columns,
    ]
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["target_date"] = pd.to_datetime(panel["target_date"])
    panel = panel[
        panel["model_eligible"].fillna(False)
        & panel["target_return_1m"].notna()
        & panel["target_date"].notna()
    ].copy()
    panel["autoencoder_target_return"] = panel["target_return_1m"].astype(float)
    if risk_free is not None:
        rf = risk_free.rename("RF_EUR").rename_axis("target_date").reset_index()
        rf["target_date"] = pd.to_datetime(rf["target_date"])
        panel = panel.merge(rf, on="target_date", how="left", validate="many_to_one")
        panel["autoencoder_target_return"] = panel["autoencoder_target_return"] - panel[
            "RF_EUR"
        ].fillna(0.0)
    for column in feature_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
    return panel.sort_values(["date", "ric"]).reset_index(drop=True)


def _standardizer(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame[columns].to_numpy(dtype=float, copy=False)
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
    return mean, scale


def _sample_month(
    month: pd.DataFrame,
    maximum: int | None,
    seed: int,
    selection: str = "random",
) -> pd.DataFrame:
    if maximum is None or len(month) <= maximum:
        return month
    if selection == "top_size":
        # Largest names by market cap, ric as a deterministic tie-break so the
        # universe does not depend on incoming row order.
        ranked = month.sort_values(
            ["company_market_cap", "ric"], ascending=[False, True]
        )
        return ranked.head(maximum).sort_values("ric")
    if selection != "random":
        raise ValueError("universe_selection must be 'random' or 'top_size'")
    return month.sample(n=maximum, random_state=seed).sort_values("ric")


def build_month_batches(
    frame: pd.DataFrame,
    feature_columns: list[str],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    config: AutoencoderAssetPricingConfig,
    seed_offset: int = 0,
) -> list[AutoencoderMonthBatch]:
    eligible = frame[
        frame["market_cap_percentile"].ge(config.minimum_size_percentile)
    ].copy()
    batches: list[AutoencoderMonthBatch] = []
    for month_number, (signal_date, month) in enumerate(
        eligible.groupby("date", sort=True)
    ):
        month = _sample_month(
            month,
            config.max_monthly_stocks,
            config.random_state + seed_offset + month_number,
            config.universe_selection,
        )
        if len(month) < config.min_monthly_stocks:
            continue
        x = month[feature_columns].to_numpy(dtype=float, copy=False)
        x = np.nan_to_num(
            (x - feature_mean) / feature_scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        evaluation_returns = month["autoencoder_target_return"].to_numpy(dtype=float)
        training_returns = evaluation_returns.copy()
        if config.training_return_clip > 0:
            training_returns = np.clip(
                training_returns,
                -config.training_return_clip,
                config.training_return_clip,
            )
        batches.append(
            AutoencoderMonthBatch(
                signal_date=pd.Timestamp(signal_date),
                target_date=pd.Timestamp(month["target_date"].max()),
                rics=month["ric"].astype(str).to_numpy(),
                features=x.astype("float32", copy=False),
                training_returns=training_returns.astype("float32", copy=False),
                evaluation_returns=evaluation_returns.astype("float32", copy=False),
                market_caps=month["company_market_cap"].to_numpy(dtype=float),
                market_cap_percentiles=month["market_cap_percentile"].to_numpy(
                    dtype=float
                ),
            )
        )
    return batches


def _month_tensors(
    batch: AutoencoderMonthBatch,
    device: torch.device,
    training_returns: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
    returns = batch.training_returns if training_returns else batch.evaluation_returns
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    return features, returns_tensor


def _factor_realization(
    betas: torch.Tensor,
    returns: torch.Tensor,
    factor_ridge: float,
) -> torch.Tensor:
    n_stocks = betas.shape[0]
    gram = betas.T @ betas / n_stocks
    rhs = betas.T @ returns / n_stocks
    ridge = factor_ridge * torch.eye(
        gram.shape[0],
        dtype=gram.dtype,
        device=gram.device,
    )
    return torch.linalg.solve(gram + ridge, rhs)


def reconstruction_tensors(
    model: ConditionalBetaNetwork,
    batch: AutoencoderMonthBatch,
    config: AutoencoderAssetPricingConfig,
    device: torch.device,
    training_returns: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    features, returns = _month_tensors(batch, device, training_returns)
    betas = model(features)
    factors = _factor_realization(betas, returns, config.factor_ridge)
    reconstructed = betas @ factors
    return returns, betas, factors, reconstructed


def _reconstruction_loss(
    model: ConditionalBetaNetwork,
    batches: list[AutoencoderMonthBatch],
    config: AutoencoderAssetPricingConfig,
    device: torch.device,
    training_returns: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    sse = []
    tss = []
    for batch in batches:
        returns, _, _, reconstructed = reconstruction_tensors(
            model,
            batch,
            config,
            device,
            training_returns=training_returns,
        )
        residual = returns - reconstructed
        losses.append(residual.square().mean())
        sse.append(residual.square().sum())
        tss.append(returns.square().sum())
    loss = torch.stack(losses).mean()
    total_sse = torch.stack(sse).sum()
    total_tss = torch.stack(tss).sum().clamp_min(1e-12)
    diagnostics = {
        "loss": float(loss.detach().cpu().item()),
        "total_r2": float((1.0 - total_sse / total_tss).detach().cpu().item()),
    }
    return loss, diagnostics


def _train_autoencoder(
    model: ConditionalBetaNetwork,
    training_batches: list[AutoencoderMonthBatch],
    validation_batches: list[AutoencoderMonthBatch],
    config: AutoencoderAssetPricingConfig,
    device: torch.device,
) -> tuple[ConditionalBetaNetwork, dict[str, Any]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in training_batches:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _reconstruction_loss(
                model,
                [batch],
                config,
                device,
                training_returns=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
            )
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            validation_loss, validation_diagnostics = _reconstruction_loss(
                model,
                validation_batches or training_batches,
                config,
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
        train_loss, train_diagnostics = _reconstruction_loss(
            model,
            training_batches,
            config,
            device,
            training_returns=True,
        )
        validation_loss, validation_diagnostics = _reconstruction_loss(
            model,
            validation_batches or training_batches,
            config,
            device,
            training_returns=True,
        )
    metadata = {
        "best_epoch": int(best_epoch),
        "fit_seconds": float(time.perf_counter() - started),
        "training_loss": float(train_loss.detach().cpu().item()),
        "validation_loss": float(validation_loss.detach().cpu().item()),
        **{f"training_{key}": value for key, value in train_diagnostics.items()},
        **{f"validation_{key}": value for key, value in validation_diagnostics.items()},
    }
    return model, metadata


def _factor_rows(
    model: ConditionalBetaNetwork,
    batches: list[AutoencoderMonthBatch],
    config: AutoencoderAssetPricingConfig,
    device: torch.device,
    training_returns: bool,
) -> pd.DataFrame:
    records = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            returns, _, factors, reconstructed = reconstruction_tensors(
                model,
                batch,
                config,
                device,
                training_returns=training_returns,
            )
            residual = returns - reconstructed
            record: dict[str, Any] = {
                "signal_date": batch.signal_date,
                "target_date": batch.target_date,
                "n_stocks": batch.n_stocks,
                "reconstruction_sse": float(residual.square().sum().cpu().item()),
                "return_ss": float(returns.square().sum().cpu().item()),
            }
            for index, value in enumerate(factors.detach().cpu().numpy()):
                record[f"factor_{index}"] = float(value)
            records.append(record)
    return pd.DataFrame.from_records(records)


def _fit_factor_sdf(
    core_factors: pd.DataFrame,
    validation_factors: pd.DataFrame,
    config: AutoencoderAssetPricingConfig,
) -> dict[str, Any]:
    factor_columns = [column for column in core_factors if column.startswith("factor_")]
    train_x = core_factors[factor_columns].to_numpy(dtype=float)
    validation_x = validation_factors[factor_columns].to_numpy(dtype=float)
    scale = train_x.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    train = train_x / scale
    validation = validation_x / scale
    y = np.ones(train.shape[0], dtype=float)
    best: dict[str, Any] | None = None
    for ridge_alpha in config.sdf_ridge_grid:
        system = train.T @ train + float(ridge_alpha) * np.eye(train.shape[1])
        rhs = train.T @ y
        try:
            scaled_coefficients = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            scaled_coefficients = np.linalg.lstsq(system, rhs, rcond=None)[0]
        validation_return = validation @ scaled_coefficients
        validation_loss = float(np.mean(np.square(1.0 - validation_return)))
        if best is None or validation_loss < best["sdf_validation_loss"]:
            best = {
                "sdf_ridge_alpha": float(ridge_alpha),
                "sdf_coefficients": scaled_coefficients / scale,
                "sdf_validation_loss": validation_loss,
                "sdf_training_loss": float(
                    np.mean(np.square(1.0 - train @ scaled_coefficients))
                ),
            }
    if best is None:
        raise ValueError("sdf_ridge_grid must contain at least one value")
    return best


def _evaluate_test_batches(
    model: ConditionalBetaNetwork,
    test_batches: list[AutoencoderMonthBatch],
    training_factor_mean: np.ndarray,
    sdf_fit: dict[str, Any],
    config: AutoencoderAssetPricingConfig,
    device: torch.device,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_records = []
    prediction_frames = []
    factor_records = []
    weight_frames = []
    sdf_coefficients = np.asarray(sdf_fit["sdf_coefficients"], dtype=float)
    model.eval()
    with torch.no_grad():
        for batch in test_batches:
            returns, betas, factors, reconstructed = reconstruction_tensors(
                model,
                batch,
                config,
                device,
                training_returns=False,
            )
            predicted = betas @ torch.as_tensor(
                training_factor_mean,
                dtype=torch.float32,
                device=device,
            )
            residual = returns - reconstructed
            predictive_residual = returns - predicted
            x = torch.as_tensor(batch.features, dtype=torch.float32, device=device)
            pricing_moments = (x * predictive_residual.unsqueeze(1)).mean(dim=0)
            factor_np = factors.detach().cpu().numpy()
            reconstructed_np = reconstructed.detach().cpu().numpy()
            predicted_np = predicted.detach().cpu().numpy()
            returns_np = returns.detach().cpu().numpy()
            residual_np = residual.detach().cpu().numpy()
            predictive_residual_np = predictive_residual.detach().cpu().numpy()
            gram = betas.T @ betas / batch.n_stocks
            ridge = config.factor_ridge * torch.eye(
                gram.shape[0],
                dtype=gram.dtype,
                device=device,
            )
            sdf_coefficients_tensor = torch.as_tensor(
                sdf_coefficients,
                dtype=torch.float32,
                device=device,
            )
            stock_sdf_direction = torch.linalg.solve(
                gram + ridge,
                sdf_coefficients_tensor,
            )
            raw_weights = (betas @ stock_sdf_direction) / batch.n_stocks
            gross = raw_weights.abs().sum().clamp_min(1e-12)
            sdf_weights = raw_weights / gross
            sdf_weights_np = sdf_weights.detach().cpu().numpy()
            raw_weights_np = raw_weights.detach().cpu().numpy()
            stock_sdf_return = float(np.dot(sdf_weights_np, returns_np))
            record: dict[str, Any] = {
                "signal_date": batch.signal_date,
                "target_date": batch.target_date,
                "model": "conditional_autoencoder",
                "n_test_stocks": batch.n_stocks,
                "reconstruction_sse": float(np.square(residual_np).sum()),
                "predictive_sse": float(np.square(predictive_residual_np).sum()),
                "return_ss": float(np.square(returns_np).sum()),
                "total_r2_month": float(
                    1.0 - np.square(residual_np).sum() / max(np.square(returns_np).sum(), 1e-12)
                ),
                "predictive_r2_month": float(
                    1.0
                    - np.square(predictive_residual_np).sum()
                    / max(np.square(returns_np).sum(), 1e-12)
                ),
                "sdf_return": float(factor_np @ sdf_coefficients),
                "stock_sdf_return": stock_sdf_return,
                "stock_sdf_gross_weight": float(np.abs(sdf_weights_np).sum()),
                "stock_sdf_net_weight": float(sdf_weights_np.sum()),
                "stock_sdf_weight_hhi": float(np.square(sdf_weights_np).sum()),
                "pricing_moment_l2": float(
                    torch.linalg.vector_norm(pricing_moments).cpu().item()
                ),
                "max_abs_pricing_moment": float(
                    pricing_moments.abs().max().cpu().item()
                ),
            }
            for index, value in enumerate(factor_np):
                record[f"factor_{index}"] = float(value)
            monthly_records.append(record)
            factor_records.append(
                {
                    "signal_date": batch.signal_date,
                    "target_date": batch.target_date,
                    **{f"factor_{index}": float(value) for index, value in enumerate(factor_np)},
                }
            )
            weight_frames.append(
                pd.DataFrame(
                    {
                        "signal_date": batch.signal_date,
                        "target_date": batch.target_date,
                        "ric": batch.rics,
                        "model": "conditional_autoencoder_stock_sdf",
                        "raw_weight": raw_weights_np,
                        "sdf_weight": sdf_weights_np,
                        "raw_score": raw_weights_np * batch.n_stocks,
                        "target_return": returns_np,
                        "market_cap": batch.market_caps,
                        "market_cap_percentile": batch.market_cap_percentiles,
                    }
                )
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "signal_date": batch.signal_date,
                        "target_date": batch.target_date,
                        "ric": batch.rics,
                        "model": "conditional_autoencoder",
                        "target_return": returns_np,
                        "reconstructed_return": reconstructed_np,
                        "predicted_return": predicted_np,
                        "reconstruction_residual": residual_np,
                        "predictive_residual": predictive_residual_np,
                        "market_cap": batch.market_caps,
                        "market_cap_percentile": batch.market_cap_percentiles,
                    }
                )
            )
    monthly = pd.DataFrame.from_records(monthly_records)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    factors = pd.DataFrame.from_records(factor_records)
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else pd.DataFrame()
    )
    return monthly, predictions, factors, weights


def _split_training_months(
    train: pd.DataFrame,
    config: AutoencoderAssetPricingConfig,
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


def run_walk_forward_autoencoder(
    panel: pd.DataFrame,
    config: AutoencoderAssetPricingConfig,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    torch.manual_seed(config.random_state)
    np.random.seed(config.random_state)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    feature_columns = feature_columns or COMPUSTAT_FEATURE_COLUMNS
    device = resolve_device(config.device)
    monthly_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    factor_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []

    for year in range(config.first_test_year, config.last_test_year + 1):
        cutoff = pd.Timestamp(year=year - 1, month=12, day=31)
        train = panel[panel["target_date"].le(cutoff)].copy()
        test = panel[panel["date"].dt.year.eq(year)].copy()
        if test.empty:
            continue
        train = train[
            train["market_cap_percentile"].ge(config.minimum_size_percentile)
        ].copy()
        if train["date"].nunique() < config.min_training_months + config.validation_months:
            continue
        core, validation = _split_training_months(train, config)
        if core["date"].nunique() < config.min_training_months or validation.empty:
            continue
        feature_mean, feature_scale = _standardizer(train, feature_columns)
        core_batches = build_month_batches(
            core,
            feature_columns,
            feature_mean,
            feature_scale,
            config,
            seed_offset=year * 10,
        )
        validation_batches = build_month_batches(
            validation,
            feature_columns,
            feature_mean,
            feature_scale,
            config,
            seed_offset=year * 20,
        )
        test_batches = build_month_batches(
            test,
            feature_columns,
            feature_mean,
            feature_scale,
            config,
            seed_offset=year * 30,
        )
        if len(core_batches) < config.min_training_months or not validation_batches:
            continue
        model = ConditionalBetaNetwork(
            n_features=len(feature_columns),
            n_factors=config.n_factors,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
        )
        model, metadata = _train_autoencoder(
            model,
            core_batches,
            validation_batches,
            config,
            device,
        )
        core_factors = _factor_rows(
            model,
            core_batches,
            config,
            device,
            training_returns=True,
        )
        validation_factors = _factor_rows(
            model,
            validation_batches,
            config,
            device,
            training_returns=True,
        )
        factor_columns = [column for column in core_factors if column.startswith("factor_")]
        training_factor_mean = core_factors[factor_columns].mean().to_numpy(dtype=float)
        sdf_fit = _fit_factor_sdf(core_factors, validation_factors, config)
        monthly, predictions, test_factors, weights = _evaluate_test_batches(
            model,
            test_batches,
            training_factor_mean,
            sdf_fit,
            config,
            device,
            feature_columns,
        )
        monthly_frames.append(monthly)
        prediction_frames.append(predictions)
        factor_frames.append(test_factors)
        weight_frames.append(weights)
        fit_record = {
            "model": "conditional_autoencoder",
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
            "n_features": int(len(feature_columns)),
            "n_factors": int(config.n_factors),
            "device": str(device),
            **metadata,
            "sdf_ridge_alpha": float(sdf_fit["sdf_ridge_alpha"]),
            "sdf_training_loss": float(sdf_fit["sdf_training_loss"]),
            "sdf_validation_loss": float(sdf_fit["sdf_validation_loss"]),
        }
        for index, value in enumerate(training_factor_mean):
            fit_record[f"training_factor_mean_{index}"] = float(value)
        fit_records.append(fit_record)

    monthly = (
        pd.concat(monthly_frames, ignore_index=True)
        if monthly_frames
        else pd.DataFrame()
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    factors = (
        pd.concat(factor_frames, ignore_index=True)
        if factor_frames
        else pd.DataFrame()
    )
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else pd.DataFrame()
    )
    fit_log = pd.DataFrame.from_records(fit_records)
    return monthly, fit_log, predictions, factors, weights


def summarize_autoencoder(monthly: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, group in monthly.groupby("model", sort=True):
        returns = group["sdf_return"].astype(float)
        total_sse = float(group["reconstruction_sse"].sum())
        predictive_sse = float(group["predictive_sse"].sum())
        return_ss = max(float(group["return_ss"].sum()), 1e-12)
        annualized_return = float(returns.mean() * 12.0)
        annualized_volatility = float(returns.std(ddof=1) * math.sqrt(12.0))
        stock_returns = (
            group["stock_sdf_return"].astype(float)
            if "stock_sdf_return" in group
            else pd.Series(dtype=float)
        )
        stock_annualized_return = float(stock_returns.mean() * 12.0)
        stock_annualized_volatility = float(
            stock_returns.std(ddof=1) * math.sqrt(12.0)
        )
        records.append(
            {
                "model": model,
                "months": int(len(group)),
                "total_r2": float(1.0 - total_sse / return_ss),
                "predictive_r2": float(1.0 - predictive_sse / return_ss),
                "mean_monthly_total_r2": float(group["total_r2_month"].mean()),
                "mean_monthly_predictive_r2": float(
                    group["predictive_r2_month"].mean()
                ),
                "annualized_sdf_return": annualized_return,
                "annualized_sdf_volatility": annualized_volatility,
                "sdf_sharpe": annualized_return / annualized_volatility
                if annualized_volatility > 0
                else np.nan,
                "annualized_stock_sdf_return": stock_annualized_return,
                "annualized_stock_sdf_volatility": stock_annualized_volatility,
                "stock_sdf_sharpe": stock_annualized_return / stock_annualized_volatility
                if stock_annualized_volatility > 0
                else np.nan,
                "average_stock_sdf_net_weight": float(
                    group["stock_sdf_net_weight"].mean()
                )
                if "stock_sdf_net_weight" in group
                else np.nan,
                "average_stock_sdf_weight_hhi": float(
                    group["stock_sdf_weight_hhi"].mean()
                )
                if "stock_sdf_weight_hhi" in group
                else np.nan,
                "average_pricing_moment_l2": float(group["pricing_moment_l2"].mean()),
                "max_abs_pricing_moment": float(group["max_abs_pricing_moment"].max()),
                "average_n_test_stocks": float(group["n_test_stocks"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def build_autoencoder_outputs(
    panel_path: Path,
    output_dir: Path,
    config: AutoencoderAssetPricingConfig,
    risk_free: pd.Series | None = None,
    feature_set: str = "compustat_enriched",
) -> dict[str, object]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    feature_columns = FEATURE_SETS[feature_set]
    panel = load_autoencoder_panel(panel_path, risk_free, feature_columns)
    monthly, fit_log, predictions, factors, weights = run_walk_forward_autoencoder(
        panel,
        config,
        feature_columns=feature_columns,
    )
    if monthly.empty:
        raise RuntimeError("Autoencoder walk-forward produced no monthly returns")
    summary = summarize_autoencoder(monthly)

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "autoencoder_monthly.csv", index=False)
    fit_log.to_csv(output_dir / "autoencoder_fit_log.csv", index=False)
    summary.to_csv(output_dir / "autoencoder_summary.csv", index=False)
    factors.to_csv(output_dir / "autoencoder_factors.csv", index=False)
    weights.to_parquet(
        output_dir / "autoencoder_weights.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    predictions.to_parquet(
        output_dir / "autoencoder_predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    manifest: dict[str, object] = {
        "config": asdict(config),
        "panel_path": str(panel_path),
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "rows": {
            "input_rows": int(len(panel)),
            "monthly": int(len(monthly)),
            "fit_log": int(len(fit_log)),
            "predictions": int(len(predictions)),
            "factors": int(len(factors)),
            "weights": int(len(weights)),
            "summary": int(len(summary)),
        },
        "return_definition": "excess_return" if risk_free is not None else "raw_return",
        "source_paper": "Gu, Kelly and Xiu, Autoencoder Asset Pricing Models",
        "objective": (
            "conditional beta autoencoder with zero-intercept latent-factor "
            "return reconstruction and factor-SDF MSRR evaluation"
        ),
        "causality_check": {
            "train_target_after_cutoff": int(
                (
                    pd.to_datetime(fit_log["train_target_end"])
                    > pd.to_datetime(fit_log["train_label_cutoff"])
                ).sum()
            )
            if not fit_log.empty
            else 0,
            "duplicate_prediction_security_months": int(
                predictions.duplicated(["signal_date", "ric", "model"]).sum()
            )
            if not predictions.empty
            else 0,
            "duplicate_weight_security_months": int(
                weights.duplicated(["signal_date", "ric", "model"]).sum()
            )
            if not weights.empty
            else 0,
        },
    }
    (output_dir / "autoencoder_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
