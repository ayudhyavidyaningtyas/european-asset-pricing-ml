"""Full AIPM transformer SDF adaptation for European equities.

This module implements the nonlinear portfolio transformer from Kelly,
Kuznetsov, Malamud and Xu's "Artificial Intelligence Asset Pricing Models"
alongside the paper's natural benchmarks:

* BSV: own-asset linear characteristic SDF.
* Linear attention: closed-form linear portfolio transformer.
* DKKM-style random-feature SDF: own-asset nonlinear random features.
* Own-asset MLP: same residual feed-forward blocks without cross-asset attention.
* Nonlinear transformer: multi-head softmax attention, residual feed-forward blocks
  and a final linear SDF layer.

The neural models minimize the MSRR objective,

    E[(1 - w(X_t; theta)' R_{t+1})^2],

using month-by-month Adam updates. Outputs include gross-normalized OOS SDF
returns and pricing moments for characteristic-managed test assets.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from torch import nn

from asset_pricing_ml import FEATURE_SETS


@dataclass(frozen=True)
class AIPMFullTransformerConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_monthly_stocks: int = 100
    min_training_months: int = 60
    validation_months: int = 12
    training_window_months: int | None = 72
    refit_frequency: str = "annual"
    minimum_size_percentile: float = 0.05
    max_monthly_stocks: int | None = 750
    gross_leverage: float = 1.0
    training_return_clip: float = 1.0
    linear_attention_features: int | None = None
    random_feature_count: int = 256
    transformer_blocks: int = 2
    attention_heads: int = 1
    feedforward_width: int = 64
    epochs: int = 8
    patience: int = 3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 5.0
    attention_temperature: float | None = None
    implementability_aware_training: bool = False
    neural_position_limit: float = 0.0
    neural_net_exposure_penalty: float = 0.0
    neural_turnover_penalty: float = 0.0
    neural_spread_cost_penalty: float = 0.0
    neural_impact_cost_penalty: float = 0.0
    neural_training_aum_eur: float = 100_000_000.0
    execution_fallback_half_spread_bps: float = 25.0
    execution_impact_coefficient: float = 0.10
    seeds: tuple[int, ...] = (0,)
    random_state: int = 42
    device: str = "cpu"
    hac_lags: int = 6
    pricing_error_ridge: float = 1e-4
    ridge_grid: tuple[float, ...] = (
        1e-10,
        1e-8,
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


@dataclass
class AIPMFullMonth:
    signal_date: pd.Timestamp
    target_date: pd.Timestamp
    rics: np.ndarray
    features: np.ndarray
    training_returns: np.ndarray
    evaluation_returns: np.ndarray
    market_caps: np.ndarray
    market_cap_percentiles: np.ndarray
    half_spread_bps: np.ndarray
    adv_eur: np.ndarray
    idio_vol_36m: np.ndarray

    @property
    def n_stocks(self) -> int:
        return int(len(self.rics))


@dataclass
class ClosedFormModel:
    model: str
    coefficients: np.ndarray
    ridge_alpha: float
    training_loss: float
    validation_loss: float
    train_mean_return: float
    validation_mean_return: float
    n_parameters: int


@dataclass
class RandomFeatureSpec:
    weights: np.ndarray
    biases: np.ndarray


@dataclass
class NeuralFit:
    model: str
    networks: list[nn.Module]
    seeds: tuple[int, ...]
    best_epochs: list[int]
    training_losses: list[float]
    validation_losses: list[float]
    fit_seconds: float
    n_parameters: int


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def load_aipm_full_panel(
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
        "turnover_12m",
        "volatility_12m",
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
    panel["aipm_target_return"] = panel["target_return_1m"].astype(float)
    if risk_free is not None:
        rf = risk_free.rename("RF_EUR").rename_axis("target_date").reset_index()
        rf["target_date"] = pd.to_datetime(rf["target_date"])
        panel = panel.merge(rf, on="target_date", how="left", validate="many_to_one")
        panel["aipm_target_return"] = panel["aipm_target_return"] - panel[
            "RF_EUR"
        ].fillna(0.0)
    for column in feature_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
    return panel.sort_values(["date", "ric"]).reset_index(drop=True)


def _select_month_stocks(
    month: pd.DataFrame,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    if maximum is None or len(month) <= maximum:
        return month
    ordered = month.sort_values(
        ["company_market_cap", "ric"],
        ascending=[False, True],
        na_position="last",
    )
    top = ordered.head(maximum)
    if len(top) < maximum:
        return month.sample(n=maximum, random_state=seed).sort_values("ric")
    return top.sort_values("ric")


def build_months(
    panel: pd.DataFrame,
    feature_columns: list[str],
    config: AIPMFullTransformerConfig,
) -> list[AIPMFullMonth]:
    eligible = panel[
        panel["market_cap_percentile"].ge(config.minimum_size_percentile)
    ].copy()
    months: list[AIPMFullMonth] = []
    for month_number, (signal_date, month) in enumerate(
        eligible.groupby("date", sort=True)
    ):
        month = _select_month_stocks(
            month,
            config.max_monthly_stocks,
            config.random_state + month_number,
        )
        if len(month) < config.min_monthly_stocks:
            continue
        features = month[feature_columns].to_numpy(dtype=float, copy=False)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        evaluation_returns = month["aipm_target_return"].to_numpy(dtype=float)
        training_returns = evaluation_returns.copy()
        if config.training_return_clip > 0:
            training_returns = np.clip(
                training_returns,
                -config.training_return_clip,
                config.training_return_clip,
            )
        months.append(
            AIPMFullMonth(
                signal_date=pd.Timestamp(signal_date),
                target_date=pd.Timestamp(month["target_date"].max()),
                rics=month["ric"].astype(str).to_numpy(),
                features=features.astype("float32", copy=False),
                training_returns=training_returns.astype("float32", copy=False),
                evaluation_returns=evaluation_returns.astype("float32", copy=False),
                market_caps=month["company_market_cap"].to_numpy(dtype=float),
                market_cap_percentiles=month["market_cap_percentile"].to_numpy(
                    dtype=float
                ),
                half_spread_bps=_execution_half_spread_bps(month, config),
                adv_eur=_execution_adv_eur(month),
                idio_vol_36m=_execution_idio_vol(month),
            )
        )
    return months


def _execution_half_spread_bps(
    month: pd.DataFrame,
    config: AIPMFullTransformerConfig,
) -> np.ndarray:
    if "half_spread_bps" in month:
        values = pd.to_numeric(month["half_spread_bps"], errors="coerce")
    else:
        values = pd.Series(np.nan, index=month.index)
    return (
        values.fillna(config.execution_fallback_half_spread_bps)
        .clip(lower=0.1, upper=500.0)
        .to_numpy(dtype="float32")
    )


def _execution_adv_eur(month: pd.DataFrame) -> np.ndarray:
    market_cap = pd.to_numeric(month["company_market_cap"], errors="coerce")
    if "turnover_12m" in month:
        turnover = pd.to_numeric(month["turnover_12m"], errors="coerce")
    else:
        turnover = pd.Series(np.nan, index=month.index)
    adv = market_cap * turnover
    fallback = market_cap * 0.0005
    adv = adv.where(adv.gt(0), fallback)
    return adv.fillna(10_000.0).clip(lower=10_000.0).to_numpy(dtype="float32")


def _execution_idio_vol(month: pd.DataFrame) -> np.ndarray:
    if "idio_vol_36m" in month:
        values = pd.to_numeric(month["idio_vol_36m"], errors="coerce")
    elif "volatility_12m" in month:
        values = pd.to_numeric(month["volatility_12m"], errors="coerce")
    else:
        values = pd.Series(np.nan, index=month.index)
    return values.fillna(0.20).clip(lower=0.02, upper=0.75).to_numpy(dtype="float32")


def _month_factor_returns(month: AIPMFullMonth, returns: np.ndarray) -> np.ndarray:
    return month.features.T @ returns / month.n_stocks


def _month_covariance(month: AIPMFullMonth, n_features: int | None = None) -> np.ndarray:
    x = month.features if n_features is None else month.features[:, :n_features]
    return x.T @ x / month.n_stocks


def _closed_form_basis_vector(
    month: AIPMFullMonth,
    model: str,
    training_returns: bool,
    linear_attention_features: int | None,
    random_feature_spec: RandomFeatureSpec | None = None,
) -> np.ndarray:
    returns = month.training_returns if training_returns else month.evaluation_returns
    if model == "bsv":
        return _month_factor_returns(month, returns)
    if model == "linear_attention":
        x = (
            month.features
            if linear_attention_features is None
            else month.features[:, :linear_attention_features]
        )
        factor_returns = x.T @ returns / month.n_stocks
        covariance = x.T @ x / month.n_stocks
        return np.kron(factor_returns, covariance.reshape(-1, order="C"))
    if model == "dkkm_random_features":
        if random_feature_spec is None:
            raise ValueError("random_feature_spec is required for DKKM basis")
        transformed = random_feature_matrix(month, random_feature_spec)
        return transformed.T @ returns / month.n_stocks
    raise ValueError(f"unknown closed-form model {model!r}")


def _basis_matrix(
    months: list[AIPMFullMonth],
    model: str,
    training_returns: bool,
    linear_attention_features: int | None,
    random_feature_spec: RandomFeatureSpec | None = None,
) -> np.ndarray:
    if not months:
        return np.empty((0, 0), dtype=float)
    rows = [
        _closed_form_basis_vector(
            month,
            model,
            training_returns,
            linear_attention_features,
            random_feature_spec,
        )
        for month in months
    ]
    return np.vstack(rows).astype("float64", copy=False)


def _ridge_msrr(
    train_basis: np.ndarray,
    validation_basis: np.ndarray,
    ridge_grid: tuple[float, ...],
) -> tuple[np.ndarray, float, float, float, float, float]:
    if train_basis.ndim != 2 or validation_basis.ndim != 2:
        raise ValueError("basis inputs must be two-dimensional")
    if train_basis.shape[0] == 0 or train_basis.shape[1] == 0:
        raise ValueError("training basis must be non-empty")
    scale = train_basis.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    train = train_basis / scale
    validation = validation_basis / scale
    target = np.ones(train.shape[0], dtype=float)
    kernel = train @ train.T
    identity = np.eye(kernel.shape[0])
    best: dict[str, Any] | None = None
    for ridge_alpha in ridge_grid:
        system = kernel + float(ridge_alpha) * identity
        try:
            dual = np.linalg.solve(system, target)
        except np.linalg.LinAlgError:
            dual = np.linalg.lstsq(system, target, rcond=None)[0]
        scaled_coefficients = train.T @ dual
        train_fitted = train @ scaled_coefficients
        validation_fitted = validation @ scaled_coefficients
        validation_loss = float(np.mean(np.square(1.0 - validation_fitted)))
        record = {
            "ridge_alpha": float(ridge_alpha),
            "scaled_coefficients": scaled_coefficients,
            "training_loss": float(np.mean(np.square(1.0 - train_fitted))),
            "validation_loss": validation_loss,
            "train_mean": float(np.mean(train_fitted)),
            "validation_mean": float(np.mean(validation_fitted)),
        }
        if best is None or record["validation_loss"] < best["validation_loss"]:
            best = record
    if best is None:
        raise ValueError("ridge grid must be non-empty")
    coefficients = best["scaled_coefficients"] / scale
    return (
        coefficients.astype("float64", copy=False),
        best["ridge_alpha"],
        best["training_loss"],
        best["validation_loss"],
        best["train_mean"],
        best["validation_mean"],
    )


def fit_closed_form_model(
    training_months: list[AIPMFullMonth],
    validation_months: list[AIPMFullMonth],
    model: str,
    config: AIPMFullTransformerConfig,
    random_feature_spec: RandomFeatureSpec | None = None,
) -> ClosedFormModel:
    train_basis = _basis_matrix(
        training_months,
        model,
        True,
        config.linear_attention_features,
        random_feature_spec,
    )
    validation_basis = _basis_matrix(
        validation_months,
        model,
        True,
        config.linear_attention_features,
        random_feature_spec,
    )
    coefficients, alpha, train_loss, val_loss, train_mean, val_mean = _ridge_msrr(
        train_basis,
        validation_basis,
        config.ridge_grid,
    )
    return ClosedFormModel(
        model=model,
        coefficients=coefficients,
        ridge_alpha=alpha,
        training_loss=train_loss,
        validation_loss=val_loss,
        train_mean_return=train_mean,
        validation_mean_return=val_mean,
        n_parameters=int(len(coefficients)),
    )


def random_feature_spec(
    n_features: int,
    n_random_features: int,
    seed: int,
) -> RandomFeatureSpec:
    rng = np.random.default_rng(seed)
    weights = rng.normal(
        loc=0.0,
        scale=1.0 / math.sqrt(max(n_features, 1)),
        size=(n_features, n_random_features),
    )
    biases = rng.normal(loc=0.0, scale=0.1, size=n_random_features)
    return RandomFeatureSpec(
        weights=weights.astype("float32", copy=False),
        biases=biases.astype("float32", copy=False),
    )


def random_feature_matrix(
    month: AIPMFullMonth,
    spec: RandomFeatureSpec,
) -> np.ndarray:
    values = month.features @ spec.weights + spec.biases
    return np.maximum(values, 0.0).astype("float32", copy=False)


def _closed_form_raw_scores(
    month: AIPMFullMonth,
    fitted: ClosedFormModel,
    config: AIPMFullTransformerConfig,
    random_feature_spec: RandomFeatureSpec | None = None,
) -> np.ndarray:
    if fitted.model == "bsv":
        return month.features @ fitted.coefficients
    if fitted.model == "linear_attention":
        x = (
            month.features
            if config.linear_attention_features is None
            else month.features[:, : config.linear_attention_features]
        )
        n_features = x.shape[1]
        covariance = x.T @ x / month.n_stocks
        covariance_vector = covariance.reshape(-1, order="C")
        coefficient_matrix = fitted.coefficients.reshape(
            n_features,
            n_features * n_features,
        )
        effective_coefficients = coefficient_matrix @ covariance_vector
        return x @ effective_coefficients
    if fitted.model == "dkkm_random_features":
        if random_feature_spec is None:
            raise ValueError("random_feature_spec is required for DKKM scores")
        return random_feature_matrix(month, random_feature_spec) @ fitted.coefficients
    raise ValueError(f"unknown closed-form model {fitted.model!r}")


class PortfolioAttention(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_heads: int,
        attention_temperature: float | None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.temperature = (
            float(attention_temperature)
            if attention_temperature is not None
            else math.sqrt(max(n_features, 1))
        )
        self.query = nn.ParameterList(
            [nn.Parameter(torch.empty(n_features, n_features)) for _ in range(n_heads)]
        )
        self.key = nn.ParameterList(
            [nn.Parameter(torch.empty(n_features, n_features)) for _ in range(n_heads)]
        )
        self.value = nn.ParameterList(
            [nn.Parameter(torch.empty(n_features, n_features)) for _ in range(n_heads)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for params in (self.query, self.key, self.value):
            for parameter in params:
                nn.init.normal_(parameter, mean=0.0, std=1.0 / math.sqrt(parameter.shape[0]))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros_like(features)
        attention_to_save: torch.Tensor | None = None
        for head_index in range(self.n_heads):
            q = features @ self.query[head_index]
            k = features @ self.key[head_index]
            v = features @ self.value[head_index]
            logits = q @ k.T / self.temperature
            attention = torch.softmax(logits, dim=1)
            if head_index == 0:
                attention_to_save = attention
            output = output + attention @ v
        if attention_to_save is None:
            attention_to_save = torch.empty(
                (features.shape[0], features.shape[0]),
                dtype=features.dtype,
                device=features.device,
            )
        return output, attention_to_save


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_heads: int,
        feedforward_width: int,
        attention_temperature: float | None,
        use_attention: bool,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.attention = (
            PortfolioAttention(n_features, n_heads, attention_temperature)
            if use_attention
            else None
        )
        self.feedforward = nn.Sequential(
            nn.Linear(n_features, feedforward_width),
            nn.ReLU(),
            nn.Linear(feedforward_width, n_features),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_matrix: torch.Tensor | None = None
        if self.attention is not None:
            attention_output, attention_matrix = self.attention(features)
            features = features + attention_output
        features = features + self.feedforward(features)
        return features, attention_matrix


class NonlinearPortfolioTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_blocks: int,
        n_heads: int,
        feedforward_width: int,
        attention_temperature: float | None,
        use_attention: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_features,
                    n_heads,
                    feedforward_width,
                    attention_temperature,
                    use_attention,
                )
                for _ in range(n_blocks)
            ]
        )
        self.final = nn.Linear(n_features, 1, bias=False)
        nn.init.normal_(self.final.weight, mean=0.0, std=1.0 / math.sqrt(n_features))

    def transformed_features(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attention_matrix: torch.Tensor | None = None
        for block in self.blocks:
            features, maybe_attention = block(features)
            if attention_matrix is None and maybe_attention is not None:
                attention_matrix = maybe_attention
        if return_attention:
            return features, attention_matrix
        return features, None

    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        transformed, attention = self.transformed_features(features, return_attention)
        scores = self.final(transformed).squeeze(-1)
        return scores, attention


def _month_tensors(
    month: AIPMFullMonth,
    device: torch.device,
    training_returns: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(month.features, dtype=torch.float32, device=device)
    returns = month.training_returns if training_returns else month.evaluation_returns
    returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    return features, returns_tensor


def _uses_implementability_training(config: AIPMFullTransformerConfig) -> bool:
    return bool(
        config.implementability_aware_training
        or config.neural_position_limit > 0
        or config.neural_net_exposure_penalty > 0
        or config.neural_turnover_penalty > 0
        or config.neural_spread_cost_penalty > 0
        or config.neural_impact_cost_penalty > 0
    )


def _torch_normalized_weights(
    scores: torch.Tensor,
    config: AIPMFullTransformerConfig,
    eps: float = 1e-12,
) -> torch.Tensor:
    gross = scores.abs().sum().clamp_min(eps)
    weights = scores / gross * float(config.gross_leverage)
    if config.neural_position_limit > 0:
        limit = float(config.neural_position_limit)
        weights = torch.clamp(weights, min=-limit, max=limit)
        capped_gross = weights.abs().sum().clamp_min(eps)
        weights = weights / capped_gross * float(config.gross_leverage)
    return weights


def _previous_weight_tensor(
    month: AIPMFullMonth,
    previous_weights: dict[str, float] | None,
    device: torch.device,
) -> torch.Tensor:
    if not previous_weights:
        return torch.zeros(month.n_stocks, dtype=torch.float32, device=device)
    values = np.array(
        [previous_weights.get(str(ric), 0.0) for ric in month.rics],
        dtype="float32",
    )
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _neural_execution_penalty(
    month: AIPMFullMonth,
    weights: torch.Tensor,
    previous_weights: dict[str, float] | None,
    config: AIPMFullTransformerConfig,
    device: torch.device,
) -> torch.Tensor:
    penalty = torch.zeros((), dtype=torch.float32, device=device)
    previous = _previous_weight_tensor(month, previous_weights, device)
    delta = weights - previous
    abs_delta = delta.abs()
    if config.neural_turnover_penalty > 0:
        penalty = penalty + float(config.neural_turnover_penalty) * abs_delta.sum()
    if config.neural_spread_cost_penalty > 0:
        half_spread = torch.as_tensor(
            month.half_spread_bps,
            dtype=torch.float32,
            device=device,
        )
        spread_cost = torch.sum(abs_delta * half_spread / 10_000.0)
        penalty = penalty + float(config.neural_spread_cost_penalty) * spread_cost
    if config.neural_impact_cost_penalty > 0:
        adv = torch.as_tensor(month.adv_eur, dtype=torch.float32, device=device)
        idio_vol = torch.as_tensor(
            month.idio_vol_36m,
            dtype=torch.float32,
            device=device,
        )
        participation = abs_delta * float(config.neural_training_aum_eur) / adv.clamp_min(1.0)
        daily_vol = idio_vol / math.sqrt(21.0)
        impact_bps = (
            float(config.execution_impact_coefficient)
            * daily_vol
            * torch.sqrt(participation.clamp_min(0.0))
            * 10_000.0
        )
        impact_cost = torch.sum(abs_delta * impact_bps / 10_000.0)
        penalty = penalty + float(config.neural_impact_cost_penalty) * impact_cost
    if config.neural_net_exposure_penalty > 0:
        penalty = penalty + float(config.neural_net_exposure_penalty) * weights.sum().square()
    return penalty


def _neural_month_return(
    network: nn.Module,
    month: AIPMFullMonth,
    device: torch.device,
    training_returns: bool,
    config: AIPMFullTransformerConfig | None = None,
    previous_weights: dict[str, float] | None = None,
    return_weights: bool = False,
) -> torch.Tensor:
    features, returns = _month_tensors(month, device, training_returns)
    scores, _ = network(features)
    if config is not None and _uses_implementability_training(config):
        weights = _torch_normalized_weights(scores, config)
        month_return = torch.sum(weights * returns)
        if training_returns:
            month_return = month_return - _neural_execution_penalty(
                month,
                weights,
                previous_weights,
                config,
                device,
            )
        if return_weights:
            return month_return, weights
        return month_return
    month_return = torch.mean(scores * returns)
    if return_weights:
        return month_return, scores.detach()
    return month_return


def _neural_loss(
    network: nn.Module,
    months: list[AIPMFullMonth],
    device: torch.device,
    training_returns: bool,
    config: AIPMFullTransformerConfig | None = None,
) -> float:
    if not months:
        return float("nan")
    losses: list[float] = []
    network.eval()
    with torch.no_grad():
        previous_weights: dict[str, float] | None = None
        for month in months:
            if config is not None and _uses_implementability_training(config):
                month_return, weights = _neural_month_return(
                    network,
                    month,
                    device,
                    training_returns,
                    config,
                    previous_weights=previous_weights,
                    return_weights=True,
                )
                previous_weights = {
                    str(ric): float(weight)
                    for ric, weight in zip(
                        month.rics,
                        weights.detach().cpu().numpy(),
                        strict=True,
                    )
                }
            else:
                month_return = _neural_month_return(
                    network,
                    month,
                    device,
                    training_returns,
                    config,
                )
            losses.append(float(torch.square(1.0 - month_return).detach().cpu()))
    return float(np.mean(losses))


def _set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_neural_model(
    training_months: list[AIPMFullMonth],
    validation_months: list[AIPMFullMonth],
    model: str,
    n_features: int,
    config: AIPMFullTransformerConfig,
    device: torch.device,
) -> NeuralFit:
    if model not in {"own_asset_mlp", "nonlinear_transformer"}:
        raise ValueError(f"unknown neural model {model!r}")
    use_attention = model == "nonlinear_transformer"
    networks: list[nn.Module] = []
    best_epochs: list[int] = []
    training_losses: list[float] = []
    validation_losses: list[float] = []
    started = time.time()
    for seed in config.seeds:
        _set_torch_seed(seed)
        network = NonlinearPortfolioTransformer(
            n_features=n_features,
            n_blocks=config.transformer_blocks,
            n_heads=config.attention_heads,
            feedforward_width=config.feedforward_width,
            attention_temperature=config.attention_temperature,
            use_attention=use_attention,
        ).to(device)
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_validation = float("inf")
        best_epoch = 0
        patience_left = config.patience
        rng = np.random.default_rng(config.random_state + seed)
        for epoch in range(1, config.epochs + 1):
            network.train()
            order = np.arange(len(training_months))
            if not _uses_implementability_training(config):
                rng.shuffle(order)
            previous_weights: dict[str, float] | None = None
            for index in order:
                month = training_months[int(index)]
                optimizer.zero_grad(set_to_none=True)
                if _uses_implementability_training(config):
                    month_return, weights = _neural_month_return(
                        network,
                        month,
                        device,
                        True,
                        config,
                        previous_weights=previous_weights,
                        return_weights=True,
                    )
                    previous_weights = {
                        str(ric): float(weight)
                        for ric, weight in zip(
                            month.rics,
                            weights.detach().cpu().numpy(),
                            strict=True,
                        )
                    }
                else:
                    month_return = _neural_month_return(
                        network,
                        month,
                        device,
                        True,
                        config,
                    )
                loss = torch.square(1.0 - month_return)
                loss.backward()
                if config.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        network.parameters(),
                        max_norm=config.gradient_clip_norm,
                    )
                optimizer.step()
            validation_loss = _neural_loss(
                network,
                validation_months,
                device,
                True,
                config,
            )
            if validation_loss + 1e-12 < best_validation:
                best_validation = validation_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in network.state_dict().items()
                }
                patience_left = config.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            network.load_state_dict(best_state)
        network = network.to(torch.device("cpu"))
        network.eval()
        networks.append(network)
        best_epochs.append(best_epoch)
        training_losses.append(
            _neural_loss(
                network,
                training_months,
                torch.device("cpu"),
                True,
                config,
            )
        )
        validation_losses.append(
            _neural_loss(
                network,
                validation_months,
                torch.device("cpu"),
                True,
                config,
            )
        )
    n_parameters = int(sum(parameter.numel() for parameter in networks[0].parameters()))
    return NeuralFit(
        model=model,
        networks=networks,
        seeds=tuple(config.seeds),
        best_epochs=best_epochs,
        training_losses=training_losses,
        validation_losses=validation_losses,
        fit_seconds=float(time.time() - started),
        n_parameters=n_parameters,
    )


def _neural_raw_scores(month: AIPMFullMonth, fitted: NeuralFit) -> np.ndarray:
    scores: list[np.ndarray] = []
    features = torch.as_tensor(month.features, dtype=torch.float32)
    with torch.no_grad():
        for network in fitted.networks:
            network.eval()
            raw_scores, _ = network(features)
            scores.append(raw_scores.detach().cpu().numpy().astype(float))
    return np.mean(np.vstack(scores), axis=0)


def extract_attention_summary(
    month: AIPMFullMonth,
    fitted: NeuralFit,
    max_pairs: int = 200,
) -> pd.DataFrame:
    if fitted.model != "nonlinear_transformer" or not fitted.networks:
        return pd.DataFrame()
    features = torch.as_tensor(month.features, dtype=torch.float32)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for seed_index, network in enumerate(fitted.networks):
            network.eval()
            _, attention = network(features, return_attention=True)
            if attention is None or attention.numel() == 0:
                continue
            matrix = attention.detach().cpu().numpy()
            rows = np.arange(matrix.shape[0])
            top_index = np.argmax(matrix, axis=1)
            top_weight = matrix[rows, top_index]
            top_count = min(max_pairs, len(rows))
            chosen = np.argsort(-top_weight)[:top_count]
            for row in chosen:
                records.append(
                    {
                        "signal_date": month.signal_date,
                        "seed": int(fitted.seeds[seed_index]),
                        "source_ric": month.rics[row],
                        "attended_ric": month.rics[top_index[row]],
                        "attention_weight": float(top_weight[row]),
                    }
                )
    return pd.DataFrame.from_records(records)


def _normalize_weights(
    raw_weights: np.ndarray,
    gross_leverage: float,
    eps: float = 1e-12,
) -> np.ndarray:
    gross = float(np.abs(raw_weights).sum())
    if gross <= eps:
        return np.zeros_like(raw_weights)
    return raw_weights / gross * gross_leverage


def evaluate_model_month(
    month: AIPMFullMonth,
    model_name: str,
    raw_scores: np.ndarray,
    config: AIPMFullTransformerConfig,
    fit_metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_weights = raw_scores.astype(float, copy=False) / month.n_stocks
    sdf_weights = _normalize_weights(raw_weights, config.gross_leverage)
    raw_sdf_return = float(raw_weights @ month.evaluation_returns)
    sdf_return = float(sdf_weights @ month.evaluation_returns)
    factor_returns = _month_factor_returns(month, month.evaluation_returns)
    sdf_moments = (1.0 - sdf_return) * factor_returns
    raw_sdf_moments = (1.0 - raw_sdf_return) * factor_returns
    record: dict[str, Any] = {
        "signal_date": month.signal_date,
        "target_date": month.target_date,
        "model": model_name,
        "raw_sdf_return": raw_sdf_return,
        "sdf_return": sdf_return,
        "n_test_stocks": month.n_stocks,
        "raw_gross_weight": float(np.abs(raw_weights).sum()),
        "gross_weight": float(np.abs(sdf_weights).sum()),
        "net_weight": float(sdf_weights.sum()),
        "long_weight": float(sdf_weights[sdf_weights > 0].sum()),
        "short_weight": float(sdf_weights[sdf_weights < 0].sum()),
        "weight_hhi": float(np.square(sdf_weights).sum()),
        "pricing_moment_l2": float(np.sqrt(np.mean(np.square(sdf_moments)))),
        "raw_pricing_moment_l2": float(np.sqrt(np.mean(np.square(raw_sdf_moments)))),
        **fit_metadata,
    }
    weights = pd.DataFrame(
        {
            "signal_date": month.signal_date,
            "target_date": month.target_date,
            "ric": month.rics,
            "model": model_name,
            "raw_weight": raw_weights,
            "sdf_weight": sdf_weights,
            "raw_score": raw_scores,
            "target_return": month.evaluation_returns,
            "market_cap": month.market_caps,
            "market_cap_percentile": month.market_cap_percentiles,
        }
    )
    return record, weights


def add_weight_turnover(monthly: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty or weights.empty:
        out = monthly.copy()
        out["weight_turnover"] = np.nan
        return out
    turnover_records: list[dict[str, Any]] = []
    frame = weights[["signal_date", "ric", "model", "sdf_weight"]].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    for model, group in frame.groupby("model", sort=True):
        wide = (
            group.pivot(index="signal_date", columns="ric", values="sdf_weight")
            .fillna(0.0)
            .sort_index()
        )
        turnover = wide.diff().abs().sum(axis=1) / 2.0
        turnover.iloc[0] = np.nan
        turnover_records.extend(
            {
                "signal_date": signal_date,
                "model": model,
                "weight_turnover": float(value) if pd.notna(value) else np.nan,
            }
            for signal_date, value in turnover.items()
        )
    out = monthly.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    return out.merge(
        pd.DataFrame.from_records(turnover_records),
        on=["signal_date", "model"],
        how="left",
    )


def _split_training_months(
    months: list[AIPMFullMonth],
    cutoff: pd.Timestamp,
    config: AIPMFullTransformerConfig,
) -> tuple[list[AIPMFullMonth], list[AIPMFullMonth]]:
    train = [month for month in months if month.target_date <= cutoff]
    if config.training_window_months is not None:
        train = train[-config.training_window_months :]
    validation_count = min(config.validation_months, max(1, len(train) // 5))
    core = train[:-validation_count]
    validation = train[-validation_count:]
    return core, validation


def _refit_jobs(
    months: list[AIPMFullMonth],
    config: AIPMFullTransformerConfig,
) -> list[tuple[str, pd.Timestamp, list[AIPMFullMonth]]]:
    test_months = [
        month
        for month in months
        if config.first_test_year <= month.signal_date.year <= config.last_test_year
    ]
    if config.refit_frequency == "annual":
        jobs = []
        for year in range(config.first_test_year, config.last_test_year + 1):
            year_months = [month for month in test_months if month.signal_date.year == year]
            if year_months:
                jobs.append(
                    (
                        str(year),
                        pd.Timestamp(year=year - 1, month=12, day=31),
                        year_months,
                    )
                )
        return jobs
    if config.refit_frequency == "monthly":
        return [
            (
                month.signal_date.strftime("%Y-%m"),
                month.signal_date,
                [month],
            )
            for month in test_months
        ]
    raise ValueError("refit_frequency must be 'annual' or 'monthly'")


def run_walk_forward_full_aipm(
    months: list[AIPMFullMonth],
    config: AIPMFullTransformerConfig,
    models: tuple[str, ...] = (
        "bsv",
        "linear_attention",
        "dkkm_random_features",
        "own_asset_mlp",
        "nonlinear_transformer",
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(months, key=lambda month: month.signal_date)
    device = resolve_device(config.device)
    n_features = months[0].features.shape[1] if months else 0
    random_spec = random_feature_spec(
        n_features,
        config.random_feature_count,
        config.random_state,
    )
    monthly_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    attention_frames: list[pd.DataFrame] = []
    jobs = _refit_jobs(months, config)
    for job_id, cutoff, test_months in jobs:
        core_months, validation_months = _split_training_months(months, cutoff, config)
        if len(core_months) < config.min_training_months or not validation_months:
            continue
        fitted_models: dict[str, ClosedFormModel | NeuralFit] = {}
        for model in models:
            started = time.time()
            if model in {"bsv", "linear_attention", "dkkm_random_features"}:
                fitted = fit_closed_form_model(
                    core_months,
                    validation_months,
                    model,
                    config,
                    random_feature_spec=random_spec
                    if model == "dkkm_random_features"
                    else None,
                )
                fit_seconds = time.time() - started
                fit_records.append(
                    {
                        "model": model,
                        "refit_id": job_id,
                        "train_start": core_months[0].signal_date,
                        "train_signal_end": core_months[-1].signal_date,
                        "train_target_end": core_months[-1].target_date,
                        "validation_start": validation_months[0].signal_date,
                        "validation_end": validation_months[-1].signal_date,
                        "validation_target_end": validation_months[-1].target_date,
                        "train_label_cutoff": cutoff,
                        "core_months": int(len(core_months)),
                        "validation_months": int(len(validation_months)),
                        "test_months": int(len(test_months)),
                        "n_parameters": fitted.n_parameters,
                        "ridge_alpha": fitted.ridge_alpha,
                        "best_epoch_mean": np.nan,
                        "training_loss": fitted.training_loss,
                        "validation_loss": fitted.validation_loss,
                        "fit_seconds": fit_seconds,
                    }
                )
            elif model in {"own_asset_mlp", "nonlinear_transformer"}:
                fitted = fit_neural_model(
                    core_months,
                    validation_months,
                    model,
                    n_features,
                    config,
                    device,
                )
                fit_records.append(
                    {
                        "model": model,
                        "refit_id": job_id,
                        "train_start": core_months[0].signal_date,
                        "train_signal_end": core_months[-1].signal_date,
                        "train_target_end": core_months[-1].target_date,
                        "validation_start": validation_months[0].signal_date,
                        "validation_end": validation_months[-1].signal_date,
                        "validation_target_end": validation_months[-1].target_date,
                        "train_label_cutoff": cutoff,
                        "core_months": int(len(core_months)),
                        "validation_months": int(len(validation_months)),
                        "test_months": int(len(test_months)),
                        "n_parameters": fitted.n_parameters,
                        "ridge_alpha": np.nan,
                        "best_epoch_mean": float(np.mean(fitted.best_epochs)),
                        "training_loss": float(np.mean(fitted.training_losses)),
                        "validation_loss": float(np.mean(fitted.validation_losses)),
                        "fit_seconds": fitted.fit_seconds,
                    }
                )
            else:
                raise ValueError(f"unknown model {model!r}")
            fitted_models[model] = fitted
        for test_month in test_months:
            for model, fitted in fitted_models.items():
                if isinstance(fitted, ClosedFormModel):
                    raw_scores = _closed_form_raw_scores(
                        test_month,
                        fitted,
                        config,
                        random_feature_spec=random_spec
                        if model == "dkkm_random_features"
                        else None,
                    )
                    metadata = {
                        "ridge_alpha": fitted.ridge_alpha,
                        "validation_loss": fitted.validation_loss,
                        "best_epoch_mean": np.nan,
                    }
                else:
                    raw_scores = _neural_raw_scores(test_month, fitted)
                    metadata = {
                        "ridge_alpha": np.nan,
                        "validation_loss": float(np.mean(fitted.validation_losses)),
                        "best_epoch_mean": float(np.mean(fitted.best_epochs)),
                    }
                    if model == "nonlinear_transformer":
                        attention = extract_attention_summary(test_month, fitted)
                        if not attention.empty:
                            attention_frames.append(attention)
                record, weights = evaluate_model_month(
                    test_month,
                    model,
                    raw_scores,
                    config,
                    metadata,
                )
                record["refit_id"] = job_id
                monthly_records.append(record)
                weight_frames.append(weights)
    monthly = pd.DataFrame.from_records(monthly_records)
    fit_log = pd.DataFrame.from_records(fit_records)
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else pd.DataFrame()
    )
    attention = (
        pd.concat(attention_frames, ignore_index=True)
        if attention_frames
        else pd.DataFrame()
    )
    test_assets = build_test_asset_returns(months, config)
    return monthly, fit_log, weights, attention, test_assets


def build_test_asset_returns(
    months: list[AIPMFullMonth],
    config: AIPMFullTransformerConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for month in months:
        if not (
            config.first_test_year
            <= month.signal_date.year
            <= config.last_test_year
        ):
            continue
        factor_returns = _month_factor_returns(month, month.evaluation_returns)
        record: dict[str, Any] = {
            "signal_date": month.signal_date,
            "target_date": month.target_date,
        }
        record.update(
            {
                f"test_asset_{index}": float(value)
                for index, value in enumerate(factor_returns)
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def pricing_error_summary(
    monthly: pd.DataFrame,
    test_assets: pd.DataFrame,
    ridge: float,
) -> pd.DataFrame:
    if monthly.empty or test_assets.empty:
        return pd.DataFrame()
    asset_columns = [column for column in test_assets if column.startswith("test_asset_")]
    test = test_assets.copy()
    test["signal_date"] = pd.to_datetime(test["signal_date"])
    records: list[dict[str, Any]] = []
    for model, group in monthly.groupby("model", sort=True):
        frame = group[["signal_date", "sdf_return", "raw_sdf_return"]].copy()
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        merged = frame.merge(test, on="signal_date", how="inner")
        if len(merged) < 3:
            continue
        factors = merged[asset_columns].to_numpy(dtype=float)
        cov = np.cov(factors, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov = cov + ridge * np.eye(cov.shape[0])
        inv_cov = np.linalg.pinv(cov)
        for return_column, prefix in [
            ("sdf_return", "normalized"),
            ("raw_sdf_return", "raw"),
        ]:
            sdf = 1.0 - merged[return_column].to_numpy(dtype=float)
            moments = factors * sdf[:, None]
            mean_moments = np.nanmean(moments, axis=0)
            records.append(
                {
                    "model": model,
                    "return_scale": prefix,
                    "months": int(len(merged)),
                    "pricing_moment_l2": float(
                        np.sqrt(np.nanmean(np.square(mean_moments)))
                    ),
                    "max_abs_pricing_moment": float(np.nanmax(np.abs(mean_moments))),
                    "hjd_pricing_error": float(
                        np.sqrt(max(mean_moments @ inv_cov @ mean_moments, 0.0))
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_full_aipm(
    monthly: pd.DataFrame,
    pricing_errors: pd.DataFrame,
) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    pricing_lookup = {}
    if not pricing_errors.empty:
        pricing_lookup = {
            (row["model"], row["return_scale"]): row
            for _, row in pricing_errors.iterrows()
        }
    for model, group in monthly.groupby("model", sort=True):
        returns = group["sdf_return"].astype(float)
        raw_returns = group["raw_sdf_return"].astype(float)
        annualized_return = float(returns.mean() * 12.0)
        annualized_volatility = float(returns.std(ddof=1) * math.sqrt(12.0))
        raw_annualized_return = float(raw_returns.mean() * 12.0)
        raw_annualized_volatility = float(raw_returns.std(ddof=1) * math.sqrt(12.0))
        normalized_pricing = pricing_lookup.get((model, "normalized"))
        raw_pricing = pricing_lookup.get((model, "raw"))
        records.append(
            {
                "model": model,
                "months": int(len(group)),
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe": annualized_return / annualized_volatility
                if annualized_volatility > 0
                else np.nan,
                "raw_annualized_return": raw_annualized_return,
                "raw_annualized_volatility": raw_annualized_volatility,
                "raw_sharpe": raw_annualized_return / raw_annualized_volatility
                if raw_annualized_volatility > 0
                else np.nan,
                "monthly_min": float(returns.min()),
                "monthly_max": float(returns.max()),
                "average_n_test_stocks": float(group["n_test_stocks"].mean()),
                "average_gross_weight": float(group["gross_weight"].mean()),
                "average_net_weight": float(group["net_weight"].mean()),
                "average_weight_hhi": float(group["weight_hhi"].mean()),
                "average_monthly_turnover": float(group["weight_turnover"].mean()),
                "median_validation_loss": float(group["validation_loss"].median()),
                "normalized_hjd_pricing_error": float(
                    normalized_pricing["hjd_pricing_error"]
                )
                if normalized_pricing is not None
                else np.nan,
                "normalized_pricing_moment_l2": float(
                    normalized_pricing["pricing_moment_l2"]
                )
                if normalized_pricing is not None
                else np.nan,
                "raw_hjd_pricing_error": float(raw_pricing["hjd_pricing_error"])
                if raw_pricing is not None
                else np.nan,
                "raw_pricing_moment_l2": float(raw_pricing["pricing_moment_l2"])
                if raw_pricing is not None
                else np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def _model_comparisons(monthly: pd.DataFrame, hac_lags: int) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    wide = monthly.pivot(index="signal_date", columns="model", values="sdf_return")
    baselines = [model for model in ["bsv", "own_asset_mlp"] if model in wide]
    records: list[dict[str, Any]] = []
    for model in wide.columns:
        for baseline in baselines:
            if model == baseline:
                continue
            common = wide[[baseline, model]].dropna()
            if len(common) < 12:
                continue
            difference = common[model] - common[baseline]
            annualized_difference = float(difference.mean() * 12.0)
            annualized_volatility = float(difference.std(ddof=1) * math.sqrt(12.0))
            lhs = _rescale_to_annual_vol(common[model].to_numpy(dtype=float), 0.15)
            rhs = _rescale_to_annual_vol(common[baseline].to_numpy(dtype=float), 0.15)
            regression = sm.OLS(lhs, sm.add_constant(rhs)).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": hac_lags},
            )
            records.append(
                {
                    "model": model,
                    "baseline": baseline,
                    "months": int(len(common)),
                    "annualized_mean_difference": annualized_difference,
                    "annualized_difference_volatility": annualized_volatility,
                    "difference_sharpe": annualized_difference / annualized_volatility
                    if annualized_volatility > 0
                    else np.nan,
                    "correlation": float(common[model].corr(common[baseline])),
                    "alpha_annualized": float(regression.params[0] * 12.0),
                    "alpha_hac_t": float(regression.tvalues[0]),
                    "alpha_hac_p": float(regression.pvalues[0]),
                    "beta": float(regression.params[1]),
                }
            )
    return pd.DataFrame.from_records(records)


def _rescale_to_annual_vol(values: np.ndarray, target_annual_vol: float) -> np.ndarray:
    volatility = float(np.nanstd(values, ddof=1) * math.sqrt(12.0))
    if volatility <= 1e-12 or not np.isfinite(volatility):
        return values.astype(float, copy=True)
    return values.astype(float, copy=True) * target_annual_vol / volatility


def build_aipm_full_transformer_outputs(
    panel_path: Path,
    output_dir: Path,
    config: AIPMFullTransformerConfig,
    risk_free: pd.Series | None = None,
    feature_set: str = "compustat_enriched",
    models: tuple[str, ...] = (
        "bsv",
        "linear_attention",
        "dkkm_random_features",
        "own_asset_mlp",
        "nonlinear_transformer",
    ),
) -> dict[str, object]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    feature_columns = list(FEATURE_SETS[feature_set])
    panel = load_aipm_full_panel(panel_path, risk_free, feature_columns)
    months = build_months(panel, feature_columns, config)
    monthly, fit_log, weights, attention, test_assets = run_walk_forward_full_aipm(
        months,
        config,
        models=models,
    )
    if monthly.empty:
        raise RuntimeError("AIPM full transformer walk-forward produced no returns")
    monthly = add_weight_turnover(monthly, weights)
    pricing_errors = pricing_error_summary(
        monthly,
        test_assets,
        config.pricing_error_ridge,
    )
    summary = summarize_full_aipm(monthly, pricing_errors)
    comparisons = _model_comparisons(monthly, config.hac_lags)

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "aipm_full_monthly.csv", index=False)
    fit_log.to_csv(output_dir / "aipm_full_fit_log.csv", index=False)
    summary.to_csv(output_dir / "aipm_full_summary.csv", index=False)
    pricing_errors.to_csv(output_dir / "aipm_full_pricing_errors.csv", index=False)
    comparisons.to_csv(output_dir / "aipm_full_comparisons.csv", index=False)
    test_assets.to_csv(output_dir / "aipm_full_test_assets.csv", index=False)
    weights.to_parquet(
        output_dir / "aipm_full_weights.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    if not attention.empty:
        attention.to_csv(output_dir / "aipm_full_attention_examples.csv", index=False)
    causality = {
        "train_target_after_cutoff": int(
            (
                pd.to_datetime(fit_log["train_target_end"])
                > pd.to_datetime(fit_log["train_label_cutoff"])
            ).sum()
        )
        if not fit_log.empty
        else 0,
        "validation_target_after_cutoff": int(
            (
                pd.to_datetime(fit_log["validation_target_end"])
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
    }
    manifest: dict[str, object] = {
        "config": asdict(config),
        "panel_path": str(panel_path),
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "models": list(models),
        "rows": {
            "input_rows": int(len(panel)),
            "months": int(len(months)),
            "monthly": int(len(monthly)),
            "fit_log": int(len(fit_log)),
            "weights": int(len(weights)),
            "attention_examples": int(len(attention)),
            "test_assets": int(len(test_assets)),
            "pricing_errors": int(len(pricing_errors)),
            "summary": int(len(summary)),
            "comparisons": int(len(comparisons)),
        },
        "return_definition": "excess_return" if risk_free is not None else "raw_return",
        "source_paper": (
            "Kelly, Kuznetsov, Malamud and Xu (2026), "
            "Artificial Intelligence Asset Pricing Models"
        ),
        "objective": (
            "Full AIPM adaptation with BSV, saturated linear attention, DKKM "
            "random features, own-asset residual MLP and nonlinear portfolio "
            "transformer trained by MSRR"
        ),
        "causality_check": causality,
    }
    (output_dir / "aipm_full_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
