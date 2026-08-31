"""Neural SDF adaptation for the European equity panel.

The implementation is intentionally self-contained.  It uses a small numpy MLP
to learn cross-sectional SDF portfolio weights from characteristics and market
state variables, then evaluates the learned weights in annual walk-forward
tests.  The objective is CPZ-inspired: the network maximizes an out-of-sample
SDF portfolio utility while penalizing Euler-equation errors on
characteristic-managed test assets.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from asset_pricing_ml import COMPUSTAT_FEATURE_COLUMNS, FEATURE_SETS


DEFAULT_STATE_COLUMNS = [
    "state_market_return_eur",
    "state_market_trend_12m",
    "state_market_volatility_12m",
]


@dataclass(frozen=True)
class NeuralSDFConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_monthly_stocks: int = 100
    min_training_months: int = 72
    validation_months: int = 24
    training_window_months: int | None = None
    hidden_sizes: tuple[int, ...] = (64, 32)
    activation: str = "tanh"
    epochs: int = 80
    patience: int = 10
    learning_rate: float = 0.001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0001
    gradient_clip_norm: float = 5.0
    risk_aversion: float = 3.0
    utility_weight: float = 1.0
    moment_penalty: float = 100.0
    gross_leverage: float = 2.0
    minimum_size_percentile: float = 0.05
    training_return_clip: float = 1.0
    cost_grid_bps: tuple[int, ...] = (0, 10, 25, 50)
    random_state: int = 42
    hac_lags: int = 6


@dataclass
class MonthBatch:
    signal_date: pd.Timestamp
    target_date: pd.Timestamp
    rics: np.ndarray
    features: np.ndarray
    moment_features: np.ndarray
    training_returns: np.ndarray
    evaluation_returns: np.ndarray
    market_caps: np.ndarray
    market_cap_percentiles: np.ndarray

    @property
    def n_stocks(self) -> int:
        return int(len(self.rics))


def load_state_features(
    market_state_path: Path | None = None,
    additional_state_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load signal-date state variables without using future returns.

    All state variables are keyed by the panel signal date.  Missing early
    observations are filled later after standardization, not forward-filled from
    future states.
    """
    frames: list[pd.DataFrame] = []
    if market_state_path is not None and market_state_path.exists():
        market = pd.read_csv(market_state_path)
        market["date"] = pd.to_datetime(market["signal_date"])
        numeric = market[
            [
                "date",
                "market_return_eur",
                "market_trend_12m",
                "market_volatility_12m",
            ]
        ].rename(
            columns={
                "market_return_eur": "state_market_return_eur",
                "market_trend_12m": "state_market_trend_12m",
                "market_volatility_12m": "state_market_volatility_12m",
            }
        )
        categoricals = []
        for column in ["volatility_state", "trend_state"]:
            if column in market:
                dummies = pd.get_dummies(
                    market[column].astype("string").fillna("missing"),
                    prefix=f"state_{column}",
                    dtype=float,
                )
                categoricals.append(dummies)
        frames.append(pd.concat([numeric, *categoricals], axis=1))

    if additional_state_path is not None and additional_state_path.exists():
        extra = pd.read_csv(additional_state_path)
        if "date" not in extra and "signal_date" in extra:
            extra = extra.rename(columns={"signal_date": "date"})
        if "date" not in extra:
            raise ValueError("Additional state feature file must contain date or signal_date")
        extra["date"] = pd.to_datetime(extra["date"])
        state_columns = [column for column in extra.columns if column != "date"]
        numeric = extra[["date", *state_columns]].copy()
        for column in state_columns:
            output_column = column if column.startswith("state_") else f"state_{column}"
            numeric[output_column] = pd.to_numeric(numeric.pop(column), errors="coerce")
        frames.append(numeric)

    if not frames:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")}), []

    state = frames[0]
    for frame in frames[1:]:
        state = state.merge(frame, on="date", how="outer", validate="one_to_one")
    state = state.sort_values("date").reset_index(drop=True)
    state_columns = [column for column in state.columns if column != "date"]
    state[state_columns] = state[state_columns].apply(pd.to_numeric, errors="coerce")
    return state, state_columns


def load_neural_sdf_panel(
    panel_path: Path,
    risk_free: pd.Series | None = None,
    feature_columns: list[str] | None = None,
    state_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load stock-month data used by the neural SDF."""
    features = feature_columns or COMPUSTAT_FEATURE_COLUMNS
    base_columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "model_eligible",
        "company_market_cap",
        "market_cap_percentile",
        *features,
    ]
    panel = pd.read_parquet(panel_path, columns=base_columns)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["target_date"] = pd.to_datetime(panel["target_date"])
    panel = panel[
        panel["model_eligible"].fillna(False)
        & panel["target_return_1m"].notna()
        & panel["target_date"].notna()
    ].copy()
    panel["sdf_target_return"] = panel["target_return_1m"].astype(float)
    if risk_free is not None:
        rf = risk_free.rename("RF_EUR").rename_axis("target_date").reset_index()
        rf["target_date"] = pd.to_datetime(rf["target_date"])
        panel = panel.merge(rf, on="target_date", how="left", validate="many_to_one")
        panel["sdf_target_return"] = panel["sdf_target_return"] - panel[
            "RF_EUR"
        ].fillna(0.0)

    state_columns: list[str] = []
    if state_features is not None and not state_features.empty:
        state = state_features.copy()
        state["date"] = pd.to_datetime(state["date"])
        state_columns = [column for column in state.columns if column != "date"]
        panel = panel.merge(state, on="date", how="left", validate="many_to_one")

    for column in [*features, *state_columns]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce").fillna(0.0)
    panel = panel.dropna(
        subset=[
            "date",
            "target_date",
            "ric",
            "sdf_target_return",
            "company_market_cap",
            "market_cap_percentile",
        ]
    )
    return panel.sort_values(["date", "ric"]).reset_index(drop=True), state_columns


def self_financing_weights(
    scores: np.ndarray,
    gross_leverage: float = 2.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert arbitrary scores to centered gross-normalized long-short weights."""
    centered = scores.astype(float, copy=False) - float(np.mean(scores))
    gross = float(np.abs(centered).sum())
    if gross <= eps:
        return np.zeros_like(centered, dtype=float)
    return centered / gross * gross_leverage


def _portfolio_return_and_gradient(
    scores: np.ndarray,
    returns: np.ndarray,
    gross_leverage: float,
    eps: float = 1e-8,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return portfolio return, weights, and d(return)/d(scores)."""
    scores = scores.astype(float, copy=False)
    returns = returns.astype(float, copy=False)
    centered = scores - float(scores.mean())
    gross = float(np.abs(centered).sum())
    if gross <= eps:
        weights = np.zeros_like(centered, dtype=float)
        gradient = np.zeros_like(centered, dtype=float)
        return 0.0, weights, gradient

    scale = gross_leverage / gross
    weights = centered * scale
    portfolio_return = float(weights @ returns)
    grad_centered = scale * returns - portfolio_return / gross * np.sign(centered)
    gradient = grad_centered - float(grad_centered.mean())
    return portfolio_return, weights, gradient


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_maximum = wealth.cummax()
    return float(wealth.div(running_maximum).sub(1.0).min())


class NumpyMLP:
    """Small dense MLP with Adam updates."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        activation: str,
        random_state: int,
    ):
        if activation not in {"tanh", "relu"}:
            raise ValueError("activation must be 'tanh' or 'relu'")
        self.activation = activation
        rng = np.random.default_rng(random_state)
        layer_sizes = [input_size, *hidden_sizes, 1]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
            scale = math.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out, dtype=float))

    def state_dict(self) -> dict[str, list[np.ndarray]]:
        return {
            "weights": [weight.copy() for weight in self.weights],
            "biases": [bias.copy() for bias in self.biases],
        }

    def load_state_dict(self, state: dict[str, list[np.ndarray]]) -> None:
        self.weights = [weight.copy() for weight in state["weights"]]
        self.biases = [bias.copy() for bias in state["biases"]]

    def _activate(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(values)
        return np.maximum(values, 0.0)

    def _activation_grad(self, preactivation: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            activated = np.tanh(preactivation)
            return 1.0 - activated * activated
        return (preactivation > 0.0).astype(float)

    def forward(
        self,
        x: np.ndarray,
        keep_cache: bool = False,
    ) -> tuple[np.ndarray, dict[str, list[np.ndarray]] | None]:
        activations = [x]
        preactivations: list[np.ndarray] = []
        current = x
        for layer_index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            linear = current @ weight + bias
            preactivations.append(linear)
            if layer_index == len(self.weights) - 1:
                current = linear
            else:
                current = self._activate(linear)
            activations.append(current)
        cache = (
            {"activations": activations, "preactivations": preactivations}
            if keep_cache
            else None
        )
        return current.reshape(-1), cache

    def empty_grads(self) -> dict[str, list[np.ndarray]]:
        return {
            "weights": [np.zeros_like(weight) for weight in self.weights],
            "biases": [np.zeros_like(bias) for bias in self.biases],
        }

    def backward(
        self,
        cache: dict[str, list[np.ndarray]],
        grad_scores: np.ndarray,
        grads: dict[str, list[np.ndarray]],
    ) -> None:
        delta = grad_scores.reshape(-1, 1)
        activations = cache["activations"]
        preactivations = cache["preactivations"]
        for layer_index in range(len(self.weights) - 1, -1, -1):
            grads["weights"][layer_index] += activations[layer_index].T @ delta
            grads["biases"][layer_index] += delta.sum(axis=0)
            if layer_index > 0:
                delta = delta @ self.weights[layer_index].T
                delta *= self._activation_grad(preactivations[layer_index - 1])


class AdamOptimizer:
    def __init__(self, network: NumpyMLP, config: NeuralSDFConfig):
        self.network = network
        self.config = config
        self.t = 0
        self.m = network.empty_grads()
        self.v = network.empty_grads()

    @staticmethod
    def _iter_arrays(grads: dict[str, list[np.ndarray]]):
        for group in ["weights", "biases"]:
            for index, value in enumerate(grads[group]):
                yield group, index, value

    def step(self, grads: dict[str, list[np.ndarray]]) -> None:
        self.t += 1
        beta1 = self.config.adam_beta1
        beta2 = self.config.adam_beta2
        arrays = [value for _, _, value in self._iter_arrays(grads)]
        norm = math.sqrt(float(sum(np.square(value).sum() for value in arrays)))
        if norm > self.config.gradient_clip_norm > 0:
            scale = self.config.gradient_clip_norm / (norm + 1e-12)
            for _, _, value in self._iter_arrays(grads):
                value *= scale

        for group, index, grad in self._iter_arrays(grads):
            self.m[group][index] = beta1 * self.m[group][index] + (1.0 - beta1) * grad
            self.v[group][index] = beta2 * self.v[group][index] + (1.0 - beta2) * (
                grad * grad
            )
            m_hat = self.m[group][index] / (1.0 - beta1**self.t)
            v_hat = self.v[group][index] / (1.0 - beta2**self.t)
            update = self.config.learning_rate * m_hat / (
                np.sqrt(v_hat) + self.config.adam_epsilon
            )
            if group == "weights":
                self.network.weights[index] -= update
            else:
                self.network.biases[index] -= update


class NeuralSDFModel:
    def __init__(self, input_size: int, config: NeuralSDFConfig):
        self.config = config
        self.network = NumpyMLP(
            input_size=input_size,
            hidden_sizes=config.hidden_sizes,
            activation=config.activation,
            random_state=config.random_state,
        )
        self.best_epoch: int | None = None
        self.best_validation_loss: float | None = None
        self.training_loss: float | None = None
        self.training_diagnostics: dict[str, float] = {}

    def _objective(
        self,
        batches: list[MonthBatch],
        require_grad: bool,
    ) -> tuple[float, dict[str, float], dict[str, list[np.ndarray]] | None]:
        if not batches:
            raise ValueError("At least one monthly batch is required")
        portfolio_returns = []
        moment_returns = []
        for batch in batches:
            scores, _ = self.network.forward(batch.features, keep_cache=False)
            portfolio_return, _, _ = _portfolio_return_and_gradient(
                scores,
                batch.training_returns,
                self.config.gross_leverage,
            )
            portfolio_returns.append(portfolio_return)
            moment_returns.append(
                batch.moment_features.T @ batch.training_returns / batch.n_stocks
            )

        p = np.asarray(portfolio_returns, dtype=float)
        g = np.vstack(moment_returns) if moment_returns else np.empty((len(p), 0))
        mean_return = float(p.mean())
        variance = float(np.mean(np.square(p - mean_return)))
        loss = (
            -self.config.utility_weight * mean_return
            + self.config.risk_aversion * variance
        )
        if g.shape[1] > 0 and self.config.moment_penalty > 0:
            moments = ((1.0 - p[:, None]) * g).mean(axis=0)
            moment_loss = float(np.mean(np.square(moments)))
            loss += self.config.moment_penalty * moment_loss
        else:
            moments = np.empty(0, dtype=float)
            moment_loss = 0.0

        weight_l2 = float(sum(np.square(weight).sum() for weight in self.network.weights))
        loss += 0.5 * self.config.weight_decay * weight_l2
        diagnostics = {
            "loss": float(loss),
            "mean_monthly_return": mean_return,
            "monthly_variance": variance,
            "moment_loss": moment_loss,
            "sdf_sharpe": (
                mean_return / float(p.std(ddof=1)) * math.sqrt(12.0)
                if len(p) > 1 and p.std(ddof=1) > 0
                else np.nan
            ),
        }
        if not require_grad:
            return float(loss), diagnostics, None

        t_count = float(len(p))
        d_loss_dp = (
            -self.config.utility_weight / t_count
            + self.config.risk_aversion * 2.0 * (p - mean_return) / t_count
        )
        if g.shape[1] > 0 and self.config.moment_penalty > 0:
            k_count = float(g.shape[1])
            d_loss_dp += (
                self.config.moment_penalty
                * 2.0
                / k_count
                * (-(g @ moments) / t_count)
            )

        grads = self.network.empty_grads()
        for batch, d_portfolio in zip(batches, d_loss_dp, strict=True):
            scores, cache = self.network.forward(batch.features, keep_cache=True)
            _, _, portfolio_gradient = _portfolio_return_and_gradient(
                scores,
                batch.training_returns,
                self.config.gross_leverage,
            )
            self.network.backward(cache, d_portfolio * portfolio_gradient, grads)
        for index, weight in enumerate(self.network.weights):
            grads["weights"][index] += self.config.weight_decay * weight
        return float(loss), diagnostics, grads

    def fit(
        self,
        training_batches: list[MonthBatch],
        validation_batches: list[MonthBatch],
    ) -> "NeuralSDFModel":
        optimizer = AdamOptimizer(self.network, self.config)
        best_state = copy.deepcopy(self.network.state_dict())
        best_loss = math.inf
        stale_epochs = 0
        for epoch in range(1, self.config.epochs + 1):
            loss, diagnostics, grads = self._objective(
                training_batches,
                require_grad=True,
            )
            if grads is None:
                raise RuntimeError("Training objective did not return gradients")
            optimizer.step(grads)
            validation_loss, _, _ = self._objective(
                validation_batches or training_batches,
                require_grad=False,
            )
            if validation_loss < best_loss - 1e-9:
                best_loss = validation_loss
                best_state = copy.deepcopy(self.network.state_dict())
                self.best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.patience:
                    break
            self.training_loss = loss
            self.training_diagnostics = diagnostics
        self.network.load_state_dict(best_state)
        self.best_validation_loss = best_loss
        return self

    def score_month(self, batch: MonthBatch) -> tuple[np.ndarray, np.ndarray, float]:
        scores, _ = self.network.forward(batch.features, keep_cache=False)
        weights = self_financing_weights(scores, self.config.gross_leverage)
        sdf_return = float(weights @ batch.evaluation_returns)
        return scores, weights, sdf_return


def _standardizer(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame[columns].to_numpy(dtype=float, copy=False)
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
    return mean, scale


def _build_month_batches(
    frame: pd.DataFrame,
    input_columns: list[str],
    moment_columns: list[str],
    mean: np.ndarray,
    scale: np.ndarray,
    config: NeuralSDFConfig,
) -> list[MonthBatch]:
    eligible = frame[
        frame["market_cap_percentile"].ge(config.minimum_size_percentile)
    ].copy()
    batches: list[MonthBatch] = []
    for signal_date, month in eligible.groupby("date", sort=True):
        if len(month) < config.min_monthly_stocks:
            continue
        x = month[input_columns].to_numpy(dtype=float, copy=False)
        x = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
        moment_x = month[moment_columns].to_numpy(dtype=float, copy=False)
        moment_x = np.nan_to_num(moment_x, nan=0.0, posinf=0.0, neginf=0.0)
        evaluation_returns = month["sdf_target_return"].to_numpy(dtype=float)
        training_returns = evaluation_returns.copy()
        if config.training_return_clip > 0:
            training_returns = np.clip(
                training_returns,
                -config.training_return_clip,
                config.training_return_clip,
            )
        batches.append(
            MonthBatch(
                signal_date=pd.Timestamp(signal_date),
                target_date=pd.Timestamp(month["target_date"].max()),
                rics=month["ric"].astype(str).to_numpy(),
                features=x,
                moment_features=moment_x,
                training_returns=training_returns,
                evaluation_returns=evaluation_returns,
                market_caps=month["company_market_cap"].to_numpy(dtype=float),
                market_cap_percentiles=month["market_cap_percentile"].to_numpy(
                    dtype=float
                ),
            )
        )
    return batches


def _split_training_months(
    train: pd.DataFrame,
    config: NeuralSDFConfig,
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


def _weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    positive = weights[weights > 0]
    negative = weights[weights < 0]
    return {
        "gross_weight": float(np.abs(weights).sum()),
        "net_weight": float(weights.sum()),
        "long_weight": float(positive.sum()),
        "short_weight": float(negative.sum()),
        "weight_hhi": float(np.square(weights).sum()),
        "active_weight_fraction": float((np.abs(weights) > 1e-8).mean()),
    }


def run_walk_forward_neural_sdf(
    panel: pd.DataFrame,
    config: NeuralSDFConfig,
    feature_columns: list[str] | None = None,
    state_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = feature_columns or COMPUSTAT_FEATURE_COLUMNS
    state_columns = state_columns or []
    input_columns = [*feature_columns, *state_columns]
    if not input_columns:
        raise ValueError("At least one input column is required")

    monthly_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    previous_weights: dict[str, float] = {}
    panel = panel.sort_values(["date", "ric"]).reset_index(drop=True)

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
        mean, scale = _standardizer(train, input_columns)
        core_batches = _build_month_batches(
            core,
            input_columns,
            feature_columns,
            mean,
            scale,
            config,
        )
        validation_batches = _build_month_batches(
            validation,
            input_columns,
            feature_columns,
            mean,
            scale,
            config,
        )
        if len(core_batches) < config.min_training_months or not validation_batches:
            continue

        model = NeuralSDFModel(len(input_columns), config).fit(
            core_batches,
            validation_batches,
        )
        test_batches = _build_month_batches(
            test,
            input_columns,
            feature_columns,
            mean,
            scale,
            config,
        )
        year_weight_frames = []
        for batch in test_batches:
            scores, weights, sdf_return = model.score_month(batch)
            weight_map = dict(zip(batch.rics, weights, strict=True))
            traded_names = set(previous_weights) | set(weight_map)
            turnover = 0.5 * sum(
                abs(weight_map.get(name, 0.0) - previous_weights.get(name, 0.0))
                for name in traded_names
            )
            previous_weights = weight_map
            monthly_records.append(
                {
                    "signal_date": batch.signal_date,
                    "target_date": batch.target_date,
                    "model": "neural_sdf",
                    "sdf_return": sdf_return,
                    "long_short_turnover": float(turnover),
                    "n_test_stocks": batch.n_stocks,
                    "score_mean": float(scores.mean()),
                    "score_std": float(scores.std(ddof=1))
                    if batch.n_stocks > 1
                    else 0.0,
                    **_weight_diagnostics(weights),
                }
            )
            year_weight_frames.append(
                pd.DataFrame(
                    {
                        "signal_date": batch.signal_date,
                        "target_date": batch.target_date,
                        "ric": batch.rics,
                        "model": "neural_sdf",
                        "score": scores,
                        "weight": weights,
                        "target_return": batch.evaluation_returns,
                        "market_cap": batch.market_caps,
                        "market_cap_percentile": batch.market_cap_percentiles,
                    }
                )
            )

        fit_records.append(
            {
                "model": "neural_sdf",
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
                "best_epoch": model.best_epoch,
                "best_validation_loss": model.best_validation_loss,
                "training_loss": model.training_loss,
                **{
                    f"training_{key}": value
                    for key, value in model.training_diagnostics.items()
                },
            }
        )
        weight_frames.extend(year_weight_frames)

    monthly = pd.DataFrame.from_records(monthly_records)
    fit_log = pd.DataFrame.from_records(fit_records)
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else pd.DataFrame()
    )
    return monthly, fit_log, weights


def summarize_neural_sdf_returns(
    monthly: pd.DataFrame,
    cost_grid_bps: tuple[int, ...] = (0, 10, 25, 50),
) -> pd.DataFrame:
    records = []
    for model, group in monthly.groupby("model", sort=True):
        gross_returns = group["sdf_return"].astype(float)
        turnover = group.get(
            "long_short_turnover",
            pd.Series(0.0, index=group.index),
        ).astype(float)
        gross_annualized_return = float(gross_returns.mean() * 12.0)
        for cost_bps in cost_grid_bps:
            returns = gross_returns - turnover * cost_bps / 10_000.0
            annualized_return = float(returns.mean() * 12.0)
            annualized_volatility = float(returns.std(ddof=1) * math.sqrt(12.0))
            records.append(
                {
                    "model": model,
                    "cost_bps": int(cost_bps),
                    "months": int(len(group)),
                    "annualized_return": annualized_return,
                    "gross_annualized_return": gross_annualized_return,
                    "annualized_volatility": annualized_volatility,
                    "sharpe": annualized_return / annualized_volatility
                    if annualized_volatility > 0
                    else np.nan,
                    "monthly_min": float(returns.min()),
                    "monthly_max": float(returns.max()),
                    "max_drawdown": _max_drawdown(returns),
                    "average_monthly_turnover": float(turnover.mean()),
                    "average_n_test_stocks": float(group["n_test_stocks"].mean()),
                    "average_gross_weight": float(group["gross_weight"].mean()),
                    "average_long_weight": float(group["long_weight"].mean()),
                    "average_short_weight": float(group["short_weight"].mean()),
                    "average_weight_hhi": float(group["weight_hhi"].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def compare_sdf_returns(
    monthly: pd.DataFrame,
    baseline: pd.DataFrame,
    baseline_name: str,
    hac_lags: int,
) -> pd.DataFrame:
    if monthly.empty or baseline.empty:
        return pd.DataFrame()
    left = monthly[["signal_date", "model", "sdf_return"]].copy()
    right = baseline[["signal_date", "model", "sdf_return"]].copy()
    left["signal_date"] = pd.to_datetime(left["signal_date"])
    right["signal_date"] = pd.to_datetime(right["signal_date"])
    records = []
    for model_name, model_group in left.groupby("model", sort=True):
        for baseline_model, baseline_group in right.groupby("model", sort=True):
            common = model_group.merge(
                baseline_group,
                on="signal_date",
                how="inner",
                suffixes=("_model", "_baseline"),
            )
            if len(common) < max(12, hac_lags + 2):
                continue
            difference = common["sdf_return_model"] - common["sdf_return_baseline"]
            fit = sm.OLS(difference.to_numpy(), np.ones((len(difference), 1))).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": hac_lags},
            )
            annualized_mean = float(difference.mean() * 12.0)
            annualized_vol = float(difference.std(ddof=1) * math.sqrt(12.0))
            records.append(
                {
                    "model": model_name,
                    "baseline_source": baseline_name,
                    "baseline_model": baseline_model,
                    "months": int(len(common)),
                    "annualized_mean_difference": annualized_mean,
                    "annualized_difference_volatility": annualized_vol,
                    "difference_sharpe": annualized_mean / annualized_vol
                    if annualized_vol > 0
                    else np.nan,
                    "hac_t": float(fit.tvalues[0]),
                    "hac_p": float(fit.pvalues[0]),
                    "correlation": float(
                        common["sdf_return_model"].corr(common["sdf_return_baseline"])
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def compare_neural_to_ml_portfolios(
    neural_monthly: pd.DataFrame,
    ml_monthly: pd.DataFrame,
    cost_bps: int = 25,
    blocks: tuple[int, ...] = (3, 6, 12),
    n_boot: int = 5_000,
    seed: int = 42,
    weighting: str = "value",
    universe_variant: str = "standard_ex_bottom_5pct",
) -> pd.DataFrame:
    """Paired stationary-bootstrap Sharpe comparisons against ML portfolios."""
    import stats as project_stats

    if neural_monthly.empty or ml_monthly.empty:
        return pd.DataFrame()
    required = {
        "model",
        "weighting",
        "universe_variant",
        "return_date",
        "gross_long_short_return",
        "long_short_turnover",
    }
    if not required.issubset(ml_monthly.columns):
        return pd.DataFrame()
    neural = neural_monthly.copy()
    neural["target_date"] = pd.to_datetime(neural["target_date"])
    ml = ml_monthly[
        ml_monthly["weighting"].eq(weighting)
        & ml_monthly["universe_variant"].eq(universe_variant)
    ].copy()
    ml["return_date"] = pd.to_datetime(ml["return_date"])
    records = []
    for baseline_model, baseline in ml.groupby("model", sort=True):
        common = neural.merge(
            baseline,
            left_on="target_date",
            right_on="return_date",
            suffixes=("_neural", "_baseline"),
        )
        if len(common) < 24:
            continue
        neural_net = common["sdf_return"] - common[
            "long_short_turnover_neural"
        ] * cost_bps / 10_000.0
        baseline_net = common["gross_long_short_return"] - common[
            "long_short_turnover_baseline"
        ] * cost_bps / 10_000.0
        rf = np.zeros(len(common), dtype=float)
        for block in blocks:
            result = project_stats.bootstrap_sharpe_diff(
                neural_net,
                baseline_net,
                rf,
                expected_block=block,
                n_boot=n_boot,
                seed=seed,
            )
            records.append(
                {
                    "model": "neural_sdf",
                    "baseline": baseline_model,
                    "weighting": weighting,
                    "universe_variant": universe_variant,
                    "cost_bps": int(cost_bps),
                    "expected_block": int(block),
                    "months": int(len(common)),
                    "neural_annualized_return": float(neural_net.mean() * 12.0),
                    "baseline_annualized_return": float(baseline_net.mean() * 12.0),
                    "neural_sharpe": float(
                        neural_net.mean() / neural_net.std(ddof=1) * math.sqrt(12.0)
                    ),
                    "baseline_sharpe": float(
                        baseline_net.mean()
                        / baseline_net.std(ddof=1)
                        * math.sqrt(12.0)
                    ),
                    "return_correlation": float(neural_net.corr(baseline_net)),
                    **result,
                }
            )
    output = pd.DataFrame.from_records(records)
    if not output.empty:
        output["p_two_sided_holm"] = output.groupby(
            ["cost_bps", "expected_block"], sort=False
        )["p_two_sided"].transform(lambda values: multipletests(values, method="holm")[1])
    return output


def build_neural_sdf_outputs(
    panel_path: Path,
    output_dir: Path,
    config: NeuralSDFConfig,
    risk_free: pd.Series | None = None,
    feature_set: str = "compustat_enriched",
    market_state_path: Path | None = None,
    additional_state_path: Path | None = None,
    baseline_monthly_paths: list[Path] | None = None,
    ml_portfolio_path: Path | None = None,
    significance_n_boot: int = 5_000,
    significance_blocks: tuple[int, ...] = (3, 6, 12),
) -> dict[str, object]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    feature_columns = FEATURE_SETS[feature_set]
    state_features, state_columns = load_state_features(
        market_state_path,
        additional_state_path,
    )
    panel, loaded_state_columns = load_neural_sdf_panel(
        panel_path,
        risk_free=risk_free,
        feature_columns=feature_columns,
        state_features=state_features,
    )
    state_columns = loaded_state_columns
    monthly, fit_log, weights = run_walk_forward_neural_sdf(
        panel,
        config,
        feature_columns=feature_columns,
        state_columns=state_columns,
    )
    if monthly.empty:
        raise RuntimeError("Neural SDF walk-forward produced no monthly returns")
    summary = summarize_neural_sdf_returns(monthly, config.cost_grid_bps)

    comparison_frames = []
    for path in baseline_monthly_paths or []:
        if not path.exists():
            continue
        baseline = pd.read_csv(path)
        if "return_date" in baseline and "signal_date" not in baseline:
            baseline = baseline.rename(columns={"return_date": "target_date"})
        if {"signal_date", "model", "sdf_return"}.issubset(baseline.columns):
            comparison_frames.append(
                compare_sdf_returns(monthly, baseline, path.parent.name, config.hac_lags)
            )
    comparison = (
        pd.concat(
            [frame for frame in comparison_frames if not frame.empty],
            ignore_index=True,
        )
        if comparison_frames
        else pd.DataFrame()
    )
    if ml_portfolio_path is not None and ml_portfolio_path.exists():
        ml_monthly = pd.read_csv(ml_portfolio_path)
        sharpe_significance = compare_neural_to_ml_portfolios(
            monthly,
            ml_monthly,
            cost_bps=25,
            blocks=significance_blocks,
            n_boot=significance_n_boot,
            seed=config.random_state,
        )
    else:
        sharpe_significance = pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "neural_sdf_monthly.csv", index=False)
    fit_log.to_csv(output_dir / "neural_sdf_fit_log.csv", index=False)
    summary.to_csv(output_dir / "neural_sdf_summary.csv", index=False)
    comparison.to_csv(output_dir / "neural_sdf_comparison.csv", index=False)
    sharpe_significance.to_csv(
        output_dir / "neural_sdf_sharpe_significance.csv",
        index=False,
    )
    weights.to_parquet(
        output_dir / "neural_sdf_weights.parquet",
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
            "comparison": int(len(comparison)),
            "sharpe_significance": int(len(sharpe_significance)),
        },
        "return_definition": "excess_return" if risk_free is not None else "raw_return",
        "objective": (
            "neural long-short SDF portfolio utility plus characteristic-managed "
            "Euler moment penalty"
        ),
        "source_paper": (
            "Chen, Pelger and Zhu (2024), Deep Learning in Asset Pricing"
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
            "duplicate_weight_security_months": int(
                weights.duplicated(["signal_date", "ric", "model"]).sum()
            )
            if not weights.empty
            else 0,
        },
    }
    (output_dir / "neural_sdf_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
