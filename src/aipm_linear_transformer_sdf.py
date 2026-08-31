"""Linear transformer SDF adaptation for European equities.

This module implements a tractable version of Kelly, Kuznetsov, Malamud and
Xu's Artificial Intelligence Pricing Model.  The paper's interpretable linear
portfolio transformer is

    w_t = N_t^{-1} (X_t W X_t') Z_t lambda,

which can be estimated as a ridge-penalized maximum-Sharpe/SDF regression.  We
use the same ranked firm characteristics for X and Z, compare the attention
portfolio to the no-attention Brandt-Santa-Clara-Valkanov (BSV) benchmark, and
evaluate both raw MSRR returns and gross-normalized OOS security weights.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from asset_pricing_ml import FEATURE_SETS


@dataclass(frozen=True)
class AIPMLinearTransformerConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_monthly_stocks: int = 100
    min_training_months: int = 72
    validation_months: int = 24
    training_window_months: int | None = None
    minimum_size_percentile: float = 0.05
    max_attention_features: int | None = 32
    gross_leverage: float = 1.0
    training_return_clip: float = 1.0
    hac_lags: int = 6
    ridge_grid: tuple[float, ...] = (
        1e-6,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
        100000.0,
        1000000.0,
    )


@dataclass
class AIPMMonth:
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


@dataclass
class FittedAIPMModel:
    model: str
    coefficients: np.ndarray
    ridge_alpha: float
    validation_loss: float
    training_loss: float
    fitted_training_return_mean: float
    fitted_validation_return_mean: float


def selected_feature_columns(
    feature_columns: list[str],
    max_attention_features: int | None,
) -> list[str]:
    if max_attention_features is None:
        return list(feature_columns)
    if max_attention_features <= 0:
        raise ValueError("max_attention_features must be positive or None")
    return list(feature_columns[:max_attention_features])


def load_aipm_panel(
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
    ].copy()
    panel["sdf_target_return"] = panel["target_return_1m"].astype(float)
    if risk_free is not None:
        rf = risk_free.rename("RF_EUR").rename_axis("target_date").reset_index()
        rf["target_date"] = pd.to_datetime(rf["target_date"])
        panel = panel.merge(rf, on="target_date", how="left", validate="many_to_one")
        panel["sdf_target_return"] = panel["sdf_target_return"] - panel[
            "RF_EUR"
        ].fillna(0.0)
    return panel


def build_months(
    panel: pd.DataFrame,
    feature_columns: list[str],
    config: AIPMLinearTransformerConfig,
) -> list[AIPMMonth]:
    eligible = panel[
        panel["market_cap_percentile"].ge(config.minimum_size_percentile)
    ].copy()
    months: list[AIPMMonth] = []
    for signal_date, month in eligible.groupby("date", sort=True):
        if len(month) < config.min_monthly_stocks:
            continue
        features = month[feature_columns].to_numpy(dtype=float, copy=False)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        evaluation_returns = month["sdf_target_return"].to_numpy(dtype=float)
        training_returns = evaluation_returns.copy()
        if config.training_return_clip > 0:
            training_returns = np.clip(
                training_returns,
                -config.training_return_clip,
                config.training_return_clip,
            )
        months.append(
            AIPMMonth(
                signal_date=pd.Timestamp(signal_date),
                target_date=pd.Timestamp(month["target_date"].max()),
                rics=month["ric"].astype(str).to_numpy(),
                features=features.astype("float64", copy=False),
                training_returns=training_returns.astype("float64", copy=False),
                evaluation_returns=evaluation_returns.astype("float64", copy=False),
                market_caps=month["company_market_cap"].to_numpy(dtype=float),
                market_cap_percentiles=month["market_cap_percentile"].to_numpy(
                    dtype=float
                ),
            )
        )
    return months


def _month_factor_returns(month: AIPMMonth, returns: np.ndarray) -> np.ndarray:
    return month.features.T @ returns / month.n_stocks


def _month_covariance(month: AIPMMonth) -> np.ndarray:
    return month.features.T @ month.features / month.n_stocks


def monthly_basis_vector(
    month: AIPMMonth,
    model: str,
    training_returns: bool,
) -> np.ndarray:
    returns = month.training_returns if training_returns else month.evaluation_returns
    factor_returns = _month_factor_returns(month, returns)
    if model == "bsv":
        return factor_returns
    if model == "linear_attention":
        covariance = _month_covariance(month)
        return np.kron(factor_returns, covariance.reshape(-1, order="C"))
    raise ValueError(f"Unknown AIPM model {model!r}")


def basis_matrix(
    months: list[AIPMMonth],
    model: str,
    training_returns: bool,
) -> np.ndarray:
    if not months:
        return np.empty((0, 0), dtype=float)
    rows = [monthly_basis_vector(month, model, training_returns) for month in months]
    return np.vstack(rows).astype("float64", copy=False)


def _ridge_fit_msrr(
    train_basis: np.ndarray,
    validation_basis: np.ndarray,
    ridge_grid: tuple[float, ...],
) -> FittedAIPMModel:
    if train_basis.ndim != 2 or validation_basis.ndim != 2:
        raise ValueError("basis inputs must be two-dimensional")
    if train_basis.shape[0] == 0 or train_basis.shape[1] == 0:
        raise ValueError("training basis must be non-empty")
    scale = train_basis.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    train = train_basis / scale
    validation = validation_basis / scale
    y = np.ones(train.shape[0], dtype=float)
    kernel = train @ train.T
    identity = np.eye(kernel.shape[0])

    best: dict[str, Any] | None = None
    for ridge_alpha in ridge_grid:
        system = kernel + float(ridge_alpha) * identity
        try:
            dual = np.linalg.solve(system, y)
        except np.linalg.LinAlgError:
            dual = np.linalg.lstsq(system, y, rcond=None)[0]
        scaled_coefficients = train.T @ dual
        train_fitted = train @ scaled_coefficients
        validation_fitted = validation @ scaled_coefficients
        validation_loss = float(np.mean(np.square(1.0 - validation_fitted)))
        record = {
            "ridge_alpha": float(ridge_alpha),
            "scaled_coefficients": scaled_coefficients,
            "validation_loss": validation_loss,
            "training_loss": float(np.mean(np.square(1.0 - train_fitted))),
            "fitted_training_return_mean": float(np.mean(train_fitted)),
            "fitted_validation_return_mean": float(np.mean(validation_fitted)),
        }
        if best is None or validation_loss < best["validation_loss"]:
            best = record
    if best is None:
        raise ValueError("ridge_grid must contain at least one value")
    coefficients = best["scaled_coefficients"] / scale
    return FittedAIPMModel(
        model="",
        coefficients=coefficients.astype("float64", copy=False),
        ridge_alpha=best["ridge_alpha"],
        validation_loss=best["validation_loss"],
        training_loss=best["training_loss"],
        fitted_training_return_mean=best["fitted_training_return_mean"],
        fitted_validation_return_mean=best["fitted_validation_return_mean"],
    )


def fit_model(
    training_months: list[AIPMMonth],
    validation_months: list[AIPMMonth],
    model: str,
    config: AIPMLinearTransformerConfig,
) -> FittedAIPMModel:
    train_basis = basis_matrix(training_months, model, training_returns=True)
    validation_basis = basis_matrix(validation_months, model, training_returns=True)
    fitted = _ridge_fit_msrr(train_basis, validation_basis, config.ridge_grid)
    fitted.model = model
    return fitted


def _raw_scores(month: AIPMMonth, fitted: FittedAIPMModel) -> np.ndarray:
    x = month.features
    n_features = x.shape[1]
    if fitted.model == "bsv":
        return x @ fitted.coefficients
    if fitted.model == "linear_attention":
        covariance = _month_covariance(month)
        covariance_vector = covariance.reshape(-1, order="C")
        coefficient_matrix = fitted.coefficients.reshape(
            n_features,
            n_features * n_features,
        )
        effective_coefficients = coefficient_matrix @ covariance_vector
        return x @ effective_coefficients
    raise ValueError(f"Unknown fitted model {fitted.model!r}")


def _normalize_weights(
    raw_weights: np.ndarray,
    gross_leverage: float,
    eps: float = 1e-12,
) -> np.ndarray:
    gross = float(np.abs(raw_weights).sum())
    if gross <= eps:
        return np.zeros_like(raw_weights)
    return raw_weights / gross * gross_leverage


def evaluate_month(
    month: AIPMMonth,
    fitted: FittedAIPMModel,
    config: AIPMLinearTransformerConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    raw_weights = _raw_scores(month, fitted) / month.n_stocks
    sdf_weights = _normalize_weights(raw_weights, config.gross_leverage)
    raw_sdf_return = float(raw_weights @ month.evaluation_returns)
    normalized_sdf_return = float(sdf_weights @ month.evaluation_returns)
    gross_weight = float(np.abs(sdf_weights).sum())
    raw_gross_weight = float(np.abs(raw_weights).sum())
    record: dict[str, Any] = {
        "signal_date": month.signal_date,
        "target_date": month.target_date,
        "model": fitted.model,
        "raw_sdf_return": raw_sdf_return,
        "sdf_return": normalized_sdf_return,
        "n_test_stocks": month.n_stocks,
        "ridge_alpha": fitted.ridge_alpha,
        "validation_loss": fitted.validation_loss,
        "raw_gross_weight": raw_gross_weight,
        "gross_weight": gross_weight,
        "net_weight": float(sdf_weights.sum()),
        "long_weight": float(sdf_weights[sdf_weights > 0].sum()),
        "short_weight": float(sdf_weights[sdf_weights < 0].sum()),
        "weight_hhi": float(np.square(sdf_weights).sum()),
    }
    weights = pd.DataFrame(
        {
            "signal_date": month.signal_date,
            "target_date": month.target_date,
            "ric": month.rics,
            "model": fitted.model,
            "raw_weight": raw_weights,
            "sdf_weight": sdf_weights,
            "target_return": month.evaluation_returns,
            "market_cap": month.market_caps,
            "market_cap_percentile": month.market_cap_percentiles,
        }
    )
    return record, weights


def _split_training_months(
    months: list[AIPMMonth],
    cutoff: pd.Timestamp,
    config: AIPMLinearTransformerConfig,
) -> tuple[list[AIPMMonth], list[AIPMMonth]]:
    train = [month for month in months if month.target_date <= cutoff]
    if config.training_window_months is not None:
        train = train[-config.training_window_months :]
    validation_count = min(config.validation_months, max(1, len(train) // 5))
    core = train[:-validation_count]
    validation = train[-validation_count:]
    return core, validation


def run_walk_forward_aipm(
    months: list[AIPMMonth],
    config: AIPMLinearTransformerConfig,
    models: tuple[str, ...] = ("bsv", "linear_attention"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = sorted(months, key=lambda month: month.signal_date)
    monthly_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    for year in range(config.first_test_year, config.last_test_year + 1):
        cutoff = pd.Timestamp(year=year - 1, month=12, day=31)
        core_months, validation_months = _split_training_months(months, cutoff, config)
        test_months = [month for month in months if month.signal_date.year == year]
        if not test_months:
            continue
        if len(core_months) < config.min_training_months or not validation_months:
            continue
        for model in models:
            fitted = fit_model(core_months, validation_months, model, config)
            fit_records.append(
                {
                    "model": model,
                    "test_year": year,
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
                    "n_parameters": int(len(fitted.coefficients)),
                    "ridge_alpha": fitted.ridge_alpha,
                    "training_loss": fitted.training_loss,
                    "validation_loss": fitted.validation_loss,
                    "fitted_training_return_mean": fitted.fitted_training_return_mean,
                    "fitted_validation_return_mean": fitted.fitted_validation_return_mean,
                }
            )
            for month in test_months:
                record, weights = evaluate_month(month, fitted, config)
                monthly_records.append(record)
                weight_frames.append(weights)
    monthly = pd.DataFrame.from_records(monthly_records)
    fit_log = pd.DataFrame.from_records(fit_records)
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else pd.DataFrame()
    )
    return monthly, fit_log, weights


def add_weight_turnover(monthly: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty or weights.empty:
        monthly = monthly.copy()
        monthly["weight_turnover"] = np.nan
        return monthly
    turnover_records: list[dict[str, Any]] = []
    weight_frame = weights[["signal_date", "ric", "model", "sdf_weight"]].copy()
    weight_frame["signal_date"] = pd.to_datetime(weight_frame["signal_date"])
    for model, group in weight_frame.groupby("model", sort=True):
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
    turnover_frame = pd.DataFrame.from_records(turnover_records)
    out = monthly.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    return out.merge(turnover_frame, on=["signal_date", "model"], how="left")


def summarize_aipm(monthly: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model, group in monthly.groupby("model", sort=True):
        returns = group["sdf_return"].astype(float)
        raw_returns = group["raw_sdf_return"].astype(float)
        annualized_return = float(returns.mean() * 12.0)
        annualized_volatility = float(returns.std(ddof=1) * math.sqrt(12.0))
        raw_annualized_return = float(raw_returns.mean() * 12.0)
        raw_annualized_volatility = float(raw_returns.std(ddof=1) * math.sqrt(12.0))
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
                "average_raw_gross_weight": float(group["raw_gross_weight"].mean()),
                "average_gross_weight": float(group["gross_weight"].mean()),
                "average_net_weight": float(group["net_weight"].mean()),
                "average_weight_hhi": float(group["weight_hhi"].mean()),
                "average_monthly_turnover": float(group["weight_turnover"].mean()),
                "median_ridge_alpha": float(group["ridge_alpha"].median()),
            }
        )
    return pd.DataFrame.from_records(records)


def compare_attention_to_bsv(
    monthly: pd.DataFrame,
    hac_lags: int,
) -> pd.DataFrame:
    wide = monthly.pivot(index="signal_date", columns="model", values="sdf_return")
    if not {"bsv", "linear_attention"}.issubset(wide.columns):
        return pd.DataFrame()
    common = wide[["bsv", "linear_attention"]].dropna()
    if len(common) < 12:
        return pd.DataFrame()
    difference = common["linear_attention"] - common["bsv"]
    fit = sm.OLS(difference.to_numpy(), np.ones((len(difference), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )
    annualized_difference = float(difference.mean() * 12.0)
    annualized_volatility = float(difference.std(ddof=1) * math.sqrt(12.0))
    return pd.DataFrame(
        [
            {
                "model": "linear_attention",
                "baseline": "bsv",
                "months": int(len(common)),
                "annualized_mean_difference": annualized_difference,
                "annualized_difference_volatility": annualized_volatility,
                "difference_sharpe": annualized_difference / annualized_volatility
                if annualized_volatility > 0
                else np.nan,
                "hac_t": float(fit.tvalues[0]),
                "hac_p": float(fit.pvalues[0]),
                "correlation": float(common["linear_attention"].corr(common["bsv"])),
            }
        ]
    )


def build_aipm_linear_transformer_outputs(
    panel_path: Path,
    output_dir: Path,
    config: AIPMLinearTransformerConfig,
    risk_free: pd.Series | None = None,
    feature_set: str = "compustat_enriched",
) -> dict[str, object]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    full_feature_columns = FEATURE_SETS[feature_set]
    feature_columns = selected_feature_columns(
        full_feature_columns,
        config.max_attention_features,
    )
    panel = load_aipm_panel(panel_path, risk_free, feature_columns)
    months = build_months(panel, feature_columns, config)
    monthly, fit_log, weights = run_walk_forward_aipm(months, config)
    if monthly.empty:
        raise RuntimeError("AIPM linear transformer walk-forward produced no returns")
    monthly = add_weight_turnover(monthly, weights)
    summary = summarize_aipm(monthly)
    comparison = compare_attention_to_bsv(monthly, config.hac_lags)

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "aipm_linear_transformer_monthly.csv", index=False)
    fit_log.to_csv(output_dir / "aipm_linear_transformer_fit_log.csv", index=False)
    summary.to_csv(output_dir / "aipm_linear_transformer_summary.csv", index=False)
    comparison.to_csv(
        output_dir / "aipm_linear_transformer_comparison.csv",
        index=False,
    )
    weights.to_parquet(
        output_dir / "aipm_linear_transformer_weights.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
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
        "full_feature_column_count": int(len(full_feature_columns)),
        "rows": {
            "input_rows": int(len(panel)),
            "months": int(len(months)),
            "monthly": int(len(monthly)),
            "fit_log": int(len(fit_log)),
            "summary": int(len(summary)),
            "comparison": int(len(comparison)),
            "weights": int(len(weights)),
        },
        "models": sorted(monthly["model"].unique().tolist()),
        "return_definition": "excess_return" if risk_free is not None else "raw_return",
        "source_paper": (
            "Kelly, Kuznetsov, Malamud and Xu (2026), "
            "Artificial Intelligence Asset Pricing Models"
        ),
        "objective": (
            "ridge-penalized MSRR for no-attention BSV and linear cross-asset "
            "attention SDFs, evaluated with gross-normalized OOS weights"
        ),
        "causality_check": causality,
    }
    (output_dir / "aipm_linear_transformer_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
