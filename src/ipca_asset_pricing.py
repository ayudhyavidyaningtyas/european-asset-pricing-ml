"""Instrumented PCA (Kelly, Pruitt and Su) for European equities.

IPCA is the linear ancestor of the conditional autoencoder: characteristics map
into factor loadings through a single matrix rather than through a network,

    r_{i,t+1} = z_{i,t}' Gamma_beta f_{t+1} + epsilon_{i,t+1},

so it is the natural "is the nonlinearity earning its keep" benchmark. The
restricted model above carries the no-arbitrage restriction that characteristics
predict returns only through exposures to the K latent factors. The unrestricted
model adds an intercept block Gamma_alpha; testing whether it is zero is the
canonical IPCA specification test and is reported here as a diagnostic.

Estimation is alternating least squares on characteristic-managed moments,

    x_t = Z_t' r_t / N_t,      W_t = Z_t' Z_t / N_t,

which reduces every step to L x K objects and is exact rather than approximate:
the ALS fixed point is the same one obtained from the full N-dimensional problem.

The walk-forward protocol, universe construction, factor-SDF fitting and
stock-level weight normalisation are deliberately shared with
``autoencoder_asset_pricing`` so that the two models are compared on identical
samples rather than merely similar ones.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asset_pricing_ml import COMPUSTAT_FEATURE_COLUMNS, FEATURE_SETS
from autoencoder_asset_pricing import (
    AutoencoderMonthBatch,
    _fit_factor_sdf,
    _split_training_months,
    _standardizer,
    build_month_batches,
    load_autoencoder_panel,
)


@dataclass(frozen=True)
class IPCAConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_monthly_stocks: int = 100
    min_training_months: int = 72
    validation_months: int = 24
    training_window_months: int | None = None
    n_factors: int = 5
    # Appends a constant instrument so the model can carry a level factor.
    include_constant: bool = True
    max_iterations: int = 500
    tolerance: float = 1e-7
    factor_ridge: float = 1e-8
    gamma_ridge: float = 1e-8
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
    universe_selection: str = "random"
    random_state: int = 42


@dataclass
class ManagedMoments:
    """Characteristic-managed first and second moments for one month."""

    signal_date: pd.Timestamp
    target_date: pd.Timestamp
    x: np.ndarray  # L
    w: np.ndarray  # L x L
    n_stocks: int


def _instruments(features: np.ndarray, include_constant: bool) -> np.ndarray:
    if not include_constant:
        return features
    constant = np.ones((features.shape[0], 1), dtype=features.dtype)
    return np.concatenate([features, constant], axis=1)


def build_managed_moments(
    batches: list[AutoencoderMonthBatch],
    include_constant: bool,
    training_returns: bool,
) -> list[ManagedMoments]:
    moments: list[ManagedMoments] = []
    for batch in batches:
        z = _instruments(batch.features.astype(float), include_constant)
        returns = (
            batch.training_returns if training_returns else batch.evaluation_returns
        ).astype(float)
        n_stocks = max(len(returns), 1)
        moments.append(
            ManagedMoments(
                signal_date=batch.signal_date,
                target_date=batch.target_date,
                x=z.T @ returns / n_stocks,
                w=z.T @ z / n_stocks,
                n_stocks=int(n_stocks),
            )
        )
    return moments


def _normalize_gamma(
    gamma: np.ndarray,
    factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Impose the KPS identification: orthonormal loadings, ordered factors.

    ``Z Gamma f`` is invariant to ``Gamma -> Gamma A``, ``f -> A^{-1} f``, so the
    fit is unchanged; this only pins down which of the observationally
    equivalent rotations gets reported.
    """

    left, singular, right = np.linalg.svd(gamma, full_matrices=False)
    gamma = left
    factors = factors @ (right.T * singular)

    # Rotate so the factor second-moment matrix is diagonal, largest first.
    second_moment = factors.T @ factors / max(len(factors), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    gamma = gamma @ eigenvectors
    factors = factors @ eigenvectors

    # Sign convention: each factor has a non-negative mean.
    signs = np.where(factors.mean(axis=0) < 0.0, -1.0, 1.0)
    return gamma * signs, factors * signs


def _solve_factors(
    gamma: np.ndarray,
    moments: list[ManagedMoments],
    factor_ridge: float,
) -> np.ndarray:
    n_factors = gamma.shape[1]
    factors = np.zeros((len(moments), n_factors), dtype=float)
    ridge = factor_ridge * np.eye(n_factors)
    for index, moment in enumerate(moments):
        system = gamma.T @ moment.w @ gamma + ridge
        rhs = gamma.T @ moment.x
        try:
            factors[index] = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            factors[index] = np.linalg.lstsq(system, rhs, rcond=None)[0]
    return factors


def _solve_gamma(
    factors: np.ndarray,
    moments: list[ManagedMoments],
    n_instruments: int,
    gamma_ridge: float,
) -> np.ndarray:
    """Least-squares update for vec(Gamma) given the factor path.

    Stacking the per-month normal equations gives
    ``sum_t (f_t f_t' kron W_t) vec(Gamma) = vec(sum_t x_t f_t')``.
    """

    n_factors = factors.shape[1]
    size = n_instruments * n_factors
    system = np.zeros((size, size), dtype=float)
    rhs = np.zeros((n_instruments, n_factors), dtype=float)
    for index, moment in enumerate(moments):
        factor = factors[index]
        system += np.kron(np.outer(factor, factor), moment.w)
        rhs += np.outer(moment.x, factor)
    system += gamma_ridge * np.eye(size)
    # vec stacks columns, so reshape with Fortran order to recover Gamma.
    try:
        solution = np.linalg.solve(system, rhs.reshape(-1, order="F"))
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, rhs.reshape(-1, order="F"), rcond=None)[0]
    return solution.reshape((n_instruments, n_factors), order="F")


def fit_ipca(
    moments: list[ManagedMoments],
    config: IPCAConfig,
) -> dict[str, Any]:
    """Alternating least squares for the restricted IPCA model."""

    if not moments:
        raise ValueError("fit_ipca requires at least one month of moments")
    n_instruments = moments[0].x.shape[0]
    if config.n_factors > n_instruments:
        raise ValueError("n_factors cannot exceed the number of instruments")

    # Initialise from the leading left singular vectors of the managed-return
    # matrix -- the PCA solution the ALS then refines.
    managed = np.column_stack([moment.x for moment in moments])
    left, _, _ = np.linalg.svd(managed, full_matrices=False)
    gamma = left[:, : config.n_factors]

    factors = _solve_factors(gamma, moments, config.factor_ridge)
    converged = False
    iterations = 0
    for iterations in range(1, config.max_iterations + 1):
        previous = gamma
        gamma = _solve_gamma(factors, moments, n_instruments, config.gamma_ridge)
        factors = _solve_factors(gamma, moments, config.factor_ridge)
        gamma, factors = _normalize_gamma(gamma, factors)
        shift = float(np.max(np.abs(gamma - previous)))
        if shift < config.tolerance:
            converged = True
            break

    # Given the converged loadings and factor path, the average managed pricing
    # error is the unrestricted intercept block Gamma_alpha. Under the
    # no-arbitrage restriction it should be indistinguishable from zero.
    residual_sum = np.zeros(n_instruments, dtype=float)
    weight_sum = np.zeros((n_instruments, n_instruments), dtype=float)
    alpha_path = np.zeros((len(moments), n_instruments), dtype=float)
    for index, moment in enumerate(moments):
        residual = moment.x - moment.w @ gamma @ factors[index]
        alpha_path[index] = residual
        residual_sum += residual
        weight_sum += moment.w
    try:
        gamma_alpha = np.linalg.solve(weight_sum, residual_sum)
    except np.linalg.LinAlgError:
        gamma_alpha = np.linalg.lstsq(weight_sum, residual_sum, rcond=None)[0]

    return {
        "gamma": gamma,
        "factors": factors,
        "gamma_alpha": gamma_alpha,
        "alpha_path": alpha_path,
        "iterations": int(iterations),
        "converged": bool(converged),
    }


def _fit_diagnostics(
    gamma: np.ndarray,
    moments: list[ManagedMoments],
    factors: np.ndarray,
) -> dict[str, float]:
    """Managed-space fit quality, comparable across cells of a grid."""

    explained = 0.0
    total = 0.0
    for index, moment in enumerate(moments):
        fitted = moment.w @ gamma @ factors[index]
        explained += float(np.sum(np.square(moment.x - fitted)))
        total += float(np.sum(np.square(moment.x)))
    return {
        "managed_sse": explained,
        "managed_ss": total,
        "managed_r2": float(1.0 - explained / max(total, 1e-18)),
        "loss": float(explained / max(len(moments), 1)),
    }


def _factor_frame(factors: np.ndarray, moments: list[ManagedMoments]) -> pd.DataFrame:
    frame = pd.DataFrame(
        factors, columns=[f"factor_{index}" for index in range(factors.shape[1])]
    )
    frame.insert(0, "signal_date", [moment.signal_date for moment in moments])
    frame.insert(1, "target_date", [moment.target_date for moment in moments])
    return frame


def _evaluate_test_batches(
    gamma: np.ndarray,
    test_batches: list[AutoencoderMonthBatch],
    training_factor_mean: np.ndarray,
    sdf_fit: dict[str, Any],
    config: IPCAConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    factor_records: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    sdf_coefficients = np.asarray(sdf_fit["sdf_coefficients"], dtype=float)

    for batch in test_batches:
        z = _instruments(batch.features.astype(float), config.include_constant)
        returns = batch.evaluation_returns.astype(float)
        n_stocks = max(batch.n_stocks, 1)
        betas = z @ gamma

        # Contemporaneous factor realisation, as in the autoencoder: the
        # cross-sectional GLS projection of realised returns on loadings.
        gram = betas.T @ betas / n_stocks
        ridge = config.factor_ridge * np.eye(gram.shape[0])
        try:
            factor = np.linalg.solve(gram + ridge, betas.T @ returns / n_stocks)
        except np.linalg.LinAlgError:
            factor = np.linalg.lstsq(
                gram + ridge, betas.T @ returns / n_stocks, rcond=None
            )[0]

        reconstructed = betas @ factor
        predicted = betas @ training_factor_mean
        residual = returns - reconstructed
        predictive_residual = returns - predicted
        pricing_moments = (z * predictive_residual[:, None]).mean(axis=0)

        try:
            stock_sdf_direction = np.linalg.solve(gram + ridge, sdf_coefficients)
        except np.linalg.LinAlgError:
            stock_sdf_direction = np.linalg.lstsq(
                gram + ridge, sdf_coefficients, rcond=None
            )[0]
        raw_weights = (betas @ stock_sdf_direction) / n_stocks
        gross = max(float(np.abs(raw_weights).sum()), 1e-12)
        sdf_weights = raw_weights / gross

        record: dict[str, Any] = {
            "signal_date": batch.signal_date,
            "target_date": batch.target_date,
            "model": "ipca",
            "n_test_stocks": batch.n_stocks,
            "reconstruction_sse": float(np.square(residual).sum()),
            "predictive_sse": float(np.square(predictive_residual).sum()),
            "return_ss": float(np.square(returns).sum()),
            "total_r2_month": float(
                1.0 - np.square(residual).sum() / max(np.square(returns).sum(), 1e-12)
            ),
            "predictive_r2_month": float(
                1.0
                - np.square(predictive_residual).sum()
                / max(np.square(returns).sum(), 1e-12)
            ),
            "sdf_return": float(factor @ sdf_coefficients),
            "stock_sdf_return": float(np.dot(sdf_weights, returns)),
            "stock_sdf_gross_weight": float(np.abs(sdf_weights).sum()),
            "stock_sdf_net_weight": float(sdf_weights.sum()),
            "stock_sdf_weight_hhi": float(np.square(sdf_weights).sum()),
            "pricing_moment_l2": float(np.linalg.norm(pricing_moments)),
            "max_abs_pricing_moment": float(np.abs(pricing_moments).max()),
        }
        for index, value in enumerate(factor):
            record[f"factor_{index}"] = float(value)
        monthly_records.append(record)
        factor_records.append(
            {
                "signal_date": batch.signal_date,
                "target_date": batch.target_date,
                **{f"factor_{index}": float(value) for index, value in enumerate(factor)},
            }
        )
        weight_frames.append(
            pd.DataFrame(
                {
                    "signal_date": batch.signal_date,
                    "target_date": batch.target_date,
                    "ric": batch.rics,
                    "model": "ipca_stock_sdf",
                    "raw_weight": raw_weights,
                    "sdf_weight": sdf_weights,
                    "raw_score": raw_weights * n_stocks,
                    "target_return": returns,
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
                    "model": "ipca",
                    "target_return": returns,
                    "reconstructed_return": reconstructed,
                    "predicted_return": predicted,
                    "reconstruction_residual": residual,
                    "predictive_residual": predictive_residual,
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
        pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    )
    return monthly, predictions, factors, weights


def run_walk_forward_ipca(
    panel: pd.DataFrame,
    config: IPCAConfig,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = feature_columns or COMPUSTAT_FEATURE_COLUMNS
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
            core, feature_columns, feature_mean, feature_scale, config, seed_offset=year * 10
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
            test, feature_columns, feature_mean, feature_scale, config, seed_offset=year * 30
        )
        if len(core_batches) < config.min_training_months or not validation_batches:
            continue

        started = time.time()
        core_moments = build_managed_moments(
            core_batches, config.include_constant, training_returns=True
        )
        validation_moments = build_managed_moments(
            validation_batches, config.include_constant, training_returns=True
        )
        fit = fit_ipca(core_moments, config)
        gamma = fit["gamma"]
        elapsed = time.time() - started

        # Validation factors are recovered with the loadings held fixed, so the
        # validation loss reflects out-of-core fit rather than a refit.
        validation_factors = _solve_factors(gamma, validation_moments, config.factor_ridge)
        core_diagnostics = _fit_diagnostics(gamma, core_moments, fit["factors"])
        validation_diagnostics = _fit_diagnostics(
            gamma, validation_moments, validation_factors
        )

        core_factor_frame = _factor_frame(fit["factors"], core_moments)
        validation_factor_frame = _factor_frame(validation_factors, validation_moments)
        sdf_fit = _fit_factor_sdf(core_factor_frame, validation_factor_frame, config)
        training_factor_mean = fit["factors"].mean(axis=0)

        monthly, predictions, factors, weights = _evaluate_test_batches(
            gamma, test_batches, training_factor_mean, sdf_fit, config
        )
        if monthly.empty:
            continue
        monthly_frames.append(monthly)
        prediction_frames.append(predictions)
        factor_frames.append(factors)
        weight_frames.append(weights)

        alpha_path = fit["alpha_path"]
        alpha_mean = alpha_path.mean(axis=0)
        alpha_se = alpha_path.std(axis=0, ddof=1) / np.sqrt(max(len(alpha_path), 1))
        alpha_t = alpha_mean / np.where(alpha_se > 0, alpha_se, np.nan)

        fit_records.append(
            {
                "model": "ipca",
                "test_year": year,
                "train_signal_start": core_batches[0].signal_date,
                "train_signal_end": core_batches[-1].signal_date,
                "train_target_end": max(batch.target_date for batch in core_batches),
                "train_label_cutoff": cutoff,
                "core_months": len(core_batches),
                "validation_months": len(validation_batches),
                "test_months": len(test_batches),
                "n_instruments": int(gamma.shape[0]),
                "n_factors": int(gamma.shape[1]),
                "iterations": fit["iterations"],
                "converged": fit["converged"],
                "fit_seconds": elapsed,
                "training_loss": core_diagnostics["loss"],
                "validation_loss": validation_diagnostics["loss"],
                "training_managed_r2": core_diagnostics["managed_r2"],
                "validation_managed_r2": validation_diagnostics["managed_r2"],
                "sdf_ridge_alpha": float(sdf_fit["sdf_ridge_alpha"]),
                "sdf_training_loss": float(sdf_fit["sdf_training_loss"]),
                "sdf_validation_loss": float(sdf_fit["sdf_validation_loss"]),
                "gamma_alpha_l2": float(np.linalg.norm(fit["gamma_alpha"])),
                "gamma_alpha_max_abs": float(np.abs(fit["gamma_alpha"]).max()),
                "alpha_max_abs_t": float(np.nanmax(np.abs(alpha_t))),
                **{
                    f"training_factor_mean_{index}": float(value)
                    for index, value in enumerate(training_factor_mean)
                },
            }
        )

    monthly = (
        pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    factors = (
        pd.concat(factor_frames, ignore_index=True) if factor_frames else pd.DataFrame()
    )
    weights = (
        pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    )
    fit_log = pd.DataFrame.from_records(fit_records)
    return monthly, predictions, factors, weights, fit_log


def summarize_ipca(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    sdf_returns = monthly["stock_sdf_return"].to_numpy(dtype=float)
    mean = float(np.mean(sdf_returns))
    volatility = float(np.std(sdf_returns, ddof=1)) if len(sdf_returns) > 1 else 0.0
    return pd.DataFrame(
        [
            {
                "model": "ipca",
                "months": int(len(monthly)),
                "total_r2": float(
                    1.0
                    - monthly["reconstruction_sse"].sum()
                    / max(monthly["return_ss"].sum(), 1e-12)
                ),
                "predictive_r2": float(
                    1.0
                    - monthly["predictive_sse"].sum()
                    / max(monthly["return_ss"].sum(), 1e-12)
                ),
                "mean_monthly_total_r2": float(monthly["total_r2_month"].mean()),
                "mean_monthly_predictive_r2": float(monthly["predictive_r2_month"].mean()),
                "annualized_sdf_return": mean * 12.0,
                "annualized_sdf_volatility": volatility * np.sqrt(12.0),
                "sdf_sharpe": float(mean / volatility * np.sqrt(12.0))
                if volatility > 0
                else np.nan,
                "average_pricing_moment_l2": float(monthly["pricing_moment_l2"].mean()),
                "max_abs_pricing_moment": float(monthly["max_abs_pricing_moment"].max()),
                "average_n_test_stocks": float(monthly["n_test_stocks"].mean()),
            }
        ]
    )


def build_ipca_outputs(
    panel_path: Path,
    output_dir: Path,
    config: IPCAConfig,
    risk_free: pd.Series | None = None,
    feature_set: str = "compustat_enriched",
) -> dict[str, Any]:
    feature_columns = list(FEATURE_SETS[feature_set])
    panel = load_autoencoder_panel(panel_path, risk_free, feature_columns)
    monthly, predictions, factors, weights, fit_log = run_walk_forward_ipca(
        panel, config, feature_columns
    )
    summary = summarize_ipca(monthly)

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "ipca_monthly.csv", index=False)
    fit_log.to_csv(output_dir / "ipca_fit_log.csv", index=False)
    factors.to_csv(output_dir / "ipca_factors.csv", index=False)
    summary.to_csv(output_dir / "ipca_summary.csv", index=False)
    if not predictions.empty:
        predictions.to_parquet(output_dir / "ipca_predictions.parquet", index=False)
    if not weights.empty:
        weights.to_parquet(output_dir / "ipca_weights.parquet", index=False)

    causality_violations = 0
    if not fit_log.empty:
        causality_violations = int(
            (
                pd.to_datetime(fit_log["train_target_end"])
                > pd.to_datetime(fit_log["train_label_cutoff"])
            ).sum()
        )
    duplicate_predictions = 0
    if not predictions.empty:
        duplicate_predictions = int(
            predictions.duplicated(subset=["signal_date", "ric"]).sum()
        )

    manifest = {
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
        "return_definition": "excess_return" if risk_free is not None else "total_return",
        "source_paper": "Kelly, Pruitt and Su, Characteristics are Covariances (IPCA)",
        "objective": (
            "restricted instrumented PCA estimated by alternating least squares on "
            "characteristic-managed moments, with the unrestricted intercept block "
            "reported as a specification diagnostic"
        ),
        "causality_check": {
            "train_target_after_cutoff": causality_violations,
            "duplicate_prediction_security_months": duplicate_predictions,
        },
    }
    (output_dir / "ipca_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
