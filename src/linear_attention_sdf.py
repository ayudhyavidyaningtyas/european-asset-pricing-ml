"""Linear attention SDF tests for European equities.

This module implements the closed-form linear portfolio transformer from
Kelly, Kuznetsov, Malamud, and Xu (2026) in a deliberately small form that
fits the existing monthly European stock panel.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from asset_pricing import RAW_FEATURES


FEATURE_COLUMNS = [f"{feature}_rank" for feature in RAW_FEATURES]


@dataclass(frozen=True)
class LinearAttentionSDFConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_monthly_stocks: int = 100
    min_training_months: int = 72
    validation_months: int = 24
    training_window_months: int | None = None
    hac_lags: int = 6
    ridge_grid: tuple[float, ...] = (
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
        10000.0,
        100000.0,
    )


def load_sdf_panel(
    panel_path: Path,
    risk_free: pd.Series | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load the minimal stock-month panel needed for SDF basis construction."""
    features = feature_columns or FEATURE_COLUMNS
    columns = [
        "date",
        "target_date",
        "target_return_1m",
        "model_eligible",
        *features,
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


def _basis_for_month(
    month: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float | pd.Timestamp | int]:
    x = month[feature_columns].to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    returns = month["sdf_target_return"].to_numpy(dtype=float)
    n_stocks = len(month)
    factor_returns = x.T @ returns / n_stocks
    characteristic_similarity = x.T @ x / n_stocks
    attention_returns = np.kron(
        characteristic_similarity.reshape(-1, order="F"),
        factor_returns,
    )
    record: dict[str, float | pd.Timestamp | int] = {
        "date": pd.Timestamp(month["date"].iloc[0]),
        "target_date": pd.Timestamp(month["target_date"].iloc[0]),
        "n_stocks": int(n_stocks),
    }
    record.update(
        {f"bsv_{index}": float(value) for index, value in enumerate(factor_returns)}
    )
    record.update(
        {
            f"linear_attention_{index}": float(value)
            for index, value in enumerate(attention_returns)
        }
    )
    return record


def build_monthly_sdf_basis(
    panel: pd.DataFrame,
    config: LinearAttentionSDFConfig,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Build monthly characteristic-managed returns for BSV and attention SDFs."""
    features = feature_columns or FEATURE_COLUMNS
    records = []
    for _, month in panel.groupby("date", sort=True):
        if len(month) < config.min_monthly_stocks:
            continue
        records.append(_basis_for_month(month, features))
    return pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)


def _ridge_predict_one(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    ridge_grid: tuple[float, ...],
) -> tuple[float, float, float]:
    scale = train_x.std(axis=0, ddof=0)
    scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
    train = train_x / scale
    validation = validation_x / scale
    test = test_x / scale

    y = np.ones(train.shape[0])
    kernel = train @ train.T
    best: tuple[float, float, np.ndarray] | None = None
    for penalty in ridge_grid:
        system = kernel + penalty * np.eye(kernel.shape[0])
        try:
            alpha = np.linalg.solve(system, y)
        except np.linalg.LinAlgError:
            alpha = np.linalg.lstsq(system, y, rcond=None)[0]
        coefficients = train.T @ alpha
        validation_return = validation @ coefficients
        validation_loss = float(np.mean(np.square(1.0 - validation_return)))
        if best is None or validation_loss < best[0]:
            best = (validation_loss, float(penalty), coefficients)
    if best is None:
        raise ValueError("ridge_grid must contain at least one penalty")
    validation_loss, penalty, coefficients = best
    prediction = float((test @ coefficients).item())
    return prediction, penalty, validation_loss


def _model_columns(basis: pd.DataFrame, model: str) -> list[str]:
    prefix = f"{model}_"
    columns = [column for column in basis.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"No basis columns found for model {model!r}")
    return columns


def run_walk_forward_sdf(
    basis: pd.DataFrame,
    config: LinearAttentionSDFConfig,
    models: tuple[str, ...] = ("bsv", "linear_attention"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate SDF weights out of sample using causal monthly walk-forward splits."""
    basis = basis.sort_values("date").reset_index(drop=True)
    signal_dates = pd.to_datetime(basis["date"])
    target_dates = pd.to_datetime(basis["target_date"])
    monthly_records = []
    fit_records = []
    for model in models:
        model_columns = _model_columns(basis, model)
        values = basis[model_columns].to_numpy(dtype=float)
        for test_index, signal_date in enumerate(signal_dates):
            if not (
                config.first_test_year <= signal_date.year <= config.last_test_year
            ):
                continue
            train_index = np.flatnonzero(target_dates.to_numpy() <= signal_date)
            train_index = train_index[train_index < test_index]
            if config.training_window_months is not None:
                train_index = train_index[-config.training_window_months :]
            if len(train_index) < config.min_training_months:
                continue
            if len(train_index) <= config.validation_months:
                continue
            core_index = train_index[: -config.validation_months]
            validation_index = train_index[-config.validation_months :]
            prediction, penalty, validation_loss = _ridge_predict_one(
                values[core_index],
                values[validation_index],
                values[test_index : test_index + 1],
                config.ridge_grid,
            )
            monthly_records.append(
                {
                    "signal_date": signal_date,
                    "target_date": target_dates.iloc[test_index],
                    "model": model,
                    "sdf_return": prediction,
                    "n_test_stocks": int(basis.loc[test_index, "n_stocks"]),
                    "ridge_z": penalty,
                    "validation_loss": validation_loss,
                }
            )
            fit_records.append(
                {
                    "model": model,
                    "signal_date": signal_date,
                    "train_start": signal_dates.iloc[train_index[0]],
                    "train_signal_end": signal_dates.iloc[train_index[-1]],
                    "train_target_end": target_dates.iloc[train_index[-1]],
                    "validation_start": signal_dates.iloc[validation_index[0]],
                    "validation_end": signal_dates.iloc[validation_index[-1]],
                    "training_months": int(len(train_index)),
                    "ridge_z": penalty,
                    "validation_loss": validation_loss,
                }
            )
    monthly = pd.DataFrame.from_records(monthly_records)
    fit_log = pd.DataFrame.from_records(fit_records)
    return monthly, fit_log


def summarize_sdf_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Annualized SDF performance by model."""
    records = []
    for model, group in monthly.groupby("model", sort=True):
        returns = group["sdf_return"].astype(float)
        annualized_return = float(returns.mean() * 12.0)
        annualized_volatility = float(returns.std(ddof=1) * np.sqrt(12.0))
        records.append(
            {
                "model": model,
                "months": int(len(group)),
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility,
                "sharpe": annualized_return / annualized_volatility
                if annualized_volatility > 0
                else np.nan,
                "monthly_min": float(returns.min()),
                "monthly_max": float(returns.max()),
                "median_ridge_z": float(group["ridge_z"].median()),
            }
        )
    return pd.DataFrame.from_records(records)


def compare_attention_to_bsv(
    monthly: pd.DataFrame,
    hac_lags: int = 6,
) -> pd.DataFrame:
    """Compare linear attention SDF returns with the BSV benchmark."""
    wide = monthly.pivot(index="signal_date", columns="model", values="sdf_return")
    required = {"bsv", "linear_attention"}
    if not required.issubset(wide.columns):
        return pd.DataFrame()
    common = wide[["bsv", "linear_attention"]].dropna()
    difference = common["linear_attention"] - common["bsv"]
    if len(difference) < 12:
        return pd.DataFrame()
    fit = sm.OLS(difference.to_numpy(), np.ones((len(difference), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )
    annualized_difference = float(difference.mean() * 12.0)
    annualized_volatility = float(difference.std(ddof=1) * np.sqrt(12.0))
    return pd.DataFrame(
        [
            {
                "model": "linear_attention",
                "baseline": "bsv",
                "months": int(len(difference)),
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


def build_linear_attention_outputs(
    panel_path: Path,
    output_dir: Path,
    config: LinearAttentionSDFConfig,
    risk_free: pd.Series | None = None,
) -> dict[str, object]:
    panel = load_sdf_panel(panel_path, risk_free=risk_free)
    basis = build_monthly_sdf_basis(panel, config)
    monthly, fit_log = run_walk_forward_sdf(basis, config)
    summary = summarize_sdf_returns(monthly)
    comparison = compare_attention_to_bsv(monthly, config.hac_lags)

    output_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_dir / "linear_attention_sdf_monthly.csv", index=False)
    fit_log.to_csv(output_dir / "linear_attention_sdf_fit_log.csv", index=False)
    summary.to_csv(output_dir / "linear_attention_sdf_summary.csv", index=False)
    comparison.to_csv(
        output_dir / "linear_attention_sdf_comparison.csv",
        index=False,
    )
    manifest: dict[str, object] = {
        "config": asdict(config),
        "panel_path": str(panel_path),
        "rows": {
            "basis_months": int(len(basis)),
            "monthly": int(len(monthly)),
            "fit_log": int(len(fit_log)),
            "summary": int(len(summary)),
            "comparison": int(len(comparison)),
        },
        "models": sorted(monthly["model"].unique().tolist()) if not monthly.empty else [],
        "return_definition": "excess_return" if risk_free is not None else "raw_return",
        "source_paper": (
            "Kelly, Kuznetsov, Malamud and Xu (2026), "
            "Artificial Intelligence Asset Pricing Models"
        ),
    }
    with (output_dir / "linear_attention_sdf_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
