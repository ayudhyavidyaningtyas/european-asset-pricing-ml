"""Peer-implied fundamental mispricing signals for European equities.

This module adapts the European agnostic-fundamental-analysis design of
Hanauer, Kononova and Rapp (2022): use accounting variables to infer a stock's
peer-implied fair value, then trade the gap between that fair value and the
observed market value.  The implementation is deliberately separate from the
return-forecasting models in ``asset_pricing_ml`` because the supervised target
is same-month relative market value, not next-month returns.
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
from scipy import stats as scipy_stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from asset_pricing_ml import (
    construct_monthly_portfolios,
    paired_sharpe_significance,
    portfolio_summary,
)


ACCOUNTING_FEATURES = [
    "fv_assets_total",
    "fv_current_assets",
    "fv_current_liabilities",
    "fv_cash",
    "fv_inventories",
    "fv_receivables",
    "fv_ppe_net",
    "fv_intangibles",
    "fv_liabilities_total",
    "fv_total_debt",
    "fv_common_equity",
    "fv_preferred_stock",
    "fv_revenue",
    "fv_cogs",
    "fv_sga",
    "fv_rd",
    "fv_depreciation",
    "fv_operating_income",
    "fv_ebit",
    "fv_net_income",
    "fv_operating_cash_flow",
    "fv_capex",
    "fv_dividends",
]
ACCOUNTING_RANK_FEATURES = [f"{feature}_rank" for feature in ACCOUNTING_FEATURES]
SUPPORTED_FAIR_VALUE_MODELS = {"linear", "rf", "hist_gbm", "ensemble"}


@dataclass(frozen=True)
class FundamentalMispricingConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    accounting_lag_months: int = 6
    training_window_months: int = 48
    min_training_rows: int = 10_000
    min_training_months: int = 24
    min_monthly_stocks: int = 100
    min_accounting_features: int = 8
    max_training_rows: int | None = 150_000
    portfolio_quantile: float = 0.10
    cost_grid_bps: tuple[int, ...] = (0, 10, 25, 50)
    random_state: int = 42
    exclude_financials: bool = False
    fair_value_target: str = "market_share"
    linear_alpha: float = 1.0
    rf_estimators: int = 120
    rf_max_depth: int | None = 10
    rf_min_samples_leaf: int = 50
    hist_learning_rate: float = 0.05
    hist_max_iter: int = 150
    hist_max_leaf_nodes: int = 31
    hist_min_samples_leaf: int = 50


def _normalise_isin(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})
    )


def _month_end(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _sum_min_count(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series(np.nan, index=frame.index)
    return frame[available].sum(axis=1, min_count=1)


def prepare_accounting_values(
    annual: pd.DataFrame,
    config: FundamentalMispricingConfig,
) -> pd.DataFrame:
    """Prepare Compustat annual accounting variables and availability dates."""

    work = annual.copy()
    work.columns = [str(column).lower() for column in work.columns]
    required = {"isin", "datadate"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Compustat annual export missing columns: {sorted(missing)}")

    numeric_columns = [
        "act",
        "at",
        "capx",
        "ceq",
        "che",
        "cogs",
        "dlc",
        "dltt",
        "dp",
        "dvc",
        "dvt",
        "ebit",
        "ib",
        "intan",
        "invt",
        "lct",
        "lt",
        "nicon",
        "oancf",
        "oiadp",
        "ppent",
        "pstk",
        "rect",
        "revt",
        "sale",
        "seq",
        "xrd",
        "xsga",
    ]
    _to_numeric(work, numeric_columns)
    work["isin_norm"] = _normalise_isin(work["isin"])
    work["period_end"] = _month_end(work["datadate"])

    base_available = work["period_end"] + pd.offsets.MonthEnd(
        config.accounting_lag_months
    )
    final_date = (
        _month_end(work["fdate"])
        if "fdate" in work
        else pd.Series(pd.NaT, index=work.index)
    )
    preliminary_date = (
        _month_end(work["pdate"])
        if "pdate" in work
        else pd.Series(pd.NaT, index=work.index)
    )
    report_date = final_date.combine_first(preliminary_date)
    work["available_date"] = pd.concat([base_available, report_date], axis=1).max(
        axis=1
    )

    value_columns = [column for column in numeric_columns if column in work]
    work["source_completeness"] = work[value_columns].notna().sum(axis=1)
    work = work.dropna(subset=["isin_norm", "period_end", "available_date"])
    work = (
        work.sort_values(["isin_norm", "period_end", "source_completeness"])
        .groupby(["isin_norm", "period_end"], as_index=False)
        .last()
        .sort_values(["isin_norm", "period_end"])
        .reset_index(drop=True)
    )

    preferred_stock = work["pstk"].fillna(0.0)
    book_equity = work["ceq"].combine_first(work["seq"].sub(preferred_stock))
    book_equity = book_equity.combine_first(work["seq"])

    values = pd.DataFrame(
        {
            "isin_norm": work["isin_norm"],
            "period_end": work["period_end"],
            "available_date": work["available_date"],
            "fv_assets_total": work["at"],
            "fv_current_assets": work["act"],
            "fv_current_liabilities": work["lct"],
            "fv_cash": work["che"],
            "fv_inventories": work["invt"],
            "fv_receivables": work["rect"],
            "fv_ppe_net": work["ppent"],
            "fv_intangibles": work["intan"],
            "fv_liabilities_total": work["lt"],
            "fv_total_debt": _sum_min_count(work, ["dltt", "dlc"]),
            "fv_common_equity": book_equity,
            "fv_preferred_stock": work["pstk"],
            "fv_revenue": work["revt"].combine_first(work["sale"]),
            "fv_cogs": work["cogs"],
            "fv_sga": work["xsga"],
            "fv_rd": work["xrd"],
            "fv_depreciation": work["dp"],
            "fv_operating_income": work["oiadp"],
            "fv_ebit": work["ebit"],
            "fv_net_income": work["nicon"].combine_first(work["ib"]),
            "fv_operating_cash_flow": work["oancf"],
            "fv_capex": work["capx"],
            "fv_dividends": work["dvt"].combine_first(work["dvc"]),
        }
    )
    if "gvkey" in work:
        values["comp_gvkey"] = work["gvkey"].astype("string")
    return values


def _rank_accounting_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    eligible = out["sample_eligible"].fillna(False)
    for feature in ACCOUNTING_FEATURES:
        rank_column = f"{feature}_rank"
        valid = eligible & out[feature].notna()
        out[rank_column] = np.nan
        if valid.any():
            out.loc[valid, rank_column] = (
                out.loc[valid].groupby("date")[feature].rank(method="average", pct=True)
                .mul(2.0)
                .sub(1.0)
            )
        out.loc[eligible & out[rank_column].isna(), rank_column] = 0.0
    out["fundamental_feature_count"] = out[ACCOUNTING_FEATURES].notna().sum(axis=1)
    return out


def _financial_mask(values: pd.Series) -> pd.Series:
    return values.astype("string").str.contains("financial", case=False, na=False)


def load_fundamental_mispricing_panel(
    panel_path: Path,
    annual_path: Path,
    config: FundamentalMispricingConfig,
) -> pd.DataFrame:
    panel_columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "target_return_rank",
        "company_market_cap",
        "market_cap_percentile",
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
        "TR.ISIN",
        "eligible",
        "model_eligible",
        "return_history_n",
        "momentum_12_2_rank",
    ]
    panel = pd.read_parquet(panel_path, columns=panel_columns)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["target_date"] = pd.to_datetime(panel["target_date"])
    panel["isin_norm"] = _normalise_isin(panel["TR.ISIN"])
    panel["sample_eligible"] = (
        panel["eligible"].fillna(False)
        & panel["model_eligible"].fillna(False)
        & panel["company_market_cap"].gt(0)
        & panel["return_history_n"].ge(24)
    )
    if config.exclude_financials:
        panel["sample_eligible"] &= ~_financial_mask(panel["TR.TRBCECONOMICSECTOR"])

    annual = pd.read_csv(annual_path, compression="gzip", low_memory=False)
    accounting = prepare_accounting_values(annual, config)
    accounting = accounting.sort_values(["available_date", "isin_norm"])
    left = panel.sort_values(["date", "isin_norm"]).copy()
    merged = pd.merge_asof(
        left,
        accounting,
        left_on="date",
        right_on="available_date",
        by="isin_norm",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = _rank_accounting_features(merged)

    market_total = (
        merged.loc[merged["sample_eligible"]]
        .groupby("date")["company_market_cap"]
        .transform("sum")
    )
    merged["total_sample_market_cap"] = market_total
    merged["actual_market_share"] = merged["company_market_cap"].div(market_total)
    merged["log_market_share"] = np.log(
        merged["actual_market_share"].where(merged["actual_market_share"].gt(0))
    )
    merged["fair_value_model_eligible"] = (
        merged["sample_eligible"]
        & merged["fundamental_feature_count"].ge(config.min_accounting_features)
        & merged["log_market_share"].notna()
    )
    return merged.sort_values(["date", "ric"]).reset_index(drop=True)


def _limit_training_rows(
    train: pd.DataFrame,
    maximum: int | None,
    random_state: int,
) -> pd.DataFrame:
    if maximum is None or len(train) <= maximum:
        return train
    fraction = maximum / len(train)
    parts = []
    for month_number, (_, group) in enumerate(train.groupby("date", sort=True)):
        sample_size = max(1, min(len(group), round(len(group) * fraction)))
        parts.append(
            group.sample(n=sample_size, random_state=random_state + month_number)
        )
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > maximum:
        sampled = sampled.sample(n=maximum, random_state=random_state)
    return sampled.sort_values(["date", "ric"])


def _walk_forward_masks(
    panel: pd.DataFrame,
    config: FundamentalMispricingConfig,
) -> list[tuple[int, pd.Timestamp, np.ndarray, np.ndarray]]:
    slices = []
    for year in range(config.first_test_year, config.last_test_year + 1):
        cutoff = pd.Timestamp(year=year - 1, month=12, day=31)
        train_mask = panel["date"].le(cutoff)
        if config.training_window_months is not None:
            train_months = np.sort(panel.loc[train_mask, "date"].dropna().unique())
            if len(train_months) > config.training_window_months:
                start = pd.Timestamp(train_months[-config.training_window_months])
                train_mask &= panel["date"].ge(start)
        test_mask = panel["date"].dt.year.eq(year)
        if test_mask.any():
            slices.append((year, cutoff, train_mask.to_numpy(), test_mask.to_numpy()))
    return slices


def _drop_sparse_training_months(
    train: pd.DataFrame,
    min_monthly_stocks: int,
) -> pd.DataFrame:
    counts = train.groupby("date")["ric"].transform("count")
    return train[counts.ge(min_monthly_stocks)]


def _fit_fair_value_model(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    config: FundamentalMispricingConfig,
    seed: int,
):
    if model_name == "linear":
        return Ridge(alpha=config.linear_alpha).fit(x, y)
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=config.rf_estimators,
            max_depth=config.rf_max_depth,
            min_samples_leaf=config.rf_min_samples_leaf,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ).fit(x, y)
    if model_name == "hist_gbm":
        return HistGradientBoostingRegressor(
            learning_rate=config.hist_learning_rate,
            max_iter=config.hist_max_iter,
            max_leaf_nodes=config.hist_max_leaf_nodes,
            min_samples_leaf=config.hist_min_samples_leaf,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=seed,
        ).fit(x, y)
    raise ValueError(f"Unsupported fair-value model: {model_name}")


def _model_metadata(model_name: str, model: Any) -> dict[str, Any]:
    if model_name == "linear":
        return {
            "selected_parameters": json.dumps({"alpha": model.alpha}),
            "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
        }
    if model_name == "rf":
        return {
            "selected_parameters": json.dumps(
                {
                    "n_estimators": model.n_estimators,
                    "max_depth": model.max_depth,
                    "min_samples_leaf": model.min_samples_leaf,
                    "max_features": model.max_features,
                },
                sort_keys=True,
            )
        }
    if model_name == "hist_gbm":
        return {
            "selected_parameters": json.dumps(
                {
                    "learning_rate": model.learning_rate,
                    "max_iter": model.max_iter,
                    "max_leaf_nodes": model.max_leaf_nodes,
                    "min_samples_leaf": model.min_samples_leaf,
                },
                sort_keys=True,
            ),
            "iterations": int(model.n_iter_),
        }
    return {}


def _prediction_frame(
    test: pd.DataFrame,
    model_label: str,
    base_model: str,
    raw_prediction: np.ndarray,
    test_year: int,
    cutoff: pd.Timestamp,
    fair_value_target: str,
) -> pd.DataFrame:
    output_columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "target_return_rank",
        "company_market_cap",
        "market_cap_percentile",
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
        "actual_market_share",
        "log_market_share",
        "fundamental_feature_count",
    ]
    output = test[output_columns].copy()
    if fair_value_target == "market_share":
        predicted_share = np.clip(raw_prediction.astype(float), 1e-12, 1.0)
        predicted_log_share = np.log(predicted_share)
    elif fair_value_target == "log_market_share":
        predicted_log_share = raw_prediction.astype(float)
        predicted_share = np.exp(np.clip(predicted_log_share, -50.0, 0.0))
    else:
        raise ValueError(
            "fair_value_target must be 'market_share' or 'log_market_share'"
        )
    actual_share = output["actual_market_share"].to_numpy(dtype=float)
    log_gap = predicted_log_share - output["log_market_share"].to_numpy(dtype=float)
    output["predicted_market_share"] = predicted_share
    output["predicted_log_market_share"] = predicted_log_share
    output["prediction"] = np.clip(log_gap, -5.0, 5.0).astype("float32")
    output["fair_value_to_market"] = np.exp(output["prediction"])
    output["mispricing_pct"] = predicted_share / actual_share - 1.0
    output["model"] = model_label
    output["base_model"] = base_model
    output["target_mode"] = "signal"
    output["test_year"] = test_year
    output["train_label_cutoff"] = cutoff
    output["is_delisting_candidate"] = False
    return output


def run_fundamental_mispricing_walk_forward(
    panel: pd.DataFrame,
    model_names: list[str],
    config: FundamentalMispricingConfig,
    include_momentum: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unknown = set(model_names) - SUPPORTED_FAIR_VALUE_MODELS
    if unknown:
        raise ValueError(f"Unknown fair-value models: {sorted(unknown)}")
    if config.fair_value_target not in {"market_share", "log_market_share"}:
        raise ValueError(
            "fair_value_target must be 'market_share' or 'log_market_share'"
        )
    if "ensemble" in model_names:
        models_to_fit = sorted((set(model_names) - {"ensemble"}) | {"rf", "hist_gbm"})
    else:
        models_to_fit = list(model_names)

    predictions: list[pd.DataFrame] = []
    fit_records: list[dict[str, Any]] = []
    importance_records: list[dict[str, Any]] = []

    for year, cutoff, train_mask, test_mask in _walk_forward_masks(panel, config):
        full_train = panel.loc[
            train_mask & panel["fair_value_model_eligible"]
        ].dropna(subset=["log_market_share"])
        full_train = _drop_sparse_training_months(
            full_train,
            config.min_monthly_stocks,
        )
        test = panel.loc[test_mask & panel["fair_value_model_eligible"]].copy()
        if len(full_train) < config.min_training_rows or test.empty:
            continue
        if full_train["date"].nunique() < config.min_training_months:
            continue

        train = _limit_training_rows(
            full_train,
            config.max_training_rows,
            config.random_state + year,
        )
        x_train = train[ACCOUNTING_RANK_FEATURES].to_numpy(dtype="float32", copy=False)
        target_column = (
            "actual_market_share"
            if config.fair_value_target == "market_share"
            else "log_market_share"
        )
        y_train = train[target_column].to_numpy(dtype="float32", copy=False)
        x_test = test[ACCOUNTING_RANK_FEATURES].to_numpy(dtype="float32", copy=False)

        model_predictions: dict[str, np.ndarray] = {}
        for model_name in models_to_fit:
            started = time.perf_counter()
            model = _fit_fair_value_model(
                model_name,
                x_train,
                y_train,
                config,
                seed=config.random_state + year,
            )
            predicted = model.predict(x_test).astype("float32", copy=False)
            model_predictions[model_name] = predicted
            elapsed = time.perf_counter() - started
            model_label = f"fv_{model_name}_signal"
            if model_name in model_names:
                predictions.append(
                    _prediction_frame(
                        test,
                        model_label,
                        model_name,
                        predicted,
                        year,
                        cutoff,
                        config.fair_value_target,
                    )
                )
                fit_records.append(
                    {
                        "model": model_label,
                        "base_model": model_name,
                        "test_year": year,
                        "train_rows_available": int(len(full_train)),
                        "train_rows_used": int(len(train)),
                        "train_months": int(train["date"].nunique()),
                        "test_rows": int(len(test)),
                        "train_signal_start": str(train["date"].min().date()),
                        "train_signal_end": str(train["date"].max().date()),
                        "train_label_cutoff": str(cutoff.date()),
                        "fit_seconds": elapsed,
                        **_model_metadata(model_name, model),
                    }
                )
            if model_name == "linear" and model_name in model_names:
                for feature, coefficient in zip(
                    ACCOUNTING_RANK_FEATURES, model.coef_, strict=True
                ):
                    importance_records.append(
                        {
                            "model": model_label,
                            "test_year": year,
                            "feature": feature.removesuffix("_rank"),
                            "importance_type": "linear_coefficient",
                            "importance": float(coefficient),
                        }
                    )
            elif model_name == "rf" and model_name in model_names:
                for feature, importance in zip(
                    ACCOUNTING_RANK_FEATURES, model.feature_importances_, strict=True
                ):
                    importance_records.append(
                        {
                            "model": model_label,
                            "test_year": year,
                            "feature": feature.removesuffix("_rank"),
                            "importance_type": "impurity_importance",
                            "importance": float(importance),
                        }
                    )

        if "ensemble" in model_names:
            members = [
                model_predictions[name]
                for name in ["rf", "hist_gbm"]
                if name in model_predictions
            ]
            if members:
                predicted = np.mean(np.column_stack(members), axis=1).astype("float32")
                predictions.append(
                    _prediction_frame(
                        test,
                        "fv_ensemble_signal",
                        "ensemble",
                        predicted,
                        year,
                        cutoff,
                        config.fair_value_target,
                    )
                )
                fit_records.append(
                    {
                        "model": "fv_ensemble_signal",
                        "base_model": "ensemble",
                        "test_year": year,
                        "train_rows_available": int(len(full_train)),
                        "train_rows_used": int(len(train)),
                        "train_months": int(train["date"].nunique()),
                        "test_rows": int(len(test)),
                        "train_signal_start": str(train["date"].min().date()),
                        "train_signal_end": str(train["date"].max().date()),
                        "train_label_cutoff": str(cutoff.date()),
                        "fit_seconds": 0.0,
                        "selected_parameters": json.dumps(
                            {"members": ["rf", "hist_gbm"]}
                        ),
                    }
                )

        if include_momentum:
            momentum = test.copy()
            output = _prediction_frame(
                momentum,
                "momentum_rank",
                "momentum",
                momentum[target_column].to_numpy(dtype="float32"),
                year,
                cutoff,
                config.fair_value_target,
            )
            output["prediction"] = momentum["momentum_12_2_rank"].to_numpy(
                dtype="float32", copy=False
            )
            output["target_mode"] = "rank"
            predictions.append(output)
            fit_records.append(
                {
                    "model": "momentum_rank",
                    "base_model": "momentum",
                    "test_year": year,
                    "train_rows_available": int(len(full_train)),
                    "train_rows_used": int(len(train)),
                    "train_months": int(train["date"].nunique()),
                    "test_rows": int(len(test)),
                    "train_signal_start": str(train["date"].min().date()),
                    "train_signal_end": str(train["date"].max().date()),
                    "train_label_cutoff": str(cutoff.date()),
                    "fit_seconds": 0.0,
                }
            )

    prediction_frame = (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    )
    return prediction_frame, pd.DataFrame(fit_records), pd.DataFrame(importance_records)


def fundamental_signal_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model_name, group in predictions.groupby("model", sort=True):
        group = group.dropna(
            subset=["target_return_1m", "target_return_rank", "prediction"]
        )
        if group.empty:
            continue
        monthly_ic = group.groupby("date").apply(
            lambda month: month["prediction"].corr(
                month["target_return_rank"], method="spearman"
            ),
            include_groups=False,
        )
        ic_standard_deviation = float(monthly_ic.std(ddof=1))
        ic_information_ratio = (
            math.copysign(math.inf, float(monthly_ic.mean()))
            if ic_standard_deviation == 0.0
            else float(monthly_ic.mean() / ic_standard_deviation * np.sqrt(12))
        )
        pearson = scipy_stats.pearsonr(
            group["prediction"].to_numpy(dtype=float),
            group["target_return_1m"].to_numpy(dtype=float),
        )
        records.append(
            {
                "model": model_name,
                "base_model": group["base_model"].iloc[0],
                "target_mode": group["target_mode"].iloc[0],
                "observations": int(len(group)),
                "months": int(group["date"].nunique()),
                "mean_monthly_spearman_ic": float(monthly_ic.mean()),
                "ic_information_ratio": ic_information_ratio,
                "positive_ic_month_fraction": float(monthly_ic.gt(0).mean()),
                "pooled_pearson_return_correlation": float(pearson.statistic),
                "pooled_pearson_return_p_value": float(pearson.pvalue),
                "mean_prediction": float(group["prediction"].mean()),
                "prediction_std": float(group["prediction"].std(ddof=1)),
                "mean_fundamental_feature_count": float(
                    group["fundamental_feature_count"].mean()
                ),
            }
        )
    return pd.DataFrame(records)


def build_fundamental_mispricing_outputs(
    panel_path: Path,
    annual_path: Path,
    output_dir: Path,
    model_names: list[str],
    config: FundamentalMispricingConfig,
    risk_free: pd.Series | None = None,
    include_momentum: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_fundamental_mispricing_panel(panel_path, annual_path, config)
    predictions, fit_log, importance = run_fundamental_mispricing_walk_forward(
        panel,
        model_names,
        config,
        include_momentum=include_momentum,
    )
    if predictions.empty:
        raise RuntimeError("Fundamental-mispricing run produced no predictions")

    metrics = fundamental_signal_metrics(predictions)
    monthly = construct_monthly_portfolios(predictions, config.portfolio_quantile)
    summary = portfolio_summary(
        monthly,
        metrics,
        config.cost_grid_bps,
        risk_free=risk_free,
    )
    baseline = "momentum_rank" if include_momentum else "fv_linear_signal"
    significance = paired_sharpe_significance(
        monthly,
        baseline_model=baseline,
        risk_free=risk_free,
    )

    predictions.to_parquet(
        output_dir / "predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    fit_log.to_csv(output_dir / "fit_log.csv", index=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    metrics.to_csv(output_dir / "prediction_metrics.csv", index=False)
    monthly.to_csv(output_dir / "monthly_portfolios.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    significance.to_csv(output_dir / "sharpe_significance.csv", index=False)

    manifest = {
        "paper": (
            "Hanauer, Kononova and Rapp (2022), Boosting agnostic fundamental "
            "analysis: using machine learning to identify mispricing in "
            "European stock markets"
        ),
        "panel_path": str(panel_path),
        "annual_path": str(annual_path),
        "models": model_names,
        "include_momentum": include_momentum,
        "accounting_features": ACCOUNTING_FEATURES,
        "fair_value_target": config.fair_value_target,
        "signal_definition": (
            "prediction is log(peer-implied market share / observed market share); "
            "higher values indicate undervaluation"
        ),
        "config": asdict(config),
        "rows": {
            "input_panel_rows": int(len(panel)),
            "fair_value_model_eligible_rows": int(
                panel["fair_value_model_eligible"].sum()
            ),
            "predictions": int(len(predictions)),
            "labelled_predictions": int(predictions["target_return_1m"].notna().sum()),
            "portfolio_months": int(len(monthly)),
            "feature_importance": int(len(importance)),
        },
        "causality_check": {
            "train_signal_after_cutoff": int(
                (
                    pd.to_datetime(fit_log["train_signal_end"])
                    > pd.to_datetime(fit_log["train_label_cutoff"])
                ).sum()
            ),
            "duplicate_model_security_month_predictions": int(
                predictions.duplicated(["model", "date", "ric"]).sum()
            ),
        },
    }
    (output_dir / "fundamental_mispricing_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
