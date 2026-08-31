"""Leakage-safe ML benchmarks for the compact European asset-pricing panel."""
from __future__ import annotations

import copy
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from asset_pricing import LIQUIDITY_EXTENSION_FEATURES, RAW_FEATURES
from compustat_features import COMPUSTAT_EXTENSION_FEATURES
from estimates_features import (
    ESTIMATES_FEATURE_FAMILIES,
    ESTIMATES_INFORMATION_TYPES,
    ESTIMATES_MISSINGNESS_FEATURES,
    ESTIMATES_MODEL_FEATURES,
)
import stats as project_stats


PPY = 12
LEVEL_METRIC_BOOTSTRAP_BLOCK = 6
LEVEL_METRIC_BOOTSTRAP_REPETITIONS = 10_000
LEVEL_METRIC_BOOTSTRAP_SEED = 42
FEATURE_COLUMNS = [f"{feature}_rank" for feature in RAW_FEATURES]
EXPANDED_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    *(f"{feature}_rank" for feature in LIQUIDITY_EXTENSION_FEATURES),
]
COMPUSTAT_FEATURE_COLUMNS = [
    *EXPANDED_FEATURE_COLUMNS,
    *(f"{feature}_rank" for feature in COMPUSTAT_EXTENSION_FEATURES),
]
ESTIMATES_FEATURE_COLUMNS = [
    *COMPUSTAT_FEATURE_COLUMNS,
    *(f"{feature}_rank" for feature in ESTIMATES_MODEL_FEATURES),
]
ESTIMATES_MISSINGNESS_FEATURE_COLUMNS = [
    f"{feature}_rank" for feature in ESTIMATES_MISSINGNESS_FEATURES
]
ESTIMATES_COVERAGE_FEATURE_COLUMNS = [
    *ESTIMATES_FEATURE_COLUMNS,
    *ESTIMATES_MISSINGNESS_FEATURE_COLUMNS,
]
ESTIMATES_REVISION_FEATURE_COLUMNS = [
    f"{feature}_rank" for feature in ESTIMATES_INFORMATION_TYPES["revisions"]
]
FEATURE_SETS = {
    "baseline": FEATURE_COLUMNS,
    "expanded_liquidity": EXPANDED_FEATURE_COLUMNS,
    "compustat_enriched": COMPUSTAT_FEATURE_COLUMNS,
    "estimates_enriched": ESTIMATES_FEATURE_COLUMNS,
    "estimates_enriched_with_coverage": ESTIMATES_COVERAGE_FEATURE_COLUMNS,
    "estimates_coverage_only": [
        *COMPUSTAT_FEATURE_COLUMNS,
        *ESTIMATES_MISSINGNESS_FEATURE_COLUMNS,
    ],
    "estimates_revisions_pure": ESTIMATES_REVISION_FEATURE_COLUMNS,
}
# Estimates-decomposition ablation sets. "X_only" adds one informative
# estimates group on top of the Compustat baseline; "ex_X" removes one source
# family from the full estimates set. Known-degenerate recommendation and
# analyst-count ranks are generated for audits but excluded from model feature
# sets; the coverage-count signal has its own explicit robustness feature set.
for _group_name, _group_features in {
    **ESTIMATES_FEATURE_FAMILIES,
    **ESTIMATES_INFORMATION_TYPES,
}.items():
    FEATURE_SETS[f"estimates_{_group_name}_only"] = [
        *COMPUSTAT_FEATURE_COLUMNS,
        *(f"{feature}_rank" for feature in _group_features),
    ]
for _group_name, _group_features in {
    **ESTIMATES_FEATURE_FAMILIES,
    **ESTIMATES_INFORMATION_TYPES,
}.items():
    _group_ranks = {f"{feature}_rank" for feature in _group_features}
    FEATURE_SETS[f"estimates_ex_{_group_name}"] = [
        column
        for column in ESTIMATES_FEATURE_COLUMNS
        if column not in _group_ranks
    ]
FEATURE_THEMES = {
    "price_trends": [
        "return_1m_rank",
        "momentum_6_2_rank",
        "momentum_12_2_rank",
        "max_return_12m_rank",
    ],
    "liquidity": [
        "turnover_1m_rank",
        "turnover_12m_rank",
        "log_trading_value_eur_rank",
        "turnover_volatility_12m_rank",
    ],
    "risk": ["volatility_12m_rank"],
    "size": ["log_size_rank", "market_cap_growth_12m_rank"],
    "fundamentals": [
        "book_to_market_rank",
        "asset_growth_rank",
        "sales_growth_rank",
        "profitability_roa_rank",
        "operating_profitability_rank",
        "leverage_rank",
        "accruals_rank",
        "capex_to_assets_rank",
        "cashflow_to_assets_rank",
    ],
    "analyst_estimates": [
        f"{feature}_rank" for feature in ESTIMATES_MODEL_FEATURES
    ],
    "analyst_coverage": [
        *ESTIMATES_MISSINGNESS_FEATURE_COLUMNS,
    ],
}
RESIDUAL_CONTROL_COLUMNS = [
    "log_size_rank",
    "book_to_market_rank",
    "momentum_12_2_rank",
    "volatility_12m_rank",
    "return_1m_rank",
]
RESIDUAL_CATEGORICAL_COLUMNS = ["screen_country", "TR.TRBCECONOMICSECTOR"]
# Named neutralization designs. A residual target that controls for a column is
# blind to any baseline whose score IS that column -- the baseline would be
# regressed against itself and score a residual IC of ~0 by construction. Use
# "country_sector" for a clean country/sector-neutral read that keeps every
# characteristic baseline (momentum included) a valid comparator, and
# "styles_ex_momentum" to strip the other styles while preserving momentum.
RESIDUAL_CONTROL_SETS = {
    "full": RESIDUAL_CONTROL_COLUMNS,
    "styles_ex_momentum": [
        column
        for column in RESIDUAL_CONTROL_COLUMNS
        if column != "momentum_12_2_rank"
    ],
    "country_sector": [],
}
# Baselines that score a single raw characteristic rather than fitting one.
MODEL_PREDICTOR_COLUMNS = {"momentum": "momentum_12_2_rank"}
RESIDUAL_TARGET_COLUMNS = [
    "target_return_residual_1m",
    "target_rank_residual",
    "target_residual_rank",
]
TARGET_COLUMN_BY_MODE = {
    "rank": "target_return_rank",
    "return": "target_return_1m",
    "residual_rank": "target_residual_rank",
    "residual_return": "target_return_residual_1m",
}
SUPPORTED_MODELS = {
    "zero",
    "momentum",
    "ridge",
    "elastic_net",
    "hist_gbm",
    "mlp",
    "dre",
}


def set_reproducible_seed(seed: int) -> None:
    """Seed stochastic libraries used by the benchmark training code."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class WalkForwardConfig:
    first_test_year: int = 2015
    last_test_year: int = 2026
    min_training_rows: int = 10_000
    max_training_rows: int | None = None
    portfolio_quantile: float = 0.10
    cost_grid_bps: tuple[int, ...] = (0, 10, 25, 50)
    random_state: int = 42
    validation_months: int = 24
    tune_hyperparameters: bool = True
    ridge_alpha: float = 10.0
    elastic_net_alpha: float = 0.0001
    elastic_net_l1_ratio: float = 0.5
    hist_learning_rate: float = 0.05
    hist_max_iter: int = 150
    hist_max_leaf_nodes: int = 31
    mlp_hidden_sizes: tuple[int, ...] = (64, 32, 16)
    mlp_dropout: float = 0.10
    mlp_epochs: int = 20
    mlp_batch_size: int = 8192
    mlp_patience: int = 4
    mlp_validation_months: int = 24
    dre_layers: int = 2
    dre_features_per_block: int = 64
    dre_gammas: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    dre_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    dre_final_alpha: float = 10.0
    dre_tune_final_alpha: bool = False
    dre_final_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)


def load_model_panel(
    path: Path,
    delisting_audit_path: Path | None = None,
    feature_columns: list[str] | None = None,
    sample_start_date: str | pd.Timestamp | None = None,
    sample_end_date: str | pd.Timestamp | None = None,
    require_estimates_feature: bool = False,
    require_revision_signal: bool = False,
    require_estimate_signal_lag_months: int | None = None,
    residual_control_set: str = "full",
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    revision_signal_columns = ESTIMATES_INFORMATION_TYPES["revisions"]
    filter_columns = []
    if require_estimates_feature:
        filter_columns.append("estimates_feature_count")
    if require_revision_signal:
        filter_columns.extend(revision_signal_columns)
    if require_estimate_signal_lag_months is not None:
        filter_columns.extend(["estimates_feature_count", "est_signal_lag_months"])
    required_columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "target_return_rank",
        "company_market_cap",
        "market_cap_percentile",
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
        "eligible",
        "model_eligible",
        "return_history_n",
        "feature_count",
        *feature_columns,
        *filter_columns,
        *(extra_columns or []),
    ]
    columns = list(dict.fromkeys(required_columns))
    panel = pd.read_parquet(path, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["target_date"] = pd.to_datetime(panel["target_date"])
    panel["is_delisting_candidate"] = False
    panel["retire_month"] = pd.NaT
    sample_audit = {"loaded_rows": int(len(panel))}

    if sample_start_date is not None:
        start_date = pd.Timestamp(sample_start_date)
        panel = panel[panel["date"].ge(start_date)].copy()
    sample_audit["after_sample_start_date"] = int(len(panel))
    if sample_end_date is not None:
        end_date = pd.Timestamp(sample_end_date)
        panel = panel[panel["date"].le(end_date)].copy()
    sample_audit["after_sample_end_date"] = int(len(panel))
    if require_estimates_feature:
        has_estimates_feature = pd.to_numeric(
            panel["estimates_feature_count"],
            errors="coerce",
        ).gt(0)
        panel = panel[has_estimates_feature].copy()
    sample_audit["after_require_estimates_feature"] = int(len(panel))
    if require_revision_signal:
        has_revision_signal = panel[revision_signal_columns].notna().any(axis=1)
        panel = panel[has_revision_signal].copy()
    sample_audit["after_require_revision_signal"] = int(len(panel))
    if require_estimate_signal_lag_months is not None:
        lag = pd.to_numeric(panel["est_signal_lag_months"], errors="coerce")
        has_estimates_feature = pd.to_numeric(
            panel["estimates_feature_count"],
            errors="coerce",
        ).gt(0)
        invalid_lag = has_estimates_feature & lag.lt(
            require_estimate_signal_lag_months,
        )
        invalid_lag = invalid_lag | (has_estimates_feature & lag.isna())
        sample_audit["estimate_signal_lag_violations"] = int(invalid_lag.sum())
        if invalid_lag.any():
            raise ValueError(
                "Estimate signal lag guard failed: "
                f"{int(invalid_lag.sum()):,} rows with non-null estimates "
                "features have est_signal_lag_months below "
                f"{require_estimate_signal_lag_months}."
            )
    else:
        sample_audit["estimate_signal_lag_violations"] = 0

    labelled = panel["model_eligible"] & panel[
        ["target_date", "target_return_1m", "target_return_rank"]
    ].notna().all(axis=1)
    scoreable_delisting = pd.Series(False, index=panel.index)
    if delisting_audit_path is not None:
        delisting_audit = pd.read_csv(delisting_audit_path, low_memory=False)
        delisting_audit = delisting_audit[
            delisting_audit["missing_retirement_month_return"]
            .fillna(False)
            .astype(bool)
        ].copy()
        delisting_audit["retire_month"] = (
            pd.to_datetime(delisting_audit["retire_month"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp("M")
        )
        delisting_audit["date"] = (
            delisting_audit["retire_month"] - pd.offsets.MonthEnd(1)
        )
        candidate_dates = delisting_audit.set_index(["ric", "date"])[
            "retire_month"
        ]
        panel_index = pd.MultiIndex.from_frame(panel[["ric", "date"]])
        retire_month = candidate_dates.reindex(panel_index).to_numpy()
        candidate = pd.notna(retire_month)
        panel.loc[candidate, "retire_month"] = retire_month[candidate]
        panel.loc[
            candidate & panel["target_date"].isna(), "target_date"
        ] = panel.loc[
            candidate & panel["target_date"].isna(), "retire_month"
        ]
        scoreable_delisting = (
            candidate
            & panel["eligible"].fillna(False)
            & panel["company_market_cap"].gt(0)
            & panel["market_cap_percentile"].ge(0.05)
            & panel["return_history_n"].ge(24)
            & panel["feature_count"].ge(8)
            & panel["target_return_1m"].isna()
            & panel["retire_month"].le(
                panel.loc[labelled, "target_date"].max()
            )
        )
        panel.loc[scoreable_delisting, "is_delisting_candidate"] = True

    panel = panel[labelled | scoreable_delisting].copy()
    panel = panel.dropna(
        subset=["date", "target_date", "ric", *feature_columns]
    )
    panel[feature_columns] = panel[feature_columns].astype("float32")
    panel["target_return_rank"] = panel["target_return_rank"].astype("float32")
    panel = add_residual_targets(panel, control_set=residual_control_set)
    residual_control_columns = list(panel.attrs.get("residual_control_columns", []))
    sample_audit["model_rows"] = int(len(panel))
    result = panel.sort_values(["date", "ric"]).reset_index(drop=True)
    result.attrs["sample_filter_audit"] = sample_audit
    result.attrs["residual_control_set"] = residual_control_set
    result.attrs["residual_control_columns"] = residual_control_columns
    return result


def _month_residuals(
    month: pd.DataFrame,
    target_column: str,
    control_columns: list[str],
    categorical_columns: list[str],
) -> pd.Series:
    labelled = month[target_column].notna()
    if labelled.sum() < 20:
        return month[target_column] - month[target_column].mean()

    y = month.loc[labelled, target_column].astype(float)
    parts: list[pd.DataFrame] = []
    available_controls = [
        column
        for column in control_columns
        if column in month.columns and month.loc[labelled, column].notna().any()
    ]
    if available_controls:
        continuous = (
            month.loc[labelled, available_controls]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        parts.append(continuous)

    available_categories = [
        column
        for column in categorical_columns
        if column in month.columns and month.loc[labelled, column].notna().any()
    ]
    if available_categories:
        categories = month.loc[labelled, available_categories].fillna("Unknown")
        categories = categories.astype(str)
        dummies = pd.get_dummies(
            categories,
            columns=available_categories,
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            parts.append(dummies)

    if parts:
        x_frame = pd.concat(parts, axis=1)
        x = np.column_stack(
            [np.ones(len(x_frame), dtype=float), x_frame.to_numpy(dtype=float)]
        )
        if x.shape[0] > x.shape[1] + 5:
            try:
                beta = np.linalg.lstsq(x, y.to_numpy(dtype=float), rcond=None)[0]
                fitted = x @ beta
                residual = y.to_numpy(dtype=float) - fitted
            except np.linalg.LinAlgError:
                residual = y.to_numpy(dtype=float) - float(y.mean())
        else:
            residual = y.to_numpy(dtype=float) - float(y.mean())
    else:
        residual = y.to_numpy(dtype=float) - float(y.mean())

    result = pd.Series(np.nan, index=month.index, dtype="float64")
    result.loc[y.index] = residual
    return result


def _monthly_residual_series(
    panel: pd.DataFrame,
    target_column: str,
    control_columns: list[str],
    categorical_columns: list[str],
) -> pd.Series:
    pieces = [
        _month_residuals(
            month,
            target_column,
            control_columns,
            categorical_columns,
        )
        for _, month in panel.groupby("date", sort=False)
    ]
    if not pieces:
        return pd.Series(np.nan, index=panel.index, dtype="float64")
    return pd.concat(pieces).reindex(panel.index)


def add_residual_targets(
    panel: pd.DataFrame,
    control_set: str = "full",
) -> pd.DataFrame:
    """Add neutralized targets for Europe-specific ML experiments.

    ``control_set`` selects a design from ``RESIDUAL_CONTROL_SETS``. Country and
    sector are always removed; the numeric style controls vary. The realised
    portfolio return remains the raw next-month return.

    Any characteristic baseline whose score appears in the chosen control set is
    invalidated on the resulting targets -- see
    ``stats.residual_target_model_eligible``.
    """
    if control_set not in RESIDUAL_CONTROL_SETS:
        raise ValueError(
            f"Unknown residual control set {control_set!r}; "
            f"expected one of {sorted(RESIDUAL_CONTROL_SETS)}"
        )
    panel = panel.copy()
    control_columns = [
        column
        for column in RESIDUAL_CONTROL_SETS[control_set]
        if column in panel.columns
    ]
    categorical_columns = [
        column
        for column in RESIDUAL_CATEGORICAL_COLUMNS
        if column in panel.columns
    ]
    panel["target_return_residual_1m"] = _monthly_residual_series(
        panel,
        target_column="target_return_1m",
        control_columns=control_columns,
        categorical_columns=categorical_columns,
    ).astype("float32")
    panel["target_rank_residual"] = _monthly_residual_series(
        panel,
        target_column="target_return_rank",
        control_columns=control_columns,
        categorical_columns=categorical_columns,
    ).astype("float32")
    residual_rank = panel.groupby("date")["target_rank_residual"].rank(
        method="average",
        pct=True,
    )
    panel["target_residual_rank"] = (2.0 * residual_rank - 1.0).astype("float32")
    panel.attrs["residual_control_set"] = control_set
    panel.attrs["residual_control_columns"] = list(control_columns)
    return panel


def walk_forward_slices(
    panel: pd.DataFrame,
    first_test_year: int,
    last_test_year: int,
) -> list[tuple[int, pd.Timestamp, np.ndarray, np.ndarray]]:
    slices = []
    for year in range(first_test_year, last_test_year + 1):
        cutoff = pd.Timestamp(year=year - 1, month=12, day=31)
        train_mask = panel["target_date"].le(cutoff).to_numpy()
        test_mask = panel["date"].dt.year.eq(year).to_numpy()
        if test_mask.any():
            slices.append((year, cutoff, train_mask, test_mask))
    return slices


def _limit_training_rows(
    train: pd.DataFrame,
    maximum: int | None,
    random_state: int,
) -> pd.DataFrame:
    if maximum is None or len(train) <= maximum:
        return train
    # Stratified monthly sampling preserves every historical period.
    fraction = maximum / len(train)
    parts = []
    for month_number, (_, group) in enumerate(train.groupby("date", sort=True)):
        sample_size = max(1, min(len(group), round(len(group) * fraction)))
        parts.append(
            group.sample(
                n=sample_size,
                random_state=random_state + month_number,
            )
        )
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > maximum:
        sampled = sampled.sample(n=maximum, random_state=random_state)
    return sampled.sort_values(["date", "ric"])


def _fit_sklearn_model(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    config: WalkForwardConfig,
    parameters: dict[str, Any] | None = None,
):
    parameters = parameters or {}
    if model_name == "ridge":
        model = Ridge(alpha=parameters.get("alpha", config.ridge_alpha))
    elif model_name == "elastic_net":
        model = ElasticNet(
            alpha=parameters.get("alpha", config.elastic_net_alpha),
            l1_ratio=parameters.get("l1_ratio", config.elastic_net_l1_ratio),
            max_iter=5_000,
            selection="cyclic",
            random_state=config.random_state,
        )
    elif model_name == "hist_gbm":
        model = HistGradientBoostingRegressor(
            learning_rate=parameters.get("learning_rate", config.hist_learning_rate),
            max_iter=parameters.get("max_iter", config.hist_max_iter),
            max_leaf_nodes=parameters.get("max_leaf_nodes", config.hist_max_leaf_nodes),
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=config.random_state,
        )
    else:
        raise ValueError(f"Unsupported sklearn model: {model_name}")
    return model.fit(x, y)


class TorchMLPRegressor:
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.model: Any = None
        self.best_epoch: int | None = None
        self.validation_loss: float | None = None
        self.validation_start: pd.Timestamp | None = None
        self.validation_end: pd.Timestamp | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dates: pd.Series,
    ) -> "TorchMLPRegressor":
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        set_reproducible_seed(self.config.random_state)
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

        unique_months = np.sort(dates.dropna().unique())
        validation_months = min(
            self.config.mlp_validation_months,
            max(1, len(unique_months) // 5),
        )
        validation_start = pd.Timestamp(unique_months[-validation_months])
        validation_mask = dates.ge(validation_start).to_numpy()
        if validation_mask.sum() == 0 or (~validation_mask).sum() == 0:
            validation_mask = np.zeros(len(dates), dtype=bool)
            validation_mask[-max(1, len(dates) // 10) :] = True
        validation_dates = pd.to_datetime(dates.loc[validation_mask])
        self.validation_start = validation_dates.min()
        self.validation_end = validation_dates.max()

        x_train = torch.from_numpy(x[~validation_mask].astype("float32", copy=False))
        y_train = torch.from_numpy(y[~validation_mask].astype("float32", copy=False)).reshape(-1, 1)
        x_valid = torch.from_numpy(x[validation_mask].astype("float32", copy=False))
        y_valid = torch.from_numpy(y[validation_mask].astype("float32", copy=False)).reshape(-1, 1)

        layers: list[nn.Module] = []
        input_size = x.shape[1]
        for hidden_size in self.config.mlp_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(self.config.mlp_dropout),
                ]
            )
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        model = nn.Sequential(*layers)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_function = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.config.random_state)
        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.config.mlp_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        best_state = copy.deepcopy(model.state_dict())
        best_loss = math.inf
        stale_epochs = 0
        for epoch in range(1, self.config.mlp_epochs + 1):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(model(x_valid), y_valid).item())
            if validation_loss < best_loss - 1e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                self.best_epoch = epoch
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.mlp_patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.validation_loss = best_loss
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self.model is None:
            raise RuntimeError("MLP has not been fitted")
        values = torch.from_numpy(x.astype("float32", copy=False))
        predictions = []
        with torch.no_grad():
            for start in range(0, len(values), self.config.mlp_batch_size):
                predictions.append(
                    self.model(values[start : start + self.config.mlp_batch_size])
                    .reshape(-1)
                    .numpy()
                )
        return np.concatenate(predictions)


class ClosedFormRidgeRegressor:
    """Small dense ridge regression used inside DRE random-feature blocks."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ClosedFormRidgeRegressor":
        x = x.astype("float64", copy=False)
        y = y.astype("float64", copy=False)
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        x_centered = x - x_mean
        y_centered = y - y_mean
        system = x_centered.T @ x_centered
        system.flat[:: system.shape[0] + 1] += self.alpha
        rhs = x_centered.T @ y_centered
        try:
            coef = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(system, rhs, rcond=None)[0]
        self.coef_ = coef.astype("float32")
        self.intercept_ = float(y_mean - x_mean @ coef)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("ClosedFormRidgeRegressor has not been fitted")
        return x @ self.coef_ + self.intercept_


class DeepRegressionEnsembleRegressor:
    """Deep regression ensemble using random-feature ridge submodels.

    The implementation follows the Didisheim-Kelly-Malamud architecture at the
    level needed here: each layer draws several random-feature blocks, fits a
    grid of myopic ridge regressions, stacks their predictions, then passes that
    ensemble representation to the next layer. A final ridge regression combines
    the last layer's predictions.
    """

    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.layers: list[dict[str, Any]] = []
        self.final_model: ClosedFormRidgeRegressor | None = None
        self.output_mean: np.ndarray | None = None
        self.output_scale: np.ndarray | None = None
        self.training_layers: int | None = None
        self.training_ensemble_width: int | None = None
        self.selected_final_alpha: float | None = None
        self.final_alpha_validation_loss: float | None = None

    @staticmethod
    def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
        return (values - mean) / scale, mean, scale

    @staticmethod
    def _standardize_apply(
        values: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        return (values - mean) / scale

    @staticmethod
    def _activation(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0.0)

    def _fit_layer(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        x_scaled, x_mean, x_scale = self._standardize_fit(x)
        input_dim = x_scaled.shape[1]
        blocks = []
        layer_predictions = []
        for gamma in self.config.dre_gammas:
            weight_scale = math.sqrt(gamma / max(1, input_dim))
            weights = rng.normal(
                loc=0.0,
                scale=weight_scale,
                size=(input_dim, self.config.dre_features_per_block),
            ).astype("float32")
            bias = rng.uniform(
                0.0,
                2.0 * math.pi,
                size=self.config.dre_features_per_block,
            ).astype("float32")
            random_features = self._activation(x_scaled @ weights + bias)
            features_scaled, features_mean, features_scale = self._standardize_fit(
                random_features
            )
            models = []
            for alpha in self.config.dre_alphas:
                ridge = ClosedFormRidgeRegressor(alpha=alpha)
                ridge.fit(features_scaled, y)
                models.append(ridge)
                layer_predictions.append(ridge.predict(features_scaled))
            blocks.append(
                {
                    "gamma": float(gamma),
                    "weights": weights,
                    "bias": bias,
                    "feature_mean": features_mean,
                    "feature_scale": features_scale,
                    "models": models,
                }
            )
        output = np.column_stack(layer_predictions).astype("float32", copy=False)
        output_scaled, output_mean, output_scale = self._standardize_fit(output)
        layer = {
            "input_mean": x_mean,
            "input_scale": x_scale,
            "output_mean": output_mean,
            "output_scale": output_scale,
            "blocks": blocks,
        }
        return output_scaled.astype("float32", copy=False), layer

    def _predict_layer(self, x: np.ndarray, layer: dict[str, Any]) -> np.ndarray:
        x_scaled = self._standardize_apply(
            x,
            layer["input_mean"],
            layer["input_scale"],
        )
        layer_predictions = []
        for block in layer["blocks"]:
            random_features = self._activation(
                x_scaled @ block["weights"] + block["bias"]
            )
            features_scaled = self._standardize_apply(
                random_features,
                block["feature_mean"],
                block["feature_scale"],
            )
            for model in block["models"]:
                layer_predictions.append(model.predict(features_scaled))
        output = np.column_stack(layer_predictions).astype("float32", copy=False)
        return self._standardize_apply(
            output,
            layer["output_mean"],
            layer["output_scale"],
        ).astype("float32", copy=False)

    def _fit_layers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        current = x.astype("float32", copy=False)
        layers = []
        for _ in range(self.config.dre_layers):
            current, layer = self._fit_layer(current, y, rng)
            layers.append(layer)
        return current, layers

    def _predict_layers(
        self,
        x: np.ndarray,
        layers: list[dict[str, Any]],
    ) -> np.ndarray:
        current = x.astype("float32", copy=False)
        for layer in layers:
            current = self._predict_layer(current, layer)
        return current

    def _select_final_alpha(
        self,
        x_core: np.ndarray,
        y_core: np.ndarray,
        x_validation: np.ndarray,
        y_validation: np.ndarray,
    ) -> tuple[float, float]:
        candidates = (
            self.config.dre_final_alphas
            if self.config.dre_final_alphas
            else (self.config.dre_final_alpha,)
        )
        rng = np.random.default_rng(self.config.random_state)
        core_current, layers = self._fit_layers(x_core, y_core, rng)
        validation_current = self._predict_layers(x_validation, layers)
        core_final_x, output_mean, output_scale = self._standardize_fit(core_current)
        validation_final_x = self._standardize_apply(
            validation_current,
            output_mean,
            output_scale,
        )
        scored = []
        for alpha in candidates:
            ridge = ClosedFormRidgeRegressor(alpha=alpha).fit(core_final_x, y_core)
            prediction = ridge.predict(validation_final_x)
            loss = float(np.mean(np.square(y_validation - prediction)))
            scored.append((loss, float(alpha)))
        validation_loss, selected_alpha = min(scored, key=lambda item: item[0])
        return selected_alpha, validation_loss

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> "DeepRegressionEnsembleRegressor":
        selected_alpha = float(self.config.dre_final_alpha)
        validation_loss = np.nan
        if self.config.dre_tune_final_alpha and validation_data is not None:
            x_core, y_core, x_validation, y_validation = validation_data
            if len(x_core) and len(x_validation):
                selected_alpha, validation_loss = self._select_final_alpha(
                    x_core.astype("float32", copy=False),
                    y_core.astype("float32", copy=False),
                    x_validation.astype("float32", copy=False),
                    y_validation.astype("float32", copy=False),
                )
        rng = np.random.default_rng(self.config.random_state)
        y = y.astype("float32", copy=False)
        current, self.layers = self._fit_layers(x, y, rng)
        final_x, self.output_mean, self.output_scale = self._standardize_fit(current)
        self.final_model = ClosedFormRidgeRegressor(
            alpha=selected_alpha,
        ).fit(final_x, y)
        self.training_layers = len(self.layers)
        self.training_ensemble_width = len(self.config.dre_gammas) * len(
            self.config.dre_alphas
        )
        self.selected_final_alpha = selected_alpha
        self.final_alpha_validation_loss = float(validation_loss)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if (
            self.final_model is None
            or self.output_mean is None
            or self.output_scale is None
        ):
            raise RuntimeError("DRE has not been fitted")
        current = x.astype("float32", copy=False)
        for layer in self.layers:
            current = self._predict_layer(current, layer)
        final_x = self._standardize_apply(
            current,
            self.output_mean,
            self.output_scale,
        )
        return self.final_model.predict(final_x)


def _parameter_grid(model_name: str, config: WalkForwardConfig) -> list[dict[str, Any]]:
    if model_name == "ridge":
        return [{"alpha": value} for value in (0.1, 1.0, 10.0, 100.0)]
    if model_name == "elastic_net":
        return [
            {"alpha": alpha, "l1_ratio": l1_ratio}
            for alpha, l1_ratio in [
                (0.00001, 0.1),
                (0.0001, 0.5),
                (0.001, 0.5),
                (0.001, 0.9),
                (0.01, 0.5),
                (0.01, 0.9),
            ]
        ]
    if model_name == "hist_gbm":
        return [
            {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": config.hist_max_iter},
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": config.hist_max_iter},
            {"learning_rate": 0.03, "max_leaf_nodes": 63, "max_iter": config.hist_max_iter},
        ]
    return [{}]


def _validation_split(
    train: pd.DataFrame,
    validation_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = np.sort(train["date"].dropna().unique())
    if len(months) < validation_months + 12:
        split = max(1, len(months) // 5)
    else:
        split = validation_months
    validation_start = pd.Timestamp(months[-split])
    core = train[train["date"].lt(validation_start)]
    validation = train[train["date"].ge(validation_start)]
    return core, validation


def _fit_model(
    model_name: str,
    train: pd.DataFrame,
    config: WalkForwardConfig,
    target_column: str,
    feature_columns: list[str],
) -> tuple[Any, dict[str, Any]]:
    x = train[feature_columns].to_numpy(dtype="float32", copy=False)
    y = train[target_column].to_numpy(dtype="float32", copy=False)
    if model_name == "mlp":
        model = TorchMLPRegressor(config).fit(x, y, train["date"])
        return model, {
            "selected_parameters": json.dumps(
                {
                    "hidden_sizes": config.mlp_hidden_sizes,
                    "dropout": config.mlp_dropout,
                    "best_epoch": model.best_epoch,
                }
            ),
            "validation_loss": model.validation_loss,
            "validation_start": (
                str(model.validation_start.date())
                if model.validation_start is not None
                else np.nan
            ),
            "validation_end": (
                str(model.validation_end.date())
                if model.validation_end is not None
                else np.nan
            ),
        }
    if model_name == "dre":
        validation_data = None
        validation_start = None
        validation_end = None
        if config.tune_hyperparameters and config.dre_tune_final_alpha:
            core, validation = _validation_split(train, config.validation_months)
            validation_data = (
                core[feature_columns].to_numpy(dtype="float32", copy=False),
                core[target_column].to_numpy(dtype="float32", copy=False),
                validation[feature_columns].to_numpy(dtype="float32", copy=False),
                validation[target_column].to_numpy(dtype="float32", copy=False),
            )
            validation_start = str(validation["date"].min().date())
            validation_end = str(validation["date"].max().date())
        model = DeepRegressionEnsembleRegressor(config).fit(
            x,
            y,
            validation_data=validation_data,
        )
        selected_parameters = {
            "layers": config.dre_layers,
            "features_per_block": config.dre_features_per_block,
            "gammas": config.dre_gammas,
            "alphas": config.dre_alphas,
            "final_alpha": model.selected_final_alpha,
            "final_alpha_grid": config.dre_final_alphas,
            "final_alpha_tuned": bool(
                config.tune_hyperparameters and config.dre_tune_final_alpha
            ),
        }
        metadata = {
            "selected_parameters": json.dumps(selected_parameters, sort_keys=True),
            "validation_loss": model.final_alpha_validation_loss,
            "selected_final_alpha": model.selected_final_alpha,
            "final_alpha_validation_loss": model.final_alpha_validation_loss,
        }
        if validation_start is not None and validation_end is not None:
            metadata["validation_start"] = validation_start
            metadata["validation_end"] = validation_end
        else:
            metadata["validation_loss"] = np.nan
            metadata["final_alpha_validation_loss"] = np.nan
        return model, {
            **metadata,
        }

    candidates = _parameter_grid(model_name, config)
    if not config.tune_hyperparameters or len(candidates) == 1:
        selected = candidates[0]
        return (
            _fit_sklearn_model(model_name, x, y, config, selected),
            {"selected_parameters": json.dumps(selected), "validation_loss": np.nan},
        )

    core, validation = _validation_split(train, config.validation_months)
    x_core = core[feature_columns].to_numpy(dtype="float32", copy=False)
    y_core = core[target_column].to_numpy(dtype="float32", copy=False)
    x_validation = validation[feature_columns].to_numpy(dtype="float32", copy=False)
    y_validation = validation[target_column].to_numpy(dtype="float32", copy=False)
    scored = []
    for parameters in candidates:
        candidate = _fit_sklearn_model(
            model_name, x_core, y_core, config, parameters
        )
        prediction = candidate.predict(x_validation)
        loss = float(np.mean(np.square(y_validation - prediction)))
        scored.append((loss, parameters))
    validation_loss, selected = min(scored, key=lambda item: item[0])
    model = _fit_sklearn_model(model_name, x, y, config, selected)
    return model, {
        "selected_parameters": json.dumps(selected, sort_keys=True),
        "validation_loss": validation_loss,
        "validation_start": str(validation["date"].min().date()),
        "validation_end": str(validation["date"].max().date()),
    }


def _model_metadata(model_name: str, model: Any) -> dict:
    metadata: dict[str, Any] = {}
    if model_name in {"ridge", "elastic_net"}:
        metadata["intercept"] = float(model.intercept_)
        metadata["nonzero_coefficients"] = int(np.count_nonzero(model.coef_))
    elif model_name == "hist_gbm":
        metadata["iterations"] = int(model.n_iter_)
    elif model_name == "mlp":
        metadata["best_epoch"] = model.best_epoch
        metadata["validation_loss"] = model.validation_loss
    elif model_name == "dre":
        metadata["layers"] = model.training_layers
        metadata["ensemble_width"] = model.training_ensemble_width
        metadata["random_features_per_block"] = model.config.dre_features_per_block
        metadata["final_alpha"] = model.selected_final_alpha
    return metadata


def _importance_subgroups(test: pd.DataFrame) -> dict[str, np.ndarray]:
    percentile = test["market_cap_percentile"].to_numpy(dtype=float)
    return {
        "all": np.ones(len(test), dtype=bool),
        "small": percentile <= 1.0 / 3.0,
        "middle": (percentile > 1.0 / 3.0) & (percentile <= 2.0 / 3.0),
        "large": percentile > 2.0 / 3.0,
    }


def _variable_importance_records(
    model: Any,
    model_label: str,
    target_mode: str,
    test_year: int,
    test: pd.DataFrame,
    x_test: np.ndarray,
    baseline_prediction: np.ndarray,
    target_column: str,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    """Fixed-model OOS ablation; zero is the monthly cross-sectional median."""
    labelled = test[target_column].notna().to_numpy()
    ablations: list[tuple[str, str, list[int]]] = []
    for index, feature in enumerate(feature_columns):
        ablations.append(("feature", feature.removesuffix("_rank"), [index]))
    for theme, members in FEATURE_THEMES.items():
        indices = [
            feature_columns.index(feature)
            for feature in members
            if feature in feature_columns
        ]
        if indices:
            ablations.append(("theme", theme, indices))

    altered_predictions: dict[tuple[str, str], np.ndarray] = {}
    for level, name, indices in ablations:
        altered = x_test.copy()
        altered[:, indices] = 0.0
        altered_predictions[(level, name)] = model.predict(altered).astype(
            "float32", copy=False
        )

    records: list[dict[str, Any]] = []
    for subgroup, subgroup_mask in _importance_subgroups(test).items():
        evaluation = labelled & subgroup_mask
        if evaluation.sum() < 30:
            continue
        actual = test.loc[evaluation, target_column].to_numpy(dtype=float)
        baseline = baseline_prediction[evaluation].astype(float)
        denominator = float(np.square(actual).sum())
        baseline_sse = float(np.square(actual - baseline).sum())
        baseline_mse = baseline_sse / len(actual)
        baseline_ic = scipy_stats.spearmanr(
            baseline,
            test.loc[evaluation, "target_return_rank"].to_numpy(dtype=float),
        ).statistic
        for level, name, _ in ablations:
            altered = altered_predictions[(level, name)][evaluation].astype(float)
            altered_sse = float(np.square(actual - altered).sum())
            altered_ic = scipy_stats.spearmanr(
                altered,
                test.loc[evaluation, "target_return_rank"].to_numpy(dtype=float),
            ).statistic
            records.append(
                {
                    "model": model_label,
                    "target_mode": target_mode,
                    "test_year": test_year,
                    "market_cap_group": subgroup,
                    "ablation_level": level,
                    "variable": name,
                    "observations": int(evaluation.sum()),
                    "baseline_mse": baseline_mse,
                    "ablated_mse": altered_sse / len(actual),
                    "mse_increase": (altered_sse - baseline_sse) / len(actual),
                    "delta_r2_zero": (
                        (altered_sse - baseline_sse) / denominator
                        if denominator > 0
                        else np.nan
                    ),
                    "baseline_spearman_ic": baseline_ic,
                    "ablated_spearman_ic": altered_ic,
                    "spearman_ic_decrease": baseline_ic - altered_ic,
                }
            )
    return records


def normalize_variable_importance(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return importance
    result = importance.copy()
    family = [
        "model",
        "target_mode",
        "test_year",
        "market_cap_group",
        "ablation_level",
    ]
    positive = result["delta_r2_zero"].clip(lower=0.0)
    denominator = positive.groupby(
        [result[column] for column in family], sort=False
    ).transform("sum")
    result["normalized_positive_importance"] = positive.div(
        denominator.where(denominator.gt(0))
    )
    return result


def _safe_annualized_sharpe(returns: pd.Series) -> float:
    standard_deviation = returns.std(ddof=1)
    if standard_deviation <= 0 or pd.isna(standard_deviation):
        return np.nan
    return float(returns.mean() / standard_deviation * np.sqrt(PPY))


def _stationary_bootstrap_columns(
    values: pd.Series,
    *,
    metric: str,
    prefix: str,
    seed: int,
    expected_block: int = LEVEL_METRIC_BOOTSTRAP_BLOCK,
    n_boot: int = LEVEL_METRIC_BOOTSTRAP_REPETITIONS,
) -> dict[str, float]:
    result = project_stats.stationary_bootstrap_metric_ci(
        values,
        metric=metric,
        expected_block=expected_block,
        n_boot=n_boot,
        seed=seed,
        ppy=PPY,
    )
    return {
        f"{prefix}_ci_low": result["ci_low"],
        f"{prefix}_ci_high": result["ci_high"],
        f"{prefix}_p_two_sided_zero": result["p_two_sided_zero"],
        f"{prefix}_bootstrap_observations": result["observations"],
    }


def _monthly_cross_sectional_r2_zero(
    group: pd.DataFrame,
    target_column: str,
) -> pd.Series:
    def month_r2(month: pd.DataFrame) -> float:
        actual = month[target_column].to_numpy(dtype=float)
        predicted = month["prediction"].to_numpy(dtype=float)
        denominator = float(np.square(actual).sum())
        if denominator <= 0 or not np.isfinite(denominator):
            return np.nan
        return 1.0 - float(np.square(actual - predicted).sum()) / denominator

    return group.groupby("date").apply(month_r2, include_groups=False)


def _safe_spearman(prediction: pd.Series, actual: pd.Series) -> float:
    if prediction.nunique(dropna=True) < 2 or actual.nunique(dropna=True) < 2:
        return np.nan
    return float(prediction.corr(actual, method="spearman"))


def _training_fit_metrics(
    train: pd.DataFrame,
    prediction: np.ndarray,
    target_column: str,
) -> dict[str, float | int]:
    actual = train[target_column].to_numpy(dtype=float)
    predicted = prediction.astype(float)
    residual = actual - predicted
    sse = float(np.square(residual).sum())
    denominator = float(np.square(actual).sum())
    monthly = pd.DataFrame(
        {
            "date": train["date"].to_numpy(),
            "prediction": predicted,
            "target": actual,
            "target_return_rank": train["target_return_rank"].to_numpy(
                dtype=float,
            ),
        }
    )
    target_ic = monthly.groupby("date").apply(
        lambda month: _safe_spearman(month["prediction"], month["target"]),
        include_groups=False,
    )
    rank_ic = monthly.groupby("date").apply(
        lambda month: _safe_spearman(
            month["prediction"],
            month["target_return_rank"],
        ),
        include_groups=False,
    )
    return {
        "in_sample_observations": int(len(train)),
        "in_sample_mse": sse / len(train) if len(train) else np.nan,
        "in_sample_sse": sse,
        "in_sample_zero_benchmark_sse": denominator,
        "in_sample_r2_zero": (
            1.0 - sse / denominator if denominator > 0 else np.nan
        ),
        "in_sample_mean_monthly_target_spearman_ic": float(target_ic.mean()),
        "in_sample_mean_monthly_spearman_ic": float(rank_ic.mean()),
    }


def run_walk_forward(
    panel: pd.DataFrame,
    model_names: list[str],
    config: WalkForwardConfig,
    target_column: str = "target_return_rank",
    target_mode: str = "rank",
    feature_columns: list[str] | None = None,
    collect_importance: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = feature_columns or FEATURE_COLUMNS
    panel = panel.copy()
    if "is_delisting_candidate" not in panel:
        panel["is_delisting_candidate"] = False
    if "retire_month" not in panel:
        panel["retire_month"] = pd.NaT
    unknown = set(model_names) - SUPPORTED_MODELS
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    predictions = []
    fit_records = []
    coefficient_records = []
    importance_records = []
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
    base_columns.extend(
        column for column in RESIDUAL_TARGET_COLUMNS if column in panel.columns
    )

    for year, cutoff, train_mask, test_mask in walk_forward_slices(
        panel, config.first_test_year, config.last_test_year
    ):
        full_train = panel.loc[train_mask & panel[target_column].notna()]
        test = panel.loc[test_mask]
        if len(full_train) < config.min_training_rows:
            continue
        train = _limit_training_rows(
            full_train,
            config.max_training_rows,
            config.random_state + year,
        )
        x_train = train[feature_columns].to_numpy(dtype="float32", copy=False)
        x_test = test[feature_columns].to_numpy(dtype="float32", copy=False)

        for model_name in model_names:
            started = time.perf_counter()
            model = None
            if model_name == "zero":
                scores = np.zeros(len(test), dtype="float32")
                train_scores = np.zeros(len(train), dtype="float32")
                tuning_metadata = {}
            elif model_name == "momentum":
                scores = test["momentum_12_2_rank"].to_numpy(dtype="float32")
                train_scores = train["momentum_12_2_rank"].to_numpy(
                    dtype="float32",
                )
                tuning_metadata = {}
            else:
                model, tuning_metadata = _fit_model(
                    model_name,
                    train,
                    config,
                    target_column,
                    feature_columns,
                )
                scores = model.predict(x_test).astype("float32", copy=False)
                train_scores = model.predict(x_train).astype(
                    "float32",
                    copy=False,
                )
                if collect_importance:
                    importance_records.extend(
                        _variable_importance_records(
                            model,
                            f"{model_name}_{target_mode}",
                            target_mode,
                            year,
                            test,
                            x_test,
                            scores,
                            target_column,
                            feature_columns,
                        )
                    )

            elapsed = time.perf_counter() - started
            output = test[base_columns].copy()
            output["prediction"] = scores
            output["model"] = f"{model_name}_{target_mode}"
            output["base_model"] = model_name
            output["target_mode"] = target_mode
            output["test_year"] = year
            output["train_label_cutoff"] = cutoff
            predictions.append(output)

            record = {
                "model": f"{model_name}_{target_mode}",
                "base_model": model_name,
                "target_mode": target_mode,
                "test_year": year,
                "train_rows_available": int(len(full_train)),
                "train_rows_used": int(len(train)),
                "test_rows": int(len(test)),
                "train_signal_start": str(train["date"].min().date()),
                "train_signal_end": str(train["date"].max().date()),
                "train_target_end": str(train["target_date"].max().date()),
                "train_label_cutoff": str(cutoff.date()),
                "fit_seconds": elapsed,
                **_training_fit_metrics(train, train_scores, target_column),
                **tuning_metadata,
                **(_model_metadata(model_name, model) if model is not None else {}),
            }
            fit_records.append(record)

            if model_name in {"ridge", "elastic_net"}:
                for feature, coefficient in zip(feature_columns, model.coef_, strict=True):
                    coefficient_records.append(
                        {
                            "model": model_name,
                            "model_label": f"{model_name}_{target_mode}",
                            "target_mode": target_mode,
                            "test_year": year,
                            "feature": feature,
                            "coefficient": float(coefficient),
                        }
                    )

    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return (
        prediction_frame,
        pd.DataFrame(fit_records),
        pd.DataFrame(coefficient_records),
        normalize_variable_importance(pd.DataFrame(importance_records)),
    )


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for model_name, group in predictions.groupby("model", sort=True):
        target_mode = group["target_mode"].iloc[0]
        target_column = TARGET_COLUMN_BY_MODE.get(target_mode, "target_return_rank")
        required = ["target_return_rank", "target_return_1m", "prediction"]
        if target_column in group.columns:
            required.append(target_column)
        group = group.dropna(subset=list(dict.fromkeys(required)))
        if group.empty:
            continue
        actual_target = group[target_column].to_numpy(dtype=float)
        predicted = group["prediction"].to_numpy(dtype=float)
        rank_r2 = np.nan
        return_r2 = np.nan
        target_r2 = np.nan
        if target_mode in {"rank", "placebo", "residual_rank"}:
            denominator = float(np.square(actual_target).sum())
            rank_r2 = (
                1.0
                - float(np.square(actual_target - predicted).sum()) / denominator
            )
            target_r2 = rank_r2
        elif target_mode in {"return", "residual_return"}:
            denominator = float(np.square(actual_target).sum())
            return_r2 = (
                1.0
                - float(np.square(actual_target - predicted).sum()) / denominator
            )
            target_r2 = return_r2
        monthly_r2 = _monthly_cross_sectional_r2_zero(group, target_column)
        r2_ci = project_stats.stationary_bootstrap_metric_ci(
            monthly_r2,
            metric="mean",
            expected_block=LEVEL_METRIC_BOOTSTRAP_BLOCK,
            n_boot=LEVEL_METRIC_BOOTSTRAP_REPETITIONS,
            seed=LEVEL_METRIC_BOOTSTRAP_SEED,
            ppy=PPY,
        )
        is_rank_r2 = target_mode in {"rank", "placebo", "residual_rank"}
        is_return_r2 = target_mode in {"return", "residual_return"}
        monthly_ic = group.groupby("date").apply(
            lambda month: month["prediction"].corr(
                month["target_return_rank"], method="spearman"
            ),
            include_groups=False,
        )
        target_monthly_ic = group.groupby("date").apply(
            lambda month: month["prediction"].corr(
                month[target_column], method="spearman"
            ),
            include_groups=False,
        )
        if "target_residual_rank" in group:
            residual_monthly_ic = group.groupby("date").apply(
                lambda month: month["prediction"].corr(
                    month["target_residual_rank"], method="spearman"
                ),
                include_groups=False,
            )
            mean_residual_ic = float(residual_monthly_ic.mean())
        else:
            mean_residual_ic = np.nan
        ic_standard_deviation = float(monthly_ic.std(ddof=1))
        if ic_standard_deviation == 0:
            ic_information_ratio = math.copysign(math.inf, float(monthly_ic.mean()))
        else:
            ic_information_ratio = float(
                monthly_ic.mean() / ic_standard_deviation * np.sqrt(12)
            )
        records.append(
            {
                "model": model_name,
                "base_model": group["base_model"].iloc[0],
                "target_mode": target_mode,
                "target_column": target_column,
                "observations": int(len(group)),
                "rank_r2_zero": rank_r2,
                "return_r2_zero": return_r2,
                "target_r2_zero": target_r2,
                "mean_monthly_spearman_ic": float(monthly_ic.mean()),
                "mean_monthly_target_spearman_ic": float(target_monthly_ic.mean()),
                "mean_monthly_residual_spearman_ic": mean_residual_ic,
                "ic_information_ratio": ic_information_ratio,
                "positive_ic_month_fraction": float(monthly_ic.gt(0).mean()),
                "rank_r2_zero_monthly_mean": (
                    r2_ci["point"] if is_rank_r2 else np.nan
                ),
                "rank_r2_zero_ci_low": (
                    r2_ci["ci_low"] if is_rank_r2 else np.nan
                ),
                "rank_r2_zero_ci_high": (
                    r2_ci["ci_high"] if is_rank_r2 else np.nan
                ),
                "rank_r2_zero_p_two_sided_zero": (
                    r2_ci["p_two_sided_zero"] if is_rank_r2 else np.nan
                ),
                "return_r2_zero_monthly_mean": (
                    r2_ci["point"] if is_return_r2 else np.nan
                ),
                "return_r2_zero_ci_low": (
                    r2_ci["ci_low"] if is_return_r2 else np.nan
                ),
                "return_r2_zero_ci_high": (
                    r2_ci["ci_high"] if is_return_r2 else np.nan
                ),
                "return_r2_zero_p_two_sided_zero": (
                    r2_ci["p_two_sided_zero"] if is_return_r2 else np.nan
                ),
                "target_r2_zero_monthly_mean": r2_ci["point"],
                "target_r2_zero_ci_low": r2_ci["ci_low"],
                "target_r2_zero_ci_high": r2_ci["ci_high"],
                "target_r2_zero_p_two_sided_zero": r2_ci["p_two_sided_zero"],
                "r2_zero_bootstrap_observations": r2_ci["observations"],
                "r2_zero_bootstrap_expected_block": LEVEL_METRIC_BOOTSTRAP_BLOCK,
                "r2_zero_bootstrap_repetitions": LEVEL_METRIC_BOOTSTRAP_REPETITIONS,
                "r2_zero_bootstrap_seed": LEVEL_METRIC_BOOTSTRAP_SEED,
                "r2_zero_bootstrap_resampling_unit": "months",
            }
        )
    return pd.DataFrame(records)


def _hac_mean_test(values: pd.Series, lags: int) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < max(12, lags + 2):
        return {
            "months": int(len(clean)),
            "mean_difference": np.nan,
            "hac_standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }
    fit = sm.OLS(clean.to_numpy(dtype=float), np.ones((len(clean), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags},
    )
    return {
        "months": int(len(clean)),
        "mean_difference": float(fit.params[0]),
        "hac_standard_error": float(fit.bse[0]),
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
    }


def _base_model_from_label(
    model_label: str,
    target_mode: str,
    fallback: str | None = None,
) -> str:
    if fallback:
        return fallback
    suffix = f"_{target_mode}"
    if model_label.endswith(suffix):
        return model_label[: -len(suffix)]
    return model_label


def _clark_west_orientation(
    model_a: str,
    base_model_a: str,
    model_b: str,
    base_model_b: str,
) -> tuple[str, str] | None:
    """Restricted/unrestricted pair for Clark-West, or None if the test does not apply.

    Nesting alone is not sufficient: the adjustment term also requires estimators
    whose forecasts differ only by sampling error in the extra parameters, so
    stochastically trained learners are excluded (see
    ``stats.clark_west_estimator_eligible``).
    """
    eligible, _ = project_stats.clark_west_estimator_eligible(
        base_model_a,
        base_model_b,
    )
    if not eligible:
        return None
    restricted_models = {"zero", "momentum"}
    if base_model_a in restricted_models and base_model_b not in restricted_models:
        return model_a, model_b
    if base_model_b in restricted_models and base_model_a not in restricted_models:
        return model_b, model_a
    if base_model_a == "zero" and base_model_b == "momentum":
        return model_a, model_b
    if base_model_b == "zero" and base_model_a == "momentum":
        return model_b, model_a
    return None


def _empty_clark_west_result() -> dict[str, Any]:
    return {
        "clark_west_restricted_model": "",
        "clark_west_unrestricted_model": "",
        "clark_west_adjusted_mean_difference": np.nan,
        "clark_west_hac_standard_error": np.nan,
        "clark_west_t_stat": np.nan,
        "clark_west_p_one_sided": np.nan,
        "clark_west_months": 0,
        "clark_west_interpretation": "",
    }


def _clark_west_adjusted_loss_test(
    common: pd.DataFrame,
    target_column: str,
    restricted_model: str,
    unrestricted_model: str,
    hac_lags: int,
) -> dict[str, Any]:
    import stats as project_stats

    restricted_prediction = (
        "prediction_a"
        if restricted_model == common.attrs["model_a"]
        else "prediction_b"
    )
    unrestricted_prediction = (
        "prediction_a"
        if unrestricted_model == common.attrs["model_a"]
        else "prediction_b"
    )
    test = project_stats.clark_west_test(
        common[target_column],
        common[restricted_prediction],
        common[unrestricted_prediction],
        dates=common["date"],
        maxlags=hac_lags,
    )
    return {
        "clark_west_restricted_model": restricted_model,
        "clark_west_unrestricted_model": unrestricted_model,
        "clark_west_adjusted_mean_difference": test["mean_difference"],
        "clark_west_hac_standard_error": test["hac_standard_error"],
        "clark_west_t_stat": test["t_stat"],
        "clark_west_p_one_sided": test["p_one_sided"],
        "clark_west_months": test["months"],
        "clark_west_interpretation": "positive_mean_favors_unrestricted_model",
    }


def _spearman_or_nan(prediction: pd.Series, actual: pd.Series) -> float:
    if prediction.nunique(dropna=True) < 2 or actual.nunique(dropna=True) < 2:
        return np.nan
    return float(prediction.corr(actual, method="spearman"))


def predictive_accuracy_tests(
    predictions: pd.DataFrame,
    hac_lags: int = 6,
    residual_control_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise monthly loss and IC comparisons within a common target scale.

    On residual targets, a comparison involving a baseline whose predictor is
    one of the neutralization controls is flagged ineligible and dropped from
    the Holm family: the baseline is regressed against itself, so the gap
    measures the control set rather than predictive skill.
    """
    if residual_control_columns is None:
        residual_control_columns = list(
            predictions.attrs.get("residual_control_columns", [])
        )
    loss_records: list[dict[str, Any]] = []
    ic_records: list[dict[str, Any]] = []
    labelled = predictions.dropna(
        subset=["target_return_1m", "target_return_rank", "prediction"]
    )
    for target_mode, target_column in [
        ("return", "target_return_1m"),
        ("rank", "target_return_rank"),
        ("residual_return", "target_return_residual_1m"),
        ("residual_rank", "target_residual_rank"),
    ]:
        if target_column not in labelled.columns:
            continue
        subset = labelled[labelled["target_mode"].eq(target_mode)]
        subset = subset.dropna(subset=[target_column])
        has_base_model = "base_model" in subset.columns
        models = sorted(subset["model"].unique())
        for model_a, model_b in itertools.combinations(models, 2):
            left_columns = ["date", "ric", target_column]
            if "target_return_rank" not in left_columns:
                left_columns.append("target_return_rank")
            ic_target_column = (
                "target_residual_rank"
                if target_mode.startswith("residual")
                and "target_residual_rank" in subset.columns
                else "target_return_rank"
            )
            if ic_target_column not in left_columns:
                left_columns.append(ic_target_column)
            left_columns.append("prediction")
            right_columns = ["date", "ric", "prediction"]
            if has_base_model:
                left_columns.append("base_model")
                right_columns.append("base_model")
            left = subset[subset["model"].eq(model_a)][left_columns].rename(
                columns={
                    "prediction": "prediction_a",
                    "base_model": "base_model_a",
                }
            )
            right = subset[subset["model"].eq(model_b)][right_columns].rename(
                columns={
                    "prediction": "prediction_b",
                    "base_model": "base_model_b",
                }
            )
            common = left.merge(
                right,
                on=["date", "ric"],
                how="inner",
                validate="one_to_one",
            )
            if common.empty:
                continue
            common.attrs["model_a"] = model_a
            common.attrs["model_b"] = model_b
            common["loss_difference"] = np.square(
                common[target_column] - common["prediction_a"]
            ) - np.square(common[target_column] - common["prediction_b"])
            monthly_loss = common.groupby("date")["loss_difference"].mean()
            base_model_a_value = None
            base_model_b_value = None
            if "base_model_a" in common and common["base_model_a"].notna().any():
                base_model_a_value = str(common["base_model_a"].dropna().iloc[0])
            if "base_model_b" in common and common["base_model_b"].notna().any():
                base_model_b_value = str(common["base_model_b"].dropna().iloc[0])
            base_model_a = _base_model_from_label(
                model_a,
                target_mode,
                base_model_a_value,
            )
            base_model_b = _base_model_from_label(
                model_b,
                target_mode,
                base_model_b_value,
            )
            clark_west = _empty_clark_west_result()
            estimator_ok, estimator_note = project_stats.clark_west_estimator_eligible(
                base_model_a,
                base_model_b,
            )
            orientation = _clark_west_orientation(
                model_a,
                base_model_a,
                model_b,
                base_model_b,
            )
            if orientation is not None:
                clark_west = _clark_west_adjusted_loss_test(
                    common,
                    target_column,
                    orientation[0],
                    orientation[1],
                    hac_lags,
                )
            elif not estimator_ok:
                clark_west["clark_west_interpretation"] = (
                    f"not_applicable: {estimator_note}"
                )
            if target_mode.startswith("residual"):
                residual_ok, residual_note = (
                    project_stats.residual_target_model_eligible(
                        residual_control_columns,
                        base_model_a,
                        base_model_b,
                    )
                )
            else:
                residual_ok, residual_note = True, "not_a_residual_target"
            residual_flags = {
                "residual_control_eligible": residual_ok,
                "residual_control_note": residual_note,
            }
            loss_records.append(
                {
                    "target_mode": target_mode,
                    "model_a": model_a,
                    "model_b": model_b,
                    "interpretation": "negative_mean_favors_model_a",
                    **_hac_mean_test(monthly_loss, hac_lags),
                    **clark_west,
                    **residual_flags,
                }
            )

            monthly_ic = common.groupby("date").apply(
                lambda month: pd.Series(
                    {
                        "ic_a": _spearman_or_nan(
                            month["prediction_a"],
                            month[ic_target_column],
                        ),
                        "ic_b": _spearman_or_nan(
                            month["prediction_b"],
                            month[ic_target_column],
                        ),
                    }
                ),
                include_groups=False,
            )
            ic_records.append(
                {
                    "target_mode": target_mode,
                    "model_a": model_a,
                    "model_b": model_b,
                    "interpretation": "positive_mean_favors_model_a",
                    **_hac_mean_test(
                        monthly_ic["ic_a"] - monthly_ic["ic_b"],
                        hac_lags,
                    ),
                    **residual_flags,
                }
            )

    loss = pd.DataFrame(loss_records)
    ic = pd.DataFrame(ic_records)
    for frame in [loss, ic]:
        if frame.empty:
            continue
        # Comparisons invalidated by a predictor/control conflict are kept for
        # audit but excluded from the Holm family, so a mechanically inflated
        # gap can neither claim significance nor inflate the correction for the
        # comparisons that are valid.
        eligible = (
            frame["residual_control_eligible"].fillna(True).astype(bool)
            if "residual_control_eligible" in frame.columns
            else pd.Series(True, index=frame.index)
        )
        frame["p_value_holm"] = np.nan
        for _, index in frame[eligible].groupby("target_mode").groups.items():
            frame.loc[index, "p_value_holm"] = multipletests(
                frame.loc[index, "p_value"].fillna(1.0), method="holm"
            )[1]
    if not loss.empty and "clark_west_p_one_sided" in loss.columns:
        loss["clark_west_p_one_sided_holm"] = np.nan
        for _, index in loss.groupby("target_mode").groups.items():
            pvalues = loss.loc[index, "clark_west_p_one_sided"]
            valid = pvalues.notna()
            if valid.sum() == 0:
                continue
            loss.loc[pvalues[valid].index, "clark_west_p_one_sided_holm"] = (
                multipletests(pvalues[valid], method="holm")[1]
            )
    return loss, ic


def binned_oos_responses(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    importance: pd.DataFrame,
    feature_columns: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Observed OOS response curves; descriptive, not a causal partial dependence."""
    if importance.empty:
        return pd.DataFrame()
    feature_importance = importance[
        importance["ablation_level"].eq("feature")
        & importance["market_cap_group"].eq("all")
    ].copy()
    ranking = (
        feature_importance.groupby(["model", "variable"], as_index=False)[
            "delta_r2_zero"
        ]
        .mean()
        .sort_values(["model", "delta_r2_zero"], ascending=[True, False])
    )
    ranking["feature_order"] = ranking.groupby("model").cumcount() + 1
    ranking = ranking[ranking["feature_order"].le(top_n)]
    if ranking.empty:
        return pd.DataFrame()

    available = panel[["date", "ric", *feature_columns]].drop_duplicates(
        ["date", "ric"]
    )
    prediction_base = predictions.drop(
        columns=[column for column in feature_columns if column in predictions],
        errors="ignore",
    )
    merged = prediction_base.merge(
        available,
        on=["date", "ric"],
        how="left",
        validate="many_to_one",
    )
    records: list[dict[str, Any]] = []
    edges = np.linspace(-1.0, 1.0, 11)
    for row in ranking.itertuples():
        feature = f"{row.variable}_rank"
        if feature not in merged:
            continue
        subset = merged[merged["model"].eq(row.model)].dropna(
            subset=[feature, "prediction", "target_return_1m", "target_return_rank"]
        ).copy()
        subset["feature_bin"] = pd.cut(
            subset[feature],
            bins=edges,
            labels=False,
            include_lowest=True,
        )
        target_column = TARGET_COLUMN_BY_MODE.get(
            subset["target_mode"].iloc[0],
            "target_return_rank",
        )
        grouped = subset.groupby("feature_bin", observed=True)
        for feature_bin, values in grouped:
            records.append(
                {
                    "model": row.model,
                    "target_mode": values["target_mode"].iloc[0],
                    "variable": row.variable,
                    "feature_order": int(row.feature_order),
                    "feature_bin": int(feature_bin),
                    "observations": int(len(values)),
                    "mean_feature_rank": float(values[feature].mean()),
                    "mean_prediction": float(values["prediction"].mean()),
                    "mean_realized_target": float(values[target_column].mean()),
                }
            )
    return pd.DataFrame(records)


def plot_binned_oos_responses(
    responses: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    if responses.empty:
        return []
    files = []
    for model, group in responses.groupby("model", sort=True):
        variables = (
            group[["variable", "feature_order"]]
            .drop_duplicates()
            .sort_values("feature_order")["variable"]
            .tolist()
        )
        figure, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
        for axis, variable in zip(axes.ravel(), variables, strict=False):
            values = group[group["variable"].eq(variable)].sort_values(
                "mean_feature_rank"
            )
            axis.plot(
                values["mean_feature_rank"],
                values["mean_prediction"],
                marker="o",
                label="Mean prediction",
            )
            axis.plot(
                values["mean_feature_rank"],
                values["mean_realized_target"],
                marker="s",
                label="Mean realized target",
            )
            axis.axhline(0.0, color="black", linewidth=0.6)
            axis.set_title(variable.replace("_", " "))
            axis.set_xlabel("Feature rank")
        for axis in axes.ravel()[len(variables) :]:
            axis.set_visible(False)
        axes[0, 0].legend(frameon=False, fontsize=8)
        figure.suptitle(f"OOS binned responses: {model}")
        figure.tight_layout()
        filename = f"oos_binned_responses_{model}.png"
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)
        files.append(filename)
    return files


def _portfolio_weights(
    month: pd.DataFrame,
    quantile: float,
    scheme: str,
) -> tuple[dict[str, float], int, int]:
    if month["prediction"].nunique() < 2:
        return {}, 0, 0
    score_rank = month["prediction"].rank(method="first", pct=True)
    long = month.loc[score_rank.gt(1.0 - quantile)].copy()
    short = month.loc[score_rank.le(quantile)].copy()
    if long.empty or short.empty:
        return {}, 0, 0

    if scheme == "equal":
        long_weights = pd.Series(1.0 / len(long), index=long.index)
        short_weights = pd.Series(-1.0 / len(short), index=short.index)
    elif scheme == "value":
        long_cap = long["company_market_cap"].clip(lower=0)
        short_cap = short["company_market_cap"].clip(lower=0)
        if long_cap.sum() <= 0 or short_cap.sum() <= 0:
            return {}, 0, 0
        long_weights = long_cap / long_cap.sum()
        short_weights = -(short_cap / short_cap.sum())
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")

    weights = {
        **dict(zip(long["ric"], long_weights, strict=True)),
        **dict(zip(short["ric"], short_weights, strict=True)),
    }
    return weights, len(long), len(short)


def construct_monthly_portfolios(
    predictions: pd.DataFrame,
    quantile: float,
) -> pd.DataFrame:
    records = []
    previous_long_short: dict[tuple[str, str, str], dict[str, float]] = {}
    previous_long_only: dict[tuple[str, str, str], dict[str, float]] = {}
    for (model_name, signal_date), month in predictions.sort_values(
        ["model", "date", "ric"]
    ).groupby(["model", "date"], sort=True):
        for universe_variant, minimum_size_percentile in [
            ("standard_ex_bottom_5pct", 0.05),
            ("ex_bottom_20pct", 0.20),
        ]:
            investable = month[
                month["market_cap_percentile"].ge(minimum_size_percentile)
                & month["target_return_1m"].notna()
            ]
            for scheme in ["equal", "value"]:
                weights, long_n, short_n = _portfolio_weights(
                    investable, quantile, scheme
                )
                if not weights:
                    continue
                returns = investable.set_index("ric")["target_return_1m"]
                weight_series = pd.Series(weights, dtype=float)
                gross_return = float(
                    (weight_series * returns.reindex(weight_series.index)).sum()
                )
                long_weights = weight_series[weight_series.gt(0)]
                long_return = float(
                    (long_weights * returns.reindex(long_weights.index)).sum()
                )
                short_weights = weight_series[weight_series.lt(0)]
                short_asset_return = float(
                    (-short_weights * returns.reindex(short_weights.index)).sum()
                )

                key = (model_name, scheme, universe_variant)
                prior_long_short = previous_long_short.get(key, {})
                names = set(prior_long_short) | set(weights)
                long_short_turnover = 0.5 * sum(
                    abs(
                        weights.get(name, 0.0)
                        - prior_long_short.get(name, 0.0)
                    )
                    for name in names
                )
                previous_long_short[key] = weights

                current_long = long_weights.to_dict()
                prior_long = previous_long_only.get(key, {})
                long_names = set(prior_long) | set(current_long)
                stock_turnover = sum(
                    abs(
                        current_long.get(name, 0.0)
                        - prior_long.get(name, 0.0)
                    )
                    for name in long_names
                )
                prior_cash = 1.0 - sum(prior_long.values())
                current_cash = 1.0 - sum(current_long.values())
                long_only_turnover = 0.5 * (
                    stock_turnover + abs(current_cash - prior_cash)
                )
                previous_long_only[key] = current_long
                candidate_flags = (
                    investable.set_index("ric")["is_delisting_candidate"]
                    if "is_delisting_candidate" in investable
                    else pd.Series(False, index=investable["ric"])
                )
                candidate_long = [
                    ric
                    for ric in long_weights.index
                    if bool(candidate_flags.get(ric, False))
                ]
                candidate_short = [
                    ric
                    for ric in short_weights.index
                    if bool(candidate_flags.get(ric, False))
                ]
                records.append(
                    {
                        "model": model_name,
                        "target_mode": investable["target_mode"].iloc[0],
                        "weighting": scheme,
                        "universe_variant": universe_variant,
                        "signal_date": signal_date,
                        "return_date": investable["target_date"].iloc[0],
                        "long_n": long_n,
                        "short_n": short_n,
                        "long_return": long_return,
                        "short_asset_return": short_asset_return,
                        "gross_long_short_return": gross_return,
                        "long_short_turnover": long_short_turnover,
                        "long_only_turnover": long_only_turnover,
                        "delisting_candidates_long_n": len(candidate_long),
                        "delisting_candidates_short_n": len(candidate_short),
                        "delisting_candidates_long_weight": float(
                            long_weights.reindex(candidate_long).sum()
                        ),
                        "delisting_candidates_short_weight": float(
                            -short_weights.reindex(candidate_short).sum()
                        ),
                    }
                )
    return pd.DataFrame(records)


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    running_maximum = wealth.cummax()
    return float(wealth.div(running_maximum).sub(1.0).min())


def portfolio_summary(
    monthly: pd.DataFrame,
    metrics: pd.DataFrame,
    cost_grid_bps: tuple[int, ...],
    risk_free: pd.Series | None = None,
) -> pd.DataFrame:
    records = []
    ci_records = []
    for (model_name, weighting, universe_variant), group in monthly.groupby(
        ["model", "weighting", "universe_variant"], sort=True
    ):
        group = group.sort_values("return_date")
        for portfolio, return_column, turnover_column in [
            ("long_short", "gross_long_short_return", "long_short_turnover"),
            ("long_only_top_decile", "long_return", "long_only_turnover"),
        ]:
            for cost_bps in cost_grid_bps:
                gross = group[return_column]
                net = (
                    gross - group[turnover_column] * cost_bps / 10_000.0
                )
                if portfolio == "long_only_top_decile" and risk_free is not None:
                    rf = risk_free.reindex(
                        pd.DatetimeIndex(group["return_date"])
                    )
                    rf.index = group.index
                    valid = rf.notna()
                    evaluation_gross = gross[valid]
                    evaluation_net = net[valid]
                    gross_excess = evaluation_gross - rf[valid]
                    excess = evaluation_net - rf[valid]
                    rf_missing_months = int((~valid).sum())
                else:
                    evaluation_gross = gross
                    evaluation_net = net
                    gross_excess = gross
                    excess = net
                    rf_missing_months = 0
                net_volatility = float(evaluation_net.std(ddof=1) * np.sqrt(12))
                gross_volatility = float(
                    evaluation_gross.std(ddof=1) * np.sqrt(12)
                )
                annualized_net_mean_return = float(evaluation_net.mean() * 12)
                annualized_gross_mean_return = float(
                    evaluation_gross.mean() * 12
                )
                annualized_net_excess_return = float(excess.mean() * 12)
                annualized_gross_excess_return = float(
                    gross_excess.mean() * 12
                )
                net_sharpe = _safe_annualized_sharpe(excess)
                gross_sharpe = _safe_annualized_sharpe(gross_excess)
                ci_seed = LEVEL_METRIC_BOOTSTRAP_SEED + 2 * len(ci_records)
                mean_ci = _stationary_bootstrap_columns(
                    evaluation_net,
                    metric="annualized_mean",
                    prefix="annualized_net_mean_return",
                    seed=ci_seed,
                )
                sharpe_ci = _stationary_bootstrap_columns(
                    excess,
                    metric="sharpe",
                    prefix="net_sharpe",
                    seed=ci_seed + 1,
                )
                records.append(
                    {
                        "model": model_name,
                        "target_mode": group["target_mode"].iloc[0],
                        "weighting": weighting,
                        "universe_variant": universe_variant,
                        "portfolio": portfolio,
                        "cost_bps": cost_bps,
                        "months": int(len(evaluation_net)),
                        "rf_missing_months": rf_missing_months,
                        "annualized_mean_return": annualized_net_mean_return,
                        "annualized_excess_return": annualized_net_excess_return,
                        "annualized_volatility": net_volatility,
                        "sharpe": net_sharpe,
                        "annualized_net_mean_return": annualized_net_mean_return,
                        "annualized_net_excess_return": (
                            annualized_net_excess_return
                        ),
                        "annualized_net_volatility": net_volatility,
                        "net_sharpe": net_sharpe,
                        "annualized_gross_mean_return": (
                            annualized_gross_mean_return
                        ),
                        "annualized_gross_excess_return": (
                            annualized_gross_excess_return
                        ),
                        "annualized_gross_volatility": gross_volatility,
                        "gross_sharpe": gross_sharpe,
                        "max_drawdown": _max_drawdown(evaluation_net),
                        "average_monthly_turnover": float(
                            group[turnover_column].mean()
                        ),
                        "gross_annualized_mean_return": (
                            annualized_gross_mean_return
                        ),
                    }
                )
                ci_records.append(
                    {
                        **mean_ci,
                        **sharpe_ci,
                        "level_metric_bootstrap_expected_block": (
                            LEVEL_METRIC_BOOTSTRAP_BLOCK
                        ),
                        "level_metric_bootstrap_repetitions": (
                            LEVEL_METRIC_BOOTSTRAP_REPETITIONS
                        ),
                        "annualized_net_mean_return_bootstrap_seed": ci_seed,
                        "net_sharpe_bootstrap_seed": ci_seed + 1,
                        "level_metric_bootstrap_resampling_unit": "months",
                    }
                )
    summary = pd.DataFrame(records)
    merged = summary.merge(metrics, on=["model", "target_mode"], how="left")
    ci_frame = pd.DataFrame(ci_records)
    for column in ci_frame.columns:
        merged[column] = ci_frame[column].to_numpy()
    return merged


def paired_sharpe_significance(
    monthly: pd.DataFrame,
    baseline_model: str = "momentum_rank",
    cost_bps: int = 25,
    blocks: tuple[int, ...] = (3, 6, 12),
    n_boot: int = 10_000,
    seed: int = 42,
    risk_free: pd.Series | None = None,
) -> pd.DataFrame:
    import stats as project_stats

    records = []
    comparison_models = [
        model for model in sorted(monthly["model"].unique()) if model != baseline_model
    ]
    for weighting in ["equal", "value"]:
        for universe_variant in sorted(monthly["universe_variant"].unique()):
            subset = monthly[
                monthly["weighting"].eq(weighting)
                & monthly["universe_variant"].eq(universe_variant)
            ]
            baseline = subset[subset["model"].eq(baseline_model)].set_index(
                "return_date"
            )
            if baseline.empty:
                continue
            for portfolio, return_column, turnover_column in [
                ("long_short", "gross_long_short_return", "long_short_turnover"),
                ("long_only_top_decile", "long_return", "long_only_turnover"),
            ]:
                baseline_net = (
                    baseline[return_column]
                    - baseline[turnover_column] * cost_bps / 10_000.0
                )
                for model_name in comparison_models:
                    model = subset[subset["model"].eq(model_name)].set_index(
                        "return_date"
                    )
                    index = baseline.index.intersection(model.index)
                    if portfolio == "long_only_top_decile" and risk_free is not None:
                        index = index.intersection(risk_free.dropna().index)
                    if len(index) < 24:
                        continue
                    model_net = (
                        model.loc[index, return_column]
                        - model.loc[index, turnover_column] * cost_bps / 10_000.0
                    )
                    comparison_net = baseline_net.reindex(index)
                    rf = (
                        risk_free.reindex(index)
                        if portfolio == "long_only_top_decile"
                        and risk_free is not None
                        else np.zeros(len(index))
                    )
                    for block in blocks:
                        result = project_stats.bootstrap_sharpe_diff(
                            model_net,
                            comparison_net,
                            rf,
                            expected_block=block,
                            n_boot=n_boot,
                            seed=seed,
                        )
                        records.append(
                            {
                                "model": model_name,
                                "baseline": baseline_model,
                                "weighting": weighting,
                                "universe_variant": universe_variant,
                                "portfolio": portfolio,
                                "cost_bps": cost_bps,
                                "expected_block": block,
                                **result,
                            }
                        )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    family = [
        "weighting",
        "universe_variant",
        "portfolio",
        "cost_bps",
        "expected_block",
    ]
    result["p_two_sided_holm"] = result.groupby(family)[
        "p_two_sided"
    ].transform(lambda values: multipletests(values, method="holm")[1])
    result["primary_family"] = (
        result["weighting"].eq("value")
        & result["universe_variant"].eq("standard_ex_bottom_5pct")
        & result["portfolio"].eq("long_short")
        & result["cost_bps"].eq(cost_bps)
        & result["expected_block"].eq(6)
    )
    return result


def placebo_rank_tests(
    panel: pd.DataFrame,
    config: WalkForwardConfig,
    repetitions: int,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    records = []
    placebo_config = replace(config, tune_hyperparameters=False)
    labelled = panel[panel["target_return_rank"].notna()].copy()
    for repetition in range(repetitions):
        rng = np.random.default_rng(config.random_state + repetition)
        placebo = labelled.copy()
        placebo["placebo_target"] = placebo.groupby("date")[
            "target_return_rank"
        ].transform(lambda values: rng.permutation(values.to_numpy()))
        predictions, _, _, _ = run_walk_forward(
            placebo,
            ["ridge"],
            placebo_config,
            target_column="placebo_target",
            target_mode="placebo",
            feature_columns=feature_columns,
            collect_importance=False,
        )
        metric = prediction_metrics(predictions).iloc[0].to_dict()
        records.append({"repetition": repetition, **metric})
    return pd.DataFrame(records)


def delisting_scenario_analysis(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: WalkForwardConfig,
    risk_free: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild portfolios after assigning stress returns to missing delistings."""
    if "is_delisting_candidate" not in predictions:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    candidate_mask = predictions["is_delisting_candidate"].fillna(False)
    candidates = predictions.loc[candidate_mask].copy()
    scenarios = [
        ("observed_only", None),
        ("missing_delisting_minus_30pct", -0.30),
        ("missing_delisting_minus_100pct", -1.00),
    ]
    monthly_frames = []
    summary_frames = []
    significance_frames = []
    for scenario, penalty in scenarios:
        if penalty is None:
            scenario_predictions = predictions
        else:
            scenario_return = predictions["target_return_1m"].mask(
                candidate_mask,
                penalty,
            )
            scenario_predictions = predictions.assign(
                target_return_1m=scenario_return
            )
        monthly = construct_monthly_portfolios(
            scenario_predictions,
            config.portfolio_quantile,
        )
        monthly["scenario"] = scenario
        monthly["missing_delisting_penalty"] = (
            np.nan if penalty is None else penalty
        )
        summary = portfolio_summary(
            monthly,
            metrics,
            config.cost_grid_bps,
            risk_free=risk_free,
        )
        summary["scenario"] = scenario
        summary["missing_delisting_penalty"] = (
            np.nan if penalty is None else penalty
        )
        significance = paired_sharpe_significance(
            monthly,
            risk_free=risk_free,
        )
        significance["scenario"] = scenario
        significance["missing_delisting_penalty"] = (
            np.nan if penalty is None else penalty
        )
        monthly_frames.append(monthly)
        summary_frames.append(summary)
        significance_frames.append(significance)
    return (
        candidates,
        pd.concat(monthly_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(significance_frames, ignore_index=True),
    )


def build_ml_outputs(
    panel_path: Path,
    output_dir: Path,
    model_names: list[str],
    config: WalkForwardConfig,
    target_modes: tuple[str, ...] = ("rank", "return"),
    placebo_repetitions: int = 20,
    delisting_audit_path: Path | None = None,
    risk_free: pd.Series | None = None,
    feature_set: str = "baseline",
    collect_importance: bool = True,
    sample_start_date: str | pd.Timestamp | None = None,
    sample_end_date: str | pd.Timestamp | None = None,
    require_estimates_feature: bool = False,
    require_revision_signal: bool = False,
    require_estimate_signal_lag_months: int | None = None,
    residual_control_set: str = "full",
) -> dict:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        )
    feature_columns = FEATURE_SETS[feature_set]
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_model_panel(
        panel_path,
        delisting_audit_path,
        feature_columns=feature_columns,
        sample_start_date=sample_start_date,
        sample_end_date=sample_end_date,
        require_estimates_feature=require_estimates_feature,
        require_revision_signal=require_revision_signal,
        require_estimate_signal_lag_months=require_estimate_signal_lag_months,
        residual_control_set=residual_control_set,
    )
    sample_filter_audit = dict(panel.attrs.get("sample_filter_audit", {}))
    panel.attrs = {}
    prediction_frames = []
    fit_frames = []
    coefficient_frames = []
    importance_frames = []
    for target_mode in target_modes:
        if target_mode == "rank":
            mode_models = model_names
            target_column = "target_return_rank"
        elif target_mode == "return":
            mode_models = [
                model for model in model_names if model not in {"momentum", "zero"}
            ]
            target_column = "target_return_1m"
        elif target_mode == "residual_rank":
            mode_models = [model for model in model_names if model != "zero"]
            target_column = "target_residual_rank"
        elif target_mode == "residual_return":
            mode_models = [
                model for model in model_names if model not in {"momentum", "zero"}
            ]
            target_column = "target_return_residual_1m"
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")
        if not mode_models:
            continue
        (
            mode_predictions,
            mode_fit_log,
            mode_coefficients,
            mode_importance,
        ) = run_walk_forward(
            panel,
            mode_models,
            config,
            target_column=target_column,
            target_mode=target_mode,
            feature_columns=feature_columns,
            collect_importance=collect_importance,
        )
        prediction_frames.append(mode_predictions)
        fit_frames.append(mode_fit_log)
        coefficient_frames.append(mode_coefficients)
        importance_frames.append(mode_importance)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    fit_log = pd.concat(fit_frames, ignore_index=True)
    in_sample_columns = [
        "model",
        "base_model",
        "target_mode",
        "test_year",
        "train_rows_available",
        "train_rows_used",
        "train_signal_start",
        "train_signal_end",
        "train_target_end",
        "train_label_cutoff",
        "in_sample_observations",
        "in_sample_mse",
        "in_sample_sse",
        "in_sample_zero_benchmark_sse",
        "in_sample_r2_zero",
        "in_sample_mean_monthly_target_spearman_ic",
        "in_sample_mean_monthly_spearman_ic",
    ]
    in_sample_metrics = fit_log[
        [column for column in in_sample_columns if column in fit_log.columns]
    ].copy()
    nonempty_coefficients = [frame for frame in coefficient_frames if not frame.empty]
    coefficients = (
        pd.concat(nonempty_coefficients, ignore_index=True)
        if nonempty_coefficients
        else pd.DataFrame(columns=["model", "model_label", "target_mode", "test_year", "feature", "coefficient"])
    )
    nonempty_importance = [frame for frame in importance_frames if not frame.empty]
    importance = (
        pd.concat(nonempty_importance, ignore_index=True)
        if nonempty_importance
        else pd.DataFrame()
    )
    if predictions.empty:
        raise RuntimeError("Walk-forward run produced no predictions")
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
        risk_free=risk_free,
    )
    loss_tests, ic_tests = predictive_accuracy_tests(
        predictions,
        residual_control_columns=list(
            panel.attrs.get("residual_control_columns", [])
        ),
    )
    responses = binned_oos_responses(
        predictions,
        panel,
        importance,
        feature_columns,
    )
    placebo = placebo_rank_tests(
        panel,
        config,
        placebo_repetitions,
        feature_columns=feature_columns,
    )
    ridge_actual = metrics.loc[
        metrics["model"].eq("ridge_rank"), "mean_monthly_spearman_ic"
    ]
    actual_ic = float(ridge_actual.iloc[0]) if not ridge_actual.empty else np.nan
    if placebo.empty or np.isnan(actual_ic):
        placebo_summary = pd.DataFrame()
    else:
        placebo_ic = placebo["mean_monthly_spearman_ic"]
        placebo_summary = pd.DataFrame(
            [
                {
                    "model": "ridge_rank",
                    "actual_mean_monthly_ic": actual_ic,
                    "placebo_repetitions": int(len(placebo)),
                    "placebo_mean_ic": float(placebo_ic.mean()),
                    "placebo_std_ic": float(placebo_ic.std(ddof=1)),
                    "placebo_95pct_quantile": float(placebo_ic.quantile(0.95)),
                    "empirical_two_sided_p": float(
                        (
                            1
                            + placebo_ic.abs().ge(abs(actual_ic)).sum()
                        )
                        / (len(placebo_ic) + 1)
                    ),
                }
            ]
        )

    predictions.to_parquet(
        output_dir / "predictions.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    fit_log.to_csv(output_dir / "fit_log.csv", index=False)
    in_sample_metrics.to_csv(output_dir / "in_sample_fit_metrics.csv", index=False)
    coefficients.to_csv(output_dir / "linear_coefficients.csv", index=False)
    importance.to_csv(output_dir / "oos_variable_importance.csv", index=False)
    responses.to_csv(output_dir / "oos_binned_responses.csv", index=False)
    response_figures = plot_binned_oos_responses(responses, output_dir)
    metrics.to_csv(output_dir / "prediction_metrics.csv", index=False)
    loss_tests.to_csv(
        output_dir / "predictive_accuracy_loss_tests.csv", index=False
    )
    ic_tests.to_csv(
        output_dir / "predictive_accuracy_ic_tests.csv", index=False
    )
    monthly.to_csv(output_dir / "monthly_portfolios.csv", index=False)
    summary.to_csv(output_dir / "model_summary.csv", index=False)
    significance.to_csv(output_dir / "sharpe_significance.csv", index=False)
    placebo.to_csv(output_dir / "placebo_rank_tests.csv", index=False)
    placebo_summary.to_csv(output_dir / "placebo_summary.csv", index=False)

    (
        delisting_candidates,
        delisting_monthly,
        delisting_summary,
        delisting_significance,
    ) = delisting_scenario_analysis(
        predictions,
        metrics,
        config,
        risk_free=risk_free,
    )
    if not delisting_candidates.empty:
        delisting_candidates.to_parquet(
            output_dir / "delisting_candidate_scores.parquet",
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        delisting_monthly.to_csv(
            output_dir / "delisting_scenario_monthly_portfolios.csv",
            index=False,
        )
        delisting_summary.to_csv(
            output_dir / "delisting_scenario_summary.csv",
            index=False,
        )
        delisting_significance.to_csv(
            output_dir / "delisting_scenario_significance.csv",
            index=False,
        )

    manifest = {
        "panel_path": str(panel_path),
        "models": model_names,
        "target_modes": target_modes,
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "residual_targets": {
            "control_set": residual_control_set,
            "control_columns": list(
                panel.attrs.get("residual_control_columns", [])
            ),
            "categorical_controls": RESIDUAL_CATEGORICAL_COLUMNS,
        },
        "sample_filter": {
            "sample_start_date": (
                str(pd.Timestamp(sample_start_date).date())
                if sample_start_date is not None
                else None
            ),
            "sample_end_date": (
                str(pd.Timestamp(sample_end_date).date())
                if sample_end_date is not None
                else None
            ),
            "require_estimates_feature": require_estimates_feature,
            "require_revision_signal": require_revision_signal,
            "require_estimate_signal_lag_months": (
                require_estimate_signal_lag_months
            ),
            "revision_signal_columns": (
                ESTIMATES_INFORMATION_TYPES["revisions"]
                if require_revision_signal
                else []
            ),
        },
        "sample_filter_audit": sample_filter_audit,
        "targets": {
            "rank": "target_return_rank",
            "return": "target_return_1m",
            "residual_rank": "target_residual_rank",
            "residual_return": "target_return_residual_1m",
        },
        "portfolio_return": "target_return_1m",
        "config": asdict(config),
        "collect_importance": collect_importance,
        "rows": {
            "input_model_rows": int(len(panel)),
            "predictions": int(len(predictions)),
            "labelled_predictions": int(
                predictions["target_return_1m"].notna().sum()
            ),
            "delisting_candidate_predictions": int(
                predictions["is_delisting_candidate"].sum()
            ),
            "delisting_candidate_securities": int(
                predictions.loc[
                    predictions["is_delisting_candidate"], "ric"
                ].nunique()
            ),
            "portfolio_months": int(len(monthly)),
            "placebo_repetitions": placebo_repetitions,
            "in_sample_fit_metrics": int(len(in_sample_metrics)),
            "variable_importance": int(len(importance)),
            "binned_responses": int(len(responses)),
        },
        "figures": response_figures,
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
            "delisting_candidates_used_for_training": int(
                (
                    panel["is_delisting_candidate"]
                    & panel["target_return_rank"].notna()
                ).sum()
            ),
        },
    }
    (output_dir / "ml_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
