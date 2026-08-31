"""Forecast-error mechanism tests for strict lagged analyst revisions.

The input panel is built by ``build_forecast_error_panel.py``.  Each row is a
monthly, lagged consensus snapshot matched to the next annual actual announced
after that snapshot.  This script asks whether the revision ranks predict the
subsequent actual-minus-consensus error, with cross-signal placebo tests.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "forecast_errors_strict_lag1_revision_signal"
    / "monthly_forecast_error_panel.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "forecast_error_mechanism"
)

CONTINUOUS_CONTROLS = [
    "log_size_rank",
    "book_to_market_rank",
    "momentum_12_2_rank",
    "volatility_12m_rank",
    "est_eps_dispersion_rank",
    "est_revenue_dispersion_rank",
    "est_coverage_composite_rank",
]
CATEGORICAL_CONTROLS = ["screen_country", "TR.TRBCECONOMICSECTOR"]


@dataclass(frozen=True)
class MechanismConfig:
    min_cross_section: int = 30
    hac_lags: int = 15
    collapsed_lead_months: int = 6
    min_hac_periods: int = 12
    include_fixed_effects: bool = True
    include_controls: bool = True


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    dependent: str
    valid_flag: str
    signal: str
    announce_date: str
    period_end: str
    expected_sign: int
    horizon: str
    channel: str
    placebo: bool


@dataclass(frozen=True)
class JointMechanismSpec:
    name: str
    dependent: str
    valid_flag: str
    primary_signal: str
    competing_signal: str
    announce_date: str
    period_end: str
    horizon: str
    channel: str


def default_specs() -> list[MechanismSpec]:
    specs: list[MechanismSpec] = []
    for horizon in ("3m", "1m"):
        specs.extend(
            [
                MechanismSpec(
                    name=f"eps_error_on_eps_revision_{horizon}",
                    dependent="eps_error_to_price_winsorized",
                    valid_flag="eps_error_valid",
                    signal=f"est_eps_revision_{horizon}_rank",
                    announce_date="eps_announce_date",
                    period_end="eps_period_end",
                    expected_sign=1,
                    horizon=horizon,
                    channel="eps",
                    placebo=False,
                ),
                MechanismSpec(
                    name=f"eps_error_on_revenue_revision_{horizon}_placebo",
                    dependent="eps_error_to_price_winsorized",
                    valid_flag="eps_error_valid",
                    signal=f"est_revenue_revision_{horizon}_rank",
                    announce_date="eps_announce_date",
                    period_end="eps_period_end",
                    expected_sign=1,
                    horizon=horizon,
                    channel="eps",
                    placebo=True,
                ),
                MechanismSpec(
                    name=f"epsfr_error_on_eps_revision_{horizon}_robustness",
                    dependent="epsfr_error_to_price_winsorized",
                    valid_flag="epsfr_error_valid",
                    signal=f"est_eps_revision_{horizon}_rank",
                    announce_date="epsfr_announce_date",
                    period_end="epsfr_period_end",
                    expected_sign=1,
                    horizon=horizon,
                    channel="epsfr",
                    placebo=False,
                ),
                MechanismSpec(
                    name=f"epsfr_error_on_revenue_revision_{horizon}_placebo",
                    dependent="epsfr_error_to_price_winsorized",
                    valid_flag="epsfr_error_valid",
                    signal=f"est_revenue_revision_{horizon}_rank",
                    announce_date="epsfr_announce_date",
                    period_end="epsfr_period_end",
                    expected_sign=1,
                    horizon=horizon,
                    channel="epsfr",
                    placebo=True,
                ),
                MechanismSpec(
                    name=f"revenue_error_on_revenue_revision_{horizon}",
                    dependent="revenue_error_to_market_cap_winsorized",
                    valid_flag="revenue_error_valid",
                    signal=f"est_revenue_revision_{horizon}_rank",
                    announce_date="revenue_announce_date",
                    period_end="revenue_period_end",
                    expected_sign=1,
                    horizon=horizon,
                    channel="revenue",
                    placebo=False,
                ),
                MechanismSpec(
                    name=f"revenue_error_on_eps_revision_{horizon}_placebo",
                    dependent="revenue_error_to_market_cap_winsorized",
                    valid_flag="revenue_error_valid",
                    signal=f"est_eps_revision_{horizon}_rank",
                    announce_date="revenue_announce_date",
                    period_end="revenue_period_end",
                    expected_sign=1,
                    horizon=horizon,
                    channel="revenue",
                    placebo=True,
                ),
            ]
        )
    return specs


def default_joint_specs() -> list[JointMechanismSpec]:
    specs = []
    for horizon in ("3m", "1m"):
        specs.extend(
            [
                JointMechanismSpec(
                    name=f"eps_error_joint_eps_revenue_revisions_{horizon}",
                    dependent="eps_error_to_price_winsorized",
                    valid_flag="eps_error_valid",
                    primary_signal=f"est_eps_revision_{horizon}_rank",
                    competing_signal=f"est_revenue_revision_{horizon}_rank",
                    announce_date="eps_announce_date",
                    period_end="eps_period_end",
                    horizon=horizon,
                    channel="eps",
                ),
                JointMechanismSpec(
                    name=f"epsfr_error_joint_eps_revenue_revisions_{horizon}",
                    dependent="epsfr_error_to_price_winsorized",
                    valid_flag="epsfr_error_valid",
                    primary_signal=f"est_eps_revision_{horizon}_rank",
                    competing_signal=f"est_revenue_revision_{horizon}_rank",
                    announce_date="epsfr_announce_date",
                    period_end="epsfr_period_end",
                    horizon=horizon,
                    channel="epsfr",
                ),
                JointMechanismSpec(
                    name=f"revenue_error_joint_revenue_eps_revisions_{horizon}",
                    dependent="revenue_error_to_market_cap_winsorized",
                    valid_flag="revenue_error_valid",
                    primary_signal=f"est_revenue_revision_{horizon}_rank",
                    competing_signal=f"est_eps_revision_{horizon}_rank",
                    announce_date="revenue_announce_date",
                    period_end="revenue_period_end",
                    horizon=horizon,
                    channel="revenue",
                ),
            ]
        )
    return specs


def specificity_pairs() -> list[dict[str, str]]:
    pairs = []
    for horizon in ("3m", "1m"):
        pairs.extend(
            [
                {
                    "name": f"eps_error_eps_minus_revenue_revision_{horizon}",
                    "primary_spec": f"eps_error_on_eps_revision_{horizon}",
                    "placebo_spec": f"eps_error_on_revenue_revision_{horizon}_placebo",
                    "channel": "eps",
                    "horizon": horizon,
                },
                {
                    "name": f"epsfr_error_eps_minus_revenue_revision_{horizon}",
                    "primary_spec": f"epsfr_error_on_eps_revision_{horizon}_robustness",
                    "placebo_spec": f"epsfr_error_on_revenue_revision_{horizon}_placebo",
                    "channel": "epsfr",
                    "horizon": horizon,
                },
                {
                    "name": f"revenue_error_revenue_minus_eps_revision_{horizon}",
                    "primary_spec": f"revenue_error_on_revenue_revision_{horizon}",
                    "placebo_spec": f"revenue_error_on_eps_revision_{horizon}_placebo",
                    "channel": "revenue",
                    "horizon": horizon,
                },
            ]
        )
    return pairs


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Forecast-error panel missing required columns: {missing}")


def _month_difference(later: pd.Series, earlier: pd.Series) -> pd.Series:
    later_date = pd.to_datetime(later)
    earlier_date = pd.to_datetime(earlier)
    return (later_date.dt.year * 12 + later_date.dt.month) - (
        earlier_date.dt.year * 12 + earlier_date.dt.month
    )


def _available_controls(frame: pd.DataFrame, config: MechanismConfig) -> list[str]:
    if not config.include_controls:
        return []
    controls = []
    for column in CONTINUOUS_CONTROLS:
        if column in frame and pd.to_numeric(frame[column], errors="coerce").notna().any():
            controls.append(column)
    return controls


def _regression_columns(
    dependent: str,
    signals: list[str],
    controls: list[str],
    config: MechanismConfig,
) -> list[str]:
    columns = [dependent, *signals, *controls]
    if config.include_fixed_effects:
        columns.extend(CATEGORICAL_CONTROLS)
    return columns


def _cross_sectional_signal_slope(
    group: pd.DataFrame,
    spec: MechanismSpec,
    controls: list[str],
    config: MechanismConfig,
) -> dict[str, Any]:
    columns = _regression_columns(spec.dependent, [spec.signal], controls, config)
    work = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < config.min_cross_section:
        return {
            "signal_slope": np.nan,
            "observations": int(len(work)),
            "regressors": np.nan,
            "rank": np.nan,
        }

    x = work[[spec.signal, *controls]].astype(float).reset_index(drop=True)
    if config.include_fixed_effects:
        dummies = pd.get_dummies(
            work[CATEGORICAL_CONTROLS].astype("string").fillna("missing"),
            drop_first=True,
            dtype=float,
        ).reset_index(drop=True)
        x = pd.concat([x, dummies], axis=1)
    x.insert(0, "constant", 1.0)
    if len(work) <= x.shape[1] + 2:
        return {
            "signal_slope": np.nan,
            "observations": int(len(work)),
            "regressors": int(x.shape[1]),
            "rank": np.nan,
        }

    y = pd.to_numeric(work[spec.dependent], errors="coerce").to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x.to_numpy(dtype=float), y, rcond=None)
    return {
        "signal_slope": float(coefficients[x.columns.get_loc(spec.signal)]),
        "observations": int(len(work)),
        "regressors": int(x.shape[1]),
        "rank": int(rank),
    }


def _hac_mean(values: pd.Series, lags: int, min_periods: int) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < max(min_periods, lags + 2):
        return {
            "mean": np.nan,
            "standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "periods": int(len(clean)),
        }
    fit = sm.OLS(clean.to_numpy(dtype=float), np.ones((len(clean), 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags},
    )
    mean = float(fit.params[0])
    standard_error = float(fit.bse[0])
    return {
        "mean": mean,
        "standard_error": standard_error,
        "t_stat": float(fit.tvalues[0]),
        "p_value": float(fit.pvalues[0]),
        "ci_low": mean - 1.96 * standard_error,
        "ci_high": mean + 1.96 * standard_error,
        "periods": int(len(clean)),
    }


def _valid_analysis_frame(
    frame: pd.DataFrame,
    *,
    dependent: str,
    valid_flag: str,
    signals: list[str],
    announce_date: str,
    period_end: str,
) -> pd.DataFrame:
    columns = [
        "date",
        "ric",
        "est_snapshot_date",
        dependent,
        valid_flag,
        *signals,
        announce_date,
        period_end,
    ]
    optional = [
        *CONTINUOUS_CONTROLS,
        *CATEGORICAL_CONTROLS,
    ]
    present = [column for column in optional if column in frame]
    _require_columns(frame, columns)
    work = frame[[*columns, *present]].copy()
    work[valid_flag] = work[valid_flag].astype(bool)
    work = work[work[valid_flag]].copy()
    needed = [
        "date",
        "ric",
        "est_snapshot_date",
        dependent,
        *signals,
        announce_date,
        period_end,
    ]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    work["date"] = pd.to_datetime(work["date"])
    work["est_snapshot_date"] = pd.to_datetime(work["est_snapshot_date"])
    work[announce_date] = pd.to_datetime(work[announce_date])
    work[period_end] = pd.to_datetime(work[period_end])
    work["announcement_month"] = work[announce_date].dt.to_period("M").dt.to_timestamp("M")
    work["lead_months"] = _month_difference(work[announce_date], work["est_snapshot_date"])
    work = work[work["lead_months"].ge(1)].copy()
    return work


def _valid_spec_frame(frame: pd.DataFrame, spec: MechanismSpec) -> pd.DataFrame:
    return _valid_analysis_frame(
        frame,
        dependent=spec.dependent,
        valid_flag=spec.valid_flag,
        signals=[spec.signal],
        announce_date=spec.announce_date,
        period_end=spec.period_end,
    )


def build_collapsed_sample(
    frame: pd.DataFrame,
    spec: MechanismSpec,
    lead_months: int,
) -> pd.DataFrame:
    """Keep one lagged snapshot per firm and realized annual actual."""
    work = _valid_spec_frame(frame, spec)
    if work.empty:
        return work
    work["lead_distance"] = (work["lead_months"] - lead_months).abs()
    work = work.sort_values(
        [
            "ric",
            spec.period_end,
            "lead_distance",
            "lead_months",
            "est_snapshot_date",
        ],
        ascending=[True, True, True, True, False],
    )
    return work.drop_duplicates(["ric", spec.period_end], keep="first").copy()


def _valid_joint_frame(frame: pd.DataFrame, spec: JointMechanismSpec) -> pd.DataFrame:
    return _valid_analysis_frame(
        frame,
        dependent=spec.dependent,
        valid_flag=spec.valid_flag,
        signals=[spec.primary_signal, spec.competing_signal],
        announce_date=spec.announce_date,
        period_end=spec.period_end,
    )


def build_collapsed_joint_sample(
    frame: pd.DataFrame,
    spec: JointMechanismSpec,
    lead_months: int,
) -> pd.DataFrame:
    work = _valid_joint_frame(frame, spec)
    if work.empty:
        return work
    work["lead_distance"] = (work["lead_months"] - lead_months).abs()
    work = work.sort_values(
        [
            "ric",
            spec.period_end,
            "lead_distance",
            "lead_months",
            "est_snapshot_date",
        ],
        ascending=[True, True, True, True, False],
    )
    return work.drop_duplicates(["ric", spec.period_end], keep="first").copy()


def _sample_frame(
    frame: pd.DataFrame,
    spec: MechanismSpec,
    sample: str,
    config: MechanismConfig,
) -> tuple[pd.DataFrame, str]:
    if sample == "overlapping_monthly":
        return _valid_spec_frame(frame, spec), "date"
    if sample == "collapsed_firm_year":
        return build_collapsed_sample(frame, spec, config.collapsed_lead_months), "announcement_month"
    raise ValueError(f"Unknown mechanism sample: {sample}")


def _joint_sample_frame(
    frame: pd.DataFrame,
    spec: JointMechanismSpec,
    sample: str,
    config: MechanismConfig,
) -> tuple[pd.DataFrame, str]:
    if sample == "overlapping_monthly":
        return _valid_joint_frame(frame, spec), "date"
    if sample == "collapsed_firm_year":
        return build_collapsed_joint_sample(frame, spec, config.collapsed_lead_months), (
            "announcement_month"
        )
    raise ValueError(f"Unknown mechanism sample: {sample}")


def run_fama_macbeth_for_spec(
    frame: pd.DataFrame,
    spec: MechanismSpec,
    sample: str,
    config: MechanismConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_frame, time_column = _sample_frame(frame, spec, sample, config)
    controls = _available_controls(sample_frame, config)
    records: list[dict[str, Any]] = []
    if sample_frame.empty:
        monthly = pd.DataFrame(
            columns=[
                "sample",
                "spec",
                "time",
                "signal_slope",
                "observations",
                "regressors",
            ]
        )
    else:
        for time_value, group in sample_frame.groupby(time_column, sort=True):
            result = _cross_sectional_signal_slope(group, spec, controls, config)
            records.append(
                {
                    "sample": sample,
                    "spec": spec.name,
                    "time": time_value,
                    **result,
                }
            )
        monthly = pd.DataFrame(records)

    inference = _hac_mean(monthly.get("signal_slope", pd.Series(dtype=float)), config.hac_lags, config.min_hac_periods)
    summary = {
        "sample": sample,
        "spec": spec.name,
        "dependent": spec.dependent,
        "signal": spec.signal,
        "channel": spec.channel,
        "horizon": spec.horizon,
        "placebo": spec.placebo,
        "expected_sign": spec.expected_sign,
        "mean_signal_slope": inference["mean"],
        "expected_signed_slope": inference["mean"] * spec.expected_sign
        if pd.notna(inference["mean"])
        else np.nan,
        "hac_standard_error": inference["standard_error"],
        "t_stat": inference["t_stat"],
        "p_value": inference["p_value"],
        "ci_low": inference["ci_low"],
        "ci_high": inference["ci_high"],
        "periods": inference["periods"],
        "rows": int(len(sample_frame)),
        "firms": int(sample_frame["ric"].nunique()) if "ric" in sample_frame else 0,
        "average_cross_section": float(monthly["observations"].mean()) if not monthly.empty else np.nan,
        "positive_period_fraction": float(monthly["signal_slope"].dropna().gt(0).mean())
        if not monthly.empty
        else np.nan,
        "controls": ",".join(controls),
        "fixed_effects": config.include_fixed_effects,
        "collapsed_lead_months": config.collapsed_lead_months if sample == "collapsed_firm_year" else np.nan,
        "median_selected_lead_months": float(sample_frame["lead_months"].median())
        if "lead_months" in sample_frame and not sample_frame.empty
        else np.nan,
    }
    return monthly, summary


def _cross_sectional_joint_slopes(
    group: pd.DataFrame,
    spec: JointMechanismSpec,
    controls: list[str],
    config: MechanismConfig,
) -> dict[str, Any]:
    signals = [spec.primary_signal, spec.competing_signal]
    columns = _regression_columns(spec.dependent, signals, controls, config)
    work = group[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < config.min_cross_section:
        return {
            "primary_slope": np.nan,
            "competing_slope": np.nan,
            "slope_difference": np.nan,
            "observations": int(len(work)),
            "regressors": np.nan,
            "rank": np.nan,
        }

    x = work[[*signals, *controls]].astype(float).reset_index(drop=True)
    if config.include_fixed_effects:
        dummies = pd.get_dummies(
            work[CATEGORICAL_CONTROLS].astype("string").fillna("missing"),
            drop_first=True,
            dtype=float,
        ).reset_index(drop=True)
        x = pd.concat([x, dummies], axis=1)
    x.insert(0, "constant", 1.0)
    if len(work) <= x.shape[1] + 2:
        return {
            "primary_slope": np.nan,
            "competing_slope": np.nan,
            "slope_difference": np.nan,
            "observations": int(len(work)),
            "regressors": int(x.shape[1]),
            "rank": np.nan,
        }

    y = pd.to_numeric(work[spec.dependent], errors="coerce").to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x.to_numpy(dtype=float), y, rcond=None)
    primary = float(coefficients[x.columns.get_loc(spec.primary_signal)])
    competing = float(coefficients[x.columns.get_loc(spec.competing_signal)])
    return {
        "primary_slope": primary,
        "competing_slope": competing,
        "slope_difference": primary - competing,
        "observations": int(len(work)),
        "regressors": int(x.shape[1]),
        "rank": int(rank),
    }


def run_joint_fama_macbeth_for_spec(
    frame: pd.DataFrame,
    spec: JointMechanismSpec,
    sample: str,
    config: MechanismConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample_frame, time_column = _joint_sample_frame(frame, spec, sample, config)
    controls = _available_controls(sample_frame, config)
    records: list[dict[str, Any]] = []
    if sample_frame.empty:
        monthly = pd.DataFrame(
            columns=[
                "sample",
                "spec",
                "time",
                "primary_slope",
                "competing_slope",
                "slope_difference",
                "observations",
                "regressors",
                "rank",
            ]
        )
    else:
        for time_value, group in sample_frame.groupby(time_column, sort=True):
            result = _cross_sectional_joint_slopes(group, spec, controls, config)
            records.append(
                {
                    "sample": sample,
                    "spec": spec.name,
                    "time": time_value,
                    **result,
                }
            )
        monthly = pd.DataFrame(records)

    primary = _hac_mean(monthly["primary_slope"], config.hac_lags, config.min_hac_periods)
    competing = _hac_mean(monthly["competing_slope"], config.hac_lags, config.min_hac_periods)
    difference = _hac_mean(
        monthly["slope_difference"],
        config.hac_lags,
        config.min_hac_periods,
    )
    summary = {
        "sample": sample,
        "spec": spec.name,
        "dependent": spec.dependent,
        "primary_signal": spec.primary_signal,
        "competing_signal": spec.competing_signal,
        "channel": spec.channel,
        "horizon": spec.horizon,
        "primary_mean_slope": primary["mean"],
        "primary_hac_standard_error": primary["standard_error"],
        "primary_t_stat": primary["t_stat"],
        "primary_p_value": primary["p_value"],
        "competing_mean_slope": competing["mean"],
        "competing_hac_standard_error": competing["standard_error"],
        "competing_t_stat": competing["t_stat"],
        "competing_p_value": competing["p_value"],
        "mean_slope_difference": difference["mean"],
        "difference_hac_standard_error": difference["standard_error"],
        "difference_t_stat": difference["t_stat"],
        "difference_p_value": difference["p_value"],
        "difference_ci_low": difference["ci_low"],
        "difference_ci_high": difference["ci_high"],
        "periods": difference["periods"],
        "rows": int(len(sample_frame)),
        "firms": int(sample_frame["ric"].nunique()) if "ric" in sample_frame else 0,
        "average_cross_section": float(monthly["observations"].mean()) if not monthly.empty else np.nan,
        "positive_difference_fraction": float(
            monthly["slope_difference"].dropna().gt(0).mean()
        )
        if not monthly.empty
        else np.nan,
        "controls": ",".join(controls),
        "fixed_effects": config.include_fixed_effects,
        "collapsed_lead_months": config.collapsed_lead_months
        if sample == "collapsed_firm_year"
        else np.nan,
        "median_selected_lead_months": float(sample_frame["lead_months"].median())
        if "lead_months" in sample_frame and not sample_frame.empty
        else np.nan,
    }
    return monthly, summary


def _monthly_quintile_spread(
    group: pd.DataFrame,
    spec: MechanismSpec,
    config: MechanismConfig,
) -> dict[str, Any]:
    work = group[[spec.dependent, spec.signal]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < config.min_cross_section:
        return {
            "high_minus_low_error": np.nan,
            "low_mean_error": np.nan,
            "high_mean_error": np.nan,
            "observations": int(len(work)),
        }
    low = work[spec.signal].quantile(0.2)
    high = work[spec.signal].quantile(0.8)
    low_group = work[work[spec.signal].le(low)]
    high_group = work[work[spec.signal].ge(high)]
    if low_group.empty or high_group.empty:
        return {
            "high_minus_low_error": np.nan,
            "low_mean_error": np.nan,
            "high_mean_error": np.nan,
            "observations": int(len(work)),
        }
    low_mean = float(low_group[spec.dependent].mean())
    high_mean = float(high_group[spec.dependent].mean())
    return {
        "high_minus_low_error": high_mean - low_mean,
        "low_mean_error": low_mean,
        "high_mean_error": high_mean,
        "observations": int(len(work)),
    }


def run_spread_tests_for_spec(
    frame: pd.DataFrame,
    spec: MechanismSpec,
    sample: str,
    config: MechanismConfig,
) -> dict[str, Any]:
    sample_frame, time_column = _sample_frame(frame, spec, sample, config)
    records = []
    if not sample_frame.empty:
        for time_value, group in sample_frame.groupby(time_column, sort=True):
            records.append(
                {
                    "sample": sample,
                    "spec": spec.name,
                    "time": time_value,
                    **_monthly_quintile_spread(group, spec, config),
                }
            )
    spreads = pd.DataFrame(records)
    inference = _hac_mean(
        spreads.get("high_minus_low_error", pd.Series(dtype=float)),
        config.hac_lags,
        config.min_hac_periods,
    )
    return {
        "sample": sample,
        "spec": spec.name,
        "dependent": spec.dependent,
        "signal": spec.signal,
        "channel": spec.channel,
        "horizon": spec.horizon,
        "placebo": spec.placebo,
        "expected_sign": spec.expected_sign,
        "mean_high_minus_low_error": inference["mean"],
        "expected_signed_spread": inference["mean"] * spec.expected_sign
        if pd.notna(inference["mean"])
        else np.nan,
        "hac_standard_error": inference["standard_error"],
        "t_stat": inference["t_stat"],
        "p_value": inference["p_value"],
        "ci_low": inference["ci_low"],
        "ci_high": inference["ci_high"],
        "periods": inference["periods"],
        "rows": int(len(sample_frame)),
        "firms": int(sample_frame["ric"].nunique()) if "ric" in sample_frame else 0,
        "average_cross_section": float(spreads["observations"].mean()) if not spreads.empty else np.nan,
    }


def sign_accuracy_for_spec(
    frame: pd.DataFrame,
    spec: MechanismSpec,
    sample: str,
    config: MechanismConfig,
) -> dict[str, Any]:
    sample_frame, _ = _sample_frame(frame, spec, sample, config)
    if sample_frame.empty:
        return {
            "sample": sample,
            "spec": spec.name,
            "dependent": spec.dependent,
            "signal": spec.signal,
            "observations": 0,
            "sign_accuracy": np.nan,
            "positive_error_fraction": np.nan,
            "positive_signal_fraction": np.nan,
        }
    work = sample_frame[[spec.dependent, spec.signal]].replace([np.inf, -np.inf], np.nan).dropna()
    nonzero = work[work[spec.dependent].ne(0) & work[spec.signal].ne(0)].copy()
    if nonzero.empty:
        accuracy = np.nan
    else:
        accuracy = float(
            (np.sign(nonzero[spec.dependent]) == np.sign(nonzero[spec.signal]) * spec.expected_sign).mean()
        )
    return {
        "sample": sample,
        "spec": spec.name,
        "dependent": spec.dependent,
        "signal": spec.signal,
        "channel": spec.channel,
        "horizon": spec.horizon,
        "placebo": spec.placebo,
        "observations": int(len(nonzero)),
        "sign_accuracy": accuracy,
        "positive_error_fraction": float(work[spec.dependent].gt(0).mean()) if not work.empty else np.nan,
        "positive_signal_fraction": float(work[spec.signal].gt(0).mean()) if not work.empty else np.nan,
    }


def run_specificity_tests(
    coefficients: pd.DataFrame,
    config: MechanismConfig,
) -> pd.DataFrame:
    records = []
    for sample in sorted(coefficients["sample"].dropna().unique()):
        sample_coefficients = coefficients[coefficients["sample"].eq(sample)]
        for pair in specificity_pairs():
            primary = sample_coefficients[
                sample_coefficients["spec"].eq(pair["primary_spec"])
            ][["time", "signal_slope"]].rename(columns={"signal_slope": "primary_slope"})
            placebo = sample_coefficients[
                sample_coefficients["spec"].eq(pair["placebo_spec"])
            ][["time", "signal_slope"]].rename(columns={"signal_slope": "placebo_slope"})
            merged = primary.merge(placebo, on="time", how="inner")
            merged["slope_difference"] = merged["primary_slope"] - merged["placebo_slope"]
            inference = _hac_mean(
                merged["slope_difference"],
                config.hac_lags,
                config.min_hac_periods,
            )
            records.append(
                {
                    "sample": sample,
                    "test": pair["name"],
                    "primary_spec": pair["primary_spec"],
                    "placebo_spec": pair["placebo_spec"],
                    "channel": pair["channel"],
                    "horizon": pair["horizon"],
                    "mean_slope_difference": inference["mean"],
                    "hac_standard_error": inference["standard_error"],
                    "t_stat": inference["t_stat"],
                    "p_value": inference["p_value"],
                    "ci_low": inference["ci_low"],
                    "ci_high": inference["ci_high"],
                    "periods": inference["periods"],
                    "positive_difference_fraction": float(
                        merged["slope_difference"].dropna().gt(0).mean()
                    )
                    if not merged.empty
                    else np.nan,
                    "primary_mean": float(merged["primary_slope"].mean())
                    if not merged.empty
                    else np.nan,
                    "placebo_mean": float(merged["placebo_slope"].mean())
                    if not merged.empty
                    else np.nan,
                }
            )
    return pd.DataFrame(records)


def run_forecast_error_mechanism_tests(
    panel_path: Path,
    output_dir: Path,
    config: MechanismConfig,
) -> dict[str, Any]:
    frame = pd.read_parquet(panel_path)
    specs = default_specs()
    joint_specs = default_joint_specs()
    required = sorted(
        {
            "ric",
            "date",
            "est_snapshot_date",
            *[spec.dependent for spec in specs],
            *[spec.valid_flag for spec in specs],
            *[spec.signal for spec in specs],
            *[spec.announce_date for spec in specs],
            *[spec.period_end for spec in specs],
            *[spec.dependent for spec in joint_specs],
            *[spec.valid_flag for spec in joint_specs],
            *[spec.primary_signal for spec in joint_specs],
            *[spec.competing_signal for spec in joint_specs],
            *[spec.announce_date for spec in joint_specs],
            *[spec.period_end for spec in joint_specs],
        }
    )
    _require_columns(frame, required)

    output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_tables = []
    joint_coefficient_tables = []
    summary_records = []
    joint_summary_records = []
    spread_records = []
    sign_records = []
    for sample in ("collapsed_firm_year", "overlapping_monthly"):
        for spec in specs:
            monthly, summary = run_fama_macbeth_for_spec(frame, spec, sample, config)
            coefficient_tables.append(monthly)
            summary_records.append(summary)
            spread_records.append(run_spread_tests_for_spec(frame, spec, sample, config))
            sign_records.append(sign_accuracy_for_spec(frame, spec, sample, config))
        for spec in joint_specs:
            monthly, summary = run_joint_fama_macbeth_for_spec(frame, spec, sample, config)
            joint_coefficient_tables.append(monthly)
            joint_summary_records.append(summary)

    coefficients = pd.concat(coefficient_tables, ignore_index=True)
    joint_coefficients = pd.concat(joint_coefficient_tables, ignore_index=True)
    summary = pd.DataFrame(summary_records)
    joint_summary = pd.DataFrame(joint_summary_records)
    spreads = pd.DataFrame(spread_records)
    sign_accuracy = pd.DataFrame(sign_records)
    specificity = run_specificity_tests(coefficients, config)

    coefficients.to_csv(output_dir / "mechanism_monthly_coefficients.csv", index=False)
    joint_coefficients.to_csv(
        output_dir / "mechanism_joint_signal_coefficients.csv",
        index=False,
    )
    summary.to_csv(output_dir / "mechanism_fama_macbeth_summary.csv", index=False)
    joint_summary.to_csv(output_dir / "mechanism_joint_signal_summary.csv", index=False)
    spreads.to_csv(output_dir / "mechanism_quintile_spread_tests.csv", index=False)
    sign_accuracy.to_csv(output_dir / "mechanism_sign_accuracy.csv", index=False)
    specificity.to_csv(output_dir / "mechanism_specificity_tests.csv", index=False)

    manifest = {
        "panel_path": str(panel_path),
        "config": asdict(config),
        "rows": {
            "input_panel": int(len(frame)),
            "specifications": int(len(specs)),
            "joint_specifications": int(len(joint_specs)),
            "coefficient_rows": int(len(coefficients)),
            "joint_coefficient_rows": int(len(joint_coefficients)),
            "summary_rows": int(len(summary)),
            "joint_summary_rows": int(len(joint_summary)),
            "spread_rows": int(len(spreads)),
            "sign_accuracy_rows": int(len(sign_accuracy)),
            "specificity_rows": int(len(specificity)),
        },
        "outputs": {
            "monthly_coefficients": str(output_dir / "mechanism_monthly_coefficients.csv"),
            "joint_signal_coefficients": str(
                output_dir / "mechanism_joint_signal_coefficients.csv"
            ),
            "fama_macbeth_summary": str(output_dir / "mechanism_fama_macbeth_summary.csv"),
            "joint_signal_summary": str(output_dir / "mechanism_joint_signal_summary.csv"),
            "quintile_spread_tests": str(output_dir / "mechanism_quintile_spread_tests.csv"),
            "sign_accuracy": str(output_dir / "mechanism_sign_accuracy.csv"),
            "specificity_tests": str(output_dir / "mechanism_specificity_tests.csv"),
            "manifest": str(output_dir / "manifest.json"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-cross-section", type=int, default=30)
    parser.add_argument("--hac-lags", type=int, default=15)
    parser.add_argument("--collapsed-lead-months", type=int, default=6)
    parser.add_argument("--min-hac-periods", type=int, default=12)
    parser.add_argument("--no-fixed-effects", action="store_true")
    parser.add_argument("--no-controls", action="store_true")
    args = parser.parse_args()

    config = MechanismConfig(
        min_cross_section=args.min_cross_section,
        hac_lags=args.hac_lags,
        collapsed_lead_months=args.collapsed_lead_months,
        min_hac_periods=args.min_hac_periods,
        include_fixed_effects=not args.no_fixed_effects,
        include_controls=not args.no_controls,
    )
    manifest = run_forecast_error_mechanism_tests(args.panel, args.output_dir, config)
    print(json.dumps(manifest["rows"], indent=2))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
