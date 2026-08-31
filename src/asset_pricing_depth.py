"""Economic-depth analysis for the European ML asset-pricing experiment.

This module works only with frozen out-of-sample predictions. It estimates
causal rolling risk measures, constructs EUR characteristic factors, and tests
whether ML scores contain information beyond momentum and conventional risks.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class DepthConfig:
    beta_window_months: int = 36
    beta_min_observations: int = 24
    factor_microcap_quantile: float = 0.05
    portfolio_cost_bps: int = 25
    hac_lags: int = 6
    minimum_cross_section: int = 100


FACTOR_COLUMNS = [
    "MKT_RF_EUR",
    "SMB_EUR",
    "HML_EUR",
    "RMW_EUR",
    "CMA_EUR",
    "MOM_EUR",
]

FMB_CHARACTERISTIC_CONTROLS = [
    "momentum_12_2_rank",
    "log_size_rank",
    "book_to_market_rank",
]

FMB_RISK_CONTROLS = [
    "beta_rank",
    "idio_vol_rank",
]


def load_eur_short_rate(path: Path) -> pd.Series:
    """Convert an annualised monthly EUR rate level to a one-month return.

    The rate observed in month t-1 is used as the cash return in month t.
    """
    frame = pd.read_csv(path)
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    annual_rate = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    rate = pd.Series(annual_rate.to_numpy(), index=dates, name="annual_rate_pct")
    rate = rate.dropna().sort_index()
    rate.index = rate.index.to_period("M").to_timestamp("M")
    return rate.shift(1).div(100.0 * 12.0).rename("RF_EUR")


def build_internal_market(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a EUR value-weighted market return using lagged market caps."""
    required = {"date", "ric", "return_1m", "company_market_cap"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Market construction missing columns: {sorted(missing)}")

    work = panel[list(required)].copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["ric", "date"])
    work["month_number"] = work["date"].dt.year * 12 + work["date"].dt.month
    grouped = work.groupby("ric", sort=False)
    previous_cap = grouped["company_market_cap"].shift(1)
    consecutive = work["month_number"].sub(
        grouped["month_number"].shift(1)
    ).eq(1)
    work["formation_cap"] = previous_cap.where(consecutive & previous_cap.gt(0))
    work["weighted_return"] = work["return_1m"] * work["formation_cap"]
    valid = work["return_1m"].notna() & work["formation_cap"].gt(0)
    market = (
        work.loc[valid]
        .groupby("date", as_index=False)
        .agg(
            weighted_return=("weighted_return", "sum"),
            market_cap=("formation_cap", "sum"),
            market_n=("ric", "nunique"),
            largest_formation_cap=("formation_cap", "max"),
        )
    )
    market["market_return_eur"] = market["weighted_return"].div(
        market["market_cap"]
    )
    market["largest_weight"] = market["largest_formation_cap"].div(
        market["market_cap"]
    )
    return market[
        ["date", "market_return_eur", "market_n", "market_cap", "largest_weight"]
    ].sort_values("date")


def _rolling_risk_for_security(
    security: pd.DataFrame,
    market: pd.Series,
    window: int,
    minimum: int,
) -> pd.DataFrame:
    start = security["date"].min()
    end = security["date"].max()
    dates = pd.date_range(start, end, freq="ME")
    returns = (
        security.drop_duplicates("date", keep="last")
        .set_index("date")["return_1m"]
        .reindex(dates)
    )
    market_returns = market.reindex(dates)
    valid = returns.notna() & market_returns.notna()
    x = market_returns.where(valid)
    y = returns.where(valid)

    n = valid.astype(float).rolling(window, min_periods=minimum).sum()
    sx = x.rolling(window, min_periods=minimum).sum()
    sy = y.rolling(window, min_periods=minimum).sum()
    sxx = x.pow(2).rolling(window, min_periods=minimum).sum()
    syy = y.pow(2).rolling(window, min_periods=minimum).sum()
    sxy = x.mul(y).rolling(window, min_periods=minimum).sum()
    centered_xx = sxx - sx.pow(2).div(n)
    centered_yy = syy - sy.pow(2).div(n)
    centered_xy = sxy - sx.mul(sy).div(n)
    beta = centered_xy.div(centered_xx.where(centered_xx.gt(0)))
    residual_ss = centered_yy - beta.mul(centered_xy)
    idio_vol = np.sqrt(
        residual_ss.clip(lower=0).div((n - 2).where(n.gt(2)))
    )

    result = pd.DataFrame(
        {
            "date": dates,
            "beta_36m": beta.to_numpy(),
            "idio_vol_36m": idio_vol.to_numpy(),
            "risk_nobs": n.to_numpy(),
        }
    )
    original_dates = pd.Index(security["date"].unique())
    return result[result["date"].isin(original_dates)]


def estimate_rolling_risk(
    panel: pd.DataFrame,
    market: pd.DataFrame,
    window: int = 36,
    minimum: int = 24,
) -> pd.DataFrame:
    """Estimate trailing market beta and residual volatility through month t."""
    required = {"date", "ric", "return_1m"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Risk estimation missing columns: {sorted(missing)}")
    if minimum < 3 or minimum > window:
        raise ValueError("minimum observations must be between 3 and window")

    market_series = market.set_index("date")["market_return_eur"].sort_index()
    work = panel[["date", "ric", "return_1m"]].copy()
    work["date"] = pd.to_datetime(work["date"])
    parts = []
    for ric, security in work.groupby("ric", sort=False):
        risk = _rolling_risk_for_security(
            security.sort_values("date"),
            market_series,
            window,
            minimum,
        )
        risk["ric"] = ric
        parts.append(risk)
    result = pd.concat(parts, ignore_index=True)
    for source, target in [
        ("beta_36m", "beta_rank"),
        ("idio_vol_36m", "idio_vol_rank"),
    ]:
        result[target] = (
            result.groupby("date")[source]
            .rank(method="average", pct=True)
            .mul(2.0)
            .sub(1.0)
        )
    return result[
        [
            "date",
            "ric",
            "beta_36m",
            "idio_vol_36m",
            "risk_nobs",
            "beta_rank",
            "idio_vol_rank",
        ]
    ].sort_values(["date", "ric"])


def _weighted_average_return(
    frame: pd.DataFrame,
    return_column: str = "target_return_1m",
) -> float:
    valid = frame[return_column].notna() & frame["company_market_cap"].gt(0)
    if not valid.any():
        return np.nan
    values = frame.loc[valid, return_column]
    weights = frame.loc[valid, "company_market_cap"]
    return float(np.average(values, weights=weights))


def _two_by_three_sort(
    month: pd.DataFrame,
    characteristic: str,
    low_minus_high: bool = False,
) -> tuple[float, float, int]:
    valid = month.dropna(
        subset=[characteristic, "company_market_cap", "target_return_1m"]
    )
    valid = valid[valid["company_market_cap"].gt(0)]
    if len(valid) < 30 or valid[characteristic].nunique() < 3:
        return np.nan, np.nan, 0

    size_break = valid["company_market_cap"].median()
    low_break, high_break = valid[characteristic].quantile([0.3, 0.7])
    size_group = np.where(
        valid["company_market_cap"].le(size_break), "small", "big"
    )
    style_group = np.select(
        [
            valid[characteristic].le(low_break),
            valid[characteristic].ge(high_break),
        ],
        ["low", "high"],
        default="neutral",
    )
    sorted_frame = valid.assign(
        size_group=size_group,
        style_group=style_group,
    )
    cell_returns = {
        (size, style): _weighted_average_return(cell)
        for (size, style), cell in sorted_frame.groupby(
            ["size_group", "style_group"], sort=False
        )
    }
    needed = {
        (size, style)
        for size in ("small", "big")
        for style in ("low", "neutral", "high")
    }
    if not needed.issubset(cell_returns):
        return np.nan, np.nan, len(valid)

    small = np.mean(
        [cell_returns[("small", style)] for style in ("low", "neutral", "high")]
    )
    big = np.mean(
        [cell_returns[("big", style)] for style in ("low", "neutral", "high")]
    )
    low = np.mean(
        [cell_returns[(size, "low")] for size in ("small", "big")]
    )
    high = np.mean(
        [cell_returns[(size, "high")] for size in ("small", "big")]
    )
    style_return = low - high if low_minus_high else high - low
    return float(small - big), float(style_return), len(valid)


def build_internal_eur_factors(
    panel: pd.DataFrame,
    eur_rf: pd.Series,
    minimum_size_percentile: float = 0.05,
) -> pd.DataFrame:
    """Construct compact Fama-French-style factors from EUR stock returns.

    This is an internally consistent spanning set, not an exact replication of
    the Kenneth French international factor methodology.
    """
    columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "company_market_cap",
        "market_cap_percentile",
        "book_to_market",
        "operating_profitability",
        "asset_growth",
        "momentum_12_2",
    ]
    work = panel[columns].copy()
    work = work[
        work["market_cap_percentile"].ge(minimum_size_percentile)
        & work["target_return_1m"].notna()
        & work["target_date"].notna()
    ]
    records = []
    for signal_date, month in work.groupby("date", sort=True):
        hml_smb, hml, hml_n = _two_by_three_sort(month, "book_to_market")
        rmw_smb, rmw, rmw_n = _two_by_three_sort(
            month, "operating_profitability"
        )
        cma_smb, cma, cma_n = _two_by_three_sort(
            month, "asset_growth", low_minus_high=True
        )
        _, momentum, momentum_n = _two_by_three_sort(
            month, "momentum_12_2"
        )
        smb_components = np.asarray(
            [hml_smb, rmw_smb, cma_smb], dtype=float
        )
        smb = (
            float(np.nanmean(smb_components))
            if np.isfinite(smb_components).any()
            else np.nan
        )
        records.append(
            {
                "signal_date": signal_date,
                "return_date": month["target_date"].iloc[0],
                "MKT_EUR": _weighted_average_return(month),
                "SMB_EUR": smb,
                "HML_EUR": hml,
                "RMW_EUR": rmw,
                "CMA_EUR": cma,
                "MOM_EUR": momentum,
                "market_n": int(month["ric"].nunique()),
                "hml_n": hml_n,
                "rmw_n": rmw_n,
                "cma_n": cma_n,
                "momentum_n": momentum_n,
            }
        )
    factors = pd.DataFrame(records).sort_values("return_date")
    rf = eur_rf.reindex(pd.DatetimeIndex(factors["return_date"])).to_numpy()
    factors["RF_EUR"] = rf
    factors["MKT_RF_EUR"] = factors["MKT_EUR"] - factors["RF_EUR"]
    return factors


def prepare_analysis_predictions(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    risk: pd.DataFrame,
) -> pd.DataFrame:
    """Attach time-t controls and causal risk estimates to OOS predictions."""
    controls = panel[
        [
            "date",
            "ric",
            "log_size_rank",
            "book_to_market_rank",
            "momentum_12_2_rank",
            "turnover_12m_rank",
        ]
    ].drop_duplicates(["date", "ric"], keep="last")
    controls = controls.merge(
        risk,
        on=["date", "ric"],
        how="left",
        validate="one_to_one",
    )
    merged = predictions.merge(
        controls,
        on=["date", "ric"],
        how="left",
        validate="many_to_one",
    )
    merged["prediction_rank"] = (
        merged.groupby(["model", "date"])["prediction"]
        .rank(method="average", pct=True)
        .mul(2.0)
        .sub(1.0)
    )
    return merged


def _hac_mean(values: pd.Series, lags: int) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < max(12, lags + 2):
        return {
            "mean": np.nan,
            "standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "months": int(len(clean)),
        }
    fit = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(
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
        "months": int(len(clean)),
    }


def _monthly_cross_sectional_slope(
    month: pd.DataFrame,
    controls: list[str],
    fixed_effects: bool,
    minimum_cross_section: int,
) -> tuple[float, int]:
    columns = ["target_return_1m", "prediction_rank", *controls]
    if fixed_effects:
        columns.extend(["screen_country", "TR.TRBCECONOMICSECTOR"])
    work = month[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < minimum_cross_section:
        return np.nan, len(work)

    x = work[["prediction_rank", *controls]].astype(float)
    if fixed_effects:
        dummies = pd.get_dummies(
            work[["screen_country", "TR.TRBCECONOMICSECTOR"]].astype(str),
            drop_first=True,
            dtype=float,
        )
        x = pd.concat([x.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    x.insert(0, "constant", 1.0)
    y = work["target_return_1m"].to_numpy(dtype=float)
    coefficients, _, _, _ = np.linalg.lstsq(
        x.to_numpy(dtype=float),
        y,
        rcond=None,
    )
    prediction_position = x.columns.get_loc("prediction_rank")
    return float(coefficients[prediction_position]), len(work)


def fama_macbeth_tests(
    analysis_predictions: pd.DataFrame,
    config: DepthConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run monthly cross-sectional regressions and HAC-average score slopes."""
    monthly_records = []
    specifications = [
        ("univariate", [], False),
        ("characteristics", FMB_CHARACTERISTIC_CONTROLS, False),
        (
            "characteristics_risk_country_sector",
            [*FMB_CHARACTERISTIC_CONTROLS, *FMB_RISK_CONTROLS],
            True,
        ),
    ]
    for (model, date), month in analysis_predictions.groupby(
        ["model", "date"], sort=True
    ):
        for specification, controls, fixed_effects in specifications:
            model_controls = list(controls)
            if model == "momentum_rank":
                model_controls = [
                    column
                    for column in model_controls
                    if column != "momentum_12_2_rank"
                ]
            slope, observations = _monthly_cross_sectional_slope(
                month,
                model_controls,
                fixed_effects,
                config.minimum_cross_section,
            )
            monthly_records.append(
                {
                    "model": model,
                    "date": date,
                    "specification": specification,
                    "score_slope": slope,
                    "observations": observations,
                    "controls": ",".join(model_controls),
                    "country_sector_fixed_effects": fixed_effects,
                }
            )

    monthly = pd.DataFrame(monthly_records)
    summary_records = []
    for (model, specification), group in monthly.groupby(
        ["model", "specification"], sort=True
    ):
        inference = _hac_mean(group["score_slope"], config.hac_lags)
        summary_records.append(
            {
                "model": model,
                "specification": specification,
                "mean_monthly_score_slope": inference["mean"],
                "annualized_score_slope": inference["mean"] * 12,
                "hac_standard_error": inference["standard_error"],
                "t_stat": inference["t_stat"],
                "p_value": inference["p_value"],
                "ci_low": inference["ci_low"],
                "ci_high": inference["ci_high"],
                "months": inference["months"],
                "average_cross_section": float(group["observations"].mean()),
                "positive_month_fraction": float(
                    group["score_slope"].dropna().gt(0).mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_records)
    summary["p_value_holm"] = summary.groupby("specification")[
        "p_value"
    ].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    return monthly, summary


def _hac_factor_regression(
    frame: pd.DataFrame,
    dependent: str,
    factor_columns: Iterable[str],
    lags: int,
) -> dict[str, float]:
    factors = list(factor_columns)
    clean = frame[[dependent, *factors]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(clean) < max(24, len(factors) + lags + 2):
        return {"observations": int(len(clean))}
    x = sm.add_constant(clean[factors], has_constant="add")
    fit = sm.OLS(clean[dependent], x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags},
    )
    result = {
        "observations": int(fit.nobs),
        "alpha_monthly": float(fit.params["const"]),
        "alpha_annualized": float(fit.params["const"] * 12),
        "alpha_t": float(fit.tvalues["const"]),
        "alpha_p": float(fit.pvalues["const"]),
        "adjusted_r2": float(fit.rsquared_adj),
    }
    for factor in factors:
        result[f"beta_{factor}"] = float(fit.params[factor])
        result[f"p_{factor}"] = float(fit.pvalues[factor])
    return result


def factor_spanning_tests(
    monthly_portfolios: pd.DataFrame,
    factors: pd.DataFrame,
    config: DepthConfig,
) -> pd.DataFrame:
    """Test absolute portfolio alpha and ML-minus-momentum spanning."""
    merged = monthly_portfolios.merge(
        factors,
        on=["signal_date", "return_date"],
        how="left",
        validate="many_to_one",
    )
    cost_rate = config.portfolio_cost_bps / 10_000.0
    merged["net_long_short"] = (
        merged["gross_long_short_return"]
        - merged["long_short_turnover"] * cost_rate
    )
    merged["net_long_only_excess"] = (
        merged["long_return"]
        - merged["long_only_turnover"] * cost_rate
        - merged["RF_EUR"]
    )
    records = []
    keys = ["weighting", "universe_variant"]
    for key, subset in merged.groupby(keys, sort=True):
        weighting, universe_variant = key
        for model, model_frame in subset.groupby("model", sort=True):
            for portfolio, dependent in [
                ("long_short", "net_long_short"),
                ("long_only_top_decile", "net_long_only_excess"),
            ]:
                regression = _hac_factor_regression(
                    model_frame,
                    dependent,
                    FACTOR_COLUMNS,
                    config.hac_lags,
                )
                records.append(
                    {
                        "comparison": "absolute",
                        "model": model,
                        "baseline": "",
                        "weighting": weighting,
                        "universe_variant": universe_variant,
                        "portfolio": portfolio,
                        "cost_bps": config.portfolio_cost_bps,
                        **regression,
                    }
                )

        baseline = subset[subset["model"].eq("momentum_rank")].set_index(
            "return_date"
        )
        if baseline.empty:
            continue
        for model in sorted(set(subset["model"]) - {"momentum_rank"}):
            candidate = subset[subset["model"].eq(model)].set_index("return_date")
            common = candidate.index.intersection(baseline.index)
            if len(common) < 24:
                continue
            for portfolio, dependent in [
                ("long_short", "net_long_short"),
                ("long_only_top_decile", "net_long_only_excess"),
            ]:
                spread = candidate.loc[common].copy()
                spread["strategy_spread"] = (
                    candidate.loc[common, dependent].to_numpy()
                    - baseline.loc[common, dependent].to_numpy()
                )
                regression = _hac_factor_regression(
                    spread,
                    "strategy_spread",
                    FACTOR_COLUMNS,
                    config.hac_lags,
                )
                records.append(
                    {
                        "comparison": "model_minus_momentum",
                        "model": model,
                        "baseline": "momentum_rank",
                        "weighting": weighting,
                        "universe_variant": universe_variant,
                        "portfolio": portfolio,
                        "cost_bps": config.portfolio_cost_bps,
                        **regression,
                    }
                )
    result = pd.DataFrame(records)
    family = [
        "comparison",
        "weighting",
        "universe_variant",
        "portfolio",
        "cost_bps",
    ]
    result["alpha_p_holm"] = result.groupby(family)["alpha_p"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    return result


def _tercile_labels(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, -1.0 / 3.0, 1.0 / 3.0, np.inf],
        labels=["low", "middle", "high"],
    )


def conditional_predictability(
    analysis_predictions: pd.DataFrame,
    config: DepthConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Locate OOS predictability across size, beta, idiosyncratic risk and turnover."""
    work = analysis_predictions.copy()
    work["size_group"] = pd.cut(
        work["market_cap_percentile"],
        bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
        labels=["small", "middle", "large"],
    )
    work["beta_group"] = _tercile_labels(work["beta_rank"])
    work["idio_vol_group"] = _tercile_labels(work["idio_vol_rank"])
    work["turnover_group"] = _tercile_labels(work["turnover_12m_rank"])

    monthly_records = []
    for dimension in [
        "size_group",
        "beta_group",
        "idio_vol_group",
        "turnover_group",
    ]:
        subset = work.dropna(
            subset=[dimension, "prediction", "target_return_rank"]
        )
        for (model, date, group_name), month in subset.groupby(
            ["model", "date", dimension],
            observed=True,
            sort=True,
        ):
            if len(month) < config.minimum_cross_section:
                continue
            ic = month["prediction"].corr(
                month["target_return_rank"],
                method="spearman",
            )
            score_rank = month["prediction"].rank(method="first", pct=True)
            long_return = month.loc[
                score_rank.gt(0.8), "target_return_1m"
            ].mean()
            short_return = month.loc[
                score_rank.le(0.2), "target_return_1m"
            ].mean()
            monthly_records.append(
                {
                    "model": model,
                    "date": date,
                    "dimension": dimension.removesuffix("_group"),
                    "group": str(group_name),
                    "observations": len(month),
                    "spearman_ic": ic,
                    "gross_equal_weight_spread": long_return - short_return,
                }
            )
    monthly = pd.DataFrame(monthly_records)
    summary_records = []
    for (model, dimension, group_name), group in monthly.groupby(
        ["model", "dimension", "group"], sort=True
    ):
        ic_inference = _hac_mean(group["spearman_ic"], config.hac_lags)
        spread = group["gross_equal_weight_spread"].dropna()
        spread_volatility = float(spread.std(ddof=1) * np.sqrt(12))
        summary_records.append(
            {
                "model": model,
                "dimension": dimension,
                "group": group_name,
                "months": int(len(group)),
                "average_cross_section": float(group["observations"].mean()),
                "mean_ic": ic_inference["mean"],
                "ic_t_stat": ic_inference["t_stat"],
                "ic_p_value": ic_inference["p_value"],
                "annualized_gross_spread": float(spread.mean() * 12),
                "annualized_spread_volatility": spread_volatility,
                "gross_spread_sharpe": (
                    float(spread.mean() / spread.std(ddof=1) * np.sqrt(12))
                    if spread.std(ddof=1) > 0
                    else np.nan
                ),
            }
        )
    contrasts = []
    contrast_definitions = {
        "size": ("small", "large"),
        "beta": ("high", "low"),
        "idio_vol": ("high", "low"),
        "turnover": ("high", "low"),
    }
    for (model, dimension), group in monthly.groupby(
        ["model", "dimension"], sort=True
    ):
        positive_group, negative_group = contrast_definitions[dimension]
        pivot_ic = group.pivot(
            index="date", columns="group", values="spearman_ic"
        )
        pivot_spread = group.pivot(
            index="date",
            columns="group",
            values="gross_equal_weight_spread",
        )
        if not {
            positive_group,
            negative_group,
        }.issubset(pivot_ic.columns):
            continue
        ic_difference = (
            pivot_ic[positive_group] - pivot_ic[negative_group]
        )
        spread_difference = (
            pivot_spread[positive_group] - pivot_spread[negative_group]
        )
        ic_inference = _hac_mean(ic_difference, config.hac_lags)
        spread_inference = _hac_mean(spread_difference, config.hac_lags)
        contrasts.append(
            {
                "model": model,
                "dimension": dimension,
                "contrast": f"{positive_group}_minus_{negative_group}",
                "months": ic_inference["months"],
                "ic_difference": ic_inference["mean"],
                "ic_difference_t": ic_inference["t_stat"],
                "ic_difference_p": ic_inference["p_value"],
                "annualized_spread_difference": (
                    spread_inference["mean"] * 12
                ),
                "spread_difference_t": spread_inference["t_stat"],
                "spread_difference_p": spread_inference["p_value"],
            }
        )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame["ic_difference_p_holm"] = contrast_frame.groupby(
        "dimension"
    )["ic_difference_p"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    contrast_frame["spread_difference_p_holm"] = contrast_frame.groupby(
        "dimension"
    )["spread_difference_p"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    return monthly, pd.DataFrame(summary_records), contrast_frame


def build_market_states(
    market: pd.DataFrame,
    minimum_history: int = 24,
) -> pd.DataFrame:
    """Construct ex-ante market states with expanding historical thresholds."""
    state = market[["date", "market_return_eur"]].sort_values("date").copy()
    log_return = np.log1p(state["market_return_eur"])
    state["market_trend_12m"] = np.expm1(
        log_return.rolling(12, min_periods=12).sum()
    )
    state["market_volatility_12m"] = (
        state["market_return_eur"].rolling(12, min_periods=12).std()
        * np.sqrt(12)
    )
    historical_volatility_median = (
        state["market_volatility_12m"]
        .expanding(min_periods=minimum_history)
        .median()
        .shift(1)
    )
    state["volatility_state"] = np.where(
        historical_volatility_median.isna(),
        None,
        np.where(
            state["market_volatility_12m"].gt(historical_volatility_median),
            "high",
            "low",
        ),
    )
    state["trend_state"] = np.where(
        state["market_trend_12m"].isna(),
        None,
        np.where(state["market_trend_12m"].lt(0), "down", "up"),
    )
    return state.rename(columns={"date": "signal_date"})


def state_dependence_tests(
    analysis_predictions: pd.DataFrame,
    monthly_portfolios: pd.DataFrame,
    states: pd.DataFrame,
    config: DepthConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate IC and value-weight net returns in pre-observable market states."""
    state_columns = ["signal_date", "volatility_state", "trend_state"]
    prediction_state = analysis_predictions.merge(
        states[state_columns],
        left_on="date",
        right_on="signal_date",
        how="left",
        validate="many_to_one",
    )
    monthly_ic = (
        prediction_state.groupby(["model", "date"], sort=True)
        .apply(
            lambda month: month["prediction"].corr(
                month["target_return_rank"], method="spearman"
            ),
            include_groups=False,
        )
        .rename("spearman_ic")
        .reset_index()
    )
    monthly_ic = monthly_ic.merge(
        states[state_columns],
        left_on="date",
        right_on="signal_date",
        how="left",
        validate="many_to_one",
    )

    portfolio = monthly_portfolios[
        monthly_portfolios["weighting"].eq("value")
        & monthly_portfolios["universe_variant"].eq(
            "standard_ex_bottom_5pct"
        )
    ].copy()
    portfolio["net_long_short"] = (
        portfolio["gross_long_short_return"]
        - portfolio["long_short_turnover"]
        * config.portfolio_cost_bps
        / 10_000.0
    )
    portfolio = portfolio.merge(
        states[state_columns],
        on="signal_date",
        how="left",
        validate="many_to_one",
    )

    monthly_records = []
    summary_records = []
    for state_variable in ["volatility_state", "trend_state"]:
        ic_part = monthly_ic.dropna(subset=[state_variable])
        return_part = portfolio.dropna(subset=[state_variable])
        for (model, state_name), group in ic_part.groupby(
            ["model", state_variable], sort=True
        ):
            inference = _hac_mean(group["spearman_ic"], config.hac_lags)
            returns = return_part[
                return_part["model"].eq(model)
                & return_part[state_variable].eq(state_name)
            ]["net_long_short"]
            monthly_records.extend(
                {
                    "model": model,
                    "date": row["date"],
                    "state_variable": state_variable,
                    "state": state_name,
                    "spearman_ic": row["spearman_ic"],
                }
                for _, row in group.iterrows()
            )
            summary_records.append(
                {
                    "model": model,
                    "state_variable": state_variable,
                    "state": state_name,
                    "months": inference["months"],
                    "mean_ic": inference["mean"],
                    "ic_t_stat": inference["t_stat"],
                    "ic_p_value": inference["p_value"],
                    "annualized_net_long_short_return": float(
                        returns.mean() * 12
                    ),
                    "net_long_short_sharpe": (
                        float(
                            returns.mean()
                            / returns.std(ddof=1)
                            * np.sqrt(12)
                        )
                        if returns.std(ddof=1) > 0
                        else np.nan
                    ),
                }
            )
    contrasts = []
    state_contrasts = {
        "volatility_state": ("high", "low"),
        "trend_state": ("down", "up"),
    }
    for model in sorted(monthly_ic["model"].unique()):
        for state_variable, (
            positive_state,
            negative_state,
        ) in state_contrasts.items():
            ic_model = monthly_ic[monthly_ic["model"].eq(model)]
            ic_pivot = ic_model.pivot(
                index="date",
                columns=state_variable,
                values="spearman_ic",
            )
            # States are mutually exclusive, so compare their sample means by
            # regressing the monthly metric on a state indicator with HAC errors.
            ic_sample = ic_model.dropna(subset=[state_variable]).copy()
            return_sample = portfolio[
                portfolio["model"].eq(model)
            ].dropna(subset=[state_variable]).copy()
            contrast_record = {
                "model": model,
                "state_variable": state_variable,
                "contrast": f"{positive_state}_minus_{negative_state}",
            }
            for label, sample, value in [
                ("ic", ic_sample, "spearman_ic"),
                ("return", return_sample, "net_long_short"),
            ]:
                indicator = sample[state_variable].eq(
                    positive_state
                ).astype(float)
                x = sm.add_constant(indicator, has_constant="add")
                fit = sm.OLS(sample[value], x).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": config.hac_lags},
                )
                coefficient = float(fit.params.iloc[1])
                contrast_record[f"{label}_difference"] = coefficient
                contrast_record[f"{label}_difference_t"] = float(
                    fit.tvalues.iloc[1]
                )
                contrast_record[f"{label}_difference_p"] = float(
                    fit.pvalues.iloc[1]
                )
            contrast_record["annualized_return_difference"] = (
                contrast_record["return_difference"] * 12
            )
            contrasts.append(contrast_record)
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame["ic_difference_p_holm"] = contrast_frame.groupby(
        "state_variable"
    )["ic_difference_p"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    contrast_frame["return_difference_p_holm"] = contrast_frame.groupby(
        "state_variable"
    )["return_difference_p"].transform(
        lambda values: multipletests(values, method="holm")[1]
    )
    return (
        pd.DataFrame(monthly_records),
        pd.DataFrame(summary_records),
        contrast_frame,
    )


def write_depth_outputs(
    output_dir: Path,
    config: DepthConfig,
    market: pd.DataFrame,
    risk: pd.DataFrame,
    factors: pd.DataFrame,
    fmb_monthly: pd.DataFrame,
    fmb_summary: pd.DataFrame,
    spanning: pd.DataFrame,
    conditional_monthly: pd.DataFrame,
    conditional_summary: pd.DataFrame,
    conditional_contrasts: pd.DataFrame,
    states: pd.DataFrame,
    state_monthly: pd.DataFrame,
    state_summary: pd.DataFrame,
    state_contrasts: pd.DataFrame,
    input_paths: dict[str, Path],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    risk.to_parquet(
        output_dir / "rolling_risk_estimates.parquet",
        index=False,
        compression="zstd",
    )
    market.to_csv(output_dir / "eur_market_return.csv", index=False)
    factors.to_csv(output_dir / "internal_eur_factors.csv", index=False)
    fmb_monthly.to_csv(output_dir / "fama_macbeth_monthly.csv", index=False)
    fmb_summary.to_csv(output_dir / "fama_macbeth_summary.csv", index=False)
    spanning.to_csv(output_dir / "factor_spanning.csv", index=False)
    conditional_monthly.to_csv(
        output_dir / "conditional_predictability_monthly.csv", index=False
    )
    conditional_summary.to_csv(
        output_dir / "conditional_predictability_summary.csv", index=False
    )
    conditional_contrasts.to_csv(
        output_dir / "conditional_predictability_contrasts.csv", index=False
    )
    states.to_csv(output_dir / "market_states.csv", index=False)
    state_monthly.to_csv(
        output_dir / "state_dependence_monthly.csv", index=False
    )
    state_summary.to_csv(
        output_dir / "state_dependence_summary.csv", index=False
    )
    state_contrasts.to_csv(
        output_dir / "state_dependence_contrasts.csv", index=False
    )
    manifest = {
        "config": asdict(config),
        "inputs": {name: str(path) for name, path in input_paths.items()},
        "rows": {
            "market_months": int(len(market)),
            "risk_estimates": int(len(risk)),
            "factor_months": int(len(factors)),
            "fama_macbeth_monthly": int(len(fmb_monthly)),
            "factor_spanning_tests": int(len(spanning)),
            "conditional_monthly": int(len(conditional_monthly)),
            "state_monthly": int(len(state_monthly)),
        },
        "causality": {
            "rolling_risk_uses_returns_through_signal_month": True,
            "factor_returns_use_signal_month_characteristics_for_next_month": True,
            "state_volatility_threshold_uses_expanding_history_shifted_one_month": True,
        },
    }
    (output_dir / "depth_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
