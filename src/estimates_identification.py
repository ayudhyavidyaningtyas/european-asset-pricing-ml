"""Identification diagnostics for the analyst-estimates layer.

Three separate questions share this machinery:

* coverage selection -- analyst coverage is not random, so the estimates lift
  measured on covered stock-months has to be shown to survive reweighting the
  covered sample back to the full investable universe;
* signal timing -- how fast the lift decays as the analyst snapshot is stale-dated;
* attribution -- which analyst feature family carries the lift.

The information-coefficient helpers accept an optional weight column so the
weighted and unweighted reads use one code path. With equal weights the
weighted formula reduces exactly to the population-moment Spearman correlation
used by the Test B interaction script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

# Two-sided 5% critical value plus the 80% power quantile: the smallest true
# effect this many standard errors wide would be detected 80% of the time.
MDE_MULTIPLIER = 1.959963985 + 0.841621234
MIN_HAC_MONTHS = 24

COVERAGE_PROPENSITY_CONTINUOUS = [
    "log_size_rank",
    "log_trading_value_eur_rank",
    "turnover_12m_rank",
    "volatility_12m_rank",
    "book_to_market_rank",
    "momentum_12_2_rank",
]
COVERAGE_PROPENSITY_CATEGORICAL = ["screen_country", "TR.TRBCECONOMICSECTOR"]


def monthly_ic(
    frame: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    target_column: str = "target_return_1m",
    group_columns: tuple[str, ...] = ("base_model", "date"),
    weight_column: str | None = None,
) -> pd.DataFrame:
    """Monthly rank IC per group, optionally weighted.

    Predictions and realised returns are ranked within each group, then combined
    with population moments so that unit weights reproduce the plain Spearman
    correlation.
    """
    keys = list(group_columns)
    working = frame.dropna(subset=[prediction_column, target_column]).copy()
    grouped = working.groupby(keys, observed=True)
    working["x"] = grouped[prediction_column].rank()
    working["y"] = grouped[target_column].rank()
    if weight_column is None:
        working["w"] = 1.0
    else:
        working["w"] = pd.to_numeric(working[weight_column], errors="coerce")
        working = working[working["w"].gt(0) & np.isfinite(working["w"])].copy()
    working["wx"] = working["w"] * working["x"]
    working["wy"] = working["w"] * working["y"]
    working["wxy"] = working["w"] * working["x"] * working["y"]
    working["wxx"] = working["w"] * working["x"] ** 2
    working["wyy"] = working["w"] * working["y"] ** 2
    working["ww"] = working["w"] ** 2

    aggregated = working.groupby(keys, observed=True).agg(
        names=("x", "size"),
        weight_sum=("w", "sum"),
        weight_square_sum=("ww", "sum"),
        sum_wx=("wx", "sum"),
        sum_wy=("wy", "sum"),
        sum_wxy=("wxy", "sum"),
        sum_wxx=("wxx", "sum"),
        sum_wyy=("wyy", "sum"),
    )
    total = aggregated["weight_sum"].replace(0.0, np.nan)
    mean_x = aggregated["sum_wx"] / total
    mean_y = aggregated["sum_wy"] / total
    covariance = aggregated["sum_wxy"] / total - mean_x * mean_y
    variance_x = aggregated["sum_wxx"] / total - mean_x**2
    variance_y = aggregated["sum_wyy"] / total - mean_y**2
    denominator = np.sqrt(variance_x.clip(lower=0.0) * variance_y.clip(lower=0.0))
    aggregated["ic"] = covariance / denominator.replace(0.0, np.nan)
    aggregated["effective_names"] = (
        aggregated["weight_sum"] ** 2 / aggregated["weight_square_sum"]
    )
    return aggregated.reset_index()[
        [*keys, "names", "weight_sum", "effective_names", "ic"]
    ]


def hac_mean(
    series: pd.Series,
    hac_lags: int,
    label: str,
    *,
    min_months: int = MIN_HAC_MONTHS,
) -> dict[str, object]:
    """HAC mean test of a monthly difference series, reported with its MDE."""
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < min_months:
        return {"quantity": label, "months": int(len(clean))}
    fit = sm.OLS(clean, np.ones(len(clean))).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    estimate = float(fit.params.iloc[0])
    standard_error = float(fit.bse.iloc[0])
    return {
        "quantity": label,
        "months": int(len(clean)),
        "estimate": estimate,
        "standard_error": standard_error,
        "t_stat": float(fit.tvalues.iloc[0]),
        "p_value": float(fit.pvalues.iloc[0]),
        "ci_low": estimate - 1.959963985 * standard_error,
        "ci_high": estimate + 1.959963985 * standard_error,
        "minimum_detectable_effect": MDE_MULTIPLIER * standard_error,
    }


def holm_within(
    frame: pd.DataFrame,
    group_columns: list[str],
    p_column: str = "p_value",
    output_column: str = "p_value_holm",
) -> pd.DataFrame:
    """Holm-adjust ``p_column`` within each group, leaving untested rows null."""
    out = frame.copy()
    out[output_column] = np.nan
    if out.empty or p_column not in out:
        return out
    testable = out[p_column].notna()
    if not testable.any():
        return out
    out.loc[testable, output_column] = (
        out[testable]
        .groupby(group_columns, observed=True)[p_column]
        .transform(lambda values: multipletests(values, method="holm")[1])
    )
    return out


def _design_matrix(
    frame: pd.DataFrame,
    continuous_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    parts = [
        frame[continuous_columns]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    ]
    available = [column for column in categorical_columns if column in frame]
    if available:
        categories = frame[available].fillna("Unknown").astype(str)
        parts.append(
            pd.get_dummies(categories, columns=available, drop_first=True, dtype=float)
        )
    return pd.concat(parts, axis=1)


def fit_monthly_coverage_propensity(
    panel: pd.DataFrame,
    *,
    covered_column: str = "is_covered",
    date_column: str = "date",
    continuous_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    regularization_c: float = 1.0,
    min_month_rows: int = 100,
    min_month_events: int = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    """Cross-sectional logit of analyst coverage on observable characteristics.

    Fitting month by month mirrors the cross-sectional design of the rest of the
    panel: the coverage rate and its country/sector composition drift over the
    sample, and a pooled fit would attribute that drift to the characteristics.
    Months that cannot support a fit fall back to the observed coverage rate.
    """
    continuous_columns = continuous_columns or COVERAGE_PROPENSITY_CONTINUOUS
    categorical_columns = (
        COVERAGE_PROPENSITY_CATEGORICAL
        if categorical_columns is None
        else categorical_columns
    )
    propensity = pd.Series(np.nan, index=panel.index, dtype=float)
    records: list[dict[str, object]] = []

    for date, month in panel.groupby(date_column, observed=True):
        covered = month[covered_column].astype(bool)
        rate = float(covered.mean())
        events = int(covered.sum())
        non_events = int((~covered).sum())
        fitted = False
        auc = np.nan
        if (
            len(month) >= min_month_rows
            and events >= min_month_events
            and non_events >= min_month_events
        ):
            x = _design_matrix(month, continuous_columns, categorical_columns)
            model = LogisticRegression(
                C=regularization_c,
                max_iter=1_000,
                solver="lbfgs",
            )
            model.fit(x.to_numpy(dtype=float), covered.to_numpy())
            scores = model.predict_proba(x.to_numpy(dtype=float))[:, 1]
            propensity.loc[month.index] = scores
            auc = float(roc_auc_score(covered.to_numpy(), scores))
            fitted = True
        else:
            propensity.loc[month.index] = rate
        records.append(
            {
                "date": date,
                "rows": int(len(month)),
                "covered_rows": events,
                "coverage_rate": rate,
                "model_fitted": fitted,
                "auc": auc,
            }
        )

    diagnostics = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return propensity, diagnostics


def coverage_weights(
    panel: pd.DataFrame,
    *,
    propensity_column: str = "coverage_propensity",
    covered_column: str = "is_covered",
    date_column: str = "date",
    min_propensity: float = 0.01,
) -> pd.Series:
    """Inverse-propensity weights taking covered rows back to the full universe.

    Covered rows are weighted by ``1 / propensity`` so the reweighted covered
    sample matches the characteristic distribution of every stock-month that was
    eligible to be covered. Weights are floored at ``min_propensity`` to stop a
    handful of near-uncovered rows dominating, then normalised to average one
    within each month so monthly ICs stay comparable across months.
    """
    covered = panel[covered_column].astype(bool)
    scores = pd.to_numeric(panel[propensity_column], errors="coerce").clip(
        lower=min_propensity, upper=1.0
    )
    weights = pd.Series(np.nan, index=panel.index, dtype=float)
    weights[covered] = 1.0 / scores[covered]
    monthly_mean = weights.groupby(panel[date_column], observed=True).transform("mean")
    return weights / monthly_mean.replace(0.0, np.nan)


def categorical_balance(
    panel: pd.DataFrame,
    columns: list[str],
    *,
    covered_column: str = "is_covered",
    weight_column: str | None = None,
) -> pd.DataFrame:
    """Per-level share of covered versus universe rows for categorical covariates.

    Country and sector enter the propensity model, so the reweighting has to be
    shown to fix their composition too, not only the continuous tilts.
    """
    covered = panel[covered_column].astype(bool)
    records = []
    for column in columns:
        values = panel[column].fillna("Unknown").astype(str)
        universe_share = values.value_counts(normalize=True)
        if weight_column is None:
            covered_share = values[covered].value_counts(normalize=True)
        else:
            weights = pd.to_numeric(panel.loc[covered, weight_column], errors="coerce")
            weights = weights.where(weights.gt(0)).fillna(0.0)
            covered_share = weights.groupby(values[covered]).sum()
            covered_share = covered_share / covered_share.sum()
        for level, share in universe_share.items():
            covered_level = float(covered_share.get(level, 0.0))
            spread = np.sqrt(share * (1.0 - share))
            records.append(
                {
                    "covariate": column,
                    "level": level,
                    "universe_share": float(share),
                    "covered_share": covered_level,
                    "standardized_mean_difference": (
                        (covered_level - share) / spread if spread > 0 else np.nan
                    ),
                    "weighted": weight_column is not None,
                }
            )
    return pd.DataFrame(records)


def standardized_mean_differences(
    panel: pd.DataFrame,
    columns: list[str],
    *,
    covered_column: str = "is_covered",
    weight_column: str | None = None,
) -> pd.DataFrame:
    """Covered-versus-universe standardised mean differences for each covariate.

    The reference is the full eligible universe rather than the uncovered
    complement, because the reweighting target is the universe the strategy would
    have to trade in.
    """
    covered = panel[covered_column].astype(bool)
    records = []
    for column in columns:
        values = pd.to_numeric(panel[column], errors="coerce")
        universe_mean = float(values.mean())
        universe_sd = float(values.std(ddof=1))
        subset = values[covered]
        if weight_column is None:
            covered_mean = float(subset.mean())
        else:
            weights = pd.to_numeric(panel.loc[covered, weight_column], errors="coerce")
            valid = subset.notna() & weights.notna() & weights.gt(0)
            covered_mean = float(
                np.average(subset[valid], weights=weights[valid])
                if valid.any()
                else np.nan
            )
        records.append(
            {
                "covariate": column,
                "universe_mean": universe_mean,
                "covered_mean": covered_mean,
                "universe_sd": universe_sd,
                "standardized_mean_difference": (
                    (covered_mean - universe_mean) / universe_sd
                    if universe_sd > 0
                    else np.nan
                ),
                "weighted": weight_column is not None,
            }
        )
    return pd.DataFrame(records)
