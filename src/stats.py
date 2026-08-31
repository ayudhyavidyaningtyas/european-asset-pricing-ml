"""Statistical inference helpers for European asset-pricing experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

PPY = 12


def sharpe_ratio(r, rf, ppy: int = PPY) -> float:
    excess = np.asarray(r, dtype=float) - np.asarray(rf, dtype=float)
    standard_deviation = excess.std(ddof=1)
    if standard_deviation <= 0:
        return np.nan
    return float(excess.mean() / standard_deviation * np.sqrt(ppy))


def hac_mean_diff_test(d, maxlags: int | None = None, alt: str = "two-sided") -> dict:
    """Newey-West HAC test of E[d]=0 for a monthly difference series."""
    values = pd.Series(d).dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return {
            "mean": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p_two_sided": np.nan,
            "p_one_sided": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "maxlags": maxlags,
            "n": 0,
        }
    if maxlags is None:
        maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    fit = sm.OLS(values, np.ones((n, 1))).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )
    mean = float(fit.params[0])
    standard_error = float(fit.bse[0])
    t_stat = float(fit.tvalues[0])
    p_two_sided = float(fit.pvalues[0])
    if alt == "less":
        p_one_sided = float(sps.norm.cdf(t_stat))
    elif alt == "greater":
        p_one_sided = float(sps.norm.sf(t_stat))
    else:
        p_one_sided = np.nan
    radius = 1.96 * standard_error
    return {
        "mean": mean,
        "se": standard_error,
        "t": t_stat,
        "p_two_sided": p_two_sided,
        "p_one_sided": p_one_sided,
        "ci_low": mean - radius,
        "ci_high": mean + radius,
        "maxlags": maxlags,
        "n": n,
    }


# Clark-West's variance adjustment assumes the unrestricted forecast differs from the
# restricted one only by sampling error in the parameters that are zero under the null.
# Estimators with stochastic training (random initialisation, SGD path, early stopping,
# subsampled boosting) also differ by training-procedure variance, which enters the
# adjustment term with no matching MSPE inflation and manufactures significance. The
# test is therefore restricted to deterministic, convex estimators.
CLARK_WEST_DETERMINISTIC_ESTIMATORS = frozenset(
    {"zero", "momentum", "ols", "ridge", "lasso", "elastic_net"}
)
CLARK_WEST_STOCHASTIC_ESTIMATORS = frozenset(
    {"mlp", "dre", "hist_gbm", "gbm", "random_forest", "lambdarank"}
)
_TARGET_MODE_SUFFIXES = ("_residual_return", "_residual_rank", "_return", "_rank")


def base_estimator_name(model_label: str) -> str:
    """Strip the target-mode suffix from a model label (``ridge_rank`` -> ``ridge``)."""
    label = str(model_label)
    for suffix in _TARGET_MODE_SUFFIXES:
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label


def clark_west_estimator_eligible(*model_labels: str) -> tuple[bool, str]:
    """Whether every estimator in a comparison meets Clark-West's fit assumption.

    Returns ``(eligible, note)``. Unknown estimators are treated as ineligible so a
    newly added model cannot silently inherit the deterministic-fit assumption.
    """
    bases = [base_estimator_name(label) for label in model_labels]
    stochastic = sorted({b for b in bases if b in CLARK_WEST_STOCHASTIC_ESTIMATORS})
    unknown = sorted(
        {
            b
            for b in bases
            if b not in CLARK_WEST_STOCHASTIC_ESTIMATORS
            and b not in CLARK_WEST_DETERMINISTIC_ESTIMATORS
        }
    )
    if stochastic:
        return False, f"stochastic_estimator={','.join(stochastic)}"
    if unknown:
        return False, f"unknown_estimator={','.join(unknown)}"
    return True, f"deterministic_estimator={','.join(sorted(set(bases)))}"


def residual_target_model_eligible(
    control_columns,
    *model_labels: str,
    predictor_columns=None,
) -> tuple[bool, str]:
    """Whether every model in a comparison is valid on a residual target.

    A characteristic baseline whose score is itself one of the neutralization
    controls is regressed against itself: its residual IC is ~0 by construction,
    so any comparison against it measures the control set, not predictive skill.

    Returns ``(eligible, note)``. Models with no known raw predictor column (all
    fitted estimators) are always eligible.
    """
    if predictor_columns is None:
        from asset_pricing_ml import MODEL_PREDICTOR_COLUMNS

        predictor_columns = MODEL_PREDICTOR_COLUMNS
    controls = set(control_columns or [])
    conflicts = sorted(
        {
            f"{base}~{predictor_columns[base]}"
            for base in (base_estimator_name(label) for label in model_labels)
            if base in predictor_columns
            and predictor_columns[base] in controls
        }
    )
    if conflicts:
        return False, f"predictor_is_residual_control={','.join(conflicts)}"
    return True, "no_predictor_control_conflict"


def clark_west_test(
    actual,
    restricted_prediction,
    unrestricted_prediction,
    dates=None,
    maxlags: int | None = None,
) -> dict:
    """Clark-West adjusted MSPE test for genuinely nested forecasts.

    Positive adjusted mean differences favor the unrestricted forecast. The
    caller is responsible for verifying both that the unrestricted model nests
    the restricted model and that the estimators are Clark-West eligible (see
    :func:`clark_west_estimator_eligible`); this helper only computes the
    adjusted statistic.
    """
    actual_series = pd.Series(actual, dtype="float64")
    restricted = pd.Series(restricted_prediction, dtype="float64")
    unrestricted = pd.Series(unrestricted_prediction, dtype="float64")
    frame = pd.DataFrame(
        {
            "actual": actual_series,
            "restricted": restricted,
            "unrestricted": unrestricted,
        }
    )
    if dates is not None:
        frame["date"] = pd.to_datetime(pd.Series(dates).to_numpy())
    frame = frame.dropna(subset=["actual", "restricted", "unrestricted"])
    if frame.empty:
        return {
            "months": 0,
            "mean_difference": np.nan,
            "hac_standard_error": np.nan,
            "t_stat": np.nan,
            "p_one_sided": np.nan,
            "p_two_sided": np.nan,
        }
    restricted_error = frame["actual"] - frame["restricted"]
    unrestricted_error = frame["actual"] - frame["unrestricted"]
    adjustment = frame["restricted"].sub(frame["unrestricted"]).pow(2)
    adjusted = restricted_error.pow(2) - unrestricted_error.pow(2) + adjustment
    if dates is not None:
        adjusted = adjusted.groupby(frame["date"]).mean()
    test = hac_mean_diff_test(adjusted, maxlags=maxlags, alt="greater")
    t_stat = test["t"]
    if (
        not np.isfinite(t_stat)
        and np.isfinite(test["mean"])
        and test["se"] == 0
    ):
        t_stat = np.sign(test["mean"]) * np.inf
    return {
        "months": test["n"],
        "mean_difference": test["mean"],
        "hac_standard_error": test["se"],
        "t_stat": t_stat,
        "p_one_sided": (
            float(sps.norm.sf(t_stat)) if not np.isnan(t_stat) else np.nan
        ),
        "p_two_sided": test["p_two_sided"],
    }


def stationary_bootstrap_indices(
    n: int,
    expected_block: float,
    n_boot: int,
    rng,
) -> np.ndarray:
    """Politis-Romano stationary-bootstrap index matrix."""
    restart_probability = 1.0 / expected_block
    restart = rng.random((n_boot, n)) < restart_probability
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    t = np.arange(n)[None, :]
    last_restart_t = np.maximum.accumulate(np.where(restart, t, 0), axis=1)
    start_value = pd.DataFrame(np.where(restart, starts, np.nan)).ffill(axis=1).to_numpy()
    return ((start_value + (t - last_restart_t)) % n).astype(int)


def metric_point(values: pd.Series | np.ndarray, metric: str, ppy: int = PPY) -> float:
    """Point estimate for level metrics used in stationary-bootstrap CIs."""
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    array = clean.to_numpy(dtype=float)
    if metric == "mean":
        return float(array.mean())
    if metric == "annualized_mean":
        return float(array.mean() * ppy)
    if metric in {"sharpe", "information_ratio"}:
        volatility = array.std(ddof=1)
        if volatility <= 0 or not np.isfinite(volatility):
            return np.nan
        return float(array.mean() / volatility * np.sqrt(ppy))
    raise ValueError(f"Unknown metric: {metric}")


def stationary_bootstrap_metric_ci(
    values: pd.Series | np.ndarray,
    *,
    metric: str,
    expected_block: int | float = 6,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
    ppy: int = PPY,
) -> dict[str, float]:
    """Stationary-bootstrap CI and sign p-value for a single level metric."""
    clean = (
        pd.Series(values)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(clean) < 3:
        return {
            "point": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided_zero": np.nan,
            "observations": int(len(clean)),
        }
    if expected_block <= 0:
        raise ValueError("expected_block must be positive")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    rng = np.random.default_rng(seed)
    indices = stationary_bootstrap_indices(
        len(clean),
        expected_block,
        n_boot,
        rng,
    )
    samples = clean[indices]
    if metric == "mean":
        draws = samples.mean(axis=1)
    elif metric == "annualized_mean":
        draws = samples.mean(axis=1) * ppy
    elif metric in {"sharpe", "information_ratio"}:
        volatility = samples.std(axis=1, ddof=1)
        draws = np.divide(
            samples.mean(axis=1),
            volatility,
            out=np.full(n_boot, np.nan),
            where=volatility > 0,
        ) * np.sqrt(ppy)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    draws = (
        pd.Series(draws)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    point = metric_point(clean, metric, ppy=ppy)
    if len(draws) == 0:
        return {
            "point": point,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided_zero": np.nan,
            "observations": int(len(clean)),
        }
    alpha = 1.0 - ci
    return {
        "point": point,
        "ci_low": float(np.quantile(draws, alpha / 2.0)),
        "ci_high": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "p_two_sided_zero": float(
            min(1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0)))
        ),
        "observations": int(len(clean)),
    }


def bootstrap_sharpe_diff(
    r_a,
    r_b,
    rf,
    expected_block: float = 6,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
    ppy: int = PPY,
) -> dict:
    """Paired stationary-bootstrap CI and two-sided p-value for Sharpe(A)-Sharpe(B)."""
    a, b, f = (np.asarray(x, dtype=float) for x in (r_a, r_b, rf))
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(f)
    a, b, f = a[valid], b[valid], f[valid]
    n = len(a)
    if n < 3:
        return {
            "delta_sharpe": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided": np.nan,
            "expected_block": expected_block,
            "n_boot": n_boot,
        }
    point = sharpe_ratio(a, f, ppy) - sharpe_ratio(b, f, ppy)
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, expected_block, n_boot, rng)
    excess_a = a[idx] - f[idx]
    excess_b = b[idx] - f[idx]
    sd_a = excess_a.std(1, ddof=1)
    sd_b = excess_b.std(1, ddof=1)
    sr_a = np.divide(
        excess_a.mean(1),
        sd_a,
        out=np.full(n_boot, np.nan),
        where=sd_a > 0,
    ) * np.sqrt(ppy)
    sr_b = np.divide(
        excess_b.mean(1),
        sd_b,
        out=np.full(n_boot, np.nan),
        where=sd_b > 0,
    ) * np.sqrt(ppy)
    differences = sr_a - sr_b
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        low = high = p_two_sided = np.nan
    else:
        low, high = np.quantile(differences, [(1 - ci) / 2, 1 - (1 - ci) / 2])
        p_two_sided = min(
            1.0,
            2 * min((differences >= 0).mean(), (differences <= 0).mean()),
        )
    return {
        "delta_sharpe": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": float(p_two_sided),
        "expected_block": expected_block,
        "n_boot": n_boot,
    }


def _bartlett_long_run_covariance(demeaned: np.ndarray, maxlags: int) -> np.ndarray:
    """Newey-West long-run covariance of a multivariate series (T x k)."""
    n = demeaned.shape[0]
    covariance = demeaned.T @ demeaned / n
    for lag in range(1, maxlags + 1):
        gamma = demeaned[lag:].T @ demeaned[:-lag] / n
        weight = 1.0 - lag / (maxlags + 1.0)
        covariance = covariance + weight * (gamma + gamma.T)
    return covariance


def ledoit_wolf_sharpe_test(r_a, r_b, rf, maxlags: int | None = None) -> dict:
    """Ledoit-Wolf (2008) HAC delta-method test of equal Sharpe ratios.

    The Sharpe difference is a smooth function of the first two moments of each
    excess-return series; its standard error combines the delta-method gradient
    with a Newey-West long-run covariance of (a, b, a^2, b^2), so serial
    correlation and non-normality enter the variance, unlike Jobson-Korkie.
    """
    a = np.asarray(r_a, dtype=float) - np.asarray(rf, dtype=float)
    b = np.asarray(r_b, dtype=float) - np.asarray(rf, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    n = len(a)
    empty = {
        "delta_sharpe_monthly": np.nan,
        "delta_sharpe_annualized": np.nan,
        "z": np.nan,
        "p_two_sided": np.nan,
        "maxlags": maxlags,
        "n": n,
    }
    if n < 12:
        return empty
    if maxlags is None:
        maxlags = int(np.floor(4 * (n / 100) ** (2 / 9)))

    mean_a, mean_b = a.mean(), b.mean()
    raw2_a, raw2_b = (a**2).mean(), (b**2).mean()
    variance_a = raw2_a - mean_a**2
    variance_b = raw2_b - mean_b**2
    if variance_a <= 0 or variance_b <= 0:
        return empty
    sharpe_a = mean_a / np.sqrt(variance_a)
    sharpe_b = mean_b / np.sqrt(variance_b)
    difference = sharpe_a - sharpe_b

    # Gradient of f(m_a, m_b, q_a, q_b) = m_a/sqrt(q_a-m_a^2) - m_b/sqrt(q_b-m_b^2)
    gradient = np.array(
        [
            raw2_a / variance_a**1.5,
            -raw2_b / variance_b**1.5,
            -0.5 * mean_a / variance_a**1.5,
            0.5 * mean_b / variance_b**1.5,
        ]
    )
    moments = np.column_stack([a, b, a**2, b**2])
    demeaned = moments - moments.mean(axis=0)
    long_run = _bartlett_long_run_covariance(demeaned, maxlags)
    variance = float(gradient @ long_run @ gradient) / n
    if variance <= 0 or not np.isfinite(variance):
        return {**empty, "delta_sharpe_monthly": float(difference)}
    z_stat = difference / np.sqrt(variance)
    return {
        "delta_sharpe_monthly": float(difference),
        "delta_sharpe_annualized": float(difference * np.sqrt(PPY)),
        "z": float(z_stat),
        "p_two_sided": float(2 * sps.norm.sf(abs(z_stat))),
        "maxlags": maxlags,
        "n": n,
    }


def jobson_korkie_memmel(r_a, r_b, rf) -> dict:
    """Iid analytical Sharpe-difference sanity check with Memmel correction."""
    a = np.asarray(r_a, dtype=float) - np.asarray(rf, dtype=float)
    b = np.asarray(r_b, dtype=float) - np.asarray(rf, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    n = len(a)
    if n < 3:
        return {
            "delta_sharpe_monthly": np.nan,
            "z": np.nan,
            "p_two_sided": np.nan,
        }
    sd_a = a.std(ddof=1)
    sd_b = b.std(ddof=1)
    if sd_a <= 0 or sd_b <= 0:
        return {
            "delta_sharpe_monthly": np.nan,
            "z": np.nan,
            "p_two_sided": np.nan,
        }
    sr_a = a.mean() / sd_a
    sr_b = b.mean() / sd_b
    rho = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(rho):
        return {
            "delta_sharpe_monthly": float(sr_a - sr_b),
            "z": np.nan,
            "p_two_sided": np.nan,
        }
    theta = (2 - 2 * rho + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * rho**2)) / n
    if theta <= 0 or not np.isfinite(theta):
        return {
            "delta_sharpe_monthly": float(sr_a - sr_b),
            "z": np.nan,
            "p_two_sided": np.nan,
        }
    z_stat = (sr_a - sr_b) / np.sqrt(theta)
    return {
        "delta_sharpe_monthly": float(sr_a - sr_b),
        "z": float(z_stat),
        "p_two_sided": float(2 * sps.norm.sf(abs(z_stat))),
    }
