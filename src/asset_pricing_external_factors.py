"""Currency-aligned external factor robustness for European ML portfolios."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from data_loader import read_french_block


EXTERNAL_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "WML"]


def load_monthly_eurusd_return(path: Path) -> pd.Series:
    """Load daily USD-per-EUR observations and calculate month-end FX returns."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    frame["usd_per_eur"] = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    frame = frame.dropna().sort_values("date")
    month_end = frame.set_index("date")["usd_per_eur"].resample("ME").last()
    return month_end.pct_change().rename("EURUSD_return")


def load_external_europe_factors(
    five_factor_path: Path,
    momentum_path: Path,
) -> pd.DataFrame:
    """Load USD-denominated monthly European factors in decimal returns."""
    five = read_french_block(five_factor_path)
    momentum = read_french_block(momentum_path)
    factors = five.join(momentum, how="inner")
    factors.index.name = "return_date"
    return factors.reset_index()


def currency_align_portfolios(
    monthly_portfolios: pd.DataFrame,
    eurusd_return: pd.Series,
    cost_bps: int = 25,
) -> pd.DataFrame:
    """Convert EUR strategy returns to USD for comparison with French factors."""
    frame = monthly_portfolios.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["return_date"] = pd.to_datetime(frame["return_date"])
    frame = frame.merge(
        eurusd_return.rename("EURUSD_return")
        .rename_axis("return_date")
        .reset_index(),
        on="return_date",
        how="left",
        validate="many_to_one",
    )
    cost_rate = cost_bps / 10_000.0
    net_long_short_eur = (
        frame["gross_long_short_return"]
        - frame["long_short_turnover"] * cost_rate
    )
    net_long_only_eur = (
        frame["long_return"]
        - frame["long_only_turnover"] * cost_rate
    )
    frame["net_long_short_usd"] = net_long_short_eur * (
        1.0 + frame["EURUSD_return"]
    )
    frame["net_long_only_usd"] = (
        (1.0 + net_long_only_eur)
        * (1.0 + frame["EURUSD_return"])
        - 1.0
    )
    return frame


def _factor_regression(
    frame: pd.DataFrame,
    dependent: str,
    hac_lags: int,
) -> dict[str, float]:
    clean = frame[[dependent, *EXTERNAL_FACTORS]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(clean) < 24:
        return {"observations": int(len(clean))}
    x = sm.add_constant(clean[EXTERNAL_FACTORS], has_constant="add")
    fit = sm.OLS(clean[dependent], x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lags},
    )
    result = {
        "observations": int(fit.nobs),
        "alpha_monthly": float(fit.params["const"]),
        "alpha_annualized": float(fit.params["const"] * 12),
        "alpha_t": float(fit.tvalues["const"]),
        "alpha_p": float(fit.pvalues["const"]),
        "adjusted_r2": float(fit.rsquared_adj),
    }
    for factor in EXTERNAL_FACTORS:
        result[f"beta_{factor}"] = float(fit.params[factor])
        result[f"p_{factor}"] = float(fit.pvalues[factor])
    return result


def external_factor_spanning(
    monthly_portfolios: pd.DataFrame,
    external_factors: pd.DataFrame,
    eurusd_return: pd.Series,
    cost_bps: int = 25,
    hac_lags: int = 6,
) -> pd.DataFrame:
    """Regress USD-aligned strategy returns on external European factors."""
    aligned = currency_align_portfolios(
        monthly_portfolios,
        eurusd_return,
        cost_bps,
    ).merge(
        external_factors,
        on="return_date",
        how="left",
        validate="many_to_one",
    )
    aligned["net_long_only_excess_usd"] = (
        aligned["net_long_only_usd"] - aligned["RF"]
    )
    records = []
    family_keys = ["weighting", "universe_variant"]
    for (weighting, universe_variant), subset in aligned.groupby(
        family_keys, sort=True
    ):
        for model, model_frame in subset.groupby("model", sort=True):
            for portfolio, dependent in [
                ("long_short", "net_long_short_usd"),
                ("long_only_top_decile", "net_long_only_excess_usd"),
            ]:
                records.append(
                    {
                        "comparison": "absolute",
                        "model": model,
                        "baseline": "",
                        "weighting": weighting,
                        "universe_variant": universe_variant,
                        "portfolio": portfolio,
                        "cost_bps": cost_bps,
                        **_factor_regression(
                            model_frame,
                            dependent,
                            hac_lags,
                        ),
                    }
                )

        baseline = subset[subset["model"].eq("momentum_rank")].set_index(
            "return_date"
        )
        for model in sorted(set(subset["model"]) - {"momentum_rank"}):
            candidate = subset[subset["model"].eq(model)].set_index(
                "return_date"
            )
            common = candidate.index.intersection(baseline.index)
            if len(common) < 24:
                continue
            for portfolio, dependent in [
                ("long_short", "net_long_short_usd"),
                ("long_only_top_decile", "net_long_only_excess_usd"),
            ]:
                spread = candidate.loc[common].copy()
                spread["strategy_spread"] = (
                    candidate.loc[common, dependent].to_numpy()
                    - baseline.loc[common, dependent].to_numpy()
                )
                records.append(
                    {
                        "comparison": "model_minus_momentum",
                        "model": model,
                        "baseline": "momentum_rank",
                        "weighting": weighting,
                        "universe_variant": universe_variant,
                        "portfolio": portfolio,
                        "cost_bps": cost_bps,
                        **_factor_regression(
                            spread,
                            "strategy_spread",
                            hac_lags,
                        ),
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
    result["primary_family"] = (
        result["comparison"].eq("model_minus_momentum")
        & result["weighting"].eq("value")
        & result["universe_variant"].eq("standard_ex_bottom_5pct")
        & result["portfolio"].eq("long_short")
    )
    return result
