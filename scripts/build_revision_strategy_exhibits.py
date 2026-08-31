"""Build consolidated exhibits for the strict lagged revisions strategy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_external_factors import (  # noqa: E402
    EXTERNAL_FACTORS,
    external_factor_spanning,
    load_external_europe_factors,
    load_monthly_eurusd_return,
)
from asset_pricing_depth import load_eur_short_rate  # noqa: E402
import stats as project_stats  # noqa: E402


DEFAULT_REVISION_DIR = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "estimates_revisions_pure_strict_lag1_revision_signal_ridge"
)
DEFAULT_CONSTRAINED_DIR = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "constrained_estimates_revisions_pure_strict_lag1_revision_signal_fixed"
)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_FRENCH_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "french"
DEFAULT_FX = PROJECT_ROOT / "data" / "raw" / "fred_DEXUSEU.csv"
DEFAULT_EUR_RATE = PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "revision_strategy_final_exhibits"
)
DEFAULT_DRE_DIR = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "estimates_revisions_pure_strict_lag1_revision_signal_dre"
)

PPY = 12
HIGHLIGHT_EXPERIMENT = "estimates_revisions_pure_strict_lag1_revision_signal_ridge"
AUM_LABELS = {"10m": 10_000_000.0, "100m": 100_000_000.0, "500m": 500_000_000.0}
REVISION_FEATURE_ORDER = [
    "est_eps_revision_1m_rank",
    "est_eps_revision_3m_rank",
    "est_revenue_revision_1m_rank",
    "est_revenue_revision_3m_rank",
    "est_price_target_revision_1m_rank",
    "est_price_target_revision_3m_rank",
]


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return float(wealth.div(peak).sub(1.0).min())


def annualized_return_summary(returns: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {
            "months": 0,
            "annualized_net_return": np.nan,
            "annualized_net_volatility": np.nan,
            "net_sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    annual_return = float(clean.mean() * PPY)
    annual_volatility = float(clean.std(ddof=1) * np.sqrt(PPY)) if len(clean) > 1 else np.nan
    return {
        "months": int(len(clean)),
        "annualized_net_return": annual_return,
        "annualized_net_volatility": annual_volatility,
        "net_sharpe": annual_return / annual_volatility
        if pd.notna(annual_volatility) and annual_volatility > 0
        else np.nan,
        "max_drawdown": _max_drawdown(clean),
    }


def annualized_portfolio_return_summary(
    returns: pd.Series,
    dates: pd.Series,
    risk_free: pd.Series | None = None,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "return": pd.to_numeric(pd.Series(returns).reset_index(drop=True), errors="coerce"),
            "date": pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="coerce"),
        }
    )
    if risk_free is not None:
        rf = risk_free.reindex(pd.DatetimeIndex(frame["date"])).reset_index(drop=True)
        frame["risk_free"] = pd.to_numeric(rf, errors="coerce")
    required = ["return", *([] if risk_free is None else ["risk_free"])]
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    if clean.empty:
        return {
            "months": 0,
            "annualized_net_return": np.nan,
            "annualized_net_volatility": np.nan,
            "net_sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    excess = (
        clean["return"] - clean["risk_free"]
        if risk_free is not None
        else clean["return"]
    )
    annual_return = float(clean["return"].mean() * PPY)
    annual_volatility = (
        float(clean["return"].std(ddof=1) * np.sqrt(PPY))
        if len(clean) > 1
        else np.nan
    )
    return {
        "months": int(len(clean)),
        "annualized_net_return": annual_return,
        "annualized_net_volatility": annual_volatility,
        "net_sharpe": sharpe_ratio(excess),
        "max_drawdown": _max_drawdown(clean["return"]),
    }


def _net_revision_returns(monthly: pd.DataFrame, cost_bps: int) -> pd.DataFrame:
    out = monthly.copy()
    cost_rate = cost_bps / 10_000.0
    out["signal_date"] = pd.to_datetime(out["signal_date"])
    out["return_date"] = pd.to_datetime(out["return_date"])
    out["net_long_short_return"] = (
        pd.to_numeric(out["gross_long_short_return"], errors="coerce")
        - pd.to_numeric(out["long_short_turnover"], errors="coerce") * cost_rate
    )
    out["net_long_only_return"] = (
        pd.to_numeric(out["long_return"], errors="coerce")
        - pd.to_numeric(out["long_only_turnover"], errors="coerce") * cost_rate
    )
    return out


def breakeven_cost_bps(
    gross_returns: pd.Series,
    turnover: pd.Series,
    target_returns: pd.Series | None = None,
) -> float:
    """Transaction-cost bps where mean net return equals the target mean."""
    clean = pd.DataFrame(
        {
            "gross": pd.to_numeric(gross_returns, errors="coerce"),
            "turnover": pd.to_numeric(turnover, errors="coerce"),
            "target": (
                0.0
                if target_returns is None
                else pd.to_numeric(target_returns, errors="coerce")
            ),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or clean["turnover"].mean() <= 0:
        return np.nan
    return float((clean["gross"].mean() - clean["target"].mean()) / clean["turnover"].mean() * 10_000.0)


def breakeven_capacity_aum(
    gross_returns: pd.Series,
    spread_cost: pd.Series,
    impact_cost_at_base_aum: pd.Series,
    base_aum: float,
    target_returns: pd.Series | None = None,
) -> float:
    """AUM where square-root impact drives mean net return to the target mean."""
    clean = pd.DataFrame(
        {
            "gross": pd.to_numeric(gross_returns, errors="coerce"),
            "spread": pd.to_numeric(spread_cost, errors="coerce"),
            "impact": pd.to_numeric(impact_cost_at_base_aum, errors="coerce"),
            "target": (
                0.0
                if target_returns is None
                else pd.to_numeric(target_returns, errors="coerce")
            ),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or clean["impact"].mean() <= 0:
        return np.nan
    residual_before_impact = clean["gross"].mean() - clean["spread"].mean() - clean["target"].mean()
    if residual_before_impact <= 0:
        return 0.0
    return float(base_aum * (residual_before_impact / clean["impact"].mean()) ** 2)


def sharpe_ratio(values: pd.Series | np.ndarray) -> float:
    clean = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(clean) < 2:
        return np.nan
    volatility = clean.std(ddof=1)
    if volatility <= 0:
        return np.nan
    return float(clean.mean() / volatility * np.sqrt(PPY))


def stationary_bootstrap_metric_ci(
    values: pd.Series | np.ndarray,
    *,
    metric: str,
    expected_block: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    return project_stats.stationary_bootstrap_metric_ci(
        values,
        metric=metric,
        expected_block=expected_block,
        n_boot=n_boot,
        seed=seed,
        ppy=PPY,
    )


def stationary_bootstrap_sharpe_difference_ci(
    returns_a: pd.Series | np.ndarray,
    returns_b: pd.Series | np.ndarray,
    *,
    expected_block: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    clean = pd.DataFrame(
        {
            "a": pd.to_numeric(pd.Series(returns_a), errors="coerce"),
            "b": pd.to_numeric(pd.Series(returns_b), errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return {
            "point": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided_zero": np.nan,
            "observations": int(len(clean)),
        }
    a = clean["a"].to_numpy(dtype=float)
    b = clean["b"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = project_stats.stationary_bootstrap_indices(len(clean), expected_block, n_boot, rng)
    sample_a = a[indices]
    sample_b = b[indices]
    vol_a = sample_a.std(axis=1, ddof=1)
    vol_b = sample_b.std(axis=1, ddof=1)
    sharpe_a = np.divide(
        sample_a.mean(axis=1),
        vol_a,
        out=np.full(n_boot, np.nan),
        where=vol_a > 0,
    ) * np.sqrt(PPY)
    sharpe_b = np.divide(
        sample_b.mean(axis=1),
        vol_b,
        out=np.full(n_boot, np.nan),
        where=vol_b > 0,
    ) * np.sqrt(PPY)
    draws = pd.Series(sharpe_a - sharpe_b).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    point = sharpe_ratio(a) - sharpe_ratio(b)
    if len(draws) == 0:
        return {
            "point": point,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_two_sided_zero": np.nan,
            "observations": int(len(clean)),
        }
    return {
        "point": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "p_two_sided_zero": float(
            min(1.0, 2.0 * min(np.mean(draws <= 0.0), np.mean(draws >= 0.0)))
        ),
        "observations": int(len(clean)),
    }


def build_implementability_exhibit(
    revision_dir: Path,
    constrained_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    model_summary = pd.read_csv(revision_dir / "model_summary.csv")
    constrained = pd.read_csv(constrained_dir / "constrained_summary.csv")
    benchmark = pd.read_csv(constrained_dir / "benchmark_relative_summary.csv")
    records: list[dict[str, Any]] = []

    model_filter = (
        model_summary["model"].eq("ridge_rank")
        & model_summary["cost_bps"].eq(25)
        & model_summary["universe_variant"].isin(
            ["standard_ex_bottom_5pct", "ex_bottom_20pct"]
        )
    )
    for _, row in model_summary[model_filter].iterrows():
        records.append(
            {
                "portfolio_object": (
                    f"unconstrained_{row['weighting']}_{row['universe_variant']}_"
                    f"{row['portfolio']}"
                ),
                "implementation": "unconstrained_decile",
                "weighting": row["weighting"],
                "universe_variant": row["universe_variant"],
                "portfolio": row["portfolio"],
                "aum_label": "",
                "months": int(row["months"]),
                "annualized_net_return": float(row["annualized_net_mean_return"]),
                "annualized_net_volatility": float(row["annualized_net_volatility"]),
                "net_sharpe": float(row["net_sharpe"]),
                "max_drawdown": float(row["max_drawdown"]),
                "average_monthly_turnover": float(row["average_monthly_turnover"]),
                "annualized_active_return": np.nan,
                "alpha_annualized": np.nan,
                "alpha_t_stat": np.nan,
                "alpha_p_two_sided": np.nan,
            }
        )

    full_constrained = constrained[constrained["subperiod"].eq("full")].copy()
    full_benchmark = benchmark[benchmark["subperiod"].eq("full")].copy()
    full_constrained = full_constrained.merge(
        full_benchmark[
            [
                "strategy",
                "constraint",
                "aum_label",
                "annualized_active_return",
                "alpha_annualized",
                "alpha_t_stat",
                "alpha_p_two_sided",
            ]
        ],
        on=["strategy", "constraint", "aum_label"],
        how="left",
        validate="one_to_one",
    )
    for _, row in full_constrained.iterrows():
        records.append(
            {
                "portfolio_object": f"constrained_top500_long_only_{row['aum_label']}",
                "implementation": "top500_constrained_long_only",
                "weighting": "optimized",
                "universe_variant": "top_500_observed_spread",
                "portfolio": "long_only",
                "aum_label": row["aum_label"],
                "months": int(row["months"]),
                "annualized_net_return": float(row["annualized_net_return"]),
                "annualized_net_volatility": float(row["annualized_net_volatility"]),
                "net_sharpe": float(row["net_sharpe"]),
                "max_drawdown": float(row["max_drawdown"]),
                "average_monthly_turnover": float(row["average_monthly_turnover"]),
                "annualized_active_return": float(row["annualized_active_return"]),
                "alpha_annualized": float(row["alpha_annualized"]),
                "alpha_t_stat": float(row["alpha_t_stat"]),
                "alpha_p_two_sided": float(row["alpha_p_two_sided"]),
            }
        )
    exhibit = pd.DataFrame(records)
    exhibit.to_csv(output_dir / "revision_implementability_exhibit.csv", index=False)
    return exhibit


def build_breakeven_cost_capacity(
    revision_dir: Path,
    constrained_dir: Path,
    output_dir: Path,
    cost_bps: int,
) -> pd.DataFrame:
    monthly = pd.read_csv(revision_dir / "monthly_portfolios.csv", parse_dates=["return_date"])
    constrained = pd.read_csv(constrained_dir / "constrained_monthly.csv", parse_dates=["target_date"])
    benchmark = pd.read_csv(
        constrained_dir / "benchmark_relative_monthly.csv",
        parse_dates=["target_date"],
    )
    records: list[dict[str, Any]] = []

    selected = monthly[
        monthly["model"].eq("ridge_rank")
        & monthly["universe_variant"].isin(["standard_ex_bottom_5pct", "ex_bottom_20pct"])
    ].copy()
    for (weighting, universe_variant), group in selected.groupby(
        ["weighting", "universe_variant"],
        sort=True,
    ):
        for portfolio, gross_column, turnover_column in [
            ("long_short", "gross_long_short_return", "long_short_turnover"),
            ("long_only_top_decile", "long_return", "long_only_turnover"),
        ]:
            gross = pd.to_numeric(group[gross_column], errors="coerce")
            turnover = pd.to_numeric(group[turnover_column], errors="coerce")
            current_net = gross - turnover * (cost_bps / 10_000.0)
            records.append(
                {
                    "portfolio_object": f"unconstrained_{weighting}_{universe_variant}_{portfolio}",
                    "implementation": "unconstrained_decile",
                    "weighting": weighting,
                    "universe_variant": universe_variant,
                    "portfolio": portfolio,
                    "aum_label": "",
                    "current_cost_bps": cost_bps,
                    "months": int(current_net.dropna().shape[0]),
                    "annualized_gross_return": float(gross.mean() * PPY),
                    "annualized_current_net_return": float(current_net.mean() * PPY),
                    "average_monthly_turnover": float(turnover.mean()),
                    "annualized_current_turnover_cost": float(
                        (turnover * (cost_bps / 10_000.0)).mean() * PPY
                    ),
                    "breakeven_cost_bps_zero_net_return": breakeven_cost_bps(
                        gross,
                        turnover,
                    ),
                    "breakeven_cost_multiple_of_current": (
                        breakeven_cost_bps(gross, turnover) / cost_bps
                        if cost_bps > 0
                        else np.nan
                    ),
                    "annualized_current_active_return": np.nan,
                    "breakeven_cost_bps_zero_active_return": np.nan,
                    "breakeven_aum_eur_zero_net_return": np.nan,
                    "breakeven_aum_eur_zero_active_return": np.nan,
                }
            )

    benchmark_column = "benchmark_return_eur"
    constrained = constrained.merge(
        benchmark[["target_date", benchmark_column]].drop_duplicates("target_date"),
        on="target_date",
        how="left",
        validate="many_to_one",
    )
    for aum_label, aum_eur in AUM_LABELS.items():
        spread_column = f"spread_cost_{aum_label}"
        impact_column = f"impact_cost_{aum_label}"
        net_column = f"net_return_{aum_label}"
        if net_column not in constrained:
            continue
        gross = pd.to_numeric(constrained["gross_return"], errors="coerce")
        spread = pd.to_numeric(constrained[spread_column], errors="coerce")
        impact = pd.to_numeric(constrained[impact_column], errors="coerce")
        net = pd.to_numeric(constrained[net_column], errors="coerce")
        benchmark_return = pd.to_numeric(constrained[benchmark_column], errors="coerce")
        active = net - benchmark_return
        records.append(
            {
                "portfolio_object": f"constrained_top500_long_only_{aum_label}",
                "implementation": "top500_constrained_long_only",
                "weighting": "optimized",
                "universe_variant": "top_500_observed_spread",
                "portfolio": "long_only",
                "aum_label": aum_label,
                "current_cost_bps": np.nan,
                "months": int(net.dropna().shape[0]),
                "annualized_gross_return": float(gross.mean() * PPY),
                "annualized_current_net_return": float(net.mean() * PPY),
                "average_monthly_turnover": float(
                    pd.to_numeric(constrained[f"turnover_{aum_label}"], errors="coerce").mean()
                ),
                "annualized_current_turnover_cost": np.nan,
                "breakeven_cost_bps_zero_net_return": np.nan,
                "breakeven_cost_multiple_of_current": np.nan,
                "annualized_current_active_return": float(active.mean() * PPY),
                "breakeven_cost_bps_zero_active_return": np.nan,
                "annualized_spread_cost": float(spread.mean() * PPY),
                "annualized_impact_cost": float(impact.mean() * PPY),
                "breakeven_aum_eur_zero_net_return": breakeven_capacity_aum(
                    gross,
                    spread,
                    impact,
                    aum_eur,
                ),
                "breakeven_aum_eur_zero_active_return": breakeven_capacity_aum(
                    gross,
                    spread,
                    impact,
                    aum_eur,
                    target_returns=benchmark_return,
                ),
            }
        )
    exhibit = pd.DataFrame(records)
    exhibit.to_csv(output_dir / "revision_breakeven_cost_capacity.csv", index=False)
    return exhibit


def _feature_family(feature: str) -> str:
    if "_eps_" in feature:
        return "eps"
    if "_revenue_" in feature:
        return "revenue"
    if "_price_target_" in feature:
        return "price_target"
    return "other"


def _feature_horizon(feature: str) -> str:
    if "_1m_" in feature:
        return "1m"
    if "_3m_" in feature:
        return "3m"
    return ""


def build_ridge_coefficient_stability(
    revision_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = pd.read_csv(revision_dir / "linear_coefficients.csv")
    coefficients = coefficients[
        coefficients["model_label"].eq("ridge_rank") & coefficients["target_mode"].eq("rank")
    ].copy()
    coefficients["feature"] = pd.Categorical(
        coefficients["feature"],
        categories=REVISION_FEATURE_ORDER,
        ordered=True,
    )
    coefficients = coefficients.sort_values(["feature", "test_year"]).reset_index(drop=True)
    coefficients["feature_family"] = coefficients["feature"].astype(str).map(_feature_family)
    coefficients["horizon"] = coefficients["feature"].astype(str).map(_feature_horizon)
    path = output_dir / "revision_ridge_coefficient_path.csv"
    coefficients.to_csv(path, index=False)

    records = []
    for feature, group in coefficients.groupby("feature", observed=True, sort=True):
        values = pd.to_numeric(group["coefficient"], errors="coerce").dropna()
        signs = np.sign(values.replace(0.0, np.nan).dropna())
        sign_changes = int((signs != signs.shift()).sum() - 1) if len(signs) > 1 else 0
        records.append(
            {
                "feature": str(feature),
                "feature_family": _feature_family(str(feature)),
                "horizon": _feature_horizon(str(feature)),
                "years": int(values.shape[0]),
                "first_coefficient": float(values.iloc[0]) if not values.empty else np.nan,
                "last_coefficient": float(values.iloc[-1]) if not values.empty else np.nan,
                "mean_coefficient": float(values.mean()) if not values.empty else np.nan,
                "median_coefficient": float(values.median()) if not values.empty else np.nan,
                "std_coefficient": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "min_coefficient": float(values.min()) if not values.empty else np.nan,
                "max_coefficient": float(values.max()) if not values.empty else np.nan,
                "mean_abs_coefficient": float(values.abs().mean()) if not values.empty else np.nan,
                "positive_year_fraction": float(values.gt(0).mean()) if not values.empty else np.nan,
                "negative_year_fraction": float(values.lt(0).mean()) if not values.empty else np.nan,
                "sign_changes": max(sign_changes, 0),
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(output_dir / "revision_ridge_coefficient_stability.csv", index=False)
    _plot_ridge_coefficients(
        coefficients,
        output_dir / "revision_ridge_coefficient_stability.png",
    )
    return coefficients, summary


def _portfolio_return_column(portfolio: str) -> tuple[str, str]:
    if portfolio == "long_short":
        return "gross_long_short_return", "long_short_turnover"
    if portfolio == "long_only_top_decile":
        return "long_return", "long_only_turnover"
    raise ValueError(f"Unknown portfolio: {portfolio}")


def _portfolio_net_returns(
    group: pd.DataFrame,
    portfolio: str,
    cost_bps: int,
    risk_free: pd.Series | None = None,
) -> pd.Series:
    gross_column, turnover_column = _portfolio_return_column(portfolio)
    gross = pd.to_numeric(group[gross_column], errors="coerce")
    turnover = pd.to_numeric(group[turnover_column], errors="coerce")
    net = gross - turnover * (cost_bps / 10_000.0)
    if portfolio == "long_only_top_decile" and risk_free is not None:
        rf = risk_free.reindex(pd.DatetimeIndex(pd.to_datetime(group["return_date"])))
        rf.index = group.index
        net = net - pd.to_numeric(rf, errors="coerce")
    return net


def build_bootstrap_uncertainty(
    revision_dir: Path,
    constrained_dir: Path,
    dre_dir: Path | None,
    output_dir: Path,
    cost_bps: int,
    bootstrap_repetitions: int,
    bootstrap_blocks: tuple[int, ...],
    random_state: int,
    risk_free: pd.Series | None = None,
) -> pd.DataFrame:
    monthly = pd.read_csv(
        revision_dir / "monthly_portfolios.csv",
        parse_dates=["signal_date", "return_date"],
    )
    records: list[dict[str, Any]] = []
    selected = monthly[
        monthly["model"].eq("ridge_rank")
        & monthly["universe_variant"].eq("standard_ex_bottom_5pct")
        & monthly["weighting"].isin(["equal", "value"])
    ].copy()
    for weighting, group in selected.groupby("weighting", sort=True):
        for portfolio in ["long_short", "long_only_top_decile"]:
            returns = _portfolio_net_returns(
                group.sort_values("return_date"),
                portfolio,
                cost_bps,
                risk_free,
            )
            for block in bootstrap_blocks:
                result = stationary_bootstrap_metric_ci(
                    returns,
                    metric="sharpe",
                    expected_block=block,
                    n_boot=bootstrap_repetitions,
                    seed=random_state + block,
                )
                records.append(
                    {
                        "test": "net_sharpe_level",
                        "portfolio_object": f"ridge_{weighting}_standard_ex_bottom_5pct_{portfolio}",
                        "comparison": "ridge_revision_strategy",
                        "model_a": "ridge_rank",
                        "model_b": "",
                        "weighting": weighting,
                        "portfolio": portfolio,
                        "aum_label": "",
                        "metric": "net_sharpe",
                        "bootstrap_block": block,
                        "bootstrap_repetitions": bootstrap_repetitions,
                        **result,
                    }
                )

    if dre_dir is not None and (dre_dir / "monthly_portfolios.csv").exists():
        challenge = pd.read_csv(
            dre_dir / "monthly_portfolios.csv",
            parse_dates=["signal_date", "return_date"],
        )
        challenge = challenge[
            challenge["universe_variant"].eq("standard_ex_bottom_5pct")
            & challenge["weighting"].isin(["equal", "value"])
            & challenge["model"].isin(["ridge_rank", "dre_rank"])
        ].copy()
        for weighting, group in challenge.groupby("weighting", sort=True):
            for portfolio in ["long_short", "long_only_top_decile"]:
                wide = {}
                for model, model_group in group.groupby("model", sort=True):
                    model_group = model_group.sort_values("return_date")
                    wide[model] = pd.DataFrame(
                        {
                            "return_date": model_group["return_date"],
                            model: _portfolio_net_returns(
                                model_group,
                                portfolio,
                                cost_bps,
                                risk_free,
                            ).to_numpy(),
                        }
                    )
                if {"ridge_rank", "dre_rank"} - set(wide):
                    continue
                paired = wide["ridge_rank"].merge(
                    wide["dre_rank"],
                    on="return_date",
                    how="inner",
                    validate="one_to_one",
                )
                for block in bootstrap_blocks:
                    result = stationary_bootstrap_sharpe_difference_ci(
                        paired["ridge_rank"],
                        paired["dre_rank"],
                        expected_block=block,
                        n_boot=bootstrap_repetitions,
                        seed=random_state + 100 + block,
                    )
                    records.append(
                        {
                            "test": "net_sharpe_difference",
                            "portfolio_object": (
                                f"ridge_minus_dre_{weighting}_standard_ex_bottom_5pct_{portfolio}"
                            ),
                            "comparison": "ridge_minus_dre",
                            "model_a": "ridge_rank",
                            "model_b": "dre_rank",
                            "weighting": weighting,
                            "portfolio": portfolio,
                            "aum_label": "",
                            "metric": "delta_net_sharpe",
                            "bootstrap_block": block,
                            "bootstrap_repetitions": bootstrap_repetitions,
                            **result,
                        }
                    )

    benchmark = pd.read_csv(
        constrained_dir / "benchmark_relative_monthly.csv",
        parse_dates=["target_date"],
    )
    for aum_label in AUM_LABELS:
        active_column = f"active_return_{aum_label}"
        net_column = f"net_return_{aum_label}"
        if active_column not in benchmark or net_column not in benchmark:
            continue
        for block in bootstrap_blocks:
            active = stationary_bootstrap_metric_ci(
                benchmark[active_column],
                metric="annualized_mean",
                expected_block=block,
                n_boot=bootstrap_repetitions,
                seed=random_state + 200 + block,
            )
            records.append(
                {
                    "test": "annualized_active_return_level",
                    "portfolio_object": f"constrained_top500_long_only_{aum_label}",
                    "comparison": "active_vs_internal_market",
                    "model_a": "smooth75_ridge_rank",
                    "model_b": "internal_eur_value_weighted_market",
                    "weighting": "optimized",
                    "portfolio": "constrained_long_only",
                    "aum_label": aum_label,
                    "metric": "annualized_active_return",
                    "bootstrap_block": block,
                    "bootstrap_repetitions": bootstrap_repetitions,
                    **active,
                }
            )
            net_sharpe = stationary_bootstrap_metric_ci(
                benchmark[net_column],
                metric="sharpe",
                expected_block=block,
                n_boot=bootstrap_repetitions,
                seed=random_state + 300 + block,
            )
            records.append(
                {
                    "test": "net_sharpe_level",
                    "portfolio_object": f"constrained_top500_long_only_{aum_label}",
                    "comparison": "constrained_revision_strategy",
                    "model_a": "smooth75_ridge_rank",
                    "model_b": "",
                    "weighting": "optimized",
                    "portfolio": "constrained_long_only",
                    "aum_label": aum_label,
                    "metric": "net_sharpe",
                    "bootstrap_block": block,
                    "bootstrap_repetitions": bootstrap_repetitions,
                    **net_sharpe,
                }
            )
    output = pd.DataFrame(records)
    output.to_csv(output_dir / "revision_bootstrap_uncertainty.csv", index=False)
    return output


def _subperiod_definitions(dates: pd.Series) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    ordered = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    midpoint = len(ordered) // 2
    first = ordered.iloc[:midpoint]
    second = ordered.iloc[midpoint:]
    definitions = [
        ("full_oos", ordered.iloc[0], ordered.iloc[-1]),
        (
            f"first_half_{first.iloc[0].year}_{first.iloc[-1].year}",
            first.iloc[0],
            first.iloc[-1],
        ),
        (
            f"second_half_{second.iloc[0].year}_{second.iloc[-1].year}",
            second.iloc[0],
            second.iloc[-1],
        ),
        ("pre_covid_2015_2019", pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31")),
        ("covid_recovery_2020_2022", pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
        ("recent_2023_2026", pd.Timestamp("2023-01-01"), pd.Timestamp("2026-12-31")),
    ]
    return definitions


def monthly_information_coefficients(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    records = []
    for (model, date), group in frame.groupby(["model", "date"], sort=True):
        clean = group[["prediction", "target_return_rank", "target_residual_rank"]].dropna()
        if len(clean) < 10:
            continue
        records.append(
            {
                "model": model,
                "date": date,
                "observations": int(len(clean)),
                "spearman_ic": float(clean["prediction"].corr(clean["target_return_rank"], method="spearman")),
                "residual_spearman_ic": float(
                    clean["prediction"].corr(clean["target_residual_rank"], method="spearman")
                ),
            }
        )
    return pd.DataFrame(records)


def build_subperiod_stability(
    revision_dir: Path,
    output_dir: Path,
    cost_bps: int,
    risk_free: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = _net_revision_returns(pd.read_csv(revision_dir / "monthly_portfolios.csv"), cost_bps)
    predictions = pd.read_parquet(revision_dir / "predictions.parquet")
    monthly_ic = monthly_information_coefficients(predictions)

    return_records = []
    definitions = _subperiod_definitions(monthly["return_date"])
    selected = monthly[
        monthly["model"].eq("ridge_rank")
        & monthly["universe_variant"].eq("standard_ex_bottom_5pct")
        & monthly["weighting"].isin(["equal", "value"])
    ].copy()
    portfolio_columns = [
        ("net_long_short_return", "long_short"),
        ("net_long_only_return", "long_only_top_decile"),
    ]
    for label, start, end in definitions:
        window = selected[selected["return_date"].between(start, end)]
        for weighting in ["equal", "value"]:
            for portfolio_column, portfolio in portfolio_columns:
                group = window[window["weighting"].eq(weighting)]
                stats = annualized_portfolio_return_summary(
                    group[portfolio_column],
                    group["return_date"],
                    risk_free if portfolio == "long_only_top_decile" else None,
                )
                return_records.append(
                    {
                        "subperiod": label,
                        "start_date": start,
                        "end_date": end,
                        "weighting": weighting,
                        "portfolio": portfolio,
                        **stats,
                    }
                )
    subperiod_returns = pd.DataFrame(return_records)

    ic_records = []
    for label, start, end in definitions:
        group = monthly_ic[monthly_ic["date"].between(start, end)]
        for column in ["spearman_ic", "residual_spearman_ic"]:
            stats = annualized_return_summary(group[column])
            ic_records.append(
                {
                    "subperiod": label,
                    "start_date": start,
                    "end_date": end,
                    "model": "ridge_rank",
                    "ic_measure": column,
                    "months": int(group[column].dropna().shape[0]),
                    "mean_monthly_ic": float(group[column].mean()),
                    "ic_information_ratio": stats["net_sharpe"],
                    "positive_ic_month_fraction": float(group[column].dropna().gt(0).mean())
                    if not group.empty
                    else np.nan,
                }
            )
    subperiod_ic = pd.DataFrame(ic_records)

    rolling_records = []
    for (weighting, _, _), group in selected.groupby(
        ["weighting", "universe_variant", "model"],
        sort=True,
    ):
        group = group.sort_values("return_date")
        for column, name in [
            ("net_long_short_return", "long_short"),
            ("net_long_only_return", "long_only_top_decile"),
        ]:
            values = pd.to_numeric(group[column], errors="coerce")
            rolling = values.rolling(36, min_periods=24)
            rolling_records.extend(
                {
                    "date": date,
                    "series": f"{weighting}_{name}_net_sharpe",
                    "rolling_months": 36,
                    "value": (
                        float(window.mean() * PPY / (window.std(ddof=1) * np.sqrt(PPY)))
                        if len(window.dropna()) >= 24 and window.std(ddof=1) > 0
                        else np.nan
                    ),
                }
                for date, window in zip(group["return_date"], rolling, strict=False)
            )
    ic_group = monthly_ic.sort_values("date")
    for column in ["spearman_ic", "residual_spearman_ic"]:
        values = pd.to_numeric(ic_group[column], errors="coerce")
        rolling = values.rolling(36, min_periods=24)
        rolling_records.extend(
            {
                "date": date,
                "series": f"{column}_rolling_ir",
                "rolling_months": 36,
                "value": (
                    float(window.mean() / window.std(ddof=1) * np.sqrt(PPY))
                    if len(window.dropna()) >= 24 and window.std(ddof=1) > 0
                    else np.nan
                ),
            }
            for date, window in zip(ic_group["date"], rolling, strict=False)
        )
    rolling = pd.DataFrame(rolling_records)

    subperiod_returns.to_csv(output_dir / "revision_subperiod_return_stability.csv", index=False)
    subperiod_ic.to_csv(output_dir / "revision_subperiod_ic_stability.csv", index=False)
    rolling.to_csv(output_dir / "revision_rolling_36m_stability.csv", index=False)
    _plot_rolling_stability(rolling, output_dir / "revision_rolling_36m_stability.png")
    return subperiod_returns, subperiod_ic, rolling


def _factor_regression(frame: pd.DataFrame, dependent: str, hac_lags: int) -> dict[str, float]:
    clean = frame[[dependent, *EXTERNAL_FACTORS]].replace([np.inf, -np.inf], np.nan).dropna()
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
        "alpha_annualized": float(fit.params["const"] * PPY),
        "alpha_t": float(fit.tvalues["const"]),
        "alpha_p": float(fit.pvalues["const"]),
        "adjusted_r2": float(fit.rsquared_adj),
    }
    for factor in EXTERNAL_FACTORS:
        result[f"beta_{factor}"] = float(fit.params[factor])
        result[f"p_{factor}"] = float(fit.pvalues[factor])
    return result


def build_factor_spanning_exhibit(
    revision_dir: Path,
    constrained_dir: Path,
    output_dir: Path,
    french_dir: Path,
    fx_file: Path,
    cost_bps: int,
    hac_lags: int,
) -> pd.DataFrame:
    factors = load_external_europe_factors(
        french_dir / "Europe_5_Factors.csv",
        french_dir / "Europe_MOM_Factor.csv",
    )
    fx = load_monthly_eurusd_return(fx_file)
    monthly = pd.read_csv(revision_dir / "monthly_portfolios.csv", parse_dates=["signal_date", "return_date"])
    unconstrained = external_factor_spanning(
        monthly,
        factors,
        fx,
        cost_bps=cost_bps,
        hac_lags=hac_lags,
    )
    unconstrained["portfolio_object"] = (
        "unconstrained_"
        + unconstrained["weighting"].astype(str)
        + "_"
        + unconstrained["universe_variant"].astype(str)
        + "_"
        + unconstrained["portfolio"].astype(str)
    )

    constrained = pd.read_csv(constrained_dir / "constrained_monthly.csv", parse_dates=["target_date"])
    constrained["return_date"] = constrained["target_date"].dt.to_period("M").dt.to_timestamp("M")
    aligned = constrained.merge(
        fx.rename("EURUSD_return").rename_axis("return_date").reset_index(),
        on="return_date",
        how="left",
        validate="many_to_one",
    ).merge(factors, on="return_date", how="left", validate="many_to_one")
    records = []
    for aum_label in ["10m", "100m", "500m"]:
        dependent = f"net_return_{aum_label}"
        if dependent not in aligned:
            continue
        aligned[f"{dependent}_usd"] = (
            (1.0 + pd.to_numeric(aligned[dependent], errors="coerce"))
            * (1.0 + aligned["EURUSD_return"])
            - 1.0
        )
        aligned[f"{dependent}_excess_usd"] = aligned[f"{dependent}_usd"] - aligned["RF"]
        records.append(
            {
                "comparison": "absolute",
                "model": aligned["model"].iloc[0],
                "baseline": "",
                "weighting": "optimized",
                "universe_variant": "top_500_observed_spread",
                "portfolio": "constrained_long_only",
                "cost_bps": cost_bps,
                "aum_label": aum_label,
                "portfolio_object": f"constrained_top500_long_only_{aum_label}",
                **_factor_regression(aligned, f"{dependent}_excess_usd", hac_lags),
            }
        )
    constrained_factors = pd.DataFrame(records)
    constrained_factors["alpha_p_holm"] = constrained_factors["alpha_p"]
    constrained_factors["primary_family"] = constrained_factors["aum_label"].eq("100m")

    combined = pd.concat([unconstrained.assign(aum_label=""), constrained_factors], ignore_index=True)
    combined.to_csv(output_dir / "revision_external_factor_spanning.csv", index=False)
    return combined


def _summary_files(results_root: Path) -> list[Path]:
    names = {
        "model_summary.csv",
        "signal_smoothing_model_summary.csv",
        "constrained_summary.csv",
    }
    return sorted(path for path in results_root.glob("*/*") if path.name in names)


def build_specification_curve(results_root: Path, output_dir: Path) -> pd.DataFrame:
    records = []
    for path in _summary_files(results_root):
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, OSError):
            continue
        if "net_sharpe" not in frame:
            continue
        experiment = path.parent.name
        for _, row in frame.replace([np.inf, -np.inf], np.nan).iterrows():
            net_sharpe = row.get("net_sharpe")
            if pd.isna(net_sharpe):
                continue
            if "subperiod" in row and row.get("subperiod") != "full":
                continue
            records.append(
                {
                    "experiment": experiment,
                    "source_file": path.name,
                    "model": row.get("model", row.get("strategy", "")),
                    "target_mode": row.get("target_mode", ""),
                    "weighting": row.get("weighting", ""),
                    "universe_variant": row.get("universe_variant", ""),
                    "portfolio": row.get("portfolio", row.get("constraint", "")),
                    "cost_bps": row.get("cost_bps", ""),
                    "aum_label": row.get("aum_label", ""),
                    "months": row.get("months", np.nan),
                    "annualized_net_return": row.get(
                        "annualized_net_mean_return",
                        row.get("annualized_net_return", np.nan),
                    ),
                    "annualized_net_volatility": row.get(
                        "annualized_net_volatility",
                        np.nan,
                    ),
                    "net_sharpe": float(net_sharpe),
                }
            )
    curve = pd.DataFrame(records)
    if curve.empty:
        curve.to_csv(output_dir / "specification_curve.csv", index=False)
        return curve
    curve["is_revision_headline_equal_long_short"] = (
        curve["experiment"].eq(HIGHLIGHT_EXPERIMENT)
        & curve["model"].eq("ridge_rank")
        & curve["weighting"].eq("equal")
        & curve["universe_variant"].eq("standard_ex_bottom_5pct")
        & curve["portfolio"].eq("long_short")
    )
    curve["is_revision_value_long_short"] = (
        curve["experiment"].eq(HIGHLIGHT_EXPERIMENT)
        & curve["model"].eq("ridge_rank")
        & curve["weighting"].eq("value")
        & curve["universe_variant"].eq("standard_ex_bottom_5pct")
        & curve["portfolio"].eq("long_short")
    )
    curve["is_revision_constrained_100m"] = (
        curve["experiment"].eq(
            "constrained_estimates_revisions_pure_strict_lag1_revision_signal_fixed"
        )
        & curve["aum_label"].eq("100m")
    )
    curve = curve.sort_values("net_sharpe").reset_index(drop=True)
    curve["specification_rank"] = np.arange(1, len(curve) + 1)
    curve.to_csv(output_dir / "specification_curve.csv", index=False)
    _plot_specification_curve(curve, output_dir / "specification_curve.png")
    return curve


def _plot_rolling_stability(rolling: pd.DataFrame, output_path: Path) -> None:
    if rolling.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    sharpe_series = [
        "equal_long_short_net_sharpe",
        "value_long_short_net_sharpe",
    ]
    for series in sharpe_series:
        subset = rolling[rolling["series"].eq(series)].dropna(subset=["value"])
        axes[0].plot(subset["date"], subset["value"], label=series)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Rolling Sharpe")
    axes[0].legend(loc="best", fontsize=8)
    for series in ["spearman_ic_rolling_ir", "residual_spearman_ic_rolling_ir"]:
        subset = rolling[rolling["series"].eq(series)].dropna(subset=["value"])
        axes[1].plot(subset["date"], subset["value"], label=series)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Rolling IC IR")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_specification_curve(curve: pd.DataFrame, output_path: Path) -> None:
    if curve.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(curve["specification_rank"], curve["net_sharpe"], s=10, alpha=0.35)
    median = curve["net_sharpe"].median()
    ax.axhline(median, color="black", linewidth=0.8, linestyle="--", label="median")
    highlights = [
        ("is_revision_headline_equal_long_short", "headline equal LS"),
        ("is_revision_value_long_short", "value-weight LS"),
        ("is_revision_constrained_100m", "constrained 100m"),
    ]
    for flag, label in highlights:
        if flag not in curve:
            continue
        subset = curve[curve[flag]]
        ax.scatter(
            subset["specification_rank"],
            subset["net_sharpe"],
            s=55,
            label=label,
        )
    ax.set_xlabel("Specification rank")
    ax.set_ylabel("Net Sharpe")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_ridge_coefficients(coefficients: pd.DataFrame, output_path: Path) -> None:
    if coefficients.empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for feature in REVISION_FEATURE_ORDER:
        subset = coefficients[coefficients["feature"].astype(str).eq(feature)]
        if subset.empty:
            continue
        ax.plot(
            subset["test_year"],
            subset["coefficient"],
            marker="o",
            linewidth=1.4,
            label=feature.replace("est_", "").replace("_rank", ""),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("OOS test year")
    ax.set_ylabel("Ridge coefficient")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_revision_strategy_exhibits(
    revision_dir: Path,
    constrained_dir: Path,
    dre_dir: Path | None,
    results_root: Path,
    output_dir: Path,
    french_dir: Path,
    fx_file: Path,
    eur_rate: Path,
    cost_bps: int,
    hac_lags: int,
    bootstrap_repetitions: int,
    bootstrap_blocks: tuple[int, ...],
    random_state: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    risk_free = load_eur_short_rate(eur_rate) if eur_rate.exists() else None
    implementability = build_implementability_exhibit(revision_dir, constrained_dir, output_dir)
    breakeven = build_breakeven_cost_capacity(
        revision_dir,
        constrained_dir,
        output_dir,
        cost_bps,
    )
    coefficient_path, coefficient_summary = build_ridge_coefficient_stability(
        revision_dir,
        output_dir,
    )
    bootstrap = build_bootstrap_uncertainty(
        revision_dir,
        constrained_dir,
        dre_dir,
        output_dir,
        cost_bps,
        bootstrap_repetitions,
        bootstrap_blocks,
        random_state,
        risk_free,
    )
    subperiod_returns, subperiod_ic, rolling = build_subperiod_stability(
        revision_dir,
        output_dir,
        cost_bps,
        risk_free,
    )
    factors = build_factor_spanning_exhibit(
        revision_dir,
        constrained_dir,
        output_dir,
        french_dir,
        fx_file,
        cost_bps,
        hac_lags,
    )
    curve = build_specification_curve(results_root, output_dir)
    manifest = {
        "inputs": {
            "revision_dir": str(revision_dir),
            "constrained_dir": str(constrained_dir),
            "dre_dir": str(dre_dir) if dre_dir is not None else None,
            "results_root": str(results_root),
            "french_dir": str(french_dir),
            "fx_file": str(fx_file),
            "eur_rate": str(eur_rate),
        },
        "cost_bps": cost_bps,
        "hac_lags": hac_lags,
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_blocks": bootstrap_blocks,
        "random_state": random_state,
        "rows": {
            "implementability": int(len(implementability)),
            "breakeven_cost_capacity": int(len(breakeven)),
            "ridge_coefficient_path": int(len(coefficient_path)),
            "ridge_coefficient_stability": int(len(coefficient_summary)),
            "bootstrap_uncertainty": int(len(bootstrap)),
            "subperiod_returns": int(len(subperiod_returns)),
            "subperiod_ic": int(len(subperiod_ic)),
            "rolling": int(len(rolling)),
            "factor_spanning": int(len(factors)),
            "specification_curve": int(len(curve)),
        },
        "outputs": {
            "implementability": str(output_dir / "revision_implementability_exhibit.csv"),
            "breakeven_cost_capacity": str(
                output_dir / "revision_breakeven_cost_capacity.csv"
            ),
            "ridge_coefficient_path": str(
                output_dir / "revision_ridge_coefficient_path.csv"
            ),
            "ridge_coefficient_stability": str(
                output_dir / "revision_ridge_coefficient_stability.csv"
            ),
            "ridge_coefficient_plot": str(
                output_dir / "revision_ridge_coefficient_stability.png"
            ),
            "bootstrap_uncertainty": str(
                output_dir / "revision_bootstrap_uncertainty.csv"
            ),
            "subperiod_returns": str(output_dir / "revision_subperiod_return_stability.csv"),
            "subperiod_ic": str(output_dir / "revision_subperiod_ic_stability.csv"),
            "rolling": str(output_dir / "revision_rolling_36m_stability.csv"),
            "rolling_plot": str(output_dir / "revision_rolling_36m_stability.png"),
            "factor_spanning": str(output_dir / "revision_external_factor_spanning.csv"),
            "specification_curve": str(output_dir / "specification_curve.csv"),
            "specification_curve_plot": str(output_dir / "specification_curve.png"),
            "manifest": str(output_dir / "manifest.json"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-dir", type=Path, default=DEFAULT_REVISION_DIR)
    parser.add_argument("--constrained-dir", type=Path, default=DEFAULT_CONSTRAINED_DIR)
    parser.add_argument("--dre-dir", type=Path, default=DEFAULT_DRE_DIR)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--french-dir", type=Path, default=DEFAULT_FRENCH_DIR)
    parser.add_argument("--fx-file", type=Path, default=DEFAULT_FX)
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5_000)
    parser.add_argument("--bootstrap-blocks", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dre_dir = args.dre_dir if args.dre_dir.exists() else None
    manifest = build_revision_strategy_exhibits(
        revision_dir=args.revision_dir,
        constrained_dir=args.constrained_dir,
        dre_dir=dre_dir,
        results_root=args.results_root,
        output_dir=args.output_dir,
        french_dir=args.french_dir,
        fx_file=args.fx_file,
        eur_rate=args.eur_rate,
        cost_bps=args.cost_bps,
        hac_lags=args.hac_lags,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_blocks=tuple(args.bootstrap_blocks),
        random_state=args.random_state,
    )
    print(json.dumps(manifest["rows"], indent=2))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
