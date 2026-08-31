"""Brinson-Fachler attribution of active return into country/sector effects.

This module answers a single question for the constrained long-only strategies:
is the measured active return a *country or sector allocation* bet, or is it
*within-group stock selection*? The distinction matters because a broad regional
or sector tilt is cheap to replicate and is not evidence that the ML signal
picks stocks.

Decomposition (Brinson-Fachler), per month t and group g:

    allocation_g  = (w_p,g - w_b,g) * (r_b,g - r_b)
    selection_g   =  w_b,g          * (r_p,g - r_b,g)
    interaction_g = (w_p,g - w_b,g) * (r_p,g - r_b,g)

with w_p / w_b the portfolio and benchmark group weights, r_p,g / r_b,g the
group returns and r_b the total benchmark return. Summing all three effects
across groups recovers the gross active return exactly:

    sum_g (allocation_g + selection_g + interaction_g) = r_p - r_b

The identity holds for any fill applied to an absent group's return as long as
both weight vectors sum to one, because the r_b terms cancel across groups. The
fill only shifts attribution between the allocation and interaction legs, so
groups the benchmark does not hold are reported separately in the audit.

Attribution is computed on *gross* returns because the benchmark is gross. The
cost drag is reported as an explicit reconciling line rather than being folded
into a group effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import stats as project_stats


PPY = 12
EFFECT_COLUMNS = ("allocation", "selection", "interaction")


def build_benchmark_group_panel(
    panel: pd.DataFrame,
    group_column: str,
    *,
    date_column: str = "date",
    return_column: str = "return_1m",
    cap_column: str = "company_market_cap",
    missing_label: str = "UNKNOWN",
) -> pd.DataFrame:
    """Cap-weighted benchmark weights and returns per group and month.

    Mirrors ``asset_pricing_depth.build_internal_market`` exactly -- lagged
    (formation) market caps, consecutive-month requirement -- so that the group
    weights aggregate back to the same internal EUR value-weighted benchmark
    used by the constrained long-only runners. Any drift between the two would
    silently break the attribution identity, so the caller should verify the
    reconstructed total against the published benchmark series.
    """
    required = {date_column, "ric", return_column, cap_column, group_column}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Benchmark group panel missing columns: {sorted(missing)}")

    work = panel[sorted(required)].copy()
    work[date_column] = pd.to_datetime(work[date_column])
    work = work.sort_values(["ric", date_column])
    work["month_number"] = (
        work[date_column].dt.year * 12 + work[date_column].dt.month
    )
    grouped = work.groupby("ric", sort=False)
    previous_cap = grouped[cap_column].shift(1)
    consecutive = (
        work["month_number"].sub(grouped["month_number"].shift(1)).eq(1)
    )
    work["formation_cap"] = previous_cap.where(consecutive & previous_cap.gt(0))
    work[group_column] = work[group_column].fillna(missing_label).astype(str)

    valid = work[return_column].notna() & work["formation_cap"].gt(0)
    work = work.loc[valid].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "group",
                "benchmark_weight",
                "benchmark_return",
                "benchmark_n",
            ]
        )

    work["weighted_return"] = work[return_column] * work["formation_cap"]
    by_group = work.groupby([date_column, group_column], as_index=False).agg(
        weighted_return=("weighted_return", "sum"),
        group_cap=("formation_cap", "sum"),
        benchmark_n=("ric", "nunique"),
    )
    total_cap = by_group.groupby(date_column)["group_cap"].transform("sum")
    by_group["benchmark_weight"] = by_group["group_cap"].div(total_cap)
    by_group["benchmark_return"] = by_group["weighted_return"].div(
        by_group["group_cap"]
    )
    by_group = by_group.rename(
        columns={date_column: "date", group_column: "group"}
    )
    return by_group[
        ["date", "group", "benchmark_weight", "benchmark_return", "benchmark_n"]
    ].sort_values(["date", "group"], ignore_index=True)


def portfolio_group_exposures(
    holdings: pd.DataFrame,
    group_column: str,
    *,
    weight_column: str = "weight",
    return_column: str = "target_return_1m",
    missing_label: str = "UNKNOWN",
) -> pd.DataFrame:
    """Collapse per-name holdings into group weights and group returns."""
    required = {group_column, weight_column, return_column}
    missing = required - set(holdings)
    if missing:
        raise ValueError(f"Holdings missing columns: {sorted(missing)}")

    work = holdings.copy()
    work[group_column] = work[group_column].fillna(missing_label).astype(str)
    work[weight_column] = pd.to_numeric(work[weight_column], errors="coerce")
    work[return_column] = pd.to_numeric(work[return_column], errors="coerce")
    work = work[work[weight_column].notna() & work[return_column].notna()]
    work["contribution"] = work[weight_column] * work[return_column]

    grouped = work.groupby(group_column, as_index=False).agg(
        portfolio_weight=(weight_column, "sum"),
        contribution=("contribution", "sum"),
        portfolio_n=(weight_column, "size"),
    )
    grouped["portfolio_return"] = np.where(
        grouped["portfolio_weight"].abs() > 1e-12,
        grouped["contribution"].div(grouped["portfolio_weight"]),
        np.nan,
    )
    return grouped.rename(columns={group_column: "group"})[
        ["group", "portfolio_weight", "portfolio_return", "portfolio_n"]
    ]


def brinson_attribution(
    portfolio_groups: pd.DataFrame,
    benchmark_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Per-month, per-group Brinson-Fachler effects.

    Both frames must carry a ``date`` and ``group`` column. Groups absent from
    one side enter with zero weight; their return is filled with the
    corresponding total benchmark return so the effect lands in the
    interaction leg rather than being silently dropped.
    """
    portfolio = portfolio_groups.copy()
    benchmark = benchmark_groups.copy()
    for frame, name in ((portfolio, "portfolio"), (benchmark, "benchmark")):
        for column in ("date", "group"):
            if column not in frame:
                raise ValueError(f"{name} groups missing '{column}' column")
        frame["date"] = pd.to_datetime(frame["date"])
        frame["group"] = frame["group"].astype(str)

    merged = portfolio.merge(benchmark, on=["date", "group"], how="outer")
    merged["portfolio_weight"] = merged["portfolio_weight"].fillna(0.0)
    merged["benchmark_weight"] = merged["benchmark_weight"].fillna(0.0)

    # Total benchmark return per month, computed before any return fill so the
    # fill cannot contaminate the r_b reference point.
    benchmark_total = (
        merged.assign(
            weighted=merged["benchmark_weight"] * merged["benchmark_return"].fillna(0.0)
        )
        .groupby("date")["weighted"]
        .sum()
        .rename("benchmark_total_return")
    )
    merged = merged.merge(benchmark_total, on="date", how="left")

    merged["benchmark_return"] = merged["benchmark_return"].fillna(
        merged["benchmark_total_return"]
    )
    merged["portfolio_return"] = merged["portfolio_return"].fillna(
        merged["benchmark_return"]
    )
    merged["active_weight"] = (
        merged["portfolio_weight"] - merged["benchmark_weight"]
    )
    merged["allocation"] = merged["active_weight"] * (
        merged["benchmark_return"] - merged["benchmark_total_return"]
    )
    merged["selection"] = merged["benchmark_weight"] * (
        merged["portfolio_return"] - merged["benchmark_return"]
    )
    merged["interaction"] = merged["active_weight"] * (
        merged["portfolio_return"] - merged["benchmark_return"]
    )
    merged["total_effect"] = merged[list(EFFECT_COLUMNS)].sum(axis=1)
    merged["held_off_benchmark"] = (
        merged["benchmark_weight"].le(0.0) & merged["portfolio_weight"].gt(0.0)
    )
    return merged.sort_values(["date", "group"], ignore_index=True)


def monthly_effect_series(attribution: pd.DataFrame) -> pd.DataFrame:
    """Aggregate group-level effects to one row per month."""
    work = attribution.copy()
    work["off_benchmark_weight"] = work["portfolio_weight"].where(
        work["held_off_benchmark"], 0.0
    )
    monthly = work.groupby("date", as_index=False).agg(
        allocation=("allocation", "sum"),
        selection=("selection", "sum"),
        interaction=("interaction", "sum"),
        gross_active_return=("total_effect", "sum"),
        groups_n=("group", "nunique"),
        off_benchmark_weight=("off_benchmark_weight", "sum"),
    )
    return monthly.sort_values("date", ignore_index=True)


def summarize_effects(
    monthly: pd.DataFrame,
    *,
    ppy: int = PPY,
    maxlags: int | None = None,
) -> pd.DataFrame:
    """Annualize each effect and HAC-test its monthly mean against zero."""
    records = []
    columns = [*EFFECT_COLUMNS, "gross_active_return"]
    for column in columns:
        if column not in monthly:
            continue
        series = pd.to_numeric(monthly[column], errors="coerce").dropna()
        if series.empty:
            continue
        test = project_stats.hac_mean_diff_test(series, maxlags=maxlags)
        records.append(
            {
                "effect": column,
                "months": int(len(series)),
                "mean_monthly": float(series.mean()),
                "annualized": float(series.mean() * ppy),
                "annualized_volatility": float(
                    series.std(ddof=1) * np.sqrt(ppy)
                ),
                "share_of_active": (
                    float(
                        series.sum()
                        / monthly["gross_active_return"].sum()
                    )
                    if column != "gross_active_return"
                    and abs(float(monthly["gross_active_return"].sum())) > 1e-12
                    else np.nan
                ),
                "hac_t_statistic": float(test["t"]),
                "hac_p_value": float(test["p_two_sided"]),
                "hac_ci_low_annualized": float(test["ci_low"] * ppy),
                "hac_ci_high_annualized": float(test["ci_high"] * ppy),
                "hac_maxlags": test["maxlags"],
            }
        )
    return pd.DataFrame(records)


def top_group_contributions(
    attribution: pd.DataFrame,
    *,
    top_n: int = 10,
    ppy: int = PPY,
) -> pd.DataFrame:
    """Rank groups by annualized total contribution to active return."""
    months = attribution["date"].nunique()
    if months == 0:
        return pd.DataFrame()
    grouped = attribution.groupby("group", as_index=False).agg(
        mean_portfolio_weight=("portfolio_weight", "mean"),
        mean_benchmark_weight=("benchmark_weight", "mean"),
        mean_active_weight=("active_weight", "mean"),
        allocation=("allocation", "sum"),
        selection=("selection", "sum"),
        interaction=("interaction", "sum"),
        total_effect=("total_effect", "sum"),
    )
    for column in (*EFFECT_COLUMNS, "total_effect"):
        grouped[f"annualized_{column}"] = grouped[column].div(months).mul(ppy)
    grouped = grouped.sort_values(
        "annualized_total_effect", ascending=False, ignore_index=True
    )
    return grouped.head(top_n)
