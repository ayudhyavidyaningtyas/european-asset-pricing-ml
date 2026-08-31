"""Compustat Global feature enrichment for the European equity panel.

The base asset-pricing panel is built from Refinitiv/Datastream data.  This
module adds a deeper Compustat Global characteristic block while preserving the
base panel and its original feature definitions.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from asset_pricing import ALL_RAW_FEATURES, PanelConfig


COMPUSTAT_ANNUAL_FEATURES = [
    "comp_asset_growth",
    "comp_sales_growth",
    "comp_equity_growth",
    "comp_gross_profitability",
    "comp_roa",
    "comp_roe",
    "comp_operating_margin",
    "comp_gross_margin",
    "comp_asset_turnover",
    "comp_leverage",
    "comp_debt_to_assets",
    "comp_debt_to_equity",
    "comp_cash_to_assets",
    "comp_cash_to_debt",
    "comp_current_ratio",
    "comp_working_capital_to_assets",
    "comp_inventory_to_assets",
    "comp_receivables_to_assets",
    "comp_ppe_to_assets",
    "comp_intangibles_to_assets",
    "comp_rd_to_assets",
    "comp_sga_to_sales",
    "comp_depreciation_to_assets",
    "comp_capex_to_assets",
    "comp_accruals_to_assets",
    "comp_payout_to_assets",
]

COMPUSTAT_MARKET_FEATURES = [
    "comp_log_price",
    "comp_log_volume",
    "comp_price_momentum_6_2",
    "comp_price_momentum_12_2",
    "comp_price_volatility_12m",
    "comp_volume_growth_12m",
]

COMPUSTAT_EXTENSION_FEATURES = [
    "comp_book_to_market",
    *COMPUSTAT_ANNUAL_FEATURES,
    *COMPUSTAT_MARKET_FEATURES,
]

COMPUSTAT_FEATURE_DICTIONARY = [
    (
        "comp_book_to_market",
        "Compustat common/ordinary equity divided by Refinitiv month-end market capitalisation",
        "Compustat annual report + lag, month t market cap",
    ),
    (
        "comp_asset_growth",
        "Annual growth in Compustat total assets, requiring a normal fiscal gap and unchanged reporting currency",
        "annual report + lag",
    ),
    (
        "comp_sales_growth",
        "Annual growth in Compustat revenue or net sales, requiring a normal fiscal gap and unchanged reporting currency",
        "annual report + lag",
    ),
    (
        "comp_equity_growth",
        "Annual growth in Compustat common/ordinary equity, requiring a normal fiscal gap and unchanged reporting currency",
        "annual report + lag",
    ),
    (
        "comp_gross_profitability",
        "Revenue less cost of goods sold, divided by total assets",
        "annual report + lag",
    ),
    ("comp_roa", "Net income divided by total assets", "annual report + lag"),
    ("comp_roe", "Net income divided by common/ordinary equity", "annual report + lag"),
    (
        "comp_operating_margin",
        "Operating income after depreciation divided by revenue",
        "annual report + lag",
    ),
    (
        "comp_gross_margin",
        "Revenue less cost of goods sold, divided by revenue",
        "annual report + lag",
    ),
    ("comp_asset_turnover", "Revenue divided by total assets", "annual report + lag"),
    ("comp_leverage", "Total liabilities divided by total assets", "annual report + lag"),
    (
        "comp_debt_to_assets",
        "Long-term debt plus debt in current liabilities, divided by total assets",
        "annual report + lag",
    ),
    (
        "comp_debt_to_equity",
        "Long-term debt plus debt in current liabilities, divided by common/ordinary equity",
        "annual report + lag",
    ),
    ("comp_cash_to_assets", "Cash and short-term investments divided by total assets", "annual report + lag"),
    (
        "comp_cash_to_debt",
        "Cash and short-term investments divided by total debt",
        "annual report + lag",
    ),
    ("comp_current_ratio", "Current assets divided by current liabilities", "annual report + lag"),
    (
        "comp_working_capital_to_assets",
        "Current assets less current liabilities, divided by total assets",
        "annual report + lag",
    ),
    ("comp_inventory_to_assets", "Inventories divided by total assets", "annual report + lag"),
    ("comp_receivables_to_assets", "Receivables divided by total assets", "annual report + lag"),
    ("comp_ppe_to_assets", "Net property, plant and equipment divided by total assets", "annual report + lag"),
    ("comp_intangibles_to_assets", "Intangible assets divided by total assets", "annual report + lag"),
    ("comp_rd_to_assets", "Research and development expense divided by total assets", "annual report + lag"),
    (
        "comp_sga_to_sales",
        "Selling, general and administrative expense divided by revenue",
        "annual report + lag",
    ),
    (
        "comp_depreciation_to_assets",
        "Depreciation and amortization divided by total assets",
        "annual report + lag",
    ),
    ("comp_capex_to_assets", "Capital expenditures divided by total assets", "annual report + lag"),
    (
        "comp_accruals_to_assets",
        "Net income less operating cash flow, divided by total assets",
        "annual report + lag",
    ),
    ("comp_payout_to_assets", "Total dividends divided by total assets", "annual report + lag"),
    ("comp_log_price", "Natural log of absolute Compustat monthly close price", "month t"),
    ("comp_log_volume", "Natural log of Compustat monthly trading volume", "month t"),
    (
        "comp_price_momentum_6_2",
        "Compounded adjusted-price returns over months t-5 to t-1",
        "excludes month t",
    ),
    (
        "comp_price_momentum_12_2",
        "Compounded adjusted-price returns over months t-11 to t-1",
        "excludes month t",
    ),
    (
        "comp_price_volatility_12m",
        "Standard deviation of adjusted-price returns over months t-11 to t",
        "month t",
    ),
    (
        "comp_volume_growth_12m",
        "Twelve-month growth in Compustat monthly trading volume",
        "month t",
    ),
]

ANNUAL_HELPER_COLUMNS = [
    "comp_gvkey",
    "comp_period_end",
    "comp_available_date",
    "comp_reporting_currency",
    "comp_book_equity",
]

MONTHLY_HELPER_COLUMNS = [
    "comp_monthly_gvkey",
    "comp_iid",
    "comp_exchange_code",
    "comp_monthly_currency",
    "comp_adjusted_price",
    "comp_price_return_1m",
]


def _normalise_isin(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})
    )


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    positive_denominator: bool = True,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = num.div(den)
    valid = den.notna() & den.ne(0)
    if positive_denominator:
        valid &= den.gt(0)
    return out.where(valid & np.isfinite(out))


def _sum_min_count(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame]
    if not available:
        return pd.Series(np.nan, index=frame.index)
    return frame[available].sum(axis=1, min_count=1)


def _month_end(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def _rolling_compound(
    values: pd.Series,
    groups: pd.Series,
    window: int,
    min_periods: int,
    shift: int = 0,
) -> pd.Series:
    clipped = values.where(values.gt(-1.0))
    log_values = np.log1p(clipped)
    if shift:
        log_values = log_values.groupby(groups, sort=False).shift(shift)
    rolled = (
        log_values.groupby(groups, sort=False)
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return np.expm1(rolled).sort_index()


def _feature_rank_frame(
    panel: pd.DataFrame,
    features: list[str],
    eligible_column: str = "eligible",
) -> pd.DataFrame:
    out = panel.copy()
    static_eligible = out[eligible_column].fillna(False)
    for feature in features:
        if feature not in out:
            out[feature] = np.nan
        rank_column = f"{feature}_rank"
        valid = static_eligible & out[feature].notna()
        out[rank_column] = np.nan
        ranks = out.loc[valid].groupby("date")[feature].rank(method="average", pct=True)
        out.loc[valid, rank_column] = ranks.mul(2).sub(1)
        out.loc[static_eligible & out[rank_column].isna(), rank_column] = 0.0
    return out


def load_compustat_exports(
    compustat_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_path = compustat_dir / "compustat_global_fundamentals_annual.csv.gz"
    monthly_path = compustat_dir / "compustat_global_security_monthly.csv.gz"
    if not annual_path.exists():
        raise FileNotFoundError(f"Missing Compustat annual export: {annual_path}")
    if not monthly_path.exists():
        raise FileNotFoundError(f"Missing Compustat monthly export: {monthly_path}")
    annual = pd.read_csv(annual_path, compression="gzip", dtype={"gvkey": str}, low_memory=False)
    monthly = pd.read_csv(
        monthly_path,
        compression="gzip",
        dtype={"gvkey": str, "iid": str},
        low_memory=False,
    )
    return annual, monthly


def prepare_compustat_annual_features(
    annual: pd.DataFrame,
    config: PanelConfig,
) -> tuple[pd.DataFrame, dict]:
    work = annual.copy()
    work.columns = [str(column).lower() for column in work.columns]
    required = {"gvkey", "isin", "datadate"}
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
        "dvt",
        "ebit",
        "ebitda",
        "ib",
        "intan",
        "invt",
        "lct",
        "lt",
        "nicon",
        "oancf",
        "oiadp",
        "oibdp",
        "ppent",
        "pstk",
        "rect",
        "revt",
        "sale",
        "seq",
        "xsga",
        "xrd",
    ]
    for column in numeric_columns:
        if column not in work:
            work[column] = np.nan
    if "curcd" not in work:
        work["curcd"] = pd.NA
    _to_numeric(work, numeric_columns)
    work["isin_norm"] = _normalise_isin(work["isin"])
    work["comp_period_end"] = _month_end(work["datadate"])

    base_available = work["comp_period_end"] + pd.offsets.MonthEnd(
        config.accounting_lag_months
    )
    final_date = _month_end(work["fdate"]) if "fdate" in work else pd.Series(pd.NaT, index=work.index)
    preliminary_date = _month_end(work["pdate"]) if "pdate" in work else pd.Series(pd.NaT, index=work.index)
    report_date = final_date.combine_first(preliminary_date)
    work["comp_available_date"] = pd.concat(
        [base_available, report_date], axis=1
    ).max(axis=1)

    value_columns = [column for column in numeric_columns if column in work]
    work["source_completeness"] = work[value_columns].notna().sum(axis=1)
    work = work.dropna(subset=["isin_norm", "comp_period_end", "comp_available_date"])
    work = (
        work.sort_values(["isin_norm", "comp_period_end", "source_completeness"])
        .groupby(["isin_norm", "comp_period_end"], as_index=False)
        .last()
        .sort_values(["isin_norm", "comp_period_end"])
        .reset_index(drop=True)
    )

    revenue = work["revt"].combine_first(work["sale"])
    total_debt = _sum_min_count(work, ["dltt", "dlc"])
    preferred_stock = work["pstk"].fillna(0.0)
    book_equity = work["ceq"].combine_first(work["seq"].sub(preferred_stock))
    book_equity = book_equity.combine_first(work["seq"])
    net_income = work["nicon"].combine_first(work["ib"])

    positive_assets = work["at"].where(work["at"].gt(0))
    positive_revenue = revenue.where(revenue.gt(0))
    positive_book_equity = book_equity.where(book_equity.gt(0))
    positive_debt = total_debt.where(total_debt.gt(0))

    work["comp_book_equity"] = book_equity
    work["comp_gross_profitability"] = revenue.sub(work["cogs"]).div(positive_assets)
    work["comp_roa"] = net_income.div(positive_assets)
    work["comp_roe"] = net_income.div(positive_book_equity)
    work["comp_operating_margin"] = work["oiadp"].div(positive_revenue)
    work["comp_gross_margin"] = revenue.sub(work["cogs"]).div(positive_revenue)
    work["comp_asset_turnover"] = revenue.div(positive_assets)
    work["comp_leverage"] = work["lt"].div(positive_assets)
    work["comp_debt_to_assets"] = total_debt.div(positive_assets)
    work["comp_debt_to_equity"] = total_debt.div(positive_book_equity)
    work["comp_cash_to_assets"] = work["che"].div(positive_assets)
    work["comp_cash_to_debt"] = work["che"].div(positive_debt)
    work["comp_current_ratio"] = work["act"].div(work["lct"].where(work["lct"].gt(0)))
    work["comp_working_capital_to_assets"] = work["act"].sub(work["lct"]).div(
        positive_assets
    )
    work["comp_inventory_to_assets"] = work["invt"].div(positive_assets)
    work["comp_receivables_to_assets"] = work["rect"].div(positive_assets)
    work["comp_ppe_to_assets"] = work["ppent"].div(positive_assets)
    work["comp_intangibles_to_assets"] = work["intan"].div(positive_assets)
    work["comp_rd_to_assets"] = work["xrd"].div(positive_assets)
    work["comp_sga_to_sales"] = work["xsga"].div(positive_revenue)
    work["comp_depreciation_to_assets"] = work["dp"].div(positive_assets)
    work["comp_capex_to_assets"] = work["capx"].div(positive_assets)
    work["comp_accruals_to_assets"] = net_income.sub(work["oancf"]).div(positive_assets)
    work["comp_payout_to_assets"] = work["dvt"].div(positive_assets)

    grouped = work.groupby("isin_norm", sort=False)
    prior_period = grouped["comp_period_end"].shift(1)
    fiscal_gap_months = (
        (work["comp_period_end"].dt.year - prior_period.dt.year) * 12
        + work["comp_period_end"].dt.month
        - prior_period.dt.month
    )
    annual_gap = fiscal_gap_months.between(9, 18)
    prior_currency = grouped["curcd"].shift(1) if "curcd" in work else pd.Series(pd.NA, index=work.index)
    same_currency = work.get("curcd", pd.Series(pd.NA, index=work.index)).eq(prior_currency)
    same_currency |= work.get("curcd", pd.Series(pd.NA, index=work.index)).isna() | prior_currency.isna()

    def growth(values: pd.Series) -> pd.Series:
        prior = values.groupby(work["isin_norm"], sort=False).shift(1)
        result = values.div(prior).sub(1)
        return result.where(annual_gap & same_currency & values.gt(0) & prior.gt(0))

    work["comp_asset_growth"] = growth(work["at"])
    work["comp_sales_growth"] = growth(revenue)
    work["comp_equity_growth"] = growth(book_equity)

    keep = [
        "isin_norm",
        "gvkey",
        "comp_period_end",
        "comp_available_date",
        "curcd",
        "comp_book_equity",
        *COMPUSTAT_ANNUAL_FEATURES,
    ]
    out = work[keep].rename(
        columns={"gvkey": "comp_gvkey", "curcd": "comp_reporting_currency"}
    )
    audit = {
        "source_rows": int(len(annual)),
        "collapsed_rows": int(len(out)),
        "unique_isins": int(out["isin_norm"].nunique()),
        "unique_gvkeys": int(out["comp_gvkey"].nunique()),
        "first_period_end": str(out["comp_period_end"].min().date()) if not out.empty else None,
        "last_period_end": str(out["comp_period_end"].max().date()) if not out.empty else None,
        "accounting_lag_months": config.accounting_lag_months,
        "feature_non_null": {
            feature: int(out[feature].notna().sum())
            for feature in COMPUSTAT_ANNUAL_FEATURES
        },
    }
    return out, audit


def prepare_compustat_monthly_features(
    monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    work = monthly.copy()
    work.columns = [str(column).lower() for column in work.columns]
    required = {"gvkey", "isin", "iid", "datadate", "prccm", "ajexm", "cshtrm"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(f"Compustat monthly export missing columns: {sorted(missing)}")

    _to_numeric(work, ["prccm", "ajexm", "cshtrm"])
    for column in ["exchg", "curcddvm"]:
        if column not in work:
            work[column] = pd.NA
    work["isin_norm"] = _normalise_isin(work["isin"])
    work["date"] = _month_end(work["datadate"])
    work["source_completeness"] = work[["prccm", "ajexm", "cshtrm"]].notna().sum(axis=1)
    work = work.dropna(subset=["isin_norm", "date"])
    work = (
        work.sort_values(["isin_norm", "date", "source_completeness"])
        .groupby(["isin_norm", "date"], as_index=False)
        .last()
        .sort_values(["isin_norm", "date"])
        .reset_index(drop=True)
    )

    price = work["prccm"].abs()
    adjustment = work["ajexm"].where(work["ajexm"].gt(0))
    work["comp_adjusted_price"] = price.div(adjustment).where(price.gt(0))
    work["comp_log_price"] = np.log(price.where(price.gt(0)))
    work["comp_log_volume"] = np.log(work["cshtrm"].where(work["cshtrm"].gt(0)))
    work["month_index"] = work["date"].dt.year * 12 + work["date"].dt.month

    grouped = work.groupby("isin_norm", sort=False)
    prior_price = grouped["comp_adjusted_price"].shift(1)
    prior_month = grouped["month_index"].shift(1)
    work["comp_price_return_1m"] = (
        work["comp_adjusted_price"].div(prior_price).sub(1)
    ).where(
        work["month_index"].sub(prior_month).eq(1)
        & work["comp_adjusted_price"].gt(0)
        & prior_price.gt(0)
    )

    groups = work["isin_norm"]
    work["comp_price_momentum_6_2"] = _rolling_compound(
        work["comp_price_return_1m"], groups, window=5, min_periods=3, shift=1
    )
    work["comp_price_momentum_12_2"] = _rolling_compound(
        work["comp_price_return_1m"], groups, window=11, min_periods=8, shift=1
    )
    work["comp_price_volatility_12m"] = (
        work["comp_price_return_1m"]
        .groupby(groups, sort=False)
        .rolling(12, min_periods=8)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    five_month_gap = work["month_index"].sub(grouped["month_index"].shift(5)).eq(5)
    twelve_month_gap = work["month_index"].sub(grouped["month_index"].shift(11)).eq(11)
    work.loc[~five_month_gap, "comp_price_momentum_6_2"] = np.nan
    work.loc[
        ~twelve_month_gap,
        ["comp_price_momentum_12_2", "comp_price_volatility_12m"],
    ] = np.nan

    lagged_volume = grouped["cshtrm"].shift(12)
    exact_year = work["month_index"].sub(grouped["month_index"].shift(12)).eq(12)
    work["comp_volume_growth_12m"] = work["cshtrm"].div(lagged_volume).sub(1).where(
        exact_year & work["cshtrm"].gt(0) & lagged_volume.gt(0)
    )

    keep = [
        "isin_norm",
        "date",
        "gvkey",
        "iid",
        "exchg",
        "curcddvm",
        "comp_adjusted_price",
        "comp_price_return_1m",
        *COMPUSTAT_MARKET_FEATURES,
    ]
    out = work[keep].rename(
        columns={
            "gvkey": "comp_monthly_gvkey",
            "iid": "comp_iid",
            "exchg": "comp_exchange_code",
            "curcddvm": "comp_monthly_currency",
        }
    )
    audit = {
        "source_rows": int(len(monthly)),
        "collapsed_rows": int(len(out)),
        "unique_isins": int(out["isin_norm"].nunique()),
        "unique_gvkeys": int(out["comp_monthly_gvkey"].nunique()),
        "first_month": str(out["date"].min().date()) if not out.empty else None,
        "last_month": str(out["date"].max().date()) if not out.empty else None,
        "feature_non_null": {
            feature: int(out[feature].notna().sum())
            for feature in COMPUSTAT_MARKET_FEATURES
        },
    }
    return out, audit


def merge_compustat_features(
    base_panel: pd.DataFrame,
    annual_features: pd.DataFrame,
    monthly_features: pd.DataFrame,
) -> pd.DataFrame:
    panel = base_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").astype("datetime64[ns]")
    panel["_row_order"] = np.arange(len(panel))
    panel["isin_norm"] = _normalise_isin(panel["TR.ISIN"])

    annual = annual_features.dropna(subset=["isin_norm", "comp_available_date"]).copy()
    annual["comp_available_date"] = pd.to_datetime(
        annual["comp_available_date"], errors="coerce"
    ).astype("datetime64[ns]")
    if annual.empty:
        for column in ANNUAL_HELPER_COLUMNS + COMPUSTAT_ANNUAL_FEATURES:
            panel[column] = np.nan
    else:
        left = panel.sort_values(["date", "isin_norm"]).copy()
        right = annual.sort_values(["comp_available_date", "isin_norm"]).copy()
        panel = pd.merge_asof(
            left,
            right,
            left_on="date",
            right_on="comp_available_date",
            by="isin_norm",
            direction="backward",
            allow_exact_matches=True,
        )

    monthly = monthly_features.dropna(subset=["isin_norm", "date"]).copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce").astype("datetime64[ns]")
    if monthly.empty:
        for column in MONTHLY_HELPER_COLUMNS + COMPUSTAT_MARKET_FEATURES:
            panel[column] = np.nan
    else:
        panel = panel.merge(
            monthly,
            on=["isin_norm", "date"],
            how="left",
            validate="many_to_one",
        )

    panel["comp_book_to_market"] = _safe_divide(
        panel["comp_book_equity"], panel["company_market_cap"]
    ).where(panel["comp_book_equity"].gt(0) & panel["company_market_cap"].gt(0))
    panel = panel.sort_values(["_row_order"]).drop(columns=["_row_order"]).reset_index(drop=True)
    return panel


def add_compustat_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    out = _feature_rank_frame(panel, COMPUSTAT_EXTENSION_FEATURES)
    for feature in ALL_RAW_FEATURES:
        if feature not in out:
            out[feature] = np.nan
    out["compustat_annual_feature_count"] = out[
        ["comp_book_to_market", *COMPUSTAT_ANNUAL_FEATURES]
    ].notna().sum(axis=1)
    out["compustat_monthly_feature_count"] = out[COMPUSTAT_MARKET_FEATURES].notna().sum(axis=1)
    out["compustat_feature_count"] = out[COMPUSTAT_EXTENSION_FEATURES].notna().sum(axis=1)
    out["deep_feature_count"] = out[[*ALL_RAW_FEATURES, *COMPUSTAT_EXTENSION_FEATURES]].notna().sum(
        axis=1
    )
    return out


def build_compustat_enriched_panel(
    base_panel: pd.DataFrame,
    annual: pd.DataFrame,
    monthly: pd.DataFrame,
    config: PanelConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    annual_features, annual_audit = prepare_compustat_annual_features(annual, config)
    monthly_features, monthly_audit = prepare_compustat_monthly_features(monthly)
    panel = merge_compustat_features(base_panel, annual_features, monthly_features)
    panel = add_compustat_cross_sectional_ranks(panel)
    eligible = panel["eligible"].fillna(False)
    audit = {
        "config": asdict(config),
        "annual": annual_audit,
        "monthly": monthly_audit,
        "panel": {
            "rows": int(len(panel)),
            "unique_rics": int(panel["ric"].nunique()),
            "unique_isins": int(panel["TR.ISIN"].nunique(dropna=True)),
            "rows_with_compustat_annual": int(panel["comp_gvkey"].notna().sum()),
            "rows_with_compustat_monthly": int(panel["comp_monthly_gvkey"].notna().sum()),
            "eligible_rows_with_compustat_annual": int((eligible & panel["comp_gvkey"].notna()).sum()),
            "eligible_rows_with_compustat_monthly": int(
                (eligible & panel["comp_monthly_gvkey"].notna()).sum()
            ),
            "unique_rics_with_compustat_annual": int(
                panel.loc[panel["comp_gvkey"].notna(), "ric"].nunique()
            ),
            "unique_rics_with_compustat_monthly": int(
                panel.loc[panel["comp_monthly_gvkey"].notna(), "ric"].nunique()
            ),
            "mean_compustat_feature_count": float(panel["compustat_feature_count"].mean()),
            "mean_deep_feature_count": float(panel["deep_feature_count"].mean()),
            "feature_non_null": {
                feature: int(panel[feature].notna().sum())
                for feature in COMPUSTAT_EXTENSION_FEATURES
            },
        },
    }
    return annual_features, monthly_features, panel, audit


def compustat_feature_dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        COMPUSTAT_FEATURE_DICTIONARY,
        columns=["feature", "definition", "information_timing"],
    ).assign(
        model_column=lambda frame: frame["feature"] + "_rank",
        transformation="Monthly cross-sectional percentile rank scaled to [-1, 1]",
        missing_value_rule="Rank set to 0 (cross-sectional median); compustat_feature_count retains disclosure",
    )


def write_compustat_outputs(
    output_dir: Path,
    annual_features: pd.DataFrame,
    monthly_features: pd.DataFrame,
    panel: pd.DataFrame,
    audit: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annual_features.to_parquet(
        output_dir / "compustat_annual_features_lagged.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    monthly_features.to_parquet(
        output_dir / "compustat_monthly_features.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    panel.to_parquet(
        output_dir / "monthly_feature_panel_compustat.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    compustat_feature_dictionary_frame().to_csv(
        output_dir / "compustat_feature_dictionary.csv", index=False
    )
    (output_dir / "compustat_enrichment_audit.json").write_text(
        json.dumps(audit, indent=2)
    )
