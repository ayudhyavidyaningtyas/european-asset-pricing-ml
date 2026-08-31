"""Cleaning and characteristic construction for the European stock panel.

The module builds a compact Gu-Kelly-Xiu-style panel. It is deliberately not a
claim to reproduce the original 94-characteristic US data set.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MONTHLY_COLUMNS = [
    "date",
    "ric",
    "company_market_cap",
    "price_close",
    "total_return_1m",
    "shares_outstanding",
    "volume",
]

FUNDAMENTAL_VALUE_MAP = {
    "TR.F.TOTASSETS": "total_assets",
    "TR.F.TOTLIAB": "total_liabilities",
    "TR.F.TOTSHHOLDEQ": "book_equity",
    "TR.F.TOTREVENUE": "revenue",
    "TR.F.OPPROFBEFNONRECURINCEXPN": "operating_profit",
    "TR.F.INCBEFDISCOPSEXORDITEMS": "net_income",
    "TR.F.NETCASHFLOWOP": "operating_cash_flow",
    "TR.F.CAPEXTOT": "capital_expenditure",
}

RAW_FEATURES = [
    "log_size",
    "book_to_market",
    "return_1m",
    "momentum_6_2",
    "momentum_12_2",
    "volatility_12m",
    "max_return_12m",
    "market_cap_growth_12m",
    "turnover_1m",
    "turnover_12m",
    "asset_growth",
    "sales_growth",
    "profitability_roa",
    "operating_profitability",
    "leverage",
    "accruals",
    "capex_to_assets",
    "cashflow_to_assets",
]

LIQUIDITY_EXTENSION_FEATURES = [
    "log_trading_value_eur",
    "turnover_volatility_12m",
]

ALL_RAW_FEATURES = [*RAW_FEATURES, *LIQUIDITY_EXTENSION_FEATURES]

FEATURE_DICTIONARY = [
    ("log_size", "Natural log of month-end company market capitalisation", "month t"),
    ("book_to_market", "Most recently available positive book equity divided by market cap", "FY end + lag"),
    ("return_1m", "Current one-month total return in decimal form", "month t"),
    ("momentum_6_2", "Compounded returns over months t-5 to t-1", "excludes month t"),
    ("momentum_12_2", "Compounded returns over months t-11 to t-1", "excludes month t"),
    ("volatility_12m", "Standard deviation of monthly returns over t-11 to t", "month t"),
    ("max_return_12m", "Maximum monthly return over t-11 to t", "month t"),
    ("market_cap_growth_12m", "Twelve-month market-cap growth", "month t"),
    ("turnover_1m", "Monthly volume divided by latest shares outstanding", "month t"),
    ("turnover_12m", "Twelve-month mean turnover", "month t"),
    ("asset_growth", "Annual total-asset growth", "FY end + lag"),
    ("sales_growth", "Annual revenue growth", "FY end + lag"),
    ("profitability_roa", "Net income divided by total assets", "FY end + lag"),
    ("operating_profitability", "Operating profit divided by total assets", "FY end + lag"),
    ("leverage", "Total liabilities divided by total assets", "FY end + lag"),
    ("accruals", "Net income less operating cash flow, divided by total assets", "FY end + lag"),
    ("capex_to_assets", "Capital expenditure divided by total assets", "FY end + lag"),
    ("cashflow_to_assets", "Operating cash flow divided by total assets", "FY end + lag"),
    (
        "log_trading_value_eur",
        "Natural log of EUR market capitalisation multiplied by trailing 12-month mean share turnover",
        "month t",
    ),
    (
        "turnover_volatility_12m",
        "Standard deviation of monthly share turnover over t-11 to t",
        "month t",
    ),
]


@dataclass(frozen=True)
class PanelConfig:
    as_of: str = "2026-07-08"
    accounting_lag_months: int = 6
    min_return_observations: int = 24
    min_market_cap_observations: int = 12
    min_features: int = 8
    microcap_quantile: float = 0.05
    maximum_monthly_return: float = 10.0


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def clean_monthly_returns(
    returns_percent: pd.Series,
    maximum_monthly_return: float,
) -> pd.Series:
    """Convert percentage returns to decimals and remove impossible/data-error tails."""
    returns = pd.to_numeric(returns_percent, errors="coerce") / 100.0
    return returns.where(returns.gt(-1.0) & returns.le(maximum_monthly_return))


def load_source_data(export_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = pd.read_csv(export_dir / "europe_equity_universe.csv", low_memory=False)
    monthly = pd.read_csv(
        export_dir / "refinitiv_monthly_panel_tidy.csv",
        usecols=MONTHLY_COLUMNS,
        parse_dates=["date"],
        low_memory=False,
    )
    fundamentals = pd.read_csv(
        export_dir / "refinitiv_fundamentals_annual.csv",
        low_memory=False,
    )
    return universe, monthly, fundamentals


def load_optional_reference_supplement(export_dir: Path) -> pd.DataFrame | None:
    path = export_dir / "supplemental" / "refinitiv_security_master_supplement.parquet"
    if not path.exists():
        return None
    reference = pd.read_parquet(path)
    if reference.empty or "Instrument" not in reference:
        return None
    return reference


def fundamental_value_coverage(fundamentals: pd.DataFrame) -> pd.DataFrame:
    value_columns = [column for column in FUNDAMENTAL_VALUE_MAP if column in fundamentals]
    if not value_columns:
        return pd.DataFrame(columns=["ric", "fundamental_rows", "fundamental_value_rows"])
    work = fundamentals[["Instrument", *value_columns]].copy()
    work["has_value"] = work[value_columns].notna().any(axis=1)
    return (
        work.groupby("Instrument", as_index=False)
        .agg(
            fundamental_rows=("Instrument", "size"),
            fundamental_value_rows=("has_value", "sum"),
        )
        .rename(columns={"Instrument": "ric"})
    )


def monthly_coverage(monthly: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    work = monthly[["ric", "date", "total_return_1m", "company_market_cap"]].copy()
    work = work[work["date"].le(pd.Timestamp(config.as_of))]
    work["valid_return"] = clean_monthly_returns(
        work["total_return_1m"], config.maximum_monthly_return
    ).notna()
    work["valid_market_cap"] = pd.to_numeric(
        work["company_market_cap"], errors="coerce"
    ).gt(0)
    return (
        work.groupby("ric", as_index=False)
        .agg(
            monthly_rows=("date", "size"),
            return_observations=("valid_return", "sum"),
            market_cap_observations=("valid_market_cap", "sum"),
            first_month=("date", "min"),
            last_month=("date", "max"),
        )
    )


def _normalise_reference_supplement(reference: pd.DataFrame | None) -> pd.DataFrame:
    if reference is None:
        return pd.DataFrame(columns=["ric"])
    columns = {
        "Instrument": "ric",
        "TR.INSTRUMENTTYPE": "instrument_type",
        "TR.ASSETCATEGORY": "asset_category",
        "TR.ASSETTYPE": "asset_type",
        "TR.ISPRIMARYQUOTE": "is_primary_quote",
        "TR.PRIMARYQUOTE": "primary_quote",
        "TR.ISDELISTEDQUOTE": "is_delisted_quote",
        "TR.FIRSTTRADEDATE": "first_trade_date",
        "TR.RETIREDATE": "retire_date",
    }
    available = [column for column in columns if column in reference]
    out = reference[available].rename(columns=columns).copy()
    return out.drop_duplicates("ric", keep="last")


def _truthy(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_clean_universe(
    universe: pd.DataFrame,
    monthly: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: PanelConfig,
    reference_supplement: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = universe.copy()
    if "ric" not in base:
        base["ric"] = base["TR.RIC"]
    base = base.drop_duplicates("ric", keep="last")

    coverage = monthly_coverage(monthly, config)
    fundamental_coverage = fundamental_value_coverage(fundamentals)
    supplement = _normalise_reference_supplement(reference_supplement)
    base = base.merge(coverage, on="ric", how="left")
    base = base.merge(fundamental_coverage, on="ric", how="left")
    base = base.merge(supplement, on="ric", how="left")

    count_columns = [
        "monthly_rows",
        "return_observations",
        "market_cap_observations",
        "fundamental_rows",
        "fundamental_value_rows",
    ]
    for column in count_columns:
        base[column] = base[column].fillna(0).astype(int)

    instrument_type = base.get(
        "instrument_type", pd.Series(index=base.index, dtype="object")
    )
    has_type = instrument_type.notna()
    ordinary_type = instrument_type.astype(str).str.contains(
        r"ordinary|common", case=False, regex=True, na=False
    )
    base["ordinary_equity_metadata"] = (~has_type) | ordinary_type
    base["metadata_provisional"] = ~has_type

    primary_value = base.get(
        "is_primary_quote", pd.Series(index=base.index, dtype="object")
    )
    active = base["screen_state"].astype(str).str.lower().eq("active")
    primary_ok = primary_value.map(_truthy)
    # Inactive RICs were selected from the historical primary-equity screener.
    base["primary_quote_metadata"] = (~active) | primary_ok

    checks = {
        "missing_isin": base["TR.ISIN"].isna(),
        "missing_sector": base["TR.TRBCECONOMICSECTOR"].isna(),
        "insufficient_returns": base["return_observations"].lt(
            config.min_return_observations
        ),
        "insufficient_market_cap": base["market_cap_observations"].lt(
            config.min_market_cap_observations
        ),
        "no_fundamental_values": base["fundamental_value_rows"].eq(0),
        "non_common_equity": ~base["ordinary_equity_metadata"],
        "non_primary_quote": ~base["primary_quote_metadata"],
    }
    for name, values in checks.items():
        base[name] = values

    failed = pd.DataFrame(checks)
    base["eligible"] = ~failed.any(axis=1)
    base["exclusion_reason"] = failed.apply(
        lambda row: ";".join(row.index[row.to_numpy(dtype=bool)]), axis=1
    )
    return base.sort_values(["eligible", "screen_country", "ric"], ascending=[False, True, True])


def canonical_period_end(fundamentals: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    date_columns = [
        column
        for column in fundamentals
        if column.startswith("TR.F.") and column.endswith(".DATE")
    ]
    if not date_columns:
        raise ValueError("Fundamental panel has no TR.F.*.DATE columns")
    parsed = pd.DataFrame(
        {column: pd.to_datetime(fundamentals[column], errors="coerce") for column in date_columns},
        index=fundamentals.index,
    )
    period_end = parsed.bfill(axis=1).iloc[:, 0]
    disagreements = parsed.nunique(axis=1).gt(1)
    if disagreements.any():
        modes = parsed.loc[disagreements].mode(axis=1, dropna=True)
        period_end.loc[disagreements] = modes.iloc[:, 0]
    return period_end, disagreements


def prepare_fundamental_features(
    fundamentals: pd.DataFrame,
    config: PanelConfig,
) -> tuple[pd.DataFrame, dict]:
    available_values = {
        source: target
        for source, target in FUNDAMENTAL_VALUE_MAP.items()
        if source in fundamentals
    }
    work = fundamentals[["Instrument", *available_values]].rename(
        columns={"Instrument": "ric", **available_values}
    )
    work = work.copy()
    _to_numeric(work, list(available_values.values()))
    work["period_end"], disagreements = canonical_period_end(fundamentals)
    work["source_completeness"] = work[list(available_values.values())].notna().sum(axis=1)
    work = work.dropna(subset=["ric", "period_end"])
    work = (
        work.sort_values(["ric", "period_end", "source_completeness"])
        .groupby(["ric", "period_end"], as_index=False)
        .last()
        .sort_values(["ric", "period_end"])
    )

    grouped = work.groupby("ric", sort=False)
    prior_period = grouped["period_end"].shift(1)
    fiscal_gap_months = (
        (work["period_end"].dt.year - prior_period.dt.year) * 12
        + work["period_end"].dt.month
        - prior_period.dt.month
    )
    annual_gap = fiscal_gap_months.between(9, 18)

    for column in ["total_assets", "revenue", "book_equity"]:
        previous = grouped[column].shift(1)
        growth = work[column].div(previous).sub(1)
        work[f"{column}_growth"] = growth.where(
            annual_gap & work[column].gt(0) & previous.gt(0)
        )

    positive_assets = work["total_assets"].where(work["total_assets"].gt(0))
    work["profitability_roa"] = work["net_income"].div(positive_assets)
    work["operating_profitability"] = work["operating_profit"].div(positive_assets)
    work["leverage"] = work["total_liabilities"].div(positive_assets)
    work["accruals"] = work["net_income"].sub(work["operating_cash_flow"]).div(
        positive_assets
    )
    work["capex_to_assets"] = work["capital_expenditure"].div(positive_assets)
    work["cashflow_to_assets"] = work["operating_cash_flow"].div(positive_assets)
    work["asset_growth"] = work["total_assets_growth"]
    work["sales_growth"] = work["revenue_growth"]
    work["available_date"] = (
        work["period_end"] + pd.offsets.MonthEnd(config.accounting_lag_months)
    )

    keep = [
        "ric",
        "period_end",
        "available_date",
        "book_equity",
        "asset_growth",
        "sales_growth",
        "profitability_roa",
        "operating_profitability",
        "leverage",
        "accruals",
        "capex_to_assets",
        "cashflow_to_assets",
    ]
    audit = {
        "source_rows": int(len(fundamentals)),
        "rows_with_disagreeing_field_dates": int(disagreements.sum()),
        "collapsed_rows": int(len(work)),
        "duplicate_ric_period_rows_removed": int(
            len(fundamentals.dropna(subset=["Instrument"])) - len(work)
        ),
        "accounting_lag_months": config.accounting_lag_months,
    }
    return work[keep], audit


def _rolling_compound(
    values: pd.Series,
    groups: pd.Series,
    window: int,
    min_periods: int,
    shift: int = 0,
) -> pd.Series:
    log_values = np.log1p(values)
    if shift:
        log_values = log_values.groupby(groups, sort=False).shift(shift)
    rolled = (
        log_values.groupby(groups, sort=False)
        .rolling(window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
    )
    return np.expm1(rolled).sort_index()


def compute_market_features(monthly: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    panel = monthly[MONTHLY_COLUMNS].copy()
    panel = panel[panel["date"].le(pd.Timestamp(config.as_of))]
    panel = panel.drop_duplicates(["ric", "date"], keep="last")
    panel = panel.sort_values(["ric", "date"]).reset_index(drop=True)
    numeric = [
        "company_market_cap",
        "price_close",
        "shares_outstanding",
        "volume",
    ]
    _to_numeric(panel, numeric)
    panel["return_1m"] = clean_monthly_returns(
        panel["total_return_1m"], config.maximum_monthly_return
    )
    panel["month_index"] = panel["date"].dt.year * 12 + panel["date"].dt.month
    groups = panel["ric"]
    grouped = panel.groupby("ric", sort=False)

    panel["log_size"] = np.log(
        panel["company_market_cap"].where(panel["company_market_cap"].gt(0))
    )
    panel["momentum_6_2"] = _rolling_compound(
        panel["return_1m"], groups, window=5, min_periods=3, shift=1
    )
    panel["momentum_12_2"] = _rolling_compound(
        panel["return_1m"], groups, window=11, min_periods=8, shift=1
    )
    panel["volatility_12m"] = (
        grouped["return_1m"]
        .rolling(12, min_periods=8)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["max_return_12m"] = (
        grouped["return_1m"]
        .rolling(12, min_periods=8)
        .max()
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    panel.loc[
        panel["month_index"].sub(grouped["month_index"].shift(5)).ne(5),
        "momentum_6_2",
    ] = np.nan
    twelve_month_gap = panel["month_index"].sub(grouped["month_index"].shift(11)).eq(11)
    panel.loc[~twelve_month_gap, ["momentum_12_2", "volatility_12m", "max_return_12m"]] = np.nan

    prior_market_cap = grouped["company_market_cap"].shift(12)
    exact_year = panel["month_index"].sub(grouped["month_index"].shift(12)).eq(12)
    panel["market_cap_growth_12m"] = (
        panel["company_market_cap"].div(prior_market_cap).sub(1)
    ).where(exact_year & prior_market_cap.gt(0) & panel["company_market_cap"].gt(0))

    shares = grouped["shares_outstanding"].ffill(limit=12)
    panel["turnover_1m"] = panel["volume"].div(shares).where(
        panel["volume"].ge(0) & shares.gt(0)
    )
    panel["turnover_12m"] = (
        panel["turnover_1m"]
        .groupby(groups, sort=False)
        .rolling(12, min_periods=6)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel["turnover_volatility_12m"] = (
        panel["turnover_1m"]
        .groupby(groups, sort=False)
        .rolling(12, min_periods=6)
        .std()
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    panel.loc[
        ~twelve_month_gap,
        ["turnover_12m", "turnover_volatility_12m"],
    ] = np.nan
    trading_value_eur = panel["company_market_cap"] * panel["turnover_12m"]
    panel["log_trading_value_eur"] = np.log(
        trading_value_eur.where(trading_value_eur.gt(0))
    )

    next_month = grouped["month_index"].shift(-1)
    panel["target_date"] = grouped["date"].shift(-1)
    panel["target_return_1m"] = grouped["return_1m"].shift(-1).where(
        next_month.sub(panel["month_index"]).eq(1)
    )
    panel["return_history_n"] = (
        panel["return_1m"].notna().groupby(groups, sort=False).cumsum()
    )
    return panel


def merge_fundamental_features(
    market_panel: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    left = market_panel.sort_values(["date", "ric"]).copy()
    right = fundamentals.sort_values(["available_date", "ric"]).copy()
    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="available_date",
        by="ric",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["book_to_market"] = merged["book_equity"].div(
        merged["company_market_cap"]
    ).where(
        merged["book_equity"].gt(0) & merged["company_market_cap"].gt(0)
    )
    return merged.sort_values(["ric", "date"]).reset_index(drop=True)


def add_cross_sectional_ranks(
    panel: pd.DataFrame,
    config: PanelConfig,
) -> pd.DataFrame:
    out = panel.copy()
    for feature in ALL_RAW_FEATURES:
        if feature not in out:
            out[feature] = np.nan
    static_eligible = out["eligible"].fillna(False)
    positive_market_cap = out["company_market_cap"].gt(0) & static_eligible
    out["market_cap_percentile"] = np.nan
    out.loc[positive_market_cap, "market_cap_percentile"] = (
        out.loc[positive_market_cap]
        .groupby("date")["company_market_cap"]
        .rank(method="average", pct=True)
    )

    out["feature_count"] = out[RAW_FEATURES].notna().sum(axis=1)
    out["expanded_feature_count"] = out[ALL_RAW_FEATURES].notna().sum(axis=1)
    for feature in ALL_RAW_FEATURES:
        rank_column = f"{feature}_rank"
        valid = static_eligible & out[feature].notna()
        out[rank_column] = np.nan
        ranks = out.loc[valid].groupby("date")[feature].rank(
            method="average", pct=True
        )
        out.loc[valid, rank_column] = ranks.mul(2).sub(1)
        # Zero is the contemporaneous cross-sectional median after rank scaling.
        out.loc[static_eligible & out[rank_column].isna(), rank_column] = 0.0

    target_valid = static_eligible & out["target_return_1m"].notna()
    out["target_return_rank"] = np.nan
    target_ranks = out.loc[target_valid].groupby("date")["target_return_1m"].rank(
        method="average", pct=True
    )
    out.loc[target_valid, "target_return_rank"] = target_ranks.mul(2).sub(1)

    out["model_eligible"] = (
        static_eligible
        & out["target_return_1m"].notna()
        & out["return_history_n"].ge(config.min_return_observations)
        & out["market_cap_percentile"].ge(config.microcap_quantile)
        & out["feature_count"].ge(config.min_features)
    )
    return out


def build_feature_panel(
    universe: pd.DataFrame,
    monthly: pd.DataFrame,
    fundamentals: pd.DataFrame,
    config: PanelConfig,
    reference_supplement: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    clean_universe = build_clean_universe(
        universe,
        monthly,
        fundamentals,
        config,
        reference_supplement,
    )
    eligible_rics = set(clean_universe.loc[clean_universe["eligible"], "ric"])
    market = compute_market_features(
        monthly[monthly["ric"].isin(eligible_rics)],
        config,
    )
    fundamental_features, fundamental_audit = prepare_fundamental_features(
        fundamentals[fundamentals["Instrument"].isin(eligible_rics)],
        config,
    )
    panel = merge_fundamental_features(market, fundamental_features)
    metadata_columns = [
        "ric",
        "TR.ISIN",
        "TR.COMMONNAME",
        "TR.EXCHANGECOUNTRY",
        "TR.TRBCECONOMICSECTOR",
        "TR.TRBCBUSINESSSECTOR",
        "TR.TRBCINDUSTRYGROUP",
        "TR.TRBCINDUSTRY",
        "screen_country",
        "screen_state",
        "eligible",
        "metadata_provisional",
    ]
    metadata_columns = [column for column in metadata_columns if column in clean_universe]
    panel = panel.merge(
        clean_universe[metadata_columns],
        on="ric",
        how="left",
        validate="many_to_one",
    )
    panel = add_cross_sectional_ranks(panel, config)

    audit = {
        "config": asdict(config),
        "universe": {
            "source_rows": int(len(universe)),
            "unique_rics": int(universe["ric"].nunique()),
            "non_null_unique_isins": int(universe["TR.ISIN"].nunique(dropna=True)),
            "duplicate_non_null_isins": int(
                universe.dropna(subset=["TR.ISIN"])
                .groupby("TR.ISIN")["ric"]
                .nunique()
                .gt(1)
                .sum()
            ),
            "eligible_rics": int(clean_universe["eligible"].sum()),
            "inactive_eligible_rics": int(
                (
                    clean_universe["eligible"]
                    & clean_universe["screen_state"].astype(str).str.lower().eq("inactive")
                ).sum()
            ),
            "provisional_metadata_rics": int(
                (
                    clean_universe["eligible"]
                    & clean_universe["metadata_provisional"]
                ).sum()
            ),
            "exclusion_counts": {
                column: int(clean_universe[column].sum())
                for column in [
                    "missing_isin",
                    "missing_sector",
                    "insufficient_returns",
                    "insufficient_market_cap",
                    "no_fundamental_values",
                    "non_common_equity",
                    "non_primary_quote",
                ]
            },
        },
        "fundamentals": fundamental_audit,
        "panel": {
            "rows": int(len(panel)),
            "model_eligible_rows": int(panel["model_eligible"].sum()),
            "unique_model_securities": int(
                panel.loc[panel["model_eligible"], "ric"].nunique()
            ),
            "first_model_month": (
                str(panel.loc[panel["model_eligible"], "date"].min().date())
                if panel["model_eligible"].any()
                else None
            ),
            "last_model_month": (
                str(panel.loc[panel["model_eligible"], "date"].max().date())
                if panel["model_eligible"].any()
                else None
            ),
            "invalid_extreme_returns_removed": int(
                clean_monthly_returns(
                    monthly["total_return_1m"], config.maximum_monthly_return
                ).isna().sum()
                - monthly["total_return_1m"].isna().sum()
            ),
        },
    }
    return clean_universe, fundamental_features, panel, audit


def feature_dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        FEATURE_DICTIONARY,
        columns=["feature", "definition", "information_timing"],
    ).assign(
        model_column=lambda frame: frame["feature"] + "_rank",
        transformation="Monthly cross-sectional percentile rank scaled to [-1, 1]",
        missing_value_rule="Rank set to 0 (cross-sectional median); feature_count retains disclosure",
    )


def write_panel_outputs(
    output_dir: Path,
    clean_universe: pd.DataFrame,
    fundamental_features: pd.DataFrame,
    panel: pd.DataFrame,
    audit: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_universe.to_csv(output_dir / "clean_universe.csv", index=False)
    fundamental_features.to_parquet(
        output_dir / "fundamental_features_lagged.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    panel.to_parquet(
        output_dir / "monthly_feature_panel.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    feature_dictionary_frame().to_csv(
        output_dir / "feature_dictionary.csv", index=False
    )
    (output_dir / "cleaning_audit.json").write_text(json.dumps(audit, indent=2))
