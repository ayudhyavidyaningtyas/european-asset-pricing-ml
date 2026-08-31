"""Refinitiv analyst-estimates feature enrichment for European equities.

The input is intentionally schema-flexible. Refinitiv/LSEG estimates fields can
arrive from Workspace, Excel, or the Python Data Library with different display
names. This module maps common aliases to a canonical monthly snapshot format,
then creates lag-safe analyst features and monthly cross-sectional ranks.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from asset_pricing import ALL_RAW_FEATURES
from compustat_features import COMPUSTAT_EXTENSION_FEATURES


ESTIMATES_EXTENSION_FEATURES = [
    "est_eps_yield",
    "est_eps_revision_1m",
    "est_eps_revision_3m",
    "est_eps_dispersion",
    "est_eps_coverage",
    "est_revenue_revision_1m",
    "est_revenue_revision_3m",
    "est_revenue_dispersion",
    "est_revenue_coverage",
    "est_price_target_upside",
    "est_price_target_revision_1m",
    "est_price_target_revision_3m",
    "est_price_target_dispersion",
    "est_price_target_coverage",
    "est_recommendation_strength",
    "est_recommendation_change_3m",
    "est_coverage_composite",
]

ESTIMATES_DEGENERATE_MODEL_FEATURES = [
    "est_eps_coverage",
    "est_revenue_coverage",
    "est_price_target_coverage",
    "est_recommendation_strength",
    "est_recommendation_change_3m",
    "est_coverage_composite",
]

ESTIMATES_MODEL_FEATURES = [
    feature
    for feature in ESTIMATES_EXTENSION_FEATURES
    if feature not in ESTIMATES_DEGENERATE_MODEL_FEATURES
]

ESTIMATES_MISSINGNESS_FEATURES = ["estimates_feature_count"]

# Decomposition groups for the estimates ablation. Only the informative
# features appear here: the coverage-count and recommendation fields are
# unavailable from the retrieved Refinitiv fields (all-null, ranks forced to
# zero), so they are excluded from every ablation subset.
ESTIMATES_FEATURE_FAMILIES = {
    "eps": [
        "est_eps_yield",
        "est_eps_revision_1m",
        "est_eps_revision_3m",
        "est_eps_dispersion",
    ],
    "revenue": [
        "est_revenue_revision_1m",
        "est_revenue_revision_3m",
        "est_revenue_dispersion",
    ],
    "price_target": [
        "est_price_target_upside",
        "est_price_target_revision_1m",
        "est_price_target_revision_3m",
        "est_price_target_dispersion",
    ],
}

ESTIMATES_INFORMATION_TYPES = {
    "levels": [
        "est_eps_yield",
        "est_price_target_upside",
    ],
    "revisions": [
        "est_eps_revision_1m",
        "est_eps_revision_3m",
        "est_revenue_revision_1m",
        "est_revenue_revision_3m",
        "est_price_target_revision_1m",
        "est_price_target_revision_3m",
    ],
    "dispersion": [
        "est_eps_dispersion",
        "est_revenue_dispersion",
        "est_price_target_dispersion",
    ],
}


def _assert_partitions_model_features(name: str, groups: dict[str, list[str]]) -> None:
    """Fail fast if a decomposition stops covering the modelled analyst layer.

    The family and information-type ablations are only interpretable as a
    decomposition if each grouping is an exact partition: "X_only" and "ex_X"
    would otherwise silently leave features unattributed.
    """
    assigned = [feature for features in groups.values() for feature in features]
    duplicates = sorted({f for f in assigned if assigned.count(f) > 1})
    missing = sorted(set(ESTIMATES_MODEL_FEATURES) - set(assigned))
    unexpected = sorted(set(assigned) - set(ESTIMATES_MODEL_FEATURES))
    if duplicates or missing or unexpected:
        raise ValueError(
            f"{name} must partition ESTIMATES_MODEL_FEATURES "
            f"(duplicates={duplicates}, missing={missing}, unexpected={unexpected})"
        )


_assert_partitions_model_features(
    "ESTIMATES_FEATURE_FAMILIES", ESTIMATES_FEATURE_FAMILIES
)
_assert_partitions_model_features(
    "ESTIMATES_INFORMATION_TYPES", ESTIMATES_INFORMATION_TYPES
)


@dataclass(frozen=True)
class EstimatesPanelConfig:
    strict_identifier_match: bool = False
    signal_lag_months: int = 0
    filter_extreme_estimates: bool = False
    max_abs_eps_yield: float = 1.0
    min_price_target_upside: float = -0.95
    max_price_target_upside: float = 5.0
    max_abs_revision: float = 5.0
    max_dispersion: float = 5.0


ESTIMATES_HELPER_COLUMNS = [
    "est_ric",
    "est_isin",
    "est_snapshot_date",
    "est_eps_mean",
    "est_revenue_mean",
    "est_price_target_mean",
    "est_recommendation_mean",
    "est_eps_count",
    "est_revenue_count",
    "est_price_target_count",
    "est_recommendation_count",
]

CANONICAL_ESTIMATES_VALUE_COLUMNS = [
    "eps_mean",
    "eps_high",
    "eps_low",
    "eps_std",
    "eps_count",
    "revenue_mean",
    "revenue_high",
    "revenue_low",
    "revenue_std",
    "revenue_count",
    "price_target_mean",
    "price_target_high",
    "price_target_low",
    "price_target_std",
    "price_target_count",
    "recommendation_mean",
    "recommendation_count",
]

ESTIMATES_RAW_COLUMNS = [
    f"est_{column}" for column in CANONICAL_ESTIMATES_VALUE_COLUMNS
]

ESTIMATES_FIELD_ALIASES = {
    "ric": [
        "ric",
        "instrument",
        "tr.ric",
        "tr_ric",
        "instrument_name",
        "identifier",
    ],
    "isin": ["isin", "tr.isin", "tr_isin"],
    "snapshot_date": [
        "date",
        "snapshot_date",
        "signal_date",
        "period_end",
        "calcdate",
        "calc_date",
        "asofdate",
        "as_of_date",
        "datadate",
    ],
    "eps_mean": [
        "eps_mean",
        "epsmean",
        "tr.epsmean",
        "tr_epsmean",
        "tr.mean_eps",
        "eps_consensus_mean",
        "earnings_per_share_mean",
    ],
    "eps_high": [
        "eps_high",
        "epshigh",
        "tr.epshigh",
        "tr_epshigh",
        "eps_consensus_high",
    ],
    "eps_low": [
        "eps_low",
        "epslow",
        "tr.epslow",
        "tr_epslow",
        "eps_consensus_low",
    ],
    "eps_std": [
        "eps_std",
        "epsstdev",
        "epsstddev",
        "eps_stddev",
        "tr.epsstdev",
        "tr.epsstddev",
        "tr_epsstdev",
        "tr_epsstddev",
    ],
    "eps_count": [
        "eps_count",
        "eps_num_estimates",
        "epsnumestimates",
        "tr.epsnumestimates",
        "tr_epsnumestimates",
        "num_eps_estimates",
    ],
    "revenue_mean": [
        "revenue_mean",
        "sales_mean",
        "rev_mean",
        "tr.revenuemean",
        "tr_salesmean",
        "tr.revenueestimate",
        "revenue_consensus_mean",
    ],
    "revenue_high": [
        "revenue_high",
        "sales_high",
        "tr.revenuehigh",
        "tr_saleshigh",
        "revenue_consensus_high",
    ],
    "revenue_low": [
        "revenue_low",
        "sales_low",
        "tr.revenuelow",
        "tr_saleslow",
        "revenue_consensus_low",
    ],
    "revenue_std": [
        "revenue_std",
        "sales_std",
        "tr.revenuestdev",
        "tr.revenuestddev",
        "tr_salesstdev",
        "tr_revenuestddev",
    ],
    "revenue_count": [
        "revenue_count",
        "sales_count",
        "revenue_num_estimates",
        "sales_num_estimates",
        "tr.revenuenumestimates",
        "tr_salesnumestimates",
    ],
    "price_target_mean": [
        "price_target_mean",
        "target_price_mean",
        "pt_mean",
        "tr.pricetargetmean",
        "tr_pricetargetmean",
        "tr.targetpricemean",
    ],
    "price_target_high": [
        "price_target_high",
        "target_price_high",
        "pt_high",
        "tr.pricetargethigh",
        "tr_pricetargethigh",
    ],
    "price_target_low": [
        "price_target_low",
        "target_price_low",
        "pt_low",
        "tr.pricetargetlow",
        "tr_pricetargetlow",
    ],
    "price_target_std": [
        "price_target_std",
        "target_price_std",
        "pt_std",
        "tr.pricetargetstdev",
        "tr.pricetargetstddev",
        "tr_pricetargetstddev",
    ],
    "price_target_count": [
        "price_target_count",
        "target_price_count",
        "pt_count",
        "tr.pricetargetnumestimates",
        "tr_pricetargetnumestimates",
    ],
    "recommendation_mean": [
        "recommendation_mean",
        "recommendation_mean_score",
        "recommendation",
        "tr.recommendationmean",
        "tr_recommendationmean",
        "tr.consensusrecommendation",
    ],
    "recommendation_count": [
        "recommendation_count",
        "recommendation_num_analysts",
        "recommendation_num_estimates",
        "tr.recommendationnumanalysts",
        "tr.recommendationnumestimates",
    ],
}

ESTIMATES_FEATURE_DICTIONARY = [
    ("est_eps_yield", "FY1/forward EPS consensus mean divided by month-end price", "month t snapshot"),
    ("est_eps_revision_1m", "One-month percentage change in EPS consensus mean", "uses t and t-1 snapshots"),
    ("est_eps_revision_3m", "Three-month percentage change in EPS consensus mean", "uses t and t-3 snapshots"),
    ("est_eps_dispersion", "EPS estimate dispersion: standard deviation or high-low range scaled by absolute mean", "month t snapshot"),
    ("est_eps_coverage", "Log one plus number of EPS estimates", "month t snapshot"),
    ("est_revenue_revision_1m", "One-month percentage change in revenue consensus mean", "uses t and t-1 snapshots"),
    ("est_revenue_revision_3m", "Three-month percentage change in revenue consensus mean", "uses t and t-3 snapshots"),
    ("est_revenue_dispersion", "Revenue estimate dispersion scaled by absolute mean", "month t snapshot"),
    ("est_revenue_coverage", "Log one plus number of revenue estimates", "month t snapshot"),
    ("est_price_target_upside", "Consensus price target divided by month-end price minus one", "month t snapshot"),
    ("est_price_target_revision_1m", "One-month percentage change in consensus price target", "uses t and t-1 snapshots"),
    ("est_price_target_revision_3m", "Three-month percentage change in consensus price target", "uses t and t-3 snapshots"),
    ("est_price_target_dispersion", "Price-target dispersion scaled by absolute mean", "month t snapshot"),
    ("est_price_target_coverage", "Log one plus number of price-target estimates", "month t snapshot"),
    ("est_recommendation_strength", "Negative consensus recommendation score, assuming lower Refinitiv scores are more bullish", "month t snapshot"),
    ("est_recommendation_change_3m", "Three-month recommendation upgrade; positive means the consensus score declined", "uses t and t-3 snapshots"),
    ("est_coverage_composite", "Log one plus total analyst-count coverage across EPS, revenue, price target and recommendation", "month t snapshot"),
]


def _normalise_name(name: object) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _normalise_identifier(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})
    )


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
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
    valid = den.notna() & den.ne(0)
    if positive_denominator:
        valid &= den.gt(0)
    out = num.div(den)
    return out.where(valid & np.isfinite(out))


def _month_end(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp("M")


def _first_existing_column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {_normalise_name(column): column for column in frame.columns}
    for alias in aliases:
        column = lookup.get(_normalise_name(alias))
        if column is not None:
            return column
    return None


def _canonical_estimates_frame(raw: pd.DataFrame) -> pd.DataFrame:
    work = raw.copy()
    columns = {}
    for canonical, aliases in ESTIMATES_FIELD_ALIASES.items():
        source = _first_existing_column(work, aliases)
        if source is not None:
            columns[canonical] = work[source]
        else:
            columns[canonical] = pd.NA
    out = pd.DataFrame(columns)
    if out["ric"].isna().all() and out["isin"].isna().all():
        raise ValueError("Estimates export needs at least one RIC or ISIN column")
    if out["snapshot_date"].isna().all():
        raise ValueError("Estimates export needs a date/snapshot_date column")
    return out


def load_estimates_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing estimates export: {path}")
    if path.suffix == ".gz" or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip", low_memory=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def prepare_estimates_snapshot_features(estimates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work = _canonical_estimates_frame(estimates)
    numeric_columns = CANONICAL_ESTIMATES_VALUE_COLUMNS
    _to_numeric(work, numeric_columns)
    work["ric_norm"] = _normalise_identifier(work["ric"])
    work["isin_norm"] = _normalise_identifier(work["isin"])
    work["date"] = _month_end(work["snapshot_date"])
    work["snapshot_date"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
    work["source_completeness"] = work[numeric_columns].notna().sum(axis=1)
    work = work.dropna(subset=["date"])
    work = work[work["ric_norm"].notna() | work["isin_norm"].notna()].copy()

    if work.empty:
        keep = ["ric_norm", "isin_norm", "date", "snapshot_date", *numeric_columns]
        return pd.DataFrame(columns=keep), {
            "source_rows": int(len(estimates)),
            "collapsed_rows": 0,
            "unique_rics": 0,
            "unique_isins": 0,
            "feature_non_null": {column: 0 for column in numeric_columns},
        }

    key = np.where(work["ric_norm"].notna(), work["ric_norm"], work["isin_norm"])
    work["_collapse_key"] = key
    work = (
        work.sort_values(["_collapse_key", "date", "snapshot_date", "source_completeness"])
        .groupby(["_collapse_key", "date"], as_index=False)
        .last()
        .sort_values(["_collapse_key", "date"])
        .reset_index(drop=True)
    )
    keep = ["ric_norm", "isin_norm", "date", "snapshot_date", *numeric_columns]
    out = work[keep].copy()
    audit = {
        "source_rows": int(len(estimates)),
        "collapsed_rows": int(len(out)),
        "unique_rics": int(out["ric_norm"].nunique(dropna=True)),
        "unique_isins": int(out["isin_norm"].nunique(dropna=True)),
        "first_month": str(out["date"].min().date()) if not out.empty else None,
        "last_month": str(out["date"].max().date()) if not out.empty else None,
        "feature_non_null": {
            column: int(out[column].notna().sum())
            for column in numeric_columns
        },
    }
    return out, audit


def _lagged_change(
    values: pd.Series,
    groups: pd.Series,
    month_index: pd.Series,
    lag: int,
) -> pd.Series:
    prior = values.groupby(groups, sort=False).shift(lag)
    prior_month = month_index.groupby(groups, sort=False).shift(lag)
    exact_gap = month_index.sub(prior_month).eq(lag)
    scaled = values.sub(prior).div(prior.abs())
    return scaled.where(exact_gap & values.notna() & prior.abs().gt(1e-8))


def _dispersion(
    mean: pd.Series,
    high: pd.Series,
    low: pd.Series,
    standard_deviation: pd.Series,
) -> pd.Series:
    by_std = _safe_divide(standard_deviation, mean.abs(), positive_denominator=True)
    by_range = _safe_divide(high.sub(low), mean.abs(), positive_denominator=True)
    return by_std.combine_first(by_range)


def merge_estimates_features(
    base_panel: pd.DataFrame,
    snapshot_features: pd.DataFrame,
    *,
    strict_identifier_match: bool = False,
) -> pd.DataFrame:
    panel = base_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").astype("datetime64[ns]")
    panel["_row_order"] = np.arange(len(panel))
    panel["ric_norm"] = _normalise_identifier(panel["ric"])
    if "TR.ISIN" in panel:
        panel["isin_norm"] = _normalise_identifier(panel["TR.ISIN"])
    elif "isin" in panel:
        panel["isin_norm"] = _normalise_identifier(panel["isin"])
    else:
        panel["isin_norm"] = pd.NA

    monthly = snapshot_features.dropna(subset=["date"]).copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce").astype("datetime64[ns]")
    if monthly.empty:
        for column in ESTIMATES_HELPER_COLUMNS:
            panel[column] = np.nan
    else:
        by_ric = monthly[monthly["ric_norm"].notna()].copy()
        if not by_ric.empty:
            panel = panel.merge(
                by_ric.add_prefix("est_"),
                left_on=["ric_norm", "date"],
                right_on=["est_ric_norm", "est_date"],
                how="left",
                validate="many_to_one",
            )
        else:
            for column in monthly.add_prefix("est_").columns:
                panel[column] = np.nan

        missing_ric_match = panel["est_snapshot_date"].isna()
        by_isin = monthly[monthly["isin_norm"].notna()].copy()
        if missing_ric_match.any() and not by_isin.empty:
            fallback = panel.loc[missing_ric_match, ["_row_order", "isin_norm", "date"]].merge(
                by_isin.add_prefix("isin_est_"),
                left_on=["isin_norm", "date"],
                right_on=["isin_est_isin_norm", "isin_est_date"],
                how="left",
                validate="many_to_one",
            )
            fallback = fallback.set_index("_row_order")
            for source_column in by_isin.add_prefix("isin_est_").columns:
                target_column = source_column.removeprefix("isin_")
                if target_column in panel and source_column in fallback:
                    values = fallback[source_column].reindex(panel["_row_order"])
                    panel[target_column] = panel[target_column].where(
                        panel[target_column].notna(),
                        values,
                    )

    rename = {
        "est_ric_norm": "est_ric",
        "est_isin_norm": "est_isin",
    }
    panel = panel.rename(columns=rename)
    for column in ESTIMATES_HELPER_COLUMNS:
        if column not in panel:
            panel[column] = np.nan
    for column in ESTIMATES_RAW_COLUMNS:
        if column not in panel:
            panel[column] = np.nan
    _to_numeric(panel, ESTIMATES_RAW_COLUMNS)
    panel["est_identifier_mismatch"] = False
    if strict_identifier_match:
        estimate_isin = _normalise_identifier(panel["est_isin"])
        panel_isin = _normalise_identifier(panel["isin_norm"])
        mismatch = (
            panel["est_snapshot_date"].notna()
            & estimate_isin.notna()
            & panel_isin.notna()
            & estimate_isin.ne(panel_isin)
        )
        panel["est_identifier_mismatch"] = mismatch.fillna(False)
        null_columns = list(
            dict.fromkeys(
                column
                for column in [
                    *ESTIMATES_HELPER_COLUMNS,
                    *ESTIMATES_RAW_COLUMNS,
                    "est_date",
                ]
                if column in panel
            )
        )
        panel.loc[mismatch, null_columns] = np.nan

    panel["month_index"] = panel["date"].dt.year * 12 + panel["date"].dt.month
    groups = panel["ric_norm"]
    if "price_close" in panel:
        price = pd.to_numeric(panel["price_close"], errors="coerce").abs()
    else:
        price = pd.Series(np.nan, index=panel.index, dtype="float64")
    panel["est_eps_yield"] = _safe_divide(panel["est_eps_mean"], price)
    panel["est_price_target_upside"] = _safe_divide(
        panel["est_price_target_mean"],
        price,
    ).sub(1.0)

    panel["est_eps_revision_1m"] = _lagged_change(panel["est_eps_mean"], groups, panel["month_index"], 1)
    panel["est_eps_revision_3m"] = _lagged_change(panel["est_eps_mean"], groups, panel["month_index"], 3)
    panel["est_revenue_revision_1m"] = _lagged_change(
        panel["est_revenue_mean"],
        groups,
        panel["month_index"],
        1,
    )
    panel["est_revenue_revision_3m"] = _lagged_change(
        panel["est_revenue_mean"],
        groups,
        panel["month_index"],
        3,
    )
    panel["est_price_target_revision_1m"] = _lagged_change(
        panel["est_price_target_mean"],
        groups,
        panel["month_index"],
        1,
    )
    panel["est_price_target_revision_3m"] = _lagged_change(
        panel["est_price_target_mean"],
        groups,
        panel["month_index"],
        3,
    )
    recommendation_prior = panel["est_recommendation_mean"].groupby(groups, sort=False).shift(3)
    recommendation_prior_month = panel["month_index"].groupby(groups, sort=False).shift(3)
    exact_three_months = panel["month_index"].sub(recommendation_prior_month).eq(3)
    panel["est_recommendation_strength"] = -pd.to_numeric(
        panel["est_recommendation_mean"], errors="coerce"
    )
    panel["est_recommendation_change_3m"] = (
        recommendation_prior.sub(panel["est_recommendation_mean"])
    ).where(exact_three_months)

    panel["est_eps_dispersion"] = _dispersion(
        panel["est_eps_mean"],
        panel["est_eps_high"],
        panel["est_eps_low"],
        panel["est_eps_std"],
    )
    panel["est_revenue_dispersion"] = _dispersion(
        panel["est_revenue_mean"],
        panel["est_revenue_high"],
        panel["est_revenue_low"],
        panel["est_revenue_std"],
    )
    panel["est_price_target_dispersion"] = _dispersion(
        panel["est_price_target_mean"],
        panel["est_price_target_high"],
        panel["est_price_target_low"],
        panel["est_price_target_std"],
    )
    panel["est_eps_coverage"] = np.log1p(panel["est_eps_count"].where(panel["est_eps_count"].ge(0)))
    panel["est_revenue_coverage"] = np.log1p(
        panel["est_revenue_count"].where(panel["est_revenue_count"].ge(0))
    )
    panel["est_price_target_coverage"] = np.log1p(
        panel["est_price_target_count"].where(panel["est_price_target_count"].ge(0))
    )
    coverage_total = (
        panel[["est_eps_count", "est_revenue_count", "est_price_target_count", "est_recommendation_count"]]
        .clip(lower=0)
        .sum(axis=1, min_count=1)
    )
    panel["est_coverage_composite"] = np.log1p(coverage_total)

    panel = (
        panel.sort_values("_row_order")
        .drop(columns=[column for column in ["_row_order", "est_date", "month_index"] if column in panel])
        .reset_index(drop=True)
    )
    return panel


def filter_extreme_estimate_features(
    panel: pd.DataFrame,
    config: EstimatesPanelConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = panel.copy()
    checks: dict[str, pd.Series] = {}
    if "est_eps_yield" in out:
        values = pd.to_numeric(out["est_eps_yield"], errors="coerce")
        checks["est_eps_yield"] = values.abs().gt(config.max_abs_eps_yield)
    if "est_price_target_upside" in out:
        values = pd.to_numeric(out["est_price_target_upside"], errors="coerce")
        checks["est_price_target_upside"] = values.lt(
            config.min_price_target_upside
        ) | values.gt(config.max_price_target_upside)
    for feature in ESTIMATES_INFORMATION_TYPES["revisions"]:
        if feature in out:
            values = pd.to_numeric(out[feature], errors="coerce")
            checks[feature] = values.abs().gt(config.max_abs_revision)
    for feature in ESTIMATES_INFORMATION_TYPES["dispersion"]:
        if feature in out:
            values = pd.to_numeric(out[feature], errors="coerce")
            checks[feature] = values.lt(0.0) | values.gt(config.max_dispersion)

    audit: dict[str, int] = {}
    for feature, invalid in checks.items():
        invalid = invalid.fillna(False) & out[feature].notna()
        audit[feature] = int(invalid.sum())
        out.loc[invalid, feature] = np.nan
    return out, audit


def apply_estimates_signal_lag(
    panel: pd.DataFrame,
    signal_lag_months: int,
) -> pd.DataFrame:
    if signal_lag_months <= 0:
        return panel
    out = panel.copy()
    out["_lag_row_order"] = np.arange(len(out))
    out["_lag_month_index"] = out["date"].dt.year * 12 + out["date"].dt.month
    out = out.sort_values(["ric_norm", "date"])
    grouped = out.groupby("ric_norm", sort=False)
    prior_month = grouped["_lag_month_index"].shift(signal_lag_months)
    exact_gap = out["_lag_month_index"].sub(prior_month).eq(signal_lag_months)
    lag_columns = [
        column
        for column in [
            *ESTIMATES_HELPER_COLUMNS,
            *ESTIMATES_RAW_COLUMNS,
            *ESTIMATES_EXTENSION_FEATURES,
            "est_identifier_mismatch",
        ]
        if column in out
    ]
    for column in lag_columns:
        shifted = grouped[column].shift(signal_lag_months)
        out[column] = shifted.where(exact_gap)
    out["est_signal_lag_months"] = np.where(
        exact_gap,
        float(signal_lag_months),
        np.nan,
    )
    return (
        out.sort_values("_lag_row_order")
        .drop(columns=["_lag_row_order", "_lag_month_index"])
        .reset_index(drop=True)
    )


def _feature_rank_frame(
    panel: pd.DataFrame,
    features: list[str],
    eligible_column: str = "eligible",
) -> pd.DataFrame:
    out = panel.copy()
    static_eligible = out[eligible_column].fillna(False) if eligible_column in out else pd.Series(True, index=out.index)
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


def add_estimates_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    out = _feature_rank_frame(panel, ESTIMATES_EXTENSION_FEATURES)
    out["estimates_feature_count"] = out[ESTIMATES_EXTENSION_FEATURES].notna().sum(axis=1)
    out = _feature_rank_frame(out, ESTIMATES_MISSINGNESS_FEATURES)
    existing_features = [
        feature
        for feature in [*ALL_RAW_FEATURES, *COMPUSTAT_EXTENSION_FEATURES]
        if feature in out
    ]
    out["deep_estimates_feature_count"] = out[[*existing_features, *ESTIMATES_EXTENSION_FEATURES]].notna().sum(
        axis=1
    )
    return out


def build_estimates_enriched_panel(
    base_panel: pd.DataFrame,
    estimates: pd.DataFrame,
    config: EstimatesPanelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = config or EstimatesPanelConfig()
    snapshot_features, snapshot_audit = prepare_estimates_snapshot_features(estimates)
    panel = merge_estimates_features(
        base_panel,
        snapshot_features,
        strict_identifier_match=config.strict_identifier_match,
    )
    identifier_mismatches = int(
        panel.get("est_identifier_mismatch", pd.Series(False, index=panel.index))
        .fillna(False)
        .sum()
    )
    rows_with_estimates_before_lag = int(panel["est_snapshot_date"].notna().sum())
    outlier_audit: dict[str, int] = {}
    if config.filter_extreme_estimates:
        panel, outlier_audit = filter_extreme_estimate_features(panel, config)
    panel = apply_estimates_signal_lag(panel, config.signal_lag_months)
    panel = add_estimates_cross_sectional_ranks(panel)
    eligible = panel["eligible"].fillna(False) if "eligible" in panel else pd.Series(True, index=panel.index)
    audit = {
        "config": asdict(config),
        "snapshot": snapshot_audit,
        "controls": {
            "identifier_mismatch_rows": identifier_mismatches,
            "rows_with_estimates_before_signal_lag": rows_with_estimates_before_lag,
            "rows_with_estimates_after_signal_lag": int(panel["est_snapshot_date"].notna().sum()),
            "outlier_filtered_rows_by_feature": outlier_audit,
        },
        "panel": {
            "rows": int(len(panel)),
            "unique_rics": int(panel["ric"].nunique()),
            "rows_with_estimates": int(panel["est_snapshot_date"].notna().sum()),
            "eligible_rows_with_estimates": int((eligible & panel["est_snapshot_date"].notna()).sum()),
            "unique_rics_with_estimates": int(
                panel.loc[panel["est_snapshot_date"].notna(), "ric"].nunique()
            ),
            "mean_estimates_feature_count": float(panel["estimates_feature_count"].mean()),
            "mean_estimates_feature_count_with_estimates": float(
                panel.loc[
                    panel["est_snapshot_date"].notna(),
                    "estimates_feature_count",
                ].mean()
            ),
            "rows_with_any_estimates_feature": int(
                panel[ESTIMATES_EXTENSION_FEATURES].notna().any(axis=1).sum()
            ),
            "feature_non_null": {
                feature: int(panel[feature].notna().sum())
                for feature in ESTIMATES_EXTENSION_FEATURES
            },
        },
    }
    return snapshot_features, panel, audit


def estimates_feature_dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        ESTIMATES_FEATURE_DICTIONARY,
        columns=["feature", "definition", "information_timing"],
    ).assign(
        model_column=lambda frame: frame["feature"] + "_rank",
        transformation="Monthly cross-sectional percentile rank scaled to [-1, 1]",
        missing_value_rule="Rank set to 0 (cross-sectional median); estimates_feature_count retains coverage",
    )


def write_estimates_outputs(
    output_dir: Path,
    snapshot_features: pd.DataFrame,
    panel: pd.DataFrame,
    audit: dict,
    panel_filename: str = "monthly_feature_panel_estimates.parquet",
    snapshot_filename: str = "refinitiv_estimates_snapshot_features.parquet",
    dictionary_filename: str = "refinitiv_estimates_feature_dictionary.csv",
    audit_filename: str = "refinitiv_estimates_enrichment_audit.json",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_features.to_parquet(
        output_dir / snapshot_filename,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    panel.to_parquet(
        output_dir / panel_filename,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    estimates_feature_dictionary_frame().to_csv(
        output_dir / dictionary_filename,
        index=False,
    )
    (output_dir / audit_filename).write_text(
        json.dumps(audit, indent=2)
    )
