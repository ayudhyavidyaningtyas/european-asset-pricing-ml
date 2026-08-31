"""Build consensus-to-actual forecast-error panel for mechanism tests."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "strict_estimates_lag1"
    / "monthly_feature_panel_estimates_strict_lag1.parquet"
)
DEFAULT_ACTUALS = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "refinitiv_actuals_annual_complete.csv.gz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "forecast_errors"
)
DEFAULT_PANEL_FILENAME = "monthly_forecast_error_panel.parquet"

REVISION_COLUMNS = [
    "est_eps_revision_1m",
    "est_eps_revision_3m",
    "est_revenue_revision_1m",
    "est_revenue_revision_3m",
    "est_price_target_revision_1m",
    "est_price_target_revision_3m",
]

PANEL_COLUMNS = [
    "date",
    "ric",
    "TR.ISIN",
    "screen_country",
    "TR.TRBCECONOMICSECTOR",
    "company_market_cap",
    "price_close",
    "log_size_rank",
    "book_to_market_rank",
    "momentum_12_2_rank",
    "volatility_12m_rank",
    "est_ric",
    "est_isin",
    "est_snapshot_date",
    "est_signal_lag_months",
    "estimates_feature_count",
    "est_eps_mean",
    "est_revenue_mean",
    "est_eps_revision_1m",
    "est_eps_revision_3m",
    "est_revenue_revision_1m",
    "est_revenue_revision_3m",
    "est_price_target_revision_1m",
    "est_price_target_revision_3m",
    "est_eps_revision_1m_rank",
    "est_eps_revision_3m_rank",
    "est_revenue_revision_1m_rank",
    "est_revenue_revision_3m_rank",
    "est_price_target_revision_1m_rank",
    "est_price_target_revision_3m_rank",
    "est_eps_dispersion_rank",
    "est_revenue_dispersion_rank",
    "est_coverage_composite_rank",
]


@dataclass(frozen=True)
class ForecastErrorConfig:
    sample_start_date: str = "2005-01-31"
    require_revision_signal: bool = True
    require_estimate_signal_lag_months: int = 1
    drop_fiscal_year_end_changes: bool = True
    min_abs_consensus_for_ratio: float = 1e-6
    max_abs_actual_to_consensus_ratio: float = 25.0
    max_abs_eps_to_price: float = 1.0
    max_abs_eps_error_to_price: float = 1.0
    max_abs_revenue_to_market_cap: float = 25.0
    max_abs_revenue_error_to_market_cap: float = 25.0
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    min_winsorize_group: int = 20


def _normalise_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA})


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def clean_actuals(actuals: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize raw Refinitiv actuals rows."""
    required = [
        "Instrument",
        "TR.EPSACTVALUE",
        "TR.EPSACTVALUE.DATE",
        "TR.EPSACTVALUE.PERIODENDDATE",
        "TR.EPSFRACTVALUE",
        "TR.EPSFRACTVALUE.DATE",
        "TR.EPSFRACTVALUE.PERIODENDDATE",
        "TR.REVENUEACTVALUE",
        "TR.REVENUEACTVALUE.DATE",
        "TR.REVENUEACTVALUE.PERIODENDDATE",
    ]
    missing = [column for column in required if column not in actuals]
    if missing:
        raise ValueError(f"Actuals file missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "actual_ric": _normalise_text(actuals["Instrument"]),
            "actual_returned_ric": _normalise_text(
                actuals.get("TR.RIC", pd.Series(pd.NA, index=actuals.index)),
            ),
            "actual_isin": _normalise_text(
                actuals.get("TR.ISIN", pd.Series(pd.NA, index=actuals.index)),
            ),
            "eps_actual": _numeric(actuals["TR.EPSACTVALUE"]),
            "eps_announce_date": _datetime(actuals["TR.EPSACTVALUE.DATE"]),
            "eps_period_end": _datetime(actuals["TR.EPSACTVALUE.PERIODENDDATE"]),
            "epsfr_actual": _numeric(actuals["TR.EPSFRACTVALUE"]),
            "epsfr_announce_date": _datetime(actuals["TR.EPSFRACTVALUE.DATE"]),
            "epsfr_period_end": _datetime(actuals["TR.EPSFRACTVALUE.PERIODENDDATE"]),
            "revenue_actual": _numeric(actuals["TR.REVENUEACTVALUE"]),
            "revenue_announce_date": _datetime(actuals["TR.REVENUEACTVALUE.DATE"]),
            "revenue_period_end": _datetime(
                actuals["TR.REVENUEACTVALUE.PERIODENDDATE"],
            ),
        },
    )
    out = out[out["actual_ric"].notna()].copy()
    return out


def actual_table(
    actuals: pd.DataFrame,
    value_column: str,
    announce_column: str,
    period_column: str,
    prefix: str,
) -> pd.DataFrame:
    table = actuals[
        [
            "actual_ric",
            "actual_returned_ric",
            "actual_isin",
            value_column,
            announce_column,
            period_column,
        ]
    ].rename(
        columns={
            value_column: f"{prefix}_actual",
            announce_column: f"{prefix}_announce_date",
            period_column: f"{prefix}_period_end",
        },
    )
    table = table.dropna(
        subset=["actual_ric", f"{prefix}_actual", f"{prefix}_announce_date", f"{prefix}_period_end"],
    ).copy()
    table[f"{prefix}_period_end_month"] = table[f"{prefix}_period_end"].dt.month
    table[f"{prefix}_fye_month_count"] = table.groupby("actual_ric")[
        f"{prefix}_period_end_month"
    ].transform("nunique")
    table[f"{prefix}_fye_month_changed"] = table[f"{prefix}_fye_month_count"].gt(1)
    table = table.sort_values(
        ["actual_ric", f"{prefix}_period_end", f"{prefix}_announce_date"],
    )
    return table.drop_duplicates(
        subset=["actual_ric", f"{prefix}_period_end"],
        keep="first",
    )


def load_panel(panel_path: Path, config: ForecastErrorConfig) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path, columns=PANEL_COLUMNS)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["est_snapshot_date"] = pd.to_datetime(panel["est_snapshot_date"])
    panel = panel[panel["date"].ge(pd.Timestamp(config.sample_start_date))].copy()
    has_estimates = pd.to_numeric(
        panel["estimates_feature_count"],
        errors="coerce",
    ).gt(0)
    lag = pd.to_numeric(panel["est_signal_lag_months"], errors="coerce")
    invalid_lag = has_estimates & (lag.isna() | lag.lt(config.require_estimate_signal_lag_months))
    if invalid_lag.any():
        raise ValueError(
            "Estimate signal lag guard failed: "
            f"{int(invalid_lag.sum()):,} rows below lag "
            f"{config.require_estimate_signal_lag_months}."
        )
    panel = panel[has_estimates & panel["est_snapshot_date"].notna()].copy()
    if config.require_revision_signal:
        panel = panel[panel[REVISION_COLUMNS].notna().any(axis=1)].copy()
    panel = panel.reset_index(drop=True)
    panel["panel_row_id"] = np.arange(len(panel), dtype=np.int64)
    return panel


def panel_match_keys(panel: pd.DataFrame) -> pd.DataFrame:
    keys = panel[
        [
            "panel_row_id",
            "ric",
            "est_ric",
            "est_snapshot_date",
        ]
    ].copy()
    left = keys.rename(columns={"ric": "match_ric"})
    left["match_source"] = "panel_ric"
    right = keys.rename(columns={"est_ric": "match_ric"})
    right["match_source"] = "est_ric"
    out = pd.concat([right, left], ignore_index=True)
    out["match_ric"] = _normalise_text(out["match_ric"])
    out = out[out["match_ric"].notna()].drop_duplicates(
        subset=["panel_row_id", "match_ric"],
    )
    return out


def match_actuals(
    panel: pd.DataFrame,
    actuals: pd.DataFrame,
    prefix: str,
    config: ForecastErrorConfig,
) -> pd.DataFrame:
    output_columns = [
        "panel_row_id",
        f"{prefix}_actual_ric",
        f"{prefix}_actual_returned_ric",
        f"{prefix}_actual_isin",
        f"{prefix}_actual",
        f"{prefix}_announce_date",
        f"{prefix}_period_end",
        f"{prefix}_fye_month_count",
        f"{prefix}_fye_month_changed",
        f"{prefix}_match_source",
    ]

    def empty_match() -> pd.DataFrame:
        return pd.DataFrame(columns=output_columns)

    keys = panel_match_keys(panel)
    merged = keys.merge(
        actuals,
        left_on="match_ric",
        right_on="actual_ric",
        how="inner",
        validate="many_to_many",
    )
    if merged.empty:
        return empty_match()
    announced_after_snapshot = merged[f"{prefix}_announce_date"].dt.normalize().gt(
        merged["est_snapshot_date"].dt.normalize(),
    )
    merged = merged[announced_after_snapshot].copy()
    if config.drop_fiscal_year_end_changes:
        merged = merged[~merged[f"{prefix}_fye_month_changed"]].copy()
    if merged.empty:
        return empty_match()
    merged["match_source_order"] = np.where(merged["match_source"].eq("est_ric"), 0, 1)
    merged = merged.sort_values(
        [
            "panel_row_id",
            f"{prefix}_announce_date",
            f"{prefix}_period_end",
            "match_source_order",
        ],
    )
    keep_columns = [
        "panel_row_id",
        "actual_ric",
        "actual_returned_ric",
        "actual_isin",
        f"{prefix}_actual",
        f"{prefix}_announce_date",
        f"{prefix}_period_end",
        f"{prefix}_fye_month_count",
        f"{prefix}_fye_month_changed",
        "match_source",
    ]
    matched = merged.drop_duplicates("panel_row_id", keep="first")[keep_columns]
    return matched.rename(
        columns={
            "actual_ric": f"{prefix}_actual_ric",
            "actual_returned_ric": f"{prefix}_actual_returned_ric",
            "actual_isin": f"{prefix}_actual_isin",
            "match_source": f"{prefix}_match_source",
        },
    )


def _ratio_filter(
    actual: pd.Series,
    consensus: pd.Series,
    config: ForecastErrorConfig,
) -> pd.Series:
    denominator = consensus.abs()
    ratio = actual.div(consensus).abs()
    return denominator.lt(config.min_abs_consensus_for_ratio) | ratio.le(
        config.max_abs_actual_to_consensus_ratio,
    )


def _winsorize_by_month(
    frame: pd.DataFrame,
    column: str,
    config: ForecastErrorConfig,
) -> pd.Series:
    def clip_group(values: pd.Series) -> pd.Series:
        clean = values.dropna()
        if len(clean) < config.min_winsorize_group:
            return values
        low, high = clean.quantile([config.winsorize_lower, config.winsorize_upper])
        return values.clip(lower=low, upper=high)

    return frame.groupby("date", group_keys=False)[column].apply(clip_group)


def add_forecast_errors(frame: pd.DataFrame, config: ForecastErrorConfig) -> pd.DataFrame:
    out = frame.copy()
    price = pd.to_numeric(out["price_close"], errors="coerce")
    market_cap = pd.to_numeric(out["company_market_cap"], errors="coerce")
    est_eps = pd.to_numeric(out["est_eps_mean"], errors="coerce")
    est_revenue = pd.to_numeric(out["est_revenue_mean"], errors="coerce")

    out["eps_error"] = out["eps_actual"] - est_eps
    out["epsfr_error"] = out["epsfr_actual"] - est_eps
    out["revenue_error"] = out["revenue_actual"] - est_revenue
    out["eps_error_to_price"] = out["eps_error"] / price
    out["epsfr_error_to_price"] = out["epsfr_error"] / price
    out["revenue_error_to_market_cap"] = out["revenue_error"] / market_cap
    out["eps_actual_to_price"] = out["eps_actual"] / price
    out["eps_consensus_to_price"] = est_eps / price
    out["epsfr_actual_to_price"] = out["epsfr_actual"] / price
    out["revenue_actual_to_market_cap"] = out["revenue_actual"] / market_cap
    out["revenue_consensus_to_market_cap"] = est_revenue / market_cap

    eps_known = out["eps_actual"].notna() & est_eps.notna()
    epsfr_known = out["epsfr_actual"].notna() & est_eps.notna()
    revenue_known = out["revenue_actual"].notna() & est_revenue.notna()

    out["eps_error_valid"] = (
        eps_known
        & price.gt(0)
        & _ratio_filter(out["eps_actual"], est_eps, config)
        & out["eps_actual_to_price"].abs().le(config.max_abs_eps_to_price)
        & out["eps_consensus_to_price"].abs().le(config.max_abs_eps_to_price)
        & out["eps_error_to_price"].abs().le(config.max_abs_eps_error_to_price)
    )
    out["epsfr_error_valid"] = (
        epsfr_known
        & price.gt(0)
        & _ratio_filter(out["epsfr_actual"], est_eps, config)
        & out["epsfr_actual_to_price"].abs().le(config.max_abs_eps_to_price)
        & out["eps_consensus_to_price"].abs().le(config.max_abs_eps_to_price)
        & out["epsfr_error_to_price"].abs().le(config.max_abs_eps_error_to_price)
    )
    out["revenue_error_valid"] = (
        revenue_known
        & market_cap.gt(0)
        & _ratio_filter(out["revenue_actual"], est_revenue, config)
        & out["revenue_actual_to_market_cap"].abs().le(
            config.max_abs_revenue_to_market_cap,
        )
        & out["revenue_consensus_to_market_cap"].abs().le(
            config.max_abs_revenue_to_market_cap,
        )
        & out["revenue_error_to_market_cap"].abs().le(
            config.max_abs_revenue_error_to_market_cap,
        )
    )

    for source, valid in [
        ("eps_error_to_price", out["eps_error_valid"]),
        ("epsfr_error_to_price", out["epsfr_error_valid"]),
        ("revenue_error_to_market_cap", out["revenue_error_valid"]),
    ]:
        target = f"{source}_winsorized"
        out[target] = np.nan
        if valid.any():
            out.loc[valid, target] = _winsorize_by_month(out.loc[valid], source, config)
    return out


def build_forecast_error_panel(
    panel_path: Path,
    actuals_path: Path,
    config: ForecastErrorConfig,
) -> tuple[pd.DataFrame, dict]:
    panel = load_panel(panel_path, config)
    raw_actuals = pd.read_csv(actuals_path, low_memory=False)
    actuals = clean_actuals(raw_actuals)
    eps_actuals = actual_table(
        actuals,
        "eps_actual",
        "eps_announce_date",
        "eps_period_end",
        "eps",
    )
    epsfr_actuals = actual_table(
        actuals,
        "epsfr_actual",
        "epsfr_announce_date",
        "epsfr_period_end",
        "epsfr",
    )
    revenue_actuals = actual_table(
        actuals,
        "revenue_actual",
        "revenue_announce_date",
        "revenue_period_end",
        "revenue",
    )
    matched = panel.merge(match_actuals(panel, eps_actuals, "eps", config), on="panel_row_id", how="left")
    matched = matched.merge(
        match_actuals(panel, epsfr_actuals, "epsfr", config),
        on="panel_row_id",
        how="left",
    )
    matched = matched.merge(
        match_actuals(panel, revenue_actuals, "revenue", config),
        on="panel_row_id",
        how="left",
    )
    result = add_forecast_errors(matched, config)
    audit = {
        "config": asdict(config),
        "rows": {
            "raw_actuals": int(len(raw_actuals)),
            "clean_actuals": int(len(actuals)),
            "panel_after_sample_filters": int(len(panel)),
            "matched_eps_actual": int(result["eps_actual"].notna().sum()),
            "matched_epsfr_actual": int(result["epsfr_actual"].notna().sum()),
            "matched_revenue_actual": int(result["revenue_actual"].notna().sum()),
            "valid_eps_error": int(result["eps_error_valid"].sum()),
            "valid_epsfr_error": int(result["epsfr_error_valid"].sum()),
            "valid_revenue_error": int(result["revenue_error_valid"].sum()),
        },
        "actuals": {
            "eps_actual_rows": int(len(eps_actuals)),
            "epsfr_actual_rows": int(len(epsfr_actuals)),
            "revenue_actual_rows": int(len(revenue_actuals)),
            "eps_ric_with_fye_month_changes": int(
                eps_actuals.loc[eps_actuals["eps_fye_month_changed"], "actual_ric"].nunique(),
            ),
            "revenue_ric_with_fye_month_changes": int(
                revenue_actuals.loc[
                    revenue_actuals["revenue_fye_month_changed"],
                    "actual_ric",
                ].nunique(),
            ),
        },
    }
    eps_bad = result["eps_announce_date"].notna() & result["eps_announce_date"].dt.normalize().le(
        result["est_snapshot_date"].dt.normalize(),
    )
    epsfr_bad = result["epsfr_announce_date"].notna() & result["epsfr_announce_date"].dt.normalize().le(
        result["est_snapshot_date"].dt.normalize(),
    )
    revenue_bad = result["revenue_announce_date"].notna() & result[
        "revenue_announce_date"
    ].dt.normalize().le(result["est_snapshot_date"].dt.normalize())
    audit["timing"] = {
        "eps_announce_on_or_before_snapshot": int(eps_bad.sum()),
        "epsfr_announce_on_or_before_snapshot": int(epsfr_bad.sum()),
        "revenue_announce_on_or_before_snapshot": int(revenue_bad.sum()),
    }
    return result, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--actuals", type=Path, default=DEFAULT_ACTUALS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-filename", default=DEFAULT_PANEL_FILENAME)
    parser.add_argument("--sample-start-date", default="2005-01-31")
    parser.add_argument("--keep-no-revision-signal", action="store_true")
    parser.add_argument("--require-estimate-signal-lag-months", type=int, default=1)
    parser.add_argument("--keep-fiscal-year-end-changes", action="store_true")
    parser.add_argument("--max-abs-actual-to-consensus-ratio", type=float, default=25.0)
    parser.add_argument("--max-abs-eps-to-price", type=float, default=1.0)
    parser.add_argument("--max-abs-eps-error-to-price", type=float, default=1.0)
    parser.add_argument("--max-abs-revenue-to-market-cap", type=float, default=25.0)
    parser.add_argument("--max-abs-revenue-error-to-market-cap", type=float, default=25.0)
    args = parser.parse_args()

    config = ForecastErrorConfig(
        sample_start_date=args.sample_start_date,
        require_revision_signal=not args.keep_no_revision_signal,
        require_estimate_signal_lag_months=args.require_estimate_signal_lag_months,
        drop_fiscal_year_end_changes=not args.keep_fiscal_year_end_changes,
        max_abs_actual_to_consensus_ratio=args.max_abs_actual_to_consensus_ratio,
        max_abs_eps_to_price=args.max_abs_eps_to_price,
        max_abs_eps_error_to_price=args.max_abs_eps_error_to_price,
        max_abs_revenue_to_market_cap=args.max_abs_revenue_to_market_cap,
        max_abs_revenue_error_to_market_cap=args.max_abs_revenue_error_to_market_cap,
    )
    panel, audit = build_forecast_error_panel(args.panel, args.actuals, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / args.panel_filename
    panel.to_parquet(panel_path, index=False, engine="pyarrow", compression="zstd")
    audit["outputs"] = {
        "panel": str(panel_path),
        "audit": str(args.output_dir / "forecast_error_panel_audit.json"),
    }
    (args.output_dir / "forecast_error_panel_audit.json").write_text(
        json.dumps(audit, indent=2, default=str),
    )
    print(json.dumps(audit["rows"], indent=2))
    print(json.dumps(audit["timing"], indent=2))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
