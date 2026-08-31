"""Audit currency metadata and missing retirement-month returns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_exports"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "asset_pricing"

CURRENCY_COLUMNS = [
    "CF_CURR",
    "TR.PRICEMOPRICECURRENCY",
    "TR.COMPANYREPORTCURRENCY",
]


def month_gap(later: pd.Period, earlier: pd.Period) -> float:
    if pd.isna(later) or pd.isna(earlier):
        return np.nan
    return float((later - earlier).n)


def build_audit(raw_dir: Path, processed_dir: Path) -> dict:
    universe = pd.read_csv(processed_dir / "clean_universe.csv", low_memory=False)
    eligible = universe.loc[universe["eligible"], ["ric", "screen_state"]].copy()
    reference_path = (
        raw_dir / "supplemental" / "refinitiv_security_master_supplement.parquet"
    )
    reference = pd.read_parquet(reference_path)
    reference = reference.rename(columns={"Instrument": "ric"})
    reference["retire_date"] = pd.to_datetime(
        reference.get("TR.RETIREDATE"), errors="coerce"
    )
    reference["retire_month"] = reference["retire_date"].dt.to_period("M")

    monthly = pd.read_csv(
        raw_dir / "refinitiv_monthly_panel_tidy.csv",
        usecols=["date", "ric", "total_return_1m"],
        parse_dates=["date"],
        low_memory=False,
    )
    monthly = monthly[monthly["ric"].isin(set(eligible["ric"]))].copy()
    monthly["month"] = monthly["date"].dt.to_period("M")
    monthly["return_1m"] = pd.to_numeric(
        monthly["total_return_1m"], errors="coerce"
    ).div(100)

    last_valid = (
        monthly.dropna(subset=["return_1m"])
        .groupby("ric", as_index=False)["month"]
        .max()
        .rename(columns={"month": "last_valid_return_month"})
    )
    retirement = eligible.merge(
        reference[
            [
                "ric",
                "TR.INSTRUMENTTYPE",
                "TR.ISDELISTEDQUOTE",
                "retire_date",
                "retire_month",
            ]
        ],
        on="ric",
        how="left",
    ).merge(last_valid, on="ric", how="left")
    retirement = retirement[retirement["retire_month"].notna()].copy()
    retirement_month_returns = monthly[["ric", "month", "return_1m"]].merge(
        retirement[["ric", "retire_month"]],
        on="ric",
        how="inner",
    )
    retirement_month_returns = retirement_month_returns[
        retirement_month_returns["month"].eq(
            retirement_month_returns["retire_month"]
        )
    ][["ric", "return_1m"]].rename(
        columns={"return_1m": "return_in_retirement_month"}
    )
    retirement = retirement.merge(
        retirement_month_returns,
        on="ric",
        how="left",
    )
    retirement["last_return_gap_months"] = retirement.apply(
        lambda row: month_gap(
            row["retire_month"], row["last_valid_return_month"]
        ),
        axis=1,
    )
    retirement["missing_retirement_month_return"] = retirement[
        "return_in_retirement_month"
    ].isna()
    retirement.to_csv(processed_dir / "delisting_return_audit.csv", index=False)

    primary_reference_path = raw_dir / "refinitiv_universe_master.csv"
    primary_reference = (
        pd.read_csv(primary_reference_path, low_memory=False)
        if primary_reference_path.exists()
        else pd.DataFrame()
    )
    primary_reference = primary_reference.rename(columns={"Instrument": "ric"})
    currency_available = [
        column for column in CURRENCY_COLUMNS if column in primary_reference
    ]
    manifest_path = raw_dir / "python_lseg_download_manifest.json"
    download_manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {}
    )
    base_currency = download_manifest.get("base_currency")
    if currency_available:
        currency = eligible.merge(
            primary_reference[
                ["ric", "TR.EXCHANGECOUNTRY", *currency_available]
            ],
            on="ric",
            how="left",
        )
        currency.to_csv(processed_dir / "currency_audit.csv", index=False)
        currency_status = {
            "status": (
                "explicit_common_currency"
                if base_currency
                else "metadata_available_currency_parameter_unverified"
            ),
            "download_base_currency": base_currency,
            "fields": currency_available,
            "missing_by_field": {
                column: int(currency[column].isna().sum())
                for column in currency_available
            },
            "unique_values_by_field": {
                column: int(currency[column].nunique(dropna=True))
                for column in currency_available
            },
        }
    else:
        currency_status = {
            "status": "not_downloaded",
            "required_fields": CURRENCY_COLUMNS,
            "conclusion": (
                "Currency metadata or an explicit download base currency "
                "could not be verified."
            ),
        }

    audit = {
        "eligible_securities": int(len(eligible)),
        "currency": currency_status,
        "delisting": {
            "eligible_inactive_securities": int(
                eligible["screen_state"].astype(str).str.lower().eq("inactive").sum()
            ),
            "eligible_with_retire_date": int(len(retirement)),
            "retirement_month_return_present": int(
                retirement["return_in_retirement_month"].notna().sum()
            ),
            "retirement_month_return_missing": int(
                retirement["return_in_retirement_month"].isna().sum()
            ),
            "last_return_within_one_month_of_retirement": int(
                retirement["last_return_gap_months"].between(0, 1).sum()
            ),
            "median_last_return_gap_months": float(
                retirement["last_return_gap_months"].median()
            ),
            "conclusion": (
                "Retirement-month returns are too incomplete to treat the "
                "short-leg backtest as free of delisting bias."
            ),
        },
    }
    (processed_dir / "data_integrity_audit.json").write_text(
        json.dumps(audit, indent=2)
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    audit = build_audit(args.raw_dir, args.processed_dir)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
