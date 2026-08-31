"""Audit Refinitiv analyst-detail spot-check vintage timestamps."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "estimates_detail_spotcheck"
    / "refinitiv_estimates_detail_spotcheck.parquet"
)


def _normalise_column_name(column: str) -> str:
    return str(column).upper().replace("__", ".")


def find_column(
    columns: list[str],
    predicate: Callable[[str], bool],
    label: str,
    required: bool = True,
) -> str | None:
    matches = [column for column in columns if predicate(_normalise_column_name(column))]
    if matches:
        return matches[0]
    if required:
        raise ValueError(f"Could not find {label} column")
    return None


def estimate_date_column(columns: list[str]) -> str:
    return find_column(
        columns,
        lambda column: "EPSESTVALUE.DATE" in column,
        "EPS estimate date",
    )


def estimate_value_column(columns: list[str]) -> str:
    return find_column(
        columns,
        lambda column: column.endswith("EPSESTVALUE"),
        "EPS estimate value",
    )


def broker_column(columns: list[str]) -> str | None:
    return find_column(
        columns,
        lambda column: "EPSESTVALUE.BROKERNAME" in column,
        "EPS estimate broker",
        required=False,
    )


def parse_datetimes(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.replace("", pd.NA), errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", pd.NA), errors="coerce")


def build_audit(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = detail.columns.tolist()
    date_col = estimate_date_column(columns)
    value_col = estimate_value_column(columns)
    broker_col = broker_column(columns)

    required_sample_columns = ["sample_ric", "sample_snapshot_date", "query_start"]
    missing = [column for column in required_sample_columns if column not in detail]
    if missing:
        raise ValueError(f"Missing required sample metadata columns: {missing}")

    work = detail.copy()
    work["_estimate_date"] = parse_datetimes(work[date_col])
    work["_estimate_value"] = parse_numeric(work[value_col])
    work["_snapshot_date"] = parse_datetimes(work["sample_snapshot_date"])
    work["_query_start"] = parse_datetimes(work["query_start"])

    has_date = work["_estimate_date"].notna()
    has_value = work["_estimate_value"].notna()
    after_snapshot = has_date & work["_snapshot_date"].notna() & (
        work["_estimate_date"].dt.normalize() > work["_snapshot_date"].dt.normalize()
    )
    before_query_start = has_date & work["_query_start"].notna() & (
        work["_estimate_date"].dt.normalize() < work["_query_start"].dt.normalize()
    )

    group_keys = ["sample_ric", "sample_snapshot_date"]
    if "sample_panel_date" in work:
        group_keys.insert(1, "sample_panel_date")

    per_sample = (
        work.assign(
            _dated_row=has_date,
            _numeric_value_row=has_value,
        )
        .groupby(group_keys, dropna=False)
        .agg(
            rows=("sample_ric", "size"),
            dated_rows=("_dated_row", "sum"),
            numeric_value_rows=("_numeric_value_row", "sum"),
            latest_estimate_date=("_estimate_date", "max"),
            earliest_estimate_date=("_estimate_date", "min"),
        )
        .reset_index()
    )

    permission_denied_rows = 0
    visible_broker_rows = 0
    if broker_col is not None:
        broker = work[broker_col].fillna("").astype(str).str.strip()
        permission_denied_rows = int(
            broker.str.contains("Permission Denied", case=False, na=False).sum(),
        )
        visible_broker_rows = int(broker.ne("").sum() - permission_denied_rows)

    audit = pd.DataFrame(
        [
            {
                "sample_firm_months": int(per_sample.shape[0]),
                "detail_rows": int(work.shape[0]),
                "detail_rows_with_estimate_date": int(has_date.sum()),
                "detail_rows_with_numeric_estimate_value": int(has_value.sum()),
                "detail_rows_with_date_and_value": int((has_date & has_value).sum()),
                "samples_with_dated_rows": int(per_sample["dated_rows"].gt(0).sum()),
                "samples_with_numeric_value_rows": int(
                    per_sample["numeric_value_rows"].gt(0).sum(),
                ),
                "samples_with_multiple_dated_rows": int(
                    per_sample["dated_rows"].gt(1).sum(),
                ),
                "dated_rows_after_snapshot": int(after_snapshot.sum()),
                "dated_rows_before_query_start": int(before_query_start.sum()),
                "broker_permission_denied_rows": permission_denied_rows,
                "visible_broker_rows": visible_broker_rows,
            },
        ],
    )
    return audit, per_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    detail = pd.read_parquet(args.input)
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    audit, per_sample = build_audit(detail)
    audit_path = output_dir / "vintage_spotcheck_audit.csv"
    per_sample_path = output_dir / "vintage_spotcheck_per_sample.csv"
    audit.to_csv(audit_path, index=False)
    per_sample.to_csv(per_sample_path, index=False)
    print(audit.to_string(index=False))
    print(f"audit -> {audit_path}")
    print(f"per-sample -> {per_sample_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
