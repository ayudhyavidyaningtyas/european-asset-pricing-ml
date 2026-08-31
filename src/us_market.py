"""US-market data helpers for the Europe-vs-US asset-pricing comparison.

The European pipeline links Refinitiv securities to Compustat Global by ISIN.
WRDS Compustat North America commonly exposes CUSIP instead.  For US ordinary
shares, the matching ISIN is mechanically ``US`` + 9-character CUSIP + the ISIN
check digit, so this module derives the Refinitiv-compatible identifier before
the existing Compustat feature builder is reused.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRDS_US_DIR = (
    PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "wrds_compustat_us_exports"
)
DEFAULT_US_ANNUAL_EXPORT = "compustat_us_fundamentals_annual.csv.gz"
DEFAULT_US_MONTHLY_EXPORT = "compustat_us_security_monthly.csv.gz"

WRDS_ANNUAL_COLUMNS = [
    "gvkey",
    "datadate",
    "fyear",
    "indfmt",
    "consol",
    "popsrc",
    "datafmt",
    "tic",
    "cusip",
    "conm",
    "curcd",
    "fic",
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

WRDS_MONTHLY_COLUMNS = [
    "gvkey",
    "iid",
    "datadate",
    "cusip",
    "exchg",
    "curcddvm",
    "prccm",
    "ajexm",
    "cshtrm",
]


def _compact_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(ch for ch in str(value).strip().upper() if ch.isalnum())


def _isin_expanded_digits(value: str) -> str:
    digits = []
    for char in value:
        if char.isdigit():
            digits.append(char)
        elif "A" <= char <= "Z":
            digits.append(str(ord(char) - ord("A") + 10))
        else:
            raise ValueError(f"Unsupported ISIN character {char!r}")
    return "".join(digits)


def isin_check_digit(country_and_nsin: str) -> str:
    """Return the ISO 6166 check digit for a 2-letter country plus 9-char NSIN."""
    base = _compact_identifier(country_and_nsin)
    if len(base) != 11:
        raise ValueError("ISIN base must contain 2 country letters and 9 NSIN characters")
    expanded = _isin_expanded_digits(base)
    total = 0
    double = True
    for char in reversed(expanded):
        value = int(char)
        if double:
            value *= 2
        total += value // 10 + value % 10
        double = not double
    return str((10 - total % 10) % 10)


def cusip_to_us_isin(cusip: object) -> str | None:
    """Convert a 9-character US CUSIP into its standard US ISIN."""
    compact = _compact_identifier(cusip)
    if len(compact) != 9:
        return None
    base = f"US{compact}"
    return f"{base}{isin_check_digit(base)}"


def add_us_isin_from_cusip(
    frame: pd.DataFrame,
    *,
    cusip_column: str = "cusip",
    isin_column: str = "isin",
) -> pd.DataFrame:
    """Fill an ``isin`` column from CUSIP where WRDS does not provide one."""
    out = frame.copy()
    out.columns = [str(column).lower() for column in out.columns]
    if isin_column not in out:
        out[isin_column] = pd.NA
    if cusip_column not in out:
        return out
    derived = out[cusip_column].map(cusip_to_us_isin)
    existing = (
        out[isin_column]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})
    )
    out[isin_column] = existing.combine_first(derived.astype("string"))
    return out


def normalize_wrds_compustat_us_annual(annual: pd.DataFrame) -> pd.DataFrame:
    """Normalize WRDS Compustat North America annual rows for feature building."""
    work = add_us_isin_from_cusip(annual)
    for column, expected in [
        ("indfmt", "INDL"),
        ("datafmt", "STD"),
        ("popsrc", "D"),
        ("consol", "C"),
        ("fic", "USA"),
    ]:
        if column in work:
            work = work[work[column].astype("string").str.upper().eq(expected)]
    return work.reset_index(drop=True)


def normalize_wrds_compustat_us_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Normalize WRDS Compustat North America monthly security rows."""
    work = add_us_isin_from_cusip(monthly)
    if "curcddvm" in work:
        currency = work["curcddvm"].astype("string").str.upper()
        work = work[currency.isna() | currency.eq("USD")]
    return work.reset_index(drop=True)


def load_wrds_compustat_us_exports(
    compustat_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_path = compustat_dir / DEFAULT_US_ANNUAL_EXPORT
    monthly_path = compustat_dir / DEFAULT_US_MONTHLY_EXPORT
    if not annual_path.exists():
        raise FileNotFoundError(f"Missing WRDS Compustat annual export: {annual_path}")
    if not monthly_path.exists():
        raise FileNotFoundError(f"Missing WRDS Compustat monthly export: {monthly_path}")
    annual = pd.read_csv(annual_path, compression="gzip", dtype={"gvkey": str}, low_memory=False)
    monthly = pd.read_csv(
        monthly_path,
        compression="gzip",
        dtype={"gvkey": str, "iid": str},
        low_memory=False,
    )
    return (
        normalize_wrds_compustat_us_annual(annual),
        normalize_wrds_compustat_us_monthly(monthly),
    )


def _sql_date(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def build_wrds_compustat_us_annual_sql(
    *,
    schema: str = "comp",
    start: str | pd.Timestamp = "2000-01-01",
    end: str | pd.Timestamp = "2026-07-08",
) -> str:
    columns = ", ".join(f"f.{column}" for column in WRDS_ANNUAL_COLUMNS)
    start_date = _sql_date(start)
    end_date = _sql_date(end)
    return f"""
select {columns}
from {schema}.funda as f
where f.datadate between date '{start_date}' and date '{end_date}'
  and f.indfmt = 'INDL'
  and f.datafmt = 'STD'
  and f.popsrc = 'D'
  and f.consol = 'C'
  and f.fic = 'USA'
order by f.gvkey, f.datadate
""".strip()


def build_wrds_compustat_us_monthly_sql(
    *,
    schema: str = "comp",
    start: str | pd.Timestamp = "2000-01-01",
    end: str | pd.Timestamp = "2026-07-08",
    primary_exchange_only: bool = True,
) -> str:
    columns = ", ".join(f"s.{column}" for column in WRDS_MONTHLY_COLUMNS)
    start_date = _sql_date(start)
    end_date = _sql_date(end)
    exchange_filter = "  and s.exchg in (11, 12, 14)\n" if primary_exchange_only else ""
    return f"""
select {columns}
from {schema}.secm as s
where s.datadate between date '{start_date}' and date '{end_date}'
  and s.curcddvm = 'USD'
{exchange_filter}order by s.gvkey, s.iid, s.datadate
""".strip()


def _write_csv_gz(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "cols": int(frame.shape[1]),
        "bytes": path.stat().st_size,
    }


def download_wrds_compustat_us(
    *,
    output_dir: Path = DEFAULT_WRDS_US_DIR,
    start: str = "2000-01-01",
    end: str = "2026-07-08",
    schema: str = "comp",
    username: str | None = None,
    skip_annual: bool = False,
    skip_monthly: bool = False,
    primary_exchange_only: bool = True,
) -> dict[str, Any]:
    """Download WRDS Compustat North America US exports into canonical local files."""
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - external dependency guard
        raise SystemExit("Install first: python -m pip install wrds") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": None,
        "schema": schema,
        "start": _sql_date(start),
        "end": _sql_date(end),
        "primary_exchange_only": primary_exchange_only,
        "outputs": {},
        "status": "started",
    }
    kwargs = {"wrds_username": username} if username else {}
    connection = wrds.Connection(**kwargs)
    try:
        if not skip_annual:
            annual_sql = build_wrds_compustat_us_annual_sql(
                schema=schema,
                start=start,
                end=end,
            )
            annual = connection.raw_sql(annual_sql)
            annual = normalize_wrds_compustat_us_annual(annual)
            manifest["outputs"]["annual"] = _write_csv_gz(
                annual,
                output_dir / DEFAULT_US_ANNUAL_EXPORT,
            )
        if not skip_monthly:
            monthly_sql = build_wrds_compustat_us_monthly_sql(
                schema=schema,
                start=start,
                end=end,
                primary_exchange_only=primary_exchange_only,
            )
            monthly = connection.raw_sql(monthly_sql)
            monthly = normalize_wrds_compustat_us_monthly(monthly)
            manifest["outputs"]["monthly"] = _write_csv_gz(
                monthly,
                output_dir / DEFAULT_US_MONTHLY_EXPORT,
            )
        manifest["status"] = "ok"
    except Exception as exc:  # pragma: no cover - depends on WRDS entitlement
        manifest["status"] = "error"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            connection.close()
        except Exception:
            pass
        (output_dir / "wrds_compustat_us_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
    return manifest
