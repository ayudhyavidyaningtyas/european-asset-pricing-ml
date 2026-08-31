"""Download supplemental LSEG data for the European asset-pricing panel.

This script leaves the completed monthly and annual downloads untouched. It
adds the Refinitiv-dependent data that is useful to preserve while Workspace
access is available:

* security type, primary quote, currency, and listing metadata;
* additional annual fundamentals and accounting availability dates;
* daily returns, prices, volume, bid, and ask in resumable Parquet batches;
* a small daily European market/regime series.

Usage:
    export LSEG_APP_KEY="your-app-key"
    python scripts/refinitiv_supplemental_downloader.py \
        --universe-csv data/raw/asset_pricing/refinitiv_exports/europe_equity_universe_rics_only.csv \
        --resume

The daily panel is intentionally not combined into one file. Read the Parquet
dataset under ``supplemental/daily`` with PyArrow, DuckDB, Polars, or pandas.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

try:
    pd.set_option("future.no_silent_downcasting", True)
except KeyError:  # pragma: no cover - older pandas compatibility
    pass

try:
    import lseg.data as ld
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Install first: python -m pip install lseg-data pyarrow") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_exports"
OUT_DIR = EXPORT_DIR / "supplemental"
BATCH_DIR = OUT_DIR / "_batches"
META_DIR = OUT_DIR / "_metadata"
MANIFEST = OUT_DIR / "refinitiv_supplemental_manifest.json"

PILOT_UNIVERSE = [
    "VOD.L",
    "BP.L",
    "AZN.L",
    "ASML.AS",
    "SAPG.DE",
    "SIEGn.DE",
    "AIR.PA",
    "NESN.S",
    "NOVOb.CO",
]

REFERENCE_FIELDS = [
    "TR.RIC",
    "TR.ISIN",
    "TR.PermID",
    "TR.CommonName",
    "TR.InstrumentType",
    "TR.AssetCategory",
    "TR.AssetType",
    "TR.IsPrimaryInstrument",
    "TR.PrimaryInstrument",
    "TR.IsPrimaryQuote",
    "TR.PrimaryQuote",
    "TR.FirstTradeDate",
    "TR.IsDelistedQuote",
    "TR.RetireDate",
    "DELIST_DAT",
    "TR.Currency",
    "CF_CURR",
    "TR.PriceMoPriceCurrency",
    "TR.CompanyReportCurrency",
    "TR.ExchangeName",
    "TR.ExchangeCountry",
    "TR.HQCountryCode",
]

SUPPLEMENTAL_FUNDAMENTAL_FIELDS = [
    "TR.F.DebtTot",
    "TR.F.CashSTInvst",
    "TR.F.CashCashEquiv",
    "TR.F.TotCurrAssets",
    "TR.F.TotCurrLiab",
    "TR.F.PPENetTot",
    "TR.F.InvntTot",
    "TR.F.TradeAcctTradeNotesRcvblNet",
    "TR.F.SGATot",
    "TR.F.EBITDA",
    "TR.F.IntrExpn",
    "TR.F.IncTax",
    "TR.F.DivPaidCashTotCF",
    "TR.F.IntangTotNet",
    "TR.F.ComShrOutsTot",
    "TR.F.PeriodEndDate",
]

REPORTING_DATE_FIELDS = [
    "TR.ISPeriodEndDate",
    "TR.ISOriginalAnnouncementDate",
    "TR.ISSourceDate",
    "TR.ExpectedReportDate",
]

DAILY_FIELD_CANDIDATES = [
    "TR.PriceClose",
    "TR.TotalReturn1D",
    "TR.Volume",
    "BID",
    "ASK",
]

REGIME_UNIVERSE = {
    ".V2TX": "vstoxx",
    ".STOXX": "stoxx_europe_600",
    ".STOXX50E": "euro_stoxx_50",
    ".FTSE": "ftse_100",
    ".GDAXI": "dax",
    ".FCHI": "cac_40",
    "EUR=": "eur_usd",
    "GBP=": "gbp_usd",
    "CHF=": "chf_usd",
    "DKK=": "dkk_usd",
    "NOK=": "nok_usd",
    "SEK=": "sek_usd",
    "DE2YT=RR": "germany_2y_yield",
    "DE10YT=RR": "germany_10y_yield",
}

DAILY_COLUMN_NAMES = {
    "TR.PRICECLOSE": "price_close",
    "TR.TOTALRETURN1D": "total_return_1d",
    "TR.VOLUME": "volume",
    "BID": "bid",
    "ASK": "ask",
}


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def year_windows(start: str, end: str) -> list[tuple[int, str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end date must not precede start date")
    windows = []
    for year in range(start_ts.year, end_ts.year + 1):
        window_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        window_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        windows.append((year, window_start.date().isoformat(), window_end.date().isoformat()))
    return windows


def safe_field_name(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")


def batch_signature(batch: list[str], **extra: object) -> dict:
    return {"universe": batch, **extra}


def metadata_path(data_path: Path) -> Path:
    relative = data_path.resolve().relative_to(OUT_DIR.resolve())
    return META_DIR / relative.parent / f"{relative.name}.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(path)


def save_parquet_batch(df: pd.DataFrame, path: Path, signature: dict, **metadata: object) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    temporary.replace(path)
    result = {
        "path": str(path),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "bytes": int(path.stat().st_size),
        "signature": signature,
        **metadata,
    }
    atomic_json(metadata_path(path), result)
    return result


def reusable_batch(path: Path, signature: dict) -> dict | None:
    meta_path = metadata_path(path)
    if not path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return metadata if metadata.get("signature") == signature else None


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "__".join(str(part) for part in values if part is not None and str(part) != "nan").strip("_")
            for values in out.columns.to_flat_index()
        ]
    else:
        out.columns = [str(column) for column in out.columns]

    if not isinstance(out.index, pd.RangeIndex):
        name = out.index.name or "date"
        while name in out.columns:
            name = f"{name}_index"
        out = out.reset_index(names=name)
    return out


def find_date_column(df: pd.DataFrame) -> str:
    candidates = [
        column
        for column in df.columns
        if str(column).lower() in {"date", "date_index", "datetime", "timestamp"}
        or str(column).lower().endswith("_date")
    ]
    scores = []
    for column in candidates:
        parsed = pd.to_datetime(df[column], errors="coerce")
        scores.append((int(parsed.notna().sum()), column))
    if not scores or max(scores)[0] == 0:
        raise ValueError(f"history response has no parseable date column: {list(df.columns)}")
    return max(scores)[1]


def history_to_tidy(df: pd.DataFrame, requested_universe: list[str]) -> pd.DataFrame:
    wide = clean_frame(df)
    if wide.empty:
        return pd.DataFrame(columns=["date", "ric"])

    date_column = find_date_column(wide)
    wide[date_column] = pd.to_datetime(wide[date_column], errors="coerce")
    value_columns = [column for column in wide.columns if column != date_column]
    pieces = []
    for column in value_columns:
        if "__" in column:
            ric, field = column.split("__", 1)
        elif len(requested_universe) == 1:
            ric, field = requested_universe[0], column
        else:
            continue
        pieces.append(
            wide[[date_column, column]]
            .rename(columns={date_column: "date", column: "value"})
            .assign(ric=ric, field=str(field).upper())
        )
    if not pieces:
        return pd.DataFrame(columns=["date", "ric"])

    long = pd.concat(pieces, ignore_index=True)
    panel = (
        long.pivot_table(index=["date", "ric"], columns="field", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(columns=DAILY_COLUMN_NAMES)
        .sort_values(["ric", "date"])
    )
    return panel


def is_identifier_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        text in message
        for text in [
            "unable to resolve",
            "requested identifiers",
            "unable to collect data for the field",
            "some specific identifier",
            "invalid universe",
        ]
    )


def run_with_retries(
    label: str,
    function: Callable[[], pd.DataFrame],
    max_retries: int,
    retry_sleep: float,
) -> pd.DataFrame:
    for attempt in range(max_retries + 1):
        try:
            return function()
        except Exception as exc:
            if is_identifier_error(exc) or attempt >= max_retries:
                raise
            wait = retry_sleep * (attempt + 1)
            print(
                f"{label}: {type(exc).__name__}: {exc}; retrying in {wait:.0f}s "
                f"({attempt + 1}/{max_retries})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


def probe_get_data_fields(
    fields: list[str],
    universe: list[str],
    parameters: dict | None,
    max_retries: int,
    retry_sleep: float,
) -> tuple[list[str], dict[str, str]]:
    available = []
    errors = {}
    for field in fields:
        try:
            frame = run_with_retries(
                f"probe {field}",
                lambda field=field: ld.get_data(
                    universe=universe,
                    fields=[field],
                    parameters=parameters,
                    header_type=ld.HeaderType.NAME,
                ),
                max_retries,
                retry_sleep,
            )
            if frame.shape[1] <= 1:
                raise ValueError("field returned no value column")
            available.append(field)
            print(f"field available: {field}", flush=True)
        except Exception as exc:
            errors[field] = f"{type(exc).__name__}: {exc}"
            print(f"field unavailable: {field}: {exc}", file=sys.stderr, flush=True)
    return available, errors


def probe_daily_fields(
    fields: list[str],
    universe: list[str],
    max_retries: int,
    retry_sleep: float,
) -> tuple[list[str], dict[str, str]]:
    available = []
    errors = {}
    probe_start = "2025-01-01"
    probe_end = "2025-03-31"
    for field in fields:
        try:
            frame = run_with_retries(
                f"probe {field}",
                lambda field=field: ld.get_history(
                    universe=universe[:2],
                    fields=[field],
                    interval="daily",
                    start=probe_start,
                    end=probe_end,
                    header_type=ld.HeaderType.NAME,
                ),
                max_retries,
                retry_sleep,
            )
            if frame.empty:
                raise ValueError("field returned no observations")
            available.append(field)
            print(f"daily field available: {field}", flush=True)
        except Exception as exc:
            errors[field] = f"{type(exc).__name__}: {exc}"
            print(f"daily field unavailable: {field}: {exc}", file=sys.stderr, flush=True)

    if available:
        try:
            run_with_retries(
                "probe combined daily fields",
                lambda: ld.get_history(
                    universe=universe[:2],
                    fields=available,
                    interval="daily",
                    start=probe_start,
                    end=probe_end,
                    header_type=ld.HeaderType.NAME,
                ),
                max_retries,
                retry_sleep,
            )
        except Exception as exc:
            core = [field for field in available if field.startswith("TR.")]
            for field in set(available) - set(core):
                errors[field] = f"excluded because combined request failed: {type(exc).__name__}: {exc}"
            available = core
    return available, errors


def tolerant_get_data(
    batch: list[str],
    fields: list[str],
    parameters: dict | None,
    label: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    try:
        frame = run_with_retries(
            label,
            lambda: ld.get_data(
                universe=batch,
                fields=fields,
                parameters=parameters,
                header_type=ld.HeaderType.NAME,
            ),
            max_retries,
            retry_sleep,
        )
        return clean_frame(frame), [], {}
    except Exception as exc:
        if not is_identifier_error(exc):
            raise
        if len(batch) == 1:
            return pd.DataFrame(), batch, {batch[0]: f"{type(exc).__name__}: {exc}"}

    middle = len(batch) // 2
    left = tolerant_get_data(
        batch[:middle], fields, parameters, f"{label} left", max_retries, retry_sleep
    )
    right = tolerant_get_data(
        batch[middle:], fields, parameters, f"{label} right", max_retries, retry_sleep
    )
    frames = [frame for frame in [left[0], right[0]] if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, [*left[1], *right[1]], {**left[2], **right[2]}


def tolerant_get_history(
    batch: list[str],
    fields: list[str],
    start: str,
    end: str,
    label: str,
    max_retries: int,
    retry_sleep: float,
    interval: str = "daily",
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    try:
        history = run_with_retries(
            label,
            lambda: ld.get_history(
                universe=batch,
                fields=fields,
                interval=interval,
                start=start,
                end=end,
                header_type=ld.HeaderType.NAME,
            ),
            max_retries,
            retry_sleep,
        )
        return history_to_tidy(history, batch), [], {}
    except Exception as exc:
        if not is_identifier_error(exc):
            raise
        if len(batch) == 1:
            return (
                pd.DataFrame(columns=["date", "ric"]),
                batch,
                {batch[0]: f"{type(exc).__name__}: {exc}"},
            )

    middle = len(batch) // 2
    left = tolerant_get_history(
        batch[:middle],
        fields,
        start,
        end,
        f"{label} left",
        max_retries,
        retry_sleep,
        interval,
    )
    right = tolerant_get_history(
        batch[middle:],
        fields,
        start,
        end,
        f"{label} right",
        max_retries,
        retry_sleep,
        interval,
    )
    frames = [frame for frame in [left[0], right[0]] if not frame.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ric"])
    return combined, [*left[1], *right[1]], {**left[2], **right[2]}


def combine_parquet_batches(paths: list[Path], output: Path) -> dict:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = combined.replace("", pd.NA)
    for column in combined.columns:
        upper = str(column).upper()
        if upper.startswith("TR.F.") and upper.endswith(".DATE"):
            combined[column] = pd.to_datetime(combined[column], errors="coerce")
        elif upper.startswith("TR.F.") and upper != "TR.F.PERIODENDDATE":
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
        elif (
            upper == "TR.F.PERIODENDDATE"
            or upper.startswith("TR.IS")
            and upper.endswith("DATE")
            or upper == "TR.EXPECTEDREPORTDATE"
        ):
            combined[column] = pd.to_datetime(combined[column], errors="coerce")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False, engine="pyarrow", compression="zstd")
    return {
        "path": str(output),
        "rows": int(combined.shape[0]),
        "cols": int(combined.shape[1]),
        "bytes": int(output.stat().st_size),
    }


def download_tabular_batches(
    component: str,
    universe: list[str],
    fields: list[str],
    parameters: dict | None,
    batch_size: int,
    resume: bool,
    max_retries: int,
    retry_sleep: float,
) -> tuple[list[dict], list[Path]]:
    component_dir = BATCH_DIR / component
    records = []
    paths = []
    all_batches = list(chunks(universe, batch_size))
    for batch_no, batch in enumerate(all_batches, start=1):
        path = component_dir / f"batch_{batch_no:04d}.parquet"
        signature = batch_signature(batch, fields=fields, parameters=parameters)
        reused = reusable_batch(path, signature) if resume else None
        if reused:
            metadata = reused
            status = "reused"
        else:
            frame, failed, errors = tolerant_get_data(
                batch,
                fields,
                parameters,
                f"{component} batch {batch_no}",
                max_retries,
                retry_sleep,
            )
            metadata = save_parquet_batch(
                frame,
                path,
                signature,
                failed_rics=failed,
                errors=errors,
            )
            status = "downloaded" if not failed else ("partial" if not frame.empty else "failed_all")
        records.append(
            {
                "batch_no": batch_no,
                "status": status,
                "n_instruments": len(batch),
                "rows": metadata["rows"],
                "bytes": metadata["bytes"],
                "failed_rics_n": len(metadata.get("failed_rics", [])),
                "path": str(path),
            }
        )
        paths.append(path)
        print(
            f"{component} {batch_no}/{len(all_batches)}: {status}; "
            f"rows={metadata['rows']}; failed={len(metadata.get('failed_rics', []))}",
            flush=True,
        )
    pd.DataFrame(records).to_csv(OUT_DIR / f"{component}_batch_index.csv", index=False)
    return records, paths


def download_daily(
    universe: list[str],
    fields: list[str],
    start: str,
    end: str,
    batch_size: int,
    resume: bool,
    max_retries: int,
    retry_sleep: float,
) -> dict:
    records = []
    universe_batches = list(chunks(universe, batch_size))
    windows = year_windows(start, end)
    total = len(universe_batches) * len(windows)
    job = 0
    index_path = OUT_DIR / "daily_batch_index.csv"

    for year, window_start, window_end in windows:
        for batch_no, batch in enumerate(universe_batches, start=1):
            job += 1
            path = OUT_DIR / "daily" / f"year={year}" / f"batch_{batch_no:04d}.parquet"
            signature = batch_signature(
                batch,
                fields=fields,
                start=window_start,
                end=window_end,
                interval="daily",
            )
            reused = reusable_batch(path, signature) if resume else None
            if reused:
                metadata = reused
                status = "reused"
            else:
                frame, failed, errors = tolerant_get_history(
                    batch,
                    fields,
                    window_start,
                    window_end,
                    f"daily {year} batch {batch_no}",
                    max_retries,
                    retry_sleep,
                )
                metadata = save_parquet_batch(
                    frame,
                    path,
                    signature,
                    failed_rics=failed,
                    errors=errors,
                )
                status = "downloaded" if not failed else ("partial" if not frame.empty else "failed_all")

            records.append(
                {
                    "job": job,
                    "year": year,
                    "batch_no": batch_no,
                    "status": status,
                    "n_instruments": len(batch),
                    "rows": metadata["rows"],
                    "bytes": metadata["bytes"],
                    "failed_rics_n": len(metadata.get("failed_rics", [])),
                    "path": str(path),
                }
            )
            if job % 25 == 0:
                pd.DataFrame(records).to_csv(index_path, index=False)
            print(
                f"daily {job}/{total}: year={year} batch={batch_no}; {status}; "
                f"rows={metadata['rows']}; failed={len(metadata.get('failed_rics', []))}",
                flush=True,
            )

    index = pd.DataFrame(records)
    index.to_csv(index_path, index=False)
    return {
        "dataset": str(OUT_DIR / "daily"),
        "batch_index": str(index_path),
        "jobs": int(len(index)),
        "rows": int(index["rows"].sum()) if not index.empty else 0,
        "bytes": int(index["bytes"].sum()) if not index.empty else 0,
        "failed_identifier_jobs": int(index["failed_rics_n"].sum()) if not index.empty else 0,
        "fields": fields,
    }


def clean_regime_history(df: pd.DataFrame, ric: str, name: str) -> pd.DataFrame:
    frame = clean_frame(df)
    if frame.empty:
        return pd.DataFrame(columns=["date", "ric", "series"])
    date_column = find_date_column(frame)
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame.insert(1, "ric", ric)
    frame.insert(2, "series", name)
    rename = {}
    used = {"date", "ric", "series"}
    for column in frame.columns[3:]:
        candidate = safe_field_name(column)
        suffix = 2
        while candidate in used:
            candidate = f"{safe_field_name(column)}_{suffix}"
            suffix += 1
        used.add(candidate)
        rename[column] = candidate
    return frame.rename(columns=rename)


def download_regime(
    start: str,
    end: str,
    resume: bool,
    max_retries: int,
    retry_sleep: float,
) -> dict:
    paths = []
    records = []
    for ric, name in REGIME_UNIVERSE.items():
        for year, window_start, window_end in year_windows(start, end):
            path = BATCH_DIR / "regime" / safe_field_name(ric) / f"{year}.parquet"
            signature = batch_signature(
                [ric], fields="all_history_fields", start=window_start, end=window_end
            )
            reused = reusable_batch(path, signature) if resume else None
            if reused:
                metadata = reused
                status = "reused"
            else:
                try:
                    raw = run_with_retries(
                        f"regime {ric} {year}",
                        lambda ric=ric, window_start=window_start, window_end=window_end: ld.get_history(
                            universe=[ric],
                            interval="daily",
                            start=window_start,
                            end=window_end,
                            header_type=ld.HeaderType.NAME,
                        ),
                        max_retries,
                        retry_sleep,
                    )
                    frame = clean_regime_history(raw, ric, name)
                    metadata = save_parquet_batch(frame, path, signature, error=None)
                    status = "downloaded"
                except Exception as exc:
                    frame = pd.DataFrame(columns=["date", "ric", "series"])
                    metadata = save_parquet_batch(
                        frame,
                        path,
                        signature,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    status = "failed"
            paths.append(path)
            records.append(
                {
                    "ric": ric,
                    "series": name,
                    "year": year,
                    "status": status,
                    "rows": metadata["rows"],
                    "bytes": metadata["bytes"],
                    "error": metadata.get("error"),
                    "path": str(path),
                }
            )
            print(f"regime {ric} {year}: {status}; rows={metadata['rows']}", flush=True)

    index_path = OUT_DIR / "regime_batch_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    final = combine_parquet_batches(paths, OUT_DIR / "refinitiv_regime_daily.parquet")
    final["batch_index"] = str(index_path)
    return final


def expand_fundamental_fields(fields: list[str]) -> list[str]:
    expanded = []
    for field in fields:
        expanded.append(field)
        if field.startswith("TR.F.") and field != "TR.F.PeriodEndDate":
            expanded.append(f"{field}.date")
    return expanded


def open_desktop_session(app_key: str):
    return ld.open_session(name="desktop.workspace", app_key=app_key)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-csv", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--start-fy", default="FY2000")
    parser.add_argument("--end-fy", default="FY2025")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--skip-daily", action="store_true")
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--reference-batch-size", type=int, default=250)
    parser.add_argument("--fundamental-batch-size", type=int, default=25)
    parser.add_argument("--daily-batch-size", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=15.0)
    args = parser.parse_args()

    app_key = os.environ.get("LSEG_APP_KEY")
    if not app_key:
        raise SystemExit("Set LSEG_APP_KEY in this terminal before running the downloader.")

    if args.pilot:
        universe = PILOT_UNIVERSE
    elif args.universe_csv:
        if not args.universe_csv.exists():
            raise SystemExit(f"Universe CSV not found: {args.universe_csv}")
        frame = pd.read_csv(args.universe_csv)
        if "ric" not in frame.columns:
            raise SystemExit("Universe CSV must contain a column named 'ric'.")
        universe = frame["ric"].dropna().astype(str).str.strip().drop_duplicates().tolist()
    else:
        raise SystemExit("Pass --universe-csv PATH or --pilot.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "started",
        "universe_size": len(universe),
        "start": args.start,
        "end": args.end,
        "start_fy": args.start_fy,
        "end_fy": args.end_fy,
        "outputs": {},
        "field_probes": {},
    }

    try:
        open_desktop_session(app_key)
        probe_universe = [ric for ric in PILOT_UNIVERSE if ric in set(universe)] or PILOT_UNIVERSE

        if not args.skip_reference:
            fields, errors = probe_get_data_fields(
                REFERENCE_FIELDS,
                probe_universe[:5],
                None,
                args.max_retries,
                args.retry_sleep,
            )
            manifest["field_probes"]["reference"] = {"available": fields, "errors": errors}
            records, paths = download_tabular_batches(
                "reference",
                universe,
                fields,
                None,
                args.reference_batch_size,
                args.resume,
                args.max_retries,
                args.retry_sleep,
            )
            output = combine_parquet_batches(paths, OUT_DIR / "refinitiv_security_master_supplement.parquet")
            output["batches"] = len(records)
            manifest["outputs"]["reference"] = output

        if not args.skip_fundamentals:
            candidates = [*SUPPLEMENTAL_FUNDAMENTAL_FIELDS, *REPORTING_DATE_FIELDS]
            probe_parameters = {"SDate": "FY2023", "EDate": "FY2025", "Frq": "FY"}
            base_fields, base_errors = probe_get_data_fields(
                candidates,
                probe_universe[:5],
                probe_parameters,
                args.max_retries,
                args.retry_sleep,
            )
            expanded_candidates = expand_fundamental_fields(base_fields)
            expanded_fields, property_errors = probe_get_data_fields(
                expanded_candidates,
                probe_universe[:5],
                probe_parameters,
                args.max_retries,
                args.retry_sleep,
            )
            manifest["field_probes"]["fundamentals"] = {
                "available": expanded_fields,
                "errors": {**base_errors, **property_errors},
            }
            parameters = {"SDate": args.start_fy, "EDate": args.end_fy, "Frq": "FY"}
            records, paths = download_tabular_batches(
                "fundamentals_supplement",
                universe,
                expanded_fields,
                parameters,
                args.fundamental_batch_size,
                args.resume,
                args.max_retries,
                args.retry_sleep,
            )
            output = combine_parquet_batches(
                paths, OUT_DIR / "refinitiv_fundamentals_supplement_annual.parquet"
            )
            output["batches"] = len(records)
            output["fields"] = expanded_fields
            manifest["outputs"]["fundamentals"] = output

        if not args.skip_regime:
            manifest["outputs"]["regime"] = download_regime(
                args.start,
                args.end,
                args.resume,
                args.max_retries,
                args.retry_sleep,
            )

        if not args.skip_daily:
            fields, errors = probe_daily_fields(
                DAILY_FIELD_CANDIDATES,
                probe_universe,
                args.max_retries,
                args.retry_sleep,
            )
            if not fields:
                raise RuntimeError("No daily fields passed the Workspace probe.")
            manifest["field_probes"]["daily"] = {"available": fields, "errors": errors}
            manifest["outputs"]["daily"] = download_daily(
                universe,
                fields,
                args.start,
                args.end,
                args.daily_batch_size,
                args.resume,
                args.max_retries,
                args.retry_sleep,
            )

        manifest["status"] = "ok"
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return 0
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        print(f"Supplemental LSEG download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
        atomic_json(MANIFEST, manifest)
        print(f"manifest -> {MANIFEST}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
