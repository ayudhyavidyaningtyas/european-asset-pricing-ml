"""Download a pilot European asset-pricing panel via LSEG Workspace desktop API.

Usage:
    export LSEG_APP_KEY="your-app-key"
    python scripts/refinitiv_python_downloader.py --pilot

This script deliberately starts with a small pilot universe. Scale only after the
pilot files import and link cleanly. It requires:
    - Refinitiv/LSEG Workspace running on the same machine;
    - an App Key generated inside Workspace;
    - desktop API entitlement/proxy access.

Do not commit app keys. Keep the key in an environment variable or local shell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    pd.set_option("future.no_silent_downcasting", True)
except KeyError:  # pragma: no cover - older pandas compatibility
    pass

warnings.filterwarnings(
    "ignore",
    message=r"Downcasting behavior in `replace` is deprecated.*",
    category=FutureWarning,
)

try:
    import lseg.data as ld
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Install first: python -m pip install lseg-data") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_exports"
MANIFEST = OUT_DIR / "python_lseg_download_manifest.json"
BATCH_DIR = OUT_DIR / "_batches"
MONTHLY_BATCH_DIR = BATCH_DIR / "monthly"
FUNDAMENTAL_BATCH_DIR = BATCH_DIR / "fundamentals"

# Common numeraire. The universe spans many listing currencies (GBP/SEK/CHF/NOK/EUR/...),
# so TR.TotalReturn1Mo / TR.CompanyMarketCap must be converted server-side to one currency,
# and accounting values must match market cap for valid valuation ratios.
BASE_CURRENCY = "EUR"

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
    "TR.CUSIP",
    "TR.TickerSymbol",
    "TR.CommonName",
    "TR.ExchangeName",
    "TR.ExchangeCountry",
    "TR.PriceMoPriceCurrency",     # listing/price currency (TR.Currency is not resolvable)
    "TR.CompanyReportCurrency",    # fundamentals reporting currency
    "TR.TRBCEconomicSector",
    "TR.TRBCBusinessSector",
    "TR.TRBCIndustryGroup",
    "TR.TRBCIndustry",
    "TR.GICSSector",
    "TR.GICSIndustry",
]

MONTHLY_FIELDS = [
    "TR.PriceClose",
    "TR.TotalReturn1Mo",
    "TR.Volume",
    "TR.CompanyMarketCap",
    "TR.TtlCmnSharesOut",
]

# These are common LSEG/Workspace financial statement fields. If a field fails under
# your entitlement, use Workspace Data Item Browser to find the exact equivalent.
FUNDAMENTAL_FIELDS = [
    "TR.F.TotAssets",
    "TR.F.TotLiab",
    "TR.F.TotShHoldEq",
    "TR.F.TotRevenue",
    "TR.F.OpProfBefNonRecurIncExpn",
    "TR.F.IncBefDiscOpsExordItems",
    "TR.F.NetCashFlowOp",
    "TR.F.CAPEXTot",
]

MONTHLY_PANEL_COLUMNS = [
    "date",
    "ric",
    "price_close",
    "total_return_1m",
    "volume",
    "company_market_cap",
    "shares_outstanding",
]


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def clean_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Convert LSEG DataFrames to plain CSV-friendly columns.

    `get_history` often returns a DatetimeIndex plus MultiIndex columns
    `(instrument, field)`. Raw `to_csv` preserves that as multiple header rows,
    which is hard to reload. This flattens it to e.g. `VOD.L__TR.PRICECLOSE`
    and moves the date index into a normal `date` column.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "__".join(str(part) for part in tup if part is not None and str(part) != "nan").strip("_")
            for tup in out.columns.to_flat_index()
        ]
    else:
        out.columns = [str(col) for col in out.columns]

    if not isinstance(out.index, pd.RangeIndex):
        index_name = out.index.name or "date"
        while index_name in out.columns:
            index_name = f"{index_name}_index"
        out = out.reset_index(names=index_name)
    if "date_index" in out.columns and "date" in out.columns:
        out = out.drop(columns=["date_index"])
    return out


def save_frame(df: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_for_csv(df)
    clean.to_csv(path, index=False)
    return {"path": str(path), "rows": int(clean.shape[0]), "cols": int(clean.shape[1]), "bytes": path.stat().st_size}


def existing_frame_meta(path: Path) -> dict:
    return {
        "path": str(path),
        "rows": int(sum(1 for _ in path.open()) - 1),
        "cols": int(pd.read_csv(path, nrows=0).shape[1]),
        "bytes": path.stat().st_size,
        "reused": True,
    }


def batch_token(batch: list[str]) -> str:
    return ",".join(batch)


def batch_file_matches(
    path: Path,
    batch: list[str],
    expected_currency: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        sample = pd.read_csv(path, nrows=1)
    except Exception:
        return False
    batch_matches = (
        "download_batch" in sample.columns
        and not sample.empty
        and sample["download_batch"].iloc[0] == batch_token(batch)
    )
    if not batch_matches or expected_currency is None:
        return batch_matches
    return (
        "download_currency" in sample.columns
        and sample["download_currency"].iloc[0] == expected_currency
    )


def failure_path_for(path: Path) -> Path:
    return path.with_suffix(".failures.json")


def load_failure_marker(path: Path, batch: list[str]) -> dict | None:
    marker = failure_path_for(path)
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text())
    except Exception:
        return None
    return payload if payload.get("download_batch") == batch_token(batch) else None


def save_failure_marker(path: Path, batch: list[str], failed_rics: list[str], errors: dict[str, str]) -> None:
    failure_path_for(path).write_text(
        json.dumps(
            {
                "download_batch": batch_token(batch),
                "failed_rics": failed_rics,
                "errors": errors,
            },
            indent=2,
        )
    )


def clear_failure_marker(path: Path) -> None:
    marker = failure_path_for(path)
    if marker.exists():
        marker.unlink()


def run_with_retries(label: str, max_retries: int, retry_sleep: float, func):
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:
            if is_identifier_resolution_error(exc) or attempt >= max_retries:
                raise
            wait = retry_sleep * (attempt + 1)
            print(
                f"{label}: {type(exc).__name__}: {exc}; retrying in {wait:.0f}s "
                f"({attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)


def is_identifier_resolution_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "unable to resolve" in message
        or "requested identifiers" in message
        or "unable to collect data for the field" in message
        or "some specific identifier" in message
    )


def find_date_column(df: pd.DataFrame) -> str:
    candidates = [
        col for col in df.columns
        if str(col).lower() in {"date", "date_index", "datetime", "timestamp"}
        or str(col).lower().endswith("_date")
    ]
    scored = []
    for col in candidates:
        parsed = pd.to_datetime(df[col], errors="coerce")
        scored.append((int(parsed.notna().sum()), col))
    if not scored:
        raise ValueError("monthly history has no parseable date column after flattening")
    valid, col = max(scored)
    if valid == 0:
        raise ValueError(f"monthly history date candidates are not parseable: {candidates}")
    return col


def tidy_monthly_panel(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return one row per instrument-date from flattened LSEG monthly history."""
    wide = clean_for_csv(monthly)
    value_cols = [c for c in wide.columns if "__TR." in c]
    if wide.empty or not value_cols:
        return pd.DataFrame(columns=MONTHLY_PANEL_COLUMNS)

    date_col = find_date_column(wide)
    wide[date_col] = pd.to_datetime(wide[date_col], errors="coerce")
    wide["month_end"] = wide[date_col].dt.to_period("M").dt.to_timestamp("M")
    # Collapse duplicate rows within each calendar month by carrying the last non-null observation.
    collapsed = wide[["month_end", *value_cols]].groupby("month_end", as_index=False).last()
    rows = []
    for col in value_cols:
        ric, field = col.split("__", 1)
        rows.append(collapsed[["month_end", col]].rename(columns={"month_end": "date", col: "value"}).assign(ric=ric, field=field))
    long = pd.concat(rows, ignore_index=True)
    panel = (
        long.pivot_table(index=["date", "ric"], columns="field", values="value", aggfunc="last")
        .reset_index()
        .rename_axis(None, axis=1)
        .rename(
            columns={
                "TR.PRICECLOSE": "price_close",
                "TR.TOTALRETURN1MO": "total_return_1m",
                "TR.VOLUME": "volume",
                "TR.COMPANYMARKETCAP": "company_market_cap",
                "TR.TTLCMNSHARESOUT": "shares_outstanding",
            }
        )
        .sort_values(["ric", "date"])
    )
    return panel


def open_desktop_session(app_key: str):
    # Desktop session uses the running Workspace app and local proxy.
    return ld.open_session(name="desktop.workspace", app_key=app_key)


def download_reference(universe: list[str]) -> pd.DataFrame:
    frames = []
    for batch in chunks(universe, 500):
        df = ld.get_data(universe=batch, fields=REFERENCE_FIELDS, header_type=ld.HeaderType.NAME)
        df["download_batch"] = ",".join(batch)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def download_monthly(universe: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for batch in chunks(universe, 25):
        df = ld.get_history(
            universe=batch,
            fields=MONTHLY_FIELDS,
            interval="monthly",
            start=start,
            end=end,
            parameters={"Curn": BASE_CURRENCY},
            header_type=ld.HeaderType.NAME,
        )
        df["download_batch"] = ",".join(batch)
        df["download_currency"] = BASE_CURRENCY
        frames.append(df)
    return pd.concat(frames) if frames else pd.DataFrame()


def combine_csvs(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(path, low_memory=False) for path in paths), ignore_index=True)


def download_monthly_batches(
    universe: list[str],
    start: str,
    end: str,
    batch_size: int,
    resume: bool,
    max_retries: int,
    retry_sleep: float,
) -> tuple[pd.DataFrame, dict]:
    MONTHLY_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    tidy_paths = []

    for batch_no, batch in enumerate(chunks(universe, batch_size), start=1):
        raw_path = MONTHLY_BATCH_DIR / f"monthly_raw_batch_{batch_no:04d}.csv"
        tidy_path = MONTHLY_BATCH_DIR / f"monthly_tidy_batch_{batch_no:04d}.csv"
        label = f"monthly batch {batch_no} ({batch[0]}..{batch[-1]}, n={len(batch)})"

        if (
            resume
            and batch_file_matches(raw_path, batch, BASE_CURRENCY)
            and batch_file_matches(tidy_path, batch, BASE_CURRENCY)
        ):
            status = "reused"
            raw_meta = existing_frame_meta(raw_path)
            tidy_meta = existing_frame_meta(tidy_path)
        else:
            df = run_with_retries(
                label,
                max_retries,
                retry_sleep,
                lambda batch=batch: ld.get_history(
                    universe=batch,
                    fields=MONTHLY_FIELDS,
                    interval="monthly",
                    start=start,
                    end=end,
                    parameters={"Curn": BASE_CURRENCY},
                    header_type=ld.HeaderType.NAME,
                ),
            )
            df["download_batch"] = batch_token(batch)
            df["download_currency"] = BASE_CURRENCY
            raw_meta = save_frame(df, raw_path)

            tidy = tidy_monthly_panel(df)
            tidy["download_batch"] = batch_token(batch)
            tidy["download_currency"] = BASE_CURRENCY
            tidy_meta = save_frame(tidy, tidy_path)
            status = "downloaded"

        records.append(
            {
                "batch_no": batch_no,
                "status": status,
                "n_instruments": len(batch),
                "first_ric": batch[0],
                "last_ric": batch[-1],
                "raw_path": str(raw_path),
                "raw_rows": raw_meta["rows"],
                "raw_cols": raw_meta["cols"],
                "raw_bytes": raw_meta["bytes"],
                "tidy_path": str(tidy_path),
                "tidy_rows": tidy_meta["rows"],
                "tidy_cols": tidy_meta["cols"],
                "tidy_bytes": tidy_meta["bytes"],
            }
        )
        tidy_paths.append(tidy_path)
        print(f"{label}: {status}; tidy rows={tidy_meta['rows']}", flush=True)

    index = pd.DataFrame(records)
    index_meta = save_frame(index, OUT_DIR / "refinitiv_monthly_batch_index.csv")
    panel = combine_csvs(tidy_paths)
    if {"ric", "date"}.issubset(panel.columns):
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel = panel.sort_values(["ric", "date"])
    return panel, {"batch_index": index_meta, "batches": records}


def download_fundamentals(universe: list[str], start_fy: str, end_fy: str) -> pd.DataFrame:
    fields = []
    for field in FUNDAMENTAL_FIELDS:
        fields.extend([field, f"{field}.date"])
    frames = []
    for batch in chunks(universe, 25):
        df = ld.get_data(
            universe=batch,
            fields=fields,
            parameters={
                "SDate": start_fy,
                "EDate": end_fy,
                "Frq": "FY",
                "Curn": BASE_CURRENCY,
            },
            header_type=ld.HeaderType.NAME,
        )
        df["download_batch"] = ",".join(batch)
        df["download_currency"] = BASE_CURRENCY
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_fundamentals_once(batch: list[str], fields: list[str], start_fy: str, end_fy: str) -> pd.DataFrame:
    return ld.get_data(
        universe=batch,
        fields=fields,
        parameters={
            "SDate": start_fy,
            "EDate": end_fy,
            "Frq": "FY",
            "Curn": BASE_CURRENCY,
        },
        header_type=ld.HeaderType.NAME,
    )


def download_fundamental_batch_tolerant(
    batch: list[str],
    fields: list[str],
    start_fy: str,
    end_fy: str,
    label: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    try:
        df = run_with_retries(
            label,
            max_retries,
            retry_sleep,
            lambda: get_fundamentals_once(batch, fields, start_fy, end_fy),
        )
        return df, [], {}
    except Exception as exc:
        if not is_identifier_resolution_error(exc):
            raise
        if len(batch) == 1:
            return pd.DataFrame(), batch, {batch[0]: f"{type(exc).__name__}: {exc}"}

    midpoint = len(batch) // 2
    left, right = batch[:midpoint], batch[midpoint:]
    left_df, left_failed, left_errors = download_fundamental_batch_tolerant(
        left,
        fields,
        start_fy,
        end_fy,
        f"{label} left",
        max_retries,
        retry_sleep,
    )
    right_df, right_failed, right_errors = download_fundamental_batch_tolerant(
        right,
        fields,
        start_fy,
        end_fy,
        f"{label} right",
        max_retries,
        retry_sleep,
    )
    frames = [df for df in [left_df, right_df] if not df.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, [*left_failed, *right_failed], {**left_errors, **right_errors}


def download_fundamental_batches(
    universe: list[str],
    start_fy: str,
    end_fy: str,
    batch_size: int,
    resume: bool,
    max_retries: int,
    retry_sleep: float,
) -> tuple[pd.DataFrame, dict]:
    FUNDAMENTAL_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    fields = []
    for field in FUNDAMENTAL_FIELDS:
        fields.extend([field, f"{field}.date"])

    records = []
    paths = []
    for batch_no, batch in enumerate(chunks(universe, batch_size), start=1):
        path = FUNDAMENTAL_BATCH_DIR / f"fundamentals_batch_{batch_no:04d}.csv"
        label = f"fundamentals batch {batch_no} ({batch[0]}..{batch[-1]}, n={len(batch)})"
        failed_rics: list[str] = []
        errors: dict[str, str] = {}

        if resume and batch_file_matches(path, batch, BASE_CURRENCY):
            status = "reused"
            meta = existing_frame_meta(path)
            marker = load_failure_marker(path, batch)
            if marker:
                failed_rics = marker.get("failed_rics", [])
                errors = marker.get("errors", {})
                status = "reused_partial" if meta["rows"] else "reused_failed_all"
        elif resume and (marker := load_failure_marker(path, batch)):
            status = "reused_failed_all"
            meta = existing_frame_meta(path) if path.exists() else save_frame(pd.DataFrame(columns=["download_batch"]), path)
            failed_rics = marker.get("failed_rics", [])
            errors = marker.get("errors", {})
        else:
            df, failed_rics, errors = download_fundamental_batch_tolerant(
                batch,
                fields,
                start_fy,
                end_fy,
                label,
                max_retries,
                retry_sleep,
            )
            df["download_batch"] = batch_token(batch)
            df["download_currency"] = BASE_CURRENCY
            meta = save_frame(df, path)
            if failed_rics:
                save_failure_marker(path, batch, failed_rics, errors)
                status = "partial" if meta["rows"] else "failed_all"
            else:
                clear_failure_marker(path)
                status = "downloaded"

        records.append(
            {
                "batch_no": batch_no,
                "status": status,
                "n_instruments": len(batch),
                "first_ric": batch[0],
                "last_ric": batch[-1],
                "path": str(path),
                "rows": meta["rows"],
                "cols": meta["cols"],
                "bytes": meta["bytes"],
                "failed_rics_n": len(failed_rics),
                "failed_rics": ",".join(failed_rics),
            }
        )
        paths.append(path)
        print(f"{label}: {status}; rows={meta['rows']}; failed={len(failed_rics)}", flush=True)

    index = pd.DataFrame(records)
    index_meta = save_frame(index, OUT_DIR / "refinitiv_fundamentals_batch_index.csv")
    fundamentals = combine_csvs(paths)
    return fundamentals, {"batch_index": index_meta, "batches": records}


def main() -> int:
    global OUT_DIR, MANIFEST, BATCH_DIR, MONTHLY_BATCH_DIR, FUNDAMENTAL_BATCH_DIR, BASE_CURRENCY

    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Use built-in small pilot universe.")
    parser.add_argument("--universe-csv", type=Path, help="CSV with a 'ric' column.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2026-07-08")
    parser.add_argument("--start-fy", default="FY2000")
    parser.add_argument("--end-fy", default="FY2025")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Raw Refinitiv output directory. Defaults to the European export folder.",
    )
    parser.add_argument(
        "--base-currency",
        default=BASE_CURRENCY,
        help="Server-side currency conversion for market data and fundamentals.",
    )
    parser.add_argument("--skip-monthly", action="store_true")
    parser.add_argument("--skip-fundamentals", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed batch files and existing universe master.")
    parser.add_argument("--monthly-batch-size", type=int, default=10)
    parser.add_argument("--fundamental-batch-size", type=int, default=25)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    args = parser.parse_args()

    OUT_DIR = args.output_dir
    MANIFEST = OUT_DIR / "python_lseg_download_manifest.json"
    BATCH_DIR = OUT_DIR / "_batches"
    MONTHLY_BATCH_DIR = BATCH_DIR / "monthly"
    FUNDAMENTAL_BATCH_DIR = BATCH_DIR / "fundamentals"
    BASE_CURRENCY = str(args.base_currency).upper()

    app_key = os.environ.get("LSEG_APP_KEY")
    if not app_key:
        raise SystemExit("Set your app key first: export LSEG_APP_KEY='...'")

    if args.pilot:
        universe = PILOT_UNIVERSE
    elif args.universe_csv:
        if not args.universe_csv.exists():
            raise SystemExit(
                f"Universe CSV not found: {args.universe_csv}\n"
                "Create a CSV with a column named 'ric', then pass its real path. "
                "Example: python scripts/refinitiv_python_downloader.py "
                "--universe-csv data/raw/asset_pricing/refinitiv_exports/my_universe.csv"
            )
        universe = pd.read_csv(args.universe_csv)["ric"].dropna().astype(str).tolist()
    else:
        raise SystemExit("Use --pilot or --universe-csv path/to/file.csv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": None,
        "base_currency": BASE_CURRENCY,
        "universe_n": len(universe),
        "universe": universe,
        "status": "started",
        "outputs": {},
    }

    try:
        open_desktop_session(app_key)
        ref_path = OUT_DIR / "refinitiv_universe_master.csv"
        if args.resume and ref_path.exists():
            manifest["outputs"]["universe_master"] = existing_frame_meta(ref_path)
        else:
            ref = download_reference(universe)
            manifest["outputs"]["universe_master"] = save_frame(ref, ref_path)

        monthly_panel_path = OUT_DIR / "refinitiv_monthly_panel_tidy.csv"
        monthly_index_path = OUT_DIR / "refinitiv_monthly_batch_index.csv"
        if args.skip_monthly:
            if monthly_panel_path.exists():
                manifest["outputs"]["monthly_panel_tidy"] = existing_frame_meta(monthly_panel_path)
            if monthly_index_path.exists():
                manifest["outputs"]["prices_market_data_monthly_batches"] = {
                    "batch_index": existing_frame_meta(monthly_index_path),
                    "skipped": True,
                }
        else:
            monthly_panel, monthly_manifest = download_monthly_batches(
                universe,
                args.start,
                args.end,
                args.monthly_batch_size,
                args.resume,
                args.max_retries,
                args.retry_sleep,
            )
            manifest["outputs"]["prices_market_data_monthly_batches"] = monthly_manifest
            manifest["outputs"]["monthly_panel_tidy"] = save_frame(
                monthly_panel, monthly_panel_path
            )

        if not args.skip_fundamentals:
            fundamentals, fundamental_manifest = download_fundamental_batches(
                universe,
                args.start_fy,
                args.end_fy,
                args.fundamental_batch_size,
                args.resume,
                args.max_retries,
                args.retry_sleep,
            )
            manifest["outputs"]["fundamentals_annual_batches"] = fundamental_manifest
            manifest["outputs"]["fundamentals_annual"] = save_frame(
                fundamentals, OUT_DIR / "refinitiv_fundamentals_annual.csv"
            )

        manifest["status"] = "ok"
        return 0
    except Exception as exc:  # pragma: no cover - depends on external entitlement
        manifest["status"] = "error"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        print(f"LSEG download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            ld.close_session()
        except Exception:
            pass
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    raise SystemExit(main())
