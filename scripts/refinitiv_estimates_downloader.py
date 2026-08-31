"""Download Refinitiv/LSEG analyst-estimates snapshots for the ML panel.

Usage:
    export LSEG_APP_KEY="your-app-key"
    python scripts/refinitiv_estimates_downloader.py --pilot --start 2024-01 --end 2024-06

For the full panel, pass a CSV with a ``ric`` column. The output is intentionally
schema-flexible; ``scripts/build_estimates_enriched_panel.py`` maps common
Refinitiv field labels to the canonical feature names.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable

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
DEFAULT_UNIVERSE_CSV = EXPORT_DIR / "europe_equity_universe_rics_only.csv"
OUT_DIR = EXPORT_DIR / "estimates"
BATCH_DIR = OUT_DIR / "_batches"
MANIFEST = OUT_DIR / "refinitiv_estimates_manifest.json"
COMBINED_OUTPUT = EXPORT_DIR / "refinitiv_analyst_estimates_monthly.csv.gz"

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

DEFAULT_FIELDS = [
    "TR.RIC",
    "TR.ISIN",
    "TR.EPSMean",
    "TR.EPSHigh",
    "TR.EPSLow",
    "TR.EPSStdDev",
    "TR.EPSNumEstimates",
    "TR.RevenueMean",
    "TR.RevenueHigh",
    "TR.RevenueLow",
    "TR.RevenueStdDev",
    "TR.RevenueNumEstimates",
    "TR.PriceTargetMean",
    "TR.PriceTargetHigh",
    "TR.PriceTargetLow",
    "TR.PriceTargetStdDev",
    "TR.PriceTargetNumEstimates",
    "TR.RecommendationMean",
    "TR.RecommendationNumEstimates",
]


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def month_ends(start: str, end: str) -> list[pd.Timestamp]:
    start_period = pd.Period(start, freq="M")
    end_period = pd.Period(end, freq="M")
    dates = pd.period_range(start_period, end_period, freq="M").to_timestamp("M")
    if dates.empty:
        raise ValueError("date range produced no month-end snapshots")
    return list(dates)


def parse_key_value(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Parameter must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "__".join(
                str(part)
                for part in values
                if part is not None and str(part) != "nan"
            ).strip("_")
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


def read_universe(path: Path) -> list[str]:
    frame = pd.read_csv(path, low_memory=False)
    lower_lookup = {str(column).strip().lower(): column for column in frame.columns}
    column = (
        lower_lookup.get("ric")
        or lower_lookup.get("tr.ric")
        or lower_lookup.get("instrument")
        or frame.columns[0]
    )
    return (
        frame[column]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .drop_duplicates()
        .tolist()
    )


def batch_path(snapshot: pd.Timestamp, batch_index: int) -> Path:
    return (
        BATCH_DIR
        / f"year={snapshot.year}"
        / f"month={snapshot.month:02d}"
        / f"batch_{batch_index:04d}.parquet"
    )


def existing_batch(path: Path, batch: list[str], snapshot: pd.Timestamp) -> dict | None:
    if not path.exists():
        return None
    try:
        sample = pd.read_parquet(path, columns=["download_batch", "snapshot_date"])
    except Exception:
        return None
    if sample.empty:
        return None
    if sample["download_batch"].iloc[0] != ",".join(batch):
        return None
    if str(sample["snapshot_date"].iloc[0])[:10] != snapshot.date().isoformat():
        return None
    return {
        "path": str(path),
        "rows": int(sample.shape[0]),
        "reused": True,
    }


def save_batch(
    frame: pd.DataFrame,
    path: Path,
    batch: list[str],
    snapshot: pd.Timestamp,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = clean_frame(frame)
    out["snapshot_date"] = snapshot.date().isoformat()
    out["download_batch"] = ",".join(batch)
    temporary = path.with_suffix(path.suffix + ".tmp")
    out.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    temporary.replace(path)
    return {
        "path": str(path),
        "rows": int(out.shape[0]),
        "cols": int(out.shape[1]),
        "bytes": int(path.stat().st_size),
    }


def open_desktop_session(app_key: str):
    return ld.open_session(name="desktop.workspace", app_key=app_key)


def error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}".lower()


def is_rate_limit_error(exc: Exception) -> bool:
    message = error_text(exc)
    return (
        "too many requests" in message
        or "rate limit" in message
        or "rate-limit" in message
        or "throttl" in message
        or "429" in message
    )


def is_connection_loss_error(exc: Exception) -> bool:
    message = error_text(exc)
    return (
        "connection refused" in message
        or "connecterror" in message
        or "localhost:9000" in message
        or "[errno 61]" in message
    )


def retry_wait_seconds(
    exc: Exception,
    attempt: int,
    retry_sleep: float,
    rate_limit_sleep: float,
    max_retry_sleep: float,
) -> float:
    base = rate_limit_sleep if is_rate_limit_error(exc) else retry_sleep
    multiplier = 2**attempt if is_rate_limit_error(exc) else attempt + 1
    wait = base * multiplier
    return min(wait, max_retry_sleep) if max_retry_sleep > 0 else wait


def stop_reason_for_error(exc: Exception) -> str | None:
    if is_connection_loss_error(exc):
        return "desktop_session_unavailable"
    if is_rate_limit_error(exc):
        return "rate_limited"
    return None


def download_snapshot_batch(
    batch: list[str],
    fields: list[str],
    snapshot: pd.Timestamp,
    parameters: dict[str, str],
    max_retries: int,
    retry_sleep: float,
    rate_limit_sleep: float,
    max_retry_sleep: float,
) -> pd.DataFrame:
    request_parameters = {
        **parameters,
        "SDate": snapshot.date().isoformat(),
        "EDate": snapshot.date().isoformat(),
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return ld.get_data(
                universe=batch,
                fields=fields,
                parameters=request_parameters,
                header_type=ld.HeaderType.NAME,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            wait = retry_wait_seconds(
                exc,
                attempt,
                retry_sleep,
                rate_limit_sleep,
                max_retry_sleep,
            )
            print(
                f"{snapshot.date()} batch failed: {type(exc).__name__}: {exc}; "
                f"retrying in {wait:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def combine_batches(paths: list[Path], output_path: Path) -> dict:
    frames = [pd.read_parquet(path) for path in paths if path.exists()]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, compression="gzip")
    return {
        "path": str(output_path),
        "rows": int(combined.shape[0]),
        "cols": int(combined.shape[1]),
        "bytes": int(output_path.stat().st_size),
    }


def main() -> int:
    global OUT_DIR, BATCH_DIR, MANIFEST, COMBINED_OUTPUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE_CSV)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--start", default="2005-01")
    parser.add_argument("--end", default="2026-06")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for per-month estimate batches and the manifest.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help=(
            "Combined CSV.GZ output. Defaults to "
            "<output-dir-parent>/refinitiv_analyst_estimates_monthly.csv.gz."
        ),
    )
    parser.add_argument(
        "--base-currency",
        default="EUR",
        help="Refinitiv currency parameter used for estimate levels.",
    )
    parser.add_argument("--batch-size", type=int, default=75)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--parameter", action="append", default=[])
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=120.0,
        help="Initial retry wait in seconds when Refinitiv/LSEG throttles the request.",
    )
    parser.add_argument(
        "--max-retry-sleep",
        type=float,
        default=900.0,
        help="Maximum retry wait in seconds; set <=0 for no cap.",
    )
    parser.add_argument(
        "--inter-batch-sleep",
        type=float,
        default=0.0,
        help="Optional pause in seconds after each successful downloaded batch.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    parser.add_argument(
        "--continue-after-rate-limit",
        action="store_true",
        help="Continue to later batches after a rate-limited batch exhausts retries.",
    )
    parser.add_argument(
        "--continue-after-connection-loss",
        action="store_true",
        help="Continue to later batches after the local desktop API stops responding.",
    )
    args = parser.parse_args()

    OUT_DIR = args.output_dir
    BATCH_DIR = OUT_DIR / "_batches"
    MANIFEST = OUT_DIR / "refinitiv_estimates_manifest.json"
    COMBINED_OUTPUT = (
        args.combined_output
        if args.combined_output is not None
        else OUT_DIR.parent / "refinitiv_analyst_estimates_monthly.csv.gz"
    )

    app_key = os.environ.get("LSEG_APP_KEY")
    if not app_key:
        raise SystemExit("Set LSEG_APP_KEY in this terminal before running the downloader.")

    universe = PILOT_UNIVERSE if args.pilot else read_universe(args.universe_csv)
    fields = args.field or DEFAULT_FIELDS
    parameters = {
        "Period": "FY1",
        "Frq": "FY",
        "Curn": str(args.base_currency).upper(),
        **parse_key_value(args.parameter),
    }
    snapshots = month_ends(args.start, args.end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": None,
        "status": "started",
        "output_dir": str(OUT_DIR),
        "combined_output": str(COMBINED_OUTPUT),
        "universe_n": len(universe),
        "fields": fields,
        "parameters": parameters,
        "retry_policy": {
            "max_retries": args.max_retries,
            "retry_sleep": args.retry_sleep,
            "rate_limit_sleep": args.rate_limit_sleep,
            "max_retry_sleep": args.max_retry_sleep,
            "inter_batch_sleep": args.inter_batch_sleep,
            "continue_after_rate_limit": args.continue_after_rate_limit,
            "continue_after_connection_loss": args.continue_after_connection_loss,
        },
        "snapshots": [snapshot.date().isoformat() for snapshot in snapshots],
        "batches": [],
        "failures": [],
        "stop_reason": None,
    }

    paths: list[Path] = []
    try:
        open_desktop_session(app_key)
        for snapshot in snapshots:
            for index, batch in enumerate(chunks(universe, args.batch_size)):
                path = batch_path(snapshot, index)
                if args.resume:
                    reused = existing_batch(path, batch, snapshot)
                    if reused is not None:
                        manifest["batches"].append(reused)
                        paths.append(path)
                        continue
                try:
                    frame = download_snapshot_batch(
                        batch,
                        fields,
                        snapshot,
                        parameters,
                        args.max_retries,
                        args.retry_sleep,
                        args.rate_limit_sleep,
                        args.max_retry_sleep,
                    )
                    meta = save_batch(frame, path, batch, snapshot)
                    manifest["batches"].append(meta)
                    paths.append(path)
                    print(
                        f"{snapshot.date()} batch {index:04d}: {meta['rows']} rows",
                        flush=True,
                    )
                    if args.inter_batch_sleep > 0:
                        time.sleep(args.inter_batch_sleep)
                except Exception as exc:
                    failure = {
                        "snapshot": snapshot.date().isoformat(),
                        "batch": index,
                        "size": len(batch),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    manifest["failures"].append(failure)
                    print(f"failed: {failure}", file=sys.stderr, flush=True)
                    stop_reason = stop_reason_for_error(exc)
                    should_stop = (
                        stop_reason == "desktop_session_unavailable"
                        and not args.continue_after_connection_loss
                    ) or (
                        stop_reason == "rate_limited"
                        and not args.continue_after_rate_limit
                    )
                    if should_stop:
                        manifest["status"] = "interrupted"
                        manifest["stop_reason"] = stop_reason
                        recovery_hint = (
                            "after the Refinitiv/LSEG throttle window cools down"
                            if stop_reason == "rate_limited"
                            else "after the Refinitiv/LSEG desktop session recovers"
                        )
                        print(
                            f"stopping after {stop_reason}; rerun with --resume {recovery_hint}",
                            file=sys.stderr,
                            flush=True,
                        )
                        return 3
                    if stop_reason == "rate_limited" and args.rate_limit_sleep > 0:
                        print(
                            f"cooling down for {args.rate_limit_sleep:.0f}s before continuing",
                            file=sys.stderr,
                            flush=True,
                        )
                        time.sleep(args.rate_limit_sleep)
        if not args.skip_combine:
            manifest["combined_output"] = combine_batches(paths, COMBINED_OUTPUT)
        manifest["status"] = "success" if not manifest["failures"] else "partial"
        return 0 if not manifest["failures"] else 2
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"manifest -> {MANIFEST}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
