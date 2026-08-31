"""Download monthly Refinitiv liquidity snapshots for implementable portfolios.

The pull is deliberately narrower than the supplemental daily downloader:
monthly bid and ask from 2013 onward are combined with close and volume already
stored in the EUR monthly panel.

Usage:
    export LSEG_APP_KEY="your-app-key"
    python scripts/refinitiv_liquidity_downloader.py --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import pandas as pd

try:
    pd.set_option("future.no_silent_downcasting", True)
except KeyError:  # pragma: no cover - older pandas compatibility
    pass

warnings.filterwarnings(
    "ignore",
    message=r"Downcasting.*deprecated.*",
    category=FutureWarning,
)

try:
    import lseg.data as ld
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Install first: python -m pip install lseg-data pyarrow") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_exports"
SUPPLEMENTAL_DIR = EXPORT_DIR / "supplemental"
OUT_DIR = SUPPLEMENTAL_DIR / "liquidity_monthly_full_period"
INDEX_PATH = SUPPLEMENTAL_DIR / "liquidity_monthly_full_period_batch_index.csv"
MANIFEST_PATH = SUPPLEMENTAL_DIR / "refinitiv_liquidity_full_period_manifest.json"
DEFAULT_UNIVERSE = EXPORT_DIR / "europe_equity_universe_rics_only.csv"

LIQUIDITY_FIELDS = ["BID", "ASK"]
PILOT_UNIVERSE = ["VOD.L", "BP.L", "AZN.L", "ASML.AS", "SAPG.DE"]

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from refinitiv_supplemental_downloader import (  # noqa: E402
    atomic_json,
    batch_signature,
    chunks,
    metadata_path,
    open_desktop_session,
    reusable_batch,
    run_with_retries,
    save_parquet_batch,
    tolerant_get_history,
    year_windows,
)


def probe_fields(
    universe: list[str],
    start: str,
    end: str,
    max_retries: int,
    retry_sleep: float,
) -> tuple[list[str], dict[str, str]]:
    probe_start = max(pd.Timestamp(start), pd.Timestamp("2024-01-01")).date().isoformat()
    probe_end = min(pd.Timestamp(end), pd.Timestamp("2024-12-31")).date().isoformat()
    if probe_end < probe_start:
        probe_start, probe_end = start, end

    available: list[str] = []
    errors: dict[str, str] = {}
    probe_universe = [ric for ric in PILOT_UNIVERSE if ric in set(universe)] or universe[:2]
    for field in LIQUIDITY_FIELDS:
        try:
            result = run_with_retries(
                f"probe monthly {field}",
                lambda field=field: ld.get_history(
                    universe=probe_universe[:2],
                    fields=[field],
                    interval="monthly",
                    start=probe_start,
                    end=probe_end,
                    header_type=ld.HeaderType.NAME,
                ),
                max_retries,
                retry_sleep,
            )
            if result.empty:
                raise ValueError("field returned no monthly observations")
            available.append(field)
            print(f"monthly field available: {field}", flush=True)
        except Exception as exc:
            errors[field] = f"{type(exc).__name__}: {exc}"
            print(f"monthly field unavailable: {field}: {exc}", file=sys.stderr, flush=True)

    if "BID" not in available or "ASK" not in available:
        raise RuntimeError("Both BID and ASK are required for spread estimation.")
    return available, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=15.0)
    args = parser.parse_args()

    app_key = os.environ.get("LSEG_APP_KEY")
    if not app_key:
        raise SystemExit("Set LSEG_APP_KEY in this terminal before running the downloader.")
    if not args.universe_csv.exists():
        raise SystemExit(f"Universe CSV not found: {args.universe_csv}")
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    else:
        output_dir = output_dir.resolve()
    index_path = args.index_path.expanduser() if args.index_path else output_dir.with_name(f"{output_dir.name}_batch_index.csv")
    if not index_path.is_absolute():
        index_path = (PROJECT_ROOT / index_path).resolve()
    else:
        index_path = index_path.resolve()
    manifest_path = args.manifest_path.expanduser() if args.manifest_path else output_dir.with_name(
        f"refinitiv_{output_dir.name}_manifest.json"
    )
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()

    universe_frame = pd.read_csv(args.universe_csv)
    if "ric" not in universe_frame.columns:
        raise SystemExit("Universe CSV must contain a column named 'ric'.")
    universe = (
        universe_frame["ric"].dropna().astype(str).str.strip().drop_duplicates().tolist()
    )
    universe_batches = list(chunks(universe, args.batch_size))
    # BID and ASK for 100 securities over the full sample remain below the
    # interday datapoint guideline and avoid fourteen separate year requests.
    windows = [("full", args.start, args.end)]
    total_jobs = len(universe_batches) * len(windows)
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "started",
        "universe": str(args.universe_csv),
        "universe_size": len(universe),
        "output_dir": str(output_dir),
        "index_path": str(index_path),
        "manifest_path": str(manifest_path),
        "start": args.start,
        "end": args.end,
        "interval": "monthly",
        "batch_size": args.batch_size,
        "workers": args.workers,
        "jobs_planned": total_jobs,
    }
    records: list[dict] = []

    try:
        open_desktop_session(app_key)
        fields, errors = probe_fields(
            universe, args.start, args.end, args.max_retries, args.retry_sleep
        )
        manifest["field_probe"] = {"available": fields, "errors": errors}

        jobs = []
        job = 0
        index_path.parent.mkdir(parents=True, exist_ok=True)
        for year, window_start, window_end in windows:
            for batch_no, batch in enumerate(universe_batches, start=1):
                job += 1
                jobs.append(
                    (job, year, window_start, window_end, batch_no, batch)
                )

        def execute_job(spec: tuple) -> dict:
            job_no, year, window_start, window_end, batch_no, batch = spec
            path = output_dir / f"year={year}" / f"batch_{batch_no:04d}.parquet"
            signature = batch_signature(
                batch,
                fields=fields,
                start=window_start,
                end=window_end,
                interval="monthly",
            )
            reused = reusable_batch(path, signature) if args.resume else None
            if reused:
                metadata = reused
                status = "reused"
            else:
                frame, failed, batch_errors = tolerant_get_history(
                    batch,
                    fields,
                    window_start,
                    window_end,
                    f"liquidity {year} batch {batch_no}",
                    args.max_retries,
                    args.retry_sleep,
                    interval="monthly",
                )
                metadata = save_parquet_batch(
                    frame,
                    path,
                    signature,
                    failed_rics=failed,
                    errors=batch_errors,
                )
                status = (
                    "downloaded"
                    if not failed
                    else ("partial" if not frame.empty else "failed_all")
                )
            return {
                "job": job_no,
                "year": year,
                "batch_no": batch_no,
                "status": status,
                "n_instruments": len(batch),
                "rows": metadata["rows"],
                "bytes": metadata["bytes"],
                "failed_rics_n": len(metadata.get("failed_rics", [])),
                "path": str(path),
                "metadata_path": str(metadata_path(path)),
            }

        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(execute_job, spec): spec for spec in jobs}
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed += 1
                if completed % 20 == 0:
                    pd.DataFrame(records).to_csv(index_path, index=False)
                print(
                    f"liquidity {completed}/{total_jobs}: "
                    f"year={record['year']} batch={record['batch_no']}; "
                    f"{record['status']}; rows={record['rows']}; "
                    f"failed={record['failed_rics_n']}",
                    flush=True,
                )

        index = pd.DataFrame(records).sort_values("job")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(index_path, index=False)
        manifest["status"] = "ok"
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["output"] = {
            "dataset": str(output_dir),
            "batch_index": str(index_path),
            "jobs": len(index),
            "rows": int(index["rows"].sum()),
            "bytes": int(index["bytes"].sum()),
            "failed_identifier_jobs": int(index["failed_rics_n"].sum()),
        }
        return 0
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        print(f"Liquidity download failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
        atomic_json(manifest_path, manifest)
        print(f"manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
