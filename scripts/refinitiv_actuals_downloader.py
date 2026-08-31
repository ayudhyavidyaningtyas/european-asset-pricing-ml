"""Download Refinitiv/LSEG reported actuals for forecast-error tests."""
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
DEFAULT_OUTPUT_DIR = EXPORT_DIR / "actuals"
DEFAULT_COMBINED_OUTPUT = EXPORT_DIR / "refinitiv_actuals_annual.csv.gz"
PILOT_UNIVERSE = [
    "UCB.BR",
    "ASML.AS",
    "SIEGn.DE",
    "NESN.S",
    "BP.L",
]
DEFAULT_FIELDS = [
    "TR.RIC",
    "TR.ISIN",
    "TR.EPSActValue",
    "TR.EPSActValue.date",
    "TR.EPSActValue.periodenddate",
    "TR.EPSFRActValue",
    "TR.EPSFRActValue.date",
    "TR.EPSFRActValue.periodenddate",
    "TR.RevenueActValue",
    "TR.RevenueActValue.date",
    "TR.RevenueActValue.periodenddate",
]


def chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


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
    for column in out.select_dtypes(include="object").columns:
        out[column] = out[column].map(lambda value: None if pd.isna(value) else str(value))
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


def batch_path(output_dir: Path, batch_index: int) -> Path:
    return output_dir / "_batches" / f"batch_{batch_index:04d}.parquet"


def existing_batch(path: Path, batch: list[str], start: str, end: str) -> dict | None:
    if not path.exists():
        return None
    try:
        sample = pd.read_parquet(
            path,
            columns=["download_batch", "download_start", "download_end"],
        )
    except Exception:
        return None
    if sample.empty:
        return None
    if sample["download_batch"].iloc[0] != ",".join(batch):
        return None
    if sample["download_start"].iloc[0] != start or sample["download_end"].iloc[0] != end:
        return None
    return {"path": str(path), "rows": int(sample.shape[0]), "reused": True}


def save_batch(
    frame: pd.DataFrame,
    path: Path,
    batch: list[str],
    start: str,
    end: str,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = clean_frame(frame)
    out["download_start"] = start
    out["download_end"] = end
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


def open_desktop_session(app_key: str | None):
    if app_key:
        return ld.open_session(name="desktop.workspace", app_key=app_key)
    return ld.open_session(name="desktop.workspace")


def download_actuals_batch(
    batch: list[str],
    fields: list[str],
    parameters: dict[str, str],
    start: str,
    end: str,
    max_retries: int,
    retry_sleep: float,
) -> pd.DataFrame:
    request_parameters = {**parameters, "SDate": start, "EDate": end}
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
            wait = retry_sleep * (attempt + 1)
            print(
                f"batch failed: {type(exc).__name__}: {exc}; retrying in {wait:.0f}s",
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


def run_download(
    universe: list[str],
    fields: list[str],
    parameters: dict[str, str],
    start: str,
    end: str,
    output_dir: Path,
    combined_output: Path,
    batch_size: int,
    max_retries: int,
    retry_sleep: float,
    resume: bool,
    skip_combine: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": None,
        "status": "started",
        "universe_n": len(universe),
        "fields": fields,
        "parameters": {**parameters, "SDate": start, "EDate": end},
        "batch_size": batch_size,
        "batches": [],
        "failures": [],
        "outputs": {},
    }
    app_key = os.environ.get("LSEG_APP_KEY")
    manifest["auth"] = {
        "session": "desktop.workspace",
        "app_key_env_set": bool(app_key),
    }
    batch_paths: list[Path] = []
    try:
        open_desktop_session(app_key)
        for batch_index, batch in enumerate(chunks(universe, batch_size), start=1):
            path = batch_path(output_dir, batch_index)
            if resume:
                reused = existing_batch(path, batch, start, end)
                if reused is not None:
                    manifest["batches"].append(reused)
                    batch_paths.append(path)
                    print(f"reused batch {batch_index:04d}: {reused['rows']} rows")
                    continue
            try:
                frame = download_actuals_batch(
                    batch,
                    fields,
                    parameters,
                    start,
                    end,
                    max_retries,
                    retry_sleep,
                )
                batch_info = save_batch(frame, path, batch, start, end)
                manifest["batches"].append(batch_info)
                batch_paths.append(path)
                print(f"batch {batch_index:04d}: {batch_info['rows']} rows", flush=True)
            except Exception as exc:
                failure = {
                    "batch_index": batch_index,
                    "batch": batch,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                manifest["failures"].append(failure)
                print(f"failed: {failure}", file=sys.stderr, flush=True)
        manifest["status"] = "success" if not manifest["failures"] else "partial"
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
        if not skip_combine:
            manifest["outputs"]["combined_csv"] = combine_batches(
                batch_paths,
                combined_output,
            )
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest_path = output_dir / "refinitiv_actuals_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"manifest -> {manifest_path}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe-csv", type=Path, default=DEFAULT_UNIVERSE_CSV)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2026-07-24")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=75)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--parameter", action="append", default=[])
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-combine", action="store_true")
    args = parser.parse_args()

    universe = PILOT_UNIVERSE if args.pilot else read_universe(args.universe_csv)
    parameters = {
        "Period": "FY0",
        "Frq": "FY",
        "Curn": "EUR",
        **parse_key_value(args.parameter),
    }
    manifest = run_download(
        universe=universe,
        fields=args.field or DEFAULT_FIELDS,
        parameters=parameters,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        combined_output=args.combined_output,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        resume=args.resume,
        skip_combine=args.skip_combine,
    )
    rows = sum(batch.get("rows", 0) for batch in manifest["batches"])
    print(f"batches: {len(manifest['batches'])}")
    print(f"rows: {rows}")
    print(f"failures: {len(manifest['failures'])}")
    return 0 if manifest["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
