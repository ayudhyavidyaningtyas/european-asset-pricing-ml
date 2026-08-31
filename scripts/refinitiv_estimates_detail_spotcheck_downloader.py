"""Pull a small Refinitiv analyst-estimate detail spot check.

This targets the vintage-timestamp concern. It samples firm-months from the
strict estimates panel, then requests analyst-level EPS estimate history around
each month-end so we can inspect whether estimate dates are available and
consistent with the lagged consensus signal.
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
import pyarrow.parquet as pq

try:
    pd.set_option("future.no_silent_downcasting", True)
except KeyError:  # pragma: no cover - older pandas compatibility
    pass

try:
    import lseg.data as ld
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Install first: python -m pip install lseg-data pyarrow") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "strict_estimates_lag1"
    / "monthly_feature_panel_estimates_strict_lag1.parquet"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "estimates_detail_spotcheck"
)
DEFAULT_FIELDS = [
    "TR.RIC",
    "TR.ISIN",
    "TR.EPSEstValue.date",
    "TR.EPSEstValue",
    "TR.EPSEstValue.BrokerName",
]
REVISION_FEATURES = [
    "est_eps_revision_1m",
    "est_eps_revision_3m",
    "est_revenue_revision_1m",
    "est_revenue_revision_3m",
    "est_price_target_revision_1m",
    "est_price_target_revision_3m",
]


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


def sample_firm_months(
    panel_path: Path,
    sample_size: int,
    random_state: int,
    start: str | None,
    end: str | None,
    require_revision_signal: bool,
    require_estimate_signal_lag_months: int | None,
    lookback_days: int,
    stratify: bool,
) -> pd.DataFrame:
    filter_columns = [
        "ric",
        "date",
        "est_snapshot_date",
        "company_market_cap",
        "market_cap_percentile",
        "estimates_feature_count",
        "est_signal_lag_months",
        *REVISION_FEATURES,
    ]
    available_columns = pq.ParquetFile(panel_path).schema_arrow.names
    columns = [column for column in filter_columns if column in available_columns]
    if "ric" not in columns or "date" not in columns:
        columns = ["ric", "date", *[c for c in columns if c not in {"ric", "date"}]]
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"])
    if start is not None:
        panel = panel[panel["date"].ge(pd.Timestamp(start))]
    if end is not None:
        panel = panel[panel["date"].le(pd.Timestamp(end))]
    if require_revision_signal:
        revision_columns = [column for column in REVISION_FEATURES if column in panel]
        if not revision_columns:
            raise ValueError("Panel has no raw revision feature columns")
        panel = panel[panel[revision_columns].notna().any(axis=1)]
    if require_estimate_signal_lag_months is not None:
        if "est_signal_lag_months" not in panel or "estimates_feature_count" not in panel:
            raise ValueError("Panel lacks estimate lag audit columns")
        lag = pd.to_numeric(panel["est_signal_lag_months"], errors="coerce")
        has_estimates = pd.to_numeric(
            panel["estimates_feature_count"],
            errors="coerce",
        ).gt(0)
        invalid = has_estimates & (lag.isna() | lag.lt(require_estimate_signal_lag_months))
        if invalid.any():
            raise ValueError(
                "Estimate signal lag guard failed for sampled panel: "
                f"{int(invalid.sum()):,} violations"
            )
    if "est_snapshot_date" in panel:
        panel["snapshot_date"] = pd.to_datetime(panel["est_snapshot_date"], errors="coerce")
    else:
        panel["snapshot_date"] = panel["date"]
    panel = panel[panel["snapshot_date"].notna()]
    sample_columns = [
        column
        for column in [
            "ric",
            "date",
            "snapshot_date",
            "company_market_cap",
            "market_cap_percentile",
        ]
        if column in panel
    ]
    panel = panel[sample_columns].drop_duplicates()
    if panel.empty:
        raise ValueError("No firm-months available after filters")
    sample_n = min(sample_size, len(panel))
    if stratify:
        sample = stratified_firm_month_sample(panel, sample_n, random_state)
    else:
        sample = panel.sample(sample_n, random_state=random_state)
    sample = sample.sort_values(["date", "ric"])
    sample = sample.rename(columns={"date": "panel_date"}).reset_index(drop=True)
    sample["query_start"] = sample["snapshot_date"] - pd.to_timedelta(
        lookback_days,
        unit="D",
    )
    sample["query_end"] = sample["snapshot_date"]
    return sample


def size_bucket(panel: pd.DataFrame) -> pd.Series:
    if "date" in panel:
        date_key = panel["date"]
    elif "panel_date" in panel:
        date_key = panel["panel_date"]
    else:
        date_key = pd.Series(0, index=panel.index)
    if "market_cap_percentile" in panel:
        size = pd.to_numeric(panel["market_cap_percentile"], errors="coerce")
        if "company_market_cap" in panel and size.isna().any():
            market_cap = pd.to_numeric(panel["company_market_cap"], errors="coerce")
            cap_rank = market_cap.groupby(date_key).rank(
                method="average",
                pct=True,
            )
            size = size.fillna(cap_rank)
    elif "company_market_cap" in panel:
        market_cap = pd.to_numeric(panel["company_market_cap"], errors="coerce")
        size = market_cap.groupby(date_key).rank(method="average", pct=True)
    else:
        return pd.Series("unknown", index=panel.index)
    return pd.cut(
        size,
        bins=[-float("inf"), 1 / 3, 2 / 3, float("inf")],
        labels=["small", "mid", "large"],
    ).astype("string").fillna("unknown")


def stratified_firm_month_sample(
    panel: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    work = panel.copy()
    work["_sample_year"] = pd.to_datetime(work["date"]).dt.year.astype(str)
    work["_size_bucket"] = size_bucket(work)
    valid_size = work["_size_bucket"].ne("unknown")
    if valid_size.sum() >= sample_size:
        work = work[valid_size].copy()
    work["_stratum"] = work["_sample_year"] + "_" + work["_size_bucket"].astype(str)

    groups = list(work.groupby("_stratum", sort=True))
    if not groups:
        return work.head(0).drop(columns=["_sample_year", "_size_bucket", "_stratum"])

    base_take = sample_size // len(groups)
    remainder = sample_size % len(groups)
    sampled_frames = []
    shortfall = 0
    for index, (_, group) in enumerate(groups):
        target = base_take + int(index < remainder)
        take = min(target, len(group))
        shortfall += target - take
        if take:
            sampled_frames.append(
                group.sample(take, random_state=random_state + index),
            )

    if sampled_frames:
        sampled = pd.concat(sampled_frames, ignore_index=False)
        used = set(sampled.index)
    else:
        sampled = work.head(0)
        used = set()

    if shortfall > 0 and len(sampled) < sample_size:
        remainder_pool = work.loc[~work.index.isin(used)]
        extra_take = min(shortfall, sample_size - len(sampled), len(remainder_pool))
        if extra_take:
            sampled = pd.concat(
                [
                    sampled,
                    remainder_pool.sample(
                        extra_take,
                        random_state=random_state + len(groups) + 1,
                    ),
                ],
                ignore_index=False,
            )

    return sampled.drop(columns=["_sample_year", "_size_bucket", "_stratum"])


def open_desktop_session(app_key: str | None):
    if app_key:
        return ld.open_session(name="desktop.workspace", app_key=app_key)
    return ld.open_session(name="desktop.workspace")


def request_detail(
    ric: str,
    fields: list[str],
    query_start: pd.Timestamp,
    query_end: pd.Timestamp,
    parameters: dict[str, str],
    max_retries: int,
    retry_sleep: float,
) -> pd.DataFrame:
    request_parameters = {
        **parameters,
        "SDate": query_start.date().isoformat(),
        "EDate": query_end.date().isoformat(),
    }
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return ld.get_data(
                universe=[ric],
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
                f"{ric} {query_end.date()} failed: {type(exc).__name__}: {exc}; "
                f"retrying in {wait:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    assert last_error is not None
    raise last_error


def write_outputs(
    output_dir: Path,
    sample: pd.DataFrame,
    frames: Iterable[pd.DataFrame],
    manifest: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_dir / "sampled_firm_months.csv", index=False)
    frame_list = list(frames)
    detail = pd.concat(frame_list, ignore_index=True) if frame_list else pd.DataFrame()
    for column in detail.select_dtypes(include="object").columns:
        detail[column] = detail[column].map(
            lambda value: None if pd.isna(value) else str(value),
        )
    detail_path = output_dir / "refinitiv_estimates_detail_spotcheck.parquet"
    detail.to_parquet(detail_path, index=False, engine="pyarrow", compression="zstd")
    detail.to_csv(
        output_dir / "refinitiv_estimates_detail_spotcheck.csv.gz",
        index=False,
        compression="gzip",
    )
    manifest["outputs"] = {
        "sample": str(output_dir / "sampled_firm_months.csv"),
        "detail_parquet": str(detail_path),
        "detail_csv": str(output_dir / "refinitiv_estimates_detail_spotcheck.csv.gz"),
    }
    manifest["rows"]["detail"] = int(len(detail))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def run_pull(
    panel: Path,
    output_dir: Path,
    fields: list[str],
    parameters: dict[str, str],
    sample_size: int,
    random_state: int,
    start: str | None,
    end: str | None,
    require_revision_signal: bool,
    require_estimate_signal_lag_months: int | None,
    lookback_days: int,
    stratify: bool,
    max_retries: int,
    retry_sleep: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": None,
        "status": "started",
        "panel": str(panel),
        "fields": fields,
        "parameters": parameters,
        "sample_size": sample_size,
        "random_state": random_state,
        "start": start,
        "end": end,
        "require_revision_signal": require_revision_signal,
        "require_estimate_signal_lag_months": require_estimate_signal_lag_months,
        "lookback_days": lookback_days,
        "stratify": stratify,
        "rows": {
            "sample": 0,
            "detail": 0,
        },
        "failures": [],
    }
    sample = sample_firm_months(
        panel,
        sample_size,
        random_state,
        start,
        end,
        require_revision_signal,
        require_estimate_signal_lag_months,
        lookback_days,
        stratify,
    )
    manifest["rows"]["sample"] = int(len(sample))
    sample.to_csv(output_dir / "sampled_firm_months.csv", index=False)

    app_key = os.environ.get("LSEG_APP_KEY")
    manifest["auth"] = {
        "session": "desktop.workspace",
        "app_key_env_set": bool(app_key),
    }

    frames = []
    try:
        open_desktop_session(app_key)
        for row in sample.itertuples(index=False):
            try:
                frame = request_detail(
                    row.ric,
                    fields,
                    pd.Timestamp(row.query_start),
                    pd.Timestamp(row.query_end),
                    parameters,
                    max_retries,
                    retry_sleep,
                )
                out = clean_frame(frame)
                out["sample_ric"] = row.ric
                out["sample_panel_date"] = pd.Timestamp(row.panel_date).date().isoformat()
                out["sample_snapshot_date"] = pd.Timestamp(
                    row.snapshot_date,
                ).date().isoformat()
                out["query_start"] = pd.Timestamp(row.query_start).date().isoformat()
                out["query_end"] = pd.Timestamp(row.query_end).date().isoformat()
                frames.append(out)
                print(
                    f"{row.ric} {pd.Timestamp(row.snapshot_date).date()}: "
                    f"{len(out)} rows",
                    flush=True,
                )
            except Exception as exc:
                failure = {
                    "ric": row.ric,
                    "snapshot_date": pd.Timestamp(row.snapshot_date).date().isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                manifest["failures"].append(failure)
                print(f"failed: {failure}", file=sys.stderr, flush=True)
        manifest["status"] = "success" if not manifest["failures"] else "partial"
        return manifest
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
        manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_outputs(output_dir, sample, frames, manifest)
        print(f"manifest -> {output_dir / 'manifest.json'}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--start", default="2015-01-31")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--field", action="append", default=[])
    parser.add_argument("--parameter", action="append", default=[])
    parser.add_argument("--period", default="FY1")
    parser.add_argument("--frq", default="FY")
    parser.add_argument("--curn", default="EUR")
    parser.add_argument("--require-revision-signal", action="store_true")
    parser.add_argument("--require-estimate-signal-lag-months", type=int)
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Sample approximately evenly across panel-year x size-tercile cells.",
    )
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    args = parser.parse_args()

    fields = args.field or DEFAULT_FIELDS
    parameters = {
        "Period": args.period,
        "Frq": args.frq,
        "Curn": args.curn,
        **parse_key_value(args.parameter),
    }
    manifest = run_pull(
        panel=args.panel,
        output_dir=args.output_dir,
        fields=fields,
        parameters=parameters,
        sample_size=args.sample_size,
        random_state=args.random_state,
        start=args.start,
        end=args.end,
        require_revision_signal=args.require_revision_signal,
        require_estimate_signal_lag_months=args.require_estimate_signal_lag_months,
        lookback_days=args.lookback_days,
        stratify=args.stratify,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"failures: {len(manifest['failures'])}", flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0 if manifest["status"] == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
