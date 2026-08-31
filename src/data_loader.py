"""Lightweight data-loading utilities shared by asset-pricing robustness scripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MISSING_SENTINEL = -99.99

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

FF3_FILE = "Europe_3_Factors.csv"
FF5_FILE = "Europe_5_Factors.csv"
PORT25_FILE = "Europe_25_Portfolios_ME_BE-ME.csv"
PORT6_FILE = "Europe_6_Portfolios_ME_BE-ME.csv"
VW_MONTHLY = "Average Value Weighted Returns -- Monthly"


def _yyyymm_to_month_end(tokens: list[str]) -> pd.DatetimeIndex:
    return pd.to_datetime(tokens, format="%Y%m").to_period("M").to_timestamp("M")


def read_french_block(path: str | Path, section_title: str | None = None) -> pd.DataFrame:
    """Read one monthly block of a Ken French CSV as decimal month-end returns."""
    lines = Path(path).read_text().splitlines()

    if section_title is not None:
        title = section_title.strip().lower()
        try:
            header_idx = next(
                i
                for i, line in enumerate(lines)
                if line.strip().lower().startswith(title)
            ) + 1
        except StopIteration as exc:
            raise ValueError(f"section {section_title!r} not found in {path}") from exc
    else:
        header_idx = next(
            i for i, line in enumerate(lines) if line.lstrip().startswith(",")
        )

    columns = [column.strip() for column in lines[header_idx].split(",")][1:]
    tokens = []
    rows = []
    for line in lines[header_idx + 1 :]:
        if line.strip() == "":
            break
        token = line.split(",")[0].strip()
        if len(token) != 6 or not token.isdigit():
            break
        values = [value.strip() for value in line.split(",")[1 : 1 + len(columns)]]
        tokens.append(token)
        rows.append(values)

    frame = pd.DataFrame(rows, columns=columns, index=_yyyymm_to_month_end(tokens))
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.mask(frame <= MISSING_SENTINEL)
    return frame / 100.0
