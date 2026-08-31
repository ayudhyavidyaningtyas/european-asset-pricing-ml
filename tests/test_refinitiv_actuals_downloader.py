from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refinitiv_actuals_downloader.py"
SPEC = importlib.util.spec_from_file_location("refinitiv_actuals_downloader", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_clean_frame_flattens_columns_and_stringifies_objects():
    frame = pd.DataFrame(
        [["AAA", pd.NA]],
        columns=pd.MultiIndex.from_tuples([("TR.RIC", "Name"), ("TR.ISIN", "Name")]),
    )

    result = MODULE.clean_frame(frame)

    assert result.columns.tolist() == ["TR.RIC__Name", "TR.ISIN__Name"]
    assert result.loc[0, "TR.RIC__Name"] == "AAA"
    assert result.loc[0, "TR.ISIN__Name"] is None


def test_existing_batch_requires_same_batch_and_window(tmp_path: Path):
    path = tmp_path / "batch.parquet"
    pd.DataFrame(
        {
            "download_batch": ["AAA,BBB"],
            "download_start": ["2020-01-01"],
            "download_end": ["2024-12-31"],
        },
    ).to_parquet(path, index=False)

    assert MODULE.existing_batch(path, ["AAA", "BBB"], "2020-01-01", "2024-12-31")
    assert MODULE.existing_batch(path, ["AAA"], "2020-01-01", "2024-12-31") is None
    assert MODULE.existing_batch(path, ["AAA", "BBB"], "2021-01-01", "2024-12-31") is None
