from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "refinitiv_supplemental_downloader.py"
SPEC = importlib.util.spec_from_file_location("refinitiv_supplemental_downloader", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_year_windows_clips_first_and_last_year():
    assert MODULE.year_windows("2020-06-15", "2022-03-04") == [
        (2020, "2020-06-15", "2020-12-31"),
        (2021, "2021-01-01", "2021-12-31"),
        (2022, "2022-01-01", "2022-03-04"),
    ]


def test_history_to_tidy_converts_multiindex_columns():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = pd.MultiIndex.from_tuples(
        [
            ("VOD.L", "TR.PRICECLOSE"),
            ("VOD.L", "TR.TOTALRETURN1D"),
            ("BP.L", "TR.PRICECLOSE"),
            ("BP.L", "TR.TOTALRETURN1D"),
        ]
    )
    raw = pd.DataFrame(
        [[70.0, 1.0, 400.0, -1.0], [71.0, 1.4, 402.0, 0.5]],
        index=dates,
        columns=columns,
    )
    raw.index.name = "Date"

    result = MODULE.history_to_tidy(raw, ["VOD.L", "BP.L"])

    assert list(result.columns) == ["date", "ric", "price_close", "total_return_1d"]
    assert len(result) == 4
    assert set(result["ric"]) == {"VOD.L", "BP.L"}
    assert result.loc[result["ric"].eq("VOD.L"), "price_close"].tolist() == [70.0, 71.0]


def test_expand_fundamental_fields_adds_dates_only_to_value_fields():
    result = MODULE.expand_fundamental_fields(
        ["TR.F.DebtTot", "TR.F.PeriodEndDate", "TR.ISOriginalAnnouncementDate"]
    )

    assert result == [
        "TR.F.DebtTot",
        "TR.F.DebtTot.date",
        "TR.F.PeriodEndDate",
        "TR.ISOriginalAnnouncementDate",
    ]


def test_identifier_error_recognizes_lseg_field_specific_message():
    error = ValueError(
        "Unable to collect data for the field 'TR.F.DebtTot' and some specific identifier(s)"
    )

    assert MODULE.is_identifier_error(error)


def test_combine_fundamental_batches_normalizes_blank_numeric_values(tmp_path):
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    output = tmp_path / "combined.parquet"
    pd.DataFrame(
        {
            "Instrument": ["AAA"],
            "TR.F.INVNTTOT": [1.0],
            "TR.F.INVNTTOT.DATE": [pd.Timestamp("2024-12-31")],
        }
    ).to_parquet(first, index=False)
    pd.DataFrame(
        {
            "Instrument": ["BBB"],
            "TR.F.INVNTTOT": [""],
            "TR.F.INVNTTOT.DATE": [""],
        }
    ).to_parquet(second, index=False)

    MODULE.combine_parquet_batches([first, second], output)
    result = pd.read_parquet(output)

    assert result["TR.F.INVNTTOT"].dtype.kind == "f"
    assert pd.isna(result.loc[1, "TR.F.INVNTTOT"])
    assert pd.api.types.is_datetime64_any_dtype(result["TR.F.INVNTTOT.DATE"])
