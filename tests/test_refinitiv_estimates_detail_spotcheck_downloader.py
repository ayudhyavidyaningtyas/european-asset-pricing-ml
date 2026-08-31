from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "refinitiv_estimates_detail_spotcheck_downloader.py"
)
SPEC = importlib.util.spec_from_file_location(
    "refinitiv_estimates_detail_spotcheck_downloader",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_sample_firm_months_filters_revision_signal_and_lag(tmp_path: Path):
    panel = pd.DataFrame(
        {
            "ric": ["AAA", "BBB", "CCC"],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "estimates_feature_count": [1, 1, 0],
            "est_signal_lag_months": [1.0, 0.0, None],
            "est_eps_revision_1m": [0.1, 0.2, None],
            "est_eps_revision_3m": [None, None, None],
            "est_revenue_revision_1m": [None, None, None],
            "est_revenue_revision_3m": [None, None, None],
            "est_price_target_revision_1m": [None, None, None],
            "est_price_target_revision_3m": [None, None, None],
        }
    )
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="lag guard failed"):
        MODULE.sample_firm_months(
            path,
            sample_size=5,
            random_state=1,
            start=None,
            end=None,
            require_revision_signal=True,
            require_estimate_signal_lag_months=1,
            lookback_days=90,
            stratify=False,
        )

    panel.loc[panel["ric"].eq("BBB"), "est_signal_lag_months"] = 1.0
    panel.to_parquet(path, index=False)

    sample = MODULE.sample_firm_months(
        path,
        sample_size=5,
        random_state=1,
        start=None,
        end=None,
        require_revision_signal=True,
        require_estimate_signal_lag_months=1,
        lookback_days=90,
        stratify=False,
    )

    assert set(sample["ric"]) == {"AAA", "BBB"}
    assert (sample["query_end"] == sample["snapshot_date"]).all()
    assert (sample["snapshot_date"] - sample["query_start"]).dt.days.eq(90).all()


def test_sample_firm_months_uses_estimate_snapshot_date_when_available(tmp_path: Path):
    panel = pd.DataFrame(
        {
            "ric": ["AAA"],
            "date": pd.to_datetime(["2024-02-29"]),
            "est_snapshot_date": pd.to_datetime(["2024-01-31"]),
            "estimates_feature_count": [1],
            "est_signal_lag_months": [1.0],
        }
    )
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)

    sample = MODULE.sample_firm_months(
        path,
        sample_size=1,
        random_state=1,
        start=None,
        end=None,
        require_revision_signal=False,
        require_estimate_signal_lag_months=1,
        lookback_days=30,
        stratify=False,
    )

    assert sample.loc[0, "panel_date"] == pd.Timestamp("2024-02-29")
    assert sample.loc[0, "snapshot_date"] == pd.Timestamp("2024-01-31")
    assert sample.loc[0, "query_end"] == pd.Timestamp("2024-01-31")


def test_sample_firm_months_can_stratify_by_year_and_size(tmp_path: Path):
    records = []
    for year in [2022, 2023]:
        for bucket, percentile in enumerate([0.2, 0.5, 0.8]):
            for stock in range(3):
                records.append(
                    {
                        "ric": f"{year}_{bucket}_{stock}",
                        "date": pd.Timestamp(f"{year}-06-30"),
                        "est_snapshot_date": pd.Timestamp(f"{year}-05-31"),
                        "company_market_cap": 100 + bucket,
                        "market_cap_percentile": percentile,
                        "estimates_feature_count": 1,
                        "est_signal_lag_months": 1.0,
                    },
                )
    panel = pd.DataFrame(records)
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)

    sample = MODULE.sample_firm_months(
        path,
        sample_size=6,
        random_state=7,
        start=None,
        end=None,
        require_revision_signal=False,
        require_estimate_signal_lag_months=1,
        lookback_days=30,
        stratify=True,
    )

    assert len(sample) == 6
    assert sample["panel_date"].dt.year.nunique() == 2
    assert set(MODULE.size_bucket(sample)) == {"small", "mid", "large"}


def test_clean_frame_flattens_multiindex_columns():
    frame = pd.DataFrame(
        [[1.2]],
        columns=pd.MultiIndex.from_tuples([("TR.EPSEstValue", "Value")]),
    )

    result = MODULE.clean_frame(frame)

    assert result.columns.tolist() == ["TR.EPSEstValue__Value"]
