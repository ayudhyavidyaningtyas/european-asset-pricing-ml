from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_refinitiv_estimates_detail_spotcheck.py"
)
SPEC = importlib.util.spec_from_file_location(
    "audit_refinitiv_estimates_detail_spotcheck",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_audit_counts_vintage_and_broker_fields():
    detail = pd.DataFrame(
        {
            "TR.EPSESTVALUE.DATE": [
                "2024-01-15T10:00:00Z",
                "2024-02-01T01:00:00Z",
                "",
            ],
            "TR.EPSESTVALUE": ["1.2", "", "3.4"],
            "TR.EPSESTVALUE.BROKERNAME": [
                "Visible Broker",
                "Permission Denied 123",
                "",
            ],
            "sample_ric": ["AAA", "AAA", "BBB"],
            "sample_panel_date": ["2024-02-29", "2024-02-29", "2024-02-29"],
            "sample_snapshot_date": ["2024-01-31", "2024-01-31", "2024-01-31"],
            "query_start": ["2024-01-01", "2024-01-01", "2024-01-01"],
        }
    )

    audit, per_sample = MODULE.build_audit(detail)
    row = audit.iloc[0]

    assert row["sample_firm_months"] == 2
    assert row["detail_rows"] == 3
    assert row["detail_rows_with_estimate_date"] == 2
    assert row["detail_rows_with_numeric_estimate_value"] == 2
    assert row["detail_rows_with_date_and_value"] == 1
    assert row["samples_with_dated_rows"] == 1
    assert row["dated_rows_after_snapshot"] == 1
    assert row["broker_permission_denied_rows"] == 1
    assert row["visible_broker_rows"] == 1
    assert set(per_sample["sample_ric"]) == {"AAA", "BBB"}
