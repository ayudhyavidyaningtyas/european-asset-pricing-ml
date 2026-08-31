from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_capacity_gradient_tests import assign_buckets  # noqa: E402
from run_closure_2026_08 import (  # noqa: E402
    build_fixed_count_long_short_legs,
    estimated_feature_signs,
    fixed_count_one_way_turnover,
    literature_feature_signs,
)


def test_estimated_feature_signs_use_supplied_training_data_only():
    rows = []
    for date in pd.to_datetime(["2020-01-31", "2020-02-29"]):
        for security in range(24):
            signal = float(security)
            rows.append(
                {
                    "date": date,
                    "ric": f"S{security:02d}",
                    "target_return_rank": signal,
                    "positive_feature": signal,
                    "negative_feature": -signal,
                }
            )
    train = pd.DataFrame(rows)

    signs, mean_ic = estimated_feature_signs(
        train,
        ["positive_feature", "negative_feature"],
    )

    assert signs.to_dict() == {"positive_feature": 1, "negative_feature": -1}
    assert mean_ic["positive_feature"] > 0
    assert mean_ic["negative_feature"] < 0


def test_literature_feature_signs_cover_declared_composite_inputs():
    signs = literature_feature_signs(
        [
            "book_to_market_rank",
            "return_1m_rank",
            "comp_gross_profitability_rank",
            "comp_log_volume_rank",
        ]
    )

    assert signs.to_dict() == {
        "book_to_market_rank": 1,
        "return_1m_rank": -1,
        "comp_gross_profitability_rank": 1,
        "comp_log_volume_rank": -1,
    }


def test_fixed_count_leg_builder_selects_exact_top_and_bottom_names():
    month = pd.DataFrame(
        {
            "ric": [f"S{security:02d}" for security in range(10)],
            "score": np.arange(10, dtype=float),
            "target_return_1m": np.linspace(-0.05, 0.05, 10),
        }
    )

    legs = build_fixed_count_long_short_legs(month, "score", leg_size=3)
    weights = legs.set_index("ric")["weight"]

    assert set(legs.loc[legs["side"].eq("long"), "ric"]) == {"S07", "S08", "S09"}
    assert set(legs.loc[legs["side"].eq("short"), "ric"]) == {"S00", "S01", "S02"}
    assert len(legs) == 6
    assert np.isclose(weights[weights.gt(0)].sum(), 1.0)
    assert np.isclose(weights[weights.lt(0)].sum(), -1.0)
    assert np.isclose(weights.sum(), 0.0)
    assert np.isclose(fixed_count_one_way_turnover(weights, None), 1.0)
    assert np.isclose(fixed_count_one_way_turnover(weights, weights), 0.0)


def test_bucket_assignment_builds_disjoint_terciles_and_top_overlay():
    date = pd.Timestamp("2021-01-31")
    frame = pd.DataFrame(
        {
            "date": date,
            "ric": [f"S{security:02d}" for security in range(60)],
            "company_market_cap": np.arange(1, 61, dtype=float),
        }
    )

    labelled = assign_buckets(frame, "company_market_cap", "market_cap", top_n=5)
    terciles = labelled[labelled["bucket"].isin(["low_cap", "mid_cap", "high_cap"])]
    top = labelled[labelled["bucket"].eq("top_5_cap")]

    assert terciles.groupby("bucket")["ric"].nunique().to_dict() == {
        "high_cap": 20,
        "low_cap": 20,
        "mid_cap": 20,
    }
    assert not terciles.duplicated(["date", "ric"]).any()
    assert set(top["ric"]) == {"S55", "S56", "S57", "S58", "S59"}
