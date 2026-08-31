from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_validation_selected_constrained_deep_hybrid import (  # noqa: E402
    ConstrainedSelectorConfig,
    select_strategy_monthly,
    validation_scores,
)


def _candidate_rows() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    rows = []
    for idx, date in enumerate(dates):
        for cap, constraint in [(500, "stable"), (2000, "future_winner")]:
            # The broader candidate is bad in validation, then excellent later.
            if constraint == "stable":
                value = 0.01
            elif idx < 24:
                value = -0.01
            else:
                value = 0.10
            rows.append(
                {
                    "strategy": "frozen_deep_hybrid_selector",
                    "constraint": constraint,
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "maximum_assets": cap,
                    "candidate_cell": f"top{cap}_{constraint}",
                    "net_return_100m": value,
                    "turnover_100m": 0.5,
                    "spread_cost_100m": 0.001,
                    "impact_cost_100m": 0.001,
                    "observed_spread_weight": 1.0,
                    "effective_n": 20.0,
                    "top_5_name_weight": 0.25,
                }
            )
    return pd.DataFrame(rows)


def test_validation_selection_uses_only_prior_completed_returns():
    config = ConstrainedSelectorConfig(
        validation_months=36,
        minimum_validation_months=24,
        risk_aversion=3.0,
        bootstrap_repetitions=10,
    )
    selected = select_strategy_monthly(
        _candidate_rows(),
        strategy="frozen_deep_hybrid_selector",
        config=config,
    )

    assert not selected.empty
    first = selected.sort_values("date").iloc[0]
    assert first["constraint"] == "stable"
    assert first["maximum_assets"] == 500


def test_validation_scores_prefer_higher_certainty_equivalent():
    config = ConstrainedSelectorConfig(minimum_validation_months=2)
    validation = pd.DataFrame(
        {
            "maximum_assets": [500, 500, 1000, 1000],
            "constraint": ["a", "a", "b", "b"],
            "net_return_100m": [0.01, 0.01, 0.03, -0.03],
            "turnover_100m": [0.2, 0.2, 0.2, 0.2],
            "spread_cost_100m": [0.0, 0.0, 0.0, 0.0],
            "impact_cost_100m": [0.0, 0.0, 0.0, 0.0],
            "observed_spread_weight": [1.0, 1.0, 1.0, 1.0],
            "effective_n": [20.0, 20.0, 20.0, 20.0],
        }
    )

    scores = validation_scores(validation, config)
    best = scores.sort_values("validation_objective", ascending=False).iloc[0]

    assert best["constraint"] == "a"
    assert np.isfinite(best["validation_certainty_equivalent"])
