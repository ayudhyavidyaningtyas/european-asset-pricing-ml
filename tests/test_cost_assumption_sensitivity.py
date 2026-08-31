from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aipm_post_analysis import simulate_weight_implementability  # noqa: E402
from scripts.run_cost_assumption_sensitivity import (  # noqa: E402
    build_scenario_runs,
)


def test_constant_spread_scenarios_rebuild_execution_inputs(tmp_path):
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    panel = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[1], dates[1]],
            "ric": ["A", "B", "A", "B"],
            "company_market_cap": [1_000_000.0, 2_000_000.0] * 2,
            "turnover_12m": [0.02, 0.02] * 2,
            "volatility_12m": [0.10, 0.10] * 2,
            "target_return_1m": [0.01, -0.01, 0.01, -0.01],
        }
    )
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    weights = pd.DataFrame(
        {
            "signal_date": [dates[0], dates[0], dates[1], dates[1]],
            "target_date": [dates[0] + pd.offsets.MonthEnd(1)] * 2
            + [dates[1] + pd.offsets.MonthEnd(1)] * 2,
            "ric": ["A", "B", "A", "B"],
            "model": ["sdf"] * 4,
            "sdf_weight": [0.5, -0.5, 0.5, -0.5],
            "target_return": [0.01, -0.01, 0.01, -0.01],
        }
    )
    scenarios = [
        {"scenario": "constant", "assumed_half_spread_bps": 5.0},
        {"scenario": "constant", "assumed_half_spread_bps": 50.0},
    ]

    runs = build_scenario_runs(
        panel_path,
        None,
        tmp_path / "missing_liquidity",
        scenarios,
        aum_eur=(1_000_000.0,),
        impact_coefficient=0.0,
    )

    assert [run["diagnostics"]["execution_input_uncovered_mean_half_spread_bps"] for run in runs] == [
        5.0,
        50.0,
    ]

    monthly_low = simulate_weight_implementability(
        weights, runs[0]["execution_inputs"], runs[0]["config"]
    )
    monthly_high = simulate_weight_implementability(
        weights, runs[1]["execution_inputs"], runs[1]["config"]
    )

    assert np.allclose(monthly_low["mean_half_spread_bps"], 5.0)
    assert np.allclose(monthly_high["mean_half_spread_bps"], 50.0)
    assert monthly_high["spread_cost"].sum() > monthly_low["spread_cost"].sum()
