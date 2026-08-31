from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_validation_selected_implementable_strategy import (  # noqa: E402
    SelectorConfig,
    select_strategy_monthly,
)


def synthetic_ladder_monthly() -> pd.DataFrame:
    records = []
    dates = pd.date_range("2020-01-31", periods=30, freq="ME")
    for index, date in enumerate(dates):
        for model in ["momentum_rank", "deep_rank"]:
            if model == "deep_rank":
                # Deep is slightly better in the completed validation history,
                # then collapses in the first selectable current month. A
                # leaky selector would avoid it.
                net_return = 0.010 if index < 24 else -1.000
            else:
                net_return = 0.009
            records.append(
                {
                    "model": model,
                    "target_mode": "rank",
                    "date": date,
                    "target_date": date + pd.offsets.MonthEnd(1),
                    "rung": "top_500_observed_spread",
                    "weighting": "value",
                    "portfolio": "long_short",
                    "universe_n": 500,
                    "gross_return": net_return,
                    "observed_spread_fraction": 1.0,
                    "median_half_spread_bps": 3.0,
                    "turnover_100m": 1.0,
                    "spread_cost_100m": 0.0,
                    "impact_cost_100m": 0.0,
                    "net_return_100m": net_return,
                }
            )
    return pd.DataFrame(records)


def test_validation_selector_uses_only_completed_prior_returns():
    selected = select_strategy_monthly(
        synthetic_ladder_monthly(),
        portfolio="long_short",
        rungs=["top_500_observed_spread"],
        candidate_models=["momentum_rank", "deep_rank"],
        config=SelectorConfig(
            validation_months=60,
            minimum_validation_months=24,
            objective="certainty_equivalent",
            aum_label="100m",
        ),
    )

    assert not selected.empty
    first = selected.sort_values("date").iloc[0]
    assert first["model"] == "deep_rank"
    assert first["net_return_100m"] == -1.0
