from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investability_ladder import (  # noqa: E402
    LadderConfig,
    investability_rungs,
    paired_ladder_inference,
    simulate_investability_ladder,
    summarize_investability_ladder,
)


def synthetic_ladder_panel() -> pd.DataFrame:
    records = []
    for month_index, date in enumerate(
        pd.date_range("2020-01-31", periods=24, freq="ME")
    ):
        month_scale = 1.0 + 0.25 * np.sin(month_index / 3.0)
        for security in range(100):
            percentile = (security + 1) / 100.0
            for model, target_mode, direction in [
                ("momentum_rank", "rank", -1.0),
                ("signal_return", "return", 1.0),
            ]:
                records.append(
                    {
                        "model": model,
                        "target_mode": target_mode,
                        "date": date,
                        "target_date": date + pd.offsets.MonthEnd(1),
                        "ric": f"S{security:03d}",
                        "prediction": direction * (security - 49.5) / 1000.0,
                        "target_return_1m": (
                            month_scale * (security - 49.5) / 2000.0
                        ),
                        "target_return_rank": 2.0 * percentile - 1.0,
                        "company_market_cap": 100.0 + security,
                        "market_cap_percentile": percentile,
                        "spread_observed": security >= 50,
                        "half_spread_bps": 30.0 - security / 10.0,
                        "adv_eur": 1_000_000.0 + security * 10_000.0,
                        "idio_vol_36m": 0.20,
                    }
                )
    return pd.DataFrame(records)


def test_investability_rungs_are_nested():
    panel = synthetic_ladder_panel()
    month = panel[panel["date"].eq(panel["date"].min())]

    rungs = investability_rungs(month, maximum_assets=50)
    sets = [set(rungs[name]["ric"]) for name in rungs]

    assert all(right.issubset(left) for left, right in zip(sets, sets[1:]))
    assert len(rungs["top_500"]) == 50
    assert rungs["top_500_observed_spread"]["spread_observed"].all()


def test_ladder_resimulates_costs_and_reports_each_rung():
    config = LadderConfig(aum_eur=(100_000_000.0,), maximum_assets=50)
    monthly = simulate_investability_ladder(
        synthetic_ladder_panel(),
        config,
    )
    summary = summarize_investability_ladder(monthly, config)

    assert set(summary["rung"]) == {
        "standard_ex_bottom_5pct",
        "top_70pct_by_market_cap",
        "top_500",
        "top_500_observed_spread",
        "large_low_spread",
    }
    assert summary["annualized_spread_cost"].gt(0).all()
    assert summary["annualized_impact_cost"].gt(0).all()
    assert np.all(
        summary["annualized_net_return"]
        < summary["gross_annualized_return"]
    )


def test_paired_ladder_inference_compares_against_momentum():
    config = LadderConfig(
        aum_eur=(100_000_000.0,),
        maximum_assets=50,
        bootstrap_repetitions=200,
        bootstrap_blocks=(3,),
        random_state=7,
    )
    monthly = simulate_investability_ladder(
        synthetic_ladder_panel(),
        config,
    )
    inference = paired_ladder_inference(monthly, config)

    assert not inference.empty
    assert set(inference["baseline"]) == {"momentum_rank"}
    assert set(inference["model"]) == {"signal_return"}
    assert inference["sharpe_p_two_sided_holm"].between(0, 1).all()
    assert inference["ce_p_two_sided_holm"].between(0, 1).all()
    primary = inference[
        inference["rung"].eq("top_500")
        & inference["weighting"].eq("value")
        & inference["portfolio"].eq("long_only")
        & inference["aum_label"].eq("100m")
    ]
    assert primary["delta_sharpe"].iloc[0] > 0
    assert primary["delta_ce"].iloc[0] > 0
