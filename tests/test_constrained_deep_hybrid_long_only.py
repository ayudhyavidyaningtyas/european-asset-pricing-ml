from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_constrained_deep_hybrid_long_only import (  # noqa: E402
    ConstraintSpec,
    solve_constrained_long_only,
)


def synthetic_universe() -> pd.DataFrame:
    records = []
    for index in range(80):
        records.append(
            {
                "ric": f"S{index:03d}",
                "prediction": float(index),
                "target_return_1m": 0.001 * index,
                "screen_country": ["GB", "FR", "DE", "CH"][index % 4],
                "TR.TRBCECONOMICSECTOR": ["Tech", "Health", "Industry", "Finance"][
                    (index // 4) % 4
                ],
                "half_spread_bps": 5.0,
                "spread_observed": True,
                "adv_eur": 1_000_000.0,
                "idio_vol_36m": 0.20,
            }
        )
    return pd.DataFrame(records)


def test_constrained_optimizer_respects_name_country_and_sector_caps():
    spec = ConstraintSpec(
        "test",
        max_name_weight=0.05,
        max_country_weight=0.35,
        max_sector_weight=0.35,
        turnover_penalty=0.0,
    )
    weights, status = solve_constrained_long_only(synthetic_universe(), {}, spec)

    assert status == "ok"
    weight_series = pd.Series(weights)
    assert np.isclose(weight_series.sum(), 1.0)
    assert weight_series.max() <= spec.max_name_weight + 1e-6

    holdings = synthetic_universe().set_index("ric").reindex(weight_series.index)
    country = weight_series.groupby(holdings["screen_country"]).sum()
    sector = weight_series.groupby(holdings["TR.TRBCECONOMICSECTOR"]).sum()
    assert country.max() <= spec.max_country_weight + 1e-6
    assert sector.max() <= spec.max_sector_weight + 1e-6


def test_turnover_penalty_keeps_previous_winners_when_scores_shift_slightly():
    universe = synthetic_universe()
    spec_no_penalty = ConstraintSpec("free", 0.10, 0.50, 0.50, 0.0)
    spec_penalty = ConstraintSpec("sticky", 0.10, 0.50, 0.50, 0.05)
    prior_names = universe.tail(10)["ric"].tolist()
    prior = {ric: 0.10 for ric in prior_names}
    shifted = universe.copy()
    shifted.loc[:9, "prediction"] = shifted["prediction"].max() + 1.0

    free_weights, free_status = solve_constrained_long_only(shifted, prior, spec_no_penalty)
    sticky_weights, sticky_status = solve_constrained_long_only(shifted, prior, spec_penalty)

    assert free_status == "ok"
    assert sticky_status == "ok"
    prior_overlap_free = sum(free_weights.get(ric, 0.0) for ric in prior_names)
    prior_overlap_sticky = sum(sticky_weights.get(ric, 0.0) for ric in prior_names)
    assert prior_overlap_sticky > prior_overlap_free
