from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_coverage_selection_test.py"
)
SPEC = importlib.util.spec_from_file_location("run_coverage_selection_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MONTHS = 36
NAMES = 60


def _cells(
    *,
    estimates_signal_by_group: dict[bool, float],
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two matched cells where the estimates lift can differ across groups.

    Half the names are tagged as thinly covered (weight-heavy under inverse
    propensity weighting), so a lift concentrated in that half must show up more
    strongly once the weights are applied.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-31", periods=MONTHS, freq="ME")
    compustat_rows = []
    estimates_rows = []
    for date in dates:
        rics = [f"S{index:03d}" for index in range(NAMES)]
        thin = np.array([index % 2 == 0 for index in range(NAMES)])
        signal = rng.normal(size=NAMES)
        noise = rng.normal(size=NAMES)
        target = 0.1 * signal + noise
        extra = np.array([estimates_signal_by_group[flag] for flag in thin])
        common = {"date": date, "ric": rics, "base_model": "ridge", "target_return_1m": target}
        compustat_rows.append(pd.DataFrame({**common, "prediction": signal}))
        estimates_rows.append(
            pd.DataFrame({**common, "prediction": signal + extra * target})
        )
    compustat = pd.concat(compustat_rows, ignore_index=True)
    estimates = pd.concat(estimates_rows, ignore_index=True)
    for frame in (compustat, estimates):
        thin = frame["ric"].str[1:].astype(int).mod(2).eq(0)
        # Thinly covered names are the ones inverse-propensity weighting upweights.
        frame["coverage_weight"] = np.where(thin, 1.6, 0.4)
    return compustat, estimates


def test_matched_cells_pass_and_report_their_stock_months():
    compustat, estimates = _cells(estimates_signal_by_group={True: 0.0, False: 0.0})

    check = MODULE.check_cells_are_matched(compustat, estimates)

    assert check["identical_cells"]
    assert check["shared_stock_months"] == MONTHS * NAMES


def test_unmatched_cells_are_refused():
    compustat, estimates = _cells(estimates_signal_by_group={True: 0.0, False: 0.0})

    with pytest.raises(SystemExit, match="not coverage-matched"):
        MODULE.check_cells_are_matched(compustat, estimates.iloc[10:])


def test_weighting_lifts_an_effect_that_sits_in_the_upweighted_names():
    compustat, estimates = _cells(estimates_signal_by_group={True: 0.35, False: 0.0})

    unweighted, _ = MODULE.data_depth_table(
        compustat, estimates, weight_column=None, hac_lags=6, label="unweighted"
    )
    weighted, _ = MODULE.data_depth_table(
        compustat,
        estimates,
        weight_column="coverage_weight",
        hac_lags=6,
        label="inverse_propensity",
    )

    assert unweighted["estimate"].iloc[0] > 0
    assert weighted["estimate"].iloc[0] > unweighted["estimate"].iloc[0]
    assert weighted["weighting"].iloc[0] == "inverse_propensity"


def test_weighting_shrinks_an_effect_that_sits_in_the_downweighted_names():
    compustat, estimates = _cells(estimates_signal_by_group={True: 0.0, False: 0.35})

    unweighted, _ = MODULE.data_depth_table(
        compustat, estimates, weight_column=None, hac_lags=6, label="unweighted"
    )
    weighted, _ = MODULE.data_depth_table(
        compustat,
        estimates,
        weight_column="coverage_weight",
        hac_lags=6,
        label="inverse_propensity",
    )

    assert weighted["estimate"].iloc[0] < unweighted["estimate"].iloc[0]


def test_data_depth_table_returns_the_monthly_ics_behind_the_test():
    compustat, estimates = _cells(estimates_signal_by_group={True: 0.2, False: 0.2})

    table, ics = MODULE.data_depth_table(
        compustat, estimates, weight_column=None, hac_lags=6, label="unweighted"
    )

    assert set(ics["cell"]) == {"compustat_only", "compustat_plus_estimates"}
    assert len(ics) == 2 * MONTHS
    assert table["months"].iloc[0] == MONTHS
