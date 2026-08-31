from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_deep_size_decomposition_exhibit.py"
)
SPEC = importlib.util.spec_from_file_location("build_deep_size_decomposition_exhibit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _synthetic_predictions() -> pd.DataFrame:
    records = []
    for date in pd.date_range("2021-01-31", periods=18, freq="ME"):
        for security in range(60):
            percentile = (security + 1) / 60
            signal = security / 60
            records.append(
                {
                    "date": date,
                    "ric": f"S{security:03d}",
                    "model": "dre_rank",
                    "prediction": signal,
                    "target_return_rank": signal,
                    "target_return_1m": signal / 100,
                    "market_cap_percentile": percentile,
                    "company_market_cap": np.exp(security / 10),
                }
            )
    return pd.DataFrame(records)


def test_size_decomposition_builds_bucket_monthly_and_summary_rows():
    monthly = MODULE.build_monthly_size_decomposition(
        _synthetic_predictions(),
        quantile=0.10,
        cost_bps=25,
        top_n=10,
        minimum_cross_section=5,
    )
    summary = MODULE.summarize_size_decomposition(monthly, hac_lags=3)

    assert {"small", "middle", "large", "top_500_by_market_cap"}.issubset(
        set(monthly["size_bucket"])
    )
    assert monthly["net_long_short_return"].notna().any()
    assert summary["mean_spearman_ic"].max() == 1.0
    assert summary["annualized_net_long_short_return"].notna().all()
