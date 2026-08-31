from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_comparison import write_market_comparison_outputs  # noqa: E402


def _write_ml_outputs(path: Path, market_offset: float) -> None:
    path.mkdir(parents=True)
    months = pd.date_range("2020-02-29", periods=30, freq="ME")
    pd.DataFrame(
        [
            {
                "model": "ridge_rank",
                "base_model": "ridge",
                "target_mode": "rank",
                "weighting": "value",
                "universe_variant": "standard_ex_bottom_5pct",
                "portfolio": "long_short",
                "cost_bps": 25,
                "months": 2,
                "observations": 100,
                "mean_monthly_spearman_ic": 0.02 + market_offset,
                "annualized_net_mean_return": 0.10 + market_offset,
                "net_sharpe": 0.50 + market_offset,
                "average_monthly_turnover": 0.30,
                "max_drawdown": -0.05,
            }
        ]
    ).to_csv(path / "model_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "ridge_rank",
                "base_model": "ridge",
                "target_mode": "rank",
                "observations": 100,
                "mean_monthly_spearman_ic": 0.02 + market_offset,
            }
        ]
    ).to_csv(path / "prediction_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "ridge_rank",
                "target_mode": "rank",
                "weighting": "value",
                "universe_variant": "standard_ex_bottom_5pct",
                "signal_date": str(month - pd.offsets.MonthEnd(1))[:10],
                "return_date": str(month.date()),
                "gross_long_short_return": 0.01
                + market_offset
                + (index % 5) * 0.002,
                "long_short_turnover": 0.20,
                "long_return": 0.02 + market_offset + (index % 5) * 0.002,
                "long_only_turnover": 0.10,
            }
            for index, month in enumerate(months)
        ]
    ).to_csv(path / "monthly_portfolios.csv", index=False)


def test_market_comparison_outputs_side_by_side_deltas(tmp_path: Path):
    europe = tmp_path / "europe"
    us = tmp_path / "us"
    output = tmp_path / "comparison"
    _write_ml_outputs(europe, 0.00)
    _write_ml_outputs(us, 0.01)

    manifest = write_market_comparison_outputs(
        {"Europe": europe, "US": us},
        output,
    )
    side_by_side = pd.read_csv(output / "side_by_side_model_summary.csv")
    correlations = pd.read_csv(output / "monthly_return_correlations.csv")

    assert manifest["rows"]["side_by_side_model_summary"] == 1
    assert np.isclose(
        side_by_side["mean_monthly_spearman_ic_us_minus_europe"].iloc[0],
        0.01,
    )
    assert np.isclose(
        side_by_side["annualized_net_mean_return_us_minus_europe"].iloc[0],
        0.01,
    )
    assert "net_sharpe_diff_bootstrap_ci_low" in side_by_side
    assert "net_sharpe_diff_bootstrap_p_two_sided" in side_by_side
    assert (output / "sharpe_difference_tests.csv").exists()
    assert correlations["common_months"].iloc[0] == 30
    assert np.isclose(correlations["return_correlation"].iloc[0], 1.0)
    assert (output / "market_comparison_report.md").exists()
