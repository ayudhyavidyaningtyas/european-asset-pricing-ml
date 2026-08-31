from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_pairwise_sharpe_inference.py"
)
SPEC = importlib.util.spec_from_file_location("run_pairwise_sharpe_inference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_monthly(path: Path, model: str, offset: float) -> None:
    path.mkdir(parents=True)
    months = pd.date_range("2020-02-29", periods=30, freq="ME")
    pd.DataFrame(
        [
            {
                "model": model,
                "target_mode": "rank",
                "weighting": "value",
                "universe_variant": "standard_ex_bottom_5pct",
                "signal_date": str(month - pd.offsets.MonthEnd(1))[:10],
                "return_date": str(month.date()),
                "gross_long_short_return": 0.01 + offset + (index % 4) * 0.002,
                "long_short_turnover": 0.20,
                "long_return": 0.02 + offset + (index % 4) * 0.002,
                "long_only_turnover": 0.10,
            }
            for index, month in enumerate(months)
        ]
    ).to_csv(path / "monthly_portfolios.csv", index=False)


def test_infer_pair_writes_sharpe_difference_fields(tmp_path: Path):
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    _write_monthly(run_a, "hist_gbm_rank", 0.01)
    _write_monthly(run_b, "ridge_rank", 0.00)

    result = MODULE.infer_pair(
        "hist_gbm_minus_ridge",
        run_a,
        "hist_gbm_rank",
        run_b,
        "ridge_rank",
        weighting="value",
        universe_variant="standard_ex_bottom_5pct",
        portfolio="long_short",
        cost_bps=25,
        expected_block=6,
        n_boot=100,
        seed=1,
        min_months=24,
    )

    assert result["comparison"] == "hist_gbm_minus_ridge"
    assert result["months"] == 30
    assert "bootstrap_ci_low" in result
    assert "jkm_p_two_sided" in result


def test_pairwise_interpretation_flags_identify_raw_only_significance(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "bootstrap_p_two_sided": [0.02, 0.20],
            "bootstrap_p_two_sided_holm": [0.10, 0.20],
            "bootstrap_ci_includes_zero": [False, True],
        }
    )

    frame["bootstrap_raw_significant_5pct"] = frame["bootstrap_p_two_sided"].le(0.05)
    frame["bootstrap_holm_significant_5pct"] = frame[
        "bootstrap_p_two_sided_holm"
    ].le(0.05)
    frame["bootstrap_ci_excludes_zero"] = ~frame[
        "bootstrap_ci_includes_zero"
    ].astype(bool)
    frame["bootstrap_ci_excludes_zero_but_holm_not_significant"] = (
        frame["bootstrap_ci_excludes_zero"]
        & ~frame["bootstrap_holm_significant_5pct"]
    )

    assert frame["bootstrap_raw_significant_5pct"].tolist() == [True, False]
    assert frame["bootstrap_holm_significant_5pct"].tolist() == [False, False]
    assert frame[
        "bootstrap_ci_excludes_zero_but_holm_not_significant"
    ].tolist() == [True, False]
