from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_revision_lag_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_revision_lag_sensitivity",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_run(root: Path, lag: int) -> None:
    run_dir = root / f"lag_sensitivity_pure_revisions_lag{lag}_ridge"
    run_dir.mkdir(parents=True)
    (run_dir / "ml_manifest.json").write_text(
        json.dumps(
            {
                "sample_filter": {"require_estimate_signal_lag_months": lag},
                "sample_filter_audit": {
                    "after_require_revision_signal": 100 - lag,
                    "model_rows": 80 - lag,
                    "estimate_signal_lag_violations": 0,
                },
                "rows": {"predictions": 70 - lag},
                "causality_check": {"train_target_after_cutoff": 0},
            },
        ),
    )
    pd.DataFrame(
        {
            "model": ["ridge_rank"],
            "target_mode": ["rank"],
            "weighting": ["equal"],
            "universe_variant": ["standard_ex_bottom_5pct"],
            "portfolio": ["long_short"],
            "cost_bps": [25],
            "months": [12],
            "observations": [1000],
            "annualized_net_mean_return": [0.1 / lag],
            "net_sharpe": [1.0 / lag],
            "annualized_gross_mean_return": [0.12 / lag],
            "gross_sharpe": [1.2 / lag],
            "average_monthly_turnover": [0.4],
            "mean_monthly_spearman_ic": [0.02 / lag],
            "ic_information_ratio": [0.5 / lag],
            "positive_ic_month_fraction": [0.7],
            "rank_r2_zero": [0.01],
        },
    ).to_csv(run_dir / "model_summary.csv", index=False)


def test_build_summary_reads_manifest_and_model_summary(tmp_path: Path):
    for lag in [1, 2]:
        _write_run(tmp_path, lag)

    summary = MODULE.build_summary(
        results_root=tmp_path,
        lags=[1, 2],
        run_template="lag_sensitivity_pure_revisions_lag{lag}_ridge",
        model="ridge_rank",
        weighting="equal",
        cost_bps=25,
    )

    assert summary["lag_months"].tolist() == [1, 2]
    assert summary["lag_guard"].tolist() == [1, 2]
    assert summary["lag_violations"].tolist() == [0, 0]
    assert summary["causality_violations"].tolist() == [0, 0]
