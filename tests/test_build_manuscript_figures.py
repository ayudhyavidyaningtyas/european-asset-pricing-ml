from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_manuscript_figures.py"
)
SPEC = importlib.util.spec_from_file_location("build_manuscript_figures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_implementability_ladder_data_combines_unconstrained_and_constrained_cis(
    tmp_path: Path,
):
    revision_summary = tmp_path / "model_summary.csv"
    pd.DataFrame(
        {
            "model": ["ridge_rank", "ridge_rank"],
            "weighting": ["equal", "value"],
            "universe_variant": [
                "standard_ex_bottom_5pct",
                "standard_ex_bottom_5pct",
            ],
            "portfolio": ["long_short", "long_short"],
            "cost_bps": [25, 25],
            "annualized_net_mean_return": [0.14, 0.09],
            "annualized_net_mean_return_ci_low": [0.08, 0.01],
            "annualized_net_mean_return_ci_high": [0.20, 0.17],
            "net_sharpe": [1.2, 0.6],
            "net_sharpe_ci_low": [0.5, -0.1],
            "net_sharpe_ci_high": [1.9, 1.2],
        }
    ).to_csv(revision_summary, index=False)
    portfolio_cis = tmp_path / "portfolio_level_bootstrap_cis.csv"
    rows = []
    for aum, ret, sharpe in [
        ("10m", 0.15, 0.85),
        ("100m", 0.14, 0.80),
        ("500m", 0.13, 0.72),
    ]:
        rows.extend(
            [
                {
                    "object_class": "constrained_long_only",
                    "strategy": "fixed_pure_revision_signal_smooth75_ridge_top500_observed",
                    "portfolio": "long_only",
                    "aum_label": aum,
                    "metric": "annualized_net_return",
                    "point": ret,
                    "ci_low": ret - 0.05,
                    "ci_high": ret + 0.05,
                },
                {
                    "object_class": "constrained_long_only",
                    "strategy": "fixed_pure_revision_signal_smooth75_ridge_top500_observed",
                    "portfolio": "long_only",
                    "aum_label": aum,
                    "metric": "net_sharpe",
                    "point": sharpe,
                    "ci_low": sharpe - 0.3,
                    "ci_high": sharpe + 0.3,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(portfolio_cis, index=False)

    data = MODULE.implementability_ladder_data(revision_summary, portfolio_cis)

    assert data["label"].tolist() == [
        "EW long-short",
        "VW long-short",
        "Constrained 10m",
        "Constrained 100m",
        "Constrained 500m",
    ]
    assert data.loc[0, "net_sharpe"] == pytest.approx(1.2)
    assert data.loc[3, "annualized_net_return"] == pytest.approx(0.14)


def _write_lag_summary(root: Path, lag: int) -> None:
    run_dir = root / f"lag_sensitivity_pure_revisions_lag{lag}_ridge"
    run_dir.mkdir()
    rows = []
    for weighting, universe in MODULE.LAG_CELLS:
        for portfolio in ["long_short", "long_only_top_decile"]:
            rows.append(
                {
                    "model": "ridge_rank",
                    "target_mode": "rank",
                    "weighting": weighting,
                    "universe_variant": universe,
                    "portfolio": portfolio,
                    "cost_bps": 25,
                    "mean_monthly_spearman_ic": 0.05 / lag,
                    "annualized_net_mean_return": (
                        0.12 / lag if portfolio == "long_short" else 0.10 / lag
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(run_dir / "model_summary.csv", index=False)


def test_lag_decay_data_reads_ic_once_and_each_portfolio_cell(tmp_path: Path):
    _write_lag_summary(tmp_path, 1)
    _write_lag_summary(tmp_path, 2)

    data = MODULE.lag_decay_data(
        tmp_path,
        "lag_sensitivity_pure_revisions_lag{lag}_ridge",
        (1, 2),
    )

    assert len(data[data["metric"].eq("monthly_ic")]) == 2
    assert len(data[data["metric"].eq("long_short_net_return")]) == 8
    assert len(data[data["metric"].eq("long_only_net_return")]) == 8
    assert set(data["lag"]) == {1, 2}
