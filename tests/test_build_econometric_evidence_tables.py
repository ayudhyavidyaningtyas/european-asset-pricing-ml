from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_econometric_evidence_tables.py"
)
SPEC = importlib.util.spec_from_file_location("build_econometric_evidence_tables", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stationary_bootstrap_metric_ci_reports_information_ratio():
    active = pd.Series([0.01, 0.02, -0.005, 0.015, 0.0, 0.012, -0.004, 0.018])

    result = MODULE.stationary_bootstrap_metric_ci(
        active,
        metric="information_ratio",
        expected_block=3,
        n_boot=200,
        seed=7,
    )

    assert result["observations"] == len(active)
    assert result["point"] == pytest.approx(MODULE.sharpe_ratio(active))
    assert result["ci_low"] < result["ci_high"]
    assert 0.0 <= result["p_two_sided_zero"] <= 1.0


def _write_lag_run(root: Path, lag: int, returns: list[float]) -> None:
    run_dir = root / f"lag_sensitivity_pure_revisions_lag{lag}_ridge"
    run_dir.mkdir()
    dates = pd.date_range("2021-01-31", periods=len(returns), freq="ME")
    prediction_rows = []
    for date, return_value in zip(dates, returns, strict=True):
        for security in range(6):
            signal = security + return_value
            prediction_rows.append(
                {
                    "date": date,
                    "ric": f"S{security}",
                    "model": "ridge_rank",
                    "target_mode": "rank",
                    "prediction": signal,
                    "target_return_rank": security,
                }
            )
    pd.DataFrame(prediction_rows).to_parquet(run_dir / "predictions.parquet")
    pd.DataFrame(
        {
            "model": ["ridge_rank"] * len(dates),
            "target_mode": ["rank"] * len(dates),
            "weighting": ["equal"] * len(dates),
            "universe_variant": ["ex_bottom_20pct"] * len(dates),
            "signal_date": dates,
            "return_date": dates + pd.offsets.MonthEnd(1),
            "gross_long_short_return": returns,
            "long_short_turnover": [1.0] * len(dates),
            "long_return": [value + 0.01 for value in returns],
            "long_only_turnover": [0.5] * len(dates),
        }
    ).to_csv(run_dir / "monthly_portfolios.csv", index=False)


def test_lag_sensitivity_tests_keep_ic_long_short_and_long_only_separate(tmp_path: Path):
    _write_lag_run(tmp_path, 1, [0.03, 0.02, 0.01, 0.00, 0.02, 0.01, 0.03, 0.02])
    _write_lag_run(tmp_path, 2, [0.01, 0.00, -0.01, 0.00, 0.01, -0.02, 0.00, 0.01])

    tests = MODULE.build_lag_sensitivity_paired_tests(
        results_root=tmp_path,
        run_template="lag_sensitivity_pure_revisions_lag{lag}_ridge",
        lags=(1, 2),
        model="ridge_rank",
        weighting="equal",
        universe_variant="ex_bottom_20pct",
        cost_bps=25,
        hac_lags=2,
    )

    assert set(tests["test_family"]) == {
        "monthly_ic",
        "net_return_long_short",
        "net_return_long_only",
    }
    assert tests["p_two_sided_holm"].between(0, 1).all()
    long_short = tests[tests["test_family"].eq("net_return_long_short")].iloc[0]
    long_only = tests[tests["test_family"].eq("net_return_long_only")].iloc[0]
    assert long_short["portfolio"] == "long_short"
    assert long_only["portfolio"] == "long_only_top_decile"


def test_lag_sensitivity_tests_emit_requested_portfolio_cells(tmp_path: Path):
    _write_lag_run(tmp_path, 1, [0.03, 0.02, 0.01, 0.00, 0.02, 0.01, 0.03, 0.02])
    _write_lag_run(tmp_path, 2, [0.01, 0.00, -0.01, 0.00, 0.01, -0.02, 0.00, 0.01])
    for lag in [1, 2]:
        path = tmp_path / f"lag_sensitivity_pure_revisions_lag{lag}_ridge" / "monthly_portfolios.csv"
        monthly = pd.read_csv(path)
        value_cell = monthly.assign(
            weighting="value",
            universe_variant="standard_ex_bottom_5pct",
            gross_long_short_return=monthly["gross_long_short_return"] + 0.001,
            long_return=monthly["long_return"] + 0.001,
        )
        pd.concat([monthly, value_cell], ignore_index=True).to_csv(path, index=False)

    tests = MODULE.build_lag_sensitivity_paired_tests(
        results_root=tmp_path,
        run_template="lag_sensitivity_pure_revisions_lag{lag}_ridge",
        lags=(1, 2),
        model="ridge_rank",
        weighting="equal",
        universe_variant="ex_bottom_20pct",
        cost_bps=25,
        hac_lags=2,
        cells=(
            ("equal", "ex_bottom_20pct"),
            ("value", "standard_ex_bottom_5pct"),
        ),
    )

    assert len(tests[tests["test_family"].eq("monthly_ic")]) == 1
    return_tests = tests[tests["test_family"].ne("monthly_ic")]
    assert set(return_tests["portfolio_cell"]) == {
        "equal_ex_bottom_20pct",
        "value_standard_ex_bottom_5pct",
    }
    assert set(return_tests["portfolio"]) == {
        "long_short",
        "long_only_top_decile",
    }


def test_summary_table_uses_conditional_selection_scope_for_validation_ci():
    portfolio_cis = pd.DataFrame(
        {
            "portfolio_object": ["validation_selected_long_only_long_only_100m"],
            "metric": ["net_sharpe"],
            "point": [0.9],
            "ci_low": [0.1],
            "ci_high": [1.5],
            "p_two_sided_zero": [0.04],
            "source_file": ["validation_selected_monthly.csv"],
            "ci_scope": ["conditional_on_saved_selection_path"],
        }
    )
    summary = MODULE.build_econometric_evidence_summary(
        fmb_path=Path("missing.csv"),
        clark_west_path=Path("missing.csv"),
        placebo_path=Path("missing.csv"),
        portfolio_cis=portfolio_cis,
        lag_tests=pd.DataFrame(),
    )

    assert len(summary) == 1
    assert summary["section"].iloc[0] == "selection"
    assert "conditional_on_saved_selection_path" in summary["notes"].iloc[0]


def test_summary_table_leads_with_revision_strategy_rows(tmp_path: Path):
    fmb_path = tmp_path / "fama_macbeth_summary.csv"
    pd.DataFrame(
        {
            "specification": ["characteristics_risk_country_sector"],
            "annualized_score_slope": [0.06],
            "ci_low": [0.004],
            "ci_high": [0.006],
            "t_stat": [7.0],
            "p_value": [0.001],
            "p_value_holm": [0.002],
        }
    ).to_csv(fmb_path, index=False)
    spanning_path = tmp_path / "revision_external_factor_spanning.csv"
    pd.DataFrame(
        {
            "comparison": ["absolute", "absolute"],
            "model": ["ridge_rank", "ridge_rank"],
            "weighting": ["equal", "value"],
            "universe_variant": ["standard_ex_bottom_5pct", "standard_ex_bottom_5pct"],
            "portfolio": ["long_short", "long_short"],
            "cost_bps": [25, 25],
            "alpha_annualized": [0.067, -0.001],
            "alpha_t": [2.87, -0.05],
            "alpha_p": [0.004, 0.96],
            "alpha_p_holm": [0.004, 0.96],
            "beta_WML": [0.78, 1.05],
        }
    ).to_csv(spanning_path, index=False)
    prediction_path = tmp_path / "prediction_metrics.csv"
    pd.DataFrame(
        {
            "model": ["ridge_rank"],
            "target_mode": ["rank"],
            "mean_monthly_spearman_ic": [0.047],
            "ic_information_ratio": [1.9],
            "observations": [1000],
        }
    ).to_csv(prediction_path, index=False)
    ic_path = tmp_path / "ablation_paired_ic_tests.csv"
    pd.DataFrame(
        {
            "test": ["monthly_ic_variant_minus_compustat"],
            "variant": ["estimates_revisions_only"],
            "reference": ["compustat_enriched"],
            "model": ["ridge_rank"],
            "delta_mean_ic": [0.006],
            "hac_t_stat": [3.8],
            "hac_p_two_sided": [0.001],
            "hac_p_holm": [0.01],
        }
    ).to_csv(ic_path, index=False)

    summary = MODULE.build_econometric_evidence_summary(
        fmb_path=fmb_path,
        clark_west_path=Path("missing.csv"),
        placebo_path=Path("missing.csv"),
        portfolio_cis=pd.DataFrame(),
        lag_tests=pd.DataFrame(),
        revision_spanning_path=spanning_path,
        pure_revision_prediction_path=prediction_path,
        ic_ablation_path=ic_path,
    )

    assert summary["section"].tolist()[:5] == [
        "predictability",
        "revision_strategy",
        "revision_strategy",
        "revision_strategy",
        "revision_strategy",
    ]
    assert "Equal-weight revision FF5+WML alpha" in set(summary["evidence"])
    assert "Revision IC increment over Compustat controls" in set(summary["evidence"])
