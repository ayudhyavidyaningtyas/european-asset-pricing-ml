from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_constrained_estimates_long_only import (  # noqa: E402
    add_benchmark_relative_returns,
    build_choice_panel,
    build_fixed_choice_panel,
    estimate_benchmark_alpha,
    load_choice_calendar_from_predictions,
    parse_fixed_choice,
    summarize_benchmark_relative,
)
from run_constrained_deep_hybrid_long_only import summarize_constrained  # noqa: E402


def test_parse_fixed_choice_requires_strategy_model_and_rung():
    parsed = parse_fixed_choice("fixed_model:smooth25_mlp_rank:large_low_spread")

    assert parsed == {
        "strategy": "fixed_model",
        "model": "smooth25_mlp_rank",
        "rung": "large_low_spread",
    }

    with pytest.raises(ValueError, match="strategy:model:rung"):
        parse_fixed_choice("bad:smooth25_mlp_rank")


def test_build_choice_panel_adds_fixed_rows_on_selected_calendar():
    selected = pd.DataFrame(
        {
            "strategy": ["selected", "selected"],
            "date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "target_date": pd.to_datetime(["2024-02-29", "2024-03-31"]),
            "model": ["smooth25_mlp_rank", "smooth50_mlp_rank"],
            "rung": ["top_500", "large_low_spread"],
        }
    )

    choices = build_choice_panel(
        selected,
        [
            {
                "strategy": "fixed",
                "model": "smooth75_ridge_rank",
                "rung": "top_500_observed_spread",
            }
        ],
    )

    fixed = choices[choices["strategy"].eq("fixed")]
    assert len(choices) == 4
    assert len(fixed) == 2
    assert set(fixed["date"]) == set(selected["date"])
    assert set(fixed["model"]) == {"smooth75_ridge_rank"}
    assert set(fixed["rung"]) == {"top_500_observed_spread"}


def test_fixed_choice_panel_can_use_prediction_calendar(tmp_path: Path):
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29"]),
            "target_date": pd.to_datetime(["2024-02-29", "2024-02-29", "2024-03-31"]),
            "ric": ["AAA", "BBB", "AAA"],
            "model": ["smooth75_ridge_rank"] * 3,
            "prediction": [0.1, 0.2, 0.3],
        }
    )
    path = tmp_path / "predictions.parquet"
    predictions.to_parquet(path, index=False)
    fixed_choices = [
        {
            "strategy": "fixed",
            "model": "smooth75_ridge_rank",
            "rung": "top_500_observed_spread",
        }
    ]

    calendar = load_choice_calendar_from_predictions(path, fixed_choices)
    choices = build_fixed_choice_panel(calendar, fixed_choices)

    assert len(calendar) == 2
    assert len(choices) == 2
    assert choices["strategy"].eq("fixed").all()
    assert choices["model"].eq("smooth75_ridge_rank").all()


def test_add_benchmark_relative_returns_aligns_on_target_date():
    monthly = pd.DataFrame(
        {
            "strategy": ["s"],
            "constraint": ["c"],
            "target_date": pd.to_datetime(["2024-02-29"]),
            "net_return_100m": [0.03],
        }
    )
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-02-29"]),
            "market_return_eur": [0.01],
        }
    )

    result = add_benchmark_relative_returns(monthly, market, (100_000_000.0,))

    assert result.loc[0, "benchmark_return_eur"] == pytest.approx(0.01)
    assert result.loc[0, "active_return_100m"] == pytest.approx(0.02)


def test_benchmark_relative_summary_reports_ir_and_alpha():
    dates = pd.date_range("2021-01-31", periods=36, freq="ME")
    market = pd.Series([0.02, -0.02] * 18)
    portfolio = 0.004 + 0.7 * market
    monthly = pd.DataFrame(
        {
            "strategy": ["strategy"] * len(dates),
            "constraint": ["constraint"] * len(dates),
            "target_date": dates,
            "net_return_100m": portfolio,
        }
    )
    benchmark = pd.DataFrame({"date": dates, "market_return_eur": market})
    relative = add_benchmark_relative_returns(
        monthly,
        benchmark,
        (100_000_000.0,),
    )

    summary = summarize_benchmark_relative(
        relative,
        (100_000_000.0,),
        hac_lags=3,
    )
    full = summary[summary["subperiod"].eq("full")].iloc[0]
    alpha = estimate_benchmark_alpha(
        relative["net_return_100m"],
        relative["benchmark_return_eur"],
        hac_lags=3,
    )

    assert full["annualized_active_return"] == pytest.approx(0.048)
    assert full["information_ratio"] > 0
    assert full["alpha_annualized"] == pytest.approx(0.048)
    assert full["benchmark_beta"] == pytest.approx(0.7)
    assert alpha["alpha_annualized"] == pytest.approx(0.048)


def test_constrained_summary_reports_gross_and_net_sharpe():
    dates = pd.date_range("2021-01-31", periods=12, freq="ME")
    gross = pd.Series([0.03, -0.01] * 6)
    monthly = pd.DataFrame(
        {
            "strategy": ["strategy"] * len(dates),
            "constraint": ["constraint"] * len(dates),
            "target_date": dates,
            "gross_return": gross,
            "net_return_100m": gross - 0.002,
            "turnover_100m": [0.4] * len(dates),
            "spread_cost_100m": [0.001] * len(dates),
            "impact_cost_100m": [0.001] * len(dates),
            "effective_n": [20.0] * len(dates),
            "max_single_name_weight": [0.05] * len(dates),
            "top_5_name_weight": [0.25] * len(dates),
            "max_country_weight": [0.40] * len(dates),
            "max_sector_weight": [0.35] * len(dates),
            "observed_spread_weight": [1.0] * len(dates),
        }
    )

    summary = summarize_constrained(monthly, (100_000_000.0,))
    full = summary[summary["subperiod"].eq("full")].iloc[0]

    assert full["annualized_gross_return"] == pytest.approx(0.12)
    assert full["annualized_net_return"] == pytest.approx(0.096)
    assert full["gross_sharpe"] > full["net_sharpe"]
