from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from summarize_aipm_full_adaptation import (  # noqa: E402
    AIPMRunSpec,
    collect_implementability,
    collect_model_summaries,
    collect_pairwise_comparisons,
    summarize_attention_mechanism,
    write_bundle,
)


def _write_run(path: Path, *, blocks: int, label_value: float = 0.08) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["own_asset_mlp", "nonlinear_transformer"],
            "months": [12, 12],
            "annualized_return": [0.07, label_value],
            "annualized_volatility": [0.10, 0.11],
            "sharpe": [0.70, label_value / 0.11],
            "average_monthly_turnover": [0.2, 0.3],
            "normalized_hjd_pricing_error": [0.20, 0.18],
        }
    ).to_csv(path / "aipm_full_summary.csv", index=False)
    pd.DataFrame(
        {
            "model": ["own_asset_mlp", "nonlinear_transformer"],
            "refit_id": ["r1", "r1"],
            "n_parameters": [100, 200 + blocks],
            "fit_seconds": [1.0, 2.0],
        }
    ).to_csv(path / "aipm_full_fit_log.csv", index=False)
    pd.DataFrame(
        {
            "model": ["nonlinear_transformer"],
            "baseline": ["own_asset_mlp"],
            "months": [12],
            "annualized_mean_difference": [label_value - 0.07],
            "annualized_difference_volatility": [0.03],
            "difference_sharpe": [0.1],
            "correlation": [0.9],
            "alpha_annualized": [0.01],
            "alpha_hac_t": [1.2],
            "alpha_hac_p": [0.23],
            "beta": [0.9],
        }
    ).to_csv(path / "aipm_full_comparisons.csv", index=False)
    manifest = {
        "config": {
            "first_test_year": 2020,
            "last_test_year": 2021,
            "refit_frequency": "annual",
            "training_window_months": 12,
            "validation_months": 3,
            "max_monthly_stocks": 100,
            "transformer_blocks": blocks,
            "attention_heads": 1,
            "feedforward_width": 8,
            "epochs": 2,
            "seeds": [0],
        },
        "rows": {"weights": 240, "attention_examples": 120},
        "causality_check": {
            "train_target_after_cutoff": 0,
            "validation_target_after_cutoff": 0,
            "duplicate_weight_security_months": 0,
        },
    }
    (path / "aipm_full_manifest.json").write_text(json.dumps(manifest))


def _write_post(path: Path) -> None:
    path.mkdir(parents=True)
    pd.DataFrame(
        {
            "model": ["own_asset_mlp", "nonlinear_transformer"],
            "aum_label": ["100m", "100m"],
            "aum_eur": [100_000_000.0, 100_000_000.0],
            "months": [12, 12],
            "annualized_gross_return": [0.07, 0.08],
            "annualized_net_return": [0.06, 0.075],
            "annualized_net_volatility": [0.1, 0.11],
            "net_sharpe": [0.6, 0.68],
            "average_monthly_turnover": [0.2, 0.3],
            "annualized_total_cost": [0.01, 0.005],
            "spread_observed_weight": [1.0, 0.99],
            "mean_half_spread_bps": [6.0, 7.0],
        }
    ).to_csv(path / "aipm_implementability_summary.csv", index=False)
    pd.DataFrame(
        {
            "signal_date": ["2020-01-31", "2020-02-29"],
            "metric": [
                "same_trbceconomicsector_weighted_mean",
                "abs_diff_log_size_rank_weighted_mean",
            ],
            "observed": [0.2, 0.1],
            "null": [0.1, 0.4],
            "lift": [0.1, -0.3],
        }
    ).to_csv(path / "aipm_attention_lift.csv", index=False)


def test_aipm_bundle_collates_core_outputs(tmp_path):
    run = tmp_path / "run"
    post = tmp_path / "post"
    _write_run(run, blocks=2)
    _write_post(post)
    specs = [AIPMRunSpec("depth2_test", run, post, "unit test")]

    summary = collect_model_summaries(specs)
    comparisons = collect_pairwise_comparisons(specs)
    implementability = collect_implementability(specs, "100m")
    attention = summarize_attention_mechanism(post)

    assert set(summary["model"]) == {"own_asset_mlp", "nonlinear_transformer"}
    assert summary["train_target_after_cutoff"].eq(0).all()
    assert comparisons.loc[0, "model"] == "nonlinear_transformer"
    assert len(implementability) == 2
    assert set(attention["metric"]) == {
        "same_trbceconomicsector_weighted_mean",
        "abs_diff_log_size_rank_weighted_mean",
    }


def test_write_bundle_creates_depth_scaling_and_brief(tmp_path):
    run1 = tmp_path / "depth1"
    run2 = tmp_path / "depth2"
    post = tmp_path / "post"
    _write_run(run1, blocks=1, label_value=0.06)
    _write_run(run2, blocks=2, label_value=0.08)
    _write_post(post)
    output = tmp_path / "bundle"
    specs = [
        AIPMRunSpec("depth1_test", run1, None, "depth"),
        AIPMRunSpec("headline_top500_three_seed", run2, post, "headline"),
    ]

    manifest = write_bundle(specs, output, "100m")
    depth = pd.read_csv(output / "depth_scaling_summary.csv")

    assert manifest["rows"]["model_hierarchy_summary"] == 4
    assert not depth.empty
    assert (output / "FULL_AIPM_ADAPTATION_BRIEF.md").exists()
