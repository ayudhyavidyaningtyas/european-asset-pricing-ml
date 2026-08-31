from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_deep_regression_ensemble.py"
)
SPEC = importlib.util.spec_from_file_location("run_deep_regression_ensemble", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ex_ante_dre_runner_defaults_lock_prespecified_design():
    args = MODULE.build_parser().parse_args([])
    config = MODULE.build_config(args)
    manifest = MODULE.ex_ante_manifest(args, config)

    assert args.panel.name == "monthly_feature_panel_estimates.parquet"
    assert args.feature_set == "estimates_enriched"
    assert args.models == ["momentum", "ridge", "hist_gbm", "mlp", "dre"]
    assert args.targets == ["rank"]
    assert args.sample_start_date == "2005-01-31"
    assert args.require_estimates_feature is False
    assert args.require_estimate_signal_lag_months is None
    assert config.dre_tune_final_alpha is True
    assert config.dre_layers == 2
    assert config.dre_features_per_block == 64
    assert config.dre_final_alphas == (0.1, 1.0, 10.0, 100.0)
    assert manifest["purpose"] == "prespecified_deep_regression_ensemble_screen"
