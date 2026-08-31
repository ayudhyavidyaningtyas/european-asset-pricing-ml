"""Regime diagnostics for sequence/blend asset-pricing predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_depth import DepthConfig, state_dependence_tests  # noqa: E402


DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_blend_experiment"
    / "blend_ladder_subset_predictions.parquet"
)
DEFAULT_MONTHLY = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_blend_experiment"
    / "blend_monthly_portfolios.csv"
)
DEFAULT_STATES = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "depth_analysis"
    / "market_states.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "deep_sequence_regime_diagnostics"
)


def _load_ladder_models(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    models = manifest.get("ladder_models")
    return list(models) if models else None


def run_regime_diagnostics(
    predictions_path: Path,
    monthly_path: Path,
    states_path: Path,
    output_dir: Path,
    models: list[str] | None,
    portfolio_cost_bps: int,
    hac_lags: int,
) -> dict[str, Any]:
    for path in [predictions_path, monthly_path, states_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(predictions_path)
    monthly = pd.read_csv(monthly_path, parse_dates=["signal_date", "return_date"])
    states = pd.read_csv(states_path, parse_dates=["signal_date"])
    if models is not None:
        predictions = predictions[predictions["model"].isin(models)].copy()
        monthly = monthly[monthly["model"].isin(models)].copy()
    predictions["date"] = pd.to_datetime(predictions["date"])
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])

    config = DepthConfig(portfolio_cost_bps=portfolio_cost_bps, hac_lags=hac_lags)
    state_monthly, state_summary, state_contrasts = state_dependence_tests(
        predictions,
        monthly,
        states,
        config,
    )

    state_monthly.to_csv(output_dir / "sequence_state_dependence_monthly.csv", index=False)
    state_summary.to_csv(output_dir / "sequence_state_dependence_summary.csv", index=False)
    state_contrasts.to_csv(
        output_dir / "sequence_state_dependence_contrasts.csv",
        index=False,
    )
    manifest = {
        "inputs": {
            "predictions": str(predictions_path),
            "monthly_portfolios": str(monthly_path),
            "states": str(states_path),
        },
        "models": sorted(predictions["model"].dropna().unique().tolist()),
        "portfolio_cost_bps": portfolio_cost_bps,
        "hac_lags": hac_lags,
        "rows": {
            "predictions": int(len(predictions)),
            "monthly_portfolios": int(len(monthly)),
            "states": int(len(states)),
            "state_monthly": int(len(state_monthly)),
            "state_summary": int(len(state_summary)),
            "state_contrasts": int(len(state_contrasts)),
        },
        "outputs": {
            "state_monthly": str(output_dir / "sequence_state_dependence_monthly.csv"),
            "state_summary": str(output_dir / "sequence_state_dependence_summary.csv"),
            "state_contrasts": str(
                output_dir / "sequence_state_dependence_contrasts.csv"
            ),
        },
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--monthly", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--states", type=Path, default=DEFAULT_STATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--models-from-manifest", type=Path, default=None)
    parser.add_argument("--portfolio-cost-bps", type=int, default=25)
    parser.add_argument("--hac-lags", type=int, default=6)
    args = parser.parse_args()

    models = args.models
    if models is None and args.models_from_manifest is not None:
        models = _load_ladder_models(args.models_from_manifest)
    manifest = run_regime_diagnostics(
        predictions_path=args.predictions,
        monthly_path=args.monthly,
        states_path=args.states,
        output_dir=args.output_dir,
        models=models,
        portfolio_cost_bps=args.portfolio_cost_bps,
        hac_lags=args.hac_lags,
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
