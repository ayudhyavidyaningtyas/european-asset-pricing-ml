"""Lag ladder for the analyst-estimates data-depth effect.

The headline runs stale-date every analyst snapshot by one month. That is enough
to rule out same-month look-ahead, but it says nothing about how quickly the
information decays -- and a signal that is worth nothing one month later is a
different economic object from one that persists.

This script reads the coverage-matched {Compustat only, Compustat + analyst}
cells at each lag on the ladder and reports the data-depth effect per lag, both
on each lag's own matched sample and on the stock-months common to every lag, so
the decay is not confounded by the sample shifting as the lag lengthens. The
longest lag is a falsification boundary rather than a candidate strategy: a
month-6 snapshot should carry little, and an effect that is flat out to month 6
would suggest the estimates layer is proxying for something slow-moving rather
than for news.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from estimates_identification import hac_mean, holm_within, monthly_ic  # noqa: E402

RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
DEFAULT_COMPUSTAT_TEMPLATE = str(
    RESULTS_ROOT / "estimates_lag_ladder_20260816_compustat_lag{lag}"
)
DEFAULT_ESTIMATES_TEMPLATE = str(
    RESULTS_ROOT / "estimates_lag_ladder_20260816_estimates_lag{lag}"
)
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "estimates_lag_ladder_20260816"

PREDICTION_COLUMNS = ["date", "ric", "base_model", "prediction", "target_return_1m"]


def load_predictions(directory: Path, models: list[str]) -> pd.DataFrame:
    path = directory / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions: {path}")
    frame = pd.read_parquet(path, columns=PREDICTION_COLUMNS)
    frame = frame[frame["base_model"].isin(models)]
    return frame.dropna(subset=["prediction", "target_return_1m"])


def stock_months(frame: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, frame[["ric", "date"]].drop_duplicates().to_numpy()))


def restrict(frame: pd.DataFrame, keys: set[tuple]) -> pd.DataFrame:
    mask = pd.MultiIndex.from_frame(frame[["ric", "date"]]).isin(
        pd.MultiIndex.from_tuples(sorted(keys), names=["ric", "date"])
    )
    return frame[mask].copy()


def lag_effects(
    compustat: pd.DataFrame,
    estimates: pd.DataFrame,
    *,
    lag: int,
    scope: str,
    hac_lags: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ics = pd.concat(
        [
            monthly_ic(compustat).assign(cell="compustat_only"),
            monthly_ic(estimates).assign(cell="compustat_plus_estimates"),
        ],
        ignore_index=True,
    ).assign(lag_months=lag, sample_scope=scope)
    wide = ics.pivot_table(index="date", columns=["cell", "base_model"], values="ic")
    records = []
    for model in sorted(compustat["base_model"].unique()):
        if ("compustat_only", model) not in wide.columns:
            continue
        if ("compustat_plus_estimates", model) not in wide.columns:
            continue
        difference = (
            wide[("compustat_plus_estimates", model)] - wide[("compustat_only", model)]
        )
        records.append(
            {
                "lag_months": lag,
                "sample_scope": scope,
                "model": model,
                "stock_months": int(
                    compustat[["ric", "date"]].drop_duplicates().shape[0]
                ),
                "mean_ic_compustat_only": float(wide[("compustat_only", model)].mean()),
                "mean_ic_with_estimates": float(
                    wide[("compustat_plus_estimates", model)].mean()
                ),
                **hac_mean(difference, hac_lags, "data_depth_effect"),
            }
        )
    return pd.DataFrame(records), ics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lags", nargs="+", type=int, default=[1, 2, 3, 6])
    parser.add_argument("--compustat-template", default=DEFAULT_COMPUSTAT_TEMPLATE)
    parser.add_argument("--estimates-template", default=DEFAULT_ESTIMATES_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=["ridge", "hist_gbm", "mlp", "dre"])
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument(
        "--allow-cell-mismatch",
        action="store_true",
        help="Intersect a lag's two cells instead of failing. Records the drop.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    sample_checks: list[dict[str, object]] = []
    for lag in args.lags:
        compustat_dir = Path(args.compustat_template.format(lag=lag))
        estimates_dir = Path(args.estimates_template.format(lag=lag))
        compustat = load_predictions(compustat_dir, args.models)
        estimates = load_predictions(estimates_dir, args.models)
        keys_compustat = stock_months(compustat)
        keys_estimates = stock_months(estimates)
        shared = keys_compustat & keys_estimates
        check = {
            "lag_months": lag,
            "compustat_dir": str(compustat_dir),
            "estimates_dir": str(estimates_dir),
            "compustat_stock_months": len(keys_compustat),
            "estimates_stock_months": len(keys_estimates),
            "shared_stock_months": len(shared),
            "identical": keys_compustat == keys_estimates,
        }
        if not check["identical"]:
            if not args.allow_cell_mismatch:
                raise SystemExit(
                    f"Lag {lag} cells do not share identical stock-months ({check}); "
                    "rerun with --allow-cell-mismatch to intersect."
                )
            compustat = restrict(compustat, shared)
            estimates = restrict(estimates, shared)
        sample_checks.append(check)
        cells[lag] = (compustat, estimates)

    common = set.intersection(*(stock_months(pair[0]) for pair in cells.values()))

    effect_frames = []
    ic_frames = []
    for lag, (compustat, estimates) in cells.items():
        own_effects, own_ics = lag_effects(
            compustat, estimates, lag=lag, scope="own_matched_sample", hac_lags=args.hac_lags
        )
        common_effects, common_ics = lag_effects(
            restrict(compustat, common),
            restrict(estimates, common),
            lag=lag,
            scope="common_across_lags",
            hac_lags=args.hac_lags,
        )
        effect_frames.extend([own_effects, common_effects])
        ic_frames.extend([own_ics, common_ics])

    effects = pd.concat(effect_frames, ignore_index=True)
    effects = holm_within(effects, ["sample_scope", "model"])
    reference = (
        effects[effects["lag_months"].eq(min(args.lags))]
        .set_index(["sample_scope", "model"])["estimate"]
        .rename("lag_min_estimate")
    )
    effects = effects.join(reference, on=["sample_scope", "model"])
    effects["share_of_shortest_lag"] = effects["estimate"].div(
        effects["lag_min_estimate"].replace(0.0, np.nan)
    )
    effects = effects.drop(columns=["lag_min_estimate"])

    monthly = pd.concat(ic_frames, ignore_index=True)
    effects.to_csv(args.output_dir / "lag_ladder_data_depth.csv", index=False)
    monthly.to_csv(args.output_dir / "lag_ladder_monthly_ics.csv", index=False)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "lags": args.lags,
        "models": args.models,
        "hac_lags": args.hac_lags,
        "sample_checks": sample_checks,
        "common_stock_months": len(common),
        "primary_scope": "common_across_lags",
        "sample_scopes": {
            "own_matched_sample": (
                "each lag uses its own coverage-matched stock-months, so the "
                "sample tracks who is covered at that lag; secondary, since "
                "sample drift can masquerade as decay"
            ),
            "common_across_lags": (
                "stock-months present at every lag, so the decay is measured "
                "on a fixed universe; the primary read for decay claims"
            ),
        },
        "interpretation": (
            "A positive effect that decays with the lag is consistent with the "
            "analyst layer carrying news; an effect that vanishes at lag 2 is "
            "fragile, and one that is flat out to the longest lag points at a "
            "slow-moving characteristic rather than news."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(json.dumps(sample_checks, indent=2, default=str))
    print(
        effects[
            [
                "sample_scope",
                "lag_months",
                "model",
                "estimate",
                "t_stat",
                "p_value",
                "p_value_holm",
                "share_of_shortest_lag",
            ]
        ].to_string(index=False)
    )
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
