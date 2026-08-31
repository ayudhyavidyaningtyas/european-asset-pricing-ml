"""Run follow-up audit checks for the final dissertation assessment.

The checks here are intentionally narrow and write their outputs under
``results/asset_pricing_ml/followup_audit_20260824`` so that the reported
numbers can be traced back to reproducible local artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from asset_pricing import PanelConfig  # noqa: E402
from asset_pricing_external_factors import (  # noqa: E402
    load_external_europe_factors,
    load_monthly_eurusd_return,
)
from asset_pricing_ml import (  # noqa: E402
    FEATURE_SETS,
    WalkForwardConfig,
    construct_monthly_portfolios,
    load_model_panel,
    predictive_accuracy_tests,
    run_walk_forward,
)
from compustat_features import (  # noqa: E402
    _normalise_isin,
    load_compustat_exports,
    prepare_compustat_annual_features,
    prepare_compustat_monthly_features,
)
from implementable_frontier import load_monthly_liquidity  # noqa: E402
from investability_ladder import investability_rungs  # noqa: E402
from run_complexity_spanning_ladder import run_ladder  # noqa: E402
from us_market import (  # noqa: E402
    add_us_isin_from_cusip,
    normalize_wrds_compustat_us_annual,
    normalize_wrds_compustat_us_monthly,
)


RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
OUTPUT_ROOT = RESULTS_ROOT / "followup_audit_20260824"
DATA_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = DATA_ROOT / "processed" / "asset_pricing"
RAW_ROOT = DATA_ROOT / "raw"
FEATURE_SET = "compustat_enriched"
PRIMARY_MODELS = ["momentum", "ridge", "elastic_net", "hist_gbm", "mlp"]
COMPLEXITY_MODELS = ["momentum", "ridge", "dre", "hist_gbm", "mlp"]
SEED_RUNS = [7, 123, 2026]
MDE_NOMINAL_ALPHA = 0.05
MDE_POWER = 0.80


_PRIMARY_PANEL: pd.DataFrame | None = None


def _json_default(value: object) -> object:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _output_dir(name: str) -> Path:
    path = OUTPUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp("M")


def _primary_panel() -> pd.DataFrame:
    global _PRIMARY_PANEL
    if _PRIMARY_PANEL is None:
        feature_columns = FEATURE_SETS[FEATURE_SET]
        _PRIMARY_PANEL = load_model_panel(
            PROCESSED_ROOT / "monthly_feature_panel_compustat.parquet",
            delisting_audit_path=None,
            feature_columns=feature_columns,
            residual_control_set="full",
        )
    return _PRIMARY_PANEL


def _walk_config(
    seed: int,
    *,
    tune: bool,
    dre_layers: int = 2,
    dre_features_per_block: int = 64,
) -> WalkForwardConfig:
    return WalkForwardConfig(
        random_state=seed,
        tune_hyperparameters=tune,
        first_test_year=2015,
        last_test_year=2026,
        validation_months=24,
        mlp_validation_months=24,
        dre_layers=dre_layers,
        dre_features_per_block=dre_features_per_block,
    )


def run_rank_models(
    *,
    seed: int,
    model_names: list[str],
    output_dir: Path,
    tune: bool,
    collect_importance: bool = False,
    dre_layers: int = 2,
    dre_features_per_block: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_path = output_dir / "predictions.parquet"
    fit_log_path = output_dir / "fit_log.csv"
    ic_path = output_dir / "predictive_accuracy_ic_tests.csv"
    if predictions_path.exists() and fit_log_path.exists():
        predictions = pd.read_parquet(predictions_path)
        fit_log = pd.read_csv(fit_log_path)
    else:
        feature_columns = FEATURE_SETS[FEATURE_SET]
        predictions, fit_log, coefficients, importance = run_walk_forward(
            _primary_panel(),
            model_names,
            _walk_config(
                seed,
                tune=tune,
                dre_layers=dre_layers,
                dre_features_per_block=dre_features_per_block,
            ),
            target_column="target_return_rank",
            target_mode="rank",
            feature_columns=feature_columns,
            collect_importance=collect_importance,
        )
        predictions.to_parquet(
            predictions_path,
            index=False,
            compression="zstd",
        )
        fit_log.to_csv(fit_log_path, index=False)
        if not coefficients.empty:
            coefficients.to_csv(output_dir / "coefficients.csv", index=False)
        if not importance.empty:
            importance.to_csv(output_dir / "importance.csv", index=False)
    if ic_path.exists():
        ic_tests = pd.read_csv(ic_path) if ic_path.stat().st_size > 1 else pd.DataFrame()
    else:
        _, ic_tests = predictive_accuracy_tests(predictions)
        ic_tests.to_csv(ic_path, index=False)
    return predictions, fit_log


def _extract_ic_pair(ic_tests: pd.DataFrame, model: str, baseline: str) -> dict[str, Any]:
    pair = ic_tests[
        ic_tests["target_mode"].eq("rank")
        & (
            (
                ic_tests["model_a"].eq(model)
                & ic_tests["model_b"].eq(baseline)
            )
            | (
                ic_tests["model_a"].eq(baseline)
                & ic_tests["model_b"].eq(model)
            )
        )
    ].copy()
    if pair.empty:
        raise RuntimeError(f"Missing IC comparison {model} vs {baseline}")
    row = pair.iloc[0].to_dict()
    sign = 1.0 if row["model_a"] == model else -1.0
    return {
        "model": model,
        "baseline": baseline,
        "mean_ic_difference": sign * float(row["mean_difference"]),
        "holm_p_value": float(row["p_value_holm"]),
        "raw_p_value": float(row["p_value"]),
        "months": int(row["months"]),
    }


def same_seed_determinism() -> pd.DataFrame:
    rows = []
    out_a = _output_dir("same_seed_mlp_a")
    out_b = _output_dir("same_seed_mlp_b")
    pred_a, _ = run_rank_models(
        seed=42,
        model_names=["mlp"],
        output_dir=out_a,
        tune=True,
    )
    pred_b, _ = run_rank_models(
        seed=42,
        model_names=["mlp"],
        output_dir=out_b,
        tune=True,
    )
    joined = pred_a[["date", "ric", "prediction"]].merge(
        pred_b[["date", "ric", "prediction"]],
        on=["date", "ric"],
        how="inner",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    rows.append(
        {
            "model": "mlp_rank",
            "seed": 42,
            "prediction_rows_a": int(len(pred_a)),
            "prediction_rows_b": int(len(pred_b)),
            "common_prediction_rows": int(len(joined)),
            "max_abs_prediction_deviation": float(
                (joined["prediction_a"] - joined["prediction_b"]).abs().max()
            ),
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_ROOT / "same_seed_determinism.csv", index=False)
    return result


def primary_seed_sensitivity(seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        out = _output_dir(f"primary_rank_seed_{seed}")
        run_rank_models(
            seed=seed,
            model_names=PRIMARY_MODELS,
            output_dir=out,
            tune=True,
        )
        ic_tests = pd.read_csv(out / "predictive_accuracy_ic_tests.csv")
        for model in ["hist_gbm_rank", "mlp_rank"]:
            rows.append({"seed": seed, **_extract_ic_pair(ic_tests, model, "ridge_rank")})
    result = pd.DataFrame(rows)
    result["holm_significant_5pct"] = result["holm_p_value"].le(0.05)
    result.to_csv(OUTPUT_ROOT / "primary_rank_seed_sensitivity.csv", index=False)
    return result


def complexity_seed_sensitivity(seeds: list[int]) -> pd.DataFrame:
    common_keys = pd.read_parquet(
        RESULTS_ROOT / "deep_sequence_common_benchmark" / "common_predictions.parquet",
        columns=["date", "ric"],
    ).drop_duplicates()
    factors = load_external_europe_factors(
        RAW_ROOT / "asset_pricing" / "french" / "Europe_5_Factors.csv",
        RAW_ROOT / "asset_pricing" / "french" / "Europe_MOM_Factor.csv",
    )
    fx = load_monthly_eurusd_return(RAW_ROOT / "fred_DEXUSEU.csv")
    rows = []
    for seed in seeds:
        out = _output_dir(f"complexity_rank_seed_{seed}")
        predictions_path = out / "predictions_common.parquet"
        monthly_path = out / "common_monthly_portfolios.csv"
        ladder_path = out / "complexity_spanning_ladder.csv"
        if predictions_path.exists() and (out / "fit_log.csv").exists():
            predictions = pd.read_parquet(predictions_path)
        else:
            predictions, fit_log = run_rank_models(
                seed=seed,
                model_names=COMPLEXITY_MODELS,
                output_dir=out,
                tune=False,
                dre_layers=1,
                dre_features_per_block=96,
            )
            predictions = predictions.merge(
                common_keys,
                on=["date", "ric"],
                how="inner",
                validate="many_to_one",
            )
            predictions.to_parquet(predictions_path, index=False, compression="zstd")
            fit_log.to_csv(out / "fit_log.csv", index=False)
        if monthly_path.exists():
            monthly = pd.read_csv(
                monthly_path,
                parse_dates=["signal_date", "return_date"],
            )
        else:
            monthly = construct_monthly_portfolios(
                predictions,
                _walk_config(seed, tune=False).portfolio_quantile,
            )
            monthly.to_csv(monthly_path, index=False)
        if ladder_path.exists():
            ladder = pd.read_csv(ladder_path)
        else:
            ladder = run_ladder(
                monthly,
                factors,
                fx,
                cost_bps=25,
                hac_lags=6,
                best_shallow="hist_gbm_rank",
            )
            ladder.to_csv(ladder_path, index=False)
        match = ladder[
            ladder["rung"].eq(3)
            & ladder["model"].eq("hist_gbm_rank")
            & ladder["weighting"].eq("equal")
            & ladder["universe_variant"].eq("standard_ex_bottom_5pct")
            & ladder["portfolio"].eq("long_short")
            & ladder["cost_bps"].eq(25)
        ]
        if match.empty:
            raise RuntimeError(f"Missing HistGBM spanning row for seed {seed}")
        row = match.iloc[0]
        rows.append(
            {
                "seed": seed,
                "alpha_annualized": float(row["alpha_annualized"]),
                "alpha_t": float(row["alpha_t"]),
                "raw_p_value": float(row["alpha_p"]),
                "holm_p_value": float(row["alpha_p_holm"]),
                "observations": int(row["observations"]),
                "holm_significant_5pct": float(row["alpha_p_holm"]) <= 0.05,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_ROOT / "complexity_spanning_seed_sensitivity.csv", index=False)
    return result


def walk_forward_boundaries() -> pd.DataFrame:
    _, fit_log = run_rank_models(
        seed=42,
        model_names=["mlp"],
        output_dir=_output_dir("same_seed_mlp_a"),
        tune=True,
    )
    predictions = pd.read_parquet(_output_dir("same_seed_mlp_a") / "predictions.parquet")
    rows = []
    for year in [2015, 2020, 2026]:
        fit = fit_log[fit_log["test_year"].eq(year)].iloc[0]
        pred = predictions[predictions["test_year"].eq(year)]
        validation_start = pd.Timestamp(fit["validation_start"])
        validation_end = pd.Timestamp(fit["validation_end"])
        core_training_end = validation_start - pd.offsets.MonthEnd(1)
        core_training_target_end = core_training_end + pd.offsets.MonthEnd(1)
        evaluation_start = pd.to_datetime(pred["date"]).min()
        evaluation_end = pd.to_datetime(pred["date"]).max()
        evaluation_target_start = pd.to_datetime(pred["target_date"]).min()
        evaluation_target_end = pd.to_datetime(pred["target_date"]).max()
        rows.append(
            {
                "test_year": year,
                "model": fit["model"],
                "core_training_signal_start": fit["train_signal_start"],
                "core_training_signal_end": str(core_training_end.date()),
                "core_training_target_end": str(core_training_target_end.date()),
                "pre_evaluation_pool_signal_start": fit["train_signal_start"],
                "pre_evaluation_pool_signal_end": fit["train_signal_end"],
                "pre_evaluation_pool_target_end": fit["train_target_end"],
                "training_label_cutoff": fit["train_label_cutoff"],
                "validation_signal_start": str(validation_start.date()),
                "validation_signal_end": str(validation_end.date()),
                "evaluation_signal_start": str(evaluation_start.date()),
                "evaluation_signal_end": str(evaluation_end.date()),
                "evaluation_target_start": str(evaluation_target_start.date()),
                "evaluation_target_end": str(evaluation_target_end.date()),
                "pre_evaluation_pool_after_cutoff_violation": bool(
                    pd.Timestamp(fit["train_target_end"])
                    > pd.Timestamp(fit["train_label_cutoff"])
                ),
                "core_training_before_validation": bool(
                    core_training_end < validation_start
                ),
                "validation_between_train_and_eval": bool(
                    core_training_end
                    < validation_start
                    <= validation_end
                    < evaluation_start
                ),
                "evaluation_excluded_from_fit": bool(
                    pd.Timestamp(fit["train_target_end"])
                    <= pd.Timestamp(fit["train_label_cutoff"])
                    and validation_end < evaluation_start
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_ROOT / "walk_forward_boundaries.csv", index=False)
    return result


def _compustat_merge_stage(region: str) -> dict[str, Any]:
    if region == "Europe":
        base_path = PROCESSED_ROOT / "monthly_feature_panel.parquet"
        annual_raw, monthly_raw = load_compustat_exports(
            RAW_ROOT / "asset_pricing" / "compustat_exports"
        )
        annual_norm = annual_raw
        monthly_norm = monthly_raw
        config = PanelConfig(as_of="2026-07-08", accounting_lag_months=6)
        audit_path = PROCESSED_ROOT / "compustat_enrichment_audit.json"
    else:
        base_path = PROCESSED_ROOT / "monthly_feature_panel_us.parquet"
        export_dir = RAW_ROOT / "asset_pricing" / "wrds_compustat_us_exports"
        annual_raw = pd.read_csv(
            export_dir / "compustat_us_fundamentals_annual.csv.gz",
            compression="gzip",
            dtype={"gvkey": str},
            low_memory=False,
        )
        monthly_raw = pd.read_csv(
            export_dir / "compustat_us_security_monthly.csv.gz",
            compression="gzip",
            dtype={"gvkey": str, "iid": str},
            low_memory=False,
        )
        annual_norm = normalize_wrds_compustat_us_annual(annual_raw)
        monthly_norm = normalize_wrds_compustat_us_monthly(monthly_raw)
        config = PanelConfig(as_of="2026-07-28", accounting_lag_months=6)
        audit_path = PROCESSED_ROOT / "wrds_compustat_us_enrichment_audit.json"

    base = pd.read_parquet(
        base_path,
        columns=["date", "ric", "TR.ISIN", "eligible", "model_eligible"],
    )
    base["date"] = _month_end(base["date"])
    base["isin_norm"] = _normalise_isin(base["TR.ISIN"])

    annual_features, annual_audit = prepare_compustat_annual_features(
        annual_norm,
        config,
    )
    monthly_features, monthly_audit = prepare_compustat_monthly_features(monthly_norm)
    raw_monthly_stage = monthly_raw.copy()
    raw_monthly_stage.columns = [str(column).lower() for column in raw_monthly_stage.columns]
    if region == "US" and "cusip" in raw_monthly_stage:
        raw_monthly_stage = add_us_isin_from_cusip(raw_monthly_stage)
    raw_monthly_stage["isin_norm"] = _normalise_isin(
        raw_monthly_stage.get("isin", pd.Series(index=raw_monthly_stage.index))
    )

    monthly_stage = monthly_norm.copy()
    monthly_stage.columns = [str(column).lower() for column in monthly_stage.columns]
    if "isin" not in monthly_stage and "cusip" in monthly_stage:
        monthly_stage = add_us_isin_from_cusip(monthly_stage)
    monthly_stage["isin_norm"] = _normalise_isin(monthly_stage.get("isin", pd.Series(index=monthly_stage.index)))
    monthly_stage["date"] = _month_end(monthly_stage["datadate"])
    monthly_keys = monthly_stage.dropna(subset=["isin_norm", "date"])
    duplicate_key_rows = int(monthly_keys.duplicated(["isin_norm", "date"]).sum())
    merged = base.merge(
        monthly_features[["isin_norm", "date", "comp_monthly_gvkey"]],
        on=["isin_norm", "date"],
        how="left",
        validate="many_to_one",
    )
    processed_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    panel_audit = processed_audit.get("panel", {})
    annual_isins = set(annual_features["isin_norm"].dropna().astype(str))
    monthly_isins = set(monthly_features["isin_norm"].dropna().astype(str))
    base_isins = set(base["isin_norm"].dropna().astype(str))
    matched = merged["comp_monthly_gvkey"].notna()
    currency_distribution = {}
    if "curcddvm" in monthly_stage:
        currency_distribution = {
            str(k): int(v)
            for k, v in monthly_stage["curcddvm"]
            .astype("string")
            .fillna("missing")
            .value_counts()
            .head(10)
            .items()
        }
    exchange_distribution = {}
    if "exchg" in monthly_stage:
        exchange_distribution = {
            str(k): int(v)
            for k, v in monthly_stage["exchg"]
            .astype("string")
            .fillna("missing")
            .value_counts()
            .head(15)
            .items()
        }
    return {
        "region": region,
        "base_panel_rows": int(len(base)),
        "base_unique_rics": int(base["ric"].nunique()),
        "base_unique_isins": int(base["isin_norm"].nunique()),
        "base_rows_with_isin": int(base["isin_norm"].notna().sum()),
        "monthly_raw_rows": int(len(monthly_raw)),
        "monthly_raw_unique_isins_after_derivation": int(raw_monthly_stage["isin_norm"].nunique()),
        "monthly_rows_after_normalization": int(len(monthly_norm)),
        "monthly_unique_isins_after_normalization": int(monthly_stage["isin_norm"].nunique()),
        "monthly_rows_with_nonmissing_merge_key": int(len(monthly_keys)),
        "monthly_duplicate_isin_date_rows_before_collapse": duplicate_key_rows,
        "monthly_collapsed_rows": int(len(monthly_features)),
        "monthly_collapsed_unique_isins": int(monthly_features["isin_norm"].nunique()),
        "monthly_first_date": monthly_audit["first_month"],
        "monthly_last_date": monthly_audit["last_month"],
        "base_unique_isins_present_in_monthly": int(len(base_isins & monthly_isins)),
        "base_unique_isins_absent_from_monthly": int(len(base_isins - monthly_isins)),
        "base_unique_isins_present_in_annual": int(len(base_isins & annual_isins)),
        "base_unique_isins_annual_yes_monthly_no": int(
            len((base_isins & annual_isins) - monthly_isins)
        ),
        "base_unique_isins_monthly_yes_annual_no": int(
            len((base_isins & monthly_isins) - annual_isins)
        ),
        "monthly_unique_isins_absent_from_base": int(len(monthly_isins - base_isins)),
        "merged_rows_with_monthly_match": int(matched.sum()),
        "merged_unique_rics_with_monthly_match": int(merged.loc[matched, "ric"].nunique()),
        "merged_unique_isins_with_monthly_match": int(merged.loc[matched, "isin_norm"].nunique()),
        "processed_audit_rows_with_monthly_match": panel_audit.get("rows_with_compustat_monthly"),
        "processed_audit_unique_rics_with_monthly_match": panel_audit.get("unique_rics_with_compustat_monthly"),
        "processed_audit_rows_with_annual_match": panel_audit.get("rows_with_compustat_annual"),
        "processed_audit_unique_rics_with_annual_match": panel_audit.get("unique_rics_with_compustat_annual"),
        "annual_source_rows_after_normalization": int(annual_audit["source_rows"]),
        "annual_collapsed_rows": int(annual_audit["collapsed_rows"]),
        "annual_unique_isins": int(annual_audit["unique_isins"]),
        "currency_distribution_top": currency_distribution,
        "exchange_distribution_top": exchange_distribution,
    }


def compustat_monthly_audit() -> pd.DataFrame:
    records = [_compustat_merge_stage("Europe"), _compustat_merge_stage("US")]
    flat = []
    for record in records:
        simple = {
            key: value
            for key, value in record.items()
            if not isinstance(value, dict)
        }
        flat.append(simple)
    result = pd.DataFrame(flat)
    result.to_csv(OUTPUT_ROOT / "compustat_monthly_merge_stage_counts.csv", index=False)
    _write_json(OUTPUT_ROOT / "compustat_monthly_merge_stage_counts.json", records)
    return result


def spread_request_audit() -> pd.DataFrame:
    request_path = (
        RAW_ROOT
        / "asset_pricing"
        / "refinitiv_exports"
        / "implementable_frontier_universe.csv"
    )
    request_rics = set(pd.read_csv(request_path)["ric"].astype(str).str.strip())
    predictions = pd.read_parquet(
        RESULTS_ROOT
        / "validation_selected_implementable_strategy"
        / "candidate_predictions.parquet",
        columns=[
            "date",
            "target_date",
            "ric",
            "model",
            "prediction",
            "target_return_1m",
            "company_market_cap",
            "market_cap_percentile",
        ],
    )
    predictions = predictions[
        predictions["model"].isin(
            [
                "momentum_rank",
                "ridge_rank",
                "smooth75_ridge_rank",
                "blend90_gbm_attn_seq24_rank",
            ]
        )
    ].copy()
    predictions["date"] = _month_end(predictions["date"])
    risk = pd.read_parquet(
        RESULTS_ROOT / "depth_analysis" / "rolling_risk_estimates.parquet",
        columns=["date", "ric", "risk_nobs"],
    )
    risk["date"] = _month_end(risk["date"])
    panel = predictions.merge(risk, on=["date", "ric"], how="left", validate="many_to_one")
    liquidity = load_monthly_liquidity(
        RAW_ROOT
        / "asset_pricing"
        / "refinitiv_exports"
        / "supplemental"
        / "liquidity_monthly_full_period"
    )
    panel = panel.merge(
        liquidity,
        on=["date", "ric"],
        how="left",
        validate="many_to_one",
    )
    panel["spread_observed"] = (
        panel["spread_observed"].astype("boolean").fillna(False).astype(bool)
    )
    panel["half_spread_bps"] = pd.to_numeric(
        panel["half_spread_bps"],
        errors="coerce",
    )

    rows = []
    for (date, model), month in panel.groupby(["date", "model"], sort=True):
        top_500 = investability_rungs(month, maximum_assets=500)["top_500"]
        if top_500.empty:
            continue
        missing = top_500[~top_500["ric"].astype(str).isin(request_rics)]
        observed = top_500[top_500["spread_observed"]]
        market_cap_total = float(top_500["company_market_cap"].sum())
        rows.append(
            {
                "date": date,
                "model": model,
                "top500_count": int(len(top_500)),
                "observed_spread_count": int(len(observed)),
                "absent_from_812_count": int(len(missing)),
                "absent_from_812_market_cap": float(missing["company_market_cap"].sum()),
                "top500_market_cap": market_cap_total,
                "absent_share_of_top500_market_cap": (
                    float(missing["company_market_cap"].sum()) / market_cap_total
                    if market_cap_total > 0
                    else np.nan
                ),
            }
        )
    monthly_model = pd.DataFrame(rows)
    monthly_model.to_csv(
        OUTPUT_ROOT / "spread_request_top500_truncation_by_model_month.csv",
        index=False,
    )
    monthly = (
        monthly_model.groupby("date", as_index=False)
        .agg(
            models=("model", "nunique"),
            any_absent=("absent_from_812_count", lambda values: bool((values > 0).any())),
            mean_absent_count=("absent_from_812_count", "mean"),
            max_absent_count=("absent_from_812_count", "max"),
            mean_absent_share_top500_market_cap=(
                "absent_share_of_top500_market_cap",
                "mean",
            ),
            max_absent_share_top500_market_cap=(
                "absent_share_of_top500_market_cap",
                "max",
            ),
            mean_observed_spread_count=("observed_spread_count", "mean"),
        )
    )
    monthly.to_csv(OUTPUT_ROOT / "spread_request_top500_truncation_by_month.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "request_rics": len(request_rics),
                "months": int(monthly["date"].nunique()),
                "months_with_absent_top500_names": int(monthly["any_absent"].sum()),
                "average_absent_names_per_month": float(monthly["mean_absent_count"].mean()),
                "average_absent_share_of_top500_market_cap": float(
                    monthly["mean_absent_share_top500_market_cap"].mean()
                ),
                "maximum_absent_names_in_any_model_month": int(
                    monthly_model["absent_from_812_count"].max()
                ),
                "maximum_absent_share_in_any_model_month": float(
                    monthly_model["absent_share_of_top500_market_cap"].max()
                ),
            }
        ]
    )
    summary.to_csv(OUTPUT_ROOT / "spread_request_top500_truncation_summary.csv", index=False)
    return summary


def holm_adjusted_mde() -> pd.DataFrame:
    table = pd.read_csv(
        RESULTS_ROOT
        / "data_depth_model_depth_interaction_refresh_20260816"
        / "data_depth_model_depth_interaction.csv"
    )
    records = []
    for quantity, group in table.dropna(subset=["p_value"]).groupby("quantity", sort=True):
        pvals = group["p_value"].to_numpy()
        order = np.argsort(pvals)
        adjusted = multipletests(pvals, method="holm")[1]
        for rank_index, row_index in enumerate(order, start=1):
            row = group.iloc[row_index]
            holm_step_tests_remaining = len(group) - rank_index + 1
            step_alpha = MDE_NOMINAL_ALPHA / holm_step_tests_remaining
            multiplier = (
                norm.ppf(1.0 - step_alpha / 2.0)
                + norm.ppf(MDE_POWER)
            )
            records.append(
                {
                    "model": row["model"],
                    "quantity": quantity,
                    "family_size": int(len(group)),
                    "holm_rank_by_raw_p": int(rank_index),
                    "holm_tests_remaining": int(holm_step_tests_remaining),
                    "raw_p_value": float(row["p_value"]),
                    "holm_p_value": float(adjusted[row_index]),
                    "standard_error": float(row["standard_error"]),
                    "nominal_mde": float(row["minimum_detectable_effect"]),
                    "holm_adjusted_mde": float(multiplier * row["standard_error"]),
                    "holm_step_alpha": float(step_alpha),
                }
            )
    result = pd.DataFrame(records)
    result.to_csv(OUTPUT_ROOT / "holm_adjusted_mde.csv", index=False)
    return result


def appendix_197_month_table() -> pd.DataFrame:
    side = pd.read_csv(
        RESULTS_ROOT
        / "market_comparison_compustat_2010"
        / "side_by_side_model_summary.csv"
    )
    model_labels = {
        "momentum_rank": "Momentum",
        "ridge_rank": "Ridge",
        "elastic_net_rank": "Elastic Net",
        "hist_gbm_rank": "HistGBM",
        "mlp_rank": "MLP",
    }
    common_filter = (
        side["model"].isin(model_labels)
        & side["target_mode"].eq("rank")
        & side["universe_variant"].eq("standard_ex_bottom_5pct")
        & side["portfolio"].eq("long_short")
        & side["cost_bps"].eq(25)
    )
    subset = side[common_filter & side["weighting"].eq("equal")].copy()
    if subset.empty:
        subset = side[common_filter & side["weighting"].eq("value")].copy()
    if subset.empty:
        raise RuntimeError("No matching 197-month US/Europe comparison rows found")
    subset["model_label"] = subset["model"].map(model_labels)
    subset["order"] = subset["model"].map({m: i for i, m in enumerate(model_labels)})
    table = subset.sort_values("order")[
        [
            "model",
            "model_label",
            "mean_monthly_spearman_ic_europe",
            "mean_monthly_spearman_ic_us",
            "net_sharpe_europe",
            "net_sharpe_us",
            "months_europe",
            "months_us",
            "observations_europe",
            "observations_us",
        ]
    ].rename(
        columns={
            "mean_monthly_spearman_ic_europe": "ic_europe",
            "mean_monthly_spearman_ic_us": "ic_us",
        }
    )
    table.to_csv(OUTPUT_ROOT / "appendix_table_197_month_us_europe.csv", index=False)

    lines = [
        "| Model | Europe IC | US IC | Europe net Sharpe | US net Sharpe | Months | Europe obs. | US obs. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            "| {label} | {ic_eu:.3f} | {ic_us:.3f} | {sh_eu:.3f} | {sh_us:.3f} | {months:d} | {obs_eu:,d} | {obs_us:,d} |".format(
                label=row["model_label"],
                ic_eu=float(row["ic_europe"]),
                ic_us=float(row["ic_us"]),
                sh_eu=float(row["net_sharpe_europe"]),
                sh_us=float(row["net_sharpe_us"]),
                months=int(row["months_europe"]),
                obs_eu=int(row["observations_europe"]),
                obs_us=int(row["observations_us"]),
            )
        )
    (OUTPUT_ROOT / "appendix_table_197_month_us_europe.md").write_text("\n".join(lines) + "\n")
    return table


def environment_determinism_audit() -> pd.DataFrame:
    try:
        import torch

        torch_available = True
        cuda_available = bool(torch.cuda.is_available())
        cudnn_available = bool(getattr(torch.backends, "cudnn", None) is not None)
    except ImportError:
        torch_available = False
        cuda_available = False
        cudnn_available = False
    tf_hits = []
    this_file = Path(__file__).resolve()
    for path in [PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"]:
        for candidate in path.rglob("*.py"):
            if candidate.resolve() == this_file:
                continue
            text = candidate.read_text(errors="ignore")
            if "tensorflow" in text or "keras" in text or "tf." in text:
                tf_hits.append(str(candidate.relative_to(PROJECT_ROOT)))
    result = pd.DataFrame(
        [
            {
                "component": "python_random",
                "status": "seeded_by_asset_pricing_ml.set_reproducible_seed",
            },
            {
                "component": "numpy",
                "status": "seeded_by_asset_pricing_ml.set_reproducible_seed",
            },
            {
                "component": "scikit_learn",
                "status": "random_state_passed_to_stochastic_estimators",
            },
            {
                "component": "torch",
                "status": (
                    "manual_seed_and_cuda_manual_seed_all_with_deterministic_algorithms"
                    if torch_available
                    else "not_installed"
                ),
            },
            {
                "component": "cuda",
                "status": (
                    "available; CUBLAS_WORKSPACE_CONFIG_set_and_cudnn_deterministic"
                    if cuda_available and cudnn_available
                    else "not_available_in_this_runtime"
                ),
            },
            {
                "component": "tensorflow",
                "status": (
                    "not_used_in_src_or_scripts"
                    if not tf_hits
                    else f"references_found: {', '.join(tf_hits)}"
                ),
            },
        ]
    )
    result.to_csv(OUTPUT_ROOT / "determinism_component_audit.csv", index=False)
    return result


def run_all(seeds: list[int], skip_seed_reruns: bool) -> dict[str, str]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    environment_determinism_audit()
    if not skip_seed_reruns:
        same_seed_determinism()
        primary_seed_sensitivity(seeds)
        complexity_seed_sensitivity(seeds)
        walk_forward_boundaries()
    compustat_monthly_audit()
    spread_request_audit()
    holm_adjusted_mde()
    appendix_197_month_table()
    manifest = {
        "output_root": str(OUTPUT_ROOT),
        "seed_runs": seeds,
        "skip_seed_reruns": skip_seed_reruns,
        "outputs": sorted(str(path.relative_to(PROJECT_ROOT)) for path in OUTPUT_ROOT.glob("*")),
    }
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEED_RUNS)
    parser.add_argument("--skip-seed-reruns", action="store_true")
    args = parser.parse_args()
    manifest = run_all(args.seeds, args.skip_seed_reruns)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
