"""Closure analyses for the August 2026 dissertation checks.

The module is intentionally a thin orchestration layer over the existing
walk-forward, constrained-portfolio, and inference helpers.  It writes only
under ``results/closure_2026_08``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stats as project_stats  # noqa: E402
from asset_pricing_ml import (  # noqa: E402
    FEATURE_SETS,
    WalkForwardConfig,
    construct_monthly_portfolios,
    load_model_panel,
    portfolio_summary,
    prediction_metrics,
    run_walk_forward,
    set_reproducible_seed,
    walk_forward_slices,
)
from run_constrained_estimates_long_only import run_experiment  # noqa: E402
from run_constrained_deep_hybrid_long_only import ConstraintSpec  # noqa: E402
from run_capacity_gradient_tests import assign_buckets  # noqa: E402


RESULTS_ROOT = PROJECT_ROOT / "results" / "asset_pricing_ml"
CLOSURE_DIR = PROJECT_ROOT / "results" / "closure_2026_08"

COMPUSTAT_PREDICTIONS = RESULTS_ROOT / "europe_compustat_benchmark" / "predictions.parquet"
REVISION_PREDICTIONS = (
    RESULTS_ROOT
    / "estimates_revisions_pure_strict_lag1_revision_signal_smooth75"
    / "smoothed_with_parents_predictions.parquet"
)
STRICT_ESTIMATES_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "strict_estimates_lag1"
    / "monthly_feature_panel_estimates_strict_lag1.parquet"
)
TASKB_LIQUIDITY = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "asset_pricing"
    / "refinitiv_exports"
    / "supplemental"
    / "liquidity_monthly_full_period_top2000"
)
TASKB_RISK = RESULTS_ROOT / "depth_analysis" / "rolling_risk_estimates.parquet"
TASKB_MARKET = RESULTS_ROOT / "depth_analysis" / "eur_market_return.csv"

TASKB_AUMS = (10_000_000.0, 100_000_000.0, 500_000_000.0)
TASKB_BOOTSTRAP_BLOCK = 6
TASKB_BOOTSTRAP_REPETITIONS = 10_000
TASKB_SEED = 42
TASKB_CONSTRAINT = "name5_country40_sector40_turnover"
TASKB_NO_TURNOVER_CONSTRAINT = "name5_country40_sector40_no_turnover"

TASKB_SIGNALS = {
    "Momentum": {
        "source": COMPUSTAT_PREDICTIONS,
        "source_model": "momentum_rank",
        "strategy": "taskB_momentum_top500_observed",
        "model": "taskB_momentum_rank",
    },
    "Ridge": {
        "source": COMPUSTAT_PREDICTIONS,
        "source_model": "ridge_rank",
        "strategy": "taskB_ridge_top500_observed",
        "model": "taskB_ridge_rank",
    },
    "HistGBM": {
        "source": COMPUSTAT_PREDICTIONS,
        "source_model": "hist_gbm_rank",
        "strategy": "taskB_histgbm_top500_observed",
        "model": "taskB_histgbm_rank",
    },
    "Revision": {
        "source": REVISION_PREDICTIONS,
        "source_model": "smooth75_ridge_rank",
        "strategy": "taskB_revision_top500_observed",
        "model": "taskB_revision_rank",
    },
}
TASKB_STRATEGY_TO_SIGNAL = {
    spec["strategy"]: signal for signal, spec in TASKB_SIGNALS.items()
}

COMPUSTAT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel_compustat.parquet"
)
DELISTING_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "delisting_return_audit.csv"
)
COMMON_SAMPLE_ROWS = 459_829
COMMON_SAMPLE_MONTHS = 137
HAC_LAGS = 6
MDE_MULTIPLIER = 1.959963985 + 0.841621234
TASKA_MICROSTRUCTURE_COLUMNS = [
    "return_1m_rank",
    "max_return_12m_rank",
    "turnover_1m_rank",
    "turnover_12m_rank",
    "turnover_volatility_12m_rank",
    "comp_log_price_rank",
    "comp_log_volume_rank",
]
TASKA_PARTITIONS = [
    ("market_cap", "company_market_cap", "low_cap", "top_500_cap"),
    ("trading_value", "log_trading_value_eur", "low_adv", "top_500_adv"),
]
TASKC_MODELS = {
    "Momentum": "momentum_rank",
    "Ridge": "ridge_rank",
}
TASKC_BUCKETS = {
    "market_cap": ["low_cap", "mid_cap", "high_cap", "top_500_cap"],
    "trading_value": ["low_adv", "mid_adv", "high_adv", "top_500_adv"],
}
TASKC_LEG_SIZES = (25, 50, 100)
TASKC_PRIMARY_LEG_SIZE = 50
TASKC_COST_BPS = 25
REVISION_CONSTRAINED_DIR = (
    RESULTS_ROOT
    / "constrained_estimates_revisions_pure_strict_lag1_revision_signal_fixed"
)
TASKD_SUBPERIODS = [
    ("early_2015_2016", "2015-02-01", "2016-12-31"),
    ("pre_covid_2017_2019", "2017-01-01", "2019-12-31"),
    ("covid_recovery_2020_2022", "2020-01-01", "2022-12-31"),
    ("recent_2023_2026", "2023-01-01", "2026-12-31"),
]
LITERATURE_SIGN_BY_FEATURE = {
    "log_size_rank": -1,
    "book_to_market_rank": 1,
    "return_1m_rank": -1,
    "momentum_6_2_rank": 1,
    "momentum_12_2_rank": 1,
    "volatility_12m_rank": -1,
    "max_return_12m_rank": -1,
    "market_cap_growth_12m_rank": -1,
    "turnover_1m_rank": -1,
    "turnover_12m_rank": -1,
    "asset_growth_rank": -1,
    "sales_growth_rank": -1,
    "profitability_roa_rank": 1,
    "operating_profitability_rank": 1,
    "leverage_rank": -1,
    "accruals_rank": -1,
    "capex_to_assets_rank": -1,
    "cashflow_to_assets_rank": 1,
    "log_trading_value_eur_rank": -1,
    "turnover_volatility_12m_rank": -1,
    "comp_book_to_market_rank": 1,
    "comp_asset_growth_rank": -1,
    "comp_sales_growth_rank": -1,
    "comp_equity_growth_rank": -1,
    "comp_gross_profitability_rank": 1,
    "comp_roa_rank": 1,
    "comp_roe_rank": 1,
    "comp_operating_margin_rank": 1,
    "comp_gross_margin_rank": 1,
    "comp_asset_turnover_rank": 1,
    "comp_leverage_rank": -1,
    "comp_debt_to_assets_rank": -1,
    "comp_debt_to_equity_rank": -1,
    "comp_cash_to_assets_rank": 1,
    "comp_cash_to_debt_rank": 1,
    "comp_current_ratio_rank": 1,
    "comp_working_capital_to_assets_rank": 1,
    "comp_inventory_to_assets_rank": -1,
    "comp_receivables_to_assets_rank": -1,
    "comp_ppe_to_assets_rank": -1,
    "comp_intangibles_to_assets_rank": 1,
    "comp_rd_to_assets_rank": 1,
    "comp_sga_to_sales_rank": -1,
    "comp_depreciation_to_assets_rank": -1,
    "comp_capex_to_assets_rank": -1,
    "comp_accruals_to_assets_rank": -1,
    "comp_payout_to_assets_rank": 1,
    "comp_log_price_rank": -1,
    "comp_log_volume_rank": -1,
    "comp_price_momentum_6_2_rank": 1,
    "comp_price_momentum_12_2_rank": 1,
    "comp_price_volatility_12m_rank": -1,
    "comp_volume_growth_12m_rank": -1,
}


def _aum_label(aum: float) -> str:
    return f"{int(round(aum / 1_000_000.0))}m"


def _hac_summary(values: pd.Series) -> dict[str, float]:
    result = project_stats.hac_mean_diff_test(values, maxlags=HAC_LAGS)
    mean = float(result["mean"])
    se = float(result["se"])
    return {
        "n_months": int(result["n"]),
        "estimate": mean,
        "standard_error": se,
        "hac_t": float(result["t"]),
        "p_value": float(result["p_two_sided"]),
        "ci_lo": float(result["ci_low"]),
        "ci_hi": float(result["ci_high"]),
        "mde": float(MDE_MULTIPLIER * se) if np.isfinite(se) else np.nan,
    }


def _common_sample_keys() -> pd.DataFrame:
    keys = pd.read_parquet(
        COMPUSTAT_PREDICTIONS,
        columns=["model", "date", "ric"],
    )
    keys = keys[keys["model"].eq("ridge_rank")][["date", "ric"]].copy()
    keys["date"] = pd.to_datetime(keys["date"])
    keys = keys.drop_duplicates(["date", "ric"]).sort_values(["date", "ric"])
    if len(keys) != COMMON_SAMPLE_ROWS or keys["date"].nunique() != COMMON_SAMPLE_MONTHS:
        raise RuntimeError(
            "Common sample mask mismatch: "
            f"rows={len(keys)}, months={keys['date'].nunique()}"
        )
    return keys.reset_index(drop=True)


def _load_common_prediction_panel() -> pd.DataFrame:
    predictions = pd.read_parquet(
        COMPUSTAT_PREDICTIONS,
        columns=[
            "date",
            "target_date",
            "ric",
            "target_return_1m",
            "target_return_rank",
            "company_market_cap",
            "market_cap_percentile",
            "prediction",
            "model",
        ],
    )
    predictions["date"] = pd.to_datetime(predictions["date"])
    rank_models = [
        "momentum_rank",
        "ridge_rank",
        "elastic_net_rank",
        "hist_gbm_rank",
        "mlp_rank",
    ]
    predictions = predictions[predictions["model"].isin(rank_models)].copy()
    counts = predictions.groupby("model").size()
    if not counts.eq(COMMON_SAMPLE_ROWS).all():
        raise RuntimeError(f"Production rank model counts are not common: {counts.to_dict()}")
    wide = predictions.pivot_table(
        index=[
            "date",
            "target_date",
            "ric",
            "target_return_1m",
            "target_return_rank",
            "company_market_cap",
            "market_cap_percentile",
        ],
        columns="model",
        values="prediction",
        aggfunc="last",
    ).reset_index()
    if len(wide) != COMMON_SAMPLE_ROWS or wide["date"].nunique() != COMMON_SAMPLE_MONTHS:
        raise RuntimeError(
            f"Wide common prediction panel mismatch: rows={len(wide)}, "
            f"months={wide['date'].nunique()}"
        )
    return wide.sort_values(["date", "ric"]).reset_index(drop=True)


def _load_common_features(extra_columns: list[str]) -> pd.DataFrame:
    keys = _common_sample_keys()
    columns = list(dict.fromkeys(["date", "ric", *extra_columns]))
    panel = pd.read_parquet(COMPUSTAT_PANEL, columns=columns)
    panel["date"] = pd.to_datetime(panel["date"])
    merged = keys.merge(panel, on=["date", "ric"], how="left", validate="one_to_one")
    if len(merged) != COMMON_SAMPLE_ROWS:
        raise RuntimeError(f"Common feature merge changed row count: {len(merged)}")
    return merged


def _monthly_rank_ic(frame: pd.DataFrame, score_column: str) -> pd.Series:
    def corr(month: pd.DataFrame) -> float:
        work = month[[score_column, "target_return_1m"]].dropna()
        if len(work) < 20 or work[score_column].nunique() < 2:
            return np.nan
        return float(
            work[score_column].rank(method="average").corr(
                work["target_return_1m"].rank(method="average")
            )
        )

    return frame.groupby("date", sort=True).apply(corr, include_groups=False)


def _mean_ic_and_ir(frame: pd.DataFrame, score_column: str) -> tuple[float, float, int]:
    monthly = _monthly_rank_ic(frame, score_column).dropna()
    if len(monthly) != COMMON_SAMPLE_MONTHS:
        raise RuntimeError(
            f"Expected {COMMON_SAMPLE_MONTHS} IC months for {score_column}, got {len(monthly)}"
        )
    std = float(monthly.std(ddof=1))
    ir = float(monthly.mean() / std * np.sqrt(12.0)) if std > 0 else np.nan
    return float(monthly.mean()), ir, int(len(monthly))


def _bucketed_frame(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
    spec = {name: (sort, least, most) for name, sort, least, most in TASKA_PARTITIONS}
    sort_column, _, _ = spec[partition]
    labelled = assign_buckets(frame.copy(), sort_column, partition, top_n=500)
    labelled["partition"] = partition
    return labelled


def _bucket_ic_series(frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (partition, bucket, date), group in frame.groupby(
        ["partition", "bucket", "date"],
        sort=True,
        observed=True,
    ):
        work = group[[score_column, "target_return_1m"]].dropna()
        if len(work) < 20 or work[score_column].nunique() < 2:
            ic = np.nan
        else:
            ic = float(
                work[score_column].rank(method="average").corr(
                    work["target_return_1m"].rank(method="average")
                )
            )
        records.append(
            {
                "partition": partition,
                "bucket": bucket,
                "date": date,
                "names": int(len(work)),
                "ic": ic,
            }
        )
    return pd.DataFrame(records)


def _taska_capacity_summary(
    frame: pd.DataFrame,
    *,
    run_label: str,
    score_column: str,
    benchmark_column: str = "momentum_rank",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bucket_tables: list[pd.DataFrame] = []
    gradient_records: list[dict[str, Any]] = []
    bucket_records: list[dict[str, Any]] = []
    for partition, _, least, most in TASKA_PARTITIONS:
        bucketed = _bucketed_frame(frame, partition)
        model_ic = _bucket_ic_series(bucketed, score_column).rename(
            columns={"ic": "model_ic", "names": "model_names"}
        )
        benchmark_ic = _bucket_ic_series(bucketed, benchmark_column).rename(
            columns={"ic": "benchmark_ic", "names": "benchmark_names"}
        )
        merged = model_ic.merge(
            benchmark_ic,
            on=["partition", "bucket", "date"],
            how="inner",
            validate="one_to_one",
        )
        merged["premium"] = merged["model_ic"] - merged["benchmark_ic"]
        for bucket, series in merged.groupby("bucket", sort=True)["premium"]:
            stats = _hac_summary(series)
            bucket_records.append(
                {
                    "run": run_label,
                    "partition": partition,
                    "bucket": bucket,
                    **stats,
                }
            )
        pivot = merged.pivot_table(index="date", columns="bucket", values="premium")
        if least not in pivot or most not in pivot:
            raise RuntimeError(f"Missing Task A gradient buckets for {run_label} {partition}")
        least_series = pivot[least].dropna()
        most_series = pivot[most].dropna()
        gradient = (pivot[least] - pivot[most]).dropna()
        gradient_stats = _hac_summary(gradient)
        gradient_records.append(
            {
                "run": run_label,
                "partition": partition,
                "least_bucket": least,
                "most_bucket": most,
                "gradient": gradient_stats["estimate"],
                "hac_t": gradient_stats["hac_t"],
                "p_value": gradient_stats["p_value"],
                "ci_lo": gradient_stats["ci_lo"],
                "ci_hi": gradient_stats["ci_hi"],
                "mde": gradient_stats["mde"],
                "ic_least": float(least_series.mean()),
                "ic_most": float(most_series.mean()),
                "n_months": gradient_stats["n_months"],
                "partition_obs": int(bucketed[["date", "ric"]].drop_duplicates().shape[0]),
            }
        )
        bucket_tables.append(merged)
    return pd.DataFrame(gradient_records), pd.DataFrame(bucket_records)


def run_task_a_diagnostics(output_dir: Path = CLOSURE_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = _load_common_prediction_panel()
    features = _load_common_features(
        [
            "return_1m_rank",
            "momentum_12_2_rank",
            "turnover_1m_rank",
            "log_trading_value_eur",
            "comp_log_price",
        ]
    )
    frame = base.merge(
        features.drop(columns=["date", "ric"], errors="ignore").assign(
            date=features["date"],
            ric=features["ric"],
        ),
        on=["date", "ric"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != COMMON_SAMPLE_ROWS:
        raise RuntimeError("Task A diagnostics lost common-sample rows")

    signal_columns = {
        "short_reversal": "_short_reversal_score",
        "momentum_12_2": "momentum_12_2_rank",
        "current_turnover": "turnover_1m_rank",
    }
    frame["_short_reversal_score"] = -pd.to_numeric(frame["return_1m_rank"], errors="coerce")

    standalone_records: list[dict[str, Any]] = []
    for partition, _, least, most in TASKA_PARTITIONS:
        bucketed = _bucketed_frame(frame, partition)
        for signal, column in signal_columns.items():
            ic = _bucket_ic_series(bucketed, column)
            for bucket, series in ic.groupby("bucket", sort=True)["ic"]:
                stats = _hac_summary(series)
                standalone_records.append(
                    {
                        "signal": signal,
                        "partition": partition,
                        "bucket": bucket,
                        "mean_ic": stats["estimate"],
                        "hac_t": stats["hac_t"],
                        "p_value": stats["p_value"],
                        "ci_lo": stats["ci_lo"],
                        "ci_hi": stats["ci_hi"],
                        "mde": stats["mde"],
                        "n_months": stats["n_months"],
                        "n_obs": int(bucketed[bucketed["bucket"].eq(bucket)].shape[0]),
                    }
                )
            pivot = ic.pivot_table(index="date", columns="bucket", values="ic")
            spread = (pivot[least] - pivot[most]).dropna()
            stats = _hac_summary(spread)
            standalone_records.append(
                {
                    "signal": signal,
                    "partition": partition,
                    "bucket": "least_minus_most",
                    "mean_ic": stats["estimate"],
                    "hac_t": stats["hac_t"],
                    "p_value": stats["p_value"],
                    "ci_lo": stats["ci_lo"],
                    "ci_hi": stats["ci_hi"],
                    "mde": stats["mde"],
                    "n_months": stats["n_months"],
                    "n_obs": int(bucketed[["date", "ric"]].drop_duplicates().shape[0]),
                }
            )
    standalone = pd.DataFrame(standalone_records)
    standalone.to_csv(output_dir / "taskA_standalone_bucket_ic.csv", index=False)

    price_screen = frame.copy()
    raw_price = pd.to_numeric(price_screen["comp_log_price"], errors="coerce")
    valid_price = raw_price.notna()
    within_month_price_rank = raw_price.groupby(price_screen["date"]).rank(
        method="average",
        pct=True,
    )
    price_screen = price_screen[valid_price & within_month_price_rank.gt(0.10)].copy()
    gradients, buckets = _taska_capacity_summary(
        price_screen,
        run_label="price_screen_ridge_vs_momentum",
        score_column="ridge_rank",
        benchmark_column="momentum_rank",
    )
    gradients["mean_ic"], gradients["ic_ir"], _ = _mean_ic_and_ir(
        price_screen,
        "ridge_rank",
    )
    gradients["n_obs"] = int(len(price_screen))
    gradients["screen_excluded_obs"] = int(COMMON_SAMPLE_ROWS - len(price_screen))
    gradients["screen_missing_price_obs"] = int((~valid_price).sum())
    gradients["holm_p"] = multipletests(gradients["p_value"], method="holm")[1]
    price_out = gradients[
        [
            "run",
            "partition",
            "mean_ic",
            "ic_ir",
            "gradient",
            "hac_t",
            "holm_p",
            "ci_lo",
            "ci_hi",
            "mde",
            "ic_least",
            "ic_most",
            "n_months",
            "n_obs",
            "screen_excluded_obs",
            "screen_missing_price_obs",
            "partition_obs",
        ]
    ].copy()
    price_out.to_csv(output_dir / "taskA_price_screen.csv", index=False)
    buckets.to_csv(output_dir / "taskA_price_screen_bucket_premia.csv", index=False)
    manifest = {
        "task": "A_diagnostics",
        "common_sample_rows": COMMON_SAMPLE_ROWS,
        "common_sample_months": COMMON_SAMPLE_MONTHS,
        "outputs": {
            "taskA_standalone_bucket_ic": str(output_dir / "taskA_standalone_bucket_ic.csv"),
            "taskA_price_screen": str(output_dir / "taskA_price_screen.csv"),
            "taskA_price_screen_bucket_premia": str(
                output_dir / "taskA_price_screen_bucket_premia.csv"
            ),
        },
        "rows": {
            "taskA_standalone_bucket_ic": int(len(standalone)),
            "taskA_price_screen": int(len(price_out)),
        },
    }
    (output_dir / "taskA_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest


def _run_taska_model(
    output_dir: Path,
    *,
    run_label: str,
    model_name: str,
    excluded_columns: list[str],
) -> Path:
    feature_columns = [
        column for column in FEATURE_SETS["compustat_enriched"] if column not in excluded_columns
    ]
    set_reproducible_seed(42)
    panel = load_model_panel(
        COMPUSTAT_PANEL,
        delisting_audit_path=DELISTING_AUDIT if DELISTING_AUDIT.exists() else None,
        feature_columns=feature_columns,
    )
    config = WalkForwardConfig(
        first_test_year=2015,
        last_test_year=2026,
        min_training_rows=10_000,
        max_training_rows=None,
        random_state=42,
        validation_months=24,
        tune_hyperparameters=True,
    )
    predictions, fit_log, coefficients, importance = run_walk_forward(
        panel,
        [model_name],
        config,
        target_column="target_return_rank",
        target_mode="rank",
        feature_columns=feature_columns,
        collect_importance=False,
    )
    if predictions.empty:
        raise RuntimeError(f"{run_label} produced no predictions")
    predictions["model"] = run_label
    predictions["base_model"] = model_name
    predictions["target_mode"] = "rank"
    run_dir = output_dir / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False, compression="zstd")
    fit_log.to_csv(run_dir / "fit_log.csv", index=False)
    coefficients.to_csv(run_dir / "linear_coefficients.csv", index=False)
    importance.to_csv(run_dir / "oos_variable_importance.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run": run_label,
                "model": model_name,
                "feature_columns": feature_columns,
                "excluded_columns": excluded_columns,
                "rows": {
                    "predictions": int(len(predictions)),
                    "fit_log": int(len(fit_log)),
                    "coefficients": int(len(coefficients)),
                },
                "causality_check": {
                    "train_target_after_cutoff": int(
                        (
                            pd.to_datetime(fit_log["train_target_end"])
                            > pd.to_datetime(fit_log["train_label_cutoff"])
                        ).sum()
                    ),
                    "duplicate_model_security_month_predictions": int(
                        predictions.duplicated(["model", "date", "ric"]).sum()
                    ),
                },
            },
            indent=2,
            default=str,
        )
    )
    return prediction_path


def _load_taska_prediction_on_common(prediction_path: Path, score_column: str) -> pd.DataFrame:
    base = _load_common_prediction_panel()
    features = _load_common_features(["log_trading_value_eur"])
    predictions = pd.read_parquet(
        prediction_path,
        columns=["date", "ric", "prediction", "model"],
    )
    predictions["date"] = pd.to_datetime(predictions["date"])
    model_counts = predictions.groupby("model").size().to_dict()
    merged = base.merge(
        predictions[["date", "ric", "prediction"]],
        on=["date", "ric"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"prediction": score_column})
    merged = merged.merge(
        features[["date", "ric", "log_trading_value_eur"]],
        on=["date", "ric"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != COMMON_SAMPLE_ROWS:
        raise RuntimeError("Task A ablation common-sample merge changed row count")
    if merged[score_column].isna().any():
        raise RuntimeError(
            f"Task A ablation {prediction_path} has missing predictions on common mask; "
            f"source counts={model_counts}"
        )
    return merged


def _evaluate_taska_ablation(
    prediction_path: Path,
    *,
    run_label: str,
    score_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _load_taska_prediction_on_common(prediction_path, score_column)
    if len(frame) != COMMON_SAMPLE_ROWS or frame["date"].nunique() != COMMON_SAMPLE_MONTHS:
        raise RuntimeError(f"{run_label} did not evaluate on the fixed common sample")
    mean_ic, ic_ir, months = _mean_ic_and_ir(frame, score_column)
    gradients, bucket_premia = _taska_capacity_summary(
        frame,
        run_label=run_label,
        score_column=score_column,
        benchmark_column="momentum_rank",
    )
    gradients["mean_ic"] = mean_ic
    gradients["ic_ir"] = ic_ir
    gradients["n_obs"] = COMMON_SAMPLE_ROWS
    gradients["full_sample_months"] = months
    return gradients, bucket_premia


def run_task_a_ablations(
    output_dir: Path = CLOSURE_DIR,
    *,
    include_histgbm_nomicro: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        ("ridge_norev", "ridge", ["return_1m_rank"]),
        ("ridge_nomicro", "ridge", TASKA_MICROSTRUCTURE_COLUMNS),
    ]
    if include_histgbm_nomicro:
        runs.append(("histgbm_nomicro", "hist_gbm", TASKA_MICROSTRUCTURE_COLUMNS))

    gradient_frames: list[pd.DataFrame] = []
    bucket_frames: list[pd.DataFrame] = []
    prediction_paths: dict[str, str] = {}
    for run_label, model_name, excluded in runs:
        prediction_path = _run_taska_model(
            output_dir,
            run_label=run_label,
            model_name=model_name,
            excluded_columns=excluded,
        )
        prediction_paths[run_label] = str(prediction_path)
        gradients, buckets = _evaluate_taska_ablation(
            prediction_path,
            run_label=run_label,
            score_column=f"{run_label}_score",
        )
        gradient_frames.append(gradients)
        bucket_frames.append(buckets)

    gradient_table = pd.concat(gradient_frames, ignore_index=True)
    gradient_table["holm_p"] = multipletests(
        gradient_table["p_value"].fillna(1.0),
        method="holm",
    )[1]
    final = gradient_table[
        [
            "run",
            "partition",
            "mean_ic",
            "ic_ir",
            "gradient",
            "hac_t",
            "holm_p",
            "ci_lo",
            "ci_hi",
            "mde",
            "ic_least",
            "ic_most",
            "n_months",
            "n_obs",
            "partition_obs",
            "full_sample_months",
        ]
    ].copy()
    if not final["n_obs"].eq(COMMON_SAMPLE_ROWS).all():
        raise RuntimeError("Task A ablation full-sample n_obs assertion failed")
    final.to_csv(output_dir / "taskA_ablation.csv", index=False)
    bucket_table = pd.concat(bucket_frames, ignore_index=True)
    bucket_table["holm_p"] = multipletests(
        bucket_table["p_value"].fillna(1.0),
        method="holm",
    )[1]
    bucket_table.to_csv(output_dir / "taskA_ablation_bucket_premia.csv", index=False)
    manifest = {
        "task": "A_ablations",
        "common_sample_rows": COMMON_SAMPLE_ROWS,
        "common_sample_months": COMMON_SAMPLE_MONTHS,
        "include_histgbm_nomicro": include_histgbm_nomicro,
        "prediction_paths": prediction_paths,
        "outputs": {
            "taskA_ablation": str(output_dir / "taskA_ablation.csv"),
            "taskA_ablation_bucket_premia": str(
                output_dir / "taskA_ablation_bucket_premia.csv"
            ),
        },
        "rows": {
            "taskA_ablation": int(len(final)),
            "taskA_ablation_bucket_premia": int(len(bucket_table)),
        },
    }
    (output_dir / "taskA_ablations_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return float(wealth.div(peak).sub(1.0).min())


def _annualized_return_to_vol(returns: pd.Series) -> float:
    standard_deviation = returns.std(ddof=1)
    if standard_deviation <= 0 or pd.isna(standard_deviation):
        return np.nan
    return float(returns.mean() / standard_deviation * np.sqrt(12.0))


def _period_slice(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    target_date = pd.to_datetime(frame["target_date"])
    return frame[target_date.ge(pd.Timestamp(start)) & target_date.le(pd.Timestamp(end))]


def _taskd_subperiod_table(monthly: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (strategy, constraint), group in monthly.groupby(["strategy", "constraint"], sort=True):
        for subperiod, start, end in TASKD_SUBPERIODS:
            part = _period_slice(group, start, end)
            if len(part) < 6:
                continue
            for aum in TASKB_AUMS:
                label = _aum_label(aum)
                returns = pd.to_numeric(part[f"net_return_{label}"], errors="coerce")
                records.append(
                    {
                        "strategy": strategy,
                        "constraint": constraint,
                        "subperiod": subperiod,
                        "aum_eur": float(aum),
                        "aum_label": label,
                        "months": int(returns.notna().sum()),
                        "annualized_net_return": float(returns.mean() * 12.0),
                        "net_sharpe": _annualized_return_to_vol(returns),
                        "max_drawdown": _max_drawdown(returns),
                        "annualized_spread_cost": float(
                            pd.to_numeric(
                                part[f"spread_cost_{label}"],
                                errors="coerce",
                            ).mean()
                            * 12.0
                        ),
                        "annualized_impact_cost": float(
                            pd.to_numeric(
                                part[f"impact_cost_{label}"],
                                errors="coerce",
                            ).mean()
                            * 12.0
                        ),
                        "average_effective_n": float(part["effective_n"].mean()),
                        "share_positive_months": float(returns.gt(0.0).mean()),
                    }
                )
    return pd.DataFrame(records)


def _assert_taskd_existing_reproduction(new_table: pd.DataFrame) -> None:
    existing = pd.read_csv(REVISION_CONSTRAINED_DIR / "constrained_summary.csv")
    compare_subperiods = [name for name, _, _ in TASKD_SUBPERIODS if name != "early_2015_2016"]
    metrics = [
        "months",
        "annualized_net_return",
        "net_sharpe",
        "max_drawdown",
        "annualized_spread_cost",
        "annualized_impact_cost",
        "average_effective_n",
    ]
    existing = existing[existing["subperiod"].isin(compare_subperiods)].copy()
    new = new_table[new_table["subperiod"].isin(compare_subperiods)].copy()
    merged = new.merge(
        existing[
            [
                "strategy",
                "constraint",
                "subperiod",
                "aum_label",
                *metrics,
            ]
        ],
        on=["strategy", "constraint", "subperiod", "aum_label"],
        how="left",
        suffixes=("", "_existing"),
        validate="one_to_one",
    )
    if merged[[f"{metric}_existing" for metric in metrics]].isna().any().any():
        raise RuntimeError("Task D could not match existing subperiod rows")
    for metric in metrics:
        left = pd.to_numeric(merged[metric], errors="coerce")
        right = pd.to_numeric(merged[f"{metric}_existing"], errors="coerce")
        if metric == "months":
            ok = left.astype(int).eq(right.astype(int)).all()
        else:
            ok = np.allclose(left, right, rtol=0.0, atol=1e-10)
        if not ok:
            diff = (left - right).abs().max()
            raise RuntimeError(f"Task D existing subperiod mismatch for {metric}: {diff}")


def run_task_d(output_dir: Path = CLOSURE_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(
        REVISION_CONSTRAINED_DIR / "constrained_monthly.csv",
        parse_dates=["date", "target_date"],
    )
    coverage = monthly[
        [
            "date",
            "target_date",
            "universe_n",
            "observed_spread_fraction",
            "median_half_spread_bps",
        ]
    ].drop_duplicates(["date", "target_date"])
    coverage = coverage.sort_values("date").rename(
        columns={"universe_n": "valid_observed_spread_names"}
    )
    coverage["coverage_period"] = np.where(
        coverage["target_date"].dt.year.le(2016),
        "early_2015_2016",
        "2017_onward",
    )
    coverage.to_csv(output_dir / "taskD_spread_coverage_by_month.csv", index=False)

    subperiods = _taskd_subperiod_table(monthly)
    _assert_taskd_existing_reproduction(subperiods)
    subperiods.to_csv(output_dir / "taskD_subperiods.csv", index=False)
    coverage_summary = (
        coverage.groupby("coverage_period")["valid_observed_spread_names"]
        .agg(["count", "mean", "min", "median", "max"])
        .reset_index()
    )
    coverage_summary.to_csv(output_dir / "taskD_spread_coverage_summary.csv", index=False)
    manifest = {
        "task": "D",
        "source_monthly": str(REVISION_CONSTRAINED_DIR / "constrained_monthly.csv"),
        "source_summary": str(REVISION_CONSTRAINED_DIR / "constrained_summary.csv"),
        "existing_subperiod_reproduction": "pass",
        "outputs": {
            "taskD_subperiods": str(output_dir / "taskD_subperiods.csv"),
            "taskD_spread_coverage_by_month": str(
                output_dir / "taskD_spread_coverage_by_month.csv"
            ),
            "taskD_spread_coverage_summary": str(
                output_dir / "taskD_spread_coverage_summary.csv"
            ),
        },
        "rows": {
            "taskD_subperiods": int(len(subperiods)),
            "taskD_spread_coverage_by_month": int(len(coverage)),
        },
    }
    (output_dir / "taskD_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _feature_mean_training_ic(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_return_rank",
) -> pd.Series:
    monthly: list[pd.Series] = []
    for _, month in train.groupby("date", sort=True):
        if len(month) < 20:
            continue
        monthly.append(month[feature_columns].corrwith(month[target_column]))
    if not monthly:
        return pd.Series(0.0, index=feature_columns)
    return pd.concat(monthly, axis=1).T.mean(axis=0).reindex(feature_columns).fillna(0.0)


def estimated_feature_signs(
    train: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_return_rank",
) -> tuple[pd.Series, pd.Series]:
    """Estimate one sign per feature from the supplied training window only."""
    mean_ic = _feature_mean_training_ic(train, feature_columns, target_column)
    signs = pd.Series(np.where(mean_ic.ge(0.0), 1, -1), index=feature_columns)
    return signs.astype(int), mean_ic


def literature_feature_signs(feature_columns: list[str]) -> pd.Series:
    missing = sorted(set(feature_columns) - set(LITERATURE_SIGN_BY_FEATURE))
    if missing:
        raise RuntimeError(f"Missing literature signs for features: {missing}")
    return pd.Series(
        {column: int(LITERATURE_SIGN_BY_FEATURE[column]) for column in feature_columns},
        dtype=int,
    )


def _composite_score(frame: pd.DataFrame, feature_columns: list[str], signs: pd.Series) -> np.ndarray:
    ordered = signs.reindex(feature_columns).to_numpy(dtype=float)
    return frame[feature_columns].to_numpy(dtype="float32", copy=False) @ ordered / len(feature_columns)


def _taske_predictions_and_signs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = FEATURE_SETS["compustat_enriched"]
    panel = load_model_panel(
        COMPUSTAT_PANEL,
        delisting_audit_path=DELISTING_AUDIT if DELISTING_AUDIT.exists() else None,
        feature_columns=feature_columns,
    )
    base_columns = [
        "date",
        "target_date",
        "ric",
        "target_return_1m",
        "target_return_rank",
        "company_market_cap",
        "market_cap_percentile",
        "screen_country",
        "TR.TRBCECONOMICSECTOR",
        "is_delisting_candidate",
        "retire_month",
    ]
    prediction_frames: list[pd.DataFrame] = []
    sign_records: list[dict[str, Any]] = []
    fixed_signs = literature_feature_signs(feature_columns)
    for year, cutoff, train_mask, test_mask in walk_forward_slices(panel, 2015, 2026):
        train = panel.loc[train_mask & panel["target_return_rank"].notna()].copy()
        test = panel.loc[test_mask].copy()
        if train.empty or test.empty:
            continue
        eval_start = pd.Timestamp(test["date"].min())
        max_train_signal_date = pd.Timestamp(train["date"].max())
        max_train_target_date = pd.Timestamp(train["target_date"].max())
        if not (max_train_signal_date < eval_start and max_train_target_date < eval_start):
            raise RuntimeError(
                f"Task E sign window leak for {year}: "
                f"train_signal={max_train_signal_date}, train_target={max_train_target_date}, "
                f"eval_start={eval_start}"
            )
        estimated_signs, mean_ic = estimated_feature_signs(train, feature_columns)
        for variant, signs, source_mean_ic in [
            ("estimated_sign", estimated_signs, mean_ic),
            (
                "literature_sign",
                fixed_signs,
                pd.Series(np.nan, index=feature_columns, dtype=float),
            ),
        ]:
            output = test[base_columns].copy()
            output["prediction"] = _composite_score(test, feature_columns, signs)
            output["model"] = f"composite_{variant}"
            output["base_model"] = "composite"
            output["target_mode"] = "rank"
            output["test_year"] = year
            output["train_label_cutoff"] = cutoff
            prediction_frames.append(output)
            for feature in feature_columns:
                sign_records.append(
                    {
                        "sign_variant": variant,
                        "test_year": year,
                        "feature": feature,
                        "sign": int(signs[feature]),
                        "mean_train_ic": float(source_mean_ic[feature])
                        if pd.notna(source_mean_ic[feature])
                        else np.nan,
                        "train_rows": int(len(train)),
                        "train_signal_start": train["date"].min(),
                        "max_train_signal_date": max_train_signal_date,
                        "max_train_target_date": max_train_target_date,
                        "train_label_cutoff": cutoff,
                        "evaluation_start_date": eval_start,
                        "evaluation_end_date": pd.Timestamp(test["date"].max()),
                        "assert_train_dates_precede_eval": True,
                    }
                )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    signs = pd.DataFrame(sign_records)
    predictions.to_parquet(
        output_dir / "taskE_composite_predictions.parquet",
        index=False,
        compression="zstd",
    )
    signs.to_csv(output_dir / "taskE_signs_by_fold.csv", index=False)
    return predictions, signs


def _taske_common_frame(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    base = _load_common_prediction_panel()
    features = _load_common_features(["log_trading_value_eur"])
    selected = predictions[predictions["model"].eq(model)][
        ["date", "ric", "prediction"]
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    frame = base.merge(
        selected,
        on=["date", "ric"],
        how="left",
        validate="one_to_one",
    ).rename(columns={"prediction": "composite_score"})
    frame = frame.merge(
        features[["date", "ric", "log_trading_value_eur"]],
        on=["date", "ric"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != COMMON_SAMPLE_ROWS or frame["composite_score"].isna().any():
        raise RuntimeError(f"Task E common frame mismatch for {model}")
    return frame


def _comparison_stats(monthly_left: pd.Series, monthly_right: pd.Series) -> dict[str, float]:
    diff = (monthly_left - monthly_right).dropna()
    stats = _hac_summary(diff)
    return {
        "estimate": stats["estimate"],
        "hac_t": stats["hac_t"],
        "p_value": stats["p_value"],
        "ci_lo": stats["ci_lo"],
        "ci_hi": stats["ci_hi"],
        "mde": stats["mde"],
        "n_months": stats["n_months"],
    }


def _taske_flat_cost_summary(predictions: pd.DataFrame, model: str) -> dict[str, float]:
    selected = predictions[predictions["model"].eq(model)].copy()
    selected = selected.merge(
        _common_sample_keys(),
        on=["date", "ric"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected) != COMMON_SAMPLE_ROWS:
        raise RuntimeError(f"Task E flat-cost sample mismatch for {model}")
    monthly = construct_monthly_portfolios(selected, quantile=0.10)
    metrics = prediction_metrics(selected)
    summary = portfolio_summary(monthly, metrics, cost_grid_bps=(25,), risk_free=None)
    row = summary[
        summary["model"].eq(model)
        & summary["weighting"].eq("equal")
        & summary["universe_variant"].eq("standard_ex_bottom_5pct")
        & summary["portfolio"].eq("long_short")
        & summary["cost_bps"].eq(25)
    ]
    if row.empty:
        raise RuntimeError(f"Task E missing flat-cost portfolio summary for {model}")
    row = row.iloc[0]
    return {
        "flat_cost_ew_ls_months": int(row["months"]),
        "flat_cost_ew_ls_annual_net_return": float(row["annualized_net_mean_return"]),
        "flat_cost_ew_ls_sharpe": float(row["net_sharpe"]),
        "flat_cost_ew_ls_turnover": float(row["average_monthly_turnover"]),
    }


def run_task_e(output_dir: Path = CLOSURE_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions, signs = _taske_predictions_and_signs(output_dir)
    rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    gradient_frames: list[pd.DataFrame] = []
    for variant in ["estimated_sign", "literature_sign"]:
        model = f"composite_{variant}"
        frame = _taske_common_frame(predictions, model)
        mean_ic, ic_ir, months = _mean_ic_and_ir(frame, "composite_score")
        composite_ic = _monthly_rank_ic(frame, "composite_score")
        momentum_ic = _monthly_rank_ic(frame, "momentum_rank")
        ridge_ic = _monthly_rank_ic(frame, "ridge_rank")
        comparisons = {
            "composite_minus_momentum": _comparison_stats(composite_ic, momentum_ic),
            "ridge_minus_composite": _comparison_stats(ridge_ic, composite_ic),
        }
        holm_values = multipletests(
            [comparisons[name]["p_value"] for name in comparisons],
            method="holm",
        )[1]
        for holm_p, name in zip(holm_values, comparisons, strict=True):
            comparisons[name]["holm_p"] = float(holm_p)
            comparison_rows.append(
                {
                    "sign_variant": variant,
                    "comparison": name,
                    **comparisons[name],
                }
            )

        gradients, bucket_premia = _taska_capacity_summary(
            frame,
            run_label=model,
            score_column="composite_score",
            benchmark_column="momentum_rank",
        )
        gradients["holm_p"] = multipletests(
            gradients["p_value"].fillna(1.0),
            method="holm",
        )[1]
        gradients["sign_variant"] = variant
        gradient_frames.append(gradients)
        flat = _taske_flat_cost_summary(predictions, model)

        row: dict[str, Any] = {
            "sign_variant": variant,
            "mean_ic": mean_ic,
            "ic_ir": ic_ir,
            "n_months": months,
            "n_obs": COMMON_SAMPLE_ROWS,
            **flat,
        }
        for name, stats in comparisons.items():
            prefix = "comp_minus_mom" if name == "composite_minus_momentum" else "ridge_minus_comp"
            row[f"{prefix}_estimate"] = stats["estimate"]
            row[f"{prefix}_hac_t"] = stats["hac_t"]
            row[f"{prefix}_holm_p"] = stats["holm_p"]
            row[f"{prefix}_ci_lo"] = stats["ci_lo"]
            row[f"{prefix}_ci_hi"] = stats["ci_hi"]
            row[f"{prefix}_mde"] = stats["mde"]
        for gradient in gradients.itertuples(index=False):
            prefix = "market_cap" if gradient.partition == "market_cap" else "trading_value"
            row[f"{prefix}_gradient"] = float(gradient.gradient)
            row[f"{prefix}_hac_t"] = float(gradient.hac_t)
            row[f"{prefix}_holm_p"] = float(gradient.holm_p)
            row[f"{prefix}_ci_lo"] = float(gradient.ci_lo)
            row[f"{prefix}_ci_hi"] = float(gradient.ci_hi)
            row[f"{prefix}_mde"] = float(gradient.mde)
            row[f"{prefix}_ic_least"] = float(gradient.ic_least)
            row[f"{prefix}_ic_most"] = float(gradient.ic_most)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "taskE_composite.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "taskE_comparison_details.csv",
        index=False,
    )
    gradient_table = pd.concat(gradient_frames, ignore_index=True)
    gradient_table.to_csv(output_dir / "taskE_gradient_details.csv", index=False)
    max_dates = signs[
        [
            "sign_variant",
            "test_year",
            "max_train_signal_date",
            "max_train_target_date",
            "evaluation_start_date",
        ]
    ].drop_duplicates()
    manifest = {
        "task": "E",
        "common_sample_rows": COMMON_SAMPLE_ROWS,
        "common_sample_months": COMMON_SAMPLE_MONTHS,
        "sign_assertion": "max training signal date and max training target date precede evaluation start in every fold",
        "sign_date_checks": max_dates.to_dict(orient="records"),
        "literature_sign_note": (
            "Fixed ex-ante signs follow standard anomaly direction conventions: "
            "value, profitability, payout, momentum and cash-flow quality positive; "
            "size, short reversal, volatility, maximum return, investment, accruals, "
            "leverage and liquidity/price proxies negative."
        ),
        "outputs": {
            "taskE_composite": str(output_dir / "taskE_composite.csv"),
            "taskE_signs_by_fold": str(output_dir / "taskE_signs_by_fold.csv"),
            "taskE_composite_predictions": str(
                output_dir / "taskE_composite_predictions.parquet"
            ),
            "taskE_comparison_details": str(
                output_dir / "taskE_comparison_details.csv"
            ),
            "taskE_gradient_details": str(output_dir / "taskE_gradient_details.csv"),
        },
        "rows": {
            "taskE_composite": int(len(table)),
            "taskE_signs_by_fold": int(len(signs)),
        },
    }
    (output_dir / "taskE_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    return manifest


def build_fixed_count_long_short_legs(
    month: pd.DataFrame,
    score_column: str,
    leg_size: int,
) -> pd.DataFrame:
    """Build equal-weighted long and short legs with exactly ``leg_size`` names."""
    if leg_size <= 0:
        raise ValueError("leg_size must be positive")
    work = month[["ric", score_column, "target_return_1m"]].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work["target_return_1m"] = pd.to_numeric(work["target_return_1m"], errors="coerce")
    work = work.dropna(subset=["ric", score_column, "target_return_1m"])
    if work["ric"].duplicated().any():
        raise RuntimeError("Fixed-count leg input contains duplicate RICs")
    if len(work) < 2 * leg_size:
        return pd.DataFrame(columns=["ric", "side", "weight"])
    work = work.sort_values([score_column, "ric"], ascending=[False, True], kind="mergesort")
    long = work.head(leg_size)[["ric"]].copy()
    short = work.tail(leg_size)[["ric"]].copy()
    if set(long["ric"]).intersection(short["ric"]):
        raise RuntimeError("Fixed-count long and short legs overlap")
    long["side"] = "long"
    long["weight"] = 1.0 / leg_size
    short["side"] = "short"
    short["weight"] = -1.0 / leg_size
    return pd.concat([long, short], ignore_index=True)


def fixed_count_one_way_turnover(
    current_weights: pd.Series,
    previous_weights: pd.Series | None,
) -> float:
    previous = (
        previous_weights.astype(float)
        if previous_weights is not None
        else pd.Series(dtype=float)
    )
    current = current_weights.astype(float)
    names = current.index.union(previous.index)
    return float(0.5 * (current.reindex(names, fill_value=0.0) - previous.reindex(names, fill_value=0.0)).abs().sum())


def _taskc_base_frame() -> pd.DataFrame:
    base = _load_common_prediction_panel()
    features = _load_common_features(["log_trading_value_eur"])
    frame = base.merge(
        features[["date", "ric", "log_trading_value_eur"]],
        on=["date", "ric"],
        how="left",
        validate="one_to_one",
    )
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    if len(frame) != COMMON_SAMPLE_ROWS:
        raise RuntimeError("Task C base frame lost common-sample rows")
    return frame


def _taskc_bucketed(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
    bucketed = _bucketed_frame(frame, partition)
    bucketed = bucketed[bucketed["bucket"].isin(TASKC_BUCKETS[partition])].copy()
    duplicates = bucketed.duplicated(["partition", "bucket", "date", "ric"]).sum()
    if duplicates:
        raise RuntimeError(f"Task C bucket assignment contains {duplicates} duplicates")
    months = bucketed.groupby("bucket", observed=True)["date"].nunique()
    missing = sorted(set(TASKC_BUCKETS[partition]) - set(months.index))
    if missing or not months.eq(COMMON_SAMPLE_MONTHS).all():
        raise RuntimeError(
            f"Task C bucket coverage mismatch for {partition}: "
            f"missing={missing}, months={months.to_dict()}"
        )
    return bucketed


def _taskc_monthly_returns(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for leg_size in TASKC_LEG_SIZES:
        for partition, _, _, _ in TASKA_PARTITIONS:
            bucketed = _taskc_bucketed(frame, partition)
            for model, score_column in TASKC_MODELS.items():
                previous: dict[str, pd.Series] = {}
                for (bucket, date), month in bucketed.groupby(
                    ["bucket", "date"],
                    sort=True,
                    observed=True,
                ):
                    legs = build_fixed_count_long_short_legs(month, score_column, leg_size)
                    if legs.empty:
                        continue
                    weights = legs.set_index("ric")["weight"].astype(float)
                    returns = month.drop_duplicates("ric").set_index("ric")[
                        "target_return_1m"
                    ].astype(float)
                    gross_return = float((weights * returns.reindex(weights.index)).sum())
                    turnover = fixed_count_one_way_turnover(weights, previous.get(bucket))
                    previous[bucket] = weights
                    records.append(
                        {
                            "model": model,
                            "partition": partition,
                            "bucket": bucket,
                            "leg_size": int(leg_size),
                            "date": pd.Timestamp(date),
                            "target_date": pd.Timestamp(month["target_date"].iloc[0]),
                            "gross_return": gross_return,
                            "one_way_turnover": turnover,
                            "net_return": gross_return
                            - turnover * TASKC_COST_BPS / 10_000.0,
                            "long_n": int((legs["side"] == "long").sum()),
                            "short_n": int((legs["side"] == "short").sum()),
                            "bucket_names": int(month[["date", "ric"]].drop_duplicates().shape[0]),
                        }
                    )
    monthly = pd.DataFrame(records).sort_values(
        ["leg_size", "partition", "bucket", "model", "date"]
    )
    expected_groups = len(TASKC_LEG_SIZES) * len(TASKC_BUCKETS) * 4 * len(TASKC_MODELS)
    grouped_months = monthly.groupby(
        ["model", "partition", "bucket", "leg_size"],
        observed=True,
    )["date"].nunique()
    if len(grouped_months) != expected_groups or not grouped_months.eq(COMMON_SAMPLE_MONTHS).all():
        raise RuntimeError(
            "Task C monthly return coverage mismatch: "
            f"groups={len(grouped_months)}, months={grouped_months.to_dict()}"
        )
    if not monthly["long_n"].eq(monthly["leg_size"]).all() or not monthly["short_n"].eq(monthly["leg_size"]).all():
        raise RuntimeError("Task C fixed-count leg assertion failed")
    return monthly.reset_index(drop=True)


def _taskc_return_summary(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for (model, partition, bucket, leg_size), group in monthly.groupby(
        ["model", "partition", "bucket", "leg_size"],
        sort=True,
        observed=True,
    ):
        group = group.sort_values("date")
        records.append(
            {
                "model": model,
                "partition": partition,
                "bucket": bucket,
                "leg_size": int(leg_size),
                "months": int(len(group)),
                "annualized_gross_return": float(group["gross_return"].mean() * 12.0),
                "annualized_net_return": float(group["net_return"].mean() * 12.0),
                "return_to_volatility": _annualized_return_to_vol(group["net_return"]),
                "one_way_turnover": float(group["one_way_turnover"].mean()),
                "long_n_min": int(group["long_n"].min()),
                "long_n_max": int(group["long_n"].max()),
                "short_n_min": int(group["short_n"].min()),
                "short_n_max": int(group["short_n"].max()),
                "bucket_names_mean": float(group["bucket_names"].mean()),
                "bucket_names_min": int(group["bucket_names"].min()),
                "bucket_names_max": int(group["bucket_names"].max()),
                "cost_bps": TASKC_COST_BPS,
            }
        )
    summary = pd.DataFrame(records)

    diff_records: list[dict[str, Any]] = []
    for (partition, bucket, leg_size), group in monthly.groupby(
        ["partition", "bucket", "leg_size"],
        sort=True,
        observed=True,
    ):
        pivot = group.pivot(index="date", columns="model", values="net_return")
        diff = (pivot["Ridge"] - pivot["Momentum"]).dropna()
        stats = _hac_summary(diff)
        diff_records.append(
            {
                "model": "Ridge",
                "partition": partition,
                "bucket": bucket,
                "leg_size": int(leg_size),
                "ridge_minus_momentum_monthly_net_diff": stats["estimate"],
                "ridge_minus_momentum_annual_net_diff": stats["estimate"] * 12.0,
                "ridge_minus_momentum_hac_t": stats["hac_t"],
                "ridge_minus_momentum_p": stats["p_value"],
                "ridge_minus_momentum_ci_lo": stats["ci_lo"],
                "ridge_minus_momentum_ci_hi": stats["ci_hi"],
                "ridge_minus_momentum_mde": stats["mde"],
                "ridge_minus_momentum_months": stats["n_months"],
            }
        )
    diff_table = pd.DataFrame(diff_records)
    diff_table["ridge_minus_momentum_holm_p"] = multipletests(
        diff_table["ridge_minus_momentum_p"].fillna(1.0),
        method="holm",
    )[1]
    summary = summary.merge(
        diff_table,
        on=["model", "partition", "bucket", "leg_size"],
        how="left",
        validate="one_to_one",
    )
    order_columns = [
        "model",
        "partition",
        "bucket",
        "leg_size",
        "months",
        "annualized_gross_return",
        "annualized_net_return",
        "return_to_volatility",
        "one_way_turnover",
        "ridge_minus_momentum_monthly_net_diff",
        "ridge_minus_momentum_annual_net_diff",
        "ridge_minus_momentum_hac_t",
        "ridge_minus_momentum_holm_p",
        "ridge_minus_momentum_ci_lo",
        "ridge_minus_momentum_ci_hi",
        "ridge_minus_momentum_mde",
        "ridge_minus_momentum_months",
        "long_n_min",
        "long_n_max",
        "short_n_min",
        "short_n_max",
        "bucket_names_mean",
        "bucket_names_min",
        "bucket_names_max",
        "cost_bps",
    ]
    return summary[order_columns], diff_table


def _taskc_did(monthly: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    bucket_diff_rows: list[dict[str, Any]] = []
    for (partition, leg_size), group in monthly.groupby(
        ["partition", "leg_size"],
        sort=True,
        observed=True,
    ):
        pivot = group.pivot_table(
            index=["date", "bucket"],
            columns="model",
            values="net_return",
            aggfunc="last",
        )
        pivot["ridge_minus_momentum"] = pivot["Ridge"] - pivot["Momentum"]
        wide = pivot["ridge_minus_momentum"].unstack("bucket")
        least = TASKA_PARTITIONS[0][2] if partition == "market_cap" else TASKA_PARTITIONS[1][2]
        most = TASKA_PARTITIONS[0][3] if partition == "market_cap" else TASKA_PARTITIONS[1][3]
        if least not in wide or most not in wide:
            raise RuntimeError(f"Task C DiD missing buckets for {partition}")
        did = (wide[least] - wide[most]).dropna()
        stats = _hac_summary(did)
        bucket_means = wide.mean().reindex(TASKC_BUCKETS[partition])
        order = ">".join(bucket_means.sort_values(ascending=False).index.astype(str))
        records.append(
            {
                "partition": partition,
                "leg_size": int(leg_size),
                "least_bucket": least,
                "most_bucket": most,
                "least_monthly_diff": float(wide[least].mean()),
                "most_monthly_diff": float(wide[most].mean()),
                "did_monthly_net_diff": stats["estimate"],
                "did_annual_net_diff": stats["estimate"] * 12.0,
                "hac_t": stats["hac_t"],
                "p_value": stats["p_value"],
                "ci_lo": stats["ci_lo"],
                "ci_hi": stats["ci_hi"],
                "mde": stats["mde"],
                "n_months": stats["n_months"],
                "least_exceeds_most": bool(stats["estimate"] > 0.0),
                "bucket_diff_order_desc": order,
            }
        )
        for bucket, value in bucket_means.items():
            bucket_diff_rows.append(
                {
                    "partition": partition,
                    "leg_size": int(leg_size),
                    "bucket": bucket,
                    "ridge_minus_momentum_monthly_net_diff": float(value),
                }
            )
    did_table = pd.DataFrame(records)
    did_table["holm_p"] = multipletests(did_table["p_value"].fillna(1.0), method="holm")[1]
    for partition, group in did_table.groupby("partition", sort=True):
        primary = group[group["leg_size"].eq(TASKC_PRIMARY_LEG_SIZE)]
        if primary.empty:
            stable = pd.Series(False, index=group.index)
        else:
            primary_positive = bool(primary.iloc[0]["least_exceeds_most"])
            stable = group["least_exceeds_most"].eq(primary_positive)
        did_table.loc[group.index, "ordering_stable_vs_leg50"] = stable.to_numpy(dtype=bool)
    pd.DataFrame(bucket_diff_rows).to_csv(
        output_dir / "taskC_bucket_net_diff_details.csv",
        index=False,
    )
    return did_table[
        [
            "partition",
            "leg_size",
            "least_bucket",
            "most_bucket",
            "least_monthly_diff",
            "most_monthly_diff",
            "did_monthly_net_diff",
            "did_annual_net_diff",
            "hac_t",
            "holm_p",
            "ci_lo",
            "ci_hi",
            "mde",
            "n_months",
            "least_exceeds_most",
            "ordering_stable_vs_leg50",
            "bucket_diff_order_desc",
        ]
    ]


def run_task_c(output_dir: Path = CLOSURE_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _taskc_base_frame()
    monthly = _taskc_monthly_returns(frame)
    summary, diff_table = _taskc_return_summary(monthly)
    did = _taskc_did(monthly, output_dir)
    monthly.to_csv(output_dir / "taskC_monthly_fixed_count_returns.csv", index=False)
    summary.to_csv(output_dir / "taskC_return_space_gradient.csv", index=False)
    did.to_csv(output_dir / "taskC_did.csv", index=False)
    manifest = {
        "task": "C",
        "common_sample_rows": COMMON_SAMPLE_ROWS,
        "common_sample_months": COMMON_SAMPLE_MONTHS,
        "models": list(TASKC_MODELS),
        "leg_sizes": list(TASKC_LEG_SIZES),
        "primary_leg_size": TASKC_PRIMARY_LEG_SIZE,
        "cost_bps": TASKC_COST_BPS,
        "hac_lags": HAC_LAGS,
        "outputs": {
            "taskC_return_space_gradient": str(
                output_dir / "taskC_return_space_gradient.csv"
            ),
            "taskC_did": str(output_dir / "taskC_did.csv"),
            "taskC_monthly_fixed_count_returns": str(
                output_dir / "taskC_monthly_fixed_count_returns.csv"
            ),
            "taskC_bucket_net_diff_details": str(
                output_dir / "taskC_bucket_net_diff_details.csv"
            ),
        },
        "rows": {
            "taskC_return_space_gradient": int(len(summary)),
            "taskC_did": int(len(did)),
            "taskC_monthly_fixed_count_returns": int(len(monthly)),
            "ridge_minus_momentum_bucket_tests": int(len(diff_table)),
        },
    }
    (output_dir / "taskC_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _taskb_combined_predictions(output_dir: Path) -> Path:
    frames: list[pd.DataFrame] = []
    for signal, spec in TASKB_SIGNALS.items():
        frame = pd.read_parquet(spec["source"])
        frame = frame[frame["model"].eq(spec["source_model"])].copy()
        if frame.empty:
            raise RuntimeError(
                f"Task B source model {spec['source_model']} missing for {signal}"
            )
        frame["model"] = spec["model"]
        frame["base_model"] = signal.lower()
        frame["target_mode"] = "rank"
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["model", "date", "ric"],
    )
    duplicates = combined.duplicated(["model", "date", "ric"]).sum()
    if duplicates:
        raise RuntimeError(f"Task B combined predictions have {duplicates} duplicates")
    path = output_dir / "taskB_combined_predictions.parquet"
    combined.to_parquet(path, index=False, compression="zstd")
    return path


def _taskb_fixed_choices() -> list[dict[str, str]]:
    return [
        {
            "strategy": spec["strategy"],
            "model": spec["model"],
            "rung": "top_500_observed_spread",
        }
        for spec in TASKB_SIGNALS.values()
    ]


def _taskb_specs() -> list[ConstraintSpec]:
    return [
        ConstraintSpec(TASKB_CONSTRAINT, 0.05, 0.40, 0.40, 0.005),
        ConstraintSpec(TASKB_NO_TURNOVER_CONSTRAINT, 0.05, 0.40, 0.40, 0.0),
    ]


def _bootstrap_annual_mean(values: pd.Series, seed: int) -> dict[str, float]:
    return project_stats.stationary_bootstrap_metric_ci(
        values,
        metric="annualized_mean",
        expected_block=TASKB_BOOTSTRAP_BLOCK,
        n_boot=TASKB_BOOTSTRAP_REPETITIONS,
        seed=seed,
    )


def _taskb_active_intervals(relative_monthly: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    monthly = relative_monthly[relative_monthly["constraint"].eq(TASKB_CONSTRAINT)]
    for strategy, group in monthly.groupby("strategy", sort=True):
        for aum in TASKB_AUMS:
            label = _aum_label(aum)
            active = pd.to_numeric(group[f"active_return_{label}"], errors="coerce")
            result = _bootstrap_annual_mean(
                active,
                TASKB_SEED + int(aum / 1_000_000),
            )
            records.append(
                {
                    "strategy": strategy,
                    "aum_label": label,
                    "active_ci_lo": result["ci_low"],
                    "active_ci_hi": result["ci_high"],
                }
            )
    return pd.DataFrame(records)


def _build_taskb_summary(raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(raw_dir / "constrained_summary.csv")
    relative = pd.read_csv(raw_dir / "benchmark_relative_summary.csv")
    relative_monthly = pd.read_csv(raw_dir / "benchmark_relative_monthly.csv")

    summary = summary[
        summary["constraint"].eq(TASKB_CONSTRAINT) & summary["subperiod"].eq("full")
    ].copy()
    relative = relative[
        relative["constraint"].eq(TASKB_CONSTRAINT) & relative["subperiod"].eq("full")
    ].copy()
    active_intervals = _taskb_active_intervals(relative_monthly)
    merged = summary.merge(
        relative,
        on=["strategy", "constraint", "subperiod", "aum_eur", "aum_label", "months"],
        how="left",
        suffixes=("", "_relative"),
        validate="one_to_one",
    ).merge(
        active_intervals,
        on=["strategy", "aum_label"],
        how="left",
        validate="one_to_one",
    )
    merged["signal"] = merged["strategy"].map(TASKB_STRATEGY_TO_SIGNAL)
    if merged["signal"].isna().any():
        raise RuntimeError("Task B summary contains unmapped strategies")

    out = pd.DataFrame(
        {
            "signal": merged["signal"],
            "aum_eur": merged["aum_eur"],
            "months": merged["months"],
            "annual_net_return": merged["annualized_net_return"],
            "return_to_vol": merged["net_sharpe"],
            "one_way_turnover": merged["average_monthly_turnover"],
            "effective_n": merged["average_effective_n"],
            "spread_cost": merged["annualized_spread_cost"],
            "impact_cost": merged["annualized_impact_cost"],
            "max_name": merged["average_max_single_name_weight"],
            "max_country": merged["average_max_country_weight"],
            "max_sector": merged["average_max_sector_weight"],
            "active_return": merged["annualized_active_return"],
            "active_ci_lo": merged["active_ci_lo"],
            "active_ci_hi": merged["active_ci_hi"],
            "information_ratio": merged["information_ratio"],
            "annual_alpha": merged["alpha_annualized"],
            "alpha_p": merged["alpha_p_two_sided"],
        }
    )
    order = {name: index for index, name in enumerate(TASKB_SIGNALS)}
    out = out.assign(_order=out["signal"].map(order)).sort_values(
        ["_order", "aum_eur"],
    )
    out = out.drop(columns="_order").reset_index(drop=True)

    expected_revision = {
        "10m": (14.99, 0.853),
        "100m": (14.26, 0.810),
        "500m": (12.94, 0.734),
    }
    revision = out[out["signal"].eq("Revision")].copy()
    for label, (expected_return, expected_rtv) in expected_revision.items():
        row = revision[revision["aum_eur"].eq(float(label.removesuffix("m")) * 1_000_000)]
        if row.empty:
            raise RuntimeError(f"Missing revision {label} row in Task B")
        values = row.iloc[0]
        if round(float(values["annual_net_return"]) * 100.0, 2) != expected_return:
            raise RuntimeError(
                f"Revision {label} net return mismatch: "
                f"{float(values['annual_net_return']) * 100.0:.4f}"
            )
        if round(float(values["return_to_vol"]), 3) != expected_rtv:
            raise RuntimeError(
                f"Revision {label} return-to-vol mismatch: "
                f"{float(values['return_to_vol']):.6f}"
            )

    out.to_csv(output_dir / "taskB_constrained_by_signal.csv", index=False)
    return out


def _taskb_difference_rows(raw_dir: Path) -> pd.DataFrame:
    monthly = pd.read_csv(raw_dir / "constrained_monthly.csv", parse_dates=["target_date"])
    relative = pd.read_csv(
        raw_dir / "benchmark_relative_monthly.csv",
        parse_dates=["target_date"],
    )
    monthly = monthly[monthly["constraint"].eq(TASKB_CONSTRAINT)].copy()
    relative = relative[relative["constraint"].eq(TASKB_CONSTRAINT)].copy()
    records: list[dict[str, Any]] = []
    momentum_strategy = TASKB_SIGNALS["Momentum"]["strategy"]
    for signal, spec in TASKB_SIGNALS.items():
        if signal == "Momentum":
            continue
        strategy = spec["strategy"]
        for aum in TASKB_AUMS:
            label = _aum_label(aum)
            left = (
                monthly[monthly["strategy"].eq(strategy)]
                .set_index("target_date")[f"net_return_{label}"]
                .astype(float)
            )
            right = (
                monthly[monthly["strategy"].eq(momentum_strategy)]
                .set_index("target_date")[f"net_return_{label}"]
                .astype(float)
            )
            left_active = (
                relative[relative["strategy"].eq(strategy)]
                .set_index("target_date")[f"active_return_{label}"]
                .astype(float)
            )
            right_active = (
                relative[relative["strategy"].eq(momentum_strategy)]
                .set_index("target_date")[f"active_return_{label}"]
                .astype(float)
            )
            dates = left.index.intersection(right.index)
            dates = dates.intersection(left_active.index).intersection(right_active.index)
            if len(dates) < 24:
                raise RuntimeError(f"Task B has too few common months for {signal} {label}")
            net_diff = left.reindex(dates) - right.reindex(dates)
            active_diff = left_active.reindex(dates) - right_active.reindex(dates)
            net_ci = _bootstrap_annual_mean(
                net_diff,
                TASKB_SEED + 1000 + int(aum / 1_000_000),
            )
            active_ci = _bootstrap_annual_mean(
                active_diff,
                TASKB_SEED + 2000 + int(aum / 1_000_000),
            )
            rtv = project_stats.bootstrap_sharpe_diff(
                left.reindex(dates),
                right.reindex(dates),
                np.zeros(len(dates)),
                expected_block=TASKB_BOOTSTRAP_BLOCK,
                n_boot=TASKB_BOOTSTRAP_REPETITIONS,
                seed=TASKB_SEED + 3000 + int(aum / 1_000_000),
            )
            records.append(
                {
                    "signal": signal,
                    "benchmark_signal": "Momentum",
                    "aum_eur": float(aum),
                    "months": int(len(dates)),
                    "annual_net_return_diff": net_ci["point"],
                    "net_ci_lo": net_ci["ci_low"],
                    "net_ci_hi": net_ci["ci_high"],
                    "net_p_two_sided": net_ci["p_two_sided_zero"],
                    "return_to_vol_diff": rtv["delta_sharpe"],
                    "return_to_vol_ci_lo": rtv["ci_low"],
                    "return_to_vol_ci_hi": rtv["ci_high"],
                    "return_to_vol_p_two_sided": rtv["p_two_sided"],
                    "active_return_diff": active_ci["point"],
                    "active_ci_lo": active_ci["ci_low"],
                    "active_ci_hi": active_ci["ci_high"],
                    "active_p_two_sided": active_ci["p_two_sided_zero"],
                    "bootstrap_expected_block": TASKB_BOOTSTRAP_BLOCK,
                    "bootstrap_repetitions": TASKB_BOOTSTRAP_REPETITIONS,
                }
            )
    return pd.DataFrame(records)


def _taskb_turnover_penalty_diagnostic(raw_dir: Path, output_dir: Path) -> pd.DataFrame:
    monthly = pd.read_csv(raw_dir / "constrained_monthly.csv", parse_dates=["target_date"])
    records: list[dict[str, Any]] = []
    for signal, spec in TASKB_SIGNALS.items():
        strategy = spec["strategy"]
        penalized = monthly[
            monthly["strategy"].eq(strategy) & monthly["constraint"].eq(TASKB_CONSTRAINT)
        ].set_index("target_date")
        no_penalty = monthly[
            monthly["strategy"].eq(strategy)
            & monthly["constraint"].eq(TASKB_NO_TURNOVER_CONSTRAINT)
        ].set_index("target_date")
        dates = penalized.index.intersection(no_penalty.index)
        for aum in TASKB_AUMS:
            label = _aum_label(aum)
            changed = (
                penalized.reindex(dates)["gross_return"].sub(
                    no_penalty.reindex(dates)["gross_return"],
                ).abs().gt(1e-10)
                | penalized.reindex(dates)[f"turnover_{label}"].sub(
                    no_penalty.reindex(dates)[f"turnover_{label}"],
                ).abs().gt(1e-8)
            )
            records.append(
                {
                    "signal": signal,
                    "aum_eur": float(aum),
                    "months": int(len(dates)),
                    "binding_months": int(changed.sum()),
                    "binding_fraction": float(changed.mean()),
                    "penalized_turnover": float(
                        penalized.reindex(dates)[f"turnover_{label}"].mean()
                    ),
                    "no_penalty_turnover": float(
                        no_penalty.reindex(dates)[f"turnover_{label}"].mean()
                    ),
                }
            )
    out = pd.DataFrame(records)
    out.to_csv(output_dir / "taskB_turnover_penalty_diagnostic.csv", index=False)
    return out


def run_task_b(output_dir: Path = CLOSURE_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = _taskb_combined_predictions(output_dir)
    raw_dir = output_dir / "taskB_constrained_runner"
    manifest = run_experiment(
        selected_path=None,
        predictions_path=combined_path,
        panel_path=STRICT_ESTIMATES_PANEL,
        liquidity_path=TASKB_LIQUIDITY if TASKB_LIQUIDITY.exists() else None,
        risk_path=TASKB_RISK if TASKB_RISK.exists() else None,
        market_path=TASKB_MARKET if TASKB_MARKET.exists() else None,
        output_dir=raw_dir,
        specs=_taskb_specs(),
        fixed_choices=_taskb_fixed_choices(),
        selected_strategy="taskB_not_used_fixed_only",
        aum_values=TASKB_AUMS,
        maximum_assets=500,
        fallback_half_spread_bps=25.0,
        impact_coefficient=0.10,
        bootstrap_repetitions=2_000,
        bootstrap_blocks=(3, 6, 12),
        random_state=TASKB_SEED,
        hac_lags=6,
        fixed_only=True,
    )
    table = _build_taskb_summary(raw_dir, output_dir)
    differences = _taskb_difference_rows(raw_dir)
    differences.to_csv(output_dir / "taskB_vs_momentum_differences.csv", index=False)
    turnover = _taskb_turnover_penalty_diagnostic(raw_dir, output_dir)
    task_manifest = {
        "task": "B",
        "combined_predictions": str(combined_path),
        "raw_runner_dir": str(raw_dir),
        "rows": {
            "taskB_constrained_by_signal": int(len(table)),
            "taskB_vs_momentum_differences": int(len(differences)),
            "taskB_turnover_penalty_diagnostic": int(len(turnover)),
        },
        "runner_rows": manifest["rows"],
        "bootstrap_expected_block": TASKB_BOOTSTRAP_BLOCK,
        "bootstrap_repetitions": TASKB_BOOTSTRAP_REPETITIONS,
        "outputs": {
            "taskB_constrained_by_signal": str(output_dir / "taskB_constrained_by_signal.csv"),
            "taskB_vs_momentum_differences": str(output_dir / "taskB_vs_momentum_differences.csv"),
            "taskB_turnover_penalty_diagnostic": str(
                output_dir / "taskB_turnover_penalty_diagnostic.csv"
            ),
        },
    }
    (output_dir / "taskB_manifest.json").write_text(json.dumps(task_manifest, indent=2))
    return task_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "taskA_diagnostics",
            "taskA_ablations",
            "taskB",
            "taskC",
            "taskD",
            "taskE",
        ],
        help="Closure task to run.",
    )
    parser.add_argument("--output-dir", type=Path, default=CLOSURE_DIR)
    parser.add_argument(
        "--include-histgbm-nomicro",
        action="store_true",
        help="Also run the optional Task A HistGBM no-microstructure ablation.",
    )
    args = parser.parse_args()

    if args.task == "taskA_diagnostics":
        manifest = run_task_a_diagnostics(args.output_dir)
    elif args.task == "taskA_ablations":
        manifest = run_task_a_ablations(
            args.output_dir,
            include_histgbm_nomicro=args.include_histgbm_nomicro,
        )
    elif args.task == "taskB":
        manifest = run_task_b(args.output_dir)
    elif args.task == "taskC":
        manifest = run_task_c(args.output_dir)
    elif args.task == "taskD":
        manifest = run_task_d(args.output_dir)
    elif args.task == "taskE":
        manifest = run_task_e(args.output_dir)
    else:
        raise ValueError(args.task)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
