"""Post-estimation diagnostics for the full AIPM European equity adaptation."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

from aipm_full_transformer_sdf import (
    AIPMFullMonth,
    AIPMFullTransformerConfig,
    _model_comparisons,
    _refit_jobs,
    _split_training_months,
    add_weight_turnover,
    build_months,
    evaluate_model_month,
    load_aipm_full_panel,
    pricing_error_summary,
    summarize_full_aipm,
)
from implementable_frontier import (
    FrontierConfig,
    attach_execution_inputs,
    execution_cost,
    load_monthly_liquidity,
)


AIPM_MODEL_BASELINES = ("bsv", "own_asset_mlp")
DEFAULT_AUM_EUR = (10_000_000.0, 100_000_000.0, 500_000_000.0)
ATTENTION_CATEGORICAL_COLUMNS = (
    "TR.EXCHANGECOUNTRY",
    "screen_country",
    "TR.TRBCECONOMICSECTOR",
    "TR.TRBCBUSINESSSECTOR",
    "TR.TRBCINDUSTRYGROUP",
    "TR.TRBCINDUSTRY",
    "comp_exchange_code",
)
ATTENTION_DISTANCE_COLUMNS = (
    "market_cap_percentile",
    "log_size_rank",
    "book_to_market_rank",
    "momentum_12_2_rank",
    "volatility_12m_rank",
    "return_1m_rank",
    "turnover_12m_rank",
    "operating_profitability_rank",
)


@dataclass(frozen=True)
class AIPMPostAnalysisConfig:
    """Settings shared by AIPM post-estimation tests."""

    aum_eur: tuple[float, ...] = DEFAULT_AUM_EUR
    fallback_half_spread_bps: float = 25.0
    impact_coefficient: float = 0.10
    null_attention_draws: int = 20
    random_state: int = 42
    hac_lags: int = 6
    # Half-spread for names the Refinitiv liquidity export never covered.
    # "constant" applies fallback_half_spread_bps to every uncovered name and is
    # the historical default. "size_conditional" instead fits
    # log(half_spread) ~ log(market_cap) each month on the covered names and
    # extrapolates down the size distribution, which matters because uncovered
    # names are far smaller than covered ones and spreads widen as size falls.
    # Both are assumptions: the export contains no quotes for these securities,
    # so the cost is imputed either way and should be reported as a band.
    spread_imputation: str = "constant"
    imputed_spread_floor_bps: float = 1.0
    imputed_spread_cap_bps: float = 500.0
    min_spread_regression_obs: int = 30
    # ADV is itself imputed (market cap x turnover) because the liquidity export
    # carries no volume column. Names with no turnover data land on this floor,
    # which drives the square-root impact term for the microcap tail.
    adv_fallback_eur: float = 10_000.0


@dataclass(frozen=True)
class PrincipalPortfolioConfig:
    """Causal characteristic-space principal-portfolio benchmark settings."""

    first_test_year: int = 2015
    last_test_year: int = 2026
    min_monthly_stocks: int = 100
    min_training_months: int = 60
    validation_months: int = 12
    training_window_months: int | None = 72
    refit_frequency: str = "annual"
    minimum_size_percentile: float = 0.05
    max_monthly_stocks: int | None = 500
    gross_leverage: float = 1.0
    training_return_clip: float = 1.0
    components: tuple[int, ...] = (1, 3, 5)
    ridge: float = 1e-6
    random_state: int = 42
    hac_lags: int = 6
    pricing_error_ridge: float = 1e-4


@dataclass(frozen=True)
class PrincipalPortfolioFit:
    refit_id: str
    train_start: pd.Timestamp
    train_signal_end: pd.Timestamp
    train_target_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    validation_target_end: pd.Timestamp
    train_label_cutoff: pd.Timestamp
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    prediction_matrix: np.ndarray
    training_loss_by_component: dict[int, float]
    validation_loss_by_component: dict[int, float]


def _month_end(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    return dates.dt.to_period("M").dt.to_timestamp("M")


def _aum_label(aum: float) -> str:
    return f"{int(round(aum / 1_000_000.0))}m"


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema_arrow.names)
    except Exception:
        return list(pd.read_parquet(path).columns)


def _available_columns(path: Path, requested: Sequence[str]) -> list[str]:
    available = set(_parquet_columns(path))
    return [column for column in requested if column in available]


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def load_aipm_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "aipm_full_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def collect_scaling_ablation(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """Collect model, scaling and ablation metrics across available AIPM runs."""

    records: list[dict[str, Any]] = []
    for run_dir in sorted({Path(path) for path in run_dirs}):
        summary_path = run_dir / "aipm_full_summary.csv"
        fit_path = run_dir / "aipm_full_fit_log.csv"
        comparison_path = run_dir / "aipm_full_comparisons.csv"
        if not summary_path.exists():
            continue
        manifest = load_aipm_manifest(run_dir)
        config = manifest.get("config", {})
        rows = manifest.get("rows", {})
        summary = pd.read_csv(summary_path)
        fit = pd.read_csv(fit_path) if fit_path.exists() else pd.DataFrame()
        comparisons = (
            pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
        )
        fit_seconds = (
            fit.groupby("model", as_index=True)["fit_seconds"].sum()
            if not fit.empty and "fit_seconds" in fit
            else pd.Series(dtype=float)
        )
        refits = (
            fit.groupby("model", as_index=True)["refit_id"].nunique()
            if not fit.empty and "refit_id" in fit
            else pd.Series(dtype=float)
        )
        comparison_lookup: dict[tuple[str, str], pd.Series] = {}
        if not comparisons.empty:
            comparison_lookup = {
                (row["model"], row["baseline"]): row
                for _, row in comparisons.iterrows()
            }

        for _, row in summary.iterrows():
            model = str(row["model"])
            record = {
                "run": run_dir.name,
                "run_dir": str(run_dir),
                "model": model,
                "feature_set": manifest.get("feature_set"),
                "feature_count": len(manifest.get("feature_columns", [])),
                "first_test_year": config.get("first_test_year"),
                "last_test_year": config.get("last_test_year"),
                "max_monthly_stocks": config.get("max_monthly_stocks"),
                "minimum_size_percentile": config.get("minimum_size_percentile"),
                "training_window_months": config.get("training_window_months"),
                "validation_months": config.get("validation_months"),
                "refit_frequency": config.get("refit_frequency"),
                "random_feature_count": config.get("random_feature_count"),
                "transformer_blocks": config.get("transformer_blocks"),
                "attention_heads": config.get("attention_heads"),
                "feedforward_width": config.get("feedforward_width"),
                "epochs": config.get("epochs"),
                "patience": config.get("patience"),
                "seeds": ",".join(str(seed) for seed in config.get("seeds", [])),
                "n_seeds": len(config.get("seeds", [])),
                "device": config.get("device"),
                "stored_weight_rows": rows.get("weights"),
                "stored_attention_rows": rows.get("attention_examples"),
                "fit_seconds_total": float(fit_seconds.get(model, np.nan)),
                "refits": int(refits.get(model, 0)) if not refits.empty else np.nan,
                **{
                    column: row[column]
                    for column in summary.columns
                    if column != "model"
                },
            }
            for baseline in AIPM_MODEL_BASELINES:
                comp = comparison_lookup.get((model, baseline))
                if comp is None:
                    continue
                prefix = f"vs_{baseline}"
                for column in [
                    "annualized_mean_difference",
                    "difference_sharpe",
                    "correlation",
                    "alpha_annualized",
                    "alpha_hac_t",
                    "alpha_hac_p",
                    "beta",
                ]:
                    if column in comp:
                        record[f"{prefix}_{column}"] = comp[column]
            records.append(record)
    return pd.DataFrame.from_records(records)


def load_attention_metadata(panel_path: Path) -> pd.DataFrame:
    requested = [
        "date",
        "ric",
        *ATTENTION_CATEGORICAL_COLUMNS,
        *ATTENTION_DISTANCE_COLUMNS,
    ]
    columns = _available_columns(panel_path, requested)
    if "date" not in columns or "ric" not in columns:
        raise ValueError(f"{panel_path} does not contain date and ric columns")
    frame = pd.read_parquet(panel_path, columns=columns)
    frame["signal_date"] = _month_end(frame["date"])
    frame = frame.drop(columns=["date"])
    frame = frame.drop_duplicates(["signal_date", "ric"], keep="last")
    return frame


def _prefix_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = {
        column: f"{prefix}{column}"
        for column in frame.columns
        if column not in {"signal_date", "ric"}
    }
    return frame.rename(columns=renamed)


def _add_pair_features(pairs: pd.DataFrame) -> pd.DataFrame:
    result = pairs.copy()
    for column in ATTENTION_CATEGORICAL_COLUMNS:
        left = f"source_{column}"
        right = f"attended_{column}"
        if left not in result or right not in result:
            continue
        valid = result[left].notna() & result[right].notna()
        safe_name = (
            column.replace("TR.", "")
            .replace(".", "_")
            .replace(" ", "_")
            .lower()
        )
        result[f"same_{safe_name}"] = (result[left] == result[right]).where(valid)
    for column in ATTENTION_DISTANCE_COLUMNS:
        left = f"source_{column}"
        right = f"attended_{column}"
        if left not in result or right not in result:
            continue
        result[f"abs_diff_{column}"] = (
            pd.to_numeric(result[left], errors="coerce")
            - pd.to_numeric(result[right], errors="coerce")
        ).abs()
    return result


def build_attention_pair_diagnostics(
    attention: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    null_draws: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Join top attention links to issuer metadata and add same/distance tests."""

    if attention.empty:
        return pd.DataFrame()
    source = _prefix_columns(metadata, "source_").rename(
        columns={"ric": "source_ric"}
    )
    attended = _prefix_columns(metadata, "attended_").rename(
        columns={"ric": "attended_ric"}
    )
    base = attention.copy()
    base["signal_date"] = _month_end(base["signal_date"])
    observed = base.merge(
        source,
        on=["signal_date", "source_ric"],
        how="left",
        validate="many_to_one",
    ).merge(
        attended,
        on=["signal_date", "attended_ric"],
        how="left",
        validate="many_to_one",
    )
    observed["sample"] = "observed"
    frames = [_add_pair_features(observed)]

    rng = np.random.default_rng(random_state)
    universe_by_month = {
        date: group["ric"].to_numpy()
        for date, group in metadata.groupby("signal_date", sort=False)
    }
    for draw in range(null_draws):
        null = base[["signal_date", "seed", "source_ric", "attention_weight"]].copy()
        sampled = pd.Series(index=null.index, dtype=object)
        for signal_date, rows in null.groupby("signal_date", sort=False):
            universe = universe_by_month.get(signal_date)
            if universe is None or len(universe) == 0:
                sampled.loc[rows.index] = np.nan
                continue
            sampled.loc[rows.index] = rng.choice(
                universe,
                size=len(rows),
                replace=True,
            )
        null["attended_ric"] = sampled.to_numpy()
        null = null.merge(
            source,
            on=["signal_date", "source_ric"],
            how="left",
            validate="many_to_one",
        ).merge(
            attended,
            on=["signal_date", "attended_ric"],
            how="left",
            validate="many_to_one",
        )
        null["sample"] = "null"
        null["null_draw"] = draw
        frames.append(_add_pair_features(null))
    return pd.concat(frames, ignore_index=True, sort=False)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    if not valid.any():
        return np.nan
    safe_weights = pd.to_numeric(weights, errors="coerce").where(valid).fillna(0.0)
    total_weight = float(safe_weights.sum())
    if total_weight <= 0:
        return float(numeric[valid].mean())
    return float(np.average(numeric[valid], weights=safe_weights[valid]))


def summarize_attention_diagnostics(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    metric_columns = [
        column
        for column in pairs.columns
        if column.startswith("same_") or column.startswith("abs_diff_")
    ]
    records: list[dict[str, Any]] = []
    for (sample, metric), group in pairs.groupby(["sample", "signal_date"], sort=True):
        record: dict[str, Any] = {
            "sample": sample,
            "signal_date": metric,
            "pairs": int(len(group)),
            "mean_attention_weight": float(group["attention_weight"].mean()),
        }
        for column in metric_columns:
            record[f"{column}_mean"] = _safe_float(
                pd.to_numeric(group[column], errors="coerce").mean()
            )
            record[f"{column}_weighted_mean"] = _weighted_mean(
                group[column], group["attention_weight"]
            )
        records.append(record)
    monthly = pd.DataFrame.from_records(records)

    overall_records: list[dict[str, Any]] = []
    for sample, group in pairs.groupby("sample", sort=True):
        record = {
            "sample": sample,
            "signal_date": "overall",
            "pairs": int(len(group)),
            "mean_attention_weight": float(group["attention_weight"].mean()),
        }
        for column in metric_columns:
            record[f"{column}_mean"] = _safe_float(
                pd.to_numeric(group[column], errors="coerce").mean()
            )
            record[f"{column}_weighted_mean"] = _weighted_mean(
                group[column], group["attention_weight"]
            )
        overall_records.append(record)
    return pd.concat([monthly, pd.DataFrame.from_records(overall_records)], ignore_index=True)


def attention_lift_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metrics = [
        column
        for column in summary.columns
        if column.endswith("_mean") and column not in {"mean_attention_weight"}
    ]
    observed = summary[summary["sample"].eq("observed")]
    null = summary[summary["sample"].eq("null")]
    merged = observed.merge(
        null,
        on="signal_date",
        suffixes=("_observed", "_null"),
        how="inner",
    )
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        for metric in metrics:
            obs = _safe_float(row.get(f"{metric}_observed"))
            nul = _safe_float(row.get(f"{metric}_null"))
            records.append(
                {
                    "signal_date": row["signal_date"],
                    "metric": metric,
                    "observed": obs,
                    "null": nul,
                    "lift": obs - nul if np.isfinite(obs) and np.isfinite(nul) else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def build_execution_input_panel(
    panel_path: Path,
    risk_path: Path | None,
    liquidity_path: Path | None,
    config: AIPMPostAnalysisConfig,
) -> pd.DataFrame:
    requested = [
        "date",
        "ric",
        "company_market_cap",
        "turnover_12m",
        "volatility_12m",
        "target_return_1m",
    ]
    columns = _available_columns(panel_path, requested)
    panel = pd.read_parquet(panel_path, columns=columns)
    panel["date"] = _month_end(panel["date"])
    if risk_path is not None and risk_path.exists():
        risk = pd.read_parquet(
            risk_path,
            columns=_available_columns(
                risk_path,
                ["date", "ric", "idio_vol_36m", "beta_36m"],
            ),
        )
        if {"date", "ric"}.issubset(risk.columns):
            risk["date"] = _month_end(risk["date"])
            panel = panel.merge(risk, on=["date", "ric"], how="left")
    if "idio_vol_36m" not in panel:
        panel["idio_vol_36m"] = np.nan
    if "volatility_12m" in panel:
        panel["idio_vol_36m"] = panel["idio_vol_36m"].fillna(panel["volatility_12m"])
    panel["idio_vol_36m"] = pd.to_numeric(
        panel["idio_vol_36m"], errors="coerce"
    ).fillna(0.20)
    if "beta_36m" not in panel:
        panel["beta_36m"] = 0.0
    panel["beta_36m"] = pd.to_numeric(panel["beta_36m"], errors="coerce").fillna(0.0)
    if "turnover_12m" not in panel:
        panel["turnover_12m"] = np.nan
    if "company_market_cap" not in panel:
        panel["company_market_cap"] = np.nan
    panel["company_market_cap"] = pd.to_numeric(
        panel["company_market_cap"], errors="coerce"
    )
    panel["turnover_12m"] = pd.to_numeric(panel["turnover_12m"], errors="coerce")

    frontier_config = FrontierConfig(
        fallback_half_spread_bps=config.fallback_half_spread_bps,
        impact_coefficient=config.impact_coefficient,
        aum_eur=config.aum_eur,
    )
    liquidity = load_monthly_liquidity(
        liquidity_path if liquidity_path is not None and liquidity_path.exists() else None
    )
    return attach_execution_inputs(panel, liquidity, frontier_config).drop_duplicates(
        ["date", "ric"], keep="last"
    )


def impute_half_spreads(
    frame: pd.DataFrame,
    config: AIPMPostAnalysisConfig,
) -> pd.DataFrame:
    """Fill half-spreads for uncovered names by extrapolating on size.

    Each month, log(half_spread_bps) is regressed on log(market_cap) across the
    names that do have observed quotes, and the fit is used to price the names
    that do not. Months without enough covered names, or without usable market
    caps, fall back to the constant half-spread so the function always returns a
    fully populated column.

    This is an extrapolation beyond the estimation sample -- covered names are
    large caps and the uncovered tail is far smaller -- so the result is a
    stated assumption, not a measurement.
    """

    frame = frame.copy()
    frame["half_spread_imputed_size_conditional"] = False
    if "market_cap" not in frame.columns:
        return frame

    observed_mask = frame["spread_observed"].to_numpy(dtype=bool)
    market_cap = pd.to_numeric(frame["market_cap"], errors="coerce").to_numpy(dtype=float)
    half_spread = frame["half_spread_bps"].to_numpy(dtype=float).copy()
    usable_cap = np.isfinite(market_cap) & (market_cap > 0.0)
    imputed_flag = np.zeros(len(frame), dtype=bool)

    for _, index in frame.groupby("signal_date", sort=False).indices.items():
        index = np.asarray(index)
        fit_rows = index[
            observed_mask[index]
            & usable_cap[index]
            & np.isfinite(half_spread[index])
            & (half_spread[index] > 0.0)
        ]
        target_rows = index[~observed_mask[index] & usable_cap[index]]
        if len(fit_rows) < config.min_spread_regression_obs or len(target_rows) == 0:
            continue
        log_cap = np.log(market_cap[fit_rows])
        log_spread = np.log(half_spread[fit_rows])
        design = np.column_stack([np.ones(len(fit_rows)), log_cap])
        coefficients, *_ = np.linalg.lstsq(design, log_spread, rcond=None)
        predicted = np.exp(
            coefficients[0] + coefficients[1] * np.log(market_cap[target_rows])
        )
        half_spread[target_rows] = np.clip(
            predicted,
            config.imputed_spread_floor_bps,
            config.imputed_spread_cap_bps,
        )
        imputed_flag[target_rows] = True

    frame["half_spread_bps"] = half_spread
    frame["half_spread_imputed_size_conditional"] = imputed_flag
    return frame


def simulate_weight_implementability(
    weights: pd.DataFrame,
    execution_inputs: pd.DataFrame,
    config: AIPMPostAnalysisConfig,
) -> pd.DataFrame:
    """Apply the existing spread plus square-root impact model to SDF weights."""

    if weights.empty:
        return pd.DataFrame()
    frame = weights.copy()
    frame["signal_date"] = _month_end(frame["signal_date"])
    frame["target_date"] = _month_end(frame["target_date"])
    frame["sdf_weight"] = pd.to_numeric(frame["sdf_weight"], errors="coerce").fillna(0.0)
    if "target_return" not in frame:
        raise ValueError("weights must contain target_return")
    frame["target_return"] = pd.to_numeric(frame["target_return"], errors="coerce")
    inputs = execution_inputs.rename(columns={"date": "signal_date"})
    needed = [
        "signal_date",
        "ric",
        "half_spread_bps",
        "spread_observed",
        "adv_eur",
        "idio_vol_36m",
    ]
    frame = frame.merge(inputs[needed], on=["signal_date", "ric"], how="left")
    frame["half_spread_bps"] = pd.to_numeric(
        frame["half_spread_bps"], errors="coerce"
    ).fillna(config.fallback_half_spread_bps)
    frame["spread_observed"] = (
        frame["spread_observed"].astype("boolean").fillna(False).astype(bool)
    )
    if config.spread_imputation == "size_conditional":
        frame = impute_half_spreads(frame, config)
    elif config.spread_imputation != "constant":
        raise ValueError("spread_imputation must be 'constant' or 'size_conditional'")
    frame["adv_floored"] = (
        pd.to_numeric(frame["adv_eur"], errors="coerce").isna()
        | pd.to_numeric(frame["adv_eur"], errors="coerce").le(config.adv_fallback_eur)
    )
    frame["adv_eur"] = (
        pd.to_numeric(frame["adv_eur"], errors="coerce")
        .fillna(config.adv_fallback_eur)
        .clip(lower=config.adv_fallback_eur)
    )
    frame["idio_vol_36m"] = pd.to_numeric(
        frame["idio_vol_36m"], errors="coerce"
    ).fillna(0.20)

    records: list[dict[str, Any]] = []
    for model, model_frame in frame.groupby("model", sort=True):
        previous_weights: dict[str, float] = {}
        previous_inputs: dict[str, tuple[float, float, float]] = {}
        for signal_date, month in model_frame.groupby("signal_date", sort=True):
            month = month.sort_values("ric")
            current_weights = dict(zip(month["ric"], month["sdf_weight"], strict=True))
            current_inputs = {
                row.ric: (
                    float(row.half_spread_bps),
                    float(row.adv_eur),
                    float(row.idio_vol_36m),
                )
                for row in month.itertuples(index=False)
            }
            all_rics = sorted(set(previous_weights).union(current_weights))
            delta = np.array(
                [
                    current_weights.get(ric, 0.0) - previous_weights.get(ric, 0.0)
                    for ric in all_rics
                ],
                dtype=float,
            )
            cost_inputs = [
                current_inputs.get(
                    ric,
                    previous_inputs.get(
                        ric,
                        (
                            config.fallback_half_spread_bps,
                            10_000.0,
                            0.20,
                        ),
                    ),
                )
                for ric in all_rics
            ]
            half_spread = np.array([item[0] for item in cost_inputs], dtype=float)
            adv = np.array([item[1] for item in cost_inputs], dtype=float)
            idio = np.array([item[2] for item in cost_inputs], dtype=float)
            gross_return = float(
                (month["sdf_weight"].to_numpy(dtype=float) * month["target_return"].to_numpy(dtype=float)).sum()
            )
            gross_exposure = float(month["sdf_weight"].abs().sum())
            row: dict[str, Any] = {
                "signal_date": signal_date,
                "target_date": month["target_date"].iloc[0],
                "model": model,
                "assets": int(len(month)),
                "gross_exposure": gross_exposure,
                "net_exposure": float(month["sdf_weight"].sum()),
                "turnover": float(np.abs(delta).sum()),
                "gross_return": gross_return,
                "spread_observed_weight": float(
                    month.loc[month["spread_observed"], "sdf_weight"].abs().sum()
                    / max(gross_exposure, 1e-12)
                ),
                "mean_half_spread_bps": float(
                    np.average(
                        month["half_spread_bps"].to_numpy(dtype=float),
                        weights=month["sdf_weight"].abs().to_numpy(dtype=float),
                    )
                )
                if gross_exposure > 0
                else np.nan,
                # Share of gross weight whose assumed daily volume is the floor
                # rather than a measured quantity. High values mean the impact
                # leg of the cost model is not identified for that weight.
                "adv_floored_weight": float(
                    month.loc[month["adv_floored"], "sdf_weight"].abs().sum()
                    / max(gross_exposure, 1e-12)
                ),
            }
            for aum in config.aum_eur:
                spread_cost, impact_cost, total_cost = execution_cost(
                    delta,
                    half_spread,
                    adv,
                    idio,
                    aum,
                    config.impact_coefficient,
                )
                label = _aum_label(aum)
                record = {
                    **row,
                    "aum_eur": float(aum),
                    "aum_label": label,
                    "spread_cost": spread_cost,
                    "impact_cost": impact_cost,
                    "total_cost": total_cost,
                    "net_return": gross_return - total_cost,
                }
                records.append(record)
            previous_weights = current_weights
            previous_inputs = current_inputs
    return pd.DataFrame.from_records(records)


def summarize_implementability(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (model, aum_label), group in monthly.groupby(["model", "aum_label"], sort=True):
        net = group["net_return"].astype(float)
        gross = group["gross_return"].astype(float)
        annual_vol = float(net.std(ddof=1) * math.sqrt(12.0))
        records.append(
            {
                "model": model,
                "aum_label": aum_label,
                "aum_eur": float(group["aum_eur"].iloc[0]),
                "months": int(len(group)),
                "annualized_gross_return": float(gross.mean() * 12.0),
                "annualized_net_return": float(net.mean() * 12.0),
                "annualized_net_volatility": annual_vol,
                "net_sharpe": float(net.mean() * 12.0 / annual_vol)
                if annual_vol > 0
                else np.nan,
                "monthly_min_net_return": float(net.min()),
                "monthly_max_net_return": float(net.max()),
                "average_monthly_turnover": float(group["turnover"].mean()),
                "average_gross_exposure": float(group["gross_exposure"].mean()),
                "average_net_exposure": float(group["net_exposure"].mean()),
                "annualized_spread_cost": float(group["spread_cost"].mean() * 12.0),
                "annualized_impact_cost": float(group["impact_cost"].mean() * 12.0),
                "annualized_total_cost": float(group["total_cost"].mean() * 12.0),
                "spread_observed_weight": float(group["spread_observed_weight"].mean()),
                "mean_half_spread_bps": float(group["mean_half_spread_bps"].mean()),
                "adv_floored_weight": float(group["adv_floored_weight"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def implementability_comparisons(monthly: pd.DataFrame, hac_lags: int = 6) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for aum_label, aum_group in monthly.groupby("aum_label", sort=True):
        wide = aum_group.pivot(index="signal_date", columns="model", values="net_return")
        baselines = [model for model in AIPM_MODEL_BASELINES if model in wide]
        for model in wide.columns:
            for baseline in baselines:
                if model == baseline:
                    continue
                common = wide[[baseline, model]].dropna()
                if len(common) < 12:
                    continue
                diff = common[model] - common[baseline]
                diff_vol = float(diff.std(ddof=1) * math.sqrt(12.0))
                regression = sm.OLS(
                    common[model].to_numpy(dtype=float),
                    sm.add_constant(common[baseline].to_numpy(dtype=float)),
                ).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
                records.append(
                    {
                        "aum_label": aum_label,
                        "model": model,
                        "baseline": baseline,
                        "months": int(len(common)),
                        "annualized_mean_difference": float(diff.mean() * 12.0),
                        "annualized_difference_volatility": diff_vol,
                        "difference_sharpe": float(diff.mean() * 12.0 / diff_vol)
                        if diff_vol > 0
                        else np.nan,
                        "correlation": float(common[model].corr(common[baseline])),
                        "alpha_annualized": float(regression.params[0] * 12.0),
                        "alpha_hac_t": float(regression.tvalues[0]),
                        "alpha_hac_p": float(regression.pvalues[0]),
                        "beta": float(regression.params[1])
                        if len(regression.params) > 1
                        else np.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def _principal_prediction_matrix(months: Sequence[AIPMFullMonth], ridge: float) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for month in months:
        x = np.asarray(month.features, dtype=float)
        r = np.asarray(month.training_returns, dtype=float)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(r)
        if valid.sum() < 2:
            continue
        x = x[valid]
        r = r[valid]
        x = x - np.nanmean(x, axis=0, keepdims=True)
        matrices.append((x.T @ (x * r[:, None])) / max(len(x), 1))
    if not matrices:
        raise ValueError("No valid training months for principal portfolios")
    matrix = np.nanmean(np.stack(matrices, axis=0), axis=0)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    symmetric = 0.5 * (matrix + matrix.T)
    return symmetric + ridge * np.eye(symmetric.shape[0])


def fit_principal_portfolio(
    refit_id: str,
    cutoff: pd.Timestamp,
    core_months: Sequence[AIPMFullMonth],
    validation_months: Sequence[AIPMFullMonth],
    config: PrincipalPortfolioConfig,
) -> PrincipalPortfolioFit:
    matrix = _principal_prediction_matrix(core_months, config.ridge)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(-np.abs(values))
    values = values[order]
    vectors = vectors[:, order]
    train_loss = {
        component: _principal_loss(core_months, values, vectors, component, config)
        for component in config.components
    }
    validation_loss = {
        component: _principal_loss(validation_months, values, vectors, component, config)
        for component in config.components
    }
    return PrincipalPortfolioFit(
        refit_id=refit_id,
        train_start=core_months[0].signal_date,
        train_signal_end=core_months[-1].signal_date,
        train_target_end=core_months[-1].target_date,
        validation_start=validation_months[0].signal_date,
        validation_end=validation_months[-1].signal_date,
        validation_target_end=validation_months[-1].target_date,
        train_label_cutoff=cutoff,
        eigenvalues=values,
        eigenvectors=vectors,
        prediction_matrix=matrix,
        training_loss_by_component=train_loss,
        validation_loss_by_component=validation_loss,
    )


def _principal_scores(
    month: AIPMFullMonth,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    components: int,
) -> np.ndarray:
    n_components = max(1, min(int(components), eigenvectors.shape[1]))
    direction = eigenvectors[:, :n_components] @ eigenvalues[:n_components]
    scores = np.asarray(month.features, dtype=float) @ direction
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores - float(np.mean(scores))


def _principal_loss(
    months: Sequence[AIPMFullMonth],
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    components: int,
    config: PrincipalPortfolioConfig,
) -> float:
    if not months:
        return np.nan
    losses: list[float] = []
    for month in months:
        scores = _principal_scores(month, eigenvalues, eigenvectors, components)
        gross = float(np.abs(scores).sum())
        if gross <= 1e-12:
            month_return = 0.0
        else:
            weights = scores / gross * config.gross_leverage
            month_return = float(weights @ month.training_returns)
        losses.append(float((1.0 - month_return) ** 2))
    return float(np.mean(losses))


def run_principal_portfolio_walk_forward(
    panel: pd.DataFrame,
    feature_columns: Sequence[str],
    config: PrincipalPortfolioConfig,
    test_assets: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aipm_config = AIPMFullTransformerConfig(
        first_test_year=config.first_test_year,
        last_test_year=config.last_test_year,
        min_monthly_stocks=config.min_monthly_stocks,
        min_training_months=config.min_training_months,
        validation_months=config.validation_months,
        training_window_months=config.training_window_months,
        refit_frequency=config.refit_frequency,
        minimum_size_percentile=config.minimum_size_percentile,
        max_monthly_stocks=config.max_monthly_stocks,
        gross_leverage=config.gross_leverage,
        training_return_clip=config.training_return_clip,
        random_state=config.random_state,
        hac_lags=config.hac_lags,
        pricing_error_ridge=config.pricing_error_ridge,
    )
    months = build_months(panel, list(feature_columns), aipm_config)
    monthly_records: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    weight_frames: list[pd.DataFrame] = []
    for refit_id, cutoff, test_months in _refit_jobs(months, aipm_config):
        core_months, validation_months = _split_training_months(months, cutoff, aipm_config)
        if len(core_months) < config.min_training_months or not validation_months:
            continue
        fitted = fit_principal_portfolio(
            refit_id,
            cutoff,
            core_months,
            validation_months,
            config,
        )
        max_components = min(max(config.components), fitted.eigenvectors.shape[1])
        for component in config.components:
            model = f"principal_portfolio_h{component}"
            fit_records.append(
                {
                    "model": model,
                    "refit_id": refit_id,
                    "train_start": fitted.train_start,
                    "train_signal_end": fitted.train_signal_end,
                    "train_target_end": fitted.train_target_end,
                    "validation_start": fitted.validation_start,
                    "validation_end": fitted.validation_end,
                    "validation_target_end": fitted.validation_target_end,
                    "train_label_cutoff": fitted.train_label_cutoff,
                    "components_requested": int(component),
                    "components_used": int(min(component, max_components)),
                    "n_parameters": int(len(feature_columns) * min(component, max_components)),
                    "top_abs_eigenvalue": float(fitted.eigenvalues[0]),
                    "training_loss": fitted.training_loss_by_component[component],
                    "validation_loss": fitted.validation_loss_by_component[component],
                }
            )
            for month in test_months:
                scores = _principal_scores(
                    month,
                    fitted.eigenvalues,
                    fitted.eigenvectors,
                    component,
                )
                record, weights = evaluate_model_month(
                    month,
                    model,
                    scores,
                    aipm_config,
                    {
                        "ridge_alpha": config.ridge,
                        "validation_loss": fitted.validation_loss_by_component[
                            component
                        ],
                        "best_epoch_mean": np.nan,
                    },
                )
                record["refit_id"] = refit_id
                record["components"] = int(component)
                monthly_records.append(record)
                weight_frames.append(weights)
    monthly = pd.DataFrame.from_records(monthly_records)
    weights = (
        pd.concat(weight_frames, ignore_index=True)
        if weight_frames
        else pd.DataFrame()
    )
    monthly = add_weight_turnover(monthly, weights)
    fit_log = pd.DataFrame.from_records(fit_records)
    pricing_errors = (
        pricing_error_summary(monthly, test_assets, config.pricing_error_ridge)
        if test_assets is not None and not test_assets.empty
        else pd.DataFrame()
    )
    summary = summarize_full_aipm(monthly, pricing_errors)
    comparisons = _model_comparisons(monthly, config.hac_lags)
    return monthly, fit_log, weights, summary, comparisons


def principal_config_from_aipm_manifest(
    manifest: dict[str, Any],
    components: Sequence[int],
) -> PrincipalPortfolioConfig:
    config = manifest.get("config", {})
    return PrincipalPortfolioConfig(
        first_test_year=int(config.get("first_test_year", 2015)),
        last_test_year=int(config.get("last_test_year", 2026)),
        min_monthly_stocks=int(config.get("min_monthly_stocks", 100)),
        min_training_months=int(config.get("min_training_months", 60)),
        validation_months=int(config.get("validation_months", 12)),
        training_window_months=config.get("training_window_months"),
        refit_frequency=str(config.get("refit_frequency", "annual")),
        minimum_size_percentile=float(config.get("minimum_size_percentile", 0.05)),
        max_monthly_stocks=config.get("max_monthly_stocks"),
        gross_leverage=float(config.get("gross_leverage", 1.0)),
        training_return_clip=float(config.get("training_return_clip", 1.0)),
        components=tuple(int(item) for item in components),
        random_state=int(config.get("random_state", 42)),
        hac_lags=int(config.get("hac_lags", 6)),
        pricing_error_ridge=float(config.get("pricing_error_ridge", 1e-4)),
    )


def write_post_analysis_outputs(
    output_dir: Path,
    *,
    run_dir: Path,
    aipm_dirs: Sequence[Path],
    panel_path: Path,
    risk_path: Path | None,
    liquidity_path: Path | None,
    risk_free: pd.Series | None,
    principal_components: Sequence[int] = (1, 3, 5),
    config: AIPMPostAnalysisConfig = AIPMPostAnalysisConfig(),
    run_principal_portfolios: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_aipm_manifest(run_dir)
    feature_columns = manifest.get("feature_columns", [])

    scaling = collect_scaling_ablation(aipm_dirs)
    scaling.to_csv(output_dir / "aipm_scaling_ablation.csv", index=False)

    attention_path = run_dir / "aipm_full_attention_examples.csv"
    metadata = load_attention_metadata(panel_path)
    attention = pd.read_csv(attention_path) if attention_path.exists() else pd.DataFrame()
    attention_pairs = build_attention_pair_diagnostics(
        attention,
        metadata,
        null_draws=config.null_attention_draws,
        random_state=config.random_state,
    )
    if not attention_pairs.empty:
        attention_pairs.to_parquet(output_dir / "aipm_attention_pair_diagnostics.parquet")
    attention_summary = summarize_attention_diagnostics(attention_pairs)
    attention_lift = attention_lift_table(attention_summary)
    attention_summary.to_csv(output_dir / "aipm_attention_summary.csv", index=False)
    attention_lift.to_csv(output_dir / "aipm_attention_lift.csv", index=False)

    execution_inputs = build_execution_input_panel(
        panel_path,
        risk_path,
        liquidity_path,
        config,
    )
    weights = pd.read_parquet(run_dir / "aipm_full_weights.parquet")
    implementability = simulate_weight_implementability(weights, execution_inputs, config)
    implementability_summary = summarize_implementability(implementability)
    implementability_comparison = implementability_comparisons(
        implementability,
        config.hac_lags,
    )
    implementability.to_csv(output_dir / "aipm_implementability_monthly.csv", index=False)
    implementability_summary.to_csv(
        output_dir / "aipm_implementability_summary.csv",
        index=False,
    )
    implementability_comparison.to_csv(
        output_dir / "aipm_implementability_comparisons.csv",
        index=False,
    )

    pp_rows: dict[str, int] = {}
    if run_principal_portfolios:
        if not feature_columns:
            raise ValueError("AIPM manifest does not contain feature columns")
        pp_config = principal_config_from_aipm_manifest(manifest, principal_components)
        panel = load_aipm_full_panel(panel_path, risk_free, list(feature_columns))
        test_assets_path = run_dir / "aipm_full_test_assets.csv"
        test_assets = (
            pd.read_csv(test_assets_path, parse_dates=["signal_date"])
            if test_assets_path.exists()
            else pd.DataFrame()
        )
        pp_monthly, pp_fit_log, pp_weights, pp_summary, pp_comparisons = (
            run_principal_portfolio_walk_forward(
                panel,
                feature_columns,
                pp_config,
                test_assets=test_assets,
            )
        )
        pp_monthly.to_csv(output_dir / "principal_portfolio_monthly.csv", index=False)
        pp_fit_log.to_csv(output_dir / "principal_portfolio_fit_log.csv", index=False)
        pp_weights.to_parquet(output_dir / "principal_portfolio_weights.parquet", index=False)
        pp_summary.to_csv(output_dir / "principal_portfolio_summary.csv", index=False)
        pp_comparisons.to_csv(
            output_dir / "principal_portfolio_comparisons.csv",
            index=False,
        )
        pp_impl = simulate_weight_implementability(pp_weights, execution_inputs, config)
        pp_impl_summary = summarize_implementability(pp_impl)
        pp_impl_comparisons = implementability_comparisons(pp_impl, config.hac_lags)
        pp_impl.to_csv(
            output_dir / "principal_portfolio_implementability_monthly.csv",
            index=False,
        )
        pp_impl_summary.to_csv(
            output_dir / "principal_portfolio_implementability_summary.csv",
            index=False,
        )
        pp_impl_comparisons.to_csv(
            output_dir / "principal_portfolio_implementability_comparisons.csv",
            index=False,
        )
        pp_rows = {
            "principal_portfolio_monthly": int(len(pp_monthly)),
            "principal_portfolio_fit_log": int(len(pp_fit_log)),
            "principal_portfolio_weights": int(len(pp_weights)),
            "principal_portfolio_summary": int(len(pp_summary)),
            "principal_portfolio_implementability": int(len(pp_impl)),
        }

    output_manifest: dict[str, Any] = {
        "run_dir": str(run_dir),
        "aipm_dirs": [str(path) for path in aipm_dirs],
        "panel_path": str(panel_path),
        "risk_path": str(risk_path) if risk_path is not None else None,
        "liquidity_path": str(liquidity_path) if liquidity_path is not None else None,
        "config": asdict(config),
        "principal_components": [int(item) for item in principal_components],
        "principal_portfolio_adaptation": (
            "Characteristic-space principal portfolios estimated from the "
            "training-window return-weighted characteristic prediction matrix."
        ),
        "rows": {
            "scaling_ablation": int(len(scaling)),
            "attention_pairs": int(len(attention_pairs)),
            "attention_summary": int(len(attention_summary)),
            "attention_lift": int(len(attention_lift)),
            "aipm_implementability": int(len(implementability)),
            "aipm_implementability_summary": int(len(implementability_summary)),
            **pp_rows,
        },
    }
    (output_dir / "aipm_post_analysis_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, default=str)
    )
    return output_manifest
