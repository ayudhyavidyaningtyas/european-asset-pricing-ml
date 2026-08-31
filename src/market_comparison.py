"""Europe-vs-US comparison utilities for fitted asset-pricing ML outputs."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import stats as project_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EUROPE_OUTPUT = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "main_compustat_benchmark"
)
DEFAULT_US_OUTPUT = PROJECT_ROOT / "results" / "asset_pricing_ml" / "us_compustat_benchmark"
DEFAULT_COMPARISON_OUTPUT = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "market_comparison"
)

SUMMARY_KEY_COLUMNS = [
    "model",
    "base_model",
    "target_mode",
    "weighting",
    "universe_variant",
    "portfolio",
    "cost_bps",
]

SUMMARY_METRIC_COLUMNS = [
    "months",
    "observations",
    "mean_monthly_spearman_ic",
    "mean_monthly_target_spearman_ic",
    "target_r2_zero_monthly_mean",
    "annualized_net_mean_return",
    "annualized_net_excess_return",
    "annualized_net_volatility",
    "net_sharpe",
    "annualized_gross_mean_return",
    "gross_sharpe",
    "average_monthly_turnover",
    "max_drawdown",
]

PRIMARY_FILTER = {
    "weighting": "value",
    "universe_variant": "standard_ex_bottom_5pct",
    "portfolio": "long_short",
    "cost_bps": 25,
}


@dataclass(frozen=True)
class MarketOutputs:
    market: str
    output_dir: Path
    summary: pd.DataFrame
    metrics: pd.DataFrame
    monthly: pd.DataFrame


def _slug(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return slug or "market"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required ML output: {path}")
    return pd.read_csv(path, low_memory=False)


def _attach_market(frame: pd.DataFrame, market: str, output_dir: Path) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "market", market)
    out.insert(1, "source_dir", str(output_dir))
    return out


def load_market_outputs(market: str, output_dir: Path) -> MarketOutputs:
    output_dir = Path(output_dir)
    summary = _attach_market(_read_csv(output_dir / "model_summary.csv"), market, output_dir)
    metrics = _attach_market(_read_csv(output_dir / "prediction_metrics.csv"), market, output_dir)
    monthly = _attach_market(_read_csv(output_dir / "monthly_portfolios.csv"), market, output_dir)
    for column in ["signal_date", "return_date"]:
        if column in monthly:
            monthly[column] = pd.to_datetime(monthly[column], errors="coerce")
    return MarketOutputs(
        market=market,
        output_dir=output_dir,
        summary=summary,
        metrics=metrics,
        monthly=monthly,
    )


def _apply_filters(frame: pd.DataFrame, filters: Mapping[str, object] | None) -> pd.DataFrame:
    out = frame.copy()
    for column, value in (filters or {}).items():
        if value is None or column not in out:
            continue
        if pd.api.types.is_numeric_dtype(out[column]):
            out = out[pd.to_numeric(out[column], errors="coerce").eq(value)]
        else:
            out = out[out[column].astype(str).eq(str(value))]
    return out


def side_by_side_model_summary(
    combined_summary: pd.DataFrame,
    *,
    baseline_market: str = "Europe",
    comparison_market: str = "US",
    filters: Mapping[str, object] | None = None,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot market model summaries and compute comparison-minus-baseline deltas."""
    metric_columns = metric_columns or SUMMARY_METRIC_COLUMNS
    filtered = combined_summary[
        combined_summary["market"].isin([baseline_market, comparison_market])
    ].copy()
    filtered = _apply_filters(filtered, filters or PRIMARY_FILTER)
    available_metrics = [column for column in metric_columns if column in filtered]
    index_columns = [column for column in SUMMARY_KEY_COLUMNS if column in filtered]
    if filtered.empty or not available_metrics:
        return pd.DataFrame(columns=index_columns)

    wide = filtered.pivot_table(
        index=index_columns,
        columns="market",
        values=available_metrics,
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{_slug(market)}" for metric, market in wide.columns]
    wide = wide.reset_index()

    baseline_slug = _slug(baseline_market)
    comparison_slug = _slug(comparison_market)
    for metric in available_metrics:
        baseline_column = f"{metric}_{baseline_slug}"
        comparison_column = f"{metric}_{comparison_slug}"
        if baseline_column in wide and comparison_column in wide:
            wide[f"{metric}_{comparison_slug}_minus_{baseline_slug}"] = (
                pd.to_numeric(wide[comparison_column], errors="coerce")
                - pd.to_numeric(wide[baseline_column], errors="coerce")
            )
    return wide.sort_values(index_columns).reset_index(drop=True)


def _monthly_net_returns(
    monthly: pd.DataFrame,
    *,
    portfolio: str,
    cost_bps: int,
) -> pd.DataFrame:
    required = {"market", "model", "target_mode", "weighting", "universe_variant", "return_date"}
    missing = required - set(monthly)
    if missing:
        raise ValueError(f"Monthly portfolio output missing columns: {sorted(missing)}")

    out = monthly.copy()
    if portfolio == "long_short":
        return_column = "gross_long_short_return"
        turnover_column = "long_short_turnover"
    elif portfolio == "long_only_top_decile":
        return_column = "long_return"
        turnover_column = "long_only_turnover"
    else:
        raise ValueError(f"Unknown portfolio: {portfolio}")
    if return_column not in out or turnover_column not in out:
        raise ValueError(
            f"Monthly portfolio output missing {return_column!r} or {turnover_column!r}"
        )
    out["portfolio"] = portfolio
    out["cost_bps"] = cost_bps
    out["net_return"] = (
        pd.to_numeric(out[return_column], errors="coerce")
        - pd.to_numeric(out[turnover_column], errors="coerce") * cost_bps / 10_000.0
    )
    return out[
        [
            "market",
            "model",
            "target_mode",
            "weighting",
            "universe_variant",
            "portfolio",
            "cost_bps",
            "return_date",
            "net_return",
        ]
    ]


def monthly_return_correlations(
    combined_monthly: pd.DataFrame,
    *,
    baseline_market: str = "Europe",
    comparison_market: str = "US",
    filters: Mapping[str, object] | None = None,
    portfolio: str = "long_short",
    cost_bps: int = 25,
) -> pd.DataFrame:
    """Compare market portfolio returns over common calendar months."""
    returns = _monthly_net_returns(
        combined_monthly,
        portfolio=portfolio,
        cost_bps=cost_bps,
    )
    filters = {
        "weighting": (filters or PRIMARY_FILTER).get("weighting"),
        "universe_variant": (filters or PRIMARY_FILTER).get("universe_variant"),
        "portfolio": portfolio,
        "cost_bps": cost_bps,
    }
    returns = _apply_filters(returns, filters)
    returns = returns[returns["market"].isin([baseline_market, comparison_market])]
    if returns.empty:
        return pd.DataFrame()

    records = []
    key_columns = [
        "model",
        "target_mode",
        "weighting",
        "universe_variant",
        "portfolio",
        "cost_bps",
    ]
    for keys, group in returns.groupby(key_columns, sort=True):
        wide = group.pivot_table(
            index="return_date",
            columns="market",
            values="net_return",
            aggfunc="first",
        )
        if baseline_market not in wide or comparison_market not in wide:
            continue
        common = wide[[baseline_market, comparison_market]].dropna()
        if common.empty:
            continue
        baseline_values = common[baseline_market]
        comparison_values = common[comparison_market]
        row = dict(zip(key_columns, keys, strict=True))
        row.update(
            {
                "common_months": int(len(common)),
                "first_common_month": str(common.index.min().date()),
                "last_common_month": str(common.index.max().date()),
                "return_correlation": float(baseline_values.corr(comparison_values)),
                f"annualized_return_{_slug(baseline_market)}": float(
                    baseline_values.mean() * 12.0
                ),
                f"annualized_return_{_slug(comparison_market)}": float(
                    comparison_values.mean() * 12.0
                ),
                f"annualized_return_{_slug(comparison_market)}_minus_{_slug(baseline_market)}": float(
                    (comparison_values.mean() - baseline_values.mean()) * 12.0
                ),
                f"volatility_{_slug(baseline_market)}": float(
                    baseline_values.std(ddof=1) * np.sqrt(12.0)
                ),
                f"volatility_{_slug(comparison_market)}": float(
                    comparison_values.std(ddof=1) * np.sqrt(12.0)
                ),
            }
        )
        records.append(row)
    return pd.DataFrame(records)


def sharpe_difference_tests(
    combined_monthly: pd.DataFrame,
    *,
    baseline_market: str = "Europe",
    comparison_market: str = "US",
    filters: Mapping[str, object] | None = None,
    portfolio: str = "long_short",
    cost_bps: int = 25,
    expected_block: float = 6.0,
    n_boot: int = 10_000,
    seed: int = 0,
    min_months: int = 24,
) -> pd.DataFrame:
    """Paired market Sharpe-difference tests over common return months.

    The reported difference is comparison minus baseline, using the same net
    monthly portfolio returns as the common-month correlation table.
    """
    returns = _monthly_net_returns(
        combined_monthly,
        portfolio=portfolio,
        cost_bps=cost_bps,
    )
    filters = {
        "weighting": (filters or PRIMARY_FILTER).get("weighting"),
        "universe_variant": (filters or PRIMARY_FILTER).get("universe_variant"),
        "portfolio": portfolio,
        "cost_bps": cost_bps,
    }
    returns = _apply_filters(returns, filters)
    returns = returns[returns["market"].isin([baseline_market, comparison_market])]
    if returns.empty:
        return pd.DataFrame()

    records = []
    key_columns = [
        "model",
        "target_mode",
        "weighting",
        "universe_variant",
        "portfolio",
        "cost_bps",
    ]
    baseline_slug = _slug(baseline_market)
    comparison_slug = _slug(comparison_market)
    for keys, group in returns.groupby(key_columns, sort=True):
        wide = group.pivot_table(
            index="return_date",
            columns="market",
            values="net_return",
            aggfunc="first",
        )
        if baseline_market not in wide or comparison_market not in wide:
            continue
        common = wide[[baseline_market, comparison_market]].dropna()
        if len(common) < min_months:
            continue

        baseline_values = common[baseline_market].to_numpy(dtype=float)
        comparison_values = common[comparison_market].to_numpy(dtype=float)
        risk_free = np.zeros(len(common), dtype=float)
        bootstrap = project_stats.bootstrap_sharpe_diff(
            comparison_values,
            baseline_values,
            risk_free,
            expected_block=expected_block,
            n_boot=n_boot,
            seed=seed,
        )
        memmel = project_stats.jobson_korkie_memmel(
            comparison_values,
            baseline_values,
            risk_free,
        )
        row = dict(zip(key_columns, keys, strict=True))
        row.update(
            {
                "common_months": int(len(common)),
                "first_common_month": str(common.index.min().date()),
                "last_common_month": str(common.index.max().date()),
                f"net_sharpe_{comparison_slug}_minus_{baseline_slug}_bootstrap": bootstrap[
                    "delta_sharpe"
                ],
                "net_sharpe_diff_bootstrap_ci_low": bootstrap["ci_low"],
                "net_sharpe_diff_bootstrap_ci_high": bootstrap["ci_high"],
                "net_sharpe_diff_bootstrap_p_two_sided": bootstrap["p_two_sided"],
                "net_sharpe_diff_bootstrap_expected_block": bootstrap[
                    "expected_block"
                ],
                "net_sharpe_diff_bootstrap_n": bootstrap["n_boot"],
                "net_sharpe_diff_bootstrap_ci_includes_zero": bool(
                    bootstrap["ci_low"] <= 0.0 <= bootstrap["ci_high"]
                ),
                "net_sharpe_diff_jkm_delta_monthly": memmel[
                    "delta_sharpe_monthly"
                ],
                "net_sharpe_diff_jkm_delta_annualized": float(
                    memmel["delta_sharpe_monthly"] * np.sqrt(12.0)
                ),
                "net_sharpe_diff_jkm_z": memmel["z"],
                "net_sharpe_diff_jkm_p_two_sided": memmel["p_two_sided"],
            }
        )
        records.append(row)

    result = pd.DataFrame(records)
    if result.empty:
        return result
    valid = result["net_sharpe_diff_bootstrap_p_two_sided"].notna()
    result["net_sharpe_diff_bootstrap_p_two_sided_holm"] = np.nan
    if valid.any():
        result.loc[valid, "net_sharpe_diff_bootstrap_p_two_sided_holm"] = (
            multipletests(
                result.loc[valid, "net_sharpe_diff_bootstrap_p_two_sided"],
                method="holm",
            )[1]
        )
    return result


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "_No matching rows._"
    available = [column for column in columns if column in frame]
    subset = frame[available].head(max_rows)
    header = "| " + " | ".join(available) + " |"
    divider = "| " + " | ".join("---" for _ in available) + " |"
    rows = [
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in subset.to_numpy()
    ]
    return "\n".join([header, divider, *rows])


def build_market_comparison_report(
    side_by_side: pd.DataFrame,
    correlations: pd.DataFrame,
    *,
    baseline_market: str,
    comparison_market: str,
    filters: Mapping[str, object],
) -> str:
    comparison_slug = _slug(comparison_market)
    baseline_slug = _slug(baseline_market)
    lines = [
        "# Europe vs US Market Comparison",
        "",
        f"Baseline market: {baseline_market}. Comparison market: {comparison_market}.",
        "Primary filter: "
        + ", ".join(f"{key}={value}" for key, value in filters.items()),
        "",
        "## Side-by-side model summary",
        "",
        _markdown_table(
            side_by_side,
            [
                "model",
                "target_mode",
                f"mean_monthly_spearman_ic_{baseline_slug}",
                f"mean_monthly_spearman_ic_{comparison_slug}",
                f"mean_monthly_spearman_ic_{comparison_slug}_minus_{baseline_slug}",
                f"annualized_net_mean_return_{baseline_slug}",
                f"annualized_net_mean_return_{comparison_slug}",
                f"annualized_net_mean_return_{comparison_slug}_minus_{baseline_slug}",
                f"net_sharpe_{baseline_slug}",
                f"net_sharpe_{comparison_slug}",
                f"net_sharpe_{comparison_slug}_minus_{baseline_slug}",
                "net_sharpe_diff_bootstrap_ci_low",
                "net_sharpe_diff_bootstrap_ci_high",
                "net_sharpe_diff_bootstrap_p_two_sided",
            ],
        ),
        "",
        "## Common-month return correlations",
        "",
        _markdown_table(
            correlations,
            [
                "model",
                "target_mode",
                "common_months",
                "return_correlation",
                f"annualized_return_{baseline_slug}",
                f"annualized_return_{comparison_slug}",
                f"annualized_return_{comparison_slug}_minus_{baseline_slug}",
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def write_market_comparison_outputs(
    market_dirs: Mapping[str, Path],
    output_dir: Path = DEFAULT_COMPARISON_OUTPUT,
    *,
    baseline_market: str = "Europe",
    comparison_market: str = "US",
    filters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    filters = dict(filters or PRIMARY_FILTER)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        load_market_outputs(market, Path(path))
        for market, path in market_dirs.items()
    ]
    combined_summary = pd.concat([item.summary for item in outputs], ignore_index=True)
    combined_metrics = pd.concat([item.metrics for item in outputs], ignore_index=True)
    combined_monthly = pd.concat([item.monthly for item in outputs], ignore_index=True)

    side_by_side = side_by_side_model_summary(
        combined_summary,
        baseline_market=baseline_market,
        comparison_market=comparison_market,
        filters=filters,
    )
    correlations = monthly_return_correlations(
        combined_monthly,
        baseline_market=baseline_market,
        comparison_market=comparison_market,
        filters=filters,
        portfolio=str(filters.get("portfolio", "long_short")),
        cost_bps=int(filters.get("cost_bps", 25)),
    )
    sharpe_tests = sharpe_difference_tests(
        combined_monthly,
        baseline_market=baseline_market,
        comparison_market=comparison_market,
        filters=filters,
        portfolio=str(filters.get("portfolio", "long_short")),
        cost_bps=int(filters.get("cost_bps", 25)),
    )
    if not sharpe_tests.empty and not side_by_side.empty:
        merge_columns = [
            column
            for column in SUMMARY_KEY_COLUMNS
            if column in side_by_side and column in sharpe_tests
        ]
        side_by_side = side_by_side.merge(
            sharpe_tests,
            on=merge_columns,
            how="left",
            validate="one_to_one",
        )

    combined_summary.to_csv(output_dir / "combined_model_summary.csv", index=False)
    combined_metrics.to_csv(output_dir / "combined_prediction_metrics.csv", index=False)
    combined_monthly.to_csv(output_dir / "combined_monthly_portfolios.csv", index=False)
    side_by_side.to_csv(output_dir / "side_by_side_model_summary.csv", index=False)
    correlations.to_csv(output_dir / "monthly_return_correlations.csv", index=False)
    sharpe_tests.to_csv(output_dir / "sharpe_difference_tests.csv", index=False)
    report = build_market_comparison_report(
        side_by_side,
        correlations,
        baseline_market=baseline_market,
        comparison_market=comparison_market,
        filters=filters,
    )
    (output_dir / "market_comparison_report.md").write_text(report)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "markets": {item.market: str(item.output_dir) for item in outputs},
        "baseline_market": baseline_market,
        "comparison_market": comparison_market,
        "filters": filters,
        "rows": {
            "combined_model_summary": int(len(combined_summary)),
            "combined_prediction_metrics": int(len(combined_metrics)),
            "combined_monthly_portfolios": int(len(combined_monthly)),
            "side_by_side_model_summary": int(len(side_by_side)),
            "monthly_return_correlations": int(len(correlations)),
            "sharpe_difference_tests": int(len(sharpe_tests)),
        },
        "outputs": {
            "combined_model_summary": str(output_dir / "combined_model_summary.csv"),
            "combined_prediction_metrics": str(output_dir / "combined_prediction_metrics.csv"),
            "combined_monthly_portfolios": str(output_dir / "combined_monthly_portfolios.csv"),
            "side_by_side_model_summary": str(output_dir / "side_by_side_model_summary.csv"),
            "monthly_return_correlations": str(output_dir / "monthly_return_correlations.csv"),
            "sharpe_difference_tests": str(output_dir / "sharpe_difference_tests.csv"),
            "report": str(output_dir / "market_comparison_report.md"),
        },
    }
    (output_dir / "market_comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest
