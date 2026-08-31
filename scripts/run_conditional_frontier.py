"""Run the 2008 implementable frontier for conditional LambdaRank signals."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from implementable_frontier import (  # noqa: E402
    FrontierConfig,
    attach_execution_inputs,
    build_market_volatility,
    causal_strategy_selection,
    frontier_dominance,
    load_eur_risk_free,
    load_monthly_liquidity,
    selected_sharpe_inference,
    simulate_frontiers,
    simulate_selected_frontiers,
    summarize_frontiers,
    summarize_selected,
    write_frontier_outputs,
)
import stats as project_stats  # noqa: E402


BASE = PROJECT_ROOT / "results/asset_pricing_ml/conditional_lambdarank_2008"
PANEL = PROJECT_ROOT / "data/processed/asset_pricing/monthly_feature_panel.parquet"
RISK = PROJECT_ROOT / "results/asset_pricing_ml/depth_analysis/rolling_risk_estimates.parquet"
MARKET = PROJECT_ROOT / "results/asset_pricing_ml/depth_analysis/eur_market_return.csv"
EUR_RATE = PROJECT_ROOT / "data/raw/fred_IR3TIB01EZM156N.csv"
LIQUIDITY = (
    PROJECT_ROOT
    / "data/raw/asset_pricing/refinitiv_exports/supplemental/liquidity_monthly_full_period"
)
DELISTING = (
    PROJECT_ROOT
    / "data/processed/asset_pricing/delisting_return_audit.csv"
)
OUTPUT = PROJECT_ROOT / "results/asset_pricing_ml/implementable_frontier_2008"


def conditional_incremental_inference(
    selected: pd.DataFrame,
    repetitions: int = 5_000,
) -> pd.DataFrame:
    records = []
    for (portfolio, aum_label), family in selected.groupby(
        ["portfolio", "aum_label"]
    ):
        conditional = family[family["signal"].eq("conditional_rank")].set_index(
            "return_date"
        )
        unconditional = family[
            family["signal"].eq("unconditional_rank")
        ].set_index("return_date")
        dates = conditional.index.intersection(unconditional.index)
        rf = (
            conditional.loc[dates, "rf_eur"].to_numpy()
            if portfolio == "long_only"
            else np.zeros(len(dates))
        )
        result = project_stats.bootstrap_sharpe_diff(
            conditional.loc[dates, "selected_net_return"],
            unconditional.loc[dates, "selected_net_return"],
            rf,
            expected_block=6,
            n_boot=repetitions,
            seed=42,
        )
        records.append(
            {
                "portfolio": portfolio,
                "aum_label": aum_label,
                "months": len(dates),
                **result,
            }
        )
    output = pd.DataFrame(records)
    output["p_two_sided_holm"] = multipletests(
        output["p_two_sided"], method="holm"
    )[1]
    return output


def load_panel() -> pd.DataFrame:
    combined = pd.read_parquet(BASE / "combined_predictions.parquet")
    signal = combined.pivot_table(
        index=["date", "target_date", "ric"],
        columns="model_variant",
        values="prediction",
        aggfunc="last",
    ).reset_index()
    signal = signal.rename(
        columns={
            "conditional_multihorizon": "conditional_rank",
            "unconditional_multihorizon": "unconditional_rank",
        }
    )
    outcomes = combined.groupby(
        ["date", "target_date", "ric"], as_index=False
    )["target_return_1m"].first()
    signal = signal.merge(
        outcomes, on=["date", "target_date", "ric"], how="left"
    )

    columns = [
        "date",
        "ric",
        "company_market_cap",
        "market_cap_percentile",
        "turnover_12m",
        "volatility_12m",
        "book_to_market_rank",
        "momentum_12_2_rank",
        "operating_profitability_rank",
    ]
    panel = pd.read_parquet(PANEL, columns=columns)
    risk = pd.read_parquet(
        RISK,
        columns=["date", "ric", "beta_36m", "idio_vol_36m", "risk_nobs"],
    )
    result = signal.merge(panel, on=["date", "ric"], how="left", validate="one_to_one")
    result = result.merge(risk, on=["date", "ric"], how="left", validate="one_to_one")
    result["momentum"] = result["momentum_12_2_rank"]
    result["sparse3"] = result[
        [
            "book_to_market_rank",
            "momentum_12_2_rank",
            "operating_profitability_rank",
        ]
    ].mean(axis=1, skipna=False)
    audit = pd.read_csv(DELISTING)
    audit = audit[
        audit["missing_retirement_month_return"].fillna(False)
    ][["ric", "retire_month"]].copy()
    audit["target_date"] = pd.PeriodIndex(
        audit["retire_month"], freq="M"
    ).to_timestamp("M")
    audit["is_delisting_candidate"] = True
    result = result.merge(audit, on=["ric", "target_date"], how="left")
    result["is_delisting_candidate"] = (
        result["is_delisting_candidate"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    result.loc[
        result["is_delisting_candidate"] & result["target_return_1m"].isna(),
        "target_return_1m",
    ] = -1.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--liquidity", type=Path, default=LIQUIDITY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()

    config = FrontierConfig(
        signals=(
            "conditional_rank",
            "unconditional_rank",
            "momentum",
            "sparse3",
        )
    )
    print("loading conditional, unconditional and benchmark signals", flush=True)
    panel = load_panel()
    panel = attach_execution_inputs(
        panel, load_monthly_liquidity(args.liquidity), config
    )
    market_volatility = build_market_volatility(MARKET)
    risk_free = load_eur_risk_free(EUR_RATE)
    print("optimizing full 2008-2026 implementable frontiers", flush=True)
    monthly = simulate_frontiers(panel, market_volatility, risk_free, config)
    summary = summarize_frontiers(monthly, config)
    dominance = frontier_dominance(summary)
    _, selection_log = causal_strategy_selection(monthly, config)
    print("replaying causal choices with continuous holdings", flush=True)
    selected = simulate_selected_frontiers(
        panel, market_volatility, risk_free, selection_log, config
    )
    selected_summary = summarize_selected(selected, config)
    inference = selected_sharpe_inference(selected, config)
    incremental = conditional_incremental_inference(selected)
    manifest = write_frontier_outputs(
        args.output_dir,
        config,
        monthly,
        summary,
        dominance,
        selected,
        selection_log,
        selected_summary,
        inference,
        {
            "panel": PANEL,
            "predictions": BASE / "combined_predictions.parquet",
            "risk": RISK,
            "market": MARKET,
            "eur_rate": EUR_RATE,
            "liquidity": args.liquidity,
        },
    )
    incremental.to_csv(
        args.output_dir / "conditional_incremental_inference.csv", index=False
    )
    subsample_summaries = []
    subsample_inference = []
    for start_year in [2008, 2010, 2015]:
        subset = selected[
            pd.to_datetime(selected["return_date"]).dt.year.ge(start_year)
        ].copy()
        summary_part = summarize_selected(subset, config)
        summary_part.insert(0, "start_year", start_year)
        subsample_summaries.append(summary_part)
        inference_part = selected_sharpe_inference(subset, config)
        inference_part.insert(0, "start_year", start_year)
        subsample_inference.append(inference_part)
    pd.concat(subsample_summaries, ignore_index=True).to_csv(
        args.output_dir / "selected_summary_by_start.csv", index=False
    )
    pd.concat(subsample_inference, ignore_index=True).to_csv(
        args.output_dir / "selected_inference_by_start.csv", index=False
    )
    print(json.dumps(manifest["rows"], indent=2), flush=True)
    print(f"outputs -> {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
