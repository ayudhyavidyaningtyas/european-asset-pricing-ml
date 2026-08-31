"""Run factor-spanning and conditional-depth analysis on frozen OOS predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asset_pricing_depth import (  # noqa: E402
    DepthConfig,
    build_internal_eur_factors,
    build_internal_market,
    build_market_states,
    conditional_predictability,
    estimate_rolling_risk,
    factor_spanning_tests,
    fama_macbeth_tests,
    load_eur_short_rate,
    prepare_analysis_predictions,
    state_dependence_tests,
    write_depth_outputs,
)


DEFAULT_PANEL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asset_pricing"
    / "monthly_feature_panel.parquet"
)
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "results"
    / "asset_pricing_ml"
    / "revised_full_eur_delisting"
)
DEFAULT_EUR_RATE = (
    PROJECT_ROOT / "data" / "raw" / "fred_IR3TIB01EZM156N.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "asset_pricing_ml" / "depth_analysis"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--eur-rate", type=Path, default=DEFAULT_EUR_RATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--beta-window-months", type=int, default=36)
    parser.add_argument("--beta-min-observations", type=int, default=24)
    parser.add_argument("--cost-bps", type=int, default=25)
    parser.add_argument("--hac-lags", type=int, default=6)
    parser.add_argument("--minimum-cross-section", type=int, default=100)
    args = parser.parse_args()

    required = [
        args.panel,
        args.baseline_dir / "predictions.parquet",
        args.baseline_dir / "monthly_portfolios.csv",
        args.eur_rate,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}")

    config = DepthConfig(
        beta_window_months=args.beta_window_months,
        beta_min_observations=args.beta_min_observations,
        portfolio_cost_bps=args.cost_bps,
        hac_lags=args.hac_lags,
        minimum_cross_section=args.minimum_cross_section,
    )
    panel_columns = [
        "date",
        "target_date",
        "ric",
        "return_1m",
        "target_return_1m",
        "company_market_cap",
        "market_cap_percentile",
        "book_to_market",
        "operating_profitability",
        "asset_growth",
        "momentum_12_2",
        "log_size_rank",
        "book_to_market_rank",
        "momentum_12_2_rank",
        "turnover_12m_rank",
    ]
    print("loading panel and frozen predictions", flush=True)
    panel = pd.read_parquet(args.panel, columns=panel_columns)
    predictions = pd.read_parquet(args.baseline_dir / "predictions.parquet")
    monthly_portfolios = pd.read_csv(
        args.baseline_dir / "monthly_portfolios.csv",
        parse_dates=["signal_date", "return_date"],
    )

    print("constructing EUR market and rolling risk estimates", flush=True)
    market = build_internal_market(panel)
    risk = estimate_rolling_risk(
        panel,
        market,
        window=config.beta_window_months,
        minimum=config.beta_min_observations,
    )

    print("constructing internal EUR factors", flush=True)
    eur_rf = load_eur_short_rate(args.eur_rate)
    factors = build_internal_eur_factors(
        panel,
        eur_rf,
        minimum_size_percentile=config.factor_microcap_quantile,
    )

    print("joining causal controls to OOS predictions", flush=True)
    analysis_predictions = prepare_analysis_predictions(
        predictions,
        panel,
        risk,
    )

    print("running Fama-MacBeth tests", flush=True)
    fmb_monthly, fmb_summary = fama_macbeth_tests(
        analysis_predictions,
        config,
    )

    print("running factor-spanning regressions", flush=True)
    spanning = factor_spanning_tests(
        monthly_portfolios,
        factors,
        config,
    )

    print("running conditional cross-sectional tests", flush=True)
    (
        conditional_monthly,
        conditional_summary,
        conditional_contrasts,
    ) = conditional_predictability(analysis_predictions, config)

    print("running ex-ante market-state tests", flush=True)
    states = build_market_states(market)
    state_monthly, state_summary, state_contrasts = state_dependence_tests(
        analysis_predictions,
        monthly_portfolios,
        states,
        config,
    )

    manifest = write_depth_outputs(
        args.output_dir,
        config,
        market,
        risk,
        factors,
        fmb_monthly,
        fmb_summary,
        spanning,
        conditional_monthly,
        conditional_summary,
        conditional_contrasts,
        states,
        state_monthly,
        state_summary,
        state_contrasts,
        {
            "panel": args.panel,
            "predictions": args.baseline_dir / "predictions.parquet",
            "monthly_portfolios": args.baseline_dir / "monthly_portfolios.csv",
            "eur_rate": args.eur_rate,
        },
    )
    print(json.dumps(manifest["rows"], indent=2))
    print(f"outputs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
