from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_us_comparison_exhibits.py"
)
SPEC = importlib.util.spec_from_file_location("build_us_comparison_exhibits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_side_by_side(path: Path) -> None:
    rows = []
    for model, base, target in [
        ("momentum_rank", "momentum", "rank"),
        ("ridge_rank", "ridge", "rank"),
        ("ridge_return", "ridge", "return"),
        ("elastic_net_rank", "elastic_net", "rank"),
        ("elastic_net_return", "elastic_net", "return"),
        ("hist_gbm_rank", "hist_gbm", "rank"),
        ("mlp_rank", "mlp", "rank"),
    ]:
        rows.append(
            {
                "model": model,
                "base_model": base,
                "target_mode": target,
                "weighting": "value",
                "universe_variant": "standard_ex_bottom_5pct",
                "portfolio": "long_short",
                "cost_bps": 25,
                "mean_monthly_spearman_ic_europe": 0.11,
                "mean_monthly_spearman_ic_us": 0.12,
                "mean_monthly_spearman_ic_us_minus_europe": 0.01,
                "annualized_net_mean_return_europe": 0.20,
                "annualized_net_mean_return_us": 0.30,
                "annualized_net_volatility_europe": 0.30,
                "annualized_net_volatility_us": 0.31,
                "net_sharpe_europe": 0.70,
                "net_sharpe_us": 1.10 if target == "rank" else -0.70,
                "average_monthly_turnover_europe": 0.85,
                "average_monthly_turnover_us": 1.00,
                "months_europe": 137,
                "months_us": 137,
                "observations_europe": 459829,
                "observations_us": 500766,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_rank_model_comparison_filters_and_orders_rank_models(tmp_path: Path):
    side = tmp_path / "side_by_side_model_summary.csv"
    _write_side_by_side(side)

    table = MODULE.rank_model_comparison_table(side, "compustat_enriched")

    assert table["model"].tolist() == MODULE.MODEL_ORDER
    assert table["feature_set"].unique().tolist() == ["compustat_enriched"]
    assert table["model_label"].tolist() == [
        "Momentum",
        "Ridge",
        "Elastic Net",
        "HistGBM",
        "MLP",
    ]
    assert (table["ic_us_minus_europe"] == 0.01).all()


def test_return_target_instability_pairs_rank_and_return(tmp_path: Path):
    side = tmp_path / "side_by_side_model_summary.csv"
    _write_side_by_side(side)

    table = MODULE.return_target_instability_table(side, "refinitiv_only")

    # momentum has no return-target counterpart and is excluded
    assert "momentum" not in table["base_model"].tolist()
    assert table["base_model"].tolist() == [
        "ridge",
        "ridge",
        "elastic_net",
        "elastic_net",
        "hist_gbm",
        "mlp",
    ]
    returns = table[table["target_mode"] == "return"]
    assert (returns["net_sharpe_us"] == -0.70).all()


def test_panel_coverage_reads_manifests_and_predictions(tmp_path: Path):
    side = tmp_path / "side_by_side_model_summary.csv"
    _write_side_by_side(side)
    benchmark_dirs = {}
    for market, rics in (("Europe", ["A.PA", "B.DE"]), ("US", ["C.N", "D.O", "E.N"])):
        run_dir = tmp_path / market.lower()
        run_dir.mkdir()
        (run_dir / "ml_manifest.json").write_text(
            json.dumps(
                {
                    "sample_filter_audit": {
                        "loaded_rows": 1000,
                        "model_rows": 400,
                    }
                }
            )
        )
        pd.DataFrame({"ric": rics * 4}).to_parquet(run_dir / "predictions.parquet")
        benchmark_dirs[market] = run_dir

    table = MODULE.panel_coverage_table(benchmark_dirs, side)

    assert table["market"].tolist() == ["Europe", "US"]
    assert table["securities_with_oos_predictions"].tolist() == [2, 3]
    assert table["oos_months"].tolist() == [137, 137]
    assert table["avg_oos_names_per_month"].tolist() == [3356.4, 3655.2]


def test_compustat_match_quality_reads_panel_section(tmp_path: Path):
    audit_paths = {}
    for market, monthly in (("Europe", 819116), ("US", 404471)):
        path = tmp_path / f"{market.lower()}_audit.json"
        path.write_text(
            json.dumps(
                {
                    "panel": {
                        "rows": 100,
                        "rows_with_compustat_annual": 60,
                        "rows_with_compustat_monthly": monthly,
                        "unique_rics_with_compustat_annual": 10,
                        "unique_rics_with_compustat_monthly": 8,
                        "mean_compustat_feature_count": 14.0713,
                    }
                }
            )
        )
        audit_paths[market] = path

    table = MODULE.compustat_match_quality_table(audit_paths)

    assert table["market"].tolist() == ["Europe", "US"]
    assert table["rows_with_compustat_monthly"].tolist() == [819116, 404471]
    assert table["mean_compustat_feature_count"].tolist() == [14.07, 14.07]


def test_return_correlation_table_keeps_rank_models_only(tmp_path: Path):
    path = tmp_path / "monthly_return_correlations.csv"
    pd.DataFrame(
        {
            "model": ["ridge_rank", "ridge_return", "momentum_rank"],
            "common_months": [137, 125, 137],
            "first_common_month": ["2015-02-28"] * 3,
            "last_common_month": ["2026-06-30"] * 3,
            "return_correlation": [0.1076, -0.0788, 0.4577],
        }
    ).to_csv(path, index=False)

    table = MODULE.return_correlation_table(path, "compustat_enriched")

    assert table["model"].tolist() == ["momentum_rank", "ridge_rank"]
    assert table["return_correlation"].tolist() == [0.46, 0.11]


def test_ic_sharpe_figure_writes_png_pdf_and_data(tmp_path: Path):
    side = tmp_path / "side_by_side_model_summary.csv"
    _write_side_by_side(side)
    data = MODULE.ic_sharpe_figure_data(side)

    record = MODULE.build_ic_sharpe_figure(
        data,
        tmp_path,
        "us_europe_rank_ic_sharpe",
        source_files=[side],
        feature_set_label="Compustat-enriched",
    )

    assert Path(record.png).exists()
    assert Path(record.pdf).exists()
    assert Path(record.data_csv).exists()
    assert data["model"].tolist() == MODULE.MODEL_ORDER
