from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing import RAW_FEATURES  # noqa: E402
from asset_pricing_ml import (  # noqa: E402
    EXPANDED_FEATURE_COLUMNS,
    FEATURE_SETS,
    WalkForwardConfig,
    _validation_split,
    _limit_training_rows,
    add_residual_targets,
    binned_oos_responses,
    construct_monthly_portfolios,
    load_model_panel,
    portfolio_summary,
    prediction_metrics,
    predictive_accuracy_tests,
    run_walk_forward,
    walk_forward_slices,
)
from estimates_features import ESTIMATES_INFORMATION_TYPES  # noqa: E402


FEATURES = [f"{feature}_rank" for feature in RAW_FEATURES]


def synthetic_panel() -> pd.DataFrame:
    records = []
    for date in pd.date_range("2018-01-31", "2021-12-31", freq="ME"):
        for security in range(20):
            signal = (security - 9.5) / 10
            record = {
                "date": date,
                "target_date": date + pd.offsets.MonthEnd(1),
                "ric": f"S{security:02d}",
                "target_return_1m": signal * 0.02,
                "target_return_rank": signal,
                "company_market_cap": 100.0 + security,
                "market_cap_percentile": (security + 1) / 20,
                "screen_country": "GB",
                "TR.TRBCECONOMICSECTOR": "Industrials",
                "model_eligible": True,
            }
            record.update({feature: signal for feature in FEATURES})
            records.append(record)
    return pd.DataFrame(records)


def synthetic_country_effect_panel() -> pd.DataFrame:
    records = []
    for date in pd.date_range("2020-01-31", periods=2, freq="ME"):
        for country, country_effect in [("GB", 0.5), ("FR", -0.5)]:
            for security in range(20):
                idiosyncratic_signal = (security - 9.5) / 20
                target_rank = country_effect + idiosyncratic_signal
                records.append(
                    {
                        "date": date,
                        "target_date": date + pd.offsets.MonthEnd(1),
                        "ric": f"{country}{security:02d}",
                        "target_return_1m": target_rank / 100,
                        "target_return_rank": target_rank,
                        "company_market_cap": 100.0 + security,
                        "market_cap_percentile": (security + 1) / 20,
                        "screen_country": country,
                        "TR.TRBCECONOMICSECTOR": "Industrials",
                        "model_eligible": True,
                    }
                )
    return pd.DataFrame(records)


def synthetic_model_panel_with_estimates() -> pd.DataFrame:
    revisions = ESTIMATES_INFORMATION_TYPES["revisions"]
    revision_ranks = [f"{feature}_rank" for feature in revisions]
    records = []
    for index, date in enumerate(
        pd.to_datetime(["2004-12-31", "2005-01-31", "2005-01-31"])
    ):
        record = {
            "date": date,
            "target_date": date + pd.offsets.MonthEnd(1),
            "ric": f"S{index:02d}",
            "target_return_1m": 0.01 * index,
            "target_return_rank": (index - 1) / 2,
            "company_market_cap": 100.0 + index,
            "market_cap_percentile": 0.50,
            "screen_country": "GB",
            "TR.TRBCECONOMICSECTOR": "Industrials",
            "eligible": True,
            "model_eligible": True,
            "return_history_n": 36,
            "feature_count": len(FEATURES),
            "estimates_feature_count": 1 if index == 2 else 0,
            "est_signal_lag_months": 1.0 if index == 2 else np.nan,
        }
        record.update({feature: 0.1 * index for feature in FEATURES})
        record.update({feature: np.nan for feature in revisions})
        record.update({feature: 0.0 for feature in revision_ranks})
        if index == 2:
            record["est_eps_revision_1m"] = 0.15
            record["est_eps_revision_1m_rank"] = 1.0
        records.append(record)
    return pd.DataFrame(records)


def test_walk_forward_training_labels_end_before_cutoff():
    panel = synthetic_panel()
    slices = walk_forward_slices(panel, 2020, 2021)

    for _, cutoff, train_mask, _ in slices:
        assert panel.loc[train_mask, "target_date"].max() <= cutoff


def test_load_model_panel_applies_sample_start_date(tmp_path):
    path = tmp_path / "panel.parquet"
    synthetic_model_panel_with_estimates().to_parquet(path, index=False)

    panel = load_model_panel(
        path,
        feature_columns=FEATURES,
        sample_start_date="2005-01-31",
    )

    assert set(panel["ric"]) == {"S01", "S02"}
    assert panel["date"].min() == pd.Timestamp("2005-01-31")
    assert panel.attrs["sample_filter_audit"]["loaded_rows"] == 3
    assert panel.attrs["sample_filter_audit"]["after_sample_start_date"] == 2


def test_load_model_panel_can_require_estimates_feature(tmp_path):
    path = tmp_path / "panel.parquet"
    synthetic_model_panel_with_estimates().to_parquet(path, index=False)

    panel = load_model_panel(
        path,
        feature_columns=FEATURES,
        sample_start_date="2005-01-31",
        require_estimates_feature=True,
    )

    assert panel["ric"].tolist() == ["S02"]
    assert panel.attrs["sample_filter_audit"]["after_require_estimates_feature"] == 1


def test_load_model_panel_can_require_raw_revision_signal(tmp_path):
    path = tmp_path / "panel.parquet"
    synthetic_model_panel_with_estimates().to_parquet(path, index=False)

    panel = load_model_panel(
        path,
        feature_columns=FEATURE_SETS["estimates_revisions_pure"],
        sample_start_date="2005-01-31",
        require_revision_signal=True,
    )

    assert panel["ric"].tolist() == ["S02"]
    assert panel.attrs["sample_filter_audit"]["after_require_revision_signal"] == 1


def test_load_model_panel_can_enforce_estimate_signal_lag(tmp_path):
    path = tmp_path / "panel.parquet"
    synthetic_model_panel_with_estimates().to_parquet(path, index=False)

    panel = load_model_panel(
        path,
        feature_columns=FEATURES,
        require_estimate_signal_lag_months=1,
    )

    assert set(panel["ric"]) == {"S00", "S01", "S02"}
    assert panel.attrs["sample_filter_audit"]["estimate_signal_lag_violations"] == 0


def test_load_model_panel_rejects_unlagged_estimates_features(tmp_path):
    path = tmp_path / "panel.parquet"
    panel = synthetic_model_panel_with_estimates()
    panel.loc[panel["ric"].eq("S02"), "est_signal_lag_months"] = 0.0
    panel.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Estimate signal lag guard failed"):
        load_model_panel(
            path,
            feature_columns=FEATURES,
            require_estimate_signal_lag_months=1,
        )


def test_training_cap_preserves_month_column_and_historical_coverage():
    panel = synthetic_panel()

    sampled = _limit_training_rows(panel, maximum=100, random_state=7)

    assert len(sampled) <= 100
    assert "date" in sampled
    assert sampled["date"].nunique() == panel["date"].nunique()


def test_residual_targets_remove_monthly_country_effects():
    panel = add_residual_targets(synthetic_country_effect_panel())

    country_means = panel.groupby(["date", "screen_country"])[
        "target_rank_residual"
    ].mean()

    assert np.allclose(country_means.to_numpy(), 0.0, atol=1e-6)
    assert panel["target_residual_rank"].between(-1.0, 1.0).all()


def test_residual_rank_mode_uses_neutralized_target_scale():
    panel = synthetic_panel().assign(
        target_residual_rank=lambda frame: frame["target_return_rank"],
    )
    config = WalkForwardConfig(
        first_test_year=2020,
        last_test_year=2021,
        min_training_rows=10,
    )

    predictions, _, _, _ = run_walk_forward(
        panel,
        ["ridge"],
        config,
        target_column="target_residual_rank",
        target_mode="residual_rank",
        collect_importance=False,
    )
    metrics = prediction_metrics(predictions)

    assert set(predictions["model"]) == {"ridge_residual_rank"}
    assert metrics.loc[0, "target_column"] == "target_residual_rank"
    assert np.isfinite(metrics.loc[0, "target_r2_zero"])


def test_validation_window_is_the_trailing_training_tail():
    panel = synthetic_panel()

    core, validation = _validation_split(panel, validation_months=12)

    assert core["date"].max() < validation["date"].min()
    assert validation["date"].nunique() == 12


def test_ridge_walk_forward_is_causal_and_unique():
    panel = synthetic_panel()
    config = WalkForwardConfig(
        first_test_year=2020,
        last_test_year=2021,
        min_training_rows=10,
    )

    predictions, fit_log, coefficients, importance = run_walk_forward(
        panel, ["ridge", "momentum"], config
    )

    assert not predictions.duplicated(["model", "date", "ric"]).any()
    assert (
        pd.to_datetime(fit_log["train_target_end"])
        <= pd.to_datetime(fit_log["train_label_cutoff"])
    ).all()
    assert fit_log["in_sample_r2_zero"].notna().all()
    assert fit_log["in_sample_observations"].eq(fit_log["train_rows_used"]).all()
    assert set(predictions["model"]) == {"ridge_rank", "momentum_rank"}
    assert len(coefficients) == 2 * len(FEATURES)
    assert set(importance["ablation_level"]) == {"feature", "theme"}
    assert "liquidity" in set(importance["variable"])


def test_dre_walk_forward_is_deterministic_and_logged():
    panel = synthetic_panel()
    config = WalkForwardConfig(
        first_test_year=2020,
        last_test_year=2021,
        min_training_rows=10,
        dre_layers=1,
        dre_features_per_block=4,
        dre_gammas=(0.5, 1.0),
        dre_alphas=(0.1, 1.0),
    )

    predictions, fit_log, coefficients, importance = run_walk_forward(
        panel,
        ["dre"],
        config,
        collect_importance=False,
    )
    repeated, _, _, _ = run_walk_forward(
        panel,
        ["dre"],
        config,
        collect_importance=False,
    )

    assert set(predictions["model"]) == {"dre_rank"}
    assert not predictions.duplicated(["model", "date", "ric"]).any()
    assert fit_log["ensemble_width"].eq(4).all()
    assert fit_log["random_features_per_block"].eq(4).all()
    assert coefficients.empty
    assert importance.empty
    assert np.allclose(predictions["prediction"], repeated["prediction"])


def test_dre_final_alpha_can_be_validation_selected_and_logged():
    panel = synthetic_panel()
    config = WalkForwardConfig(
        first_test_year=2021,
        last_test_year=2021,
        min_training_rows=10,
        validation_months=6,
        dre_layers=1,
        dre_features_per_block=4,
        dre_gammas=(0.5, 1.0),
        dre_alphas=(0.1, 1.0),
        dre_tune_final_alpha=True,
        dre_final_alpha=99.0,
        dre_final_alphas=(0.01, 1.0),
    )

    _, fit_log, _, _ = run_walk_forward(
        panel,
        ["dre"],
        config,
        collect_importance=False,
    )

    parameters = json.loads(fit_log.loc[0, "selected_parameters"])
    assert parameters["final_alpha_tuned"] is True
    assert parameters["final_alpha_grid"] == [0.01, 1.0]
    assert fit_log.loc[0, "selected_final_alpha"] in {0.01, 1.0}
    assert fit_log.loc[0, "final_alpha"] in {0.01, 1.0}
    assert np.isfinite(fit_log.loc[0, "final_alpha_validation_loss"])
    assert np.isfinite(fit_log.loc[0, "validation_loss"])


def test_portfolio_cost_is_turnover_times_cost_rate():
    panel = synthetic_panel()
    predictions = panel.assign(
        prediction=panel["target_return_rank"],
        model="signal_rank",
        base_model="signal",
        target_mode="rank",
    )
    metrics = prediction_metrics(predictions)
    monthly = construct_monthly_portfolios(predictions, quantile=0.10)
    summary = portfolio_summary(monthly, metrics, (0, 25))

    equal = summary[
        summary["weighting"].eq("equal")
        & summary["universe_variant"].eq("standard_ex_bottom_5pct")
        & summary["portfolio"].eq("long_short")
    ].set_index("cost_bps")
    assert equal.loc[25, "net_sharpe"] == equal.loc[25, "sharpe"]
    assert equal.loc[25, "annualized_net_mean_return"] == equal.loc[
        25,
        "annualized_mean_return",
    ]
    assert equal.loc[25, "level_metric_bootstrap_resampling_unit"] == "months"
    assert equal.loc[25, "annualized_net_mean_return_ci_low"] < equal.loc[
        25,
        "annualized_net_mean_return_ci_high",
    ]
    assert equal.loc[25, "net_sharpe_ci_low"] < equal.loc[
        25,
        "net_sharpe_ci_high",
    ]
    assert 0.0 <= equal.loc[
        25,
        "annualized_net_mean_return_p_two_sided_zero",
    ] <= 1.0
    assert equal.loc[25, "annualized_gross_mean_return"] == equal.loc[
        25,
        "gross_annualized_mean_return",
    ]
    assert equal.loc[25, "gross_sharpe"] > equal.loc[25, "net_sharpe"]
    expected_difference = (
        monthly.loc[
            monthly["weighting"].eq("equal")
            & monthly["universe_variant"].eq("standard_ex_bottom_5pct"),
            "long_short_turnover",
        ].mean()
        * 0.0025
        * 12
    )
    actual_difference = (
        equal.loc[0, "annualized_mean_return"]
        - equal.loc[25, "annualized_mean_return"]
    )
    assert np.isclose(actual_difference, expected_difference)


def test_prediction_metrics_reward_correct_cross_sectional_ordering():
    panel = synthetic_panel()
    predictions = panel.assign(
        prediction=panel["target_return_rank"],
        model="perfect_rank",
        base_model="perfect",
        target_mode="rank",
    )

    metrics = prediction_metrics(predictions)

    assert np.isclose(metrics.loc[0, "rank_r2_zero"], 1.0)
    assert np.isclose(metrics.loc[0, "rank_r2_zero_monthly_mean"], 1.0)
    assert metrics.loc[0, "r2_zero_bootstrap_resampling_unit"] == "months"
    assert metrics.loc[0, "rank_r2_zero_ci_low"] <= metrics.loc[
        0,
        "rank_r2_zero_ci_high",
    ]
    assert np.isclose(metrics.loc[0, "mean_monthly_spearman_ic"], 1.0)


def test_prediction_metrics_annualizes_ic_ir_with_sqrt_12():
    panel = synthetic_panel().copy()
    month_number = panel.groupby("date").ngroup().to_numpy(dtype=float)
    security_number = panel["ric"].str.extract(r"(\d+)").astype(int)[0].to_numpy(dtype=float)
    noise = 0.30 * np.sin(security_number + month_number / 3.0)
    predictions = panel.assign(
        prediction=panel["target_return_rank"] + noise,
        model="ridge_rank",
        base_model="ridge",
        target_mode="rank",
    )

    metrics = prediction_metrics(predictions)
    monthly_ic = predictions.groupby("date").apply(
        lambda month: month["prediction"].corr(
            month["target_return_rank"],
            method="spearman",
        ),
        include_groups=False,
    )
    expected = monthly_ic.mean() / monthly_ic.std(ddof=1) * np.sqrt(12.0)
    incorrect_linear_annualization = monthly_ic.mean() / monthly_ic.std(ddof=1) * 12.0

    observed = metrics.loc[0, "ic_information_ratio"]
    assert observed == pytest.approx(expected)
    assert observed != pytest.approx(incorrect_linear_annualization)


def test_raw_return_model_reports_return_r2_not_rank_r2():
    panel = synthetic_panel()
    predictions = panel.assign(
        prediction=panel["target_return_1m"],
        model="perfect_return",
        base_model="perfect",
        target_mode="return",
    )

    metrics = prediction_metrics(predictions)

    assert np.isclose(metrics.loc[0, "return_r2_zero"], 1.0)
    assert np.isclose(metrics.loc[0, "return_r2_zero_monthly_mean"], 1.0)
    assert metrics.loc[0, "r2_zero_bootstrap_resampling_unit"] == "months"
    assert pd.isna(metrics.loc[0, "rank_r2_zero"])


def test_predictive_accuracy_ic_holm_family_uses_all_pairwise_model_comparisons():
    panel = synthetic_panel().copy()
    month_number = panel.groupby("date").ngroup().to_numpy(dtype=float)
    security_number = panel["ric"].str.extract(r"(\d+)").astype(int)[0].to_numpy(dtype=float)
    model_noise = {
        "momentum": 0.75,
        "ridge": 0.35,
        "elastic_net": 0.40,
        "hist_gbm": 0.25,
        "mlp": 0.27,
    }
    frames = []
    for model, amplitude in model_noise.items():
        frames.append(
            panel.assign(
                prediction=panel["target_return_rank"]
                + amplitude * np.sin(security_number + month_number / 5.0),
                model=f"{model}_rank",
                base_model=model,
                target_mode="rank",
            )
        )

    _, ic = predictive_accuracy_tests(pd.concat(frames, ignore_index=True))

    expected_pairs = {
        tuple(sorted((f"{left}_rank", f"{right}_rank")))
        for left, right in itertools.combinations(model_noise, 2)
    }
    observed_pairs = {
        tuple(sorted((row.model_a, row.model_b)))
        for row in ic.itertuples(index=False)
    }
    assert observed_pairs == expected_pairs

    expected_holm = multipletests(
        ic["p_value"].fillna(1.0),
        method="holm",
    )[1]
    assert np.allclose(ic["p_value_holm"], expected_holm, equal_nan=True)


def test_predictive_accuracy_tests_compare_models_on_common_target_scale():
    panel = synthetic_panel()
    ridge = panel.assign(
        prediction=panel["target_return_1m"],
        model="ridge_return",
        base_model="ridge",
        target_mode="return",
    )
    zero = panel.assign(
        prediction=0.0,
        model="zero_return",
        base_model="zero",
        target_mode="return",
    )
    ridge_rank = panel.assign(
        prediction=panel["target_return_rank"],
        model="ridge_rank",
        base_model="ridge",
        target_mode="rank",
    )
    zero_rank = panel.assign(
        prediction=0.0,
        model="zero_rank",
        base_model="zero",
        target_mode="rank",
    )

    loss, ic = predictive_accuracy_tests(
        pd.concat([ridge, zero, ridge_rank, zero_rank])
    )

    comparison = loss[loss["target_mode"].eq("return")].iloc[0]
    assert comparison["model_a"] == "ridge_return"
    assert comparison["mean_difference"] < 0
    assert comparison["p_value"] < 0.05
    assert comparison["clark_west_restricted_model"] == "zero_return"
    assert comparison["clark_west_unrestricted_model"] == "ridge_return"
    assert comparison["clark_west_adjusted_mean_difference"] > 0
    assert comparison["clark_west_p_one_sided"] < 0.05
    assert not ic.empty
    assert set(loss["target_mode"]) == {"rank", "return"}


def test_predictive_accuracy_tests_infer_clark_west_orientation_without_base_model():
    panel = synthetic_panel()
    momentum = panel.assign(
        prediction=0.0,
        model="momentum_return",
        base_model="momentum",
        target_mode="return",
    )
    ridge = panel.assign(
        prediction=panel["target_return_1m"],
        model="ridge_return",
        base_model="ridge",
        target_mode="return",
    )

    loss, _ = predictive_accuracy_tests(
        pd.concat([momentum, ridge]).drop(columns=["base_model"])
    )

    comparison = loss.iloc[0]
    assert comparison["clark_west_restricted_model"] == "momentum_return"
    assert comparison["clark_west_unrestricted_model"] == "ridge_return"
    assert comparison["clark_west_adjusted_mean_difference"] > 0


def test_predictive_accuracy_tests_include_residual_rank_family():
    panel = synthetic_panel().assign(
        target_residual_rank=lambda frame: frame["target_return_rank"],
        target_return_residual_1m=lambda frame: frame["target_return_1m"],
    )
    perfect = panel.assign(
        prediction=panel["target_residual_rank"],
        model="perfect_residual_rank",
        base_model="perfect",
        target_mode="residual_rank",
    )
    zero = panel.assign(
        prediction=0.0,
        model="zero_residual_rank",
        base_model="zero",
        target_mode="residual_rank",
    )

    loss, ic = predictive_accuracy_tests(pd.concat([perfect, zero]))

    assert set(loss["target_mode"]) == {"residual_rank"}
    assert set(ic["target_mode"]) == {"residual_rank"}


def test_expanded_feature_set_adds_only_prespecified_liquidity_features():
    assert set(EXPANDED_FEATURE_COLUMNS) - set(FEATURES) == {
        "log_trading_value_eur_rank",
        "turnover_volatility_12m_rank",
    }


def test_binned_oos_responses_use_top_fixed_model_importance_features():
    panel = synthetic_panel()
    predictions = panel.assign(
        prediction=panel["target_return_rank"],
        model="ridge_rank",
        base_model="ridge",
        target_mode="rank",
    )
    importance = pd.DataFrame(
        [
            {
                "model": "ridge_rank",
                "ablation_level": "feature",
                "market_cap_group": "all",
                "variable": "momentum_12_2",
                "delta_r2_zero": 0.10,
            }
        ]
    )

    result = binned_oos_responses(
        predictions,
        panel,
        importance,
        FEATURES,
    )

    assert set(result["variable"]) == {"momentum_12_2"}
    assert result["feature_bin"].nunique() > 1


def test_load_model_panel_includes_unlabelled_scoreable_delisting(tmp_path):
    date = pd.Timestamp("2025-10-31")
    labelled = {
        "date": pd.Timestamp("2025-09-30"),
        "target_date": date,
        "ric": "LIVE",
        "target_return_1m": 0.01,
        "target_return_rank": 0.2,
        "company_market_cap": 100.0,
        "market_cap_percentile": 0.5,
        "screen_country": "GB",
        "TR.TRBCECONOMICSECTOR": "Industrials",
        "eligible": True,
        "model_eligible": True,
        "return_history_n": 30,
        "feature_count": 18,
    }
    candidate = {
        **labelled,
        "date": date,
        "target_date": pd.NaT,
        "ric": "DEAD^K25",
        "target_return_1m": np.nan,
        "target_return_rank": np.nan,
        "model_eligible": False,
    }
    for feature in FEATURES:
        labelled[feature] = 0.1
        candidate[feature] = -0.1
    horizon = {
        **labelled,
        "date": date,
        "target_date": pd.Timestamp("2025-11-30"),
        "ric": "LIVE2",
    }
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame([labelled, horizon, candidate]).to_parquet(panel_path, index=False)
    audit_path = tmp_path / "audit.csv"
    pd.DataFrame(
        {
            "ric": ["DEAD^K25"],
            "retire_month": ["2025-11"],
            "missing_retirement_month_return": [True],
        }
    ).to_csv(audit_path, index=False)

    result = load_model_panel(panel_path, audit_path)
    delisting = result[result["is_delisting_candidate"]].iloc[0]

    assert len(result) == 3
    assert delisting["target_date"] == pd.Timestamp("2025-11-30")
    assert pd.isna(delisting["target_return_1m"])


def test_delisting_penalty_enters_short_leg_only_after_assignment():
    date = pd.Timestamp("2025-10-31")
    records = []
    for security in range(100):
        is_candidate = security == 0
        records.append(
            {
                "date": date,
                "target_date": date + pd.offsets.MonthEnd(1),
                "ric": f"S{security:03d}",
                "target_return_1m": (
                    np.nan if is_candidate else security / 10_000
                ),
                "prediction": float(security),
                "model": "ridge_rank",
                "target_mode": "rank",
                "company_market_cap": 100.0,
                "market_cap_percentile": 0.5,
                "is_delisting_candidate": is_candidate,
            }
        )
    predictions = pd.DataFrame(records)
    observed = construct_monthly_portfolios(predictions, 0.10)
    stressed_predictions = predictions.assign(
        target_return_1m=predictions["target_return_1m"].fillna(-1.0)
    )
    stressed = construct_monthly_portfolios(stressed_predictions, 0.10)

    observed_row = observed[
        observed["weighting"].eq("equal")
        & observed["universe_variant"].eq("standard_ex_bottom_5pct")
    ].iloc[0]
    stressed_row = stressed[
        stressed["weighting"].eq("equal")
        & stressed["universe_variant"].eq("standard_ex_bottom_5pct")
    ].iloc[0]
    assert observed_row["delisting_candidates_short_n"] == 0
    assert stressed_row["delisting_candidates_short_n"] == 1
    assert (
        stressed_row["gross_long_short_return"]
        > observed_row["gross_long_short_return"]
    )


def test_long_only_summary_uses_eur_excess_returns():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    monthly = pd.DataFrame(
        {
            "model": "ridge_rank",
            "target_mode": "rank",
            "weighting": "value",
            "universe_variant": "standard_ex_bottom_5pct",
            "signal_date": dates - pd.offsets.MonthEnd(1),
            "return_date": dates,
            "gross_long_short_return": np.linspace(-0.01, 0.02, len(dates)),
            "long_return": np.linspace(0.0, 0.03, len(dates)),
            "long_short_turnover": 0.2,
            "long_only_turnover": 0.2,
        }
    )
    metrics = pd.DataFrame(
        {
            "model": ["ridge_rank"],
            "target_mode": ["rank"],
        }
    )
    risk_free = pd.Series(0.001, index=dates)

    summary = portfolio_summary(
        monthly,
        metrics,
        (25,),
        risk_free=risk_free,
    )
    long_only = summary[
        summary["portfolio"].eq("long_only_top_decile")
    ].iloc[0]

    assert np.isclose(
        long_only["annualized_mean_return"]
        - long_only["annualized_excess_return"],
        0.012,
    )
    assert long_only["rf_missing_months"] == 0
