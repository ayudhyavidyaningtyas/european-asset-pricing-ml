from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from estimates_identification import (
    categorical_balance,
    coverage_weights,
    fit_monthly_coverage_propensity,
    hac_mean,
    holm_within,
    monthly_ic,
    standardized_mean_differences,
)


INTERACTION_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_data_depth_model_depth_interaction.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_data_depth_model_depth_interaction", INTERACTION_SCRIPT
)
INTERACTION = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = INTERACTION
SPEC.loader.exec_module(INTERACTION)


def _prediction_frame(seed: int = 0, months: int = 8, names: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=months, freq="ME")
    records = []
    for date in dates:
        signal = rng.normal(size=names)
        noise = rng.normal(size=names)
        for model, weight in {"ridge": 0.3, "hist_gbm": 0.6}.items():
            records.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "ric": [f"S{index:03d}" for index in range(names)],
                        "base_model": model,
                        "prediction": signal,
                        "target_return_1m": weight * signal + noise,
                    }
                )
            )
    return pd.concat(records, ignore_index=True)


def _coverage_panel(seed: int = 7, months: int = 6, names: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=months, freq="ME")
    frames = []
    for date in dates:
        size = rng.uniform(-1.0, 1.0, size=names)
        frame = pd.DataFrame(
            {
                "date": date,
                "ric": [f"S{index:03d}" for index in range(names)],
                "log_size_rank": size,
                "log_trading_value_eur_rank": size + rng.normal(scale=0.2, size=names),
                "turnover_12m_rank": rng.uniform(-1.0, 1.0, size=names),
                "volatility_12m_rank": rng.uniform(-1.0, 1.0, size=names),
                "book_to_market_rank": rng.uniform(-1.0, 1.0, size=names),
                "momentum_12_2_rank": rng.uniform(-1.0, 1.0, size=names),
                "screen_country": rng.choice(["DE", "FR", "GB"], size=names),
                "TR.TRBCECONOMICSECTOR": rng.choice(["Industrials", "Energy"], size=names),
            }
        )
        # Coverage is strongly increasing in size: the selection this test exists
        # to detect.
        probability = 1.0 / (1.0 + np.exp(-3.0 * size))
        frame["is_covered"] = rng.uniform(size=names) < probability
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_unit_weight_ic_matches_test_b_interaction_script():
    frame = _prediction_frame()

    shared = monthly_ic(frame).sort_values(["base_model", "date"]).reset_index(drop=True)
    reference = (
        INTERACTION.monthly_ic(frame)
        .sort_values(["base_model", "date"])
        .reset_index(drop=True)
    )

    pd.testing.assert_series_equal(shared["ic"], reference["ic"])
    pd.testing.assert_series_equal(
        shared["names"].astype(int), reference["names"].astype(int)
    )


def test_binary_weights_select_a_subsample_of_the_month():
    frame = _prediction_frame(seed=3, months=2, names=30)
    frame["weight"] = np.where(frame["ric"].str.endswith(("0", "1", "2", "3")), 1.0, 0.0)

    weighted = monthly_ic(frame, weight_column="weight")

    month = frame[frame["base_model"].eq("ridge")].copy()
    first_date = month["date"].min()
    month = month[month["date"].eq(first_date)]
    ranks_x = month["prediction"].rank()
    ranks_y = month["target_return_1m"].rank()
    selected = month["weight"].gt(0)
    expected = np.corrcoef(ranks_x[selected], ranks_y[selected])[0, 1]

    actual = weighted[
        weighted["base_model"].eq("ridge") & weighted["date"].eq(first_date)
    ]["ic"].iloc[0]
    assert actual == pytest.approx(expected)


def test_effective_names_falls_below_the_raw_count_when_weights_are_uneven():
    frame = _prediction_frame(seed=5, months=2, names=20)
    frame["weight"] = np.where(frame["ric"].eq("S000"), 50.0, 1.0)

    result = monthly_ic(frame, weight_column="weight")

    assert (result["effective_names"] < result["names"]).all()


def test_hac_mean_reports_only_the_month_count_when_the_series_is_short():
    short = pd.Series(np.random.default_rng(0).normal(size=12))

    result = hac_mean(short, 6, "data_depth_effect")

    assert result == {"quantity": "data_depth_effect", "months": 12}


def test_hac_mean_matches_the_test_b_interaction_script():
    series = pd.Series(np.random.default_rng(1).normal(scale=0.01, size=60))

    assert hac_mean(series, 6, "data_depth_effect") == INTERACTION._hac_mean(
        series, 6, "data_depth_effect"
    )


def test_holm_within_adjusts_inside_groups_and_skips_untested_rows():
    frame = pd.DataFrame(
        {
            "scope": ["a", "a", "b", "b"],
            "p_value": [0.01, 0.04, 0.01, np.nan],
        }
    )

    result = holm_within(frame, ["scope"])

    assert result.loc[0, "p_value_holm"] == pytest.approx(0.02)
    assert result.loc[1, "p_value_holm"] == pytest.approx(0.04)
    # The single tested row in group "b" is adjusted against itself only.
    assert result.loc[2, "p_value_holm"] == pytest.approx(0.01)
    assert np.isnan(result.loc[3, "p_value_holm"])


def test_coverage_propensity_recovers_a_size_driven_selection_rule():
    panel = _coverage_panel()

    propensity, diagnostics = fit_monthly_coverage_propensity(panel)

    assert diagnostics["model_fitted"].all()
    assert diagnostics["auc"].min() > 0.75
    assert propensity.between(0.0, 1.0).all()
    covered = panel["is_covered"]
    assert propensity[covered].mean() > propensity[~covered].mean()


def test_coverage_propensity_falls_back_to_the_observed_rate_without_variation():
    panel = _coverage_panel(months=1, names=150)
    panel["is_covered"] = True

    propensity, diagnostics = fit_monthly_coverage_propensity(panel)

    assert not diagnostics["model_fitted"].any()
    assert np.isnan(diagnostics["auc"]).all()
    assert propensity.eq(1.0).all()


def test_coverage_weights_average_one_per_month_and_ignore_uncovered_rows():
    panel = _coverage_panel()
    propensity, _ = fit_monthly_coverage_propensity(panel)
    panel["coverage_propensity"] = propensity

    weights = coverage_weights(panel)

    assert weights[~panel["is_covered"]].isna().all()
    monthly_mean = weights.groupby(panel["date"]).mean()
    assert monthly_mean.sub(1.0).abs().max() == pytest.approx(0.0, abs=1e-9)
    # Sparsely covered names carry the most weight.
    covered = panel["is_covered"]
    assert (
        weights[covered].corr(panel.loc[covered, "coverage_propensity"]) < 0
    )


def test_coverage_weights_floor_the_propensity_before_inverting():
    panel = _coverage_panel(months=1, names=200)
    panel["coverage_propensity"] = 0.0001

    weights = coverage_weights(panel, min_propensity=0.05)

    covered = panel["is_covered"]
    # Constant propensity plus normalisation leaves every weight at one; the
    # floor is what stops the raw 1/p from exploding beforehand.
    assert weights[covered].max() == pytest.approx(1.0)


def test_inverse_propensity_weighting_shrinks_the_coverage_size_tilt():
    panel = _coverage_panel()
    propensity, _ = fit_monthly_coverage_propensity(panel)
    panel["coverage_propensity"] = propensity
    panel["coverage_weight"] = coverage_weights(panel)

    raw = standardized_mean_differences(panel, ["log_size_rank"])
    weighted = standardized_mean_differences(
        panel, ["log_size_rank"], weight_column="coverage_weight"
    )

    raw_gap = abs(raw["standardized_mean_difference"].iloc[0])
    weighted_gap = abs(weighted["standardized_mean_difference"].iloc[0])
    assert raw_gap > 0.3
    assert weighted_gap < raw_gap / 2
    assert not raw["weighted"].iloc[0]
    assert weighted["weighted"].iloc[0]


def test_categorical_balance_reports_shares_and_is_fixed_by_weighting():
    panel = _coverage_panel()
    # Coverage is concentrated in one country on top of the size tilt.
    panel.loc[panel["screen_country"].eq("GB"), "is_covered"] = False
    propensity, _ = fit_monthly_coverage_propensity(panel)
    panel["coverage_propensity"] = propensity
    panel["coverage_weight"] = coverage_weights(panel)

    raw = categorical_balance(panel, ["screen_country"])
    weighted = categorical_balance(
        panel, ["screen_country"], weight_column="coverage_weight"
    )

    raw_gb = raw[raw["level"].eq("GB")].iloc[0]
    weighted_gb = weighted[weighted["level"].eq("GB")].iloc[0]
    assert raw_gb["covered_share"] == 0.0
    assert raw_gb["standardized_mean_difference"] < -0.5
    assert weighted_gb["covered_share"] == 0.0
    assert raw["universe_share"].sum() == pytest.approx(1.0)
    assert weighted[weighted["covariate"].eq("screen_country")][
        "covered_share"
    ].sum() == pytest.approx(1.0)
