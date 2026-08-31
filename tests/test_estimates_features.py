from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_pricing import ALL_RAW_FEATURES  # noqa: E402
from asset_pricing_ml import (  # noqa: E402
    COMPUSTAT_FEATURE_COLUMNS,
    ESTIMATES_COVERAGE_FEATURE_COLUMNS,
    ESTIMATES_FEATURE_COLUMNS,
    ESTIMATES_MISSINGNESS_FEATURE_COLUMNS,
    FEATURE_SETS,
)
from estimates_features import (  # noqa: E402
    ESTIMATES_DEGENERATE_MODEL_FEATURES,
    ESTIMATES_FEATURE_FAMILIES,
    ESTIMATES_INFORMATION_TYPES,
    ESTIMATES_MODEL_FEATURES,
    EstimatesPanelConfig,
    build_estimates_enriched_panel,
    prepare_estimates_snapshot_features,
    write_estimates_outputs,
)


def base_panel() -> pd.DataFrame:
    rows = []
    for ric, isin, price in [
        ("AAA.L", "GB0000000001", 10.0),
        ("BBB.L", "GB0000000002", 20.0),
    ]:
        for date in pd.to_datetime(
            ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"]
        ):
            row = {
                "date": date,
                "ric": ric,
                "TR.ISIN": isin,
                "price_close": price,
                "company_market_cap": price * 100.0,
                "eligible": True,
            }
            row.update({feature: 1.0 for feature in ALL_RAW_FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)


def estimates_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Instrument": "AAA.L",
                "TR.ISIN": "GB0000000001",
                "Date": "2024-01-31",
                "TR.EPSMean": 1.0,
                "TR.EPSHigh": 1.2,
                "TR.EPSLow": 0.8,
                "TR.EPSNumEstimates": 4,
                "TR.RevenueMean": 100.0,
                "TR.RevenueHigh": 110.0,
                "TR.RevenueLow": 90.0,
                "TR.RevenueNumEstimates": 5,
                "TR.PriceTargetMean": 12.0,
                "TR.PriceTargetHigh": 14.0,
                "TR.PriceTargetLow": 10.0,
                "TR.PriceTargetNumEstimates": 3,
                "TR.RecommendationMean": 2.0,
                "TR.RecommendationNumEstimates": 6,
            },
            {
                "Instrument": "AAA.L",
                "TR.ISIN": "GB0000000001",
                "Date": "2024-02-29",
                "TR.EPSMean": 1.1,
                "TR.EPSHigh": 1.3,
                "TR.EPSLow": 0.9,
                "TR.EPSNumEstimates": 5,
                "TR.RevenueMean": 105.0,
                "TR.RevenueHigh": 115.0,
                "TR.RevenueLow": 95.0,
                "TR.RevenueNumEstimates": 5,
                "TR.PriceTargetMean": 13.0,
                "TR.PriceTargetHigh": 15.0,
                "TR.PriceTargetLow": 11.0,
                "TR.PriceTargetNumEstimates": 4,
                "TR.RecommendationMean": 1.8,
                "TR.RecommendationNumEstimates": 6,
            },
            {
                "Instrument": "AAA.L",
                "TR.ISIN": "GB0000000001",
                "Date": "2024-04-30",
                "TR.EPSMean": 1.4,
                "TR.EPSHigh": 1.6,
                "TR.EPSLow": 1.2,
                "TR.EPSNumEstimates": 6,
                "TR.RevenueMean": 112.0,
                "TR.RevenueHigh": 120.0,
                "TR.RevenueLow": 104.0,
                "TR.RevenueNumEstimates": 6,
                "TR.PriceTargetMean": 15.0,
                "TR.PriceTargetHigh": 18.0,
                "TR.PriceTargetLow": 12.0,
                "TR.PriceTargetNumEstimates": 5,
                "TR.RecommendationMean": 1.6,
                "TR.RecommendationNumEstimates": 7,
            },
            {
                "Instrument": "BBB.L",
                "TR.ISIN": "GB0000000002",
                "Date": "2024-01-31",
                "TR.EPSMean": 2.0,
                "TR.EPSHigh": 2.3,
                "TR.EPSLow": 1.7,
                "TR.EPSNumEstimates": 8,
                "TR.RevenueMean": 200.0,
                "TR.RevenueHigh": 220.0,
                "TR.RevenueLow": 180.0,
                "TR.RevenueNumEstimates": 8,
                "TR.PriceTargetMean": 24.0,
                "TR.PriceTargetHigh": 28.0,
                "TR.PriceTargetLow": 20.0,
                "TR.PriceTargetNumEstimates": 7,
                "TR.RecommendationMean": 3.0,
                "TR.RecommendationNumEstimates": 9,
            },
        ]
    )


def test_estimates_features_are_lag_safe_and_ranked():
    _, panel, audit = build_estimates_enriched_panel(base_panel(), estimates_rows())

    feb = panel[panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-02-29"))].iloc[0]
    april = panel[panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-04-30"))].iloc[0]

    assert np.isclose(feb["est_eps_revision_1m"], 0.1)
    assert pd.isna(april["est_eps_revision_1m"])
    assert np.isclose(april["est_eps_revision_3m"], 0.4)
    assert np.isclose(april["est_price_target_upside"], 0.5)
    assert panel["est_eps_yield_rank"].notna().all()
    assert april["estimates_feature_count"] > 0
    assert "estimates_feature_count_rank" in panel.columns
    assert panel["estimates_feature_count_rank"].notna().all()
    assert audit["panel"]["unique_rics_with_estimates"] == 2


def test_write_estimates_outputs_accepts_custom_filenames(tmp_path: Path):
    snapshot, panel, audit = build_estimates_enriched_panel(base_panel(), estimates_rows())

    write_estimates_outputs(
        tmp_path,
        snapshot,
        panel,
        audit,
        panel_filename="us_panel.parquet",
        snapshot_filename="us_snapshot.parquet",
        dictionary_filename="us_dictionary.csv",
        audit_filename="us_audit.json",
    )

    assert (tmp_path / "us_panel.parquet").exists()
    assert (tmp_path / "us_snapshot.parquet").exists()
    assert (tmp_path / "us_dictionary.csv").exists()
    assert (tmp_path / "us_audit.json").exists()


def test_estimates_features_fall_back_to_isin_when_ric_is_missing():
    raw = estimates_rows().drop(columns=["Instrument"])
    _, panel, audit = build_estimates_enriched_panel(base_panel(), raw)

    matched = panel[panel["est_snapshot_date"].notna()]
    assert matched["ric"].nunique() == 2
    assert audit["snapshot"]["unique_rics"] == 0
    assert audit["snapshot"]["unique_isins"] == 2


def test_strict_identifier_match_rejects_conflicting_estimate_isin():
    raw = estimates_rows().copy()
    raw.loc[raw["Date"].eq("2024-01-31") & raw["Instrument"].eq("AAA.L"), "TR.ISIN"] = (
        "GB9999999999"
    )

    _, panel, audit = build_estimates_enriched_panel(
        base_panel(),
        raw,
        config=EstimatesPanelConfig(strict_identifier_match=True),
    )

    january = panel[
        panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-01-31"))
    ].iloc[0]
    february = panel[
        panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-02-29"))
    ].iloc[0]

    assert pd.isna(january["est_snapshot_date"])
    assert january["est_identifier_mismatch"]
    assert pd.isna(february["est_eps_revision_1m"])
    assert audit["controls"]["identifier_mismatch_rows"] == 1


def test_estimates_signal_lag_uses_previous_month_features_before_ranking():
    _, panel, audit = build_estimates_enriched_panel(
        base_panel(),
        estimates_rows(),
        config=EstimatesPanelConfig(signal_lag_months=1),
    )

    february = panel[
        panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-02-29"))
    ].iloc[0]
    march = panel[
        panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-03-31"))
    ].iloc[0]

    assert february["est_snapshot_date"] == pd.Timestamp("2024-01-31")
    assert np.isclose(february["est_eps_yield"], 0.1)
    assert pd.isna(february["est_eps_revision_1m"])
    assert march["est_snapshot_date"] == pd.Timestamp("2024-02-29")
    assert np.isclose(march["est_eps_revision_1m"], 0.1)
    assert audit["controls"]["rows_with_estimates_after_signal_lag"] == 3


def test_extreme_estimates_filter_nulls_implausible_values_before_ranking():
    raw = estimates_rows().copy()
    raw.loc[raw["Date"].eq("2024-01-31") & raw["Instrument"].eq("AAA.L"), "TR.PriceTargetMean"] = (
        10_000.0
    )

    _, panel, audit = build_estimates_enriched_panel(
        base_panel(),
        raw,
        config=EstimatesPanelConfig(filter_extreme_estimates=True),
    )

    january = panel[
        panel["ric"].eq("AAA.L") & panel["date"].eq(pd.Timestamp("2024-01-31"))
    ].iloc[0]

    assert pd.isna(january["est_price_target_upside"])
    assert january["est_price_target_upside_rank"] == 0.0
    assert (
        audit["controls"]["outlier_filtered_rows_by_feature"][
            "est_price_target_upside"
        ]
        == 1
    )


def test_prepare_estimates_snapshot_features_collapses_duplicate_snapshots():
    raw = pd.concat(
        [
            estimates_rows().iloc[[0]],
            estimates_rows().iloc[[0]].assign(**{"TR.EPSMean": 1.05}),
        ],
        ignore_index=True,
    )

    snapshots, audit = prepare_estimates_snapshot_features(raw)

    assert len(snapshots) == 1
    assert np.isclose(snapshots["eps_mean"].iloc[0], 1.05)
    assert audit["source_rows"] == 2


def test_ml_feature_set_exposes_estimates_rank_columns():
    added = set(ESTIMATES_FEATURE_COLUMNS) - set(COMPUSTAT_FEATURE_COLUMNS)
    degenerate = {f"{feature}_rank" for feature in ESTIMATES_DEGENERATE_MODEL_FEATURES}

    assert added == {f"{feature}_rank" for feature in ESTIMATES_MODEL_FEATURES}
    assert not added.intersection(degenerate)
    assert FEATURE_SETS["estimates_enriched"] == ESTIMATES_FEATURE_COLUMNS


def test_coverage_aware_feature_set_adds_only_feature_count_rank():
    added = set(ESTIMATES_COVERAGE_FEATURE_COLUMNS) - set(ESTIMATES_FEATURE_COLUMNS)

    assert added == set(ESTIMATES_MISSINGNESS_FEATURE_COLUMNS)
    assert FEATURE_SETS["estimates_enriched_with_coverage"] == (
        ESTIMATES_COVERAGE_FEATURE_COLUMNS
    )
    assert FEATURE_SETS["estimates_coverage_only"] == [
        *COMPUSTAT_FEATURE_COLUMNS,
        *ESTIMATES_MISSINGNESS_FEATURE_COLUMNS,
    ]


def test_estimates_only_ablation_sets_are_compustat_plus_group_columns():
    groups = {
        **ESTIMATES_FEATURE_FAMILIES,
        **ESTIMATES_INFORMATION_TYPES,
    }

    for group_name, features in groups.items():
        columns = FEATURE_SETS[f"estimates_{group_name}_only"]

        assert columns[: len(COMPUSTAT_FEATURE_COLUMNS)] == COMPUSTAT_FEATURE_COLUMNS
        assert columns[len(COMPUSTAT_FEATURE_COLUMNS) :] == [
            f"{feature}_rank" for feature in features
        ]


def test_estimates_leave_one_family_sets_remove_only_that_family():
    full = set(ESTIMATES_FEATURE_COLUMNS)

    for family_name, features in ESTIMATES_FEATURE_FAMILIES.items():
        removed = {f"{feature}_rank" for feature in features}

        assert set(FEATURE_SETS[f"estimates_ex_{family_name}"]) == full - removed


def test_estimates_revisions_only_is_compustat_plus_six_revision_columns():
    revisions = [
        f"{feature}_rank" for feature in ESTIMATES_INFORMATION_TYPES["revisions"]
    ]

    assert len(revisions) == 6
    assert FEATURE_SETS["estimates_revisions_only"] == [
        *COMPUSTAT_FEATURE_COLUMNS,
        *revisions,
    ]
    assert FEATURE_SETS["estimates_revisions_pure"] == revisions


def test_estimates_leave_one_information_type_sets_remove_only_that_type():
    full = set(ESTIMATES_FEATURE_COLUMNS)

    for type_name, features in ESTIMATES_INFORMATION_TYPES.items():
        removed = {f"{feature}_rank" for feature in features}

        assert set(FEATURE_SETS[f"estimates_ex_{type_name}"]) == full - removed


def test_both_decompositions_partition_the_modelled_analyst_features():
    for groups in (ESTIMATES_FEATURE_FAMILIES, ESTIMATES_INFORMATION_TYPES):
        assigned = [feature for features in groups.values() for feature in features]

        assert sorted(assigned) == sorted(ESTIMATES_MODEL_FEATURES)
        assert len(assigned) == len(set(assigned))


def test_partition_guard_rejects_an_incomplete_decomposition():
    from estimates_features import _assert_partitions_model_features

    incomplete = {"eps": ["est_eps_yield"]}

    with pytest.raises(ValueError, match="must partition"):
        _assert_partitions_model_features("ESTIMATES_FEATURE_FAMILIES", incomplete)
