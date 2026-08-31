from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_identification_evidence_exhibit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_identification_evidence_exhibit", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_inputs(root: Path, *, matched: bool = True) -> dict[str, Path]:
    coverage_dir = root / "coverage"
    ladder_dir = root / "ladder"
    ablation_dir = root / "ablation"
    for directory in (coverage_dir, ladder_dir, ablation_dir):
        directory.mkdir()

    pd.DataFrame(
        {
            "weighting": ["unweighted", "inverse_propensity"],
            "min_propensity_floor": [float("nan"), 0.01],
            "model": ["ridge", "ridge"],
            "months": [137, 137],
            "estimate": [0.0036, 0.0044],
            "standard_error": [0.0014, 0.0012],
            "t_stat": [2.53, 3.72],
            "p_value": [0.011, 0.0002],
            "p_value_holm": [0.034, 0.0006],
        }
    ).to_csv(coverage_dir / "coverage_selection_data_depth_tests.csv", index=False)

    pd.DataFrame(
        {
            "sample_scope": ["common_across_lags", "common_across_lags", "own_matched_sample", "own_matched_sample"],
            "lag_months": [1, 2, 1, 2],
            "model": ["ridge"] * 4,
            "stock_months": [1000, 1000, 1200, 1100],
            "months": [137] * 4,
            "estimate": [0.004, 0.002, 0.0042, 0.0021],
            "standard_error": [0.001] * 4,
            "t_stat": [4.0, 2.0, 4.2, 2.1],
            "p_value": [0.0001, 0.04, 0.0001, 0.04],
            "p_value_holm": [0.0002, 0.04, 0.0002, 0.04],
            "share_of_shortest_lag": [1.0, 0.5, 1.0, 0.5],
        }
    ).to_csv(ladder_dir / "lag_ladder_data_depth.csv", index=False)

    pd.DataFrame(
        {
            "test": ["variant_minus_compustat", "full_minus_leave_one_out"],
            "variant": ["estimates_revisions_only", "estimates_enriched"],
            "reference": ["compustat_enriched", "estimates_ex_revisions"],
            "variant_stock_months": [1000, 1000],
            "reference_stock_months": [1000, 1000],
            "shared_stock_months": [1000, 1000],
            "samples_identical": [matched, matched],
        }
    ).to_csv(ablation_dir / "ablation_sample_checks.csv", index=False)

    pd.DataFrame(
        {
            "test": [
                "monthly_ic_variant_minus_compustat",
                "monthly_ic_variant_minus_compustat",
                "monthly_ic_variant_minus_compustat",
                "monthly_ic_full_minus_leave_one_out",
            ],
            "variant": [
                "estimates_revisions_only",
                "estimates_ex_revisions",
                "estimates_enriched",
                "estimates_enriched",
            ],
            "reference": [
                "compustat_enriched",
                "compustat_enriched",
                "compustat_enriched",
                "estimates_ex_revisions",
            ],
            "model": ["ridge_rank"] * 4,
            "months": [137] * 4,
            "mean_ic_variant": [0.074, 0.072, 0.0745, 0.075],
            "mean_ic_reference": [0.071, 0.071, 0.071, 0.073],
            "delta_mean_ic": [0.003, 0.001, 0.0035, 0.002],
            "hac_t_stat": [2.8, 1.0, 2.6, 2.1],
            "hac_p_two_sided": [0.005, 0.3, 0.01, 0.036],
            "hac_p_holm": [0.02, 0.9, 0.03, 0.09],
        }
    ).to_csv(ablation_dir / "ablation_paired_ic_tests.csv", index=False)

    return {
        "coverage": coverage_dir,
        "ladder": ladder_dir,
        "ablation": ablation_dir,
    }


def _run(root: Path, dirs: dict[str, Path]) -> Path:
    output_dir = root / "out"
    argv = [
        "build_identification_evidence_exhibit.py",
        "--coverage-dir",
        str(dirs["coverage"]),
        "--ladder-dir",
        str(dirs["ladder"]),
        "--ablation-dir",
        str(dirs["ablation"]),
        "--output-dir",
        str(output_dir),
    ]
    original = sys.argv
    sys.argv = argv
    try:
        assert MODULE.main() == 0
    finally:
        sys.argv = original
    return output_dir


def test_exhibit_combines_all_four_panels(tmp_path):
    dirs = _write_inputs(tmp_path)

    output_dir = _run(tmp_path, dirs)

    combined = pd.read_csv(output_dir / "identification_evidence_panels.csv")
    assert set(combined["panel"]) == {
        "A_coverage_selection",
        "B_lag_decay_common_sample",
        "C_standalone_attribution",
        "D_marginal_attribution",
    }
    markdown = (output_dir / "identification_evidence.md").read_text()
    assert "Panel A" in markdown and "Panel D" in markdown
    assert "not interchangeable" in markdown


def test_lag_panel_marks_own_sample_as_robustness_columns(tmp_path):
    dirs = _write_inputs(tmp_path)

    lag = MODULE.lag_panel(dirs["ladder"])

    assert lag["panel"].eq("B_lag_decay_common_sample").all()
    # Primary estimates come from the common sample; own-sample sits beside.
    assert lag.set_index("lag_months").loc[1, "estimate"] == pytest.approx(0.004)
    assert lag.set_index("lag_months").loc[1, "own_sample_estimate"] == pytest.approx(
        0.0042
    )


def test_attribution_panels_split_standalone_from_marginal(tmp_path):
    dirs = _write_inputs(tmp_path)

    standalone, marginal = MODULE.attribution_panels(dirs["ablation"])

    # Panel C keeps the X_only cells and the full layer as reference, and
    # excludes the ex_X-vs-Compustat rows, which are Panel D territory.
    assert sorted(standalone["group"]) == ["all_11_features", "revisions"]
    assert "estimates_ex_revisions" not in standalone["variant"].tolist()
    assert marginal["group"].tolist() == ["revisions"]
    assert marginal["variant"].iloc[0] == "estimates_enriched"


def test_exhibit_refuses_unmatched_ablation_samples(tmp_path):
    dirs = _write_inputs(tmp_path, matched=False)

    with pytest.raises(SystemExit, match="non-identical stock-months"):
        MODULE.attribution_panels(dirs["ablation"])
