#!/usr/bin/env bash
# Identification tests for the analyst-estimates layer.
#
# Five blocks, all on the strict lag-1 (or lag-L) enriched panels with the
# estimates-covered matched sample:
#   1. feature-family / information-type ablation on the cleaned 11-feature set
#   2. analyst-signal lag ladder (1, 2, 3, 6 months)
#   3. coverage-selection test (analysis only; reuses the Test B refresh runs)
#   4. missingness negative control (coverage counts only)
#   5. combined four-panel identification-evidence exhibit
#
# Every model run keeps the same sample filters as the Test B refresh cells --
# including --skip-delisting-scenarios, which those cells used and which would
# otherwise add 32 unlabelled delisting stock-months and break the match.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="${STAMP:-20260816}"
RESULTS="results/asset_pricing_ml"
PANEL_ROOT="data/processed/asset_pricing"
LAG1_PANEL="${PANEL_ROOT}/strict_estimates_lag1/monthly_feature_panel_estimates_strict_lag1.parquet"
ABLATION_PREFIX="estimates_ablation_refresh_${STAMP}_"
LADDER_PREFIX="estimates_lag_ladder_${STAMP}_"
BASELINE_DIR="${RESULTS}/test_b_datadepth_compustat_enriched_refresh_${STAMP}"
FULL_DIR="${RESULTS}/test_b_datadepth_estimates_enriched_refresh_${STAMP}"

ABLATION_VARIANTS=(
  eps_only revenue_only price_target_only
  levels_only revisions_only dispersion_only
  ex_eps ex_revenue ex_price_target
  ex_levels ex_revisions ex_dispersion
)
LADDER_LAGS=(1 2 3 6)

run_cell() {
  local panel="$1" feature_set="$2" output_dir="$3"
  shift 3
  if [[ -f "${output_dir}/ml_manifest.json" ]]; then
    echo "skip (exists): ${output_dir}"
    return 0
  fi
  echo "=== ${output_dir}"
  python scripts/run_asset_pricing_ml.py \
    --panel "${panel}" \
    --output-dir "${output_dir}" \
    --feature-set "${feature_set}" \
    --targets rank \
    --require-estimates-feature \
    --require-estimate-signal-lag-months "${LAG_GUARD}" \
    --skip-importance \
    --skip-delisting-scenarios \
    --models "$@"
}

# ---------------------------------------------------------------- 1. ablation
LAG_GUARD=1
for variant in "${ABLATION_VARIANTS[@]}"; do
  run_cell "${LAG1_PANEL}" "estimates_${variant}" \
    "${RESULTS}/${ABLATION_PREFIX}${variant}" ridge hist_gbm mlp
done

python scripts/run_estimates_ablation_paired_tests.py \
  --baseline-dir "${BASELINE_DIR}" \
  --full-dir "${FULL_DIR}" \
  --variant-prefix "${ABLATION_PREFIX}" \
  --output-dir "${RESULTS}/estimates_family_ablation_refresh_${STAMP}" \
  --models ridge_rank hist_gbm_rank mlp_rank \
  --require-matched-samples

# ------------------------------------------------------------- 2. lag ladder
for lag in "${LADDER_LAGS[@]}"; do
  LAG_GUARD="${lag}"
  panel="${PANEL_ROOT}/strict_estimates_lag${lag}/monthly_feature_panel_estimates_strict_lag${lag}.parquet"
  run_cell "${panel}" compustat_enriched \
    "${RESULTS}/${LADDER_PREFIX}compustat_lag${lag}" ridge hist_gbm mlp dre
  run_cell "${panel}" estimates_enriched \
    "${RESULTS}/${LADDER_PREFIX}estimates_lag${lag}" ridge hist_gbm mlp dre
done
LAG_GUARD=1

python scripts/run_estimates_lag_ladder.py \
  --lags "${LADDER_LAGS[@]}" \
  --compustat-template "${RESULTS}/${LADDER_PREFIX}compustat_lag{lag}" \
  --estimates-template "${RESULTS}/${LADDER_PREFIX}estimates_lag{lag}" \
  --output-dir "${RESULTS}/estimates_lag_ladder_${STAMP}"

# ------------------------------------------------------ 3. coverage selection
python scripts/run_coverage_selection_test.py \
  --panel "${LAG1_PANEL}" \
  --compustat-dir "${BASELINE_DIR}" \
  --estimates-dir "${FULL_DIR}" \
  --output-dir "${RESULTS}/estimates_coverage_selection_${STAMP}"

# -------------------------------------- 4. missingness negative control
# The coverage-only feature set needs estimates_feature_count_rank, which the
# original strict lag-1 panel predates. The _missrank rebuild is byte-identical
# on all shared columns and adds only that rank column.
NEGCONTROL_PANEL="${PANEL_ROOT}/strict_estimates_lag1_missrank/monthly_feature_panel_estimates_strict_lag1.parquet"
NEGCONTROL_DIR="${RESULTS}/estimates_negcontrol_coverage_only_${STAMP}"
if [[ ! -f "${NEGCONTROL_PANEL}" ]]; then
  python scripts/build_estimates_enriched_panel.py \
    --output-dir "${PANEL_ROOT}/strict_estimates_lag1_missrank" \
    --panel-filename monthly_feature_panel_estimates_strict_lag1.parquet \
    --strict-identifier-match \
    --estimate-signal-lag-months 1 \
    --filter-extreme-estimates
fi
run_cell "${NEGCONTROL_PANEL}" estimates_coverage_only \
  "${NEGCONTROL_DIR}" ridge hist_gbm mlp
python scripts/run_coverage_missingness_control.py \
  --compustat-dir "${BASELINE_DIR}" \
  --coverage-dir "${NEGCONTROL_DIR}" \
  --estimates-dir "${FULL_DIR}" \
  --output-dir "${RESULTS}/estimates_missingness_control_${STAMP}"

# ---------------------------------------------------- 5. combined exhibit
python scripts/build_identification_evidence_exhibit.py \
  --coverage-dir "${RESULTS}/estimates_coverage_selection_${STAMP}" \
  --ladder-dir "${RESULTS}/estimates_lag_ladder_${STAMP}" \
  --ablation-dir "${RESULTS}/estimates_family_ablation_refresh_${STAMP}" \
  --output-dir "${RESULTS}/estimates_identification_evidence_${STAMP}"

echo "identification tests complete"
