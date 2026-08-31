#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${LSEG_APP_KEY:-}" ]]; then
  echo "Set LSEG_APP_KEY in this terminal before running this script." >&2
  exit 2
fi

BOOTSTRAP_REPETITIONS="${BOOTSTRAP_REPETITIONS:-2000}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-100}"

LIQUIDITY_DIR="data/raw/asset_pricing/refinitiv_exports/supplemental/liquidity_monthly_full_period_top2000"

python scripts/build_broader_liquidity_universe.py

python scripts/refinitiv_liquidity_downloader.py \
  --universe-csv data/raw/asset_pricing/refinitiv_exports/implementable_frontier_universe_top2000.csv \
  --output-dir "${LIQUIDITY_DIR}" \
  --workers "${WORKERS}" \
  --batch-size "${BATCH_SIZE}" \
  --resume

for MAX_ASSETS in 1000 2000; do
  python scripts/run_frozen_deep_hybrid_robustness.py \
    --maximum-assets "${MAX_ASSETS}" \
    --liquidity "${LIQUIDITY_DIR}" \
    --output-dir "results/asset_pricing_ml/frozen_deep_hybrid_long_only_robustness_top${MAX_ASSETS}" \
    --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"

  python scripts/run_constrained_deep_hybrid_long_only.py \
    --maximum-assets "${MAX_ASSETS}" \
    --liquidity "${LIQUIDITY_DIR}" \
    --output-dir "results/asset_pricing_ml/constrained_deep_hybrid_long_only_top${MAX_ASSETS}" \
    --bootstrap-repetitions "${BOOTSTRAP_REPETITIONS}"
done
