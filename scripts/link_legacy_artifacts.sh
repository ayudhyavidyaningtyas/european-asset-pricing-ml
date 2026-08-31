#!/usr/bin/env bash
set -euo pipefail

LEGACY_ROOT="${1:-../jump-model-europe}"

if [[ ! -d "${LEGACY_ROOT}" ]]; then
  echo "Legacy project root not found: ${LEGACY_ROOT}" >&2
  exit 1
fi

LEGACY_ROOT="$(cd "${LEGACY_ROOT}" && pwd)"

mkdir -p data

rm -rf data/raw data/processed
ln -s "${LEGACY_ROOT}/data/raw" data/raw
ln -s "${LEGACY_ROOT}/data/processed" data/processed

if [[ -L results ]]; then
  rm results
fi
mkdir -p results
touch results/.gitkeep
if [[ -e results/asset_pricing_ml || -L results/asset_pricing_ml ]]; then
  rm -rf results/asset_pricing_ml
fi
if [[ -d "${LEGACY_ROOT}/results/asset_pricing_ml" ]]; then
  ln -s "${LEGACY_ROOT}/results/asset_pricing_ml" results/asset_pricing_ml
fi

echo "Linked data/raw, data/processed and results/asset_pricing_ml to ${LEGACY_ROOT}"
