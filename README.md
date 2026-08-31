# European Asset Pricing with Machine Learning

This repository contains the dissertation code for European equity return
prediction, Kelly-style asset-pricing models, deep sequence models and
implementability tests.

Dissertation title:

**Deep Learning and the Tradability Gradient in European Equity Return Prediction**

This is the clean code repository for the dissertation. Licensed raw data,
processed panels, generated model outputs and submission documents are not
tracked in Git.

## Core Question

Can machine learning produce economically useful stock-selection signals in
European equities once liquidity, turnover, transaction costs and capacity are
made explicit?

The empirical story so far is not just "ML predicts returns." It is:

1. ML and deep models do contain out-of-sample information about next-month
   European stock returns.
2. More data depth, especially Compustat Global fundamentals and Refinitiv
   liquidity, improves the research design.
3. Advanced Kelly-style models are feasible: DRE, neural SDF, adversarial
   SDF, AIPM transformer, IPCA and conditional autoencoder.
4. Strong gross predictability often weakens after realistic portfolio
   construction, turnover, spread, impact and capacity diagnostics.
5. The dissertation contribution is the European
   **predictability-implementability gap**.

## Repository Layout

```text
src/       Research modules and model implementations
scripts/   Reproducible command-line runners
tests/     Focused unit/regression tests
data/      Ignored raw and processed data location
results/   Ignored experiment output location
figures/   Ignored generated figure location
outputs/   Ignored presentation/report artifacts
```

Important documentation:

- `RESULTS_ASSET_PRICING.md` records the main experiment log and findings.
- `LIQUIDITY_MECHANISM_EXTENSION.md` records the implementability/liquidity work.
- `REFINITIV_ASSET_PRICING_DOWNLOAD_GUIDE.md` records Refinitiv/LSEG download steps.
- `DATA.md` describes expected local data files and what is intentionally not tracked.

## Main Experiment Families

- GKX-style characteristic ML baselines: ridge, elastic net, histogram GBM, MLP.
- Compustat data-depth extension.
- Didisheim-Kelly-Malamud deep regression ensemble.
- Kelly-style neural SDF and adversarial test-asset SDF.
- Kelly-Kuznetsov-Malamud-Xu AIPM transformer adaptation.
- Gu-Kelly-Xiu conditional autoencoder and IPCA comparison.
- Deep sequence modelling: LSTM, GRU and attention-LSTM.
- Refinitiv analyst estimates: EPS/revenue/price-target revisions, dispersion,
  coverage and recommendation signals.
- Turnover-aware smoothing and constrained long-only construction.
- Refinitiv spread/ADV/capacity diagnostics and broader top-1000/top-2000 liquidity
  coverage.
- Residualized European ML target screen.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For LSEG/Refinitiv downloads, set credentials in your shell; do not commit keys:

```bash
read -s LSEG_APP_KEY
export LSEG_APP_KEY
```

## Sanity Checks

```bash
pytest -q
```

For a focused check after changing the main ML runner:

```bash
pytest -q tests/test_asset_pricing_ml.py
python -m compileall -q src scripts
```

## Reproduction Entry Points

Build the base Refinitiv/Datastream panel:

```bash
python scripts/build_asset_pricing_panel.py
```

Build the Compustat-enriched panel:

```bash
python scripts/build_compustat_enriched_panel.py
```

## US Market Comparison

The US comparison mirrors the European design while keeping the data sources
explicit:

- Refinitiv/LSEG: US ordinary-equity RIC universe, monthly returns, market data
  and Refinitiv fundamentals used to build the base panel.
- WRDS Compustat North America: annual fundamentals and monthly security data
  used for the Compustat enrichment.

Build a US RIC universe, then download the Refinitiv panel in USD:

```bash
python scripts/refinitiv_build_us_universe.py

python scripts/refinitiv_python_downloader.py \
  --universe-csv data/raw/asset_pricing/refinitiv_us_exports/us_equity_universe_rics_only.csv \
  --output-dir data/raw/asset_pricing/refinitiv_us_exports \
  --base-currency USD \
  --start 2000-01-01 --end 2026-07-08 \
  --start-fy FY2000 --end-fy FY2025 \
  --resume
```

Build the US base panel, download WRDS Compustat US, then enrich the panel:

```bash
python scripts/build_us_asset_pricing_panel.py

python scripts/download_wrds_compustat_us.py \
  --start 2000-01-01 --end 2026-07-08

python scripts/build_us_compustat_enriched_panel.py
```

Run the US ML benchmark with the same feature set and model family. Use
`--no-risk-free` unless a comparable USD monthly cash-rate CSV has been supplied
through `--risk-free-rate`.

```bash
python scripts/run_asset_pricing_ml.py \
  --panel data/processed/asset_pricing/monthly_feature_panel_us_compustat.parquet \
  --feature-set compustat_enriched \
  --models momentum ridge elastic_net hist_gbm mlp \
  --targets rank return \
  --first-test-year 2015 --last-test-year 2026 \
  --no-risk-free \
  --output-dir results/asset_pricing_ml/us_compustat_benchmark
```

After the European and US benchmarks exist, build the side-by-side comparison:

```bash
python scripts/run_market_comparison.py \
  --europe-output-dir results/asset_pricing_ml/main_compustat_benchmark \
  --us-output-dir results/asset_pricing_ml/us_compustat_benchmark \
  --output-dir results/asset_pricing_ml/market_comparison
```

Download a pilot analyst-estimates extract, then build the estimates-enriched
panel:

```bash
python scripts/refinitiv_estimates_downloader.py --pilot --start 2024-01 --end 2024-06
python scripts/build_estimates_enriched_panel.py
```

For the full analyst-estimates experiment, run the downloader with the saved
European RIC universe:

```bash
python scripts/refinitiv_estimates_downloader.py \
  --universe-csv data/raw/asset_pricing/refinitiv_exports/europe_equity_universe_rics_only.csv \
  --start 2005-01 --end 2026-06 --resume

python scripts/build_estimates_enriched_panel.py
```

Run the main ML benchmark:

```bash
python scripts/run_asset_pricing_ml.py \
  --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
  --feature-set compustat_enriched \
  --models momentum ridge elastic_net hist_gbm mlp \
  --targets rank return \
  --first-test-year 2015 --last-test-year 2026 \
  --output-dir results/asset_pricing_ml/main_compustat_benchmark
```

Run the same benchmark with analyst-estimates features added:

```bash
python scripts/run_asset_pricing_ml.py \
  --panel data/processed/asset_pricing/monthly_feature_panel_estimates.parquet \
  --feature-set estimates_enriched \
  --models momentum ridge elastic_net hist_gbm mlp \
  --targets rank return \
  --first-test-year 2015 --last-test-year 2026 \
  --output-dir results/asset_pricing_ml/main_estimates_benchmark
```

Run the residualized European target screen:

```bash
python scripts/run_asset_pricing_ml.py \
  --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
  --feature-set compustat_enriched \
  --models momentum ridge hist_gbm \
  --targets rank residual_rank \
  --first-test-year 2015 --last-test-year 2026 \
  --max-training-rows 250000 \
  --no-tuning --placebo-repetitions 0 \
  --skip-delisting-scenarios --skip-importance \
  --output-dir results/asset_pricing_ml/residual_target_compustat_screen
```

See `RESULTS_ASSET_PRICING.md` for the full run list.
