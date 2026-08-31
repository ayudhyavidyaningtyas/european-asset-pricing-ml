# Reproduction command reference

Commands for the data builds and model runs. All of them expect the local data
layout described in `DATA.md`.

## European panels

Build the base Refinitiv/Datastream panel:

```bash
python scripts/build_asset_pricing_panel.py
```

Build the Compustat-enriched panel:

```bash
python scripts/build_compustat_enriched_panel.py
```

## Main European ML benchmark

```bash
python scripts/run_asset_pricing_ml.py \
  --panel data/processed/asset_pricing/monthly_feature_panel_compustat.parquet \
  --feature-set compustat_enriched \
  --models momentum ridge elastic_net hist_gbm mlp \
  --targets rank return \
  --first-test-year 2015 --last-test-year 2026 \
  --output-dir results/asset_pricing_ml/main_compustat_benchmark
```

## Analyst-estimates runs

Download a pilot analyst-estimates extract, then build the estimates-enriched
panel:

```bash
python scripts/refinitiv_estimates_downloader.py --pilot --start 2024-01 --end 2024-06
python scripts/build_estimates_enriched_panel.py
```

For the full download, run the downloader with the saved European RIC universe:

```bash
python scripts/refinitiv_estimates_downloader.py \
  --universe-csv data/raw/asset_pricing/refinitiv_exports/europe_equity_universe_rics_only.csv \
  --start 2005-01 --end 2026-06 --resume

python scripts/build_estimates_enriched_panel.py
```

Run the benchmark with analyst-estimates features added:

```bash
python scripts/run_asset_pricing_ml.py \
  --panel data/processed/asset_pricing/monthly_feature_panel_estimates.parquet \
  --feature-set estimates_enriched \
  --models momentum ridge elastic_net hist_gbm mlp \
  --targets rank return \
  --first-test-year 2015 --last-test-year 2026 \
  --output-dir results/asset_pricing_ml/main_estimates_benchmark
```

## US market comparison

The US runs mirror the European design. Data sources:

- Refinitiv/LSEG: US ordinary-equity RIC universe, monthly returns, market data
  and Refinitiv fundamentals for the base panel.
- WRDS Compustat North America: annual fundamentals and monthly security data
  for the Compustat enrichment.

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

Run the US benchmark with the same feature set and model family. Use
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

## Residualized target screen

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
