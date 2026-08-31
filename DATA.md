# Data Policy

Raw data, processed panels and experiment outputs are intentionally not tracked
in Git. Most files come from licensed Refinitiv/LSEG and Compustat access, so
they should stay local.

Expected local directories:

```text
data/raw/asset_pricing/refinitiv_exports/
data/raw/asset_pricing/refinitiv_us_exports/
data/raw/asset_pricing/compustat_exports/
data/raw/asset_pricing/wrds_compustat_us_exports/
data/processed/asset_pricing/
results/asset_pricing_ml/
```

Key generated panels:

- `data/processed/asset_pricing/monthly_feature_panel.parquet`
- `data/processed/asset_pricing/monthly_feature_panel_compustat.parquet`
- `data/processed/asset_pricing/monthly_feature_panel_estimates.parquet`
- `data/processed/asset_pricing/monthly_feature_panel_us.parquet`
- `data/processed/asset_pricing/monthly_feature_panel_us_compustat.parquet`
- `data/processed/asset_pricing/compustat_enrichment_audit.json`
- `data/processed/asset_pricing/wrds_compustat_us_enrichment_audit.json`
- `data/processed/asset_pricing/compustat_feature_dictionary.csv`
- `data/processed/asset_pricing/wrds_compustat_us_feature_dictionary.csv`
- `data/processed/asset_pricing/refinitiv_estimates_enrichment_audit.json`
- `data/processed/asset_pricing/refinitiv_estimates_feature_dictionary.csv`

Key external inputs:

- Refinitiv/LSEG European equity universe, monthly prices, market caps, returns,
  volumes, fundamentals and liquidity fields.
- Refinitiv/LSEG US equity universe and USD monthly prices, market caps, returns,
  volumes and fundamentals for the US comparison.
- Refinitiv/LSEG analyst-estimates snapshots saved as
  `data/raw/asset_pricing/refinitiv_exports/refinitiv_analyst_estimates_monthly.csv.gz`.
- Compustat Global annual fundamentals and security monthly descriptors.
- WRDS Compustat North America annual fundamentals and monthly security
  descriptors saved as
  `data/raw/asset_pricing/wrds_compustat_us_exports/compustat_us_fundamentals_annual.csv.gz`
  and
  `data/raw/asset_pricing/wrds_compustat_us_exports/compustat_us_security_monthly.csv.gz`.
- Ken French Europe factors for external-factor robustness.
- EUR short-rate series for long-only excess-return calculations.
- Optional USD short-rate series for US long-only excess-return calculations.

If this repo was split from `jump-model-europe` on the same machine, you can
reuse existing artifacts without duplicating them by symlinking:

```bash
./scripts/link_legacy_artifacts.sh ../jump-model-europe
```

Use symlinks for local work only. Do not commit licensed data or large result
folders.
