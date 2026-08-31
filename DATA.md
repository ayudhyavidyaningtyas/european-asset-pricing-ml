# Data Policy

Licensed raw data, processed panels and generated model outputs are not
uploaded in Git due to license. Most files come from Refinitiv/LSEG and
Compustat subscriptions, so they stay local.

Expected local directories:

```text
data/raw/asset_pricing/refinitiv_exports/
data/raw/asset_pricing/refinitiv_us_exports/
data/raw/asset_pricing/compustat_exports/
data/raw/asset_pricing/wrds_compustat_us_exports/
data/processed/asset_pricing/
results/asset_pricing_ml/
```

Generated panels:

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

External inputs:

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
