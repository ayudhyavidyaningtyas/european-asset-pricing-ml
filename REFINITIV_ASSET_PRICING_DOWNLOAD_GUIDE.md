# Refinitiv Workspace Excel Download Guide for European ML Asset Pricing

This guide is for building a monthly European stock-level panel suitable for a
Gu-Kelly-Xiu-style or regime-conditional asset-pricing dissertation.

## Scope to Use

Start with this. It is large enough for a serious MSc project, but not so large that the
data build becomes the dissertation.

- Region: developed Europe
- Countries: Austria, Belgium, Denmark, Finland, France, Germany, Ireland, Italy,
  Netherlands, Norway, Portugal, Spain, Sweden, Switzerland, United Kingdom
- Security type: ordinary/common equities only
- Listing status: active plus inactive/dead/delisted if Workspace allows it
- Frequency: monthly for prices and market data; annual for accounting fundamentals
- Sample: 2000-01 to latest available
- Currency: local currency first; optionally also USD if matching Kenneth French Europe

Do a pilot first: 20-50 active stocks plus at least 5 inactive/dead stocks. Scale only
after the pilot files import cleanly.

## Folder and File Names

Save raw exports unchanged here:

```text
data/raw/asset_pricing/refinitiv_exports/
```

Create these files exactly:

| Order | File | Purpose |
|---:|---|---|
| 1 | `refinitiv_universe_master.csv` | One row per security; identifiers, country, sector, listing status |
| 2 | `refinitiv_prices_monthly.csv` | Monthly return/price panel |
| 3 | `refinitiv_market_data_monthly.csv` | Monthly size, shares, volume, liquidity |
| 4 | `refinitiv_fundamentals_annual.csv` | Annual Worldscope/accounting fundamentals |
| 5 | `refinitiv_fundamentals_quarterly.csv` | Optional later; do not start here |
| 6 | `refinitiv_analyst_estimates_monthly.csv.gz` | Optional analyst-estimates extension |

## File 1: Universe Master

Export one row per security.

Canonical columns to save:

```text
ric
isin
ds_code
perm_id
ticker
company_name
country
exchange
currency
security_type
listing_status
listing_date
delisting_date
trbc_economic_sector
trbc_business_sector
trbc_industry_group
trbc_industry
gics_sector
gics_industry
```

Workspace/Data Item Browser fields to search:

| Canonical column | Workspace/TR field to try | Search term in Data Item Browser |
|---|---|---|
| `ric` | `TR.RIC` | RIC |
| `isin` | `TR.ISIN` | ISIN |
| `ticker` | `TR.TickerSymbol` | Ticker Symbol |
| `company_name` | `TR.CommonName` | Common Name |
| `country` | `TR.ExchangeCountry` or country of primary listing | Exchange Country / Country |
| `exchange` | `TR.ExchangeName` | Exchange Name |
| `currency` | `TR.Currency` | Currency |
| `trbc_economic_sector` | `TR.TRBCEconomicSector` | TRBC Economic Sector |
| `trbc_business_sector` | `TR.TRBCBusinessSector` | TRBC Business Sector |
| `trbc_industry_group` | `TR.TRBCIndustryGroup` | TRBC Industry Group |
| `trbc_industry` | `TR.TRBCIndustry` | TRBC Industry |
| `gics_sector` | `TR.GICSSector` | GICS Sector |
| `gics_industry` | `TR.GICSIndustry` | GICS Industry |
| `listing_status` | use Screener/List export field if available | Status / Active / Inactive / Delisted |
| `listing_date` | use Screener/List export field if available | Listing Date |
| `delisting_date` | use Screener/List export field if available | Delisting Date |

If the exact listing-status fields are not available in Workspace Lite, preserve whatever
Workspace exports from Screener/List Manager, and name the resulting column clearly.

## File 2: Monthly Prices and Returns

Preferred route: Datastream Request Table, because `RI` is the clean total-return index.

Canonical columns:

```text
date
ric
ri
price_close
total_return_1m
currency
```

Datastream datatypes:

| Canonical column | Datastream datatype | Actual name |
|---|---|---|
| `ri` | `RI` | Total Return Index |
| `price_close` | `P` | Price, adjusted |
| `total_return_1m` | calculate from `RI` | `ri_t / ri_t-1 - 1` |

Workspace/TR fallback fields:

| Canonical column | Workspace/TR field to try | Parameters |
|---|---|---|
| `date` | `TR.PriceClose.date` | `SDate=2000-01-01 EDate=<latest> Frq=M` |
| `price_close` | `TR.PriceClose` | `SDate=2000-01-01 EDate=<latest> Frq=M` |
| `total_return_1m` | `TR.TotalReturn1Mo` | `SDate=2000-01-01 EDate=<latest> Frq=M` |

If Datastream gives a wide file, that is fine. Save it raw. The analysis code can reshape it.

## File 3: Monthly Market Data

Canonical columns:

```text
date
ric
market_cap
shares_outstanding
free_float_shares
volume
turnover_value
bid
ask
currency
```

Datastream datatypes:

| Canonical column | Datastream datatype | Actual name |
|---|---|---|
| `market_cap` | `MV` | Market Value |
| `shares_outstanding` | `NOSH` | Number of Shares |
| `free_float_shares` | `NOSHFF` | Free-float Number of Shares |
| `volume` | `VO` | Turnover by Volume |
| `turnover_value` | `VA` | Turnover by Value |
| `price_close` | `P` | Price, adjusted |

Workspace/TR fallback fields:

| Canonical column | Workspace/TR field to try | Search term |
|---|---|---|
| `market_cap` | `TR.CompanyMarketCap` | Company Market Cap |
| `shares_outstanding` | `TR.TtlCmnSharesOut` | Total Common Shares Outstanding |
| `volume` | `TR.Volume` | Volume |
| `price_close` | `TR.PriceClose` | Price Close |
| `bid` | use Data Item Browser result | Bid |
| `ask` | use Data Item Browser result | Ask |

Bid/ask fields are optional. If Workspace Lite does not return them, volume and turnover are
enough for a dissertation-level implementability proxy.

## File 4: Annual Fundamentals

Preferred route: Worldscope/Datastream datatypes. These names are stable and easier to audit.

Canonical columns:

```text
ric
fiscal_year
fiscal_period_end
report_date
document_currency
total_assets
common_equity
total_shareholders_equity
total_liabilities
total_debt
cash_short_term_investments
net_sales_revenue
operating_income
ebit
ebitda
net_income_before_extraordinary_items
net_income_available_to_common
cash_flow_operations
capital_expenditures
inventories
ppe_net
research_development
depreciation_amortization
income_taxes
preferred_stock
common_shares_outstanding
```

Worldscope/Datastream datatypes:

| Canonical column | Datastream / Worldscope code | Actual name |
|---|---|---|
| `document_currency` | `WC06099` | Currency of Document |
| `fiscal_period_end` | `WC05350` | Date of Fiscal Year End |
| `total_assets` | `WC02999` | Total Assets |
| `common_equity` | `WC03501` | Common Equity |
| `total_shareholders_equity` | `WC03995` | Total Shareholders Equity |
| `total_liabilities` | `WC03351` | Total Liabilities |
| `total_debt` | `WC03255` | Total Debt |
| `cash_short_term_investments` | `WC02001` | Cash & Short Term Investments |
| `net_sales_revenue` | `WC01001` | Net Sales or Revenues |
| `operating_income` | `WC01250` | Operating Income |
| `ebit` | `WC18191` | Earnings Before Interest and Taxes |
| `ebitda` | `WC18198` | Earnings Before Interest, Taxes & Depreciation |
| `net_income_before_extraordinary_items` | `WC01551` | Net Income Before Extra Items / Preferred Dividends |
| `net_income_available_to_common` | `WC01751` | Net Income Available to Common |
| `cash_flow_operations` | `WC04860` | Net Cash Flow Operating Activities |
| `capital_expenditures` | `WC04601` | Capital Expenditures |
| `inventories` | `WC02101` | Inventories Total |
| `ppe_net` | `WC02501` | Property, Plant and Equipment Net |
| `research_development` | `WC01201` | Research & Development |
| `depreciation_amortization` | `WC01151` | Depreciation, Depletion and Amortization |
| `income_taxes` | `WC01451` | Income Taxes |
| `preferred_stock` | `WC03451` | Preferred Stock |
| `common_shares_outstanding` | `WC05301` | Common Shares Outstanding |

Workspace/TR fallback fields to search:

| Concept | Workspace/TR field to try first | Search term |
|---|---|---|
| Total assets | `TR.F.TotAssets` | Total Assets |
| Total liabilities | `TR.F.TotLiab` | Total Liabilities |
| Total shareholders equity | `TR.F.TotShHoldEq` | Total Shareholders Equity |
| Total revenue | `TR.F.TotRevenue` | Total Revenue |
| Net income | `TR.F.IncBefDiscOpsExordItems` | Income Before Discontinued Operations / Extraordinary Items |
| Market cap | `TR.CompanyMarketCap` | Company Market Cap |

For the TR.F fields, use annual/fiscal-year parameters in Formula Builder. If Workspace asks for
period parameters, use `Period=FY0` for the latest fiscal year, then historical fiscal periods through
the Formula Builder or Request Table. Do not manually paste only the latest year.

## File 5: Optional Quarterly Fundamentals

Only download this after annual data works.

Canonical file name:

```text
refinitiv_fundamentals_quarterly.csv
```

Use the same fields as annual fundamentals, but quarterly/fiscal-quarter frequency. This is optional
because quarterly availability is less consistent across European firms and can waste time.

## File 6: Analyst Estimates Extension

This is the preferred next predictor expansion because it adds forward-looking
information without mechanically neutralizing the broad return structure that
currently helps the portfolios.

Canonical output file:

```text
refinitiv_analyst_estimates_monthly.csv.gz
```

Start with a small pilot through the Python downloader:

```bash
python scripts/refinitiv_estimates_downloader.py --pilot --start 2024-01 --end 2024-06
python scripts/build_estimates_enriched_panel.py
```

Then scale to the full European RIC universe:

```bash
python scripts/refinitiv_estimates_downloader.py \
  --universe-csv data/raw/asset_pricing/refinitiv_exports/europe_equity_universe_rics_only.csv \
  --start 2005-01 --end 2026-06 --resume

python scripts/build_estimates_enriched_panel.py
```

Fields to try first:

| Concept | Workspace/TR field to try | Search term |
|---|---|---|
| RIC | `TR.RIC` | RIC |
| ISIN | `TR.ISIN` | ISIN |
| EPS consensus mean | `TR.EPSMean` | EPS Mean |
| EPS consensus high/low | `TR.EPSHigh`, `TR.EPSLow` | EPS High / EPS Low |
| EPS estimate dispersion | `TR.EPSStdDev` | EPS Standard Deviation |
| EPS analyst count | `TR.EPSNumEstimates` | EPS Number of Estimates |
| Revenue consensus mean | `TR.RevenueMean` | Revenue Mean |
| Revenue high/low | `TR.RevenueHigh`, `TR.RevenueLow` | Revenue High / Revenue Low |
| Revenue dispersion | `TR.RevenueStdDev` | Revenue Standard Deviation |
| Revenue analyst count | `TR.RevenueNumEstimates` | Revenue Number of Estimates |
| Price-target mean | `TR.PriceTargetMean` | Price Target Mean |
| Price-target high/low | `TR.PriceTargetHigh`, `TR.PriceTargetLow` | Price Target High / Low |
| Price-target dispersion | `TR.PriceTargetStdDev` | Price Target Standard Deviation |
| Price-target analyst count | `TR.PriceTargetNumEstimates` | Price Target Number of Estimates |
| Recommendation mean | `TR.RecommendationMean` | Recommendation Mean |
| Recommendation analyst count | `TR.RecommendationNumEstimates` | Recommendation Number of Estimates |

Use monthly snapshots with `Period=FY1`, `Frq=FY`, and the same currency as the
price panel, currently `Curn=EUR` in the downloader. If any mnemonic fails under
your entitlement, use Workspace Data Item Browser to find the equivalent and
rerun with repeated `--field` overrides.

## Excel Add-in Workflow

### A. If You Have Datastream Menu

Use this route first.

1. Excel -> Workspace tab -> Datastream -> New Request Table or Datastream Formula.
2. Build the universe/list in Navigator or paste RICs from `refinitiv_universe_master.csv`.
3. For prices, set:
   - Datatypes: `RI,P`
   - Frequency: Monthly
   - Start: `2000-01-01`
   - End: latest available
   - Currency: local currency first
4. For market data, set:
   - Datatypes: `MV,NOSH,NOSHFF,VO,VA`
   - Frequency: Monthly
5. For fundamentals, set:
   - Datatypes:
     `WC06099,WC05350,WC02999,WC03501,WC03995,WC03351,WC03255,WC02001,WC01001,WC01250,WC18191,WC18198,WC01551,WC01751,WC04860,WC04601,WC02101,WC02501,WC01201,WC01151,WC01451,WC03451,WC05301`
   - Frequency: Annual / Fiscal year
6. Save each raw output as CSV/XLSX using the filenames above.

### B. If You Only Have Workspace Formula Builder

Use Build Formula / Data Item Browser.

1. Put the pilot RICs in column A.
2. Search each field in Data Item Browser using the names above.
3. Insert the formula for a small pilot first.
4. For time series, use monthly frequency and start/end parameters.
5. Export the resulting sheet unchanged.

Example field set for prices:

```text
TR.PriceClose.date
TR.PriceClose
TR.TotalReturn1Mo
TR.Volume
TR.CompanyMarketCap
TR.TtlCmnSharesOut
```

Example field set for reference/classification:

```text
TR.RIC
TR.ISIN
TR.TickerSymbol
TR.CommonName
TR.ExchangeName
TR.ExchangeCountry
TR.Currency
TR.TRBCEconomicSector
TR.TRBCBusinessSector
TR.TRBCIndustryGroup
TR.TRBCIndustry
TR.GICSSector
TR.GICSIndustry
```

## Minimum Viable Dissertation Panel

If time is short, this is enough:

| File | Must-have fields |
|---|---|
| `refinitiv_universe_master.csv` | RIC, ISIN, company name, country, exchange, currency, status, sector |
| `refinitiv_prices_monthly.csv` | RIC, date, total return index or 1-month total return, price |
| `refinitiv_market_data_monthly.csv` | RIC, date, market cap, volume |
| `refinitiv_fundamentals_annual.csv` | RIC, fiscal year, total assets, common equity, revenue, operating income, net income, capex |

This supports size, value, profitability, investment, momentum, reversal, volatility and liquidity.

## Implementable-Frontier Liquidity Pull

The final cost-aware stage uses a restricted universe containing every
security that can enter the monthly top-500 optimization. Generate or retain:

```text
data/raw/asset_pricing/refinitiv_exports/implementable_frontier_universe.csv
```

Then run:

```bash
export LSEG_APP_KEY="your-full-workspace-app-key"

python scripts/refinitiv_liquidity_downloader.py \
  --universe-csv \
  data/raw/asset_pricing/refinitiv_exports/implementable_frontier_universe.csv \
  --resume
```

The script probes and downloads monthly `BID` and `ASK` from 2013 onward.
Existing EUR panel data supply `TR.PriceClose`, `TR.Volume`,
`TR.CompanyMarketCap` and trailing turnover, so these are not downloaded
again. Full-period batches keep each request below the intended interday
datapoint size while avoiding slow year-by-year requests.

Expected outputs:

```text
supplemental/liquidity_monthly_full_period/
supplemental/liquidity_monthly_full_period_batch_index.csv
supplemental/refinitiv_liquidity_full_period_manifest.json
```

The app key may be longer than examples shown in older Workspace material.
Always export the complete value provided by Workspace; do not truncate it.

## Red Flags

Do not proceed to full modelling until these are solved:

- only active/current constituents are available;
- no inactive/dead securities;
- fundamentals have no fiscal dates;
- price data lacks either total return index or dividend-adjusted return;
- identifiers cannot link price and fundamentals reliably;
- currencies are mixed but not labelled.

If those problems cannot be solved in Workspace Lite, use the French factor/portfolio route instead:
regime-conditional European factor timing.
