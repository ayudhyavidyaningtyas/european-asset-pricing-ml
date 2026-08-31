"""Download the European equity panel from Refinitiv/LSEG into canonical CSVs.

This mirrors REFINITIV_ASSET_PRICING_DOWNLOAD_GUIDE.md but pulls the data over the
Data Library API instead of the Excel add-in. It writes four files into
data/raw/asset_pricing/refinitiv/:

    refinitiv_universe_master.csv       one row per security (identifiers, sector, ...)
    refinitiv_prices_monthly.csv        monthly total return / price panel
    refinitiv_market_data_monthly.csv   monthly market cap, shares, volume
    refinitiv_fundamentals_annual.csv   annual accounting fundamentals

Setup
-----
    pip install refinitiv-data pandas
    cp refinitiv-data.config.json.template refinitiv-data.config.json   # fill in app-key
    # (Desktop mode) open + log into Refinitiv Workspace

Usage
-----
Run from the project root so the config file is found:

    # Pilot: put a handful of RICs in a text file, one per line
    python scripts/download_refinitiv.py --rics-file pilot_rics.txt --start 2000-01-01

    # Or pass RICs inline
    python scripts/download_refinitiv.py --rics VOD.L SAP.DE MC.PA --start 2015-01-01

    # Skip parts you already have
    python scripts/download_refinitiv.py --rics-file rics.txt --only universe prices

Start with a pilot (20-50 active + a few dead RICs). Only scale up once the pilot
files import cleanly.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas not installed. Run: pip install pandas")

try:
    import refinitiv.data as rd
except ImportError:
    sys.exit("refinitiv.data not installed. Run: pip install refinitiv-data")


# Output location, relative to the project root.
OUT_DIR = Path("data/raw/asset_pricing/refinitiv")

# --- Field maps (from REFINITIV_ASSET_PRICING_DOWNLOAD_GUIDE.md) --------------

UNIVERSE_FIELDS = [
    "TR.RIC",
    "TR.ISIN",
    "TR.TickerSymbol",
    "TR.CommonName",
    "TR.ExchangeName",
    "TR.ExchangeCountry",
    "TR.Currency",
    "TR.TRBCEconomicSector",
    "TR.TRBCBusinessSector",
    "TR.TRBCIndustryGroup",
    "TR.TRBCIndustry",
    "TR.GICSSector",
    "TR.GICSIndustry",
]

# Monthly time series. TR.<field>.date gives the observation date column.
PRICE_FIELDS = [
    "TR.PriceClose.date",
    "TR.PriceClose",
    "TR.TotalReturn1Mo",
]

MARKET_FIELDS = [
    "TR.CompanyMarketCap.date",
    "TR.CompanyMarketCap",
    "TR.TtlCmnSharesOut",
    "TR.Volume",
]

# Annual fundamentals via the TR.F fiscal-statement fields.
FUNDAMENTAL_FIELDS = [
    "TR.F.TotAssets.fperiod",
    "TR.F.TotAssets",
    "TR.F.TotLiab",
    "TR.F.TotShHoldEq",
    "TR.F.TotRevenue",
    "TR.F.OpProfit",
    "TR.F.IncBefDiscOpsExordItems",
    "TR.F.NetIncAfterMinIntr",
    "TR.F.CAPEX",
    "TR.CompanyMarketCap",
]


def read_rics(args: argparse.Namespace) -> list[str]:
    if args.rics:
        rics = list(args.rics)
    elif args.rics_file:
        path = Path(args.rics_file)
        if not path.exists():
            sys.exit(f"RICs file not found: {path}")
        rics = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    else:
        sys.exit("Provide --rics or --rics-file.")
    if not rics:
        sys.exit("No RICs to download.")
    print(f"{len(rics)} RIC(s) to download.")
    return rics


def save(df: pd.DataFrame, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    df.to_csv(out, index=False)
    print(f"  wrote {out}  ({len(df)} rows, {len(df.columns)} cols)")


def fetch_universe(rics: list[str]) -> None:
    print("Universe master...")
    df = rd.get_data(universe=rics, fields=UNIVERSE_FIELDS)
    save(df, "refinitiv_universe_master.csv")


def fetch_timeseries(rics: list[str], fields: list[str], start: str, end: str,
                     name: str) -> None:
    # Frq=M -> monthly. get_data with SDate/EDate parameters returns a long panel
    # with one row per security-date, which is what the analysis code expects.
    params = {"SDate": start, "EDate": end, "Frq": "M"}
    df = rd.get_data(universe=rics, fields=fields, parameters=params)
    save(df, name)


def fetch_fundamentals(rics: list[str], start_year: int, end_year: int) -> None:
    print("Annual fundamentals...")
    # Period=FY0..FY-n pulls a run of fiscal years. Adjust the span to your sample.
    n_years = end_year - start_year
    params = {
        "SDate": "FY0",
        "EDate": f"FY-{n_years}",
        "Frq": "FY",
        "Period": "FY0",
    }
    df = rd.get_data(universe=rics, fields=FUNDAMENTAL_FIELDS, parameters=params)
    save(df, "refinitiv_fundamentals_annual.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--rics", nargs="+", help="RICs inline, e.g. VOD.L SAP.DE")
    src.add_argument("--rics-file", help="Text file with one RIC per line")
    parser.add_argument("--start", default="2000-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--only", nargs="+",
                        choices=["universe", "prices", "market", "fundamentals"],
                        help="Download only these parts (default: all)")
    args = parser.parse_args()

    rics = read_rics(args)
    parts = args.only or ["universe", "prices", "market", "fundamentals"]

    print("Opening Refinitiv session...")
    rd.open_session()
    try:
        if "universe" in parts:
            fetch_universe(rics)
        if "prices" in parts:
            print("Monthly prices/returns...")
            fetch_timeseries(rics, PRICE_FIELDS, args.start, args.end,
                             "refinitiv_prices_monthly.csv")
        if "market" in parts:
            print("Monthly market data...")
            fetch_timeseries(rics, MARKET_FIELDS, args.start, args.end,
                             "refinitiv_market_data_monthly.csv")
        if "fundamentals" in parts:
            fetch_fundamentals(rics, int(args.start[:4]), int(args.end[:4]))
    finally:
        rd.close_session()
        print("Session closed.")

    print("Done. Raw exports are in", OUT_DIR)


if __name__ == "__main__":
    main()
