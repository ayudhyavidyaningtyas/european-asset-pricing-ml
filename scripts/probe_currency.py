"""Probe the reporting/price currency of the equity universe (read-only, no re-pull).

Answers the open data-integrity question: are TR.TotalReturn1Mo returns in a common
numeraire, or in each stock's local listing currency? Pulls the currency *metadata*
fields (cheap `get_data`, not history) and reports the distribution.

Run (LSEG Workspace must be open and logged in):

    export LSEG_APP_KEY=your_desktop_app_key
    python scripts/probe_currency.py

Writes data/processed/asset_pricing/currency_probe.csv and prints a summary.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

try:
    pd.set_option("future.no_silent_downcasting", True)
except KeyError:  # pragma: no cover - older pandas compatibility
    pass

warnings.simplefilter("ignore", FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = (
    PROJECT_ROOT
    / "data/raw/asset_pricing/refinitiv_exports/europe_equity_universe_rics_only.csv"
)
SECURITY_MASTER = (
    PROJECT_ROOT
    / "data/raw/asset_pricing/refinitiv_exports/supplemental/refinitiv_security_master_supplement.parquet"
)
OUT_CSV = PROJECT_ROOT / "data/processed/asset_pricing/currency_probe.csv"

# The correct currency metadata fields. TR.Currency is NOT resolvable — do not use it.
CURRENCY_FIELDS = ["TR.PriceMoPriceCurrency", "TR.CompanyReportCurrency"]
BATCH_SIZE = 50


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_universe() -> list[str]:
    rics = pd.read_csv(UNIVERSE_CSV)["ric"].dropna().astype(str).str.strip()
    return rics[rics != ""].drop_duplicates().tolist()


def main() -> int:
    app_key = os.environ.get("LSEG_APP_KEY")
    if not app_key:
        print("ERROR: set LSEG_APP_KEY and make sure Workspace is running.", file=sys.stderr)
        return 1

    import lseg.data as ld

    try:
        ld.open_session(name="desktop.workspace", app_key=app_key)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not open Workspace session: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    try:
        # Fail fast: verify the session actually works before looping the whole universe.
        try:
            ld.get_data(["VOD.L"], ["TR.PriceMoPriceCurrency"], header_type=ld.HeaderType.NAME)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: session not usable ({type(exc).__name__}: {exc}).", file=sys.stderr)
            print("Most likely LSEG_APP_KEY is not your real key, or Workspace is not logged in.", file=sys.stderr)
            print("Get the key from Workspace -> search 'App Key Generator' (a ~24-char hex string).", file=sys.stderr)
            return 1
        universe = load_universe()
        print(f"universe: {len(universe)} RICs; pulling {CURRENCY_FIELDS} ...", flush=True)

        frames = []
        for batch_no, batch in enumerate(chunks(universe, BATCH_SIZE), start=1):
            for attempt in range(3):
                try:
                    df = ld.get_data(batch, CURRENCY_FIELDS, header_type=ld.HeaderType.NAME)
                    frames.append(df)
                    break
                except Exception as exc:  # noqa: BLE001 - tolerate transient LSEG errors
                    if attempt == 2:
                        print(f"  batch {batch_no} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                    else:
                        time.sleep(1.5)
            if batch_no % 20 == 0:
                print(f"  {batch_no} batches done", flush=True)

        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    finally:
        ld.close_session()

    if data.empty:
        print("No data returned.", file=sys.stderr)
        return 2

    # Normalise column names (HeaderType.NAME returns upper-cased field names).
    by_upper = {str(c).upper(): c for c in data.columns}
    inst = by_upper.get("INSTRUMENT", data.columns[0])
    price_ccy = by_upper.get("TR.PRICEMOPRICECURRENCY")
    report_ccy = by_upper.get("TR.COMPANYREPORTCURRENCY")

    data = data.rename(
        columns={inst: "ric", price_ccy: "price_currency", report_ccy: "report_currency"}
    )

    # Join exchange country for a country x currency view.
    if SECURITY_MASTER.exists():
        ref = pd.read_parquet(SECURITY_MASTER, columns=["TR.RIC", "TR.EXCHANGECOUNTRY"]).rename(
            columns={"TR.RIC": "ric", "TR.EXCHANGECOUNTRY": "exchange_country"}
        )
        data = data.merge(ref, on="ric", how="left")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_CSV, index=False)

    print("\n=== PRICE / RETURN CURRENCY (TR.PriceMoPriceCurrency) ===")
    if "price_currency" in data:
        vc = data["price_currency"].fillna("<missing>").value_counts()
        print(vc.to_string())
        n_distinct = data["price_currency"].dropna().nunique()
        print(f"\ndistinct price currencies: {n_distinct}  ->  "
              + ("MIXED: returns are in local currency, re-pull with Curn=EUR/USD needed."
                 if n_distinct > 1 else "single currency: returns already common, no re-pull needed."))

    print("\n=== REPORTING CURRENCY (TR.CompanyReportCurrency) ===")
    if "report_currency" in data:
        print(data["report_currency"].fillna("<missing>").value_counts().head(15).to_string())

    if "exchange_country" in data and "price_currency" in data:
        print("\n=== price currency x exchange country (top 15) ===")
        ct = data.groupby(["exchange_country", "price_currency"]).size().sort_values(ascending=False)
        print(ct.head(15).to_string())

    print(f"\nwritten -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
