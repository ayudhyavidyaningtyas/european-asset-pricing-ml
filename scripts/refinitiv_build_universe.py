"""Build a European equity RIC universe through LSEG Workspace Screener.

Usage:
    export LSEG_APP_KEY="your-app-key"
    python scripts/refinitiv_build_universe.py

Output:
    data/raw/asset_pricing/refinitiv_exports/europe_equity_universe.csv

The script queries active and inactive public primary equities by headquarters
country. This is a pragmatic starting point for a survivorship-aware universe,
but you should still audit inactive/dead coverage against Workspace Screener.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    pd.set_option("future.no_silent_downcasting", True)
except KeyError:  # pragma: no cover - older pandas compatibility
    pass

try:
    import lseg.data as ld
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install first: python -m pip install lseg-data") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_exports"
OUT_FILE = OUT_DIR / "europe_equity_universe.csv"
MANIFEST = OUT_DIR / "europe_equity_universe_manifest.json"

COUNTRIES = {
    "AT": "Austria",
    "BE": "Belgium",
    "DK": "Denmark",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "IE": "Ireland",
    "IT": "Italy",
    "NL": "Netherlands",
    "NO": "Norway",
    "PT": "Portugal",
    "ES": "Spain",
    "SE": "Sweden",
    "CH": "Switzerland",
    "GB": "United Kingdom",
}

FIELDS = [
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


def screen_expression(country: str, state: str) -> str:
    return (
        f'SCREEN(U(IN(Equity({state},public,primary))/*UNV:Public*/), '
        f'IN(TR.HQCountryCode,"{country}"), CURN=USD)'
    )


def open_session(app_key: str):
    return ld.open_session(name="desktop.workspace", app_key=app_key)


def fetch_country_state(country: str, state: str) -> pd.DataFrame:
    expr = screen_expression(country, state)
    df = ld.get_data(universe=[expr], fields=FIELDS, header_type=ld.HeaderType.NAME)
    df["screen_country"] = country
    df["screen_state"] = state
    df["screen_expression"] = expr
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TR.RIC" in out.columns:
        out["ric"] = out["TR.RIC"]
    elif "Instrument" in out.columns:
        out["ric"] = out["Instrument"]
    else:
        raise ValueError(f"No RIC-like column returned. Columns: {list(out.columns)}")
    out["ric"] = out["ric"].astype(str).str.strip()
    out = out[out["ric"].ne("") & out["ric"].ne("nan")]
    return out.drop_duplicates(subset=["ric"]).sort_values(["screen_country", "ric"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", default=",".join(COUNTRIES), help="Comma-separated ISO country codes.")
    parser.add_argument("--active-only", action="store_true", help="Skip inactive equities.")
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    app_key = os.environ.get("LSEG_APP_KEY")
    if not app_key:
        raise SystemExit("Set your app key first: export LSEG_APP_KEY='...'")

    countries = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    states = ["active"] if args.active_only else ["active", "inactive"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "countries": countries,
        "states": states,
        "status": "started",
        "screens": [],
    }

    frames = []
    try:
        open_session(app_key)
        for country in countries:
            for state in states:
                entry = {"country": country, "state": state, "expression": screen_expression(country, state)}
                try:
                    df = fetch_country_state(country, state)
                    entry["rows_raw"] = int(df.shape[0])
                    frames.append(df)
                except Exception as exc:
                    entry["error_type"] = type(exc).__name__
                    entry["error"] = str(exc)
                    print(f"screen failed {country}/{state}: {type(exc).__name__}: {exc}", file=sys.stderr)
                manifest["screens"].append(entry)

        if not frames:
            raise RuntimeError("No screener rows returned.")
        universe = normalize(pd.concat(frames, ignore_index=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        universe.to_csv(args.output, index=False)
        # Downloader needs only a lowercase `ric` column; this file keeps metadata too.
        (args.output.parent / "europe_equity_universe_rics_only.csv").write_text(
            "ric\n" + "\n".join(universe["ric"].tolist()) + "\n"
        )
        manifest["status"] = "ok"
        manifest["rows_unique_ric"] = int(universe["ric"].nunique())
        manifest["output"] = str(args.output)
        manifest["rics_only_output"] = str(args.output.parent / "europe_equity_universe_rics_only.csv")
        return 0
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        print(f"universe build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            ld.close_session()
        except Exception:
            pass
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    raise SystemExit(main())
