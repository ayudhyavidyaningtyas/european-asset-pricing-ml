"""Build a US ordinary-equity RIC universe through LSEG Workspace Screener."""
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
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from us_market import cusip_to_us_isin  # noqa: E402


OUT_DIR = PROJECT_ROOT / "data" / "raw" / "asset_pricing" / "refinitiv_us_exports"
OUT_FILE = OUT_DIR / "us_equity_universe.csv"
MANIFEST = OUT_DIR / "us_equity_universe_manifest.json"

FIELDS = [
    "TR.RIC",
    "TR.ISIN",
    "TR.CUSIP",
    "TR.TickerSymbol",
    "TR.CommonName",
    "TR.ExchangeName",
    "TR.ExchangeCountry",
    "TR.PriceMoPriceCurrency",
    "TR.CompanyReportCurrency",
    "TR.TRBCEconomicSector",
    "TR.TRBCBusinessSector",
    "TR.TRBCIndustryGroup",
    "TR.TRBCIndustry",
    "TR.GICSSector",
    "TR.GICSIndustry",
]


def screen_expression(state: str) -> str:
    return (
        f'SCREEN(U(IN(Equity({state},public,primary))/*UNV:Public*/), '
        'IN(TR.HQCountryCode,"US"), CURN=USD)'
    )


def open_session(app_key: str):
    return ld.open_session(name="desktop.workspace", app_key=app_key)


def fetch_state(state: str) -> pd.DataFrame:
    expr = screen_expression(state)
    df = ld.get_data(universe=[expr], fields=FIELDS, header_type=ld.HeaderType.NAME)
    df["screen_country"] = "US"
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
    if {"TR.ISIN", "TR.CUSIP"}.issubset(out.columns):
        existing_isin = (
            out["TR.ISIN"]
            .astype("string")
            .str.strip()
            .str.upper()
            .replace({"": pd.NA, "NAN": pd.NA, "<NA>": pd.NA})
        )
        derived_isin = out["TR.CUSIP"].map(cusip_to_us_isin)
        out["TR.ISIN"] = existing_isin.combine_first(derived_isin.astype("string"))
    return out.drop_duplicates(subset=["ric"]).sort_values(["screen_state", "ric"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-only", action="store_true", help="Skip inactive equities.")
    parser.add_argument("--output", type=Path, default=OUT_FILE)
    args = parser.parse_args()

    app_key = os.environ.get("LSEG_APP_KEY")
    if not app_key:
        raise SystemExit("Set your app key first: export LSEG_APP_KEY='...'")

    states = ["active"] if args.active_only else ["active", "inactive"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "market": "US",
        "states": states,
        "status": "started",
        "screens": [],
    }

    frames = []
    try:
        open_session(app_key)
        for state in states:
            entry = {"state": state, "expression": screen_expression(state)}
            try:
                df = fetch_state(state)
                entry["rows_raw"] = int(df.shape[0])
                frames.append(df)
            except Exception as exc:
                entry["error_type"] = type(exc).__name__
                entry["error"] = str(exc)
                print(f"screen failed US/{state}: {type(exc).__name__}: {exc}", file=sys.stderr)
            manifest["screens"].append(entry)

        if not frames:
            raise RuntimeError("No screener rows returned.")
        universe = normalize(pd.concat(frames, ignore_index=True))
        universe.to_csv(args.output, index=False)
        rics_only = args.output.parent / "us_equity_universe_rics_only.csv"
        rics_only.write_text("ric\n" + "\n".join(universe["ric"].tolist()) + "\n")
        manifest["status"] = "ok"
        manifest["rows_unique_ric"] = int(universe["ric"].nunique())
        manifest["output"] = str(args.output)
        manifest["rics_only_output"] = str(rics_only)
        return 0
    except Exception as exc:
        manifest["status"] = "error"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        print(f"US universe build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
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
