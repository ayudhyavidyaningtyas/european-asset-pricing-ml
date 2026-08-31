"""Quick check that Refinitiv/LSEG Data Library can open a session.

Usage:
    1. pip install refinitiv-data
    2. Copy refinitiv-data.config.json.template -> refinitiv-data.config.json and
       fill in your app-key (+ RDP username/password if using platform mode).
    3. Run from the project root so the config file is found:
           python scripts/test_refinitiv_connection.py

Desktop mode requires Refinitiv Workspace to be open and logged in.
"""

import sys

try:
    import refinitiv.data as rd
except ImportError:
    sys.exit("refinitiv.data not installed. Run: pip install refinitiv-data")


def main() -> None:
    print("Opening session (reads refinitiv-data.config.json in the working dir)...")
    rd.open_session()
    print("Session opened OK.")

    # One tiny request confirms data actually flows, not just that auth succeeded.
    df = rd.get_data(universe=["VOD.L"], fields=["TR.CommonName", "TR.PriceClose"])
    print("Sample response:")
    print(df.to_string(index=False))

    rd.close_session()
    print("Session closed. You're connected.")


if __name__ == "__main__":
    main()
