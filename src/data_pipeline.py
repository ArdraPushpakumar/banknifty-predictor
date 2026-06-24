"""
Fetches Bank Nifty close prices from Yahoo Finance and keeps a local
CSV up to date.

First run: downloads full history from HISTORY_START_DATE to today.
Every later run: only downloads days newer than what's already saved,
so this is safe to run daily without re-downloading everything.
"""

import os

import pandas as pd
import yfinance as yf

from config import TICKER, HISTORY_START_DATE, DATA_PATH


def _strip_timezone(df, col="Date"):
    if df[col].dt.tz is not None:
        df[col] = df[col].dt.tz_localize(None)
    return df


def fetch_history(start):
    raw = yf.download(TICKER, start=start, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["Date", "Close"])
    raw = raw.reset_index()
    # yfinance sometimes returns MultiIndex columns for a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = _strip_timezone(df)
    return df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)


def update_local_data():
    """Returns the full up-to-date history as a DataFrame, and writes it to DATA_PATH."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"No local history found. Downloading full history from {HISTORY_START_DATE}...")
        df = fetch_history(start=HISTORY_START_DATE)
        df.to_csv(DATA_PATH, index=False)
        print(f"Saved {len(df)} rows to {DATA_PATH}.")
        return df

    existing = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    last_date = existing["Date"].max()
    new_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    new_data = fetch_history(start=new_start)
    if new_data.empty:
        print("No new trading days available since last update.")
        return existing

    combined = (
        pd.concat([existing, new_data])
        .drop_duplicates(subset="Date")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    combined.to_csv(DATA_PATH, index=False)
    print(f"Added {len(new_data)} new row(s). Local history now has {len(combined)} rows.")
    return combined


if __name__ == "__main__":
    update_local_data()
