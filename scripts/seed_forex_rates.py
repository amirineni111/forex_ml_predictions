#!/usr/bin/env python3
"""
Seed / refresh the shared ``forex_rates_daily`` table from FRED.

This table powers the rate- and yield-differential features (the dominant
medium-term FX driver) consumed by
``ExternalDataSources.get_rates_data`` → ``RelativeForexFeatures``.

It creates the table if missing and UPSERTs, per currency, the policy rate and
2Y / 10Y sovereign yields. FRED series ids live in ``src/forex_config.py``
(``FRED_RATE_SERIES``); US series are daily, most non-US long-term yields are
monthly and are forward-filled onto the business-day grid.

Usage
-----
    python scripts/seed_forex_rates.py                 # full history (default 2y)
    python scripts/seed_forex_rates.py --start 2018-01-01
    python scripts/seed_forex_rates.py --days-back 10  # daily incremental refresh

Requires a (free) FRED API key in ``.env`` as ``FRED_API_KEY``
(https://fredaccount.stlouisfed.org/apikey).
"""

import os
import sys
import time
import argparse
from datetime import datetime, timedelta

import requests
import pandas as pd
from dotenv import load_dotenv

# Make src/ importable (callers elsewhere in the repo use this same pattern)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from forex_config import FRED_RATE_SERIES, FRED_API_KEY, RATES_TABLE  # noqa: E402

load_dotenv()

try:
    import pyodbc
except ImportError:
    pyodbc = None

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
RATE_FIELDS = ['policy_rate', 'yield_2y', 'yield_10y']


def _get_connection():
    """pyodbc connection using the same env vars as the rest of the repo."""
    if pyodbc is None:
        raise SystemExit("pyodbc is required (pip install pyodbc)")
    server = os.getenv('SQL_SERVER', '192.168.86.28,1444')
    database = os.getenv('SQL_DATABASE', 'stockdata_db')
    driver = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
    username = os.getenv('SQL_USERNAME', '')
    password = os.getenv('SQL_PASSWORD', '')
    trusted = os.getenv('SQL_TRUSTED_CONNECTION', 'no').lower() == 'yes'
    if trusted:
        conn_str = (f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
                    f"Trusted_Connection=yes;")
    else:
        conn_str = (f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};"
                    f"UID={username};PWD={password};")
    return pyodbc.connect(conn_str)


def fetch_fred_series(series_id: str, api_key: str, start: str) -> pd.Series:
    """Return a numeric, date-indexed Series for a FRED series (missing → dropped)."""
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start,
    }
    resp = requests.get(FRED_URL, params=params, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get('observations', [])
    if not obs:
        return pd.Series(dtype=float)
    s = pd.Series({o['date']: o['value'] for o in obs})
    s.index = pd.to_datetime(s.index)
    # FRED encodes missing values as '.'
    return pd.to_numeric(s, errors='coerce').dropna().sort_index()


def build_rates_frame(start: str, api_key: str) -> pd.DataFrame:
    """Assemble a long DataFrame: trading_date, ccy, policy_rate, yield_2y, yield_10y."""
    idx = pd.date_range(start, datetime.today(), freq='B')  # business days
    frames = []
    for ccy, series in FRED_RATE_SERIES.items():
        cols = {}
        for field, sid in series.items():
            if not sid:
                continue
            try:
                s = fetch_fred_series(sid, api_key, start)
                if not s.empty:
                    # forward-fill (monthly long-term yields → daily) onto B-day grid
                    cols[field] = s.reindex(idx.union(s.index)).ffill().reindex(idx)
                print(f"  [OK] {ccy} {field:11s} <- {sid} ({len(s)} obs)")
            except Exception as e:
                print(f"  [WARN] {ccy} {field} ({sid}) failed: {e}")
            time.sleep(0.25)  # be polite to the FRED API
        if not cols:
            continue
        df = pd.DataFrame(cols, index=idx)
        for f in RATE_FIELDS:           # ensure all three columns exist
            if f not in df.columns:
                df[f] = pd.NA
        df['ccy'] = ccy
        df['trading_date'] = df.index
        frames.append(df.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(columns=['trading_date', 'ccy'] + RATE_FIELDS)

    long = pd.concat(frames, ignore_index=True)
    # Drop rows with no rate at all (e.g. dates before a series starts)
    long = long.dropna(subset=RATE_FIELDS, how='all')
    return long[['trading_date', 'ccy'] + RATE_FIELDS]


def ensure_table(cur):
    cur.execute(f"""
        IF NOT EXISTS (SELECT 1 FROM INFORMATION_SCHEMA.TABLES
                       WHERE TABLE_NAME = '{RATES_TABLE}')
        CREATE TABLE {RATES_TABLE} (
            trading_date DATE        NOT NULL,
            ccy          VARCHAR(3)  NOT NULL,
            policy_rate  DECIMAL(7,4) NULL,
            yield_2y     DECIMAL(7,4) NULL,
            yield_10y    DECIMAL(7,4) NULL,
            updated_at   DATETIME    DEFAULT GETDATE(),
            CONSTRAINT PK_{RATES_TABLE} PRIMARY KEY (trading_date, ccy)
        );
    """)


def upsert(cur, df: pd.DataFrame) -> int:
    merge_sql = f"""
        MERGE {RATES_TABLE} AS t
        USING (SELECT ? AS trading_date, ? AS ccy,
                      ? AS policy_rate, ? AS yield_2y, ? AS yield_10y) AS s
        ON t.trading_date = s.trading_date AND t.ccy = s.ccy
        WHEN MATCHED THEN UPDATE SET
            policy_rate = s.policy_rate, yield_2y = s.yield_2y,
            yield_10y = s.yield_10y, updated_at = GETDATE()
        WHEN NOT MATCHED THEN
            INSERT (trading_date, ccy, policy_rate, yield_2y, yield_10y)
            VALUES (s.trading_date, s.ccy, s.policy_rate, s.yield_2y, s.yield_10y);
    """

    def _v(x):
        return None if pd.isna(x) else float(x)

    params = [
        (row.trading_date.date() if hasattr(row.trading_date, 'date') else row.trading_date,
         row.ccy, _v(row.policy_rate), _v(row.yield_2y), _v(row.yield_10y))
        for row in df.itertuples(index=False)
    ]
    cur.fast_executemany = False  # MERGE + executemany: keep simple/correct
    cur.executemany(merge_sql, params)
    return len(params)


def main():
    ap = argparse.ArgumentParser(description="Seed forex_rates_daily from FRED")
    ap.add_argument('--start', default=None,
                    help="Start date YYYY-MM-DD (default: 2 years ago)")
    ap.add_argument('--days-back', type=int, default=None,
                    help="Incremental: only fetch the last N days")
    args = ap.parse_args()

    api_key = os.getenv('FRED_API_KEY', FRED_API_KEY)
    if not api_key:
        raise SystemExit(
            "FRED_API_KEY is not set. Get a free key at "
            "https://fredaccount.stlouisfed.org/apikey and add it to .env:\n"
            "    FRED_API_KEY=your_key_here")

    if args.days_back:
        start = (datetime.today() - timedelta(days=args.days_back)).strftime('%Y-%m-%d')
    else:
        start = args.start or (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')

    print(f"[INFO] Fetching FRED rate/yield series from {start} ...")
    df = build_rates_frame(start, api_key)
    if df.empty:
        print("[WARN] No rate data assembled — nothing to write.")
        return

    print(f"[INFO] Assembled {len(df)} rows across {df['ccy'].nunique()} currencies. Upserting...")
    conn = _get_connection()
    try:
        cur = conn.cursor()
        ensure_table(cur)
        n = upsert(cur, df)
        conn.commit()
        print(f"[OK] Upserted {n} rows into {RATES_TABLE}.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
