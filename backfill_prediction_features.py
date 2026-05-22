"""
Backfill script for dbo.forex_prediction_features

Re-populates TA feature columns (OHLCV, SMA, EMA, RSI, MACD, BB, ATR) for the
last N business days where those columns are NULL/0 due to the column-name mismatch
bug fixed in May 2026.

Usage:
    python backfill_prediction_features.py              # backfill last 30 days
    python backfill_prediction_features.py --days 60    # backfill last 60 days
    python backfill_prediction_features.py --dry-run    # print what would be inserted, no DB writes
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import ForexSQLServerConnection
from database.export_results import ForexResultsExporter
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


CURRENCY_PAIRS = [
    'USD/INR', 'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD',
    'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY'
]

# Symbol format used by the DB (no slash)
PAIR_SYMBOLS = {
    'USD/INR': 'USDINR', 'EUR/USD': 'EURUSD', 'GBP/USD': 'GBPUSD',
    'USD/JPY': 'USDJPY', 'AUD/USD': 'AUDUSD', 'USD/CAD': 'USDCAD',
    'NZD/USD': 'NZDUSD', 'EUR/GBP': 'EURGBP', 'EUR/JPY': 'EURJPY',
    'GBP/JPY': 'GBPJPY'
}

# TA columns this table stores (must match DB schema)
TA_COLUMNS = [
    'open_price', 'high_price', 'low_price', 'close_price', 'volume',
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
    'rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
    'daily_return', 'gap', 'volume_ratio',
]


def business_days_back(n: int):
    """Return list of business dates for the last n business days (most recent last)."""
    days = []
    d = date.today() - timedelta(days=1)   # start from yesterday
    while len(days) < n:
        if d.weekday() < 5:   # Mon-Fri
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    return days


def next_business_day(d: date) -> date:
    """Return the next business day after d."""
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:
        nd += timedelta(days=1)
    return nd


def normalise_ta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename variant column names produced by different DB view versions so they
    match the forex_prediction_features schema:
      rsi_14  -> rsi
      atr_14  -> atr
      macd_histogram derived when absent
    """
    rename_map = {}
    if 'rsi_14' in df.columns and 'rsi' not in df.columns:
        rename_map['rsi_14'] = 'rsi'
    if 'atr_14' in df.columns and 'atr' not in df.columns:
        rename_map['atr_14'] = 'atr'
    if rename_map:
        df = df.rename(columns=rename_map)
    if 'macd_histogram' not in df.columns and 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_histogram'] = df['macd'] - df['macd_signal']

    # sma_5 / ema_5 / sma_10 / ema_10 are often not in the view — compute from close_price
    if 'close_price' in df.columns:
        for window, col in [(5, 'sma_5'), (10, 'sma_10')]:
            if col not in df.columns:
                df[col] = df['close_price'].rolling(window).mean()
        for window, col in [(5, 'ema_5'), (10, 'ema_10'), (20, 'ema_20'), (50, 'ema_50')]:
            if col not in df.columns:
                df[col] = df['close_price'].ewm(span=window, adjust=False).mean()

    # bb_width / bb_percent derived when absent
    if 'bb_width' not in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
    if 'bb_percent' not in df.columns and 'bb_upper' in df.columns and 'bb_lower' in df.columns and 'close_price' in df.columns:
        band_range = df['bb_upper'] - df['bb_lower']
        df['bb_percent'] = (df['close_price'] - df['bb_lower']) / band_range.replace(0, np.nan)

    # gap = open - prev_close
    if 'gap' not in df.columns and 'open_price' in df.columns and 'close_price' in df.columns:
        df['gap'] = df['open_price'] - df['close_price'].shift(1)

    # volume_ratio = volume / 20-day avg volume
    if 'volume_ratio' not in df.columns and 'volume' in df.columns:
        avg_vol = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / avg_vol.replace(0, np.nan)

    return df


def fetch_full_history(db: ForexSQLServerConnection, symbol: str) -> pd.DataFrame:
    """Fetch all available history for a symbol via the indicators view."""
    try:
        df = db.get_forex_data_with_indicators(currency_pair=symbol)
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def get_row_as_of(full_df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """
    Slice full_df to rows on/before as_of, run column normalisation,
    and return the last row (the TA snapshot for that date).
    """
    date_col = 'date_time'
    if date_col not in full_df.columns:
        return pd.DataFrame()

    df = full_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].dt.date <= as_of].copy()

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(date_col)
    df = normalise_ta_columns(df)
    return df.tail(1)


def delete_existing_rows(engine, prediction_date: date, currency_pair: str):
    """Delete any existing row for this prediction_date + pair so we can re-insert cleanly."""
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM dbo.forex_prediction_features WHERE prediction_date = :pd AND currency_pair = :cp"),
            {"pd": prediction_date, "cp": currency_pair}
        )
        conn.commit()


def insert_feature_row(engine, row: dict, dry_run: bool):
    """Insert a single feature row into forex_prediction_features."""
    if dry_run:
        logger.info(f"[DRY-RUN] Would insert: {row.get('currency_pair')} / {row.get('prediction_date')} — "
                    f"close={row.get('close_price')} rsi={row.get('rsi')} macd={row.get('macd')}")
        return

    cols = list(row.keys())
    placeholders = ', '.join(f':{c}' for c in cols)
    col_list = ', '.join(cols)
    sql = f"INSERT INTO dbo.forex_prediction_features ({col_list}) VALUES ({placeholders})"
    with engine.connect() as conn:
        conn.execute(text(sql), row)
        conn.commit()


def run_backfill(days: int = 30, dry_run: bool = False):
    db = ForexSQLServerConnection()
    engine = db.get_sqlalchemy_engine()
    target_dates = business_days_back(days)

    logger.info(f"Backfilling forex_prediction_features for {len(target_dates)} business days "
                f"({target_dates[0]} → {target_dates[-1]}){' [DRY-RUN]' if dry_run else ''}")

    total_inserted = 0
    total_skipped = 0

    for pair_slash, symbol in PAIR_SYMBOLS.items():
        logger.info(f"--- Processing {pair_slash} ({symbol}) ---")
        full_df = fetch_full_history(db, symbol)
        if full_df.empty:
            logger.warning(f"Skipping {pair_slash} — no historical data")
            continue

        # Ensure currency_pair column uses slash format to match existing table records
        full_df['currency_pair'] = pair_slash

        for as_of in target_dates:
            prediction_date = next_business_day(as_of)
            row_df = get_row_as_of(full_df, as_of)

            if row_df.empty:
                logger.warning(f"  {pair_slash} {as_of}: no data — skipping")
                total_skipped += 1
                continue

            # Build the row dict with only columns that exist in the DB schema
            all_cols = ['prediction_date', 'currency_pair'] + TA_COLUMNS
            row = {'prediction_date': prediction_date, 'currency_pair': pair_slash}
            for col in TA_COLUMNS:
                val = row_df[col].iloc[0] if col in row_df.columns else None
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    row[col] = float(val)
                else:
                    row[col] = None

            if not dry_run:
                delete_existing_rows(engine, prediction_date, pair_slash)

            insert_feature_row(engine, row, dry_run)
            total_inserted += 1

        logger.info(f"  {pair_slash}: done")

    logger.info(f"\nBackfill complete — {total_inserted} rows inserted/updated, {total_skipped} skipped")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill forex_prediction_features TA columns')
    parser.add_argument('--days', type=int, default=30, help='Number of business days to backfill (default: 30)')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be inserted without writing to DB')
    args = parser.parse_args()

    run_backfill(days=args.days, dry_run=args.dry_run)
