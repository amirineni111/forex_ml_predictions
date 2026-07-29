"""
Backfill / repair script for dbo.forex_prediction_features

Re-derives the TA snapshot (OHLCV, SMA, EMA, RSI, MACD, BB, ATR) for a range of
prediction_dates and rewrites those rows, replacing whatever is there.

Why it exists (2026-07-29): the daily export block in daily_forex_automation.py
ran `dropna(subset=model_features)` WITHOUT the median-fill that the prediction
path applies. Because `rate_yield_10y_diff_chg_5d` is structurally NaN on the
most recent bar, the exported row silently fell back to the newest *complete*
bar instead of the newest bar:
  - EURUSD/EURJPY/EURCHF/EURGBP froze on the 2026-07-16 bar,
  - every other pair was stamped one bar stale,
  - USDHKD/USDSGD (no FRED coverage -> all-NaN rate diffs) lost every row.
The daily path is fixed; this script repairs the rows already written.

Rows are written through ForexResultsExporter.export_prediction_features() —
the SAME writer the daily run uses — so backfilled and live rows cannot drift.

Anchoring: a feature row stamped prediction_date D was produced by the run on
R = the previous business day, from the latest bar available on R. This script
reproduces that: bar = latest bar <= R, signal = the real forex_ml_predictions
row for (R, pair). Dates with no prediction row are skipped rather than
invented — that keeps genuinely-skipped pairs (e.g. USDINR, stale since
2026-05-14 and blocked by the freshness gate) out of the table.

Usage:
    python backfill_prediction_features.py --start 2026-07-07 --end 2026-07-29 --dry-run
    python backfill_prediction_features.py --start 2026-07-07 --end 2026-07-29
    python backfill_prediction_features.py --start 2026-07-07 --end 2026-07-29 --pairs USDHKD,USDSGD
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import timedelta, date

sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import ForexSQLServerConnection
from database.export_results import ForexResultsExporter
from predict_forex_signals import ForexTradingSignalPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

MODEL_PATH = './data/best_forex_model.joblib'

# Raw TA columns the features table stores. Mirrors the ta_columns list in
# daily_forex_automation.py — the exporter does the rsi_14->rsi / atr_14->atr
# normalisation and drops anything not in the DB schema.
TA_COLUMNS = [
    'open_price', 'high_price', 'low_price', 'close_price', 'volume',
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
    'rsi', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'atr', 'atr_14',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
    'daily_return', 'gap', 'volume_ratio', 'currency_pair',
]


def business_days(start: date, end: date):
    """All Mon-Fri dates in [start, end]."""
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def prev_business_day(d: date) -> date:
    pd_ = d - timedelta(days=1)
    while pd_.weekday() >= 5:
        pd_ -= timedelta(days=1)
    return pd_


def load_predictions(engine, start_run: date, end_run: date) -> pd.DataFrame:
    """The real signals written on each run date — the join source for backfilled rows."""
    sql = """
        SELECT CAST(prediction_date AS DATE) AS run_date, currency_pair,
               predicted_signal, signal_confidence, model_name
        FROM dbo.forex_ml_predictions
        WHERE CAST(prediction_date AS DATE) BETWEEN ? AND ?
    """
    df = pd.read_sql(sql, engine, params=(start_run, end_run))
    df['run_date'] = pd.to_datetime(df['run_date']).dt.date
    # A re-run can leave >1 prediction row per (run_date, pair): keep the last.
    return df.drop_duplicates(subset=['run_date', 'currency_pair'], keep='last')


def build_feature_history(pair: str, days_back: int) -> pd.DataFrame:
    """
    Run the production feature pipeline once over full history for `pair`,
    applying the same median-fill the prediction path uses, and return the
    frame indexed by bar date. Rolling features are causal (create_advanced_
    features sorts ascending), so slicing after the fact matches what the run
    on any given date would have computed.
    """
    predictor = ForexTradingSignalPredictor(model_path=MODEL_PATH, currency_pair=pair)
    df = predictor.get_forex_data(currency_pair=pair, days_back=days_back)
    if df.empty:
        return pd.DataFrame()

    df_features, available_features = predictor.prepare_features(df)
    if df_features.empty or not available_features:
        return pd.DataFrame()

    df_features = df_features.sort_values('date_time')
    fill_values = getattr(predictor, 'feature_fill_values', None) or {}
    if fill_values:
        fillable = [c for c in available_features
                    if c in df_features.columns and c in fill_values]
        if fillable:
            df_features[fillable] = df_features[fillable].fillna(pd.Series(fill_values))

    df_features = df_features.dropna(subset=available_features)
    if df_features.empty:
        return pd.DataFrame()

    df_features['bar_date'] = pd.to_datetime(df_features['date_time']).dt.date
    return df_features


def run_backfill(start: date, end: date, pairs=None, days_back: int = 500,
                 dry_run: bool = False):
    db = ForexSQLServerConnection()
    engine = db.get_sqlalchemy_engine()
    exporter = ForexResultsExporter()

    target_dates = business_days(start, end)
    run_dates = [prev_business_day(d) for d in target_dates]

    preds = load_predictions(engine, min(run_dates), max(run_dates))
    if preds.empty:
        logger.error("No rows in forex_ml_predictions for the requested window — nothing to anchor to")
        return

    if pairs:
        preds = preds[preds['currency_pair'].isin(pairs)]
    discovered = sorted(preds['currency_pair'].unique())

    logger.info(f"Backfilling prediction_date {start} -> {end} "
                f"({len(target_dates)} business days){' [DRY-RUN]' if dry_run else ''}")
    logger.info(f"Pairs ({len(discovered)}): {', '.join(discovered)}")

    histories = {}
    for pair in discovered:
        hist = build_feature_history(pair, days_back)
        if hist.empty:
            logger.warning(f"  {pair}: no usable feature history — skipping pair")
        histories[pair] = hist

    written = skipped = 0
    for target_date, run_date in zip(target_dates, run_dates):
        feature_rows, signal_rows = [], []

        for pair in discovered:
            hist = histories.get(pair)
            if hist is None or hist.empty:
                skipped += 1
                continue

            sig = preds[(preds['run_date'] == run_date) & (preds['currency_pair'] == pair)]
            if sig.empty:
                skipped += 1
                continue

            bar = hist[hist['bar_date'] <= run_date].tail(1)
            if bar.empty:
                logger.warning(f"  {pair} {run_date}: no bar on/before run date — skipping")
                skipped += 1
                continue

            cols = [c for c in TA_COLUMNS if c in bar.columns]
            feature_rows.append(bar[cols].copy())
            signal_rows.append(sig[['currency_pair', 'predicted_signal',
                                    'signal_confidence', 'model_name']])

        if not feature_rows:
            logger.warning(f"{target_date}: nothing to write")
            continue

        features_df = pd.concat(feature_rows, ignore_index=True)
        predictions_df = pd.concat(signal_rows, ignore_index=True)

        if dry_run:
            sample = features_df[['currency_pair', 'close_price']].head(3).to_dict('records')
            logger.info(f"[DRY-RUN] {target_date}: would write {len(features_df)} rows "
                        f"(bar {run_date}) e.g. {sample}")
            written += len(features_df)
            continue

        ok = exporter.export_prediction_features(features_df, predictions_df,
                                                 prediction_date=target_date)
        if ok:
            logger.info(f"{target_date}: wrote {len(features_df)} rows (bar {run_date})")
            written += len(features_df)
        else:
            logger.error(f"{target_date}: export FAILED")

    logger.info(f"\nBackfill complete — {written} rows written, {skipped} pair-days skipped")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill/repair forex_prediction_features TA rows')
    parser.add_argument('--start', required=True, help='First prediction_date to rewrite (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='Last prediction_date to rewrite (YYYY-MM-DD)')
    parser.add_argument('--pairs', default=None,
                        help='Comma-separated pair filter, e.g. USDHKD,USDSGD (default: all pairs with predictions)')
    parser.add_argument('--days-back', type=int, default=500,
                        help='History depth to fetch per pair (default: 500, enough for 200-period warmup)')
    parser.add_argument('--dry-run', action='store_true', help='Report what would be written, no DB writes')
    args = parser.parse_args()

    run_backfill(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        pairs=[p.strip() for p in args.pairs.split(',')] if args.pairs else None,
        days_back=args.days_back,
        dry_run=args.dry_run,
    )
