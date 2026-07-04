"""
Shared external-feature merge for BOTH training and prediction.

Any external feature (market-context sentiment, rate differentials) must be
added ONLY through add_external_features() below. Training and the predict
paths both call it, so the feature set cannot diverge between them.

History: training used to merge market_context_daily sentiment features while
the predict path did not — the model's sentiment features were silently
zero-filled at predict time (train/serve skew, the same failure class as the
rolled-back 2026-06-25 "relative features" incident). Adding a merge to the
training path alone recreates that bug; don't.
"""

import logging

import numpy as np
import pandas as pd

import forex_config

logger = logging.getLogger(__name__)


def add_external_features(df: pd.DataFrame, currency_pair: str,
                          ext, include_rates: bool = None) -> pd.DataFrame:
    """Merge external features onto a per-pair frame keyed by 'date_time'.

    Args:
        df: per-pair feature frame with a 'date_time' column
        currency_pair: e.g. 'EURUSD' or 'EUR/USD' (used for rate differentials)
        ext: an ExternalDataSources instance
        include_rates: merge rate-differential features from forex_rates_daily;
            None means use forex_config.INCLUDE_RATES_FEATURES

    Left-merges only — rows without external data get NaN, which the training
    pipeline median-fills and the predict paths fill with the SAME medians
    stored in the model artifact (feature_fill_values).
    """
    if df.empty or 'date_time' not in df.columns:
        return df

    if include_rates is None:
        include_rates = forex_config.INCLUDE_RATES_FEATURES

    start_date = pd.to_datetime(df['date_time']).min().strftime('%Y-%m-%d')
    end_date = pd.to_datetime(df['date_time']).max().strftime('%Y-%m-%d')

    # 1) Market-context sentiment (VIX/DXY/S&P/US10Y from market_context_daily).
    # This merge must stay identical to the original training-path merge.
    try:
        sentiment_data = ext.get_market_sentiment_data(start_date, end_date)
        if not sentiment_data.empty:
            df = df.merge(sentiment_data, left_on='date_time',
                          right_index=True, how='left')
    except Exception as e:
        logger.warning(f"Error adding sentiment features for {currency_pair}: {e}")

    # 2) Rate / yield differentials (forex_rates_daily, FRED-seeded).
    if include_rates:
        try:
            rates_wide = ext.get_rates_data(start_date, end_date)
            df = merge_rate_features(df, currency_pair, rates_wide)
        except Exception as e:
            logger.warning(f"Error adding rate features for {currency_pair}: {e}")

    return df


def merge_rate_features(df: pd.DataFrame, currency_pair: str,
                        rates_wide: pd.DataFrame) -> pd.DataFrame:
    """Compute base-minus-quote rate differentials for one pair and left-merge.

    `rates_wide` is the wide frame from ExternalDataSources.get_rates_data():
    business-day index, columns '{CCY}_policy_rate' / '{CCY}_yield_2y' /
    '{CCY}_yield_10y'. All outputs are 'rate_'-prefixed and NaN-tolerant: a
    currency with no FRED coverage (HKD/SGD/INR) simply leaves its diff
    columns NaN, and features with >50% missing are dropped at training time.

    Diffs are computed on the rates frame's own business-day index BEFORE
    merging onto forex trading dates — computing them after the merge would
    silently change the diff horizon wherever dates don't line up. Values for
    date t are known end-of-day t (FRED publishes with a lag), so this adds
    no look-ahead.
    """
    if not currency_pair or rates_wide is None or rates_wide.empty:
        return df
    if 'date_time' not in df.columns:
        return df

    key = currency_pair.replace('/', '').replace('_', '').upper()
    base, quote = key[:3], key[3:6]

    def leg(ccy: str, field: str) -> pd.Series:
        col = f'{ccy}_{field}'
        if col in rates_wide.columns:
            return rates_wide[col]
        return pd.Series(np.nan, index=rates_wide.index)

    feats = pd.DataFrame(index=rates_wide.index)

    # Base-minus-quote differentials (carry / rate-expectation proxies)
    feats['rate_policy_diff'] = leg(base, 'policy_rate') - leg(quote, 'policy_rate')
    feats['rate_yield_2y_diff'] = leg(base, 'yield_2y') - leg(quote, 'yield_2y')
    feats['rate_yield_10y_diff'] = leg(base, 'yield_10y') - leg(quote, 'yield_10y')
    feats['rate_yield_10y_diff_chg_5d'] = feats['rate_yield_10y_diff'].diff(5)
    feats['rate_yield_10y_diff_chg_20d'] = feats['rate_yield_10y_diff'].diff(20)

    # Relative curve slope (10Y-2Y of base vs quote) — sparse until FRED
    # coverage grows, kept for the day it does
    feats['rate_carry_slope'] = (
        (leg(base, 'yield_10y') - leg(base, 'yield_2y'))
        - (leg(quote, 'yield_10y') - leg(quote, 'yield_2y'))
    )

    # Global monetary-regime signal, defined for EVERY pair (USD has full FRED
    # coverage) — gives the Asian pairs at least one macro-rate feature
    usd_policy = leg('USD', 'policy_rate')
    feats['rate_usd_policy'] = usd_policy
    feats['rate_usd_policy_chg_20d'] = usd_policy.diff(20)

    feats.index = pd.to_datetime(feats.index).normalize()

    df = df.copy()
    df['_rate_date'] = pd.to_datetime(df['date_time']).dt.normalize()
    df = df.merge(feats, left_on='_rate_date', right_index=True, how='left')
    df = df.drop(columns=['_rate_date'])
    return df
