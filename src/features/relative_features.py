"""
Relative / cross-pair forex features (plan Phase 3).

An FX pair is the *relative* price of two economies, so the strongest
directional signals are cross-sectional — they require the whole basket of
pairs at once and therefore cannot live in the per-pair `create_advanced_features`
call. This module operates on the COMBINED panel (all pairs stacked, one row per
pair per date) and adds:

1. Currency-strength indices  — per-currency strength derived from the basket,
   and per-pair `base_strength`, `quote_strength`, `strength_diff` (+ momentum).
   The real version of the crude `usd_strength_proxy` in advanced_features.py.
2. Rate / yield differentials  — base-minus-quote policy-rate & 2Y/10Y yield
   spreads + carry sign (from forex_rates_daily, the dominant medium-term driver).
3. Risk / commodity archetype routing — interact macro signals (VIX/DXY/gold/oil)
   with whether the pair carries a safe-haven (JPY/CHF) or commodity (AUD/CAD/NZD)
   currency.
4. Cross-pair relative momentum — pair return vs. its cluster-average return.

All steps degrade gracefully: if a required input (e.g. rates table) is missing
the corresponding features are simply skipped, never raising.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    from forex_config import (
        SAFE_HAVEN_CCY, COMMODITY_CCY, cluster_for_pair,
    )
except Exception:  # pragma: no cover
    SAFE_HAVEN_CCY = {'JPY', 'CHF'}
    COMMODITY_CCY = {'AUD', 'NZD', 'CAD'}

    def cluster_for_pair(pair: str) -> str:
        return 'usd_majors'

logger = logging.getLogger(__name__)


class RelativeForexFeatures:
    """Cross-sectional features computed over the combined multi-pair panel."""

    def __init__(self, pair_col: str = 'currency_pair', date_col: str = 'date_time',
                 price_col: str = 'close_price'):
        self.pair_col = pair_col
        self.date_col = date_col
        self.price_col = price_col

    # ------------------------------------------------------------------
    def add_relative_features(
        self,
        df: pd.DataFrame,
        rates_wide: Optional[pd.DataFrame] = None,
        intermarket: Optional[pd.DataFrame] = None,
        events: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Add all relative features to the combined panel and return it."""
        if df is None or df.empty:
            return df
        for col in (self.pair_col, self.date_col, self.price_col):
            if col not in df.columns:
                logger.warning("RelativeForexFeatures: missing column '%s' — skipping", col)
                return df

        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df['_pair'] = df[self.pair_col].astype(str).str.replace('/', '', regex=False).str.upper()
        df['_base'] = df['_pair'].str[:3]
        df['_quote'] = df['_pair'].str[3:6]
        # Per-pair daily return (chronological within each pair)
        df = df.sort_values([self.pair_col, self.date_col])
        df['_ret'] = df.groupby(self.pair_col)[self.price_col].pct_change()

        df = self._add_currency_strength(df)
        df = self._add_rate_differentials(df, rates_wide)
        df = self._merge_date_indexed(df, intermarket, label='intermarket')
        df = self._merge_date_indexed(df, events, label='events')
        df = self._add_archetype_features(df)
        df = self._add_cross_pair_momentum(df)

        df = df.drop(columns=['_pair', '_base', '_quote', '_ret'], errors='ignore')
        return df

    # ------------------------------------------------------------------
    def _add_currency_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional currency strength index from the basket of pair returns.

        A pair rising means its base strengthened and quote weakened, so each pair
        contributes +ret to its base currency and -ret to its quote currency. A
        currency's daily strength is the mean of these contributions across every
        pair it appears in.
        """
        base_c = df[[self.date_col, '_base', '_ret']].rename(columns={'_base': 'ccy'})
        quote_c = df[[self.date_col, '_quote', '_ret']].copy()
        quote_c['_ret'] = -quote_c['_ret']
        quote_c = quote_c.rename(columns={'_quote': 'ccy'})
        contrib = pd.concat([base_c, quote_c], ignore_index=True)

        strength = (
            contrib.dropna(subset=['_ret'])
            .groupby([self.date_col, 'ccy'])['_ret']
            .mean()
            .rename('strength')
            .reset_index()
        )
        # Cumulative strength index per currency = running sum of daily strength
        strength = strength.sort_values([self.date_col, 'ccy'])
        strength['strength_idx'] = strength.groupby('ccy')['strength'].cumsum()

        # Merge base + quote strength back onto each row
        for leg in ('base', 'quote'):
            leg_col = f'_{leg}'
            merged = df.merge(
                strength.rename(columns={
                    'ccy': leg_col,
                    'strength': f'{leg}_strength',
                    'strength_idx': f'{leg}_strength_idx',
                }),
                on=[self.date_col, leg_col], how='left',
            )
            df[f'{leg}_strength'] = merged[f'{leg}_strength'].values
            df[f'{leg}_strength_idx'] = merged[f'{leg}_strength_idx'].values

        df['strength_diff'] = df['base_strength'] - df['quote_strength']
        df['strength_idx_diff'] = df['base_strength_idx'] - df['quote_strength_idx']
        # Rolling momentum of the strength differential per pair
        df['strength_diff_mom_5'] = (
            df.groupby(self.pair_col)['strength_diff']
            .transform(lambda s: s.rolling(5, min_periods=2).mean())
        )
        return df

    # ------------------------------------------------------------------
    def _add_rate_differentials(self, df: pd.DataFrame,
                                rates_wide: Optional[pd.DataFrame]) -> pd.DataFrame:
        """base-minus-quote policy rate & 2Y/10Y yield spreads + carry sign."""
        if rates_wide is None or rates_wide.empty:
            return df

        rw = rates_wide.copy()
        rw.index = pd.to_datetime(rw.index)
        # Merge the wide per-currency rate columns on date
        df = df.merge(rw, left_on=self.date_col, right_index=True, how='left')

        for field in ('policy_rate', 'yield_2y', 'yield_10y'):
            base_vals = df.apply(
                lambda r: r.get(f"{r['_base']}_{field}", np.nan), axis=1)
            quote_vals = df.apply(
                lambda r: r.get(f"{r['_quote']}_{field}", np.nan), axis=1)
            diff = pd.to_numeric(base_vals, errors='coerce') - pd.to_numeric(quote_vals, errors='coerce')
            df[f'{field}_diff'] = diff.values

        if 'policy_rate_diff' in df.columns:
            df['carry_sign'] = np.sign(df['policy_rate_diff']).fillna(0)
        return df

    # ------------------------------------------------------------------
    def _merge_date_indexed(self, df: pd.DataFrame,
                            extra: Optional[pd.DataFrame], label: str) -> pd.DataFrame:
        """Merge a date-indexed frame (intermarket / events) onto the panel by date."""
        if extra is None or extra.empty:
            return df
        ex = extra.copy()
        ex.index = pd.to_datetime(ex.index)
        # Avoid clobbering existing columns
        new_cols = [c for c in ex.columns if c not in df.columns]
        if not new_cols:
            return df
        df = df.merge(ex[new_cols], left_on=self.date_col, right_index=True, how='left')
        logger.info("RelativeForexFeatures: merged %d %s columns", len(new_cols), label)
        return df

    # ------------------------------------------------------------------
    def _add_archetype_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tag each pair by archetype and interact macro signals accordingly."""
        has_sh = df['_base'].isin(SAFE_HAVEN_CCY) | df['_quote'].isin(SAFE_HAVEN_CCY)
        has_cmdty = df['_base'].isin(COMMODITY_CCY) | df['_quote'].isin(COMMODITY_CCY)
        df['arch_safe_haven'] = has_sh.astype(int)
        df['arch_commodity'] = has_cmdty.astype(int)
        # +1 if the safe-haven currency is the QUOTE (pair falls in risk-off),
        # -1 if it is the BASE (pair rises in risk-off). 0 if none.
        sh_quote = df['_quote'].isin(SAFE_HAVEN_CCY).astype(int)
        sh_base = df['_base'].isin(SAFE_HAVEN_CCY).astype(int)
        df['safe_haven_dir'] = sh_quote - sh_base

        # Interact risk signals with archetype so the model can route by character.
        if 'vix_close' in df.columns:
            df['vix_x_safehaven'] = pd.to_numeric(df['vix_close'], errors='coerce') * df['safe_haven_dir']
        if 'vix_return_1d' in df.columns:
            df['vixchg_x_commodity'] = pd.to_numeric(df['vix_return_1d'], errors='coerce') * df['arch_commodity']
        for cmdty in ('gold', 'oil', 'copper'):
            ret_col = f'{cmdty}_return_1d'
            if ret_col in df.columns:
                df[f'{cmdty}ret_x_commodity'] = pd.to_numeric(df[ret_col], errors='coerce') * df['arch_commodity']
        if 'dxy_return_1d' in df.columns:
            # USD majors react most strongly to DXY moves
            is_usd = (df['_base'].eq('USD') | df['_quote'].eq('USD')).astype(int)
            df['dxy_x_usd'] = pd.to_numeric(df['dxy_return_1d'], errors='coerce') * is_usd
        return df

    # ------------------------------------------------------------------
    def _add_cross_pair_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pair return relative to its cluster-average return (divergence signal)."""
        df['_cluster'] = df['_pair'].map(cluster_for_pair)
        cluster_mean = (
            df.groupby([self.date_col, '_cluster'])['_ret']
            .transform('mean')
        )
        df['ret_vs_cluster'] = df['_ret'] - cluster_mean
        # Rolling 5d divergence
        df['ret_vs_cluster_5'] = (
            df.groupby(self.pair_col)['ret_vs_cluster']
            .transform(lambda s: s.rolling(5, min_periods=2).mean())
        )
        df = df.drop(columns=['_cluster'], errors='ignore')
        return df
