"""
Advanced Feature Engineering for Forex ML Models

Enhanced with:
- Market regime detection (trending vs ranging)
- Momentum quality indicators
- Mean-reversion signals
- Noise-reduced features using smoothing
- Feature quality scoring to prioritize predictive features
- Forex-specific session and correlation features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder

# Try importing TA-Lib (optional - fallback calculations if not available)
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class AdvancedForexFeatures:
    """Advanced feature engineering specifically designed for forex trading."""
    
    def __init__(self):
        self.label_encoders = {}
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features that can significantly boost model accuracy.
        
        Enhanced with regime detection, momentum quality, and noise-reduced features.
        
        Args:
            df: Input dataframe with basic forex data
            
        Returns:
            DataFrame with advanced engineered features
        """
        df_enhanced = df.copy()
        
        # Ensure we have required columns
        required_cols = ['close_price', 'high_price', 'low_price', 'open_price']
        missing_cols = [col for col in required_cols if col not in df_enhanced.columns]
        if missing_cols:
            print(f"[WARN] Missing columns: {missing_cols}")
            return df_enhanced
        
        print("[INFO] Creating advanced features...")
        initial_cols = len(df_enhanced.columns)
        
        # 1. Market Microstructure Features
        df_enhanced = self._create_microstructure_features(df_enhanced)
        
        # 2. Multi-Timeframe Features  
        df_enhanced = self._create_multi_timeframe_features(df_enhanced)
        
        # 3. Volatility Regime Features (ENHANCED)
        df_enhanced = self._create_volatility_features(df_enhanced)
        
        # 4. Order Flow Indicators
        df_enhanced = self._create_order_flow_features(df_enhanced)
        
        # 5. Market Sentiment Features
        df_enhanced = self._create_sentiment_features(df_enhanced)
        
        # 6. Advanced Technical Patterns
        df_enhanced = self._create_pattern_features(df_enhanced)
        
        # 7. Statistical Features (ENHANCED with noise reduction)
        df_enhanced = self._create_statistical_features(df_enhanced)
        
        # 8. Time-based Features (ENHANCED with session features)
        df_enhanced = self._create_time_features(df_enhanced)
        
        # 9. Cross-Asset Features
        df_enhanced = self._create_cross_asset_features(df_enhanced)
        
        # 10. NEW: Market Regime Detection
        df_enhanced = self._create_regime_features(df_enhanced)
        
        # 11. NEW: Momentum Quality Features
        df_enhanced = self._create_momentum_quality_features(df_enhanced)
        
        # 12. NEW: Mean Reversion Features
        df_enhanced = self._create_mean_reversion_features(df_enhanced)
        
        # 13. NEW: Noise-Reduced Composite Features
        df_enhanced = self._create_composite_features(df_enhanced)
        
        new_features = len(df_enhanced.columns) - initial_cols
        print(f"[OK] Created {new_features} new features (total: {len(df_enhanced.columns)})")
        return df_enhanced
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features."""
        
        # Price action patterns
        body = df['close_price'] - df['open_price']
        full_range = df['high_price'] - df['low_price'] + 1e-8
        
        df['doji'] = (np.abs(body) / full_range < 0.1).astype(int)
        df['hammer'] = ((full_range > 3 * np.abs(body)) & (df['close_price'] > df['open_price'])).astype(int)
        df['shooting_star'] = ((full_range > 3 * np.abs(body)) & (df['close_price'] < df['open_price'])).astype(int)
        
        # Wick analysis
        df['upper_wick'] = df['high_price'] - np.maximum(df['close_price'], df['open_price'])
        df['lower_wick'] = np.minimum(df['close_price'], df['open_price']) - df['low_price']
        df['body_size'] = np.abs(body)
        df['wick_to_body_ratio'] = (df['upper_wick'] + df['lower_wick']) / (df['body_size'] + 1e-8)
        
        # Intraday momentum
        df['intraday_return'] = (df['close_price'] - df['open_price']) / (df['open_price'] + 1e-8)
        df['intraday_range'] = (df['high_price'] - df['low_price']) / (df['open_price'] + 1e-8)
        
        # Close position within bar (0 = close at low, 1 = close at high)
        df['close_position'] = (df['close_price'] - df['low_price']) / (full_range)
        
        return df
    
    def _create_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-timeframe analysis with noise reduction."""
        
        # Multiple period returns - SMOOTHED to reduce noise
        periods = [1, 2, 3, 5, 8, 13, 21]
        for period in periods:
            raw_return = df['close_price'].pct_change(period)
            df[f'return_{period}d'] = raw_return
            
            # Smoothed returns (EMA of returns) - reduces noise
            if period >= 3:
                df[f'return_{period}d_smooth'] = raw_return.ewm(span=3, min_periods=1).mean()
        
        # High/Low channels
        for period in [5, 10, 20]:
            df[f'high_{period}d'] = df['high_price'].rolling(period).max()
            df[f'low_{period}d'] = df['low_price'].rolling(period).min()
            channel_range = df[f'high_{period}d'] - df[f'low_{period}d']
            df[f'channel_position_{period}d'] = (
                (df['close_price'] - df[f'low_{period}d']) / (channel_range + 1e-8)
            )
        
        # Price relative to recent highs/lows
        for period in [5, 10, 20]:
            df[f'price_vs_high_{period}d'] = df['close_price'] / (df['high_price'].rolling(period).max() + 1e-8)
            df[f'price_vs_low_{period}d'] = df['close_price'] / (df['low_price'].rolling(period).min() + 1e-8)
        
        return df
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volatility features with regime detection."""
        
        # Parkinson volatility estimator (more efficient than close-to-close)
        log_hl = np.log(df['high_price'] / (df['low_price'] + 1e-8))
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * log_hl ** 2)
        
        # Garman-Klass volatility estimator
        log_hl_sq = log_hl ** 2
        log_co = np.log(df['close_price'] / (df['open_price'] + 1e-8))
        df['garman_klass_vol'] = np.sqrt(0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co ** 2)
        
        # Rolling volatility at multiple horizons
        returns = df['close_price'].pct_change()
        df['vol_5d'] = returns.rolling(5).std()
        df['vol_10d'] = returns.rolling(10).std()
        df['vol_20d'] = returns.rolling(20).std()
        df['vol_30d'] = returns.rolling(30).std()
        
        # Volatility ratio (short vs long term)
        df['vol_regime'] = df['vol_5d'] / (df['vol_20d'] + 1e-8)
        df['vol_regime_10_30'] = df['vol_10d'] / (df['vol_30d'] + 1e-8)
        
        # Volatility expansion/contraction
        df['vol_expanding'] = (df['vol_5d'] > df['vol_20d']).astype(int)
        df['vol_contracting'] = (df['vol_5d'] < df['vol_20d'] * 0.7).astype(int)
        
        # Volatility breakouts
        vol_20_percentile_80 = df['vol_10d'].rolling(60, min_periods=20).quantile(0.8)
        df['vol_breakout'] = (df['vol_10d'] > vol_20_percentile_80).astype(int)
        
        # ATR-based volatility normalization
        if 'atr_14' in df.columns:
            df['atr_normalized'] = df['atr_14'] / (df['close_price'] + 1e-8)
        elif 'atr' in df.columns:
            df['atr_normalized'] = df['atr'] / (df['close_price'] + 1e-8)
        
        return df
    
    def _create_order_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Order flow and volume analysis."""
        
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return df
        
        # Volume-price analysis
        typical_price = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        cumvol = df['volume'].cumsum()
        df['vwap'] = (df['volume'] * typical_price).cumsum() / (cumvol + 1e-8)
        df['price_vs_vwap'] = df['close_price'] / (df['vwap'] + 1e-8)
        
        # Volume patterns
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-8)
        df['volume_trend'] = df['volume_ma_10'] / (df['volume_ma_20'] + 1e-8)
        
        # High volume indicator
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        
        # Money flow
        df['money_flow'] = typical_price * df['volume']
        df['money_flow_ratio'] = df['money_flow'] / (df['money_flow'].rolling(20).mean() + 1e-8)
        
        # On-Balance Volume trend
        returns = df['close_price'].pct_change()
        obv_direction = np.sign(returns)
        df['obv_direction'] = obv_direction
        
        return df
    
    def _create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market sentiment indicators."""
        
        returns = df['close_price'].pct_change()
        
        # Up/down ratio
        df['up_days_10'] = (returns > 0).rolling(10).sum()
        df['down_days_10'] = (returns < 0).rolling(10).sum()
        df['up_days_20'] = (returns > 0).rolling(20).sum()
        df['down_days_20'] = (returns < 0).rolling(20).sum()
        df['sentiment_ratio_10'] = df['up_days_10'] / (df['down_days_10'] + 1)
        df['sentiment_ratio_20'] = df['up_days_20'] / (df['down_days_20'] + 1)
        
        # Consecutive patterns
        df['consecutive_down'] = (returns < 0).astype(int).groupby(
            (returns >= 0).cumsum()
        ).cumsum()
        df['consecutive_up'] = (returns > 0).astype(int).groupby(
            (returns <= 0).cumsum()
        ).cumsum()
        
        # Average gain vs average loss (like RSI internals)
        gains = returns.clip(lower=0)
        losses = (-returns).clip(lower=0)
        df['avg_gain_14'] = gains.rolling(14).mean()
        df['avg_loss_14'] = losses.rolling(14).mean()
        df['gain_loss_ratio'] = df['avg_gain_14'] / (df['avg_loss_14'] + 1e-8)
        
        return df
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced technical pattern recognition."""
        
        # Support/Resistance levels
        df['local_high'] = (df['high_price'].rolling(5, center=True).max() == df['high_price']).astype(int)
        df['local_low'] = (df['low_price'].rolling(5, center=True).min() == df['low_price']).astype(int)
        
        # Moving average convergence/divergence
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['ma_convergence'] = np.abs(df['sma_20'] - df['sma_50']) / (df['close_price'] + 1e-8)
        else:
            sma_20 = df['close_price'].rolling(20).mean()
            sma_50 = df['close_price'].rolling(50).mean()
            df['ma_convergence'] = np.abs(sma_20 - sma_50) / (df['close_price'] + 1e-8)
        
        # Price channels
        df['price_channel_high'] = df['high_price'].rolling(20).max()
        df['price_channel_low'] = df['low_price'].rolling(20).min()
        channel_range = df['price_channel_high'] - df['price_channel_low']
        df['price_channel_position'] = (
            (df['close_price'] - df['price_channel_low']) / (channel_range + 1e-8)
        )
        
        # Breakout indicators
        df['breaking_high_20'] = (df['close_price'] >= df['price_channel_high']).astype(int)
        df['breaking_low_20'] = (df['close_price'] <= df['price_channel_low']).astype(int)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical and mathematical features with noise reduction."""
        
        # Z-scores (how many standard deviations from mean)
        for period in [10, 20, 50]:
            rolling_mean = df['close_price'].rolling(period).mean()
            rolling_std = df['close_price'].rolling(period).std()
            df[f'price_zscore_{period}'] = (df['close_price'] - rolling_mean) / (rolling_std + 1e-8)
        
        # Percentile rankings
        for period in [10, 20]:
            df[f'price_percentile_{period}'] = df['close_price'].rolling(period).rank(pct=True)
        
        # Autocorrelation features (measure of trend persistence)
        returns = df['close_price'].pct_change()
        for lag in [1, 5]:
            df[f'return_autocorr_{lag}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
            )
        
        # Skewness and kurtosis of recent returns
        df['return_skew_20'] = returns.rolling(20).skew()
        df['return_kurt_20'] = returns.rolling(20).kurt()
        
        # Hurst exponent proxy (mean-reverting vs trending)
        # If H > 0.5 = trending, H < 0.5 = mean-reverting
        def hurst_proxy(series):
            if len(series) < 10:
                return 0.5
            lags = range(2, min(10, len(series) // 2))
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            if all(t == 0 for t in tau):
                return 0.5
            try:
                poly = np.polyfit(np.log(list(lags)), np.log([t + 1e-8 for t in tau]), 1)
                return poly[0]
            except Exception:
                return 0.5
        
        df['hurst_proxy'] = returns.rolling(30).apply(hurst_proxy, raw=True)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features with forex session awareness."""
        
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
            
            # Basic time features
            df['hour'] = df['date_time'].dt.hour
            df['day_of_week'] = df['date_time'].dt.dayofweek
            df['month'] = df['date_time'].dt.month
            df['quarter'] = df['date_time'].dt.quarter
            
            # Market session features
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
            df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
            df['overlap_session'] = (df['london_session'] & df['ny_session']).astype(int)
            df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
            
            # Day-of-week effects (forex has known patterns)
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)
            df['mid_week'] = ((df['day_of_week'] >= 1) & (df['day_of_week'] <= 3)).astype(int)
            
            # Month-end effect
            df['is_month_end'] = (df['date_time'].dt.day >= 25).astype(int)
            df['is_month_start'] = (df['date_time'].dt.day <= 5).astype(int)
        
        return df
    
    def _create_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-asset correlation features."""
        
        # Dollar strength proxy
        if 'currency_pair' in df.columns:
            is_usd_base = df['currency_pair'].str.startswith('USD')
            is_usd_quote = df['currency_pair'].str.endswith('USD')
            df['usd_strength_proxy'] = np.where(
                is_usd_base, df['close_price'], 
                np.where(is_usd_quote, 1 / (df['close_price'] + 1e-8), 1)
            )
        
        return df
    
    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Market regime detection features.
        
        Identifies whether the market is trending, ranging, or volatile.
        This helps models adapt to different market conditions.
        """
        returns = df['close_price'].pct_change()
        
        # ADX-like trend strength (simplified without DI+/DI-)
        # Uses the ratio of directional movement to total movement
        price_change = df['close_price'].diff().abs()
        total_range = df['high_price'] - df['low_price']
        df['direction_strength'] = (price_change / (total_range + 1e-8)).rolling(14).mean()
        
        # Trend consistency: how often returns are in the same direction
        positive_returns = (returns > 0).astype(float)
        df['trend_consistency_10'] = positive_returns.rolling(10).mean()
        df['trend_consistency_20'] = positive_returns.rolling(20).mean()
        
        # Deviation from 0.5 indicates strong trend
        df['trend_bias_10'] = np.abs(df['trend_consistency_10'] - 0.5) * 2
        df['trend_bias_20'] = np.abs(df['trend_consistency_20'] - 0.5) * 2
        
        # Efficiency ratio (Kaufman): measures trending vs noise
        # ER = abs(price change over N) / sum of abs(daily changes over N)
        for period in [10, 20]:
            price_change_n = df['close_price'].diff(period).abs()
            sum_daily_changes = returns.abs().rolling(period).sum()
            df[f'efficiency_ratio_{period}'] = price_change_n / (sum_daily_changes + 1e-8)
        
        # Regime classification features
        # High efficiency + high trend consistency = strong trend
        # Low efficiency + low trend consistency = choppy/ranging
        if 'vol_5d' in df.columns and 'vol_20d' in df.columns:
            df['trending_regime'] = (
                (df['efficiency_ratio_10'] > 0.3) & 
                (df['trend_bias_10'] > 0.3)
            ).astype(int)
            df['ranging_regime'] = (
                (df['efficiency_ratio_10'] < 0.15) & 
                (df['vol_5d'] < df['vol_20d'])
            ).astype(int)
            df['volatile_regime'] = (
                (df['vol_5d'] > df['vol_20d'] * 1.5) & 
                (df['efficiency_ratio_10'] < 0.2)
            ).astype(int)
        
        return df
    
    def _create_momentum_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Momentum quality features.
        
        Not all momentum is equal. These features measure the QUALITY
        of momentum to distinguish between real trends and fake-outs.
        """
        returns = df['close_price'].pct_change()
        
        # Momentum breadth: are multiple timeframes agreeing?
        short_mom = returns.rolling(5).mean()
        med_mom = returns.rolling(10).mean()
        long_mom = returns.rolling(20).mean()
        
        # Momentum alignment (all positive or all negative = strong)
        mom_signs = np.sign(pd.DataFrame({
            'short': short_mom, 'med': med_mom, 'long': long_mom
        }))
        df['momentum_alignment'] = mom_signs.sum(axis=1) / 3  # -1 to 1
        
        # Momentum acceleration/deceleration
        df['momentum_accel_5'] = short_mom.diff()
        df['momentum_accel_10'] = med_mom.diff()
        
        # Smoothed momentum (reduces noise)
        df['momentum_smooth_5'] = short_mom.ewm(span=3).mean()
        df['momentum_smooth_10'] = med_mom.ewm(span=3).mean()
        
        # Momentum divergence (price making new highs but momentum fading)
        price_high_20 = df['close_price'].rolling(20).max()
        mom_high_20 = short_mom.rolling(20).max()
        df['price_at_high'] = (df['close_price'] >= price_high_20 * 0.99).astype(int)
        df['momentum_fading'] = (
            (df['close_price'] > df['close_price'].shift(5)) & 
            (short_mom < short_mom.shift(5))
        ).astype(int)
        
        return df
    
    def _create_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Mean reversion features.
        
        Identifies oversold/overbought conditions where price is likely
        to snap back toward the mean.
        """
        returns = df['close_price'].pct_change()
        
        # Distance from various moving averages (in terms of ATR or std)
        for period in [10, 20, 50]:
            ma = df['close_price'].rolling(period).mean()
            std = df['close_price'].rolling(period).std()
            df[f'distance_from_ma_{period}'] = (df['close_price'] - ma) / (std + 1e-8)
        
        # Extreme extension indicators
        df['extended_up'] = (df.get('price_zscore_20', pd.Series(0, index=df.index)) > 2).astype(int)
        df['extended_down'] = (df.get('price_zscore_20', pd.Series(0, index=df.index)) < -2).astype(int)
        
        # Mean reversion potential (how far from mean + volatility contraction)
        if 'vol_5d' in df.columns and 'vol_20d' in df.columns:
            distance = df.get('distance_from_ma_20', pd.Series(0, index=df.index)).abs()
            vol_contraction = df['vol_5d'] < df['vol_20d']
            df['reversion_potential'] = distance * vol_contraction.astype(float)
        
        # Rubber band effect: larger moves tend to revert
        df['large_move_5d'] = (returns.rolling(5).sum().abs() > returns.rolling(20).std() * 2).astype(int)
        
        return df
    
    def _create_composite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        NEW: Noise-reduced composite features.
        
        Combines multiple indicators into composite scores that are
        more robust than individual features.
        """
        
        # Trend composite score (-1 to 1)
        trend_components = []
        
        if 'trend_consistency_20' in df.columns:
            trend_components.append((df['trend_consistency_20'] - 0.5) * 2)
        
        if 'momentum_alignment' in df.columns:
            trend_components.append(df['momentum_alignment'])
        
        if 'efficiency_ratio_20' in df.columns:
            # Convert efficiency to directional efficiency
            direction = np.sign(df['close_price'].pct_change(20))
            trend_components.append(df['efficiency_ratio_20'] * direction)
        
        if trend_components:
            df['trend_composite'] = pd.concat(trend_components, axis=1).mean(axis=1)
        
        # Momentum composite score
        momentum_components = []
        
        if 'momentum_smooth_5' in df.columns:
            # Normalize momentum to -1, 1 range using tanh
            mom_normalized = np.tanh(df['momentum_smooth_5'] * 100)
            momentum_components.append(mom_normalized)
        
        if 'rsi_14' in df.columns:
            # Convert RSI to -1, 1 range
            rsi_normalized = (df['rsi_14'] - 50) / 50
            momentum_components.append(rsi_normalized)
        
        if momentum_components:
            df['momentum_composite'] = pd.concat(
                [pd.Series(c, index=df.index) if not isinstance(c, pd.Series) else c 
                 for c in momentum_components], axis=1
            ).mean(axis=1)
        
        # Volatility composite
        vol_components = []
        if 'vol_regime' in df.columns:
            vol_components.append(df['vol_regime'])
        if 'atr_normalized' in df.columns:
            vol_components.append(df['atr_normalized'] * 100)
        
        if vol_components:
            df['volatility_composite'] = pd.concat(
                [pd.Series(c, index=df.index) if not isinstance(c, pd.Series) else c 
                 for c in vol_components], axis=1
            ).mean(axis=1)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for ML models."""
        
        categorical_columns = df.select_dtypes(include=['object', 'bool']).columns
        
        for col in categorical_columns:
            if col not in ['currency_pair', 'date_time']:  # Keep these as-is
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    except Exception:
                        df[f'{col}_encoded'] = 0
                else:
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except Exception:
                        df[f'{col}_encoded'] = 0
        
        return df
    
    def get_feature_quality_scores(self, df: pd.DataFrame, target_col: str = 'signal') -> pd.DataFrame:
        """
        Score features by their predictive quality.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            
        Returns:
            DataFrame with feature quality scores
        """
        if target_col not in df.columns:
            return pd.DataFrame()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c != target_col]
        
        scores = []
        for col in feature_cols:
            try:
                # Correlation with target direction
                corr = df[col].corr(df[target_col].map({'BUY': 1, 'SELL': -1, 'HOLD': 0}) 
                                   if df[target_col].dtype == 'object'
                                   else df[target_col])
                
                # Variance (features with very low variance are uninformative)
                variance = df[col].var()
                
                # Missing ratio
                missing_ratio = df[col].isna().mean()
                
                scores.append({
                    'feature': col,
                    'abs_correlation': abs(corr) if not pd.isna(corr) else 0,
                    'variance': variance,
                    'missing_ratio': missing_ratio,
                    'quality_score': abs(corr) * (1 - missing_ratio) if not pd.isna(corr) else 0
                })
            except Exception:
                continue
        
        quality_df = pd.DataFrame(scores).sort_values('quality_score', ascending=False)
        return quality_df
