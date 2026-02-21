"""
Additional Data Sources for Forex ML Models
These data sources can significantly boost prediction accuracy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
import requests
from datetime import datetime, timedelta

# Optional: pyodbc for reading from shared market_context_daily table
try:
    import pyodbc
    _HAS_PYODBC = True
except ImportError:
    _HAS_PYODBC = False

class ExternalDataSources:
    """Collect external data that impacts forex movements"""
    
    def __init__(self, use_db=True):
        """
        Args:
            use_db: If True, read market sentiment from shared market_context_daily table
                    in SQL Server (faster, no rate limits). Falls back to yfinance if unavailable.
        """
        self.data_cache = {}
        self.use_db = use_db and _HAS_PYODBC
    
    def get_economic_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get economic indicators that drive forex movements
        
        Key indicators to add to your database:
        1. Interest Rates (Fed Funds, ECB, BOJ, BOE)
        2. GDP Growth Rates
        3. Inflation Rates (CPI, PPI)
        4. Employment Data (NFP, Unemployment)
        5. Central Bank Meeting Dates
        6. Economic Calendar Events
        """
        
        # This would connect to economic data APIs like:
        # - Federal Reserve Economic Data (FRED)
        # - European Central Bank API
        # - Bank of England API
        # - Economic Calendar APIs
        
        indicators = {
            'fed_funds_rate': self._get_fed_funds_rate(start_date, end_date),
            'ecb_rate': self._get_ecb_rate(start_date, end_date),
            'us_cpi': self._get_us_inflation(start_date, end_date),
            'eur_cpi': self._get_eur_inflation(start_date, end_date),
            'us_gdp': self._get_us_gdp(start_date, end_date),
            'eur_gdp': self._get_eur_gdp(start_date, end_date)
        }
        
        return pd.DataFrame(indicators)
    
    def get_market_sentiment_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get market sentiment indicators from shared DB table or yfinance fallback.
        
        When use_db=True, reads from market_context_daily table which has:
        VIX, DXY, S&P 500, NASDAQ, US 10Y yield - all pre-computed with returns.
        This is faster and avoids yfinance rate limits during training.
        """
        
        # Try reading from shared database table first
        if self.use_db:
            try:
                return self._get_sentiment_from_db(start_date, end_date)
            except Exception as e:
                print(f"[WARN] Could not read market context from DB: {e}")
                print("[INFO] Falling back to yfinance download...")
        
        return self._get_sentiment_from_yfinance(start_date, end_date)
    
    def _get_sentiment_from_db(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Read market sentiment data from shared market_context_daily table."""
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=192.168.86.55\\MSSQLSERVER01;"
            "DATABASE=stockdata_db;"
            "Trusted_Connection=yes;"
        )
        
        query = f"""
        SELECT trading_date,
               vix_close, vix_change_pct,
               sp500_close, sp500_return_1d,
               nasdaq_comp_close, nasdaq_comp_return_1d,
               dxy_close, dxy_return_1d,
               us_10y_yield_close, us_10y_yield_change
        FROM dbo.market_context_daily
        WHERE trading_date >= '{start_date}' AND trading_date <= '{end_date}'
        ORDER BY trading_date
        """
        
        df = pd.read_sql(query, conn, parse_dates=['trading_date'])
        conn.close()
        
        if df.empty:
            raise ValueError("No data in market_context_daily for date range")
        
        # Rename columns to match existing forex feature naming convention
        rename_map = {
            'vix_close': 'vix_close',
            'vix_change_pct': 'vix_return_1d',
            'sp500_close': 'spx_close',
            'sp500_return_1d': 'spx_return_1d',
            'dxy_close': 'dxy_close',
            'dxy_return_1d': 'dxy_return_1d',
            'us_10y_yield_close': 'us_10y_close',
            'us_10y_yield_change': 'us_10y_return_1d',
        }
        df = df.rename(columns=rename_map)
        
        # Compute rolling volatility features (same as yfinance path)
        for col in ['vix', 'spx', 'dxy', 'us_10y']:
            close_col = f'{col}_close'
            if close_col in df.columns:
                df[f'{col}_return_5d'] = df[close_col].pct_change(5)
                df[f'{col}_volatility'] = df[close_col].pct_change().rolling(20).std()
                df[f'{col}_volume'] = 0  # No volume from DB — set to 0 for compatibility
        
        df = df.set_index('trading_date')
        print(f"  [OK] Market context from DB: {len(df)} rows ({start_date} to {end_date})")
        return df
    
    def _get_sentiment_from_yfinance(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Original yfinance download path (fallback)."""
        
        sentiment_tickers = {
            'vix': '^VIX',          # Fear index
            'dxy': 'DX-Y.NYB',      # USD Index  
            'gold': 'GC=F',         # Gold futures
            'us_10y': '^TNX',       # 10Y Treasury yield
            'oil': 'CL=F',          # Oil futures
            'spx': '^GSPC'          # S&P 500
        }
        
        sentiment_data = {}
        
        for name, ticker in sentiment_tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                sentiment_data[f'{name}_close'] = data['Close']
                sentiment_data[f'{name}_volume'] = data.get('Volume', 0)
                
                # Calculate momentum features
                sentiment_data[f'{name}_return_1d'] = data['Close'].pct_change()
                sentiment_data[f'{name}_return_5d'] = data['Close'].pct_change(5)
                sentiment_data[f'{name}_volatility'] = data['Close'].pct_change().rolling(20).std()
                
            except Exception as e:
                print(f"Error downloading {name}: {e}")
        
        return pd.DataFrame(sentiment_data)
    
    def get_news_sentiment(self, currency_pair: str, date: str) -> Dict:
        """
        Get news sentiment for specific currency pair
        
        Sources to implement:
        1. Financial news APIs (Bloomberg, Reuters)
        2. Social media sentiment (Twitter, Reddit)
        3. Central bank communications
        4. Economic news impact scores
        """
        
        # This would use APIs like:
        # - Alpha Vantage News Sentiment API
        # - NewsAPI
        # - Twitter API
        # - Reddit API
        
        # Placeholder implementation
        return {
            'news_sentiment_score': 0.0,  # -1 (very negative) to +1 (very positive)
            'news_volume': 0,             # Number of news articles
            'social_sentiment': 0.0,      # Social media sentiment
            'news_importance': 0.0        # Importance score of news events
        }
    
    def get_intermarket_data(self, date: str) -> Dict:
        """
        Get intermarket relationships that affect forex
        
        Key relationships:
        1. Bond yield differentials
        2. Commodity currencies correlation
        3. Risk-on/Risk-off sentiment
        4. Cross-currency correlations
        """
        
        intermarket_features = {
            'yield_spread_us_de': 0.0,    # US-Germany 10Y yield spread
            'yield_spread_us_jp': 0.0,    # US-Japan 10Y yield spread
            'risk_on_sentiment': 0.0,     # Risk appetite measure
            'commodity_index': 0.0,       # Commodity price index
            'em_currency_index': 0.0      # Emerging market currency index
        }
        
        return intermarket_features
    
    def get_technical_regime_data(self, currency_pair: str) -> Dict:
        """
        Get technical regime indicators
        
        Regime features:
        1. Trend regime (trending vs ranging)
        2. Volatility regime (high/low vol)
        3. Momentum regime
        4. Mean reversion patterns
        """
        
        regime_features = {
            'trend_regime': 0,        # 0=ranging, 1=trending
            'volatility_regime': 0,   # 0=low vol, 1=high vol
            'momentum_regime': 0,     # 0=weak, 1=strong momentum
            'market_phase': 0         # Market cycle phase
        }
        
        return regime_features
    
    # Helper methods for economic indicators
    def _get_fed_funds_rate(self, start_date: str, end_date: str) -> pd.Series:
        """Get Federal Funds Rate from FRED API"""
        # Implementation would use FRED API
        # For now, return dummy data
        return pd.Series(dtype=float)
    
    def _get_ecb_rate(self, start_date: str, end_date: str) -> pd.Series:
        """Get ECB main refinancing rate"""
        return pd.Series(dtype=float)
    
    def _get_us_inflation(self, start_date: str, end_date: str) -> pd.Series:
        """Get US CPI data"""
        return pd.Series(dtype=float)
    
    def _get_eur_inflation(self, start_date: str, end_date: str) -> pd.Series:
        """Get Eurozone CPI data"""
        return pd.Series(dtype=float)
    
    def _get_us_gdp(self, start_date: str, end_date: str) -> pd.Series:
        """Get US GDP growth data"""
        return pd.Series(dtype=float)
    
    def _get_eur_gdp(self, start_date: str, end_date: str) -> pd.Series:
        """Get Eurozone GDP growth data"""
        return pd.Series(dtype=float)


# SQL queries to add these tables to your database
CREATE_EXTERNAL_DATA_TABLES = """
-- Economic Indicators Table
CREATE TABLE forex_economic_indicators (
    date DATE PRIMARY KEY,
    fed_funds_rate DECIMAL(5,3),
    ecb_rate DECIMAL(5,3),
    boj_rate DECIMAL(5,3),
    boe_rate DECIMAL(5,3),
    us_cpi_yoy DECIMAL(5,2),
    eur_cpi_yoy DECIMAL(5,2),
    us_gdp_qoq DECIMAL(5,2),
    eur_gdp_qoq DECIMAL(5,2),
    us_unemployment DECIMAL(4,1),
    eur_unemployment DECIMAL(4,1),
    us_nfp_change INT,
    created_at DATETIME DEFAULT GETDATE()
);

-- Market Sentiment Table  
CREATE TABLE forex_market_sentiment (
    date DATE PRIMARY KEY,
    vix_close DECIMAL(8,2),
    dxy_close DECIMAL(8,4),
    gold_close DECIMAL(8,2),
    us_10y_yield DECIMAL(5,3),
    oil_close DECIMAL(8,2),
    spx_close DECIMAL(8,2),
    risk_sentiment DECIMAL(3,2), -- -1 to +1
    created_at DATETIME DEFAULT GETDATE()
);

-- News Sentiment Table
CREATE TABLE forex_news_sentiment (
    date DATE,
    currency_pair VARCHAR(10),
    news_sentiment_score DECIMAL(3,2), -- -1 to +1
    news_volume INT,
    social_sentiment DECIMAL(3,2),
    news_importance DECIMAL(3,2),
    PRIMARY KEY (date, currency_pair)
);

-- Intermarket Data Table
CREATE TABLE forex_intermarket (
    date DATE PRIMARY KEY,
    yield_spread_us_de DECIMAL(5,3),
    yield_spread_us_jp DECIMAL(5,3),
    yield_spread_us_uk DECIMAL(5,3),
    commodity_index DECIMAL(8,2),
    em_currency_index DECIMAL(8,2),
    created_at DATETIME DEFAULT GETDATE()
);
"""