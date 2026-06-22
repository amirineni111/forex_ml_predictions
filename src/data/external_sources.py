"""
Additional Data Sources for Forex ML Models
These data sources can significantly boost prediction accuracy
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Optional: pyodbc for reading from shared market_context_daily table
try:
    import pyodbc
    _HAS_PYODBC = True
except ImportError:
    _HAS_PYODBC = False

# Structural config: external-data table names (Phase 1/2). Imported lazily so
# this module still loads if forex_config is absent on the path.
try:
    from forex_config import RATES_TABLE, INTERMARKET_TABLE, ECON_EVENTS_TABLE
except Exception:  # pragma: no cover - fallback to defaults
    RATES_TABLE = 'forex_rates_daily'
    INTERMARKET_TABLE = 'forex_intermarket_daily'
    ECON_EVENTS_TABLE = 'forex_econ_events'

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
    
    def _get_connection(self):
        """Build a pyodbc connection from the same env vars the rest of the repo uses."""
        server = os.getenv('SQL_SERVER', '192.168.86.28,1444')
        database = os.getenv('SQL_DATABASE', 'stockdata_db')
        driver = os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
        username = os.getenv('SQL_USERNAME', '')
        password = os.getenv('SQL_PASSWORD', '')
        trusted = os.getenv('SQL_TRUSTED_CONNECTION', 'no').lower() == 'yes'

        if trusted:
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Trusted_Connection=yes;"
            )
        else:
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
            )
        return pyodbc.connect(conn_str)

    def _table_exists(self, conn, table_name: str) -> bool:
        """Return True if `table_name` exists in the connected database."""
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?",
                [table_name],
            )
            return cur.fetchone() is not None
        except Exception:
            return False

    def _get_sentiment_from_db(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Read market sentiment data from shared market_context_daily table."""
        conn = self._get_connection()

        query = """
        SELECT trading_date,
               vix_close, vix_change_pct,
               sp500_close, sp500_return_1d,
               nasdaq_comp_close, nasdaq_comp_return_1d,
               dxy_close, dxy_return_1d,
               us_10y_yield_close, us_10y_yield_change
        FROM dbo.market_context_daily
        WHERE trading_date >= ? AND trading_date <= ?
        ORDER BY trading_date
        """
        
        df = pd.read_sql(query, conn, params=[start_date, end_date], parse_dates=['trading_date'])
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
    
    def get_rates_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Per-currency policy rate + 2Y/10Y sovereign yields, indexed by trading_date.

        Reads the shared `forex_rates_daily` table (columns: trading_date, ccy,
        policy_rate, yield_2y, yield_10y) and pivots to WIDE form so each currency
        becomes columns like `USD_policy_rate`, `EUR_yield_10y`, etc. — ready to
        merge on date and difference per pair (see relative_features).

        Returns an empty DataFrame (graceful) when the table or DB is unavailable;
        callers should treat missing rate features as "not yet ingested".
        """
        if not self.use_db:
            return pd.DataFrame()
        try:
            conn = self._get_connection()
            if not self._table_exists(conn, RATES_TABLE):
                conn.close()
                print(f"[INFO] {RATES_TABLE} not present — skipping rate/yield features")
                return pd.DataFrame()

            query = f"""
            SELECT trading_date, ccy, policy_rate, yield_2y, yield_10y
            FROM dbo.{RATES_TABLE}
            WHERE trading_date >= ? AND trading_date <= ?
            ORDER BY trading_date
            """
            df = pd.read_sql(query, conn, params=[start_date, end_date],
                             parse_dates=['trading_date'])
            conn.close()
            if df.empty:
                return pd.DataFrame()

            wide = df.pivot_table(index='trading_date', columns='ccy',
                                  values=['policy_rate', 'yield_2y', 'yield_10y'])
            # Flatten MultiIndex columns: ('yield_10y','USD') -> 'USD_yield_10y'
            wide.columns = [f'{ccy}_{field}' for field, ccy in wide.columns]
            wide = wide.sort_index().ffill()
            print(f"  [OK] Rates from DB: {len(wide)} rows, {wide.shape[1]} columns")
            return wide
        except Exception as e:
            print(f"[WARN] Could not read {RATES_TABLE}: {e}")
            return pd.DataFrame()

    def get_intermarket_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Intermarket relationships that affect forex, indexed by trading_date.

        Columns: gold/oil/copper closes + returns, commodity index, EM-FX index,
        risk_on_sentiment. Prefers the shared `forex_intermarket_daily` table;
        falls back to yfinance for gold/oil/copper so training still gets a signal.

        NOTE: signature changed from a single-date dict to a date-range DataFrame
        so it merges on date like get_market_sentiment_data.
        """
        # 1. Preferred: shared DB table
        if self.use_db:
            try:
                conn = self._get_connection()
                if self._table_exists(conn, INTERMARKET_TABLE):
                    query = f"""
                    SELECT trading_date, gold_close, oil_close, copper_close,
                           commodity_index, em_currency_index, risk_on_sentiment
                    FROM dbo.{INTERMARKET_TABLE}
                    WHERE trading_date >= ? AND trading_date <= ?
                    ORDER BY trading_date
                    """
                    df = pd.read_sql(query, conn, params=[start_date, end_date],
                                     parse_dates=['trading_date'])
                    conn.close()
                    if not df.empty:
                        df = df.set_index('trading_date').sort_index()
                        for c in ['gold_close', 'oil_close', 'copper_close']:
                            if c in df.columns:
                                base = c.replace('_close', '')
                                df[f'{base}_return_1d'] = df[c].pct_change()
                                df[f'{base}_return_5d'] = df[c].pct_change(5)
                        print(f"  [OK] Intermarket from DB: {len(df)} rows")
                        return df
                else:
                    conn.close()
            except Exception as e:
                print(f"[WARN] Could not read {INTERMARKET_TABLE}: {e}; trying yfinance")

        # 2. Fallback: yfinance for commodities
        return self._get_intermarket_from_yfinance(start_date, end_date)

    def _get_intermarket_from_yfinance(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fallback commodity/intermarket proxies from yfinance (never raises)."""
        tickers = {'gold': 'GC=F', 'oil': 'CL=F', 'copper': 'HG=F'}
        out = {}
        for name, ticker in tickers.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if data is None or data.empty or 'Close' not in data.columns:
                    continue
                close = data['Close']
                # Newer yfinance can return a 1-col DataFrame for Close — squeeze it
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = pd.Series(close).astype(float)
                if close.empty:
                    continue
                out[f'{name}_close'] = close
                out[f'{name}_return_1d'] = close.pct_change()
                out[f'{name}_return_5d'] = close.pct_change(5)
            except Exception as e:
                print(f"[WARN] intermarket {name} download failed: {e}")
        if not out:
            return pd.DataFrame()
        try:
            df = pd.DataFrame(out)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        except Exception as e:
            print(f"[WARN] intermarket assembly failed: {e}")
            return pd.DataFrame()

    def get_econ_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Daily macro-event flags per currency, indexed by trading_date.

        Reads `forex_econ_events` (event_date, ccy, event_type, importance) and
        derives, per date, per-currency flags the model can consume:
          <CCY>_is_cb_meeting, <CCY>_is_cpi, <CCY>_is_nfp, <CCY>_event_importance,
          <CCY>_days_to_cb_meeting
        Returns empty DataFrame when the table/DB is unavailable.
        """
        if not self.use_db:
            return pd.DataFrame()
        try:
            conn = self._get_connection()
            if not self._table_exists(conn, ECON_EVENTS_TABLE):
                conn.close()
                print(f"[INFO] {ECON_EVENTS_TABLE} not present — skipping event features")
                return pd.DataFrame()

            query = f"""
            SELECT event_date, ccy, event_type, importance
            FROM dbo.{ECON_EVENTS_TABLE}
            WHERE event_date >= ? AND event_date <= ?
            ORDER BY event_date
            """
            ev = pd.read_sql(query, conn, params=[start_date, end_date],
                             parse_dates=['event_date'])
            conn.close()
            if ev.empty:
                return pd.DataFrame()

            ev['event_type'] = ev['event_type'].str.upper()
            frames = []
            for ccy, grp in ev.groupby('ccy'):
                g = grp.set_index('event_date')
                daily = pd.DataFrame(index=g.index.unique())
                daily[f'{ccy}_is_cb_meeting'] = g['event_type'].eq('CB_MEETING').groupby(level=0).max().astype(int)
                daily[f'{ccy}_is_cpi'] = g['event_type'].eq('CPI').groupby(level=0).max().astype(int)
                daily[f'{ccy}_is_nfp'] = g['event_type'].eq('NFP').groupby(level=0).max().astype(int)
                daily[f'{ccy}_event_importance'] = g['importance'].groupby(level=0).max()
                frames.append(daily)

            if not frames:
                return pd.DataFrame()
            out = pd.concat(frames, axis=1).sort_index().fillna(0)
            print(f"  [OK] Econ events from DB: {len(out)} dated rows")
            return out
        except Exception as e:
            print(f"[WARN] Could not read {ECON_EVENTS_TABLE}: {e}")
            return pd.DataFrame()
    
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