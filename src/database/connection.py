"""
Forex Database Connection Module

This module provides specialized database connection utilities for forex trading data,
including access to technical indicators like BB, EMA, SMA, RSI, MACD, ATR from SQL Server.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import quote_plus
from datetime import datetime, timedelta

import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ForexSQLServerConnection:
    """
    Specialized SQL Server connection class for forex trading data.
    
    Handles connections to forex databases containing technical indicators
    and provides methods to fetch forex-specific data from existing views.
    """
    
    def __init__(
        self,
        server: str = "localhost\\MSSQLSERVER01",
        database: str = "stockdata_db", 
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver: str = "ODBC Driver 17 for SQL Server",
        trusted_connection: bool = True
    ):
        """
        Initialize the Forex SQL Server connection.
        
        Args:
            server: SQL Server instance name or IP address
            database: Database name containing forex data
            username: SQL Server username
            password: SQL Server password
            driver: ODBC driver name
            trusted_connection: Use Windows authentication if True
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.driver = driver
        self.trusted_connection = trusted_connection
        
        self._engine: Optional[Engine] = None
        self.connection = None
        
    def connect(self):
        """Establish database connection."""
        try:
            if self.trusted_connection:
                connection_string = (
                    f"DRIVER={{{self.driver}}};"
                    f"SERVER={self.server};"
                    f"DATABASE={self.database};"
                    f"Trusted_Connection=yes;"
                )
            else:
                connection_string = (
                    f"DRIVER={{{self.driver}}};"
                    f"SERVER={self.server};"
                    f"DATABASE={self.database};"
                    f"UID={self.username};"
                    f"PWD={self.password};"
                )
            
            self.connection = pyodbc.connect(connection_string)
            logger.info(f"Successfully connected to {self.server}\\{self.database}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def get_sqlalchemy_engine(self) -> Engine:
        """Create and return a SQLAlchemy engine."""
        if self._engine:
            return self._engine
            
        if self.trusted_connection:
            connection_string = (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
            )
        else:
            connection_string = (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
            )
        
        encoded_connection_string = quote_plus(connection_string)
        engine_url = f"mssql+pyodbc:///?odbc_connect={encoded_connection_string}"
        
        self._engine = create_engine(
            engine_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        return self._engine
    
    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            if not self.connection:
                self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_forex_pairs(self) -> List[str]:
        """
        Get list of available forex pairs from the database.
        
        Returns:
            List of forex pair symbols
        """
        query = """
        SELECT DISTINCT symbol as currency_pair 
        FROM forex_hist_data
        WHERE symbol IS NOT NULL
        ORDER BY symbol
        """
        
        try:
            df = pd.read_sql(query, self.get_sqlalchemy_engine())
            return df['currency_pair'].tolist()
        except Exception as e:
            logger.error(f"Error fetching forex pairs: {e}")
            return []
    
    def get_forex_data_with_indicators(self, currency_pair: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch forex data with technical indicators for a specific currency pair using existing SQL Server views.
        
        Args:
            currency_pair (str): Currency pair (e.g., 'EURUSD', 'GBPUSD')
            limit (int, optional): Number of records to fetch (latest first)
            
        Returns:
            pd.DataFrame: DataFrame with forex data and indicators
        """
        try:
            # Base query with limit
            limit_clause = f"TOP {limit}" if limit else ""
            
            # Comprehensive query using your existing forex views with signal strength indicators
            query = f"""
            SELECT {limit_clause}
                h.symbol as currency_pair,
                h.trading_date as date_time,
                h.open_price,
                h.high_price,
                h.low_price,
                h.close_price,
                h.volume,
                
                -- Basic price features
                (h.high_price - h.low_price) as daily_range,
                ((h.close_price - h.open_price) / NULLIF(h.open_price, 0) * 100) as daily_return,
                
                -- RSI indicators with signal strength
                ISNULL(rsi.RSI, 50) as rsi_14,
                ISNULL(rsi_signals.rsi_trade_signal, 'Neutral') as rsi_signal_strength,
                CASE WHEN rsi.RSI > 70 THEN 'Overbought'
                     WHEN rsi.RSI < 30 THEN 'Oversold'
                     ELSE 'Neutral' END as rsi_signal,
                     
                -- MACD indicators with signal strength
                ISNULL(macd.EMA_12, h.close_price) as ema_12,
                ISNULL(macd.EMA_26, h.close_price) as ema_26,
                ISNULL(macd.MACD, 0) as macd,
                ISNULL(macd.Signal_Line, 0) as macd_signal,
                ISNULL(macd.MACD_Signal, 'Neutral') as macd_signal_strength,
                ISNULL(macd_signals.MACD_Signal, 'Neutral') as macd_trade_signal,
                
                -- EMA/SMA indicators with signal flags
                ISNULL(ema.SMA_200, h.close_price) as sma_200,
                ISNULL(ema.SMA_100, h.close_price) as sma_100,
                ISNULL(ema.SMA_50, h.close_price) as sma_50,
                ISNULL(ema.SMA_20, h.close_price) as sma_20,
                ISNULL(ema.EMA_200, h.close_price) as ema_200,
                ISNULL(ema.EMA_100, h.close_price) as ema_100,
                ISNULL(ema.EMA_50, h.close_price) as ema_50,
                ISNULL(ema.EMA_20, h.close_price) as ema_20,
                -- SMA/EMA signal strength flags
                ISNULL(ema.SMA_200_Flag, 'Neutral') as sma_200_signal,
                ISNULL(ema.SMA_100_Flag, 'Neutral') as sma_100_signal,
                ISNULL(ema.SMA_50_Flag, 'Neutral') as sma_50_signal,
                ISNULL(ema.SMA_20_Flag, 'Neutral') as sma_20_signal,
                ISNULL(ema.EMA_200_Flag, 'Neutral') as ema_200_signal,
                ISNULL(ema.EMA_100_Flag, 'Neutral') as ema_100_signal,
                ISNULL(ema.EMA_50_Flag, 'Neutral') as ema_50_signal,
                ISNULL(ema.EMA_20_Flag, 'Neutral') as ema_20_signal,
                ISNULL(sma_signals.sma_trade_signal, 'Neutral') as sma_trade_signal,
                
                -- Bollinger Bands with signal strength
                ISNULL(bb.SMA_20, h.close_price) as bb_middle,
                ISNULL(bb.Upper_Band, h.close_price * 1.02) as bb_upper,
                ISNULL(bb.Lower_Band, h.close_price * 0.98) as bb_lower,
                ISNULL(bb_signals.bb_trade_signal, 'Neutral') as bb_signal_strength,
                
                -- ATR for volatility with signal strength
                ISNULL(atr.ATR_14, (h.high_price - h.low_price)) as atr_14,
                ISNULL(atr_spikes.atr_volatility_signal, 'Normal') as atr_signal_strength,
                
                -- Additional features for ML
                CASE WHEN h.close_price > ISNULL(ema.SMA_20, h.close_price) THEN 1 ELSE 0 END as price_above_sma20,
                CASE WHEN h.close_price > ISNULL(ema.SMA_50, h.close_price) THEN 1 ELSE 0 END as price_above_sma50,
                CASE WHEN ISNULL(ema.EMA_20, h.close_price) > ISNULL(ema.EMA_50, h.close_price) THEN 1 ELSE 0 END as ema_bullish,
                CASE WHEN ISNULL(macd.MACD, 0) > ISNULL(macd.Signal_Line, 0) THEN 1 ELSE 0 END as macd_bullish
                
            FROM forex_hist_data h
            LEFT JOIN forex_RSI_calculation rsi ON h.symbol = rsi.symbol AND h.trading_date = rsi.trading_date
            LEFT JOIN forex_rsi_signals rsi_signals ON h.symbol = rsi_signals.symbol AND h.trading_date = rsi_signals.trading_date
            LEFT JOIN forex_macd macd ON h.symbol = macd.symbol AND h.trading_date = macd.trading_date
            LEFT JOIN forex_macd_signals macd_signals ON h.symbol = macd_signals.symbol AND h.trading_date = macd_signals.trading_date
            LEFT JOIN forex_ema_sma_view ema ON h.symbol = ema.symbol AND h.trading_date = ema.trading_date
            LEFT JOIN forex_sma_signals sma_signals ON h.symbol = sma_signals.symbol AND h.trading_date = sma_signals.trading_date
            LEFT JOIN forex_bollingerband bb ON h.symbol = bb.symbol AND h.trading_date = bb.trading_date
            LEFT JOIN forex_bb_signals bb_signals ON h.symbol = bb_signals.symbol AND h.trading_date = bb_signals.trading_date
            LEFT JOIN forex_atr atr ON h.symbol = atr.symbol AND h.trading_date = atr.trading_date
            LEFT JOIN forex_atr_spikes atr_spikes ON h.symbol = atr_spikes.symbol AND h.trading_date = atr_spikes.trading_date
            
            WHERE h.symbol = '{currency_pair}'
            ORDER BY h.trading_date DESC
            """
            
            # Execute query
            df = pd.read_sql_query(query, self.get_sqlalchemy_engine())
            
            # Convert date_time to datetime if needed
            if 'date_time' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'])
            
            logger.info(f"Fetched {len(df)} forex records with indicators for {currency_pair}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching forex data for {currency_pair}: {str(e)}")
            raise
    
    def get_latest_forex_data(self, currency_pair: str, limit: int = 100) -> pd.DataFrame:
        """
        Get the most recent forex data for a specific currency pair.
        
        Args:
            currency_pair: Currency pair symbol
            limit: Number of recent records to fetch
            
        Returns:
            DataFrame with recent forex data
        """
        return self.get_forex_data_with_indicators(
            currency_pair=currency_pair,
            limit=limit
        )
    
    def get_forex_data_for_training(
        self, 
        currency_pair: str,
        days_back: int = 365,
        min_records: int = 100,
        include_today: bool = True
    ) -> pd.DataFrame:
        """
        Get forex data suitable for ML model training.
        
        Args:
            currency_pair: Specific currency pair
            days_back: Number of days to look back
            min_records: Minimum number of records required
            include_today: Whether to include today's data if available
            
        Returns:
            DataFrame with training data
        """
        # For training, we get all data and let the model use what it needs
        df = self.get_forex_data_with_indicators(currency_pair=currency_pair)
        
        if include_today:
            # Ensure we have the latest possible data by checking today's date
            latest_date_query = f"""
            SELECT MAX(trading_date) as latest_date
            FROM forex_hist_data 
            WHERE symbol = '{currency_pair}'
            """
            
            try:
                latest_df = pd.read_sql_query(latest_date_query, self.get_sqlalchemy_engine())
                if not latest_df.empty and latest_df['latest_date'].iloc[0]:
                    latest_date = latest_df['latest_date'].iloc[0]
                    logger.info(f"Latest data available for {currency_pair}: {latest_date}")
            except Exception as e:
                logger.warning(f"Could not check latest date for {currency_pair}: {e}")
        
        if len(df) < min_records:
            logger.warning(f"Only {len(df)} records found, less than minimum {min_records}")
        
        return df
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query and return results as DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame with query results
        """
        try:
            return pd.read_sql(query, self.get_sqlalchemy_engine())
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def get_forex_table_info(self) -> Dict[str, Any]:
        """
        Get information about forex tables and views.
        
        Returns:
            Dictionary with table information
        """
        tables_query = """
        SELECT 
            TABLE_NAME,
            TABLE_TYPE,
            TABLE_SCHEMA
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME LIKE '%forex%'
        ORDER BY TABLE_NAME
        """
        
        try:
            tables_df = pd.read_sql(tables_query, self.get_sqlalchemy_engine())
            
            return {
                'tables': tables_df.to_dict('records')
            }
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {'tables': []}
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Convenience function to create a connection
def create_forex_connection(**kwargs) -> ForexSQLServerConnection:
    """
    Create a forex database connection with optional parameters.
    
    Returns:
        ForexSQLServerConnection instance
    """
    return ForexSQLServerConnection(**kwargs)