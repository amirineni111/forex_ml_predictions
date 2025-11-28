#!/usr/bin/env python3
"""
Test the updated Forex ML connection with real data from SQL Server views
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import ForexSQLServerConnection
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_updated_connection():
    """Test the updated connection with real forex data"""
    
    try:
        print("=== Testing Updated Forex ML Connection ===")
        
        # Create connection
        conn = ForexSQLServerConnection()
        
        # Test basic connection
        print("\n1. Testing database connection...")
        if conn.test_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            return False
        
        # Test forex pairs
        print("\n2. Getting available forex pairs...")
        pairs = conn.get_forex_pairs()
        print(f"✅ Found {len(pairs)} forex pairs: {pairs[:5]}...")  # Show first 5
        
        if pairs:
            # Test data fetch with indicators for first pair
            test_pair = pairs[0]  # Use first available pair
            print(f"\n3. Testing data fetch with indicators for {test_pair}...")
            
            df = conn.get_forex_data_with_indicators(test_pair, limit=5)
            
            if not df.empty:
                print(f"✅ Successfully fetched {len(df)} records with indicators")
                print(f"\nColumns: {list(df.columns)}")
                print(f"\nSample data:")
                print(df[['currency_pair', 'date_time', 'close_price', 'rsi_14', 'macd', 'sma_20']].head())
                
                # Check for technical indicators
                print(f"\n4. Validating technical indicators...")
                indicators_check = {
                    'RSI': not df['rsi_14'].isna().all(),
                    'MACD': not df['macd'].isna().all(),
                    'SMA_20': not df['sma_20'].isna().all(),
                    'BB_Upper': not df['bb_upper'].isna().all(),
                    'ATR': not df['atr_14'].isna().all()
                }
                
                for indicator, has_data in indicators_check.items():
                    status = "✅" if has_data else "⚠️"
                    print(f"{status} {indicator}: {'Available' if has_data else 'Missing/Null'}")
                
                print(f"\n5. Testing ML model preparation...")
                # Test ML features
                ml_features = [
                    'close_price', 'rsi_14', 'macd', 'macd_signal', 
                    'sma_20', 'sma_50', 'ema_20', 'ema_50',
                    'bb_upper', 'bb_lower', 'atr_14',
                    'price_above_sma20', 'price_above_sma50', 'ema_bullish', 'macd_bullish'
                ]
                
                available_features = [col for col in ml_features if col in df.columns]
                print(f"✅ Available ML features: {len(available_features)}/{len(ml_features)}")
                print(f"Features: {available_features}")
                
                return True
            else:
                print(f"❌ No data returned for {test_pair}")
                return False
        else:
            print("❌ No forex pairs found")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    success = test_updated_connection()
    print(f"\n=== Test {'PASSED' if success else 'FAILED'} ===")