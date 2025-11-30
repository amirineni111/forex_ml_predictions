"""
Debug script to check MACD values and test calculation
"""
import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import ForexSQLServerConnection
from models.ml_models import ForexMLModelManager

def debug_macd():
    print("🔍 Debug: MACD Values Analysis")
    print("=" * 50)
    
    # Initialize database connection
    db = ForexSQLServerConnection()
    
    # Fetch AUDUSD data
    df = db.get_forex_data_for_training(currency_pair='AUDUSD', days_back=60, min_records=50)
    
    if df.empty:
        print("❌ No data retrieved")
        return
        
    print(f"📊 Retrieved {len(df)} records for AUDUSD")
    
    # Check MACD values in raw data
    macd_cols = ['macd', 'macd_signal', 'macd_histogram']
    existing_macd_cols = [col for col in macd_cols if col in df.columns]
    
    print(f"\n📊 MACD columns in raw data: {existing_macd_cols}")
    
    if 'macd' in df.columns:
        macd_values = df['macd'].fillna(0)
        print(f"📊 MACD values - Count: {len(macd_values)}")
        print(f"📊 MACD values - Sum: {macd_values.sum()}")
        print(f"📊 MACD values - Abs Sum: {macd_values.abs().sum()}")
        print(f"📊 MACD values - Max Abs: {macd_values.abs().max()}")
        print(f"📊 MACD values - Non-zero count: {(macd_values != 0).sum()}")
        
        print(f"\n📊 Last 5 MACD values:")
        print(df[['date_time', 'close_price', 'macd']].tail())
    else:
        print("❌ No 'macd' column in raw data")
    
    # Test feature preparation
    print(f"\n🔧 Testing feature preparation...")
    model_manager = ForexMLModelManager()
    df_features = model_manager.prepare_forex_features(df)
    
    if 'macd' in df_features.columns:
        macd_after = df_features['macd'].fillna(0)
        print(f"📊 After feature prep - MACD Sum: {macd_after.sum()}")
        print(f"📊 After feature prep - MACD Abs Sum: {macd_after.abs().sum()}")
        print(f"📊 After feature prep - MACD Max Abs: {macd_after.abs().max()}")
        
        print(f"\n📊 Last 5 MACD values after feature prep:")
        print(df_features[['close_price', 'macd', 'macd_signal', 'macd_histogram']].tail())
    
if __name__ == "__main__":
    debug_macd()