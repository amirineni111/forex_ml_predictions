"""
Check today's signal distribution from the new 3-class model
"""
import pandas as pd
import pyodbc
from datetime import datetime

print("="*80)
print("TODAY'S FOREX PREDICTIONS - SIGNAL DISTRIBUTION")
print("="*80)

try:
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=192.168.86.28,1444;"
        "DATABASE=stockdata_db;"
        "UID=remote_user;"
        "PWD=YourStrongPassword123!;"
        "TrustServerCertificate=yes;"
    )
    
    conn = pyodbc.connect(conn_str)
    
    # Get today's predictions
    query = """
    SELECT 
        currency_pair,
        prediction_date,
        predicted_signal,
        signal_confidence,
        prob_buy,
        prob_sell,
        prob_hold
    FROM forex_ml_predictions
    WHERE CAST(prediction_date AS DATE) = CAST(GETDATE() AS DATE)
    ORDER BY currency_pair
    """
    
    df = pd.read_sql(query, conn)
    
    if len(df) > 0:
        print(f"\n✅ Found {len(df)} predictions for TODAY ({datetime.now().date()})\n")
        
        # Signal distribution
        print("SIGNAL DISTRIBUTION:")
        print("-" * 80)
        signal_dist = df['predicted_signal'].value_counts()
        for signal, count in signal_dist.items():
            pct = (count / len(df)) * 100
            print(f"  {signal}: {count} ({pct:.1f}%)")
        
        print("\n\nDETAILED PREDICTIONS:")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['currency_pair']:8} | {row['predicted_signal']:4} | "
                  f"Conf: {row['signal_confidence']:.3f} | "
                  f"BUY:{row['prob_buy']:.3f} SELL:{row['prob_sell']:.3f} HOLD:{row['prob_hold']:.3f}")
        
    else:
        print(f"\n❌ No predictions found for TODAY ({datetime.now().date()})")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
