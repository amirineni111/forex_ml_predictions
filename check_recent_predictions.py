"""
Check most recent predictions with model version details
"""
import pyodbc
from datetime import datetime

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
    
    print("="*80)
    print("MOST RECENT FOREX PREDICTIONS")
    print("="*80)
    
    # Get most recent predictions (top 10)
    query = """
    SELECT TOP 10
        prediction_date,
        currency_pair,
        predicted_signal,
        signal_confidence,
        model_name,
        model_version
    FROM forex_ml_predictions
    ORDER BY prediction_date DESC
    """
    
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    
    if rows:
        print(f"\n📊 Found {len(rows)} most recent predictions:\n")
        for row in rows:
            print(f"Date: {row.prediction_date}")
            print(f"  Pair: {row.currency_pair:8} | Signal: {row.predicted_signal:4} | Conf: {row.signal_confidence:.3f}")
            print(f"  Model: {row.model_name} | Version: {row.model_version}")
            print()
    else:
        print("\n❌ No predictions found in table")
    
    # Check distinct model versions
    query2 = """
    SELECT DISTINCT 
        model_version,
        COUNT(*) as count,
        MAX(prediction_date) as last_date
    FROM forex_ml_predictions
    GROUP BY model_version
    ORDER BY last_date DESC
    """
    
    print("\n" + "="*80)
    print("MODEL VERSIONS IN DATABASE")
    print("="*80)
    cursor.execute(query2)
    rows = cursor.fetchall()
    
    for row in rows:
        print(f"\nVersion: {row.model_version}")
        print(f"  Count: {row.count} predictions")
        print(f"  Last used: {row.last_date}")
    
    conn.close()
    print("\n" + "="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
