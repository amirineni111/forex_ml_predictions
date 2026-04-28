"""
Check which model names exist in forex_ml_predictions table
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
    cursor = conn.cursor()
    
    print("="*80)
    print("MODEL NAMES IN FOREX_ML_PREDICTIONS TABLE")
    print("="*80)
    
    # Check today's predictions
    query_today = """
    SELECT DISTINCT 
        model_name,
        model_version,
        COUNT(*) as count
    FROM forex_ml_predictions
    WHERE CAST(prediction_date AS DATE) = CAST(GETDATE() AS DATE)
    GROUP BY model_name, model_version
    """
    
    print(f"\n📅 TODAY'S PREDICTIONS ({datetime.now().date()}):")
    print("-" * 80)
    cursor.execute(query_today)
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"  Model: {row.model_name}")
            print(f"  Version: {row.model_version}")
            print(f"  Count: {row.count} predictions")
            print()
    else:
        print("  No predictions found for today")
    
    # Check all model names in history
    query_all = """
    SELECT 
        model_name,
        COUNT(*) as total_count,
        MIN(prediction_date) as first_date,
        MAX(prediction_date) as last_date
    FROM forex_ml_predictions
    GROUP BY model_name
    ORDER BY last_date DESC
    """
    
    print("\n📊 ALL MODEL NAMES IN HISTORY:")
    print("-" * 80)
    cursor.execute(query_all)
    rows = cursor.fetchall()
    for row in rows:
        print(f"  Model: {row.model_name}")
        print(f"  Total predictions: {row.total_count}")
        print(f"  Date range: {row.first_date} to {row.last_date}")
        print()
    
    conn.close()
    print("="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
