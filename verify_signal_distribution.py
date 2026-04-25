"""
Quick script to verify the current signal distribution in the database
"""
import pyodbc
import pandas as pd

# Connection string
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.86.28,1444;"
    "DATABASE=stockdata_db;"
    "UID=remote_user;"
    "PWD=YourStrongPassword123!;"
    "TrustServerCertificate=yes;"
)

try:
    conn = pyodbc.connect(conn_str)
    
    # Check signal distribution for last 50 predictions per currency pair
    query = """
    WITH RankedPredictions AS (
        SELECT 
            currency_pair,
            prediction_date,
            predicted_signal,
            ROW_NUMBER() OVER (PARTITION BY currency_pair ORDER BY prediction_date DESC) as rn
        FROM forex_ml_predictions
    )
    SELECT 
        currency_pair,
        predicted_signal,
        COUNT(*) as count
    FROM RankedPredictions
    WHERE rn <= 50
    GROUP BY currency_pair, predicted_signal
    ORDER BY currency_pair, predicted_signal
    """
    
    df = pd.read_sql(query, conn)
    
    if df.empty:
        print("No predictions found in database")
    else:
        print("\n" + "="*80)
        print("SIGNAL DISTRIBUTION - LAST 50 PREDICTIONS PER CURRENCY PAIR")
        print("="*80)
        
        # Calculate distribution per pair
        for pair in df['currency_pair'].unique():
            pair_data = df[df['currency_pair'] == pair]
            total = pair_data['count'].sum()
            
            print(f"\n{pair}:")
            print(f"  Total predictions: {total}")
            
            for _, row in pair_data.iterrows():
                signal = row['predicted_signal']
                count = row['count']
                pct = (count / total * 100) if total > 0 else 0
                print(f"    {signal}: {count} ({pct:.1f}%)")
        
        # Overall distribution
        print("\n" + "="*80)
        print("OVERALL SIGNAL DISTRIBUTION")
        print("="*80)
        overall = df.groupby('predicted_signal')['count'].sum()
        total_all = overall.sum()
        
        for signal, count in overall.items():
            pct = (count / total_all * 100) if total_all > 0 else 0
            print(f"  {signal}: {count} ({pct:.1f}%)")
        
        print(f"\n  Total predictions analyzed: {total_all}")
        print("="*80)
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
