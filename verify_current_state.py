"""
Quick verification script to check current model state and recent predictions
"""
import joblib
import pandas as pd
import pyodbc
from datetime import datetime, timedelta

print("="*80)
print("FOREX ML SYSTEM - CURRENT STATE VERIFICATION")
print("="*80)

# 1. Check model files
print("\n[1] CHECKING MODEL FILES")
print("-" * 80)
import os
from pathlib import Path

model_files = [
    'data/enhanced_forex_model.joblib',
    'data/best_forex_model.joblib'
]

for model_path in model_files:
    if os.path.exists(model_path):
        stat = os.stat(model_path)
        modified = datetime.fromtimestamp(stat.st_mtime)
        print(f"✅ {model_path}")
        print(f"   Last Modified: {modified}")
        print(f"   Size: {stat.st_size / 1024:.2f} KB")
        
        # Load and check classes
        try:
            model_data = joblib.load(model_path)
            if 'label_encoder' in model_data:
                classes = model_data['label_encoder'].classes_
                print(f"   Classes: {classes} ({len(classes)} classes)")
            print()
        except Exception as e:
            print(f"   ❌ Error loading: {e}\n")
    else:
        print(f"❌ {model_path} - NOT FOUND\n")

# 2. Check recent predictions from database
print("\n[2] CHECKING RECENT PREDICTIONS FROM DATABASE")
print("-" * 80)

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
    
    # Get predictions from last 7 days
    query = """
    SELECT 
        prediction_date,
        currency_pair,
        predicted_signal,
        signal_confidence,
        model_name
    FROM forex_ml_predictions
    WHERE prediction_date >= DATEADD(day, -7, GETDATE())
    ORDER BY prediction_date DESC, currency_pair
    """
    
    df = pd.read_sql(query, conn)
    
    if len(df) > 0:
        print(f"Found {len(df)} predictions from last 7 days\n")
        
        # Signal distribution
        print("Signal Distribution:")
        signal_dist = df['predicted_signal'].value_counts()
        for signal, count in signal_dist.items():
            pct = (count / len(df)) * 100
            print(f"  {signal}: {count} ({pct:.1f}%)")
        
        print("\nMost Recent Predictions (last 10):")
        print(df.head(10).to_string(index=False))
        
        # Check if SELL bias exists
        sell_pct = (signal_dist.get('SELL', 0) / len(df)) * 100
        if sell_pct > 75:
            print(f"\n⚠️  WARNING: SELL BIAS DETECTED ({sell_pct:.1f}%)")
        else:
            print(f"\n✅ Signal distribution looks balanced")
    else:
        print("❌ No predictions found in last 7 days")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Database error: {e}")

# 3. Check configuration files
print("\n[3] CHECKING CONFIGURATION IN KEY FILES")
print("-" * 80)

files_to_check = [
    ('daily_forex_automation.py', 556, 'signal_type'),
    ('predict_forex_signals.py', 307, 'signal_type'),
    ('predict_forex_signals.py', 444, 'signal_type'),
    ('predict_forex_signals.py', 594, 'default'),
    ('train_enhanced_model.py', 272, 'use_binary_direction'),
    ('train_enhanced_model.py', 692, 'use_binary_direction'),
]

for filepath, line_num, check_type in files_to_check:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if line_num <= len(lines):
                line_content = lines[line_num - 1].strip()
                print(f"{filepath}:{line_num}")
                print(f"  {line_content}")
                
                # Check for correct values
                if check_type == 'signal_type':
                    if "'trend'" in line_content or '"trend"' in line_content:
                        print("  ✅ Correct: using 'trend'")
                    elif "'direction'" in line_content or '"direction"' in line_content:
                        print("  ❌ WRONG: using 'direction' (should be 'trend')")
                    else:
                        print("  ⚠️  Cannot determine signal_type value")
                elif check_type == 'use_binary_direction':
                    if 'False' in line_content:
                        print("  ✅ Correct: use_binary_direction=False")
                    elif 'True' in line_content:
                        print("  ❌ WRONG: use_binary_direction=True (should be False)")
                elif check_type == 'default':
                    if "'trend'" in line_content:
                        print("  ✅ Correct: default='trend'")
                    elif "'direction'" in line_content:
                        print("  ❌ WRONG: default='direction' (should be 'trend')")
                print()
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}\n")

print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
