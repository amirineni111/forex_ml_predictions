#!/usr/bin/env python3
"""
Run forex ML predictions for all available currency pairs
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import ForexSQLServerConnection

def get_all_forex_pairs():
    """Get all forex pairs from database"""
    try:
        conn = ForexSQLServerConnection()
        if not conn.test_connection():
            print("❌ Database connection failed")
            return []
        
        pairs = conn.get_forex_pairs()
        conn.close()
        return pairs
        
    except Exception as e:
        print(f"Error getting forex pairs: {e}")
        return []

def run_prediction_for_pair(currency_pair, use_existing_model=True):
    """Run prediction for a specific currency pair"""
    
    print(f"\n{'='*60}")
    print(f"🔄 Processing {currency_pair}")
    print(f"{'='*60}")
    
    try:
        # Build command
        cmd = [
            "python", 
            "predict_forex_signals.py", 
            "--currency-pair", currency_pair,
            "--export-db"
        ]
        
        # Add train-new flag only for first pair or if specifically requested
        if not use_existing_model:
            cmd.append("--train-new")
        
        # Run the prediction
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {currency_pair} completed successfully in {duration:.1f}s")
            # Print key output lines
            lines = result.stdout.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ['generated', 'exported', 'signals', 'records']):
                    print(f"   📊 {line.strip()}")
            return True
        else:
            print(f"❌ {currency_pair} failed:")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {currency_pair} timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ {currency_pair} error: {e}")
        return False

def main():
    """Run predictions for all forex pairs"""
    
    print("🚀 Forex ML Batch Prediction System")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Get all forex pairs
    pairs = get_all_forex_pairs()
    
    if not pairs:
        print("❌ No forex pairs found in database")
        return
    
    print(f"📊 Found {len(pairs)} forex pairs to process:")
    for i, pair in enumerate(pairs, 1):
        print(f"   {i:2}. {pair}")
    
    # Track results
    successful_pairs = []
    failed_pairs = []
    start_time = time.time()
    
    # Process each pair
    for i, pair in enumerate(pairs):
        print(f"\n📈 Progress: {i+1}/{len(pairs)} pairs")
        
        # Use existing model for subsequent pairs to speed up processing
        use_existing = i > 0
        
        if run_prediction_for_pair(pair, use_existing_model=use_existing):
            successful_pairs.append(pair)
        else:
            failed_pairs.append(pair)
        
        # Small delay between pairs
        time.sleep(2)
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"⏰ Total time: {total_time/60:.1f} minutes")
    print(f"✅ Successful: {len(successful_pairs)}/{len(pairs)} pairs")
    
    if successful_pairs:
        print(f"\n✅ Successfully processed pairs:")
        for pair in successful_pairs:
            print(f"   • {pair}")
    
    if failed_pairs:
        print(f"\n❌ Failed pairs:")
        for pair in failed_pairs:
            print(f"   • {pair}")
    
    print(f"\n🎉 Batch processing completed!")
    print(f"💾 Check database tables: forex_ml_predictions, forex_model_performance")

if __name__ == "__main__":
    main()