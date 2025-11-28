#!/usr/bin/env python3
"""
Check the results of batch forex predictions in database
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.database.connection import ForexSQLServerConnection
import pandas as pd

def check_prediction_results():
    """Check how many predictions were generated for each currency pair"""
    
    try:
        conn = ForexSQLServerConnection()
        if not conn.test_connection():
            print("❌ Database connection failed")
            return
        
        print("🔍 Checking Forex ML Prediction Results")
        print("="*60)
        
        # Query to get prediction counts by currency pair
        query = """
        SELECT 
            currency_pair,
            COUNT(*) as prediction_count,
            COUNT(DISTINCT prediction_date) as unique_dates,
            MIN(prediction_date) as earliest_prediction,
            MAX(prediction_date) as latest_prediction,
            SUM(CASE WHEN predicted_signal = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
            SUM(CASE WHEN predicted_signal = 'SELL' THEN 1 ELSE 0 END) as sell_signals,
            SUM(CASE WHEN predicted_signal = 'HOLD' THEN 1 ELSE 0 END) as hold_signals
        FROM forex_ml_predictions 
        GROUP BY currency_pair
        ORDER BY currency_pair
        """
        
        results_df = pd.read_sql(query, conn.get_sqlalchemy_engine())
        
        if results_df.empty:
            print("❌ No prediction results found in forex_ml_predictions table")
            return
        
        print(f"📊 Found predictions for {len(results_df)} currency pairs:\n")
        
        # Display results
        for _, row in results_df.iterrows():
            pair = row['currency_pair']
            count = row['prediction_count']
            dates = row['unique_dates']
            buy = row['buy_signals']
            sell = row['sell_signals']
            hold = row['hold_signals']
            
            print(f"💱 {pair:8} | {count:3} predictions | {dates:2} dates | BUY: {buy:2} | SELL: {sell:2} | HOLD: {hold:2}")
        
        # Summary statistics
        total_predictions = results_df['prediction_count'].sum()
        total_pairs = len(results_df)
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total predictions: {total_predictions}")
        print(f"   Currency pairs: {total_pairs}")
        print(f"   Average per pair: {total_predictions/total_pairs:.1f}")
        
        # Check model performance table
        print(f"\n🎯 Model Performance Summary:")
        perf_query = """
        SELECT 
            currency_pair,
            COUNT(*) as model_count,
            MAX(training_timestamp) as latest_training
        FROM forex_model_performance 
        GROUP BY currency_pair
        ORDER BY currency_pair
        """
        
        perf_df = pd.read_sql(perf_query, conn.get_sqlalchemy_engine())
        
        if not perf_df.empty:
            print(f"   Models trained for {len(perf_df)} currency pairs")
            for _, row in perf_df.iterrows():
                print(f"   📊 {row['currency_pair']}: {row['model_count']} models trained")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking results: {e}")

if __name__ == "__main__":
    check_prediction_results()