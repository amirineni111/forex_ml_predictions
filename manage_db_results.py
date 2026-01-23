"""
Database Results Management Script

This script provides utilities for managing forex prediction results in SQL Server.
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.export_results import ForexResultsExporter

def safe_print(text):
    """Print text with safe encoding handling."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', errors='ignore').decode('ascii')
        print(safe_text)


def setup_tables():
    """Setup the results tables in SQL Server."""
    safe_print("🔧 Setting up results tables in SQL Server...")
    
    exporter = ForexResultsExporter()
    
    if exporter.create_results_tables():
        safe_print("✅ Results tables created successfully!")
        safe_print("📊 Created tables:")
        safe_print("   - forex_ml_predictions")
        safe_print("   - forex_model_performance") 
        safe_print("   - forex_daily_summary")
    else:
        safe_print("❌ Failed to create results tables")


def view_recent_predictions(currency_pair=None, days=7):
    """View recent predictions from the database."""
    safe_print(f"📊 Viewing recent predictions (last {days} days)...")
    
    exporter = ForexResultsExporter()
    
    try:
        df = exporter.get_recent_predictions(currency_pair=currency_pair, days_back=days)
        
        if df.empty:
            safe_print("📭 No recent predictions found")
            return
        
        safe_print(f"📈 Found {len(df)} recent predictions")
        
        if currency_pair:
            safe_print(f"💱 Currency Pair: {currency_pair}")
        else:
            pairs = df['currency_pair'].unique()
            safe_print(f"💱 Currency Pairs: {', '.join(pairs)}")
        
        # Show sample of recent predictions
        safe_print("\n📋 Recent Predictions Sample:")
        safe_print("-" * 80)
        
        display_cols = ['prediction_date', 'currency_pair', 'close_price', 'predicted_signal', 'signal_confidence']
        available_cols = [col for col in display_cols if col in df.columns]
        
        sample_df = df[available_cols].head(10)
        safe_print(sample_df.to_string(index=False))
        
        # Show signal distribution
        safe_print(f"\n📊 Signal Distribution:")
        signal_counts = df['predicted_signal'].value_counts()
        for signal, count in signal_counts.items():
            safe_print(f"   {signal}: {count} ({count/len(df)*100:.1f}%)")
        
    except Exception as e:
        safe_print(f"❌ Error retrieving predictions: {e}")


def view_model_performance(model_name=None, currency_pair=None):
    """View model performance history from the database."""
    safe_print("📊 Viewing model performance history...")
    
    exporter = ForexResultsExporter()
    
    try:
        df = exporter.get_model_performance_history(
            model_name=model_name, 
            currency_pair=currency_pair
        )
        
        if df.empty:
            safe_print("📭 No model performance history found")
            return
        
        safe_print(f"📈 Found {len(df)} performance records")
        
        if model_name:
            safe_print(f"🤖 Model: {model_name}")
        if currency_pair:
            safe_print(f"💱 Currency Pair: {currency_pair}")
        
        # Show performance summary
        safe_print("\n📋 Model Performance History:")
        safe_print("-" * 100)
        
        display_cols = ['training_date', 'model_name', 'currency_pair', 'cv_accuracy_mean', 'train_f1', 'training_samples']
        available_cols = [col for col in display_cols if col in df.columns]
        
        safe_print(df[available_cols].to_string(index=False))
        
        # Show best performing models
        if 'cv_accuracy_mean' in df.columns:
            safe_print(f"\n🏆 Best Performing Models by Accuracy:")
            best_models = df.nlargest(5, 'cv_accuracy_mean')[['model_name', 'currency_pair', 'cv_accuracy_mean']]
            safe_print(best_models.to_string(index=False))
        
    except Exception as e:
        safe_print(f"❌ Error retrieving performance history: {e}")


def cleanup_old_predictions(days_to_keep=30):
    """Clean up old prediction records from the database."""
    safe_print(f"🗑️ Cleaning up predictions older than {days_to_keep} days...")
    
    exporter = ForexResultsExporter()
    
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        engine = exporter.db.get_sqlalchemy_engine()
        
        with engine.connect() as conn:
            # Delete old predictions
            delete_sql = f"""
            DELETE FROM {exporter.predictions_table} 
            WHERE prediction_date < '{cutoff_date.strftime('%Y-%m-%d')}'
            """
            
            result = conn.execute(delete_sql)
            deleted_count = result.rowcount
            conn.commit()
        
        safe_print(f"✅ Cleaned up {deleted_count} old prediction records")
        
    except Exception as e:
        safe_print(f"❌ Error cleaning up predictions: {e}")


def main():
    """Main function for database results management."""
    parser = argparse.ArgumentParser(description='Forex Database Results Management')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup tables command
    setup_parser = subparsers.add_parser('setup', help='Setup results tables in SQL Server')
    
    # View predictions command
    view_pred_parser = subparsers.add_parser('predictions', help='View recent predictions')
    view_pred_parser.add_argument('--currency-pair', type=str, help='Filter by currency pair')
    view_pred_parser.add_argument('--days', type=int, default=7, help='Number of days to look back')
    
    # View performance command
    view_perf_parser = subparsers.add_parser('performance', help='View model performance history')
    view_perf_parser.add_argument('--model-name', type=str, help='Filter by model name')
    view_perf_parser.add_argument('--currency-pair', type=str, help='Filter by currency pair')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old predictions')
    cleanup_parser.add_argument('--days-to-keep', type=int, default=30, help='Days of data to keep')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    safe_print("💱 Forex Database Results Management")
    safe_print("=" * 50)
    
    if args.command == 'setup':
        setup_tables()
        
    elif args.command == 'predictions':
        view_recent_predictions(
            currency_pair=args.currency_pair,
            days=args.days
        )
        
    elif args.command == 'performance':
        view_model_performance(
            model_name=args.model_name,
            currency_pair=args.currency_pair
        )
        
    elif args.command == 'cleanup':
        cleanup_old_predictions(days_to_keep=args.days_to_keep)


if __name__ == "__main__":
    main()