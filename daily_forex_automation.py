"""
Daily Forex Automation Script

Automates daily forex signal generation, model training, and reporting.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import schedule
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import ForexSQLServerConnection
from database.export_results import ForexResultsExporter

# Setup logging
log_dir = Path('./logs')
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'./logs/forex_automation_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def safe_print(text):
    """Print text with safe encoding handling."""
    try:
        print(text)
        logger.info(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', errors='ignore').decode('ascii')
        print(safe_text)
        logger.info(safe_text)


class ForexDailyAutomation:
    """Handles daily forex ML automation tasks."""
    
    def __init__(self):
        """Initialize the automation system."""
        self.db = ForexSQLServerConnection()
        self.results_exporter = ForexResultsExporter(self.db)
        self.reports_dir = Path('./daily_reports')
        self.reports_dir.mkdir(exist_ok=True)
        
    def drop_previous_prediction_tables(self):
        """Drop previous forex prediction tables to start fresh."""
        try:
            safe_print("🗑️ Dropping previous prediction tables...")
            
            # Tables to drop
            tables_to_drop = [
                'forex_ml_predictions',
                'forex_daily_summary', 
                'forex_model_performance'
            ]
            
            engine = self.db.get_sqlalchemy_engine()
            
            for table in tables_to_drop:
                try:
                    with engine.connect() as conn:
                        conn.execute(f"DROP TABLE IF EXISTS {table}")
                        conn.commit()
                    safe_print(f"✅ Dropped table: {table}")
                except Exception as e:
                    safe_print(f"⚠️ Table {table} not found or already dropped: {e}")
                    
            safe_print("✅ Previous prediction tables dropped successfully")
            return True
            
        except Exception as e:
            safe_print(f"❌ Error dropping tables: {e}")
            return False
        
    def check_data_availability(self):
        """Check if forex data is available for today."""
        safe_print("🔄 Checking forex data availability...")
        
        try:
            # Get forex pairs
            pairs = self.db.get_forex_pairs()
            safe_print(f"📊 Found {len(pairs)} forex pairs")
            
            # Check recent data for each pair
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            data_status = {}
            
            for pair in pairs[:10]:  # Check top 10 pairs
                df = self.db.get_forex_data_with_indicators(
                    currency_pair=pair,
                    start_date=yesterday,
                    limit=10
                )
                
                data_status[pair] = {
                    'records': len(df),
                    'latest_date': df['date_time'].max() if not df.empty else None
                }
            
            # Report data status
            safe_print("📈 Data Status:")
            for pair, status in data_status.items():
                safe_print(f"  {pair}: {status['records']} records, latest: {status['latest_date']}")
            
            return data_status
            
        except Exception as e:
            safe_print(f"❌ Error checking data: {e}")
            return {}
    
    def check_data_freshness(self):
        """Check if the latest data is available before running predictions"""
        try:
            safe_print("🔍 Checking data freshness...")
            
            # Check a sample currency pair to see latest data
            test_query = """
            SELECT 
                MAX(trading_date) as latest_date,
                COUNT(*) as record_count
            FROM forex_hist_data 
            WHERE trading_date >= CAST(GETDATE() - 1 AS DATE)  -- Check for yesterday/today
            """
            
            df = pd.read_sql_query(test_query, self.db.get_sqlalchemy_engine())
            
            if not df.empty:
                latest_date = df['latest_date'].iloc[0]
                record_count = df['record_count'].iloc[0]
                
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                
                if latest_date:
                    latest_date_obj = pd.to_datetime(latest_date).date()
                    days_old = (today - latest_date_obj).days
                    
                    safe_print(f"📅 Latest forex data: {latest_date} ({record_count} records)")
                    
                    if days_old <= 1:
                        safe_print("✅ Data is fresh (within 1 day)")
                        return True
                    else:
                        safe_print(f"⚠️ Data is {days_old} days old - predictions may use stale data")
                        return True  # Still proceed but warn user
                else:
                    safe_print("⚠️ No recent data found")
                    return False
            else:
                safe_print("❌ No data available")
                return False
                
        except Exception as e:
            safe_print(f"⚠️ Could not check data freshness: {e}")
            return True  # Proceed anyway if check fails

    def run_daily_predictions(self):
        """Run daily forex predictions for all major pairs."""
        safe_print("🔄 Running daily forex predictions...")
        
        try:
            # Check if fresh data is available
            if not self.check_data_freshness():
                safe_print("⚠️ Proceeding with available data...")
            
            # Drop previous prediction tables to start fresh
            self.drop_previous_prediction_tables()
            
            # Import prediction module
            sys.path.append('.')
            from predict_forex_signals import ForexTradingSignalPredictor
            
            # Get all available pairs from database for enhanced signal strength analysis
            available_pairs = self.db.get_forex_pairs()
            pairs_to_analyze = available_pairs
            
            safe_print(f"📊 Processing all {len(pairs_to_analyze)} available currency pairs with enhanced signal strength features")
            
            safe_print(f"💱 Analyzing pairs: {', '.join(pairs_to_analyze)}")
            
            all_predictions = []
            
            for pair in pairs_to_analyze:
                try:
                    safe_print(f"📊 Processing {pair}...")
                    
                    # Initialize predictor for this pair using enhanced model
                    predictor = ForexTradingSignalPredictor(
                        model_path='./data/best_forex_model.joblib',
                        currency_pair=pair
                    )
                    
                    # Generate predictions
                    predictions = predictor.predict_signals(currency_pair=pair)
                    
                    if not predictions.empty:
                        predictions['processed_date'] = datetime.now()
                        all_predictions.append(predictions)
                        safe_print(f"✅ Generated {len(predictions)} signals for {pair}")
                    else:
                        safe_print(f"⚠️ No predictions generated for {pair}")
                        
                except Exception as e:
                    safe_print(f"❌ Error processing {pair}: {e}")
            
            # Combine all predictions
            if all_predictions:
                import pandas as pd
                combined_predictions = pd.concat(all_predictions, ignore_index=True)
                
                # Export to database
                safe_print("📊 Exporting predictions to SQL Server...")
                self.results_exporter.create_results_tables()
                db_success = self.results_exporter.export_predictions(
                    combined_predictions, 
                    model_name='daily_automation_model'
                )
                
                if db_success:
                    # Export daily summary
                    self.results_exporter.export_daily_summary(
                        combined_predictions,
                        model_name='daily_automation_model'
                    )
                    safe_print("✅ Predictions exported to database")
                
                # Save daily report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = self.reports_dir / f"forex_signals_{timestamp}.csv"
                combined_predictions.to_csv(report_file, index=False)
                
                safe_print(f"📄 Daily report saved: {report_file}")
                
                # Generate summary
                self.generate_summary_report(combined_predictions)
                
                return True
            else:
                safe_print("❌ No predictions generated for any pairs")
                return False
                
        except Exception as e:
            safe_print(f"❌ Error in daily predictions: {e}")
            return False
    
    def generate_summary_report(self, predictions_df):
        """Generate a summary report of predictions."""
        try:
            safe_print("📄 Generating summary report...")
            
            # Signal distribution
            signal_summary = predictions_df.groupby(['currency_pair', 'signal']).size().unstack(fill_value=0)
            
            # Recent signals
            recent_signals = predictions_df.sort_values('date_time').tail(20)
            
            # Create summary report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = self.reports_dir / f"forex_summary_{timestamp}.txt"
            
            with open(summary_file, 'w') as f:
                f.write(f"Forex Trading Signals Summary\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Signal Distribution by Currency Pair:\n")
                f.write("-" * 40 + "\n")
                f.write(signal_summary.to_string())
                f.write("\n\n")
                
                f.write("Recent Signals:\n")
                f.write("-" * 20 + "\n")
                cols = ['date_time', 'currency_pair', 'close_price', 'signal']
                if 'confidence' in recent_signals.columns:
                    cols.append('confidence')
                display_cols = [col for col in cols if col in recent_signals.columns]
                f.write(recent_signals[display_cols].to_string(index=False))
                
            safe_print(f"📋 Summary report saved: {summary_file}")
            
        except Exception as e:
            safe_print(f"❌ Error generating summary: {e}")
    
    def retrain_models_weekly(self):
        """Retrain models weekly with fresh data."""
        safe_print("🔄 Starting weekly model retraining...")
        
        try:
            # Import prediction module
            sys.path.append('.')
            from predict_forex_signals import ForexTradingSignalPredictor
            
            # Use EURUSD as the representative pair to retrain the shared enhanced model
            pairs = self.db.get_forex_pairs()
            representative_pair = 'EURUSD' if 'EURUSD' in pairs else pairs[0]
            
            safe_print(f"🔄 Retraining shared enhanced model using {representative_pair} data...")
            
            try:
                predictor = ForexTradingSignalPredictor(
                    model_path='./data/best_forex_model.joblib',
                    currency_pair=representative_pair
                )
                
                # Get extended historical data for retraining with signal strength features
                safe_print("📊 Fetching extended historical data with signal strength indicators...")
                forex_data = predictor.get_forex_data(currency_pair=representative_pair, days_back=200)
                
                if not forex_data.empty:
                    safe_print("🤖 Training enhanced model with signal strength features...")
                    success = predictor.train_model(forex_data, signal_type='trend')
                    if success:
                        safe_print("✅ Enhanced forex model retrained successfully for all currency pairs")
                    else:
                        safe_print("❌ Failed to retrain enhanced model")
                else:
                    safe_print(f"⚠️ No data available for {representative_pair} retraining")
                    
            except Exception as e:
                safe_print(f"❌ Error retraining enhanced model: {e}")
            
            safe_print("✅ Weekly model retraining completed")
            
        except Exception as e:
            safe_print(f"❌ Error in weekly retraining: {e}")
    
    def cleanup_old_reports(self, days_to_keep=30):
        """Clean up old report files."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            deleted_count = 0
            for report_file in self.reports_dir.glob("*.csv"):
                if report_file.stat().st_mtime < cutoff_date.timestamp():
                    report_file.unlink()
                    deleted_count += 1
            
            for report_file in self.reports_dir.glob("*.txt"):
                if report_file.stat().st_mtime < cutoff_date.timestamp():
                    report_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                safe_print(f"🗑️ Cleaned up {deleted_count} old report files")
                
        except Exception as e:
            safe_print(f"❌ Error cleaning up files: {e}")


def run_daily_job():
    """Run the daily automation job."""
    safe_print("🚀 Starting daily forex automation job...")
    
    automation = ForexDailyAutomation()
    
    # Check data availability
    data_status = automation.check_data_availability()
    
    # Run daily predictions
    success = automation.run_daily_predictions()
    
    # Cleanup old files
    automation.cleanup_old_reports()
    
    if success:
        safe_print("✅ Daily automation completed successfully")
    else:
        safe_print("⚠️ Daily automation completed with issues")


def run_weekly_job():
    """Run the weekly automation job."""
    safe_print("🚀 Starting weekly forex automation job...")
    
    automation = ForexDailyAutomation()
    
    # Retrain models
    automation.retrain_models_weekly()
    
    safe_print("✅ Weekly automation completed")


def main():
    """Main automation scheduler."""
    safe_print("💱 Forex Daily Automation Started")
    safe_print("=" * 50)
    
    # Schedule daily job at 7:00 AM
    schedule.every().day.at("07:00").do(run_daily_job)
    
    # Schedule weekly job on Sundays at 6:00 AM
    schedule.every().sunday.at("06:00").do(run_weekly_job)
    
    safe_print("⏰ Scheduled jobs:")
    safe_print("  - Daily predictions: 7:00 AM")
    safe_print("  - Weekly retraining: Sunday 6:00 AM")
    
    # Run immediately for testing
    if len(sys.argv) > 1 and sys.argv[1] == '--run-now':
        safe_print("🔄 Running daily job immediately...")
        run_daily_job()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--retrain-now':
        safe_print("🔄 Running weekly job immediately...")
        run_weekly_job()
        
    else:
        # Keep running and check for scheduled jobs
        safe_print("⏳ Waiting for scheduled jobs... (Ctrl+C to stop)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            safe_print("👋 Automation stopped by user")


if __name__ == "__main__":
    main()