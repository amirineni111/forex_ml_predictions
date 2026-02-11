"""
Daily Forex Automation Script - Enhanced for Direction Accuracy

Key improvements:
- Model drift detection and adaptive retraining
- Performance monitoring with accuracy tracking
- Automatic retraining when accuracy drops below threshold
- Walk-forward model validation before deployment
- Health checks and diagnostics
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
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
    """
    Handles daily forex ML automation tasks.
    
    Enhanced with:
    - Model drift detection
    - Adaptive retraining triggers
    - Performance monitoring
    - Walk-forward validation
    """
    
    # Accuracy threshold below which automatic retraining is triggered
    RETRAIN_ACCURACY_THRESHOLD = 0.50
    # Minimum number of predictions needed to evaluate accuracy
    MIN_PREDICTIONS_FOR_EVAL = 20
    
    def __init__(self):
        """Initialize the automation system."""
        self.db = ForexSQLServerConnection()
        self.results_exporter = ForexResultsExporter(self.db)
        
    def drop_previous_prediction_tables(self):
        """Clean previous forex prediction data for today (preserves historical backfill data).
        
        Instead of DROP TABLE (which destroys outcome tracking columns and backfill data),
        we delete today's rows and drop/recreate only the summary and performance tables.
        The forex_ml_predictions table schema is preserved so that:
        - backfill_strategy1_outcomes.py can populate actual_return_* and direction_correct_* columns
        - vw_strategy2_unified_ml_predictions view stays valid (no binding errors)
        """
        try:
            safe_print("[INFO] Cleaning previous prediction data...")
            
            engine = self.db.get_sqlalchemy_engine()
            from sqlalchemy import text
            
            # For forex_ml_predictions: DELETE today's rows only (preserve history + backfill data)
            try:
                with engine.connect() as conn:
                    result = conn.execute(text(
                        "DELETE FROM forex_ml_predictions WHERE CAST(prediction_date AS date) = CAST(GETDATE() AS date)"
                    ))
                    conn.commit()
                    safe_print(f"[OK] Cleaned forex_ml_predictions: removed {result.rowcount} rows for today")
            except Exception as e:
                # Table might not exist yet on first run -- that's OK, create_results_tables will handle it
                safe_print(f"[WARN] forex_ml_predictions cleanup: {e}")
            
            # For summary and performance tables: safe to drop and recreate (no backfill dependency)
            tables_to_drop = [
                'forex_daily_summary', 
                'forex_model_performance'
            ]
            
            for table in tables_to_drop:
                try:
                    with engine.connect() as conn:
                        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                        conn.commit()
                    safe_print(f"[OK] Dropped table: {table}")
                except Exception as e:
                    safe_print(f"[WARN] Table {table} not found or already dropped: {e}")
                    
            safe_print("[OK] Previous prediction data cleaned successfully")
            return True
            
        except Exception as e:
            safe_print(f"[ERROR] Error cleaning prediction data: {e}")
            return False
        
    def check_data_availability(self):
        """Check if forex data is available for today."""
        safe_print("[PROCESSING] Checking forex data availability...")
        
        try:
            # Get forex pairs
            pairs = self.db.get_forex_pairs()
            safe_print(f"[DATA] Found {len(pairs)} forex pairs")
            
            # Check recent data for each pair
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            data_status = {}
            
            for pair in pairs[:10]:  # Check top 10 pairs
                df = self.db.get_forex_data_with_indicators(
                    currency_pair=pair,
                    limit=10
                )
                
                data_status[pair] = {
                    'records': len(df),
                    'latest_date': df['date_time'].max() if not df.empty else None
                }
            
            # Report data status
            safe_print("[DATA] Data Status:")
            for pair, status in data_status.items():
                safe_print(f"  {pair}: {status['records']} records, latest: {status['latest_date']}")
            
            return data_status
            
        except Exception as e:
            safe_print(f"[ERROR] Error checking data: {e}")
            return {}
    
    def check_data_freshness(self):
        """Check if the latest data is available before running predictions."""
        try:
            safe_print("[CHECK] Checking data freshness...")
            
            from sqlalchemy import text
            test_query = text("""
            SELECT 
                MAX(trading_date) as latest_date,
                COUNT(*) as record_count
            FROM forex_hist_data 
            WHERE trading_date >= CAST(GETDATE() - 1 AS DATE)
            """)
            
            df = pd.read_sql_query(test_query, self.db.get_sqlalchemy_engine())
            
            if not df.empty:
                latest_date = df['latest_date'].iloc[0]
                record_count = df['record_count'].iloc[0]
                
                today = datetime.now().date()
                
                if latest_date:
                    latest_date_obj = pd.to_datetime(latest_date).date()
                    days_old = (today - latest_date_obj).days
                    
                    safe_print(f"[DATA] Latest forex data: {latest_date} ({record_count} records)")
                    
                    if days_old <= 1:
                        safe_print("[OK] Data is fresh (within 1 day)")
                        return True
                    else:
                        safe_print(f"[WARN] Data is {days_old} days old - predictions may use stale data")
                        return True  # Still proceed but warn
                else:
                    safe_print("[WARN] No recent data found")
                    return False
            else:
                safe_print("[ERROR] No data available")
                return False
                
        except Exception as e:
            safe_print(f"[WARN] Could not check data freshness: {e}")
            return True  # Proceed anyway if check fails
    
    def check_model_drift(self) -> dict:
        """
        Check if model performance has degraded (model drift detection).
        
        Queries ai_prediction_history to see recent direction accuracy.
        Returns drift status and recommendations.
        """
        safe_print("[CHECK] Checking model drift...")
        
        drift_report = {
            'drift_detected': False,
            'current_accuracy': None,
            'should_retrain': False,
            'details': {}
        }
        
        try:
            from sqlalchemy import text
            
            # Check recent Forex prediction accuracy
            query = text("""
            SELECT 
                model_name,
                COUNT(*) as total_predictions,
                SUM(CAST(direction_correct AS INT)) as correct_predictions,
                ROUND(100.0 * SUM(CAST(direction_correct AS INT)) / NULLIF(COUNT(*), 0), 2) as accuracy_pct,
                AVG(ABS(percentage_error)) as avg_abs_pct_error
            FROM ai_prediction_history 
            WHERE market = 'Forex' 
              AND direction_correct IS NOT NULL
              AND prediction_date >= DATEADD(day, -7, GETDATE())
            GROUP BY model_name
            ORDER BY accuracy_pct DESC
            """)
            
            df = pd.read_sql_query(query, self.db.get_sqlalchemy_engine())
            
            if df.empty or len(df) == 0:
                safe_print("[INFO] No recent prediction history to evaluate drift")
                return drift_report
            
            safe_print("[DATA] Recent Model Performance (last 7 days):")
            safe_print("  " + "-" * 60)
            
            overall_correct = 0
            overall_total = 0
            
            for _, row in df.iterrows():
                model = row['model_name']
                total = row['total_predictions']
                correct = row['correct_predictions']
                accuracy = row['accuracy_pct']
                avg_error = row['avg_abs_pct_error']
                
                overall_correct += correct
                overall_total += total
                
                status = ""
                if accuracy < 45:
                    status = " [POOR - Below random]"
                elif accuracy < 50:
                    status = " [WEAK - Near random]"
                elif accuracy > 55:
                    status = " [GOOD]"
                
                safe_print(f"  {model:20s}: {accuracy:.1f}% ({correct}/{total}), "
                          f"Avg Error: {avg_error:.2f}%{status}")
            
            # Overall accuracy
            if overall_total >= self.MIN_PREDICTIONS_FOR_EVAL:
                overall_accuracy = (overall_correct / overall_total) * 100
                drift_report['current_accuracy'] = overall_accuracy
                
                safe_print(f"\n  Overall: {overall_accuracy:.1f}% ({overall_correct}/{overall_total})")
                
                if overall_accuracy < self.RETRAIN_ACCURACY_THRESHOLD * 100:
                    drift_report['drift_detected'] = True
                    drift_report['should_retrain'] = True
                    safe_print(f"\n[WARN] MODEL DRIFT DETECTED! Accuracy ({overall_accuracy:.1f}%) is below "
                              f"threshold ({self.RETRAIN_ACCURACY_THRESHOLD * 100}%)")
                    safe_print("[INFO] Automatic retraining will be triggered")
                else:
                    safe_print(f"\n[OK] Model performance is acceptable ({overall_accuracy:.1f}%)")
            else:
                safe_print(f"\n[INFO] Insufficient predictions ({overall_total}) for drift evaluation "
                          f"(need {self.MIN_PREDICTIONS_FOR_EVAL})")
            
            drift_report['details'] = df.to_dict('records')
            
        except Exception as e:
            safe_print(f"[WARN] Could not check model drift: {e}")
        
        return drift_report

    def run_daily_predictions(self):
        """
        Run daily forex predictions for all major pairs.
        
        Enhanced with model drift detection and adaptive retraining.
        """
        safe_print("[PROCESSING] Running daily forex predictions...")
        
        try:
            # Step 1: Check data freshness
            if not self.check_data_freshness():
                safe_print("[WARN] Proceeding with available data...")
            
            # Step 2: Check model drift - may trigger retraining
            drift_report = self.check_model_drift()
            
            if drift_report['should_retrain']:
                safe_print("[INFO] Drift detected - retraining model before predictions...")
                self.retrain_models_weekly()
            
            # Step 3: Drop previous prediction tables to start fresh
            self.drop_previous_prediction_tables()
            
            # Step 4: Import prediction module
            sys.path.append('.')
            from predict_forex_signals import ForexTradingSignalPredictor
            
            # Get all available pairs from database
            available_pairs = self.db.get_forex_pairs()
            pairs_to_analyze = available_pairs
            
            safe_print(f"[DATA] Processing {len(pairs_to_analyze)} currency pairs")
            safe_print(f"[FOREX] Pairs: {', '.join(pairs_to_analyze)}")
            
            all_predictions = []
            all_features = []
            
            for pair in pairs_to_analyze:
                try:
                    safe_print(f"[DATA] Processing {pair}...")
                    
                    # Initialize predictor for this pair
                    predictor = ForexTradingSignalPredictor(
                        model_path='./data/best_forex_model.joblib',
                        currency_pair=pair
                    )
                    
                    # Get feature data for export
                    df = predictor.get_forex_data(currency_pair=pair, days_back=100)
                    df_features, available_features = predictor.prepare_features(df)
                    df_recent = df_features.dropna(subset=available_features).tail(1)
                    
                    if not df_recent.empty:
                        feature_record = df_recent[available_features + ['currency_pair']].copy()
                        all_features.append(feature_record)
                    
                    # Generate FORWARD-LOOKING prediction for next trading day
                    from src.utils.forward_prediction import predict_next_day_signals
                    predictions = predict_next_day_signals(predictor, pair)
                    
                    if not predictions.empty:
                        predictions['processed_date'] = datetime.now()
                        all_predictions.append(predictions)
                        
                        signal = predictions['predicted_signal'].iloc[0]
                        confidence = predictions['signal_confidence'].iloc[0]
                        safe_print(f"[OK] {pair}: {signal} (confidence: {confidence:.3f})")
                    else:
                        safe_print(f"[WARN] No prediction generated for {pair}")
                        
                except Exception as e:
                    safe_print(f"[ERROR] Error processing {pair}: {e}")
            
            # Combine and export predictions
            if all_predictions:
                combined_predictions = pd.concat(all_predictions, ignore_index=True)
                
                # Combine features
                if all_features:
                    combined_features = pd.concat(all_features, ignore_index=True)
                    safe_print(f"[DATA] Collected {len(combined_features)} feature records")
                else:
                    combined_features = pd.DataFrame()
                
                # Export to database
                safe_print("[DATA] Exporting predictions to SQL Server...")
                self.results_exporter.create_results_tables()
                db_success = self.results_exporter.export_predictions(
                    combined_predictions, 
                    model_name='daily_automation_model'
                )
                
                # Export feature values
                if not combined_features.empty:
                    safe_print("[DATA] Exporting feature values to SQL Server...")
                    feature_success = self.results_exporter.export_prediction_features(
                        combined_features, 
                        combined_predictions
                    )
                    if feature_success:
                        safe_print("[OK] Feature values exported to database")
                    else:
                        safe_print("[WARN] Could not export feature values")
                
                if db_success:
                    self.results_exporter.export_daily_summary(
                        combined_predictions,
                        model_name='daily_automation_model'
                    )
                    safe_print("[OK] Predictions exported to database")
                
                # Log summary
                self._log_prediction_summary(combined_predictions)
                
                return True
            else:
                safe_print("[ERROR] No predictions generated for any pairs")
                return False
                
        except Exception as e:
            safe_print(f"[ERROR] Error in daily predictions: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _log_prediction_summary(self, predictions_df):
        """Log a summary of today's predictions."""
        safe_print("\n[SUMMARY] Prediction Summary:")
        safe_print("  " + "-" * 50)
        
        if 'predicted_signal' in predictions_df.columns:
            signal_counts = predictions_df['predicted_signal'].value_counts()
            for signal, count in signal_counts.items():
                safe_print(f"  {signal}: {count} pairs")
        
        if 'signal_confidence' in predictions_df.columns:
            avg_conf = predictions_df['signal_confidence'].mean()
            high_conf = (predictions_df['signal_confidence'] > 0.65).sum()
            safe_print(f"  Average confidence: {avg_conf:.3f}")
            safe_print(f"  High-confidence signals (>65%): {high_conf}")
    
    def retrain_models_weekly(self):
        """
        Retrain models weekly with fresh data from ALL currency pairs.
        
        Enhanced with:
        - Walk-forward validation before deployment
        - Binary direction prediction for better accuracy
        - Feature selection to reduce noise
        - Model stability checks
        """
        safe_print("[PROCESSING] Starting model retraining with ALL currency pairs...")
        
        try:
            # Try to use the enhanced trainer first
            sys.path.append('.')
            
            try:
                from train_enhanced_model import EnhancedForexTrainer
                
                safe_print("[INFO] Using enhanced training pipeline...")
                trainer = EnhancedForexTrainer()
                
                # Test database connection
                if not trainer.db.test_connection():
                    safe_print("[ERROR] Database connection failed")
                    return
                
                # Get ALL available currency pairs
                pairs = self.db.get_forex_pairs()
                safe_print(f"[DATA] Found {len(pairs)} currency pairs for retraining")
                
                # Prepare enhanced dataset
                enhanced_data = trainer.prepare_enhanced_dataset(pairs, lookback_days=800)
                
                # Train with binary direction prediction (UP/DOWN)
                results = trainer.train_enhanced_models(
                    test_size=0.2, 
                    use_binary_direction=True
                )
                
                if results:
                    safe_print(f"[OK] Enhanced model retrained: {results['best_model_name']}")
                    
                    # Export performance
                    try:
                        model_results = results.get('results', {})
                        perf_data = {}
                        for name, res in model_results.items():
                            if isinstance(res, dict) and 'cv_accuracy_mean' in res:
                                perf_data[name] = {
                                    'cv_accuracy_mean': res['cv_accuracy_mean'],
                                    'cv_accuracy_std': res.get('cv_accuracy_std', 0),
                                    'train_accuracy': res.get('train_accuracy', 0),
                                    'train_precision': 0,
                                    'train_recall': 0,
                                    'train_f1': 0
                                }
                        
                        if perf_data:
                            self.results_exporter.export_model_performance(
                                model_results=perf_data,
                                model_name='enhanced_direction_model',
                                training_pairs=pairs,
                                training_date=datetime.now()
                            )
                            safe_print("[OK] Model performance exported to database")
                    except Exception as e:
                        safe_print(f"[WARN] Could not export model performance: {e}")
                    
                    return
                else:
                    safe_print("[WARN] Enhanced training produced no results, falling back...")
                    
            except ImportError as e:
                safe_print(f"[WARN] Enhanced trainer not available ({e}), using standard training...")
            except Exception as e:
                safe_print(f"[WARN] Enhanced training failed ({e}), falling back to standard...")
            
            # Fallback: Standard training
            from predict_forex_signals import ForexTradingSignalPredictor
            
            pairs = self.db.get_forex_pairs()
            safe_print(f"[DATA] Retraining with {len(pairs)} currency pairs (standard pipeline)")
            
            all_forex_data = []
            
            for pair in pairs:
                try:
                    predictor = ForexTradingSignalPredictor(
                        model_path='./data/best_forex_model.joblib',
                        currency_pair=pair
                    )
                    
                    forex_data = predictor.get_forex_data(currency_pair=pair, days_back=300)
                    
                    if not forex_data.empty:
                        forex_data['currency_pair'] = pair
                        all_forex_data.append(forex_data)
                        safe_print(f"[OK] {pair}: {len(forex_data)} records")
                        
                except Exception as e:
                    safe_print(f"[ERROR] Error collecting data for {pair}: {e}")
                    continue
            
            if all_forex_data:
                combined_data = pd.concat(all_forex_data, ignore_index=True)
                safe_print(f"[DATA] Combined: {len(combined_data)} records from {len(all_forex_data)} pairs")
                
                # Train with direction signal type for better accuracy
                main_predictor = ForexTradingSignalPredictor(
                    model_path='./data/best_forex_model.joblib',
                    currency_pair=pairs[0]
                )
                
                success = main_predictor.train_model(
                    combined_data, 
                    signal_type='direction',  # Binary direction for better accuracy
                    future_periods=1           # 1-day ahead
                )
                
                if success:
                    safe_print("[OK] Model retrained successfully")
                else:
                    safe_print("[ERROR] Failed to retrain model")
            else:
                safe_print("[ERROR] No data collected from any currency pairs")
            
            safe_print("[OK] Weekly model retraining completed")
            
        except Exception as e:
            safe_print(f"[ERROR] Error in weekly retraining: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup_old_reports(self, days_to_keep=30):
        """Clean up old report files."""
        try:
            reports_dir = Path('./logs')
            if not reports_dir.exists():
                return
                
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            deleted_count = 0
            for report_file in reports_dir.glob("*.log"):
                if report_file.stat().st_mtime < cutoff_date.timestamp():
                    report_file.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                safe_print(f"[INFO] Cleaned up {deleted_count} old log files")
                
        except Exception as e:
            safe_print(f"[ERROR] Error cleaning up files: {e}")


def run_daily_job():
    """Run the daily automation job."""
    safe_print("=" * 60)
    safe_print("[INFO] Starting daily forex automation job...")
    safe_print(f"[INFO] Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 60)
    
    automation = ForexDailyAutomation()
    
    # Check data availability
    data_status = automation.check_data_availability()
    
    # Run daily predictions (includes drift detection)
    success = automation.run_daily_predictions()
    
    # Cleanup old files
    automation.cleanup_old_reports()
    
    if success:
        safe_print("\n[OK] Daily automation completed successfully")
    else:
        safe_print("\n[WARN] Daily automation completed with issues")


def run_weekly_job():
    """Run the weekly automation job."""
    safe_print("=" * 60)
    safe_print("[INFO] Starting weekly forex automation job...")
    safe_print(f"[INFO] Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("=" * 60)
    
    automation = ForexDailyAutomation()
    
    # Retrain models with enhanced pipeline
    automation.retrain_models_weekly()
    
    safe_print("\n[OK] Weekly automation completed")


def main():
    """Main automation scheduler."""
    safe_print("[FOREX] Forex Daily Automation - Enhanced v3.0")
    safe_print("=" * 60)
    
    # Schedule daily job at 7:00 AM
    schedule.every().day.at("07:00").do(run_daily_job)
    
    # Schedule weekly job on Sundays at 6:00 AM
    schedule.every().sunday.at("06:00").do(run_weekly_job)
    
    safe_print("[INFO] Scheduled jobs:")
    safe_print("  - Daily predictions: 7:00 AM (with drift detection)")
    safe_print("  - Weekly retraining: Sunday 6:00 AM (enhanced pipeline)")
    
    # Run immediately for testing
    if len(sys.argv) > 1 and sys.argv[1] == '--run-now':
        safe_print("[PROCESSING] Running daily job immediately...")
        run_daily_job()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--retrain-now':
        safe_print("[PROCESSING] Running weekly retraining immediately...")
        run_weekly_job()
        
    elif len(sys.argv) > 1 and sys.argv[1] == '--check-drift':
        safe_print("[CHECK] Checking model drift...")
        automation = ForexDailyAutomation()
        drift_report = automation.check_model_drift()
        if drift_report['drift_detected']:
            safe_print("\n[WARN] Drift detected! Run --retrain-now to fix.")
        else:
            safe_print("\n[OK] No significant drift detected.")
        
    else:
        # Keep running and check for scheduled jobs
        safe_print("[INFO] Waiting for scheduled jobs... (Ctrl+C to stop)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            safe_print("[INFO] Automation stopped by user")


if __name__ == "__main__":
    main()
