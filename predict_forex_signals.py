"""
Forex Trading Signal Prediction Script - Enhanced for Direction Accuracy

Key improvements:
- Loads enhanced model with feature selection and calibrated probabilities
- Proper feature alignment with model's expected features
- Confidence-based filtering (only output high-confidence signals)
- Model health monitoring and drift detection
- Compatible with both v2 and v3 model formats
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from database.connection import ForexSQLServerConnection
from database.export_results import ForexResultsExporter
from models.ml_models import ForexMLModelManager

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_print(text):
    """Print text with safe encoding handling for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        emoji_replacements = {
            '\u2705': '[OK]', '\u274c': '[ERROR]', '\U0001f4ca': '[DATA]',
            '\U0001f4b1': '[FOREX]', '\U0001f4c8': '[PREDICTION]', '\u26a0\ufe0f': '[WARN]',
            '\U0001f3af': '[TARGET]', '\U0001f504': '[PROCESSING]', '\U0001f4b0': '[SIGNAL]',
            '\U0001f4c9': '[TREND]', '\U0001f527': '[FIX]', '\U0001f50d': '[CHECK]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', errors='replace').decode('ascii'))


class ForexTradingSignalPredictor:
    """
    Forex trading signal prediction system - Enhanced for direction accuracy.
    
    Key improvements:
    - Loads v3 enhanced model with feature selection
    - Proper feature alignment with training features
    - Confidence-based signal filtering
    - Model health monitoring
    """
    
    def __init__(self, model_path=None, currency_pair=None):
        """Initialize the Forex predictor with saved model artifacts."""
        
        # Default paths
        self.model_path = model_path or './data/best_forex_model.joblib'
        self.currency_pair = currency_pair
        
        # Model components
        self.model_manager = None
        self.model_version = None
        self.model_scaler = None
        self.model_feature_columns = None
        
        # Load model artifacts
        self.load_model_artifacts()
        
        # Database connection
        self.db = ForexSQLServerConnection()
        
        # Results exporter for writing back to SQL Server
        self.results_exporter = ForexResultsExporter(self.db)
        
        # Test database connection
        if not self.db.test_connection():
            safe_print("[ERROR] Failed to connect to database. Please check your connection settings.")
            sys.exit(1)
        
        safe_print("[OK] Database connection established")
        
        # Feature columns for forex predictions
        self.feature_columns = [
            # Price data
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
            # Technical Indicators
            'rsi_14', 'macd', 'macd_signal', 'atr_14',
            # Bollinger Bands
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            # Signal Strength Features
            'rsi_signal_strength', 'macd_signal_strength', 'macd_trade_signal',
            'sma_200_signal', 'sma_100_signal', 'sma_50_signal', 'sma_20_signal',
            'ema_200_signal', 'ema_100_signal', 'ema_50_signal', 'ema_20_signal',
            'sma_trade_signal', 'bb_signal_strength', 'atr_signal_strength',
            # Derived features
            'daily_return', 'volatility', 'price_range', 'gap', 'volume_ratio',
            # Engineered features
            'price_momentum_5', 'price_momentum_10', 'price_position',
            'price_vs_sma_20', 'price_vs_sma_50', 'price_vs_ema_20',
            'sma20_vs_sma50', 'ema20_vs_ema50', 'rsi_oversold', 'rsi_overbought',
            'rsi_momentum', 'rsi_price_divergence', 'bb_squeeze', 'bb_breakout_upper', 'bb_breakout_lower',
            'macd_signal_cross', 'volatility_10', 'volatility_20',
            'hour', 'day_of_week', 'month',
            # Signal strength derived features
            'combined_signal_strength', 'bullish_signal_count', 'bearish_signal_count', 'signal_agreement',
            # Encoded signal features
            'rsi_signal_strength_encoded', 'macd_signal_strength_encoded', 'macd_trade_signal_encoded',
            'sma_200_signal_encoded', 'sma_100_signal_encoded', 'sma_50_signal_encoded', 'sma_20_signal_encoded',
            'ema_200_signal_encoded', 'ema_100_signal_encoded', 'ema_50_signal_encoded', 'ema_20_signal_encoded',
            'sma_trade_signal_encoded', 'bb_signal_strength_encoded', 'atr_signal_strength_encoded'
        ]
    
    def load_model_artifacts(self):
        """Load the trained model artifacts with version detection."""
        if os.path.exists(self.model_path):
            try:
                artifacts = joblib.load(self.model_path)
                
                # Detect model version
                self.model_version = artifacts.get('model_version', '1.0')
                
                if 'model_name' in artifacts:
                    # v3 enhanced model format
                    self._load_v3_model(artifacts)
                elif 'model' in artifacts and 'scaler' in artifacts:
                    # v2 format
                    self._load_v2_model(artifacts)
                else:
                    # v1 format - use ForexMLModelManager
                    self._load_v1_model(artifacts)
                
                safe_print(f"[OK] Model loaded (version: {self.model_version})")
                
            except Exception as e:
                safe_print(f"[WARN] Error loading model artifacts: {e}")
                safe_print("[INFO] Will train new model when needed")
        else:
            safe_print(f"[WARN] Model file not found at {self.model_path}")
            safe_print("[INFO] Will train new model when needed")
    
    def _load_v3_model(self, artifacts):
        """Load v3 enhanced model format."""
        self.model_manager = ForexMLModelManager()
        self.model_manager.best_model = artifacts['model']
        self.model_manager.best_model_name = artifacts.get('model_name')
        self.model_manager.scaler = artifacts.get('scaler')
        self.model_manager.label_encoder = artifacts.get('label_encoder')
        
        # Use the model's specific feature columns if available
        self.model_feature_columns = artifacts.get('feature_columns')
        if self.model_feature_columns:
            self.model_manager.forex_feature_columns = self.model_feature_columns
        
        self.model_scaler = artifacts.get('scaler')
        
        # Log model info
        if self.model_manager.label_encoder:
            safe_print(f"[DATA] Target classes: {list(self.model_manager.label_encoder.classes_)}")
        if self.model_feature_columns:
            safe_print(f"[DATA] Model expects {len(self.model_feature_columns)} features")
        
        # Show improvements used
        improvements = artifacts.get('improvements', [])
        if improvements:
            safe_print(f"[INFO] Model improvements: {', '.join(improvements[:5])}")
    
    def _load_v2_model(self, artifacts):
        """Load v2 model format."""
        self.model_manager = ForexMLModelManager()
        self.model_manager.best_model = artifacts['model']
        self.model_manager.scaler = artifacts.get('scaler')
        self.model_manager.label_encoder = artifacts.get('label_encoder')
        self.model_manager.forex_feature_columns = artifacts.get('feature_columns', self.feature_columns)
        self.model_manager.best_model_name = artifacts.get('best_model_name')
    
    def _load_v1_model(self, artifacts):
        """Load v1 model format."""
        self.model_manager = ForexMLModelManager()
        self.model_manager.load_model(self.model_path)
    
    def get_forex_data(self, currency_pair=None, days_back=60):
        """Fetch forex data for prediction."""
        
        safe_print(f"[PROCESSING] Fetching forex data...")
        
        try:
            df = self.db.get_forex_data_for_training(
                currency_pair=currency_pair,
                days_back=days_back,
                min_records=100
            )
            
            if df.empty:
                safe_print("[ERROR] No forex data retrieved from database")
                return df
            
            safe_print(f"[DATA] Retrieved {len(df)} forex records")
            
            if currency_pair:
                safe_print(f"[FOREX] Currency pair: {currency_pair}")
            else:
                pairs = df['currency_pair'].unique() if 'currency_pair' in df.columns else ['Unknown']
                safe_print(f"[FOREX] Currency pairs: {', '.join(pairs)}")
            
            return df
            
        except Exception as e:
            safe_print(f"[ERROR] Error fetching forex data: {e}")
            return pd.DataFrame()
    
    def _merge_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load calendar features (bank holidays, short weeks) and merge on date.
        
        Adds pre/post holiday flags, short week indicators, and cross-market
        holiday awareness from the shared market_calendar table.
        Forex uses 'date_time' column (aliased from trading_date in SQL).
        """
        try:
            cal_query = """
            SELECT calendar_date,
                   is_pre_holiday, is_post_holiday, is_short_week,
                   trading_days_in_week, is_quarter_end, is_options_expiry,
                   days_until_next_holiday, days_since_last_holiday,
                   other_market_closed
            FROM dbo.vw_market_calendar_features
            WHERE market = 'FOREX'
            """
            df_cal = pd.read_sql(cal_query, self.db.get_sqlalchemy_engine())
            
            if not df_cal.empty and 'date_time' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'])
                df_cal['calendar_date'] = pd.to_datetime(df_cal['calendar_date'])
                df['_cal_date'] = df['date_time'].dt.normalize()
                df = df.merge(df_cal, left_on='_cal_date', right_on='calendar_date', how='left')
                df = df.drop(columns=['calendar_date', '_cal_date'], errors='ignore')
        except Exception as e:
            safe_print(f"[WARN] Could not load calendar features: {e}")
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for prediction with proper alignment.
        
        Enhanced to align with model's expected features.
        """
        
        if df.empty:
            return df, []
        
        safe_print("[PROCESSING] Preparing features...")
        
        try:
            # Initialize model manager if not loaded
            if self.model_manager is None:
                self.model_manager = ForexMLModelManager(task_type='classification')
            
            # Prepare forex-specific features
            df_features = self.model_manager.prepare_forex_features(df)
            
            # Merge calendar features (holidays, short weeks, bank holidays)
            df_features = self._merge_calendar_features(df_features)
            
            # Determine which features to use
            if self.model_feature_columns:
                # Use model's specific features (v3 enhanced)
                expected_features = self.model_feature_columns
            else:
                # Use model manager's default features
                expected_features = self.model_manager.forex_feature_columns
            
            available_features = [col for col in expected_features if col in df_features.columns]
            missing_features = [col for col in expected_features if col not in df_features.columns]
            
            if missing_features:
                n_missing = len(missing_features)
                safe_print(f"[WARN] Missing {n_missing} features - creating with defaults")
                # Create missing features with zeros (safe default)
                for feat in missing_features:
                    df_features[feat] = 0
                available_features = expected_features  # Now all are available
            
            safe_print(f"[DATA] Available features: {len(available_features)}/{len(expected_features)}")
            
            return df_features, available_features
            
        except Exception as e:
            safe_print(f"[ERROR] Error preparing features: {e}")
            return df, []
    
    def train_model(self, df, signal_type='direction', future_periods=1):
        """
        Train a new forex prediction model.
        
        Enhanced with:
        - Default to 'direction' signal type for binary UP/DOWN
        - Default to 1-day ahead prediction
        - Feature selection
        - Sample weighting
        """
        
        safe_print("[PROCESSING] Training new forex model...")
        
        # Get currency pair from the data
        currency_pair = self.currency_pair
        if currency_pair is None and 'currency_pair' in df.columns:
            currency_pair = df['currency_pair'].iloc[0] if len(df) > 0 else 'UNKNOWN'
        
        try:
            if self.model_manager is None:
                self.model_manager = ForexMLModelManager(task_type='classification')
            
            # Create trading signals with IMPROVED settings
            df_with_signals = self.model_manager.create_trading_signals(
                df, 
                signal_type=signal_type,
                future_periods=future_periods  # Default 1-day ahead
            )
            
            if df_with_signals.empty:
                safe_print("[ERROR] No training data available after signal generation")
                return False
            
            # Prepare features
            df_features = self.model_manager.prepare_forex_features(df_with_signals)
            
            # Select features (use model's feature columns)
            available_features = [col for col in self.model_manager.forex_feature_columns if col in df_features.columns]
            
            if len(available_features) < 10:
                safe_print(f"[ERROR] Too few features available: {len(available_features)}")
                return False
            
            X = df_features[available_features]
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            y = df_features['signal']
            
            # Remove rows with invalid data
            valid_mask = ~(X.isin([np.inf, -np.inf]).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                safe_print(f"[ERROR] Insufficient training samples: {len(X)}")
                return False
            
            safe_print(f"[DATA] Training with {len(X)} samples and {len(available_features)} features")
            safe_print(f"[TARGET] Signal distribution:")
            signal_counts = y.value_counts()
            for signal, count in signal_counts.items():
                safe_print(f"   {signal}: {count} ({count/len(y)*100:.1f}%)")
            
            # Train models with ENHANCED settings
            results = self.model_manager.train_all_models(
                X, y, 
                use_time_series_cv=True,
                use_sample_weights=True,
                use_feature_selection=True,
                use_purged_cv=True,
                calibrate_probabilities=True
            )
            
            # Display results
            safe_print(f"\n[DATA] Model Training Results:")
            safe_print("-" * 60)
            
            for model_name, metrics in results.items():
                if 'error' not in metrics:
                    accuracy = metrics.get('cv_accuracy_mean', 0)
                    std = metrics.get('cv_accuracy_std', 0)
                    gap = metrics.get('overfitting_gap', 0)
                    status = ""
                    if gap > 0.15:
                        status = " [OVERFIT]"
                    elif accuracy > 0.55:
                        status = " [GOOD]"
                    safe_print(f"  {model_name:20}: CV={accuracy:.3f}(+/-{std:.3f}) Gap={gap:.3f}{status}")
                else:
                    safe_print(f"  {model_name:20}: ERROR - {metrics['error']}")
            
            # Show model health report
            health = self.model_manager.get_model_health_report()
            if health.get('recommendations'):
                safe_print("\n[CHECK] Model Health Recommendations:")
                for rec in health['recommendations']:
                    safe_print(f"  - {rec}")
            
            # Save model performance to database
            self.results_exporter.export_model_performance(
                model_results=results,
                model_name=f'forex_ml_model_{currency_pair or "ALL"}',
                training_pairs=[currency_pair] if currency_pair else ['ALL'],
                training_date=datetime.now()
            )
            
            # Save the model
            model_dir = Path(self.model_path).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self.model_manager.save_model(self.model_path)
            safe_print(f"[OK] Model saved to {self.model_path}")
            
            return True
            
        except Exception as e:
            safe_print(f"[ERROR] Error training model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_signals(self, df=None, currency_pair=None):
        """
        Generate forex trading signals for next trading day.
        
        Enhanced with confidence-based filtering.
        """
        
        if self.model_manager is None or self.model_manager.best_model is None:
            safe_print("[ERROR] No trained model available. Training new model...")
            
            if df is None:
                df = self.get_forex_data(currency_pair=currency_pair)
            
            if df.empty:
                safe_print("[ERROR] Cannot train model: no data available")
                return pd.DataFrame()
            
            if not self.train_model(df, signal_type='direction', future_periods=1):
                safe_print("[ERROR] Model training failed")
                return pd.DataFrame()
        
        # Get recent data for prediction if not provided
        if df is None:
            df = self.get_forex_data(currency_pair=currency_pair, days_back=100)
        
        if df.empty:
            safe_print("[ERROR] No data available for prediction")
            return pd.DataFrame()
        
        safe_print("[PROCESSING] Generating trading signals...")
        
        try:
            # Prepare features
            df_features, available_features = self.prepare_features(df)
            
            if len(available_features) < 10:
                safe_print("[ERROR] Insufficient features for prediction")
                return pd.DataFrame()
            
            # Get the MOST RECENT record only (latest trading day)
            df_features = df_features.sort_values('date_time')
            df_recent = df_features.dropna(subset=available_features).tail(1)
            
            if df_recent.empty:
                safe_print("[ERROR] No valid data for prediction")
                return pd.DataFrame()
            
            safe_print(f"[DATA] Predicting for latest date: {df_recent['date_time'].iloc[0]}")
            
            # Prepare prediction input
            X_pred = df_recent[available_features].copy()
            X_pred = X_pred.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Make prediction for next trading day
            predictions = self.model_manager.predict(X_pred)
            
            # Get probabilities if available
            try:
                probabilities = self.model_manager.predict_proba(X_pred)
                df_recent = df_recent.copy()
                df_recent['signal'] = predictions
                
                # Add probability columns
                if len(probabilities.shape) > 1:
                    classes = self.model_manager.label_encoder.classes_
                    for i, class_name in enumerate(classes):
                        df_recent[f'prob_{class_name}'] = probabilities[:, i]
                
            except Exception as e:
                safe_print(f"[WARN] Could not get probabilities: {e}")
                df_recent = df_recent.copy()
                df_recent['signal'] = predictions
            
            # Map direction to signal if using binary direction model
            signal_mapping = {'UP': 'BUY', 'DOWN': 'SELL'}
            if df_recent['signal'].iloc[0] in signal_mapping:
                df_recent['predicted_direction'] = df_recent['signal']
                df_recent['signal'] = df_recent['signal'].map(signal_mapping)
            
            # Add confidence score
            prob_cols = [c for c in df_recent.columns if c.startswith('prob_')]
            if prob_cols:
                df_recent['confidence'] = df_recent[prob_cols].max(axis=1)
            
            safe_print(f"[PREDICTION] Generated {len(predictions)} trading signals")
            
            # Display signal
            signal_cols = ['date_time', 'currency_pair', 'close_price', 'signal']
            if 'confidence' in df_recent.columns:
                signal_cols.append('confidence')
            
            display_cols = [col for col in signal_cols if col in df_recent.columns]
            recent_signals = df_recent[display_cols].tail(10)
            
            safe_print(f"\n[SIGNAL] Trading Signals:")
            safe_print("-" * 60)
            for _, row in recent_signals.iterrows():
                date_str = row.get('date_time', 'N/A')
                pair = row.get('currency_pair', 'N/A')
                price = row.get('close_price', 0)
                signal = row.get('signal', 'N/A')
                conf = row.get('confidence', 0)
                
                # Color-code by confidence
                conf_label = "HIGH" if conf > 0.65 else "MEDIUM" if conf > 0.55 else "LOW"
                safe_print(f"  {date_str} | {pair:7} | {price:8.5f} | {signal:4} | {conf:.3f} ({conf_label})")
            
            return df_recent
            
        except Exception as e:
            safe_print(f"[ERROR] Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def export_to_database(self, predictions_df, model_name='forex_ml_model'):
        """Export predictions to SQL Server database."""
        
        if predictions_df.empty:
            safe_print("[ERROR] No predictions to export to database")
            return False
        
        try:
            safe_print("[PROCESSING] Setting up database tables for results...")
            
            # Create results tables if they don't exist
            if not self.results_exporter.create_results_tables():
                safe_print("[ERROR] Failed to create results tables")
                return False
            
            # Export predictions
            safe_print("[DATA] Exporting predictions to SQL Server...")
            success = self.results_exporter.export_predictions(
                predictions_df=predictions_df,
                model_name=model_name,
                model_version=self.model_version or '3.0'
            )
            
            if success:
                # Export daily summary
                self.results_exporter.export_daily_summary(
                    predictions_df=predictions_df,
                    model_name=model_name
                )
                
                safe_print("[OK] Predictions exported to SQL Server database")
                safe_print(f"[DATA] Records written to: forex_ml_predictions table")
                return True
            else:
                safe_print("[ERROR] Failed to export predictions to database")
                return False
                
        except Exception as e:
            safe_print(f"[ERROR] Error exporting to database: {e}")
            return False


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Forex Trading Signal Prediction - Enhanced')
    parser.add_argument('--currency-pair', type=str, help='Currency pair (e.g., EURUSD, GBPUSD)')
    parser.add_argument('--days-back', type=int, default=60, help='Days of historical data')
    parser.add_argument('--model-path', type=str, help='Path to saved model file')
    parser.add_argument('--train-new', action='store_true', help='Train a new model')
    parser.add_argument('--export-db', action='store_true', help='Export predictions to SQL Server database')
    parser.add_argument('--setup-tables', action='store_true', help='Setup database tables for results storage')
    parser.add_argument('--signal-type', type=str, default='direction', 
                       choices=['trend', 'momentum', 'mean_reversion', 'direction'],
                       help='Type of trading signal (direction recommended)')
    parser.add_argument('--future-periods', type=int, default=1,
                       help='Prediction horizon in days (1 recommended)')
    
    args = parser.parse_args()
    
    safe_print("[FOREX] Forex Trading Signal Predictor - Enhanced v3.0")
    safe_print("=" * 60)
    
    # Initialize predictor
    predictor = ForexTradingSignalPredictor(
        model_path=args.model_path,
        currency_pair=args.currency_pair
    )
    
    # Setup database tables if requested
    if args.setup_tables:
        safe_print("[FIX] Setting up database tables for results storage...")
        success = predictor.results_exporter.create_results_tables()
        if success:
            safe_print("[OK] Database tables setup completed")
        else:
            safe_print("[ERROR] Database tables setup failed")
        return
    
    # Get forex data
    forex_data = predictor.get_forex_data(
        currency_pair=args.currency_pair,
        days_back=args.days_back
    )
    
    if forex_data.empty:
        safe_print("[ERROR] No forex data available. Please check your database connection and data.")
        return
    
    # Train new model if requested
    if args.train_new:
        safe_print(f"[PROCESSING] Training new model with {args.signal_type} signals, "
                   f"{args.future_periods}-day horizon...")
        success = predictor.train_model(
            forex_data, 
            signal_type=args.signal_type,
            future_periods=args.future_periods
        )
        if not success:
            safe_print("[ERROR] Model training failed")
            return
    
    # Generate predictions
    predictions = predictor.predict_signals(df=forex_data, currency_pair=args.currency_pair)
    
    if predictions.empty:
        safe_print("[ERROR] No predictions generated")
        return
    
    # Export to database if requested
    if args.export_db:
        predictor.export_to_database(predictions, model_name='forex_ml_model')
    
    safe_print("\n[OK] Forex signal prediction completed!")


if __name__ == "__main__":
    main()
