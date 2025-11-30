"""
Forex Trading Signal Prediction Script

This script generates trading signals for forex currency pairs using machine learning models
with technical indicators (BB, EMA, SMA, RSI, MACD, ATR).
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
            '✅': '[SUCCESS]',
            '❌': '[ERROR]',
            '📊': '[DATA]',
            '💱': '[FOREX]',
            '📈': '[PREDICTION]',
            '⚠️': '[WARN]',
            '🎯': '[TARGET]',
            '🔄': '[PROCESSING]',
            '💰': '[SIGNAL]',
            '📉': '[TREND]'
        }
        for emoji, replacement in emoji_replacements.items():
            text = text.replace(emoji, replacement)
        print(text)


class ForexTradingSignalPredictor:
    """Forex trading signal prediction system"""
    
    def __init__(self, model_path=None, currency_pair=None):
        """Initialize the Forex predictor with saved model artifacts"""
        
        # Default paths
        self.model_path = model_path or './data/best_forex_model.joblib'
        self.currency_pair = currency_pair
        
        # Load model artifacts if exists
        self.model_manager = None
        self.load_model_artifacts()
        
        # Database connection
        self.db = ForexSQLServerConnection()
        
        # Results exporter for writing back to SQL Server
        self.results_exporter = ForexResultsExporter(self.db)
        
        # Test database connection
        if not self.db.test_connection():
            safe_print("❌ Failed to connect to database. Please check your connection settings.")
            sys.exit(1)
        
        safe_print("✅ Database connection established")
        
        # Feature columns for forex predictions (including signal strength)
        self.feature_columns = [
            # Price data
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            
            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
            
            # Technical Indicators (updated column names)
            'rsi_14', 'macd', 'macd_signal', 'atr_14',
            
            # Bollinger Bands
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            
            # Signal Strength Features (like NSE/NASDAQ models)
            'rsi_signal_strength', 'macd_signal_strength', 'macd_trade_signal',
            'sma_200_signal', 'sma_100_signal', 'sma_50_signal', 'sma_20_signal',
            'ema_200_signal', 'ema_100_signal', 'ema_50_signal', 'ema_20_signal',
            'sma_trade_signal', 'bb_signal_strength', 'atr_signal_strength',
            
            # Derived features
            'daily_return', 'volatility', 'price_range', 'gap', 'volume_ratio',
            
            # Engineered features (will be calculated)
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
        """Load the trained model artifacts"""
        if os.path.exists(self.model_path):
            try:
                self.model_manager = ForexMLModelManager()
                self.model_manager.load_model(self.model_path)
                safe_print(f"✅ Model artifacts loaded from {self.model_path}")
                
                if self.model_manager.label_encoder:
                    safe_print(f"📊 Target classes: {list(self.model_manager.label_encoder.classes_)}")
                
            except Exception as e:
                safe_print(f"⚠️ Error loading model artifacts: {e}")
                safe_print("📊 Will train new model when needed")
        else:
            safe_print(f"⚠️ Model file not found at {self.model_path}")
            safe_print("📊 Will train new model when needed")
    
    def get_forex_data(self, currency_pair=None, days_back=60):
        """Fetch forex data for prediction"""
        
        safe_print(f"🔄 Fetching forex data...")
        
        try:
            df = self.db.get_forex_data_for_training(
                currency_pair=currency_pair,
                days_back=days_back,
                min_records=100
            )
            
            if df.empty:
                safe_print("❌ No forex data retrieved from database")
                return df
            
            safe_print(f"📊 Retrieved {len(df)} forex records")
            
            if currency_pair:
                safe_print(f"💱 Currency pair: {currency_pair}")
            else:
                pairs = df['currency_pair'].unique() if 'currency_pair' in df.columns else ['Unknown']
                safe_print(f"💱 Currency pairs: {', '.join(pairs)}")
            
            return df
            
        except Exception as e:
            safe_print(f"❌ Error fetching forex data: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df):
        """Prepare features for prediction"""
        
        if df.empty:
            return df, []
        
        safe_print("🔄 Preparing features...")
        
        try:
            # Initialize model manager if not loaded
            if self.model_manager is None:
                self.model_manager = ForexMLModelManager(task_type='classification')
            
            # Prepare forex-specific features
            df_features = self.model_manager.prepare_forex_features(df)
            
            # Select available feature columns (use model's expected features)
            model_feature_columns = self.model_manager.forex_feature_columns
            available_features = [col for col in model_feature_columns if col in df_features.columns]
            missing_features = [col for col in model_feature_columns if col not in df_features.columns]
            
            if missing_features:
                safe_print(f"⚠️ Missing features: {', '.join(missing_features[:5])}...")
            
            safe_print(f"📊 Available features: {len(available_features)}")
            
            return df_features, available_features
            
        except Exception as e:
            safe_print(f"❌ Error preparing features: {e}")
            return df, []
    
    def train_model(self, df, signal_type='trend', future_periods=5):
        """Train a new forex prediction model"""
        
        safe_print("🔄 Training new forex model...")
        
        # Get currency pair from the data
        currency_pair = self.currency_pair
        if currency_pair is None and 'currency_pair' in df.columns:
            currency_pair = df['currency_pair'].iloc[0] if len(df) > 0 else 'UNKNOWN'
        
        try:
            if self.model_manager is None:
                self.model_manager = ForexMLModelManager(task_type='classification')
            
            # Create trading signals
            df_with_signals = self.model_manager.create_trading_signals(
                df, 
                signal_type=signal_type,
                future_periods=future_periods
            )
            
            if df_with_signals.empty:
                safe_print("❌ No training data available after signal generation")
                return False
            
            # Prepare features
            df_features = self.model_manager.prepare_forex_features(df_with_signals)
            
            # Select features and target (use model's feature columns)
            available_features = [col for col in self.model_manager.forex_feature_columns if col in df_features.columns]
            
            if len(available_features) < 10:
                safe_print(f"❌ Too few features available: {len(available_features)}")
                return False
            
            X = df_features[available_features].fillna(0)
            y = df_features['signal']
            
            # Remove rows with invalid data
            valid_mask = ~(X.isin([np.inf, -np.inf]).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                safe_print(f"❌ Insufficient training samples: {len(X)}")
                return False
            
            safe_print(f"📊 Training with {len(X)} samples and {len(available_features)} features")
            safe_print(f"🎯 Signal distribution:")
            signal_counts = y.value_counts()
            for signal, count in signal_counts.items():
                safe_print(f"   {signal}: {count} ({count/len(y)*100:.1f}%)")
            
            # Train models
            results = self.model_manager.train_all_models(X, y, use_time_series_cv=True)
            
            # Display results
            safe_print(f"\n📈 Model Training Results:")
            safe_print("-" * 50)
            
            for model_name, metrics in results.items():
                if 'error' not in metrics:
                    accuracy = metrics.get('cv_accuracy_mean', 0)
                    std = metrics.get('cv_accuracy_std', 0)
                    safe_print(f"{model_name:15}: {accuracy:.3f} (+/- {std:.3f})")
                else:
                    safe_print(f"{model_name:15}: ERROR - {metrics['error']}")
            
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
            safe_print(f"✅ Model saved to {self.model_path}")
            
            return True
            
        except Exception as e:
            safe_print(f"❌ Error training model: {e}")
            return False
    
    def predict_signals(self, df=None, currency_pair=None):
        """Generate forex trading signals for next trading day"""
        
        if self.model_manager is None or self.model_manager.best_model is None:
            safe_print("❌ No trained model available. Training new model...")
            
            if df is None:
                df = self.get_forex_data(currency_pair=currency_pair)
            
            if df.empty:
                safe_print("❌ Cannot train model: no data available")
                return pd.DataFrame()
            
            if not self.train_model(df):
                safe_print("❌ Model training failed")
                return pd.DataFrame()
        
        # Get recent data for prediction if not provided
        if df is None:
            df = self.get_forex_data(currency_pair=currency_pair, days_back=100)  # Get enough history for features
        
        if df.empty:
            safe_print("❌ No data available for prediction")
            return pd.DataFrame()
        
        safe_print("🔄 Generating trading signals...")
        
        try:
            # Prepare features
            df_features, available_features = self.prepare_features(df)
            
            if len(available_features) < 10:
                safe_print("❌ Insufficient features for prediction")
                return pd.DataFrame()
            
            # Get the MOST RECENT record only (latest trading day)
            df_features = df_features.sort_values('date_time')
            df_recent = df_features.dropna(subset=available_features).tail(1)  # Only predict for latest day
            
            if df_recent.empty:
                safe_print("❌ No valid data for prediction")
                return pd.DataFrame()
            
            safe_print(f"📊 Predicting for latest date: {df_recent['date_time'].iloc[0]}")
            
            # Make prediction for next trading day
            X_pred = df_recent[available_features].fillna(0)
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
                safe_print(f"⚠️ Could not get probabilities: {e}")
                df_recent = df_recent.copy()
                df_recent['signal'] = predictions
            
            # Add confidence score
            if 'prob_BUY' in df_recent.columns and 'prob_SELL' in df_recent.columns:
                df_recent['confidence'] = np.maximum(
                    df_recent.get('prob_BUY', 0),
                    df_recent.get('prob_SELL', 0)
                )
            
            safe_print(f"📈 Generated {len(predictions)} trading signals")
            
            # Display recent signals
            signal_cols = ['date_time', 'currency_pair', 'close_price', 'signal']
            if 'confidence' in df_recent.columns:
                signal_cols.append('confidence')
            
            display_cols = [col for col in signal_cols if col in df_recent.columns]
            recent_signals = df_recent[display_cols].tail(10)
            
            safe_print(f"\n💰 Recent Trading Signals:")
            safe_print("-" * 60)
            for _, row in recent_signals.iterrows():
                date_str = row.get('date_time', 'N/A')
                pair = row.get('currency_pair', 'N/A')
                price = row.get('close_price', 0)
                signal = row.get('signal', 'N/A')
                conf = row.get('confidence', 0)
                
                safe_print(f"{date_str} | {pair:7} | {price:8.5f} | {signal:4} | {conf:.3f}")
            
            return df_recent
            
        except Exception as e:
            safe_print(f"❌ Error generating predictions: {e}")
            return pd.DataFrame()
    
    def export_to_database(self, predictions_df, model_name='forex_ml_model'):
        """Export predictions to SQL Server database"""
        
        if predictions_df.empty:
            safe_print("❌ No predictions to export to database")
            return False
        
        try:
            safe_print("🔄 Setting up database tables for results...")
            
            # Create results tables if they don't exist
            if not self.results_exporter.create_results_tables():
                safe_print("❌ Failed to create results tables")
                return False
            
            # Export predictions
            safe_print("📊 Exporting predictions to SQL Server...")
            success = self.results_exporter.export_predictions(
                predictions_df=predictions_df,
                model_name=model_name,
                model_version='1.0'
            )
            
            if success:
                # Export daily summary
                self.results_exporter.export_daily_summary(
                    predictions_df=predictions_df,
                    model_name=model_name
                )
                
                safe_print("✅ Predictions exported to SQL Server database")
                safe_print(f"📊 Records written to: forex_ml_predictions table")
                return True
            else:
                safe_print("❌ Failed to export predictions to database")
                return False
                
        except Exception as e:
            safe_print(f"❌ Error exporting to database: {e}")
            return False
    
    def export_predictions(self, predictions_df, filename=None):
        """Export predictions to CSV file"""
        
        if predictions_df.empty:
            safe_print("❌ No predictions to export")
            return False
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"./results/forex_predictions_{timestamp}.csv"
            
            # Create results directory
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Export to CSV
            predictions_df.to_csv(filename, index=False)
            safe_print(f"✅ Predictions exported to {filename}")
            
            return True
            
        except Exception as e:
            safe_print(f"❌ Error exporting predictions: {e}")
            return False


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Forex Trading Signal Prediction')
    parser.add_argument('--currency-pair', type=str, help='Currency pair (e.g., EURUSD, GBPUSD)')
    parser.add_argument('--days-back', type=int, default=60, help='Days of historical data')
    parser.add_argument('--model-path', type=str, help='Path to saved model file')
    parser.add_argument('--train-new', action='store_true', help='Train a new model')
    parser.add_argument('--export', action='store_true', help='Export predictions to CSV')
    parser.add_argument('--export-db', action='store_true', help='Export predictions to SQL Server database')
    parser.add_argument('--setup-tables', action='store_true', help='Setup database tables for results storage')
    parser.add_argument('--signal-type', type=str, default='trend', 
                       choices=['trend', 'momentum', 'mean_reversion'],
                       help='Type of trading signal')
    
    args = parser.parse_args()
    
    safe_print("💱 Forex Trading Signal Predictor")
    safe_print("=" * 50)
    
    # Initialize predictor
    predictor = ForexTradingSignalPredictor(
        model_path=args.model_path,
        currency_pair=args.currency_pair
    )
    
    # Setup database tables if requested
    if args.setup_tables:
        safe_print("🔧 Setting up database tables for results storage...")
        success = predictor.results_exporter.create_results_tables()
        if success:
            safe_print("✅ Database tables setup completed")
        else:
            safe_print("❌ Database tables setup failed")
        return
    
    # Get forex data
    forex_data = predictor.get_forex_data(
        currency_pair=args.currency_pair,
        days_back=args.days_back
    )
    
    if forex_data.empty:
        safe_print("❌ No forex data available. Please check your database connection and data.")
        return
    
    # Train new model if requested
    if args.train_new:
        safe_print(f"🔄 Training new model with {args.signal_type} signals...")
        success = predictor.train_model(forex_data, signal_type=args.signal_type)
        if not success:
            safe_print("❌ Model training failed")
            return
    
    # Generate predictions
    predictions = predictor.predict_signals(df=forex_data, currency_pair=args.currency_pair)
    
    if predictions.empty:
        safe_print("❌ No predictions generated")
        return
    
    # Export predictions if requested
    if args.export:
        predictor.export_predictions(predictions)
    
    # Export to database if requested
    if args.export_db:
        predictor.export_to_database(predictions, model_name='forex_ml_model')
    
    safe_print("\n✅ Forex signal prediction completed!")


if __name__ == "__main__":
    main()