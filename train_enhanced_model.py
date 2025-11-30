"""
Enhanced Forex ML Training Script for 70%+ Accuracy
Integrates advanced features, models, and data sources
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from database.connection import ForexSQLServerConnection
from features.advanced_features import AdvancedForexFeatures
from models.advanced_models import AdvancedForexModels
from data.external_sources import ExternalDataSources

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_print(message):
    """Safe printing for different environments"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode('ascii', 'ignore').decode('ascii'))
    logger.info(message)

class EnhancedForexTrainer:
    """Enhanced forex ML trainer for high accuracy models"""
    
    def __init__(self):
        self.db = ForexSQLServerConnection()
        self.feature_engineer = AdvancedForexFeatures()
        self.model_trainer = AdvancedForexModels()
        self.external_data = ExternalDataSources()
        self.enhanced_features_df = None
        
    def prepare_enhanced_dataset(self, currency_pairs: list = None, lookback_days: int = 1000) -> pd.DataFrame:
        """Prepare enhanced dataset with all available features"""
        
        safe_print("🔄 Preparing enhanced dataset for training...")
        
        if currency_pairs is None:
            currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
                            'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY']
        
        all_data = []
        
        for pair in currency_pairs:
            safe_print(f"📊 Processing {pair}...")
            
            try:
                # Get base forex data with indicators
                base_data = self.db.get_forex_data_with_indicators(pair)
                
                if base_data.empty:
                    safe_print(f"⚠️ No data found for {pair}")
                    continue
                
                # Add advanced features
                enhanced_data = self.feature_engineer.create_advanced_features(base_data)
                
                # Add external market data (sentiment, economic indicators)
                enhanced_data = self._add_external_features(enhanced_data, pair)
                
                # Create target variable with better logic
                enhanced_data = self._create_enhanced_target(enhanced_data)
                
                # Add pair identifier
                enhanced_data['currency_pair'] = pair
                
                all_data.append(enhanced_data)
                safe_print(f"✅ {pair}: {len(enhanced_data)} records with {len(enhanced_data.columns)} features")
                
            except Exception as e:
                safe_print(f"❌ Error processing {pair}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be processed for any currency pair")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        safe_print(f"📈 Combined dataset: {len(combined_data)} records, {len(combined_data.columns)} features")
        
        # Feature engineering and encoding
        combined_data = self.feature_engineer.encode_categorical_features(combined_data)
        
        self.enhanced_features_df = combined_data
        return combined_data
    
    def _add_external_features(self, df: pd.DataFrame, currency_pair: str) -> pd.DataFrame:
        """Add external market features"""
        
        # Add market sentiment features (VIX, DXY, etc.)
        try:
            if 'date_time' in df.columns:
                start_date = df['date_time'].min().strftime('%Y-%m-%d')
                end_date = df['date_time'].max().strftime('%Y-%m-%d')
                
                # Get market sentiment data
                sentiment_data = self.external_data.get_market_sentiment_data(start_date, end_date)
                
                if not sentiment_data.empty:
                    # Merge sentiment data with main dataset
                    df_with_sentiment = df.merge(sentiment_data, left_on='date_time', 
                                                right_index=True, how='left')
                    return df_with_sentiment
        except Exception as e:
            logger.warning(f\"Error adding external features for {currency_pair}: {e}\")
        
        return df
    
    def _create_enhanced_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced target variable with better prediction logic"""
        
        if 'close_price' not in df.columns:
            return df
        
        # Calculate future returns for different horizons
        df['future_return_1d'] = df['close_price'].shift(-1) / df['close_price'] - 1
        df['future_return_3d'] = df['close_price'].shift(-3) / df['close_price'] - 1
        df['future_return_5d'] = df['close_price'].shift(-5) / df['close_price'] - 1
        
        # Enhanced target with multiple criteria
        buy_threshold = 0.002   # 0.2% gain
        sell_threshold = -0.002 # 0.2% loss
        
        def determine_signal(row):
            ret_1d = row['future_return_1d']
            ret_3d = row['future_return_3d'] 
            ret_5d = row['future_return_5d']
            
            # Strong buy signal
            if ret_1d > buy_threshold and ret_3d > buy_threshold:
                return 'BUY'
            # Strong sell signal  
            elif ret_1d < sell_threshold and ret_3d < sell_threshold:
                return 'SELL'
            # Hold for unclear signals
            else:
                return 'HOLD'
        
        df['target_signal'] = df.apply(determine_signal, axis=1)
        
        # Remove rows without target (last few rows)
        df = df.dropna(subset=['target_signal'])
        
        return df
    
    def train_enhanced_models(self, test_size: float = 0.2) -> dict:
        \"\"\"Train enhanced models with all features\"\"\"
        
        if self.enhanced_features_df is None:
            raise ValueError(\"Must call prepare_enhanced_dataset first\")
        
        safe_print(\"🚀 Training enhanced models for high accuracy...\")
        
        # Prepare features and target
        feature_columns = self._select_feature_columns()
        X = self.enhanced_features_df[feature_columns].copy()
        y = self.enhanced_features_df['target_signal'].copy()
        
        safe_print(f\"📊 Training with {len(feature_columns)} features on {len(X)} samples\")
        safe_print(f\"📈 Target distribution: {y.value_counts().to_dict()}\")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode target
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Time-based train/test split (important for time series)
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y_encoded[:split_point], y_encoded[split_point:]
        
        safe_print(f\"📊 Train set: {len(X_train)} samples\")
        safe_print(f\"📊 Test set: {len(X_test)} samples\")
        
        # Train models with feature selection
        results, X_selected = self.model_trainer.train_with_feature_selection(X_train, y_train)
        
        # Evaluate on test set
        test_results = self._evaluate_on_test_set(results, X_selected, X_test, y_test, label_encoder)
        
        # Find best model
        best_model_name = max(test_results.keys(), key=lambda k: test_results[k]['test_accuracy'])
        best_model = results[best_model_name]['model']
        
        safe_print(f\"\\n🏆 Best Model: {best_model_name}\")
        safe_print(f\"📊 Test Accuracy: {test_results[best_model_name]['test_accuracy']:.4f}\")
        
        # Save enhanced model
        self._save_enhanced_model(best_model, best_model_name, X_selected.columns.tolist(), 
                                label_encoder, results)
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'results': results,
            'test_results': test_results,
            'feature_columns': X_selected.columns.tolist(),
            'label_encoder': label_encoder
        }
    
    def _select_feature_columns(self) -> list:
        \"\"\"Select relevant feature columns for training\"\"\"
        
        # Exclude non-feature columns
        exclude_columns = [
            'currency_pair', 'date_time', 'target_signal',
            'future_return_1d', 'future_return_3d', 'future_return_5d'
        ]
        
        feature_columns = [col for col in self.enhanced_features_df.columns 
                          if col not in exclude_columns and not col.endswith('_signal')]
        
        # Select only numeric columns
        numeric_features = []
        for col in feature_columns:
            if self.enhanced_features_df[col].dtype in ['int64', 'float64', 'bool']:
                numeric_features.append(col)
            elif col.endswith('_encoded'):
                numeric_features.append(col)
        
        return numeric_features
    
    def _evaluate_on_test_set(self, results: dict, X_train_selected: pd.DataFrame, 
                            X_test: pd.DataFrame, y_test: np.ndarray, 
                            label_encoder) -> dict:
        \"\"\"Evaluate all models on test set\"\"\"
        
        from sklearn.metrics import accuracy_score, classification_report
        
        test_results = {}
        
        # Use same features as training
        feature_columns = X_train_selected.columns
        X_test_selected = X_test[feature_columns].fillna(X_test[feature_columns].median())
        
        for model_name, result in results.items():
            model = result['model']
            
            try:
                y_pred = model.predict(X_test_selected)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                test_results[model_name] = {
                    'test_accuracy': test_accuracy,
                    'classification_report': classification_report(y_test, y_pred, 
                                                                  target_names=label_encoder.classes_)
                }
                
                safe_print(f\"{model_name}: Test Accuracy = {test_accuracy:.4f}\")
                
            except Exception as e:
                safe_print(f\"Error evaluating {model_name}: {e}\")
                continue
        
        return test_results
    
    def _save_enhanced_model(self, model, model_name: str, feature_columns: list,
                           label_encoder, all_results: dict):
        \"\"\"Save the enhanced model with all artifacts\"\"\"
        
        import joblib
        from pathlib import Path
        
        # Create data directory
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Prepare model artifacts
        model_artifacts = {
            'model': model,
            'model_name': model_name,
            'feature_columns': feature_columns,
            'label_encoder': label_encoder,
            'training_results': all_results,
            'training_date': datetime.now().isoformat(),
            'model_version': '2.0_enhanced'
        }
        
        # Save model
        model_path = data_dir / 'enhanced_forex_model.joblib'
        joblib.dump(model_artifacts, model_path)
        
        safe_print(f\"💾 Enhanced model saved to: {model_path}\")
        
        # Also update the main model file
        main_model_path = data_dir / 'best_forex_model.joblib'
        joblib.dump(model_artifacts, main_model_path)
        safe_print(f\"💾 Updated main model: {main_model_path}\")

def main():
    \"\"\"Main training function\"\"\"
    
    safe_print(\"🚀 Starting Enhanced Forex ML Training\")
    safe_print(\"=\" * 60)
    
    try:
        # Initialize trainer
        trainer = EnhancedForexTrainer()
        
        # Test database connection
        if not trainer.db.test_connection():
            safe_print(\"❌ Database connection failed\")
            return
        
        # Prepare enhanced dataset
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        enhanced_data = trainer.prepare_enhanced_dataset(currency_pairs, lookback_days=800)
        
        # Train enhanced models
        results = trainer.train_enhanced_models(test_size=0.2)
        
        safe_print(\"\\n\" + \"=\" * 60)
        safe_print(\"🎯 TRAINING COMPLETED SUCCESSFULLY!\")
        safe_print(f\"🏆 Best Model: {results['best_model_name']}\")
        safe_print(f\"📊 Features Used: {len(results['feature_columns'])}\")
        safe_print(\"\\n📈 Top Test Accuracies:\")
        
        # Show top results
        test_results = results['test_results']
        sorted_results = sorted(test_results.items(), 
                              key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        for model_name, result in sorted_results[:5]:
            accuracy = result['test_accuracy']
            safe_print(f\"  {model_name}: {accuracy:.1%}\")
        
        safe_print(\"\\n💡 If accuracy is still below 70%, consider:\")
        safe_print(\"  1. Adding more external data sources\")
        safe_print(\"  2. Increasing training data (more currency pairs/history)\")
        safe_print(\"  3. Fine-tuning hyperparameters\")
        safe_print(\"  4. Using ensemble methods\")
        
    except Exception as e:
        safe_print(f\"❌ Training failed: {e}\")
        import traceback
        traceback.print_exc()

if __name__ == \"__main__\":
    main()
"