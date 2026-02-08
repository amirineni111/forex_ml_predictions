"""
Enhanced Forex ML Training Script for Direction Accuracy

Key improvements over previous version:
- Binary direction prediction (UP/DOWN) for better accuracy
- Walk-forward validation for realistic performance estimates
- Adaptive thresholds based on recent volatility
- Feature selection to reduce noise and overfitting
- Sample weighting to prioritize recent market data
- Purged cross-validation to prevent data leakage
- Model stability checks before saving
- Overfitting detection and warning
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
    """
    Enhanced forex ML trainer optimized for direction accuracy.
    
    Key design principles:
    1. Predict direction (UP/DOWN) instead of 3-class (BUY/SELL/HOLD)
    2. Use 1-day ahead prediction instead of 5-day
    3. Constrained models to prevent overfitting
    4. Walk-forward validation for realistic estimates
    5. Feature selection to reduce noise
    """
    
    def __init__(self):
        self.db = ForexSQLServerConnection()
        self.feature_engineer = AdvancedForexFeatures()
        self.model_trainer = AdvancedForexModels()
        self.external_data = ExternalDataSources()
        self.enhanced_features_df = None
        self.feature_quality = None
        
    def prepare_enhanced_dataset(self, currency_pairs: list = None, lookback_days: int = 1000) -> pd.DataFrame:
        """
        Prepare enhanced dataset with all available features.
        
        Enhanced with better data quality checks and feature engineering.
        """
        
        safe_print("[INFO] Preparing enhanced dataset for training...")
        
        if currency_pairs is None:
            currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 
                            'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY']
        
        all_data = []
        
        for pair in currency_pairs:
            safe_print(f"[DATA] Processing {pair}...")
            
            try:
                # Get base forex data with indicators
                base_data = self.db.get_forex_data_with_indicators(pair)
                
                if base_data.empty:
                    safe_print(f"[WARN] No data found for {pair}")
                    continue
                
                # Data quality check
                if len(base_data) < 50:
                    safe_print(f"[WARN] Insufficient data for {pair}: only {len(base_data)} records")
                    continue
                
                # Add advanced features
                enhanced_data = self.feature_engineer.create_advanced_features(base_data)
                
                # Add external market data (sentiment, economic indicators)
                enhanced_data = self._add_external_features(enhanced_data, pair)
                
                # Create target variable with IMPROVED logic
                enhanced_data = self._create_enhanced_target(enhanced_data)
                
                # Add pair identifier
                enhanced_data['currency_pair'] = pair
                
                all_data.append(enhanced_data)
                safe_print(f"[OK] {pair}: {len(enhanced_data)} records with {len(enhanced_data.columns)} features")
                
            except Exception as e:
                safe_print(f"[ERROR] Error processing {pair}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be processed for any currency pair")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        safe_print(f"[DATA] Combined dataset: {len(combined_data)} records, {len(combined_data.columns)} features")
        
        # Feature engineering and encoding
        combined_data = self.feature_engineer.encode_categorical_features(combined_data)
        
        # Data quality report
        self._report_data_quality(combined_data)
        
        self.enhanced_features_df = combined_data
        return combined_data
    
    def _add_external_features(self, df: pd.DataFrame, currency_pair: str) -> pd.DataFrame:
        """Add external market features."""
        try:
            if 'date_time' in df.columns:
                start_date = df['date_time'].min().strftime('%Y-%m-%d')
                end_date = df['date_time'].max().strftime('%Y-%m-%d')
                
                # Get market sentiment data
                sentiment_data = self.external_data.get_market_sentiment_data(start_date, end_date)
                
                if not sentiment_data.empty:
                    df_with_sentiment = df.merge(sentiment_data, left_on='date_time', 
                                                right_index=True, how='left')
                    return df_with_sentiment
        except Exception as e:
            logger.warning(f"Error adding external features for {currency_pair}: {e}")
        
        return df
    
    def _create_enhanced_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced target variable with ADAPTIVE thresholds.
        
        Key improvements:
        - 1-day ahead prediction (was multi-day)
        - Adaptive threshold based on recent volatility
        - Binary direction (UP/DOWN) option for better accuracy
        - Multi-timeframe confirmation for 3-class (BUY/SELL/HOLD)
        """
        
        if 'close_price' not in df.columns:
            return df
        
        # Calculate future returns for different horizons
        df['future_return_1d'] = df['close_price'].shift(-1) / df['close_price'] - 1
        df['future_return_3d'] = df['close_price'].shift(-3) / df['close_price'] - 1
        df['future_return_5d'] = df['close_price'].shift(-5) / df['close_price'] - 1
        
        # Calculate ADAPTIVE thresholds based on recent volatility
        daily_returns = df['close_price'].pct_change()
        rolling_vol = daily_returns.rolling(window=20, min_periods=5).std()
        
        # Adaptive threshold: 30% of daily volatility, bounded
        adaptive_buy = (rolling_vol * 0.3).clip(lower=0.0005, upper=0.005)
        adaptive_sell = (-rolling_vol * 0.3).clip(upper=-0.0005, lower=-0.005)
        
        def determine_signal(row_idx):
            """Determine signal with adaptive threshold and multi-timeframe confirmation."""
            ret_1d = df['future_return_1d'].iloc[row_idx]
            
            if pd.isna(ret_1d):
                return np.nan
            
            buy_thresh = adaptive_buy.iloc[row_idx] if row_idx < len(adaptive_buy) else 0.001
            sell_thresh = adaptive_sell.iloc[row_idx] if row_idx < len(adaptive_sell) else -0.001
            
            if pd.isna(buy_thresh):
                buy_thresh = 0.001
            if pd.isna(sell_thresh):
                sell_thresh = -0.001
            
            # Primary signal based on 1-day return with adaptive threshold
            if ret_1d > buy_thresh:
                return 'BUY'
            elif ret_1d < sell_thresh:
                return 'SELL'
            else:
                return 'HOLD'
        
        # Apply signal determination
        df['target_signal'] = [determine_signal(i) for i in range(len(df))]
        
        # Also create a binary direction target (simpler, often more accurate)
        df['target_direction'] = np.where(df['future_return_1d'] > 0, 'UP', 'DOWN')
        
        # Remove rows without target
        df = df.dropna(subset=['future_return_1d'])
        
        return df
    
    def _report_data_quality(self, df: pd.DataFrame):
        """Report data quality metrics."""
        safe_print("\n[DATA QUALITY REPORT]")
        safe_print(f"  Total records: {len(df)}")
        safe_print(f"  Total features: {len(df.columns)}")
        
        # Check for target distribution
        if 'target_signal' in df.columns:
            dist = df['target_signal'].value_counts()
            safe_print(f"  Target distribution (3-class):")
            for signal, count in dist.items():
                safe_print(f"    {signal}: {count} ({count/len(df)*100:.1f}%)")
        
        if 'target_direction' in df.columns:
            dist = df['target_direction'].value_counts()
            safe_print(f"  Target distribution (binary):")
            for signal, count in dist.items():
                safe_print(f"    {signal}: {count} ({count/len(df)*100:.1f}%)")
        
        # Missing value summary
        missing = df.isnull().sum()
        high_missing = missing[missing > len(df) * 0.3]
        if len(high_missing) > 0:
            safe_print(f"  Features with >30% missing: {len(high_missing)}")
        
        safe_print("")
    
    def train_enhanced_models(self, test_size: float = 0.2, 
                             use_binary_direction: bool = True) -> dict:
        """
        Train enhanced models with all improvements.
        
        Args:
            test_size: Fraction of data for testing
            use_binary_direction: If True, use UP/DOWN binary prediction (recommended)
        
        Key improvements:
        - Binary direction prediction for better accuracy
        - Walk-forward validation
        - Feature selection
        - Sample weighting
        - Overfitting checks
        """
        
        if self.enhanced_features_df is None:
            raise ValueError("Must call prepare_enhanced_dataset first")
        
        safe_print("[INFO] Training enhanced models for direction accuracy...")
        
        # Choose target variable
        if use_binary_direction:
            target_col = 'target_direction'
            safe_print("[INFO] Using BINARY direction prediction (UP/DOWN) for better accuracy")
        else:
            target_col = 'target_signal'
            safe_print("[INFO] Using 3-class prediction (BUY/SELL/HOLD)")
        
        if target_col not in self.enhanced_features_df.columns:
            safe_print(f"[ERROR] Target column {target_col} not found. Available: {list(self.enhanced_features_df.columns)}")
            return {}
        
        # Prepare features and target
        feature_columns = self._select_feature_columns()
        
        # Remove rows with missing target
        valid_data = self.enhanced_features_df.dropna(subset=[target_col])
        
        X = valid_data[feature_columns].copy()
        y = valid_data[target_col].copy()
        
        # Handle missing and infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        safe_print(f"[DATA] Training with {len(feature_columns)} features on {len(X)} samples")
        safe_print(f"[DATA] Target distribution: {y.value_counts().to_dict()}")
        
        # Encode target
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Time-based train/test split (IMPORTANT: don't shuffle time series!)
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y_encoded[:split_point], y_encoded[split_point:]
        
        safe_print(f"[DATA] Train set: {len(X_train)} samples")
        safe_print(f"[DATA] Test set: {len(X_test)} samples")
        
        # Step 1: Feature selection with walk-forward awareness
        safe_print("[INFO] Step 1: Feature selection...")
        results, X_selected = self.model_trainer.train_with_feature_selection(X_train, y_train)
        
        # Step 2: Walk-forward validation on full dataset
        safe_print("\n[INFO] Step 2: Walk-forward validation...")
        wf_results = self.model_trainer.walk_forward_validate(
            X[X_selected.columns], y_encoded, 
            n_windows=5, embargo_gap=3
        )
        
        # Step 3: Evaluate on held-out test set
        safe_print("\n[INFO] Step 3: Test set evaluation...")
        test_results = self._evaluate_on_test_set(results, X_selected, X_test, y_test, label_encoder)
        
        # Step 4: Select best model considering overfitting
        safe_print("\n[INFO] Step 4: Model selection...")
        best_model_name = self._select_best_model(results, test_results, wf_results)
        
        if best_model_name is None:
            safe_print("[ERROR] No valid model found")
            return {}
        
        best_model = results[best_model_name]['model']
        
        safe_print(f"\n[BEST] Best Model: {best_model_name}")
        
        if best_model_name in test_results:
            safe_print(f"[BEST] Test Accuracy: {test_results[best_model_name]['test_accuracy']:.4f}")
        if best_model_name in wf_results:
            safe_print(f"[BEST] Walk-Forward Accuracy: {wf_results[best_model_name]['wf_accuracy_mean']:.4f}")
        if best_model_name in results:
            safe_print(f"[BEST] CV Accuracy: {results[best_model_name]['cv_accuracy_mean']:.4f}")
            safe_print(f"[BEST] Overfitting Gap: {results[best_model_name].get('overfitting_gap', 'N/A')}")
        
        # Step 5: Model stability check
        stability_ok = self._check_model_stability(results, test_results, wf_results, best_model_name)
        
        if not stability_ok:
            safe_print("[WARN] Model stability check failed. Consider more data or simpler models.")
        
        # Save enhanced model
        self._save_enhanced_model(
            best_model, best_model_name, 
            X_selected.columns.tolist(), 
            label_encoder, results, 
            self.model_trainer.scaler,
            wf_results
        )
        
        return {
            'best_model': best_model,
            'best_model_name': best_model_name,
            'results': results,
            'test_results': test_results,
            'wf_results': wf_results,
            'feature_columns': X_selected.columns.tolist(),
            'label_encoder': label_encoder,
            'scaler': self.model_trainer.scaler,
            'stability_passed': stability_ok
        }
    
    def _select_feature_columns(self) -> list:
        """Select relevant feature columns for training."""
        
        # Exclude non-feature columns (including future-looking data!)
        exclude_columns = [
            'currency_pair', 'date_time', 'target_signal', 'target_direction',
            'future_return_1d', 'future_return_3d', 'future_return_5d',
            'signal', 'future_return'  # Any target-related columns
        ]
        
        # Also exclude any column with 'signal' in the name (except encoded ones)
        feature_columns = [
            col for col in self.enhanced_features_df.columns 
            if col not in exclude_columns 
            and not (col.endswith('_signal') and not col.endswith('_encoded'))
        ]
        
        # Select only numeric columns
        numeric_features = []
        for col in feature_columns:
            dtype = self.enhanced_features_df[col].dtype
            if dtype in ['int64', 'float64', 'bool', 'int32', 'float32']:
                numeric_features.append(col)
            elif col.endswith('_encoded'):
                numeric_features.append(col)
        
        # Remove features with very low variance (uninformative)
        variances = self.enhanced_features_df[numeric_features].var()
        low_var_features = variances[variances < 1e-10].index.tolist()
        if low_var_features:
            safe_print(f"[INFO] Removing {len(low_var_features)} zero-variance features")
            numeric_features = [f for f in numeric_features if f not in low_var_features]
        
        # Remove features with >50% missing values
        missing_rates = self.enhanced_features_df[numeric_features].isnull().mean()
        high_missing = missing_rates[missing_rates > 0.5].index.tolist()
        if high_missing:
            safe_print(f"[INFO] Removing {len(high_missing)} features with >50% missing values")
            numeric_features = [f for f in numeric_features if f not in high_missing]
        
        safe_print(f"[INFO] Selected {len(numeric_features)} numeric features for training")
        return numeric_features
    
    def _evaluate_on_test_set(self, results: dict, X_train_selected: pd.DataFrame, 
                            X_test: pd.DataFrame, y_test: np.ndarray, 
                            label_encoder) -> dict:
        """Evaluate all models on held-out test set."""
        
        from sklearn.metrics import accuracy_score, classification_report
        
        test_results = {}
        
        # Use same features as training
        feature_columns = X_train_selected.columns
        
        # Ensure test set has the same features
        available_features = [f for f in feature_columns if f in X_test.columns]
        X_test_selected = X_test[available_features].copy()
        X_test_selected = X_test_selected.replace([np.inf, -np.inf], np.nan)
        X_test_selected = X_test_selected.fillna(X_test_selected.median())
        
        # Scale test data using the same scaler
        if self.model_trainer.scaler is not None:
            X_test_scaled = pd.DataFrame(
                self.model_trainer.scaler.transform(X_test_selected),
                columns=X_test_selected.columns, index=X_test_selected.index
            )
        else:
            X_test_scaled = X_test_selected
        
        for model_name, result in results.items():
            if 'model' not in result:
                continue
            
            model = result['model']
            
            try:
                y_pred = model.predict(X_test_scaled)
                test_accuracy = accuracy_score(y_test, y_pred)
                
                # Calculate direction accuracy specifically
                cv_accuracy = result.get('cv_accuracy_mean', 0)
                train_accuracy = result.get('train_accuracy', 0)
                overfit_gap = train_accuracy - test_accuracy
                
                test_results[model_name] = {
                    'test_accuracy': test_accuracy,
                    'cv_accuracy': cv_accuracy,
                    'train_accuracy': train_accuracy,
                    'overfit_gap': overfit_gap,
                    'classification_report': classification_report(
                        y_test, y_pred, 
                        target_names=label_encoder.classes_,
                        output_dict=True
                    )
                }
                
                status = "[OK]"
                if overfit_gap > 0.15:
                    status = "[OVERFIT]"
                elif test_accuracy > 0.55:
                    status = "[GOOD]"
                
                safe_print(f"  {model_name}: Test={test_accuracy:.4f}, CV={cv_accuracy:.4f}, "
                          f"Train={train_accuracy:.4f}, Gap={overfit_gap:.3f} {status}")
                
            except Exception as e:
                safe_print(f"  [ERROR] evaluating {model_name}: {e}")
                continue
        
        return test_results
    
    def _select_best_model(self, results: dict, test_results: dict, wf_results: dict) -> str:
        """
        Select best model with anti-overfitting criteria.
        
        Priority: Walk-forward accuracy > Test accuracy > CV accuracy
        Penalize models with large overfitting gaps.
        """
        
        candidates = {}
        
        for model_name in results.keys():
            if 'model' not in results[model_name]:
                continue
            
            score = 0
            n_scores = 0
            
            # Walk-forward accuracy (most realistic)
            if model_name in wf_results:
                wf_acc = wf_results[model_name]['wf_accuracy_mean']
                score += wf_acc * 2  # Double weight for WF
                n_scores += 2
            
            # Test accuracy
            if model_name in test_results:
                test_acc = test_results[model_name]['test_accuracy']
                score += test_acc
                n_scores += 1
                
                # Penalize overfitting
                overfit_gap = test_results[model_name].get('overfit_gap', 0)
                if overfit_gap > 0.15:
                    score -= (overfit_gap - 0.10) * 0.5
            
            # CV accuracy
            cv_acc = results[model_name].get('cv_accuracy_mean', 0)
            score += cv_acc
            n_scores += 1
            
            if n_scores > 0:
                candidates[model_name] = score / n_scores
        
        if not candidates:
            return None
        
        best = max(candidates, key=candidates.get)
        safe_print(f"\n[MODEL SELECTION] Adjusted scores:")
        for name, score in sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:5]:
            marker = " <-- BEST" if name == best else ""
            safe_print(f"  {name}: {score:.4f}{marker}")
        
        return best
    
    def _check_model_stability(self, results, test_results, wf_results, best_model_name) -> bool:
        """
        Check if the best model is stable enough for production use.
        
        Returns True if model passes stability checks.
        """
        safe_print("\n[STABILITY CHECK]")
        passed = True
        
        # Check 1: CV accuracy should be > 50% (better than random)
        cv_acc = results[best_model_name].get('cv_accuracy_mean', 0)
        if cv_acc < 0.50:
            safe_print(f"  [FAIL] CV accuracy ({cv_acc:.3f}) below random chance (0.50)")
            passed = False
        else:
            safe_print(f"  [PASS] CV accuracy: {cv_acc:.3f}")
        
        # Check 2: CV standard deviation should be < 0.15
        cv_std = results[best_model_name].get('cv_accuracy_std', 0)
        if cv_std > 0.15:
            safe_print(f"  [WARN] CV accuracy very unstable (std={cv_std:.3f})")
        else:
            safe_print(f"  [PASS] CV stability: std={cv_std:.3f}")
        
        # Check 3: Test accuracy should be within reasonable range of CV
        if best_model_name in test_results:
            test_acc = test_results[best_model_name]['test_accuracy']
            if abs(test_acc - cv_acc) > 0.15:
                safe_print(f"  [WARN] Large CV-Test gap: {abs(test_acc - cv_acc):.3f}")
            else:
                safe_print(f"  [PASS] CV-Test consistency: gap={abs(test_acc - cv_acc):.3f}")
        
        # Check 4: Walk-forward should confirm predictive ability
        if best_model_name in wf_results:
            wf_acc = wf_results[best_model_name]['wf_accuracy_mean']
            if wf_acc < 0.50:
                safe_print(f"  [FAIL] Walk-forward accuracy ({wf_acc:.3f}) below random")
                passed = False
            else:
                safe_print(f"  [PASS] Walk-forward accuracy: {wf_acc:.3f}")
        
        # Check 5: Overfitting gap
        overfit_gap = results[best_model_name].get('overfitting_gap', 0)
        if overfit_gap > 0.20:
            safe_print(f"  [FAIL] Severe overfitting detected (gap={overfit_gap:.3f})")
            passed = False
        elif overfit_gap > 0.10:
            safe_print(f"  [WARN] Moderate overfitting (gap={overfit_gap:.3f})")
        else:
            safe_print(f"  [PASS] Overfitting gap: {overfit_gap:.3f}")
        
        overall = "PASSED" if passed else "FAILED"
        safe_print(f"\n  Overall stability: {overall}")
        
        return passed
    
    def _save_enhanced_model(self, model, model_name: str, feature_columns: list,
                           label_encoder, all_results: dict, scaler=None,
                           wf_results: dict = None):
        """Save the enhanced model with all artifacts."""
        
        import joblib
        from pathlib import Path
        
        # Create data directory
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Prepare model artifacts
        model_artifacts = {
            'model': model,
            'model_name': model_name,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'label_encoder': label_encoder,
            'training_results': all_results,
            'walk_forward_results': wf_results,
            'training_date': datetime.now().isoformat(),
            'model_version': '3.0_enhanced_direction',
            'improvements': [
                'binary_direction_prediction',
                'adaptive_thresholds',
                'purged_cv',
                'sample_weighting',
                'feature_selection',
                'walk_forward_validation',
                'reduced_complexity',
                'probability_calibration'
            ]
        }
        
        # Save model
        model_path = data_dir / 'enhanced_forex_model.joblib'
        joblib.dump(model_artifacts, model_path)
        safe_print(f"[SAVE] Enhanced model saved to: {model_path}")
        
        # Also update the main model file
        main_model_path = data_dir / 'best_forex_model.joblib'
        joblib.dump(model_artifacts, main_model_path)
        safe_print(f"[SAVE] Updated main model: {main_model_path}")


def main():
    """Main training function."""
    
    safe_print("=" * 70)
    safe_print("  ENHANCED FOREX ML TRAINING - Direction Accuracy Optimizer")
    safe_print("=" * 70)
    safe_print("")
    
    try:
        # Initialize trainer
        trainer = EnhancedForexTrainer()
        
        # Test database connection
        if not trainer.db.test_connection():
            safe_print("[ERROR] Database connection failed")
            return
        
        safe_print("[OK] Database connection established")
        
        # Prepare enhanced dataset
        currency_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        safe_print(f"[INFO] Training on currency pairs: {', '.join(currency_pairs)}")
        
        enhanced_data = trainer.prepare_enhanced_dataset(currency_pairs, lookback_days=800)
        
        # Train enhanced models with BINARY direction prediction
        safe_print("\n" + "=" * 70)
        safe_print("  TRAINING WITH BINARY DIRECTION (UP/DOWN)")
        safe_print("=" * 70)
        results = trainer.train_enhanced_models(test_size=0.2, use_binary_direction=True)
        
        if not results:
            safe_print("[ERROR] Training produced no results")
            return
        
        safe_print("\n" + "=" * 70)
        safe_print("  TRAINING COMPLETED!")
        safe_print("=" * 70)
        safe_print(f"  Best Model: {results['best_model_name']}")
        safe_print(f"  Features Used: {len(results['feature_columns'])}")
        safe_print(f"  Stability Check: {'PASSED' if results.get('stability_passed') else 'NEEDS ATTENTION'}")
        
        # Show top results
        safe_print("\n  Model Performance Summary:")
        safe_print("  " + "-" * 60)
        
        test_results = results.get('test_results', {})
        wf_results = results.get('wf_results', {})
        
        for model_name in sorted(test_results.keys(), 
                                key=lambda x: test_results[x]['test_accuracy'], 
                                reverse=True)[:5]:
            test_acc = test_results[model_name]['test_accuracy']
            wf_acc = wf_results.get(model_name, {}).get('wf_accuracy_mean', 'N/A')
            if isinstance(wf_acc, float):
                wf_acc = f"{wf_acc:.1%}"
            safe_print(f"  {model_name:30s}: Test={test_acc:.1%}, WF={wf_acc}")
        
        safe_print("\n  Improvement tips if accuracy is still below target:")
        safe_print("  1. Increase training data (more currency pairs + longer history)")
        safe_print("  2. Try different signal types (trend vs direction)")
        safe_print("  3. Add external data sources (economic calendar, sentiment)")
        safe_print("  4. Run hyperparameter optimization")
        safe_print("  5. Consider pair-specific models instead of universal")
        
    except Exception as e:
        safe_print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
