"""
Forex ML Models Module

This module provides machine learning models specifically designed for forex trading
signal prediction using technical indicators like BB, EMA, SMA, RSI, MACD, ATR.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
import pickle
import joblib
from pathlib import Path
import os
import logging
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, 
    TimeSeriesSplit, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler

# Advanced ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ForexMLModelManager:
    """
    Comprehensive machine learning model manager for forex trading signals.
    
    Specialized for forex data with technical indicators and time series patterns.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the Forex ML Model Manager.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
        """
        self.task_type = task_type.lower()
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
        # Forex-specific feature columns
        self.forex_feature_columns = [
            # Price data
            'open_price', 'high_price', 'low_price', 'close_price', 'volume',
            
            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
            
            # Technical Indicators
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'atr',
            
            # Bollinger Bands
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent',
            
            # Derived features
            'daily_return', 'volatility', 'price_range', 'gap', 'volume_ratio'
        ]
        
        # Initialize models based on task type
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize forex-specific models based on task type."""
        
        base_models = {}
        
        if self.task_type == 'classification':
            base_models = {
                'logistic_regression': LogisticRegression(
                    random_state=42, max_iter=1000, class_weight='balanced'
                ),
                'random_forest': RandomForestClassifier(
                    random_state=42, n_estimators=200, max_depth=10, 
                    class_weight='balanced'
                ),
                'extra_trees': ExtraTreesClassifier(
                    random_state=42, n_estimators=200, max_depth=10,
                    class_weight='balanced'
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    random_state=42, n_estimators=200, max_depth=6
                ),
                'svm': SVC(
                    random_state=42, probability=True, class_weight='balanced'
                ),
                'naive_bayes': GaussianNB(),
                'knn': KNeighborsClassifier(n_neighbors=7)
            }
            
        elif self.task_type == 'regression':
            base_models = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(random_state=42, alpha=1.0),
                'lasso_regression': Lasso(random_state=42, alpha=1.0),
                'random_forest': RandomForestRegressor(
                    random_state=42, n_estimators=200, max_depth=10
                ),
                'extra_trees': ExtraTreesRegressor(
                    random_state=42, n_estimators=200, max_depth=10
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    random_state=42, n_estimators=200, max_depth=6
                ),
                'svm': SVR(),
                'knn': KNeighborsRegressor(n_neighbors=7)
            }
        
        # Add advanced models if available
        if HAS_XGBOOST:
            if self.task_type == 'classification':
                base_models['xgboost'] = xgb.XGBClassifier(
                    random_state=42, n_estimators=200, max_depth=6,
                    eval_metric='logloss'
                )
            else:
                base_models['xgboost'] = xgb.XGBRegressor(
                    random_state=42, n_estimators=200, max_depth=6
                )
        
        if HAS_LIGHTGBM:
            if self.task_type == 'classification':
                base_models['lightgbm'] = lgb.LGBMClassifier(
                    random_state=42, n_estimators=200, max_depth=6,
                    class_weight='balanced', verbosity=-1
                )
            else:
                base_models['lightgbm'] = lgb.LGBMRegressor(
                    random_state=42, n_estimators=200, max_depth=6,
                    verbosity=-1
                )
        
        self.models = base_models
        
    def prepare_forex_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and engineer features specifically for forex data.
        
        Args:
            df: Raw forex dataframe
            
        Returns:
            DataFrame with engineered features
        """
        df_processed = df.copy()
        
        # Sort by currency_pair and date_time for time series calculations
        if 'currency_pair' in df_processed.columns and 'date_time' in df_processed.columns:
            df_processed = df_processed.sort_values(['currency_pair', 'date_time'])
        
        # Price-based features
        if all(col in df_processed.columns for col in ['high_price', 'low_price', 'close_price', 'open_price']):
            # Price momentum features
            df_processed['price_momentum_5'] = df_processed['close_price'].pct_change(periods=5)
            df_processed['price_momentum_10'] = df_processed['close_price'].pct_change(periods=10)
            
            # Price position within range
            df_processed['price_position'] = (
                (df_processed['close_price'] - df_processed['low_price']) / 
                (df_processed['high_price'] - df_processed['low_price'] + 1e-8)
            )
            
            # Gap analysis
            if 'open_price' in df_processed.columns:
                df_processed['gap'] = (
                    (df_processed['open_price'] - df_processed['close_price'].shift(1)) / 
                    df_processed['close_price'].shift(1)
                )
        
        # Moving average ratios and crossovers
        ma_columns = [col for col in df_processed.columns if 'sma_' in col or 'ema_' in col]
        
        if 'close_price' in df_processed.columns:
            for ma_col in ma_columns:
                if ma_col in df_processed.columns:
                    # Price vs MA ratio
                    df_processed[f'price_vs_{ma_col}'] = (
                        df_processed['close_price'] / (df_processed[ma_col] + 1e-8)
                    )
        
        # MA crossover signals
        if 'sma_20' in df_processed.columns and 'sma_50' in df_processed.columns:
            df_processed['sma20_vs_sma50'] = (
                df_processed['sma_20'] / (df_processed['sma_50'] + 1e-8)
            )
        
        if 'ema_20' in df_processed.columns and 'ema_50' in df_processed.columns:
            df_processed['ema20_vs_ema50'] = (
                df_processed['ema_20'] / (df_processed['ema_50'] + 1e-8)
            )
        
        # RSI-based features
        if 'rsi_14' in df_processed.columns:
            df_processed['rsi_oversold'] = (df_processed['rsi_14'] < 30).astype(int)
            df_processed['rsi_overbought'] = (df_processed['rsi_14'] > 70).astype(int)
            df_processed['rsi_momentum'] = df_processed['rsi_14'].diff()
        
        # Bollinger Bands features
        if all(col in df_processed.columns for col in ['bb_upper', 'bb_lower', 'close_price']):
            # Calculate Bollinger Band width and position
            df_processed['bb_width'] = df_processed['bb_upper'] - df_processed['bb_lower']
            df_processed['bb_percent'] = (
                (df_processed['close_price'] - df_processed['bb_lower']) / 
                (df_processed['bb_width'] + 1e-8)
            )
            df_processed['bb_squeeze'] = (df_processed['bb_width'] < df_processed['bb_width'].rolling(20).quantile(0.2)).astype(int)
            df_processed['bb_breakout_upper'] = (df_processed['close_price'] > df_processed['bb_upper']).astype(int)
            df_processed['bb_breakout_lower'] = (df_processed['close_price'] < df_processed['bb_lower']).astype(int)
        
        # MACD features
        if all(col in df_processed.columns for col in ['macd', 'macd_signal']):
            df_processed['macd_signal_cross'] = (
                (df_processed['macd'] > df_processed['macd_signal']).astype(int).diff()
            )
        
        # Volume features (if volume data available)
        if 'volume' in df_processed.columns:
            df_processed['volume_ma_20'] = df_processed['volume'].rolling(20).mean()
            df_processed['volume_ratio'] = (
                df_processed['volume'] / (df_processed['volume_ma_20'] + 1e-8)
            )
        
        # Volatility features
        if 'close_price' in df_processed.columns:
            df_processed['volatility_10'] = df_processed['close_price'].pct_change().rolling(10).std()
            df_processed['volatility_20'] = df_processed['close_price'].pct_change().rolling(20).std()
        
        # Time-based features
        if 'date_time' in df_processed.columns:
            df_processed['hour'] = pd.to_datetime(df_processed['date_time']).dt.hour
            df_processed['day_of_week'] = pd.to_datetime(df_processed['date_time']).dt.dayofweek
            df_processed['month'] = pd.to_datetime(df_processed['date_time']).dt.month
        
        # Remove infinite and NaN values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        
        return df_processed
    
    def create_trading_signals(
        self, 
        df: pd.DataFrame, 
        signal_type: str = 'trend',
        future_periods: int = 5
    ) -> pd.DataFrame:
        """
        Create trading signals for forex data.
        
        Args:
            df: DataFrame with forex data
            signal_type: Type of signal ('trend', 'momentum', 'mean_reversion')
            future_periods: Number of periods to look ahead
            
        Returns:
            DataFrame with trading signals
        """
        df_signals = df.copy()
        
        if 'close_price' not in df_signals.columns:
            raise ValueError("close_price column is required for signal generation")
        
        # Calculate future returns
        df_signals['future_return'] = (
            df_signals['close_price'].shift(-future_periods) / df_signals['close_price'] - 1
        )
        
        if signal_type == 'trend':
            # Trend following signals
            df_signals['signal'] = np.where(df_signals['future_return'] > 0.001, 'BUY',
                                   np.where(df_signals['future_return'] < -0.001, 'SELL', 'HOLD'))
        
        elif signal_type == 'momentum':
            # Momentum-based signals with higher thresholds
            df_signals['signal'] = np.where(df_signals['future_return'] > 0.002, 'BUY',
                                   np.where(df_signals['future_return'] < -0.002, 'SELL', 'HOLD'))
        
        elif signal_type == 'mean_reversion':
            # Mean reversion signals (opposite direction)
            df_signals['signal'] = np.where(df_signals['future_return'] > 0.001, 'SELL',
                                   np.where(df_signals['future_return'] < -0.001, 'BUY', 'HOLD'))
        
        # Remove rows where we can't calculate future returns
        df_signals = df_signals.dropna(subset=['future_return', 'signal'])
        
        return df_signals
    
    def add_model(self, name: str, model: Any):
        """Add a custom model to the manager."""
        self.models[name] = model
        logger.info(f"Added forex model: {name}")
    
    def train_all_models(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        cv_folds: int = 5,
        use_time_series_cv: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models and evaluate using cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            cv_folds: Number of CV folds
            use_time_series_cv: Use TimeSeriesSplit for CV
            
        Returns:
            Dictionary of model results
        """
        
        # Setup cross-validation
        if use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            cv = cv_folds
        
        # Initialize scaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)
        
        # Encode labels if classification
        if self.task_type == 'classification':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y_train)
        else:
            y_encoded = y_train.values
        
        results = {}
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                
                if self.task_type == 'classification':
                    # Classification metrics
                    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
                    
                    # Fit model for additional metrics
                    model.fit(X_scaled, y_encoded)
                    y_pred = model.predict(X_scaled)
                    
                    results[name] = {
                        'cv_accuracy_mean': cv_scores.mean(),
                        'cv_accuracy_std': cv_scores.std(),
                        'train_accuracy': accuracy_score(y_encoded, y_pred),
                        'train_precision': precision_score(y_encoded, y_pred, average='weighted', zero_division=0),
                        'train_recall': recall_score(y_encoded, y_pred, average='weighted', zero_division=0),
                        'train_f1': f1_score(y_encoded, y_pred, average='weighted', zero_division=0)
                    }
                    
                    # Add AUC if binary classification
                    if len(np.unique(y_encoded)) == 2:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_scaled)[:, 1]
                            results[name]['train_auc'] = roc_auc_score(y_encoded, y_proba)
                
                else:
                    # Regression metrics
                    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='neg_mean_squared_error')
                    
                    # Fit model for additional metrics
                    model.fit(X_scaled, y_encoded)
                    y_pred = model.predict(X_scaled)
                    
                    results[name] = {
                        'cv_mse_mean': -cv_scores.mean(),
                        'cv_mse_std': cv_scores.std(),
                        'train_mse': mean_squared_error(y_encoded, y_pred),
                        'train_mae': mean_absolute_error(y_encoded, y_pred),
                        'train_r2': r2_score(y_encoded, y_pred)
                    }
                
                logger.info(f"✅ {name} trained successfully")
                
            except Exception as e:
                logger.error(f"❌ Error training {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        self.results = results
        
        # Find best model
        self._find_best_model()
        
        return results
    
    def _find_best_model(self):
        """Find the best performing model based on results."""
        if not self.results:
            return
        
        if self.task_type == 'classification':
            metric = 'cv_accuracy_mean'
        else:
            metric = 'cv_mse_mean'
            
        valid_results = {
            name: scores for name, scores in self.results.items() 
            if isinstance(scores, dict) and metric in scores and 'error' not in scores
        }
        
        if valid_results:
            if self.task_type == 'classification':
                best_name = max(valid_results, key=lambda x: valid_results[x][metric])
            else:
                best_name = min(valid_results, key=lambda x: valid_results[x][metric])
            
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            
            logger.info(f"🎯 Best model: {best_name}")
    
    def save_model(self, filepath: str, model_name: Optional[str] = None):
        """Save the best model or specified model to disk."""
        model_to_save = self.models.get(model_name) if model_name else self.best_model
        
        if model_to_save is None:
            logger.error("No model to save")
            return
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model artifacts
        artifacts = {
            'model': model_to_save,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.forex_feature_columns,
            'task_type': self.task_type,
            'best_model_name': self.best_model_name or model_name,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(artifacts, filepath)
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model from disk."""
        try:
            artifacts = joblib.load(filepath)
            
            self.best_model = artifacts['model']
            self.scaler = artifacts.get('scaler')
            self.label_encoder = artifacts.get('label_encoder')
            self.forex_feature_columns = artifacts.get('feature_columns', self.forex_feature_columns)
            self.best_model_name = artifacts.get('best_model_name')
            self.results = artifacts.get('results', {})
            
            logger.info(f"✅ Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best model."""
        if self.best_model is None:
            raise ValueError("No trained model available for prediction")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        predictions = self.best_model.predict(X_scaled)
        
        # Decode labels if classification
        if self.task_type == 'classification' and self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if self.best_model is None:
            raise ValueError("No trained model available for prediction")
        
        if not hasattr(self.best_model, 'predict_proba'):
            raise ValueError("Model does not support probability prediction")
        
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return self.best_model.predict_proba(X_scaled)


# Convenience functions
def create_forex_classifier() -> ForexMLModelManager:
    """Create a forex classification model manager."""
    return ForexMLModelManager(task_type='classification')


def create_forex_regressor() -> ForexMLModelManager:
    """Create a forex regression model manager."""
    return ForexMLModelManager(task_type='regression')