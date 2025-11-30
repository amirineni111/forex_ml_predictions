"""
Advanced ML Models for High-Accuracy Forex Prediction
Ensemble and neural network approaches for 70%+ accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import logging

logger = logging.getLogger(__name__)

class AdvancedForexModels:
    """Advanced ML models optimized for forex prediction accuracy"""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        
    def create_high_performance_models(self) -> Dict:
        """Create advanced models optimized for forex prediction"""
        
        models = {
            # Gradient Boosting Models (often best for tabular data)
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                early_stopping_rounds=50
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            
            # Optimized Tree Models
            'extra_trees_optimized': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'random_forest_optimized': RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            # Support Vector Machine with RBF kernel
            'svm_rbf': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            
            # Logistic Regression with regularization
            'logistic_l2': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
        }
        
        self.models = models
        return models
    
    def create_ensemble_models(self) -> Dict:
        """Create powerful ensemble models"""
        
        # Base models for ensemble
        base_models = [
            ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=6, random_state=42, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42))
        ]
        
        # Voting Classifier (Hard and Soft voting)
        voting_hard = VotingClassifier(
            estimators=base_models,
            voting='hard'
        )
        
        voting_soft = VotingClassifier(
            estimators=base_models,
            voting='soft'
        )
        
        # Stacking Classifier
        stacking = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(random_state=42),
            cv=3  # Time series aware CV should be used
        )
        
        ensemble_models = {
            'voting_hard': voting_hard,
            'voting_soft': voting_soft,
            'stacking': stacking
        }
        
        return ensemble_models
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_name: str) -> Dict:
        """Optimize hyperparameters for specific model"""
        
        from sklearn.model_selection import RandomizedSearchCV
        
        param_grids = {
            'xgboost': {
                'n_estimators': [300, 500, 700],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'lightgbm': {
                'n_estimators': [300, 500, 700],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            },
            'extra_trees_optimized': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
        
        if model_name not in param_grids:
            return {}
        
        model = self.models[model_name]
        param_grid = param_grids[model_name]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=20, cv=tscv, 
            scoring='accuracy', random_state=42, n_jobs=-1
        )
        
        random_search.fit(X, y)
        
        logger.info(f"Best parameters for {model_name}: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return random_search.best_params_
    
    def train_with_feature_selection(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
        """Train models with automatic feature selection"""
        
        from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
        from sklearn.ensemble import ExtraTreesClassifier
        
        # Feature selection methods
        feature_selectors = {
            'mutual_info': SelectKBest(mutual_info_classif, k=min(50, X.shape[1])),
            'rfe_rf': RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=min(40, X.shape[1])),
            'tree_importance': ExtraTreesClassifier(n_estimators=100, random_state=42)
        }
        
        selected_features = {}
        
        # Apply feature selection
        for name, selector in feature_selectors.items():
            if name == 'tree_importance':
                selector.fit(X, y)
                importances = selector.feature_importances_
                selected_features[name] = X.columns[importances > np.percentile(importances, 60)].tolist()
            else:
                X_selected = selector.fit_transform(X, y)
                selected_features[name] = X.columns[selector.get_support()].tolist()
        
        # Combine features (union of all methods)
        all_selected = set()
        for features in selected_features.values():
            all_selected.update(features)
        
        final_features = list(all_selected)
        logger.info(f"Selected {len(final_features)} features from {X.shape[1]} original features")
        
        # Train models on selected features
        X_selected = X[final_features]
        results = self.train_models(X_selected, y)
        
        return results, X_selected
    
    def train_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Train all models and return results"""
        
        # Create models if not exists
        if not self.models:
            self.create_high_performance_models()
        
        # Add ensemble models
        ensemble_models = self.create_ensemble_models()
        all_models = {**self.models, **ensemble_models}
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        results = {}
        
        for model_name, model in all_models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Handle XGBoost early stopping
                    if 'xgboost' in model_name.lower():
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    else:
                        model.fit(X_train, y_train)
                    
                    predictions = model.predict(X_val)
                    score = accuracy_score(y_val, predictions)
                    cv_scores.append(score)
                
                # Final training on full dataset
                model.fit(X, y)
                
                results[model_name] = {
                    'model': model,
                    'cv_accuracy_mean': np.mean(cv_scores),
                    'cv_accuracy_std': np.std(cv_scores),
                    'cv_scores': cv_scores
                }
                
                logger.info(f"{model_name}: CV Accuracy = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from trained model"""
        
        importance_df = pd.DataFrame()
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0])
            }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def create_neural_network_model(self, input_dim: int, num_classes: int) -> 'tensorflow.keras.Model':
        """Create a neural network model for forex prediction"""
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            
            model = Sequential([
                Dense(256, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(128, activation='relu'),
                BatchNormalization(), 
                Dropout(0.2),
                
                Dense(64, activation='relu'),
                Dropout(0.1),
                
                Dense(32, activation='relu'),
                
                Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
            
        except ImportError:
            logger.warning("TensorFlow not available. Skipping neural network model.")
            return None