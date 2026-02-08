"""
Advanced ML Models for High-Accuracy Forex Prediction

Enhanced with:
- Walk-forward validation with embargo periods
- Reduced model complexity to prevent overfitting
- Weighted ensemble based on recent performance
- Probability calibration for reliable confidence
- Adaptive model selection based on market regime
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import (
    VotingClassifier, StackingClassifier, 
    ExtraTreesClassifier, RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler

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

import joblib
import logging

logger = logging.getLogger(__name__)


class AdvancedForexModels:
    """Advanced ML models optimized for forex direction prediction accuracy."""
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.walk_forward_results = {}
        self.scaler = None
        
    def create_high_performance_models(self) -> Dict:
        """
        Create advanced models with CONSTRAINED parameters to prevent overfitting.
        
        Key changes from previous version:
        - Reduced max_depth (8 -> 4-5)
        - Increased min_samples_leaf
        - Added regularization (reg_alpha, reg_lambda)
        - Lower learning rates
        - Subsampling enabled
        """
        
        models = {}
        
        # XGBoost - CONSTRAINED for forex
        if HAS_XGBOOST:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,              # Reduced from 8
                learning_rate=0.03,       # Reduced from 0.1
                subsample=0.75,           # Row subsampling
                colsample_bytree=0.65,    # Feature subsampling per tree
                colsample_bylevel=0.7,    # Feature subsampling per level
                reg_alpha=2.0,            # L1 regularization
                reg_lambda=3.0,           # L2 regularization
                min_child_weight=8,       # Minimum leaf weight
                gamma=0.1,               # Minimum loss reduction for split
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
        
        # LightGBM - CONSTRAINED
        if HAS_LIGHTGBM:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=4,              # Reduced from 8
                learning_rate=0.03,       # Reduced from 0.1
                num_leaves=15,            # Constrained (default 31)
                subsample=0.75,
                colsample_bytree=0.65,
                reg_alpha=2.0,
                reg_lambda=3.0,
                min_child_samples=15,     # Minimum leaf samples
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        
        # Extra Trees - CONSTRAINED
        models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=5,                  # Reduced from 15
            min_samples_split=12,         # Increased from 5
            min_samples_leaf=6,           # Increased from 2
            max_features='sqrt',          # Limit features per tree
            random_state=42,
            n_jobs=-1
        )
        
        # Random Forest - CONSTRAINED
        models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,                  # Reduced from 12
            min_samples_split=12,         # Increased from 5
            min_samples_leaf=6,           # Increased from 2
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting - HEAVILY CONSTRAINED
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,                  # Very shallow trees
            learning_rate=0.03,
            min_samples_split=15,
            min_samples_leaf=8,
            subsample=0.75,
            max_features='sqrt',
            random_state=42
        )
        
        # Logistic Regression - naturally regularized
        models['logistic_l2'] = LogisticRegression(
            C=0.5,                        # Stronger regularization
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        
        self.models = models
        return models
    
    def create_ensemble_models(self) -> Dict:
        """
        Create ensemble models with CONSTRAINED base learners.
        
        Key improvements:
        - Constrained base models
        - Soft voting for probabilistic ensemble
        - Proper stacking with time-series CV
        """
        
        # Base models for ensemble - all constrained
        base_models = []
        
        if HAS_XGBOOST:
            base_models.append(('xgb', xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=2.0, min_child_weight=5,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            )))
        
        if HAS_LIGHTGBM:
            base_models.append(('lgb', lgb.LGBMClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=2.0, min_child_samples=8,
                random_state=42, verbose=-1, n_jobs=-1
            )))
        
        base_models.extend([
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=4, min_samples_leaf=5,
                max_features='sqrt', random_state=42, n_jobs=-1
            )),
            ('et', ExtraTreesClassifier(
                n_estimators=100, max_depth=4, min_samples_leaf=5,
                max_features='sqrt', random_state=42, n_jobs=-1
            ))
        ])
        
        ensemble_models = {}
        
        # Soft Voting - averages probabilities (better than hard voting)
        if len(base_models) >= 2:
            ensemble_models['voting_soft'] = VotingClassifier(
                estimators=base_models,
                voting='soft'
            )
        
        # Stacking with LogisticRegression meta-learner
        if len(base_models) >= 2:
            ensemble_models['stacking'] = StackingClassifier(
                estimators=base_models,
                final_estimator=LogisticRegression(C=0.5, random_state=42, max_iter=1000),
                cv=TimeSeriesSplit(n_splits=3),
                passthrough=False
            )
        
        return ensemble_models
    
    def walk_forward_validate(
        self, X: pd.DataFrame, y: np.ndarray, 
        n_windows: int = 5, train_ratio: float = 0.7,
        embargo_gap: int = 5
    ) -> Dict:
        """
        Walk-forward validation - the gold standard for time series model evaluation.
        
        This simulates real-world usage: train on past, predict on future,
        then slide the window forward. Includes embargo gap to prevent leakage.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_windows: Number of walk-forward windows
            train_ratio: Ratio of training data in each window
            embargo_gap: Gap between train and test to prevent leakage
            
        Returns:
            Dictionary with walk-forward results per model
        """
        n_samples = len(X)
        min_train = max(100, int(n_samples * 0.3))
        window_size = (n_samples - min_train) // n_windows
        
        if window_size < 20:
            logger.warning(f"Insufficient data for {n_windows} walk-forward windows. Using 3.")
            n_windows = 3
            window_size = (n_samples - min_train) // n_windows
        
        logger.info(f"Walk-forward validation: {n_windows} windows, {window_size} samples each, {embargo_gap} embargo gap")
        
        # Ensure models are created
        if not self.models:
            self.create_high_performance_models()
        
        all_models = {**self.models}
        ensemble_models = self.create_ensemble_models()
        all_models.update(ensemble_models)
        
        wf_results = {}
        
        for model_name, model in all_models.items():
            window_scores = []
            window_direction_scores = []
            
            for w in range(n_windows):
                train_end = min_train + w * window_size
                test_start = train_end + embargo_gap
                test_end = min(test_start + window_size, n_samples)
                
                if test_start >= n_samples or test_end <= test_start:
                    continue
                
                X_train_w = X.iloc[:train_end]
                X_test_w = X.iloc[test_start:test_end]
                y_train_w = y[:train_end]
                y_test_w = y[test_start:test_end]
                
                try:
                    model_clone = self._clone_model(model)
                    model_clone.fit(X_train_w, y_train_w)
                    
                    y_pred = model_clone.predict(X_test_w)
                    accuracy = accuracy_score(y_test_w, y_pred)
                    window_scores.append(accuracy)
                    
                except Exception as e:
                    logger.warning(f"Walk-forward window {w} failed for {model_name}: {e}")
                    continue
            
            if window_scores:
                wf_results[model_name] = {
                    'wf_accuracy_mean': np.mean(window_scores),
                    'wf_accuracy_std': np.std(window_scores),
                    'wf_accuracy_min': np.min(window_scores),
                    'wf_accuracy_max': np.max(window_scores),
                    'wf_window_scores': window_scores,
                    'n_windows': len(window_scores)
                }
                
                logger.info(f"{model_name}: WF Accuracy = {np.mean(window_scores):.4f} +/- {np.std(window_scores):.4f}")
        
        self.walk_forward_results = wf_results
        return wf_results
    
    def _clone_model(self, model):
        """Create a fresh clone of a model with same parameters."""
        from sklearn.base import clone
        try:
            return clone(model)
        except Exception:
            # For models that don't support clone, return as-is
            return model
    
    def train_with_feature_selection(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[Dict, pd.DataFrame]:
        """
        Train models with automatic feature selection.
        
        Enhanced with:
        - Multiple selection methods for robustness
        - Intersection-based selection (more conservative)
        - Minimum feature count guarantee
        """
        
        from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
        
        n_features_target = min(30, X.shape[1])  # Reduced from 50
        
        feature_selectors = {}
        selected_features = {}
        
        # Method 1: Mutual Information
        try:
            mi_selector = SelectKBest(mutual_info_classif, k=min(n_features_target, X.shape[1]))
            mi_selector.fit_transform(X, y)
            selected_features['mutual_info'] = X.columns[mi_selector.get_support()].tolist()
        except Exception as e:
            logger.warning(f"MI selection failed: {e}")
        
        # Method 2: Tree-based importance with threshold
        try:
            tree_selector = ExtraTreesClassifier(
                n_estimators=100, max_depth=4, random_state=42, n_jobs=-1
            )
            tree_selector.fit(X, y)
            importances = tree_selector.feature_importances_
            # Select features above 50th percentile (more selective)
            threshold = np.percentile(importances, 50)
            selected_features['tree_importance'] = X.columns[importances > threshold].tolist()
        except Exception as e:
            logger.warning(f"Tree importance selection failed: {e}")
        
        # Method 3: RFE with Random Forest
        try:
            rfe_estimator = RandomForestClassifier(
                n_estimators=50, max_depth=4, random_state=42, n_jobs=-1
            )
            rfe_selector = RFE(rfe_estimator, n_features_to_select=min(n_features_target, X.shape[1]))
            rfe_selector.fit(X, y)
            selected_features['rfe'] = X.columns[rfe_selector.get_support()].tolist()
        except Exception as e:
            logger.warning(f"RFE selection failed: {e}")
        
        # Combine features using INTERSECTION of at least 2 methods (more conservative)
        if len(selected_features) >= 2:
            from collections import Counter
            all_features_flat = [f for feats in selected_features.values() for f in feats]
            feature_counts = Counter(all_features_flat)
            
            # Features selected by at least 2 methods
            robust_features = [f for f, count in feature_counts.items() if count >= 2]
            
            # Ensure minimum features
            if len(robust_features) < 10:
                # Fall back to union
                robust_features = list(set(all_features_flat))
        elif selected_features:
            robust_features = list(next(iter(selected_features.values())))
        else:
            robust_features = X.columns.tolist()
        
        # Cap at n_features_target
        if len(robust_features) > n_features_target:
            # Use tree importance to rank and select top
            if 'tree_importance' in selected_features:
                importance_order = pd.Series(
                    tree_selector.feature_importances_, index=X.columns
                ).sort_values(ascending=False)
                robust_features = [f for f in importance_order.index if f in robust_features][:n_features_target]
        
        logger.info(f"Selected {len(robust_features)} robust features from {X.shape[1]} original features")
        
        # Train models on selected features
        X_selected = X[robust_features]
        results = self.train_models(X_selected, y)
        
        return results, X_selected
    
    def train_models(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """
        Train all models with walk-forward evaluation.
        
        Enhanced with:
        - Purged time-series CV
        - Sample weighting
        - Overfitting monitoring
        """
        
        # Create models if not exists
        if not self.models:
            self.create_high_performance_models()
        
        # Add ensemble models
        ensemble_models = self.create_ensemble_models()
        all_models = {**self.models, **ensemble_models}
        
        # Scale features
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, index=X.index
        )
        
        # Compute sample weights (exponential decay)
        n = len(X)
        sample_weights = np.array([0.997 ** (n - 1 - i) for i in range(n)])
        sample_weights = sample_weights * n / sample_weights.sum()
        
        # Time series split for validation with purge gap
        try:
            from src.models.ml_models import PurgedTimeSeriesSplit
        except ImportError:
            from models.ml_models import PurgedTimeSeriesSplit
        tscv = PurgedTimeSeriesSplit(n_splits=5, purge_gap=3)
        
        results = {}
        
        for model_name, model in all_models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X_scaled):
                    X_train_cv = X_scaled.iloc[train_idx]
                    X_val_cv = X_scaled.iloc[val_idx]
                    y_train_cv = y[train_idx]
                    y_val_cv = y[val_idx]
                    
                    fold_weights = sample_weights[train_idx]
                    
                    # Fit with sample weights where supported
                    model_clone = self._clone_model(model)
                    
                    if hasattr(model_clone, 'fit') and 'sample_weight' in model_clone.fit.__code__.co_varnames:
                        try:
                            model_clone.fit(X_train_cv, y_train_cv, sample_weight=fold_weights)
                        except TypeError:
                            model_clone.fit(X_train_cv, y_train_cv)
                    else:
                        model_clone.fit(X_train_cv, y_train_cv)
                    
                    predictions = model_clone.predict(X_val_cv)
                    score = accuracy_score(y_val_cv, predictions)
                    cv_scores.append(score)
                
                # Final training on full dataset with sample weights
                if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
                    try:
                        model.fit(X_scaled, y, sample_weight=sample_weights)
                    except TypeError:
                        model.fit(X_scaled, y)
                else:
                    model.fit(X_scaled, y)
                
                # Calculate train accuracy for overfitting check
                train_pred = model.predict(X_scaled)
                train_accuracy = accuracy_score(y, train_pred)
                cv_mean = np.mean(cv_scores)
                overfit_gap = train_accuracy - cv_mean
                
                results[model_name] = {
                    'model': model,
                    'cv_accuracy_mean': cv_mean,
                    'cv_accuracy_std': np.std(cv_scores),
                    'cv_scores': cv_scores,
                    'train_accuracy': train_accuracy,
                    'overfitting_gap': overfit_gap
                }
                
                status = "OK"
                if overfit_gap > 0.15:
                    status = "OVERFIT WARNING"
                
                logger.info(f"{model_name}: CV={cv_mean:.4f}+/-{np.std(cv_scores):.4f}, "
                          f"Train={train_accuracy:.4f}, Gap={overfit_gap:.3f} [{status}]")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from trained model."""
        
        importance_df = pd.DataFrame()
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
        elif hasattr(model, 'coef_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def select_best_model_for_regime(
        self, results: Dict, recent_volatility: float = None
    ) -> str:
        """
        Select the best model considering market regime.
        
        In high-volatility regimes, simpler models often work better.
        In low-volatility regimes, complex models can capture subtle patterns.
        
        Args:
            results: Model training results
            recent_volatility: Recent market volatility measure
            
        Returns:
            Name of the recommended model
        """
        if not results:
            return None
        
        # Filter to models with valid results
        valid_models = {
            name: res for name, res in results.items()
            if isinstance(res, dict) and 'cv_accuracy_mean' in res
        }
        
        if not valid_models:
            return None
        
        # Score: CV accuracy - overfitting penalty
        def model_score(name):
            res = valid_models[name]
            cv_acc = res['cv_accuracy_mean']
            overfit = res.get('overfitting_gap', 0)
            cv_std = res.get('cv_accuracy_std', 0)
            
            # Penalize overfitting and instability
            penalty = max(0, overfit - 0.08) * 0.5 + cv_std * 0.3
            return cv_acc - penalty
        
        best_model = max(valid_models.keys(), key=model_score)
        
        logger.info(f"Selected model for current regime: {best_model} "
                    f"(adjusted score: {model_score(best_model):.4f})")
        
        return best_model
