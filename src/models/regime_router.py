"""
Regime Router — Fix 5: Separate bull/bear model routing.

Detects the current market regime from recent close prices and routes the
prediction request to a regime-specific model when one is available,
falling back to the general stacking/best model otherwise.

Usage in predict_forex_signals.py
----------------------------------
from models.regime_router import RegimeRouter

router = RegimeRouter()
regime = router.detect_regime(recent_closes)          # 'bull', 'bear', 'neutral'
model, model_label = router.route(regime, models)     # picks best model for regime
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum number of samples needed to train a regime-specific model
_MIN_REGIME_SAMPLES = 50

# Lookback window (in periods) for slope-based regime detection
_REGIME_LOOKBACK = 20

# Annualised daily-return slope threshold that separates bull from bear
# +0.1% per day => ~25% annualised  (treat as bull)
# -0.1% per day => ~-25% annualised (treat as bear)
_SLOPE_THRESHOLD = 0.001


class RegimeRouter:
    """
    Detects market regime and routes to the appropriate trained model.

    Attributes
    ----------
    regime_models : dict
        Maps regime labels ('bull', 'bear', 'neutral') to fitted sklearn models.
    fallback_model_key : str
        Key used when no regime model is available.
    """

    def __init__(self, lookback: int = _REGIME_LOOKBACK):
        self.lookback = lookback
        self.regime_models: Dict[str, Any] = {}
        self.fallback_model_key = "stacking"

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def detect_regime(self, recent_closes: List[float]) -> str:
        """
        Classify the current market regime from recent close prices.

        Uses the normalised linear-regression slope over the last
        ``self.lookback`` periods:

        - slope  >  +THRESHOLD → 'bull'
        - slope  <  -THRESHOLD → 'bear'
        - otherwise             → 'neutral'

        Parameters
        ----------
        recent_closes : list[float]
            Chronological close prices (oldest first).

        Returns
        -------
        str
            'bull', 'bear', or 'neutral'.
        """
        if len(recent_closes) < self.lookback:
            logger.warning(
                "RegimeRouter: only %d closes available (need %d) — defaulting to 'neutral'",
                len(recent_closes), self.lookback
            )
            return "neutral"

        prices = np.asarray(recent_closes[-self.lookback:], dtype=float)

        if np.isnan(prices).any() or prices.mean() == 0:
            return "neutral"

        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        normalised_slope = slope / prices.mean()   # as fraction of mean price per period

        if normalised_slope > _SLOPE_THRESHOLD:
            regime = "bull"
        elif normalised_slope < -_SLOPE_THRESHOLD:
            regime = "bear"
        else:
            regime = "neutral"

        logger.info(
            "RegimeRouter: slope=%.6f (norm=%.6f) → regime='%s'",
            slope, normalised_slope, regime
        )
        return regime

    def detect_regime_from_df(self, df: pd.DataFrame, price_col: str = "close_price") -> str:
        """
        Convenience wrapper: extract close prices from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a column named ``price_col`` and ideally a date column.
        price_col : str
            Name of the close-price column.

        Returns
        -------
        str
            'bull', 'bear', or 'neutral'.
        """
        if price_col not in df.columns:
            logger.warning("RegimeRouter: column '%s' not found — returning 'neutral'", price_col)
            return "neutral"

        closes = df[price_col].dropna().tolist()
        return self.detect_regime(closes)

    # ------------------------------------------------------------------
    # Model routing
    # ------------------------------------------------------------------

    def register_regime_model(self, regime: str, model: Any) -> None:
        """
        Register a pre-trained model for a specific regime.

        Parameters
        ----------
        regime : str
            'bull', 'bear', or 'neutral'.
        model : sklearn estimator
            A fitted estimator with ``predict`` / ``predict_proba`` methods.
        """
        if regime not in ("bull", "bear", "neutral"):
            raise ValueError(f"regime must be 'bull', 'bear', or 'neutral'; got {regime!r}")
        self.regime_models[regime] = model
        logger.info("RegimeRouter: registered model for regime='%s'", regime)

    def route(
        self,
        regime: str,
        models: Dict[str, Any],
    ) -> Tuple[Any, str]:
        """
        Return the best model for the given regime.

        Looks for a regime-specific model first; falls back to the
        ``fallback_model_key`` (usually 'stacking') and then to the
        first available model in ``models``.

        Parameters
        ----------
        regime : str
            Current market regime from ``detect_regime``.
        models : dict
            Dictionary of fitted models keyed by name.

        Returns
        -------
        (model, label) : tuple
            The selected model and a string label for logging.
        """
        # 1. Regime-specific model (registered via register_regime_model)
        if regime in self.regime_models:
            logger.info("RegimeRouter: using regime-specific model for '%s'", regime)
            return self.regime_models[regime], f"regime_{regime}"

        # 2. Fallback: general stacking / best model from the models dict
        if self.fallback_model_key in models:
            logger.info(
                "RegimeRouter: no regime model for '%s' — using fallback '%s'",
                regime, self.fallback_model_key
            )
            return models[self.fallback_model_key], self.fallback_model_key

        # 3. Last resort: first available model
        if models:
            first_key = next(iter(models))
            logger.warning(
                "RegimeRouter: fallback key '%s' not found — using '%s'",
                self.fallback_model_key, first_key
            )
            return models[first_key], first_key

        raise ValueError("RegimeRouter.route: no models available at all")

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def build_regime_models(
        self,
        df: pd.DataFrame,
        X: pd.DataFrame,
        y: pd.Series,
        model_factory,
        regime_col: str = "regime",
    ) -> Dict[str, Any]:
        """
        Train separate models for each regime label in ``df[regime_col]``.

        This is an optional enhancement — call during weekly retraining after
        you have detected and labelled regimes in the historical dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Full training dataframe with a 'regime' column.
        X : pd.DataFrame
            Feature matrix (same row order as ``df``).
        y : pd.Series
            Target labels (same row order as ``df``).
        model_factory : callable
            A function that returns a fresh unfitted sklearn estimator, e.g.
            ``lambda: XGBClassifier(...)``.
        regime_col : str
            Column in ``df`` that contains the regime label.

        Returns
        -------
        dict
            Maps regime names to fitted models.  Also auto-registers them
            via ``register_regime_model``.
        """
        if regime_col not in df.columns:
            logger.warning(
                "RegimeRouter.build_regime_models: column '%s' not in df — skipping",
                regime_col
            )
            return {}

        trained: Dict[str, Any] = {}

        for regime in ("bull", "bear", "neutral"):
            mask = df[regime_col] == regime
            n = mask.sum()

            if n < _MIN_REGIME_SAMPLES:
                logger.warning(
                    "RegimeRouter: only %d samples for regime='%s' (min=%d) — skipping",
                    n, regime, _MIN_REGIME_SAMPLES
                )
                continue

            X_r, y_r = X[mask], y[mask]
            model = model_factory()

            try:
                model.fit(X_r, y_r)
                trained[regime] = model
                self.register_regime_model(regime, model)
                logger.info(
                    "RegimeRouter: trained '%s' model on %d samples", regime, n
                )
            except Exception as exc:
                logger.error(
                    "RegimeRouter: failed to train model for regime='%s': %s",
                    regime, exc
                )

        return trained

    def label_regimes(
        self,
        df: pd.DataFrame,
        price_col: str = "close_price",
        lookback: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Add a 'regime' column to a training dataframe using a rolling slope.

        Each row is labelled with the regime that was in effect at that date,
        allowing ``build_regime_models`` to split training data by regime.

        Parameters
        ----------
        df : pd.DataFrame
            Must be sorted chronologically and contain ``price_col``.
        price_col : str
            Name of the close-price column.
        lookback : int, optional
            Override the instance-level lookback window.

        Returns
        -------
        pd.DataFrame
            A copy with an added 'regime' column.
        """
        lb = lookback or self.lookback
        df = df.copy()

        prices = df[price_col].values.astype(float)
        regimes = []

        for i in range(len(prices)):
            start = max(0, i - lb + 1)
            window = prices[start: i + 1]

            if len(window) < max(2, lb // 2) or np.isnan(window).any() or window.mean() == 0:
                regimes.append("neutral")
                continue

            x = np.arange(len(window))
            slope = np.polyfit(x, window, 1)[0]
            norm_slope = slope / window.mean()

            if norm_slope > _SLOPE_THRESHOLD:
                regimes.append("bull")
            elif norm_slope < -_SLOPE_THRESHOLD:
                regimes.append("bear")
            else:
                regimes.append("neutral")

        df["regime"] = regimes
        return df
