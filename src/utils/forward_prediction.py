"""
Fixed Forex Prediction Logic for Forward-Looking Signals
This creates predictions for TOMORROW based on TODAY's data.

Enhanced to support:
- Binary direction model (UP/DOWN -> BUY/SELL mapping)
- Calibrated probability outputs
- Confidence-based signal strength
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

try:
    from .signal_policy import apply_signal_thresholds, technical_veto, gate_binary_signal
except ImportError:  # imported without package context (e.g. sys.path points at src/utils)
    from signal_policy import apply_signal_thresholds, technical_veto, gate_binary_signal


def create_forward_prediction(df_recent, predictions, probabilities, currency_pair,
                            model_name='enhanced_forex_model', model_version='3.0',
                            label_classes=None):
    """
    Create forward-looking prediction for next trading day.

    Applies the shared signal policy (asymmetric thresholds + technical veto,
    src/utils/signal_policy.py) so the scheduled daily run enforces the same
    rules as predict_forex_signals.py. 'HOLD' in the output means the model
    ABSTAINED (no side cleared its threshold, or a SELL was vetoed) — it is
    not a predicted class of the binary model, and prob_hold stays 0.0.

    Args:
        df_recent: Most recent market data (today)
        predictions: Model predictions
        probabilities: Prediction probabilities
        currency_pair: Currency pair symbol
        model_name: Name of the model used
        model_version: Version of the model used
        label_classes: The model's label_encoder.classes_ — used to map
            probability columns by class NAME instead of assuming positions

    Returns:
        DataFrame with tomorrow's prediction
    """
    
    # Get today's date and calculate next trading day
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    # Skip weekends for next trading day
    while tomorrow.weekday() >= 5:  # Saturday = 5, Sunday = 6
        tomorrow = tomorrow + timedelta(days=1)
    
    # Get recent close price
    recent_close = df_recent['close_price'].iloc[0]
    
    # Check if current data looks artificial (all OHLC are identical)
    current_open = df_recent['open_price'].iloc[0]
    current_high = df_recent['high_price'].iloc[0]
    current_low = df_recent['low_price'].iloc[0]
    
    is_artificial_data = (current_open == current_high == current_low == recent_close)
    
    if is_artificial_data:
        estimated_high = round(recent_close * 1.001, 5)
        estimated_low = round(recent_close * 0.999, 5)
        print(f"[WARN] Detected artificial data for {currency_pair}, using conservative estimates")
    else:
        estimated_high = current_high
        estimated_low = current_low
    
    # Map direction predictions to signals (raw argmax; may be overridden by the
    # signal policy below when probabilities are available)
    raw_signal = predictions[0]
    signal_mapping = {'UP': 'BUY', 'DOWN': 'SELL'}
    predicted_signal = signal_mapping.get(raw_signal, raw_signal)

    # Mark the artifact as policy-gated in the version string
    if 'gated' not in str(model_version):
        model_version = f"{model_version}+gated"

    # Create prediction record for TOMORROW
    prediction_record = {
        'prediction_date': tomorrow,
        'currency_pair': currency_pair,
        'date_time': tomorrow,
        'open_price': recent_close,
        'high_price': estimated_high,
        'low_price': estimated_low,
        'close_price': recent_close,
        'volume': df_recent['volume'].iloc[0] if 'volume' in df_recent.columns else 0,
        'predicted_signal': predicted_signal,
        'signal_confidence': 0.0,
        'model_name': model_name,
        'model_version': model_version,
        'features_used': 'technical_indicators_enhanced',
        'gate_reason': 'ungated',  # transient; exporter whitelists table columns
    }

    # Add probability columns and apply the shared signal policy
    if probabilities is not None and len(probabilities.shape) > 1:
        n_classes = probabilities.shape[1]
        probs = probabilities[0]

        # Resolve class names from the model's label encoder rather than
        # assuming positions (LabelEncoder happens to sort alphabetically
        # today, but nothing guarantees a given artifact's order).
        if label_classes is not None and len(label_classes) == n_classes:
            classes = [str(c).upper() for c in label_classes]
        elif n_classes == 2:
            classes = ['DOWN', 'UP']
        else:
            classes = ['BUY', 'HOLD', 'SELL'][:n_classes]

        prob_by_class = {c: float(p) for c, p in zip(classes, probs)}
        today_row = df_recent.iloc[0]

        if n_classes == 2:
            # Binary direction model: DOWN -> SELL side, UP -> BUY side.
            prob_sell = prob_by_class.get('DOWN', 0.0)
            prob_buy = prob_by_class.get('UP', 0.0)
            prediction_record['prob_sell'] = prob_sell
            prediction_record['prob_buy'] = prob_buy
            prediction_record['prob_hold'] = 0.0  # HOLD = abstain, never a class here

            gated_signal, gate_reason = gate_binary_signal(prob_buy, prob_sell, today_row)
        else:
            # 3-class artifact (defensive: e.g. an old model restored from backup)
            prediction_record['prob_buy'] = prob_by_class.get('BUY', 0.0)
            prediction_record['prob_hold'] = prob_by_class.get('HOLD', 0.0)
            prediction_record['prob_sell'] = prob_by_class.get('SELL', 0.0)

            thresholded = apply_signal_thresholds({
                'BUY': prediction_record['prob_buy'],
                'HOLD': prediction_record['prob_hold'],
                'SELL': prediction_record['prob_sell'],
            })
            gated_signal = technical_veto(thresholded, today_row)
            if gated_signal != thresholded:
                gate_reason = 'veto'
            elif gated_signal == 'HOLD' and predicted_signal != 'HOLD':
                gate_reason = 'abstain'
            else:
                gate_reason = 'threshold'

        if gated_signal != predicted_signal:
            print(f"[INFO] {currency_pair}: gated {predicted_signal} -> {gated_signal} "
                  f"(buy={prediction_record.get('prob_buy', 0):.3f}, "
                  f"sell={prediction_record.get('prob_sell', 0):.3f}, reason={gate_reason})")

        prediction_record['predicted_signal'] = gated_signal
        prediction_record['gate_reason'] = gate_reason

        # Confidence stays the max class probability even when abstaining, so a
        # 0.60/0.40 abstain is distinguishable from a 0.51/0.49 one downstream.
        prediction_record['signal_confidence'] = float(np.max(probs))

    return pd.DataFrame([prediction_record])


def predict_next_day_signals(predictor, currency_pair):
    """
    Generate prediction for NEXT trading day only.
    
    Enhanced to work with both v2 and v3 model formats.
    """
    
    # Get sufficient historical data for feature calculation
    df = predictor.get_forex_data(currency_pair=currency_pair, days_back=100)
    
    if df.empty:
        print("[ERROR] No data available for prediction")
        return pd.DataFrame()
    
    print("[PROCESSING] Generating NEXT DAY trading signal...")
    
    try:
        # Prepare features using historical data
        df_features, available_features = predictor.prepare_features(df)
        
        if len(available_features) < 10:
            print("[ERROR] Insufficient features for prediction") 
            return pd.DataFrame()
        
        # Get the MOST RECENT record (today's data). Features that are
        # structurally NaN for some pairs (e.g. rate_* diffs for currencies
        # with no FRED coverage: HKD/SGD/INR) are median-filled here exactly
        # as in training — the fill must happen BEFORE the dropna row filter,
        # otherwise those pairs lose every row and get no prediction.
        df_features = df_features.sort_values('date_time')
        fill_values = getattr(predictor, 'feature_fill_values', None) or {}
        if fill_values:
            fillable = [c for c in available_features
                        if c in df_features.columns and c in fill_values]
            if fillable:
                df_features[fillable] = df_features[fillable].fillna(pd.Series(fill_values))
        df_today = df_features.dropna(subset=available_features).tail(1)
        
        if df_today.empty:
            print("[ERROR] No valid data for prediction")
            return pd.DataFrame()
        
        latest_date = df_today['date_time'].iloc[0]
        
        # Check for artificial data in recent records
        recent_records = df_features.tail(3)
        artificial_count = 0
        for _, row in recent_records.iterrows():
            if (row.get('open_price') == row.get('high_price') == 
                row.get('low_price') == row.get('close_price')):
                artificial_count += 1
        
        if artificial_count >= 2:
            print(f"[WARN] Detected {artificial_count} artificial data records in recent {currency_pair} data")
        
        print(f"[DATA] Using data from: {latest_date} to predict NEXT trading day")
        
        # Make prediction for TOMORROW using TODAY's data. NaN fill uses the
        # training-time medians from the artifact (fill-value parity); 0 only
        # as last resort for pre-5.1 artifacts without feature_fill_values.
        X_pred = df_today[available_features].copy()
        X_pred = X_pred.replace([np.inf, -np.inf], np.nan)
        fill_values = getattr(predictor, 'feature_fill_values', None) or {}
        if fill_values:
            X_pred = X_pred.fillna(pd.Series(fill_values))
        X_pred = X_pred.fillna(0)
        
        predictions = predictor.model_manager.predict(X_pred)
        
        # Get probabilities
        try:
            probabilities = predictor.model_manager.predict_proba(X_pred)
        except:
            probabilities = None
        
        # Create forward-looking prediction record
        # Get model info from predictor
        model_name = getattr(predictor.model_manager, 'best_model_name', 'enhanced_forex_model')
        model_version = getattr(predictor, 'model_version', '3.0')

        label_encoder = getattr(predictor.model_manager, 'label_encoder', None)
        label_classes = getattr(label_encoder, 'classes_', None)

        prediction_df = create_forward_prediction(
            df_today, predictions, probabilities, currency_pair,
            model_name=model_name, model_version=model_version,
            label_classes=label_classes
        )
        
        signal = prediction_df['predicted_signal'].iloc[0]
        confidence = prediction_df['signal_confidence'].iloc[0]
        pred_date = prediction_df['prediction_date'].iloc[0]
        
        # Confidence label
        conf_label = "HIGH" if confidence > 0.65 else "MEDIUM" if confidence > 0.55 else "LOW"
        
        print(f"[OK] Prediction for {pred_date}: {signal} (Confidence: {confidence:.3f} - {conf_label})")
        
        return prediction_df
        
    except Exception as e:
        print(f"[ERROR] Error generating prediction: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
