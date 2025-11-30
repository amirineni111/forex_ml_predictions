"""
Fixed Forex Prediction Logic for Forward-Looking Signals
This creates predictions for TOMORROW based on TODAY's data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

def create_forward_prediction(df_recent, predictions, probabilities, currency_pair):
    """
    Create forward-looking prediction for next trading day
    
    Args:
        df_recent: Most recent market data (today)
        predictions: Model predictions 
        probabilities: Prediction probabilities
        currency_pair: Currency pair symbol
    
    Returns:
        DataFrame with tomorrow's prediction
    """
    
    # Get today's date and calculate next trading day
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    
    # Skip weekends for next trading day
    while tomorrow.weekday() >= 5:  # Saturday = 5, Sunday = 6
        tomorrow = tomorrow + timedelta(days=1)
    
    # Get recent close price for realistic estimates
    recent_close = df_recent['close_price'].iloc[0]
    
    # Check if current data looks artificial (all OHLC are identical)
    current_open = df_recent['open_price'].iloc[0]
    current_high = df_recent['high_price'].iloc[0]
    current_low = df_recent['low_price'].iloc[0]
    
    is_artificial_data = (current_open == current_high == current_low == recent_close)
    
    if is_artificial_data:
        # Use more conservative estimates for artificial data
        estimated_high = round(recent_close * 1.001, 5)  # 0.1% range
        estimated_low = round(recent_close * 0.999, 5)
        print(f"⚠️ Detected artificial data for {currency_pair}, using conservative estimates")
    else:
        # Use actual high/low as reference for real market data
        estimated_high = current_high
        estimated_low = current_low
    
    # Create prediction record for TOMORROW
    prediction_record = {
        'prediction_date': tomorrow,  # TOMORROW's date
        'currency_pair': currency_pair,
        'date_time': tomorrow,  # Prediction is for tomorrow
        'open_price': recent_close,  # Use today's close as tomorrow's expected open
        'high_price': estimated_high,  # Either real high or conservative estimate
        'low_price': estimated_low,   # Either real low or conservative estimate
        'close_price': recent_close,  # Expected close same as open initially
        'volume': df_recent['volume'].iloc[0] if 'volume' in df_recent.columns else 0,
        'predicted_signal': predictions[0],
        'signal_confidence': 0.0,
        'model_name': 'enhanced_forex_model',
        'model_version': '2.0',
        'features_used': 'technical_indicators_enhanced'
    }
    
    # Add probability columns if available
    if probabilities is not None and len(probabilities.shape) > 1:
        prob_cols = ['BUY', 'HOLD', 'SELL']  # Adjust based on your classes
        for i, col in enumerate(prob_cols):
            if i < probabilities.shape[1]:
                prediction_record[f'prob_{col.lower()}'] = probabilities[0][i]
                
        # Calculate confidence as max probability
        prediction_record['signal_confidence'] = np.max(probabilities[0])
    
    return pd.DataFrame([prediction_record])

# Create a fixed version of the predict_signals method
def predict_next_day_signals(predictor, currency_pair):
    """
    Generate prediction for NEXT trading day only
    """
    from datetime import datetime, timedelta
    
    # Get sufficient historical data for feature calculation
    df = predictor.get_forex_data(currency_pair=currency_pair, days_back=100)
    
    if df.empty:
        print("❌ No data available for prediction")
        return pd.DataFrame()
    
    print("🔄 Generating NEXT DAY trading signal...")
    
    try:
        # Prepare features using historical data
        df_features, available_features = predictor.prepare_features(df)
        
        if len(available_features) < 10:
            print("❌ Insufficient features for prediction") 
            return pd.DataFrame()
        
        # Get the MOST RECENT record (today's data)
        df_features = df_features.sort_values('date_time')
        df_today = df_features.dropna(subset=available_features).tail(1)
        
        if df_today.empty:
            print("❌ No valid data for prediction")
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
            print(f"⚠️ Warning: Detected {artificial_count} artificial data records in recent {currency_pair} data")
        
        print(f"📊 Using data from: {latest_date} to predict NEXT trading day")
        
        # Make prediction for TOMORROW using TODAY's data
        X_pred = df_today[available_features].fillna(0)
        predictions = predictor.model_manager.predict(X_pred)
        
        # Get probabilities
        try:
            probabilities = predictor.model_manager.predict_proba(X_pred)
        except:
            probabilities = None
        
        # Create forward-looking prediction record
        prediction_df = create_forward_prediction(
            df_today, predictions, probabilities, currency_pair
        )
        
        print(f"✅ Generated prediction for NEXT trading day: {prediction_df['prediction_date'].iloc[0]}")
        print(f"📈 Signal: {prediction_df['predicted_signal'].iloc[0]} (Confidence: {prediction_df['signal_confidence'].iloc[0]:.3f})")
        
        return prediction_df
        
    except Exception as e:
        print(f"❌ Error generating prediction: {e}")
        return pd.DataFrame()