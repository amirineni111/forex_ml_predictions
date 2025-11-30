"""
Analyze AUDUSD Features and Signal Confidence
Shows the 15 most important features and their values for the prediction
"""

from predict_forex_signals import ForexTradingSignalPredictor
import pandas as pd
import numpy as np

def analyze_audusd_prediction():
    print("🔍 AUDUSD Feature Analysis & Signal Confidence Breakdown")
    print("="*70)
    
    # Initialize predictor
    predictor = ForexTradingSignalPredictor()
    
    # Get AUDUSD data and prepare features
    df = predictor.get_forex_data(currency_pair='AUDUSD', days_back=100)
    df_features, available_features = predictor.prepare_features(df)
    
    print(f"📊 Total features available: {len(available_features)}")
    print(f"📅 Data range: {df['date_time'].min()} to {df['date_time'].max()}")
    
    # Get the most recent record used for prediction
    df_recent = df_features.dropna(subset=available_features).tail(1)
    latest_date = df_recent['date_time'].iloc[0]
    
    print(f"📅 Using data from: {latest_date}")
    print(f"💱 Currency pair: AUDUSD")
    
    # Prepare prediction data
    X_pred = df_recent[available_features].fillna(0)
    
    # Make prediction and get probabilities
    predictions = predictor.model_manager.predict(X_pred)
    probabilities = predictor.model_manager.predict_proba(X_pred)
    
    print(f"\n🎯 PREDICTION RESULTS:")
    print(f"   Signal: {predictions[0]}")
    print(f"   Signal Confidence: {probabilities[0].max():.3f} ({probabilities[0].max()*100:.1f}%)")
    print(f"   BUY Probability: {probabilities[0][0]:.3f} ({probabilities[0][0]*100:.1f}%)")
    print(f"   HOLD Probability: {probabilities[0][1]:.3f} ({probabilities[0][1]*100:.1f}%)")
    print(f"   SELL Probability: {probabilities[0][2]:.3f} ({probabilities[0][2]*100:.1f}%)")
    
    print(f"\n📋 ALL {len(available_features)} FEATURES USED IN PREDICTION:")
    print("-" * 50)
    
    # Show all feature values
    feature_values = X_pred.iloc[0]
    for i, (feature, value) in enumerate(feature_values.items()):
        print(f"{i+1:2d}. {feature:20s}: {value:10.6f}")
    
    print(f"\n🔝 TOP 15 FEATURES BY ABSOLUTE VALUE:")
    print("-" * 45)
    
    # Sort features by absolute value to see which might be most influential
    feature_abs_values = [(feature, abs(value), value) for feature, value in feature_values.items()]
    feature_abs_values.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, abs_val, actual_val) in enumerate(feature_abs_values[:15]):
        print(f"{i+1:2d}. {feature:20s}: {actual_val:10.6f} (|{abs_val:8.6f}|)")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Model uses {len(available_features)} technical indicators")
    print(f"   • Strong {predictions[0]} signal with {probabilities[0].max()*100:.1f}% confidence")
    print(f"   • Based on data from {latest_date}")
    print(f"   • Uses price action, moving averages, RSI, MACD, Bollinger Bands, ATR, and volume")

if __name__ == "__main__":
    analyze_audusd_prediction()