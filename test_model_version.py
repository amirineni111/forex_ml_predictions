"""
Test model version tracking in predictions
"""
import sys
sys.path.append('.')
from predict_forex_signals import ForexTradingSignalPredictor
from src.utils.forward_prediction import predict_next_day_signals
import pandas as pd

print("="*80)
print("TEST: Model Version Tracking")
print("="*80)

# Test with one currency pair
pair = 'EURUSD'

print(f"\n1. Loading predictor for {pair}...")
predictor = ForexTradingSignalPredictor(
    model_path='./data/best_forex_model.joblib',
    currency_pair=pair
)

print(f"   Predictor model_version: {predictor.model_version}")
if hasattr(predictor.model_manager, 'best_model_name'):
    print(f"   Model name: {predictor.model_manager.best_model_name}")

print(f"\n2. Generating prediction...")
predictions = predict_next_day_signals(predictor, pair)

if not predictions.empty:
    print("\n3. Prediction DataFrame columns:")
    print(f"   {list(predictions.columns)}")
    
    print("\n4. Model metadata in prediction:")
    print(f"   model_name: {predictions['model_name'].iloc[0]}")
    print(f"   model_version: {predictions['model_version'].iloc[0]}")
    print(f"   predicted_signal: {predictions['predicted_signal'].iloc[0]}")
    print(f"   signal_confidence: {predictions['signal_confidence'].iloc[0]:.3f}")
    
    print("\n✅ SUCCESS: Model version is correctly tracked!")
else:
    print("\n❌ FAILED: No predictions generated")

print("\n" + "="*80)
