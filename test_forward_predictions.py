"""
Test Forward-Looking Predictions
Verify that predictions are generated for TOMORROW, not past days
"""

import sys
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.append('.')

def test_forward_predictions():
    """Test that predictions are forward-looking"""
    
    print("🧪 Testing Forward-Looking Predictions")
    print("=" * 50)
    
    try:
        from predict_forex_signals import ForexTradingSignalPredictor
        from src.utils.forward_prediction import predict_next_day_signals
        
        # Initialize predictor
        predictor = ForexTradingSignalPredictor(
            model_path='./data/best_forex_model.joblib'
        )
        
        # Test with a currency pair
        test_pair = 'EURUSD'
        print(f"📊 Testing predictions for {test_pair}")
        
        # Generate forward-looking prediction
        prediction = predict_next_day_signals(predictor, test_pair)
        
        if not prediction.empty:
            pred_date = prediction['prediction_date'].iloc[0]
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            
            # Skip weekends
            while tomorrow.weekday() >= 5:
                tomorrow = tomorrow + timedelta(days=1)
            
            print(f"\n✅ SUCCESS! Prediction generated for: {pred_date}")
            print(f"📅 Today's date: {today}")
            print(f"📅 Expected prediction date: {tomorrow}")
            print(f"📈 Predicted signal: {prediction['predicted_signal'].iloc[0]}")
            print(f"🎯 Signal confidence: {prediction['signal_confidence'].iloc[0]:.3f}")
            
            # Verify it's forward-looking
            if pd.to_datetime(pred_date).date() >= tomorrow:
                print("\n🎉 CORRECT: Prediction is for future date!")
            else:
                print(f"\n❌ ERROR: Prediction is for past date: {pred_date}")
                
            return True
        else:
            print("❌ No prediction generated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forward_predictions()
    
    if success:
        print("\n✅ Test passed! Predictions are now forward-looking.")
        print("\n💡 Next steps:")
        print("  1. Run: python daily_forex_automation.py --run-now")
        print("  2. Check database for tomorrow's predictions")
        print("  3. Set up automation for daily forward predictions")
    else:
        print("\n❌ Test failed. Check the error messages above.")