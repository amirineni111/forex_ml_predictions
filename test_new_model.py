"""
Test script to verify the new 3-class model (BUY/SELL/HOLD) structure.
"""
import joblib
import pandas as pd
import numpy as np

print("=" * 80)
print("TESTING NEW FOREX MODEL STRUCTURE")
print("=" * 80)

# Load the enhanced model (the one we just trained)
model_path = 'data/enhanced_forex_model.joblib'
print(f"\n[1] Loading model from: {model_path}")

try:
    model_data = joblib.load(model_path)
    print(f"✅ Model loaded successfully")
    
    # Check what's in the model data
    print(f"\n[2] Model contents:")
    for key in model_data.keys():
        print(f"   - {key}: {type(model_data[key])}")
    
    # Get the model and check its classes
    model = model_data.get('model') or model_data.get('best_model')
    if model and hasattr(model, 'classes_'):
        print(f"\n[3] Model target classes: {model.classes_}")
        print(f"   Number of classes: {len(model.classes_)}")
        
        if len(model.classes_) == 3:
            print("   ✅ This is a 3-class model (BUY/SELL/HOLD)!")
        elif len(model.classes_) == 2:
            print("   ⚠️ This is a binary model (UP/DOWN) - OLD MODEL")
    else:
        print("\n[3] Could not access model classes")
    
    # Check version/metadata
    version = model_data.get('version', 'unknown')
    print(f"\n[4] Model version: {version}")
    
    features = model_data.get('selected_features') or model_data.get('feature_names')
    if features:
        print(f"\n[5] Number of features: {len(features)}")
        print(f"   First 5 features: {features[:5]}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)

# Now check the "best" model file
print("\nCHECKING 'BEST' MODEL FILE")
print("=" * 80)

model_path = 'data/best_forex_model.joblib'
print(f"\n[1] Loading model from: {model_path}")

try:
    model_data = joblib.load(model_path)
    print(f"✅ Model loaded successfully")
    
    model = model_data.get('model') or model_data.get('best_model')
    if model and hasattr(model, 'classes_'):
        print(f"\n[2] Model target classes: {model.classes_}")
        print(f"   Number of classes: {len(model.classes_)}")
        
        if len(model.classes_) == 3:
            print("   ✅ This is a 3-class model (BUY/SELL/HOLD)!")
        elif len(model.classes_) == 2:
            print("   ⚠️ This is a binary model (UP/DOWN) - OLD MODEL")
    
    version = model_data.get('version', 'unknown')
    print(f"\n[3] Model version: {version}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("If 'enhanced_forex_model.joblib' is 3-class and 'best_forex_model.joblib' is binary:")
print("  → Copy enhanced_forex_model.joblib to best_forex_model.joblib")
print("  → OR update prediction scripts to use enhanced_forex_model.joblib")
print("=" * 80)
