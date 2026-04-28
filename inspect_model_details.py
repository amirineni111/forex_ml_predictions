"""
Detailed model inspection to understand the current model state
"""
import joblib
import numpy as np
from datetime import datetime

print("="*80)
print("DETAILED MODEL INSPECTION")
print("="*80)

model_path = 'data/best_forex_model.joblib'
print(f"\nLoading: {model_path}")
print("-" * 80)

try:
    model_data = joblib.load(model_path)
    
    print("\n[1] MODEL COMPONENTS:")
    print("-" * 80)
    for key in model_data.keys():
        print(f"  - {key}: {type(model_data[key]).__name__}")
    
    print("\n[2] LABEL ENCODER (Target Classes):")
    print("-" * 80)
    if 'label_encoder' in model_data:
        encoder = model_data['label_encoder']
        classes = encoder.classes_
        print(f"  Number of classes: {len(classes)}")
        print(f"  Classes: {classes}")
        print(f"  Class mapping:")
        for i, cls in enumerate(classes):
            print(f"    {i} -> {cls}")
    else:
        print("  ❌ No label_encoder found!")
    
    print("\n[3] MODEL TYPE:")
    print("-" * 80)
    if 'model' in model_data:
        model = model_data['model']
        print(f"  Model type: {type(model).__name__}")
        print(f"  Model: {model}")
        
        # Check if it's an ensemble
        if hasattr(model, 'estimators_'):
            print(f"\n  Ensemble with {len(model.estimators_)} estimators:")
            for i, est in enumerate(model.estimators_):
                print(f"    {i}: {type(est).__name__}")
        
        # Check n_classes
        if hasattr(model, 'n_classes_'):
            print(f"\n  Model n_classes_: {model.n_classes_}")
        if hasattr(model, 'classes_'):
            print(f"  Model classes_: {model.classes_}")
    
    print("\n[4] FEATURE INFORMATION:")
    print("-" * 80)
    if 'feature_columns' in model_data:
        features = model_data['feature_columns']
        print(f"  Number of features: {len(features)}")
        print(f"  Feature columns: {features[:5]}... (showing first 5)")
    
    print("\n[5] MODEL METADATA:")
    print("-" * 80)
    if 'model_version' in model_data:
        print(f"  Model version: {model_data['model_version']}")
    if 'trained_date' in model_data:
        print(f"  Trained date: {model_data['trained_date']}")
    if 'improvements' in model_data:
        print(f"  Improvements: {model_data['improvements']}")
    
    print("\n[6] TEST PREDICTION:")
    print("-" * 80)
    # Create dummy feature data
    n_features = len(model_data['feature_columns'])
    dummy_data = np.random.rand(1, n_features)
    
    # Scale if scaler exists
    if 'scaler' in model_data:
        dummy_data = model_data['scaler'].transform(dummy_data)
    
    # Predict
    model = model_data['model']
    prediction = model.predict(dummy_data)
    prediction_proba = model.predict_proba(dummy_data)
    
    print(f"  Predicted class index: {prediction[0]}")
    print(f"  Predicted probabilities shape: {prediction_proba.shape}")
    print(f"  Predicted probabilities: {prediction_proba[0]}")
    
    # Decode prediction
    if 'label_encoder' in model_data:
        predicted_label = model_data['label_encoder'].inverse_transform(prediction)
        print(f"  Decoded prediction: {predicted_label[0]}")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    
    if 'label_encoder' in model_data:
        n_classes = len(model_data['label_encoder'].classes_)
        if n_classes == 2:
            print("⚠️  THIS IS A 2-CLASS (BINARY) MODEL")
            print("    Classes:", model_data['label_encoder'].classes_)
            print("    This explains the SELL bias!")
        elif n_classes == 3:
            print("✅ THIS IS A 3-CLASS MODEL")
            print("    Classes:", model_data['label_encoder'].classes_)
            print("    Model should produce balanced predictions")
        else:
            print(f"⚠️  UNEXPECTED: {n_classes} classes")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
