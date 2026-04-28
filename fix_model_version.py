"""
Update model version to a shorter string that fits database column
"""
import joblib
import os

model_path = './data/best_forex_model.joblib'

if os.path.exists(model_path):
    print(f"Loading model from {model_path}...")
    artifacts = joblib.load(model_path)
    
    # Check current version
    current_version = artifacts.get('model_version', 'unknown')
    print(f"Current version: {current_version} ({len(current_version)} characters)")
    
    # Update to shorter version
    artifacts['model_version'] = '3.0'
    
    # Save back
    joblib.dump(artifacts, model_path)
    print(f"✅ Updated model_version to: 3.0 (3 characters)")
    
    # Also update enhanced model
    enhanced_path = './data/enhanced_forex_model.joblib'
    if os.path.exists(enhanced_path):
        joblib.dump(artifacts, enhanced_path)
        print(f"✅ Updated {enhanced_path} as well")
    
    print("\n✅ Model version updated successfully")
else:
    print(f"❌ Model file not found at {model_path}")
