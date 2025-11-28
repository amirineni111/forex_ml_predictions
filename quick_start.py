"""
Quick Start Script for Forex ML Models

This script helps users get started quickly with forex prediction models.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

def safe_print(text):
    """Print text with safe encoding handling."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe_text = text.encode('ascii', errors='ignore').decode('ascii')
        print(safe_text)


def check_requirements():
    """Check if required packages are installed."""
    safe_print("🔍 Checking requirements...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'pyodbc', 
        'sqlalchemy', 'dotenv', 'joblib', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        safe_print(f"❌ Missing packages: {', '.join(missing_packages)}")
        safe_print("📦 Please install missing packages:")
        safe_print("   pip install -r requirements.txt")
        return False
    else:
        safe_print("✅ All required packages are installed")
        return True


def check_environment():
    """Check if environment is properly configured."""
    safe_print("🔍 Checking environment configuration...")
    
    env_file = Path('.env')
    
    if not env_file.exists():
        safe_print("⚠️ .env file not found")
        safe_print("📝 Please copy .env.example to .env and configure your settings")
        return False
    
    # Check for required environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['SQL_SERVER', 'SQL_DATABASE']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        safe_print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        safe_print("📝 Please configure these variables in your .env file")
        return False
    else:
        safe_print("✅ Environment configuration looks good")
        return True


def test_database_connection():
    """Test database connection."""
    safe_print("🔍 Testing database connection...")
    
    try:
        from database.connection import ForexSQLServerConnection
        
        db = ForexSQLServerConnection()
        
        if db.test_connection():
            safe_print("✅ Database connection successful")
            
            # Check for forex data
            pairs = db.get_forex_pairs()
            if pairs:
                safe_print(f"📊 Found {len(pairs)} forex pairs")
                safe_print(f"   Sample pairs: {', '.join(pairs[:5])}")
            else:
                safe_print("⚠️ No forex pairs found in database")
                
            return True
        else:
            safe_print("❌ Database connection failed")
            return False
            
    except Exception as e:
        safe_print(f"❌ Database connection error: {e}")
        return False


def create_sample_prediction():
    """Create a sample prediction to verify everything works."""
    safe_print("🔍 Creating sample prediction...")
    
    try:
        from predict_forex_signals import ForexTradingSignalPredictor
        
        # Initialize predictor
        predictor = ForexTradingSignalPredictor()
        
        # Get available forex pairs
        pairs = predictor.db.get_forex_pairs()
        
        if not pairs:
            safe_print("❌ No forex pairs available for testing")
            return False
        
        # Use first available pair for testing
        test_pair = pairs[0]
        safe_print(f"📊 Testing with currency pair: {test_pair}")
        
        # Get some data
        forex_data = predictor.get_forex_data(currency_pair=test_pair, days_back=30)
        
        if forex_data.empty:
            safe_print("❌ No forex data available for testing")
            return False
        
        safe_print(f"📈 Retrieved {len(forex_data)} records for testing")
        
        # Train a simple model
        safe_print("🔧 Training a test model...")
        success = predictor.train_model(forex_data, signal_type='trend')
        
        if success:
            safe_print("✅ Test model trained successfully")
            
            # Generate sample predictions
            predictions = predictor.predict_signals(df=forex_data, currency_pair=test_pair)
            
            if not predictions.empty:
                safe_print(f"📈 Generated {len(predictions)} test predictions")
                safe_print("✅ Sample prediction completed successfully!")
                
                # Show sample of predictions
                sample_cols = ['date_time', 'currency_pair', 'close_price', 'signal']
                available_cols = [col for col in sample_cols if col in predictions.columns]
                sample_predictions = predictions[available_cols].tail(5)
                
                safe_print("\n📋 Sample Predictions:")
                safe_print("-" * 40)
                safe_print(sample_predictions.to_string(index=False))
                
                return True
            else:
                safe_print("❌ No predictions generated")
                return False
        else:
            safe_print("❌ Test model training failed")
            return False
            
    except Exception as e:
        safe_print(f"❌ Error creating sample prediction: {e}")
        return False


def setup_directories():
    """Create necessary directories."""
    safe_print("📁 Setting up directories...")
    
    directories = [
        './data',
        './results', 
        './reports',
        './daily_reports',
        './logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    safe_print("✅ Directories created")


def main():
    """Main quick start function."""
    safe_print("💱 Forex ML Models Quick Start")
    safe_print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Check requirements
    if not check_requirements():
        safe_print("\n❌ Setup incomplete: Missing required packages")
        return
    
    # Check environment
    if not check_environment():
        safe_print("\n❌ Setup incomplete: Environment not configured")
        return
    
    # Test database connection
    if not test_database_connection():
        safe_print("\n❌ Setup incomplete: Database connection failed")
        return
    
    # Create sample prediction
    if not create_sample_prediction():
        safe_print("\n⚠️ Setup mostly complete, but sample prediction failed")
        safe_print("   You may need to check your forex data or adjust the configuration")
    else:
        safe_print("\n✅ Setup completed successfully!")
    
    # Show next steps
    safe_print("\n📋 Next Steps:")
    safe_print("-" * 20)
    safe_print("1. Run daily predictions:")
    safe_print("   python predict_forex_signals.py --currency-pair EURUSD --export")
    safe_print("")
    safe_print("2. Train models for specific pairs:")
    safe_print("   python predict_forex_signals.py --currency-pair GBPUSD --train-new")
    safe_print("")
    safe_print("3. Set up daily automation:")
    safe_print("   python daily_forex_automation.py --run-now")
    safe_print("")
    safe_print("4. Check available forex pairs:")
    safe_print("   python -c \"from src.database.connection import ForexSQLServerConnection; db=ForexSQLServerConnection(); print(db.get_forex_pairs())\"")


if __name__ == "__main__":
    main()