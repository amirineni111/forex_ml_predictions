# Forex ML Predictions

A comprehensive machine learning system for forex trading signal prediction using technical indicators stored in SQL Server database.

🔗 **Repository**: https://github.com/amirineni111/forex_ml_predictions

## Features

- **Forex-specific ML models** using technical indicators (BB, EMA, SMA, RSI, MACD, ATR)
- **Real-time signal prediction** for major currency pairs
- **Daily automation** with scheduled model retraining
- **Comprehensive reporting** and performance monitoring
- **SQL Server integration** with optimized queries for forex data

## Technical Indicators Supported

- **Moving Averages**: SMA (5,10,20,50,200), EMA (5,10,20,50,200)
- **Momentum Indicators**: RSI, MACD (line, signal, histogram)
- **Volatility Indicators**: ATR, Bollinger Bands (upper, middle, lower, width, %)
- **Price Action**: Daily returns, volatility, price ranges, gaps
- **Volume Analysis**: Volume ratios and trends

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your SQL Server connection details
   ```

3. **Verify setup:**
```bash
python quick_start.py
```

## Database Results Storage

The system now includes comprehensive database storage capabilities for prediction results, model performance, and daily summaries.

### Results Tables

The following tables are automatically created in your SQL Server database:

1. **forex_ml_predictions** - Stores individual prediction records
   - Prediction metadata, signals, confidence scores
   - Currency pair, timestamps, price data
   - Model information and probabilities

2. **forex_model_performance** - Tracks model training performance
   - Cross-validation scores, accuracy metrics
   - Training parameters and sample counts
   - Historical performance tracking

3. **forex_daily_summary** - Daily aggregated statistics
   - Signal distributions by currency pair
   - Confidence statistics and model usage
   - Daily performance summaries

### Database Export Features

- **Automatic table creation** during first use
- **Prediction storage** with full metadata
- **Performance tracking** across model retraining
- **Daily summaries** for reporting and analysis
- **Data cleanup utilities** for maintenance
- **CSV export capabilities** from database4. **Generate forex signals:**
   ```bash
   python predict_forex_signals.py --currency-pair EURUSD --export
   ```

## Project Structure

```
├── src/
│   ├── database/         # Database connection modules
│   ├── models/          # ML model implementations
│   └── data/            # Data processing utilities
├── data/                # Model artifacts and processed data
├── results/             # Prediction outputs
├── reports/             # Analysis reports
├── daily_reports/       # Automated daily reports
├── logs/                # Application logs
└── notebooks/           # Jupyter notebooks for analysis
```

## Main Scripts

### predict_forex_signals.py
Main prediction script for generating forex trading signals.

**Usage:**
```bash
# Predict signals for specific currency pair
python predict_forex_signals.py --currency-pair EURUSD

# Train new model with momentum signals
python predict_forex_signals.py --train-new --signal-type momentum

# Export predictions to CSV
python predict_forex_signals.py --currency-pair GBPUSD --export

# Export predictions to SQL Server database
python predict_forex_signals.py --currency-pair EURUSD --export-db

# Setup database tables for results storage
python predict_forex_signals.py --setup-tables

# Use custom model path
python predict_forex_signals.py --model-path ./data/custom_model.joblib
```

**Arguments:**
- `--currency-pair`: Specific currency pair (e.g., EURUSD, GBPUSD)
- `--days-back`: Days of historical data (default: 60)
- `--model-path`: Path to saved model file
- `--train-new`: Train a new model
- `--export`: Export predictions to CSV
- `--export-db`: Export predictions to SQL Server database
- `--setup-tables`: Setup database tables for results storage
- `--signal-type`: Type of trading signal (trend, momentum, mean_reversion)

### daily_forex_automation.py
Automated daily processing and model management.

**Usage:**
```bash
# Run daily automation immediately
python daily_forex_automation.py --run-now

# Retrain models immediately
python daily_forex_automation.py --retrain-now

# Start scheduled automation (runs in background)
python daily_forex_automation.py
```

**Features:**
- Daily signal generation for major forex pairs
- Weekly model retraining with fresh data
- Automated report generation
- Data availability monitoring
- Old file cleanup

### manage_db_results.py
Database results management and analysis.

```bash
# Setup results tables in SQL Server
python manage_db_results.py setup

# View recent predictions
python manage_db_results.py predictions --days 7

# View predictions for specific currency pair
python manage_db_results.py predictions --currency-pair EURUSD --days 3

# View model performance history
python manage_db_results.py performance

# View performance for specific model
python manage_db_results.py performance --model-name forex_ml_model

# Export recent predictions to CSV
python manage_db_results.py export --days 10

# Cleanup old predictions (keep last 30 days)
python manage_db_results.py cleanup --days-to-keep 30
```

## Database Schema

The system expects forex data with the following structure:

### Required Tables/Views:
- **forex_data**: Base forex price data
- **forex_indicators_view**: View with technical indicators

### Expected Columns:
```sql
-- Price Data
currency_pair, date_time, open_price, high_price, low_price, close_price, volume

-- Moving Averages  
sma_5, sma_10, sma_20, sma_50, sma_200
ema_5, ema_10, ema_20, ema_50, ema_200

-- Technical Indicators
rsi_14, macd_line, macd_signal, macd_histogram, atr_14

-- Bollinger Bands
bb_upper, bb_middle, bb_lower, bb_width, bb_percent

-- Derived Features
daily_return, volatility, price_range, gap, volume_ratio
```

## Configuration

### Environment Variables (.env)
```bash
# SQL Server Settings
SQL_SERVER=your-sql-server-instance
SQL_DATABASE=your-forex-database
SQL_USERNAME=your-username
SQL_PASSWORD=your-password
SQL_DRIVER=ODBC Driver 17 for SQL Server
SQL_TRUSTED_CONNECTION=no

# Forex Data Settings
FOREX_BASE_TABLE=forex_data
FOREX_INDICATORS_VIEW=forex_indicators_view

# ML Model Settings
MODEL_SAVE_PATH=./data/models/
RESULTS_SAVE_PATH=./results/
DEFAULT_LOOKBACK_DAYS=60
```

## Machine Learning Models

The system supports multiple ML algorithms:

### Classification Models:
- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- SVM
- Naive Bayes
- K-Nearest Neighbors
- XGBoost (if installed)
- LightGBM (if installed)

### Signal Types:
- **Trend Following**: Predicts continuation of current trends
- **Momentum**: Higher threshold momentum-based signals  
- **Mean Reversion**: Counter-trend signals

### Feature Engineering:
- Price momentum calculations
- Moving average ratios and crossovers
- RSI-based conditions (oversold/overbought)
- Bollinger Band squeeze and breakout detection
- MACD signal crossovers
- Volume-price relationships
- Volatility measures
- Time-based features (hour, day of week, month)

## Output Examples

### Prediction CSV Format:
```
date_time,currency_pair,close_price,signal,prob_BUY,prob_SELL,prob_HOLD,confidence
2024-01-15 09:00:00,EURUSD,1.0985,BUY,0.75,0.15,0.10,0.75
2024-01-15 10:00:00,EURUSD,1.0992,HOLD,0.30,0.25,0.45,0.45
```

### Daily Report Structure:
- Signal distribution by currency pair
- Recent trading signals with confidence scores
- Model performance metrics
- Data availability status

## Performance Monitoring

The system includes comprehensive logging and monitoring:

- **Model Performance**: Cross-validation scores, accuracy metrics
- **Data Quality**: Missing data detection, outlier identification  
- **System Health**: Database connectivity, file system status
- **Trading Signals**: Signal distribution, confidence analysis

## Customization

### Adding New Currency Pairs:
1. Ensure forex data is available in your SQL Server database
2. Update currency pair lists in automation scripts
3. Train specific models for new pairs

### Custom Technical Indicators:
1. Add indicators to your SQL Server view
2. Update feature columns in `ForexMLModelManager`
3. Modify feature engineering in `prepare_forex_features()`

### Custom Signal Logic:
1. Modify signal generation in `create_trading_signals()`
2. Adjust thresholds for different trading strategies
3. Add new signal types as needed

## Troubleshooting

### Common Issues:

1. **Database Connection Failed**
   - Check SQL Server credentials in .env file
   - Verify ODBC driver installation
   - Ensure SQL Server allows connections

2. **No Forex Data Found**
   - Verify table/view names in configuration
   - Check data availability in your database
   - Ensure proper column naming

3. **Model Training Failed**
   - Check for sufficient historical data (minimum 1000 records)
   - Verify technical indicators are properly calculated
   - Review log files for specific error details

4. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Install TA-Lib for additional technical indicators
   - Check Python version compatibility (3.8+)

### Getting Help:

1. Check log files in `./logs/` directory
2. Run `python quick_start.py` for setup verification
3. Review database schema and data availability
4. Verify all required environment variables are set

## License

This project is designed for educational and research purposes. Ensure compliance with your organization's trading and data usage policies.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Update documentation
5. Submit pull request

---

**Note**: This system is for educational/research purposes. Always validate trading signals against market conditions and implement proper risk management before live trading.