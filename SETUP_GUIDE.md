# Forex ML Setup Guide

## ✅ Status: Project Structure Complete!

Your forex ML models project is now set up and ready to use. The old NSE and ML model folders have been cleaned up.

## 🔧 Next Steps to Configure Database Connection:

### 1. Configure Your SQL Server Connection

Edit the `.env` file with your actual SQL Server details:

```bash
# Update these values in .env file:
SQL_SERVER=your-actual-server-name-or-ip
SQL_DATABASE=your-forex-database-name
SQL_USERNAME=your-sql-username
SQL_PASSWORD=your-sql-password

# For Windows Authentication instead:
# SQL_TRUSTED_CONNECTION=yes
# (and remove/comment out username/password)
```

### 2. Verify Database Schema

Make sure your SQL Server database contains forex data with these tables/views:

**Required Tables:**
- `forex_data` (or update `FOREX_BASE_TABLE` in .env)
- `forex_indicators_view` (or update `FOREX_INDICATORS_VIEW` in .env)

**Expected Columns in forex_indicators_view:**
```sql
-- Price Data
currency_pair, date_time, open_price, high_price, low_price, close_price, volume

-- Moving Averages  
sma_5, sma_10, sma_20, sma_50, sma_200
ema_5, ema_10, ema_20, ema_50, ema_200

-- Technical Indicators (your BB, EMA, SMA, RSI, MACD, ATR data)
rsi_14 (or rsi), macd_line (or macd), macd_signal, macd_histogram, atr_14 (or atr)

-- Bollinger Bands
bb_upper, bb_middle, bb_lower, bb_width, bb_percent

-- Derived Features (optional, will be calculated if missing)
daily_return, volatility, price_range, gap, volume_ratio
```

### 3. Test Your Connection

```bash
python test_connection.py
```

This will show you:
- Your current configuration
- Database connection status  
- Available forex tables
- Currency pairs in your database

### 4. Once Connection Works, Run Quick Start

```bash
python quick_start.py
```

This will:
- Verify all packages are installed
- Test database connection
- Create a sample prediction
- Show you how to use the system

### 5. Generate Your First Forex Signals

```bash
# For a specific currency pair
python predict_forex_signals.py --currency-pair EURUSD --export

# Train a new model first
python predict_forex_signals.py --currency-pair EURUSD --train-new --export

# See all available options
python predict_forex_signals.py --help
```

## 🚀 Ready to Use Features:

1. **Forex Signal Prediction** - `predict_forex_signals.py`
2. **Daily Automation** - `daily_forex_automation.py` 
3. **Connection Testing** - `test_connection.py`
4. **Setup Verification** - `quick_start.py`

## 📊 What's Different from NSE/Nasdaq Models:

- ✅ Specialized forex database connections
- ✅ Forex-specific technical indicators (BB, EMA, SMA, RSI, MACD, ATR)
- ✅ Currency pair-based modeling
- ✅ Forex market time series features
- ✅ Clean project structure without old cloned repos

## 🆘 Need Help?

1. **Connection Issues**: Run `python test_connection.py` for diagnostics
2. **Missing Tables**: Update table names in `.env` file  
3. **Data Issues**: Check that your forex data contains required indicators
4. **Package Issues**: Run `pip install -r requirements.txt`

The project is ready to go - just configure your database connection and start predicting forex signals! 🚀