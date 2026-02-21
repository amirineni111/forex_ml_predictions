# CLAUDE.md — sqlserver_copilot_forex (Forex ML Training Pipeline)

> **Project context file for AI assistants (Claude, Copilot, Cursor).**

---

## 1. SYSTEM OVERVIEW

This is the **Forex ML training pipeline** — one of **7 interconnected repositories** that form an AI-powered stock trading analytics platform. All repos share a single SQL Server database (`stockdata_db`).

### Repository Map

| Layer | Repo | Purpose |
|-------|------|---------|
| Data Ingestion | `stockanalysis` | ETL: yfinance/Alpha Vantage → SQL Server |
| SQL Infrastructure | `sqlserver_mcp` | .NET 8 MCP Server (Microsoft MssqlMcp) — 7 tools (ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData) via stdio transport for AI IDE ↔ SQL Server |
| Dashboard | `streamlit-trading-dashboard` | 40+ views, signal tracking, Streamlit UI |
| ML: NASDAQ | `sqlserver_copilot` | Gradient Boosting → `ml_trading_predictions` |
| ML: NSE | `sqlserver_copilot_nse` | 5-model ensemble → `ml_nse_trading_predictions` |
| **ML: Forex** ⭐ | **`sqlserver_copilot_forex`** | **THIS REPO** — XGBoost/LightGBM/Stacking → `forex_ml_predictions` |
| Agentic AI | `stockdata_agenticai` | 7 CrewAI agents, daily briefing email |

---

## 2. THIS REPO: sqlserver_copilot_forex

### Purpose
Trains a **multi-model ensemble** (XGBoost, LightGBM, VotingClassifier, StackingClassifier) on 10 forex currency pairs to predict Buy/Sell/Hold signals, then writes predictions to `forex_ml_predictions`.

### Daily Schedule (Windows Task Scheduler)
```
07:00 AM  Daily prediction run     → forex_ml_predictions
Sunday 6 AM  Weekly full retrain   → Updated model files
```

### Key Files

```
sqlserver_copilot_forex/
├── src/
│   ├── predict_daily.py           # Daily prediction entry point
│   ├── train_model.py             # Full training pipeline
│   ├── feature_engineering.py     # 100+ feature calculations
│   ├── sql_queries.py             # SQL queries for data retrieval
│   ├── model_utils.py             # Model save/load utilities
│   └── ensemble_builder.py        # XGBoost/LightGBM/Stacking ensemble
├── models/
│   ├── forex_xgb_model.pkl        # XGBoost model
│   ├── forex_lgbm_model.pkl       # LightGBM model
│   ├── forex_voting_model.pkl     # VotingClassifier
│   ├── forex_stacking_model.pkl   # StackingClassifier (meta-learner)
│   └── feature_columns.pkl        # Selected feature names
├── config/
│   └── settings.py                # .env configuration loader
├── logs/
│   └── *.log                      # Execution logs
└── notebooks/
    └── exploratory_analysis.ipynb # EDA notebooks
```

---

## 3. ML MODEL DETAILS

### Model Architecture
- **Models**: XGBoost, LightGBM, VotingClassifier, StackingClassifier
- **Target**: Buy/Sell/Hold signal (3-class classification)
- **Features**: 100+ engineered features from forex OHLCV + economic indicators
- **Training Data**: `forex_hist_data` for 10 currency pairs
- **Best Performer**: Stacking model (uses XGBoost + LightGBM as base learners)

### Feature Categories (100+)
| Category | Examples |
|----------|---------|
| Price-based | Returns (1d/5d/10d/20d), pip changes, range ratios |
| Moving Averages | SMA (5/10/20/50/200), EMA (12/26), MA crossovers |
| Momentum | RSI (14), MACD, Stochastic, ROC, CCI, Williams %R, MFI |
| Volatility | Bollinger Bands, ATR, Keltner Channels, historical volatility |
| Volume | Volume ratios, on-balance volume proxies |
| Forex-specific | Pip volatility, session overlaps, carry trade proxies |
| Lag features | Lagged returns, lagged indicators (1-5 periods) |
| **Market context** | VIX, DXY, S&P 500, US 10Y yield — now reads from shared `market_context_daily` DB table via `ExternalDataSources(use_db=True)`, with yfinance fallback |

### Output Table: `forex_ml_predictions`
| Column | Type | Description |
|--------|------|-------------|
| currency_pair | VARCHAR | e.g., 'USD/INR', 'EUR/USD' |
| trading_date | DATE | Prediction date |
| predicted_signal | VARCHAR | 'Buy', 'Sell', or 'Hold' |
| signal_confidence | FLOAT | Model confidence (0-100) |
| prob_buy | FLOAT | P(Buy) from model |
| prob_sell | FLOAT | P(Sell) from model |
| prob_hold | FLOAT | P(Hold) from model |
| model_name | VARCHAR | Which model produced prediction |
| model_version | VARCHAR | Model version identifier |

### Currency Pairs (10)
USD/INR, EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY

---

## 4. DATABASE CONTEXT

### Shared SQL Server
- **Server**: `192.168.86.55\MSSQLSERVER01` (Machine A LAN IP)
- **Database**: `stockdata_db`
- **Auth**: SQL Auth (`remote_user`, `SQL_TRUSTED_CONNECTION=no`)

### Tables This Repo READS
| Table | Purpose |
|-------|---------|
| `forex_hist_data` | Historical OHLCV + daily changes + moving averages |
| `forex_master` | Currency pair master list (10 active pairs) |

### Tables This Repo WRITES
| Table | Purpose |
|-------|---------|
| `forex_ml_predictions` | Daily Buy/Sell/Hold predictions per pair |

---

## 5. CODING CONVENTIONS

### Key Notes
- Forex data has DECIMAL columns (not VARCHAR like equity tables)
- 3-class classification (Buy/Sell/Hold) — unlike equity repos which are binary
- Multiple models trained and compared — best model selected for production
- Model versioning via model_name + model_version columns in predictions table

### Testing
```bash
python src/predict_daily.py --test
python -m pytest tests/
```

---

## 6. DOWNSTREAM CONSUMERS
- **stockdata_agenticai** — Forex Agent reads `forex_ml_predictions` for daily briefing
- **streamlit-trading-dashboard** — Displays forex predictions and trends
- Note: Forex is **excluded from Strategy 2 cross-analysis** (regression model underperformance)

---

## 7. MCP SERVER FOR DEVELOPMENT

The `sqlserver_mcp` repo provides an MCP server for AI IDEs to query `stockdata_db` directly during development.

### VS Code Configuration
```json
"MSSQL MCP": {
    "type": "stdio",
    "command": "C:\\Users\\sreea\\OneDrive\\Desktop\\sqlserver_mcp\\SQL-AI-samples\\MssqlMcp\\dotnet\\MssqlMcp\\bin\\Debug\\net8.0\\MssqlMcp.exe",
    "env": {
        "CONNECTION_STRING": "Server=192.168.86.55\\MSSQLSERVER01;Database=stockdata_db;User Id=remote_user;Password=YourStrongPassword123!;TrustServerCertificate=True"
    }
}
```

### 7 MCP Tools: ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData

Useful for: checking `forex_ml_predictions` output format, verifying `forex_hist_data` schema, exploring currency pair data and model signal distribution.
