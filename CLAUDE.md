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
Trains a **single global model** (best of XGBoost / LightGBM / VotingClassifier /
StackingClassifier, selected by walk-forward accuracy + stability) on 10 forex
currency pairs to predict **binary direction (UP→Buy / DOWN→Sell)**, then writes
predictions to `forex_ml_predictions`.

> **⚠️ History (2026-06-25):** an earlier per-cluster + cross-pair "relative
> features" design (commit `d1b0986`) was **rolled back** — it introduced
> train/serve skew that dropped backtest accuracy to ~25%. In the process a
> long-standing **look-ahead leakage** bug was also found and fixed (see §5): the
> DB returns rows newest-first and rolling/shift features were computed without
> sorting ascending, which inflated reported accuracy to ~80–89%. With leakage
> removed, the 3-class BUY/HOLD/SELL target showed **no edge** (~41.5% WF vs 41%
> baseline), so production switched to the **binary** target, which has a real
> out-of-sample edge (~59–62% WF vs 50% coin-flip). The previous tag
> `forex-relative-features-d1b0986` preserves the rolled-back work.

### Daily Schedule (Windows Task Scheduler)
```
6:00 PM ET   FRED rate ingestion      → forex_rates_daily   (scripts/seed_forex_rates.py)
8:55 PM ET   Daily prediction run     → forex_ml_predictions
Sunday 10 AM Weekly full retrain      → Updated global model file
```
Register tasks via `scripts/setup_automation.ps1` (run as Administrator). The FRED
rate-ingestion infra is **retained** (additive); its rate/yield-differential
features are not currently consumed by the model but the table stays seeded for
future work.

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
├── logs/
│   └── *.log                      # Execution logs
└── notebooks/
    └── exploratory_analysis.ipynb # EDA notebooks
```

> **Note:** the layout above is the original design sketch. The actual entry
> points are `train_enhanced_model.py` (training), `predict_forex_signals.py` and
> `daily_forex_automation.py --run-now` (prediction; the scheduled task uses the
> latter), and `run_all_forex_predictions.py` (dev batch). Config is `.env` +
> `src/forex_config.py` (no `config/` package). Key modules:
>
> | File | Purpose |
> |------|---------|
> | `train_enhanced_model.py` | Trains the global binary model on a 400-day window |
> | `src/features/advanced_features.py` | ~150 per-pair technical features. **Sorts ascending by `date_time` first** (critical — see §5). |
> | `src/forex_config.py` | FRED series + table names (also retains an unused cluster/archetype map for possible future use) |
> | `src/data/external_sources.py` | Market-context + (optional) rates/intermarket/event reads |
> | `scripts/seed_forex_rates.py` | FRED ingestion → `forex_rates_daily` (retained, additive) |
> | `data/best_forex_model.joblib` | The single production model artifact (git-ignored — retrain to regenerate) |
>
> Removed in the 2026-06-25 rollback: `src/features/relative_features.py` and the
> per-cluster `data/forex_model_<cluster>.joblib` artifacts.

---

## 3. ML MODEL DETAILS

### Model Architecture
- **Single global model**: one model trained across all pairs (the per-cluster
  design was rolled back). Candidates XGBoost / LightGBM / VotingClassifier /
  StackingClassifier; the best is selected by walk-forward accuracy + stability
  checks and saved to `data/best_forex_model.joblib`.
- **Target**: **binary direction** — `target_direction` UP/DOWN from the sign of
  the 1-day forward return, mapped to **Buy/Sell** at predict time (UP→Buy,
  DOWN→Sell). **No Hold class.** The 3-class BUY/HOLD/SELL target (`target_signal`,
  adaptive volatility thresholds) still exists in the code but is **not used in
  production** — it had no measurable edge once leakage was removed.
- **Features**: ~150 engineered per-pair technicals (`create_advanced_features`),
  with ~30 selected for the model after variance/missingness filtering. Cross-pair
  "relative" features were removed in the rollback.
- **Measured performance (honest, leakage-free):** walk-forward / CV / test all
  ~0.59–0.61 (vs 0.50 coin-flip). Known caveat: train/test overfitting gap ~0.20;
  generalization is fine (the three estimates agree) but worth tuning down.
- **Training Data**: `forex_hist_data` for all active pairs, 400-day window.

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
| **Market context** | VIX, DXY, S&P 500, US 10Y yield — reads from shared `market_context_daily` DB table via `ExternalDataSources(use_db=True)`, with yfinance fallback |
| ~~Relative / cross-pair~~ | **Removed in the 2026-06-25 rollback** (caused train/serve skew). Code lives only under tag `forex-relative-features-d1b0986`. |
| ~~Rate / yield differentials~~ | **Not consumed by the current model.** `forex_rates_daily` is still seeded (FRED) for future use; the model does not read it. |

### Output Table: `forex_ml_predictions`
| Column | Type | Description |
|--------|------|-------------|
| currency_pair | VARCHAR | e.g., 'USD/INR', 'EUR/USD' |
| trading_date | DATE | Prediction date |
| predicted_signal | VARCHAR | **'Buy' or 'Sell'** (binary; Hold no longer produced) |
| signal_confidence | FLOAT | Model confidence (0-100) |
| prob_buy | FLOAT | P(Buy) = P(UP) from model |
| prob_sell | FLOAT | P(Sell) = P(DOWN) from model |
| prob_hold | FLOAT | Always 0.0 under the binary model |
| model_name | VARCHAR | `daily_automation_model` (daily run) |
| model_version | VARCHAR | `5.0_binary_noleak` (current) |

### Currency Pairs (actual, from `forex_hist_data`)
> ⚠️ The pairs actually present in the DB differ from earlier docs. There is **no**
> USD/CHF, USD/CAD, EUR/GBP, or GBP/JPY; instead EUR/CHF and the Asian pairs
> USD/HKD, USD/SGD, USD/INR are present. `train_enhanced_model.py` fetches the live
> list via `get_forex_pairs()`.

**10 active pairs:** EUR/USD, GBP/USD, AUD/USD, NZD/USD, USD/JPY, EUR/JPY,
EUR/CHF, USD/HKD, USD/SGD, USD/INR. All train into the single global model.

> The cluster grouping below is **no longer used for modeling** (per-cluster
> models were rolled back). It remains in `src/forex_config.py` only as reference /
> for possible future use:
>
> | Cluster | Pairs |
> |---------|-------|
> | `usd_majors` | EUR/USD, GBP/USD |
> | `commodity` | AUD/USD, NZD/USD |
> | `jpy_crosses` | USD/JPY, EUR/JPY |
> | `eur_crosses` | EUR/CHF |
> | `usd_asia` | USD/HKD, USD/SGD, USD/INR |

---

## 4. DATABASE CONTEXT

### Shared SQL Server
- **Server**: `192.168.86.28,1444` (Machine A LAN IP)
- **Database**: `stockdata_db`
- **Auth**: SQL Auth (`remote_user`, `SQL_TRUSTED_CONNECTION=no`)

### Tables This Repo READS
| Table | Purpose |
|-------|---------|
| `forex_hist_data` | Historical OHLCV + daily changes + moving averages |
| `forex_master` | Currency pair master list |
| `market_context_daily` | VIX/DXY/S&P/NASDAQ/US10Y market context |
| `forex_rates_daily` ⭐ | Per-currency policy rate + 2Y/10Y yields (FRED). Seeded by `scripts/seed_forex_rates.py`. |
| `forex_intermarket_daily` | Gold/oil/copper, commodity & EM-FX indices, risk-on (optional; yfinance fallback) |
| `forex_econ_events` | Central-bank/CPI/NFP/GDP event flags per currency (optional) |

> `forex_rates_daily` / `forex_intermarket_daily` / `forex_econ_events` are read by
> `ExternalDataSources` and **skip gracefully when absent**. Only `forex_rates_daily`
> has an ingestion script in this repo so far.

### Tables This Repo WRITES
| Table | Purpose |
|-------|---------|
| `forex_ml_predictions` | Daily **Buy/Sell** predictions per pair (`model_name` = `daily_automation_model`, `model_version` = `5.0_binary_noleak`) |
| `forex_model_performance` | Per-model training metrics |

---

## 5. CODING CONVENTIONS

### Key Notes
- Forex data has DECIMAL columns (not VARCHAR like equity tables)
- **Binary classification (Buy/Sell)** in production — like the equity repos
- Multiple candidate models trained and compared — best selected for production
- Model versioning via model_name + model_version columns in predictions table

### ⚠️ Critical: row order before feature engineering (look-ahead leakage)
`ForexSQLServerConnection.get_forex_data_with_indicators` returns rows
**newest-first** (`ORDER BY trading_date DESC`). Any `rolling()` / `shift()`
feature math **must** run on date-ascending data, otherwise each row's window
peeks into its own future and the `shift(-1)` target is time-reversed — this
silently inflated accuracy to ~80–89% and anchored predictions ~20 days stale.
`create_advanced_features` now sorts ascending first
(`df.sort_values('date_time')`); **do not remove that sort.** When validating,
trust walk-forward / CV — a suspiciously high (>70%) daily-FX accuracy is the
leakage signature.

### Testing
```bash
python src/predict_daily.py --test
python -m pytest tests/
```

---

## 6. DOWNSTREAM CONSUMERS
- **stockdata_agenticai** — Forex Agent reads `forex_ml_predictions` for daily briefing
- **streamlit-trading-dashboard** — Displays forex predictions and trends
- ⚠️ Signals are now **Buy/Sell only** (binary). Consumers that branched on `Hold`
  will simply never see it; `prob_hold` is always 0.0.
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
        "CONNECTION_STRING": "Server=192.168.86.28,1444;Database=stockdata_db;User Id=remote_user;Password=YourStrongPassword123!;TrustServerCertificate=True"
    }
}
```

### 7 MCP Tools: ListTables, DescribeTable, ReadData, CreateTable, DropTable, InsertData, UpdateData

Useful for: checking `forex_ml_predictions` output format, verifying `forex_hist_data` schema, exploring currency pair data and model signal distribution.
