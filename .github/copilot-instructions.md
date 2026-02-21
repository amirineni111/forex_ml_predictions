# Copilot Instructions — sqlserver_copilot_forex

## Project Context
This is the **Forex ML training pipeline** — part of a 7-repo stock trading analytics platform. Trains XGBoost/LightGBM/Stacking ensemble to predict Buy/Sell/Hold signals for 10 forex currency pairs.

## Key Architecture Rules
- Reads from `forex_hist_data` (DECIMAL prices, no casting needed)
- Writes to `forex_ml_predictions`
- Multi-model ensemble: XGBoost, LightGBM, VotingClassifier, StackingClassifier
- 3-class classification (Buy/Sell/Hold) — NOT binary like equity pipelines
- Connected to shared database `stockdata_db` on `localhost\MSSQLSERVER01` (Windows Auth)

## Pipeline Flow
1. `feature_engineering.py` — 100+ technical features from OHLCV
2. `ensemble_builder.py` — Trains 4 models (XGBoost, LightGBM, Voting, Stacking)
3. `predict_daily.py` — Daily predictions → forex_ml_predictions table

## Schedule
- Daily 7:00 AM: Forex prediction run
- Sunday 6:00 AM: Weekly full retrain

## Currency Pairs (10)
USD/INR, EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY, GBP/JPY

## Sibling Repositories (same database)
- `sqlserver_copilot` — NASDAQ ML pipeline (Gradient Boosting)
- `sqlserver_copilot_nse` — NSE ML pipeline (5-model ensemble)
- `stockdata_agenticai` — CrewAI agents (Forex Agent consumes predictions)
- `streamlit-trading-dashboard` — Visualization
- `sqlserver_mcp` — .NET MCP bridge
- `stockanalysis` — Data ingestion ETL
