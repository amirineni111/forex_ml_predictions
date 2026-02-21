# AGENTS.md — sqlserver_copilot_forex (Forex ML Pipeline)

## Overview
This repo does NOT contain CrewAI agents. It is a **multi-model ML training pipeline** for Forex currency pair predictions.

## ML Pipeline Architecture

```
[forex_hist_data] (SQL Server, 10 currency pairs)
        │
        ▼
  feature_engineering.py  (100+ features)
        │
        ▼
  ensemble_builder.py  (XGBoost, LightGBM, Voting, Stacking)
        │
        ▼
  models/*.pkl  (4 serialized models)
        │
        ▼
  predict_daily.py  (daily predictions → SQL Server)
        │
        ▼
  [forex_ml_predictions]
```

## Model Ensemble
| Model | Type | Role |
|-------|------|------|
| XGBoost | Base learner | Fast gradient boosting |
| LightGBM | Base learner | Light gradient boosting |
| VotingClassifier | Ensemble | Combines base learners |
| StackingClassifier | Meta-ensemble | Uses XGB+LGBM as base, LogReg as meta |

## Key Differences from Equity Pipelines
- **3-class** classification (Buy/Sell/Hold) vs binary (Buy/Sell)
- **DECIMAL** price columns (vs VARCHAR in equity tables)
- **100+** features (vs 50+ for NASDAQ)
- **Not included** in Strategy 2 cross-analysis

## Downstream Consumers
- **stockdata_agenticai** — Forex Agent reads predictions
- **streamlit-trading-dashboard** — Displays forex signals
