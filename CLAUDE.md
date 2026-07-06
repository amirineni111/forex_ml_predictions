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

> **⚠️ History (2026-07-04):** a 12-day production audit exposed that the Sunday
> **weekly retrain silently reverted production to the 3-class model** (it called
> `train_enhanced_models(use_binary_direction=False, lookback_days=90)`),
> explaining the post-Jun-29 confidence collapse to 41–48%. Fixed by funnelling
> ALL production retrains through `EnhancedForexTrainer.train_production_model()`
> (config in `src/forex_config.py`; the 3-class fallback trainer was deleted).
> Two more latent bugs fixed at the same time: (a) the 0.75-SELL-threshold "May
> fix" only existed in `predict_forex_signals.py` while the scheduled daily run
> used raw argmax via `src/utils/forward_prediction.py` — thresholds/veto now
> live in shared `src/utils/signal_policy.py` and gate BOTH paths; (b) training
> merged `market_context_daily` features but the predict path didn't (they were
> silently zero-filled = train/serve skew) — external merges now go through
> shared `src/features/external_merge.py`, and training NaN medians are stored
> in the artifact (`feature_fill_values`) and reused at predict time.

### Daily Schedule (Windows Task Scheduler)
```
6:00 PM ET   FRED rate ingestion      → forex_rates_daily   (scripts/seed_forex_rates.py)
8:55 PM ET   Daily prediction run     → forex_ml_predictions
Sunday 10 AM Weekly full retrain      → Updated global model file
```
Register tasks via `scripts/setup_automation.ps1` (run as Administrator). The
FRED-seeded `forex_rates_daily` table is **consumed by the model since
2026-07-04**: per-pair rate/yield differentials are built by
`src/features/external_merge.py` and were accepted by the A/B gate
(`scripts/compare_rates_features.py`, WF 0.6256 vs 0.6241 baseline). Keep the
6 PM ingestion task healthy — the features degrade to stale/NaN without it.

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
> | `train_enhanced_model.py` | Training. **Production entry:** `EnhancedForexTrainer.train_production_model()` (live pair list, binary target, 400-day window — constants in `src/forex_config.py`). Never call the lower-level `prepare_enhanced_dataset`/`train_enhanced_models` with ad-hoc args for production. |
> | `src/utils/signal_policy.py` | **Single source of truth** for signal thresholds (SELL ≥ 0.75, BUY ≥ 0.55) + Pattern-B technical veto + `gate_binary_signal()`. Both prediction paths import from here. |
> | `src/utils/forward_prediction.py` | The scheduled daily path — now applies `signal_policy` gating (was raw argmax) |
> | `src/features/external_merge.py` | **The only place external features are merged** (market context + rate differentials), called by BOTH training and prediction — see §5 |
> | `src/features/advanced_features.py` | ~150 per-pair technical features. **Sorts ascending by `date_time` first** (critical — see §5). |
> | `src/forex_config.py` | Production training config (`TRAINING_LOOKBACK_DAYS`, `USE_BINARY_DIRECTION`, `INCLUDE_RATES_FEATURES`, `MODEL_VERSION`) + FRED series + table names (+ unused cluster/archetype map) |
> | `src/data/external_sources.py` | Market-context reads + `get_rates_data()` (wide per-currency frame from `forex_rates_daily`) |
> | `scripts/seed_forex_rates.py` | FRED ingestion → `forex_rates_daily` |
> | `scripts/compare_rates_features.py` | A/B gate: trains with/without rates features, saves the walk-forward winner |
> | `scripts/check_train_serve_parity.py` | Regression guard: asserts train and predict pipelines produce identical model features |
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
  the 1-day forward return, mapped to **BUY/SELL** at predict time (UP→BUY,
  DOWN→SELL). The 3-class BUY/HOLD/SELL target (`target_signal`, adaptive
  volatility thresholds) still exists in the code but is **not used in
  production** — it had no measurable edge once leakage was removed.
- **Signal gate (predict time)**: `src/utils/signal_policy.py` applies asymmetric
  thresholds — SELL fires only at `prob_sell ≥ 0.75`, BUY at `prob_buy ≥ 0.55`;
  otherwise the output is **'HOLD' = ABSTAIN** (low conviction), and a clearing
  SELL can still be demoted to HOLD by the Pattern-B technical veto. HOLD is not
  a model class: `prob_hold` stays 0.0. The `gate_reason` (threshold/abstain/veto)
  is logged in the run summary.
- **Freshness gate**: the daily run skips (with `[ERROR]` + summary entry) any
  pair whose `MAX(trading_date)` in `forex_hist_data` is >1 business day old —
  stale prices produced the silent bad-signal episodes of Jun 22–24 and the
  USDINR May-14 predictions.
- **Features**: ~150 engineered per-pair technicals (`create_advanced_features`)
  + market-context + `rate_*` differentials, with ~30 selected for the model
  after variance/missingness filtering + multi-method selection. Cross-pair
  "relative" features were removed in the rollback.
- **Measured performance (honest, leakage-free, 2026-07-06 retrain — first with
  the 5 new pairs):** walk-forward 0.655 / test 0.644 / CV 0.626 (vs 0.50
  coin-flip), best model `xgboost`, stability PASSED, overfit gap ~0.14.
  (Previous 2026-07-04 retrain: WF 0.626 / test 0.612, `voting_soft`.)
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
| **Rate / yield differentials** | **Consumed since 2026-07-04** (gated by `INCLUDE_RATES_FEATURES`). 8 `rate_*` candidates built per pair from `forex_rates_daily` (base−quote policy/2Y/10Y diffs, 5d/20d changes, USD policy level); `rate_yield_10y_diff_chg_5d` and `_chg_20d` survived feature selection. HKD/SGD/INR have no FRED series → their diff columns stay NaN (median-filled consistently on both paths). |

### Output Table: `forex_ml_predictions`
| Column | Type | Description |
|--------|------|-------------|
| currency_pair | VARCHAR | e.g., 'USD/INR', 'EUR/USD' |
| trading_date | DATE | Prediction date |
| predicted_signal | VARCHAR | **'BUY', 'SELL', or 'HOLD'** — HOLD means the gate ABSTAINED (below threshold / vetoed), not a predicted class |
| signal_confidence | FLOAT | Max class probability (kept even when abstaining, so a 0.60 abstain is distinguishable from a 0.51 one) |
| prob_buy | FLOAT | P(BUY) = P(UP) from model |
| prob_sell | FLOAT | P(SELL) = P(DOWN) from model |
| prob_hold | FLOAT | Always 0.0 under the binary model (even for HOLD/abstain rows) |
| model_name | VARCHAR | `daily_automation_model` (daily run) |
| model_version | VARCHAR | `5.2_binary_rates` (current; daily rows carry a `+gated` suffix when the artifact predates the gate) |

### Currency Pairs (actual, from `forex_hist_data`)
> Pair discovery is live (`get_forex_pairs()` = `SELECT DISTINCT symbol FROM
> forex_hist_data`) on both the training and daily-prediction paths, so pairs
> added to the DB are picked up automatically. On **2026-07-06** five pairs were
> added with ~1 year of history (AUD/NZD, EUR/GBP, GBP/JPY, USD/CAD, USD/CHF)
> and the model was retrained the same day (WF 0.655, best model `xgboost`,
> stability PASSED).

**15 active pairs:** EUR/USD, GBP/USD, AUD/USD, NZD/USD, USD/JPY, EUR/JPY,
EUR/CHF, USD/HKD, USD/SGD, USD/INR, AUD/NZD, EUR/GBP, GBP/JPY, USD/CAD,
USD/CHF. All train into the single global model.

> The cluster grouping below is **no longer used for modeling** (per-cluster
> models were rolled back). It remains in `src/forex_config.py` only as reference /
> for possible future use:
>
> | Cluster | Pairs |
> |---------|-------|
> | `usd_majors` | EUR/USD, GBP/USD |
> | `commodity` | AUD/USD, NZD/USD, USD/CAD, AUD/NZD |
> | `jpy_crosses` | USD/JPY, EUR/JPY, GBP/JPY |
> | `eur_crosses` | EUR/CHF, EUR/GBP, USD/CHF |
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
| `forex_rates_daily` ⭐ | Per-currency policy rate + 2Y/10Y yields (FRED). Seeded by `scripts/seed_forex_rates.py`; **read by the model** via `ExternalDataSources.get_rates_data()` since 2026-07-04. |
| `forex_intermarket_daily` | Gold/oil/copper, commodity & EM-FX indices, risk-on (optional; yfinance fallback) |
| `forex_econ_events` | Central-bank/CPI/NFP/GDP event flags per currency (optional) |

> `forex_rates_daily` / `forex_intermarket_daily` / `forex_econ_events` are read by
> `ExternalDataSources` and **skip gracefully when absent**. Only `forex_rates_daily`
> has an ingestion script in this repo so far.

### Tables This Repo WRITES
| Table | Purpose |
|-------|---------|
| `forex_ml_predictions` | Daily **BUY/SELL/HOLD-abstain** predictions per pair (`model_name` = `daily_automation_model`, `model_version` = `5.2_binary_rates`) |
| `forex_model_performance` | Per-model training metrics |

---

## 5. CODING CONVENTIONS

### Key Notes
- Forex data has DECIMAL columns (not VARCHAR like equity tables)
- **Binary classification (BUY/SELL)** in production, with a HOLD/abstain gate — like the equity repos
- Multiple candidate models trained and compared — best selected for production
- Model versioning via model_name + model_version columns in predictions table
- **Production retrains go ONLY through `train_production_model()`** — passing
  ad-hoc args to the lower-level training APIs is how the 2026-06-28 3-class
  regression reached production
- **External features (market context, rates) are merged ONLY via
  `src/features/external_merge.py::add_external_features`**, which both the
  training and predict paths call. Adding a merge to one path alone recreates
  train/serve skew (this repo has been bitten twice)
- **NaN fill parity**: training median-fills features and stores the medians in
  the artifact (`feature_fill_values`); the predict paths fill with those same
  medians (0 only as last resort for old artifacts). Never `fillna(0)` a new
  feature on the predict side only
- **Per-pair freshness gate**: pairs with `forex_hist_data` older than 1 business
  day are skipped with `[ERROR]` + a run-summary entry (never predicted on)

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

### Testing / Verification
There is no pytest suite; use the targeted smoke checks:
```bash
python scripts/check_train_serve_parity.py EURUSD   # train vs serve feature parity (run after ANY feature change)
python train_enhanced_model.py                      # production retrain (expect WF ~0.55-0.65; >0.70 = leakage)
python daily_forex_automation.py --run-now          # full daily run (freshness gate + signal gate + export)
python scripts/compare_rates_features.py            # A/B before enabling/disabling rates features
```

---

## 6. DOWNSTREAM CONSUMERS
- **stockdata_agenticai** — Forex Agent reads `forex_ml_predictions` for daily briefing
- **streamlit-trading-dashboard** — Displays forex predictions and trends
- ⚠️ Signals are **BUY/SELL/HOLD**, where HOLD = the gate **abstained** (low
  conviction or technical veto), not a model prediction; `prob_hold` is always
  0.0. Consumers should treat HOLD as "no actionable signal today".
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
