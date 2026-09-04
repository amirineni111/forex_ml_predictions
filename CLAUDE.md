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
> | `src/utils/signal_policy.py` | **Single source of truth** for signal thresholds (SELL ≥ 0.62, BUY ≥ 0.55; env-overridable via `FOREX_SELL_THRESHOLD`/`FOREX_BUY_THRESHOLD`) + Pattern-B technical veto + `gate_binary_signal()`. Both prediction paths import from here. |
> | `src/utils/forward_prediction.py` | The scheduled daily path — now applies `signal_policy` gating (was raw argmax) |
> | `src/features/external_merge.py` | **The only place external features are merged** (market context + rate differentials), called by BOTH training and prediction — see §5 |
> | `src/features/advanced_features.py` | ~150 per-pair technical features. **Sorts ascending by `date_time` first** (critical — see §5). |
> | `src/forex_config.py` | Production training config (`TRAINING_LOOKBACK_DAYS`, `USE_BINARY_DIRECTION`, `INCLUDE_RATES_FEATURES`, `MODEL_VERSION`) + FRED series + table names (+ unused cluster/archetype map) |
> | `src/data/external_sources.py` | Market-context reads + `get_rates_data()` (wide per-currency frame from `forex_rates_daily`) |
> | `scripts/seed_forex_rates.py` | FRED ingestion → `forex_rates_daily` |
> | `scripts/compare_rates_features.py` | A/B gate: trains with/without rates features, saves the walk-forward winner |
> | `scripts/check_train_serve_parity.py` | Regression guard: asserts train and predict pipelines produce identical model features |
> | `scripts/audit_live_performance.py` | Live accuracy audit from `forex_ml_predictions`: BUY edge vs always-long base rate, confidence-bucket inversion, signal mix. Enforces the §5 scoring rules |
> | `scripts/check_probability_range.py` | Post-retrain guard: probability spread of the current artifact — is SELL still reachable, did the ≥ 0.80 share jump? (§3.1e) |
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
  thresholds — SELL fires only at `prob_sell ≥ 0.62`, BUY at `prob_buy ≥ 0.55`;
  otherwise the output is **'HOLD' = ABSTAIN** (low conviction), and a clearing
  SELL can still be demoted to HOLD by the Pattern-B technical veto. HOLD is not
  a model class: `prob_hold` stays 0.0. The `gate_reason` (threshold/abstain/veto)
  is logged in the run summary. Both thresholds are env-overridable
  (`FOREX_SELL_THRESHOLD`, `FOREX_BUY_THRESHOLD`).
  > The SELL bar was **0.75 until 2026-07-26**. After the retrain compressed the
  > probability distribution (`prob_sell` tops out near 0.70) a 0.75 gate was
  > mathematically unreachable and SELL stopped firing entirely; it was lowered
  > to 0.62. SELL is still nearly dormant — **9 SELL rows out of 611** in
  > Jul 6–Sep 4. Any change here must be re-checked against the live
  > `prob_sell` range, not assumed.
- **Freshness gate**: the daily run skips (with `[ERROR]` + summary entry) any
  pair whose `MAX(trading_date)` in `forex_hist_data` is >1 business day old —
  stale prices produced the silent bad-signal episodes of Jun 22–24 and the
  USDINR May-14 predictions.
- **Features**: ~150 engineered per-pair technicals (`create_advanced_features`)
  + market-context + `rate_*` differentials, with ~30 selected for the model
  after variance/missingness filtering + multi-method selection. Cross-pair
  "relative" features were removed in the rollback.
- **Backtest performance (current artifact, 2026-09-04 retrain):** best model
  `xgboost`, walk-forward 0.6309 (std 0.0249) / test 0.6276 / CV 0.6200 /
  overfit gap 0.134, stability PASSED. Read these from the artifact
  (`walk_forward_results` / `training_results` in `data/best_forex_model.joblib`),
  not from this file — it goes stale every Sunday retrain.
  (History: 2026-08-30 WF 0.6272 `voting_soft`; 2026-07-06 WF 0.655 `xgboost`;
  2026-07-04 WF 0.626.)
  > **⚠️ Feature selection is unstable across retrains.** The 2026-09-04 retrain
  > replaced **11 of 30** selected features vs 2026-08-30, five days earlier on
  > nearly identical data — including dropping BOTH `rate_*` features that the
  > 2026-07-04 A/B gate was run to justify. Backtest scores barely moved
  > (WF 0.627 → 0.631), so the selector is choosing between near-equivalent
  > correlated features rather than finding signal. Treat WF differences under
  > ~0.02 between candidate models or retrains as noise, and do not read the
  > selected-feature list as evidence about what drives the market.
- **⚠️ LIVE performance does NOT match backtest — see §3.1 below.** Walk-forward
  0.63 has not reproduced out-of-sample: live BUY accuracy is 0.534 against a
  0.548 always-long base rate, i.e. **no measurable 1-day edge**.
- **Training Data**: `forex_hist_data` for all active pairs, 400-day window.

### 3.1 Live measured performance (audit 2026-09-04) — READ BEFORE TUNING

Scored from `forex_ml_predictions` where `model_version='5.2_binary_rates+gated'`
(571 scored rows, 2026-07-06 → 2026-09-04). These are the numbers that matter;
the walk-forward figures above have **not** reproduced live.

**(a) There is no 1-day edge. There may be a 5-day edge.**

| Horizon | BUY accuracy | always-long base rate | edge |
|---------|--------------|-----------------------|------|
| 1-day | 0.534 | 0.548 | **−1.5 pts** |
| 5-day | 0.638 | 0.531 | **+9.4 pts** |

The base rate is the trap: over this window 54.8% of pair-days closed up, so a
raw 0.534 BUY accuracy is *worse than always buying*. **Never evaluate this model
against 0.50** — always compare to the realised up-rate on the same rows.
Only AUD/NZD (+17.9 pts) and NZD/USD (+17.5 pts) beat their base rate
meaningfully at 1 day; 8 of 14 pairs are negative. Caveat: 5-day windows on daily
predictions overlap heavily, so the +9.4 is autocorrelated and its effective
sample is far below n=234 — directionally real, magnitude soft.

**(b) Confidence is ANTI-correlated with accuracy.**

| `signal_confidence` | n | 1-day accuracy |
|---------------------|---|----------------|
| 0.55–0.60 | 129 | **0.574** |
| 0.60–0.65 | 68 | 0.500 |
| 0.65–0.70 | 31 | 0.484 |
| 0.70+ | 15 | **0.400** |

Monotone decline across the whole live range. Any downstream tier/sizing logic
that favours *higher* confidence is selecting the worse half of the book.
Do not "fix" this by lowering a confidence gate to make it reachable.

**(c) The 0.80 confidence ceiling was an artifact of model choice — and it is
now gone.** Through 2026-09-04 no live v5.2 signal had ever reached 0.80 (max BUY
0.783, max SELL 0.719), which made downstream ≥ 0.80 tier gates dead by
construction. That ceiling belonged to the `voting_soft` artifact, not to the
model family: the `xgboost` artifact deployed 2026-09-04 clears 0.80 on **~20% of
firing signals**. The pre-leakage-fix v3.0/v4.0 rows (max 0.993) remain a
separate, non-comparable era.

> ⚠️ **Downstream gates that were safely dormant are now live.** Any consumer
> logic keyed to ≥ 0.80 confidence started firing with this artifact. It must
> stay disabled until §3.1(b)'s inversion is re-measured on LIVE rows from this
> artifact (`scripts/audit_live_performance.py`, ~2–4 weeks of scored rows).
> The inversion was measured on older, compressed artifacts; whether it holds for
> a better-calibrated model is an open question, and a test-split ECE of 0.021
> is not an answer to it.

**(d) ⚠️ HOLD rows are effectively auto-correct.** The upstream scoring job
(`backfill_strategy1_outcomes.py`, in the `STREAMLIT_TRADING_DASHBOARD` repo)
marks a HOLD "Correct" when the move is `< 1%` — a band calibrated for equities.
**97% of FX pair-days move less than 1%** (mean absolute move: 0.243%), so HOLD
scores 0.976 across 422 rows and the flag carries almost no information.
**Any pooled accuracy over this table is inflated** and mostly measures how often
the model abstained. Always filter `predicted_signal <> 'HOLD'` when scoring.
The proper upstream fix is an FX-scaled band (≈ 0.25%, or ATR-relative per pair),
not the equity 1%.

**(e) ⚠️ Model choice rescales confidence — but "narrower" is NOT "better".**
Candidates sit within noise of each other on walk-forward yet produce very
different probability spreads, which changes signal VOLUME and re-arms downstream
confidence gates. Measured on identical data (last 60 bars × 15 pairs):

| artifact | WF | p90 prob | fires SELL | ≥ 0.80 conf | **test ECE** |
|---|---|---|---|---|---|
| `voting_soft` (2026-08-30) | 0.627 | 0.62–0.63 | 10.1% | 6.5% | **0.0433** |
| `xgboost` (2026-09-04) | 0.635 | 0.70–0.72 | 15.2% | 20.3% | **0.0206** |

**The intuitive reading of this table is wrong.** Soft-voting averages its
members and compresses probabilities, so it *looks* conservative — but by
Expected Calibration Error it is the WORSE of the two: it says 0.58 when it is
right 64% of the time. Under-confidence is a calibration error just as
over-confidence is. Do not infer calibration from probability spread; measure it.

**What this does and does not settle:** ECE here is computed on the held-out test
split, and test-split metrics in this repo have a track record of not reproducing
live (WF 0.63 → live 0.53, §3.1a). So a low test ECE is a reason to prefer a
model, not proof its 0.80 signals are right 80% of the time live. Until
`scripts/audit_live_performance.py` shows a positive confidence/accuracy slope on
LIVE rows, downstream confidence gates should stay off regardless of which
artifact is deployed.

**Mitigation (2026-09-04):** `_select_best_model` now applies a **calibration
tie-break** — candidates within `forex_config.MODEL_SELECTION_NOISE_BAND` (0.02)
of the top composite score are re-ranked by test-set ECE, best-calibrated wins,
and the override is logged. ECE is reported per candidate during training.
Run `scripts/check_probability_range.py` after each retrain and before the next
daily run to confirm (i) SELL is reachable at its threshold and (ii) you know
what the ≥ 0.80 share is.

**Run-to-run variance:** two retrains on identical config and data the same day
gave WF 0.6309 and 0.6348 for the same model. Run-to-run noise (~0.01 WF) exceeds
most between-model differences. One retrain's numbers are not a measurement.

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
| date_time | DATETIME | Prediction date (**not** `trading_date` — that column name belongs to `forex_hist_data`) |
| predicted_signal | VARCHAR | **'BUY', 'SELL', or 'HOLD'** — HOLD means the gate ABSTAINED (below threshold / vetoed), not a predicted class |
| signal_confidence | FLOAT | Max class probability (kept even when abstaining, so a 0.60 abstain is distinguishable from a 0.51 one) |
| prob_buy | FLOAT | P(BUY) = P(UP) from model |
| prob_sell | FLOAT | P(SELL) = P(DOWN) from model |
| prob_hold | FLOAT | Always 0.0 under the binary model (even for HOLD/abstain rows) |
| model_name | VARCHAR | `daily_automation_model` (daily run) |
| model_version | VARCHAR | `5.2_binary_rates` (current; daily rows carry a `+gated` suffix when the artifact predates the gate) |
| actual_return_1d / _5d / _10d | FLOAT | Realised forward return, backfilled after the fact. **Outcome columns, not predictions** — see the note below |
| direction_correct_1d / _5d | BIT | Outcome flag — **auto-`True` for HOLD rows**, see §3.1(d) |
| prediction_accuracy | VARCHAR | 'Correct'/'Incorrect', same HOLD caveat |

> **⚠️ There is no 5-day prediction.** The `_5d` / `_10d` columns are
> **retrospective outcomes** written by the upstream backfill job; nothing in this
> repo predicts a multi-day horizon. The model's only target is the sign of the
> **1-day** forward return (`future_return_1d`). `future_return_3d` and
> `future_return_5d` are computed in `train_enhanced_model.py` but appear solely
> in the feature-exclusion list — they are never labels.
>
> So the "+9.4 pt 5-day edge" in §3.1(a) is **the same 1-day signal scored over a
> longer window**, not the output of a 5-day model. It is knowable only in
> retrospect and cannot be published as a signal. Realising it would require
> re-labelling the target and retraining.
>
> `DEFAULT_PREDICTION_HORIZON=5` in `.env` is **dead config** — no code reads it.
> Do not assume it controls anything.

### Currency Pairs (actual, from `forex_hist_data`)
> Pair discovery is live (`get_forex_pairs()` = `SELECT DISTINCT symbol FROM
> forex_hist_data`) on both the training and daily-prediction paths, so pairs
> added to the DB are picked up automatically. On **2026-07-06** five pairs were
> added with ~1 year of history (AUD/NZD, EUR/GBP, GBP/JPY, USD/CAD, USD/CHF)
> and the model was retrained the same day (WF 0.655, best model `xgboost`,
> stability PASSED).

**14 live pairs:** EUR/USD, GBP/USD, AUD/USD, NZD/USD, USD/JPY, EUR/JPY,
EUR/CHF, USD/HKD, USD/SGD, AUD/NZD, EUR/GBP, GBP/JPY, USD/CAD, USD/CHF.
All train into the single global model.

**USD/INR is a 15th pair in the DB but is NOT live** — its `forex_hist_data`
stops at **2026-05-14** (upstream ingestion broken in the `stockanalysis` repo),
so the freshness gate skips it every day. It is still picked up by
`get_forex_pairs()` and **still contributes its stale rows to training**. Restore
ingestion or exclude it explicitly before reading anything into USD/INR output.

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
- **Scoring rule**: when measuring accuracy, ALWAYS (a) filter
  `predicted_signal <> 'HOLD'` — HOLD is auto-scored correct upstream, and
  (b) filter to a single `model_version` — pooling across versions mixes the
  pre- and post-leakage-fix eras and produces meaningless numbers, and
  (c) compare against the realised up-rate on the same rows, never against 0.50.
  `scripts/audit_live_performance.py` does all three.

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
python scripts/audit_live_performance.py           # LIVE accuracy vs base rate + confidence inversion
python scripts/check_probability_range.py          # post-retrain: SELL reachable? >=0.80 share stable?
```

> `audit_live_performance.py` is the check that matters after a retrain —
> walk-forward has consistently overstated live performance by ~10 pts. Run it
> ~2 weeks after any model change, once enough rows have been scored, and
> compare `edge_1d` / `edge_5d` (not raw accuracy) against the previous run.

---

## 6. DOWNSTREAM CONSUMERS
- **stockdata_agenticai** — Forex Agent reads `forex_ml_predictions` for daily briefing
- **streamlit-trading-dashboard** — Displays forex predictions and trends
- ⚠️ Signals are **BUY/SELL/HOLD**, where HOLD = the gate **abstained** (low
  conviction or technical veto), not a model prediction; `prob_hold` is always
  0.0. Consumers should treat HOLD as "no actionable signal today".
- Note: Forex is **excluded from Strategy 2 cross-analysis** (regression model underperformance)

### ⚠️ Contract notes for consumers (added 2026-09-04, see §3.1)
1. **`signal_confidence` must not be used as a quality/sizing input.** It is
   anti-correlated with live accuracy (0.55–0.60 → 57.4%; 0.70+ → 40.0%).
   A tier gate keyed to *higher* confidence selects the worse signals.
2. **≥ 0.80 confidence signals now exist again — do not treat that as a green
   light.** The pre-2026-09-04 `voting_soft` artifact never exceeded 0.783, so
   ≥ 0.80 gates were dormant; the `xgboost` artifact deployed 2026-09-04 clears
   0.80 on ~20% of firing signals. Those gates must remain OFF until the
   confidence/accuracy relationship is re-measured on live rows from this
   artifact. Re-arming them on backtest calibration alone repeats the mistake
   that produced §3.1(a).
3. **Do not compute accuracy over pooled `model_version`s**, and do not include
   HOLD rows — both inflate the result (see §3.1(d)).
4. **1-day direction has no live edge** (0.534 vs a 0.548 base rate). Consumers
   presenting these as actionable 1-day calls are overstating them. The 5-day
   column carries what edge exists.
5. **Rolling-window accuracy needs n ≥ 5** before it is displayed — at ~1–2 BUY
   signals per pair per week, a 7-day per-pair window is usually n=1–2.
6. **`forex_cluster_*` v4.0 rows are dead** (30 rows, 2026-06-23–25, rolled back
   for train/serve skew; they scored 12–14%). Consumers must filter to
   `model_name='daily_automation_model'` — they are not a live model family.

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
