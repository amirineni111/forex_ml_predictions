"""
Train/serve feature-parity check (read-only against the DB).

Builds the SAME pair's feature row through BOTH pipelines — the training path
(EnhancedForexTrainer.prepare_enhanced_dataset) and the predict path
(ForexTradingSignalPredictor.prepare_features) — restricted to the model
artifact's feature_columns, aligned on the latest common date, and compares
value by value.

This is the regression guard for the skew class that has bitten this repo
twice (rolled-back relative features 2026-06-25; sentiment features zero-
filled at predict time). Run it after any change to feature engineering,
external merges, or the training pipeline.

Exit codes: 0 = parity OK, 1 = mismatches found, 2 = could not compare.

Usage:  python scripts/check_train_serve_parity.py [PAIR]   (default EURUSD)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = './data/best_forex_model.joblib'
LOOKBACK_DAYS = 150


def main() -> int:
    pair = sys.argv[1] if len(sys.argv) > 1 else 'EURUSD'

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] {MODEL_PATH} not found - train a model first")
        return 2
    artifacts = joblib.load(MODEL_PATH)
    feature_columns = artifacts.get('feature_columns') or []
    if not feature_columns:
        print("[ERROR] Artifact has no feature_columns")
        return 2
    print(f"[INFO] Artifact: {artifacts.get('model_name')} "
          f"v{artifacts.get('model_version')} - {len(feature_columns)} features")

    # --- Training path ---
    from train_enhanced_model import EnhancedForexTrainer
    trainer = EnhancedForexTrainer()
    train_df = trainer.prepare_enhanced_dataset([pair], lookback_days=LOOKBACK_DAYS)
    train_df['date_time'] = pd.to_datetime(train_df['date_time']).dt.normalize()

    # --- Predict path ---
    from predict_forex_signals import ForexTradingSignalPredictor
    predictor = ForexTradingSignalPredictor(model_path=MODEL_PATH, currency_pair=pair)
    raw = predictor.get_forex_data(currency_pair=pair, days_back=LOOKBACK_DAYS)
    pred_df, _ = predictor.prepare_features(raw)
    pred_df['date_time'] = pd.to_datetime(pred_df['date_time']).dt.normalize()

    common = sorted(set(train_df['date_time']) & set(pred_df['date_time']))
    if not common:
        print("[ERROR] No common dates between the two pipelines")
        return 2
    # Training drops its last row (needs t+1 for the target), so the latest
    # common date is typically yesterday — that's fine for parity purposes.
    day = common[-1]
    t_row = train_df[train_df['date_time'] == day].iloc[-1]
    p_row = pred_df[pred_df['date_time'] == day].iloc[-1]
    print(f"[INFO] Comparing {pair} features on {day.date()}")

    mismatches, one_sided, missing = [], [], []
    for col in feature_columns:
        in_t, in_p = col in t_row.index, col in p_row.index
        if not in_t or not in_p:
            missing.append((col, 'train' if not in_t else 'predict'))
            continue
        tv, pv = t_row[col], p_row[col]
        try:
            tv, pv = float(tv), float(pv)
        except (TypeError, ValueError):
            if str(tv) != str(pv):
                mismatches.append((col, tv, pv))
            continue
        t_nan, p_nan = np.isnan(tv), np.isnan(pv)
        if t_nan and p_nan:
            continue
        if t_nan != p_nan:
            one_sided.append((col, tv, pv))
            continue
        if not np.isclose(tv, pv, rtol=1e-4, atol=1e-8):
            mismatches.append((col, tv, pv))

    print(f"\n[RESULT] {len(feature_columns)} features checked: "
          f"{len(mismatches)} value mismatches, {len(one_sided)} one-sided NaN, "
          f"{len(missing)} missing from a pipeline")

    for name, group in (('VALUE MISMATCH', mismatches), ('ONE-SIDED NaN', one_sided)):
        for col, tv, pv in group[:20]:
            print(f"  [{name}] {col}: train={tv} predict={pv}")
    for col, side in missing[:20]:
        print(f"  [MISSING] {col}: absent from {side} pipeline")

    if mismatches or one_sided or missing:
        print("\n[FAIL] Train/serve feature parity violated - fix before deploying")
        return 1
    print("\n[PASS] Train and serve pipelines produce identical model features")
    return 0


if __name__ == '__main__':
    sys.exit(main())
