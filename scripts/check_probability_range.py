"""Post-retrain guard: does the current artifact still produce a usable
probability range?

Model choice rescales confidence even when walk-forward accuracy is unchanged
(see CLAUDE.md 3.1e). Two failure modes this catches:

  1. SELL becomes unreachable -- the probability spread compresses below the
     SELL threshold and SELL silently stops firing (this happened in Jul 2026).
  2. The >=0.80 confidence share jumps -- re-arming downstream tier gates on
     signals that are, per the live audit, the LEAST accurate ones.

Read-only: builds serve-path features and calls predict_proba. No DB writes.
Run after every retrain, before the next daily run.

    python scripts/check_probability_range.py
"""
import sys, os, warnings, logging
warnings.filterwarnings('ignore'); logging.disable(logging.INFO)
R=r"c:\Users\sreea\OneDrive\Desktop\sqlserver_copilot_forex"
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,'src')); os.chdir(R)
import numpy as np, pandas as pd, joblib
from predict_forex_signals import ForexTradingSignalPredictor
from src.utils.signal_policy import SIGNAL_THRESHOLDS

art=joblib.load(os.path.join(R,'data','best_forex_model.joblib'))
model, scaler, le = art['model'], art.get('scaler'), art['label_encoder']
feats, fills = art['feature_columns'], art.get('feature_fill_values') or {}
cls=list(le.classes_); iu,idn=cls.index('UP'),cls.index('DOWN')

p=ForexTradingSignalPredictor(); p.load_model_artifacts()
rows=[]
for pair in p.db.get_forex_pairs():
    try:
        df=p.db.get_forex_data_with_indicators(pair, days_back=400)
        if df is None or df.empty: continue
        p.currency_pair=pair
        X,_=p.prepare_features(df)
        if X is None or len(X)==0: continue
        Xm=X.reindex(columns=feats)
        for c in feats:
            Xm[c]=pd.to_numeric(Xm[c],errors='coerce').fillna(fills.get(c,0))
        Xs=scaler.transform(Xm) if scaler is not None else Xm.values
        pr=model.predict_proba(Xs)[-60:]
        for b,s in zip(pr[:,iu],pr[:,idn]): rows.append({'pair':pair,'pb':b,'ps':s})
    except Exception as e:
        print(f"  [skip] {pair}: {type(e).__name__}: {e}")
d=pd.DataFrame(rows)
print(f"\nartifact : {art['model_name']} trained {art['training_date'][:10]}")
print(f"thresholds: BUY>={SIGNAL_THRESHOLDS['BUY']}  SELL>={SIGNAL_THRESHOLDS['SELL']}")
if d.empty: print("NO ROWS SCORED"); sys.exit(1)
print(f"rows: {len(d)} over {d['pair'].nunique()} pairs (last ~60 bars each)\n")
print("prob_buy  min %.3f  max %.3f" % (d.pb.min(), d.pb.max()))
print("prob_sell min %.3f  max %.3f" % (d.ps.min(), d.ps.max()))
nb=(d.pb>=SIGNAL_THRESHOLDS['BUY']).sum(); ns=(d.ps>=SIGNAL_THRESHOLDS['SELL']).sum()
print(f"\nwould fire BUY : {nb:4d}/{len(d)} ({nb/len(d):5.1%})")
print(f"would fire SELL: {ns:4d}/{len(d)} ({ns/len(d):5.1%})" + ("   <-- SELL UNREACHABLE" if ns==0 else ""))
print("\nprob_sell pctiles:", {f"p{q}": round(float(np.percentile(d.ps,q)),3) for q in (50,90,95,99,100)})
print("prob_buy  pctiles:", {f"p{q}": round(float(np.percentile(d.pb,q)),3) for q in (50,90,95,99,100)})
fired=d[(d.pb>=SIGNAL_THRESHOLDS['BUY'])|(d.ps>=SIGNAL_THRESHOLDS['SELL'])].copy()
fired['conf']=fired[['pb','ps']].max(axis=1)
print(f"CONF80: of {len(fired)} firing signals, {(fired.conf>=0.80).sum()} ({(fired.conf>=0.80).mean():.1%}) have confidence >= 0.80")
