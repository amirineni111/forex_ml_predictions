"""
Audit LIVE prediction performance from forex_ml_predictions.

Backtest walk-forward accuracy has not reproduced out-of-sample (see CLAUDE.md
section 3.1), so this is the check that matters. It enforces the three scoring
rules that pooled/naive queries get wrong:

  1. HOLD rows are excluded -- the upstream scoring job marks every HOLD
     `direction_correct_1d = True` regardless of outcome, so including them
     inflates accuracy toward the abstain rate.
  2. One model_version at a time -- pooling mixes the pre- and post-leakage-fix
     eras, whose confidence scales are not comparable.
  3. Accuracy is reported against the realised up-rate on the SAME rows, never
     against 0.50. Over Jul-Sep 2026 the base rate was 0.548, so a "53% BUY
     accuracy" is a losing signal, not a marginal edge.

Usage:
    python scripts/audit_live_performance.py                    # current version
    python scripts/audit_live_performance.py --version 5.0_binary_noleak
    python scripts/audit_live_performance.py --days 30
"""

import argparse
import os
import sys
from urllib.parse import quote_plus

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_VERSION = '5.2_binary_rates+gated'


def get_engine():
    load_dotenv()
    cs = (
        f"DRIVER={{{os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')}}};"
        f"SERVER={os.getenv('SQL_SERVER')};DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};PWD={os.getenv('SQL_PASSWORD')};"
        f"TrustServerCertificate=yes"
    )
    return create_engine(f"mssql+pyodbc:///?odbc_connect={quote_plus(cs)}")


def load(engine, version, days):
    where = ["model_version = :version", "actual_return_1d IS NOT NULL"]
    params = {'version': version}
    if days:
        where.append("date_time >= DATEADD(day, :days, GETDATE())")
        params['days'] = -abs(days)
    sql = text(f"""
        SELECT currency_pair, date_time, predicted_signal, signal_confidence,
               actual_return_1d, actual_return_5d,
               direction_correct_1d, direction_correct_5d
        FROM forex_ml_predictions
        WHERE {' AND '.join(where)}
    """)
    return pd.read_sql(sql, engine, params=params)


def edge_table(df):
    """BUY accuracy vs the always-long base rate, per pair."""
    rows = []
    for pair, g in df.groupby('currency_pair'):
        buys = g[g.predicted_signal == 'BUY']
        if buys.empty:
            continue
        rows.append({
            'pair': pair,
            'n_buy': len(buys),
            'acc_1d': buys.direction_correct_1d.mean(),
            'base_1d': (g.actual_return_1d > 0).mean(),
            'acc_5d': buys.direction_correct_5d.mean(),
            'base_5d': (g.actual_return_5d > 0).mean(),
        })
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t['edge_1d'] = t.acc_1d - t.base_1d
    t['edge_5d'] = t.acc_5d - t.base_5d
    return t.sort_values('edge_1d').round(3)


def confidence_table(df):
    """Accuracy by confidence bucket -- expected to be INVERTED (see 3.1b)."""
    acted = df[df.predicted_signal != 'HOLD'].copy()
    if acted.empty:
        return pd.DataFrame()
    bins = [0, 0.55, 0.60, 0.65, 0.70, 1.01]
    labels = ['<0.55', '0.55-0.60', '0.60-0.65', '0.65-0.70', '0.70+']
    acted['bucket'] = pd.cut(acted.signal_confidence.astype(float),
                             bins=bins, labels=labels, right=False)
    return (acted.groupby('bucket', observed=True)
                 .agg(n=('direction_correct_1d', 'size'),
                      acc_1d=('direction_correct_1d', 'mean'))
                 .round(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default=DEFAULT_VERSION)
    ap.add_argument('--days', type=int, default=None,
                    help='limit to the trailing N days (default: all rows)')
    args = ap.parse_args()

    df = load(get_engine(), args.version, args.days)
    if df.empty:
        print(f"[ERROR] no scored rows for model_version={args.version!r}")
        return 1

    acted = df[df.predicted_signal != 'HOLD']
    print(f"\nmodel_version : {args.version}")
    print(f"window        : {df.date_time.min():%Y-%m-%d} -> {df.date_time.max():%Y-%m-%d}")
    print(f"rows          : {len(df)} scored ({len(acted)} acted, "
          f"{len(df) - len(acted)} HOLD/abstain excluded)")

    base_1d = (df.actual_return_1d > 0).mean()
    base_5d = (df.actual_return_5d > 0).mean()
    print("\n== headline: acted signals vs always-long base rate ==")
    for h, base in (('1d', base_1d), ('5d', base_5d)):
        acc = acted[f'direction_correct_{h}'].mean()
        print(f"  {h}: accuracy {acc:.3f} | base {base:.3f} | "
              f"edge {acc - base:+.3f}{'   <-- NO EDGE' if acc - base <= 0 else ''}")

    print("\n== per-pair BUY edge over base rate ==")
    t = edge_table(df)
    print(t.to_string(index=False) if not t.empty else "  (no BUY signals)")

    print("\n== accuracy by confidence bucket (expect INVERSION) ==")
    c = confidence_table(df)
    print(c.to_string() if not c.empty else "  (no acted signals)")
    if len(c) >= 2 and c.acc_1d.iloc[-1] < c.acc_1d.iloc[0]:
        print("  [WARN] confidence is anti-correlated with accuracy -- do NOT "
              "use signal_confidence for sizing or tiering (CLAUDE.md 3.1b)")

    print("\n== signal mix ==")
    print(df.predicted_signal.value_counts().to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
