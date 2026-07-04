"""
A/B acceptance gate for the forex_rates_daily rate-differential features.

Trains the production model twice on the same data window — baseline (no
rates features) vs candidate (with rates features) — WITHOUT touching the
model files, compares walk-forward accuracy, and saves ONLY the winner.

Accept rule: candidate best-model wf_accuracy_mean >= baseline - 0.005.
Sanity rule: baseline WF must be in the known honest band (0.52-0.70);
             >0.70 is the leakage signature (see CLAUDE.md §5) — abort.

Exit codes: 0 = candidate accepted (rates model saved)
            1 = candidate rejected (baseline model saved)
            2 = sanity failure / error (nothing saved)

If the candidate is ACCEPTED you must also flip INCLUDE_RATES_FEATURES=True
in src/forex_config.py (and bump MODEL_VERSION) so future weekly retrains
reproduce the accepted configuration. The predict path keys off the saved
artifact's feature list, so it follows the winner automatically either way.

Usage:  python scripts/compare_rates_features.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from train_enhanced_model import EnhancedForexTrainer, safe_print  # noqa: E402

WF_TOLERANCE = 0.005        # candidate may be at most this much worse
SANE_WF_LOW, SANE_WF_HIGH = 0.52, 0.70


def run_arm(label: str, include_rates: bool, pairs: list) -> dict:
    """Train one arm (fresh trainer per arm so no state leaks between them)."""
    safe_print("\n" + "=" * 70)
    safe_print(f"  ARM: {label} (include_rates={include_rates})")
    safe_print("=" * 70)
    trainer = EnhancedForexTrainer()
    results = trainer.train_production_model(
        pairs=pairs, include_rates=include_rates, save_artifact=False)
    if not results:
        raise RuntimeError(f"{label} arm produced no results")
    results['_trainer'] = trainer
    return results


def best_wf(results: dict) -> float:
    name = results['best_model_name']
    wf = results.get('wf_results', {}).get(name, {})
    return float(wf.get('wf_accuracy_mean', 0.0))


def save_winner(results: dict):
    trainer = results.pop('_trainer')
    trainer._save_enhanced_model(
        results['best_model'], results['best_model_name'],
        results['feature_columns'], results['label_encoder'],
        results['results'], results['scaler'],
        results['wf_results'],
        feature_fill_values=results.get('feature_fill_values'),
    )


def main() -> int:
    trainer = EnhancedForexTrainer()
    if not trainer.db.test_connection():
        safe_print("[ERROR] Database connection failed")
        return 2
    pairs = trainer.db.get_forex_pairs()
    safe_print(f"[INFO] A/B on {len(pairs)} pairs: {', '.join(pairs)}")

    baseline = run_arm('BASELINE (no rates)', include_rates=False, pairs=pairs)
    candidate = run_arm('CANDIDATE (rates)', include_rates=True, pairs=pairs)

    b_name, c_name = baseline['best_model_name'], candidate['best_model_name']
    b_wf, c_wf = best_wf(baseline), best_wf(candidate)
    b_test = baseline.get('test_results', {}).get(b_name, {}).get('test_accuracy', float('nan'))
    c_test = candidate.get('test_results', {}).get(c_name, {}).get('test_accuracy', float('nan'))
    rate_feats = [f for f in candidate['feature_columns'] if str(f).startswith('rate_')]

    safe_print("\n" + "=" * 70)
    safe_print("  A/B COMPARISON")
    safe_print("=" * 70)
    safe_print(f"  {'':22s} {'baseline':>12s} {'candidate':>12s}")
    safe_print(f"  {'best model':22s} {b_name:>12s} {c_name:>12s}")
    safe_print(f"  {'walk-forward acc':22s} {b_wf:>12.4f} {c_wf:>12.4f}")
    safe_print(f"  {'test acc':22s} {b_test:>12.4f} {c_test:>12.4f}")
    safe_print(f"  {'stability passed':22s} {str(baseline.get('stability_passed')):>12s} "
               f"{str(candidate.get('stability_passed')):>12s}")
    safe_print(f"  {'n features':22s} {len(baseline['feature_columns']):>12d} "
               f"{len(candidate['feature_columns']):>12d}")
    safe_print(f"  rate_* features selected in candidate: "
               f"{', '.join(rate_feats) if rate_feats else '(none survived selection)'}")

    # Sanity: an out-of-band baseline means something else broke — accept neither.
    if not (SANE_WF_LOW <= b_wf <= SANE_WF_HIGH):
        safe_print(f"\n[ERROR] Baseline WF {b_wf:.4f} outside sane band "
                   f"[{SANE_WF_LOW}, {SANE_WF_HIGH}] — >0.70 is the leakage "
                   f"signature. NOT saving either model. Investigate first.")
        return 2

    if c_wf >= b_wf - WF_TOLERANCE and rate_feats:
        safe_print(f"\n[ACCEPT] Candidate WF {c_wf:.4f} >= baseline {b_wf:.4f} - {WF_TOLERANCE} "
                   f"and {len(rate_feats)} rate feature(s) in use. Saving CANDIDATE.")
        save_winner(candidate)
        safe_print("[ACTION REQUIRED] Flip INCLUDE_RATES_FEATURES=True in "
                   "src/forex_config.py so weekly retrains keep the rates features.")
        return 0

    if c_wf >= b_wf - WF_TOLERANCE and not rate_feats:
        safe_print(f"\n[REJECT] Candidate WF is fine ({c_wf:.4f}) but feature selection "
                   f"dropped every rate_* feature — rates add nothing yet. Saving BASELINE.")
    else:
        safe_print(f"\n[REJECT] Candidate WF {c_wf:.4f} < baseline {b_wf:.4f} - {WF_TOLERANCE}. "
                   f"Saving BASELINE.")
    save_winner(baseline)
    safe_print("[INFO] Leave INCLUDE_RATES_FEATURES=False in src/forex_config.py.")
    return 1


if __name__ == '__main__':
    sys.exit(main())
