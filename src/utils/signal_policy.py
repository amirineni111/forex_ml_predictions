"""
Signal policy: the single source of truth for confidence thresholds and the
technical veto, shared by BOTH prediction paths (predict_forex_signals.py and
src/utils/forward_prediction.py — the scheduled daily run).

History: these rules originally lived only in predict_forex_signals.py, so the
production daily path (forward_prediction) shipped raw argmax signals and the
0.75 SELL threshold was silently dead in production. Keep all threshold/veto
logic in this module so the two paths cannot diverge again.
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Asymmetric signal thresholds (configurable via environment).
#
# SELL was over-predicted, so its bar was originally raised to 0.75. But after
# the 2026-07-06 retrain the current model's probability distribution compressed:
# prob_sell now tops out near ~0.70 (trailing-30-day max), so a 0.75 gate was
# mathematically unreachable and SELL never fired at all. Lowered the default to
# 0.62 (2026-07-26) — still asymmetrically stricter than BUY (0.55) so SELL only
# fires with clear conviction, but now actually reachable.
#
# Override without a code change via env vars, e.g. FOREX_SELL_THRESHOLD=0.60.
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == '':
        return default
    try:
        return float(raw)
    except ValueError:
        return default


SIGNAL_THRESHOLDS = {
    'BUY':  _env_float('FOREX_BUY_THRESHOLD', 0.55),   # BUY accuracy is solid
    'SELL': _env_float('FOREX_SELL_THRESHOLD', 0.62),  # reachable + still strict
    'HOLD': 0.50,                                       # neutral baseline
}


def apply_signal_thresholds(prob_map: dict) -> str:
    """
    Convert raw class probabilities to a signal using asymmetric thresholds.

    Falls back to 'HOLD' when no class clears its threshold (no conviction).
    """
    candidates = {
        sig: prob for sig, prob in prob_map.items()
        if prob >= SIGNAL_THRESHOLDS.get(sig, 0.50)
    }
    if not candidates:
        return 'HOLD'
    return max(candidates, key=candidates.get)


# ---------------------------------------------------------------------------
# Technical veto layer
# Pattern B: RSI<55, MACD<0, price<SMA20, BB%<40 → SELL accuracy only 37%.
# When all 4 conditions hold, demote SELL → HOLD.
# ---------------------------------------------------------------------------
def technical_veto(signal: str, row: 'pd.Series') -> str:
    """
    Suppress SELL signals when technical indicators contradict the bearish call.
    Only SELL signals are evaluated — BUY and HOLD pass through unchanged.
    """
    if signal != 'SELL':
        return signal

    rsi    = float(row.get('rsi_14', 50) or 50)
    macd   = float(row.get('macd',    0) or 0)
    close  = float(row.get('close_price', 0) or 0)
    sma20  = float(row.get('sma_20',  close) or close)
    bb_pct = float(row.get('bb_percent', 50) or 50)   # 0–1 scale from prepare_features

    # bb_percent is stored as 0–1 ratio; convert to 0–100 if needed
    if bb_pct <= 1.0:
        bb_pct *= 100.0

    pattern_b = (
        rsi   < 55    and
        macd  < 0     and
        close < sma20 and
        bb_pct < 40.0
    )

    if pattern_b:
        return 'HOLD'   # veto: model is in Pattern B — historically only 37% accurate
    return signal


def gate_binary_signal(prob_buy: float, prob_sell: float, row: 'pd.Series' = None) -> tuple:
    """
    Apply the asymmetric thresholds + technical veto to a BINARY model output.

    The binary model only knows UP/DOWN; 'HOLD' here means ABSTAIN (no
    conviction), not a predicted class — prob_hold stays 0.0 in the output.

    Returns (signal, reason):
        ('BUY'|'SELL', 'threshold') — the winning side cleared its bar
        ('HOLD', 'abstain')         — neither side cleared its threshold
        ('HOLD', 'veto')            — SELL cleared 0.75 but Pattern B vetoed it
    """
    signal = apply_signal_thresholds({'BUY': prob_buy, 'SELL': prob_sell})
    if signal == 'HOLD':
        return 'HOLD', 'abstain'

    if signal == 'SELL' and row is not None:
        if technical_veto('SELL', row) == 'HOLD':
            return 'HOLD', 'veto'

    return signal, 'threshold'
