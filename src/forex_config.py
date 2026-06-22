"""
Forex configuration: currency-pair clusters, currency archetypes, external
data table names, and FRED series references.

This repo keeps config in `.env` + flat modules under `src/` (imported as
top-level packages because callers do `sys.path.append('src')`). This module
centralises the *structural* config that does not belong in `.env`:

- which cluster each pair is trained in (per-cluster models, see plan Phase 4)
- currency archetypes for risk/commodity feature routing (Phase 3)
- names of the new external-data tables (Phase 1/2)
- FRED series ids used to seed rate/yield data (Phase 1 ingestion reference)
"""

from __future__ import annotations

import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# Currency pairs in production. Reflects the symbols actually present in
# forex_hist_data; main() fetches the live list via get_forex_pairs() and this
# is only a fallback.
# ---------------------------------------------------------------------------
DEFAULT_PAIRS: List[str] = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD',
    'EURJPY', 'EURCHF', 'USDHKD', 'USDSGD', 'USDINR',
]

# ---------------------------------------------------------------------------
# Per-cluster model mapping (Phase 4).
# Each pair maps to EXACTLY ONE cluster so it trains in exactly one model.
#   usd_majors  — USD strength / Fed / DXY driven (EUR/USD, GBP/USD)
#   commodity   — risk-on & commodity driven (AUD, NZD)
#   jpy_crosses — JPY legs, safe-haven / risk-off driven
#   eur_crosses — intra-European, rate-differential driven (EUR/CHF)
#   usd_asia    — USD vs pegged/managed Asian currencies (HKD peg, SGD basket,
#                 INR managed) — fundamentally different dynamics from free floats
# Unknown pairs fall back to FALLBACK_CLUSTER.
# ---------------------------------------------------------------------------
PAIR_TO_CLUSTER: Dict[str, str] = {
    # usd majors (free-floating vs USD)
    'EURUSD': 'usd_majors',
    'GBPUSD': 'usd_majors',
    # commodity block
    'AUDUSD': 'commodity',
    'NZDUSD': 'commodity',
    'USDCAD': 'commodity',
    # JPY crosses
    'USDJPY': 'jpy_crosses',
    'EURJPY': 'jpy_crosses',
    'GBPJPY': 'jpy_crosses',
    # intra-European crosses
    'EURCHF': 'eur_crosses',
    'EURGBP': 'eur_crosses',
    'USDCHF': 'eur_crosses',   # CHF-driven; group with European crosses
    # USD vs pegged/managed Asian currencies
    'USDHKD': 'usd_asia',
    'USDSGD': 'usd_asia',
    'USDINR': 'usd_asia',
}

FALLBACK_CLUSTER = 'usd_majors'

# Minimum training rows required to train a dedicated cluster model; below this
# the cluster is skipped and the global model (best_forex_model) covers it.
# Sized for ~1 cluster of 1-3 pairs over the available daily history.
MIN_CLUSTER_SAMPLES = 250


def cluster_for_pair(pair: str) -> str:
    """Return the cluster id for a currency pair (normalises EUR/USD → EURUSD)."""
    key = (pair or '').replace('/', '').replace('_', '').upper()
    return PAIR_TO_CLUSTER.get(key, FALLBACK_CLUSTER)


def all_clusters() -> List[str]:
    """Distinct cluster ids in a stable order."""
    seen: List[str] = []
    for c in PAIR_TO_CLUSTER.values():
        if c not in seen:
            seen.append(c)
    return seen


def cluster_model_path(cluster: str, model_dir: str = 'data') -> str:
    """Path of the per-cluster model artifact, e.g. data/forex_model_usd_majors.joblib."""
    return os.path.join(model_dir, f'forex_model_{cluster}.joblib')


# ---------------------------------------------------------------------------
# Currency archetypes (Phase 3 risk/commodity routing).
# Used to interact macro signals (VIX/DXY/gold/oil) with the pair's character.
# ---------------------------------------------------------------------------
SAFE_HAVEN_CCY = {'JPY', 'CHF'}        # rise in risk-off
COMMODITY_CCY = {'AUD', 'NZD', 'CAD'}  # track risk-on / commodities

# All currencies that appear across the basket (base + quote of every pair)
def currencies_in(pairs: List[str]) -> List[str]:
    """Unique 3-letter currencies appearing as base or quote across `pairs`."""
    ccys: List[str] = []
    for p in pairs:
        key = p.replace('/', '').replace('_', '').upper()
        for ccy in (key[:3], key[3:6]):
            if ccy and ccy not in ccys:
                ccys.append(ccy)
    return ccys


# ---------------------------------------------------------------------------
# External data tables (Phase 1/2). Read by ExternalDataSources; populated by
# the ingestion repo (stockanalysis) or a FRED seed script.
# ---------------------------------------------------------------------------
RATES_TABLE = os.getenv('FOREX_RATES_TABLE', 'forex_rates_daily')
INTERMARKET_TABLE = os.getenv('FOREX_INTERMARKET_TABLE', 'forex_intermarket_daily')
ECON_EVENTS_TABLE = os.getenv('FOREX_ECON_EVENTS_TABLE', 'forex_econ_events')

# FRED series reference for seeding forex_rates_daily (ingestion-side reference).
# policy rate + 2Y + 10Y per currency where a daily FRED series exists.
FRED_API_KEY = os.getenv('FRED_API_KEY', '')
FRED_RATE_SERIES: Dict[str, Dict[str, str]] = {
    'USD': {'policy_rate': 'DFF',    'yield_2y': 'DGS2',          'yield_10y': 'DGS10'},
    'EUR': {'policy_rate': 'ECBDFR', 'yield_2y': '',              'yield_10y': 'IRLTLT01EZM156N'},
    'JPY': {'policy_rate': '',       'yield_2y': '',              'yield_10y': 'IRLTLT01JPM156N'},
    'GBP': {'policy_rate': '',       'yield_2y': '',              'yield_10y': 'IRLTLT01GBM156N'},
    'AUD': {'policy_rate': '',       'yield_2y': '',              'yield_10y': 'IRLTLT01AUM156N'},
    'CAD': {'policy_rate': '',       'yield_2y': '',              'yield_10y': 'IRLTLT01CAM156N'},
    'NZD': {'policy_rate': '',       'yield_2y': '',              'yield_10y': 'IRLTLT01NZM156N'},
    'CHF': {'policy_rate': '',       'yield_2y': '',              'yield_10y': 'IRLTLT01CHM156N'},
}
