"""ng2.1 — Bull/Bear specialist trainers (regime-filtered, DD-penalized).

Reuses ng1.0.1 feature cache (no new schema). Specialization happens through:
  1. Training data filter (regime_v2 ∈ {+1, -1} from market_regime_signals)
  2. Bear-only DD-penalty label transform

Risk control (L1-L5) lives in tomorrow_stock_selector.py, not here.
"""
from ml_models.ng21.ng21_trainer import NG21Trainer

__all__ = ['NG21Trainer']
