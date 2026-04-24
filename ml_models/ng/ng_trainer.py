#!/usr/bin/env python3
"""
NG v1.0.3 Trainer — inherits V485Trainer, overrides feature loading to use ng_feature_cache.

v1.0.3 changes from v1.0.0 (ng1.0.0):
  - 58 stock features + 10 market = 68 total (was 52+10=62)
  - Labels are industry excess returns (not absolute)
  - ICIR adaptive composite weights (not hardcoded)
  - WF summary JSON generation for L4 scoring
  - Removed 11 low-efficiency factors, added 10 CS rank + 5 residual + 3 sector activity
"""

import json
import logging
import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.training.train_v395_multi_target import V485Trainer
from ml_models.ng.ng_schema import (
    get_table_name, version_ge, get_schema_version,
    _is_1_2_branch, _is_1_3_branch, _is_1_4_branch, _is_1_5_branch, _is_1_6_branch,
    _version_in_range,
)
from ml_models.common.lgb_rank_utils import RANK_BASE_PARAMS, build_groups_per_date
from ml_models.ng.ng_margin_loss import make_margin_objective, make_margin_eval_metric
from ml_models.ng.ng_quintile_ce import (
    N_CLASSES, QUINTILE_MODEL_KEY, QuintileStrongBuyModel, make_quintile_dataset,
)
import lightgbm as lgb

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads

logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# ---------------------------------------------------------------------------
# Feature name constants — NG v1.0.3 (66 total = 56 stock + 10 market)
# ---------------------------------------------------------------------------

STOCK_FEATURE_NAMES: List[str] = [
    # Trend state (4, was 5 — removed pullback_from_high: IC flips pre/post-2020)
    'trend_strength_20d', 'days_since_breakout', 'adx_proxy',
    'volume_contraction',
    # Pullback entry (6)
    'pullback_to_ma10', 'pullback_to_ma20', 'rsi_14',
    'kdj_j_value', 'lower_shadow_ratio', 'intraday_recovery',
    # Volume confirmation (8)
    'volume_ratio_5d', 'volume_price_corr', 'obv_trend', 'volume_breakout',
    'log_amount_ma5', 'turnover_rate', 'up_volume_ratio', 'volume_cv',
    # Fundamental quality (13, was 14 — removed log_market_cap: IC flips pre/post-2020)
    'roe_ttm', 'roe_change', 'revenue_growth', 'net_profit_margin', 'ocf_quality',
    'pe_ttm', 'pb', 'pe_percentile_60d', 'debt_to_assets', 'current_ratio',
    'log_adv_20d', 'free_float_ratio', 'dv_ratio',
    # Industry rotation (11)
    'industry_return_5d', 'industry_return_20d', 'industry_relative_strength',
    'industry_breadth', 'industry_volume_change', 'industry_rank_return_5d',
    'sw_index_return_5d', 'industry_hhi',
    'sector_breadth_vs_market', 'sector_volume_vs_market', 'n_sectors_strong',
    # Cross-sectional rank (9, was 10 — removed cs_rank_market_cap: IC flips pre/post-2020)
    'cs_rank_return_5d', 'cs_rank_return_20d', 'cs_rank_volume_surge',
    'cs_rank_turnover', 'cs_rank_rsi', 'cs_rank_new_high',
    'cs_rank_pullback', 'cs_rank_volatility', 'cs_rank_pe',
    # Residual factors (5)
    'residual_return_20d', 'residual_volume', 'idiosyncratic_volatility',
    'residual_skewness', 'relative_strength_vs_peers',
]

MARKET_FEATURE_NAMES: List[str] = [
    'market_return_5d', 'market_return_20d', 'market_volatility_20d',
    'market_breadth', 'market_new_high_ratio', 'northbound_flow_5d',
    'market_volume_ratio', 'market_drawdown', 'vix_proxy', 'market_momentum_diff',
]

ALL_FEATURE_NAMES: List[str] = STOCK_FEATURE_NAMES + MARKET_FEATURE_NAMES  # 66 total (56 stock + 10 market)

SMOOTHING_FEATURE_NAMES: List[str] = [
    # Long-horizon trend (3)
    'trend_strength_60d', 'ma60_distance', 'price_channel_pos_40d',
    # Volatility regime (3)
    'vol_ratio_5d_60d', 'vol_regime', 'downside_vol_20d',
    # Drawdown state (3)
    'current_drawdown', 'recovery_speed_20d', 'gap_risk_20d',
]

# ng1.0.4 = ng1.0.3 base (56 stock) + 9 smoothing = 65 stock + 10 market = 75 total
NG104_STOCK_FEATURES: List[str] = STOCK_FEATURE_NAMES + SMOOTHING_FEATURE_NAMES
NG104_ALL_FEATURES: List[str] = NG104_STOCK_FEATURES + MARKET_FEATURE_NAMES

MONEYFLOW_FEATURE_NAMES: List[str] = [
    'net_mf_ratio_5d', 'big_order_ratio', 'big_order_trend_5d',
    'small_vs_big_divergence', 'mf_concentration', 'mf_momentum_10d',
    'mf_volume_divergence', 'northbound_stock_5d',
]

INTERACTION_FEATURE_NAMES: List[str] = [
    'ix_vol_pullback', 'ix_big_trend', 'ix_rsi_mf', 'ix_ind_big',
    'ix_mf_efficiency', 'ix_vol_surge_pullback', 'ix_alpha_conc',
    'ix_north_cap',
]

# ng1.0.7: Extended market state features (8)
EXTENDED_MARKET_FEATURE_NAMES: List[str] = [
    'amv_var1', 'amv_macd', 'amv_regime_days',
    'market_ret_60d', 'market_vol_ratio', 'breadth_momentum_5d',
    'market_skewness_20d', 'liquidity_stress',
]

# ng1.0.7: Conditional interaction features (7, IC-screened)
CONDITIONAL_IX_FEATURE_NAMES: List[str] = [
    'cx_beta_mkt_vol', 'cx_momentum_trend', 'cx_ind_mkt_dir',
    'cx_vol_stress', 'cx_drawdown_regime', 'cx_value_bear',
    'cx_quality_stress',
]

# ng1.0.7 = ng1.0.3 base (56 stock) + 18 market (10 base + 8 extended) + 7 cond_ix = 81 total
NG107_MARKET_FEATURES: List[str] = MARKET_FEATURE_NAMES + EXTENDED_MARKET_FEATURE_NAMES
NG107_ALL_FEATURES: List[str] = STOCK_FEATURE_NAMES + NG107_MARKET_FEATURES + CONDITIONAL_IX_FEATURE_NAMES

# v1.0.0 constants (for backward compatibility reference)
NG_V1_VERSION = 'ng1.0.0'
NG_VERSION = 'ng1.0.3'
NG104_VERSION = 'ng1.0.4'
NG107_VERSION = 'ng1.0.7'


def _describe_series(s: pd.Series) -> str:
    """One-line summary for label/feature series logging."""
    d = s.describe(percentiles=[0.10, 0.90])
    return (f"n={int(d['count']):,} mean={d['mean']:.4f} std={d['std']:.4f} "
            f"p10={d['10%']:.4f} p90={d['90%']:.4f}")

# ---------------------------------------------------------------------------
# ng1.1.0: 基于ng1.0.1精简 + bug修复 + P2新因子
# ---------------------------------------------------------------------------
# P0移除: roe_change(5版本近零), n_sectors_strong(3版本全零),
#          days_since_breakout(4版本近零, 被adx_proxy替代)
# Bug修复移除: volume_contraction(≡volume_ratio_5d), sw_index_return_5d(≡industry_return_5d),
#              industry_relative_strength(≡residual_return_20d)
# Bug修复重命名: revenue_growth→profit_margin_ratio (原值是margin非growth)
# 新增: revenue_growth(真正的or_yoy) + P2的4个因子
_NG110_PRUNED = frozenset({
    'roe_change', 'n_sectors_strong', 'days_since_breakout',   # P0
    'volume_contraction', 'sw_index_return_5d',                 # Bug #1, #2
    'industry_relative_strength',                               # Bug #3
    'revenue_growth',                                           # Bug #4: was margin, replaced by real or_yoy
})
_NG101_STOCK_FEATURES: List[str] = [
    # Trend (5→3 after pruning volume_contraction + days_since_breakout)
    'trend_strength_20d', 'days_since_breakout', 'adx_proxy',
    'pullback_from_high', 'volume_contraction',
    # Pullback entry (6)
    'pullback_to_ma10', 'pullback_to_ma20', 'rsi_14',
    'kdj_j_value', 'lower_shadow_ratio', 'intraday_recovery',
    # Volume confirmation (8→7 after pruning volume_contraction)
    'volume_ratio_5d', 'volume_price_corr', 'obv_trend', 'volume_breakout',
    'log_amount_ma5', 'turnover_rate', 'up_volume_ratio', 'volume_cv',
    # Fundamental (14→13 after removing roe_change; revenue_growth renamed)
    'roe_ttm', 'roe_change', 'revenue_growth', 'net_profit_margin', 'ocf_quality',
    'pe_ttm', 'pb', 'pe_percentile_60d', 'debt_to_assets', 'current_ratio',
    'log_market_cap', 'log_adv_20d', 'free_float_ratio', 'dv_ratio',
    # Industry (11→9 after removing sw_index_return_5d + industry_relative_strength + n_sectors_strong)
    'industry_return_5d', 'industry_return_20d', 'industry_relative_strength',
    'industry_breadth', 'industry_volume_change', 'industry_rank_return_5d',
    'sw_index_return_5d', 'industry_hhi',
    'sector_breadth_vs_market', 'sector_volume_vs_market', 'n_sectors_strong',
    # Cross-sectional rank (10)
    'cs_rank_return_5d', 'cs_rank_return_20d', 'cs_rank_volume_surge',
    'cs_rank_turnover', 'cs_rank_rsi', 'cs_rank_new_high',
    'cs_rank_pullback', 'cs_rank_volatility', 'cs_rank_market_cap', 'cs_rank_pe',
    # Residual factors (5)
    'residual_return_20d', 'residual_volume', 'idiosyncratic_volatility',
    'residual_skewness', 'relative_strength_vs_peers',
]
NG110_STOCK_FEATURES: List[str] = [f for f in _NG101_STOCK_FEATURES if f not in _NG110_PRUNED]
# Bug #4 fix: add corrected features + P2 new alpha factors
NG110_BUGFIX_FEATURES: List[str] = [
    'profit_margin_ratio',               # Bug #4: renamed from revenue_growth (was margin, not growth)
    'revenue_growth',                     # Bug #4: now real or_yoy (营收同比增长率)
]
NG110_P2_FEATURES: List[str] = [
    'cs_rank_pb', 'cs_rank_dv',          # industry-within valuation rank
    'peg_proxy', 'pb_roe_ratio',         # value efficiency composites
]
NG110_STOCK_FEATURES = NG110_STOCK_FEATURES + NG110_BUGFIX_FEATURES + NG110_P2_FEATURES
# ng1.1.0 = 59 base - 7 pruned + 2 bugfix + 4 P2 = 58 stock + 10 market = 68 total
NG110_ALL_FEATURES: List[str] = NG110_STOCK_FEATURES + MARKET_FEATURE_NAMES
NG110_VERSION = 'ng1.1.0'

# ---------------------------------------------------------------------------
# ng1.3.0: Multi-task 双头 (excess + downside) + β composite
# Spec: docs/superpowers/specs/2026-04-18-ng130-multitask-design.md
# ---------------------------------------------------------------------------
# Tier A (7): 4 downside (from ng1.0.4 smoothing subset) + 3 AMV (from ng1.0.6)
# Tier B (3): moneyflow factors (require moneyflow_daily.code_6 fix)
# Tier C (0 accepted): EMT 4-gate rejected all 4 candidates (2026-04-18)
#   → ng1.3.0 final: 66 ng1.0.1 base + 4 Tier A downside + 3 Tier A AMV + 3 Tier B mf = 76 features

NG130_TIER_A_DOWNSIDE: List[str] = [
    'current_drawdown',       # 60d 距高点距离 (from ng1.0.4 smoothing)
    'downside_vol_20d',       # 下行波动率
    'recovery_speed_20d',     # 20d 反弹速率
    'gap_risk_20d',           # 跳空风险
]

NG130_TIER_A_AMV: List[str] = [
    'amv_var1',               # 0AMV 活跃筹码连续值 (from ng1.0.6)
    'amv_macd',               # 0AMV MACD 强度
    'amv_regime_days',        # 当前 regime 持续天数
]

# Tier B: moneyflow factors (defined in ml_models/ng/ng130_moneyflow_factors.py)
NG130_TIER_B_MF: List[str] = [
    'elg_net_inflow_20d_z',   # 特大单 20d 净流入 CS z-score
    'mf_main_ratio_20d',      # 主力资金占比 (lg + elg) / total
    'mf_concentration_20d',   # 资金流波动 (std / mean_abs)
]

# ng1.3.0 feature assembly:
#   stock features (63) = ng1.0.1 base (56) + 4 downside + 3 moneyflow
#   market features (13) = ng1.0.1 base (10) + 3 AMV
#   cond_ix (0) = not used
NG130_STOCK_FEATURES: List[str] = (
    STOCK_FEATURE_NAMES + NG130_TIER_A_DOWNSIDE + NG130_TIER_B_MF
)
NG130_MARKET_FEATURES: List[str] = MARKET_FEATURE_NAMES + NG130_TIER_A_AMV
NG130_ALL_FEATURES: List[str] = NG130_STOCK_FEATURES + NG130_MARKET_FEATURES
NG130_VERSION = 'ng1.3.0'

# ---------------------------------------------------------------------------
# ng1.4.0: ng1.0.1 稳定底座 + Tier A (downside + AMV), 无 dual-head, 无 mf
# 设计: docs/ng140_plan.md
# ---------------------------------------------------------------------------
# 保留 ng1.0.1: 53 stock + 10 market + industry excess labels + 6-algo MSE + 3-seed
# 新增 Tier A (ng1.3.x 验证有用): 4 downside stock + 3 AMV market
# 丢弃 (ng1.3.x 验证无用): mf (top-30 零命中), dual-head, β composite
# 丢弃 (ng1.1.0 已识别重复): volume_contraction, sw_index_return_5d,
#     industry_relative_strength (canonical 版本 volume_ratio_5d / industry_return_5d /
#     residual_return_20d 保留, ng130 cache 里就不含这 3 个重复特征)
_NG140_PRUNED_DUPES = frozenset({
    'volume_contraction', 'sw_index_return_5d', 'industry_relative_strength',
})
NG140_STOCK_FEATURES: List[str] = (
    [f for f in STOCK_FEATURE_NAMES if f not in _NG140_PRUNED_DUPES] + NG130_TIER_A_DOWNSIDE
)
NG140_MARKET_FEATURES: List[str] = MARKET_FEATURE_NAMES + NG130_TIER_A_AMV
NG140_ALL_FEATURES: List[str] = NG140_STOCK_FEATURES + NG140_MARKET_FEATURES
NG140_VERSION = 'ng1.4.0'

# ng1.4.1: ng1.4.0 minus 4 Tier A downside stock features (keeps 3 AMV market)
# Ablation: is the AMV signal alone (without per-stock downside) enough?
NG141_STOCK_FEATURES: List[str] = [f for f in STOCK_FEATURE_NAMES if f not in _NG140_PRUNED_DUPES]
NG141_MARKET_FEATURES: List[str] = MARKET_FEATURE_NAMES + NG130_TIER_A_AMV
NG141_ALL_FEATURES: List[str] = NG141_STOCK_FEATURES + NG141_MARKET_FEATURES
NG141_VERSION = 'ng1.4.1'

# ng1.4.2: ng1.4.0 minus 3 AMV market features (keeps 4 downside stock)
# Ablation: is the downside signal alone (without market regime) enough?
NG142_STOCK_FEATURES: List[str] = NG140_STOCK_FEATURES
NG142_MARKET_FEATURES: List[str] = list(MARKET_FEATURE_NAMES)
NG142_ALL_FEATURES: List[str] = NG142_STOCK_FEATURES + NG142_MARKET_FEATURES
NG142_VERSION = 'ng1.4.2'

# ---------------------------------------------------------------------------
# ng1.5.0: ng1.4.0 底座 + 5 Tier B regime-refined features
# 设计: docs/superpowers/specs/2026-04-20-ng150-regime-refined-design.md
# ---------------------------------------------------------------------------
# Stock Tier B (4): industry_regime_agreement, recent_maxdd_60d,
#                     volatility_skew_20d, upside_capture_60d
# Market Tier B (1): amv_regime_bull_prob
NG150_STOCK_TIER_B: List[str] = [
    'industry_regime_agreement',
    'recent_maxdd_60d',
    'volatility_skew_20d',
    'upside_capture_60d',
]
NG150_MARKET_TIER_B: List[str] = [
    'amv_regime_bull_prob',
]
NG150_STOCK_FEATURES: List[str] = NG140_STOCK_FEATURES + NG150_STOCK_TIER_B
NG150_MARKET_FEATURES: List[str] = NG140_MARKET_FEATURES + NG150_MARKET_TIER_B
NG150_ALL_FEATURES: List[str] = NG150_STOCK_FEATURES + NG150_MARKET_FEATURES
NG150_VERSION = 'ng1.5.0'
# 53 stock (ng101 - 3 dupes) + 4 downside + 4 regime-stock
# + 10 market + 3 AMV + 1 regime-market = 75. Spec §2.1 listed 78 assuming
# ng1.0.1 was 56 stock; the 3 ng1.4.0-pruned dupes explain the delta.
assert len(NG150_ALL_FEATURES) == 75, f"ng150 feature count drift: {len(NG150_ALL_FEATURES)} != 75"

# ---------------------------------------------------------------------------
# ng1.6.1: ng1.0.1 底座 + cross-sectional factor-residual labels (F2)
# 设计: docs/ng_next_iteration_plan.md#F2
# ---------------------------------------------------------------------------
# 目标: 复刻 ng1.0.6 β_UMD≈0 跨 regime 鲁棒性, 通过 label 端残差化 (不动架构).
# 特征集同 ng1.0.1 (66 features). 只改 label: 每日 cross-sectional 对
# (cs_rank_market_cap, cs_rank_pb, cs_rank_return_20d, industry_return_20d)
# 回归 label_Nd, 用残差作为 "pure alpha" label.
# 这是 pre-1.4.x 路线的分支 — 不继承 ng1.3.x/ng1.4.x 的 cache schema.
NG161_STOCK_FEATURES: List[str] = list(STOCK_FEATURE_NAMES)
NG161_MARKET_FEATURES: List[str] = list(MARKET_FEATURE_NAMES)
NG161_ALL_FEATURES: List[str] = NG161_STOCK_FEATURES + NG161_MARKET_FEATURES
NG161_VERSION = 'ng1.6.1'
# Factor exposure proxies (subset of STOCK_FEATURE_NAMES). Used as X in
# cross-sectional regression to residualize labels per trade_date.
NG161_FACTOR_PROXIES = [
    'log_adv_20d',            # SMB proxy (size via liquidity; no raw market_cap in base)
    'pb',                     # HML proxy (value)
    'cs_rank_return_20d',     # UMD proxy (momentum)
    'industry_return_20d',    # MKT proxy (within-industry systematic)
]

# ---------------------------------------------------------------------------
# ng1.7.0: ng1.0.1 底座 (66) + 4 alt-alpha 因子 (70 total)
# 设计: memory/ng17_candidate_factors.md
# ---------------------------------------------------------------------------
# 4 factors passed 2026-04-23 fast-check (IC/ICIR gate |IC|>=0.015 AND |ICIR|>=0.25):
#   altdata_rzmre_5d_ratio:   IC=-0.1024 ICIR=-0.72 (散户融资追高反向)
#   altdata_rzye_chg_10d:     IC=-0.0462 ICIR=-0.92 (融资加速反向)
#   altdata_lhb_inst_net_5d:  IC=+0.0391 ICIR=+0.26 (机构 smart money)
#   altdata_lhb_count_20d:    IC=-0.0724 ICIR=-0.83 (过度炒作反向)
#
# Reuses ng101_feature_cache (ng1.0.1 base) — the 4 altdata factors are joined
# from altdata_factor_cache at data-load time (trainer) and inference-time (scorer).
# Cross-factor Spearman corr ≤ 0.32, language-independent信号, 3 反向 + 1 正 alpha.
NG170_ALTDATA_FEATURES: List[str] = [
    'altdata_rzmre_5d_ratio',
    'altdata_rzye_chg_10d',
    'altdata_lhb_inst_net_5d',
    'altdata_lhb_count_20d',
]
NG170_STOCK_FEATURES: List[str] = list(STOCK_FEATURE_NAMES) + NG170_ALTDATA_FEATURES
NG170_MARKET_FEATURES: List[str] = list(MARKET_FEATURE_NAMES)
NG170_ALL_FEATURES: List[str] = NG170_STOCK_FEATURES + NG170_MARKET_FEATURES
NG170_VERSION = 'ng1.7.0'
assert len(NG170_ALL_FEATURES) == 70, f"ng170 feature count drift: {len(NG170_ALL_FEATURES)} != 70"

# ng1.0.9: Persistent features (10-day rank autocorrelation >= 0.5)
# 22 features that produce stable cross-sectional rankings over 10 days
PERSISTENT_STOCK_FEATURES: List[str] = [
    # Fundamentals (autocorr 0.94-1.00)
    'debt_to_assets', 'current_ratio', 'free_float_ratio', 'pb', 'dv_ratio',
    'pe_ttm', 'net_profit_margin', 'profit_margin_ratio', 'roe_ttm', 'ocf_quality',
    # Liquidity (0.80-0.97)
    'log_adv_20d', 'log_amount_ma5', 'turnover_rate',
    # Volatility (0.74-0.79)
    'idiosyncratic_volatility', 'cs_rank_volatility',
    # Valuation rank (0.60-0.96)
    'cs_rank_pe', 'pe_percentile_60d', 'cs_rank_turnover', 'cs_rank_rsi',
    # Industry structure (0.90-0.95)
    'industry_hhi', 'residual_volume',
]


# ---------------------------------------------------------------------------
# NGTrainer
# ---------------------------------------------------------------------------

class NGTrainer(V485Trainer):
    """NG v1.0.3 Trainer — 66 features (3 flipping factors removed), delegates training to V485."""

    VERSION_TAG = NG_VERSION

    # Default target weights — will be overridden by ICIR adaptive in walk_forward_train
    TARGET_WEIGHTS = {
        'label_3d': 0.10,
        'label_5d': 0.20,
        'label_10d': 0.35,
        'label_15d': 0.35,
    }

    def __init__(self, db_path: str = DB_PATH, version: str = None, head: str = 'excess'):
        super().__init__(db_path)
        self._ng_version = version or NG_VERSION
        self.schema_version = get_schema_version(self._ng_version)
        self._train_start_ts = time.time()  # P0.3 Check 9: pkl duration metadata
        self.target_weights = dict(self.TARGET_WEIGHTS)
        self._turbo_skip_etf = True
        self._head = head  # 'excess' (default) or 'downside' (ng1.3.x dual-head training)
        self.cache_table = get_table_name(self._ng_version)
        version_feature_table = [
            ('ng1.7.0', NG170_ALL_FEATURES, NG170_STOCK_FEATURES, NG170_MARKET_FEATURES, []),
            ('ng1.6.1', NG161_ALL_FEATURES, NG161_STOCK_FEATURES, NG161_MARKET_FEATURES, []),
            ('ng1.5.0', NG150_ALL_FEATURES, NG150_STOCK_FEATURES, NG150_MARKET_FEATURES, []),
            ('ng1.4.2', NG142_ALL_FEATURES, NG142_STOCK_FEATURES, NG142_MARKET_FEATURES, []),
            ('ng1.4.1', NG141_ALL_FEATURES, NG141_STOCK_FEATURES, NG141_MARKET_FEATURES, []),
            ('ng1.4.0', NG140_ALL_FEATURES, NG140_STOCK_FEATURES, NG140_MARKET_FEATURES, []),
            ('ng1.3.0', NG130_ALL_FEATURES, NG130_STOCK_FEATURES, NG130_MARKET_FEATURES, []),
            ('ng1.1.0', NG110_ALL_FEATURES, NG110_STOCK_FEATURES, MARKET_FEATURE_NAMES, []),
            ('ng1.0.7', NG107_ALL_FEATURES, STOCK_FEATURE_NAMES,  NG107_MARKET_FEATURES, CONDITIONAL_IX_FEATURE_NAMES),
            ('ng1.0.4', NG104_ALL_FEATURES, NG104_STOCK_FEATURES, MARKET_FEATURE_NAMES,  []),
            ('ng0.0.0', ALL_FEATURE_NAMES,  STOCK_FEATURE_NAMES,  MARKET_FEATURE_NAMES,  []),
        ]
        for min_ver, all_f, stock_f, macro_f, cond_f in version_feature_table:
            if version_ge(self._ng_version, min_ver):
                self.feature_names = list(all_f)
                self.stock_feature_cols = list(stock_f)
                self.macro_feature_cols = list(macro_f)
                self._cond_ix_cols = list(cond_f)
                break
        # ng1.1.0+: disable V475 PRUNE_FEATURES — NG manages its own feature set
        if version_ge(self._ng_version, 'ng1.1.0'):
            self.PRUNE_FEATURES = []
        # Stub market_calculator for V475 model_data serialization
        class _StubMC:
            class market_features:
                columns = ['date'] + list(MARKET_FEATURE_NAMES)
        self.market_calculator = _StubMC()

    def _compute_cache_key(self, start_date, end_date):
        """Override: include NG version + head (ng1.3.x) + ng1.2.3 lambda_downside
        + feature-set hash in cache key to invalidate on version/head/label/feature changes.

        CRITICAL: ng1.3.x excess vs downside heads MUST have different cache keys,
        otherwise the second head run will reuse the first head's cached labels
        (seen in first training batch: excess and downside models bit-identical).

        Feature-set hash guards against stale caches written by an older feature
        pipeline (root cause of ng1.3.0 pkls carrying 18-col NG107 macro instead
        of 13-col NG130 macro — stale cache predated ext_market exclusion in e37eb263).
        """
        import hashlib
        db_mtime = os.path.getmtime(self.db_path) if os.path.exists(self.db_path) else 0
        lam_suffix = ''
        if self.schema_version == 'ng1.2.3':
            lam = getattr(self, '_lambda_downside', 0.3)
            lam_suffix = f"_lam{lam:.4f}"
        # head suffix when training with non-default head (current: ng1.3.x downside).
        # Future versions using --head downside get the same cache invalidation automatically.
        head_suffix = f"_{self._head}" if self._head != 'excess' else ''
        feat_hash = hashlib.md5(
            ('|'.join(self.stock_feature_cols) + '#' + '|'.join(self.macro_feature_cols)).encode()
        ).hexdigest()[:8]
        key_str = (f"{self.__class__.__name__}_{self._ng_version}_{start_date}_{end_date}"
                   f"_{db_mtime:.0f}{lam_suffix}{head_suffix}_{feat_hash}")
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def _residualize_labels_cross_section(
        self, df: pd.DataFrame, proxy_cols: List[str]
    ) -> None:
        """Replace label_Nd columns in-place with per-date cross-sectional
        residuals vs factor exposure proxies.

        For each trade_date:
            y_i = label_Nd_i            (stock i's industry-excess return)
            X_i = [proxy_cols values]   (size/value/momentum/industry proxies)
            fit OLS y ~ X; label_Nd_i  <- y_i - predicted_i

        Residual = "pure alpha" after removing systematic factor exposure.
        Only labels change; features are untouched.
        """
        from numpy.linalg import lstsq
        available = [c for c in proxy_cols if c in df.columns]
        if len(available) < 2:
            logger.warning(
                f"  ng1.6.x: only {len(available)} factor proxies in df, "
                f"skipping label residualization (need ≥2)"
            )
            return
        label_cols = [f'label_{h}' for h in ('3d', '5d', '10d', '15d') if f'label_{h}' in df.columns]
        logger.info(
            f"  ng1.6.x F2: cross-sectional factor residualization — "
            f"proxies={available}, labels={label_cols}"
        )
        date_groups = df.groupby('trade_date', sort=False).indices
        n_residualized = 0
        for date, idx in date_groups.items():
            if len(idx) < 30:
                continue
            X = df.loc[idx, available].values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            X_aug = np.column_stack([X, np.ones(len(idx))])
            for lab in label_cols:
                y = df.loc[idx, lab].values.astype(float)
                mask = ~np.isnan(y)
                if mask.sum() < 30:
                    continue
                try:
                    beta, *_ = lstsq(X_aug[mask], y[mask], rcond=None)
                    pred = X_aug @ beta
                    df.loc[idx, lab] = y - pred
                except Exception as e:
                    logger.debug(f'residualize failed {date}/{lab}: {e}')
            n_residualized += 1
        logger.info(f"  ng1.6.x F2: residualized {n_residualized} trade dates")

    def _warn_on_feature_count_mismatch(self, model_data: dict) -> None:
        """Probe first probeable booster; warn if its n_features != len(self.feature_names)."""
        try:
            models = model_data.get('models', {})
            for tk in ('3d', '5d', '10d', '15d'):
                boosters = (models.get(tk, {}) or {}).get('models', {}) or {}
                for algo, b in boosters.items():
                    if hasattr(b, 'num_feature'):
                        nf = b.num_feature()
                    else:
                        nfi = getattr(b, 'n_features_in_', None)
                        nf = int(nfi) if nfi else None
                    if nf is None:
                        continue
                    if nf != len(self.feature_names):
                        logger.warning(
                            f"  ⚠️ pkl feature_names ({len(self.feature_names)}) != "
                            f"booster {tk}/{algo} num_feature ({nf}). "
                            "Inference may misalign."
                        )
                    return
        except Exception as e:
            logger.debug(f"  feature_names sanity skipped: {e}")

    # ------------------------------------------------------------------
    # Feature name accessors
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_names() -> List[str]:
        return list(ALL_FEATURE_NAMES)

    @staticmethod
    def get_stock_feature_names() -> List[str]:
        return list(STOCK_FEATURE_NAMES)

    @staticmethod
    def get_market_feature_names() -> List[str]:
        return list(MARKET_FEATURE_NAMES)

    # 10-day rank autocorrelation lookup (pre-computed from ng1.0.1 data 2020-2025)
    # Note: 'revenue_growth' was measured on old profit_to_gr (margin); real or_yoy TBD
    _FEATURE_AUTOCORR = {
        'debt_to_assets': 0.998, 'current_ratio': 0.997, 'free_float_ratio': 0.995,
        'pb': 0.994, 'dv_ratio': 0.991, 'pe_ttm': 0.988, 'net_profit_margin': 0.986,
        'profit_margin_ratio': 0.986, 'log_adv_20d': 0.970, 'roe_ttm': 0.970,
        'cs_rank_pe': 0.955, 'industry_hhi': 0.951,
        'ocf_quality': 0.940, 'log_amount_ma5': 0.913, 'residual_volume': 0.904,
        'turnover_rate': 0.797, 'idiosyncratic_volatility': 0.790,
        'cs_rank_turnover': 0.742, 'cs_rank_volatility': 0.735,
        'cs_rank_rsi': 0.623, 'pe_percentile_60d': 0.597,
        'up_volume_ratio': 0.480,
        'residual_return_20d': 0.437, 'relative_strength_vs_peers': 0.437,
        'industry_return_20d': 0.436, 'volume_cv': 0.432,
        'cs_rank_return_20d': 0.422, 'volume_price_corr': 0.416,
        'residual_skewness': 0.400, 'cs_rank_new_high': 0.351,
        'trend_strength_20d': 0.321, 'rsi_14': 0.281, 'obv_trend': 0.270,
        'pullback_to_ma20': 0.249, 'industry_breadth': 0.051,
        'sector_breadth_vs_market': 0.051, 'cs_rank_pullback': 0.049,
        'intraday_recovery': 0.040,
        'industry_return_5d': 0.023, 'industry_rank_return_5d': 0.023,
        'lower_shadow_ratio': 0.013, 'volume_breakout': 0.011,
        'adx_proxy': 0.000, 'cs_rank_return_5d': -0.004,
        'industry_volume_change': -0.014, 'sector_volume_vs_market': -0.014,
        'pullback_to_ma10': -0.041, 'cs_rank_volume_surge': -0.053,
        'volume_ratio_5d': -0.058,
        'kdj_j_value': -0.116,
    }

    def _get_active_stock_features(self) -> List[str]:
        """Build active stock feature list based on CLI switches."""
        # ng1.0.9: use only persistent features (22 slow-changing features)
        if getattr(self, '_persistent_features', False):
            features = [f for f in PERSISTENT_STOCK_FEATURES if f in self.stock_feature_cols
                        or f in STOCK_FEATURE_NAMES]
            return features

        # ng1.0.9: filter by min autocorrelation threshold
        min_ac = getattr(self, '_min_autocorr', 0.0)
        if min_ac > 0:
            base = list(self.stock_feature_cols)
            features = [f for f in base if self._FEATURE_AUTOCORR.get(f, 0.0) >= min_ac]
            dropped = len(base) - len(features)
            if dropped > 0:
                logger.info(f"  Autocorr filter (>={min_ac}): {len(features)} kept, {dropped} dropped")
            return features

        features = list(self.stock_feature_cols)
        if getattr(self, '_enable_moneyflow', False):
            features += MONEYFLOW_FEATURE_NAMES
        if getattr(self, '_enable_interaction', False):
            selected = getattr(self, '_selected_ix', None)
            if selected:
                features += selected
            else:
                features += INTERACTION_FEATURE_NAMES
        # Conditional interaction features (set per-version in __init__)
        if self._cond_ix_cols:
            selected_cx = getattr(self, '_selected_cx', None)
            if selected_cx:
                features += selected_cx
            else:
                features += self._cond_ix_cols
        return features

    # ------------------------------------------------------------------
    # P1: Ensemble weight floor + shrinkage (ng1.1.0+)
    # ------------------------------------------------------------------

    def calculate_ensemble_weights(self, predictions_val: dict, y_val):
        """Override: add weight floor and equal-weight shrinkage for ng1.1.0+."""
        weights, mean_ics = super().calculate_ensemble_weights(predictions_val, y_val)

        if not version_ge(self._ng_version, 'ng1.1.0'):
            return weights, mean_ics

        n = len(weights)
        if n <= 1:
            return weights, mean_ics

        # Shrinkage: blend IC-weights with equal weights (70% IC + 30% equal)
        shrinkage = 0.3
        equal_w = 1.0 / n
        shrunk = {k: (1 - shrinkage) * v + shrinkage * equal_w for k, v in weights.items()}

        # Weight floor: no algorithm below 1/N * 0.3 (roughly 5% for 6 algos)
        floor = equal_w * 0.3
        clipped = {k: max(v, floor) for k, v in shrunk.items()}
        total = sum(clipped.values())
        final = {k: v / total for k, v in clipped.items()}

        logger.info(f"  P1 shrinkage+floor: {', '.join(f'{k}={v:.3f}' for k, v in final.items())}")
        return final, mean_ics

    # ------------------------------------------------------------------
    # ICIR Adaptive Composite Weights
    # ------------------------------------------------------------------

    def _compute_icir_adaptive_weights(self, history: dict) -> dict:
        """
        Compute composite weights proportional to OOS ICIR for each target.
        Falls back to default weights if ICIR data unavailable.

        Reads from history['summary']['walk_forward_summary'] (V475+ format).

        Returns dict like {'label_3d': 0.10, 'label_5d': 0.20, ...}
        """
        try:
            # V475+ stores WF metrics in summary.walk_forward_summary
            wf_summary = history.get('summary', {}).get('walk_forward_summary', {})

            target_ics: dict = {}
            for target_key, wf_key in [('label_3d', '3d'), ('label_5d', '5d'),
                                        ('label_10d', '10d'), ('label_15d', '15d')]:
                tw = wf_summary.get(wf_key, {})
                icir = tw.get('mean_icir')
                if icir is not None and not np.isnan(icir):
                    target_ics[target_key] = max(float(icir), 0.0)
                else:
                    target_ics[target_key] = 0.0

            total = sum(target_ics.values())
            if total < 1e-8:
                logger.warning("No WF ICIR data in history, using default weights")
                return dict(self.TARGET_WEIGHTS)

            weights = {k: v / total for k, v in target_ics.items()}
            logger.info(f"ICIR adaptive weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
            logger.info(f"  Raw ICIR: {', '.join(f'{k}={v:.3f}' for k, v in target_ics.items())}")
            return weights

        except Exception as e:
            logger.warning(f"ICIR adaptive weights failed: {e}, using defaults")
            return dict(self.TARGET_WEIGHTS)

    # ------------------------------------------------------------------
    # Factor Quality Analysis
    # ------------------------------------------------------------------

    # Group mapping derived from module-level feature name constants
    _FEATURE_GROUP_MAP: dict = {}
    for _f in STOCK_FEATURE_NAMES[:4]:
        _FEATURE_GROUP_MAP[_f] = 'trend'
    # v1.0.0 had pullback_from_high in trend; keep mapping for older models
    _FEATURE_GROUP_MAP['pullback_from_high'] = 'trend'
    for _f in STOCK_FEATURE_NAMES[4:10]:
        _FEATURE_GROUP_MAP[_f] = 'pullback'
    for _f in STOCK_FEATURE_NAMES[10:18]:
        _FEATURE_GROUP_MAP[_f] = 'volume'
    for _f in STOCK_FEATURE_NAMES[18:31]:
        _FEATURE_GROUP_MAP[_f] = 'fundamental'
    # v1.0.0 had log_market_cap; keep mapping for older models
    _FEATURE_GROUP_MAP['log_market_cap'] = 'fundamental'
    _FEATURE_GROUP_MAP['turnover_rate'] = 'fundamental'
    for _f in STOCK_FEATURE_NAMES[31:42]:
        _FEATURE_GROUP_MAP[_f] = 'industry'
    for _f in STOCK_FEATURE_NAMES[42:51]:
        _FEATURE_GROUP_MAP[_f] = 'cs_rank'
    # v1.0.0 had cs_rank_market_cap; keep mapping for older models
    _FEATURE_GROUP_MAP['cs_rank_market_cap'] = 'cs_rank'
    for _f in STOCK_FEATURE_NAMES[51:]:
        _FEATURE_GROUP_MAP[_f] = 'residual'
    for _f in MARKET_FEATURE_NAMES:
        _FEATURE_GROUP_MAP[_f] = 'market'
    for _f in SMOOTHING_FEATURE_NAMES:
        _FEATURE_GROUP_MAP[_f] = 'smoothing'
    for _f in MONEYFLOW_FEATURE_NAMES:
        _FEATURE_GROUP_MAP[_f] = 'moneyflow'
    for _f in INTERACTION_FEATURE_NAMES + CONDITIONAL_IX_FEATURE_NAMES:
        _FEATURE_GROUP_MAP[_f] = 'interaction'
    for _f in EXTENDED_MARKET_FEATURE_NAMES:
        _FEATURE_GROUP_MAP[_f] = 'market'
    del _f  # clean up loop variable from class namespace

    @staticmethod
    def _detect_trained_feature_count(models_dict: dict, fallback: int) -> int:
        """Detect actual feature count from first available model importance."""
        for target_data in models_dict.values():
            if not isinstance(target_data, dict):
                continue
            for m in target_data.get('models', {}).values():
                tp = type(m).__name__
                try:
                    if tp == 'Booster' and hasattr(m, 'feature_importance'):
                        return len(m.feature_importance(importance_type='gain'))
                    elif hasattr(m, 'feature_importances_'):
                        return len(m.feature_importances_)
                except Exception:
                    continue
        return fallback

    def _log_factor_quality(self, model_data: dict, adaptive_weights: dict,
                            history: dict, model_dir: Path, timestamp: str):
        """Extract and log per-factor weighted importance after training.

        Saves factor_quality_{timestamp}.json alongside the model, and prints
        a summary to the training log.  The JSON is the canonical source for
        updating docs/wiki/models/ng-factor-quality.md.
        """
        feature_names = model_data.get('feature_names', [])
        n_feat = len(feature_names)
        if n_feat == 0:
            return

        # V485 pipeline trains on a pruned feature subset — detect actual
        # training features from the first model's importance length
        models_dict = model_data.get('models', {})
        trained_feat_count = self._detect_trained_feature_count(models_dict, n_feat)
        if trained_feat_count < n_feat:
            # Models were trained on V485 pruned features, not the full NG set.
            # Use only the first trained_feat_count feature names (they match
            # the V485 training order).
            feature_names = feature_names[:trained_feat_count]
            n_feat = trained_feat_count
            logger.info(f"  Factor quality: using {n_feat} features (V485 pruned)")

        all_imp = {feat: 0.0 for feat in feature_names}

        for target, target_data in models_dict.items():
            if not isinstance(target_data, dict):
                continue
            inner = target_data.get('models', {})
            algo_weights = target_data.get('weights', {})
            target_w = adaptive_weights.get(f'label_{target}',
                                            adaptive_weights.get(target, 0.25))

            for algo, m in inner.items():
                tp = type(m).__name__
                algo_w = algo_weights.get(algo, 0.2)
                imp_arr = np.zeros(n_feat)

                try:
                    if tp == 'Booster' and 'lgb' in algo:
                        raw = m.feature_importance(importance_type='gain')
                        s = raw.sum()
                        if s > 0:
                            imp_arr = raw / s
                    elif algo == 'xgb' or (tp == 'Booster' and 'xgb' in algo):
                        score = m.get_score(importance_type='gain')
                        raw = np.zeros(n_feat)
                        for feat_key, val in score.items():
                            idx = int(feat_key.replace('f', ''))
                            if idx < n_feat:
                                raw[idx] = val
                        s = raw.sum()
                        if s > 0:
                            imp_arr = raw / s
                    elif 'CatBoost' in tp:
                        if hasattr(m, 'get_feature_importance'):
                            raw = np.array(m.get_feature_importance(), dtype=float)
                        elif hasattr(m, 'feature_importances_'):
                            raw = m.feature_importances_
                        else:
                            raw = None
                        if raw is not None:
                            s = raw.sum()
                            if s > 0:
                                imp_arr = raw / s
                    elif hasattr(m, 'feature_importances_'):
                        # RF, HGB, and other sklearn estimators (already normalized)
                        imp_arr = m.feature_importances_
                except Exception as e:
                    logger.debug(f"  Factor quality: {target}/{algo} importance extraction failed: {e}")
                    continue

                imp_len = len(imp_arr) if hasattr(imp_arr, '__len__') else 0
                if imp_len == 0:
                    continue
                for i in range(min(n_feat, imp_len)):
                    all_imp[feature_names[i]] += float(imp_arr[i]) * algo_w * target_w

        total_imp = sum(all_imp.values())
        if total_imp == 0:
            logger.warning("Factor quality: all importances are zero, skipping")
            return

        # Sort and compute ranks
        sorted_factors = sorted(all_imp.items(), key=lambda x: x[1], reverse=True)
        factor_list = []
        cum = 0.0
        for rank, (fname, imp) in enumerate(sorted_factors, 1):
            pct = imp / total_imp * 100
            cum += pct
            factor_list.append({
                'rank': rank, 'feature': fname,
                'weighted_importance': round(imp, 8),
                'pct': round(pct, 3), 'cum_pct': round(cum, 3),
            })

        # Group summary using class-level mapping
        group_pcts = {}
        for feat, imp in all_imp.items():
            g = self._FEATURE_GROUP_MAP.get(feat, 'other')
            group_pcts[g] = group_pcts.get(g, 0.0) + imp / total_imp * 100

        # WF ICIR from history
        wf = history.get('summary', {}).get('walk_forward_summary', {})
        icir_summary = {}
        for t in ['3d', '5d', '10d', '15d']:
            wf_entry = wf.get(t, {})
            if 'mean_icir' in wf_entry:
                icir_summary[t] = {'ic': round(wf_entry['mean_ic'], 4),
                                   'icir': round(wf_entry['mean_icir'], 4)}

        near_zero = [entry for entry in factor_list if entry['pct'] < 0.1]

        quality_data = {
            'version': self._ng_version,
            'timestamp': timestamp,
            'n_features': n_feat,
            'wf_icir': icir_summary,
            'adaptive_weights': {k: round(v, 4) for k, v in adaptive_weights.items()},
            'group_pcts': {k: round(v, 1) for k, v in sorted(group_pcts.items(), key=lambda x: -x[1])},
            'top_10': factor_list[:10],
            'near_zero': near_zero,
            'all_factors': factor_list,
        }

        # Save JSON
        out_path = model_dir / f'factor_quality_{timestamp}.json'
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(quality_data, fh, indent=2, ensure_ascii=False)

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Factor Quality Report — NG {self._ng_version}")
        logger.info(f"{'='*60}")
        logger.info(f"  Features: {n_feat}")
        for t, v in icir_summary.items():
            logger.info(f"  {t} ICIR: {v['icir']}")
        logger.info(f"\n  Group weights:")
        for g, pct in sorted(group_pcts.items(), key=lambda x: -x[1]):
            logger.info(f"    {g:15s}: {pct:5.1f}%")
        logger.info(f"\n  Top 10 factors:")
        for entry in factor_list[:10]:
            logger.info(f"    {entry['rank']:2d}. {entry['feature']:35s} {entry['pct']:5.1f}% (cum {entry['cum_pct']:5.1f}%)")
        if near_zero:
            logger.info(f"\n  Near-zero factors (<0.1%): {len(near_zero)}")
            for entry in near_zero:
                logger.info(f"    {entry['feature']:35s} {entry['pct']:.3f}%")
        logger.info(f"\n  Saved: {out_path}")
        logger.info(f"  → Update docs/wiki/models/ng-factor-quality.md with this data")

    # ------------------------------------------------------------------
    # WF Summary Generation
    # ------------------------------------------------------------------

    def _generate_wf_summary(self, history: dict, model_dir: Path) -> dict:
        """
        Generate wf_summary.json for L4 scoring (WFER + OOS monthly IC).
        Returns the summary dict.
        """
        summary = {
            'version': self._ng_version,
            'generated_at': datetime.now().isoformat(),
            'wf_windows': [],
            'aggregate': {},
        }

        try:
            # V475+ stores WF metrics in summary.walk_forward_summary
            wf_data = history.get('summary', {}).get('walk_forward_summary', {})
            if not wf_data:
                logger.warning("No WF summary for wf_summary generation")
                return summary

            # Build per-window summaries from OOS monthly ICs
            all_oos_ics = {'label_3d': [], 'label_5d': [], 'label_10d': [], 'label_15d': []}
            n_windows = wf_data.get('3d', {}).get('n_windows', 0)
            window_summaries = []

            for i in range(n_windows):
                ws = {'window_id': i, 'metrics': {}}
                for target_key, wf_key in [('label_3d', '3d'), ('label_5d', '5d'),
                                            ('label_10d', '10d'), ('label_15d', '15d')]:
                    tw = wf_data.get(wf_key, {})
                    oos_icirs = tw.get('oos_icir_per_window', [])
                    oos_ics = tw.get('oos_monthly_ics', [])
                    if i < len(oos_icirs):
                        ws['metrics'][f'{target_key}_icir'] = float(oos_icirs[i])
                    if i < len(oos_ics):
                        monthly = oos_ics[i]
                        mean_ic = float(np.mean(monthly)) if monthly else 0.0
                        ws['metrics'][f'{target_key}_ic'] = mean_ic
                        all_oos_ics[target_key].append(mean_ic)
                window_summaries.append(ws)

            summary['wf_windows'] = window_summaries

            # Aggregate metrics from pre-computed walk_forward_summary values
            for target_key, wf_key in [('label_3d', '3d'), ('label_5d', '5d'),
                                        ('label_10d', '10d'), ('label_15d', '15d')]:
                tw = wf_data.get(wf_key, {})
                if 'mean_ic' in tw:
                    summary['aggregate'][f'{target_key}_mean_ic'] = float(tw['mean_ic'])
                    summary['aggregate'][f'{target_key}_std_ic'] = float(tw.get('std_ic', 0))
                    summary['aggregate'][f'{target_key}_icir'] = float(tw.get('mean_icir', 0))
                # Also compute from per-window OOS ICs if available
                ics = all_oos_ics.get(target_key, [])
                if ics:
                    summary['aggregate'][f'{target_key}_ic_positive_ratio'] = float(np.mean(np.array(ics) > 0))

            # WF Efficiency Ratio (WFER) from IS vs OOS ICIR
            is_icirs = wf_data.get('10d', {}).get('is_icir_per_window', [])
            oos_icirs = wf_data.get('10d', {}).get('oos_icir_per_window', [])
            if is_icirs and oos_icirs:
                wfer = float(np.mean(oos_icirs) / (np.mean(is_icirs) + 1e-8))
                summary['aggregate']['wfer'] = wfer
            else:
                summary['aggregate']['wfer'] = None

            # OOS IC half-life (months until IC decays to half)
            oos_ics_10d = all_oos_ics.get('label_10d', [])
            if len(oos_ics_10d) >= 3:
                initial_ic = oos_ics_10d[0]
                half_target = initial_ic / 2
                half_life_months = None
                for j, ic in enumerate(oos_ics_10d[1:], 1):
                    if ic <= half_target:
                        half_life_months = j * 4
                        break
                summary['aggregate']['oos_ic_half_life_months'] = half_life_months
            else:
                summary['aggregate']['oos_ic_half_life_months'] = None

            summary['n_windows'] = n_windows

            # Save to file
            summary_path = model_dir / 'wf_summary.json'
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"WF summary saved: {summary_path}")

        except Exception as e:
            logger.warning(f"WF summary generation failed: {e}")

        return summary

    # ------------------------------------------------------------------
    # IC Screening for Interaction Features (ng1.0.3)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Market Regime Sample Weighting (ng1.0.3)
    # ------------------------------------------------------------------

    def _compute_regime_weights(self, df):
        """Compute sample weights based on market regime.
        ng1.0.7: enhanced weights (bull=0.7, sideways=1.0, bear=1.5, crisis=2.0)
        Earlier versions: bull=0.8, sideways=1.0, bear=1.2
        """
        mkt_ret = df['market_return_20d'].values if 'market_return_20d' in df.columns else np.zeros(len(df))
        weights = np.ones(len(df))
        if version_ge(self._ng_version, 'ng1.0.7'):
            weights[mkt_ret > 0.05] = 0.7
            weights[(mkt_ret < -0.05) & (mkt_ret >= -0.10)] = 1.5
            weights[mkt_ret < -0.10] = 2.0
            logger.info(f"  ng1.0.7 regime weights: bull={np.sum(weights == 0.7):,}, "
                        f"sideways={np.sum(weights == 1.0):,}, "
                        f"bear={np.sum(weights == 1.5):,}, "
                        f"crisis={np.sum(weights == 2.0):,}")
        else:
            weights[mkt_ret > 0.05] = 0.8
            weights[mkt_ret < -0.05] = 1.2
            logger.info(f"  Regime weights: bull={np.sum(weights == 0.8):,}, sideways={np.sum(weights == 1.0):,}, bear={np.sum(weights == 1.2):,}")
        return weights

    def compute_sample_weights(self, df, y):
        """NG v1.0.3: parent weights + optional regime weighting."""
        weights = super().compute_sample_weights(df, y)
        if getattr(self, '_regime_weight', False):
            regime_w = self._compute_regime_weights(df)
            weights = weights * regime_w
        return weights

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _append_altdata_factors(self, result: pd.DataFrame) -> int:
        """ng1.7.0 only: LEFT-JOIN altdata_factor_cache on (code, trade_date).

        Missing cells become 0.0 (sparse factors, especially LHB — legitimate
        "no signal on this stock/day"). Mutates `result` in place.
        Returns number of factor columns added.
        """
        from ml_models.ng.altdata_factor_cache import FACTOR_COLS
        # Drop any placeholder altdata cols upstream loader may have added as NaN —
        # otherwise the merge would suffix-rename and break column lookup.
        for col in FACTOR_COLS:
            if col in result.columns:
                result.drop(columns=[col], inplace=True)
        if result.empty:
            for col in FACTOR_COLS:
                result[col] = 0.0
            return len(FACTOR_COLS)
        dates = result['trade_date'].unique().tolist()
        in_dates = ",".join("?" * len(dates))
        cols_select = ", ".join(FACTOR_COLS)
        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            factor_df = pd.read_sql(
                f"""
                SELECT code, trade_date, {cols_select}
                FROM altdata_factor_cache
                WHERE trade_date IN ({in_dates})
                """,
                conn,
                params=dates,
            )
        finally:
            conn.close()
        merged = result.merge(factor_df, on=['code', 'trade_date'], how='left')
        for col in FACTOR_COLS:
            result[col] = merged[col].fillna(0.0).values
        return len(FACTOR_COLS)

    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load training data from ng_feature_cache.

        v1.0.3: Labels are now industry excess returns.
        Features include 10 CS rank + 5 residual + 3 sector activity.
        """
        logger.info(f"NG {self._ng_version} Trainer: Loading data from {self.cache_table} ...")

        conn = sqlite3.connect(self.db_path, timeout=30)

        date_filter = ""
        params = []
        if start_date:
            date_filter += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            date_filter += " AND trade_date <= ?"
            params.append(end_date)

        extra_select = ""
        # ng1.2.x branches from ng1.0.1 and skips ng1.0.4/1.0.7 columns; detect
        # separately so the numeric version_ge doesn't pull in non-existent cols.
        is_12 = _is_1_2_branch(self.schema_version)
        is_13 = _is_1_3_branch(self._ng_version)
        is_14 = _is_1_4_branch(self._ng_version)
        is_15 = _is_1_5_branch(self._ng_version)
        # ng1.3.x dual-head, ng1.4.x (reuses ng130 cache), ng1.5.x (own schema
        # same shape as ng130 + 5 features in features_json). Downside labels are
        # harmless/unused for ng1.4/1.5 single-head training.
        if is_13 or is_14 or is_15:
            extra_select = ", downside_3d, downside_5d, downside_10d, downside_15d"
        # ng1.2.3: 4-horizon downside_kd columns; ng1.2.4 has no downside cols
        elif is_12 and _version_in_range(self.schema_version, 'ng1.2.3', 'ng1.2.4'):
            extra_select = ", downside_3d, downside_5d, downside_10d, downside_15d"
        # ng1.2.1: Sharpe-style path-based labels (not inherited by ng1.2.3+)
        elif is_12 and _version_in_range(self.schema_version, 'ng1.2.1', 'ng1.2.3'):
            extra_select = ", vn_label_3d, vn_label_5d, vn_label_10d, vn_label_15d"
            extra_select += ", path_mean_10d, path_std_10d, downside_std_10d"
        elif version_ge(self.schema_version, 'ng1.0.7') and not is_12:
            extra_select = ", ra_label_3d, ra_label_5d, ra_label_10d, ra_label_15d"
            extra_select += ", cond_label_3d, cond_label_5d, cond_label_10d, cond_label_15d"
            extra_select += ", amv_var1, amv_macd, amv_regime_days"
        elif version_ge(self.schema_version, 'ng1.0.4') and not is_12:
            extra_select = ", ra_label_3d, ra_label_5d, ra_label_10d, ra_label_15d"

        # ng1.0.2 linear lineage has a single downside_10d column.
        # ng1.2.3/ng1.3.x/ng1.4.x/ng1.5.x already have downside_10d inside the
        # 4-horizon extra_select block — avoid duplicate column selection.
        downside_col = (
            ", downside_10d"
            if version_ge(self.schema_version, 'ng1.0.2')
               and not is_12 and not is_13 and not is_14 and not is_15
            else ""
        )
        query = f"""
        SELECT code, trade_date, features_json,
               label_3d, label_5d, label_10d, label_15d{downside_col}{extra_select},
               market_return_5d, market_return_20d, market_volatility_20d,
               market_breadth, market_new_high_ratio, northbound_flow_5d,
               market_volume_ratio, market_drawdown, vix_proxy,
               market_momentum_diff
        FROM {self.cache_table}
        WHERE label_5d IS NOT NULL {date_filter}
        ORDER BY trade_date, code
        """

        df_raw = pd.read_sql(query, conn, params=params)
        conn.close()

        n_raw = len(df_raw)
        logger.info(f"  Raw rows from {self.cache_table}: {n_raw:,}")

        if n_raw == 0:
            logger.warning(f"  {self.cache_table} returned 0 rows!")
            return pd.DataFrame()

        # Parse features_json
        logger.info("  Parsing features_json ...")
        parsed_rows = df_raw['features_json'].apply(_json_loads).tolist()
        df_stock_features = pd.DataFrame(parsed_rows)

        active_stock_features = self._get_active_stock_features()

        # Exclude market feature names that may appear in features_json
        # (they are loaded from dedicated SQL columns instead — except ng1.3.x/ng1.4.x/ng1.5.x,
        # which store AMV in features_json only since those caches have no amv_* columns)
        market_cols_to_load = list(MARKET_FEATURE_NAMES)
        if is_13 or is_14 or is_15:
            # ng1.3.x/ng1.4.x/ng1.5.x: 10 base + 3 AMV in features_json.
            # ng1.5.x also has 1 regime feature (amv_regime_bull_prob) in features_json.
            market_cols_to_load += NG130_TIER_A_AMV
            if is_15:
                market_cols_to_load += NG150_MARKET_TIER_B
        elif version_ge(self.schema_version, 'ng1.0.7'):
            market_cols_to_load += EXTENDED_MARKET_FEATURE_NAMES
        market_set = set(market_cols_to_load)
        active_stock_features = [c for c in active_stock_features if c not in market_set]

        for col in active_stock_features:
            if col not in df_stock_features.columns:
                df_stock_features[col] = np.nan

        df_stock_features = df_stock_features[[c for c in active_stock_features if c in df_stock_features.columns]]

        n_extra = len(active_stock_features) - len(STOCK_FEATURE_NAMES)
        if n_extra > 0:
            logger.info(f"  Dynamic features enabled: +{n_extra} "
                         f"(moneyflow={getattr(self, '_enable_moneyflow', False)}, "
                         f"interaction={getattr(self, '_enable_interaction', False)})")

        # Assemble result
        result = pd.DataFrame()
        result['code'] = df_raw['code'].values
        result['trade_date'] = df_raw['trade_date'].values

        for col in active_stock_features:
            if col in df_stock_features.columns:
                result[col] = df_stock_features[col].values

        # ng1.3.x/ng1.4.x/ng1.5.x: AMV features live in features_json (not SQL columns).
        # ng1.5.x additionally has amv_regime_bull_prob in features_json.
        amv_from_json = {}
        if is_13 or is_14 or is_15:
            for col in NG130_TIER_A_AMV:
                amv_from_json[col] = [d.get(col, np.nan) for d in parsed_rows]
            if is_15:
                for col in NG150_MARKET_TIER_B:
                    amv_from_json[col] = [d.get(col, np.nan) for d in parsed_rows]

        for col in market_cols_to_load:
            if col in df_raw.columns:
                result[col] = df_raw[col].values
            elif col in amv_from_json:
                result[col] = amv_from_json[col]
            else:
                result[col] = np.nan

        # ng1.7.0: JOIN altdata_factor_cache to append 4 alt-alpha factors.
        # Missing (code, date) rows get 0.0 (sparse factors like LHB legitimately miss most stocks).
        if self._ng_version == 'ng1.7.0':
            n_added = self._append_altdata_factors(result)
            logger.info(f"  ng1.7.0: appended {n_added} altdata factor columns")

        # Labels (industry excess returns in v1.0.3)
        result['label_3d'] = pd.to_numeric(df_raw['label_3d'], errors='coerce').values
        result['label_5d'] = pd.to_numeric(df_raw['label_5d'], errors='coerce').values
        result['label_10d'] = pd.to_numeric(df_raw['label_10d'], errors='coerce').values
        result['label_15d'] = pd.to_numeric(df_raw.get('label_15d'), errors='coerce').values

        # ng1.0.4: Override labels with risk-adjusted (RA) versions where available
        if version_ge(self._ng_version, 'ng1.0.4') and not version_ge(self._ng_version, 'ng1.0.7'):
            for h in ['3d', '5d', '10d', '15d']:
                ra_col = f'ra_label_{h}'
                if ra_col in df_raw.columns:
                    ra_vals = pd.to_numeric(df_raw[ra_col], errors='coerce')
                    mask = ra_vals.notna()
                    if mask.any():
                        result.loc[mask.values, f'label_{h}'] = ra_vals[mask].values
                        logger.info(f"  Using risk-adjusted label_{h}: {mask.sum():,} values")

        # ng1.2.1: cross-sectional rank of vn_label per trade_date → [0, 1].
        # Robust to cross-date vol regime shifts; what listwise ranking needs.
        if self._ng_version == 'ng1.2.1':
            ranked = 0
            for h in ['3d', '5d', '10d', '15d']:
                vn_col = f'vn_label_{h}'
                if vn_col not in df_raw.columns:
                    continue
                vn_vals = pd.to_numeric(df_raw[vn_col], errors='coerce')
                ranks = vn_vals.groupby(df_raw['trade_date']).rank(pct=True)
                mask = ranks.notna()
                if mask.any():
                    result.loc[mask.values, f'label_{h}'] = ranks[mask].values
                    ranked += 1
            if ranked:
                logger.info(f"  ng1.2.1: cross-sectional rank applied to {ranked} vn_label_* columns")

        # ng1.0.7: Use conditional labels (bear: rank-based, bull: industry excess).
        # Skip for ng1.3.x / ng1.4.x / ng1.5.x — those caches do not have cond_label_*.
        if (version_ge(self._ng_version, 'ng1.0.7')
                and not is_13 and not is_14 and not is_15):
            for h in ['3d', '5d', '10d', '15d']:
                cond_col = f'cond_label_{h}'
                if cond_col in df_raw.columns:
                    cond_vals = pd.to_numeric(df_raw[cond_col], errors='coerce')
                    mask = cond_vals.notna()
                    if mask.any():
                        result.loc[mask.values, f'label_{h}'] = cond_vals[mask].values
                        logger.info(f"  Using conditional label_{h}: {mask.sum():,} values")

        # downside_10d (v1.0.2 linear lineage): backward compat with ng101 cache
        if 'downside_10d' in df_raw.columns and not _is_1_2_branch(self.schema_version):
            result['downside_10d'] = pd.to_numeric(df_raw['downside_10d'], errors='coerce').fillna(0.0).values
        elif not _is_1_2_branch(self.schema_version):
            result['downside_10d'] = 0.0

        # ng1.2.3: propagate 4-horizon downside_kd columns from df_raw into result
        if _is_1_2_branch(self.schema_version) and _version_in_range(self.schema_version, 'ng1.2.3', 'ng1.2.4'):
            for h in [3, 5, 10, 15]:
                col = f'downside_{h}d'
                if col in df_raw.columns:
                    result[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0.0).values
                else:
                    result[col] = 0.0
                null_count = int(pd.isna(result[col]).sum())
                if null_count > 0:
                    logger.warning(f"  ng1.2.3: {null_count} NULL {col} rows (partial cache? penalty=0 for those)")

        # ng1.3.x/ng1.5.x: propagate 4-horizon downside columns from df_raw into result
        if is_13 or is_15:
            for h in [3, 5, 10, 15]:
                col = f'downside_{h}d'
                if col in df_raw.columns:
                    result[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0.0).values
                else:
                    result[col] = 0.0
                    logger.warning(f"  {self._ng_version}: {col} missing from cache — fill=0.0 (run ng_cache_updater first)")

        # ng1.3.x dual-head: if head='downside', swap downside_Nd → label_Nd so
        # V485Trainer.walk_forward_train trains on downside values unchanged.
        # Log pre/post label_10d stats to catch silent residual transforms.
        if _is_1_3_branch(self._ng_version) and self._head == 'downside':
            logger.info("  ng1.3.x head=downside: overriding label_Nd with downside_Nd values")
            logger.info(f"    pre-swap  label_10d: {_describe_series(result['label_10d'])}")
            logger.info(f"    source    downside_10d: {_describe_series(result['downside_10d'])}")
            for h in [3, 5, 10, 15]:
                ds_col = f'downside_{h}d'
                lb_col = f'label_{h}d'
                if ds_col in result.columns:
                    result[lb_col] = result[ds_col].values
                    logger.info(f"    label_{h}d ← downside_{h}d (non-zero: "
                                f"{int((result[lb_col] != 0).sum()):,})")
            logger.info(f"    post-swap label_10d: {_describe_series(result['label_10d'])}")

        # Market features: ffill
        result = result.sort_values('trade_date')
        for col in market_cols_to_load:
            result[col] = pd.to_numeric(result[col], errors='coerce')
            result[col] = result[col].ffill()
        result = result.dropna(subset=MARKET_FEATURE_NAMES)

        # Stock features: fill NaN with 0
        for col in active_stock_features:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0.0)

        # ng1.2.3: apply soft downside penalty to multi-target labels (ng1.2.4 skips)
        if _is_1_2_branch(self.schema_version) and _version_in_range(self.schema_version, 'ng1.2.3', 'ng1.2.4'):
            from ml_models.ng.ng123_label_transform import apply_downside_penalty
            lam = float(getattr(self, '_lambda_downside', 0.3))
            penalized_horizons = []
            for h in [3, 5, 10, 15]:
                excess_col = f'label_{h}d'
                ds_col = f'downside_{h}d'
                if ds_col in result.columns and excess_col in result.columns:
                    result[excess_col] = apply_downside_penalty(
                        excess=result[excess_col].values,
                        downside=result[ds_col].values,
                        lam=lam,
                    )
                    penalized_horizons.append(h)
            logger.info(f"  ng1.2.3: applied downside penalty (λ={lam}) to labels for horizons {penalized_horizons}")

        # ng1.0.9: Smooth labels — average return over entry window
        smooth_window = getattr(self, '_smooth_label', 0)
        if smooth_window > 1:
            logger.info(f"  Smoothing labels over {smooth_window}-day entry window...")
            result = result.sort_values(['code', 'trade_date']).reset_index(drop=True)
            for h in ['3d', '5d', '10d', '15d']:
                col = f'label_{h}'
                if col in result.columns:
                    # Rolling mean per stock over smooth_window consecutive dates
                    result[col] = result.groupby('code')[col].transform(
                        lambda x: x.rolling(smooth_window, min_periods=1).mean()
                    )
            logger.info(f"  Labels smoothed: window={smooth_window}")

        result = result.dropna(subset=['label_3d', 'label_5d', 'label_10d'])
        result = result.sort_values(['trade_date', 'code']).reset_index(drop=True)

        # ng1.6.1 F2: cross-sectional factor residualization on labels.
        # For each trade_date, regress label_Nd across stocks on factor exposure
        # proxies (size/value/momentum/industry), use residual as "pure alpha".
        if _is_1_6_branch(self._ng_version):
            self._residualize_labels_cross_section(result, NG161_FACTOR_PROXIES)

        # Stub market_calculator for V475 serialization
        mkt_df = result[['trade_date'] + [c for c in MARKET_FEATURE_NAMES if c in result.columns]].drop_duplicates('trade_date')
        if self.market_calculator is not None:
            self.market_calculator.market_features = mkt_df

        n_stocks = result['code'].nunique()
        n_dates = result['trade_date'].nunique()
        n_total_features = len(active_stock_features) + len(MARKET_FEATURE_NAMES)
        logger.info(f"  NG load_data complete: {len(result):,} rows, "
                     f"{n_stocks:,} stocks, {n_dates} dates, {n_total_features} features")

        return result

    # ------------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare NG feature matrix (dynamic: base + moneyflow + interaction + conditional_ix)."""
        active_stock_features = self._get_active_stock_features()

        # Determine market feature columns (version-specific to avoid ext_market
        # leaking into ng1.3.x/ng1.4.x/ng1.5.x training — each version has its own
        # exact market feature list; no overflow from broader schema additions).
        if _is_1_5_branch(self._ng_version):
            active_market_cols = [c for c in NG150_MARKET_FEATURES if c in df.columns]
        elif _is_1_3_branch(self._ng_version):
            active_market_cols = [c for c in NG130_MARKET_FEATURES if c in df.columns]
        elif self._ng_version == 'ng1.4.2':
            active_market_cols = [c for c in NG142_MARKET_FEATURES if c in df.columns]
        elif _is_1_4_branch(self._ng_version):
            active_market_cols = [c for c in NG140_MARKET_FEATURES if c in df.columns]
        elif version_ge(self._ng_version, 'ng1.0.7'):
            active_market_cols = [c for c in NG107_MARKET_FEATURES if c in df.columns]
        else:
            active_market_cols = [c for c in MARKET_FEATURE_NAMES if c in df.columns]

        logger.info(f"NG {self._ng_version} prepare_features: "
                     f"{len(active_stock_features)} stock + {len(active_market_cols)} market "
                     f"= {len(active_stock_features) + len(active_market_cols)} total")

        self.stock_feature_cols = [c for c in active_stock_features if c in df.columns]
        self.macro_feature_cols = active_market_cols
        self.feature_names = self.stock_feature_cols + self.macro_feature_cols

        logger.info(f"  Stock features: {len(self.stock_feature_cols)}, "
                     f"Market features: {len(self.macro_feature_cols)}")

        # Cross-sectional Robust Z-Score on stock features
        # Note: CS rank features are already [0,1] but z-scoring won't hurt much
        logger.info("  Applying cross-sectional Robust Z-Score to stock features ...")
        stock_data = df[self.stock_feature_cols].values.copy()
        dates_arr = df['trade_date'].values
        stock_data = self._robust_zscore_cross_section(stock_data, dates_arr)
        df[self.stock_feature_cols] = stock_data
        df[self.stock_feature_cols] = df[self.stock_feature_cols].fillna(0.0)

        self.rank_normalized = False
        self.robust_zscore = True
        self.dual_stream = False

        # P3: Market orthogonalization — regress out market factors from stock factors
        # so the model must find alpha from stock-level signals, not market timing
        if version_ge(self._ng_version, 'ng1.1.0') and self.macro_feature_cols:
            logger.info("  P3: Orthogonalizing stock features against market factors ...")
            from numpy.linalg import lstsq
            market_vals = df[self.macro_feature_cols].values
            stock_vals = df[self.stock_feature_cols].values
            market_with_intercept = np.column_stack([market_vals, np.ones(len(market_vals))])
            # Valid mask is the same for all stock features (market side only)
            market_valid = ~np.any(np.isnan(market_with_intercept), axis=1)
            # After z-score + fillna, stock features have no NaNs → batched lstsq
            valid = market_valid & ~np.any(np.isnan(stock_vals), axis=1)
            if valid.sum() >= 100:
                beta_all, _, _, _ = lstsq(market_with_intercept[valid], stock_vals[valid], rcond=None)
                stock_vals[valid] = stock_vals[valid] - market_with_intercept[valid] @ beta_all
            df[self.stock_feature_cols] = np.nan_to_num(stock_vals, nan=0.0)
            logger.info(f"  P3: {len(self.stock_feature_cols)} stock features orthogonalized")

        X = df[self.feature_names].values
        y_3d = df['label_3d'].values
        y_5d = df['label_5d'].values
        y_10d = df['label_10d'].values
        # label_15d: use actual values where available, fallback to label_10d
        y_15d_raw = pd.to_numeric(df['label_15d'], errors='coerce').values
        y_10d_vals = df['label_10d'].values.copy()
        y_15d = np.where(np.isnan(y_15d_raw), y_10d_vals, y_15d_raw)

        # v1.0.2: downside target (stored as instance var for V485 compatibility)
        self._y_downside = df['downside_10d'].values.copy() if 'downside_10d' in df.columns else np.zeros(len(df))

        self.winsorize_bounds = None

        logger.info(f"  Feature matrix: {X.shape[0]:,} x {X.shape[1]}")
        return X, y_3d, y_5d, y_10d, y_15d, df

    # ------------------------------------------------------------------
    # Downside Model Training (v1.0.2)
    # ------------------------------------------------------------------

    def _train_downside_model(self, X_train, y_train, X_val, y_val, feature_names):
        """Train a standalone LightGBM for downside_10d prediction."""
        import lightgbm as lgb
        from ml_models.training.train_v395_multi_target import _GLOBAL_RANDOM_SEED

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 200,
            'seed': _GLOBAL_RANDOM_SEED,
            'verbose': -1,
        }

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        return model

    def train_single_target_models(self, X_train, X_val, y_train, y_val,
                                    target_name: str, sample_weights_train=None):
        """Override: append margin_rank (ng1.2.0) or lgb_quintile (ng1.2.2).

        Each ng1.2.x variant augments the base ensemble {lgb,xgb,cb,rf,hgb,
        lgb_rank} with a single additional model aligned to that variant's
        loss experiment. Adaptive ICIR weighting decides whether the new
        model earns weight vs the base members — same mechanism as ng1.2.0.
        """
        models, pred_train, pred_val = super().train_single_target_models(
            X_train, X_val, y_train, y_val, target_name,
            sample_weights_train=sample_weights_train
        )

        if self._ng_version == 'ng1.2.2':
            self._append_quintile_ce(
                models, pred_train, pred_val,
                X_train, X_val, y_train, y_val, target_name
            )
            return models, pred_train, pred_val

        if self._ng_version != 'ng1.2.0':
            return models, pred_train, pred_val

        train_dates = getattr(self, 'train_dates', None)
        val_dates = getattr(self, 'val_dates', None)
        if train_dates is None or len(train_dates) != len(y_train):
            logger.warning(f"  [ng1.2.0] missing train_dates, skip margin_rank for {target_name}")
            return models, pred_train, pred_val

        margin = float(getattr(self, '_margin', 0.05))
        # Wrap entire margin training: mirrors _train_downside_model's pattern
        # (degrade gracefully — losing one ensemble member for one WF window
        # shouldn't abort the other 5 models + remaining windows).
        try:
            group_train = build_groups_per_date(train_dates)
            if sum(group_train) != len(y_train):
                raise ValueError(
                    f"group_train sum {sum(group_train)} != len(y_train) {len(y_train)} "
                    f"(train_dates must be contiguous per date)"
                )

            group_val = None
            if val_dates is not None:
                group_val = build_groups_per_date(val_dates)
                if sum(group_val) != len(y_val):
                    raise ValueError(
                        f"group_val sum {sum(group_val)} != len(y_val) {len(y_val)}"
                    )

            params = {
                **RANK_BASE_PARAMS,
                'objective': make_margin_objective(margin=margin),
            }

            dtrain = lgb.Dataset(
                X_train, label=y_train, group=group_train,
                weight=sample_weights_train, free_raw_data=True,
            )
            # valid_sets: val only; monitoring train loss with early_stopping just
            # rides overfitting down to zero loss instead of generalization.
            callbacks = [lgb.log_evaluation(0)]
            if group_val is not None:
                dval = lgb.Dataset(
                    X_val, label=y_val, group=group_val,
                    reference=dtrain, free_raw_data=True,
                )
                valid_sets = [dval]
                callbacks.insert(0, lgb.early_stopping(30))
            else:
                valid_sets = [dtrain]  # eval only — no early_stopping

            # num_boost_round=300: the pairwise-hinge objective is O(N²) per group
            # per iteration. At ~2500 stocks/date × 1300 dates, one iteration costs
            # ~8B ops. 1000 rounds would push WF fast-check past 2h; 300 still gives
            # early_stopping room (patience=30) without blowing the time budget.
            logger.info(f"  训练 margin_rank ({target_name}, margin={margin})...")
            margin_model = lgb.train(
                params, dtrain,
                num_boost_round=300,
                feval=make_margin_eval_metric(margin=margin),
                valid_sets=valid_sets,
                callbacks=callbacks,
            )
            models['margin_rank'] = margin_model
            pred_train['margin_rank'] = margin_model.predict(X_train)
            pred_val['margin_rank'] = margin_model.predict(X_val)
            logger.info(f"    margin_rank done: groups={len(group_train)}, margin={margin}")
        except Exception as e:
            logger.warning(
                f"  margin_rank training failed ({target_name}, margin={margin}): "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )

        return models, pred_train, pred_val

    def _append_quintile_ce(self, models, pred_train, pred_val,
                             X_train, X_val, y_train, y_val, target_name: str):
        """Train ng1.2.2 quintile-CE classifier and register P(strong_buy)."""
        train_dates = getattr(self, 'train_dates', None)
        val_dates = getattr(self, 'val_dates', None)
        if train_dates is None or len(train_dates) != len(y_train):
            logger.warning(f"  [ng1.2.2] missing train_dates, skip lgb_quintile for {target_name}")
            return

        try:
            cls_train, w_train, valid_tr = make_quintile_dataset(y_train, train_dates)
            if valid_tr.sum() < 1000:
                logger.warning(
                    f"  [ng1.2.2] too few valid training rows ({int(valid_tr.sum())}), skip lgb_quintile"
                )
                return

            X_tr_v = X_train[valid_tr] if hasattr(X_train, 'shape') else np.asarray(X_train)[valid_tr]
            cls_tr_v = cls_train[valid_tr].astype(np.int32)
            w_tr_v = w_train[valid_tr]

            # Validation set — allow empty valid mask to fall back to no early stop
            if val_dates is not None and len(val_dates) == len(y_val):
                cls_val, w_val, valid_va = make_quintile_dataset(y_val, val_dates)
            else:
                cls_val, w_val, valid_va = None, None, None

            params = {
                'objective': 'multiclass',
                'num_class': N_CLASSES,
                'metric': 'multi_logloss',
                'learning_rate': 0.03,
                'num_leaves': 31,
                'min_data_in_leaf': 200,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.5,
                'reg_lambda': 3.0,
                'verbose': -1,
            }

            dtrain = lgb.Dataset(X_tr_v, label=cls_tr_v, weight=w_tr_v, free_raw_data=True)
            callbacks = [lgb.log_evaluation(0)]
            has_val = valid_va is not None and valid_va.sum() >= 500
            if has_val:
                X_va_v = X_val[valid_va] if hasattr(X_val, 'shape') else np.asarray(X_val)[valid_va]
                dval = lgb.Dataset(
                    X_va_v, label=cls_val[valid_va].astype(np.int32),
                    weight=w_val[valid_va], reference=dtrain, free_raw_data=True,
                )
                valid_sets = [dval]
                callbacks.insert(0, lgb.early_stopping(30))
            else:
                valid_sets = [dtrain]

            # Cap rounds lower when no early_stopping guard is available.
            num_rounds = 500 if has_val else 300
            logger.info(f"  训练 lgb_quintile ({target_name}, n_train_valid={int(valid_tr.sum())})...")
            booster = lgb.train(
                params, dtrain,
                num_boost_round=num_rounds,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )
            # Predict on full X_train/X_val so pred_train stays row-aligned
            # with y_train — downstream IS IC expects full-length predictions.
            wrapper = QuintileStrongBuyModel(booster)
            models[QUINTILE_MODEL_KEY] = wrapper
            pred_train[QUINTILE_MODEL_KEY] = wrapper.predict(X_train)
            pred_val[QUINTILE_MODEL_KEY] = wrapper.predict(X_val)
            logger.info(
                f"    lgb_quintile done: rounds={booster.num_trees() // N_CLASSES}, "
                f"best_iter={booster.best_iteration}"
            )
        except Exception as e:
            logger.warning(
                f"  lgb_quintile training failed ({target_name}): {type(e).__name__}: {e}",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Walk-Forward Training
    # ------------------------------------------------------------------

    def walk_forward_train(self, start_date: str = None, end_date: str = None,
                           purge_days: int = 15, min_train_days: int = 900,
                           val_days: int = 120, test_days: int = 120,
                           step_days: int = 120):
        """NG Walk-Forward Training with ICIR adaptive weights and WF summary."""
        import shutil

        logger.info("=" * 60)
        logger.info(f"NG {self._ng_version} Walk-Forward Training")
        logger.info("=" * 60)
        logger.info(f"  Base: V4.8.5 machinery (6-model ensemble, LambdaRank, Bear Specialist)")
        logger.info(f"  Data: {self.cache_table}")
        # ng1.0.7: auto-enable regime weighting
        if version_ge(self._ng_version, 'ng1.0.7'):
            self._regime_weight = True

        logger.info(f"  Switches: moneyflow={getattr(self, '_enable_moneyflow', False)}, "
                     f"interaction={getattr(self, '_enable_interaction', False)}, "
                     f"regime_weight={getattr(self, '_regime_weight', False)}")
        if version_ge(self._ng_version, 'ng1.0.7'):
            logger.info(f"  Labels: CONDITIONAL (bear: rank-blend, bull: industry excess)")
            logger.info(f"  Features: {len(NG107_ALL_FEATURES)} total (56 stock + 18 market + 7 cond_ix)")
        else:
            logger.info(f"  Labels: INDUSTRY EXCESS returns (stock - industry median)")
        logger.info(f"  Initial weights: {', '.join(f'{k}={v:.2f}' for k, v in self.target_weights.items())}")

        # 数据泄露修复 (2026-04-12): 禁用全期 IC screening
        # 之前会用 [start_date, end_date] 全量数据 (含未来的 walk-forward test 窗口)
        # 计算 IC 来选 interaction/conditional_ix 特征, 导致 OOS IC 虚高 5-15%.
        # 现在保持 _selected_ix/_selected_cx 为空, 下游 get_active_feature_names()
        # 会自动 fallback 到 INTERACTION_FEATURE_NAMES / CONDITIONAL_IX_FEATURE_NAMES 全集.
        self._selected_ix = []
        self._selected_cx = []
        if getattr(self, '_enable_interaction', False) or version_ge(self._ng_version, 'ng1.0.7'):
            logger.info("IC screening disabled (data leakage fix) — using full INTERACTION/CONDITIONAL_IX feature lists")

        model_data, history = super().walk_forward_train(
            start_date=start_date, end_date=end_date,
            purge_days=purge_days, min_train_days=min_train_days,
            val_days=val_days, test_days=test_days, step_days=step_days)

        # fast-check: skip model save
        if model_data.get('fast_check'):
            return model_data, history

        # --- v1.0.2: Train downside model ---
        # ng1.3.x has dual-head (4-horizon) architecture via --head downside.
        # ng1.4.x explicitly drops the downside auxiliary model (clean ng1.0.1-style).
        # ng1.5.x inherits ng1.4.x — also drops the legacy aux model. Running it
        # here overwrites self.stock_feature_cols with NG110_STOCK_FEATURES (58)
        # instead of the NG150/NG140 list, losing the new Tier A/B features.
        skip_legacy_downside = (
            _is_1_3_branch(self._ng_version)
            or _is_1_4_branch(self._ng_version)
            or _is_1_5_branch(self._ng_version)
        )
        if skip_legacy_downside:
            logger.info(f"{self._ng_version}: skipping legacy downside_10d model")
        else:
            # Restore feature cols (WF parent may have modified them)
            if version_ge(self._ng_version, 'ng1.1.0'):
                self.stock_feature_cols = list(NG110_STOCK_FEATURES)
                self.macro_feature_cols = list(MARKET_FEATURE_NAMES)
                self.feature_names = list(NG110_ALL_FEATURES)
            elif version_ge(self._ng_version, 'ng1.0.7'):
                self.stock_feature_cols = list(STOCK_FEATURE_NAMES)
                self.macro_feature_cols = list(NG107_MARKET_FEATURES)
                self.feature_names = self.stock_feature_cols + self.macro_feature_cols
            logger.info("Training downside_10d model (separate LightGBM pass)...")
        if not skip_legacy_downside:
            try:
                df_full = self.load_data(start_date=start_date, end_date=end_date)
                _result = self.prepare_features(df_full)
                X, y_3d, y_5d, y_10d, y_15d, df_full = _result
                y_downside = self._y_downside

                # Use last portion as val (matching WF logic)
                unique_dates = sorted(df_full['trade_date'].unique())
                n = len(unique_dates)
                val_start_idx = max(0, n - test_days - val_days)
                val_end_idx = max(0, n - test_days)

                train_dates = set(unique_dates[:val_start_idx])
                val_dates = set(unique_dates[val_start_idx:val_end_idx])
                test_dates = set(unique_dates[val_end_idx:])

                train_mask = df_full['trade_date'].isin(train_dates).values
                val_mask = df_full['trade_date'].isin(val_dates).values
                test_mask = df_full['trade_date'].isin(test_dates).values

                if train_mask.sum() > 1000 and val_mask.sum() > 100:
                    downside_model = self._train_downside_model(
                        X[train_mask], y_downside[train_mask],
                        X[val_mask], y_downside[val_mask],
                        feature_names=self.feature_names,
                    )
                    model_data['downside_model'] = downside_model

                    # Compute OOS IC
                    if test_mask.sum() > 0:
                        pred_ds = downside_model.predict(X[test_mask])
                        from scipy.stats import spearmanr
                        ic, _ = spearmanr(pred_ds, y_downside[test_mask])
                        logger.info(f"  Downside 10d OOS IC: {ic:.4f}")
                        model_data['downside_ic'] = float(ic)

                    logger.info(f"  Downside model trained: {downside_model.num_trees()} trees")
                else:
                    logger.warning("  Not enough data for downside model training")
                    model_data['downside_model'] = None
            except Exception as e:
                logger.warning(f"  Downside model training failed: {e}")
                model_data['downside_model'] = None

        # Compute ICIR adaptive weights from WF history
        adaptive_weights = self._compute_icir_adaptive_weights(history)
        self.target_weights = adaptive_weights

        # Move from v485/ to ng/
        v485_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v485'
        ng_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'ng'
        ng_dir.mkdir(parents=True, exist_ok=True)

        v485_files = sorted(v485_dir.glob('v485_*.pkl'), key=lambda f: f.stat().st_mtime)
        if v485_files:
            latest = v485_files[-1]
            timestamp = latest.stem.replace('v485_multi_target_', '')
            version_tag = self._ng_version.replace('.', '')  # e.g. ng104
            # Include seed tag in filename if a global seed was set
            seed_val = 42
            try:
                import ml_models.training.train_v395_multi_target as _tm
                seed_val = getattr(_tm, '_GLOBAL_RANDOM_SEED', 42)
                seed_tag = f'_seed{seed_val}'
            except Exception:
                seed_tag = ''
            # ng1.2.0 grid search: encode margin in filename so margin=0.03 and
            # margin=0.05 runs don't silently overwrite each other's .pkl.
            margin_tag = ''
            if self._ng_version == 'ng1.2.0':
                margin = float(getattr(self, '_margin', 0.05))
                margin_tag = f'_m{int(round(margin * 100)):03d}'
            # ng1.3.x dual-head: encode head in filename so excess/downside don't overwrite each other
            head_tag = ''
            if _is_1_3_branch(self._ng_version):
                head_tag = f'_{self._head}'
            new_path = ng_dir / f'{version_tag}{seed_tag}{head_tag}{margin_tag}_multi_target_{timestamp}.pkl'

            # Update model metadata
            model_data['version'] = self._ng_version
            model_data['lambda_risk'] = getattr(self, '_lambda_risk', 0.5)
            model_data['risk_filter_quantile'] = getattr(self, '_risk_filter_quantile', 0.20)
            model_data['ng_innovations'] = {
                'base': 'V4.8.5 ensemble machinery',
                'version': self._ng_version,
                'prev_version': NG_V1_VERSION,
                'data_source': f'{self.cache_table} (industry excess labels)',
                'feature_set': f'{len(self.stock_feature_cols)} stock + {len(self.macro_feature_cols)} market features',
                'stock_features': list(self.stock_feature_cols),
                'market_features': list(self.macro_feature_cols),
                'targets': ['3d', '5d', '10d', '15d'],
                'target_weights': adaptive_weights,
                'label_type': 'industry_excess_return',
                'downside_model': model_data.get('downside_model') is not None,
                'lambda_risk': model_data.get('lambda_risk', 0.5),
                'new_in_v110': {
                    'cs_rank_factors': 10,
                    'residual_factors': 5,
                    'sector_activity_factors': 3,
                    'removed_factors': 11,
                    'icir_adaptive_weights': True,
                    'wf_summary': True,
                },
                'ng110_switches': {
                    'moneyflow': getattr(self, '_enable_moneyflow', False),
                    'interaction': getattr(self, '_enable_interaction', False),
                    'interaction_selected': self._selected_ix if getattr(self, '_enable_interaction', False) else [],
                    'residual_label': True,  # ng1.0.3 cache always has residual labels
                    'regime_weight': getattr(self, '_regime_weight', False),
                    'wf_step_days': step_days,
                },
            }
            # Warn if pkl feature_names mismatches the trained booster's column count
            # (stale cache or V485 internals can silently mutate feature cols mid-WF).
            self._warn_on_feature_count_mismatch(model_data)

            model_data['feature_names'] = self.feature_names
            model_data['stock_feature_cols'] = self.stock_feature_cols
            model_data['macro_feature_cols'] = self.macro_feature_cols
            model_data['target_weights'] = adaptive_weights

            # P0.3 Check 9: reproducibility metadata
            try:
                model_data['git_commit_hash'] = subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=str(PROJECT_ROOT), text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except Exception:
                model_data['git_commit_hash'] = 'unknown'
            model_data['host'] = socket.gethostname()
            model_data['training_duration_sec'] = time.time() - self._train_start_ts
            model_data['schema_version'] = self.schema_version
            model_data['seed'] = seed_val
            model_data['wf_mode'] = getattr(self, '_wf_mode', 'expanding')
            model_data['purge_days'] = getattr(self, '_purge_days', None)

            joblib.dump(model_data, new_path)
            logger.info(f"\nNG {self._ng_version} model saved: {new_path}")
            logger.info(f"  Size: {new_path.stat().st_size / 1024 / 1024:.1f} MB")

            # Copy auxiliary files
            for aux in ['global_quantiles.npy', 'recommendation_thresholds.json']:
                src = v485_dir / aux
                if src.exists():
                    shutil.copy2(str(src), str(ng_dir / aux))

            # Clean up v485 artifacts
            latest.unlink()
            for hf in v485_dir.glob(f'training_history_{timestamp}*'):
                hf.unlink()

            # Save training history
            history['version'] = self._ng_version
            history['base'] = f'NG {self._ng_version} ({len(self.feature_names)} features, industry excess labels)'
            history['ng_innovations'] = model_data['ng_innovations']
            history['adaptive_weights'] = adaptive_weights

            history_path = ng_dir / f'training_history_{timestamp}.json'
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            latest_hist_path = ng_dir / 'training_history_latest.json'
            with open(latest_hist_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

            # Generate WF summary (v1.0.3)
            wf_summary = self._generate_wf_summary(history, ng_dir)

            # Factor quality analysis
            self._log_factor_quality(model_data, adaptive_weights, history, ng_dir, timestamp)

            logger.info(f"\nNG {self._ng_version} training complete!")
            logger.info(f"  Features: {len(self.feature_names)}")
            logger.info(f"  ICIR weights: {', '.join(f'{k}={v:.3f}' for k, v in adaptive_weights.items())}")
            logger.info(f"  Model: {new_path.name}")
            if wf_summary.get('aggregate', {}).get('wfer') is not None:
                logger.info(f"  WFER: {wf_summary['aggregate']['wfer']:.3f}")
        else:
            logger.warning("No v485 model file found to relocate to ng/")

        return model_data, history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=f'NG {NG_VERSION} Trainer')
    parser.add_argument('--start-date', default='2020-01-01')
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--purge-days', type=int, default=15)
    parser.add_argument('--min-train-days', type=int, default=900)
    parser.add_argument('--val-days', type=int, default=120)
    parser.add_argument('--test-days', type=int, default=120)
    parser.add_argument('--step-days', type=int, default=120)
    parser.add_argument('--fast-check', action='store_true',
                        help='Fast check mode: 2 WF windows, no model save')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel WF workers')
    parser.add_argument('--target-parallel', type=int, default=1,
                        help='Targets trained concurrently per window (1=serial, 4=3d/5d/10d/15d in parallel). '
                             'Measured ~1.4x speedup on M5 Max with N=4.')
    parser.add_argument('--lambda-risk', type=float, default=0.5,
                        help='Risk discount factor for downside model (default: 0.5)')
    # ng1.0.3 new switches
    parser.add_argument('--enable-moneyflow', action='store_true',
                        help='Enable moneyflow features (8 factors)')
    parser.add_argument('--enable-interaction', action='store_true',
                        help='Enable interaction features with IC screening')
    parser.add_argument('--residual-label', action='store_true',
                        help='Use style-residual labels (ng1.0.3 default)')
    parser.add_argument('--wf-windows', type=int, default=3,
                        help='Target WF windows (3 or 8)')
    parser.add_argument('--regime-weight', action='store_true',
                        help='Enable market regime sample weighting')
    parser.add_argument('--seed', type=int, default=None,
                        help='Global random seed (for multi-seed ensemble)')
    # ng1.0.4 new arguments
    parser.add_argument('--version', default=None,
                        help='NG version to train (e.g., ng1.0.3, ng1.0.4)')
    parser.add_argument('--penalty-power', type=float, default=1.5,
                        help='Risk-adjusted label penalty power (ng1.0.4)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds for multi-seed training (e.g., 42,123,456,789,2024)')
    # ng1.0.7 new arguments
    parser.add_argument('--risk-filter-quantile', type=float, default=0.20,
                        help='Pareto risk filter: exclude worst N%% stocks by predicted maxdd (ng1.0.7, default: 0.20)')
    # ng1.0.9: signal persistence
    parser.add_argument('--persistent-features', action='store_true',
                        help='ng1.0.9: only use 22 persistent features (10d rank autocorr >= 0.5)')
    parser.add_argument('--min-autocorr', type=float, default=0.0,
                        help='ng1.0.9: minimum 10d rank autocorrelation threshold (0=off, 0.3-0.5 recommended)')
    parser.add_argument('--smooth-label', type=int, default=0,
                        help='ng1.0.9: label smoothing window (0=off, 5=average over 5-day entry window)')
    parser.add_argument('--margin', type=float, default=0.05,
                        help='ng1.2.0: pairwise margin ranking loss margin value (0.03-0.10 typical)')
    parser.add_argument('--lambda-downside', type=float, default=0.3,
                        help='ng1.2.3: downside penalty multiplier for label transform (default 0.3 per spec §5). '
                             'λ ∈ {0, 0.15, 0.3, 0.45, 0.6} for ablation.')
    # ng1.3.x dual-head training
    parser.add_argument('--head', choices=['excess', 'downside'], default='excess',
                        help='ng1.3.x multi-task head selector. "excess" = industry excess labels (default). '
                             '"downside" = min-cumret downside labels (requires ng1.3.x cache with downside_Nd cols).')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    version = args.version or NG_VERSION

    def _apply_trainer_switches(trainer):
        """Apply CLI switches to a trainer instance."""
        trainer._enable_moneyflow = args.enable_moneyflow
        trainer._enable_interaction = args.enable_interaction
        trainer._regime_weight = args.regime_weight
        trainer._lambda_risk = args.lambda_risk
        trainer._risk_filter_quantile = args.risk_filter_quantile
        trainer._persistent_features = args.persistent_features
        trainer._smooth_label = args.smooth_label
        trainer._min_autocorr = args.min_autocorr
        trainer._margin = args.margin
        trainer._lambda_downside = args.lambda_downside
        if args.fast_check:
            trainer._fast_check = True
            trainer._fast_check_max_windows = 2
            trainer._fast_check_min_train = min(args.min_train_days, 600)
            trainer._fast_check_val_days = 60
            trainer._fast_check_test_days = 60
            trainer._fast_check_step_days = 60
        if args.parallel > 1:
            trainer._parallel_wf_workers = args.parallel
        if args.target_parallel > 1:
            trainer._target_parallel = args.target_parallel

    if args.wf_windows > 3:
        args.step_days = 90
        logger.info(f"WF windows target: {args.wf_windows}, step_days=90")

    if args.seeds:
        # Multi-seed training loop
        import random
        import ml_models.training.train_v395_multi_target as _trainer_mod
        seed_list = [int(s.strip()) for s in args.seeds.split(',')]
        for i, seed_val in enumerate(seed_list):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training seed {seed_val} ({i+1}/{len(seed_list)})")
            logger.info(f"{'='*60}")
            random.seed(seed_val)
            np.random.seed(seed_val)
            _trainer_mod._GLOBAL_RANDOM_SEED = seed_val

            trainer = NGTrainer(version=version, head=args.head)
            _apply_trainer_switches(trainer)

            model_data, history = trainer.walk_forward_train(
                start_date=args.start_date,
                end_date=args.end_date,
                purge_days=args.purge_days,
                min_train_days=args.min_train_days,
                val_days=args.val_days,
                test_days=args.test_days,
                step_days=args.step_days,
            )
    else:
        # Single-seed training
        if args.seed is not None:
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            import ml_models.training.train_v395_multi_target as _trainer_mod
            _trainer_mod._GLOBAL_RANDOM_SEED = args.seed
            logger.info(f"Global random seed set to {args.seed} (numpy + LGB/XGB/CB/RF/HGB)")

        trainer = NGTrainer(version=version, head=args.head)
        _apply_trainer_switches(trainer)

        model_data, history = trainer.walk_forward_train(
            start_date=args.start_date,
            end_date=args.end_date,
            purge_days=args.purge_days,
            min_train_days=args.min_train_days,
            val_days=args.val_days,
            test_days=args.test_days,
            step_days=args.step_days,
        )
