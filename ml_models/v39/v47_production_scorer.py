#!/usr/bin/env python3
"""
V4.7 生产评分器
基于 V4.6.1 底座, 但:
1. 模型从 v47 目录加载 (无小盘训练加权)
2. 移除小盘评分加成 (Step 9)
3. 保留所有其他增强: ICIR权重, Combined Isotonic, Meta-Learner, 流动性折扣
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from .v46_production_scorer import V46ProductionScorer

import logging
logger = logging.getLogger(__name__)


class V47ProductionScorer(V46ProductionScorer):
    """V4.7 生产评分器 — V4.6底座 - 小盘加权/加成 + percentile-label LGB"""

    def __init__(self, model_type: str = 'small_data'):
        self._v47_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v47'
        # Override v46 model dir before parent __init__
        self._v46_model_dir = self._v47_model_dir
        super().__init__(model_type=model_type)

    def _load_v46_model(self):
        """加载v4.7模型 (文件名v47_*.pkl)"""
        import joblib
        import pickle

        model_files = list(self.model_dir.glob('v47_*.pkl'))
        if not model_files:
            # Fallback to v46 pattern
            model_files = list(self.model_dir.glob('v46_*.pkl'))
        if not model_files:
            print(f"V4.7 未找到模型文件: {self.model_dir}/v47_*.pkl")
            return

        latest = max(model_files, key=lambda f: f.stat().st_mtime)
        try:
            model_data = joblib.load(latest)
        except Exception:
            with open(latest, 'rb') as f:
                model_data = pickle.load(f)

        raw_models = model_data.get('models', {})
        self.models = {}
        self.weights = model_data.get('ensemble_weights', {})
        for target, target_data in raw_models.items():
            if isinstance(target_data, dict) and 'models' in target_data:
                self.models[target] = target_data['models']
                if not self.weights:
                    self.weights[f'label_{target}'] = target_data.get('weights', {})
            else:
                self.models[target] = target_data

        self.scaler = model_data.get('scaler')
        self.feature_cols = model_data.get('feature_names', model_data.get('feature_cols', []))
        self.market_feature_cols = model_data.get('market_features', model_data.get('market_feature_cols', []))
        self.target_weights = model_data.get('target_weights', {
            'label_3d': 0.20, 'label_5d': 0.25, 'label_10d': 0.35, 'label_15d': 0.20
        })

        self.cascade = False
        self.cascade_feature_names = None
        self.dual_stream = False
        self.rank_normalized = False
        self.robust_zscore = model_data.get('robust_zscore', True)
        self.extra_features_from_daily_basic = model_data.get('extra_features_from_daily_basic', None)
        self.extra_tech_features = model_data.get('extra_features_from_tech_indicators', None)
        self.stock_rank_cols = model_data.get('stock_feature_cols', None)

        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        self.bear_models = model_data.get('bear_models', {})
        self.isotonic_calibration = model_data.get('isotonic_calibration', {})
        self.combined_isotonic = model_data.get('combined_isotonic')
        self.meta_learner = model_data.get('meta_learner')
        self.meta_feature_names = model_data.get('meta_feature_names')

        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        self.weights = self._clip_icir_weights(self.weights)

        wf = model_data.get('walk_forward_metrics', {})
        gq_status = "全局评分" if self.global_quantiles is not None else "截面评分"
        meta_status = "有" if self.meta_learner is not None else "无"
        ciso_status = "有" if self.combined_isotonic is not None else "无"
        has_pct = model_data.get('has_percentile_lgb', False)
        print(f"V4.7 模型加载完成: {list(self.models.keys())} [V4.6底座-小盘+{gq_status}]")
        print(f"  模型文件: {latest.name}")
        print(f"  Percentile-LGB: {'有' if has_pct else '无'}")
        print(f"  Combined Isotonic: {ciso_status}, Meta-Learner: {meta_status}")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}±{m.get('std_icir', 0):.4f}")

    def _apply_small_cap_bonus(self, results: Dict[str, Dict], date: str,
                                codes: List[str]) -> Dict[str, Dict]:
        """V4.7: 不应用小盘加成 (训练层已无小盘加权)"""
        return results
