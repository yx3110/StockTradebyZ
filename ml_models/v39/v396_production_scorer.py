#!/usr/bin/env python3
"""
V3.96 生产评分器
基于 V3.95 Robust Z-Score + Industry-Excess Labels 架构

改进点 (相比 V3.95 旧模型):
  - Robust Z-Score 截面归一化 (保留幅度信息, 替代 Rank)
  - 行业超额收益标签 (去除市场/行业噪声)
  - 新增 daily_basic 特征 (pe_ttm, pb, ps_ttm, turnover_rate, log_market_cap)
  - 独立训练各目标 (非级联)
  - 49 个特征, 5 模型 ensemble (LGB+XGB+CB+RF+HGB)
"""

import json
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from .v395_production_scorer import V395ProductionScorer


class V396ProductionScorer(V395ProductionScorer):
    """V3.96 生产评分器 — Robust Z-Score + Industry-Excess Labels"""

    def __init__(self, model_type: str = 'small_data'):
        self._v396_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v396'
        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v396 模型目录"""
        self.model_dir = self._v396_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v396_model()

    def _load_v396_model(self):
        """加载v3.96模型 (支持 v396_*.pkl 文件名)"""
        model_files = list(self.model_dir.glob('v396_*.pkl'))
        if not model_files:
            print(f"V3.96 未找到模型文件: {self.model_dir}/v396_*.pkl")
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
                if f'label_{target}' not in self.weights:
                    self.weights[f'label_{target}'] = target_data.get('weights', {})
            else:
                self.models[target] = target_data

        self.scaler = model_data.get('scaler')
        self.feature_cols = model_data.get('feature_names', model_data.get('feature_cols', []))
        self.market_feature_cols = model_data.get('market_features', model_data.get('market_feature_cols', []))
        self.target_weights = model_data.get('target_weights', self.target_weights)

        # 元数据标志
        self.cascade = model_data.get('cascade', False)
        self.cascade_feature_names = model_data.get('cascade_feature_names', None)
        self.dual_stream = model_data.get('dual_stream', False)
        self.stock_feature_cols_raw = model_data.get('stock_feature_cols_raw', None)
        self.stock_feature_cols_rank = model_data.get('stock_feature_cols_rank', None)
        self.rank_normalized = model_data.get('rank_normalized', False)
        self.cross_sectional_neutralization = model_data.get('cross_sectional_neutralization', False)
        self.stock_rank_cols = model_data.get('stock_feature_cols', None)
        self.robust_zscore = model_data.get('robust_zscore', False)
        self.extra_features_from_daily_basic = model_data.get('extra_features_from_daily_basic', None)

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        # 全局分位数
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                import numpy as np
                self.global_quantiles = np.load(quantiles_path)

        # Recommendation thresholds
        self.recommendation_thresholds = model_data.get('recommendation_thresholds')
        if not self.recommendation_thresholds:
            import json as _json
            rec_path = self.model_dir / 'recommendation_thresholds.json'
            if rec_path.exists():
                with open(rec_path) as f:
                    self.recommendation_thresholds = _json.load(f)

        suffix = " [robust_zscore+industry_excess]" if self.robust_zscore else ""
        print(f"V3.96 模型加载完成: {list(self.models.keys())}{suffix}")
        print(f"  模型文件: {latest.name}")
