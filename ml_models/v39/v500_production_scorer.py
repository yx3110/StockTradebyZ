#!/usr/bin/env python3
"""
V5.0 Unified Production Scorer

特征融合: v39_feature_cache (基础) + v40_feature_cache (精选) + daily_basic
继承 V396ProductionScorer, 扩展 v40 特征加载
"""

import json
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from .v395_production_scorer import V395ProductionScorer


# V4.0 精选特征 (与 train_v500_unified.py 保持一致)
V40_RANK_FEATURES = [
    'xs_kdj_j_rank', 'xs_rsi6_rank', 'xs_boll_position_rank',
    'xs_macd_hist_rank', 'xs_atr14_pct_rank',
    'xs_turnover_rank', 'xs_volume_ratio_rank',
    'xs_return_5d_rank', 'xs_return_10d_rank',
    'xs_volatility_10d_rank',
    'momentum_volume_confirm', 'tech_momentum_confirm',
    'contrarian_signal', 'boll_position',
]

V40_CONTINUOUS_FEATURES = [
    'amihud_illiquidity', 'volume_price_corr_10d',
    'updown_asymmetry_10d', 'max_drawdown_20d',
    'industry_momentum_rank', 'industry_rotation_signal',
    'industry_breadth', 'industry_concentration',
    'industry_volume_change',
    'squeeze_momentum', 'zhixing_short_trend',
]


class V500ProductionScorer(V395ProductionScorer):
    """V5.0 Unified Production Scorer"""

    def __init__(self, model_type: str = 'small_data'):
        self._v500_model_dir = (Path(__file__).parent.parent.parent /
                                'ml_models' / 'trained_models' / 'v500')
        # V5.0 specific metadata
        self.v40_selected_features = None
        self.v40_rank_cols = None
        self.v40_continuous_cols = None
        self.neural_cols = None
        self.include_neural = False

        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v500 模型"""
        self.model_dir = self._v500_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v500_model()

    def _load_v500_model(self):
        """加载 v5.0 模型"""
        import pickle
        import joblib

        # Prefer explicit 'latest' symlink/copy over mtime-based selection
        latest_link = self.model_dir / 'v500_unified_latest.pkl'
        if latest_link.exists():
            latest = latest_link
        else:
            model_files = list(self.model_dir.glob('v500_unified_2*.pkl'))
            if not model_files:
                print(f"V5.0 未找到模型: {self.model_dir}")
                return
            latest = max(model_files, key=lambda f: f.name)
        try:
            model_data = joblib.load(latest)
        except Exception:
            with open(latest, 'rb') as f:
                model_data = pickle.load(f)

        # Standard model loading (same as V396)
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
        self.feature_cols = model_data.get('feature_names', [])
        self.market_feature_cols = model_data.get('market_features',
                                                  model_data.get('macro_feature_cols', []))
        self.target_weights = model_data.get('target_weights', self.target_weights)

        # Standard metadata
        self.cascade = False
        self.dual_stream = False
        self.rank_normalized = False
        self.robust_zscore = model_data.get('robust_zscore', True)
        self.stock_rank_cols = model_data.get('stock_feature_cols', None)
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
                self.global_quantiles = np.load(quantiles_path)

        # V5.0 specific metadata
        self.v40_selected_features = model_data.get('v40_selected_features', [])
        self.v40_rank_cols = model_data.get('v40_rank_cols', [])
        self.v40_continuous_cols = model_data.get('v40_continuous_cols', [])
        self.neural_cols = model_data.get('neural_cols', [])
        self.include_neural = model_data.get('include_neural', False)

        n_v40 = len(self.v40_rank_cols) + len(self.v40_continuous_cols)
        n_neural = len(self.neural_cols)
        print(f"V5.0 模型加载完成: {list(self.models.keys())} "
              f"[v39+v40({n_v40})+neural({n_neural})]")
        print(f"  模型文件: {latest.name}")
        print(f"  特征数: {len(self.feature_cols)}")

    def _get_features(self, stock_codes: Optional[List[str]], date: str,
                      load_full_cross_section: bool = False) -> Optional[pd.DataFrame]:
        """获取 v39 + v40 融合特征"""
        # Step 1: 加载 v39 基础特征 (via parent)
        features_df = super()._get_features(stock_codes, date, load_full_cross_section)

        if features_df is None or len(features_df) == 0:
            return features_df

        # Step 2: 加载 v40 精选特征并 JOIN
        if self.v40_selected_features:
            v40_df = self._load_v40_features(features_df['code'].tolist(), date)
            if v40_df is not None and len(v40_df) > 0:
                features_df = features_df.merge(v40_df, on='code', how='left')
                # 缺失的 v40 特征填 0
                v40_cols = [c for c in features_df.columns if c.startswith('v40_')]
                features_df[v40_cols] = features_df[v40_cols].fillna(0)

        # Step 3: 加载 GRU embeddings (if model uses them)
        if self.include_neural and self.neural_cols:
            emb_df = self._load_neural_features(features_df['code'].tolist(), date)
            if emb_df is not None and len(emb_df) > 0:
                features_df = features_df.merge(emb_df, on='code', how='left')
                for col in self.neural_cols:
                    if col in features_df.columns:
                        features_df[col] = features_df[col].fillna(0)

        return features_df

    def _load_v40_features(self, codes: List[str], date: str) -> Optional[pd.DataFrame]:
        """从 v40_feature_cache 加载精选特征"""
        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ','.join(['?' for _ in codes])
            query = f"""
            SELECT code, features_json
            FROM v40_feature_cache
            WHERE code IN ({placeholders}) AND trade_date = ?
            """
            df_raw = pd.read_sql_query(query, conn, params=list(codes) + [date])
        finally:
            conn.close()

        if df_raw.empty:
            return None

        records = []
        for _, row in df_raw.iterrows():
            try:
                v40_all = json.loads(row['features_json'])
                record = {'code': row['code']}
                for feat in self.v40_selected_features:
                    record[f'v40_{feat}'] = v40_all.get(feat, 0)
                records.append(record)
            except Exception:
                continue

        return pd.DataFrame(records) if records else None

    def _load_neural_features(self, codes: List[str], date: str) -> Optional[pd.DataFrame]:
        """加载 GRU embeddings"""
        try:
            from ml_models.neural.embedding_cache_manager import EmbeddingCacheManager
            cache_mgr = EmbeddingCacheManager(str(self.db_path))
            emb_dict = cache_mgr.batch_load(codes, date, model_version='gru_v1')
            if not emb_dict:
                return None

            records = []
            for code, emb in emb_dict.items():
                record = {'code': code}
                for i, val in enumerate(emb):
                    record[f'gru_emb_{i}'] = val
                records.append(record)
            return pd.DataFrame(records)
        except Exception:
            return None

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V5.0 预测: v39 z-score + v40 注入 + daily_basic"""
        # robust_zscore=True 时, 父类会:
        # 1. load_full_cross_section → _get_features (已覆盖, 会注入v40)
        # 2. _robust_zscore_normalize_features (仅对 stock_rank_cols 做 z-score)
        # 3. _load_daily_basic_features
        # 4. 过滤到 stock_codes
        # 这正好是 V5.0 需要的流程
        return super().predict_scores(stock_codes, date)

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """使用预加载特征 + 动态注入 v40"""
        if features_df is not None and len(features_df) > 0:
            # 注入 v40 特征到预加载数据
            if self.v40_selected_features:
                codes = features_df['code'].tolist()
                v40_df = self._load_v40_features(codes, date)
                if v40_df is not None and len(v40_df) > 0:
                    features_df = features_df.merge(v40_df, on='code', how='left')
                    v40_cols = [c for c in features_df.columns if c.startswith('v40_')]
                    features_df[v40_cols] = features_df[v40_cols].fillna(0)

            # 注入 neural embeddings
            if self.include_neural and self.neural_cols:
                codes = features_df['code'].tolist()
                emb_df = self._load_neural_features(codes, date)
                if emb_df is not None and len(emb_df) > 0:
                    features_df = features_df.merge(emb_df, on='code', how='left')
                    for col in self.neural_cols:
                        if col in features_df.columns:
                            features_df[col] = features_df[col].fillna(0)

        return super().predict_scores_from_preloaded(stock_codes, date, features_df)

    def preload_feature_cache(self, dates: List[str]) -> Dict[str, pd.DataFrame]:
        """批量预加载 v39 特征 (v40 在 predict 时动态注入)"""
        # v39 预加载走父类, v40 由于每天数据量大不做预加载
        # 而是在 predict_scores_from_preloaded 中按需加载
        return super().preload_feature_cache(dates)
