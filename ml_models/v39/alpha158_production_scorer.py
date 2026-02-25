#!/usr/bin/env python3
"""
Alpha158 Baseline 生产评分器

独立实现，不继承V395 (避免复杂依赖)。
接口与V395兼容: predict_scores(), preload_feature_cache()

模型文件: ml_models/trained_models/alpha158/alpha158_*.pkl
"""

import json
import joblib
import pickle
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    from core.config import get_db_path as _get_db_path
    _DEFAULT_DB_PATH = _get_db_path()
except ImportError:
    _DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / 'data_adapter' / 'stock_data.db'


class Alpha158ProductionScorer:
    """Alpha158 Baseline 生产评分器"""

    def __init__(self, mode: str = 'ensemble'):
        """初始化

        Args:
            mode: 'ensemble' 或 'qlib_standard'
        """
        self.mode = mode
        self.model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'alpha158'
        self.db_path = _DEFAULT_DB_PATH

        self.models = {}
        self.weights = {}
        self.scaler = None
        self.feature_cols = None
        self.target_weights = {'label_3d': 0.35, 'label_5d': 0.35, 'label_10d': 0.30}

        # batch_generate 兼容字段
        self.robust_zscore = False
        self.extra_features_from_daily_basic = None

        # 预加载缓存
        self._feature_cache = {}  # date -> DataFrame

        self._load_model()

    def _load_model(self):
        """加载模型"""
        # 优先加载 latest
        latest_path = self.model_dir / f'alpha158_{self.mode}_latest.pkl'
        if latest_path.exists():
            model_path = latest_path
        else:
            # 查找最新模型
            pattern = f'alpha158_{self.mode}_*.pkl'
            model_files = list(self.model_dir.glob(pattern))
            if not model_files:
                # 尝试任意 alpha158 模型
                model_files = list(self.model_dir.glob('alpha158_*.pkl'))
            if not model_files:
                print(f"Alpha158 未找到模型: {self.model_dir}")
                return
            model_path = max(model_files, key=lambda f: f.stat().st_mtime)

        try:
            model_data = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

        raw_models = model_data.get('models', {})
        self.models = {}
        ensemble_weights = model_data.get('ensemble_weights', {})
        self.weights = ensemble_weights

        for target, target_data in raw_models.items():
            if isinstance(target_data, dict) and 'models' in target_data:
                self.models[target] = target_data['models']
                if f'label_{target}' not in self.weights:
                    self.weights[f'label_{target}'] = target_data.get('weights', {})
            else:
                self.models[target] = target_data

        self.scaler = model_data.get('scaler')
        self.feature_cols = model_data.get('feature_names', [])
        self.target_weights = model_data.get('target_weights', self.target_weights)

        n_models = sum(len(m) for m in self.models.values() if isinstance(m, dict))
        print(f"Alpha158 模型加载完成: {list(self.models.keys())}, "
              f"{n_models}个子模型, {len(self.feature_cols)}特征 ({model_path.name})")

    def preload_feature_cache(self, dates: List[str]):
        """批量预加载特征缓存 (供 batch_generate 使用)"""
        if not dates:
            return

        conn = sqlite3.connect(str(self.db_path))
        CHUNK_SIZE = 50
        total = 0

        for chunk_start in range(0, len(dates), CHUNK_SIZE):
            chunk = dates[chunk_start:chunk_start + CHUNK_SIZE]
            placeholders = ','.join(['?' for _ in chunk])

            query = f"""
            SELECT code, trade_date, features_json
            FROM alpha158_feature_cache
            WHERE trade_date IN ({placeholders})
            """
            df = pd.read_sql_query(query, conn, params=chunk)

            if df.empty:
                continue

            # 向量化解析
            parsed = df['features_json'].apply(json.loads)
            features_all = pd.DataFrame(parsed.tolist())
            features_all['code'] = df['code'].values
            features_all['trade_date'] = df['trade_date'].values

            for date, group in features_all.groupby('trade_date'):
                self._feature_cache[date] = group.drop(columns=['trade_date']).reset_index(drop=True)
                total += len(group)

        conn.close()
        print(f"Alpha158 特征预加载: {total:,} 条, {len(self._feature_cache)} 天")

    def predict_scores(self, codes: List[str], date: str) -> Dict[str, Dict]:
        """对指定股票预测评分

        Args:
            codes: 股票代码列表
            date: 交易日期

        Returns:
            {code: {score, pred_3d, pred_5d, pred_10d}}
        """
        # 尝试从缓存获取
        features_df = self._feature_cache.get(date)
        if features_df is None:
            features_df = self._load_features_from_db(date)

        if features_df is None or features_df.empty:
            return {}

        # 过滤到请求的股票
        if codes:
            features_df = features_df[features_df['code'].isin(codes)]

        return self._score_dataframe(features_df)

    def predict_scores_from_preloaded(self, codes: List[str], date: str,
                                       features_df: pd.DataFrame) -> Dict[str, Dict]:
        """使用预加载的特征 DataFrame 评分 (batch_generate 兼容)"""
        if features_df is None or features_df.empty:
            return {}
        return self._score_dataframe(features_df)

    def _score_dataframe(self, features_df: pd.DataFrame) -> Dict[str, Dict]:
        """对一个 DataFrame 做评分"""
        if not self.feature_cols:
            return {}

        df = features_df.copy()
        codes = df['code'].tolist()

        # 准备特征矩阵
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_cols].fillna(0).values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # RobustZScore
        if self.scaler is not None:
            X = self.scaler.transform(X)
            X = np.clip(X, -5, 5)
            X = np.nan_to_num(X, nan=0.0)

        # 预测各目标
        predictions = {}
        import xgboost as xgb
        for target in ['3d', '5d', '10d']:
            if target not in self.models:
                predictions[target] = np.zeros(len(X))
                continue

            target_models = self.models[target]
            if not isinstance(target_models, dict):
                predictions[target] = np.zeros(len(X))
                continue

            target_weights = self.weights.get(f'label_{target}', {})
            pred = np.zeros(len(X))
            total_w = 0

            for name, model in target_models.items():
                w = target_weights.get(name, 1.0 / len(target_models))
                try:
                    if hasattr(model, 'predict'):
                        p = model.predict(X)
                    else:
                        p = model.predict(xgb.DMatrix(X))
                    pred += w * p
                    total_w += w
                except Exception:
                    continue

            if total_w > 0:
                pred /= total_w
            predictions[target] = pred

        # 综合分数
        combined = np.zeros(len(X))
        for target, w in [('3d', 0.35), ('5d', 0.35), ('10d', 0.30)]:
            combined += w * predictions.get(target, np.zeros(len(X)))

        # 百分位排名 -> 30~90 分
        if len(combined) > 1:
            from scipy import stats as sp_stats
            ranks = sp_stats.rankdata(combined)
            percentiles = (ranks - 1) / (len(ranks) - 1) * 100
            scores = 30 + percentiles * 0.6
        else:
            scores = np.array([60.0])

        results = {}
        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]),
                'pred_5d': float(predictions['5d'][i]),
                'pred_10d': float(predictions['10d'][i]),
            }

        return results

    def _load_features_from_db(self, date: str) -> Optional[pd.DataFrame]:
        """从数据库加载单日特征"""
        conn = sqlite3.connect(str(self.db_path))
        query = """
        SELECT code, features_json
        FROM alpha158_feature_cache
        WHERE trade_date = ?
        """
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()

        if df.empty:
            return None

        parsed = df['features_json'].apply(json.loads)
        features_df = pd.DataFrame(parsed.tolist())
        features_df['code'] = df['code'].values
        return features_df
