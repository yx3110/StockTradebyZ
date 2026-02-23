#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V4.0/V4.2 Cross-Sectional Alpha Model 生产评分器

核心改进:
- 从 v40_feature_cache 读取 cross-sectional 排名特征
- 使用当天全部已评分股票的 percentile rank 映射到 30-90 分
- 预测超额收益 (个股 - 沪深300)
- 接口与 V390ProductionScorer 完全一致

V4.2 新增:
- 自动检测v42模型，启用v39市场特征注入 + robust z-score归一化
- 动态扩展MARKET_FEATURES以包含v39_*特征
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class V400ProductionScorer:
    """V4.0 Cross-Sectional Alpha Model 生产评分器"""

    # 市场级特征 (推理时应用缩放)
    MARKET_FEATURES = {
        'market_regime', 'market_vol_regime', 'market_breadth_5d',
        'northbound_flow_zscore', 'market_volume_regime', 'market_trend_strength'
    }

    # 行业级特征
    INDUSTRY_FEATURES = {
        'sw_l1_code', 'industry_breadth', 'industry_volume_change',
        'industry_kdj_avg', 'industry_macd_bullish_pct',
        'industry_concentration', 'industry_momentum_rank',
        'industry_rotation_signal'
    }

    def __init__(self, model_path: str = None, db_path: str = None):
        self.project_root = Path(__file__).parent.parent.parent

        if model_path is None:
            model_path = str(self.project_root / 'ml_models' / 'trained_models' / 'v400' / 'v400_full_system_latest.pkl')
        self.model_path = model_path

        if db_path is None:
            db_path = str(self.project_root / 'data_adapter' / 'stock_data.db')
        self.db_path = db_path

        # 模型组件
        self.base_models = None
        self.meta_model = None
        self.feature_names = None
        self.winsorize_bounds = None
        self.n_features = 0

        # V4.2 字段
        self.v42_mode = False
        self.join_v39_market = False
        self.robust_zscore_features = False
        self.v39_market_feature_names = []

        self._load_model()

        version_str = "V4.2 Hybrid Alpha" if self.v42_mode else "V4.0 Cross-Sectional"
        logger.info(f"✅ {version_str}评分系统初始化完成")
        logger.info(f"   模型: {model_path}")
        logger.info(f"   特征数: {self.n_features}")
        if self.v42_mode:
            logger.info(f"   V4.2: v39市场特征={self.join_v39_market}, robust_zscore={self.robust_zscore_features}")

    def _load_model(self):
        """加载训练好的模型"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.base_models = model_data['base_models']
        self.meta_model = model_data['meta_model']
        self.feature_names = model_data.get('feature_names')
        self.winsorize_bounds = model_data.get('winsorize_bounds')

        # v4.0.1 新字段 (向后兼容旧模型)
        self.meta_model_type = model_data.get('meta_model_type', 'gbm')
        self.market_scale = model_data.get('market_scale', 1.0)
        self.industry_scale = model_data.get('industry_scale', 1.0)

        # V4.2 字段检测
        self.v42_mode = model_data.get('v42_mode', False)
        self.join_v39_market = model_data.get('join_v39_market', False)
        self.robust_zscore_features = model_data.get('robust_zscore_features', False)
        self.v39_market_feature_names = model_data.get('v39_market_feature_names', [])

        # V4.2: 动态扩展MARKET_FEATURES
        if self.v39_market_feature_names:
            self.MARKET_FEATURES = set(self.MARKET_FEATURES) | set(self.v39_market_feature_names)

        # 从第一个基础模型推断特征名
        if self.feature_names is None:
            first_model = list(self.base_models.values())[0]
            if hasattr(first_model, 'feature_name_'):
                self.feature_names = first_model.feature_name_
            elif hasattr(first_model, 'feature_names_in_'):
                self.feature_names = list(first_model.feature_names_in_)

        self.n_features = len(self.feature_names) if self.feature_names else 0

        version = model_data.get('version', 'unknown')
        logger.info(f"✅ 加载V4.0模型 (version={version}): {list(self.base_models.keys())}, "
                     f"特征数={self.n_features}, meta={self.meta_model_type}, "
                     f"market_scale={self.market_scale}, industry_scale={self.industry_scale}")
        if self.v42_mode:
            logger.info(f"   V4.2模式: v39_market={self.join_v39_market}, "
                         f"robust_zscore={self.robust_zscore_features}, "
                         f"v39特征={len(self.v39_market_feature_names)}个")

    def _load_v39_market_for_date(self, trade_date: str) -> Optional[Dict]:
        """
        V4.2: 从v39_feature_cache加载1行市场特征

        Returns:
            dict {v39_market_return_5d: ..., ...} or None
        """
        if not self.join_v39_market or not self.v39_market_feature_names:
            return None

        conn = sqlite3.connect(self.db_path)
        try:
            v39_cols_raw = [name.replace('v39_', '') for name in self.v39_market_feature_names]
            cols_sql = ', '.join(v39_cols_raw)

            cursor = conn.execute(f"""
                SELECT {cols_sql} FROM v39_feature_cache
                WHERE trade_date = ?
                LIMIT 1
            """, (trade_date,))
            row = cursor.fetchone()
            if not row:
                return None

            result = {}
            for i, col_raw in enumerate(v39_cols_raw):
                result[f'v39_{col_raw}'] = float(row[i]) if row[i] is not None else 0.0
            return result
        except Exception as e:
            logger.debug(f"V4.2加载v39市场特征失败 {trade_date}: {e}")
            return None
        finally:
            conn.close()

    def _load_v39_market_batch(self, dates: List[str]) -> Dict[str, Dict]:
        """
        V4.2: 批量加载多日期的v39市场特征

        Returns:
            {date: {v39_market_return_5d: ..., ...}}
        """
        if not self.join_v39_market or not self.v39_market_feature_names:
            return {}

        conn = sqlite3.connect(self.db_path)
        try:
            v39_cols_raw = [name.replace('v39_', '') for name in self.v39_market_feature_names]
            cols_sql = ', '.join(v39_cols_raw)
            placeholders = ','.join(['?' for _ in dates])

            df = pd.read_sql_query(f"""
                SELECT trade_date, {cols_sql} FROM v39_feature_cache
                WHERE trade_date IN ({placeholders})
                GROUP BY trade_date
            """, conn, params=dates)
            conn.close()

            result = {}
            for _, row in df.iterrows():
                date = row['trade_date']
                result[date] = {}
                for col_raw in v39_cols_raw:
                    val = row[col_raw]
                    result[date][f'v39_{col_raw}'] = float(val) if val is not None else 0.0

            return result
        except Exception as e:
            logger.debug(f"V4.2批量加载v39市场特征失败: {e}")
            try:
                conn.close()
            except Exception:
                pass
            return {}

    def _inject_v39_market_features(self, features_df: pd.DataFrame, trade_date: str,
                                      v39_cache: Dict = None) -> pd.DataFrame:
        """
        V4.2: 向特征DataFrame注入v39市场特征，删除旧V40分类市场特征

        Args:
            features_df: 含code列+特征列的DataFrame
            trade_date: 交易日期
            v39_cache: 可选的预加载缓存 {date: {feat: val}}
        """
        if not self.join_v39_market:
            return features_df

        # 获取v39市场特征
        if v39_cache and trade_date in v39_cache:
            v39_data = v39_cache[trade_date]
        else:
            v39_data = self._load_v39_market_for_date(trade_date)

        if not v39_data:
            # 填0
            for col in self.v39_market_feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0.0
            return features_df

        # 注入v39特征 (所有股票共享同一市场值)
        for col, val in v39_data.items():
            features_df[col] = val

        # 删除V40分类市场特征
        v40_market_cat = ['market_regime', 'market_vol_regime', 'market_breadth_5d',
                          'northbound_flow_zscore', 'market_volume_regime', 'market_trend_strength']
        for col in v40_market_cat:
            if col in features_df.columns and col not in (self.feature_names or []):
                features_df = features_df.drop(columns=[col], errors='ignore')

        return features_df

    def _apply_robust_zscore(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        V4.2: 对当日全市场特征矩阵应用cross-sectional robust z-score

        Args:
            features_df: 特征DataFrame (不含code列)
        """
        if not self.robust_zscore_features:
            return features_df

        if len(features_df) < 10:
            return features_df

        categorical_cols = {'sw_l1_code'}
        features_df = features_df.copy()

        for col in features_df.columns:
            if col in categorical_cols or col == 'code':
                continue

            vals = features_df[col].values
            median = np.median(vals)
            mad = np.median(np.abs(vals - median))
            scale = 1.4826 * mad

            if scale < 1e-10:
                features_df[col] = 0.0
            else:
                features_df[col] = np.clip((vals - median) / scale, -3, 3)

        return features_df

    def _apply_feature_scaling(self, features):
        """应用市场/行业特征缩放 (与训练时一致)"""
        if self.market_scale == 1.0 and self.industry_scale == 1.0:
            return features

        features = features.copy()
        for col in features.columns:
            if col in self.MARKET_FEATURES:
                features[col] = features[col] * self.market_scale
            elif col in self.INDUSTRY_FEATURES:
                features[col] = features[col] * self.industry_scale
        return features

    def _predict_raw(self, features):
        """内部预测: 返回原始超额收益预测值"""
        # 应用市场/行业特征缩放
        features = self._apply_feature_scaling(features)

        base_preds = np.column_stack([
            model.predict(features) for model in self.base_models.values()
        ])

        # 根据 meta_model_type 选择预测方式
        if self.meta_model_type == 'avg' or self.meta_model is None:
            return np.mean(base_preds, axis=1)
        else:
            return self.meta_model.predict(base_preds)

    def _get_features_from_cache(self, code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """从 v40_feature_cache 读取预计算特征"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("""
                SELECT features_json FROM v40_feature_cache
                WHERE code = ? AND trade_date = ?
            """, (code, trade_date))
            row = cursor.fetchone()
            if not row or not row[0]:
                return None

            features_dict = json.loads(row[0])
            feature_df = pd.DataFrame([features_dict])

            # 对齐列顺序
            if self.feature_names:
                for col in self.feature_names:
                    if col not in feature_df.columns:
                        feature_df[col] = 0
                feature_df = feature_df[self.feature_names]

            # Winsorization
            if self.winsorize_bounds:
                for col, (lower, upper) in self.winsorize_bounds.items():
                    if col in feature_df.columns:
                        feature_df[col] = feature_df[col].clip(lower, upper)

            return feature_df
        except Exception as e:
            logger.debug(f"从V4.0缓存读取特征失败 {code}: {e}")
            return None
        finally:
            conn.close()

    def _get_features_from_cache_batch(self, codes: List[str], trade_date: str) -> Dict[str, Optional[pd.DataFrame]]:
        """批量从 v40_feature_cache 读取特征"""
        result = {code: None for code in codes}
        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ','.join(['?' for _ in codes])
            cursor = conn.execute(f"""
                SELECT code, features_json FROM v40_feature_cache
                WHERE code IN ({placeholders}) AND trade_date = ?
            """, codes + [trade_date])

            for code, features_json in cursor.fetchall():
                try:
                    features_dict = json.loads(features_json)
                    feature_df = pd.DataFrame([features_dict])

                    if self.feature_names:
                        for col in self.feature_names:
                            if col not in feature_df.columns:
                                feature_df[col] = 0
                        feature_df = feature_df[self.feature_names]

                    if self.winsorize_bounds:
                        for col, (lower, upper) in self.winsorize_bounds.items():
                            if col in feature_df.columns:
                                feature_df[col] = feature_df[col].clip(lower, upper)

                    result[code] = feature_df
                except Exception as e:
                    logger.debug(f"解析V4.0缓存失败 {code}: {e}")
        except Exception as e:
            logger.warning(f"批量V4.0缓存读取失败: {e}")
        finally:
            conn.close()

        return result

    def _convert_prediction_to_score(self, prediction: float,
                                       all_predictions: Optional[np.ndarray] = None) -> float:
        """
        使用percentile rank映射到30-90分

        如果提供了all_predictions (当天全部评分), 使用percentile rank
        否则回退到线性映射
        """
        if all_predictions is not None and len(all_predictions) > 10:
            # Percentile rank映射
            rank = np.mean(all_predictions <= prediction)
            score = 30 + rank * 60  # 映射到 30-90
        else:
            # 线性映射回退
            prediction = np.clip(prediction, -0.10, 0.10)
            score = (prediction + 0.10) / 0.20 * 60 + 30

        return float(np.clip(score, 30, 90))

    def predict_score(self, code: str, trade_date: str) -> Optional[Dict]:
        """预测单只股票的评分"""
        features = self._get_features_from_cache(code, trade_date)
        if features is None:
            logger.debug(f"V4.0缓存无数据: {code} {trade_date}")
            return None

        features = features.fillna(0)

        try:
            prediction = self._predict_raw(features)[0]
            score = self._convert_prediction_to_score(prediction)

            return {
                'code': code,
                'trade_date': trade_date,
                'score': score,
                'predicted_excess_return_5d': float(prediction),
                'confidence': self._calculate_confidence(features, prediction),
                'recommendation': self._get_recommendation(score),
                'scoring_method': 'V4.2_HybridAlpha' if self.v42_mode else 'V4.0_CrossSectional',
                'model_grade': 'TBD',
            }
        except Exception as e:
            logger.error(f"V4.0预测错误 {code}: {e}")
            return None

    def _load_all_features_for_date(self, trade_date: str) -> Optional[pd.DataFrame]:
        """
        加载当天全市场所有股票的特征 (用于计算全市场percentile rank)

        Returns:
            DataFrame with columns: code + feature columns, or None
        """
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query("""
                SELECT code, features_json FROM v40_feature_cache
                WHERE trade_date = ?
            """, conn, params=(trade_date,))

            if df.empty:
                logger.warning(f"V4.0 cache无当天数据: {trade_date}")
                return None

            features_list = []
            valid_codes = []
            for _, row in df.iterrows():
                try:
                    features_dict = json.loads(row['features_json'])
                    features_list.append(features_dict)
                    valid_codes.append(row['code'])
                except (json.JSONDecodeError, TypeError):
                    continue

            if not features_list:
                return None

            features_df = pd.DataFrame(features_list)
            features_df.insert(0, 'code', valid_codes)

            # 对齐列顺序 + winsorize
            if self.feature_names:
                for col in self.feature_names:
                    if col not in features_df.columns:
                        features_df[col] = 0

            if self.winsorize_bounds and not self.robust_zscore_features:
                for col, (lower, upper) in self.winsorize_bounds.items():
                    if col in features_df.columns:
                        features_df[col] = features_df[col].clip(lower, upper)

            # V4.2: 注入v39市场特征
            if self.v42_mode:
                features_df = self._inject_v39_market_features(features_df, trade_date)

            # V4.2: 全市场robust z-score
            if self.robust_zscore_features:
                code_col = features_df['code'].copy()
                feat_cols = [c for c in features_df.columns if c != 'code']
                features_df[feat_cols] = self._apply_robust_zscore(features_df[feat_cols])
                features_df['code'] = code_col

            logger.info(f"✅ V4.0全市场特征加载: {len(features_df)} 只股票 ({trade_date})")
            return features_df

        except Exception as e:
            logger.warning(f"V4.0全市场特征加载失败: {e}")
            return None
        finally:
            conn.close()

    def predict_scores(self, codes: List[str], trade_date: str) -> Dict[str, Dict]:
        """
        批量预测 (使用全市场percentile rank评分)

        关键改进: 先对全市场所有股票预测，再用全市场percentile rank映射分数
        这确保排名含义与训练时一致 (全市场4000+只股票的相对位置)
        """
        if not codes:
            return {}

        # 加载全市场特征 (不仅仅是请求的codes)
        all_market_features = self._load_all_features_for_date(trade_date)

        if all_market_features is None or len(all_market_features) == 0:
            # 回退: 只使用请求的codes
            logger.warning("V4.0全市场特征加载失败，回退到局部排名模式")
            return self._predict_scores_local(codes, trade_date)

        try:
            market_codes = all_market_features['code'].values
            feature_cols = self.feature_names if self.feature_names else [
                c for c in all_market_features.columns if c != 'code']
            X_all = all_market_features[feature_cols].fillna(0)

            # 全市场预测
            all_predictions = self._predict_raw(X_all)
            logger.info(f"  全市场预测完成: {len(all_predictions)} 只, "
                        f"均值={np.mean(all_predictions):.5f}, "
                        f"std={np.std(all_predictions):.5f}")

            # 构建 code -> prediction 映射
            pred_map = dict(zip(market_codes, all_predictions))

            # 对请求的codes提取结果，使用全市场percentile rank
            results = {}
            for code in codes:
                if code not in pred_map:
                    continue

                pred = pred_map[code]
                score = self._convert_prediction_to_score(pred, all_predictions)

                # 从缓存的features构建confidence
                code_idx = np.where(market_codes == code)[0]
                if len(code_idx) > 0:
                    feat_row = all_market_features.iloc[[code_idx[0]]][feature_cols]
                    confidence = self._calculate_confidence(feat_row, pred)
                else:
                    confidence = 0.5

                results[code] = {
                    'code': code,
                    'trade_date': trade_date,
                    'score': score,
                    'predicted_excess_return_5d': float(pred),
                    'confidence': confidence,
                    'recommendation': self._get_recommendation(score),
                    'scoring_method': 'V4.2_HybridAlpha' if self.v42_mode else 'V4.0_CrossSectional',
                    'model_grade': 'TBD',
                }

            logger.info(f"  返回 {len(results)}/{len(codes)} 只请求股票的评分 "
                        f"(全市场 {len(all_predictions)} 只参与排名)")
            return results

        except Exception as e:
            logger.error(f"V4.0全市场predict失败: {e}")
            return self._predict_scores_local(codes, trade_date)

    def _predict_scores_local(self, codes: List[str], trade_date: str) -> Dict[str, Dict]:
        """局部排名模式回退 (仅在全市场加载失败时使用)"""
        batch_features = self._get_features_from_cache_batch(codes, trade_date)

        valid_codes = []
        feature_rows = []
        for code in codes:
            feat = batch_features.get(code)
            if feat is not None:
                valid_codes.append(code)
                feature_rows.append(feat)

        if not valid_codes:
            return {}

        try:
            all_features = pd.concat(feature_rows, ignore_index=True)
            all_features = all_features.fillna(0)
            predictions = self._predict_raw(all_features)

            results = {}
            for i, code in enumerate(valid_codes):
                pred = predictions[i]
                score = self._convert_prediction_to_score(pred, predictions)
                results[code] = {
                    'code': code,
                    'trade_date': trade_date,
                    'score': score,
                    'predicted_excess_return_5d': float(pred),
                    'confidence': self._calculate_confidence(feature_rows[i], pred),
                    'recommendation': self._get_recommendation(score),
                    'scoring_method': 'V4.2_HybridAlpha' if self.v42_mode else 'V4.0_CrossSectional',
                    'model_grade': 'TBD',
                }
            return results
        except Exception as e:
            logger.error(f"V4.0局部predict失败: {e}")
            return {}

    def preload_feature_cache(self, dates: List[str]) -> Dict[str, pd.DataFrame]:
        """
        批量预加载多个日期的特征缓存

        Returns:
            {date: features_DataFrame} 字典，每个 DataFrame 含 code 列和特征列
        """
        result = {d: None for d in dates}
        if not dates:
            return result

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ','.join(['?' for _ in dates])
            df = pd.read_sql_query(f"""
                SELECT code, trade_date, features_json FROM v40_feature_cache
                WHERE trade_date IN ({placeholders})
            """, conn, params=dates)
            conn.close()

            if df.empty:
                return result

            for date, date_df in df.groupby('trade_date'):
                features_list = []
                valid_codes = []

                for _, row in date_df.iterrows():
                    try:
                        features = json.loads(row['features_json'])
                        features_list.append(features)
                        valid_codes.append(row['code'])
                    except (json.JSONDecodeError, TypeError):
                        continue

                if not features_list:
                    continue

                features_df = pd.DataFrame(features_list)
                features_df['code'] = valid_codes

                if self.feature_names:
                    for col in self.feature_names:
                        if col not in features_df.columns:
                            features_df[col] = 0

                if self.winsorize_bounds and not self.robust_zscore_features:
                    for col, (lower, upper) in self.winsorize_bounds.items():
                        if col in features_df.columns:
                            features_df[col] = features_df[col].clip(lower, upper)

                result[date] = features_df

            # V4.2: 批量注入v39市场特征 + robust z-score
            if self.v42_mode:
                v39_cache = self._load_v39_market_batch(dates)
                for date in dates:
                    if result[date] is not None:
                        result[date] = self._inject_v39_market_features(
                            result[date], date, v39_cache)
                        if self.robust_zscore_features:
                            code_col = result[date]['code'].copy()
                            feat_cols = [c for c in result[date].columns if c != 'code']
                            result[date][feat_cols] = self._apply_robust_zscore(result[date][feat_cols])
                            result[date]['code'] = code_col

            total = sum(len(v) for v in result.values() if v is not None)
            logger.info(f"✅ V4.0特征缓存预加载: {len(dates)}天, {total}条记录")

        except Exception as e:
            logger.warning(f"V4.0特征缓存预加载失败: {e}")
            try:
                conn.close()
            except Exception:
                pass

        return result

    def predict_scores_from_preloaded(self, codes: List[str], trade_date: str,
                                       preloaded_features) -> Dict[str, Dict]:
        """
        使用预加载的特征进行批量预测

        关键改进: 对preloaded_features中的全部股票做预测(而非仅codes),
        然后用全市场percentile rank映射分数
        """
        if not codes:
            return {}

        results = {}

        if preloaded_features is not None and len(preloaded_features) > 0:
            try:
                if self.feature_names:
                    feature_cols = self.feature_names
                else:
                    feature_cols = [c for c in preloaded_features.columns if c != 'code']

                # 对全部预加载股票做预测 (用于全市场percentile rank)
                all_features = preloaded_features[feature_cols].fillna(0)
                all_predictions = self._predict_raw(all_features)
                all_codes = preloaded_features['code'].values

                logger.info(f"  预加载全市场预测: {len(all_predictions)} 只, "
                            f"均值={np.mean(all_predictions):.5f}")

                # 构建映射
                pred_map = dict(zip(all_codes, all_predictions))

                # 对请求的codes提取结果
                for code in codes:
                    if code in pred_map:
                        pred = pred_map[code]
                        score = self._convert_prediction_to_score(pred, all_predictions)

                        code_mask = preloaded_features['code'] == code
                        if code_mask.any():
                            feat_row = preloaded_features.loc[code_mask, feature_cols].iloc[[0]]
                            confidence = self._calculate_confidence(feat_row, pred)
                        else:
                            confidence = 0.5

                        results[code] = {
                            'code': code,
                            'trade_date': trade_date,
                            'score': score,
                            'predicted_excess_return_5d': float(pred),
                            'confidence': confidence,
                            'recommendation': self._get_recommendation(score),
                            'scoring_method': 'V4.0_CrossSectional',
                            'model_grade': 'TBD',
                        }

            except Exception as e:
                logger.error(f"V4.0预加载批量predict失败: {e}")

        # 对缺失特征的股票走单只 fallback
        for code in codes:
            if code not in results:
                result = self.predict_score(code, trade_date)
                if result:
                    results[code] = result

        return results

    def _calculate_confidence(self, features: pd.DataFrame, prediction: float) -> float:
        """计算预测置信度"""
        missing_rate = features.isna().sum().sum() / max(features.shape[0] * features.shape[1], 1)
        feature_quality = 1.0 - missing_rate
        prediction_strength = min(abs(prediction) / 0.05, 1.0)
        confidence = feature_quality * 0.4 + prediction_strength * 0.6
        return float(np.clip(confidence, 0.3, 0.95))

    def _get_recommendation(self, score: float) -> str:
        """根据评分给出投资建议 (基于percentile分布)"""
        if score >= 80:
            return "强烈买入"
        elif score >= 72:
            return "买入"
        elif score >= 65:
            return "谨慎买入"
        elif score >= 55:
            return "持有观望"
        elif score >= 45:
            return "谨慎卖出"
        else:
            return "卖出"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scorer = V400ProductionScorer()
    result = scorer.predict_score('000001', '2026-02-13')
    print("\n测试结果:", result)
