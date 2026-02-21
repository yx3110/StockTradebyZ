#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9.4生产版评分系统
- 使用训练好的v394_full_model.pkl模型
- 48个特征 = 42个v3.9.0基础特征 + 6个活跃市值特征
- IC提升166% (0.05 → 0.14), Top20胜率+8.93%

活跃市值特征:
- market_active_mv_ratio: 市场活跃市值比率
- market_active_mv_zscore: 市场活跃市值Z-score
- market_active_mv_trend: 市场活跃市值趋势
- stock_active_mv_rank: 个股活跃市值排名
- stock_relative_liquidity: 相对流动性
- market_cap_quality_score: 市值质量分(小市值惩罚)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sqlite3
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# V390EnhancedFeatureMLSystem用于实时特征计算（延迟初始化避免循环导入）
_v390_system_instance = None

def _get_v390_system():
    """获取V390增强特征系统实例（用于实时特征计算）"""
    global _v390_system_instance
    if _v390_system_instance is None:
        from .v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem
        _v390_system_instance = V390EnhancedFeatureMLSystem(lookback_days=10, lookahead_days=5)
    return _v390_system_instance


class V394ProductionScorer:
    """V3.9.4生产版评分系统 - 带活跃市值特征"""

    def __init__(self, model_path: str = None, db_path: str = None):
        """
        初始化V3.9.4评分系统

        Args:
            model_path: 模型文件路径
            db_path: 数据库路径
        """
        # 确定项目根目录
        self.project_root = Path(__file__).parent.parent.parent

        # 设置模型路径
        if model_path is None:
            model_path = str(self.project_root / 'ml_models' / 'trained_models' / 'v394' / 'v394_full_model.pkl')
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.n_features = 48  # 42 + 6
        self.model_info = None

        # 活跃市值特征名
        self.active_mv_features = [
            'market_active_mv_ratio',
            'market_active_mv_zscore',
            'market_active_mv_trend',
            'stock_active_mv_rank',
            'stock_relative_liquidity',
            'market_cap_quality_score'
        ]

        # 加载模型
        self._load_model()

        # 数据库路径 - 使用绝对路径
        if db_path is None:
            db_path = str(self.project_root / 'data_adapter' / 'stock_data.db')
        self.db_path = db_path

        logger.info("V3.9.4生产版评分系统初始化完成")
        logger.info(f"   模型: {model_path}")
        logger.info(f"   特征数: {self.n_features} (42基础 + 6活跃市值)")

    def _load_model(self):
        """加载训练好的模型"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        model_data = joblib.load(self.model_path)

        self.model = model_data['model']
        self.feature_names = model_data['feature_cols']
        self.n_features = len(self.feature_names)
        self.model_info = model_data

        logger.info(f"加载模型成功: {self.model_path}")
        logger.info(f"   特征数: {self.n_features}")

    def extract_features(self, code: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        提取股票的48个特征 (42基础 + 6活跃市值)

        Args:
            code: 股票代码
            trade_date: 交易日期

        Returns:
            特征DataFrame (1行48列) 或 None
        """
        conn = sqlite3.connect(self.db_path)

        try:
            # 1. 先从v39_feature_cache获取42个基础特征
            base_features = self._get_v39_features(conn, code, trade_date)

            # 如果缓存缺失，使用V390EnhancedFeatureMLSystem实时计算42个基础特征
            if base_features is None:
                try:
                    v390_system = _get_v390_system()
                    v390_features_df = v390_system.extract_features(code, trade_date)
                    if v390_features_df is not None and not v390_features_df.empty:
                        base_features = v390_features_df.iloc[0].to_dict()
                        logger.debug(f"V3.9.4使用V390EnhancedSystem实时计算特征: {code}")
                except Exception as e:
                    logger.warning(f"V390EnhancedSystem实时特征计算失败 {code}: {e}")

            if base_features is None:
                return None

            # 2. 从active_mv_feature_cache获取6个活跃市值特征
            active_mv_features = self._get_active_mv_features(conn, code, trade_date)

            # 3. 合并特征
            if active_mv_features is not None:
                for feat, value in active_mv_features.items():
                    base_features[feat] = value
            else:
                # 如果缓存没有,实时计算
                active_mv_features = self._calculate_active_mv_features(conn, code, trade_date)
                if active_mv_features:
                    for feat, value in active_mv_features.items():
                        base_features[feat] = value

            # 确保所有特征都存在
            for feat in self.feature_names:
                if feat not in base_features:
                    base_features[feat] = 0.0

            # 转换为DataFrame
            feature_df = pd.DataFrame([base_features])[self.feature_names]

            return feature_df

        except Exception as e:
            logger.error(f"特征提取错误 {code}: {e}")
            return None
        finally:
            conn.close()

    def _get_v39_features(self, conn, code: str, trade_date: str) -> Optional[Dict]:
        """从v39_feature_cache获取42个基础特征"""
        import json

        cursor = conn.execute("""
            SELECT features_json FROM v39_feature_cache
            WHERE code = ? AND trade_date = ?
        """, (code, trade_date))

        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except:
                pass

        # 如果缓存没有,尝试找最近的日期
        cursor = conn.execute("""
            SELECT features_json, trade_date FROM v39_feature_cache
            WHERE code = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 1
        """, (code, trade_date))

        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except:
                pass

        return None

    def _get_active_mv_features(self, conn, code: str, trade_date: str) -> Optional[Dict]:
        """从active_mv_feature_cache获取6个活跃市值特征"""
        cursor = conn.execute("""
            SELECT market_active_mv_ratio, market_active_mv_zscore, market_active_mv_trend,
                   stock_active_mv_rank, stock_relative_liquidity, market_cap_quality_score
            FROM active_mv_feature_cache
            WHERE code = ? AND trade_date = ?
        """, (code, trade_date))

        row = cursor.fetchone()
        if row:
            return {
                'market_active_mv_ratio': row[0] or 0.0,
                'market_active_mv_zscore': row[1] or 0.0,
                'market_active_mv_trend': row[2] or 0.0,
                'stock_active_mv_rank': row[3] or 0.5,
                'stock_relative_liquidity': row[4] or 0.5,
                'market_cap_quality_score': row[5] or 0.5
            }

        # 尝试找最近的日期
        cursor = conn.execute("""
            SELECT market_active_mv_ratio, market_active_mv_zscore, market_active_mv_trend,
                   stock_active_mv_rank, stock_relative_liquidity, market_cap_quality_score
            FROM active_mv_feature_cache
            WHERE code = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 1
        """, (code, trade_date))

        row = cursor.fetchone()
        if row:
            return {
                'market_active_mv_ratio': row[0] or 0.0,
                'market_active_mv_zscore': row[1] or 0.0,
                'market_active_mv_trend': row[2] or 0.0,
                'stock_active_mv_rank': row[3] or 0.5,
                'stock_relative_liquidity': row[4] or 0.5,
                'market_cap_quality_score': row[5] or 0.5
            }

        return None

    def _calculate_active_mv_features(self, conn, code: str, trade_date: str) -> Optional[Dict]:
        """实时计算活跃市值特征(当缓存不可用时)"""
        try:
            # 获取股票的流通市值和换手率
            cursor = conn.execute("""
                SELECT db.circ_mv, db.turnover_rate
                FROM daily_basic db
                JOIN securities s ON db.security_id = s.id
                WHERE s.code = ? AND db.trade_date = ?
            """, (code, trade_date))

            row = cursor.fetchone()
            if not row or row[0] is None or row[1] is None:
                return None

            circ_mv = row[0]  # 万元
            turnover_rate = row[1]  # %

            circ_mv_yi = circ_mv / 10000  # 亿元
            stock_active_mv = circ_mv * turnover_rate / 100  # 万元

            # 市值质量分 (sigmoid惩罚小市值)
            market_cap_quality_score = 1 / (1 + np.exp(
                -(np.log(max(circ_mv_yi, 0.1)) - np.log(50)) / 0.8
            ))

            # 获取市场平均活跃市值
            cursor = conn.execute("""
                SELECT AVG(db.circ_mv * db.turnover_rate / 100) as avg_active_mv
                FROM daily_basic db
                JOIN securities s ON db.security_id = s.id
                WHERE s.type = 'A股'
                  AND db.circ_mv IS NOT NULL
                  AND db.turnover_rate IS NOT NULL
                  AND db.trade_date = ?
            """, (trade_date,))

            row = cursor.fetchone()
            avg_active_mv = row[0] if row and row[0] else stock_active_mv

            # 相对流动性
            if avg_active_mv > 0:
                stock_relative_liquidity = np.tanh(stock_active_mv / avg_active_mv / 2)
            else:
                stock_relative_liquidity = 0.5

            return {
                'market_active_mv_ratio': 0.05,  # 默认值
                'market_active_mv_zscore': 0.0,  # 默认值
                'market_active_mv_trend': 0.0,   # 默认值
                'stock_active_mv_rank': 0.5,     # 默认值
                'stock_relative_liquidity': stock_relative_liquidity,
                'market_cap_quality_score': market_cap_quality_score
            }

        except Exception as e:
            logger.warning(f"实时计算活跃市值特征失败 {code}: {e}")
            return None

    def predict_score(self, code: str, trade_date: str) -> Optional[Dict]:
        """
        预测单只股票的评分

        Args:
            code: 股票代码
            trade_date: 交易日期

        Returns:
            评分结果字典
        """
        # 提取特征
        features = self.extract_features(code, trade_date)

        if features is None:
            logger.warning(f"无法提取特征: {code}")
            return None

        # 处理缺失值
        features = features.fillna(0)

        # 预测
        try:
            prediction = self.model.predict(features)[0]

            # 转换为0-100评分
            score = self._convert_prediction_to_score(prediction)

            return {
                'code': code,
                'trade_date': trade_date,
                'score': score,
                'predicted_return_5d': prediction,
                'confidence': self._calculate_confidence(features, prediction),
                'recommendation': self._get_recommendation(score),
                'scoring_method': 'V3.9.4_Production',
                'model_grade': 'A+',
                'model_ic': 0.1363,  # 13.63% IC
                'model_top20_win_rate': 0.5643  # 56.43%
            }

        except Exception as e:
            logger.error(f"预测错误 {code}: {e}")
            return None

    def predict_scores(self, codes: List[str], trade_date: str) -> Dict[str, Dict]:
        """
        批量预测多只股票

        Args:
            codes: 股票代码列表
            trade_date: 交易日期

        Returns:
            {code: 评分结果} 字典
        """
        results = {}

        for code in codes:
            result = self.predict_score(code, trade_date)
            if result:
                results[code] = result

        return results

    def predict_scores_with_ranking(self, codes: List[str], trade_date: str) -> Dict[str, Dict]:
        """
        批量预测并使用百分位排名评分（解决评分集中问题）

        使用百分位排名将预测值映射到0-100分，确保评分有良好的区分度。

        Args:
            codes: 股票代码列表
            trade_date: 交易日期

        Returns:
            {code: 评分结果} 字典，包含ranked_score字段
        """
        from scipy import stats

        # 第一遍：收集所有预测值
        predictions = {}
        features_cache = {}

        for code in codes:
            features = self.extract_features(code, trade_date)
            if features is not None and not features.empty:
                try:
                    # 处理数据类型
                    features = features.apply(pd.to_numeric, errors='coerce').fillna(0)
                    pred = float(self.model.predict(features)[0])
                    predictions[code] = pred
                    features_cache[code] = features
                except Exception as e:
                    logger.debug(f"预测失败 {code}: {e}")
                    continue

        if not predictions:
            return {}

        # 计算百分位排名 (0-100)
        pred_values = list(predictions.values())
        pred_codes = list(predictions.keys())

        # 使用scipy.stats.rankdata计算排名，然后转换为百分位
        ranks = stats.rankdata(pred_values, method='average')
        percentiles = (ranks - 1) / (len(ranks) - 1) * 100 if len(ranks) > 1 else [50.0] * len(ranks)

        # 构建结果
        results = {}
        for i, code in enumerate(pred_codes):
            pred = predictions[code]
            percentile_score = percentiles[i]

            # 映射到30-90分范围（避免极端值）
            # 百分位0% -> 30分, 百分位100% -> 90分
            ranked_score = 30 + percentile_score * 0.6

            # 原始评分（保留用于对比）
            original_score = self._convert_prediction_to_score(pred)

            # 根据百分位排名给出建议
            if percentile_score >= 80:
                recommendation = "强烈买入"
            elif percentile_score >= 60:
                recommendation = "买入"
            elif percentile_score >= 40:
                recommendation = "持有观望"
            elif percentile_score >= 20:
                recommendation = "谨慎"
            else:
                recommendation = "回避"

            # 置信度基于特征质量
            features = features_cache[code]
            confidence = self._calculate_confidence(features, pred)

            results[code] = {
                'code': code,
                'trade_date': trade_date,
                'score': ranked_score,  # 使用排名评分作为主评分
                'original_score': original_score,
                'percentile_rank': percentile_score,
                'predicted_return_5d': pred,
                'confidence': confidence,
                'recommendation': recommendation,
                'scoring_method': 'V3.9.4_Percentile_Ranking',
                'model_grade': 'A+',
                'model_ic': 0.1363,
                'model_top20_win_rate': 0.5643
            }

        logger.info(f"百分位排名评分完成: {len(results)}只股票, 分数范围{min(r['score'] for r in results.values()):.1f}-{max(r['score'] for r in results.values()):.1f}")

        return results

    def _convert_prediction_to_score(self, prediction: float) -> float:
        """
        将5日收益率预测转换为0-100评分

        预测值分布: -10% ~ +10%
        映射到: 0 ~ 100分
        """
        # 截断到-15% ~ +15%
        prediction = np.clip(prediction, -0.15, 0.15)

        # 线性映射
        score = (prediction + 0.15) / 0.30 * 100

        return np.clip(score, 0, 100)

    def _calculate_confidence(self, features: pd.DataFrame, prediction: float) -> float:
        """
        计算预测置信度

        基于特征质量和预测强度
        """
        # 特征缺失率
        missing_rate = features.isna().sum().sum() / (features.shape[0] * features.shape[1])
        feature_quality = 1.0 - missing_rate

        # 预测强度 (离0越远置信度越高)
        prediction_strength = min(abs(prediction) / 0.10, 1.0)

        # 综合置信度
        confidence = (feature_quality * 0.4 + prediction_strength * 0.6)

        return np.clip(confidence, 0.3, 0.95)

    def _get_recommendation(self, score: float) -> str:
        """根据评分给出投资建议

        V3.9.4优化阈值 (基于IC=0.1363, Top20胜率=56.43%)
        """
        if score >= 68:
            return "强烈买入"
        elif score >= 64:
            return "买入"
        elif score >= 60:
            return "谨慎买入"
        elif score >= 55:
            return "持有观望"
        elif score >= 50:
            return "谨慎卖出"
        else:
            return "卖出"


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scorer = V394ProductionScorer()

    # 测试单只股票
    result = scorer.predict_score('000001', '2025-10-28')
    print("\n测试结果:")
    print(result)
