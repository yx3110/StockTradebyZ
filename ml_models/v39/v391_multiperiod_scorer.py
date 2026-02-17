#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.91 多周期生产评分系统
综合5天、10天、15天收益预测，给出更全面的投资建议
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import sqlite3
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta

# 尝试导入实时特征计算器
try:
    from incremental_learning.features.realtime_calculator import RealtimeFeatureCalculator
    REALTIME_AVAILABLE = True
    print("✅ Phase 2 实时特征组件加载成功")
except ImportError:
    REALTIME_AVAILABLE = False

# 尝试导入增量学习组件
try:
    from incremental_learning.engines.incremental_engine import IncrementalLearningEngine
    INCREMENTAL_AVAILABLE = True
    print("✅ Phase 3 增量学习组件加载成功")
except ImportError:
    INCREMENTAL_AVAILABLE = False


class V391MultiPeriodScorer:
    """V3.91 多周期评分器"""

    # 多周期权重
    PERIOD_WEIGHTS = {
        '5d': 0.40,
        '10d': 0.35,
        '15d': 0.25
    }

    # 预测范围配置
    RETURN_RANGES = {
        '5d': (-0.15, 0.15),    # 5天：±15%
        '10d': (-0.25, 0.25),   # 10天：±25%
        '15d': (-0.30, 0.30),   # 15天：±30%
    }

    def __init__(self,
                 model_path: str = 'models/v391/v391_multiperiod_latest.pkl',
                 db_path: str = 'data_adapter/stock_data.db'):
        """
        初始化多周期评分器

        Args:
            model_path: 模型文件路径
            db_path: 数据库路径
        """
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.model_path = model_path

        # 加载模型
        self.models = None
        self.meta_models = None
        self.feature_columns = None
        self.period_weights = self.PERIOD_WEIGHTS

        self._load_model()

        # 初始化特征提取器
        self._init_feature_extractor()

    def _load_model(self):
        """加载多周期模型"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.models = model_data['base_models']
            self.meta_models = model_data['meta_models']
            self.feature_columns = model_data.get('feature_columns', [])
            self.period_weights = model_data.get('period_weights', self.PERIOD_WEIGHTS)

            self.logger.info(f"✅ V3.91模型加载成功: {self.model_path}")
            self.logger.info(f"   周期权重: 5d={self.period_weights['5d']}, "
                           f"10d={self.period_weights['10d']}, 15d={self.period_weights['15d']}")

        except FileNotFoundError:
            self.logger.warning(f"⚠️ 模型文件不存在: {self.model_path}")
            self.logger.warning("   请先运行 train_v391_multiperiod.py 训练模型")
        except Exception as e:
            self.logger.error(f"❌ 加载模型失败: {e}")

    def _init_feature_extractor(self):
        """初始化特征提取器（复用V3.9的特征提取）"""
        try:
            from ml_models.v39.v390_production_scorer import V390ProductionScorer
            self._v390_scorer = V390ProductionScorer()
            self.logger.info("✅ 特征提取器初始化成功（复用V3.90）")
        except Exception as e:
            self.logger.error(f"❌ 特征提取器初始化失败: {e}")
            self._v390_scorer = None

    def extract_features(self, code: str, date: str) -> Optional[pd.DataFrame]:
        """
        提取股票特征（复用V3.9特征提取）

        Args:
            code: 股票代码
            date: 日期 YYYY-MM-DD

        Returns:
            特征DataFrame或None
        """
        if self._v390_scorer is None:
            return None

        return self._v390_scorer.extract_features(code, date)

    def predict_returns(self, code: str, date: str) -> Optional[Dict[str, float]]:
        """
        预测多周期收益率

        Args:
            code: 股票代码
            date: 日期 YYYY-MM-DD

        Returns:
            {
                'return_5d': float,
                'return_10d': float,
                'return_15d': float,
            }
        """
        if self.models is None:
            return None

        # 提取特征
        features = self.extract_features(code, date)
        if features is None or features.empty:
            return None

        # 对齐特征列
        X = features.reindex(columns=self.feature_columns, fill_value=0)

        predictions = {}

        for period in ['5d', '10d', '15d']:
            try:
                # 基础模型预测
                base_preds = np.array([
                    model.predict(X)[0]
                    for model in self.models[period].values()
                ])

                # 元模型预测
                meta_features = base_preds.reshape(1, -1)
                pred = self.meta_models[period].predict(meta_features)[0]

                predictions[f'return_{period}'] = pred

            except Exception as e:
                self.logger.error(f"预测{period}失败: {e}")
                return None

        return predictions

    def _return_to_score(self, return_val: float, period: str) -> float:
        """
        将收益率预测转换为0-100评分

        Args:
            return_val: 预测收益率
            period: 周期 ('5d', '10d', '15d')

        Returns:
            0-100分
        """
        min_ret, max_ret = self.RETURN_RANGES[period]
        # 线性映射到0-100
        score = (return_val - min_ret) / (max_ret - min_ret) * 100
        return np.clip(score, 0, 100)

    def predict_score(self, code: str, date: str) -> Optional[Dict[str, Any]]:
        """
        综合多周期评分

        Args:
            code: 股票代码
            date: 日期 YYYY-MM-DD

        Returns:
            {
                'code': str,
                'date': str,
                'composite_score': float,      # 综合评分（0-100）
                'composite_return': float,     # 综合预期收益
                'score_5d': float,             # 5天评分
                'score_10d': float,            # 10天评分
                'score_15d': float,            # 15天评分
                'return_5d': float,            # 5天预期收益
                'return_10d': float,           # 10天预期收益
                'return_15d': float,           # 15天预期收益
                'trend_consistency': float,    # 趋势一致性 (-1到1)
                'recommendation': str,         # 投资建议
                'period_outlook': str,         # 多周期展望
            }
        """
        # 预测收益率
        returns = self.predict_returns(code, date)
        if returns is None:
            return None

        # 计算各周期评分
        score_5d = self._return_to_score(returns['return_5d'], '5d')
        score_10d = self._return_to_score(returns['return_10d'], '10d')
        score_15d = self._return_to_score(returns['return_15d'], '15d')

        # 计算综合评分（加权平均）
        composite_score = (
            self.period_weights['5d'] * score_5d +
            self.period_weights['10d'] * score_10d +
            self.period_weights['15d'] * score_15d
        )

        # 计算综合预期收益（加权平均）
        composite_return = (
            self.period_weights['5d'] * returns['return_5d'] +
            self.period_weights['10d'] * returns['return_10d'] +
            self.period_weights['15d'] * returns['return_15d']
        )

        # 趋势一致性分析
        trend_consistency = self._analyze_trend_consistency(
            returns['return_5d'],
            returns['return_10d'],
            returns['return_15d']
        )

        # 生成投资建议
        recommendation = self._get_recommendation(composite_score)

        # 多周期展望
        period_outlook = self._get_period_outlook(
            returns['return_5d'],
            returns['return_10d'],
            returns['return_15d'],
            trend_consistency
        )

        return {
            'code': code,
            'date': date,
            'composite_score': round(composite_score, 2),
            'composite_return': round(composite_return * 100, 2),  # 转为百分比
            'score_5d': round(score_5d, 2),
            'score_10d': round(score_10d, 2),
            'score_15d': round(score_15d, 2),
            'return_5d': round(returns['return_5d'] * 100, 2),
            'return_10d': round(returns['return_10d'] * 100, 2),
            'return_15d': round(returns['return_15d'] * 100, 2),
            'trend_consistency': round(trend_consistency, 2),
            'recommendation': recommendation,
            'period_outlook': period_outlook
        }

    def _analyze_trend_consistency(self, r5d: float, r10d: float, r15d: float) -> float:
        """
        分析趋势一致性

        Returns:
            -1到1之间的值
            1: 完全一致看涨（5d < 10d < 15d 且都为正）
            -1: 完全一致看跌（5d > 10d > 15d 且都为负）
            0: 无明显趋势或矛盾
        """
        # 方向一致性
        signs = [np.sign(r5d), np.sign(r10d), np.sign(r15d)]
        if len(set(signs)) == 1 and signs[0] != 0:
            direction = signs[0]
        else:
            direction = 0

        # 累进性：长期预期是否比短期更强
        if direction > 0:
            # 看涨时，15d > 10d > 5d 表示趋势加强
            if r15d > r10d > r5d:
                consistency = 1.0
            elif r15d > r5d:
                consistency = 0.5
            else:
                consistency = 0.3
        elif direction < 0:
            # 看跌时，15d < 10d < 5d 表示趋势加强
            if r15d < r10d < r5d:
                consistency = -1.0
            elif r15d < r5d:
                consistency = -0.5
            else:
                consistency = -0.3
        else:
            # 混合信号
            consistency = 0

        return consistency

    def _get_recommendation(self, score: float) -> str:
        """
        根据综合评分给出投资建议

        阈值基于多周期综合分布优化
        """
        if score >= 65:
            return "强烈买入"
        elif score >= 62:
            return "买入"
        elif score >= 58:
            return "谨慎买入"
        elif score >= 52:
            return "持有观望"
        elif score >= 48:
            return "谨慎卖出"
        else:
            return "卖出"

    def _get_period_outlook(self, r5d: float, r10d: float, r15d: float,
                            consistency: float) -> str:
        """
        生成多周期展望描述

        Args:
            r5d, r10d, r15d: 各周期预期收益率
            consistency: 趋势一致性

        Returns:
            展望描述字符串
        """
        # 基本趋势判断
        avg_return = (r5d + r10d + r15d) / 3

        if consistency >= 0.8:
            trend = "强劲上升"
        elif consistency >= 0.3:
            trend = "温和上涨"
        elif consistency <= -0.8:
            trend = "持续下跌"
        elif consistency <= -0.3:
            trend = "小幅走弱"
        elif abs(avg_return) < 0.02:
            trend = "横盘整理"
        else:
            trend = "震荡"

        # 短中期比较
        if r5d > r10d > 0:
            short_term = "短期动能强于中期"
        elif r10d > r5d > 0:
            short_term = "中期趋势优于短期"
        elif r5d < r10d < 0:
            short_term = "短期压力大于中期"
        elif r10d < r5d < 0:
            short_term = "中期风险大于短期"
        else:
            short_term = "短中期信号混合"

        return f"{trend}，{short_term}"

    def batch_predict(self, codes: list, date: str) -> pd.DataFrame:
        """
        批量预测多只股票

        Args:
            codes: 股票代码列表
            date: 日期

        Returns:
            预测结果DataFrame
        """
        results = []
        for code in codes:
            try:
                result = self.predict_score(code, date)
                if result:
                    results.append(result)
            except Exception as e:
                self.logger.warning(f"预测{code}失败: {e}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df.sort_values('composite_score', ascending=False)


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    scorer = V391MultiPeriodScorer()

    # 测试单只股票
    test_codes = ['000001', '600519', '000858']
    test_date = '2025-11-20'

    print("\n" + "=" * 80)
    print(f"V3.91 多周期评分测试 - {test_date}")
    print("=" * 80)

    for code in test_codes:
        result = scorer.predict_score(code, test_date)
        if result:
            print(f"\n📊 {code}")
            print(f"   综合评分: {result['composite_score']:.1f}分")
            print(f"   综合预期: {result['composite_return']:.2f}%")
            print(f"   5天评分: {result['score_5d']:.1f}分 (预期{result['return_5d']:.2f}%)")
            print(f"   10天评分: {result['score_10d']:.1f}分 (预期{result['return_10d']:.2f}%)")
            print(f"   15天评分: {result['score_15d']:.1f}分 (预期{result['return_15d']:.2f}%)")
            print(f"   趋势一致性: {result['trend_consistency']:.2f}")
            print(f"   建议: {result['recommendation']}")
            print(f"   展望: {result['period_outlook']}")
        else:
            print(f"\n❌ {code} 预测失败")
