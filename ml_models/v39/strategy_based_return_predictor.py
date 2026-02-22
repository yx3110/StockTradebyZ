#!/usr/bin/env python3
"""
V3.95 策略驱动的预测收益计算器 (v2.0 科学版)

基于22,334个历史样本的详细分布分析，提供保守且科学的预测。

核心改进（相比v1.0）：
1. 使用中位数而非均值作为基准（更稳健）
2. 引入保守系数，避免过度乐观
3. 技术调整使用乘法系数，可正可负
4. 策略调整影响较小（历史数据显示策略数量影响不大）
5. 投资建议阈值基于实际分布的分位数

历史统计（22,334样本，5日收益）：
┌─────────────┬────────┬────────┬────────┬────────┐
│ 评分区间    │ 平均值  │ 中位数  │ 胜率   │ 样本数  │
├─────────────┼────────┼────────┼────────┼────────┤
│ 85-100      │ +9.84% │ +8.96% │ 88.3%  │ 145    │
│ 75-85       │ +4.61% │ +4.06% │ 77.5%  │ 1,873  │
│ 65-75       │ +2.31% │ +1.72% │ 65.8%  │ 3,461  │
│ 55-65       │ +1.55% │ +1.38% │ 68.9%  │ 4,338  │
│ 0-55        │ +0.63% │ +0.37% │ 54.1%  │ 12,517 │
├─────────────┼────────┼────────┼────────┼────────┤
│ 整体        │ +1.46% │ +1.03% │ 61.0%  │ 22,334 │
│ 高分(>=75)  │ +4.99% │ +4.34% │ 78.3%  │ 2,018  │
└─────────────┴────────┴────────┴────────┴────────┘

投资建议阈值（基于分布分位数）：
- 强烈推荐：预测 > 3.5%（75分位附近）
- 推荐买入：预测 > 1.5%（中位数的1.5倍）
- 谨慎买入：预测 > 0.5%
- 观望：预测 0~0.5%
- 回避：预测 < 0%
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class StrategyBasedReturnPredictor:
    """基于策略信号的预测收益计算器 (v2.0 科学版)"""

    # 历史统计数据（基于22,334个样本的中位数，更保守）
    # 使用 中位数 × 保守系数 作为预测基准
    SCORE_RETURN_STATS = {
        # (score_min, score_max): (median_return, conservative_factor, win_rate, std)
        (85, 100): (0.0896, 0.50, 0.883, 0.0906),  # 中位数8.96% × 0.5 = 4.48%
        (75, 85):  (0.0406, 0.60, 0.775, 0.0756),  # 中位数4.06% × 0.6 = 2.44%
        (65, 75):  (0.0172, 0.80, 0.658, 0.0607),  # 中位数1.72% × 0.8 = 1.38%
        (55, 65):  (0.0138, 0.90, 0.689, 0.0451),  # 中位数1.38% × 0.9 = 1.24%
        (0, 55):   (0.0037, 1.00, 0.541, 0.0485),  # 中位数0.37% × 1.0 = 0.37%
    }

    # 策略调整系数（乘法，历史数据显示策略数量影响较小）
    STRATEGY_MULTIPLIERS = {
        '补票': 1.08,      # 补票战法历史表现最好
        '填坑': 1.06,      # 填坑战法表现较好
        '上穿60': 1.04,    # 上穿60表现较好
        'SuperB1': 1.05,   # SuperB1战法
        '暴力K': 1.03,     # 暴力K战法
        '知行': 1.01,      # 知行战法（样本多，接近平均）
        '少负': 1.01,      # 少负战法
    }

    # 多策略选中的额外乘数
    MULTI_STRATEGY_MULTIPLIER = {
        1: 1.00,  # 单策略
        2: 1.03,  # 双策略
        3: 1.06,  # 三策略及以上
    }

    # 投资建议阈值（基于分布分位数）
    RECOMMENDATION_THRESHOLDS = {
        'strong_buy': (0.035, 0.70),    # 预测>3.5%且置信度>70% → 强烈推荐
        'buy': (0.015, 0.60),           # 预测>1.5%且置信度>60% → 推荐买入
        'cautious_buy': (0.005, 0.50),  # 预测>0.5% → 谨慎买入
        'hold': (0.0, 0.0),             # 预测>0% → 观望
        # 其他 → 回避
    }

    def __init__(self, db_path: str = None):
        """初始化预测器"""
        self.db_path = db_path or str(Path(__file__).parent.parent.parent / 'data_adapter' / 'stock_data.db')
        self._features_cache = {}

    def _get_features_from_db(self, code: str, date: str) -> Dict:
        """从数据库获取股票的技术特征"""
        cache_key = f"{code}_{date}"
        if cache_key in self._features_cache:
            return self._features_cache[cache_key]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT features_json FROM v39_feature_cache
                WHERE code = ? AND trade_date = ?
            """, (code, date))

            row = cursor.fetchone()
            conn.close()

            if row:
                features = json.loads(row[0])
                self._features_cache[cache_key] = features
                return features
        except Exception as e:
            pass

        return {}

    def _get_base_return(self, score: float) -> Tuple[float, float, float]:
        """
        根据评分获取基准预测收益

        Returns:
            (base_return, win_rate, std)
        """
        for (score_min, score_max), (median, conservative_factor, win_rate, std) in self.SCORE_RETURN_STATS.items():
            if score_min <= score < score_max:
                # 使用 中位数 × 保守系数 作为基准预测
                base_return = median * conservative_factor
                return base_return, win_rate, std

        # 默认值（低分区间）
        return 0.0037, 0.541, 0.0485

    def _get_strategy_multiplier(self, strategies: str) -> Tuple[float, int]:
        """
        根据选中的策略计算收益乘数

        Returns:
            (strategy_multiplier, matched_count)
        """
        if not strategies:
            return 1.0, 0

        multiplier = 1.0
        strategy_list = [s.strip() for s in strategies.split(',')]
        matched_count = 0

        # 找到最高的策略乘数（不叠加，取最大值）
        max_strategy_mult = 1.0
        for strategy in strategy_list:
            for key, mult in self.STRATEGY_MULTIPLIERS.items():
                if key in strategy:
                    max_strategy_mult = max(max_strategy_mult, mult)
                    matched_count += 1
                    break

        # 应用最高策略乘数
        multiplier *= max_strategy_mult

        # 多策略选中的额外乘数
        count_key = min(matched_count, 3)
        multiplier *= self.MULTI_STRATEGY_MULTIPLIER.get(count_key, 1.0)

        return multiplier, matched_count

    def _get_technical_multiplier(self, features: Dict) -> float:
        """
        根据技术指标计算收益乘数（可大于1也可小于1）

        Returns:
            technical_multiplier (0.85 ~ 1.15)
        """
        multiplier = 1.0

        # 1. RSI调整
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            multiplier *= 1.06  # RSI超卖，反弹空间大
        elif rsi < 40:
            multiplier *= 1.03  # RSI偏低
        elif rsi > 70:
            multiplier *= 0.95  # RSI超买，上涨空间有限
        elif rsi > 60:
            multiplier *= 0.98  # RSI偏高

        # 2. 价格位置调整
        ma5_ratio = features.get('ma5_ratio', 0)
        ma10_ratio = features.get('ma10_ratio', 0)
        avg_ma_ratio = (ma5_ratio + ma10_ratio) / 2 if ma5_ratio and ma10_ratio else 0

        if avg_ma_ratio < -0.03:
            multiplier *= 1.05  # 明显低于均线，反弹空间大
        elif avg_ma_ratio < -0.01:
            multiplier *= 1.02  # 略低于均线
        elif avg_ma_ratio > 0.05:
            multiplier *= 0.93  # 大幅高于均线，追高风险
        elif avg_ma_ratio > 0.02:
            multiplier *= 0.97  # 略高于均线

        # 3. 近期涨跌幅调整
        return_5d = features.get('return_5d', 0)
        if return_5d < -0.05:
            multiplier *= 1.05  # 近5日跌幅超过5%，超卖反弹
        elif return_5d < -0.03:
            multiplier *= 1.02  # 近5日跌幅3-5%
        elif return_5d > 0.08:
            multiplier *= 0.90  # 近5日已涨超8%，追高风险大
        elif return_5d > 0.05:
            multiplier *= 0.95  # 近5日已涨超5%

        # 4. 成交量调整
        volume_ratio = features.get('volume_ratio', 1)
        if 0.6 < volume_ratio < 1.2:
            multiplier *= 1.02  # 温和量能，企稳信号
        elif volume_ratio > 3.0:
            multiplier *= 0.95  # 放量过大，可能是出货

        # 限制乘数范围
        return max(0.85, min(1.15, multiplier))

    def _calculate_confidence(self, score: float, strategy_count: int, features: Dict) -> float:
        """
        计算预测置信度

        考虑因素：
        1. 评分高低
        2. 策略数量
        3. 特征完整性
        """
        # 基础置信度（基于评分）
        if score >= 85:
            base_confidence = 0.75
        elif score >= 75:
            base_confidence = 0.68
        elif score >= 65:
            base_confidence = 0.60
        elif score >= 55:
            base_confidence = 0.55
        else:
            base_confidence = 0.50

        # 策略数量加成
        if strategy_count >= 3:
            base_confidence += 0.08
        elif strategy_count >= 2:
            base_confidence += 0.04

        # 特征完整性检查
        key_features = ['rsi_14', 'return_5d', 'ma5_ratio', 'volume_ratio']
        feature_count = sum(1 for f in key_features if f in features and features[f] is not None)
        feature_bonus = (feature_count / len(key_features)) * 0.05
        base_confidence += feature_bonus

        return min(0.90, base_confidence)

    def _get_recommendation(self, predicted_return: float, confidence: float) -> str:
        """
        根据预测收益和置信度生成投资建议
        """
        thresholds = self.RECOMMENDATION_THRESHOLDS

        if predicted_return >= thresholds['strong_buy'][0] and confidence >= thresholds['strong_buy'][1]:
            return '强烈推荐'
        elif predicted_return >= thresholds['buy'][0] and confidence >= thresholds['buy'][1]:
            return '推荐买入'
        elif predicted_return >= thresholds['cautious_buy'][0]:
            return '谨慎买入'
        elif predicted_return >= thresholds['hold'][0]:
            return '观望'
        else:
            return '回避'

    def predict_return(
        self,
        score: float,
        strategies: str = '',
        features: Dict = None,
        period: str = '5d'
    ) -> Dict:
        """
        预测选股后的收益

        Args:
            score: 综合评分 (0-100)
            strategies: 选中的策略，逗号分隔
            features: 股票的技术特征
            period: 预测周期 ('5d')

        Returns:
            {
                'predicted_return': 预测收益率,
                'confidence': 置信度,
                'win_rate': 预测胜率,
                'return_range': (下限, 上限),
                'recommendation': 投资建议
            }
        """
        features = features or {}

        # 1. 获取基准收益（基于评分）
        base_return, base_win_rate, base_std = self._get_base_return(score)

        # 2. 策略乘数
        strategy_mult, strategy_count = self._get_strategy_multiplier(strategies)

        # 3. 技术指标乘数
        tech_mult = self._get_technical_multiplier(features)

        # 4. 计算最终预测
        predicted_return = base_return * strategy_mult * tech_mult

        # 5. 计算置信度
        confidence = self._calculate_confidence(score, strategy_count, features)

        # 6. 计算收益区间（使用标准差）
        adjusted_std = base_std * 0.8  # 略微收紧
        return_low = predicted_return - adjusted_std
        return_high = predicted_return + adjusted_std

        # 7. 调整胜率
        win_rate = base_win_rate
        if strategy_mult > 1.05:
            win_rate = min(0.90, win_rate + 0.03)
        if tech_mult > 1.05:
            win_rate = min(0.90, win_rate + 0.02)
        elif tech_mult < 0.95:
            win_rate = max(0.40, win_rate - 0.03)

        # 8. 生成投资建议
        recommendation = self._get_recommendation(predicted_return, confidence)

        return {
            'predicted_return': predicted_return,
            'predicted_return_5d': predicted_return,
            'confidence': confidence,
            'win_rate': win_rate,
            'return_range': (return_low, return_high),
            'recommendation': recommendation,
            'components': {
                'base_return': base_return,
                'strategy_multiplier': strategy_mult,
                'technical_multiplier': tech_mult,
                'strategy_count': strategy_count
            }
        }


def test_predictor():
    """测试预测器"""
    predictor = StrategyBasedReturnPredictor()

    print("=" * 70)
    print("V2.0 科学版预测器测试")
    print("=" * 70)

    # 测试不同评分和策略组合
    test_cases = [
        (90, '少负战法, 知行', {'rsi_14': 52, 'return_5d': -0.0375, 'ma5_ratio': -0.015}),
        (85, '补票', {'rsi_14': 35, 'return_5d': -0.05, 'ma5_ratio': -0.03}),
        (80, '知行', {'rsi_14': 55, 'return_5d': 0.02, 'ma5_ratio': 0.01}),
        (75, '少负战法', {'rsi_14': 65, 'return_5d': 0.05, 'ma5_ratio': 0.03}),
        (70, '知行', {'rsi_14': 50, 'return_5d': 0, 'ma5_ratio': 0}),
        (65, '少负战法, 知行, 上穿60', {'rsi_14': 28, 'return_5d': -0.08, 'ma5_ratio': -0.05}),
        (60, '', {'rsi_14': 50, 'return_5d': 0, 'ma5_ratio': 0}),
        (50, '', {'rsi_14': 70, 'return_5d': 0.08, 'ma5_ratio': 0.05}),
    ]

    print("\n| 评分 | 策略 | 预测收益 | 置信度 | 胜率 | 建议 |")
    print("|------|------|----------|--------|------|------|")

    for score, strategies, features in test_cases:
        result = predictor.predict_return(score=score, strategies=strategies, features=features)
        strategy_short = strategies[:12] + '...' if len(strategies) > 12 else strategies
        print(f"| {score} | {strategy_short:12} | {result['predicted_return']*100:+5.2f}% | "
              f"{result['confidence']*100:4.1f}% | {result['win_rate']*100:4.1f}% | {result['recommendation']} |")

    # 详细测试史丹利
    print("\n" + "=" * 70)
    print("史丹利 (002588) 详细分析")
    print("=" * 70)

    features = {
        'return_5d': -0.0375,
        'return_10d': -0.007,
        'return_20d': 0.033,
        'rsi_14': 51.7,
        'ma5_ratio': -0.015,
        'ma10_ratio': -0.014,
        'volume_ratio': 1.5
    }

    result = predictor.predict_return(
        score=90.0,
        strategies='少负战法, 知行',
        features=features
    )

    print(f"\n预测收益: {result['predicted_return']*100:+.2f}%")
    print(f"置信度: {result['confidence']*100:.1f}%")
    print(f"预测胜率: {result['win_rate']*100:.1f}%")
    print(f"收益区间: [{result['return_range'][0]*100:.2f}%, {result['return_range'][1]*100:.2f}%]")
    print(f"投资建议: {result['recommendation']}")
    print(f"\n分解:")
    for k, v in result['components'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    test_predictor()
