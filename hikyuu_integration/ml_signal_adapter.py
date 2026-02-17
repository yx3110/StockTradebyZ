#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLScoringSignal - ML评分系统Signal适配器

将v3.7/v3.8/v3.81 ML评分系统转换为Hikyuu风格的Signal
"""

import logging
from typing import Optional, Dict
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .signal_base import SignalBase
from .kdata import KData

logger = logging.getLogger(__name__)


class MLScoringSignal(SignalBase):
    """
    ML评分Signal适配器

    将ML评分系统(v3.7/v3.8/v3.81)转换为Hikyuu Signal

    策略逻辑:
    - 买入: ML评分 >= min_score
    - 卖出: ML评分 < min_score - sell_margin

    参数:
    - ml_version: ML版本 ('v3.7', 'v3.8', 'v3.81')
    - min_score: 最低买入评分阈值（默认80）
    - sell_margin: 卖出评分余量（默认10，即评分<70时卖出）

    示例:
        # 使用v3.81，评分80以上买入
        signal = MLScoringSignal(ml_version='v3.81', min_score=80)

        # 计算信号
        signal.calculate(kdata)

        # 判断买卖
        if signal.should_buy('2025-09-30'):
            print("买入!")
    """

    def __init__(self,
                 ml_version: str = 'v3.81',
                 min_score: float = 80.0,
                 sell_margin: float = 10.0,
                 name: Optional[str] = None):
        """
        初始化ML评分Signal

        参数:
            ml_version: ML版本 ('v3.7', 'v3.8', 'v3.81')
            min_score: 最低买入评分阈值
            sell_margin: 卖出评分余量
            name: 信号名称（默认自动生成）
        """
        if name is None:
            name = f'MLScoring_{ml_version}_Signal'

        super().__init__(name=name, params={
            'ml_version': ml_version,
            'min_score': min_score,
            'sell_margin': sell_margin
        })

        self.ml_version = ml_version
        self.min_score = min_score
        self.sell_margin = sell_margin
        self.sell_score = min_score - sell_margin

        # 初始化ML系统
        self.ml_system = self._init_ml_system()

        # 评分缓存
        self._score_cache = {}

    def _init_ml_system(self):
        """
        初始化对应版本的ML系统

        返回:
            ML系统实例
        """
        try:
            if self.ml_version == 'v3.7':
                from ml_models.v37 import V370AdvancedMLSystem
                logger.info(f"✅ 加载 {self.ml_version} ML系统")
                return V370AdvancedMLSystem()

            elif self.ml_version == 'v3.8':
                from ml_models.v38 import V380AdvancedIncrementalMLSystem
                logger.info(f"✅ 加载 {self.ml_version} ML系统")
                return V380AdvancedIncrementalMLSystem()

            elif self.ml_version == 'v3.81':
                from ml_models.v381 import V380Level4IntegratedSystem
                logger.info(f"✅ 加载 {self.ml_version} ML系统")
                return V380Level4IntegratedSystem()

            else:
                raise ValueError(f"不支持的ML版本: {self.ml_version}，"
                               f"支持的版本: v3.7, v3.8, v3.81")

        except ImportError as e:
            logger.error(f"❌ 导入{self.ml_version} ML系统失败: {e}")
            logger.warning("将使用模拟评分模式（仅用于测试）")
            return None

    def _calculate(self, kdata: KData):
        """
        计算ML评分信号

        对每个交易日计算ML评分，然后根据阈值生成买入/卖出信号

        参数:
            kdata: K线数据对象
        """
        stock_code = kdata.stock_code

        logger.debug(f"计算 {stock_code} 的ML评分信号 ({self.ml_version})")

        # 遍历每个交易日
        for i in range(len(kdata)):
            date = kdata.datetime[i]

            # 计算ML评分
            score = self._get_ml_score(stock_code, date)

            # 缓存评分
            self._score_cache[date] = score

            # 根据评分生成信号
            if score >= self.min_score:
                # 买入信号
                # 信号强度: 评分越高，强度越大
                strength = min(1.0, (score - self.min_score) / (100 - self.min_score))
                self._add_buy_signal(date, strength)

                logger.debug(f"{date}: 买入信号，评分={score:.1f}, 强度={strength:.2f}")

            elif score < self.sell_score:
                # 卖出信号
                # 信号强度: 评分越低，强度越大
                strength = min(1.0, (self.sell_score - score) / self.sell_score)
                self._add_sell_signal(date, strength)

                logger.debug(f"{date}: 卖出信号，评分={score:.1f}, 强度={strength:.2f}")

        logger.info(f"{stock_code} 信号计算完成: "
                   f"买入{len(self._buy_signals)}次, "
                   f"卖出{len(self._sell_signals)}次")

    def _get_ml_score(self, stock_code: str, date: str) -> float:
        """
        获取ML评分

        参数:
            stock_code: 股票代码
            date: 日期

        返回:
            评分 (0-100)
        """
        if self.ml_system is None:
            # 模拟评分模式（仅用于测试）
            return self._mock_score(stock_code, date)

        try:
            # 调用ML系统计算评分
            if self.ml_version == 'v3.7':
                # V3.7需要特征提取和预测
                score = self._calculate_v37_score(stock_code, date)

            elif self.ml_version == 'v3.8':
                # V3.8有predict_scores方法
                score = self._calculate_v38_score(stock_code, date)

            elif self.ml_version == 'v3.81':
                # V3.81有predict_scores方法，返回overall_score
                score = self._calculate_v381_score(stock_code, date)

            else:
                score = 50.0

            return float(score) if score is not None else 50.0

        except Exception as e:
            logger.warning(f"计算{stock_code} {date}评分失败: {e}")
            return 50.0

    def _calculate_v37_score(self, stock_code: str, date: str) -> float:
        """计算V3.7评分"""
        try:
            # V3.7需要提取特征然后预测
            # 提取单日特征（使用较小的日期范围）
            features_df = self.ml_system.extract_advanced_features(
                codes=[stock_code],
                start_date=date,
                end_date=date,
                target_only=True
            )

            if features_df is None or len(features_df) == 0:
                logger.debug(f"V3.7: {stock_code} {date} 无特征数据")
                return 50.0

            # 使用三层Ensemble预测
            predictions = self.ml_system.predict_three_layer_ensemble(features_df)

            if predictions is None or len(predictions) == 0:
                return 50.0

            # 将预测值映射到0-100评分
            # V3.7预测的是收益率，需要转换为评分
            pred_value = predictions.iloc[0] if hasattr(predictions, 'iloc') else predictions[0]

            # 映射：预测收益率越高，评分越高
            # 假设预测收益率在[-0.1, 0.1]范围，映射到[0, 100]
            score = max(0, min(100, (pred_value + 0.1) / 0.2 * 100))

            logger.debug(f"V3.7评分: {stock_code} {date} = {score:.1f}")
            return float(score)

        except Exception as e:
            logger.warning(f"V3.7评分计算失败 {stock_code} {date}: {e}")
            return 50.0

    def _calculate_v38_score(self, stock_code: str, date: str) -> float:
        """计算V3.8评分"""
        try:
            # V3.8有predict_scores方法，直接返回评分
            result = self.ml_system.predict_scores([stock_code], date)

            if result is None or len(result) == 0:
                logger.debug(f"V3.8: {stock_code} {date} 无评分数据")
                return 50.0

            # 获取overall_score
            score = result[0].get('overall_score', 50.0) if isinstance(result[0], dict) else 50.0

            logger.debug(f"V3.8评分: {stock_code} {date} = {score:.1f}")
            return float(score)

        except Exception as e:
            logger.warning(f"V3.8评分计算失败 {stock_code} {date}: {e}")
            return 50.0

    def _calculate_v381_score(self, stock_code: str, date: str) -> float:
        """计算V3.81评分"""
        try:
            # V3.81有predict_scores_with_quality方法，返回完整评分信息
            # 如果没有，降级使用predict_scores
            if hasattr(self.ml_system, 'predict_scores_with_quality'):
                result = self.ml_system.predict_scores_with_quality([stock_code], date)
            else:
                result = self.ml_system.predict_scores([stock_code], date)

            if result is None or len(result) == 0:
                logger.debug(f"V3.81: {stock_code} {date} 无评分数据")
                return 50.0

            # 获取overall_score
            if isinstance(result[0], dict):
                # 优先使用quality_score，其次overall_score
                score = result[0].get('quality_score', result[0].get('overall_score', 50.0))
            else:
                score = 50.0

            logger.debug(f"V3.81评分: {stock_code} {date} = {score:.1f}")
            return float(score)

        except Exception as e:
            logger.warning(f"V3.81评分计算失败 {stock_code} {date}: {e}")
            return 50.0

    def _mock_score(self, stock_code: str, date: str) -> float:
        """
        模拟评分（仅用于测试）

        生成一个伪随机但稳定的评分
        """
        import hashlib

        # 使用hash生成稳定的伪随机数
        hash_input = f"{stock_code}_{date}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)

        # 映射到60-90分范围
        score = 60 + (hash_value % 30)

        return float(score)

    def get_score(self, date: str) -> Optional[float]:
        """
        获取指定日期的评分

        参数:
            date: 日期字符串

        返回:
            评分，如果不存在返回None
        """
        return self._score_cache.get(date)

    def get_all_scores(self) -> Dict[str, float]:
        """获取所有日期的评分"""
        return self._score_cache.copy()


# ==================== 组合Signal示例 ====================

class MLCombinedSignal(SignalBase):
    """
    ML + 技术指标组合Signal

    结合ML评分和技术指标生成信号

    买入条件:
    - ML评分 >= min_score
    - 收盘价 > BBI (多头市场)
    - KDJ_K < 20 (超卖)

    示例:
        signal = MLCombinedSignal(ml_version='v3.81', min_score=80)
    """

    def __init__(self,
                 ml_version: str = 'v3.81',
                 min_score: float = 80.0,
                 use_bbi: bool = True,
                 use_kdj: bool = True):
        """
        参数:
            ml_version: ML版本
            min_score: ML最低评分
            use_bbi: 是否使用BBI条件
            use_kdj: 是否使用KDJ条件
        """
        super().__init__(name=f'MLCombined_{ml_version}_Signal', params={
            'ml_version': ml_version,
            'min_score': min_score,
            'use_bbi': use_bbi,
            'use_kdj': use_kdj
        })

        self.ml_signal = MLScoringSignal(ml_version=ml_version, min_score=min_score)
        self.min_score = min_score
        self.use_bbi = use_bbi
        self.use_kdj = use_kdj

    def _calculate(self, kdata: KData):
        """计算组合信号"""
        # 先计算ML信号
        self.ml_signal.calculate(kdata)

        # 获取技术指标
        bbi = kdata.get_indicator('BBI') if self.use_bbi else None
        kdj_k = kdata.get_indicator('KDJ_K') if self.use_kdj else None

        # 遍历每个交易日
        for i in range(len(kdata)):
            date = kdata.datetime[i]

            # 条件1: ML评分
            ml_score = self.ml_signal.get_score(date)
            if ml_score is None or ml_score < self.min_score:
                continue

            # 条件2: BBI（可选）
            if self.use_bbi and bbi is not None:
                if kdata.close[i] <= bbi[i]:
                    continue  # 价格不在BBI之上，跳过

            # 条件3: KDJ（可选）
            if self.use_kdj and kdj_k is not None:
                if kdj_k[i] >= 20:
                    continue  # KDJ不在超卖区，跳过

            # 所有条件满足，生成买入信号
            strength = (ml_score - self.min_score) / (100 - self.min_score)
            self._add_buy_signal(date, strength)

            logger.debug(f"{date}: 组合买入信号，ML={ml_score:.1f}, "
                        f"BBI={'✓' if self.use_bbi else 'N/A'}, "
                        f"KDJ={'✓' if self.use_kdj else 'N/A'}")
