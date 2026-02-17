#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8自适应评分系统 - 主集成器

整合动态归一化、多时间维度评分和置信度评估
解决V3.7评分固化和敏感性不足的问题

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from adaptive_scoring.normalizers.dynamic_normalizer import DynamicNormalizer
from adaptive_scoring.temporal.multi_temporal_scorer import MultiTemporalScorer
from adaptive_scoring.confidence.confidence_estimator import ConfidenceEstimator

class AdaptiveScoringSystem:
    """
    V3.8自适应评分系统

    核心功能:
    1. 集成三大核心组件
    2. 提供统一的评分接口
    3. 自适应参数调优
    4. 系统性能监控
    """

    def __init__(self,
                 normalization_strategy: str = 'adaptive_sigmoid',
                 temporal_windows: Dict[str, int] = None,
                 confidence_levels: List[float] = None,
                 adaptation_mode: str = 'full',
                 logger: Optional[logging.Logger] = None):
        """
        初始化自适应评分系统

        Args:
            normalization_strategy: 归一化策略
            temporal_windows: 时间窗口配置
            confidence_levels: 置信度水平
            adaptation_mode: 适应模式 ('full', 'conservative', 'aggressive')
            logger: 日志记录器
        """
        self.normalization_strategy = normalization_strategy
        self.adaptation_mode = adaptation_mode
        self.logger = logger or logging.getLogger(__name__)

        # 默认配置
        if temporal_windows is None:
            temporal_windows = {'short_term': 5, 'medium_term': 20, 'long_term': 60}

        if confidence_levels is None:
            confidence_levels = [0.68, 0.95]

        # 初始化核心组件
        try:
            self.dynamic_normalizer = DynamicNormalizer(
                market_volatility_window=20,
                adaptation_sensitivity=0.3 if adaptation_mode == 'conservative' else 0.5 if adaptation_mode == 'full' else 0.7,
                logger=self.logger
            )

            self.multi_temporal_scorer = MultiTemporalScorer(
                short_term_window=temporal_windows['short_term'],
                medium_term_window=temporal_windows['medium_term'],
                long_term_window=temporal_windows['long_term'],
                logger=self.logger
            )

            self.confidence_estimator = ConfidenceEstimator(
                confidence_levels=confidence_levels,
                logger=self.logger
            )

            self.logger.info("V3.8自适应评分系统初始化完成")

        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            raise

        # 系统状态跟踪
        self.system_stats = {
            'total_scorings': 0,
            'successful_scorings': 0,
            'failed_scorings': 0,
            'last_scoring_time': None,
            'performance_history': []
        }

        # 缓存机制
        self.cache = {
            'market_context': None,
            'market_context_timestamp': None,
            'cache_ttl': 300  # 5分钟缓存
        }

    def calculate_adaptive_scores(self,
                                stock_code: str,
                                stock_data: pd.DataFrame,
                                fundamental_data: Optional[pd.DataFrame] = None,
                                market_data: Optional[pd.DataFrame] = None,
                                custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        计算自适应评分

        Args:
            stock_code: 股票代码
            stock_data: 股票技术数据
            fundamental_data: 基本面数据
            market_data: 市场环境数据
            custom_config: 自定义配置

        Returns:
            完整的自适应评分结果
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"开始计算 {stock_code} 的自适应评分")
            self.system_stats['total_scorings'] += 1

            # 数据验证
            if stock_data.empty:
                raise ValueError(f"股票数据为空: {stock_code}")

            # 1. 多时间维度评分
            self.logger.info("  步骤1: 计算多时间维度评分...")
            temporal_result = self.multi_temporal_scorer.calculate_multi_temporal_scores(
                stock_data=stock_data,
                fundamental_data=fundamental_data,
                market_data=market_data,
                custom_weights=custom_config.get('temporal_weights') if custom_config else None
            )

            # 2. 动态归一化
            self.logger.info("  步骤2: 执行动态归一化...")

            # 准备原始评分
            raw_scores = np.array([
                temporal_result['temporal_scores']['short_term']['overall_score'],
                temporal_result['temporal_scores']['medium_term']['overall_score'],
                temporal_result['temporal_scores']['long_term']['overall_score']
            ])

            normalization_result = self.dynamic_normalizer.normalize_scores(
                raw_scores=raw_scores,
                market_data=market_data,
                strategy=custom_config.get('normalization_strategy', self.normalization_strategy) if custom_config else self.normalization_strategy
            )

            # 3. 置信度评估
            self.logger.info("  步骤3: 评估预测置信度...")

            # 准备预测评分用于置信度评估
            prediction_scores = {
                'short_term': temporal_result['temporal_scores']['short_term']['overall_score'],
                'medium_term': temporal_result['temporal_scores']['medium_term']['overall_score'],
                'long_term': temporal_result['temporal_scores']['long_term']['overall_score'],
                'composite': temporal_result['composite_score']
            }

            # 准备模型元数据
            model_metadata = {
                'model_complexity': 0.6,  # 中等复杂度
                'output_stability': 0.8,
                'feature_importances': [0.3, 0.4, 0.3]  # 短期、中期、长期权重
            }

            confidence_result = self.confidence_estimator.estimate_confidence(
                prediction_scores=prediction_scores,
                input_data=stock_data,
                model_metadata=model_metadata,
                market_context=normalization_result.get('market_context', {})
            )

            # 4. 综合结果整合
            self.logger.info("  步骤4: 整合最终结果...")
            final_result = self._integrate_results(
                stock_code=stock_code,
                temporal_result=temporal_result,
                normalization_result=normalization_result,
                confidence_result=confidence_result,
                processing_time=(datetime.now() - start_time).total_seconds()
            )

            # 5. 系统性能记录
            self._record_system_performance(final_result, True, datetime.now() - start_time)
            self.system_stats['successful_scorings'] += 1
            self.system_stats['last_scoring_time'] = datetime.now()

            self.logger.info(f"  ✅ {stock_code} 自适应评分完成 - 最终评分: {final_result['final_score']:.3f}, 置信度: {final_result['confidence']['confidence_score']:.3f}")

            return final_result

        except Exception as e:
            self.system_stats['failed_scorings'] += 1
            self.logger.error(f"自适应评分失败 {stock_code}: {e}")

            # 返回保守的默认结果
            return self._generate_fallback_result(stock_code, str(e))

    def _integrate_results(self,
                          stock_code: str,
                          temporal_result: Dict,
                          normalization_result: Dict,
                          confidence_result: Dict,
                          processing_time: float) -> Dict[str, Any]:
        """整合所有组件的结果"""

        try:
            # 主要评分：使用归一化后的综合评分
            normalized_scores = normalization_result['normalized_scores']
            final_score = np.mean(normalized_scores)  # 归一化后各时间维度的均值

            # 临时禁用置信度调整以解决评分固化问题
            confidence_score = confidence_result['confidence_score']
            # 直接使用归一化后的评分，不做置信度调整
            confidence_adjusted_score = final_score

            # 构建详细结果
            result = {
                'stock_code': stock_code,
                'final_score': confidence_adjusted_score,
                'raw_final_score': final_score,
                'processing_time': processing_time,
                'timestamp': datetime.now(),

                # 时间维度分解
                'temporal_breakdown': {
                    'short_term': {
                        'raw_score': temporal_result['temporal_scores']['short_term']['overall_score'],
                        'normalized_score': normalized_scores[0],
                        'weight': temporal_result['adaptive_weights']['short_term'],
                        'components': temporal_result['temporal_scores']['short_term']['component_scores']
                    },
                    'medium_term': {
                        'raw_score': temporal_result['temporal_scores']['medium_term']['overall_score'],
                        'normalized_score': normalized_scores[1],
                        'weight': temporal_result['adaptive_weights']['medium_term'],
                        'components': temporal_result['temporal_scores']['medium_term']['component_scores']
                    },
                    'long_term': {
                        'raw_score': temporal_result['temporal_scores']['long_term']['overall_score'],
                        'normalized_score': normalized_scores[2],
                        'weight': temporal_result['adaptive_weights']['long_term'],
                        'components': temporal_result['temporal_scores']['long_term']['component_scores']
                    }
                },

                # 归一化详情
                'normalization': {
                    'strategy_used': normalization_result['strategy_used'],
                    'parameters': normalization_result['normalization_params'],
                    'quality_metrics': normalization_result['quality_metrics'],
                    'market_context': normalization_result['market_context']
                },

                # 置信度详情
                'confidence': {
                    'confidence_score': confidence_score,
                    'confidence_level': confidence_result['confidence_level'],
                    'confidence_intervals': confidence_result.get('confidence_intervals', {}),
                    'risk_assessment': confidence_result['risk_assessment'],
                    'reliability_factors': confidence_result['reliability_factors']
                },

                # 质量指标
                'quality_metrics': {
                    'temporal_quality': temporal_result['quality_metrics']['overall_quality'],
                    'normalization_quality': normalization_result['quality_metrics']['overall_quality'],
                    'confidence_quality': confidence_score,
                    'overall_quality': (
                        temporal_result['quality_metrics']['overall_quality'] * 0.4 +
                        normalization_result['quality_metrics']['overall_quality'] * 0.3 +
                        confidence_score * 0.3
                    )
                },

                # 系统状态
                'system_status': {
                    'adaptation_mode': self.adaptation_mode,
                    'components_healthy': True,
                    'cache_used': self.cache['market_context'] is not None
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"结果整合失败: {e}")
            raise

    def _generate_fallback_result(self, stock_code: str, error_msg: str) -> Dict[str, Any]:
        """生成后备结果"""

        return {
            'stock_code': stock_code,
            'final_score': 0.5,  # 中性评分
            'raw_final_score': 0.5,
            'processing_time': 0.0,
            'timestamp': datetime.now(),
            'error': error_msg,
            'temporal_breakdown': {
                'short_term': {'raw_score': 0.5, 'normalized_score': 0.5, 'weight': 0.3, 'components': {}},
                'medium_term': {'raw_score': 0.5, 'normalized_score': 0.5, 'weight': 0.4, 'components': {}},
                'long_term': {'raw_score': 0.5, 'normalized_score': 0.5, 'weight': 0.3, 'components': {}}
            },
            'confidence': {
                'confidence_score': 0.2,  # 低置信度
                'confidence_level': 'very_low',
                'risk_assessment': {'overall_risk': 'high'},
                'reliability_factors': ['评分计算失败']
            },
            'quality_metrics': {
                'overall_quality': 0.1
            },
            'system_status': {
                'components_healthy': False,
                'error': True
            }
        }

    def _record_system_performance(self, result: Dict, success: bool, execution_time: timedelta):
        """记录系统性能"""

        performance_record = {
            'timestamp': datetime.now(),
            'success': success,
            'execution_time': execution_time.total_seconds(),
            'final_score': result.get('final_score', 0.5),
            'confidence': result.get('confidence', {}).get('confidence_score', 0.5),
            'quality': result.get('quality_metrics', {}).get('overall_quality', 0.5)
        }

        self.system_stats['performance_history'].append(performance_record)

        # 保持历史记录在合理范围
        if len(self.system_stats['performance_history']) > 1000:
            self.system_stats['performance_history'] = self.system_stats['performance_history'][-1000:]

    def batch_calculate_scores(self,
                              stock_list: List[str],
                              data_provider: Any,
                              parallel: bool = False,
                              max_workers: int = 4) -> Dict[str, Dict]:
        """批量计算自适应评分"""

        try:
            self.logger.info(f"开始批量计算 {len(stock_list)} 只股票的自适应评分")

            results = {}

            if parallel and len(stock_list) > 1:
                # 并行处理
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交所有任务
                    future_to_stock = {}
                    for stock_code in stock_list:
                        try:
                            stock_data = data_provider.get_stock_data(stock_code)
                            fundamental_data = data_provider.get_fundamental_data(stock_code)
                            market_data = data_provider.get_market_data()

                            future = executor.submit(
                                self.calculate_adaptive_scores,
                                stock_code, stock_data, fundamental_data, market_data
                            )
                            future_to_stock[future] = stock_code
                        except Exception as e:
                            self.logger.warning(f"提交任务失败 {stock_code}: {e}")
                            results[stock_code] = self._generate_fallback_result(stock_code, str(e))

                    # 收集结果
                    for future in concurrent.futures.as_completed(future_to_stock):
                        stock_code = future_to_stock[future]
                        try:
                            result = future.result(timeout=60)  # 60秒超时
                            results[stock_code] = result
                        except Exception as e:
                            self.logger.warning(f"并行计算失败 {stock_code}: {e}")
                            results[stock_code] = self._generate_fallback_result(stock_code, str(e))

            else:
                # 串行处理
                for stock_code in stock_list:
                    try:
                        stock_data = data_provider.get_stock_data(stock_code)
                        fundamental_data = data_provider.get_fundamental_data(stock_code)
                        market_data = data_provider.get_market_data()

                        result = self.calculate_adaptive_scores(
                            stock_code, stock_data, fundamental_data, market_data
                        )
                        results[stock_code] = result

                    except Exception as e:
                        self.logger.warning(f"串行计算失败 {stock_code}: {e}")
                        results[stock_code] = self._generate_fallback_result(stock_code, str(e))

            self.logger.info(f"批量评分完成 - 成功: {len([r for r in results.values() if not r.get('error')])}/{len(stock_list)}")

            return results

        except Exception as e:
            self.logger.error(f"批量计算失败: {e}")
            return {}

    def get_system_performance(self) -> Dict[str, Any]:
        """获取系统性能统计"""

        if not self.system_stats['performance_history']:
            return {
                'status': 'no_data',
                'total_scorings': self.system_stats['total_scorings'],
                'success_rate': 0.0
            }

        recent_performance = self.system_stats['performance_history'][-100:]  # 最近100次

        # 计算统计指标
        success_rate = self.system_stats['successful_scorings'] / max(self.system_stats['total_scorings'], 1)
        avg_execution_time = np.mean([p['execution_time'] for p in recent_performance])
        avg_confidence = np.mean([p['confidence'] for p in recent_performance])
        avg_quality = np.mean([p['quality'] for p in recent_performance])

        return {
            'status': 'healthy' if success_rate > 0.9 else 'warning' if success_rate > 0.7 else 'error',
            'total_scorings': self.system_stats['total_scorings'],
            'successful_scorings': self.system_stats['successful_scorings'],
            'failed_scorings': self.system_stats['failed_scorings'],
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'average_confidence': avg_confidence,
            'average_quality': avg_quality,
            'last_scoring_time': self.system_stats['last_scoring_time'],
            'component_status': {
                'dynamic_normalizer': 'healthy',
                'multi_temporal_scorer': 'healthy',
                'confidence_estimator': 'healthy'
            }
        }

    def optimize_system_parameters(self, historical_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """优化系统参数"""

        try:
            self.logger.info("开始系统参数优化")

            optimization_results = {}

            # 1. 归一化参数优化
            if len(self.dynamic_normalizer.parameter_history) > 50:
                norm_suggestions = self.dynamic_normalizer.get_optimization_suggestions()
                optimization_results['normalization'] = norm_suggestions

            # 2. 时间维度权重优化
            if len(self.multi_temporal_scorer.weight_history) > 30:
                # 分析各时间维度的历史表现
                weight_performance = {}
                for record in self.multi_temporal_scorer.weight_history[-50:]:
                    for timeframe, weight in record['adaptive_weights'].items():
                        if timeframe not in weight_performance:
                            weight_performance[timeframe] = []
                        weight_performance[timeframe].append({
                            'weight': weight,
                            'quality': record['quality']
                        })

                # 计算最优权重建议
                optimal_weights = {}
                for timeframe, performance_data in weight_performance.items():
                    if performance_data:
                        # 找到质量评分最高的权重配置
                        best_config = max(performance_data, key=lambda x: x['quality'])
                        optimal_weights[timeframe] = best_config['weight']

                optimization_results['temporal_weights'] = optimal_weights

            # 3. 置信度校准优化
            if len(self.confidence_estimator.prediction_history) > 50:
                confidence_summary = self.confidence_estimator.get_confidence_summary()
                optimization_results['confidence_calibration'] = confidence_summary

            # 4. 系统整体建议
            system_suggestions = []

            performance_stats = self.get_system_performance()
            if performance_stats['success_rate'] < 0.9:
                system_suggestions.append("系统成功率偏低，建议检查输入数据质量")

            if performance_stats['average_execution_time'] > 5.0:
                system_suggestions.append("执行时间较长，建议启用缓存机制")

            if performance_stats['average_confidence'] < 0.6:
                system_suggestions.append("整体置信度偏低，建议增加历史数据用于校准")

            optimization_results['system_suggestions'] = system_suggestions

            self.logger.info("系统参数优化完成")
            return optimization_results

        except Exception as e:
            self.logger.error(f"系统优化失败: {e}")
            return {'error': str(e)}

    def export_system_config(self) -> Dict[str, Any]:
        """导出当前系统配置"""

        return {
            'system_version': '3.8.0',
            'adaptation_mode': self.adaptation_mode,
            'normalization_strategy': self.normalization_strategy,
            'temporal_windows': {
                'short_term': self.multi_temporal_scorer.short_window,
                'medium_term': self.multi_temporal_scorer.medium_window,
                'long_term': self.multi_temporal_scorer.long_window
            },
            'confidence_levels': self.confidence_estimator.confidence_levels,
            'system_stats': self.system_stats.copy(),
            'export_timestamp': datetime.now()
        }

    def __repr__(self):
        return f"AdaptiveScoringSystem(mode={self.adaptation_mode}, strategy={self.normalization_strategy}, scorings={self.system_stats['total_scorings']})"