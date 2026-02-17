#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8自适应评分系统选股适配器

为tomorrow_stock_selector.py提供V3.8系统的标准化接口
兼容现有选股框架的API设计

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from adaptive_scoring.adaptive_scoring_system import AdaptiveScoringSystem
from adaptive_scoring.utils.data_adapter import AdaptiveScoringDataAdapter
from adaptive_scoring.utils.performance_optimizer import V38PerformanceOptimizer

class V38SelectorAdapter:
    """
    V3.8自适应评分系统选股适配器

    功能:
    1. 提供与现有选股系统兼容的API
    2. 批量股票评分处理
    3. 结果格式化和排序
    4. 性能优化和缓存
    """

    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        初始化V3.8选股适配器

        Args:
            config_path: 配置文件路径
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)

        # 加载配置
        if config_path is None:
            config_path = "adaptive_scoring/config/v38_config.json"

        self.config = self._load_config(config_path)

        # 初始化核心组件
        try:
            # 性能优化器
            self.performance_optimizer = V38PerformanceOptimizer(
                cache_dir=self.config.get('performance_config', {}).get('cache_dir', 'adaptive_scoring/cache'),
                cache_ttl_seconds=self.config.get('performance_config', {}).get('caching', {}).get('cache_ttl_seconds', 300),
                max_cache_size=self.config.get('performance_config', {}).get('caching', {}).get('max_cache_size', 1000),
                logger=self.logger
            )

            # 数据适配器
            self.data_adapter = AdaptiveScoringDataAdapter(
                db_path=self.config['data_adapter_config']['db_path'],
                logger=self.logger
            )

            # 自适应评分系统
            self.scoring_system = AdaptiveScoringSystem(
                normalization_strategy=self.config['normalization_config']['default_strategy'],
                temporal_windows=self.config['temporal_scoring_config']['time_windows'],
                confidence_levels=self.config['confidence_config']['confidence_levels'],
                adaptation_mode='full',
                logger=self.logger
            )

            # 性能统计
            self.performance_stats = {
                'total_evaluations': 0,
                'successful_evaluations': 0,
                'failed_evaluations': 0,
                'average_processing_time': 0.0,
                'last_batch_time': None
            }

            self.logger.info("V3.8自适应评分系统选股适配器初始化完成")

        except Exception as e:
            self.logger.error(f"V3.8适配器初始化失败: {e}")
            raise

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"配置文件加载失败: {e}")
            # 返回默认配置
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'normalization_config': {'default_strategy': 'adaptive_sigmoid'},
            'temporal_scoring_config': {'time_windows': {'short_term': 5, 'medium_term': 20, 'long_term': 60}},
            'confidence_config': {'confidence_levels': [0.68, 0.95]},
            'data_adapter_config': {'db_path': 'data_adapter/stock_data.db'},
            'performance_config': {'batch_processing': {'max_workers': 4, 'parallel_threshold': 5}},
            'scoring_thresholds': {'min_score': 0.0, 'max_score': 1.0, 'confidence_threshold': 0.3}
        }

    def evaluate_stocks(self,
                       stock_codes: List[str],
                       date_str: Optional[str] = None,
                       parallel: bool = True) -> Dict[str, Any]:
        """
        批量评估股票评分 - 主要接口方法

        Args:
            stock_codes: 股票代码列表
            date_str: 评估日期，格式YYYY-MM-DD
            parallel: 是否启用并行处理

        Returns:
            包含评分结果的字典
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"开始V3.8批量股票评估 - 股票数量: {len(stock_codes)}")

            # 检查缓存
            cached_results = {}
            uncached_codes = []
            score_params = {
                'normalization_strategy': self.config['normalization_config']['default_strategy'],
                'temporal_windows': self.config['temporal_scoring_config']['time_windows']
            }

            for code in stock_codes:
                cached_result = self.performance_optimizer.get_cached_score(code, date_str or datetime.now().strftime('%Y-%m-%d'), score_params)
                if cached_result:
                    cached_results[code] = cached_result
                    self.logger.debug(f"使用缓存结果: {code}")
                else:
                    uncached_codes.append(code)

            self.logger.info(f"缓存命中: {len(cached_results)}/{len(stock_codes)}, 需要计算: {len(uncached_codes)}")

            # 对需要计算的股票进行评分
            scoring_results = cached_results.copy()

            if uncached_codes:
                # 创建数据提供者
                data_provider = self._create_data_provider(date_str)

                # 预加载数据以提升性能
                if len(uncached_codes) >= 5:
                    self.performance_optimizer.preload_stock_data(uncached_codes, data_provider)

                # 根据股票数量决定是否使用并行处理
                use_parallel = parallel and len(uncached_codes) >= self.config['performance_config']['batch_processing']['parallel_threshold']

                # 批量评分
                computation_start = time.time()
                new_results = self.scoring_system.batch_calculate_scores(
                    stock_list=uncached_codes,
                    data_provider=data_provider,
                    parallel=use_parallel,
                    max_workers=self.config['performance_config']['batch_processing']['max_workers']
                )
                computation_time = time.time() - computation_start

                # 记录计算时间
                self.performance_optimizer.record_computation_time(computation_time)

                # 缓存新结果
                for code, result in new_results.items():
                    if not result.get('error'):
                        self.performance_optimizer.cache_score(code, date_str or datetime.now().strftime('%Y-%m-%d'), score_params, result)

                scoring_results.update(new_results)

            # 格式化结果
            formatted_results = self._format_results(scoring_results, date_str)

            # 更新性能统计
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(len(stock_codes), len(scoring_results), processing_time)

            self.logger.info(f"V3.8批量评估完成 - 成功: {len(scoring_results)}/{len(stock_codes)}, 耗时: {processing_time:.2f}秒")

            return formatted_results

        except Exception as e:
            self.logger.error(f"V3.8批量评估失败: {e}")
            return self._generate_fallback_results(stock_codes, date_str)

    def _create_data_provider(self, evaluation_date: Optional[str] = None):
        """创建数据提供者"""
        class DataProvider:
            def __init__(self, data_adapter, eval_date):
                self.data_adapter = data_adapter
                self.eval_date = eval_date

            def get_stock_data(self, stock_code):
                return self.data_adapter.get_stock_data(stock_code, end_date=self.eval_date)

            def get_fundamental_data(self, stock_code):
                return self.data_adapter.get_fundamental_data(stock_code, end_date=self.eval_date)

            def get_market_data(self):
                return self.data_adapter.get_market_data('000001.SH', end_date=self.eval_date)

        return DataProvider(self.data_adapter, evaluation_date)

    def _format_results(self, scoring_results: Dict[str, Dict], date_str: Optional[str]) -> Dict[str, Any]:
        """格式化结果以兼容现有选股系统"""

        try:
            formatted_stocks = []

            for stock_code, result in scoring_results.items():
                if result.get('error'):
                    continue

                # 提取关键信息
                final_score = result.get('final_score', 0.5)
                confidence_score = result.get('confidence', {}).get('confidence_score', 0.0)
                temporal_breakdown = result.get('temporal_breakdown', {})

                # 转换为与现有系统兼容的格式
                stock_info = {
                    'code': stock_code,
                    'name': self._get_stock_name(stock_code),
                    'final_score': final_score,
                    'confidence_score': confidence_score,
                    'confidence_level': result.get('confidence', {}).get('confidence_level', 'unknown'),

                    # 时间维度评分
                    'short_term_score': temporal_breakdown.get('short_term', {}).get('raw_score', 0.5),
                    'medium_term_score': temporal_breakdown.get('medium_term', {}).get('raw_score', 0.5),
                    'long_term_score': temporal_breakdown.get('long_term', {}).get('raw_score', 0.5),

                    # 权重信息
                    'short_term_weight': temporal_breakdown.get('short_term', {}).get('weight', 0.3),
                    'medium_term_weight': temporal_breakdown.get('medium_term', {}).get('weight', 0.4),
                    'long_term_weight': temporal_breakdown.get('long_term', {}).get('weight', 0.3),

                    # 质量指标
                    'overall_quality': result.get('quality_metrics', {}).get('overall_quality', 0.5),
                    'processing_time': result.get('processing_time', 0.0),

                    # 风险评估
                    'risk_level': result.get('confidence', {}).get('risk_assessment', {}).get('overall_risk', 'medium'),

                    # 元数据
                    'evaluation_date': date_str or datetime.now().strftime('%Y-%m-%d'),
                    'version': '3.8.0'
                }

                formatted_stocks.append(stock_info)

            # 按最终评分排序
            formatted_stocks.sort(key=lambda x: x['final_score'], reverse=True)

            # 生成摘要统计
            if formatted_stocks:
                summary_stats = {
                    'total_evaluated': len(formatted_stocks),
                    'average_score': np.mean([s['final_score'] for s in formatted_stocks]),
                    'average_confidence': np.mean([s['confidence_score'] for s in formatted_stocks]),
                    'high_confidence_count': len([s for s in formatted_stocks if s['confidence_score'] > 0.7]),
                    'score_range': {
                        'min': min([s['final_score'] for s in formatted_stocks]),
                        'max': max([s['final_score'] for s in formatted_stocks]),
                        'std': np.std([s['final_score'] for s in formatted_stocks])
                    }
                }
            else:
                summary_stats = {'total_evaluated': 0, 'average_score': 0.0, 'average_confidence': 0.0}

            return {
                'stocks': formatted_stocks,
                'summary': summary_stats,
                'metadata': {
                    'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'version': '3.8.0',
                    'total_candidates': len(scoring_results)
                }
            }

        except Exception as e:
            self.logger.error(f"结果格式化失败: {e}")
            return {'stocks': [], 'summary': {'total_evaluated': 0}, 'error': str(e)}

    def _get_stock_name(self, stock_code: str) -> str:
        """获取股票名称"""
        try:
            # 简化实现，实际可以从数据库获取
            return f"股票{stock_code}"
        except Exception:
            return stock_code

    def _update_performance_stats(self, total_stocks: int, successful_stocks: int, processing_time: float):
        """更新性能统计"""
        try:
            self.performance_stats['total_evaluations'] += total_stocks
            self.performance_stats['successful_evaluations'] += successful_stocks
            self.performance_stats['failed_evaluations'] += (total_stocks - successful_stocks)

            # 更新平均处理时间
            if self.performance_stats['total_evaluations'] > 0:
                total_time = self.performance_stats['average_processing_time'] * (self.performance_stats['total_evaluations'] - total_stocks)
                self.performance_stats['average_processing_time'] = (total_time + processing_time) / self.performance_stats['total_evaluations']

            self.performance_stats['last_batch_time'] = datetime.now()

        except Exception as e:
            self.logger.warning(f"性能统计更新失败: {e}")

    def _generate_fallback_results(self, stock_codes: List[str], date_str: Optional[str]) -> Dict[str, Any]:
        """生成后备结果"""
        fallback_stocks = []
        for code in stock_codes:
            fallback_stocks.append({
                'code': code,
                'name': self._get_stock_name(code),
                'final_score': 0.5,  # 中性评分
                'confidence_score': 0.1,  # 低置信度
                'confidence_level': 'very_low',
                'short_term_score': 0.5,
                'medium_term_score': 0.5,
                'long_term_score': 0.5,
                'overall_quality': 0.1,
                'risk_level': 'high',
                'evaluation_date': date_str or datetime.now().strftime('%Y-%m-%d'),
                'version': '3.8.0',
                'error': True
            })

        return {
            'stocks': fallback_stocks,
            'summary': {'total_evaluated': len(fallback_stocks), 'average_score': 0.5, 'average_confidence': 0.1},
            'metadata': {'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'version': '3.8.0', 'error': True}
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            # 获取系统级别的性能数据
            system_performance = self.scoring_system.get_system_performance()
            optimizer_performance = self.performance_optimizer.get_performance_report()

            return {
                'adapter_stats': self.performance_stats.copy(),
                'system_stats': system_performance,
                'optimizer_stats': optimizer_performance,
                'config_summary': {
                    'version': self.config['system_config']['version'],
                    'normalization_strategy': self.config['normalization_config']['default_strategy'],
                    'temporal_windows': self.config['temporal_scoring_config']['time_windows'],
                    'parallel_threshold': self.config['performance_config']['batch_processing']['parallel_threshold']
                },
                'recommendations': self._generate_performance_recommendations()
            }
        except Exception as e:
            self.logger.error(f"性能报告生成失败: {e}")
            return {'error': str(e)}

    def _generate_performance_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        try:
            if self.performance_stats['total_evaluations'] == 0:
                recommendations.append("尚未进行任何评估")
                return recommendations

            success_rate = self.performance_stats['successful_evaluations'] / self.performance_stats['total_evaluations']

            if success_rate < 0.9:
                recommendations.append(f"成功率偏低 ({success_rate:.1%})，建议检查数据源")

            avg_time = self.performance_stats['average_processing_time']
            if avg_time > 1.0:
                recommendations.append(f"平均处理时间较长 ({avg_time:.2f}秒)，建议启用并行处理")

            if not recommendations:
                recommendations.append("系统性能良好")

        except Exception as e:
            recommendations.append(f"性能分析失败: {e}")

        return recommendations

    def __repr__(self):
        return f"V38SelectorAdapter(version=3.8.0, evaluations={self.performance_stats['total_evaluations']})"