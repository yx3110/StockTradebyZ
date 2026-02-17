#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8自适应评分系统性能优化器

提供缓存、批处理和性能监控功能
解决大规模股票评分的性能瓶颈

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import time
import pickle
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3


class V38PerformanceOptimizer:
    """
    V3.8性能优化器

    功能:
    1. 智能缓存系统 - 减少重复计算
    2. 批处理优化 - 提升并发性能
    3. 数据预加载 - 减少数据库查询
    4. 性能监控 - 实时性能统计
    """

    def __init__(self,
                 cache_dir: str = "adaptive_scoring/cache",
                 cache_ttl_seconds: int = 300,
                 max_cache_size: int = 1000,
                 logger: Optional[logging.Logger] = None):
        """
        初始化性能优化器

        Args:
            cache_dir: 缓存目录
            cache_ttl_seconds: 缓存生存时间(秒)
            max_cache_size: 最大缓存条目数
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)

        # 缓存配置
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_cache_size = max_cache_size

        # 内存缓存
        self._memory_cache = {}
        self._cache_metadata = {}
        self._cache_lock = threading.RLock()

        # 性能统计
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'total_computation_time': 0.0,
            'total_cache_time': 0.0,
            'batch_processing_count': 0,
            'average_batch_size': 0.0
        }

        # 数据预加载缓存
        self._preloaded_data = {}
        self._preload_lock = threading.RLock()

        self.logger.info(f"V3.8性能优化器初始化完成 - 缓存目录: {cache_dir}")

    def get_cached_score(self,
                        stock_code: str,
                        date_str: str,
                        score_params: Dict) -> Optional[Dict[str, Any]]:
        """
        获取缓存的评分结果

        Args:
            stock_code: 股票代码
            date_str: 评估日期
            score_params: 评分参数

        Returns:
            缓存的评分结果，如果不存在或过期则返回None
        """
        cache_start = time.time()

        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(stock_code, date_str, score_params)

            with self._cache_lock:
                self.performance_stats['total_requests'] += 1

                # 检查内存缓存
                if cache_key in self._memory_cache:
                    metadata = self._cache_metadata.get(cache_key)
                    if metadata and datetime.now() - metadata['created_at'] < self.cache_ttl:
                        self.performance_stats['cache_hits'] += 1
                        self.performance_stats['total_cache_time'] += time.time() - cache_start

                        self.logger.debug(f"内存缓存命中: {stock_code}")
                        return self._memory_cache[cache_key].copy()
                    else:
                        # 过期缓存清理
                        self._remove_cache_entry(cache_key)

                # 检查磁盘缓存
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_data = pickle.load(f)

                        # 检查是否过期
                        if datetime.now() - cached_data['created_at'] < self.cache_ttl:
                            # 加载到内存缓存
                            self._add_to_memory_cache(cache_key, cached_data['result'])

                            self.performance_stats['cache_hits'] += 1
                            self.performance_stats['total_cache_time'] += time.time() - cache_start

                            self.logger.debug(f"磁盘缓存命中: {stock_code}")
                            return cached_data['result'].copy()
                        else:
                            # 删除过期文件
                            cache_file.unlink()

                    except Exception as e:
                        self.logger.warning(f"磁盘缓存读取失败: {e}")
                        if cache_file.exists():
                            cache_file.unlink()

                # 缓存未命中
                self.performance_stats['cache_misses'] += 1
                self.performance_stats['total_cache_time'] += time.time() - cache_start
                return None

        except Exception as e:
            self.logger.error(f"缓存获取失败: {e}")
            self.performance_stats['total_cache_time'] += time.time() - cache_start
            return None

    def cache_score(self,
                   stock_code: str,
                   date_str: str,
                   score_params: Dict,
                   result: Dict[str, Any]):
        """
        缓存评分结果

        Args:
            stock_code: 股票代码
            date_str: 评估日期
            score_params: 评分参数
            result: 评分结果
        """
        try:
            cache_key = self._generate_cache_key(stock_code, date_str, score_params)

            with self._cache_lock:
                # 添加到内存缓存
                self._add_to_memory_cache(cache_key, result)

                # 异步保存到磁盘
                threading.Thread(
                    target=self._save_to_disk_cache,
                    args=(cache_key, result),
                    daemon=True
                ).start()

                self.logger.debug(f"缓存已保存: {stock_code}")

        except Exception as e:
            self.logger.error(f"缓存保存失败: {e}")

    def _generate_cache_key(self, stock_code: str, date_str: str, params: Dict) -> str:
        """生成缓存键"""
        # 创建参数的哈希值
        params_str = str(sorted(params.items()))
        combined_str = f"{stock_code}_{date_str}_{params_str}"

        return hashlib.md5(combined_str.encode()).hexdigest()

    def _add_to_memory_cache(self, cache_key: str, result: Dict[str, Any]):
        """添加到内存缓存"""
        # 检查缓存大小限制
        if len(self._memory_cache) >= self.max_cache_size:
            # 删除最旧的缓存项
            oldest_key = min(self._cache_metadata.keys(),
                           key=lambda k: self._cache_metadata[k]['created_at'])
            self._remove_cache_entry(oldest_key)

        # 添加新缓存项
        self._memory_cache[cache_key] = result.copy()
        self._cache_metadata[cache_key] = {
            'created_at': datetime.now(),
            'access_count': 1
        }

    def _remove_cache_entry(self, cache_key: str):
        """删除缓存条目"""
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        if cache_key in self._cache_metadata:
            del self._cache_metadata[cache_key]

    def _save_to_disk_cache(self, cache_key: str, result: Dict[str, Any]):
        """保存到磁盘缓存"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_data = {
                'result': result,
                'created_at': datetime.now()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

        except Exception as e:
            self.logger.error(f"磁盘缓存保存失败: {e}")

    def preload_stock_data(self,
                          stock_codes: List[str],
                          data_provider,
                          days: int = 120) -> Dict[str, Any]:
        """
        预加载股票数据

        Args:
            stock_codes: 股票代码列表
            data_provider: 数据提供者
            days: 数据天数

        Returns:
            预加载的数据字典
        """
        start_time = time.time()

        try:
            with self._preload_lock:
                preloaded_data = {}

                self.logger.info(f"开始预加载 {len(stock_codes)} 只股票数据")

                # 批量数据库查询优化
                if hasattr(data_provider, 'batch_get_stock_data'):
                    # 使用批量查询接口
                    batch_data = data_provider.batch_get_stock_data(stock_codes, days)
                    preloaded_data.update(batch_data)
                else:
                    # 并行单个查询
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        future_to_code = {
                            executor.submit(data_provider.get_stock_data, code): code
                            for code in stock_codes
                        }

                        for future in as_completed(future_to_code):
                            code = future_to_code[future]
                            try:
                                data = future.result(timeout=10)
                                if data is not None and not data.empty:
                                    preloaded_data[code] = data
                                    self.logger.debug(f"预加载成功: {code}")
                                else:
                                    self.logger.warning(f"预加载数据为空: {code}")
                            except Exception as e:
                                self.logger.error(f"预加载失败 {code}: {e}")

                # 更新预加载缓存
                self._preloaded_data.update(preloaded_data)

                processing_time = time.time() - start_time
                success_rate = len(preloaded_data) / len(stock_codes)

                self.logger.info(
                    f"数据预加载完成 - 成功: {len(preloaded_data)}/{len(stock_codes)} "
                    f"({success_rate:.1%}), 耗时: {processing_time:.2f}秒"
                )

                return preloaded_data

        except Exception as e:
            self.logger.error(f"数据预加载失败: {e}")
            return {}

    def get_preloaded_data(self, stock_code: str) -> Optional[Any]:
        """获取预加载的股票数据"""
        with self._preload_lock:
            return self._preloaded_data.get(stock_code)

    def optimize_batch_processing(self,
                                 stock_codes: List[str],
                                 batch_size: int = 50,
                                 max_workers: int = 4) -> List[List[str]]:
        """
        优化批处理分组

        Args:
            stock_codes: 股票代码列表
            batch_size: 批处理大小
            max_workers: 最大工作线程数

        Returns:
            优化后的批处理分组
        """
        try:
            # 根据可用性能调整批处理大小
            total_stocks = len(stock_codes)

            # 动态调整批处理大小
            if total_stocks < 100:
                # 小规模: 减小批处理大小，增加并发
                optimal_batch_size = max(10, batch_size // 2)
            elif total_stocks > 1000:
                # 大规模: 增大批处理大小，减少调度开销
                optimal_batch_size = min(100, batch_size * 2)
            else:
                optimal_batch_size = batch_size

            # 创建批处理分组
            batches = []
            for i in range(0, total_stocks, optimal_batch_size):
                batch = stock_codes[i:i + optimal_batch_size]
                batches.append(batch)

            # 更新统计
            self.performance_stats['batch_processing_count'] += 1
            current_avg = self.performance_stats['average_batch_size']
            count = self.performance_stats['batch_processing_count']
            self.performance_stats['average_batch_size'] = (
                current_avg * (count - 1) + len(batches)
            ) / count

            self.logger.info(
                f"批处理优化完成 - {total_stocks}只股票分为{len(batches)}批, "
                f"每批{optimal_batch_size}只"
            )

            return batches

        except Exception as e:
            self.logger.error(f"批处理优化失败: {e}")
            # 返回简单分组
            return [stock_codes[i:i+batch_size] for i in range(0, len(stock_codes), batch_size)]

    def clear_cache(self, expired_only: bool = True):
        """
        清理缓存

        Args:
            expired_only: 是否只清理过期缓存
        """
        try:
            with self._cache_lock:
                if expired_only:
                    # 清理过期的内存缓存
                    expired_keys = []
                    current_time = datetime.now()

                    for cache_key, metadata in self._cache_metadata.items():
                        if current_time - metadata['created_at'] > self.cache_ttl:
                            expired_keys.append(cache_key)

                    for key in expired_keys:
                        self._remove_cache_entry(key)

                    # 清理过期的磁盘缓存
                    for cache_file in self.cache_dir.glob("*.pkl"):
                        try:
                            with open(cache_file, 'rb') as f:
                                cached_data = pickle.load(f)
                            if current_time - cached_data['created_at'] > self.cache_ttl:
                                cache_file.unlink()
                        except Exception:
                            cache_file.unlink()  # 删除损坏的缓存文件

                    self.logger.info(f"清理过期缓存完成 - 内存: {len(expired_keys)}项")

                else:
                    # 清理所有缓存
                    self._memory_cache.clear()
                    self._cache_metadata.clear()

                    for cache_file in self.cache_dir.glob("*.pkl"):
                        cache_file.unlink()

                    self.logger.info("清理所有缓存完成")

        except Exception as e:
            self.logger.error(f"缓存清理失败: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        try:
            total_requests = self.performance_stats['total_requests']
            cache_hits = self.performance_stats['cache_hits']
            cache_misses = self.performance_stats['cache_misses']

            # 计算缓存命中率
            cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0

            # 计算平均响应时间
            avg_computation_time = (
                self.performance_stats['total_computation_time'] / cache_misses
                if cache_misses > 0 else 0.0
            )
            avg_cache_time = (
                self.performance_stats['total_cache_time'] / total_requests
                if total_requests > 0 else 0.0
            )

            return {
                'cache_performance': {
                    'hit_rate': cache_hit_rate,
                    'total_requests': total_requests,
                    'cache_hits': cache_hits,
                    'cache_misses': cache_misses,
                    'cache_size': len(self._memory_cache),
                    'average_cache_time_ms': avg_cache_time * 1000
                },
                'computation_performance': {
                    'average_computation_time_ms': avg_computation_time * 1000,
                    'total_computation_time_s': self.performance_stats['total_computation_time']
                },
                'batch_processing': {
                    'batch_count': self.performance_stats['batch_processing_count'],
                    'average_batch_size': self.performance_stats['average_batch_size']
                },
                'preloaded_data': {
                    'stocks_preloaded': len(self._preloaded_data)
                },
                'recommendations': self._generate_performance_recommendations(cache_hit_rate)
            }

        except Exception as e:
            self.logger.error(f"性能报告生成失败: {e}")
            return {'error': str(e)}

    def _generate_performance_recommendations(self, cache_hit_rate: float) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        try:
            if cache_hit_rate < 0.3:
                recommendations.append("缓存命中率偏低，建议增加缓存生存时间")
            elif cache_hit_rate > 0.8:
                recommendations.append("缓存性能良好")

            if len(self._memory_cache) >= self.max_cache_size * 0.9:
                recommendations.append("内存缓存接近上限，考虑增加缓存大小")

            avg_batch_size = self.performance_stats['average_batch_size']
            if avg_batch_size > 0:
                if avg_batch_size < 5:
                    recommendations.append("批处理大小较小，可能影响性能")
                elif avg_batch_size > 100:
                    recommendations.append("批处理大小较大，考虑减小以提升响应速度")

            if not recommendations:
                recommendations.append("系统性能优化状态良好")

        except Exception as e:
            recommendations.append(f"性能分析失败: {e}")

        return recommendations

    def record_computation_time(self, computation_time: float):
        """记录计算时间"""
        self.performance_stats['total_computation_time'] += computation_time

    def __repr__(self):
        return (f"V38PerformanceOptimizer(cache_size={len(self._memory_cache)}, "
                f"hit_rate={self.performance_stats['cache_hits']/max(1, self.performance_stats['total_requests']):.1%})")