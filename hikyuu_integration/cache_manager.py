#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CacheManager - 智能缓存管理器

为数据适配器提供高效的缓存管理功能
"""

from collections import OrderedDict
from typing import Any, Optional, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """
    LRU (Least Recently Used) 缓存实现

    特点:
    - 自动淘汰最久未使用的缓存项
    - O(1)时间复杂度的get/put操作
    - 线程安全（单线程）
    """

    def __init__(self, capacity: int = 1000):
        """
        初始化LRU缓存

        参数:
            capacity: 最大缓存容量
        """
        self.capacity = capacity
        self.cache = OrderedDict()

        # 统计信息
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存项

        参数:
            key: 缓存键

        返回:
            缓存值，如果不存在返回None
        """
        if key in self.cache:
            # 移到末尾（标记为最近使用）
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return None

    def put(self, key: str, value: Any):
        """
        添加缓存项

        参数:
            key: 缓存键
            value: 缓存值
        """
        if key in self.cache:
            # 更新值并移到末尾
            self.cache.move_to_end(key)
        else:
            # 新增项
            if len(self.cache) >= self.capacity:
                # 淘汰最久未使用的项（第一个）
                self.cache.popitem(last=False)
                self.evictions += 1

        self.cache[key] = value

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        """
        获取缓存统计信息

        返回:
            统计信息字典
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0

        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

    def print_stats(self):
        """打印缓存统计信息"""
        stats = self.get_stats()

        print("\n📊 缓存统计信息")
        print("=" * 60)
        print(f"缓存大小: {stats['size']}/{stats['capacity']}")
        print(f"命中次数: {stats['hits']}")
        print(f"未命中:   {stats['misses']}")
        print(f"淘汰次数: {stats['evictions']}")
        print(f"命中率:   {stats['hit_rate']:.2f}%")
        print(f"总请求:   {stats['total_requests']}")
        print("=" * 60)

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache


class SmartCacheManager:
    """
    智能缓存管理器

    特点:
    - LRU缓存策略
    - 智能缓存键匹配
    - 缓存预热
    - 性能监控
    """

    def __init__(self, capacity: int = 1000):
        """
        初始化智能缓存管理器

        参数:
            capacity: 缓存容量
        """
        self.lru_cache = LRUCache(capacity)
        self.preload_info = {}  # 记录预加载的数据范围

        logger.info(f"SmartCacheManager initialized with capacity={capacity}")

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存

        参数:
            key: 缓存键

        返回:
            缓存值
        """
        return self.lru_cache.get(key)

    def put(self, key: str, value: Any, preload_info: Optional[Dict] = None):
        """
        添加缓存

        参数:
            key: 缓存键
            value: 缓存值
            preload_info: 预加载信息（如日期范围）
        """
        self.lru_cache.put(key, value)

        if preload_info:
            self.preload_info[key] = preload_info

    def find_matching_cache(self,
                           stock_code: str,
                           start_date: str,
                           end_date: str) -> Optional[Any]:
        """
        智能查找匹配的缓存

        如果请求的日期范围在已缓存的范围内，直接返回缓存数据的子集

        参数:
            stock_code: 股票代码
            start_date: 请求开始日期
            end_date: 请求结束日期

        返回:
            匹配的缓存数据
        """
        # 遍历预加载信息，查找匹配的缓存
        for cache_key, info in self.preload_info.items():
            if info.get('stock_code') == stock_code:
                cached_start = info.get('start_date')
                cached_end = info.get('end_date')

                # 检查请求范围是否在缓存范围内
                if (cached_start and cached_end and
                    start_date >= cached_start and
                    end_date <= cached_end):

                    # 从缓存获取完整数据
                    cached_data = self.lru_cache.get(cache_key)

                    if cached_data is not None:
                        # 过滤出请求的日期范围
                        import pandas as pd
                        if isinstance(cached_data, pd.DataFrame):
                            # 确保日期类型一致（转换为字符串比较）
                            trade_dates = cached_data['trade_date'].astype(str)
                            filtered = cached_data[
                                (trade_dates >= start_date) &
                                (trade_dates <= end_date)
                            ].copy()

                            if not filtered.empty:
                                # 将trade_date列也转换为字符串，避免后续比较问题
                                filtered['trade_date'] = filtered['trade_date'].astype(str)
                                logger.debug(f"Cache hit: {stock_code} [{start_date}→{end_date}] "
                                           f"from cached [{cached_start}→{cached_end}]")
                                return filtered

        return None

    def clear(self):
        """清空所有缓存"""
        self.lru_cache.clear()
        self.preload_info.clear()

    def get_stats(self) -> Dict:
        """获取缓存统计"""
        stats = self.lru_cache.get_stats()
        stats['preload_ranges'] = len(self.preload_info)
        return stats

    def print_stats(self):
        """打印缓存统计"""
        self.lru_cache.print_stats()
        print(f"预加载范围: {len(self.preload_info)}")

    def warm_up(self, data_loader_func, warm_up_items: list):
        """
        缓存预热

        参数:
            data_loader_func: 数据加载函数
            warm_up_items: 预热项列表 [(stock_code, start_date, end_date), ...]
        """
        logger.info(f"开始缓存预热，共{len(warm_up_items)}项...")

        success_count = 0
        for item in warm_up_items:
            try:
                stock_code, start_date, end_date = item

                # 加载数据
                data = data_loader_func(stock_code, start_date, end_date)

                if data is not None:
                    cache_key = f"{stock_code}_{start_date}_{end_date}"
                    self.put(cache_key, data, {
                        'stock_code': stock_code,
                        'start_date': start_date,
                        'end_date': end_date
                    })
                    success_count += 1

            except Exception as e:
                logger.warning(f"预热失败 {item}: {e}")

        logger.info(f"✅ 缓存预热完成: {success_count}/{len(warm_up_items)}")

    def __len__(self):
        return len(self.lru_cache)


# 使用示例
if __name__ == "__main__":
    # 创建缓存管理器
    cache_mgr = SmartCacheManager(capacity=100)

    # 模拟数据
    import pandas as pd

    # 添加缓存
    df = pd.DataFrame({
        'trade_date': ['2025-09-01', '2025-09-02', '2025-09-03'],
        'close': [10.0, 10.5, 11.0]
    })

    cache_mgr.put('000001_2025-09-01_2025-09-03', df, {
        'stock_code': '000001',
        'start_date': '2025-09-01',
        'end_date': '2025-09-03'
    })

    # 智能匹配（请求子范围）
    result = cache_mgr.find_matching_cache('000001', '2025-09-01', '2025-09-02')
    print(f"匹配结果: {len(result) if result is not None else 0}条")

    # 打印统计
    cache_mgr.print_stats()
