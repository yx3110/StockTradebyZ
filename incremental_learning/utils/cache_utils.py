#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8缓存工具
"""

import time
from typing import Dict, Any, Optional

class FeatureCache:
    """特征缓存工具基础实现"""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key not in self.cache:
            return None

        # 检查过期
        if time.time() - self.timestamps[key] > self.ttl:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            return None

        return self.cache[key]

    def set(self, key: str, value: Any):
        """设置缓存"""
        # 如果超过最大大小，删除最老的
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            self.cache.pop(oldest_key, None)
            self.timestamps.pop(oldest_key, None)

        self.cache[key] = value
        self.timestamps[key] = time.time()

    def clear(self):
        """清理所有缓存"""
        self.cache.clear()
        self.timestamps.clear()