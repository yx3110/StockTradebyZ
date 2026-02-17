#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query查询对象 - Hikyuu风格API

模仿Hikyuu的Query对象，用于指定K线数据查询条件
"""

from datetime import datetime, timedelta
from typing import Optional


class Query:
    """
    K线数据查询对象

    用法示例:
        Query(-150)                    # 最近150天
        Query(start='2024-01-01')      # 从指定日期开始
        Query(start='2024-01-01', end='2025-09-30')  # 指定日期区间
    """

    def __init__(self,
                 days: Optional[int] = None,
                 start: Optional[str] = None,
                 end: Optional[str] = None):
        """
        初始化查询对象

        参数:
            days: 查询天数，负数表示最近N天，正数表示从第N天开始
            start: 开始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)
        """
        self.days = days
        self.start_date = start
        self.end_date = end

        # 如果指定了days，计算实际的日期区间
        if days is not None and days < 0:
            # 最近N天
            if not end:
                self.end_date = datetime.now().strftime('%Y-%m-%d')
            # 开始日期会在查询时根据实际交易日计算
            self._is_recent_days = True
        else:
            self._is_recent_days = False

    def __repr__(self):
        if self.days is not None:
            return f"Query(days={self.days})"
        elif self.start_date and self.end_date:
            return f"Query(start='{self.start_date}', end='{self.end_date}')"
        elif self.start_date:
            return f"Query(start='{self.start_date}')"
        else:
            return "Query(all)"

    def is_recent_days(self) -> bool:
        """是否是最近N天查询"""
        return self._is_recent_days

    def get_days_count(self) -> Optional[int]:
        """获取查询天数（绝对值）"""
        if self.days is not None:
            return abs(self.days)
        return None
