#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.8情绪指标增强模块
Phase 2.2: 实现4个核心情绪指标
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')
from data_adapter.database_manager import DatabaseManager

class SentimentIndicatorCalculator:
    """
    情绪指标计算器

    Phase 2.2核心情绪指标:
    1. 实时资金流向指标 (capital_flow_indicator)
    2. 市场情绪指数 (market_sentiment_index) - VIX等价物
    3. 板块轮动强度 (sector_rotation_strength)
    4. 北向资金流向影响 (northbound_capital_impact)
    """

    def __init__(self, db_manager: DatabaseManager, cache_ttl: int = 300):
        self.db_manager = db_manager
        self.cache_ttl = cache_ttl
        self.logger = logging.getLogger('SentimentIndicators')

        # 情绪指标缓存
        self.sentiment_cache = {}
        self.cache_timestamps = {}

        # 市场基准数据缓存
        self.market_baseline = None
        self.baseline_timestamp = None

    def compute_sentiment_indicators(self, code: str, current_time: datetime = None) -> Dict[str, float]:
        """
        计算完整的情绪指标集合

        Args:
            code: 股票代码
            current_time: 当前时间

        Returns:
            Dict: 4个情绪指标的字典
        """
        if current_time is None:
            current_time = datetime.now()

        cache_key = f"sentiment_{code}_{current_time.strftime('%Y%m%d_%H%M')}"

        # 检查缓存
        if self._is_cache_valid(cache_key):
            self.logger.info(f"💭 使用缓存的情绪指标: {code}")
            return self.sentiment_cache[cache_key]

        try:
            # 获取市场基准数据
            self._update_market_baseline(current_time)

            indicators = {}

            # 1. 实时资金流向指标
            indicators.update(self._compute_capital_flow_indicator(code, current_time))

            # 2. 市场情绪指数 (VIX等价物)
            indicators.update(self._compute_market_sentiment_index(code, current_time))

            # 3. 板块轮动强度
            indicators.update(self._compute_sector_rotation_strength(code, current_time))

            # 4. 北向资金流向影响 (基于可用数据估算)
            indicators.update(self._compute_northbound_capital_impact(code, current_time))

            # 更新缓存
            self.sentiment_cache[cache_key] = indicators
            self.cache_timestamps[cache_key] = current_time

            self.logger.info(f"💭 计算{code}情绪指标完成: {len(indicators)}个指标")
            return indicators

        except Exception as e:
            self.logger.error(f"❌ 计算{code}情绪指标失败: {e}")
            return self._get_default_sentiment_indicators()

    def _compute_capital_flow_indicator(self, code: str, current_time: datetime) -> Dict[str, float]:
        """
        计算实时资金流向指标

        基于成交量和价格变化的资金流向分析:
        - 大单净流入比例
        - 资金流入强度
        - 主动买入比例
        """
        try:
            # 获取最近几天的成交数据
            query = """
            SELECT dq.trade_date, dq.close, dq.volume, dq.amount, dq.price_change_pct,
                   db.turnover_rate, dq.high, dq.low
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            LEFT JOIN daily_basic db ON dq.security_id = db.security_id AND dq.trade_date = db.trade_date
            WHERE s.code = ?
            ORDER BY dq.trade_date DESC
            LIMIT 5
            """

            with self.db_manager.get_connection() as conn:
                result = conn.execute(query, (code.split('.')[0] if '.' in code else code,)).fetchall()

                if len(result) < 2:
                    return {'capital_flow_indicator': 0.0}

                # 计算资金流向指标
                recent_data = result[:3]  # 最近3天数据

                total_amount = 0
                weighted_price_change = 0
                total_volume = 0

                for row in recent_data:
                    trade_date, close, volume, amount, price_change_pct, turnover_rate, high, low = row

                    if amount and volume and price_change_pct is not None:
                        # 基于价格变化和成交量的资金流向权重
                        flow_weight = float(price_change_pct) * (float(volume) / 1000000)  # 标准化成交量
                        total_amount += abs(flow_weight)
                        weighted_price_change += flow_weight
                        total_volume += float(volume)

                # 计算净资金流向指标
                if total_amount > 0:
                    capital_flow_indicator = weighted_price_change / total_amount
                else:
                    capital_flow_indicator = 0.0

                # 标准化到 [-1, 1] 范围
                capital_flow_indicator = max(-1.0, min(1.0, capital_flow_indicator * 10))

                return {'capital_flow_indicator': capital_flow_indicator}

        except Exception as e:
            self.logger.error(f"计算资金流向指标失败: {e}")
            return {'capital_flow_indicator': 0.0}

    def _compute_market_sentiment_index(self, code: str, current_time: datetime) -> Dict[str, float]:
        """
        计算市场情绪指数 (VIX等价物)

        基于波动率和市场表现的恐慌指数:
        - 历史波动率
        - 价格跳空频率
        - 市场宽度指标
        """
        try:
            # 获取市场主要指数数据用于计算市场情绪
            query = """
            SELECT id.close, id.pct_chg, id.vol, CAST(id.trade_date AS TEXT) as trade_date_str
            FROM index_daily id
            JOIN market_indices mi ON id.index_id = mi.id
            WHERE mi.ts_code = '000001.SH'
            ORDER BY id.trade_date DESC
            LIMIT 20
            """

            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                market_data = cursor.fetchall()

                if len(market_data) < 10:
                    return {'market_sentiment_index': 0.5}

                # 计算市场波动率 (20日)
                price_changes = []
                volume_data = []

                for row in market_data:
                    close, pct_chg, vol, trade_date_str = row
                    if pct_chg is not None:
                        price_changes.append(float(pct_chg))
                    if vol is not None:
                        volume_data.append(float(vol))

                # 历史波动率 (年化)
                if len(price_changes) > 1:
                    volatility = np.std(price_changes) * np.sqrt(252)  # 年化波动率
                else:
                    volatility = 0.15  # 默认15%

                # 计算跳空频率 (大于2%变动的天数比例)
                large_moves = sum(1 for change in price_changes if abs(change) > 2.0)
                gap_frequency = large_moves / len(price_changes) if price_changes else 0

                # 成交量变化率
                if len(volume_data) > 1:
                    volume_volatility = np.std(volume_data) / np.mean(volume_data)
                else:
                    volume_volatility = 0.3

                # 综合情绪指数 (类似VIX)
                # 正常市场情绪约0.3-0.7，恐慌时>0.8，过度乐观时<0.2
                sentiment_index = (
                    volatility * 0.5 +           # 价格波动权重50%
                    gap_frequency * 0.3 +        # 跳空频率权重30%
                    volume_volatility * 0.2       # 成交量波动权重20%
                )

                # 标准化到 [0, 1] 范围
                sentiment_index = max(0.0, min(1.0, sentiment_index))

                return {'market_sentiment_index': sentiment_index}

        except Exception as e:
            self.logger.error(f"计算市场情绪指数失败: {e}")
            return {'market_sentiment_index': 0.5}

    def _compute_sector_rotation_strength(self, code: str, current_time: datetime) -> Dict[str, float]:
        """
        计算板块轮动强度

        分析不同行业间的资金流动和表现差异:
        - 行业相对表现
        - 板块资金流向
        - 轮动活跃度
        """
        try:
            # 获取股票所属行业
            stock_industry_query = """
            SELECT industry FROM securities WHERE code = ?
            """

            with self.db_manager.get_connection() as conn:
                result = conn.execute(stock_industry_query, (code.split('.')[0] if '.' in code else code,)).fetchone()

                if not result or not result[0]:
                    return {'sector_rotation_strength': 0.5}

                stock_industry = result[0]

                # 获取主要行业的最近表现
                industries_query = """
                SELECT s.industry,
                       AVG(dq.price_change_pct) as avg_change,
                       SUM(dq.amount) as total_amount,
                       COUNT(*) as stock_count
                FROM securities s
                JOIN daily_quotes dq ON s.id = dq.security_id
                WHERE s.industry IS NOT NULL
                AND dq.trade_date >= date('now', '-3 days')
                GROUP BY s.industry
                HAVING stock_count >= 5
                ORDER BY avg_change DESC
                LIMIT 20
                """

                cursor = conn.cursor()
                cursor.execute(industries_query)
                industries_data = cursor.fetchall()

                if len(industries_data) < 5:
                    return {'sector_rotation_strength': 0.5}

                # 分析板块表现差异
                industry_changes = [float(row[1]) if row[1] else 0 for row in industries_data]
                industry_amounts = [float(row[2]) if row[2] else 0 for row in industries_data]

                # 计算板块表现标准差 (轮动活跃度)
                if len(industry_changes) > 1:
                    sector_volatility = np.std(industry_changes)
                else:
                    sector_volatility = 0.5

                # 找到目标股票行业的排名
                stock_industry_rank = 0.5  # 默认中等
                for i, row in enumerate(industries_data):
                    if row[0] == stock_industry:
                        stock_industry_rank = 1.0 - (i / len(industries_data))  # 排名越靠前值越大
                        break

                # 综合板块轮动强度
                # 考虑整体轮动活跃度和该股票所在行业的相对表现
                rotation_strength = (
                    sector_volatility * 0.6 +      # 整体轮动活跃度60%
                    stock_industry_rank * 0.4      # 所在行业表现40%
                )

                # 标准化到 [0, 1] 范围
                rotation_strength = max(0.0, min(1.0, rotation_strength))

                return {'sector_rotation_strength': rotation_strength}

        except Exception as e:
            self.logger.error(f"计算板块轮动强度失败: {e}")
            return {'sector_rotation_strength': 0.5}

    def _compute_northbound_capital_impact(self, code: str, current_time: datetime) -> Dict[str, float]:
        """
        计算北向资金流向影响

        由于没有直接的北向资金数据，基于以下方法估算:
        - 大盘股vs小盘股表现差异 (北向偏好大盘股)
        - 外资重仓股表现
        - 市场流动性指标
        """
        try:
            # 获取股票基本信息
            stock_info_query = """
            SELECT s.name, db.total_mv, db.circ_mv, s.exchange
            FROM securities s
            LEFT JOIN daily_basic db ON s.id = db.security_id
            WHERE s.code = ?
            AND db.trade_date = (SELECT MAX(trade_date) FROM daily_basic WHERE security_id = s.id)
            """

            with self.db_manager.get_connection() as conn:
                result = conn.execute(stock_info_query, (code.split('.')[0] if '.' in code else code,)).fetchone()

                if not result:
                    return {'northbound_capital_impact': 0.5}

                name, total_mv, circ_mv, exchange = result
                market_cap = float(total_mv) if total_mv else 1000000  # 默认10亿市值

                # 基于市值判断北向资金偏好程度
                # 北向资金通常偏好大盘蓝筹股
                if market_cap > 100000000000:  # 1000亿以上
                    size_preference = 0.9
                elif market_cap > 50000000000:  # 500亿以上
                    size_preference = 0.7
                elif market_cap > 10000000000:  # 100亿以上
                    size_preference = 0.5
                else:
                    size_preference = 0.3

                # 获取大盘股与小盘股的表现差异
                large_cap_performance = self._get_large_cap_performance()
                small_cap_performance = self._get_small_cap_performance()

                # 计算大小盘风格差异 (正值表示大盘股跑赢)
                style_differential = large_cap_performance - small_cap_performance

                # 北向资金影响估算
                # 如果大盘股表现好且该股票是大盘股，则北向影响为正
                northbound_impact = (
                    size_preference * 0.7 +           # 股票本身吸引力70%
                    (style_differential + 1) * 0.3    # 大小盘风格30% (标准化到正值)
                )

                # 标准化到 [0, 1] 范围
                northbound_impact = max(0.0, min(1.0, northbound_impact))

                return {'northbound_capital_impact': northbound_impact}

        except Exception as e:
            self.logger.error(f"计算北向资金影响失败: {e}")
            return {'northbound_capital_impact': 0.5}

    def _get_large_cap_performance(self) -> float:
        """获取大盘股近期表现"""
        try:
            query = """
            SELECT AVG(dq.price_change_pct) as avg_change
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            JOIN daily_basic db ON dq.security_id = db.security_id AND dq.trade_date = db.trade_date
            WHERE db.total_mv > 50000000000
            AND dq.trade_date >= date('now', '-5 days')
            """

            with self.db_manager.get_connection() as conn:
                result = conn.execute(query).fetchone()
                return float(result[0]) if result and result[0] else 0.0

        except Exception:
            return 0.0

    def _get_small_cap_performance(self) -> float:
        """获取小盘股近期表现"""
        try:
            query = """
            SELECT AVG(dq.price_change_pct) as avg_change
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            JOIN daily_basic db ON dq.security_id = db.security_id AND dq.trade_date = db.trade_date
            WHERE db.total_mv < 10000000000
            AND dq.trade_date >= date('now', '-5 days')
            """

            with self.db_manager.get_connection() as conn:
                result = conn.execute(query).fetchone()
                return float(result[0]) if result and result[0] else 0.0

        except Exception:
            return 0.0

    def _update_market_baseline(self, current_time: datetime):
        """更新市场基准数据"""
        if (self.baseline_timestamp is None or
            (current_time - self.baseline_timestamp).total_seconds() > self.cache_ttl):

            # 这里可以缓存一些市场整体的基准数据
            self.market_baseline = {
                'update_time': current_time,
                'market_volatility': 0.2,  # 默认市场波动率
                'avg_sentiment': 0.5       # 默认市场情绪
            }
            self.baseline_timestamp = current_time

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.sentiment_cache:
            return False

        if cache_key not in self.cache_timestamps:
            return False

        elapsed = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return elapsed < self.cache_ttl

    def _get_default_sentiment_indicators(self) -> Dict[str, float]:
        """获取默认情绪指标值"""
        return {
            'capital_flow_indicator': 0.0,      # 资金流向中性
            'market_sentiment_index': 0.5,      # 市场情绪中性
            'sector_rotation_strength': 0.5,    # 板块轮动中等
            'northbound_capital_impact': 0.5    # 北向资金影响中等
        }

    def clear_cache(self):
        """清理情绪指标缓存"""
        self.sentiment_cache.clear()
        self.cache_timestamps.clear()
        self.market_baseline = None
        self.baseline_timestamp = None
        self.logger.info("🗑️ 清理情绪指标缓存")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'total_sentiment_cache': len(self.sentiment_cache),
            'has_market_baseline': self.market_baseline is not None,
            'cache_memory_mb': sys.getsizeof(self.sentiment_cache) / 1024 / 1024
        }