#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据适配器 - 为自适应评分系统提供标准化数据接口

将数据库中的字段名映射到评分系统期望的标准字段名
处理缺失数据和数据质量问题

Created: 2025-09-16
Author: Claude Code
Version: 3.8.0
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager

class AdaptiveScoringDataAdapter:
    """
    自适应评分系统数据适配器

    功能：
    1. 标准化数据库字段名
    2. 处理缺失数据
    3. 计算衍生指标
    4. 数据质量验证
    """

    def __init__(self, db_path: str = "data_adapter/stock_data.db", logger: Optional[logging.Logger] = None):
        self.db_manager = DatabaseManager(db_path)
        self.logger = logger or logging.getLogger(__name__)

        # 字段映射配置
        self.field_mapping = {
            # 基础价格数据映射
            'price_fields': {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'trade_date': 'trade_date'
            },

            # 技术指标映射
            'technical_fields': {
                'rsi': 'rsi6',  # 使用6日RSI作为默认RSI
                'rsi_12': 'rsi12',
                'rsi_24': 'rsi24',
                'kdj_k': 'kdj_k',
                'kdj_d': 'kdj_d',
                'kdj_j': 'kdj_j',
                'macd': 'macd_dif',  # MACD DIF线
                'macd_signal': 'macd_dea',  # MACD DEA线
                'macd_histogram': 'macd_macd',  # MACD柱状图
                'bb_upper': 'boll_upper',  # 布林带上轨
                'bb_middle': 'boll_middle',  # 布林带中轨
                'bb_lower': 'boll_lower',  # 布林带下轨
                'bbi': 'bbi',
                'volume_ma5': 'volume_ma5',
                'volume_ratio': 'volume_ratio'
            },

            # 移动平均线映射
            'ma_fields': {
                'ma5': 'ma5',
                'ma10': 'ma10',
                'ma20': 'ma20',
                'ma60': 'ma60'
            }
        }

        self.logger.info("自适应评分数据适配器初始化完成")

    def get_stock_data(self, stock_code: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       days: int = 120) -> pd.DataFrame:
        """
        获取股票的标准化技术数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数（如果没有指定日期范围）

        Returns:
            标准化的股票技术数据DataFrame
        """
        try:
            # 如果没有指定日期范围，获取最近指定天数的数据
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            self.logger.info(f"获取股票 {stock_code} 从 {start_date} 到 {end_date} 的数据")

            with self.db_manager.get_connection() as conn:
                # 构建查询SQL - 联合daily_quotes和technical_indicators
                sql = """
                    SELECT
                        dq.trade_date,
                        dq.open, dq.high, dq.low, dq.close, dq.volume,
                        dq.ma5, dq.ma10, dq.ma20, dq.ma60,
                        ti.kdj_k, ti.kdj_d, ti.kdj_j,
                        ti.rsi6, ti.rsi12, ti.rsi24,
                        ti.macd_dif, ti.macd_dea, ti.macd_macd,
                        ti.boll_upper, ti.boll_middle, ti.boll_lower,
                        ti.bbi, ti.volume_ma5, ti.volume_ratio
                    FROM daily_quotes dq
                    LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ?
                        AND dq.trade_date >= ?
                        AND dq.trade_date <= ?
                    ORDER BY dq.trade_date ASC
                """

                raw_data = pd.read_sql(sql, conn, params=[stock_code, start_date, end_date])

                if raw_data.empty:
                    self.logger.warning(f"未找到股票 {stock_code} 的数据")
                    return pd.DataFrame()

                # 应用字段映射和数据处理
                processed_data = self._process_stock_data(raw_data)

                self.logger.info(f"成功获取股票 {stock_code} 数据，共 {len(processed_data)} 天")
                return processed_data

        except Exception as e:
            self.logger.error(f"获取股票数据失败 {stock_code}: {e}")
            return pd.DataFrame()

    def _process_stock_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """处理和标准化股票数据"""

        try:
            # 创建标准化数据框
            processed = pd.DataFrame()

            # 1. 复制基础字段
            for standard_field, db_field in self.field_mapping['price_fields'].items():
                if db_field in raw_data.columns:
                    processed[standard_field] = raw_data[db_field]

            # 2. 复制移动平均线
            for standard_field, db_field in self.field_mapping['ma_fields'].items():
                if db_field in raw_data.columns:
                    processed[standard_field] = raw_data[db_field]

            # 3. 映射技术指标字段
            for standard_field, db_field in self.field_mapping['technical_fields'].items():
                if db_field in raw_data.columns:
                    processed[standard_field] = raw_data[db_field]

            # 4. 计算衍生指标
            processed = self._calculate_derived_indicators(processed)

            # 5. 数据清理和质量检查
            processed = self._clean_and_validate_data(processed)

            return processed

        except Exception as e:
            self.logger.error(f"数据处理失败: {e}")
            return pd.DataFrame()

    def _calculate_derived_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算衍生技术指标"""

        try:
            # 确保数据按日期排序
            if 'trade_date' in data.columns:
                data = data.sort_values('trade_date').reset_index(drop=True)

            # 计算EMA (如果没有的话)
            if 'close' in data.columns:
                # 12日和26日EMA用于MACD计算
                data['ema12'] = data['close'].ewm(span=12).mean()
                data['ema26'] = data['close'].ewm(span=26).mean()

                # 如果没有MACD数据，计算MACD
                if 'macd' not in data.columns or data['macd'].isna().all():
                    data['macd'] = data['ema12'] - data['ema26']
                    if 'macd_signal' not in data.columns or data['macd_signal'].isna().all():
                        data['macd_signal'] = data['macd'].ewm(span=9).mean()
                    data['macd_histogram'] = data['macd'] - data['macd_signal']

            # 计算布林带位置百分比
            if all(col in data.columns for col in ['close', 'bb_upper', 'bb_lower']):
                bb_width = data['bb_upper'] - data['bb_lower']
                data['bb_position'] = np.where(
                    bb_width != 0,
                    (data['close'] - data['bb_lower']) / bb_width,
                    0.5
                )

            # 计算价格相对于移动平均线的位置
            for ma_period in ['ma5', 'ma10', 'ma20', 'ma60']:
                if ma_period in data.columns and 'close' in data.columns:
                    ma_ratio_col = f'{ma_period}_ratio'
                    data[ma_ratio_col] = data['close'] / data[ma_period]

            return data

        except Exception as e:
            self.logger.error(f"衍生指标计算失败: {e}")
            return data

    def _clean_and_validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据清理和验证"""

        try:
            original_rows = len(data)

            # 1. 移除全为NaN的行
            data = data.dropna(how='all')

            # 2. 处理关键字段的NaN值
            critical_fields = ['open', 'high', 'low', 'close', 'volume']
            for field in critical_fields:
                if field in data.columns:
                    # 如果关键字段有NaN，尝试向前填充
                    data[field] = data[field].fillna(method='ffill')

            # 3. 技术指标NaN值处理 - 不要简单填充，保持NaN让上层处理
            # 这样上层可以知道哪些指标无法计算

            # 4. 数据类型转换
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # 5. 日期格式处理
            if 'trade_date' in data.columns:
                data['trade_date'] = pd.to_datetime(data['trade_date'])

            cleaned_rows = len(data)
            if original_rows != cleaned_rows:
                self.logger.info(f"数据清理：{original_rows} -> {cleaned_rows} 行")

            return data

        except Exception as e:
            self.logger.error(f"数据清理失败: {e}")
            return data

    def get_fundamental_data(self, stock_code: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """获取基本面数据 - 使用可用数据构建基本面指标"""

        try:
            # 由于数据库中没有专门的financial_indicator表
            # 我们从daily_quotes和technical_indicators构建基本面相关指标

            with self.db_manager.get_connection() as conn:
                # 构建基本面相关指标
                sql = """
                    SELECT
                        dq.trade_date,
                        dq.close * dq.volume / 1000000.0 as turnover_amount,  -- 成交额(百万)
                        CASE
                            WHEN ti.volume_ma5 > 0 THEN dq.volume / ti.volume_ma5
                            ELSE 1.0
                        END as volume_ratio,  -- 量比（使用technical_indicators表的volume_ma5）
                        (dq.high - dq.low) / dq.close as volatility,  -- 日内波动率
                        dq.price_change_pct / 100.0 as daily_return,  -- 日收益率
                        CASE
                            WHEN dq.ma60 > 0 THEN dq.ma20 / dq.ma60
                            ELSE 1.0
                        END as trend_strength,  -- 趋势强度
                        CASE
                            WHEN ti.rsi6 > 70 THEN 'overbought'
                            WHEN ti.rsi6 < 30 THEN 'oversold'
                            ELSE 'normal'
                        END as rsi_signal,
                        CASE
                            WHEN ti.rsi6 IS NOT NULL THEN ti.rsi6 / 50.0 - 1.0
                            ELSE 0.0
                        END as rsi_normalized,  -- RSI标准化(-1到1)
                        CASE
                            WHEN dq.ma20 > 0 THEN (dq.close - dq.ma20) / dq.ma20
                            ELSE 0.0
                        END as ma20_deviation,  -- 偏离20日均线程度
                        -- 添加基本面指标
                        db.pe_ttm,
                        db.pb,
                        db.ps_ttm,
                        db.turnover_rate,
                        db.total_mv,
                        db.circ_mv
                    FROM daily_quotes dq
                    LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
                    LEFT JOIN daily_basic db ON dq.security_id = db.security_id AND dq.trade_date = db.trade_date
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ?
                """

                params = [stock_code]
                if start_date and end_date:
                    sql += " AND dq.trade_date >= ? AND dq.trade_date <= ?"
                    params.extend([start_date, end_date])

                sql += " ORDER BY dq.trade_date DESC LIMIT 20"  # 最近20个交易日

                fundamental_data = pd.read_sql(sql, conn, params=params)

                # 添加一些计算指标
                if not fundamental_data.empty:
                    # 计算移动平均收益率
                    fundamental_data['avg_return_5d'] = fundamental_data['daily_return'].rolling(5).mean()
                    fundamental_data['avg_return_10d'] = fundamental_data['daily_return'].rolling(10).mean()

                    # 计算波动率
                    fundamental_data['volatility_5d'] = fundamental_data['daily_return'].rolling(5).std()
                    fundamental_data['volatility_10d'] = fundamental_data['daily_return'].rolling(10).std()

                    # 计算动量指标
                    fundamental_data['momentum_5d'] = fundamental_data['daily_return'].rolling(5).sum()
                    fundamental_data['momentum_10d'] = fundamental_data['daily_return'].rolling(10).sum()

                return fundamental_data

        except Exception as e:
            self.logger.error(f"获取基本面数据失败 {stock_code}: {e}")
            return pd.DataFrame()

    def get_market_data(self, index_code: str = '000001',
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       days: int = 120) -> pd.DataFrame:
        """获取市场指数数据 - 使用大盘股代表市场趋势"""

        try:
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            with self.db_manager.get_connection() as conn:
                # 由于没有index_daily表，使用大盘股代表市场数据
                # 选择平安银行(000001)作为市场指标的代表
                sql = """
                    SELECT
                        dq.trade_date,
                        dq.open, dq.high, dq.low, dq.close, dq.volume,
                        dq.price_change_pct,
                        dq.ma5, dq.ma10, dq.ma20,
                        ti.rsi6 as market_rsi
                    FROM daily_quotes dq
                    LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ?
                        AND dq.trade_date >= ?
                        AND dq.trade_date <= ?
                    ORDER BY dq.trade_date ASC
                """

                market_data = pd.read_sql(sql, conn, params=[index_code, start_date, end_date])

                if not market_data.empty:
                    # 计算市场动量和波动率指标
                    market_data['market_momentum_5d'] = market_data['price_change_pct'].rolling(5).mean()
                    market_data['market_volatility'] = market_data['price_change_pct'].rolling(10).std()
                    market_data['market_trend'] = market_data['close'] / market_data['ma20'] - 1

                return market_data

        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            return pd.DataFrame()

    def validate_data_quality(self, data: pd.DataFrame, data_type: str = 'stock') -> Dict[str, any]:
        """验证数据质量"""

        if data.empty:
            return {'status': 'empty', 'issues': ['数据为空']}

        issues = []
        metrics = {}

        # 1. 基本统计
        metrics['total_rows'] = len(data)
        metrics['total_columns'] = len(data.columns)

        # 2. 缺失值检查
        missing_ratio = data.isnull().sum() / len(data)
        high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
        if high_missing_cols:
            issues.append(f"高缺失率字段: {high_missing_cols}")

        metrics['missing_ratios'] = missing_ratio.to_dict()

        # 3. 数据连续性检查
        if 'trade_date' in data.columns:
            data_sorted = data.sort_values('trade_date')
            date_gaps = pd.to_datetime(data_sorted['trade_date']).diff().dt.days
            large_gaps = date_gaps[date_gaps > 5]  # 超过5天的空隙
            if not large_gaps.empty:
                issues.append(f"发现 {len(large_gaps)} 个大的日期空隙")

        # 4. 数值合理性检查
        if data_type == 'stock':
            # 价格字段应该为正
            price_fields = ['open', 'high', 'low', 'close']
            for field in price_fields:
                if field in data.columns:
                    negative_count = (data[field] <= 0).sum()
                    if negative_count > 0:
                        issues.append(f"{field} 有 {negative_count} 个非正值")

            # 成交量应该为非负
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    issues.append(f"成交量有 {negative_volume} 个负值")

        # 5. 综合质量评分
        issue_weight = len(issues) * 0.1
        missing_weight = missing_ratio.mean() * 0.5
        quality_score = max(0, 1 - issue_weight - missing_weight)

        return {
            'status': 'good' if quality_score > 0.8 else 'warning' if quality_score > 0.5 else 'poor',
            'quality_score': quality_score,
            'issues': issues,
            'metrics': metrics
        }

    def get_data_summary(self, stock_code: str) -> Dict[str, any]:
        """获取股票数据摘要"""

        try:
            with self.db_manager.get_connection() as conn:
                # 检查数据可用性
                sql = """
                    SELECT
                        MIN(dq.trade_date) as earliest_date,
                        MAX(dq.trade_date) as latest_date,
                        COUNT(dq.id) as price_records,
                        COUNT(ti.id) as tech_records
                    FROM daily_quotes dq
                    LEFT JOIN technical_indicators ti ON dq.security_id = ti.security_id AND dq.trade_date = ti.trade_date
                    JOIN securities s ON dq.security_id = s.id
                    WHERE s.code = ?
                """

                summary = pd.read_sql(sql, conn, params=[stock_code])

                if not summary.empty:
                    result = summary.iloc[0].to_dict()
                    result['tech_coverage'] = result['tech_records'] / result['price_records'] if result['price_records'] > 0 else 0
                    return result
                else:
                    return {'error': '未找到股票数据'}

        except Exception as e:
            self.logger.error(f"获取数据摘要失败: {e}")
            return {'error': str(e)}

    def __repr__(self):
        return f"AdaptiveScoringDataAdapter(db_path={self.db_manager.db_path})"