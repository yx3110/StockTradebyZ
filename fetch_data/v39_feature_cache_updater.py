#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9/V3.95 特征缓存每日更新器 (优化版)
用于每日数据更新后，更新v39_feature_cache表中的特征和市场状态

优化内容:
1. 批量预加载数据，避免5000+次独立SQL查询
2. 缓存市场数据，避免重复计算
3. 简化特征计算流程，使用轻量级方法
4. 添加进度日志和超时处理
"""

import os
import sys
import json
import sqlite3
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
import bisect
import time
import signal

# 添加项目路径
try:
    from core.config import PROJECT_ROOT, get_db_path
    _DB_PATH = str(get_db_path())
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent
    _DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class V39FeatureCacheUpdaterOptimized:
    """V3.9/V3.95 特征缓存更新器 (优化版)"""

    def __init__(self, db_path: str = None):
        """初始化更新器"""
        if db_path is None:
            db_path = _DB_PATH
        self.db_path = db_path

        # 市场指数代码 (用于计算市场状态特征)
        self.market_index = '000300.SH'  # 沪深300

        # 特征系统 (延迟加载) - 使用轻量级版本
        self._feature_system = None
        self._use_lightweight = True  # 默认使用轻量级模式

        # 数据缓存
        self._stock_data_cache = {}
        self._market_data_cache = None

        # 批量加载的数据
        self._batch_stock_data = None
        self._batch_market_features = None

        # 行业数据缓存
        self._sw_industry_mapping = None  # code -> l1_name
        self._sw_l1_label_encoding = None  # l1_name -> int
        self._industry_valuation_cache = None  # {code: {pe_rank, pb_rank, ps_rank}}
        self._industry_return_cache = None  # {code: {return_5d, return_20d, relative_strength}}
        self._industry_daily_stats_cache = None  # {l1_name: {breadth, volume_change, ...}}
        self._sw_index_return_cache = None  # {l1_name: {return_1d, return_5d}}
        self._northbound_flow_cache = None  # {flow_5d: float}

    def get_stock_list(self) -> List[str]:
        """获取所有A股代码"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT code FROM securities
            WHERE type = 'A股'
            ORDER BY code
        """)
        stocks = [row[0] for row in cursor.fetchall()]
        conn.close()
        return stocks

    def _batch_load_stock_data(self, date: str, lookback: int = 60) -> Dict[str, pd.DataFrame]:
        """
        批量加载所有股票的历史数据

        优化: 一次查询获取所有需要的数据，而不是每只股票单独查询
        """
        logger.info(f"批量预加载股票数据 (lookback={lookback}天)...")
        start_time = time.time()

        # 计算开始日期
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=lookback + 30)  # 多加一些余量

        conn = sqlite3.connect(self.db_path)

        # 一次性获取所有A股的历史数据
        query = """
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume, q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股'
        AND q.trade_date >= ?
        AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
        """

        df = pd.read_sql_query(query, conn, params=(start_date.strftime('%Y-%m-%d'), date))
        conn.close()

        # 按股票代码分组
        stock_data = {}
        for code, group in df.groupby('code'):
            stock_data[code] = group.reset_index(drop=True)

        elapsed = time.time() - start_time
        logger.info(f"批量加载完成: {len(stock_data)} 只股票, 耗时 {elapsed:.1f}秒")

        return stock_data

    def calculate_market_features(self, date: str, lookback_days: int = 30) -> Dict[str, float]:
        """
        计算市场状态特征 (优化版 - 使用缓存)
        """
        # 检查缓存
        if self._batch_market_features is not None:
            return self._batch_market_features

        # 获取足够的历史数据
        end_date = datetime.strptime(date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=lookback_days + 10)

        conn = sqlite3.connect(self.db_path)

        # 获取市场数据
        market_query = """
        SELECT q.trade_date, q.close, q.volume
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE (s.code = '000300.SH' OR s.name = '沪深300')
        AND q.trade_date >= ?
        AND q.trade_date <= ?
        ORDER BY q.trade_date
        """

        market_df = pd.read_sql_query(market_query, conn,
                                       params=(start_date.strftime('%Y-%m-%d'), date))

        # 获取上涨/下跌股票数量
        up_down_query = """
        SELECT q.trade_date,
               SUM(CASE WHEN q.price_change_pct > 0 THEN 1 ELSE 0 END) as up_count,
               COUNT(*) as total_count
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股'
        AND q.trade_date >= ?
        AND q.trade_date <= ?
        GROUP BY q.trade_date
        ORDER BY q.trade_date
        """

        up_down_df = pd.read_sql_query(up_down_query, conn,
                                        params=(start_date.strftime('%Y-%m-%d'), date))
        conn.close()

        # 初始化特征字典
        features = {
            'market_return_20d': 0.0,
            'market_return_10d': 0.0,
            'market_return_5d': 0.0,
            'market_volatility_20d': 0.0,
            'market_volatility_10d': 0.0,
            'market_up_ratio_20d': 0.5,
            'market_up_ratio_10d': 0.5,
            'market_drawdown_20d': 0.0,
            'market_volume_ratio': 1.0,
            'market_position_20d': 0.5,
            'market_momentum_20d': 0.0,
            'market_momentum_5d': 0.0
        }

        if market_df.empty or len(market_df) < 5:
            return features

        # 计算市场收益
        closes = market_df['close'].values
        if len(closes) >= 20:
            features['market_return_20d'] = (closes[-1] / closes[-20] - 1)
        if len(closes) >= 10:
            features['market_return_10d'] = (closes[-1] / closes[-10] - 1)
        if len(closes) >= 5:
            features['market_return_5d'] = (closes[-1] / closes[-5] - 1)

        # 计算波动率 (日收益率标准差 * sqrt(252))
        returns = np.diff(np.log(closes))
        if len(returns) >= 20:
            features['market_volatility_20d'] = np.std(returns[-20:]) * np.sqrt(252)
        if len(returns) >= 10:
            features['market_volatility_10d'] = np.std(returns[-10:]) * np.sqrt(252)

        # 计算上涨比例
        if not up_down_df.empty:
            up_down_df['up_ratio'] = up_down_df['up_count'] / up_down_df['total_count']
            if len(up_down_df) >= 20:
                features['market_up_ratio_20d'] = up_down_df['up_ratio'].tail(20).mean()
            if len(up_down_df) >= 10:
                features['market_up_ratio_10d'] = up_down_df['up_ratio'].tail(10).mean()

        # 计算最大回撤
        if len(closes) >= 20:
            rolling_max = np.maximum.accumulate(closes[-20:])
            drawdowns = (closes[-20:] - rolling_max) / rolling_max
            features['market_drawdown_20d'] = np.min(drawdowns)

        # 计算成交量比
        volumes = market_df['volume'].values
        if len(volumes) >= 20:
            avg_vol = np.mean(volumes[-20:])
            if avg_vol > 0:
                features['market_volume_ratio'] = volumes[-1] / avg_vol

        # 计算20日相对位置
        if len(closes) >= 20:
            high_20d = np.max(closes[-20:])
            low_20d = np.min(closes[-20:])
            if high_20d > low_20d:
                features['market_position_20d'] = (closes[-1] - low_20d) / (high_20d - low_20d)

        # 计算动量 (当前价格相对于N日均线的偏离)
        if len(closes) >= 20:
            ma20 = np.mean(closes[-20:])
            features['market_momentum_20d'] = (closes[-1] / ma20 - 1)
        if len(closes) >= 5:
            ma5 = np.mean(closes[-5:])
            features['market_momentum_5d'] = (closes[-1] / ma5 - 1)

        # 缓存结果
        self._batch_market_features = features
        return features

    def calculate_labels(self, code: str, date: str) -> Dict[str, Optional[float]]:
        """计算3d, 5d, 10d, 15d未来收益标签"""
        conn = sqlite3.connect(self.db_path)

        # 获取当天和未来的收盘价
        query = """
        SELECT q.trade_date, q.close
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.code = ?
        AND q.trade_date >= ?
        ORDER BY q.trade_date
        LIMIT 20
        """

        df = pd.read_sql_query(query, conn, params=(code, date))
        conn.close()

        labels = {
            'label_3d': None,
            'label_5d': None,
            'label_10d': None,
            'label_15d': None
        }

        if df.empty or len(df) < 2:
            return labels

        base_close = df.iloc[0]['close']
        if base_close <= 0:
            return labels

        closes = df['close'].tolist()

        # 计算各期限收益
        if len(closes) > 3:
            labels['label_3d'] = (closes[3] / base_close - 1)
        if len(closes) > 5:
            labels['label_5d'] = (closes[5] / base_close - 1)
        if len(closes) > 10:
            labels['label_10d'] = (closes[10] / base_close - 1)
        if len(closes) > 15:
            labels['label_15d'] = (closes[15] / base_close - 1)

        return labels

    def _load_sw_industry_mapping(self):
        """从sw_industry表加载行业映射 (code -> l1_name, l1_name -> label)"""
        if self._sw_industry_mapping is not None:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sw_industry'")
        if not cursor.fetchone():
            logger.warning("sw_industry表不存在，行业特征将使用默认值")
            self._sw_industry_mapping = {}
            self._sw_l1_label_encoding = {}
            conn.close()
            return

        # code -> l1_name
        cursor.execute("SELECT code, l1_name FROM sw_industry WHERE is_new = 'Y'")
        self._sw_industry_mapping = {row[0]: row[1] for row in cursor.fetchall()}

        # l1_name -> label (排序后编码)
        cursor.execute("SELECT DISTINCT l1_name FROM sw_industry WHERE is_new = 'Y' ORDER BY l1_name")
        names = [row[0] for row in cursor.fetchall()]
        self._sw_l1_label_encoding = {name: i for i, name in enumerate(names)}

        conn.close()
        logger.info(f"加载申万行业映射: {len(self._sw_industry_mapping)} 只股票, "
                     f"{len(self._sw_l1_label_encoding)} 个行业")

    def _batch_load_industry_valuation(self, date: str):
        """
        批量加载行业内估值排名数据

        一次SQL获取所有股票的PE/PB/PS，按行业分组计算分位数
        """
        if self._industry_valuation_cache is not None:
            return

        self._load_sw_industry_mapping()
        if not self._sw_industry_mapping:
            self._industry_valuation_cache = {}
            return

        conn = sqlite3.connect(self.db_path)

        # 获取当天或最近的daily_basic数据
        query = """
        SELECT s.code, db.pe_ttm, db.pb, db.ps_ttm
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE s.type = 'A股'
        AND db.trade_date = (
            SELECT MAX(trade_date) FROM daily_basic WHERE trade_date <= ?
        )
        """
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()

        if df.empty:
            self._industry_valuation_cache = {}
            return

        # 添加行业列
        df['l1_name'] = df['code'].map(self._sw_industry_mapping)
        df = df.dropna(subset=['l1_name'])

        # 按行业分组计算分位数
        result = {}
        for col, rank_name in [('pe_ttm', 'pe_rank'), ('pb', 'pb_rank'), ('ps_ttm', 'ps_rank')]:
            # 去掉异常值 (负值或极端值)
            valid = df[df[col].notna() & (df[col] > 0)].copy()
            if valid.empty:
                continue
            valid[rank_name] = valid.groupby('l1_name')[col].rank(pct=True)
            for _, row in valid.iterrows():
                code = row['code']
                if code not in result:
                    result[code] = {}
                result[code][rank_name] = row[rank_name]

        self._industry_valuation_cache = result
        logger.info(f"加载行业估值排名: {len(result)} 只股票")

    def _batch_load_industry_returns(self, date: str, stock_data_map: Dict[str, pd.DataFrame]):
        """
        利用已加载的行情数据，计算行业平均收益率和个股相对强度

        Args:
            date: 当前日期
            stock_data_map: 已预加载的 {code: DataFrame} 股票数据
        """
        if self._industry_return_cache is not None:
            return

        self._load_sw_industry_mapping()
        if not self._sw_industry_mapping:
            self._industry_return_cache = {}
            return

        # 收集每只股票的5d和20d收益率
        stock_returns = {}
        for code, df in stock_data_map.items():
            if df is None or len(df) < 5:
                continue
            closes = df['close'].values
            r5 = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 and closes[-5] > 0 else 0.0
            r20 = (closes[-1] / closes[-20] - 1) if len(closes) >= 20 and closes[-20] > 0 else 0.0
            stock_returns[code] = {'return_5d': r5, 'return_20d': r20}

        # 按行业分组计算平均收益
        industry_avg = {}  # l1_name -> {avg_5d, avg_20d}
        industry_stocks = {}  # l1_name -> [returns...]

        for code, ret in stock_returns.items():
            l1_name = self._sw_industry_mapping.get(code)
            if not l1_name:
                continue
            if l1_name not in industry_stocks:
                industry_stocks[l1_name] = []
            industry_stocks[l1_name].append(ret)

        for l1_name, stocks in industry_stocks.items():
            if not stocks:
                continue
            avg_5d = np.mean([s['return_5d'] for s in stocks])
            avg_20d = np.mean([s['return_20d'] for s in stocks])
            industry_avg[l1_name] = {'avg_5d': avg_5d, 'avg_20d': avg_20d}

        # 计算每只股票的行业收益和相对强度
        result = {}
        for code, ret in stock_returns.items():
            l1_name = self._sw_industry_mapping.get(code)
            if not l1_name or l1_name not in industry_avg:
                continue
            avg = industry_avg[l1_name]
            # 相对强度 = 个股5d收益 - 行业平均5d收益
            relative_strength = ret['return_5d'] - avg['avg_5d']
            result[code] = {
                'industry_return_5d': avg['avg_5d'],
                'industry_return_20d': avg['avg_20d'],
                'industry_relative_strength': relative_strength,
            }

        self._industry_return_cache = result
        logger.info(f"计算行业收益特征: {len(result)} 只股票, {len(industry_avg)} 个行业")

    def _batch_load_industry_daily_stats(self, date: str, stock_data_map: Dict[str, pd.DataFrame]):
        """
        批量计算行业日度统计特征 (6个新特征)

        Features:
            industry_breadth: 行业内上涨股票比例 (0-1)
            industry_volume_change: 行业成交量 vs 5日均量 (0+)
            industry_limit_up_ratio: 行业内涨停比例 (0-1)
            industry_kdj_avg: 行业平均KDJ_J (-∞~+∞)
            industry_macd_bullish_pct: 行业MACD金叉比例 (0-1)
            industry_concentration: 行业HHI集中度 (0-1)

        Args:
            date: 交易日期 YYYY-MM-DD
            stock_data_map: 已预加载的 {code: DataFrame}
        """
        if self._industry_daily_stats_cache is not None:
            return

        self._load_sw_industry_mapping()
        if not self._sw_industry_mapping:
            self._industry_daily_stats_cache = {}
            return

        start_time = time.time()
        conn = sqlite3.connect(self.db_path)

        # --- Query 1: breadth + limit_up_ratio from daily_quotes ---
        breadth_query = """
        SELECT sw.l1_name,
               COUNT(*) as total,
               SUM(CASE WHEN q.price_change_pct > 0 THEN 1 ELSE 0 END) as up_count,
               SUM(CASE WHEN q.is_limit_up = 1 THEN 1 ELSE 0 END) as limit_up_count
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        JOIN sw_industry sw ON s.code = sw.code AND sw.is_new = 'Y'
        WHERE s.type = 'A股'
        AND q.trade_date = ?
        GROUP BY sw.l1_name
        """
        breadth_df = pd.read_sql_query(breadth_query, conn, params=(date,))

        # Initialize stats dict
        stats = {}  # l1_name -> {breadth, limit_up_ratio, ...}
        for _, row in breadth_df.iterrows():
            l1 = row['l1_name']
            total = row['total']
            stats[l1] = {
                'breadth': row['up_count'] / total if total > 0 else 0.5,
                'limit_up_ratio': row['limit_up_count'] / total if total > 0 else 0.0,
            }

        # --- Python aggregation: volume_change from stock_data_map ---
        # Group stocks by industry, compute volume vs 5d avg volume
        industry_volumes = {}  # l1_name -> [(today_vol, avg_5d_vol), ...]
        for code, df in stock_data_map.items():
            l1 = self._sw_industry_mapping.get(code)
            if not l1 or df is None or len(df) < 5:
                continue
            vols = df['volume'].values
            today_vol = vols[-1]
            avg_5d = np.mean(vols[-5:])
            if l1 not in industry_volumes:
                industry_volumes[l1] = []
            industry_volumes[l1].append((today_vol, avg_5d))

        for l1, vol_pairs in industry_volumes.items():
            if l1 not in stats:
                stats[l1] = {'breadth': 0.5, 'limit_up_ratio': 0.0}
            total_today = sum(p[0] for p in vol_pairs)
            total_avg = sum(p[1] for p in vol_pairs)
            stats[l1]['volume_change'] = total_today / total_avg if total_avg > 0 else 1.0

        # --- Query 2: kdj_avg + macd_bullish_pct from technical_indicators ---
        tech_query = """
        SELECT sw.l1_name,
               AVG(ti.kdj_j) as kdj_avg,
               COUNT(*) as total,
               SUM(CASE WHEN ti.macd_dif > ti.macd_dea THEN 1 ELSE 0 END) as macd_bull_count
        FROM technical_indicators ti
        JOIN securities s ON ti.security_id = s.id
        JOIN sw_industry sw ON s.code = sw.code AND sw.is_new = 'Y'
        WHERE s.type = 'A股'
        AND ti.trade_date = ?
        AND ti.kdj_j IS NOT NULL
        GROUP BY sw.l1_name
        """
        tech_df = pd.read_sql_query(tech_query, conn, params=(date,))

        for _, row in tech_df.iterrows():
            l1 = row['l1_name']
            if l1 not in stats:
                stats[l1] = {'breadth': 0.5, 'limit_up_ratio': 0.0}
            stats[l1]['kdj_avg'] = row['kdj_avg'] if row['kdj_avg'] is not None else 50.0
            total = row['total']
            stats[l1]['macd_bullish_pct'] = row['macd_bull_count'] / total if total > 0 else 0.5

        # --- Query 3: HHI concentration from daily_basic (circ_mv) ---
        hhi_query = """
        SELECT sw.l1_name, db.circ_mv
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        JOIN sw_industry sw ON s.code = sw.code AND sw.is_new = 'Y'
        WHERE s.type = 'A股'
        AND db.trade_date = ?
        AND db.circ_mv IS NOT NULL AND db.circ_mv > 0
        """
        hhi_df = pd.read_sql_query(hhi_query, conn, params=(date,))

        if not hhi_df.empty:
            for l1, group in hhi_df.groupby('l1_name'):
                if l1 not in stats:
                    stats[l1] = {'breadth': 0.5, 'limit_up_ratio': 0.0}
                mv_values = group['circ_mv'].values
                total_mv = mv_values.sum()
                if total_mv > 0:
                    shares = mv_values / total_mv
                    hhi = float(np.sum(shares ** 2))
                else:
                    hhi = 0.03
                stats[l1]['concentration'] = hhi

        conn.close()

        # Fill defaults for missing keys
        for l1 in stats:
            stats[l1].setdefault('volume_change', 1.0)
            stats[l1].setdefault('kdj_avg', 50.0)
            stats[l1].setdefault('macd_bullish_pct', 0.5)
            stats[l1].setdefault('concentration', 0.03)

        self._industry_daily_stats_cache = stats
        elapsed = time.time() - start_time
        logger.info(f"计算行业日度统计: {len(stats)} 个行业, 耗时 {elapsed:.1f}秒")

    def _batch_load_sw_index_returns(self, date: str):
        """
        从 sw_index_daily 表加载申万行业指数收益率

        Features:
            sw_index_return_1d: 官方行业指数当日涨幅
            sw_index_return_5d: 官方行业指数5日涨幅
        """
        if self._sw_index_return_cache is not None:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sw_index_daily'")
        if not cursor.fetchone():
            self._sw_index_return_cache = {}
            conn.close()
            return

        # Load l1_code -> l1_name mapping
        self._load_sw_industry_mapping()
        cursor.execute("""
            SELECT DISTINCT l1_code, l1_name FROM sw_industry WHERE is_new = 'Y'
        """)
        code_to_name = {row[0]: row[1] for row in cursor.fetchall()}

        # Get 5 recent trading days for 5d return calc
        query = """
        SELECT l1_code, trade_date, close, pct_change
        FROM sw_index_daily
        WHERE trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
        """
        # 31 industries × 5 days = 155 rows max
        df = pd.read_sql_query(query, conn, params=(date, 31 * 6))
        conn.close()

        if df.empty:
            self._sw_index_return_cache = {}
            return

        result = {}  # l1_name -> {return_1d, return_5d}
        for l1_code, group in df.groupby('l1_code'):
            l1_name = code_to_name.get(l1_code)
            if not l1_name:
                continue
            group = group.sort_values('trade_date')
            closes = group['close'].values
            pct = group['pct_change'].values

            ret_1d = pct[-1] / 100.0 if len(pct) >= 1 and pct[-1] is not None else 0.0
            ret_5d = (closes[-1] / closes[0] - 1) if len(closes) >= 2 and closes[0] > 0 else 0.0

            result[l1_name] = {'return_1d': float(ret_1d), 'return_5d': float(ret_5d)}

        self._sw_index_return_cache = result
        logger.info(f"加载申万指数收益: {len(result)} 个行业")

    def _batch_load_northbound_flow(self, date: str):
        """
        从 hsgt_daily 表加载北向资金流入数据

        Feature:
            northbound_flow_5d: 5日北向资金累计净流入(归一化, 单位: 亿元)
        """
        if self._northbound_flow_cache is not None:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hsgt_daily'")
        if not cursor.fetchone():
            self._northbound_flow_cache = {'flow_5d': 0.0}
            conn.close()
            return

        cursor.execute("""
            SELECT north_money FROM hsgt_daily
            WHERE trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT 5
        """, (date,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            self._northbound_flow_cache = {'flow_5d': 0.0}
            return

        # north_money in 万元, convert to 亿元 for readability
        total = sum(float(r[0]) for r in rows if r[0] is not None) / 10000.0
        self._northbound_flow_cache = {'flow_5d': total}
        logger.info(f"加载北向资金: 5日累计 {total:.2f} 亿元")

    def _compute_lightweight_features(self, code: str, stock_df: pd.DataFrame,
                                      trade_date: str = None,
                                      _skip_copy: bool = False) -> Optional[Dict]:
        """
        计算轻量级特征 (不依赖V390系统)

        使用预加载的数据，避免额外的数据库查询

        Args:
            code: 股票代码
            stock_df: 预加载的行情 DataFrame
            trade_date: 目标交易日期 (YYYY-MM-DD)，用于验证数据对齐
            _skip_copy: 跳过 DataFrame 复制 (批量回填时使用，节省内存)
        """
        if stock_df is None or len(stock_df) < 20:
            return None

        # 验证最后一行确实是 trade_date（停牌股最后交易日可能更早）
        if trade_date is not None:
            last_date = str(stock_df.iloc[-1]['trade_date'])[:10]
            if last_date != trade_date:
                logger.debug(f"跳过 {code}: 最后交易日 {last_date} != 目标日 {trade_date} (可能停牌)")
                return None

        try:
            df = stock_df if _skip_copy else stock_df.copy()
            features = {}

            closes = df['close'].values
            volumes = df['volume'].values
            highs = df['high'].values
            lows = df['low'].values

            # 基础技术指标
            # 1. 收益率
            if len(closes) >= 5:
                features['return_5d'] = (closes[-1] / closes[-5] - 1) if closes[-5] > 0 else 0
            if len(closes) >= 10:
                features['return_10d'] = (closes[-1] / closes[-10] - 1) if closes[-10] > 0 else 0
            if len(closes) >= 20:
                features['return_20d'] = (closes[-1] / closes[-20] - 1) if closes[-20] > 0 else 0

            # 2. 波动率
            returns = np.diff(np.log(closes[max(-21, -len(closes)):]))
            if len(returns) >= 10:
                features['volatility_10d'] = np.std(returns[-10:]) * np.sqrt(252)
                features['volatility_20d'] = np.std(returns) * np.sqrt(252) if len(returns) >= 20 else features['volatility_10d']
            else:
                features['volatility_10d'] = 0.3
                features['volatility_20d'] = 0.3

            # 3. 成交量指标
            if len(volumes) >= 20:
                avg_vol_20 = np.mean(volumes[-20:])
                features['volume_ratio'] = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0
                features['volume_trend'] = (np.mean(volumes[-5:]) / np.mean(volumes[-20:]) - 1) if np.mean(volumes[-20:]) > 0 else 0
            else:
                features['volume_ratio'] = 1.0
                features['volume_trend'] = 0.0

            # 4. 价格位置
            if len(closes) >= 20:
                high_20d = np.max(highs[-20:])
                low_20d = np.min(lows[-20:])
                features['price_position_20d'] = (closes[-1] - low_20d) / (high_20d - low_20d) if high_20d > low_20d else 0.5
            else:
                features['price_position_20d'] = 0.5

            # 5. 均线特征
            if len(closes) >= 20:
                ma5 = np.mean(closes[-5:])
                ma10 = np.mean(closes[-10:])
                ma20 = np.mean(closes[-20:])
                features['ma5_ratio'] = closes[-1] / ma5 - 1 if ma5 > 0 else 0
                features['ma10_ratio'] = closes[-1] / ma10 - 1 if ma10 > 0 else 0
                features['ma20_ratio'] = closes[-1] / ma20 - 1 if ma20 > 0 else 0
                features['ma_cross'] = 1 if ma5 > ma10 > ma20 else (-1 if ma5 < ma10 < ma20 else 0)
            else:
                features['ma5_ratio'] = 0
                features['ma10_ratio'] = 0
                features['ma20_ratio'] = 0
                features['ma_cross'] = 0

            # 6. 动量指标
            if len(closes) >= 14:
                # RSI
                delta = np.diff(closes[-15:])
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gain)
                avg_loss = np.mean(loss)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features['rsi_14'] = 100 - (100 / (1 + rs))
                else:
                    features['rsi_14'] = 100 if avg_gain > 0 else 50
            else:
                features['rsi_14'] = 50

            # 7. 涨跌幅 (从收盘价直接计算，不依赖 price_change_pct 列)
            if len(closes) >= 6:  # 需要至少6个价格点才能算5个日涨跌幅
                pct_changes = np.diff(closes) / closes[:-1]  # 日涨跌幅
                features['avg_pct_change_5d'] = float(np.mean(pct_changes[-5:]))
                features['max_pct_change_5d'] = float(np.max(pct_changes[-5:]))
                features['min_pct_change_5d'] = float(np.min(pct_changes[-5:]))
            else:
                features['avg_pct_change_5d'] = 0
                features['max_pct_change_5d'] = 0
                features['min_pct_change_5d'] = 0

            # 8. 行业特征 (申万2021)
            # sw_l1_code: 行业label编码
            l1_name = self._sw_industry_mapping.get(code) if self._sw_industry_mapping else None
            if l1_name and self._sw_l1_label_encoding:
                features['sw_l1_code'] = self._sw_l1_label_encoding.get(l1_name, -1)
            else:
                features['sw_l1_code'] = -1

            # 行业内估值分位数
            val = self._industry_valuation_cache.get(code, {}) if self._industry_valuation_cache else {}
            features['pe_industry_rank'] = val.get('pe_rank', 0.5)
            features['pb_industry_rank'] = val.get('pb_rank', 0.5)
            features['ps_industry_rank'] = val.get('ps_rank', 0.5)

            # 行业收益率和相对强度
            ind_ret = self._industry_return_cache.get(code, {}) if self._industry_return_cache else {}
            features['industry_return_5d'] = ind_ret.get('industry_return_5d', 0.0)
            features['industry_return_20d'] = ind_ret.get('industry_return_20d', 0.0)
            features['industry_relative_strength'] = ind_ret.get('industry_relative_strength', 0.0)

            # 9. 行业日度统计特征 (Tier 1: 6个新特征)
            ind_stats = self._industry_daily_stats_cache.get(l1_name, {}) if (
                l1_name and self._industry_daily_stats_cache) else {}
            features['industry_breadth'] = ind_stats.get('breadth', 0.5)
            features['industry_volume_change'] = ind_stats.get('volume_change', 1.0)
            features['industry_limit_up_ratio'] = ind_stats.get('limit_up_ratio', 0.0)
            features['industry_kdj_avg'] = ind_stats.get('kdj_avg', 50.0)
            features['industry_macd_bullish_pct'] = ind_stats.get('macd_bullish_pct', 0.5)
            features['industry_concentration'] = ind_stats.get('concentration', 0.03)

            # 10. 申万行业指数收益 (Tier 2C)
            sw_ret = self._sw_index_return_cache.get(l1_name, {}) if (
                l1_name and self._sw_index_return_cache) else {}
            features['sw_index_return_1d'] = sw_ret.get('return_1d', 0.0)
            features['sw_index_return_5d'] = sw_ret.get('return_5d', 0.0)

            # 11. 北向资金 (Tier 2C)
            nb = self._northbound_flow_cache if self._northbound_flow_cache else {}
            features['northbound_flow_5d'] = nb.get('flow_5d', 0.0)

            return features

        except Exception as e:
            logger.debug(f"计算 {code} 轻量级特征失败: {e}")
            return None

    def update_single_date(self, date: str, max_workers: int = 4) -> int:
        """
        更新单日的特征缓存 (优化版)

        Args:
            date: 交易日期 YYYY-MM-DD
            max_workers: 并行线程数

        Returns:
            更新的记录数
        """
        start_time = time.time()
        logger.info(f"开始更新 {date} 的V3.9特征缓存 (优化版)...")

        # 1. 获取股票列表
        stocks = self.get_stock_list()
        logger.info(f"股票数量: {len(stocks)}")

        # 2. 计算市场状态特征 (所有股票共享)
        market_features = self.calculate_market_features(date)
        logger.info(f"市场特征计算完成: {len(market_features)} 个")

        # 3. 批量预加载所有股票数据
        stock_data_map = self._batch_load_stock_data(date, lookback=60)

        # 3.5. 批量预加载行业数据 (申万2021)
        self._load_sw_industry_mapping()
        self._batch_load_industry_valuation(date)
        self._batch_load_industry_returns(date, stock_data_map)
        self._batch_load_industry_daily_stats(date, stock_data_map)
        self._batch_load_sw_index_returns(date)
        self._batch_load_northbound_flow(date)

        # 4. 计算每只股票的特征 (使用轻量级方法)
        results = []
        processed = 0
        failed = 0

        for code in stocks:
            try:
                stock_df = stock_data_map.get(code)
                if stock_df is None or len(stock_df) < 20:
                    failed += 1
                    continue

                # 使用轻量级特征计算
                features = self._compute_lightweight_features(code, stock_df, trade_date=date)
                if features:
                    results.append({
                        'code': code,
                        'trade_date': date,
                        'features': features
                    })
                else:
                    failed += 1

                processed += 1

                # 进度日志
                if processed % 1000 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"已处理 {processed}/{len(stocks)} 只股票, 成功 {len(results)}, 失败 {failed}, 耗时 {elapsed:.1f}秒")

            except Exception as e:
                failed += 1
                continue

        logger.info(f"特征计算完成: {len(results)} 只股票有有效特征, 失败 {failed}")

        # 5. 批量计算标签 (跳过，标签会在之后的回测中使用)
        # 每日更新时不需要标签，只需要特征

        # 6. 批量插入数据库
        if results:
            inserted = self._batch_insert(results, market_features)
            elapsed = time.time() - start_time
            logger.info(f"写入 {inserted} 条记录到 v39_feature_cache, 总耗时 {elapsed:.1f}秒")
            return inserted

        return 0

    def _batch_insert(self, results: List[Dict], market_features: Dict) -> int:
        """批量插入特征到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 确保表存在
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS v39_feature_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            trade_date DATE NOT NULL,
            features_json TEXT,
            label_3d REAL,
            label_5d REAL,
            label_10d REAL,
            label_15d REAL,
            market_return_20d REAL,
            market_return_10d REAL,
            market_return_5d REAL,
            market_volatility_20d REAL,
            market_volatility_10d REAL,
            market_up_ratio_20d REAL,
            market_up_ratio_10d REAL,
            market_drawdown_20d REAL,
            market_volume_ratio REAL,
            market_position_20d REAL,
            market_momentum_20d REAL,
            market_momentum_5d REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code, trade_date)
        )
        """)

        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_v39_code_date
        ON v39_feature_cache(code, trade_date)
        """)

        sql = """
        INSERT OR REPLACE INTO v39_feature_cache
        (code, trade_date, features_json,
         label_3d, label_5d, label_10d, label_15d,
         market_return_20d, market_return_10d, market_return_5d,
         market_volatility_20d, market_volatility_10d,
         market_up_ratio_20d, market_up_ratio_10d,
         market_drawdown_20d, market_volume_ratio,
         market_position_20d, market_momentum_20d, market_momentum_5d)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        values = []
        for result in results:
            features_json = json.dumps(result['features'], ensure_ascii=False)
            labels = result.get('labels', {})

            values.append((
                result['code'],
                result['trade_date'],
                features_json,
                labels.get('label_3d'),
                labels.get('label_5d'),
                labels.get('label_10d'),
                labels.get('label_15d'),
                market_features.get('market_return_20d'),
                market_features.get('market_return_10d'),
                market_features.get('market_return_5d'),
                market_features.get('market_volatility_20d'),
                market_features.get('market_volatility_10d'),
                market_features.get('market_up_ratio_20d'),
                market_features.get('market_up_ratio_10d'),
                market_features.get('market_drawdown_20d'),
                market_features.get('market_volume_ratio'),
                market_features.get('market_position_20d'),
                market_features.get('market_momentum_20d'),
                market_features.get('market_momentum_5d')
            ))

        cursor.executemany(sql, values)
        conn.commit()
        conn.close()

        return len(values)

    def backfill_labels(self, batch_size: int = 1000) -> int:
        """
        回填 v39_feature_cache 中 NULL 的 label_3d/5d/10d/15d

        从 daily_quotes 查询未来价格，计算收益率标签。
        只回填距今 >15 个交易日的记录（近期记录没有足够未来数据）。

        Args:
            batch_size: 每批 commit 的记录数

        Returns:
            更新的记录数
        """
        logger.info("=" * 80)
        logger.info("📊 开始回填 v39_feature_cache 标签 (label_3d/5d/10d/15d)...")
        logger.info("=" * 80)
        start_time = time.time()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 找到最新交易日，往前推 16 个交易日作为截止日期（确保有足够未来数据）
        cursor.execute("""
            SELECT DISTINCT trade_date FROM daily_quotes
            ORDER BY trade_date DESC LIMIT 20
        """)
        recent_dates = [row[0] for row in cursor.fetchall()]
        if len(recent_dates) < 16:
            logger.warning("交易日数据不足 16 天，无法回填标签")
            conn.close()
            return 0
        cutoff_date = recent_dates[15]  # 第16个最新交易日
        logger.info(f"截止日期: {cutoff_date} (距今 >=16 个交易日)")

        # 查询所有需要回填的记录 (label_5d IS NULL 且 trade_date <= cutoff)
        cursor.execute("""
            SELECT id, code, trade_date FROM v39_feature_cache
            WHERE label_5d IS NULL AND trade_date <= ?
            ORDER BY trade_date, code
        """, (cutoff_date,))
        rows = cursor.fetchall()
        total = len(rows)
        logger.info(f"需要回填的记录数: {total:,}")

        if total == 0:
            conn.close()
            return 0

        # 预加载所有股票的 security_id 映射
        cursor.execute("SELECT code, id FROM securities WHERE type = 'A股'")
        code_to_sid = {row[0]: row[1] for row in cursor.fetchall()}

        # 预加载停牌日数据: (security_id, trade_date) 的 volume
        # 批量获取太大，改为按需查询

        updated = 0
        skipped_suspended = 0
        skipped_insufficient = 0

        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]

            for cache_id, code, trade_date in batch:
                sid = code_to_sid.get(code)
                if not sid:
                    skipped_insufficient += 1
                    continue

                # 查询 base_date 的 volume（检查停牌）和未来 16 天收盘价
                cursor.execute("""
                    SELECT trade_date, close, volume
                    FROM daily_quotes
                    WHERE security_id = ? AND trade_date >= ?
                    ORDER BY trade_date
                    LIMIT 20
                """, (sid, trade_date))
                future_rows = cursor.fetchall()

                if len(future_rows) < 2:
                    skipped_insufficient += 1
                    continue

                # 第一行应该是 base_date 本身
                base_date_str, base_close, base_volume = future_rows[0]

                # 停牌检测: base_date 成交量为 0
                if base_volume is not None and base_volume == 0:
                    skipped_suspended += 1
                    continue

                if base_close is None or base_close <= 0:
                    skipped_insufficient += 1
                    continue

                # 计算标签
                label_3d = None
                label_5d = None
                label_10d = None
                label_15d = None

                if len(future_rows) > 3:
                    label_3d = future_rows[3][1] / base_close - 1
                if len(future_rows) > 5:
                    label_5d = future_rows[5][1] / base_close - 1
                if len(future_rows) > 10:
                    label_10d = future_rows[10][1] / base_close - 1
                if len(future_rows) > 15:
                    label_15d = future_rows[15][1] / base_close - 1

                # 至少有 label_3d 才更新
                if label_3d is not None:
                    cursor.execute("""
                        UPDATE v39_feature_cache
                        SET label_3d = ?, label_5d = ?, label_10d = ?, label_15d = ?
                        WHERE id = ?
                    """, (label_3d, label_5d, label_10d, label_15d, cache_id))
                    updated += 1

            conn.commit()

            elapsed = time.time() - start_time
            progress = min(i + batch_size, total)
            rate = progress / elapsed if elapsed > 0 else 0
            eta = (total - progress) / rate if rate > 0 else 0
            logger.info(f"进度: {progress:,}/{total:,} ({progress/total*100:.1f}%), "
                        f"已更新: {updated:,}, 停牌跳过: {skipped_suspended:,}, "
                        f"数据不足: {skipped_insufficient:,}, "
                        f"耗时: {elapsed:.0f}秒, ETA: {eta:.0f}秒")

        conn.close()

        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ 标签回填完成!")
        logger.info(f"   总记录: {total:,}")
        logger.info(f"   已更新: {updated:,}")
        logger.info(f"   停牌跳过: {skipped_suspended:,}")
        logger.info(f"   数据不足: {skipped_insufficient:,}")
        logger.info(f"   总耗时: {elapsed:.1f}秒")
        logger.info("=" * 80)

        return updated

    def backfill_pct_change_features(self, batch_size: int = 1000) -> int:
        """
        修复已有缓存中的 pct_change 特征 (avg/max/min_pct_change_5d)

        这些特征之前依赖 price_change_pct 列（可能为 NULL），
        现在改为从 close 价格直接计算。

        Returns:
            更新的记录数
        """
        logger.info("=" * 80)
        logger.info("📊 修复已有缓存中的 pct_change 特征...")
        logger.info("=" * 80)
        start_time = time.time()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 找出 pct_change 特征为 0 或异常的记录（即需要修复的记录）
        cursor.execute("""
            SELECT id, code, trade_date, features_json FROM v39_feature_cache
            ORDER BY trade_date, code
        """)
        rows = cursor.fetchall()
        total = len(rows)
        logger.info(f"总记录数: {total:,}")

        # 预加载 security_id 映射
        cursor.execute("SELECT code, id FROM securities WHERE type = 'A股'")
        code_to_sid = {row[0]: row[1] for row in cursor.fetchall()}

        updated = 0

        for i in range(0, total, batch_size):
            batch = rows[i:i + batch_size]

            for cache_id, code, trade_date, features_json in batch:
                try:
                    features = json.loads(features_json)
                except:
                    continue

                # 检查是否需要修复 (NaN 或 0 表示之前用的是NULL列)
                import math
                avg_val = features.get('avg_pct_change_5d', 0)
                max_val = features.get('max_pct_change_5d', 0)
                needs_fix = (avg_val == 0 and max_val == 0) or \
                            (isinstance(avg_val, float) and math.isnan(avg_val)) or \
                            (isinstance(max_val, float) and math.isnan(max_val))
                if not needs_fix:
                    continue  # 已有有效值，跳过

                sid = code_to_sid.get(code)
                if not sid:
                    continue

                # 查询历史收盘价（需要至少6天来计算5个日涨跌幅）
                cursor.execute("""
                    SELECT close FROM daily_quotes
                    WHERE security_id = ? AND trade_date <= ?
                    ORDER BY trade_date DESC
                    LIMIT 10
                """, (sid, trade_date))
                close_rows = cursor.fetchall()

                if len(close_rows) < 6:
                    continue

                # 反转为时间顺序
                closes = np.array([r[0] for r in reversed(close_rows)])
                pct_changes = np.diff(closes) / closes[:-1]

                features['avg_pct_change_5d'] = float(np.mean(pct_changes[-5:]))
                features['max_pct_change_5d'] = float(np.max(pct_changes[-5:]))
                features['min_pct_change_5d'] = float(np.min(pct_changes[-5:]))

                new_json = json.dumps(features, ensure_ascii=False)
                cursor.execute("""
                    UPDATE v39_feature_cache SET features_json = ? WHERE id = ?
                """, (new_json, cache_id))
                updated += 1

            conn.commit()

            if (i + batch_size) % 10000 == 0 or i + batch_size >= total:
                elapsed = time.time() - start_time
                progress = min(i + batch_size, total)
                logger.info(f"进度: {progress:,}/{total:,}, 修复: {updated:,}, 耗时: {elapsed:.0f}秒")

        conn.close()

        elapsed = time.time() - start_time
        logger.info(f"✅ pct_change 特征修复完成: {updated:,} 条记录, 耗时: {elapsed:.1f}秒")
        return updated

    def update_date_range(self, start_date: str, end_date: str) -> int:
        """更新日期范围内的特征缓存"""
        conn = sqlite3.connect(self.db_path)

        # 获取交易日列表
        query = """
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date >= ?
        AND trade_date <= ?
        ORDER BY trade_date
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        dates = df['trade_date'].tolist()
        logger.info(f"日期范围: {start_date} ~ {end_date}, 共 {len(dates)} 个交易日")

        total_inserted = 0
        for i, date in enumerate(dates, 1):
            logger.info(f"\n[{i}/{len(dates)}] 处理日期: {date}")
            # 清除每日缓存 (行业映射保留，估值/收益需要重算)
            self._batch_market_features = None
            self._industry_valuation_cache = None
            self._industry_return_cache = None
            self._industry_daily_stats_cache = None
            self._sw_index_return_cache = None
            self._northbound_flow_cache = None
            inserted = self.update_single_date(date)
            total_inserted += inserted

        logger.info(f"\n总共更新 {total_inserted} 条记录")
        return total_inserted

    # ================================================================
    # 快速批量回填 (一次性预加载 + 滑动窗口 + 同步标签计算)
    # ================================================================

    def _default_market_features(self) -> Dict[str, float]:
        """返回默认市场特征"""
        return {
            'market_return_20d': 0.0, 'market_return_10d': 0.0, 'market_return_5d': 0.0,
            'market_volatility_20d': 0.0, 'market_volatility_10d': 0.0,
            'market_up_ratio_20d': 0.5, 'market_up_ratio_10d': 0.5,
            'market_drawdown_20d': 0.0, 'market_volume_ratio': 1.0,
            'market_position_20d': 0.5, 'market_momentum_20d': 0.0, 'market_momentum_5d': 0.0
        }

    def _precompute_all_market_features(self, trading_dates: List[str],
                                         market_df: pd.DataFrame,
                                         up_down_df: pd.DataFrame) -> Dict[str, Dict]:
        """批量预计算所有日期的市场特征 (替代逐日 calculate_market_features)"""
        result = {}

        mkt_dates = market_df['trade_date'].values
        mkt_closes = market_df['close'].values.astype(float)
        mkt_volumes = market_df['volume'].values.astype(float)

        ud_dates = up_down_df['trade_date'].values
        ud_up = up_down_df['up_count'].values.astype(float)
        ud_total = up_down_df['total_count'].values.astype(float)

        for date in trading_dates:
            mkt_mask = mkt_dates <= date
            closes = mkt_closes[mkt_mask]
            volumes = mkt_volumes[mkt_mask]

            features = self._default_market_features()

            if len(closes) < 5:
                result[date] = features
                continue

            c = closes[-30:] if len(closes) >= 30 else closes
            v = volumes[-30:] if len(volumes) >= 30 else volumes

            if len(c) >= 20:
                features['market_return_20d'] = float(c[-1] / c[-20] - 1)
            if len(c) >= 10:
                features['market_return_10d'] = float(c[-1] / c[-10] - 1)
            if len(c) >= 5:
                features['market_return_5d'] = float(c[-1] / c[-5] - 1)

            log_returns = np.diff(np.log(c))
            if len(log_returns) >= 20:
                features['market_volatility_20d'] = float(np.std(log_returns[-20:]) * np.sqrt(252))
            if len(log_returns) >= 10:
                features['market_volatility_10d'] = float(np.std(log_returns[-10:]) * np.sqrt(252))

            ud_mask = ud_dates <= date
            up_vals = ud_up[ud_mask]
            total_vals = ud_total[ud_mask]
            if len(up_vals) >= 20:
                ratios = up_vals[-20:] / np.maximum(total_vals[-20:], 1)
                features['market_up_ratio_20d'] = float(np.mean(ratios))
            if len(up_vals) >= 10:
                ratios = up_vals[-10:] / np.maximum(total_vals[-10:], 1)
                features['market_up_ratio_10d'] = float(np.mean(ratios))

            if len(c) >= 20:
                rolling_max = np.maximum.accumulate(c[-20:])
                drawdowns = (c[-20:] - rolling_max) / rolling_max
                features['market_drawdown_20d'] = float(np.min(drawdowns))

            if len(v) >= 20:
                avg_vol = np.mean(v[-20:])
                if avg_vol > 0:
                    features['market_volume_ratio'] = float(v[-1] / avg_vol)

            if len(c) >= 20:
                h, l = float(np.max(c[-20:])), float(np.min(c[-20:]))
                if h > l:
                    features['market_position_20d'] = float((c[-1] - l) / (h - l))

            if len(c) >= 20:
                features['market_momentum_20d'] = float(c[-1] / np.mean(c[-20:]) - 1)
            if len(c) >= 5:
                features['market_momentum_5d'] = float(c[-1] / np.mean(c[-5:]) - 1)

            result[date] = features

        return result

    def _compute_industry_valuation_from_preloaded(self, date: str,
                                                    basic_by_date: Dict[str, pd.DataFrame],
                                                    basic_dates_sorted: List[str]):
        """从预加载的 daily_basic 数据计算行业估值排名 (替代逐日 SQL 查询)"""
        if not self._sw_industry_mapping:
            self._industry_valuation_cache = {}
            return

        pos = bisect.bisect_right(basic_dates_sorted, date)
        if pos == 0:
            self._industry_valuation_cache = {}
            return

        closest_date = basic_dates_sorted[pos - 1]
        df_basic = basic_by_date.get(closest_date)
        if df_basic is None or df_basic.empty:
            self._industry_valuation_cache = {}
            return

        df_basic = df_basic.copy()
        df_basic['l1_name'] = df_basic['code'].map(self._sw_industry_mapping)
        df_basic = df_basic.dropna(subset=['l1_name'])

        result = {}
        for col, rank_name in [('pe_ttm', 'pe_rank'), ('pb', 'pb_rank'), ('ps_ttm', 'ps_rank')]:
            valid = df_basic[df_basic[col].notna() & (df_basic[col] > 0)].copy()
            if valid.empty:
                continue
            valid[rank_name] = valid.groupby('l1_name')[col].rank(pct=True)
            codes = valid['code'].values
            ranks = valid[rank_name].values
            for j in range(len(codes)):
                c = codes[j]
                if c not in result:
                    result[c] = {}
                result[c][rank_name] = float(ranks[j])

        self._industry_valuation_cache = result

    def _compute_labels_from_preloaded(self, stock_df_full: pd.DataFrame,
                                        dates_list: List[str],
                                        date: str) -> Dict[str, Optional[float]]:
        """从预加载数据直接计算标签 (无需 SQL 查询)"""
        labels = {'label_3d': None, 'label_5d': None, 'label_10d': None, 'label_15d': None}

        pos = bisect.bisect_left(dates_list, date)
        if pos >= len(dates_list) or dates_list[pos] != date:
            return labels

        closes = stock_df_full['close'].values
        base_close = closes[pos]
        if base_close <= 0:
            return labels

        # 停牌检测
        volumes = stock_df_full['volume'].values
        if volumes[pos] == 0:
            return labels

        remaining = len(dates_list) - pos
        if remaining > 3:
            labels['label_3d'] = float(closes[pos + 3] / base_close - 1)
        if remaining > 5:
            labels['label_5d'] = float(closes[pos + 5] / base_close - 1)
        if remaining > 10:
            labels['label_10d'] = float(closes[pos + 10] / base_close - 1)
        if remaining > 15:
            labels['label_15d'] = float(closes[pos + 15] / base_close - 1)

        return labels

    def _ensure_cache_table(self, conn):
        """确保 v39_feature_cache 表和索引存在"""
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS v39_feature_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL,
            trade_date DATE NOT NULL,
            features_json TEXT,
            label_3d REAL, label_5d REAL, label_10d REAL, label_15d REAL,
            market_return_20d REAL, market_return_10d REAL, market_return_5d REAL,
            market_volatility_20d REAL, market_volatility_10d REAL,
            market_up_ratio_20d REAL, market_up_ratio_10d REAL,
            market_drawdown_20d REAL, market_volume_ratio REAL,
            market_position_20d REAL, market_momentum_20d REAL, market_momentum_5d REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(code, trade_date)
        )
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_v39_code_date ON v39_feature_cache(code, trade_date)
        """)
        conn.commit()

    def _batch_insert_with_conn(self, conn, results: List[Dict], market_features: Dict) -> int:
        """使用已有连接批量插入 (不关闭连接、不单独 commit)"""
        cursor = conn.cursor()
        sql = """
        INSERT OR REPLACE INTO v39_feature_cache
        (code, trade_date, features_json,
         label_3d, label_5d, label_10d, label_15d,
         market_return_20d, market_return_10d, market_return_5d,
         market_volatility_20d, market_volatility_10d,
         market_up_ratio_20d, market_up_ratio_10d,
         market_drawdown_20d, market_volume_ratio,
         market_position_20d, market_momentum_20d, market_momentum_5d)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        values = []
        for r in results:
            labels = r.get('labels', {})
            values.append((
                r['code'], r['trade_date'],
                json.dumps(r['features'], ensure_ascii=False),
                labels.get('label_3d'), labels.get('label_5d'),
                labels.get('label_10d'), labels.get('label_15d'),
                market_features.get('market_return_20d'),
                market_features.get('market_return_10d'),
                market_features.get('market_return_5d'),
                market_features.get('market_volatility_20d'),
                market_features.get('market_volatility_10d'),
                market_features.get('market_up_ratio_20d'),
                market_features.get('market_up_ratio_10d'),
                market_features.get('market_drawdown_20d'),
                market_features.get('market_volume_ratio'),
                market_features.get('market_position_20d'),
                market_features.get('market_momentum_20d'),
                market_features.get('market_momentum_5d')
            ))
        cursor.executemany(sql, values)
        return len(values)

    def update_date_range_fast(self, start_date: str, end_date: str,
                                num_workers: int = 1) -> int:
        """
        优化版批量回填：一次性预加载数据 + 滑动窗口 + 同步计算标签

        比 update_date_range() 快 5-10 倍:
        - 一次性加载全部行情数据 (避免每天 342K 行的重复 SQL)
        - 批量预算所有日期的市场特征
        - 预加载 daily_basic，按日期索引行业估值
        - 同步计算 labels (利用已加载的未来数据，省去 backfill_labels)
        - 可选多进程并行 (每进程独立加载各自日期段)

        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            num_workers: 并行进程数 (1=顺序, >1=多进程)

        Returns:
            插入/更新的总记录数
        """
        overall_start = time.time()

        # ========== 获取交易日列表 ==========
        conn = sqlite3.connect(self.db_path)
        df_dates = pd.read_sql_query(
            "SELECT DISTINCT trade_date FROM daily_quotes "
            "WHERE trade_date >= ? AND trade_date <= ? ORDER BY trade_date",
            conn, params=(start_date, end_date)
        )
        conn.close()
        trading_dates = df_dates['trade_date'].tolist()

        if not trading_dates:
            logger.warning(f"日期范围 {start_date} ~ {end_date} 无交易日数据")
            return 0

        logger.info("=" * 80)
        logger.info(f"V3.9 特征缓存快速批量回填")
        logger.info(f"   日期范围: {start_date} ~ {end_date} ({len(trading_dates)} 个交易日)")
        logger.info(f"   并行进程: {num_workers}")
        logger.info("=" * 80)

        # ========== 多进程分发 ==========
        if num_workers > 1:
            return self._parallel_backfill(trading_dates, num_workers)

        # ========== Phase 1: 一次性预加载全部数据 ==========
        logger.info("Phase 1: 一次性预加载全部数据...")
        phase1_start = time.time()

        lookback_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=120)
        future_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=30)
        lookback_start = lookback_dt.strftime('%Y-%m-%d')
        future_end = future_dt.strftime('%Y-%m-%d')

        conn = sqlite3.connect(self.db_path)

        # 1a. 全部A股行情
        logger.info(f"  加载行情数据 ({lookback_start} ~ {future_end})...")
        all_quotes = pd.read_sql_query("""
            SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume, q.price_change_pct
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
            ORDER BY s.code, q.trade_date
        """, conn, params=(lookback_start, future_end))
        logger.info(f"  行情: {len(all_quotes):,} 条")

        # 1b. 市场指数数据
        market_df = pd.read_sql_query("""
            SELECT q.trade_date, q.close, q.volume
            FROM daily_quotes q JOIN securities s ON q.security_id = s.id
            WHERE (s.code = '000300.SH' OR s.name = '沪深300')
            AND q.trade_date >= ? AND q.trade_date <= ?
            ORDER BY q.trade_date
        """, conn, params=(lookback_start, future_end))

        # 1c. 涨跌家数
        up_down_df = pd.read_sql_query("""
            SELECT q.trade_date,
                   SUM(CASE WHEN q.price_change_pct > 0 THEN 1 ELSE 0 END) as up_count,
                   COUNT(*) as total_count
            FROM daily_quotes q JOIN securities s ON q.security_id = s.id
            WHERE s.type = 'A股' AND q.trade_date >= ? AND q.trade_date <= ?
            GROUP BY q.trade_date ORDER BY q.trade_date
        """, conn, params=(lookback_start, future_end))

        # 1d. daily_basic (行业估值用)
        logger.info(f"  加载基本面数据...")
        all_basic = pd.read_sql_query("""
            SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm
            FROM daily_basic db JOIN securities s ON db.security_id = s.id
            WHERE s.type = 'A股' AND db.trade_date >= ? AND db.trade_date <= ?
            ORDER BY db.trade_date
        """, conn, params=(lookback_start, end_date))
        logger.info(f"  基本面: {len(all_basic):,} 条")

        conn.close()

        # 按股票代码分组 (一次性 groupby)
        stock_data_all = {}    # code -> DataFrame
        stock_dates_idx = {}   # code -> list[str] (for bisect)
        for code, group in all_quotes.groupby('code'):
            df = group.reset_index(drop=True)
            stock_data_all[code] = df
            stock_dates_idx[code] = df['trade_date'].tolist()

        # daily_basic 按日期分组
        basic_by_date = {}
        for d, g in all_basic.groupby('trade_date'):
            basic_by_date[d] = g
        basic_dates_sorted = sorted(basic_by_date.keys())

        # 释放原始大 DataFrame
        total_rows = len(all_quotes)
        del all_quotes, all_basic

        phase1_time = time.time() - phase1_start
        est_mem_mb = total_rows * 150 / 1e6  # ~150 bytes/row estimate
        logger.info(f"Phase 1 完成: {len(stock_data_all)} 只股票, "
                     f"~{est_mem_mb:.0f}MB, 耗时 {phase1_time:.1f}秒")

        # ========== Phase 2: 批量预计算市场特征 ==========
        logger.info("Phase 2: 批量预计算市场特征...")
        phase2_start = time.time()
        all_market_features = self._precompute_all_market_features(
            trading_dates, market_df, up_down_df)
        del market_df, up_down_df
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2 完成: {len(all_market_features)} 天, 耗时 {phase2_time:.1f}秒")

        # ========== Phase 3: 加载行业映射 ==========
        self._load_sw_industry_mapping()

        # ========== Phase 4: 逐日计算特征 + 标签 ==========
        logger.info("Phase 4: 逐日计算特征 + 标签...")
        phase4_start = time.time()

        # 持久化连接 + SQLite 写入优化
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA busy_timeout=60000")  # 60s wait for concurrent writes
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._ensure_cache_table(conn)

        total_inserted = 0

        for i, date in enumerate(trading_dates):
            # 4a. 滑动窗口切片 (bisect O(log n))
            stock_data_map = {}
            for code, dates_list in stock_dates_idx.items():
                pos = bisect.bisect_right(dates_list, date)
                if pos < 20:
                    continue
                if dates_list[pos - 1] != date:
                    continue  # 停牌
                start_idx = max(0, pos - 60)
                stock_data_map[code] = stock_data_all[code].iloc[start_idx:pos]

            # 4b. 行业估值排名 (从预加载数据)
            self._compute_industry_valuation_from_preloaded(
                date, basic_by_date, basic_dates_sorted)

            # 4c. 行业收益率 + 行业日度统计 + SW指数 + 北向资金
            self._industry_return_cache = None
            self._industry_daily_stats_cache = None
            self._sw_index_return_cache = None
            self._northbound_flow_cache = None
            self._batch_load_industry_returns(date, stock_data_map)
            self._batch_load_industry_daily_stats(date, stock_data_map)
            self._batch_load_sw_index_returns(date)
            self._batch_load_northbound_flow(date)

            # 4d. 市场特征 (预计算)
            market_features = all_market_features.get(date, self._default_market_features())

            # 4e. 计算特征 + 标签
            results = []
            for code, stock_df in stock_data_map.items():
                features = self._compute_lightweight_features(
                    code, stock_df, trade_date=date, _skip_copy=True)
                if features:
                    labels = self._compute_labels_from_preloaded(
                        stock_data_all[code], stock_dates_idx[code], date)
                    results.append({
                        'code': code,
                        'trade_date': date,
                        'features': features,
                        'labels': labels
                    })

            # 4f. 批量插入
            if results:
                inserted = self._batch_insert_with_conn(conn, results, market_features)
                total_inserted += inserted

            # 定期 commit (每10天)
            if (i + 1) % 10 == 0:
                conn.commit()

            # 进度报告 (每20天)
            if (i + 1) % 20 == 0 or (i + 1) == len(trading_dates):
                elapsed = time.time() - phase4_start
                rate = (i + 1) / elapsed
                eta = (len(trading_dates) - i - 1) / rate if rate > 0 else 0
                logger.info(f"  [{i+1}/{len(trading_dates)}] {date}: "
                           f"{len(results)} stocks, 总计 {total_inserted:,}, "
                           f"{rate:.1f} 天/秒, ETA {eta:.0f}秒")

        conn.commit()
        conn.close()

        phase4_time = time.time() - phase4_start
        overall_time = time.time() - overall_start

        logger.info("=" * 80)
        logger.info(f"快速批量回填完成!")
        logger.info(f"   日期范围: {trading_dates[0]} ~ {trading_dates[-1]} "
                     f"({len(trading_dates)} 个交易日)")
        logger.info(f"   总记录: {total_inserted:,}")
        logger.info(f"   Phase 1 (数据预加载): {phase1_time:.1f}秒")
        logger.info(f"   Phase 2 (市场特征): {phase2_time:.1f}秒")
        logger.info(f"   Phase 4 (特征+标签): {phase4_time:.1f}秒 "
                     f"({phase4_time/len(trading_dates):.2f}秒/天)")
        logger.info(f"   总耗时: {overall_time:.1f}秒 ({overall_time/60:.1f}分钟)")
        old_est = len(trading_dates) * 1.8
        logger.info(f"   vs 旧方法预估: {old_est:.0f}秒 ({old_est/60:.1f}分钟), "
                     f"加速比: {old_est/overall_time:.1f}x")
        logger.info("=" * 80)

        return total_inserted

    def _parallel_backfill(self, trading_dates: List[str], num_workers: int) -> int:
        """多进程并行回填 (每进程独立加载数据段，避免序列化瓶颈)"""
        chunk_size = max(1, len(trading_dates) // num_workers)
        chunks = []
        for i in range(0, len(trading_dates), chunk_size):
            chunk = trading_dates[i:i + chunk_size]
            if chunk:
                chunks.append((chunk[0], chunk[-1]))

        logger.info(f"并行回填: {len(chunks)} 个进程")
        for i, (s, e) in enumerate(chunks):
            logger.info(f"  Worker {i}: {s} ~ {e}")

        total_inserted = 0
        with ProcessPoolExecutor(max_workers=min(num_workers, len(chunks))) as executor:
            futures = {}
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                f = executor.submit(
                    _worker_backfill_chunk,
                    self.db_path, chunk_start, chunk_end, i
                )
                futures[f] = i

            for f in as_completed(futures):
                worker_id = futures[f]
                try:
                    count = f.result(timeout=7200)
                    total_inserted += count
                    logger.info(f"Worker {worker_id} 完成: {count:,} 条记录")
                except Exception as e:
                    logger.error(f"Worker {worker_id} 失败: {e}")

        logger.info(f"并行回填总计: {total_inserted:,} 条记录")
        return total_inserted


# 保持向后兼容的别名
V39FeatureCacheUpdater = V39FeatureCacheUpdaterOptimized


def _worker_backfill_chunk(db_path, start_date, end_date, worker_id):
    """多进程回填 worker (每个进程独立创建实例、加载数据)"""
    try:
        logger.info(f"Worker {worker_id}: 开始处理 {start_date} ~ {end_date}")
        updater = V39FeatureCacheUpdaterOptimized(db_path)
        count = updater.update_date_range_fast(start_date, end_date, num_workers=1)
        return count
    except Exception as e:
        logger.error(f"Worker {worker_id} 异常: {e}")
        import traceback
        traceback.print_exc()
        return 0


def update_v39_feature_cache(date_str: str) -> int:
    """
    每日更新入口函数 (供quick_daily_update.py调用)

    Args:
        date_str: 日期字符串，格式 YYYYMMDD

    Returns:
        更新的记录数
    """
    # 转换日期格式
    date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    try:
        updater = V39FeatureCacheUpdaterOptimized()
        count = updater.update_single_date(date_dash)
        return count
    except Exception as e:
        logger.error(f"更新V3.9特征缓存失败: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='V3.9/V3.95特征缓存更新 (优化版)')
    parser.add_argument('--date', type=str, help='单日更新日期 (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str, help='批量更新开始日期')
    parser.add_argument('--end-date', type=str, help='批量更新结束日期')
    parser.add_argument('--max-workers', type=int, default=4, help='并行线程数 (单日模式)')
    parser.add_argument('--workers', type=int, default=1,
                        help='多进程并行数 (批量快速模式, 默认1=顺序, >1=多进程)')
    parser.add_argument('--backfill-labels', action='store_true',
                        help='回填已有缓存的 label_3d/5d/10d/15d (从 daily_quotes 计算)')
    parser.add_argument('--fix-pct-change', action='store_true',
                        help='修复已有缓存中的 pct_change 特征 (从 close 价格重算)')
    parser.add_argument('--slow', action='store_true',
                        help='使用旧版逐日模式 (用于调试对比)')

    args = parser.parse_args()

    updater = V39FeatureCacheUpdaterOptimized()

    if args.backfill_labels:
        updater.backfill_labels()
    elif args.fix_pct_change:
        updater.backfill_pct_change_features()
    elif args.date:
        # 单日更新
        updater.update_single_date(args.date, max_workers=args.max_workers)
    elif args.start_date and args.end_date:
        if args.slow:
            # 旧版逐日模式
            updater.update_date_range(args.start_date, args.end_date)
        else:
            # 快速批量模式 (默认)
            updater.update_date_range_fast(
                args.start_date, args.end_date, num_workers=args.workers)
    else:
        # 默认更新今天
        today = datetime.now().strftime('%Y-%m-%d')
        updater.update_single_date(today, max_workers=args.max_workers)


if __name__ == '__main__':
    main()
