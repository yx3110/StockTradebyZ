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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
import signal

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
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
            db_path = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
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

    def _compute_lightweight_features(self, code: str, stock_df: pd.DataFrame) -> Optional[Dict]:
        """
        计算轻量级特征 (不依赖V390系统)

        使用预加载的数据，避免额外的数据库查询
        """
        if stock_df is None or len(stock_df) < 20:
            return None

        try:
            df = stock_df.copy()
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

            # 7. 涨跌幅
            if 'price_change_pct' in df.columns:
                pct_changes = df['price_change_pct'].values
                if len(pct_changes) >= 5:
                    features['avg_pct_change_5d'] = np.mean(pct_changes[-5:])
                    features['max_pct_change_5d'] = np.max(pct_changes[-5:])
                    features['min_pct_change_5d'] = np.min(pct_changes[-5:])
                else:
                    features['avg_pct_change_5d'] = 0
                    features['max_pct_change_5d'] = 0
                    features['min_pct_change_5d'] = 0
            else:
                features['avg_pct_change_5d'] = 0
                features['max_pct_change_5d'] = 0
                features['min_pct_change_5d'] = 0

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
                features = self._compute_lightweight_features(code, stock_df)
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
            # 清除缓存
            self._batch_market_features = None
            inserted = self.update_single_date(date)
            total_inserted += inserted

        logger.info(f"\n总共更新 {total_inserted} 条记录")
        return total_inserted


# 保持向后兼容的别名
V39FeatureCacheUpdater = V39FeatureCacheUpdaterOptimized


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
    parser.add_argument('--max-workers', type=int, default=4, help='并行线程数')

    args = parser.parse_args()

    updater = V39FeatureCacheUpdaterOptimized()

    if args.date:
        # 单日更新
        updater.update_single_date(args.date, max_workers=args.max_workers)
    elif args.start_date and args.end_date:
        # 批量更新
        updater.update_date_range(args.start_date, args.end_date)
    else:
        # 默认更新今天
        today = datetime.now().strftime('%Y-%m-%d')
        updater.update_single_date(today, max_workers=args.max_workers)


if __name__ == '__main__':
    main()
