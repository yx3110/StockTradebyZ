#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
活跃市值特征预计算脚本

将6个活跃市值特征批量计算并存储到数据库缓存表中：
- 市场层面: market_active_mv_ratio, market_active_mv_zscore, market_active_mv_trend
- 个股层面: stock_active_mv_rank, stock_relative_liquidity, market_cap_quality_score

使用批量SQL计算，高效处理大量数据。

作者: Claude Code
创建时间: 2025-11-28
"""

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import logging
from pathlib import Path
from tqdm import tqdm
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = 'data_adapter/stock_data.db'


def create_cache_table(conn):
    """创建活跃市值特征缓存表"""
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS active_mv_feature_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT NOT NULL,
        trade_date DATE NOT NULL,
        -- 市场层面特征 (3个)
        market_active_mv_ratio REAL,
        market_active_mv_zscore REAL,
        market_active_mv_trend REAL,
        -- 个股层面特征 (3个)
        stock_active_mv_rank REAL,
        stock_relative_liquidity REAL,
        market_cap_quality_score REAL,
        -- 元数据
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(code, trade_date)
    )
    """)

    # 创建索引
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_active_mv_code_date
    ON active_mv_feature_cache(code, trade_date)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_active_mv_date
    ON active_mv_feature_cache(trade_date)
    """)

    conn.commit()
    logger.info("✅ 缓存表创建/检查完成")


def calculate_market_features(conn, start_date, end_date):
    """
    计算市场层面活跃市值特征

    Returns:
        DataFrame with columns: trade_date, market_active_mv_ratio,
                               market_active_mv_zscore, market_active_mv_trend,
                               avg_active_mv (用于个股相对流动性计算)
    """
    logger.info(f"计算市场层面特征: {start_date} ~ {end_date}")

    # 获取更早的数据用于计算滚动统计量
    lookback_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=120)).strftime('%Y-%m-%d')

    query = f"""
    SELECT
        db.trade_date,
        SUM(db.circ_mv) as total_circ_mv,
        SUM(db.circ_mv * db.turnover_rate / 100) as total_active_mv,
        AVG(db.circ_mv * db.turnover_rate / 100) as avg_active_mv,
        COUNT(*) as stock_count
    FROM daily_basic db
    JOIN securities s ON db.security_id = s.id
    WHERE s.type = 'A股'
        AND db.circ_mv IS NOT NULL
        AND db.turnover_rate IS NOT NULL
        AND db.trade_date >= '{lookback_start}'
        AND db.trade_date <= '{end_date}'
    GROUP BY db.trade_date
    ORDER BY db.trade_date
    """

    df = pd.read_sql(query, conn)
    logger.info(f"获取市场数据: {len(df)} 天")

    # 计算活跃度比率
    df['market_active_mv_ratio'] = df['total_active_mv'] / df['total_circ_mv']

    # 计算Z-score (60日滚动)
    df['active_mv_mean_60'] = df['total_active_mv'].rolling(60, min_periods=20).mean()
    df['active_mv_std_60'] = df['total_active_mv'].rolling(60, min_periods=20).std()
    df['market_active_mv_zscore'] = (
        (df['total_active_mv'] - df['active_mv_mean_60']) /
        df['active_mv_std_60'].clip(lower=1e-10)
    ).clip(-3, 3)

    # 计算趋势 (MA5/MA20 - 1)
    df['active_mv_ma5'] = df['total_active_mv'].rolling(5, min_periods=1).mean()
    df['active_mv_ma20'] = df['total_active_mv'].rolling(20, min_periods=5).mean()
    df['market_active_mv_trend'] = (
        (df['active_mv_ma5'] / df['active_mv_ma20']) - 1
    ).clip(-0.5, 0.5)

    # 只保留目标日期范围
    df = df[df['trade_date'] >= start_date]

    # 填充NaN
    df = df.fillna(0)

    result = df[['trade_date', 'market_active_mv_ratio', 'market_active_mv_zscore',
                 'market_active_mv_trend', 'avg_active_mv']]

    logger.info(f"市场特征计算完成: {len(result)} 天")
    return result


def calculate_stock_features_batch(conn, trade_date, market_features):
    """
    批量计算某一天所有股票的个股层面特征

    Args:
        conn: 数据库连接
        trade_date: 交易日期
        market_features: 市场层面特征DataFrame

    Returns:
        DataFrame with all stock features for the day
    """
    # 获取市场数据
    market_row = market_features[market_features['trade_date'] == trade_date]
    if market_row.empty:
        return None

    avg_active_mv = market_row['avg_active_mv'].iloc[0]
    market_active_mv_ratio = market_row['market_active_mv_ratio'].iloc[0]
    market_active_mv_zscore = market_row['market_active_mv_zscore'].iloc[0]
    market_active_mv_trend = market_row['market_active_mv_trend'].iloc[0]

    # 获取当日所有股票数据
    query = f"""
    SELECT
        s.code,
        db.circ_mv / 10000 as circ_mv_yi,
        db.turnover_rate,
        db.circ_mv * db.turnover_rate / 100 / 10000 as stock_active_mv
    FROM daily_basic db
    JOIN securities s ON db.security_id = s.id
    WHERE s.type = 'A股'
        AND db.circ_mv IS NOT NULL
        AND db.turnover_rate IS NOT NULL
        AND db.trade_date = '{trade_date}'
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        return None

    df['trade_date'] = trade_date

    # 1. 市值质量分 (sigmoid惩罚小市值)
    df['market_cap_quality_score'] = 1 / (1 + np.exp(
        -(np.log(df['circ_mv_yi'].clip(lower=0.1)) - np.log(50)) / 0.8
    ))

    # 2. 活跃市值排名 (百分位)
    df['stock_active_mv_rank'] = df['stock_active_mv'].rank(pct=True)

    # 3. 相对流动性
    if avg_active_mv > 0:
        df['stock_relative_liquidity'] = np.tanh(
            df['stock_active_mv'] / avg_active_mv / 2
        )
    else:
        df['stock_relative_liquidity'] = 0.5

    # 4. 添加市场层面特征
    df['market_active_mv_ratio'] = market_active_mv_ratio
    df['market_active_mv_zscore'] = market_active_mv_zscore
    df['market_active_mv_trend'] = market_active_mv_trend

    # 选择需要的列
    result = df[['code', 'trade_date',
                 'market_active_mv_ratio', 'market_active_mv_zscore', 'market_active_mv_trend',
                 'stock_active_mv_rank', 'stock_relative_liquidity', 'market_cap_quality_score']]

    return result


def precompute_features(start_date, end_date, batch_size=10):
    """
    预计算活跃市值特征并存储到数据库

    Args:
        start_date: 开始日期
        end_date: 结束日期
        batch_size: 每批处理的天数
    """
    logger.info("=" * 60)
    logger.info("活跃市值特征预计算")
    logger.info("=" * 60)
    logger.info(f"日期范围: {start_date} ~ {end_date}")

    conn = sqlite3.connect(DB_PATH)

    # 创建缓存表
    create_cache_table(conn)

    # 计算市场层面特征
    market_features = calculate_market_features(conn, start_date, end_date)

    # 获取所有交易日
    trading_dates = market_features['trade_date'].tolist()
    logger.info(f"需要处理: {len(trading_dates)} 个交易日")

    # 检查已缓存的日期
    existing_dates = pd.read_sql(
        f"SELECT DISTINCT trade_date FROM active_mv_feature_cache WHERE trade_date >= '{start_date}'",
        conn
    )['trade_date'].tolist()

    # 过滤掉已处理的日期
    dates_to_process = [d for d in trading_dates if d not in existing_dates]
    logger.info(f"已缓存: {len(existing_dates)} 天, 待处理: {len(dates_to_process)} 天")

    if not dates_to_process:
        logger.info("✅ 所有数据已缓存，无需处理")
        conn.close()
        return

    # 批量处理
    total_records = 0

    for date in tqdm(dates_to_process, desc="预计算特征"):
        try:
            # 计算当日所有股票的特征
            df = calculate_stock_features_batch(conn, date, market_features)

            if df is not None and len(df) > 0:
                # 插入数据库
                df.to_sql('active_mv_feature_cache', conn, if_exists='append', index=False)
                total_records += len(df)

        except Exception as e:
            logger.error(f"处理 {date} 失败: {e}")
            continue

    conn.commit()
    conn.close()

    logger.info(f"\n✅ 预计算完成!")
    logger.info(f"   新增记录: {total_records}")
    logger.info(f"   存储位置: {DB_PATH} -> active_mv_feature_cache")


def get_cached_features(codes, trade_dates, db_path=DB_PATH):
    """
    从缓存获取活跃市值特征

    Args:
        codes: 股票代码列表
        trade_dates: 交易日期列表

    Returns:
        DataFrame with cached features
    """
    conn = sqlite3.connect(db_path)

    # 构建查询
    codes_str = "','".join(codes)
    dates_str = "','".join(trade_dates)

    query = f"""
    SELECT * FROM active_mv_feature_cache
    WHERE code IN ('{codes_str}')
    AND trade_date IN ('{dates_str}')
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def verify_cache(start_date, end_date):
    """验证缓存完整性"""
    conn = sqlite3.connect(DB_PATH)

    # 统计缓存情况
    stats = pd.read_sql(f"""
    SELECT
        trade_date,
        COUNT(*) as stock_count,
        AVG(market_cap_quality_score) as avg_quality_score,
        AVG(stock_active_mv_rank) as avg_rank
    FROM active_mv_feature_cache
    WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY trade_date
    ORDER BY trade_date
    """, conn)

    conn.close()

    logger.info(f"\n缓存验证 ({start_date} ~ {end_date}):")
    logger.info(f"  总天数: {len(stats)}")
    logger.info(f"  日均股票数: {stats['stock_count'].mean():.0f}")
    logger.info(f"  平均质量分: {stats['avg_quality_score'].mean():.4f}")

    return stats


def main():
    parser = argparse.ArgumentParser(description='预计算活跃市值特征')
    parser.add_argument('--start-date', default='2024-01-01', help='开始日期')
    parser.add_argument('--end-date', default=None, help='结束日期（默认今天）')
    parser.add_argument('--verify', action='store_true', help='仅验证缓存')

    args = parser.parse_args()

    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')

    if args.verify:
        verify_cache(args.start_date, args.end_date)
    else:
        precompute_features(args.start_date, args.end_date)
        verify_cache(args.start_date, args.end_date)


if __name__ == '__main__':
    main()
