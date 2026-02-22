#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为v39特征缓存的所有数据backfill活跃市值特征

高效处理日期格式不一致问题：
- v39_feature_cache: YYYY-MM-DD 格式
- daily_basic: 混合格式 (YYYYMMDD 和 YYYY-MM-DD)

作者: Claude Code
创建时间: 2025-11-28
"""

import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import logging
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')


def normalize_date(date_str):
    """统一日期格式为 YYYY-MM-DD"""
    if len(date_str) == 8:  # YYYYMMDD
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str


def get_v39_dates():
    """获取v39缓存的所有日期"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT DISTINCT trade_date FROM v39_feature_cache ORDER BY trade_date", conn)
    conn.close()
    return df['trade_date'].tolist()


def get_existing_active_mv_dates():
    """获取已计算活跃市值特征的日期"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT DISTINCT trade_date FROM active_mv_feature_cache", conn)
    conn.close()
    return set(df['trade_date'].tolist())


def calculate_market_features_for_date(conn, trade_date, lookback_days=60):
    """
    计算单个日期的市场层面活跃市值特征

    Args:
        conn: 数据库连接
        trade_date: 交易日期 (YYYY-MM-DD格式)
        lookback_days: 用于计算滚动统计的回望天数
    """
    # 尝试两种日期格式
    date_variants = [trade_date, trade_date.replace('-', '')]

    # 获取历史数据用于计算滚动统计
    query = f"""
    SELECT
        trade_date,
        SUM(circ_mv) as total_circ_mv,
        SUM(circ_mv * turnover_rate / 100) as total_active_mv,
        AVG(circ_mv * turnover_rate / 100) as avg_active_mv
    FROM daily_basic db
    JOIN securities s ON db.security_id = s.id
    WHERE s.type = 'A股'
        AND db.circ_mv IS NOT NULL
        AND db.turnover_rate IS NOT NULL
        AND (trade_date <= '{trade_date}' OR trade_date <= '{date_variants[1]}')
    GROUP BY trade_date
    ORDER BY trade_date DESC
    LIMIT {lookback_days}
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        return None

    # 统一日期格式
    df['trade_date'] = df['trade_date'].apply(normalize_date)
    df = df.drop_duplicates(subset=['trade_date'])
    df = df.sort_values('trade_date')

    # 找到目标日期的数据
    target_row = df[df['trade_date'] == trade_date]
    if target_row.empty:
        return None

    idx = target_row.index[0]

    # 计算市场层面特征
    current_active_mv = df.loc[idx, 'total_active_mv']
    current_circ_mv = df.loc[idx, 'total_circ_mv']
    avg_active_mv = df.loc[idx, 'avg_active_mv']

    # 活跃度比率
    market_active_mv_ratio = current_active_mv / current_circ_mv if current_circ_mv > 0 else 0

    # Z-score (使用可用的历史数据)
    if len(df) >= 20:
        mean_val = df['total_active_mv'].mean()
        std_val = df['total_active_mv'].std()
        if std_val > 0:
            market_active_mv_zscore = np.clip((current_active_mv - mean_val) / std_val, -3, 3)
        else:
            market_active_mv_zscore = 0
    else:
        market_active_mv_zscore = 0

    # 趋势 (MA5/MA20 - 1)
    if len(df) >= 5:
        ma5 = df['total_active_mv'].tail(5).mean()
        ma20 = df['total_active_mv'].tail(min(20, len(df))).mean()
        if ma20 > 0:
            market_active_mv_trend = np.clip((ma5 / ma20) - 1, -0.5, 0.5)
        else:
            market_active_mv_trend = 0
    else:
        market_active_mv_trend = 0

    return {
        'market_active_mv_ratio': market_active_mv_ratio,
        'market_active_mv_zscore': market_active_mv_zscore,
        'market_active_mv_trend': market_active_mv_trend,
        'avg_active_mv': avg_active_mv
    }


def calculate_stock_features_for_date(conn, trade_date, market_features):
    """
    计算单个日期所有股票的个股层面特征
    """
    if market_features is None:
        return None

    # 尝试两种日期格式
    date_variants = [trade_date, trade_date.replace('-', '')]

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
        AND (db.trade_date = '{trade_date}' OR db.trade_date = '{date_variants[1]}')
    """

    df = pd.read_sql(query, conn)

    if df.empty:
        return None

    df['trade_date'] = trade_date

    # 市值质量分 (sigmoid惩罚小市值)
    df['market_cap_quality_score'] = 1 / (1 + np.exp(
        -(np.log(df['circ_mv_yi'].clip(lower=0.1)) - np.log(50)) / 0.8
    ))

    # 活跃市值排名
    df['stock_active_mv_rank'] = df['stock_active_mv'].rank(pct=True)

    # 相对流动性
    avg_active_mv = market_features['avg_active_mv']
    if avg_active_mv > 0:
        df['stock_relative_liquidity'] = np.tanh(
            df['stock_active_mv'] / avg_active_mv / 2
        )
    else:
        df['stock_relative_liquidity'] = 0.5

    # 添加市场层面特征
    df['market_active_mv_ratio'] = market_features['market_active_mv_ratio']
    df['market_active_mv_zscore'] = market_features['market_active_mv_zscore']
    df['market_active_mv_trend'] = market_features['market_active_mv_trend']

    # 选择需要的列
    result = df[['code', 'trade_date',
                 'market_active_mv_ratio', 'market_active_mv_zscore', 'market_active_mv_trend',
                 'stock_active_mv_rank', 'stock_relative_liquidity', 'market_cap_quality_score']]

    return result


def backfill_active_mv_features():
    """执行backfill"""
    logger.info("=" * 70)
    logger.info("Backfill 活跃市值特征")
    logger.info("=" * 70)

    # 获取需要处理的日期
    v39_dates = get_v39_dates()
    existing_dates = get_existing_active_mv_dates()

    dates_to_process = [d for d in v39_dates if d not in existing_dates]

    logger.info(f"v39缓存日期总数: {len(v39_dates)}")
    logger.info(f"已处理日期数: {len(existing_dates)}")
    logger.info(f"待处理日期数: {len(dates_to_process)}")

    if not dates_to_process:
        logger.info("✅ 所有日期已处理完成")
        return

    conn = sqlite3.connect(DB_PATH)

    total_records = 0
    successful_dates = 0

    for date in tqdm(dates_to_process, desc="Backfill进度"):
        try:
            # 计算市场层面特征
            market_features = calculate_market_features_for_date(conn, date)

            if market_features is None:
                continue

            # 计算个股层面特征
            stock_df = calculate_stock_features_for_date(conn, date, market_features)

            if stock_df is not None and len(stock_df) > 0:
                # 插入数据库
                stock_df.to_sql('active_mv_feature_cache', conn, if_exists='append', index=False)
                total_records += len(stock_df)
                successful_dates += 1

        except Exception as e:
            logger.warning(f"处理 {date} 失败: {e}")
            continue

    conn.commit()
    conn.close()

    logger.info(f"\n✅ Backfill完成!")
    logger.info(f"   成功处理日期: {successful_dates}")
    logger.info(f"   新增记录数: {total_records}")


def verify_coverage():
    """验证覆盖情况"""
    conn = sqlite3.connect(DB_PATH)

    # v39缓存的日期
    v39_dates = set(pd.read_sql(
        "SELECT DISTINCT trade_date FROM v39_feature_cache", conn
    )['trade_date'].tolist())

    # 活跃市值缓存的日期
    active_mv_dates = set(pd.read_sql(
        "SELECT DISTINCT trade_date FROM active_mv_feature_cache", conn
    )['trade_date'].tolist())

    # 重叠
    overlap = v39_dates & active_mv_dates

    logger.info(f"\n=== 覆盖情况 ===")
    logger.info(f"v39缓存日期: {len(v39_dates)}")
    logger.info(f"活跃市值缓存日期: {len(active_mv_dates)}")
    logger.info(f"重叠日期: {len(overlap)}")
    logger.info(f"覆盖率: {len(overlap)/len(v39_dates)*100:.1f}%")

    # 统计重叠数据量
    if overlap:
        df = pd.read_sql(f"""
            SELECT COUNT(*) as cnt
            FROM v39_feature_cache v
            JOIN active_mv_feature_cache a ON v.code = a.code AND v.trade_date = a.trade_date
        """, conn)
        logger.info(f"可合并的数据量: {df['cnt'].iloc[0]}")

    conn.close()


if __name__ == '__main__':
    backfill_active_mv_features()
    verify_coverage()
