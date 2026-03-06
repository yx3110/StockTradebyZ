#!/usr/bin/env python3
"""
ETF专用v39_feature_cache回填脚本

只处理ETF，跳过已有A股数据的重算。
从已有缓存中复用市场特征，ETF行业特征使用默认值。
"""

import os
import sys
import json
import sqlite3
import logging
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_etf_features(code: str, closes: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray, volumes: np.ndarray) -> Optional[Dict]:
    """计算ETF的轻量级特征（与v39_feature_cache_updater一致）"""
    n = len(closes)
    if n < 20:
        return None

    try:
        features = {}

        # 1. 收益率
        if n >= 5:
            features['return_5d'] = float((closes[-1] / closes[-5] - 1)) if closes[-5] > 0 else 0
        if n >= 10:
            features['return_10d'] = float((closes[-1] / closes[-10] - 1)) if closes[-10] > 0 else 0
        if n >= 20:
            features['return_20d'] = float((closes[-1] / closes[-20] - 1)) if closes[-20] > 0 else 0

        # 2. 波动率
        log_returns = np.diff(np.log(closes[max(-21, -n):]))
        if len(log_returns) >= 10:
            features['volatility_10d'] = float(np.std(log_returns[-10:]) * np.sqrt(252))
            features['volatility_20d'] = float(np.std(log_returns) * np.sqrt(252)) if len(log_returns) >= 20 else features['volatility_10d']
        else:
            features['volatility_10d'] = 0.3
            features['volatility_20d'] = 0.3

        # 3. 成交量指标
        if n >= 20:
            avg_vol_20 = np.mean(volumes[-20:])
            features['volume_ratio'] = float(volumes[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0
            features['volume_trend'] = float(np.mean(volumes[-5:]) / np.mean(volumes[-20:]) - 1) if np.mean(volumes[-20:]) > 0 else 0.0
        else:
            features['volume_ratio'] = 1.0
            features['volume_trend'] = 0.0

        # 4. 价格位置
        if n >= 20:
            high_20d = np.max(highs[-20:])
            low_20d = np.min(lows[-20:])
            features['price_position_20d'] = float((closes[-1] - low_20d) / (high_20d - low_20d)) if high_20d > low_20d else 0.5
        else:
            features['price_position_20d'] = 0.5

        # 5. 均线特征
        if n >= 20:
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            ma20 = np.mean(closes[-20:])
            features['ma5_ratio'] = float(closes[-1] / ma5 - 1) if ma5 > 0 else 0
            features['ma10_ratio'] = float(closes[-1] / ma10 - 1) if ma10 > 0 else 0
            features['ma20_ratio'] = float(closes[-1] / ma20 - 1) if ma20 > 0 else 0
            features['ma_cross'] = 1 if ma5 > ma10 > ma20 else (-1 if ma5 < ma10 < ma20 else 0)
        else:
            features['ma5_ratio'] = 0
            features['ma10_ratio'] = 0
            features['ma20_ratio'] = 0
            features['ma_cross'] = 0

        # 6. RSI
        if n >= 14:
            delta = np.diff(closes[-15:])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain)
            avg_loss = np.mean(loss)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features['rsi_14'] = float(100 - (100 / (1 + rs)))
            else:
                features['rsi_14'] = 100.0 if avg_gain > 0 else 50.0
        else:
            features['rsi_14'] = 50.0

        # 7. 涨跌幅
        if n >= 6:
            pct_changes = np.diff(closes) / closes[:-1]
            features['avg_pct_change_5d'] = float(np.mean(pct_changes[-5:]))
            features['max_pct_change_5d'] = float(np.max(pct_changes[-5:]))
            features['min_pct_change_5d'] = float(np.min(pct_changes[-5:]))
        else:
            features['avg_pct_change_5d'] = 0
            features['max_pct_change_5d'] = 0
            features['min_pct_change_5d'] = 0

        # 8. 行业特征 — ETF使用默认值
        features['sw_l1_code'] = -1
        features['pe_industry_rank'] = 0.5
        features['pb_industry_rank'] = 0.5
        features['ps_industry_rank'] = 0.5
        features['industry_return_5d'] = 0.0
        features['industry_return_20d'] = 0.0
        features['industry_relative_strength'] = 0.0

        # 9. 行业日度统计 — 默认值
        features['industry_breadth'] = 0.5
        features['industry_volume_change'] = 1.0
        features['industry_limit_up_ratio'] = 0.0
        features['industry_kdj_avg'] = 50.0
        features['industry_macd_bullish_pct'] = 0.5
        features['industry_concentration'] = 0.03

        # 10. 申万指数收益 — 默认值
        features['sw_index_return_1d'] = 0.0
        features['sw_index_return_5d'] = 0.0

        # 11. 北向资金 — 默认值 (ETF不单独拆分北向)
        features['northbound_flow_5d'] = 0.0

        return features

    except Exception as e:
        return None


def backfill_etf_cache(start_date: str = '2020-01-02', end_date: str = None):
    """
    回填ETF的v39_feature_cache数据

    策略：
    1. 一次性加载所有ETF的全部行情 (~2M行)
    2. 从已有A股缓存中读取每日市场特征 (避免重算)
    3. 逐日计算ETF特征，批量写入
    """
    total_start = time.time()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. 获取已有缓存的交易日期列表
    cursor.execute('SELECT DISTINCT trade_date FROM v39_feature_cache ORDER BY trade_date')
    all_cache_dates = [r[0] for r in cursor.fetchall()]
    logger.info(f"已有缓存交易日: {len(all_cache_dates)} 天 ({all_cache_dates[0]} ~ {all_cache_dates[-1]})")

    # 过滤日期范围
    if end_date is None:
        end_date = all_cache_dates[-1]
    target_dates = [d for d in all_cache_dates if d >= start_date and d <= end_date]
    logger.info(f"目标回填范围: {start_date} ~ {end_date}, 共 {len(target_dates)} 个交易日")

    # 2. 检查已有ETF缓存（跳过已回填的日期）
    cursor.execute('''
        SELECT DISTINCT trade_date FROM v39_feature_cache fc
        JOIN securities s ON fc.code = s.code
        WHERE s.type = 'ETF_基金'
    ''')
    existing_etf_dates = set(r[0] for r in cursor.fetchall())
    dates_to_fill = [d for d in target_dates if d not in existing_etf_dates]
    logger.info(f"已有ETF缓存: {len(existing_etf_dates)} 天, 需回填: {len(dates_to_fill)} 天")

    if not dates_to_fill:
        logger.info("无需回填!")
        conn.close()
        return

    # 3. 一次性加载所有ETF行情数据
    logger.info("加载所有ETF行情数据...")
    load_start = time.time()
    lookback_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=120)
    lookback_start = lookback_dt.strftime('%Y-%m-%d')

    all_etf_quotes = pd.read_sql_query("""
        SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'ETF_基金'
        AND q.trade_date >= ? AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
    """, conn, params=(lookback_start, end_date))
    logger.info(f"ETF行情: {len(all_etf_quotes):,} 条, 耗时 {time.time()-load_start:.1f}秒")

    # 按ETF代码分组
    etf_data_all = {}
    etf_dates_idx = {}
    for code, group in all_etf_quotes.groupby('code'):
        df = group.reset_index(drop=True)
        etf_data_all[code] = df
        etf_dates_idx[code] = df['trade_date'].tolist()
    del all_etf_quotes
    logger.info(f"ETF数量: {len(etf_data_all)}")

    # 4. 预加载每日市场特征（从已有A股缓存中读取）
    logger.info("加载每日市场特征...")
    market_cols = [
        'market_return_20d', 'market_return_10d', 'market_return_5d',
        'market_volatility_20d', 'market_volatility_10d',
        'market_up_ratio_20d', 'market_up_ratio_10d',
        'market_drawdown_20d', 'market_volume_ratio',
        'market_position_20d', 'market_momentum_20d', 'market_momentum_5d'
    ]

    placeholders = ','.join(['?' for _ in dates_to_fill])
    market_df = pd.read_sql_query(f"""
        SELECT trade_date, {', '.join(market_cols)}
        FROM v39_feature_cache
        WHERE trade_date IN ({placeholders})
        GROUP BY trade_date
    """, conn, params=dates_to_fill)

    market_by_date = {}
    for _, row in market_df.iterrows():
        d = row['trade_date']
        market_by_date[d] = {col: row[col] for col in market_cols}
    del market_df
    logger.info(f"市场特征: {len(market_by_date)} 天")

    conn.close()

    # 5. 逐日计算ETF特征并批量写入
    import bisect

    total_inserted = 0
    batch_values = []
    batch_size = 5000

    insert_sql = """
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

    for idx, date in enumerate(dates_to_fill):
        mf = market_by_date.get(date, {})
        success = 0

        for code, full_df in etf_data_all.items():
            dates_list = etf_dates_idx[code]

            # 找到date在该ETF时间序列中的位置
            pos = bisect.bisect_right(dates_list, date)
            if pos == 0:
                continue
            # 确认最后一行就是target date
            if dates_list[pos - 1] != date:
                continue

            # 截取到date为止的数据 (最多60行lookback)
            start_idx = max(0, pos - 60)
            sub_df = full_df.iloc[start_idx:pos]

            if len(sub_df) < 20:
                continue

            closes = sub_df['close'].values
            highs = sub_df['high'].values
            lows = sub_df['low'].values
            volumes = sub_df['volume'].values

            features = compute_etf_features(code, closes, highs, lows, volumes)
            if features is None:
                continue

            features_json = json.dumps(features, ensure_ascii=False)
            batch_values.append((
                code, date, features_json,
                None, None, None, None,  # labels (回填后再算)
                mf.get('market_return_20d'),
                mf.get('market_return_10d'),
                mf.get('market_return_5d'),
                mf.get('market_volatility_20d'),
                mf.get('market_volatility_10d'),
                mf.get('market_up_ratio_20d'),
                mf.get('market_up_ratio_10d'),
                mf.get('market_drawdown_20d'),
                mf.get('market_volume_ratio'),
                mf.get('market_position_20d'),
                mf.get('market_momentum_20d'),
                mf.get('market_momentum_5d'),
            ))
            success += 1

        # 批量写入
        if len(batch_values) >= batch_size:
            conn = sqlite3.connect(DB_PATH)
            conn.executemany(insert_sql, batch_values)
            conn.commit()
            conn.close()
            total_inserted += len(batch_values)
            batch_values = []

        # 进度日志
        if (idx + 1) % 50 == 0 or idx == len(dates_to_fill) - 1:
            elapsed = time.time() - total_start
            pct = (idx + 1) / len(dates_to_fill) * 100
            rate = (idx + 1) / elapsed
            eta = (len(dates_to_fill) - idx - 1) / rate if rate > 0 else 0
            logger.info(f"进度: {idx+1}/{len(dates_to_fill)} ({pct:.1f}%), "
                        f"已写入: {total_inserted + len(batch_values):,}, "
                        f"本日成功: {success}, "
                        f"耗时: {elapsed:.0f}秒, ETA: {eta:.0f}秒")

    # 写入剩余
    if batch_values:
        conn = sqlite3.connect(DB_PATH)
        conn.executemany(insert_sql, batch_values)
        conn.commit()
        conn.close()
        total_inserted += len(batch_values)

    elapsed = time.time() - total_start
    logger.info("=" * 80)
    logger.info(f"✅ ETF特征缓存回填完成!")
    logger.info(f"   回填天数: {len(dates_to_fill)}")
    logger.info(f"   写入记录: {total_inserted:,}")
    logger.info(f"   总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    logger.info("=" * 80)

    return total_inserted


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='回填ETF的v39_feature_cache')
    parser.add_argument('--start-date', default='2020-01-02', help='开始日期')
    parser.add_argument('--end-date', default=None, help='结束日期 (默认到最新)')
    args = parser.parse_args()

    backfill_etf_cache(start_date=args.start_date, end_date=args.end_date)
