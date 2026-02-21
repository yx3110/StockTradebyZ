#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9特征快速补全脚本 - 优化版

优化策略:
1. 预加载所有行情数据到内存
2. 按股票批量计算（减少数据库访问）
3. 简化特征计算（只保留核心特征）
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from multiprocessing import Pool, cpu_count
import json
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')

# 全局数据缓存
_global_data = {}

def load_all_data(start_date, end_date):
    """预加载所有需要的数据"""
    logger.info("预加载行情数据到内存...")

    conn = sqlite3.connect(DB_PATH)

    # 加载所有行情数据
    quotes = pd.read_sql(f"""
        SELECT
            s.code,
            q.trade_date,
            q.open, q.high, q.low, q.close, q.volume,
            q.price_change_pct
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE q.trade_date BETWEEN '{start_date}' AND '{end_date}'
        AND s.type = 'A股'
        ORDER BY s.code, q.trade_date
    """, conn)

    logger.info(f"  行情数据: {len(quotes):,} 条")

    # 加载基本面数据
    basic = pd.read_sql(f"""
        SELECT
            s.code,
            d.trade_date,
            d.pe_ttm, d.pb, d.ps_ttm,
            d.total_mv, d.circ_mv,
            d.turnover_rate
        FROM daily_basic d
        JOIN securities s ON d.security_id = s.id
        WHERE d.trade_date BETWEEN '{start_date}' AND '{end_date}'
    """, conn)

    logger.info(f"  基本面数据: {len(basic):,} 条")

    conn.close()

    return quotes, basic

def calculate_features_batch(args):
    """批量计算一只股票的所有特征"""
    code, quotes_df, basic_df, trade_dates, lookahead = args

    results = []

    # 获取该股票的数据
    stock_quotes = quotes_df[quotes_df['code'] == code].copy()
    stock_basic = basic_df[basic_df['code'] == code].copy()

    if len(stock_quotes) < 60:  # 需要足够的历史数据
        return results

    stock_quotes = stock_quotes.sort_values('trade_date')
    stock_quotes.set_index('trade_date', inplace=True)

    if not stock_basic.empty:
        stock_basic = stock_basic.sort_values('trade_date')
        stock_basic.set_index('trade_date', inplace=True)

    for date in trade_dates:
        if date not in stock_quotes.index:
            continue

        # 获取历史数据
        hist_mask = stock_quotes.index <= date
        hist = stock_quotes[hist_mask].tail(60)

        if len(hist) < 30:
            continue

        try:
            features = {}
            close = hist['close'].values
            high = hist['high'].values
            low = hist['low'].values
            volume = hist['volume'].values

            # === 技术特征 ===
            # 动量
            features['momentum_5d'] = (close[-1] / close[-5] - 1) if len(close) >= 5 else 0
            features['momentum_10d'] = (close[-1] / close[-10] - 1) if len(close) >= 10 else 0
            features['momentum_20d'] = (close[-1] / close[-20] - 1) if len(close) >= 20 else 0

            # 均线
            ma5 = np.mean(close[-5:])
            ma10 = np.mean(close[-10:])
            ma20 = np.mean(close[-20:])
            features['price_ma5_ratio'] = close[-1] / ma5 - 1 if ma5 > 0 else 0
            features['price_ma10_ratio'] = close[-1] / ma10 - 1 if ma10 > 0 else 0
            features['price_ma20_ratio'] = close[-1] / ma20 - 1 if ma20 > 0 else 0

            # 波动率
            returns = np.diff(close) / close[:-1]
            features['volatility_10d'] = np.std(returns[-10:]) if len(returns) >= 10 else 0
            features['volatility_20d'] = np.std(returns[-20:]) if len(returns) >= 20 else 0

            # RSI
            gains = np.maximum(np.diff(close), 0)
            losses = np.maximum(-np.diff(close), 0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
            features['rsi_14'] = 100 - 100/(1 + avg_gain/(avg_loss + 1e-10))

            # 布林带位置
            bb_mid = ma20
            bb_std = np.std(close[-20:])
            features['bb_position'] = (close[-1] - bb_mid) / (2 * bb_std + 1e-10)

            # 成交量特征
            vol_ma5 = np.mean(volume[-5:])
            vol_ma20 = np.mean(volume[-20:])
            features['volume_ratio_5d'] = volume[-1] / (vol_ma5 + 1e-10)
            features['volume_ratio_20d'] = volume[-1] / (vol_ma20 + 1e-10)
            features['volume_trend'] = vol_ma5 / (vol_ma20 + 1e-10)

            # ATR
            tr = np.maximum(high[1:] - low[1:],
                          np.abs(high[1:] - close[:-1]),
                          np.abs(low[1:] - close[:-1]))
            features['atr_14'] = np.mean(tr[-14:]) / close[-1] if len(tr) >= 14 else 0

            # === 基本面特征 ===
            if date in stock_basic.index:
                basic_row = stock_basic.loc[date]
                features['pe_ttm'] = basic_row.get('pe_ttm', 0) or 0
                features['pb'] = basic_row.get('pb', 0) or 0
                features['turnover_rate'] = basic_row.get('turnover_rate', 0) or 0
                features['market_cap'] = np.log1p(basic_row.get('total_mv', 0) or 0)
            else:
                features['pe_ttm'] = 0
                features['pb'] = 0
                features['turnover_rate'] = 0
                features['market_cap'] = 0

            # === 计算标签 ===
            future_mask = stock_quotes.index > date
            future_data = stock_quotes[future_mask].head(lookahead)

            if len(future_data) >= lookahead:
                future_return = future_data['close'].iloc[-1] / close[-1] - 1

                results.append({
                    'code': code,
                    'trade_date': date,
                    'features': features,
                    'label_5d': future_return
                })
        except:
            continue

    return results

def run_fast_backfill(start_date, end_date, sample_stocks=None, num_workers=None, lookahead=5):
    """运行快速数据补全"""

    if num_workers is None:
        num_workers = cpu_count()

    logger.info("=" * 70)
    logger.info(f"V3.9 快速数据补全: {start_date} ~ {end_date}")
    logger.info(f"并行进程数: {num_workers}")
    logger.info("=" * 70)

    # 预加载数据
    # 需要额外加载lookahead天的数据用于计算标签
    quotes, basic = load_all_data(start_date, end_date)

    # 获取股票列表
    conn = sqlite3.connect(DB_PATH)
    if sample_stocks:
        stocks = pd.read_sql(f"""
            SELECT code FROM securities
            WHERE type='A股'
            ORDER BY RANDOM()
            LIMIT {sample_stocks}
        """, conn)['code'].tolist()
    else:
        stocks = quotes['code'].unique().tolist()

    # 获取交易日列表
    trade_dates = sorted(quotes['trade_date'].unique())
    trade_dates = trade_dates[:-lookahead]  # 排除最后几天

    # 检查已有数据
    existing = set()
    try:
        existing_df = pd.read_sql(f"""
            SELECT code, trade_date
            FROM v39_feature_cache
            WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        """, conn)
        existing = set(zip(existing_df['code'], existing_df['trade_date']))
    except:
        pass
    conn.close()

    # 过滤已有数据
    stocks_to_process = []
    for code in stocks:
        stock_dates = [d for d in trade_dates if (code, d) not in existing]
        if stock_dates:
            stocks_to_process.append((code, stock_dates))

    logger.info(f"股票数: {len(stocks)}, 需处理: {len(stocks_to_process)}")
    logger.info(f"交易日: {len(trade_dates)}")

    if not stocks_to_process:
        logger.info("无需处理的数据")
        return 0

    # 准备任务
    tasks = [(code, quotes, basic, dates, lookahead) for code, dates in stocks_to_process]

    total_inserted = 0
    start_time = time.time()

    # 并行处理
    logger.info(f"\n开始处理 {len(tasks)} 只股票...")

    with Pool(processes=num_workers) as pool:
        for i, results in enumerate(pool.imap_unordered(calculate_features_batch, tasks), 1):
            if results:
                # 批量写入
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                values = []
                for r in results:
                    features_json = json.dumps(r['features'], ensure_ascii=False)
                    values.append((r['code'], r['trade_date'], features_json, r['label_5d']))

                cursor.executemany("""
                    INSERT OR REPLACE INTO v39_feature_cache
                    (code, trade_date, features_json, label_5d)
                    VALUES (?, ?, ?, ?)
                """, values)
                conn.commit()
                conn.close()

                total_inserted += len(values)

            # 进度报告
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                rate = i / elapsed * 60 if elapsed > 0 else 0
                eta = (len(tasks) - i) / (i / elapsed) / 60 if elapsed > 0 else 0
                pct = i / len(tasks) * 100
                logger.info(f"进度: {pct:.1f}% ({i}/{len(tasks)}股票) | "
                           f"已插入: {total_inserted:,} | "
                           f"速率: {rate:.0f}股票/分钟 | "
                           f"ETA: {eta:.1f}分钟")

    elapsed_total = (time.time() - start_time) / 60
    logger.info("=" * 70)
    logger.info(f"✅ 完成! 插入: {total_inserted:,}, 耗时: {elapsed_total:.1f}分钟")
    logger.info("=" * 70)

    return total_inserted

def main():
    import argparse
    parser = argparse.ArgumentParser(description='V3.9特征快速补全')
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--end-date', default='2024-12-31')
    parser.add_argument('--sample-stocks', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)

    args = parser.parse_args()

    run_fast_backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_stocks=args.sample_stocks,
        num_workers=args.num_workers
    )

if __name__ == '__main__':
    main()
