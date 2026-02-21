#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9特征批量预计算脚本 - 分层采样版
改进：确保采样覆盖各种类型的股票（市值、板块、行业）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from precompute_v39_features import V39FeaturePrecomputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V39StratifiedPrecomputer(V39FeaturePrecomputer):
    """分层采样版预计算器"""

    def get_stratified_stock_list(self, sample_stocks=1000):
        """
        分层采样股票列表

        策略：
        1. 按板块分层（沪市、深市、创业板、科创板）
        2. 按市值分层（大中小盘）
        3. 排除ST股票
        4. 确保流动性足够

        Returns:
            list: 股票代码列表
        """
        logger.info("="*80)
        logger.info("🎯 分层采样股票...")
        logger.info("="*80)

        conn = sqlite3.connect(self.db_path)

        # 1. 获取所有A股基础信息
        query = """
            SELECT
                s.code,
                s.name,
                CASE
                    WHEN s.code LIKE '60%' THEN '沪市主板'
                    WHEN s.code LIKE '000%' OR s.code LIKE '001%' THEN '深市主板'
                    WHEN s.code LIKE '002%' THEN '中小板'
                    WHEN s.code LIKE '300%' OR s.code LIKE '301%' THEN '创业板'
                    WHEN s.code LIKE '688%' OR s.code LIKE '689%' THEN '科创板'
                    ELSE '其他'
                END as market,
                db.total_mv as market_cap,
                db.turnover_rate_f as turnover
            FROM securities s
            LEFT JOIN (
                SELECT security_id, total_mv, turnover_rate_f
                FROM daily_basic
                WHERE trade_date = (SELECT MAX(trade_date) FROM daily_basic)
            ) db ON s.id = db.security_id
            WHERE s.type = 'A股'
              AND s.name NOT LIKE '%ST%'  -- 排除ST股票
              AND s.name NOT LIKE '%退%'  -- 排除退市股
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"初步筛选: {len(df)} 只股票（已排除ST和退市风险股）")

        # 2. 过滤流动性不足的股票
        df = df.dropna(subset=['market_cap', 'turnover'])
        df = df[df['turnover'] > 0.1]  # 换手率 > 0.1%

        logger.info(f"流动性筛选后: {len(df)} 只股票")

        # 3. 市值分层
        df['cap_class'] = pd.cut(
            df['market_cap'],
            bins=[0, 5000, 10000, 20000, 50000, 100000, np.inf],
            labels=['超小盘(<50亿)', '小盘(50-100亿)', '中小盘(100-200亿)',
                   '中盘(200-500亿)', '大盘(500-1000亿)', '超大盘(>1000亿)']
        )

        # 4. 分层采样
        # 各板块按市场占比采样
        market_weights = {
            '沪市主板': 0.30,
            '创业板': 0.25,
            '中小板': 0.17,
            '科创板': 0.10,
            '深市主板': 0.10,
            '其他': 0.08
        }

        sampled_stocks = []

        for market, weight in market_weights.items():
            market_df = df[df['market'] == market]
            n_samples = int(sample_stocks * weight)

            if len(market_df) == 0:
                logger.warning(f"  ⚠️  {market}: 无可用股票")
                continue

            if len(market_df) < n_samples:
                # 如果该板块股票不够，全部采样
                logger.warning(f"  ⚠️  {market}: 只有{len(market_df)}只股票，少于目标{n_samples}只")
                selected = market_df['code'].tolist()
            else:
                # 在该板块内按市值分层采样
                cap_groups = market_df.groupby('cap_class', observed=True)
                selected = []

                for cap_class, group in cap_groups:
                    # 每个市值级别采样比例相同
                    n_cap_samples = max(1, int(n_samples * len(group) / len(market_df)))
                    sampled = group.sample(n=min(n_cap_samples, len(group)), random_state=42)
                    selected.extend(sampled['code'].tolist())

                # 如果采样不足，随机补充
                if len(selected) < n_samples:
                    remaining = market_df[~market_df['code'].isin(selected)]
                    additional = remaining.sample(
                        n=min(n_samples - len(selected), len(remaining)),
                        random_state=42
                    )
                    selected.extend(additional['code'].tolist())

            sampled_stocks.extend(selected)
            logger.info(f"  ✅ {market:10s}: 采样 {len(selected):4d} 只")

        # 5. 汇总统计
        logger.info("\n" + "="*80)
        logger.info(f"✅ 分层采样完成: 共 {len(sampled_stocks)} 只股票")
        logger.info("="*80)

        # 显示市值分布
        sampled_df = df[df['code'].isin(sampled_stocks)]
        logger.info("\n市值分布:")
        cap_dist = sampled_df.groupby('cap_class', observed=True).size()
        for cap_class, count in cap_dist.items():
            logger.info(f"  {cap_class:20s}: {count:4d} 只 ({count/len(sampled_stocks)*100:.1f}%)")

        # 显示板块分布
        logger.info("\n板块分布:")
        market_dist = sampled_df.groupby('market').size()
        for market, count in market_dist.items():
            logger.info(f"  {market:10s}: {count:4d} 只 ({count/len(sampled_stocks)*100:.1f}%)")

        return sampled_stocks

    def get_stocks_and_dates(self, start_date, end_date, sample_stocks=None):
        """
        获取股票列表和交易日列表（使用分层采样）

        Returns:
            (stock_list, trade_dates): 股票列表和交易日列表
        """
        # 使用分层采样
        if sample_stocks:
            stock_list = self.get_stratified_stock_list(sample_stocks)
        else:
            # 如果不限制数量，获取全部股票
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT code FROM securities
                WHERE type='A股'
                  AND name NOT LIKE '%ST%'
                  AND name NOT LIKE '%退%'
            """)
            stock_list = [row[0] for row in cursor.fetchall()]
            conn.close()

        # 获取交易日列表
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT trade_date
            FROM daily_quotes
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, (start_date, end_date))
        trade_dates = [row[0] for row in cursor.fetchall()]
        conn.close()

        return stock_list, trade_dates


def main():
    parser = argparse.ArgumentParser(description='V3.9特征批量预计算（分层采样版）')
    parser.add_argument('--start-date', type=str, default='2022-01-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2025-11-15', help='结束日期')
    parser.add_argument('--lookback-days', type=int, default=10, help='回望天数')
    parser.add_argument('--lookahead-days', type=int, default=5, help='前瞻天数')
    parser.add_argument('--sample-stocks', type=int, default=1000, help='采样股票数量')
    parser.add_argument('--batch-size', type=int, default=20, help='每批处理的股票数')
    parser.add_argument('--num-workers', type=int, default=8, help='并行进程数')

    args = parser.parse_args()

    precomputer = V39StratifiedPrecomputer()
    precomputer.num_workers = args.num_workers

    precomputer.precompute(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_stocks=args.sample_stocks,
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
