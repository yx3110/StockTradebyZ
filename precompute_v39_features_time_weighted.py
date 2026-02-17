#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9特征预计算 - 时间加权版本
重点采样近期数据，降低远期数据采样率
"""
import os
import sys
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_models.v39.v390_enhanced_feature_ml_system import V390EnhancedFeatureMLSystem

# 全局变量，用于多进程worker
_worker_system = None

def worker_init():
    """初始化worker进程"""
    global _worker_system
    _worker_system = V390EnhancedFeatureMLSystem()
    print(f"Worker initialized (PID: {os.getpid()})")


def compute_features_batch(args):
    """批量计算特征"""
    codes, date = args
    global _worker_system

    if _worker_system is None:
        worker_init()

    results = []
    for code in codes:
        try:
            # 提取特征
            features = _worker_system.extract_features(code, date)
            if features is None or len(features) == 0:
                continue

            # 计算标签
            label = _worker_system.calculate_label(code, date, lookahead_days=5)

            if label is not None:
                # 转换为字典并保存
                feature_dict = features.iloc[0].to_dict()
                feature_dict['code'] = code
                feature_dict['trade_date'] = date
                feature_dict['label_5d'] = label
                results.append(feature_dict)

        except Exception as e:
            # 静默处理单个股票错误
            pass

    return results


class TimeWeightedPrecompute:
    """时间加权特征预计算器"""

    def __init__(self, db_path='data_adapter/stock_data.db'):
        self.db_path = db_path
        self.system = V390EnhancedFeatureMLSystem()

    def get_stratified_stock_list(self, sample_stocks=1000):
        """
        分层采样股票列表
        按市场和市值分层
        """
        conn = sqlite3.connect(self.db_path)

        # 获取所有A股，排除ST、退市风险等
        query = """
            SELECT DISTINCT s.code, s.name,
                   CASE
                       WHEN s.code LIKE '688%' THEN '科创板'
                       WHEN s.code LIKE '300%' THEN '创业板'
                       WHEN s.code LIKE '002%' OR s.code LIKE '003%' THEN '中小板'
                       WHEN s.code LIKE '000%' THEN '深市主板'
                       WHEN s.code LIKE '60%' THEN '沪市主板'
                       ELSE '其他'
                   END as market,
                   db.total_mv
            FROM securities s
            LEFT JOIN (
                SELECT security_id, total_mv, trade_date
                FROM daily_basic
                WHERE trade_date = (SELECT MAX(trade_date) FROM daily_basic)
            ) db ON s.id = db.security_id
            WHERE s.type = 'A股'
              AND s.name NOT LIKE '%ST%'
              AND s.name NOT LIKE '%退%'
              AND db.total_mv IS NOT NULL
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        # 添加市值分类
        df['cap_class'] = pd.qcut(df['total_mv'], q=3, labels=['小盘', '中盘', '大盘'])

        # 市场权重（根据实际A股分布）
        market_weights = {
            '沪市主板': 0.30,
            '创业板': 0.25,
            '中小板': 0.17,
            '科创板': 0.10,
            '深市主板': 0.10,
            '其他': 0.08
        }

        selected_stocks = []

        for market, weight in market_weights.items():
            market_df = df[df['market'] == market]
            if len(market_df) == 0:
                continue

            # 每个市场内按市值分层
            n_samples = int(sample_stocks * weight)

            # 每个市值类别平均采样
            cap_samples_per_class = n_samples // 3

            for cap_class in ['小盘', '中盘', '大盘']:
                cap_df = market_df[market_df['cap_class'] == cap_class]
                if len(cap_df) == 0:
                    continue

                # 随机采样
                n = min(cap_samples_per_class, len(cap_df))
                sampled = cap_df.sample(n=n, random_state=42)
                selected_stocks.extend(sampled['code'].tolist())

        print(f"\n分层采样完成:")
        print(f"  总股票数: {len(selected_stocks)}")
        for market in df['market'].unique():
            market_count = len([s for s in selected_stocks if s in df[df['market']==market]['code'].values])
            print(f"  {market}: {market_count}")

        return selected_stocks

    def get_time_weighted_dates(self, start_date, end_date):
        """
        时间加权日期选择

        策略:
        - 最近3个月: 100% 采样
        - 3-6个月: 70% 采样 (每3天跳过1天)
        - 6-12个月: 40% 采样 (每5天取2天)
        - 12个月以上: 0% (不使用)
        """
        conn = sqlite3.connect(self.db_path)

        # 获取所有交易日
        query = """
            SELECT DISTINCT trade_date
            FROM daily_quotes
            WHERE trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
        """

        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()

        all_dates = df['trade_date'].tolist()
        today = datetime.now()

        selected_dates = []
        sampling_stats = {
            'recent_3m': 0,
            'medium_6m': 0,
            'older_12m': 0,
            'skipped': 0
        }

        for i, date_str in enumerate(all_dates):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            days_ago = (today - date_obj).days

            # 最近3个月: 100%
            if days_ago <= 90:
                selected_dates.append(date_str)
                sampling_stats['recent_3m'] += 1

            # 3-6个月: 70% (每3天跳过1天)
            elif days_ago <= 180:
                if i % 3 != 2:  # 跳过每第3个
                    selected_dates.append(date_str)
                    sampling_stats['medium_6m'] += 1
                else:
                    sampling_stats['skipped'] += 1

            # 6-12个月: 40% (每5天取2天)
            elif days_ago <= 365:
                if i % 5 < 2:  # 只取每5天的前2天
                    selected_dates.append(date_str)
                    sampling_stats['older_12m'] += 1
                else:
                    sampling_stats['skipped'] += 1

            # 12个月以上: 0%
            else:
                sampling_stats['skipped'] += 1

        print(f"\n时间加权采样统计:")
        print(f"  最近3个月 (100%): {sampling_stats['recent_3m']} 天")
        print(f"  3-6个月 (70%):    {sampling_stats['medium_6m']} 天")
        print(f"  6-12个月 (40%):   {sampling_stats['older_12m']} 天")
        print(f"  跳过的天数:       {sampling_stats['skipped']} 天")
        print(f"  总选择天数:       {len(selected_dates)} 天")

        return selected_dates

    def run_precompute(self, start_date, end_date, sample_stocks=1000,
                      max_workers=None, batch_size=10):
        """
        执行时间加权预计算

        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            sample_stocks: 采样股票数量
            max_workers: 并行进程数
            batch_size: 每批处理的股票数
        """
        print("=" * 80)
        print("V3.9特征预计算 - 时间加权版本")
        print("=" * 80)

        # 1. 分层采样股票
        print("\n[1/5] 分层采样股票...")
        stock_list = self.get_stratified_stock_list(sample_stocks)

        # 2. 时间加权选择日期
        print("\n[2/5] 时间加权选择交易日...")
        date_list = self.get_time_weighted_dates(start_date, end_date)

        # 3. 检查已有缓存
        print("\n[3/5] 检查已有缓存...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM v39_feature_cache")
        existing_count = cursor.fetchone()[0]
        print(f"  已有缓存记录: {existing_count:,}")

        # 4. 生成任务列表
        print("\n[4/5] 生成计算任务...")
        tasks = []

        for date in date_list:
            # 将股票列表分成小批次
            for i in range(0, len(stock_list), batch_size):
                batch_codes = stock_list[i:i+batch_size]
                tasks.append((batch_codes, date))

        total_tasks = len(tasks)
        total_stocks_dates = len(stock_list) * len(date_list)

        print(f"  总任务数: {total_tasks:,}")
        print(f"  预计样本数: {total_stocks_dates:,}")
        print(f"  批次大小: {batch_size}")

        # 5. 并行计算
        print("\n[5/5] 并行计算特征...")

        if max_workers is None:
            max_workers = max(1, cpu_count() - 1)

        print(f"  使用进程数: {max_workers}")

        start_time = datetime.now()
        total_saved = 0

        with Pool(processes=max_workers, initializer=worker_init) as pool:
            with tqdm(total=total_tasks, desc="预计算进度") as pbar:
                for results in pool.imap_unordered(compute_features_batch, tasks):
                    if results:
                        # 批量插入数据库
                        for record in results:
                            try:
                                features_json = json.dumps({
                                    k: v for k, v in record.items()
                                    if k not in ['code', 'trade_date', 'label_5d']
                                })

                                cursor.execute("""
                                    INSERT OR REPLACE INTO v39_feature_cache
                                    (code, trade_date, features_json, label_5d)
                                    VALUES (?, ?, ?, ?)
                                """, (
                                    record['code'],
                                    record['trade_date'],
                                    features_json,
                                    record['label_5d']
                                ))
                                total_saved += 1
                            except Exception as e:
                                pass

                        # 定期提交
                        if total_saved % 1000 == 0:
                            conn.commit()

                    pbar.update(1)
                    pbar.set_postfix({
                        '已保存': f'{total_saved:,}',
                        '速度': f'{total_saved / (datetime.now()-start_time).total_seconds():.1f} samples/s'
                    })

        conn.commit()
        conn.close()

        # 统计结果
        elapsed = (datetime.now() - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("预计算完成!")
        print("=" * 80)
        print(f"  总耗时: {elapsed/60:.1f} 分钟")
        print(f"  保存样本: {total_saved:,}")
        print(f"  计算速度: {total_saved/elapsed:.1f} samples/s")
        print(f"  覆盖率: {total_saved/total_stocks_dates*100:.1f}%")
        print("=" * 80)

        return total_saved


def main():
    parser = argparse.ArgumentParser(description='V3.9时间加权特征预计算')
    parser.add_argument('--start-date', type=str, default='2024-11-01',
                       help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-11-15',
                       help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--sample-stocks', type=int, default=1000,
                       help='采样股票数量')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='并行进程数 (默认: CPU核心数-1)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='每批处理的股票数')

    args = parser.parse_args()

    precompute = TimeWeightedPrecompute()
    precompute.run_precompute(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_stocks=args.sample_stocks,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
