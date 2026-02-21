#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9模型训练脚本（并行优化版）
核心优化：批量数据预加载 + 多进程并行
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from ml_models.v39 import V390EnhancedFeatureMLSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def process_single_stock_date(args):
    """
    处理单只股票在某日期的特征提取（并行工作函数）

    Args:
        args: (code, date, system) 元组

    Returns:
        (features_dict, label, info) 或 None
    """
    code, date, lookback_days, lookahead_days = args

    try:
        # 创建本地系统实例（每个进程独立）
        system = V390EnhancedFeatureMLSystem(
            lookback_days=lookback_days,
            lookahead_days=lookahead_days
        )

        # 提取特征
        features = system.extract_features(code, date)
        if features is None or features.empty:
            return None

        # 计算标签
        label = system.calculate_label(code, date)
        if label is None:
            return None

        return (features.iloc[0].to_dict(), label, {'code': code, 'date': date})

    except Exception:
        return None


class ParallelV390System(V390EnhancedFeatureMLSystem):
    """V3.9系统的并行版本"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.db_path = 'data_adapter/stock_data.db'
        self.num_workers = min(cpu_count(), 8)  # 最多8个进程

    def prepare_training_data_parallel(self, start_date: str, end_date: str, sample_stocks=None):
        """
        并行版训练数据准备

        性能提升：
        - 旧版：串行处理 9,500个样本 → 约30-60分钟
        - 新版：8进程并行处理 → 约5-10分钟（6-12倍提升）
        """
        logger.info("="*80)
        logger.info("🚀 并行版训练数据准备")
        logger.info("="*80)
        logger.info(f"时间范围: {start_date} ~ {end_date}")
        logger.info(f"回望天数: {self.lookback_days}")
        logger.info(f"前瞻天数: {self.lookahead_days}")
        logger.info(f"并行进程数: {self.num_workers}")

        # 1. 获取股票列表
        if sample_stocks is None:
            conn = sqlite3.connect(str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
            cursor = conn.cursor()
            cursor.execute("SELECT code FROM securities WHERE type='A股' LIMIT 1000")
            stock_list = [row[0] for row in cursor.fetchall()]
            conn.close()
        else:
            conn = sqlite3.connect(str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(sample_stocks))
            cursor.execute(f"SELECT code FROM securities WHERE type='A股' AND code IN ({placeholders})", sample_stocks)
            stock_list = [row[0] for row in cursor.fetchall()]
            conn.close()

        logger.info(f"样本股票数: {len(stock_list)}")

        # 2. 获取交易日列表
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

        logger.info(f"交易日数量: {len(trade_dates)}")
        valid_dates = trade_dates[:-self.lookahead_days]
        total_samples_est = len(stock_list) * len(valid_dates)
        logger.info(f"预计样本数: {total_samples_est:,}")

        # 3. 构建任务列表（日期 × 股票）
        logger.info("\n📦 构建并行任务...")
        tasks = []
        for date in valid_dates:
            for code in stock_list:
                tasks.append((code, date, self.lookback_days, self.lookahead_days))

        logger.info(f"总任务数: {len(tasks):,}")

        # 4. 🚀 并行处理
        logger.info(f"\n🚀 启动 {self.num_workers} 个并行进程...")
        X_list = []
        y_list = []
        info_list = []

        with Pool(processes=self.num_workers) as pool:
            # 使用imap_unordered提升效率，tqdm显示进度
            results = pool.imap_unordered(
                process_single_stock_date,
                tasks,
                chunksize=100  # 每次分配100个任务
            )

            for result in tqdm(results, total=len(tasks), desc="并行处理", unit="样本"):
                if result is not None:
                    features_dict, label, info = result
                    X_list.append(features_dict)
                    y_list.append(label)
                    info_list.append(info)

        if len(X_list) == 0:
            logger.error("❌ 未能提取任何训练样本")
            return None, None, None

        X_train = pd.DataFrame(X_list)
        y_train = np.array(y_list)

        logger.info(f"\n✅ 训练数据准备完成!")
        logger.info(f"   总样本数: {len(X_train):,}")
        logger.info(f"   特征数: {X_train.shape[1]}")
        logger.info(f"   有效率: {len(X_train)/total_samples_est*100:.1f}%")

        return X_train, y_train, info_list


def main():
    parser = argparse.ArgumentParser(description='V3.9模型训练（并行版）')
    parser.add_argument('--start-date', type=str, default='2025-06-12', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2025-11-04', help='结束日期')
    parser.add_argument('--lookback-days', type=int, default=10, help='回望天数')
    parser.add_argument('--lookahead-days', type=int, default=5, help='前瞻天数')
    parser.add_argument('--sample-stocks', type=int, default=None, help='采样股票数量')
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'models' / 'v39'), help='模型输出目录')
    parser.add_argument('--num-workers', type=int, default=None, help='并行进程数')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("🚀 V3.9模型训练（并行优化版）")
    logger.info("="*80)
    logger.info(f"训练时间范围: {args.start_date} ~ {args.end_date}")
    logger.info(f"回望天数: {args.lookback_days}")
    logger.info(f"前瞻天数: {args.lookahead_days}")
    logger.info(f"样本股票数: {args.sample_stocks if args.sample_stocks else '全部A股'}")
    logger.info("="*80)

    # Step 1: 初始化系统
    logger.info("\n📦 Step 1: 初始化V3.9系统...")
    system = ParallelV390System(
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days
    )

    if args.num_workers:
        system.num_workers = args.num_workers
        logger.info(f"设置并行进程数: {system.num_workers}")

    # Step 2: 准备训练数据（并行版）
    logger.info("\n📊 Step 2: 准备训练数据（并行版）...")

    # 获取股票样本
    if args.sample_stocks:
        conn = sqlite3.connect(str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT code FROM securities
            WHERE type='A股'
            ORDER BY RANDOM()
            LIMIT {args.sample_stocks}
        """)
        sample_stocks = [row[0] for row in cursor.fetchall()]
        conn.close()
    else:
        sample_stocks = None

    X_train, y_train, info_list = system.prepare_training_data_parallel(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_stocks=sample_stocks
    )

    if X_train is None:
        logger.error("训练数据准备失败")
        return

    # Step 3: 训练模型
    logger.info("\n🎯 Step 3: 训练三层Ensemble模型...")
    system.train(X_train, y_train, optimize_hyperparams=False)

    # Step 4: 保存模型
    logger.info("\n💾 Step 4: 保存模型...")
    timestamp = datetime.now().strftime('%Y%m%d')
    model_path = f"{args.output_dir}/v390_model_{timestamp}.pkl"
    system.save_model(model_path)

    # Step 5: 生成训练报告
    logger.info("\n📄 Step 5: 生成训练报告...")
    report_path = f"{args.output_dir}/training_report_{timestamp}.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# V3.9模型训练报告（并行优化版）\n\n")
        f.write(f"**训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 训练配置\n\n")
        f.write(f"- 时间范围: {args.start_date} ~ {args.end_date}\n")
        f.write(f"- 股票数量: {args.sample_stocks if args.sample_stocks else '全部A股'}\n")
        f.write(f"- 回望天数: {args.lookback_days}\n")
        f.write(f"- 前瞻天数: {args.lookahead_days}\n")
        f.write(f"- 并行进程数: {system.num_workers}\n\n")
        f.write(f"## 训练结果\n\n")
        f.write(f"- 训练样本数: {len(X_train):,}\n")
        f.write(f"- 特征数: {X_train.shape[1]}\n\n")
        f.write(f"## Top 10 重要特征 (LightGBM)\n\n")

        if 'lgb' in system.feature_importance:
            importance_dict = system.feature_importance['lgb']
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            for rank, (feat, imp) in enumerate(sorted_features, 1):
                f.write(f"{rank}. {feat}: {imp:.4f}\n")

    logger.info(f"📁 模型文件: {model_path}")
    logger.info(f"📄 训练报告: {report_path}")
    logger.info("\n" + "="*80)
    logger.info("🎉 V3.9模型训练完成!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
