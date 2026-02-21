#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.9模型训练脚本

训练v3.9增强特征机器学习系统
- 42个增强特征
- 三层Ensemble架构
- 历史数据训练

使用方法:
    python3 train_v390.py --start-date 2024-01-01 --end-date 2025-11-03

作者: Claude Code
创建时间: 2025-11-03
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml_models.v39 import V390EnhancedFeatureMLSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'logs/v39_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练V3.9模型')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-11-03',
                       help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--lookback-days', type=int, default=10,
                       help='回望天数')
    parser.add_argument('--lookahead-days', type=int, default=5,
                       help='前瞻天数（收益计算）')
    parser.add_argument('--sample-stocks', type=int, default=None,
                       help='样本股票数量（None=全部）')
    parser.add_argument('--output-dir', type=str, default=str(PROJECT_ROOT / 'models' / 'v39'),
                       help='模型输出目录')
    parser.add_argument('--optimize-hyperparams', action='store_true',
                       help='优化超参数')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("🚀 V3.9模型训练开始")
    logger.info("=" * 80)
    logger.info(f"训练时间范围: {args.start_date} ~ {args.end_date}")
    logger.info(f"回望天数: {args.lookback_days}")
    logger.info(f"前瞻天数: {args.lookahead_days}")
    logger.info(f"样本股票数: {args.sample_stocks or '全部'}")
    logger.info("=" * 80)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化系统
    logger.info("\n📦 Step 1: 初始化V3.9系统...")
    system = V390EnhancedFeatureMLSystem(
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days
    )

    # 准备训练数据
    logger.info("\n📊 Step 2: 准备训练数据...")
    sample_stocks = None
    if args.sample_stocks:
        # 获取样本股票列表
        import sqlite3
        conn = sqlite3.connect(str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db'))  # 修复：使用正确的数据库路径
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT code FROM securities
            WHERE type='A股'
            ORDER BY RANDOM()
            LIMIT {args.sample_stocks}
        """)
        sample_stocks = [row[0] for row in cursor.fetchall()]
        conn.close()

    X_train, y_train, info_list = system.prepare_training_data(
        start_date=args.start_date,
        end_date=args.end_date,
        sample_stocks=sample_stocks
    )

    if X_train is None:
        logger.error("❌ 训练数据准备失败，退出")
        return 1

    logger.info(f"✅ 训练样本数: {len(X_train)}")
    logger.info(f"✅ 特征数量: {X_train.shape[1]}")

    # 训练模型
    logger.info("\n🤖 Step 3: 训练模型...")
    system.train(
        X_train,
        y_train,
        optimize_hyperparams=args.optimize_hyperparams
    )

    # 保存模型
    logger.info("\n💾 Step 4: 保存模型...")
    model_filename = f"v390_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    model_path = output_dir / model_filename
    system.save_model(str(model_path))

    # 输出特征重要性
    logger.info("\n📊 Step 5: 特征重要性分析...")
    if system.feature_importance:
        logger.info("\nTop 10 重要特征:")
        for model_name, importance_dict in system.feature_importance.items():
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            logger.info(f"\n{model_name}:")
            for feat, imp in sorted_features:
                logger.info(f"  {feat}: {imp:.4f}")

    # 保存训练报告
    logger.info("\n📝 Step 6: 生成训练报告...")
    report_path = output_dir / f"training_report_{datetime.now().strftime('%Y%m%d')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# V3.9模型训练报告\n\n")
        f.write(f"**训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 训练参数\n\n")
        f.write(f"- 时间范围: {args.start_date} ~ {args.end_date}\n")
        f.write(f"- 回望天数: {args.lookback_days}\n")
        f.write(f"- 前瞻天数: {args.lookahead_days}\n")
        f.write(f"- 训练样本数: {len(X_train)}\n")
        f.write(f"- 特征数量: {X_train.shape[1]}\n")
        f.write(f"- 模型版本: v3.9\n\n")
        f.write(f"## 模型文件\n\n")
        f.write(f"- 保存路径: {model_path}\n\n")
        f.write(f"## 特征列表\n\n")
        for i, feat in enumerate(system.feature_names, 1):
            f.write(f"{i}. {feat}\n")

    logger.info(f"✅ 训练报告已保存: {report_path}")

    logger.info("\n" + "=" * 80)
    logger.info("🎉 V3.9模型训练完成!")
    logger.info("=" * 80)
    logger.info(f"📁 模型文件: {model_path}")
    logger.info(f"📄 训练报告: {report_path}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
