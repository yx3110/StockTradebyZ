#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3.80参数化训练脚本
支持自定义日期范围和自动日期计算

创建时间: 2025-09-30
作者: Claude Code
版本: 1.0
"""
import sys
import argparse
import os
from datetime import datetime, timedelta
import logging

sys.path.append('/Users/yangxu/StockTradebyZ')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_auto_date_range(months=6):
    """自动计算训练日期范围（最近N个月）"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def load_stock_list():
    """加载训练股票列表"""
    logger.info("📋 加载训练股票列表...")

    # 尝试从文件加载
    stock_file = 'archive/logs/v380_2025_focused_stocks.txt'
    if os.path.exists(stock_file):
        try:
            with open(stock_file, 'r') as f:
                stock_list = [
                    line.strip().split()[0]
                    for line in f
                    if line.strip() and not line.startswith('#')
                ]
            logger.info(f"✅ 从文件加载: {len(stock_list)}只股票")
            return stock_list
        except Exception as e:
            logger.warning(f"从文件加载失败: {e}")

    # 从数据库加载活跃股票
    logger.info("从数据库加载活跃A股...")
    from data_adapter.database_manager import DatabaseManager
    import pandas as pd

    db = DatabaseManager()
    with db.get_connection() as conn:
        query = """
        SELECT DISTINCT s.code
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股'
        AND dq.trade_date >= '2025-01-01'
        GROUP BY s.code
        HAVING COUNT(dq.trade_date) >= 100
        ORDER BY COUNT(dq.trade_date) DESC
        LIMIT 1500
        """
        df = pd.read_sql(query, conn)
        stock_list = df['code'].tolist()

    logger.info(f"✅ 从数据库加载: {len(stock_list)}只活跃A股")
    return stock_list

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(
        description='V3.80模型参数化训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动模式：使用最近6个月数据
  python3 train_v380_parameterized.py --auto

  # 指定日期范围
  python3 train_v380_parameterized.py --start-date 2025-04-01 --end-date 2025-09-30

  # 自动模式：使用最近3个月数据
  python3 train_v380_parameterized.py --auto --months 3
        """
    )
    parser.add_argument('--start-date', type=str, help='训练开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='训练结束日期 (YYYY-MM-DD)')
    parser.add_argument('--months', type=int, default=6, help='自动计算：使用最近N个月数据 (默认: 6)')
    parser.add_argument('--auto', action='store_true', help='自动计算日期范围')
    parser.add_argument('--output-dir', type=str, default='models/v380', help='模型输出目录')

    args = parser.parse_args()

    # 确定日期范围
    if args.auto or (not args.start_date and not args.end_date):
        start_date, end_date = calculate_auto_date_range(args.months)
        logger.info(f"🤖 自动模式：使用最近{args.months}个月数据")
    else:
        if not args.start_date or not args.end_date:
            logger.error("❌ 请同时指定 --start-date 和 --end-date，或使用 --auto")
            return False
        start_date = args.start_date
        end_date = args.end_date

    logger.info("="*80)
    logger.info("🚀 V3.80模型训练开始")
    logger.info("="*80)
    logger.info(f"📅 训练日期范围: {start_date} 到 {end_date}")
    logger.info(f"📁 输出目录: {args.output_dir}")

    try:
        # 导入V3.80系统
        from ml_models.v38 import V380AdvancedIncrementalMLSystem

        system = V380AdvancedIncrementalMLSystem()
        logger.info(f"✅ {system.version} 系统初始化成功")

        # 加载股票列表
        stock_list = load_stock_list()
        if not stock_list:
            logger.error("❌ 无法加载股票列表")
            return False

        logger.info(f"📊 训练股票数量: {len(stock_list)}只")

        # ==================== 步骤1：特征提取 ====================
        logger.info("\n" + "="*80)
        logger.info("🔍 步骤1/4: 特征提取")
        logger.info("="*80)

        features = system.extract_advanced_features(
            codes=stock_list,
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
            target_only=False
        )

        if features is None or len(features) == 0:
            logger.error("❌ 特征提取失败")
            return False

        logger.info(f"✅ 特征提取完成:")
        logger.info(f"  - 样本数量: {len(features)}条")
        logger.info(f"  - 特征维度: {len(features.columns)-2}维")
        logger.info(f"  - 时间跨度: {features['trade_date'].min()} 到 {features['trade_date'].max()}")

        # ==================== 步骤2：准备训练数据 ====================
        logger.info("\n" + "="*80)
        logger.info("🎯 步骤2/4: 准备训练数据")
        logger.info("="*80)

        training_result = system.prepare_training_data(
            features_df=features,
            target_days=[1, 3, 5, 10]
        )

        if isinstance(training_result, tuple):
            training_data, feature_groups = training_result
        else:
            training_data = training_result
            feature_groups = system._group_features_for_experts()

        if training_data is None or len(training_data) == 0:
            logger.error("❌ 训练数据准备失败")
            return False

        logger.info(f"✅ 训练数据准备完成:")
        logger.info(f"  - 有效样本: {len(training_data)}条")
        logger.info(f"  - 目标变量: 1日、3日、5日、10日收益")

        # ==================== 步骤3：模型训练 ====================
        logger.info("\n" + "="*80)
        logger.info("🚀 步骤3/4: 三层Ensemble模型训练")
        logger.info("="*80)

        training_results = {}

        for target_period in [1, 3, 5, 10]:
            target_col = f'target_{target_period}d'

            if target_col not in training_data.columns:
                logger.warning(f"⚠️ 跳过{target_period}日目标：数据不足")
                continue

            logger.info(f"\n📈 训练{target_period}日预测模型...")

            result = system.train_three_layer_ensemble(
                training_data=training_data,
                feature_groups=feature_groups,
                target_col=target_col
            )

            training_results[target_period] = result

            if result.get('success', False):
                logger.info(f"✅ {target_period}日模型训练成功")
                logger.info(f"  - 训练样本: {result.get('training_samples', 0)}条")
                logger.info(f"  - Meta模型性能: {result.get('meta_performance', 0):.4f}")
            else:
                logger.error(f"❌ {target_period}日模型训练失败")

        # ==================== 步骤4：保存模型 ====================
        logger.info("\n" + "="*80)
        logger.info("💾 步骤4/4: 保存模型")
        logger.info("="*80)

        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存模型（只传递后缀名，不传递完整路径）
        model_file = system.save_models("_retrained")
        logger.info(f"✅ 模型已保存: {model_file}")

        # ==================== 训练总结 ====================
        logger.info("\n" + "="*80)
        logger.info("🎉 训练完成！")
        logger.info("="*80)
        logger.info("📊 训练结果总结:")

        success_count = sum(1 for r in training_results.values() if r.get('success'))
        logger.info(f"  - 成功训练模型: {success_count}/{len(training_results)}")

        for period, result in training_results.items():
            if result.get('success'):
                perf = result.get('meta_performance', 0)
                samples = result.get('training_samples', 0)
                logger.info(f"  - {period}日模型: ✅ 性能={perf:.4f}, 样本={samples}条")
            else:
                logger.info(f"  - {period}日模型: ❌ 失败")

        logger.info(f"\n💾 模型文件: {model_file}")
        logger.info(f"📏 模型大小: {os.path.getsize(model_file) / (1024**3):.2f} GB")

        return True

    except Exception as e:
        logger.error(f"❌ 训练过程发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)