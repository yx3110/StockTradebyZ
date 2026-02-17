#!/usr/bin/env python3
"""
V3.7高级机器学习系统增强训练脚本 - 阶段2优化
包含：行业均衡采样、数据质量筛选、内存优化
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta
from collections import defaultdict
import gc

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from v370_advanced_ml_system import V370AdvancedMLSystem
from data_adapter.database_manager import DatabaseManager

def intelligent_stock_sampling(db_manager, target_count=1000, industry_balance=True, quality_filter=True):
    """
    智能股票采样策略

    Args:
        db_manager: 数据库管理器
        target_count: 目标股票数量
        industry_balance: 是否进行行业均衡
        quality_filter: 是否进行质量筛选

    Returns:
        list: 筛选后的股票代码列表
    """
    print("🎯 开始智能股票采样...")

    # 获取所有A股信息
    securities_df = db_manager.get_all_securities('A股')

    if quality_filter:
        print("🔍 应用数据质量筛选...")
        # 排除ST、*ST、退市等异常股票
        excluded_patterns = ['ST', r'\*ST', 'N ', 'C ', 'U ', '退']
        for pattern in excluded_patterns:
            securities_df = securities_df[~securities_df['name'].str.contains(pattern, na=False, regex=True)]

        # 简化停牌股票筛选 - 批量查询最近有交易数据的股票
        recent_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')

        try:
            # 批量查询最近有交易的股票
            active_stocks_data = db_manager.execute_query('''
                SELECT DISTINCT s.code
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.type = 'A股' AND dq.trade_date >= ?
            ''', (recent_date,))

            if active_stocks_data:
                active_codes = [row[0] for row in active_stocks_data]
                securities_df = securities_df[securities_df['code'].isin(active_codes)]
                print(f"✅ 质量筛选后剩余: {len(securities_df)}只股票")
            else:
                print("⚠️ 无法获取活跃股票数据，跳过停牌筛选")

        except Exception as e:
            print(f"⚠️ 停牌筛选失败: {e}，跳过此步骤")

    if industry_balance and len(securities_df) > target_count:
        print("⚖️ 应用行业均衡采样...")

        # 获取行业分布
        industry_counts = securities_df['industry'].value_counts()

        # 计算每个行业应分配的股票数
        total_industries = len(industry_counts)
        base_allocation = target_count // total_industries
        remaining = target_count % total_industries

        selected_stocks = []
        industry_allocations = {}

        # 为每个行业分配股票数量
        for i, (industry, count) in enumerate(industry_counts.items()):
            allocation = base_allocation + (1 if i < remaining else 0)
            # 确保不超过该行业实际股票数
            allocation = min(allocation, count)
            industry_allocations[industry] = allocation

        # 从每个行业中随机选择股票
        for industry, allocation in industry_allocations.items():
            industry_stocks = securities_df[securities_df['industry'] == industry]['code'].tolist()
            if len(industry_stocks) > allocation:
                selected = np.random.choice(industry_stocks, allocation, replace=False).tolist()
            else:
                selected = industry_stocks
            selected_stocks.extend(selected)

        print(f"✅ 行业均衡采样完成: {len(selected_stocks)}只股票")
        return selected_stocks
    else:
        # 简单随机采样
        if len(securities_df) > target_count:
            selected = securities_df.sample(n=target_count)['code'].tolist()
        else:
            selected = securities_df['code'].tolist()

        print(f"✅ 随机采样完成: {len(selected)}只股票")
        return selected

def batch_feature_extraction(v37_system, stock_codes, batch_size=50, max_samples=30000):
    """
    批量特征提取，优化内存使用

    Args:
        v37_system: V3.7系统实例
        stock_codes: 股票代码列表
        batch_size: 批次大小
        max_samples: 最大样本数

    Returns:
        pd.DataFrame: 特征和目标数据
    """
    print("🔄 开始批量特征提取...")

    all_features_data = []
    processed_count = 0
    batch_count = 0

    # 分批处理股票
    for i in range(0, len(stock_codes), batch_size):
        batch_codes = stock_codes[i:i+batch_size]
        batch_count += 1

        print(f"处理第{batch_count}批: {i+1}-{min(i+batch_size, len(stock_codes))}/{len(stock_codes)}")

        batch_features = []

        for j, code in enumerate(batch_codes):
            if processed_count >= max_samples:
                break

            try:
                # 获取股票数据 - 扩展时间范围以覆盖更多市场周期
                stock_data = v37_system._get_stock_data(code, '2020-01-01', '2025-09-23')

                if stock_data is None or len(stock_data) < 120:
                    continue

                # 采样策略：不是每个时间点都用，而是间隔采样
                sample_indices = range(60, len(stock_data) - 10, 3)  # 每3天采样一次

                for idx in sample_indices:
                    if processed_count >= max_samples:
                        break

                    try:
                        # 提取特征
                        features_df = v37_system._compute_advanced_features(code, stock_data, idx)
                        if features_df.empty:
                            continue

                        # 转换为字典
                        current_features = features_df.iloc[0].to_dict()

                        # 添加基本信息
                        trade_date = stock_data.iloc[idx]['trade_date']
                        current_features['code'] = code
                        current_features['trade_date'] = trade_date

                        # 计算目标（未来收益）
                        current_price = stock_data.iloc[idx]['close']
                        targets = {}

                        for days in [1, 3, 5]:
                            if idx + days < len(stock_data):
                                future_price = stock_data.iloc[idx + days]['close']
                                return_rate = (future_price / current_price - 1) * 100
                                targets[f'target_{days}d'] = return_rate
                            else:
                                targets[f'target_{days}d'] = 0.0

                        # 合并特征和目标
                        combined_row = {**current_features, **targets}
                        batch_features.append(combined_row)
                        processed_count += 1

                    except Exception as e:
                        continue

            except Exception as e:
                continue

        # 将当前批次数据添加到总数据中
        all_features_data.extend(batch_features)

        # 定期清理内存
        if batch_count % 10 == 0:
            gc.collect()
            print(f"内存清理完成，当前样本数: {processed_count}")

        if processed_count >= max_samples:
            break

    print(f"✅ 特征提取完成，总样本数: {len(all_features_data)}")
    return pd.DataFrame(all_features_data)

def main():
    print("🚀 启动V3.7增强训练 - 阶段2优化...")

    # 初始化系统
    v37_system = V370AdvancedMLSystem()
    # 清空已加载的模型，强制重新训练
    v37_system.base_models = {}
    v37_system.expert_models = {}
    v37_system.meta_learner = {}
    v37_system.scalers = {}
    print("🔄 已清空现有模型，将重新训练...")

    db_manager = DatabaseManager()

    # 智能股票采样 - 减少压力避免数据库锁定
    stock_codes = intelligent_stock_sampling(
        db_manager,
        target_count=800,  # 先用800只股票避免数据库查询压力
        industry_balance=True,
        quality_filter=True
    )

    print(f"📈 准备训练{len(stock_codes)}只股票")

    # 批量特征提取 - 减少内存压力
    training_df = batch_feature_extraction(
        v37_system,
        stock_codes,
        batch_size=30,  # 减少批次大小避免内存问题
        max_samples=35000  # 减少到35K样本避免内存压力
    )

    if training_df.empty:
        print("❌ 没有提取到训练数据")
        return False

    # 分离特征和目标
    target_cols = [col for col in training_df.columns if col.startswith('target_')]
    feature_cols = [col for col in training_df.columns
                   if col not in target_cols and col not in ['code', 'trade_date']]

    print(f"📊 特征数: {len(feature_cols)}, 目标数: {len(target_cols)}")
    print(f"📊 有效训练数据: {len(training_df)}条记录")

    # 准备训练数据
    X = training_df[feature_cols].fillna(0)
    y = training_df[target_cols].fillna(0)

    # 过滤无效数据
    valid_mask = (~X.isna().all(axis=1)) & (~y.isna().all(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"🎯 有效训练数据: {len(X)}条记录")

    if len(X) < 1000:
        print("❌ 训练数据不足")
        return False

    # 训练模型
    print("🤖 开始训练增强V3.7模型...")
    try:
        # 准备训练数据格式
        combined_training_data = pd.concat([X, y], axis=1)

        # 为每个目标训练模型
        for target_col in target_cols:
            print(f"训练目标: {target_col}")
            v37_system.build_three_layer_architecture(target_col)

            # 准备特征分组
            feature_groups = v37_system._group_features_for_experts()

            # 训练三层ensemble
            v37_system.train_three_layer_ensemble(combined_training_data, feature_groups, target_col)

        print("✅ 所有目标模型训练完成")

        # 保存模型
        print("💾 保存增强模型...")
        Path('models/v370').mkdir(parents=True, exist_ok=True)

        # 使用特殊命名表示这是增强版本
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'v370_enhanced_models_{timestamp}.pkl'

        # 手动构造保存路径
        model_path = f'models/v370/{model_filename}'

        # 使用系统自带的保存方法，但指定文件名
        saved_path = v37_system.save_models()
        if saved_path:
            # 重命名为增强版本
            import shutil
            shutil.move(saved_path, model_path)
            print(f"✅ V3.7增强模型训练完成: {model_path}")

            # 测试加载
            test_system = V370AdvancedMLSystem()
            if test_system.load_models(model_path):
                print("✅ 增强模型加载测试成功")
                return True, model_path
            else:
                print("❌ 增强模型加载测试失败")
                return False, None
        else:
            print("❌ 增强模型保存失败")
            return False, None

    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, model_path = main()
    if success:
        print(f"\n🎉 V3.7增强模型训练成功完成！")
        print(f"📁 模型路径: {model_path}")
        print(f"🔧 包含优化: 行业均衡采样 + 数据质量筛选 + 内存优化")
    else:
        print("\n💥 V3.7增强模型训练失败")
        sys.exit(1)