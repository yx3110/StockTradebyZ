#!/usr/bin/env python3
"""
V3.7高级机器学习系统 - 样本质量优化版
重点：智能采样策略 + 时间加权 + 质量筛选
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

def intelligent_quality_sampling(v37_system, stock_codes, max_samples=35000):
    """
    智能样本质量优化采样

    策略:
    1. 时间加权: 近期样本权重更高
    2. 波动率加权: 高波动期样本更有信息价值
    3. 均匀分布: 确保各股票贡献均衡
    """
    print("🎯 开始智能质量优化采样...")

    all_samples = []
    samples_per_stock = max_samples // len(stock_codes)

    for i, code in enumerate(stock_codes):
        if (i + 1) % 50 == 0:
            print(f"质量采样进度: {i+1}/{len(stock_codes)} ({(i+1)/len(stock_codes)*100:.1f}%)")

        try:
            # 获取股票数据
            stock_data = v37_system._get_stock_data(code, '2022-01-01', '2025-09-15')

            if stock_data is None or len(stock_data) < 120:
                continue

            # 计算每日波动率 (作为样本重要性指标)
            stock_data['volatility'] = stock_data['close'].pct_change().rolling(10).std()

            # 计算时间权重 (近期权重更高)
            total_days = len(stock_data)
            time_weights = np.linspace(0.5, 2.0, total_days)  # 线性增长

            # 可采样的时间点
            valid_indices = range(60, len(stock_data) - 10)

            if len(valid_indices) == 0:
                continue

            # 计算每个时间点的采样权重
            sample_weights = []
            valid_samples = []

            for idx in valid_indices:
                volatility = stock_data.iloc[idx]['volatility']
                time_weight = time_weights[idx]

                # 综合权重 = 波动率权重 × 时间权重
                if pd.notna(volatility) and volatility > 0:
                    weight = volatility * time_weight
                    sample_weights.append(weight)
                    valid_samples.append(idx)

            if len(valid_samples) == 0:
                continue

            # 归一化权重
            sample_weights = np.array(sample_weights)
            sample_weights = sample_weights / np.sum(sample_weights)

            # 基于权重采样 (替代均匀采样)
            num_samples = min(samples_per_stock, len(valid_samples))

            if num_samples > 0:
                selected_indices = np.random.choice(
                    valid_samples,
                    size=num_samples,
                    replace=False,
                    p=sample_weights
                )

                # 提取选中样本的特征
                for idx in selected_indices:
                    try:
                        # 提取特征
                        features_df = v37_system._compute_advanced_features(code, stock_data, idx)
                        if features_df.empty:
                            continue

                        current_features = features_df.iloc[0].to_dict()

                        # 添加基本信息
                        trade_date = stock_data.iloc[idx]['trade_date']
                        current_features['code'] = code
                        current_features['trade_date'] = trade_date

                        # 添加样本质量权重
                        current_features['sample_weight'] = sample_weights[valid_samples.index(idx)]

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
                        all_samples.append(combined_row)

                    except Exception as e:
                        continue

        except Exception as e:
            print(f"处理股票 {code} 失败: {e}")
            continue

    print(f"✅ 质量优化采样完成，总样本数: {len(all_samples)}")
    return pd.DataFrame(all_samples)

def weighted_model_training(v37_system, training_df):
    """
    加权模型训练，考虑样本质量权重
    """
    print("🎯 开始加权模型训练...")

    # 分离特征和目标
    target_cols = [col for col in training_df.columns if col.startswith('target_')]
    feature_cols = [col for col in training_df.columns
                   if col not in target_cols and col not in ['code', 'trade_date', 'sample_weight']]

    print(f"📊 特征数: {len(feature_cols)}, 目标数: {len(target_cols)}")
    print(f"📊 加权训练数据: {len(training_df)}条记录")

    # 准备训练数据
    X = training_df[feature_cols].fillna(0)
    y = training_df[target_cols].fillna(0)
    weights = training_df['sample_weight'].fillna(1.0)

    # 过滤无效数据
    valid_mask = (~X.isna().all(axis=1)) & (~y.isna().all(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    weights = weights[valid_mask]

    print(f"🎯 有效加权训练数据: {len(X)}条记录")

    if len(X) < 1000:
        print("❌ 训练数据不足")
        return False, None

    # 为每个目标训练模型
    for target_col in target_cols:
        print(f"训练目标: {target_col}")
        v37_system.build_three_layer_architecture(target_col)

        # 准备特征分组
        feature_groups = v37_system._group_features_for_experts()

        # 准备加权训练数据（包含权重信息）
        weighted_training_data = pd.concat([X, y[[target_col]], weights.rename('sample_weight')], axis=1)

        # 训练三层ensemble（支持样本权重）
        v37_system.train_three_layer_ensemble(weighted_training_data, feature_groups, target_col)

    return True, X

def main():
    print("🚀 启动V3.7样本质量优化训练...")

    # 初始化系统
    v37_system = V370AdvancedMLSystem()
    # 清空已加载的模型，强制重新训练
    v37_system.base_models = {}
    v37_system.expert_models = {}
    v37_system.meta_learner = {}
    v37_system.scalers = {}
    print("🔄 已清空现有模型，将重新训练...")

    db_manager = DatabaseManager()

    # 获取股票列表（继续使用阶段2的行业均衡结果）
    print("📊 复用阶段2行业均衡股票列表...")
    securities_df = db_manager.get_all_securities('A股')

    # 应用质量筛选
    excluded_patterns = ['ST', r'\*ST', 'N ', 'C ', 'U ', '退']
    for pattern in excluded_patterns:
        securities_df = securities_df[~securities_df['name'].str.contains(pattern, na=False, regex=True)]

    # 批量查询活跃股票
    recent_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
    try:
        active_stocks_data = db_manager.execute_query('''
            SELECT DISTINCT s.code
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.type = 'A股' AND dq.trade_date >= ?
        ''', (recent_date,))

        if active_stocks_data:
            active_codes = [row[0] for row in active_stocks_data]
            securities_df = securities_df[securities_df['code'].isin(active_codes)]
    except Exception as e:
        print(f"⚠️ 活跃股票筛选失败: {e}")

    # 选择800只优质股票（减少数量，提升质量）
    stock_codes = securities_df['code'].head(800).tolist()
    print(f"📈 选择{len(stock_codes)}只优质股票进行质量优化训练")

    # 智能质量采样
    training_df = intelligent_quality_sampling(
        v37_system,
        stock_codes,
        max_samples=35000  # 适度增加到35K
    )

    if training_df.empty:
        print("❌ 没有提取到训练数据")
        return False, None

    # 加权模型训练
    success, X = weighted_model_training(v37_system, training_df)

    if not success:
        print("❌ 模型训练失败")
        return False, None

    # 保存模型
    print("💾 保存质量优化模型...")
    Path('models/v370').mkdir(parents=True, exist_ok=True)

    # 使用特殊命名表示这是质量优化版本
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f'v370_quality_optimized_{timestamp}.pkl'

    # 手动构造保存路径
    model_path = f'models/v370/{model_filename}'

    # 使用系统自带的保存方法
    saved_path = v37_system.save_models()
    if saved_path:
        # 重命名为质量优化版本
        import shutil
        shutil.move(saved_path, model_path)
        print(f"✅ V3.7质量优化模型训练完成: {model_path}")

        # 测试加载
        test_system = V370AdvancedMLSystem()
        if test_system.load_models(model_path):
            print("✅ 质量优化模型加载测试成功")
            return True, model_path
        else:
            print("❌ 质量优化模型加载测试失败")
            return False, None
    else:
        print("❌ 质量优化模型保存失败")
        return False, None

if __name__ == "__main__":
    success, model_path = main()
    if success:
        print(f"\n🎉 V3.7质量优化模型训练成功完成！")
        print(f"📁 模型路径: {model_path}")
        print(f"🔧 优化特性: 智能采样 + 时间加权 + 质量筛选")
    else:
        print("\n💥 V3.7质量优化模型训练失败")
        sys.exit(1)