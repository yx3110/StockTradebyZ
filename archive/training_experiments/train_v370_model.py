#!/usr/bin/env python3
"""
V3.7高级机器学习系统训练脚本
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from v370_advanced_ml_system import V370AdvancedMLSystem
from data_adapter.database_manager import DatabaseManager

def main():
    print("🚀 启动V3.7模型训练...")
    
    # 初始化系统（强制重新训练模式）
    v37_system = V370AdvancedMLSystem()
    # 清空已加载的模型，强制重新训练
    v37_system.base_models = {}
    v37_system.expert_models = {}
    v37_system.meta_learner = {}
    v37_system.scalers = {}
    print("🔄 已清空现有模型，将重新训练...")

    db_manager = DatabaseManager()
    
    # 获取股票列表
    print("📊 获取股票列表...")
    securities_df = db_manager.get_all_securities('A股')
    stock_codes = securities_df['code'].head(500).tolist()  # 使用500只股票训练（优化后）
    
    print(f"📈 准备训练{len(stock_codes)}只股票")
    
    # 手动提取特征和目标
    print("🔍 提取特征和目标数据...")
    features_data = []
    targets_data = []
    
    processed_count = 0
    for i, code in enumerate(stock_codes):
        if (i + 1) % 25 == 0:
            print(f"进度: {i+1}/{len(stock_codes)} ({(i+1)/len(stock_codes)*100:.1f}%)")
            
        try:
            # 获取股票数据
            stock_data = v37_system._get_stock_data(code, '2023-01-01', '2025-09-11')
            
            if stock_data is None or len(stock_data) < 120:
                continue
                
            # 为每个时间点提取特征
            for idx in range(60, len(stock_data) - 10):  # 留出足够的历史和未来数据
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
                    features_data.append(combined_row)
                    processed_count += 1
                    
                    # 限制数据量避免内存问题（优化后增加到20K）
                    if processed_count >= 20000:
                        break
                        
                except Exception as e:
                    continue
            
            if processed_count >= 20000:
                break
                
        except Exception as e:
            print(f"处理股票 {code} 失败: {e}")
            continue
    
    if not features_data:
        print("❌ 没有提取到训练数据")
        return False
        
    # 转换为DataFrame
    print(f"🔄 整理训练数据: {len(features_data)}条记录")
    training_df = pd.DataFrame(features_data)
    
    # 分离特征和目标
    target_cols = [col for col in training_df.columns if col.startswith('target_')]
    feature_cols = [col for col in training_df.columns 
                   if col not in target_cols and col not in ['code', 'trade_date']]
    
    print(f"📊 特征数: {len(feature_cols)}, 目标数: {len(target_cols)}")
    
    # 准备训练数据
    X = training_df[feature_cols].fillna(0)
    y = training_df[target_cols].fillna(0)
    
    # 过滤无效数据
    valid_mask = (~X.isna().all(axis=1)) & (~y.isna().all(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"🎯 有效训练数据: {len(X)}条记录")
    
    if len(X) < 100:
        print("❌ 训练数据不足")
        return False
    
    # 训练模型
    print("🤖 开始训练V3.7模型...")
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
        print("💾 保存模型...")
        Path('models/v370').mkdir(parents=True, exist_ok=True)
        
        model_path = v37_system.save_models()
        if model_path:
            print(f"✅ V3.7模型训练完成: {model_path}")
            
            # 测试加载
            test_system = V370AdvancedMLSystem()
            if test_system.load_models(model_path):
                print("✅ 模型加载测试成功")
                return True
            else:
                print("❌ 模型加载测试失败")
                return False
        else:
            print("❌ 模型保存失败")
            return False
            
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 V3.7模型训练成功完成！")
    else:
        print("\n💥 V3.7模型训练失败")
        sys.exit(1)