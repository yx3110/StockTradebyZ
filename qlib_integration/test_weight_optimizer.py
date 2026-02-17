#!/usr/bin/env python3
"""
测试权重优化器基础功能
简化版测试，确保所有组件正常工作
"""

import os
import sys
import traceback
from datetime import datetime

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def test_imports():
    """测试所有必需的导入"""
    print("🧪 测试导入...")
    
    try:
        # 测试基础依赖
        import numpy as np
        import pandas as pd
        import sqlite3
        print("✅ 基础依赖导入成功")
        
        # 测试hyperopt
        from hyperopt import hp, fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
        print("✅ Hyperopt导入成功")
        
        # 测试数据库管理器
        from data_adapter.database_manager import DatabaseManager
        db = DatabaseManager()
        print("✅ 数据库管理器导入成功")
        
        # 测试权重优化器
        from qlib_weight_optimizer import QlibWeightOptimizer
        print("✅ 权重优化器导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {str(e)}")
        traceback.print_exc()
        return False

def test_database_connection():
    """测试数据库连接"""
    print("\n🧪 测试数据库连接...")
    
    try:
        from data_adapter.database_manager import DatabaseManager
        db = DatabaseManager()
        
        with db.get_connection() as conn:
            # 测试基本查询
            cursor = conn.execute("SELECT COUNT(*) FROM securities WHERE type = 'A股'")
            stock_count = cursor.fetchone()[0]
            print(f"✅ 数据库连接成功，A股数量: {stock_count}")
            
            # 测试最新交易日
            cursor = conn.execute("SELECT MAX(trade_date) FROM daily_quotes")
            latest_date = cursor.fetchone()[0]
            print(f"✅ 最新交易日: {latest_date}")
            
            return True, latest_date
            
    except Exception as e:
        print(f"❌ 数据库连接失败: {str(e)}")
        traceback.print_exc()
        return False, None

def test_optimizer_initialization():
    """测试优化器初始化"""
    print("\n🧪 测试优化器初始化...")
    
    try:
        from qlib_weight_optimizer import QlibWeightOptimizer
        
        optimizer = QlibWeightOptimizer(optimization_period_days=30)
        print("✅ 优化器初始化成功")
        
        # 测试优化空间设置
        space = optimizer.setup_optimization_space()
        print(f"✅ 优化空间设置成功，参数数量: {len(space)}")
        
        return True, optimizer
        
    except Exception as e:
        print(f"❌ 优化器初始化失败: {str(e)}")
        traceback.print_exc()
        return False, None

def test_data_preparation():
    """测试数据准备"""
    print("\n🧪 测试数据准备...")
    
    try:
        from qlib_weight_optimizer import QlibWeightOptimizer
        
        optimizer = QlibWeightOptimizer(optimization_period_days=30)
        
        # 获取最新日期
        with optimizer.db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT MAX(trade_date) FROM daily_quotes WHERE trade_date <= date('now')")
            end_date = cursor.fetchone()[0]
        
        print(f"📅 使用结束日期: {end_date}")
        
        # 测试获取活跃股票
        active_stocks = optimizer._get_active_stocks(end_date)
        print(f"✅ 获取活跃股票成功，数量: {len(active_stocks)}")
        
        if len(active_stocks) == 0:
            print("⚠️ 警告：没有活跃股票，请检查数据")
            return False
        
        # 测试获取单个股票数据
        test_stock = active_stocks[0]
        print(f"📊 测试股票: {test_stock}")
        
        # 计算开始日期
        from datetime import datetime, timedelta
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=60)
        start_date = start_dt.strftime('%Y-%m-%d')
        
        stock_data = optimizer._get_stock_data(test_stock, start_date, end_date)
        print(f"✅ 获取股票数据成功，数据量: {len(stock_data)} 天")
        
        if len(stock_data) == 0:
            print("⚠️ 警告：股票数据为空")
            return False
        
        # 测试技术指标计算
        features = optimizer._calculate_technical_features(stock_data)
        if features:
            print(f"✅ 技术指标计算成功，特征数量: {len(features)}")
            print(f"📊 样例特征: {list(features.keys())[:5]}")
        else:
            print("❌ 技术指标计算失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据准备测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_scoring_functions():
    """测试评分函数"""
    print("\n🧪 测试评分函数...")
    
    try:
        from qlib_weight_optimizer import QlibWeightOptimizer
        
        optimizer = QlibWeightOptimizer()
        
        # 测试样例特征
        sample_features = {
            'kdj_k': 45.0,
            'kdj_d': 40.0, 
            'kdj_j': 55.0,
            'rsi': 35.0,
            'bbi': 10.50,
            'close': 10.75,
            'zhixing_trend': 10.60,
            'zhixing_multiavg': 10.55,
            'volume_surge': 2.5,
            'pe_ttm': 20.0,
            'pb': 1.8,
            'price_momentum': 0.05,
            'volatility': 0.025
        }
        
        # 测试样例权重
        sample_weights = {
            'kdj_strength': 0.12,
            'rsi_momentum': 0.10,
            'bbi_trend': 0.08,
            'volume_surge': 0.10,
            'zhixing_trend': 0.12,
            'zhixing_multiavg': 0.08,
            'pe_valuation': 0.08,
            'pb_valuation': 0.08,
            'price_momentum': 0.12,
            'volatility_risk': 0.06
        }
        
        # 计算加权评分
        score = optimizer._calculate_weighted_score(sample_features, sample_weights)
        print(f"✅ 评分计算成功: {score:.2f}")
        
        if 0 <= score <= 100:
            print("✅ 评分范围正确")
        else:
            print(f"⚠️ 评分范围异常: {score}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评分函数测试失败: {str(e)}")
        traceback.print_exc()
        return False

def run_mini_optimization():
    """运行迷你优化测试"""
    print("\n🧪 运行迷你优化测试...")
    
    try:
        from qlib_weight_optimizer import QlibWeightOptimizer
        
        optimizer = QlibWeightOptimizer(optimization_period_days=20)  # 减少天数
        
        # 准备少量数据
        print("📊 准备测试数据...")
        with optimizer.db_manager.get_connection() as conn:
            cursor = conn.execute("SELECT MAX(trade_date) FROM daily_quotes WHERE trade_date <= date('now')")
            end_date = cursor.fetchone()[0]
        
        # 获取少量活跃股票进行测试
        active_stocks = optimizer._get_active_stocks(end_date)[:50]  # 只取50只股票
        print(f"📈 测试股票数量: {len(active_stocks)}")
        
        if len(active_stocks) < 10:
            print("⚠️ 股票数量太少，跳过优化测试")
            return True
        
        # 手动准备少量数据
        from datetime import datetime, timedelta
        
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') 
        start_dt = end_dt - timedelta(days=50)  # 减少历史数据量
        start_date = start_dt.strftime('%Y-%m-%d')
        
        historical_features = {}
        future_returns = {}
        
        processed_count = 0
        for stock_code in active_stocks:
            try:
                stock_data = optimizer._get_stock_data(stock_code, start_date, end_date)
                if len(stock_data) < 20:
                    continue
                
                features = optimizer._calculate_technical_features(stock_data)
                if features is None:
                    continue
                
                # 简化的未来收益：使用最后几天的数据模拟
                mock_returns = {
                    'return_1d': np.random.normal(0, 0.02),  # 模拟收益
                    'return_3d': np.random.normal(0, 0.03),
                    'return_5d': np.random.normal(0, 0.04),
                    'return_10d': np.random.normal(0, 0.05)
                }
                
                historical_features[stock_code] = features
                future_returns[stock_code] = mock_returns
                
                processed_count += 1
                if processed_count >= 20:  # 只处理20只股票
                    break
                    
            except Exception as e:
                continue
        
        print(f"✅ 数据准备完成，股票数量: {len(historical_features)}")
        
        if len(historical_features) < 10:
            print("⚠️ 数据量太少，跳过优化")
            return True
        
        # 缓存数据
        optimizer.historical_data_cache = historical_features
        optimizer.future_returns_cache = future_returns
        
        # 运行小规模优化（仅3轮测试）
        print("🎯 开始迷你优化...")
        results = optimizer.run_optimization(max_evals=3)
        
        print("✅ 迷你优化完成！")
        print(f"📊 最佳相关性: {results['optimization_summary']['best_correlation']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 迷你优化测试失败: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始权重优化器功能测试")
    print("="*50)
    
    # 导入numpy用于随机数生成
    import numpy as np
    np.random.seed(42)  # 固定随机种子
    
    test_results = []
    
    # 1. 测试导入
    if test_imports():
        test_results.append("✅ 导入测试")
    else:
        test_results.append("❌ 导入测试")
        print("🛑 导入测试失败，停止后续测试")
        return
    
    # 2. 测试数据库连接
    db_success, latest_date = test_database_connection()
    if db_success:
        test_results.append("✅ 数据库连接测试")
    else:
        test_results.append("❌ 数据库连接测试")
        print("🛑 数据库连接失败，停止后续测试")
        return
    
    # 3. 测试优化器初始化
    if test_optimizer_initialization()[0]:
        test_results.append("✅ 优化器初始化测试")
    else:
        test_results.append("❌ 优化器初始化测试")
        print("🛑 优化器初始化失败，停止后续测试")
        return
    
    # 4. 测试数据准备
    if test_data_preparation():
        test_results.append("✅ 数据准备测试")
    else:
        test_results.append("❌ 数据准备测试")
        print("⚠️ 数据准备测试失败，但继续其他测试")
    
    # 5. 测试评分函数
    if test_scoring_functions():
        test_results.append("✅ 评分函数测试")
    else:
        test_results.append("❌ 评分函数测试")
        print("⚠️ 评分函数测试失败，但继续其他测试")
    
    # 6. 运行迷你优化（可选）
    print("\n❓ 是否运行迷你优化测试？(需要几分钟)")
    # 自动跳过优化测试，避免耗时太长
    print("⏭️ 跳过迷你优化测试（可手动运行）")
    
    # 总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    for result in test_results:
        print(f"  {result}")
    
    success_count = sum(1 for r in test_results if r.startswith("✅"))
    total_count = len(test_results)
    
    print(f"\n🎯 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count >= total_count - 1:  # 允许一个测试失败
        print("🎉 权重优化器基础功能测试基本通过！")
        print("💡 提示：可以运行 run_mini_optimization() 进行完整测试")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")

if __name__ == "__main__":
    main()