#!/usr/bin/env python3
"""
简化的相似度搜索测试
用于快速验证算法功能
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.search_engine import SimilaritySearchEngine
from data_preprocessing.data_loader import DataLoader


def simple_test():
    """简化测试"""
    print("=" * 60)
    print("股票相似度搜索 - 简化测试")
    print("=" * 60)
    
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'default_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 减少候选股票数量以加快测试
    config['filters']['min_daily_volume'] = 50000000  # 5千万
    
    try:
        # 创建数据加载器测试
        print("\n1. 测试数据加载...")
        data_loader = DataLoader(config=config)
        
        # 测试加载股票数据
        stock_data = data_loader.load_stock_data('000001', '2025-08-01', '2025-08-08')
        print(f"   ✅ 成功加载数据: {stock_data.shape}")
        print(f"   - 数据列: {list(stock_data.columns[:10])}...")
        
        # 测试筛选股票
        candidate_stocks = data_loader.filter_stocks_by_criteria(
            min_volume=config['filters']['min_daily_volume'],
            date='2025-08-08'
        )
        print(f"   ✅ 候选股票数量: {len(candidate_stocks)}")
        print(f"   - 前10只: {candidate_stocks[:10]}")
        
    except Exception as e:
        print(f"   ❌ 数据加载失败: {str(e)}")
        return False
    
    try:
        # 算法测试
        print("\n2. 测试相似度算法...")
        from algorithms.matrix_profile import MatrixProfileSimilarity
        from algorithms.dtw_similarity import DTWSimilarity
        from algorithms.mass_similarity import MASSimilarity
        
        # 创建测试数据
        np.random.seed(42)
        query = np.sin(np.linspace(0, 2*np.pi, 10)) + 0.1 * np.random.randn(10)
        candidate = np.sin(np.linspace(0, 2*np.pi, 10) + 0.1) + 0.1 * np.random.randn(10)
        
        # 测试Matrix Profile
        mp = MatrixProfileSimilarity({'window_length': 5})
        mp_sim = mp.compute_similarity(query, candidate)
        print(f"   ✅ Matrix Profile相似度: {mp_sim:.4f}")
        
        # 测试DTW  
        dtw = DTWSimilarity({'window_type': 'none'})
        dtw_sim = dtw.compute_similarity(query, candidate)
        print(f"   ✅ DTW相似度: {dtw_sim:.4f}")
        
        # 测试MASS
        mass = MASSimilarity()
        mass_sim = mass.compute_similarity(query, candidate)
        print(f"   ✅ MASS相似度: {mass_sim:.4f}")
        
    except Exception as e:
        print(f"   ❌ 算法测试失败: {str(e)}")
        return False
    
    try:
        # 搜索引擎测试（简化版）
        print("\n3. 测试搜索引擎（简化版）...")
        
        # 使用更小的候选池
        limited_candidates = candidate_stocks[:10] if len(candidate_stocks) > 10 else candidate_stocks
        
        # 手动构建简化的搜索
        search_engine = SimilaritySearchEngine(config)
        
        # 测试数据提取
        query_data = data_loader.load_stock_data('000001', '2025-08-04', '2025-08-08')
        if len(query_data) >= 4:
            query_series = search_engine._extract_query_series(query_data, 4)
            print(f"   ✅ 查询序列提取成功: {list(query_series.keys())}")
            print(f"   - 价格序列长度: {len(query_series.get('price', []))}")
            
            # 测试单只候选股票的相似度计算
            if limited_candidates:
                test_candidate = limited_candidates[0]
                print(f"   📊 测试候选股票: {test_candidate}")
                
                # 加载候选股票数据
                candidate_data = data_loader.load_stock_data(
                    test_candidate, '2025-07-01', '2025-08-08'
                )
                
                if len(candidate_data) >= 4:
                    candidate_series = search_engine._extract_candidate_series(
                        candidate_data, query_series
                    )
                    print(f"   ✅ 候选序列提取成功: {list(candidate_series.keys())}")
                    
                    # 计算相似度
                    matches = search_engine._compute_similarities(
                        query_series, candidate_series, test_candidate, candidate_data
                    )
                    print(f"   ✅ 找到 {len(matches)} 个匹配")
                    
                    if matches:
                        best_match = max(matches, key=lambda x: x['similarity_score'])
                        print(f"   🎯 最佳匹配相似度: {best_match['similarity_score']:.4f}")
                        print(f"   - 时期: {best_match['period_start']} 至 {best_match['period_end']}")
                
        else:
            print("   ⚠️  数据不足，跳过搜索引擎测试")
        
    except Exception as e:
        print(f"   ❌ 搜索引擎测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成！相似度搜索系统基本功能正常。")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = simple_test()
    sys.exit(0 if success else 1)