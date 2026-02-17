#!/usr/bin/env python3
"""
小盘股相似度搜索测试
使用002215（诺普信）作为查询股票，测试相似市值和成交额的股票
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import time

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.search_engine import SimilaritySearchEngine
from data_preprocessing.data_loader import DataLoader


def test_small_cap_similarity():
    """小盘股相似度测试"""
    print("=" * 60)
    print("小盘股相似度搜索测试 - 002215 诺普信")
    print("=" * 60)
    
    # 查询股票
    query_stock = '002215'  # 诺普信
    query_date = '2025-08-08'
    window_length = 5
    
    # 相似成交额的候选股票（手动指定以确保测试效果）
    candidate_stocks = [
        '002204',  # 大连重工
        '002432',  # 九安医疗  
        '002467',  # 二六三
        '000880',  # 潍柴重机
        '002178',  # 延华智能
        '002564',  # 天沃科技
        '000762',  # 西藏矿业
        '000547',  # 航天发展
        '000953',  # 河化股份
        '002045'   # 国光电器
    ]
    
    # 加载配置
    config_path = Path(__file__).parent / 'configs' / 'default_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 调整配置以适应小盘股测试
    config['filters']['min_daily_volume'] = 1000000    # 降低到100万
    config['filters']['min_market_cap'] = 100000       # 降低到10万
    config['similarity']['search']['min_similarity'] = 0.0           # 不设最低相似度门槛
    config['similarity']['search']['parallel_workers'] = 1           # 单线程避免日志混乱
    
    try:
        print(f"\n1. 加载查询股票数据: {query_stock}")
        data_loader = DataLoader(config=config)
        
        # 加载查询股票数据
        start_date = pd.to_datetime(query_date) - pd.Timedelta(days=15)
        query_data = data_loader.load_stock_data(
            query_stock, 
            start_date.strftime('%Y-%m-%d'), 
            query_date
        )
        
        print(f"   ✅ 查询数据: {query_data.shape}")
        print(f"   📊 价格范围: {query_data['close'].iloc[0]:.2f} -> {query_data['close'].iloc[-1]:.2f}")
        print(f"   📈 收益率: {((query_data['close'].iloc[-1] / query_data['close'].iloc[0]) - 1) * 100:.2f}%")
        
        # 检查每只候选股票的数据
        print(f"\n2. 检查候选股票数据...")
        valid_candidates = []
        
        for stock in candidate_stocks:
            try:
                candidate_data = data_loader.load_stock_data(
                    stock, 
                    start_date.strftime('%Y-%m-%d'), 
                    query_date
                )
                if len(candidate_data) >= window_length:
                    valid_candidates.append(stock)
                    last_close = candidate_data['close'].iloc[-1]
                    first_close = candidate_data['close'].iloc[0] 
                    return_pct = ((last_close / first_close) - 1) * 100
                    print(f"   ✅ {stock}: {candidate_data.shape[0]}天, 收益率 {return_pct:+.2f}%")
                else:
                    print(f"   ❌ {stock}: 数据不足 ({len(candidate_data)}天)")
            except Exception as e:
                print(f"   ❌ {stock}: 加载失败 - {str(e)}")
        
        print(f"\n   📋 有效候选股票: {len(valid_candidates)} 只")
        
        if not valid_candidates:
            print("   ⚠️  没有有效的候选股票，退出测试")
            return False
            
    except Exception as e:
        print(f"   ❌ 数据准备失败: {str(e)}")
        return False
    
    try:
        print(f"\n3. 手动相似度计算测试...")
        
        # 创建搜索引擎
        search_engine = SimilaritySearchEngine(config)
        
        # 提取查询序列
        recent_query_data = query_data.tail(window_length)
        query_series = search_engine._extract_query_series(recent_query_data, window_length)
        print(f"   ✅ 查询序列: {list(query_series.keys())}")
        print(f"   📊 价格序列长度: {len(query_series.get('price', []))}")
        
        # 手动计算相似度
        similarities = []
        
        for stock in valid_candidates[:5]:  # 只测试前5只以节省时间
            try:
                print(f"\n   🔍 分析 {stock}...")
                
                # 加载候选股票历史数据 (更长时间用于搜索)
                search_start = pd.to_datetime(query_date) - pd.Timedelta(days=60)
                candidate_data = data_loader.load_stock_data(
                    stock,
                    search_start.strftime('%Y-%m-%d'), 
                    query_date
                )
                
                # 提取候选序列
                candidate_series = search_engine._extract_candidate_series(
                    candidate_data, query_series
                )
                
                # 计算相似度
                stock_matches = search_engine._compute_similarities(
                    query_series, candidate_series, stock, candidate_data
                )
                
                if stock_matches:
                    best_match = max(stock_matches, key=lambda x: x['similarity_score'])
                    similarities.append({
                        'stock': stock,
                        'similarity': best_match['similarity_score'],
                        'period': f"{best_match['period_start']} to {best_match['period_end']}",
                        'algorithm_scores': best_match['algorithm_scores']
                    })
                    print(f"     ✅ 最佳相似度: {best_match['similarity_score']:.4f}")
                    print(f"     📅 匹配期间: {best_match['period_start']} to {best_match['period_end']}")
                    print(f"     🔧 算法分数: {best_match['algorithm_scores']}")
                else:
                    print(f"     ❌ 未找到匹配")
                    
            except Exception as e:
                print(f"     ❌ 计算失败: {str(e)}")
                continue
        
        # 排序显示结果
        if similarities:
            print(f"\n4. 相似度排名 (Top-{len(similarities)}):")
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            for i, sim in enumerate(similarities, 1):
                print(f"\n   第 {i} 名: {sim['stock']}")
                print(f"   🎯 相似度: {sim['similarity']:.4f}")
                print(f"   📅 匹配期间: {sim['period']}")
                print(f"   📊 算法详情:")
                for alg, score in sim['algorithm_scores'].items():
                    print(f"      - {alg}: {score:.4f}")
        else:
            print(f"\n   ⚠️  没有找到有效的相似匹配")
            
    except Exception as e:
        print(f"   ❌ 相似度计算失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print(f"\n5. 完整搜索引擎测试...")
        
        # 临时修改候选股票池（避免搜索所有股票）
        original_filter_method = search_engine._get_candidate_stocks
        
        def mock_get_candidate_stocks(query_stock_code, kwargs):
            """模拟获取候选股票池"""
            candidates = [s for s in valid_candidates if s != query_stock_code]
            print(f"   📋 模拟候选池: {len(candidates)} 只股票")
            return candidates
        
        search_engine._get_candidate_stocks = mock_get_candidate_stocks
        
        # 运行完整搜索
        start_time = time.time()
        result = search_engine.search_similar_patterns(
            stock_code=query_stock,
            query_date=query_date,
            window_length=window_length
        )
        search_time = time.time() - start_time
        
        print(f"   ✅ 搜索完成: {search_time:.2f}秒")
        print(f"   📊 状态: {result['status']}")
        
        if result['status'] == 'success':
            patterns = result['similar_patterns']
            print(f"   🎯 找到 {len(patterns)} 个相似模式")
            
            for pattern in patterns[:3]:  # 显示前3个
                print(f"\n     排名 {pattern['rank']}: {pattern['stock']}")
                print(f"     相似度: {pattern['similarity_score']:.4f}")
                print(f"     时期: {pattern['period_start']} to {pattern['period_end']}")
        else:
            print(f"   ❌ 搜索失败: {result.get('message', '未知错误')}")
            
    except Exception as e:
        print(f"   ❌ 完整搜索测试失败: {str(e)}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 小盘股相似度搜索测试完成！")
    print(f"🎯 查询股票: {query_stock} (诺普信)")
    print(f"📊 测试窗口: {window_length} 天")
    print(f"🔍 候选股票: {len(valid_candidates)} 只")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_small_cap_similarity()
    sys.exit(0 if success else 1)