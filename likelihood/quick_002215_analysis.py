#!/usr/bin/env python3
"""
002215快速相似度分析
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from data_adapter.database_manager import DatabaseManager
from algorithms.matrix_profile import MatrixProfileSimilarity
from algorithms.dtw_similarity import DTWSimilarity
from algorithms.mass_similarity import MASSimilarity


def get_stock_data(db, stock_code, start_date='2025-01-01', end_date='2025-08-08'):
    """获取股票数据"""
    query = '''
    SELECT dq.trade_date, dq.close, dq.volume
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
    ORDER BY dq.trade_date
    '''
    
    result = db.execute_query(query, (stock_code, start_date, end_date))
    if not result:
        return None
        
    df = pd.DataFrame(result, columns=['trade_date', 'close', 'volume'])
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    
    return df


def get_candidate_stocks(db, limit=50):
    """获取候选股票"""
    query = '''
    SELECT DISTINCT s.code, s.name, s.industry
    FROM securities s
    JOIN daily_quotes dq ON s.id = dq.security_id
    WHERE s.type = 'A股' 
      AND dq.trade_date = '2025-08-08'
      AND dq.volume > 500000
      AND dq.close > 2
      AND s.code != '002215'
    ORDER BY dq.volume * dq.close DESC
    LIMIT ?
    '''
    
    result = db.execute_query(query, (limit,))
    return [(row[0], row[1], row[2]) for row in result] if result else []


def compute_similarity(query_series, candidate_series):
    """计算相似度"""
    try:
        mp_algo = MatrixProfileSimilarity({'window_length': 8})
        dtw_algo = DTWSimilarity({'window_type': 'none'})  # 不使用约束
        mass_algo = MASSimilarity()
        
        mp_sim = mp_algo.compute_similarity(query_series, candidate_series)
        dtw_sim = dtw_algo.compute_similarity(query_series, candidate_series)
        mass_sim = mass_algo.compute_similarity(query_series, candidate_series)
        
        # 简单平均
        overall_sim = (mp_sim + dtw_sim + mass_sim) / 3.0
        
        return {
            'overall': overall_sim,
            'mp': mp_sim,
            'dtw': dtw_sim,
            'mass': mass_sim
        }
    except Exception as e:
        print(f"计算相似度失败: {e}")
        return None


def main():
    """主函数"""
    print("="*60)
    print("002215 快速相似度分析")
    print("="*60)
    
    db = DatabaseManager()
    
    # 1. 获取目标股票数据
    print(f"\n1. 获取002215数据...")
    target_data = get_stock_data(db, '002215')
    
    if target_data is None or len(target_data) < 15:
        print("❌ 002215数据不足")
        return
    
    print(f"✅ 获取到 {len(target_data)} 条数据")
    print(f"   日期范围: {target_data.index.min().date()} 至 {target_data.index.max().date()}")
    
    # 2. 提取查询序列（最近15天）
    window_length = 15
    query_returns = target_data['close'].pct_change().fillna(0).tail(window_length).values
    
    print(f"\\n2. 查询序列:")
    print(f"   窗口长度: {window_length} 天")
    print(f"   序列长度: {len(query_returns)}")
    
    # 3. 获取候选股票
    print(f"\\n3. 获取候选股票池...")
    candidates = get_candidate_stocks(db, 50)
    print(f"✅ 候选股票数量: {len(candidates)}")
    
    if not candidates:
        print("❌ 没有候选股票")
        return
    
    # 4. 计算相似度
    print(f"\\n4. 计算相似度...")
    similar_stocks = []
    
    for i, (code, name, industry) in enumerate(candidates):
        try:
            # 获取候选股票数据
            candidate_data = get_stock_data(db, code)
            if candidate_data is None or len(candidate_data) < window_length:
                continue
            
            # 在候选股票历史数据中寻找最佳匹配
            best_similarity = 0
            best_period = None
            
            candidate_returns = candidate_data['close'].pct_change().fillna(0).values
            
            # 滑动窗口
            for j in range(len(candidate_returns) - window_length + 1):
                window_returns = candidate_returns[j:j+window_length]
                
                if len(window_returns) != window_length:
                    continue
                
                # 计算相似度
                sim_result = compute_similarity(query_returns, window_returns)
                if sim_result is None:
                    continue
                
                if sim_result['overall'] > best_similarity:
                    best_similarity = sim_result['overall']
                    period_start = candidate_data.index[j]
                    period_end = candidate_data.index[j + window_length - 1]
                    best_period = {
                        'start': period_start,
                        'end': period_end,
                        'details': sim_result
                    }
            
            # 如果相似度足够高
            if best_similarity > 0.15 and best_period:  # 设置阈值
                similar_stocks.append({
                    'code': code,
                    'name': name,
                    'industry': industry,
                    'similarity_score': best_similarity,
                    'best_period': best_period
                })
            
            if (i + 1) % 10 == 0:
                print(f"   已处理 {i + 1}/{len(candidates)} 只股票，找到 {len(similar_stocks)} 只相似股票")
                
        except Exception as e:
            print(f"   处理 {code} 时出错: {e}")
            continue
    
    # 5. 排序并显示结果
    similar_stocks.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    print(f"\\n" + "="*60)
    print(f"发现 {len(similar_stocks)} 只相似股票")
    print("="*60)
    
    if not similar_stocks:
        print("❌ 未找到足够相似的股票")
        return
    
    # 显示前10只
    print(f"\\n相似度排行榜:")
    print("-"*80)
    print(f"{'排名':<4} {'代码':<8} {'名称':<12} {'行业':<10} {'相似度':<8} {'匹配期间'}")
    print("-"*80)
    
    for i, stock in enumerate(similar_stocks[:10], 1):
        period_str = f"{stock['best_period']['start'].strftime('%m-%d')} 至 {stock['best_period']['end'].strftime('%m-%d')}"
        print(f"{i:<4} {stock['code']:<8} {stock['name'][:10]:<12} {stock['industry'][:8]:<10} "
              f"{stock['similarity_score']:.4f}    {period_str}")
    
    # 6. 分析后续表现
    print(f"\\n📈 后续表现分析...")
    
    performance_results = []
    
    for stock in similar_stocks[:10]:
        try:
            code = stock['code']
            match_end_date = stock['best_period']['end']
            
            # 获取匹配期结束后的数据
            extended_data = get_stock_data(db, code)
            if extended_data is None:
                continue
            
            # 找到匹配期结束后的数据
            future_data = extended_data[extended_data.index > match_end_date]
            
            if len(future_data) < 3:
                continue
            
            # 分析后续表现（取前15天或所有可用数据）
            analysis_data = future_data.head(min(15, len(future_data)))
            
            start_price = analysis_data['close'].iloc[0]
            end_price = analysis_data['close'].iloc[-1]
            max_price = analysis_data['close'].max()
            min_price = analysis_data['close'].min()
            
            total_return = (end_price - start_price) / start_price
            max_gain = (max_price - start_price) / start_price
            max_loss = (min_price - start_price) / start_price
            
            performance_results.append({
                'code': code,
                'name': stock['name'],
                'similarity_score': stock['similarity_score'],
                'analysis_days': len(analysis_data),
                'total_return': total_return,
                'max_gain': max_gain,
                'max_loss': max_loss,
                'match_period': f"{stock['best_period']['start'].strftime('%m-%d')} 至 {stock['best_period']['end'].strftime('%m-%d')}"
            })
            
        except Exception as e:
            print(f"   分析 {stock['code']} 后续表现时出错: {e}")
            continue
    
    # 显示后续表现
    if performance_results:
        print(f"\\n后续表现统计:")
        print("-"*90)
        print(f"{'代码':<8} {'名称':<10} {'相似度':<8} {'天数':<6} {'总收益率':<10} {'最大涨幅':<10} {'最大跌幅'}")
        print("-"*90)
        
        for result in performance_results:
            print(f"{result['code']:<8} {result['name'][:8]:<10} {result['similarity_score']:.4f}    "
                  f"{result['analysis_days']:<6} {result['total_return']:>8.2%} "
                  f"{result['max_gain']:>8.2%} {result['max_loss']:>8.2%}")
        
        # 统计摘要
        returns = [r['total_return'] for r in performance_results]
        if returns:
            avg_return = np.mean(returns)
            positive_count = sum(1 for r in returns if r > 0)
            win_rate = positive_count / len(returns)
            
            print(f"\\n📊 统计摘要:")
            print(f"   平均收益率: {avg_return:.2%}")
            print(f"   胜率: {win_rate:.1%} ({positive_count}/{len(returns)})")
    
    # 7. 生成报告
    print(f"\\n💾 生成详细报告...")
    
    report_path = Path(__file__).parent.parent / 'reports' / 'similarity_analysis'
    report_path.mkdir(parents=True, exist_ok=True)
    report_file = report_path / f'002215_quick_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.md'
    
    # 生成报告内容
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_content = f"""# 002215 (诺普信) 快速相似度分析报告

**生成时间**: {current_time}  
**分析方法**: Matrix Profile + DTW + MASS 算法融合  
**分析窗口**: {window_length} 个交易日  
**候选股票**: {len(candidates)} 只  
**相似股票**: {len(similar_stocks)} 只  

## 目标股票信息

- **股票代码**: 002215
- **股票名称**: 诺普信
- **所属行业**: 农药化肥
- **数据范围**: {target_data.index.min().date()} 至 {target_data.index.max().date()}
- **数据量**: {len(target_data)} 个交易日

## 相似股票发现结果

| 排名 | 代码 | 名称 | 行业 | 相似度 | 匹配期间 | MP相似度 | DTW相似度 | MASS相似度 |
|------|------|------|------|--------|----------|----------|-----------|------------|
"""
    
    for i, stock in enumerate(similar_stocks, 1):
        details = stock['best_period']['details']
        period = f"{stock['best_period']['start'].strftime('%m-%d')} 至 {stock['best_period']['end'].strftime('%m-%d')}"
        report_content += (f"| {i} | {stock['code']} | {stock['name']} | {stock['industry']} | "
                          f"{stock['similarity_score']:.4f} | {period} | "
                          f"{details['mp']:.4f} | {details['dtw']:.4f} | {details['mass']:.4f} |\\n")
    
    if performance_results:
        report_content += f"""

## 后续走势表现分析

| 代码 | 名称 | 相似度 | 匹配期间 | 分析天数 | 总收益率 | 最大涨幅 | 最大跌幅 |
|------|------|--------|----------|----------|----------|----------|----------|
"""
        
        for result in performance_results:
            report_content += (f"| {result['code']} | {result['name']} | {result['similarity_score']:.4f} | "
                              f"{result['match_period']} | {result['analysis_days']} | "
                              f"{result['total_return']:.2%} | {result['max_gain']:.2%} | "
                              f"{result['max_loss']:.2%} |\\n")
        
        # 添加统计摘要
        returns = [r['total_return'] for r in performance_results]
        if returns:
            avg_return = np.mean(returns)
            positive_count = sum(1 for r in returns if r > 0)
            win_rate = positive_count / len(returns)
            
            report_content += f"""

### 统计摘要

- **样本数量**: {len(performance_results)} 只股票
- **平均收益率**: {avg_return:.2%}
- **胜率**: {win_rate:.1%} ({positive_count}/{len(returns)})
"""
    
    report_content += f"""

## 分析方法说明

1. **数据源**: 使用股票历史交易数据
2. **分析窗口**: 最近{window_length}个交易日的收盘价收益率序列
3. **相似度算法**:
   - Matrix Profile: 时间序列模式发现
   - DTW: 动态时间规整，处理时间扭曲
   - MASS: 高效相似子序列搜索
4. **筛选条件**: 
   - 日成交量 > 50万股
   - 股价 > 2元
   - 相似度阈值 > 0.15

## 风险提示

⚠️ 本分析基于历史数据，不构成投资建议。股票投资有风险，请谨慎决策。

---
*生成时间: {current_time}*
"""
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✅ 报告已保存: {report_file}")
    
    print(f"\\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == '__main__':
    main()