#!/usr/bin/env python3
"""
002215 快速长期相似度分析 (2020-2025)
优化版本：使用30天窗口，高效搜索
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加路径
sys.path.append('/Users/yangxu/StockTradebyZ')
sys.path.append('/Users/yangxu/StockTradebyZ/likelihood')

from data_adapter.database_manager import DatabaseManager
from algorithms.matrix_profile import MatrixProfileSimilarity
from algorithms.dtw_similarity import DTWSimilarity
from algorithms.mass_similarity import MASSimilarity


def run_quick_extended_analysis():
    """快速长期相似度分析"""
    print("="*80)
    print("002215 (诺普信) 快速长期相似度分析 (2020-2025)")
    print("分析窗口: 30天 | 时间范围: 5年 | 优化搜索")
    print("="*80)
    
    db = DatabaseManager()
    
    # 参数设置
    window_length = 30
    start_date = '2020-01-01'
    end_date = '2025-08-08'
    
    # 1. 获取002215数据
    print(f"\\n1. 获取002215历史数据...")
    
    target_query = '''
    SELECT dq.trade_date, dq.close, dq.volume
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = '002215'
      AND dq.trade_date >= ?
      AND dq.trade_date <= ?
    ORDER BY dq.trade_date
    '''
    
    target_result = db.execute_query(target_query, (start_date, end_date))
    if not target_result or len(target_result) < window_length * 3:
        print(f"❌ 数据不足: {len(target_result) if target_result else 0} 条")
        return
    
    target_df = pd.DataFrame(target_result, columns=['trade_date', 'close', 'volume'])
    target_df['trade_date'] = pd.to_datetime(target_df['trade_date'])
    target_df.set_index('trade_date', inplace=True)
    
    print(f"✅ 获取到 {len(target_df)} 条数据 ({target_df.index.min().date()} 至 {target_df.index.max().date()})")
    
    # 2. 计算收益率序列
    target_returns = target_df['close'].pct_change().fillna(0).values
    
    # 3. 定义查询窗口（选择3个代表性时期）
    query_windows = []
    
    # 最近、中期、早期各选一个30天窗口
    periods = [
        ('最近期', len(target_returns) - window_length, len(target_returns)),  # 最近30天
        ('中期', len(target_returns) - 250, len(target_returns) - 220),  # 约1年前
        ('早期', len(target_returns) - 500, len(target_returns) - 470),  # 约2年前
    ]
    
    for period_name, start_idx, end_idx in periods:
        if start_idx >= 0 and end_idx <= len(target_returns):
            query_window = target_returns[start_idx:end_idx]
            if len(query_window) == window_length:
                query_windows.append({
                    'name': period_name,
                    'data': query_window,
                    'start_date': target_df.index[start_idx],
                    'end_date': target_df.index[end_idx-1]
                })
    
    print(f"\\n2. 定义 {len(query_windows)} 个查询窗口:")
    for window in query_windows:
        print(f"   - {window['name']}: {window['start_date'].date()} 至 {window['end_date'].date()}")
    
    # 4. 获取候选股票（限制数量提高速度）
    print(f"\\n3. 获取候选股票...")
    
    candidates_query = '''
    SELECT s.code, s.name, s.industry, COUNT(*) as data_count
    FROM securities s
    JOIN daily_quotes dq ON s.id = dq.security_id
    WHERE s.type = 'A股'
      AND dq.trade_date >= ?
      AND dq.trade_date <= ?
      AND s.code != '002215'
    GROUP BY s.code, s.name, s.industry
    HAVING COUNT(*) >= 1000
    ORDER BY COUNT(*) DESC
    LIMIT 50
    '''
    
    candidates_result = db.execute_query(candidates_query, (start_date, end_date))
    if not candidates_result:
        print("❌ 没有候选股票")
        return
    
    candidates = [(row[0], row[1], row[2], row[3]) for row in candidates_result]
    print(f"✅ 选定 {len(candidates)} 只候选股票")
    
    # 初始化算法
    print(f"\\n4. 初始化相似度算法...")
    try:
        mp_algo = MatrixProfileSimilarity({'window_length': 15})
        dtw_algo = DTWSimilarity({'window_type': 'sakoe_chiba', 'sakoe_chiba_radius': 8})
        mass_algo = MASSimilarity()
        print("✅ 算法初始化成功")
    except Exception as e:
        print(f"❌ 算法初始化失败: {e}")
        return
    
    # 5. 执行相似度分析
    all_similar_stocks = {}
    
    for window_info in query_windows:
        window_name = window_info['name']
        query_data = window_info['data']
        
        print(f"\\n5. 分析{window_name}期间...")
        
        similar_stocks = []
        processed_count = 0
        
        for code, name, industry, data_count in candidates:
            try:
                # 获取候选股票数据
                cand_query = '''
                SELECT dq.trade_date, dq.close
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ?
                  AND dq.trade_date >= ?
                  AND dq.trade_date <= ?
                ORDER BY dq.trade_date
                '''
                
                cand_result = db.execute_query(cand_query, (code, start_date, end_date))
                if not cand_result or len(cand_result) < 800:
                    continue
                
                cand_df = pd.DataFrame(cand_result, columns=['trade_date', 'close'])
                cand_df['trade_date'] = pd.to_datetime(cand_df['trade_date'])
                cand_df.set_index('trade_date', inplace=True)
                
                cand_returns = cand_df['close'].pct_change().fillna(0).values
                
                # 滑动窗口搜索（加大步长提高效率）
                best_similarity = 0
                best_period = None
                
                step_size = 10  # 加大步长
                
                for j in range(0, len(cand_returns) - window_length + 1, step_size):
                    cand_window = cand_returns[j:j + window_length]
                    
                    if len(cand_window) != window_length:
                        continue
                    
                    try:
                        # 计算相似度（简化版本）
                        mp_sim = mp_algo.compute_similarity(query_data, cand_window)
                        dtw_sim = dtw_algo.compute_similarity(query_data, cand_window)
                        mass_sim = mass_algo.compute_similarity(query_data, cand_window)
                        
                        # 简单平均
                        overall_sim = (mp_sim + dtw_sim + mass_sim) / 3.0
                        
                        if overall_sim > best_similarity:
                            best_similarity = overall_sim
                            best_period = {
                                'start': cand_df.index[j],
                                'end': cand_df.index[j + window_length - 1],
                                'mp': mp_sim,
                                'dtw': dtw_sim,
                                'mass': mass_sim
                            }
                    
                    except Exception:
                        continue
                
                # 如果相似度足够高
                if best_similarity > 0.3 and best_period:  # 提高阈值
                    similar_stocks.append({
                        'code': code,
                        'name': name,
                        'industry': industry,
                        'similarity_score': best_similarity,
                        'best_period': best_period,
                        'query_window': window_name
                    })
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"   已处理 {processed_count}/{len(candidates)}，找到 {len(similar_stocks)} 只")
            
            except Exception as e:
                continue
        
        # 排序并保存结果
        similar_stocks.sort(key=lambda x: x['similarity_score'], reverse=True)
        all_similar_stocks[window_name] = similar_stocks[:15]  # 每个窗口保留前15只
        
        print(f"   ✅ {window_name}找到 {len(similar_stocks)} 只相似股票")
        
        # 显示该窗口的前5只
        if similar_stocks:
            print(f"   前5只:")
            for i, stock in enumerate(similar_stocks[:5], 1):
                print(f"     {i}. {stock['code']} ({stock['name'][:8]}) - 相似度 {stock['similarity_score']:.4f}")
    
    # 6. 分析后续表现
    print(f"\\n📈 后续表现分析...")
    
    all_performance = {}
    
    for window_name, similar_stocks in all_similar_stocks.items():
        performance_results = []
        
        for stock in similar_stocks[:10]:  # 分析前10只
            try:
                code = stock['code']
                match_end_date = stock['best_period']['end']
                
                # 获取匹配期后的数据
                future_query = '''
                SELECT dq.trade_date, dq.close
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ?
                  AND dq.trade_date > ?
                  AND dq.trade_date <= ?
                ORDER BY dq.trade_date
                LIMIT 30
                '''
                
                future_result = db.execute_query(future_query, (code, match_end_date.strftime('%Y-%m-%d'), end_date))
                if not future_result or len(future_result) < 5:
                    continue
                
                future_df = pd.DataFrame(future_result, columns=['trade_date', 'close'])
                
                # 计算收益率
                start_price = future_df['close'].iloc[0]
                end_price = future_df['close'].iloc[-1]
                max_price = future_df['close'].max()
                min_price = future_df['close'].min()
                
                total_return = (end_price - start_price) / start_price
                max_gain = (max_price - start_price) / start_price
                max_loss = (min_price - start_price) / start_price
                
                performance_results.append({
                    'code': code,
                    'name': stock['name'],
                    'similarity_score': stock['similarity_score'],
                    'match_period': f"{stock['best_period']['start'].strftime('%Y-%m-%d')} 至 {stock['best_period']['end'].strftime('%Y-%m-%d')}",
                    'analysis_days': len(future_df),
                    'total_return': total_return,
                    'max_gain': max_gain,
                    'max_loss': max_loss
                })
            
            except Exception:
                continue
        
        all_performance[window_name] = performance_results
        
        # 显示该窗口表现统计
        if performance_results:
            returns = [r['total_return'] for r in performance_results]
            avg_return = np.mean(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            
            print(f"   {window_name}表现: 平均收益 {avg_return:.2%}, 胜率 {win_rate:.1%} ({len(returns)}只)")
    
    # 7. 生成简化报告
    print(f"\\n💾 生成报告...")
    
    report_path = Path('/Users/yangxu/StockTradebyZ/reports/similarity_analysis')
    report_path.mkdir(parents=True, exist_ok=True)
    report_file = report_path / f'002215_quick_extended_{datetime.now().strftime("%Y%m%d_%H%M")}.md'
    
    # 生成报告内容
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_content = f"""# 002215 (诺普信) 快速长期相似度分析报告

**生成时间**: {current_time}  
**分析工具**: Matrix Profile + DTW + MASS  
**分析窗口**: {window_length} 个交易日  
**时间范围**: {start_date} 至 {end_date} (5年)  
**候选股票**: {len(candidates)} 只  

---

## 🎯 分析概述

本报告在2020-2025年的5年时间范围内，使用30天滑动窗口寻找与002215（诺普信）相似的走势模式。

---

## 📊 分析结果

"""
    
    # 统计总览
    total_found = sum(len(stocks) for stocks in all_similar_stocks.values())
    unique_stocks = set()
    for stocks in all_similar_stocks.values():
        for stock in stocks:
            unique_stocks.add(stock['code'])
    
    report_content += f"""
### 总体发现

- **查询时期**: {len(all_similar_stocks)} 个
- **相似匹配**: {total_found} 次  
- **独特股票**: {len(unique_stocks)} 只

"""
    
    # 各时期详细结果
    for window_name, similar_stocks in all_similar_stocks.items():
        if not similar_stocks:
            continue
        
        report_content += f"""

### {window_name} 相似股票

| 排名 | 代码 | 名称 | 行业 | 相似度 | 匹配期间 | MP | DTW | MASS |
|------|------|------|------|--------|----------|----|----|------|
"""
        
        for i, stock in enumerate(similar_stocks, 1):
            period = f"{stock['best_period']['start'].strftime('%Y-%m-%d')} 至 {stock['best_period']['end'].strftime('%Y-%m-%d')}"
            details = stock['best_period']
            
            report_content += (f"| {i} | {stock['code']} | {stock['name']} | {stock['industry']} | "
                              f"{stock['similarity_score']:.4f} | {period} | "
                              f"{details['mp']:.3f} | {details['dtw']:.3f} | {details['mass']:.3f} |\\n")
        
        # 后续表现
        if window_name in all_performance and all_performance[window_name]:
            perf_data = all_performance[window_name]
            returns = [r['total_return'] for r in perf_data]
            
            if returns:
                avg_return = np.mean(returns)
                median_return = np.median(returns)
                win_rate = sum(1 for r in returns if r > 0) / len(returns)
                max_return = max(returns)
                min_return = min(returns)
                
                report_content += f"""

#### {window_name} 后续表现 ({len(returns)} 样本)

- 平均收益率: {avg_return:.2%}
- 中位数收益率: {median_return:.2%}  
- 胜率: {win_rate:.1%}
- 最高收益: {max_return:.2%}
- 最低收益: {min_return:.2%}

"""
    
    # 高频股票统计
    stock_frequency = {}
    stock_info = {}
    
    for window_name, similar_stocks in all_similar_stocks.items():
        for stock in similar_stocks:
            code = stock['code']
            if code not in stock_frequency:
                stock_frequency[code] = 0
                stock_info[code] = {
                    'name': stock['name'],
                    'industry': stock['industry'],
                    'total_similarity': 0
                }
            stock_frequency[code] += 1
            stock_info[code]['total_similarity'] += stock['similarity_score']
    
    frequent_stocks = sorted(stock_frequency.items(), key=lambda x: x[1], reverse=True)
    
    if frequent_stocks:
        report_content += f"""

---

## 🏆 跨期间高频相似股票

| 股票代码 | 股票名称 | 行业 | 出现次数 | 平均相似度 |
|----------|----------|------|----------|------------|
"""
        
        for code, frequency in frequent_stocks[:15]:
            info = stock_info[code]
            avg_sim = info['total_similarity'] / frequency
            
            report_content += (f"| {code} | {info['name']} | {info['industry']} | "
                              f"{frequency} | {avg_sim:.4f} |\\n")
    
    report_content += f"""

---

## 🔬 方法说明

### 算法组合
- **Matrix Profile**: 时间序列模式发现
- **DTW**: 动态时间规整
- **MASS**: 快速相似搜索

### 参数设置
- **分析窗口**: 30天
- **滑动步长**: 10天 (平衡精度与效率)
- **相似度阈值**: 0.3
- **候选股票**: 50只 (数据最完整)

---

## ⚠️ 重要说明

1. **快速版本**: 为提高效率，使用了较大步长和较少候选股票
2. **历史相似性**: 不保证未来重现相同模式
3. **仅供参考**: 不构成投资建议
4. **需要验证**: 建议结合基本面分析

---

**生成时间**: {current_time}  
**版本**: 快速长期分析 v1.0  
"""
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 8. 显示最终总结
    print(f"\\n" + "="*80)
    print("🎉 002215快速长期分析完成！")
    print("="*80)
    
    print(f"📊 总体发现:")
    print(f"   时间跨度: 5年 ({start_date} 至 {end_date})")
    print(f"   查询时期: {len(all_similar_stocks)} 个")
    print(f"   相似匹配: {total_found} 次")
    print(f"   独特股票: {len(unique_stocks)} 只")
    
    if frequent_stocks:
        print(f"\\n🏆 高频相似股票 (前5):")
        for i, (code, freq) in enumerate(frequent_stocks[:5], 1):
            name = stock_info[code]['name']
            print(f"   {i}. {code} ({name[:8]}) - 出现{freq}次")
    
    # 综合表现统计
    all_returns = []
    for perf_data in all_performance.values():
        for result in perf_data:
            all_returns.append(result['total_return'])
    
    if all_returns:
        overall_avg = np.mean(all_returns)
        overall_win_rate = sum(1 for r in all_returns if r > 0) / len(all_returns)
        print(f"\\n📈 综合表现 ({len(all_returns)}个样本):")
        print(f"   平均收益率: {overall_avg:.2%}")
        print(f"   胜率: {overall_win_rate:.1%}")
    
    print(f"\\n📄 详细报告: {report_file}")


if __name__ == '__main__':
    run_quick_extended_analysis()