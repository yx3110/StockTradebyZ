#!/usr/bin/env python3
"""
增强型股票相似度分析工具 - 包含后续走势跟踪
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# 添加项目根目录到路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager
# 直接导入需要的函数
import stumpy
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr

def track_future_performance(db, stock_code, query_date, days_forward=[5, 10]):
    """
    跟踪股票在特定日期后的走势表现
    
    返回:
    {
        '5_days': {
            'max_gain': 最大浮盈百分比,
            'max_loss': 最大浮亏百分比,
            'final_return': 最终收益率
        },
        '10_days': {...}
    }
    """
    results = {}
    
    # 获取基准日期的收盘价
    base_query = """
    SELECT close 
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date = ?
    """
    
    base_price = db.execute_query(base_query, (stock_code, query_date))
    if not base_price:
        return results
    
    base_price = base_price[0][0]
    
    # 获取后续N天的数据
    future_query = """
    SELECT trade_date, high, low, close
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date > ?
    ORDER BY trade_date ASC
    LIMIT ?
    """
    
    for days in days_forward:
        future_data = db.execute_query(future_query, (stock_code, query_date, days))
        
        if future_data:
            highs = [row[1] for row in future_data]
            lows = [row[2] for row in future_data]
            closes = [row[3] for row in future_data]
            
            # 计算最大浮盈和浮亏
            max_gain = (max(highs) / base_price - 1) * 100
            max_loss = (min(lows) / base_price - 1) * 100
            
            # 计算最终收益
            final_return = (closes[-1] / base_price - 1) * 100 if len(closes) >= days else None
            
            results[f'{days}_days'] = {
                'max_gain': round(max_gain, 2),
                'max_loss': round(max_loss, 2),
                'final_return': round(final_return, 2) if final_return else None,
                'actual_days': len(future_data)
            }
        else:
            results[f'{days}_days'] = {
                'max_gain': None,
                'max_loss': None,
                'final_return': None,
                'actual_days': 0
            }
    
    return results

def get_stock_fundamentals(db, stock_code, trade_date=None):
    """获取股票的基本面、技术面、财务面数据"""
    
    if not trade_date:
        trade_date = datetime.now().strftime('%Y-%m-%d')
    
    # 基本信息
    basic_query = """
    SELECT s.name, sbi.industry, sbi.area, sbi.market, sbi.list_date
    FROM securities s
    LEFT JOIN stock_basic_info sbi ON s.code = sbi.code
    WHERE s.code = ?
    """
    
    # 最新财务数据
    financial_query = """
    SELECT 
        db.pe_ttm, db.pb, db.ps_ttm, db.dv_ratio, db.dv_ttm,
        db.total_mv, db.circ_mv, db.turnover_rate, db.volume_ratio
    FROM daily_basic db
    JOIN securities s ON db.security_id = s.id
    WHERE s.code = ? AND db.trade_date <= ?
    ORDER BY db.trade_date DESC
    LIMIT 1
    """
    
    # 技术指标
    technical_query = """
    SELECT 
        ti.ma_5, ti.ma_10, ti.ma_20, ti.ma_60,
        ti.rsi_6, ti.rsi_12, ti.macd, ti.signal, ti.kdj_k, ti.kdj_d, ti.kdj_j,
        ti.bbi
    FROM technical_indicators ti
    JOIN securities s ON ti.security_id = s.id
    WHERE s.code = ? AND ti.trade_date <= ?
    ORDER BY ti.trade_date DESC
    LIMIT 1
    """
    
    # 最新价格数据
    price_query = """
    SELECT 
        dq.close, dq.volume, dq.price_change_pct,
        dq.high, dq.low, dq.open
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date <= ?
    ORDER BY dq.trade_date DESC
    LIMIT 1
    """
    
    # 执行查询
    basic_info = db.execute_query(basic_query, (stock_code,))
    financial_data = db.execute_query(financial_query, (stock_code, trade_date))
    technical_data = db.execute_query(technical_query, (stock_code, trade_date))
    price_data = db.execute_query(price_query, (stock_code, trade_date))
    
    result = {
        'code': stock_code,
        'basic': {},
        'financial': {},
        'technical': {},
        'price': {}
    }
    
    if basic_info:
        result['basic'] = {
            'name': basic_info[0][0],
            'industry': basic_info[0][1],
            'area': basic_info[0][2],
            'market': basic_info[0][3],
            'list_date': basic_info[0][4]
        }
    
    if financial_data:
        result['financial'] = {
            'pe_ttm': financial_data[0][0],
            'pb': financial_data[0][1],
            'ps_ttm': financial_data[0][2],
            'dv_ratio': financial_data[0][3],
            'dv_ttm': financial_data[0][4],
            'total_mv': financial_data[0][5],
            'circ_mv': financial_data[0][6],
            'turnover_rate': financial_data[0][7],
            'volume_ratio': financial_data[0][8]
        }
    
    if technical_data:
        result['technical'] = {
            'ma_5': technical_data[0][0],
            'ma_10': technical_data[0][1],
            'ma_20': technical_data[0][2],
            'ma_60': technical_data[0][3],
            'rsi_6': technical_data[0][4],
            'rsi_12': technical_data[0][5],
            'macd': technical_data[0][6],
            'signal': technical_data[0][7],
            'kdj_k': technical_data[0][8],
            'kdj_d': technical_data[0][9],
            'kdj_j': technical_data[0][10],
            'bbi': technical_data[0][11]
        }
    
    if price_data:
        result['price'] = {
            'close': price_data[0][0],
            'volume': price_data[0][1],
            'change_pct': price_data[0][2],
            'high': price_data[0][3],
            'low': price_data[0][4],
            'open': price_data[0][5]
        }
    
    return result

def run_enhanced_similarity_analysis(
    target_code='002215',
    window_length=15,
    start_date='2020-01-01',
    end_date='2025-08-11',
    similarity_threshold=0.12,
    candidate_limit=None,
    multi_period=True,
    n_processes=None
):
    """运行增强型相似度分析，包含后续走势跟踪"""
    
    start_time = time.time()
    
    # 初始化数据库
    db = DatabaseManager()
    
    print("="*80)
    print("🚀 增强型股票相似度分析工具 - 投资决策版")
    print(f"目标股票: {target_code} | 分析窗口: {window_length}天")
    print(f"扫描范围: {'全A股' if candidate_limit is None else f'{candidate_limit}只'} | 分析模式: {'多时期' if multi_period else '单时期'}")
    print("="*80)
    
    # 获取目标股票数据
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 获取目标股票数据...")
    
    target_query = """
    SELECT dq.trade_date, dq.close, dq.volume
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
    ORDER BY dq.trade_date ASC
    """
    
    target_data = db.execute_query(target_query, (target_code, start_date, end_date))
    
    if not target_data:
        print(f"❌ 未找到股票 {target_code} 的数据")
        return
    
    # 构建查询窗口
    query_windows = []
    if multi_period:
        # 多时期分析：每季度一个窗口
        current_date = datetime.strptime(end_date, '%Y-%m-%d')
        for i in range(4):
            window_end = current_date - timedelta(days=90*i)
            window_end_str = window_end.strftime('%Y-%m-%d')
            query_windows.append({
                'name': f"Q{i+1}-{window_end.year}",
                'end_date': window_end_str
            })
    else:
        # 单时期分析：只分析最近期
        query_windows.append({
            'name': '最近期',
            'end_date': end_date
        })
    
    # 获取候选股票
    candidates = get_candidates_from_db(db, target_code, candidate_limit)
    
    if not candidates:
        print("❌ 没有找到候选股票")
        return
    
    print(f"✅ 获取候选股票: {len(candidates)} 只")
    
    # 预加载所有候选股票数据
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 批量预加载候选股票数据...")
    preloaded_data = preload_all_candidates_data(db, candidates, start_date, end_date)
    
    # 设置进程数
    if n_processes is None:
        n_processes = min(cpu_count(), 8)
    
    # 分批处理
    batch_size = max(1, len(candidates) // (n_processes * 2))
    candidate_batches = [candidates[i:i+batch_size] for i in range(0, len(candidates), batch_size)]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ 开始并行相似度计算...")
    print(f"   分批设置: {len(candidate_batches)}批, 每批约{batch_size}只股票")
    
    # 准备任务参数
    tasks = []
    for batch_id, batch in enumerate(candidate_batches):
        tasks.append((
            batch, query_windows, preloaded_data,
            window_length, similarity_threshold,
            f"Worker-{batch_id+1}", batch_id
        ))
    
    # 并行执行
    all_results = {}
    total_processed = 0
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        future_to_batch = {executor.submit(worker_process_similarity, task): task for task in tasks}
        
        completed = 0
        for future in as_completed(future_to_batch):
            result = future.result()
            completed += 1
            
            if 'error' not in result:
                total_processed += result['processed_count']
                
                # 合并结果
                for window_name, window_results in result['results'].items():
                    if window_name not in all_results:
                        all_results[window_name] = []
                    all_results[window_name].extend(window_results)
            
            print_progress_bar(completed, len(tasks),
                             prefix="   计算进度",
                             suffix=f"已处理{total_processed}次操作")
    
    # 结果排序
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📊 分析结果并跟踪后续走势...")
    
    for window_name in all_results:
        all_results[window_name].sort(key=lambda x: x['similarity_score'], reverse=True)
        all_results[window_name] = all_results[window_name][:20]  # 保留前20
        
        # 为每个相似股票添加后续走势数据
        for stock in all_results[window_name]:
            # 获取相似度计算时的日期
            query_date = stock['best_period']['end_date']
            
            # 跟踪后续走势
            future_performance = track_future_performance(db, stock['code'], query_date)
            stock['future_performance'] = future_performance
    
    # 获取目标股票的完整数据
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📈 获取目标股票基本面数据...")
    target_fundamentals = get_stock_fundamentals(db, target_code)
    
    # 生成报告
    total_elapsed = time.time() - start_time
    
    report_path = Path('/Users/yangxu/StockTradebyZ/reports/similarity_analysis')
    report_path.mkdir(parents=True, exist_ok=True)
    report_file = report_path / f'{target_code}_enhanced_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.md'
    
    # 生成报告内容
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report_content = f"""# {target_code} 增强型相似度分析报告

**生成时间**: {current_time}  
**分析耗时**: {total_elapsed/60:.1f} 分钟  
**分析窗口**: {window_length} 个交易日  
**时间范围**: {start_date} 至 {end_date}  
**候选股票**: {len(candidates)} 只  
**相似度阈值**: {similarity_threshold}  

---

## 📊 目标股票基本信息

### 基础信息
- **股票代码**: {target_code}
- **股票名称**: {target_fundamentals['basic'].get('name', 'N/A')}
- **所属行业**: {target_fundamentals['basic'].get('industry', 'N/A')}
- **地区**: {target_fundamentals['basic'].get('area', 'N/A')}
- **上市日期**: {target_fundamentals['basic'].get('list_date', 'N/A')}

### 财务指标
- **市盈率(TTM)**: {target_fundamentals['financial'].get('pe_ttm', 'N/A')}
- **市净率**: {target_fundamentals['financial'].get('pb', 'N/A')}
- **市销率(TTM)**: {target_fundamentals['financial'].get('ps_ttm', 'N/A')}
- **总市值**: {target_fundamentals['financial'].get('total_mv', 'N/A')} 万元
- **流通市值**: {target_fundamentals['financial'].get('circ_mv', 'N/A')} 万元
- **换手率**: {target_fundamentals['financial'].get('turnover_rate', 'N/A')}%

### 技术指标
- **MA5**: {target_fundamentals['technical'].get('ma_5', 'N/A')}
- **MA20**: {target_fundamentals['technical'].get('ma_20', 'N/A')}
- **RSI(6)**: {target_fundamentals['technical'].get('rsi_6', 'N/A')}
- **KDJ_K**: {target_fundamentals['technical'].get('kdj_k', 'N/A')}
- **MACD**: {target_fundamentals['technical'].get('macd', 'N/A')}
- **BBI**: {target_fundamentals['technical'].get('bbi', 'N/A')}

---
"""
    
    # 各时期相似股票及后续走势
    for window_name, similar_stocks in all_results.items():
        if similar_stocks:
            report_content += f"""
## 📅 {window_name} 相似股票分析

**发现数量**: {len(similar_stocks)} 只

### 相似股票排行榜及后续走势

| 排名 | 代码 | 名称 | 行业 | 相似度 | 5日最大浮盈 | 5日最大浮亏 | 5日收益 | 10日最大浮盈 | 10日最大浮亏 | 10日收益 |
|------|------|------|------|--------|------------|------------|---------|-------------|-------------|----------|
"""
            
            for i, stock in enumerate(similar_stocks, 1):
                perf_5 = stock['future_performance'].get('5_days', {})
                perf_10 = stock['future_performance'].get('10_days', {})
                
                report_content += (
                    f"| {i} | {stock['code']} | {stock['name']} | {stock['industry']} | "
                    f"{stock['similarity_score']:.4f} | "
                    f"{perf_5.get('max_gain', 'N/A')}% | "
                    f"{perf_5.get('max_loss', 'N/A')}% | "
                    f"{perf_5.get('final_return', 'N/A')}% | "
                    f"{perf_10.get('max_gain', 'N/A')}% | "
                    f"{perf_10.get('max_loss', 'N/A')}% | "
                    f"{perf_10.get('final_return', 'N/A')}% |\n"
                )
            
            # 统计分析
            valid_5d = [s['future_performance']['5_days']['final_return'] 
                       for s in similar_stocks 
                       if s['future_performance'].get('5_days', {}).get('final_return') is not None]
            
            valid_10d = [s['future_performance']['10_days']['final_return'] 
                        for s in similar_stocks 
                        if s['future_performance'].get('10_days', {}).get('final_return') is not None]
            
            if valid_5d:
                avg_5d = np.mean(valid_5d)
                positive_5d = sum(1 for x in valid_5d if x > 0)
                report_content += f"""
### 📊 5日走势统计
- **平均收益率**: {avg_5d:.2f}%
- **上涨概率**: {positive_5d/len(valid_5d)*100:.1f}% ({positive_5d}/{len(valid_5d)})
- **最大收益**: {max(valid_5d):.2f}%
- **最大亏损**: {min(valid_5d):.2f}%
"""
            
            if valid_10d:
                avg_10d = np.mean(valid_10d)
                positive_10d = sum(1 for x in valid_10d if x > 0)
                report_content += f"""
### 📊 10日走势统计
- **平均收益率**: {avg_10d:.2f}%
- **上涨概率**: {positive_10d/len(valid_10d)*100:.1f}% ({positive_10d}/{len(valid_10d)})
- **最大收益**: {max(valid_10d):.2f}%
- **最大亏损**: {min(valid_10d):.2f}%
"""
    
    report_content += f"""
---

**生成时间**: {current_time}  
**分析引擎**: StockTradebyZ Enhanced System v2.0  
"""
    
    # 保存报告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 生成Claude分析prompt
    prompt_file = report_path / f'{target_code}_claude_prompt_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    
    claude_prompt = generate_claude_prompt(target_code, target_fundamentals, all_results)
    
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(claude_prompt)
    
    print("\n" + "="*80)
    print(f"🎉 {target_code} 增强型分析完成！")
    print("="*80)
    print(f"⏱️  总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"📄 分析报告: {report_file}")
    print(f"🤖 Claude Prompt: {prompt_file}")

def generate_claude_prompt(target_code, fundamentals, similarity_results):
    """生成用于Claude分析的prompt"""
    
    # 收集相似股票的统计数据
    all_5d_returns = []
    all_10d_returns = []
    
    for window_name, stocks in similarity_results.items():
        for stock in stocks[:10]:  # 只取前10个最相似的
            if stock['future_performance'].get('5_days', {}).get('final_return') is not None:
                all_5d_returns.append(stock['future_performance']['5_days']['final_return'])
            if stock['future_performance'].get('10_days', {}).get('final_return') is not None:
                all_10d_returns.append(stock['future_performance']['10_days']['final_return'])
    
    prompt = f"""请分析以下股票是否值得买入：

## 股票基本信息
- 代码: {target_code}
- 名称: {fundamentals['basic'].get('name', 'N/A')}
- 行业: {fundamentals['basic'].get('industry', 'N/A')}
- 地区: {fundamentals['basic'].get('area', 'N/A')}

## 财务指标
- 市盈率(TTM): {fundamentals['financial'].get('pe_ttm', 'N/A')}
- 市净率: {fundamentals['financial'].get('pb', 'N/A')}
- 市销率(TTM): {fundamentals['financial'].get('ps_ttm', 'N/A')}
- 总市值: {fundamentals['financial'].get('total_mv', 'N/A')} 万元
- 流通市值: {fundamentals['financial'].get('circ_mv', 'N/A')} 万元
- 换手率: {fundamentals['financial'].get('turnover_rate', 'N/A')}%

## 技术指标
- 当前价格: {fundamentals['price'].get('close', 'N/A')}
- 涨跌幅: {fundamentals['price'].get('change_pct', 'N/A')}%
- MA5: {fundamentals['technical'].get('ma_5', 'N/A')}
- MA20: {fundamentals['technical'].get('ma_20', 'N/A')}
- MA60: {fundamentals['technical'].get('ma_60', 'N/A')}
- RSI(6): {fundamentals['technical'].get('rsi_6', 'N/A')}
- RSI(12): {fundamentals['technical'].get('rsi_12', 'N/A')}
- KDJ_K: {fundamentals['technical'].get('kdj_k', 'N/A')}
- KDJ_D: {fundamentals['technical'].get('kdj_d', 'N/A')}
- KDJ_J: {fundamentals['technical'].get('kdj_j', 'N/A')}
- MACD: {fundamentals['technical'].get('macd', 'N/A')}
- Signal: {fundamentals['technical'].get('signal', 'N/A')}
- BBI: {fundamentals['technical'].get('bbi', 'N/A')}

## 历史相似走势股票表现
基于历史数据，找到了与该股票走势高度相似的股票，这些相似股票在相似走势后的表现如下：

### 5日后续表现统计
- 平均收益率: {np.mean(all_5d_returns) if all_5d_returns else 0:.2f}%
- 上涨概率: {sum(1 for x in all_5d_returns if x > 0)/len(all_5d_returns)*100 if all_5d_returns else 0:.1f}%
- 最大收益: {max(all_5d_returns) if all_5d_returns else 0:.2f}%
- 最大亏损: {min(all_5d_returns) if all_5d_returns else 0:.2f}%
- 样本数量: {len(all_5d_returns)}

### 10日后续表现统计
- 平均收益率: {np.mean(all_10d_returns) if all_10d_returns else 0:.2f}%
- 上涨概率: {sum(1 for x in all_10d_returns if x > 0)/len(all_10d_returns)*100 if all_10d_returns else 0:.1f}%
- 最大收益: {max(all_10d_returns) if all_10d_returns else 0:.2f}%
- 最大亏损: {min(all_10d_returns) if all_10d_returns else 0:.2f}%
- 样本数量: {len(all_10d_returns)}

## 分析要求
请基于以上数据，分析：
1. 该股票当前的投资价值如何？
2. 是否建议买入？给出明确的建议（强烈推荐/推荐/观望/不推荐）
3. 如果建议买入：
   - 建议的买入价格区间
   - 目标价位（基于技术面和历史相似走势）
   - 止损价位（风险控制）
   - 建议持有周期
4. 主要风险提示
5. 综合评分（1-10分）

请给出专业、客观的分析建议。
"""
    
    return prompt

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='增强型股票相似度分析工具')
    parser.add_argument('--code', default='002215', help='目标股票代码')
    parser.add_argument('--window', type=int, default=15, help='分析窗口天数')
    parser.add_argument('--start-date', default='2020-01-01', help='开始日期')
    parser.add_argument('--end-date', default='2025-08-11', help='结束日期')
    parser.add_argument('--threshold', type=float, default=0.12, help='相似度阈值')
    parser.add_argument('--candidates', type=int, default=None, help='候选股票数量')
    parser.add_argument('--all-stocks', action='store_true', help='扫描全A股所有股票')
    parser.add_argument('--single-period', action='store_true', help='只分析最近期')
    parser.add_argument('--processes', type=int, default=None, help='并行进程数')
    
    args = parser.parse_args()
    
    # 处理参数
    if args.all_stocks:
        candidate_limit = None
    else:
        candidate_limit = args.candidates if args.candidates is not None else 500
    
    multi_period = not args.single_period
    
    print("🚀 启动增强型股票相似度分析工具")
    print(f"参数: 股票={args.code}, 窗口={args.window}天, 阈值={args.threshold}")
    print(f"扫描: {'全A股' if candidate_limit is None else f'{candidate_limit}只'}, {'多时期' if multi_period else '单时期'}")
    
    run_enhanced_similarity_analysis(
        target_code=args.code,
        window_length=args.window,
        start_date=args.start_date,
        end_date=args.end_date,
        similarity_threshold=args.threshold,
        candidate_limit=candidate_limit,
        multi_period=multi_period,
        n_processes=args.processes
    )