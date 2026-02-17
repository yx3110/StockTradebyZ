#!/usr/bin/env python3
"""
002215 长期相似度分析 (2020-2025)
使用30天窗口，在5年时间范围内寻找相似走势
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 添加路径
sys.path.append('/Users/yangxu/StockTradebyZ')
sys.path.append('/Users/yangxu/StockTradebyZ/likelihood')

from data_adapter.database_manager import DatabaseManager
from algorithms.matrix_profile import MatrixProfileSimilarity
from algorithms.dtw_similarity import DTWSimilarity
from algorithms.mass_similarity import MASSimilarity


class ExtendedStock002215Analyzer:
    """002215长期相似度分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.db = DatabaseManager()
        
        # 设置参数
        self.window_length = 30  # 30天窗口
        self.start_date = '2020-01-01'
        self.end_date = '2025-08-08'
        self.min_data_threshold = 1200  # 至少1200个交易日
        
        # 初始化算法
        self.mp_algo = MatrixProfileSimilarity({'window_length': 15})  # MP窗口稍小
        self.dtw_algo = DTWSimilarity({'window_type': 'sakoe_chiba', 'sakoe_chiba_radius': 8})
        self.mass_algo = MASSimilarity()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def get_stock_data_long_term(self, stock_code):
        """获取股票长期数据"""
        query = '''
        SELECT dq.trade_date, dq.close, dq.volume, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? 
          AND dq.trade_date >= ?
          AND dq.trade_date <= ?
        ORDER BY dq.trade_date
        '''
        
        result = self.db.execute_query(query, (stock_code, self.start_date, self.end_date))
        if not result:
            return None
        
        df = pd.DataFrame(result, columns=['trade_date', 'close', 'volume', 'price_change_pct'])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        return df
    
    def get_candidate_stocks_extended(self, limit=150):
        """获取数据完整的候选股票（扩展版）"""
        query = '''
        SELECT s.code, s.name, s.industry, COUNT(*) as data_count
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股'
          AND dq.trade_date >= ?
          AND dq.trade_date <= ?
          AND s.code != '002215'
        GROUP BY s.code, s.name, s.industry
        HAVING COUNT(*) >= ?
        ORDER BY COUNT(*) DESC
        LIMIT ?
        '''
        
        result = self.db.execute_query(query, (self.start_date, self.end_date, 
                                             self.min_data_threshold, limit))
        return [(row[0], row[1], row[2], row[3]) for row in result] if result else []
    
    def compute_enhanced_similarity(self, query_series, candidate_series):
        """计算增强相似度"""
        try:
            # 确保序列长度一致
            if len(query_series) != len(candidate_series):
                return None
            
            # 计算各种相似度
            mp_sim = self.mp_algo.compute_similarity(query_series, candidate_series)
            dtw_sim = self.dtw_algo.compute_similarity(query_series, candidate_series)
            mass_sim = self.mass_algo.compute_similarity(query_series, candidate_series)
            
            # 计算统计相似度指标
            correlation = np.corrcoef(query_series, candidate_series)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            # 计算方向一致性（涨跌方向匹配度）
            query_directions = np.sign(query_series)
            candidate_directions = np.sign(candidate_series)
            direction_match = np.mean(query_directions == candidate_directions)
            
            # 加权综合相似度
            # MP权重35%、DTW权重25%、MASS权重20%、相关性15%、方向一致性5%
            overall_sim = (mp_sim * 0.35 + 
                          dtw_sim * 0.25 + 
                          mass_sim * 0.20 + 
                          abs(correlation) * 0.15 + 
                          direction_match * 0.05)
            
            return {
                'overall': overall_sim,
                'mp': mp_sim,
                'dtw': dtw_sim,
                'mass': mass_sim,
                'correlation': correlation,
                'direction_match': direction_match
            }
            
        except Exception as e:
            self.logger.error(f"计算相似度失败: {str(e)}")
            return None
    
    def find_similar_patterns_long_term(self, target_code='002215'):
        """在长期时间范围内寻找相似模式"""
        self.logger.info(f"开始长期相似度分析: {target_code}")
        print("="*80)
        print("002215 (诺普信) 长期相似度分析 (2020-2025)")
        print(f"分析窗口: {self.window_length} 天")
        print("="*80)
        
        # 1. 获取目标股票数据
        print(f"\\n1. 获取 {target_code} 长期数据...")
        target_data = self.get_stock_data_long_term(target_code)
        
        if target_data is None or len(target_data) < self.window_length * 2:
            self.logger.error(f"目标股票数据不足: {len(target_data) if target_data is not None else 0}")
            return []
        
        print(f"✅ 获取到 {len(target_data)} 条数据 ({target_data.index.min().date()} 至 {target_data.index.max().date()})")
        
        # 2. 计算目标股票的收益率序列
        target_returns = target_data['close'].pct_change().fillna(0).values
        
        # 3. 定义多个查询窗口（不同时期的30天窗口）
        query_windows = []
        
        # 选择不同时期的代表性窗口
        periods = [
            ('2024年末', len(target_returns) - self.window_length, len(target_returns)),  # 最近30天
            ('2024年中', len(target_returns) - 120, len(target_returns) - 90),  # 4个月前
            ('2023年末', len(target_returns) - 250, len(target_returns) - 220),  # 约1年前
            ('2023年中', len(target_returns) - 400, len(target_returns) - 370),  # 约1.5年前
            ('2022年末', len(target_returns) - 550, len(target_returns) - 520),  # 约2年前
            ('2021年末', len(target_returns) - 800, len(target_returns) - 770),  # 约3年前
        ]
        
        for period_name, start_idx, end_idx in periods:
            if start_idx >= 0 and end_idx <= len(target_returns):
                query_window = target_returns[start_idx:end_idx]
                if len(query_window) == self.window_length:
                    query_windows.append({
                        'name': period_name,
                        'data': query_window,
                        'start_date': target_data.index[start_idx],
                        'end_date': target_data.index[end_idx-1]
                    })
        
        print(f"\\n2. 定义 {len(query_windows)} 个查询窗口:")
        for window in query_windows:
            print(f"   - {window['name']}: {window['start_date'].date()} 至 {window['end_date'].date()}")
        
        # 4. 获取候选股票
        print(f"\\n3. 获取候选股票池...")
        candidates = self.get_candidate_stocks_extended(150)
        print(f"✅ 候选股票数量: {len(candidates)}")
        
        if not candidates:
            print("❌ 没有符合条件的候选股票")
            return []
        
        print(f"   数据量前10的候选股票:")
        for i, (code, name, industry, count) in enumerate(candidates[:10]):
            print(f"     {code} ({name[:8]:<8}) {industry[:8]:<8} - {count}天")
        
        # 5. 并行分析所有查询窗口
        all_similar_stocks = {}
        
        for window_info in query_windows:
            window_name = window_info['name']
            query_data = window_info['data']
            
            print(f"\\n4. 分析{window_name}期间的相似股票...")
            
            similar_stocks = []
            processed_count = 0
            
            for code, name, industry, data_count in candidates:
                try:
                    # 获取候选股票数据
                    candidate_data = self.get_stock_data_long_term(code)
                    if candidate_data is None or len(candidate_data) < self.min_data_threshold:
                        continue
                    
                    candidate_returns = candidate_data['close'].pct_change().fillna(0).values
                    
                    # 在候选股票的历史数据中滑动窗口寻找最佳匹配
                    best_similarity = 0
                    best_period = None
                    
                    # 滑动窗口搜索（每10个交易日采样一次以提高效率）
                    step_size = 5  # 步长为5天，在精度和效率间平衡
                    
                    for j in range(0, len(candidate_returns) - self.window_length + 1, step_size):
                        candidate_window = candidate_returns[j:j + self.window_length]
                        
                        if len(candidate_window) != self.window_length:
                            continue
                        
                        # 计算相似度
                        sim_result = self.compute_enhanced_similarity(query_data, candidate_window)
                        if sim_result is None:
                            continue
                        
                        if sim_result['overall'] > best_similarity:
                            best_similarity = sim_result['overall']
                            best_period = {
                                'start': candidate_data.index[j],
                                'end': candidate_data.index[j + self.window_length - 1],
                                'similarity_details': sim_result
                            }
                    
                    # 如果相似度足够高，加入结果
                    if best_similarity > 0.25 and best_period:  # 提高阈值确保质量
                        similar_stocks.append({
                            'code': code,
                            'name': name,
                            'industry': industry,
                            'data_count': data_count,
                            'similarity_score': best_similarity,
                            'best_period': best_period,
                            'query_window': window_name
                        })
                    
                    processed_count += 1
                    if processed_count % 30 == 0:
                        print(f"     已处理 {processed_count}/{len(candidates)} 只股票，找到 {len(similar_stocks)} 只相似股票")
                
                except Exception as e:
                    self.logger.error(f"处理股票 {code} 时出错: {str(e)}")
                    continue
            
            # 按相似度排序
            similar_stocks.sort(key=lambda x: x['similarity_score'], reverse=True)
            all_similar_stocks[window_name] = similar_stocks[:20]  # 每个窗口保留前20只
            
            print(f"   {window_name}期间找到 {len(similar_stocks)} 只相似股票")
        
        return all_similar_stocks
    
    def analyze_long_term_performance(self, all_similar_stocks):
        """分析长期表现"""
        print(f"\\n📈 长期表现分析...")
        
        performance_summary = {}
        
        for window_name, similar_stocks in all_similar_stocks.items():
            print(f"\\n--- {window_name} 期间相似股票后续表现 ---")
            
            performance_results = []
            
            for stock in similar_stocks[:10]:  # 分析每个窗口前10只
                try:
                    code = stock['code']
                    match_end_date = stock['best_period']['end']
                    
                    # 获取完整数据
                    full_data = self.get_stock_data_long_term(code)
                    if full_data is None:
                        continue
                    
                    # 分析匹配期结束后的表现
                    future_data = full_data[full_data.index > match_end_date]
                    
                    if len(future_data) < 10:
                        continue
                    
                    # 分析不同时间窗口的表现：10天、30天、60天
                    for days in [10, 30, 60]:
                        if len(future_data) >= days:
                            analysis_data = future_data.head(days)
                            
                            start_price = analysis_data['close'].iloc[0]
                            end_price = analysis_data['close'].iloc[-1]
                            max_price = analysis_data['close'].max()
                            min_price = analysis_data['close'].min()
                            
                            total_return = (end_price - start_price) / start_price
                            max_gain = (max_price - start_price) / start_price
                            max_loss = (min_price - start_price) / start_price
                            
                            # 计算夏普比率（简化版）
                            returns = analysis_data['close'].pct_change().dropna()
                            if len(returns) > 1:
                                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
                            else:
                                sharpe = 0
                            
                            performance_results.append({
                                'code': code,
                                'name': stock['name'],
                                'industry': stock['industry'],
                                'similarity_score': stock['similarity_score'],
                                'match_period': f"{stock['best_period']['start'].strftime('%Y-%m-%d')} 至 {stock['best_period']['end'].strftime('%Y-%m-%d')}",
                                'analysis_days': days,
                                'total_return': total_return,
                                'max_gain': max_gain,
                                'max_loss': max_loss,
                                'sharpe_ratio': sharpe,
                                'similarity_details': stock['best_period']['similarity_details']
                            })
                
                except Exception as e:
                    self.logger.error(f"分析 {stock['code']} 长期表现时出错: {str(e)}")
                    continue
            
            performance_summary[window_name] = performance_results
            
            # 显示该窗口的表现摘要
            if performance_results:
                returns_30d = [r['total_return'] for r in performance_results if r['analysis_days'] == 30]
                if returns_30d:
                    avg_return = np.mean(returns_30d)
                    win_rate = sum(1 for r in returns_30d if r > 0) / len(returns_30d)
                    print(f"   30天表现: 平均收益 {avg_return:.2%}, 胜率 {win_rate:.1%}")
        
        return performance_summary
    
    def generate_extended_report(self, all_similar_stocks, performance_summary):
        """生成扩展报告"""
        print(f"\\n💾 生成长期相似度分析报告...")
        
        # 创建报告目录
        report_path = Path('/Users/yangxu/StockTradebyZ/reports/similarity_analysis')
        report_path.mkdir(parents=True, exist_ok=True)
        report_file = report_path / f'002215_extended_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.md'
        
        # 生成报告内容
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_content = f"""# 002215 (诺普信) 长期相似度分析报告 (2020-2025)

**生成时间**: {current_time}  
**分析工具**: 多算法融合相似度分析 (Matrix Profile + DTW + MASS + 统计指标)  
**分析窗口**: {self.window_length} 个交易日  
**时间范围**: {self.start_date} 至 {self.end_date}  
**分析维度**: 多时期查询窗口 × 长期历史匹配  

---

## 🎯 分析概述

本报告采用先进的时间序列相似度算法，在2020-2025年的5年时间范围内，寻找与002215（诺普信）具有相似走势模式的股票。

### 分析特点

1. **长期视角**: 覆盖5年完整市场周期
2. **多时期查询**: 选择6个不同时期的30天窗口作为查询模式
3. **高精度算法**: Matrix Profile、DTW、MASS算法融合
4. **增强指标**: 加入相关性和方向一致性分析
5. **严格筛选**: 相似度阈值0.25，确保结果质量

---

## 📊 分析结果汇总

"""
        
        # 统计所有窗口的结果
        total_found = sum(len(stocks) for stocks in all_similar_stocks.values())
        unique_stocks = set()
        for stocks in all_similar_stocks.values():
            for stock in stocks:
                unique_stocks.add(stock['code'])
        
        report_content += f"""
### 总体发现

- **查询窗口数量**: {len(all_similar_stocks)} 个
- **总相似匹配**: {total_found} 次
- **独特相似股票**: {len(unique_stocks)} 只
- **平均每窗口匹配**: {total_found / len(all_similar_stocks):.1f} 只

"""
        
        # 各时期详细结果
        for window_name, similar_stocks in all_similar_stocks.items():
            if not similar_stocks:
                continue
                
            report_content += f"""
---

## 📅 {window_name} 期间相似股票分析

**相似股票数量**: {len(similar_stocks)} 只

### 相似度排行榜

| 排名 | 股票代码 | 股票名称 | 行业 | 相似度 | 匹配期间 | MP | DTW | MASS | 相关性 | 方向匹配 |
|------|----------|----------|------|--------|----------|----|----|------|--------|----------|
"""
            
            for i, stock in enumerate(similar_stocks[:15], 1):  # 显示前15只
                details = stock['best_period']['similarity_details']
                period = f"{stock['best_period']['start'].strftime('%Y-%m-%d')} 至 {stock['best_period']['end'].strftime('%Y-%m-%d')}"
                
                report_content += (f"| {i} | {stock['code']} | {stock['name']} | {stock['industry']} | "
                                  f"{stock['similarity_score']:.4f} | {period} | "
                                  f"{details['mp']:.3f} | {details['dtw']:.3f} | {details['mass']:.3f} | "
                                  f"{details['correlation']:.3f} | {details['direction_match']:.3f} |\\n")
            
            # 该时期的后续表现
            if window_name in performance_summary:
                perf_data = performance_summary[window_name]
                
                # 30天表现统计
                returns_30d = [r['total_return'] for r in perf_data if r['analysis_days'] == 30]
                returns_60d = [r['total_return'] for r in perf_data if r['analysis_days'] == 60]
                
                if returns_30d or returns_60d:
                    report_content += f"""

### 后续表现分析

#### 30天后续表现 ({len(returns_30d)} 样本)
"""
                    if returns_30d:
                        avg_30d = np.mean(returns_30d)
                        median_30d = np.median(returns_30d)
                        win_rate_30d = sum(1 for r in returns_30d if r > 0) / len(returns_30d)
                        max_30d = max(returns_30d)
                        min_30d = min(returns_30d)
                        
                        report_content += f"""
- **平均收益率**: {avg_30d:.2%}
- **中位数收益率**: {median_30d:.2%}
- **胜率**: {win_rate_30d:.1%}
- **最高收益**: {max_30d:.2%}
- **最低收益**: {min_30d:.2%}
"""
                    
                    if returns_60d:
                        avg_60d = np.mean(returns_60d)
                        win_rate_60d = sum(1 for r in returns_60d if r > 0) / len(returns_60d)
                        
                        report_content += f"""
#### 60天后续表现 ({len(returns_60d)} 样本)

- **平均收益率**: {avg_60d:.2%}
- **胜率**: {win_rate_60d:.1%}
"""
        
        # 跨时期分析
        report_content += f"""

---

## 🔍 跨时期分析

### 高频出现的相似股票

"""
        
        # 统计跨时期出现频率
        stock_frequency = {}
        stock_details = {}
        
        for window_name, similar_stocks in all_similar_stocks.items():
            for stock in similar_stocks:
                code = stock['code']
                if code not in stock_frequency:
                    stock_frequency[code] = 0
                    stock_details[code] = {
                        'name': stock['name'],
                        'industry': stock['industry'],
                        'windows': []
                    }
                stock_frequency[code] += 1
                stock_details[code]['windows'].append({
                    'window': window_name,
                    'similarity': stock['similarity_score']
                })
        
        # 按出现频率排序
        frequent_stocks = sorted(stock_frequency.items(), key=lambda x: x[1], reverse=True)
        
        report_content += f"""
| 股票代码 | 股票名称 | 行业 | 出现次数 | 平均相似度 | 出现时期 |
|----------|----------|------|----------|------------|----------|
"""
        
        for code, frequency in frequent_stocks[:20]:  # 显示前20只高频股票
            details = stock_details[code]
            avg_similarity = np.mean([w['similarity'] for w in details['windows']])
            windows_str = ', '.join([w['window'] for w in details['windows']])
            
            report_content += (f"| {code} | {details['name']} | {details['industry']} | "
                              f"{frequency} | {avg_similarity:.4f} | {windows_str} |\\n")
        
        # 方法说明
        report_content += f"""

---

## 🔬 分析方法详解

### 相似度算法组合

1. **Matrix Profile (35%权重)**
   - 时间序列模式发现的金标准算法
   - 高效识别重复和异常模式
   - 适合发现结构性相似

2. **Dynamic Time Warping (25%权重)**
   - 处理时间轴伸缩变形
   - 识别不同速度的相似走势
   - Sakoe-Chiba约束提高效率

3. **MASS Algorithm (20%权重)**
   - 快速相似子序列搜索
   - 基于标准化欧几里德距离
   - 高计算效率

4. **统计相关性 (15%权重)**
   - 皮尔逊相关系数
   - 衡量线性相关程度
   - 补充非线性算法

5. **方向一致性 (5%权重)**
   - 涨跌方向匹配度
   - 反映趋势同步性
   - 强化模式识别

### 数据质量控制

- **时间范围**: 2020-2025年完整市场周期
- **数据要求**: 至少1200个交易日
- **候选筛选**: 150只数据最完整的A股
- **滑动步长**: 5个交易日，平衡精度与效率
- **相似度阈值**: 0.25，确保高质量匹配

### 多时期查询策略

通过选择不同历史时期的30天窗口作为查询模式：
- 捕获不同市场环境下的相似性
- 识别稳定的长期相似关系
- 提高分析结果的稳健性

---

## ⚠️ 风险提示

1. **历史相似性局限**: 过往模式不能保证未来重现
2. **市场环境变化**: 宏观环境变化可能改变相似性有效性
3. **算法假设**: 基于价格走势相似，不考虑基本面差异
4. **时间窗口限制**: 30天窗口可能无法捕获更长期的相似性
5. **数据挖掘风险**: 大量比较可能产生偶然的虚假相似性

---

## 📈 投资启示

### 积极信号
- 多时期出现的相似股票可能具有结构性相似特征
- 高相似度匹配的后续表现可为参考依据
- 算法融合提高了识别准确性

### 注意事项
- 需结合基本面分析验证相似性合理性
- 关注相似股票的行业背景和商业模式
- 考虑市场环境变化对相似性的影响

---

**报告生成**: {current_time}  
**数据来源**: 股票历史交易数据  
**分析引擎**: StockTradebyZ Extended Likelihood System  
**算法版本**: v2.0 (长期多时期分析版)  
"""
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ 长期分析报告已保存: {report_file}")
        
        return report_file
    
    def run_complete_analysis(self):
        """运行完整的长期相似度分析"""
        try:
            # 1. 寻找相似模式
            all_similar_stocks = self.find_similar_patterns_long_term()
            
            if not all_similar_stocks:
                print("❌ 未找到相似股票")
                return None
            
            # 2. 分析长期表现
            performance_summary = self.analyze_long_term_performance(all_similar_stocks)
            
            # 3. 生成报告
            report_file = self.generate_extended_report(all_similar_stocks, performance_summary)
            
            # 4. 显示总结
            print(f"\\n" + "="*80)
            print("🎉 002215 长期相似度分析完成！")
            print("="*80)
            
            # 统计总览
            total_matches = sum(len(stocks) for stocks in all_similar_stocks.values())
            unique_stocks = set()
            for stocks in all_similar_stocks.values():
                for stock in stocks:
                    unique_stocks.add((stock['code'], stock['name']))
            
            print(f"📊 分析总览:")
            print(f"   时间范围: {self.start_date} 至 {self.end_date}")
            print(f"   查询窗口: {len(all_similar_stocks)} 个")
            print(f"   总匹配数: {total_matches} 次")
            print(f"   独特股票: {len(unique_stocks)} 只")
            
            print(f"\\n🏆 高频相似股票 (前10):")
            stock_frequency = {}
            for stocks in all_similar_stocks.values():
                for stock in stocks:
                    code = stock['code']
                    name = stock['name']
                    if (code, name) not in stock_frequency:
                        stock_frequency[(code, name)] = 0
                    stock_frequency[(code, name)] += 1
            
            frequent_stocks = sorted(stock_frequency.items(), key=lambda x: x[1], reverse=True)
            for i, ((code, name), freq) in enumerate(frequent_stocks[:10], 1):
                print(f"   {i}. {code} ({name[:8]}) - 出现{freq}次")
            
            print(f"\\n📄 详细报告: {report_file}")
            
            return report_file
            
        except Exception as e:
            self.logger.error(f"分析过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数"""
    print("启动002215长期相似度分析系统...")
    
    analyzer = ExtendedStock002215Analyzer()
    result = analyzer.run_complete_analysis()
    
    if result:
        print(f"\\n✅ 分析成功完成，报告已保存")
    else:
        print(f"\\n❌ 分析失败")


if __name__ == '__main__':
    main()