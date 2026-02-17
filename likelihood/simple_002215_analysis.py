#!/usr/bin/env python3
"""
002215简化相似度分析 - 直接使用数据库查询
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))

from data_adapter.database_manager import DatabaseManager
from algorithms.matrix_profile import MatrixProfileSimilarity
from algorithms.dtw_similarity import DTWSimilarity
from algorithms.mass_similarity import MASSimilarity


class SimpleStock002215Analyzer:
    def __init__(self):
        """初始化分析器"""
        self.db = DatabaseManager()
        
        # 初始化算法
        self.mp_algo = MatrixProfileSimilarity({'window_length': 10})
        self.dtw_algo = DTWSimilarity({'window_type': 'sakoe_chiba', 'sakoe_chiba_radius': 5})
        self.mass_algo = MASSimilarity()
        
    def get_stock_data(self, stock_code, start_date='2025-01-01', end_date='2025-08-08'):
        """获取股票数据"""
        query = '''
        SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = ? AND dq.trade_date BETWEEN ? AND ?
        ORDER BY dq.trade_date
        '''
        
        result = self.db.execute_query(query, (stock_code, start_date, end_date))
        if not result:
            return None
            
        df = pd.DataFrame(result, columns=['trade_date', 'open', 'high', 'low', 'close', 'volume', 'price_change_pct'])
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        return df
    
    def get_candidate_stocks(self, limit=100):
        """获取候选股票池"""
        query = '''
        SELECT DISTINCT s.code, s.name, s.industry
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股' 
          AND dq.trade_date = '2025-08-08'
          AND dq.volume > 1000000  -- 成交量大于100万股
          AND dq.close > 1  -- 价格大于1元
        ORDER BY dq.volume * dq.close DESC
        LIMIT ?
        '''
        
        result = self.db.execute_query(query, (limit,))
        if not result:
            return []
            
        return [(row[0], row[1], row[2]) for row in result]
    
    def compute_similarity_score(self, query_series, candidate_series):
        """计算综合相似度"""
        try:
            # 计算价格相似度
            mp_sim = self.mp_algo.compute_similarity(query_series, candidate_series)
            dtw_sim = self.dtw_algo.compute_similarity(query_series, candidate_series) 
            mass_sim = self.mass_algo.compute_similarity(query_series, candidate_series)
            
            # 加权平均
            price_sim = mp_sim * 0.4 + dtw_sim * 0.3 + mass_sim * 0.3
            
            return {
                'overall': price_sim,
                'mp': mp_sim,
                'dtw': dtw_sim,
                'mass': mass_sim
            }
        except Exception as e:
            print(f"计算相似度失败: {e}")
            return None
    
    def find_similar_stocks(self, target_code='002215', window_length=15, top_k=10):
        """寻找相似股票"""
        print(f"开始分析 {target_code} 的相似股票...")
        
        # 获取目标股票数据
        target_data = self.get_stock_data(target_code)
        if target_data is None or len(target_data) < window_length:
            print(f"目标股票 {target_code} 数据不足")
            return []
        
        # 提取查询序列（最近15天的收益率）
        query_returns = target_data['close'].pct_change().fillna(0).tail(window_length).values
        query_volumes = target_data['volume'].pct_change().fillna(0).tail(window_length).values
        
        print(f"查询序列长度: {len(query_returns)}")
        
        # 获取候选股票
        candidates = self.get_candidate_stocks(200)
        print(f"候选股票数量: {len(candidates)}")
        
        similar_stocks = []
        
        for i, (code, name, industry) in enumerate(candidates):
            if code == target_code:
                continue
                
            try:
                # 获取候选股票数据
                candidate_data = self.get_stock_data(code, '2025-01-01', '2025-08-08')
                if candidate_data is None or len(candidate_data) < window_length:
                    continue
                
                # 在候选股票历史数据中找最佳匹配
                best_similarity = 0
                best_period = None
                
                candidate_returns = candidate_data['close'].pct_change().fillna(0).values
                candidate_volumes = candidate_data['volume'].pct_change().fillna(0).values
                
                # 滑动窗口寻找最佳匹配
                for j in range(len(candidate_returns) - window_length + 1):
                    window_returns = candidate_returns[j:j+window_length]
                    window_volumes = candidate_volumes[j:j+window_length]
                    
                    # 计算价格相似度
                    price_sim = self.compute_similarity_score(query_returns, window_returns)
                    if price_sim is None:
                        continue
                    
                    # 计算成交量相似度
                    volume_sim = self.compute_similarity_score(query_volumes, window_volumes)
                    if volume_sim is None:
                        continue
                    
                    # 综合相似度（价格权重0.8，成交量权重0.2）
                    overall_sim = price_sim['overall'] * 0.8 + volume_sim['overall'] * 0.2
                    
                    if overall_sim > best_similarity:
                        best_similarity = overall_sim
                        period_start = candidate_data.index[j]
                        period_end = candidate_data.index[j + window_length - 1]
                        best_period = {
                            'start': period_start,
                            'end': period_end,
                            'price_sim': price_sim,
                            'volume_sim': volume_sim
                        }
                
                # 如果相似度足够高，加入结果
                if best_similarity > 0.1 and best_period:  # 设置一个较低的阈值
                    similar_stocks.append({
                        'code': code,
                        'name': name,
                        'industry': industry,
                        'similarity_score': best_similarity,
                        'best_period': best_period
                    })
                
                if (i + 1) % 20 == 0:
                    print(f"已处理 {i + 1}/{len(candidates)} 只股票，找到 {len(similar_stocks)} 只相似股票")
                    
            except Exception as e:
                print(f"处理 {code} 时出错: {e}")
                continue
        
        # 按相似度排序
        similar_stocks.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_stocks[:top_k]
    
    def analyze_subsequent_performance(self, similar_stocks, forecast_days=20):
        """分析后续表现"""
        results = []
        
        for stock in similar_stocks:
            try:
                code = stock['code']
                end_period = stock['best_period']['end']
                
                # 获取匹配期结束后的数据
                extended_data = self.get_stock_data(code, '2025-01-01', '2025-08-08')
                if extended_data is None:
                    continue
                
                # 找到匹配期结束后的数据
                future_data = extended_data[extended_data.index > end_period]
                
                if len(future_data) < 5:  # 至少需要5天数据
                    continue
                    
                # 取预测期间的数据
                analysis_data = future_data.head(min(forecast_days, len(future_data)))
                
                # 计算后续表现
                start_price = analysis_data['close'].iloc[0]
                end_price = analysis_data['close'].iloc[-1]
                max_price = analysis_data['close'].max()
                min_price = analysis_data['close'].min()
                
                total_return = (end_price - start_price) / start_price
                max_gain = (max_price - start_price) / start_price
                max_loss = (min_price - start_price) / start_price
                
                # 计算波动率
                returns = analysis_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                
                results.append({
                    'code': code,
                    'name': stock['name'],
                    'industry': stock['industry'],
                    'similarity_score': stock['similarity_score'],
                    'match_period': f"{stock['best_period']['start'].strftime('%Y-%m-%d')} 至 {stock['best_period']['end'].strftime('%Y-%m-%d')}",
                    'analysis_days': len(analysis_data),
                    'total_return': total_return,
                    'max_gain': max_gain,
                    'max_loss': max_loss,
                    'volatility': volatility,
                    'start_price': start_price,
                    'end_price': end_price,
                    'price_similarity': stock['best_period']['price_sim'],
                    'volume_similarity': stock['best_period']['volume_sim']
                })
                
            except Exception as e:
                print(f"分析 {stock['code']} 后续表现时出错: {e}")
                continue
        
        return results
    
    def generate_report(self):
        """生成分析报告"""
        print("\n" + "="*60)
        print("002215 (诺普信) 股票相似度分析报告")
        print("="*60)
        
        # 获取目标股票信息
        target_info_query = '''
        SELECT code, name, industry, area
        FROM securities 
        WHERE code = '002215'
        '''
        
        target_result = self.db.execute_query(target_info_query)
        if target_result:
            target_info = target_result[0]
            print(f"\n目标股票: {target_info[0]} ({target_info[1]})")
            print(f"行业: {target_info[2]} | 地区: {target_info[3]}")
        
        # 查找相似股票
        similar_stocks = self.find_similar_stocks()
        
        if not similar_stocks:
            print("\n❌ 未找到相似股票")
            return None
        
        print(f"\n✅ 找到 {len(similar_stocks)} 只相似股票")
        print("\n相似度排行:")
        print("-" * 80)
        print(f"{'排名':<4} {'代码':<8} {'名称':<12} {'行业':<12} {'相似度':<8} {'匹配期间':<20}")
        print("-" * 80)
        
        for i, stock in enumerate(similar_stocks, 1):
            period_str = f"{stock['best_period']['start'].strftime('%m-%d')} 至 {stock['best_period']['end'].strftime('%m-%d')}"
            print(f"{i:<4} {stock['code']:<8} {stock['name'][:10]:<12} {stock['industry'][:10]:<12} {stock['similarity_score']:.4f}    {period_str:<20}")
        
        # 分析后续表现
        print(f"\n📈 后续走势分析...")
        performance_results = self.analyze_subsequent_performance(similar_stocks)
        
        if performance_results:
            print(f"\n后续表现统计:")
            print("-" * 100)
            print(f"{'代码':<8} {'名称':<10} {'相似度':<8} {'分析天数':<8} {'总收益率':<10} {'最大涨幅':<10} {'最大跌幅':<10}")
            print("-" * 100)
            
            total_returns = []
            for result in performance_results:
                print(f"{result['code']:<8} {result['name'][:8]:<10} {result['similarity_score']:.4f}    {result['analysis_days']:<8} "
                      f"{result['total_return']:>8.2%} {result['max_gain']:>8.2%} {result['max_loss']:>8.2%}")
                total_returns.append(result['total_return'])
            
            # 统计摘要
            if total_returns:
                avg_return = np.mean(total_returns)
                median_return = np.median(total_returns)
                positive_count = sum(1 for r in total_returns if r > 0)
                win_rate = positive_count / len(total_returns)
                
                print(f"\n📊 统计摘要:")
                print(f"平均收益率: {avg_return:.2%}")
                print(f"中位数收益率: {median_return:.2%}")
                print(f"胜率: {win_rate:.1%} ({positive_count}/{len(total_returns)})")
        
        # 保存详细报告
        report_path = Path(__file__).parent.parent / 'reports' / 'similarity_analysis'
        report_path.mkdir(parents=True, exist_ok=True)
        report_file = report_path / f'002215_similarity_simple_{datetime.now().strftime("%Y%m%d_%H%M")}.md'
        
        self._save_detailed_report(similar_stocks, performance_results, report_file)
        
        print(f"\n💾 详细报告已保存: {report_file}")
        
        return report_file
    
    def _save_detailed_report(self, similar_stocks, performance_results, output_path):
        """保存详细报告"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_content = f"""# 002215 (诺普信) 股票相似度分析报告

**生成时间**: {current_time}
**分析方法**: Matrix Profile + DTW + MASS 多算法融合
**分析窗口**: 15个交易日
**预测期间**: 20个交易日

## 目标股票信息

- **股票代码**: 002215
- **股票名称**: 诺普信  
- **所属行业**: 农药化肥
- **分析期间**: 最近15个交易日

## 相似股票发现结果

共发现 {len(similar_stocks)} 只相似股票：

| 排名 | 代码 | 名称 | 行业 | 相似度 | 匹配期间 |
|------|------|------|------|--------|----------|
"""
        
        for i, stock in enumerate(similar_stocks, 1):
            period_str = f"{stock['best_period']['start'].strftime('%Y-%m-%d')} 至 {stock['best_period']['end'].strftime('%Y-%m-%d')}"
            report_content += f"| {i} | {stock['code']} | {stock['name']} | {stock['industry']} | {stock['similarity_score']:.4f} | {period_str} |\n"
        
        if performance_results:
            report_content += f"""

## 后续走势表现分析

| 代码 | 名称 | 相似度 | 分析天数 | 总收益率 | 最大涨幅 | 最大跌幅 | 波动率 |
|------|------|--------|----------|----------|----------|----------|---------|
"""
            
            for result in performance_results:
                report_content += (f"| {result['code']} | {result['name']} | {result['similarity_score']:.4f} | "
                                 f"{result['analysis_days']} | {result['total_return']:.2%} | "
                                 f"{result['max_gain']:.2%} | {result['max_loss']:.2%} | {result['volatility']:.2%} |\n")
            
            # 添加统计摘要
            returns = [r['total_return'] for r in performance_results]
            if returns:
                avg_return = np.mean(returns)
                median_return = np.median(returns)
                positive_count = sum(1 for r in returns if r > 0)
                win_rate = positive_count / len(returns)
                
                report_content += f"""

### 统计摘要

- **平均收益率**: {avg_return:.2%}
- **中位数收益率**: {median_return:.2%}  
- **胜率**: {win_rate:.1%} ({positive_count}/{len(returns)})
- **样本数量**: {len(performance_results)} 只股票
"""
        
        report_content += f"""

## 算法说明

本分析使用了三种时间序列相似度算法：

1. **Matrix Profile**: 时间序列模式发现算法，权重40%
2. **Dynamic Time Warping (DTW)**: 处理时间扭曲的相似度算法，权重30%  
3. **MASS**: 高效的相似子序列搜索算法，权重30%

价格相似度权重80%，成交量相似度权重20%。

## 风险提示

1. 历史相似性不代表未来表现
2. 市场环境变化可能影响相似模式的有效性
3. 本报告仅供参考，不构成投资建议

---
*报告生成时间: {current_time}*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)


def main():
    """主函数"""
    analyzer = SimpleStock002215Analyzer()
    analyzer.generate_report()


if __name__ == '__main__':
    main()