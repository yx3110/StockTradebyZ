#!/usr/bin/env python3
"""
002215（诺普信）相似股票分析器
使用多种时间序列相似度算法找出相似股票并分析后续走势
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import logging

# 添加路径
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from algorithms.search_engine import SimilaritySearchEngine
from data_preprocessing.data_loader import DataLoader
from algorithms.matrix_profile import MatrixProfileSimilarity
from algorithms.dtw_similarity import DTWSimilarity
from algorithms.mass_similarity import MASSimilarity


class Stock002215Analyzer:
    def __init__(self, config_path=None):
        """初始化分析器"""
        if config_path is None:
            config_path = Path(__file__).parent / 'configs' / 'default_config.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 调整配置以提高分析效果
        self.config['filters']['min_daily_volume'] = 5000000  # 降低到500万成交额
        self.config['similarity']['search']['top_k'] = 15  # 增加返回结果数
        self.config['similarity']['search']['min_similarity'] = 0.05  # 降低阈值
        
        self.data_loader = DataLoader(config=self.config)
        self.search_engine = SimilaritySearchEngine(self.config)
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_stock_info(self, stock_code):
        """获取股票基本信息"""
        from data_adapter.database_manager import DatabaseManager
        
        db = DatabaseManager()
        query = '''
        SELECT code, name, type, exchange, industry, area, list_date
        FROM securities
        WHERE code = ?
        '''
        
        result = db.execute_query(query, (stock_code,))
        if result:
            return {
                'code': result[0][0],
                'name': result[0][1],
                'type': result[0][2],
                'exchange': result[0][3],
                'industry': result[0][4],
                'area': result[0][5],
                'list_date': result[0][6]
            }
        return None
    
    def get_recent_data(self, stock_code, days=60):
        """获取股票最近的数据"""
        end_date = '2025-08-08'
        start_date = '2025-06-01'  # 约60个交易日
        
        try:
            data = self.data_loader.load_stock_data(stock_code, start_date, end_date)
            return data
        except Exception as e:
            self.logger.error(f"加载{stock_code}数据失败: {str(e)}")
            return None
    
    def find_similar_stocks(self, target_code='002215', analysis_period=30):
        """找出与目标股票相似的股票"""
        self.logger.info(f"开始分析股票 {target_code} 的相似股票...")
        
        # 获取目标股票数据
        target_data = self.get_recent_data(target_code, 60)
        if target_data is None or len(target_data) < analysis_period:
            self.logger.error(f"目标股票 {target_code} 数据不足")
            return []
        
        # 获取候选股票池
        candidate_stocks = self.data_loader.filter_stocks_by_criteria(
            min_volume=self.config['filters']['min_daily_volume'],
            date='2025-08-08'
        )
        
        self.logger.info(f"候选股票池大小: {len(candidate_stocks)}")
        
        if not candidate_stocks:
            # 如果没有候选股票，放宽条件
            self.logger.info("放宽筛选条件重新搜索...")
            candidate_stocks = self.data_loader.filter_stocks_by_criteria(
                min_volume=1000000,  # 进一步降低到100万
                date='2025-08-08'
            )
            self.logger.info(f"放宽后候选股票池大小: {len(candidate_stocks)}")
        
        # 提取查询序列（最近30天）
        query_data = target_data.tail(analysis_period)
        if len(query_data) < analysis_period:
            self.logger.warning(f"查询数据长度不足: {len(query_data)} < {analysis_period}")
            return []
        
        # 计算相似度
        similar_stocks = []
        
        # 准备算法
        mp_algo = MatrixProfileSimilarity({'window_length': min(15, analysis_period//2)})
        dtw_algo = DTWSimilarity({'window_type': 'sakoe_chiba', 'sakoe_chiba_radius': 5})
        mass_algo = MASSimilarity()
        
        # 提取查询序列
        query_price = query_data['close'].pct_change().fillna(0).values  # 收益率
        query_volume = query_data['volume'].pct_change().fillna(0).values  # 成交量变化率
        
        self.logger.info(f"查询序列长度 - 价格: {len(query_price)}, 成交量: {len(query_volume)}")
        
        # 遍历候选股票
        processed_count = 0
        for candidate_code in candidate_stocks[:200]:  # 限制处理数量
            if candidate_code == target_code:
                continue
                
            try:
                candidate_data = self.get_recent_data(candidate_code, 60)
                if candidate_data is None or len(candidate_data) < analysis_period + 10:
                    continue
                
                # 寻找最佳匹配期间
                best_similarity = 0
                best_period = None
                
                # 在候选股票的历史数据中滑动窗口寻找最佳匹配
                for i in range(len(candidate_data) - analysis_period + 1):
                    candidate_window = candidate_data.iloc[i:i+analysis_period]
                    
                    if len(candidate_window) != analysis_period:
                        continue
                    
                    # 提取候选序列
                    cand_price = candidate_window['close'].pct_change().fillna(0).values
                    cand_volume = candidate_window['volume'].pct_change().fillna(0).values
                    
                    if len(cand_price) != len(query_price):
                        continue
                    
                    # 计算多种相似度
                    similarities = []
                    
                    try:
                        # 价格相似度
                        mp_sim = mp_algo.compute_similarity(query_price, cand_price)
                        dtw_sim = dtw_algo.compute_similarity(query_price, cand_price)
                        mass_sim = mass_algo.compute_similarity(query_price, cand_price)
                        
                        # 成交量相似度
                        vol_mp_sim = mp_algo.compute_similarity(query_volume, cand_volume)
                        vol_dtw_sim = dtw_algo.compute_similarity(query_volume, cand_volume)
                        
                        # 综合相似度（价格权重0.7，成交量权重0.3）
                        price_sim = (mp_sim * 0.4 + dtw_sim * 0.3 + mass_sim * 0.3)
                        volume_sim = (vol_mp_sim * 0.5 + vol_dtw_sim * 0.5)
                        overall_sim = price_sim * 0.7 + volume_sim * 0.3
                        
                        if overall_sim > best_similarity:
                            best_similarity = overall_sim
                            best_period = {
                                'start_date': candidate_window.index[0],
                                'end_date': candidate_window.index[-1],
                                'price_similarity': price_sim,
                                'volume_similarity': volume_sim,
                                'mp_similarity': mp_sim,
                                'dtw_similarity': dtw_sim,
                                'mass_similarity': mass_sim
                            }
                    
                    except Exception as e:
                        continue
                
                # 如果找到了足够相似的股票
                if best_similarity > self.config['similarity']['search']['min_similarity'] and best_period:
                    stock_info = self.get_stock_info(candidate_code)
                    
                    similar_stocks.append({
                        'code': candidate_code,
                        'name': stock_info['name'] if stock_info else 'N/A',
                        'industry': stock_info['industry'] if stock_info else 'N/A',
                        'similarity_score': best_similarity,
                        'best_period': best_period,
                        'details': {
                            'price_similarity': best_period['price_similarity'],
                            'volume_similarity': best_period['volume_similarity'],
                            'mp_similarity': best_period['mp_similarity'],
                            'dtw_similarity': best_period['dtw_similarity'],
                            'mass_similarity': best_period['mass_similarity']
                        }
                    })
                
                processed_count += 1
                if processed_count % 50 == 0:
                    self.logger.info(f"已处理 {processed_count} 只股票，找到 {len(similar_stocks)} 只相似股票")
                    
            except Exception as e:
                self.logger.error(f"处理股票 {candidate_code} 时出错: {str(e)}")
                continue
        
        # 按相似度排序
        similar_stocks.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        self.logger.info(f"分析完成，共找到 {len(similar_stocks)} 只相似股票")
        return similar_stocks[:self.config['similarity']['search']['top_k']]
    
    def analyze_subsequent_performance(self, similar_stocks, target_code='002215', forecast_days=20):
        """分析相似股票的后续表现"""
        self.logger.info("开始分析后续表现...")
        
        results = []
        
        for stock in similar_stocks:
            try:
                # 获取匹配期间后的数据
                match_end_date = pd.to_datetime(stock['best_period']['end_date'])
                
                # 获取该股票的更长历史数据用于分析后续走势
                extended_data = self.get_recent_data(stock['code'], 90)
                if extended_data is None:
                    continue
                
                # 找到匹配结束日期在数据中的位置
                extended_data.index = pd.to_datetime(extended_data.index)
                
                # 找到最接近匹配结束日期的位置
                match_end_idx = None
                for i, date in enumerate(extended_data.index):
                    if date >= match_end_date:
                        match_end_idx = i
                        break
                
                if match_end_idx is None or match_end_idx >= len(extended_data) - 5:
                    continue
                
                # 分析后续走势
                subsequent_data = extended_data.iloc[match_end_idx:match_end_idx+min(forecast_days, len(extended_data)-match_end_idx)]
                
                if len(subsequent_data) < 5:  # 至少需要5天数据
                    continue
                
                # 计算后续表现指标
                start_price = subsequent_data['close'].iloc[0]
                end_price = subsequent_data['close'].iloc[-1]
                max_price = subsequent_data['close'].max()
                min_price = subsequent_data['close'].min()
                
                total_return = (end_price - start_price) / start_price
                max_gain = (max_price - start_price) / start_price
                max_loss = (min_price - start_price) / start_price
                
                # 计算波动率
                returns = subsequent_data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
                
                results.append({
                    'code': stock['code'],
                    'name': stock['name'],
                    'industry': stock['industry'],
                    'similarity_score': stock['similarity_score'],
                    'match_period': f"{stock['best_period']['start_date']} 至 {stock['best_period']['end_date']}",
                    'subsequent_days': len(subsequent_data),
                    'total_return': total_return,
                    'max_gain': max_gain,
                    'max_loss': max_loss,
                    'volatility': volatility,
                    'start_price': start_price,
                    'end_price': end_price,
                    'algorithm_details': stock['details']
                })
                
            except Exception as e:
                self.logger.error(f"分析股票 {stock['code']} 后续表现时出错: {str(e)}")
                continue
        
        return results
    
    def generate_report(self, target_code='002215', output_path=None):
        """生成完整的相似度分析报告"""
        if output_path is None:
            output_path = Path(__file__).parent.parent / 'reports' / 'similarity_analysis'
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / f'{target_code}_similarity_analysis_{datetime.now().strftime("%Y%m%d_%H%M")}.md'
        
        # 获取目标股票信息
        target_info = self.get_stock_info(target_code)
        
        # 找相似股票
        similar_stocks = self.find_similar_stocks(target_code)
        
        if not similar_stocks:
            self.logger.warning("未找到相似股票")
            return None
        
        # 分析后续表现
        performance_results = self.analyze_subsequent_performance(similar_stocks, target_code)
        
        # 生成报告
        report_content = self._generate_report_content(target_info, similar_stocks, performance_results)
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"报告已生成: {output_path}")
        return output_path
    
    def _generate_report_content(self, target_info, similar_stocks, performance_results):
        """生成报告内容"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# {target_info['code']} ({target_info['name']}) 股票相似度分析报告

**生成时间**: {current_time}
**分析工具**: 多维度时间序列相似度算法 (Matrix Profile + DTW + MASS)
**分析期间**: 最近30个交易日
**预测期间**: 后续20个交易日

---

## 🎯 目标股票基本信息

- **股票代码**: {target_info['code']}
- **股票名称**: {target_info['name']}
- **所属行业**: {target_info['industry']}
- **交易所**: {target_info['exchange']}
- **所在地区**: {target_info['area']}
- **上市日期**: {target_info['list_date']}

---

## 📊 相似股票发现结果

共发现 **{len(similar_stocks)}** 只与 {target_info['code']} 具有相似走势的股票：

### 相似度排行榜

| 排名 | 股票代码 | 股票名称 | 行业 | 综合相似度 | 价格相似度 | 成交量相似度 |
|------|----------|----------|------|------------|------------|--------------|
"""
        
        for i, stock in enumerate(similar_stocks, 1):
            report += f"| {i} | {stock['code']} | {stock['name']} | {stock['industry']} | {stock['similarity_score']:.4f} | {stock['details']['price_similarity']:.4f} | {stock['details']['volume_similarity']:.4f} |\n"
        
        report += f"""

### 算法详细分析

"""
        
        for i, stock in enumerate(similar_stocks, 1):
            details = stock['details']
            report += f"""
#### {i}. {stock['code']} ({stock['name']})

- **综合相似度**: {stock['similarity_score']:.4f}
- **Matrix Profile 相似度**: {details['mp_similarity']:.4f}
- **DTW 相似度**: {details['dtw_similarity']:.4f}  
- **MASS 相似度**: {details['mass_similarity']:.4f}
- **最佳匹配期间**: {stock['best_period']['start_date']} 至 {stock['best_period']['end_date']}
"""

        report += f"""

---

## 📈 后续走势表现分析

基于历史相似期间的后续走势分析：

### 业绩表现汇总

| 股票代码 | 股票名称 | 相似度 | 后续收益率 | 最大涨幅 | 最大跌幅 | 波动率 | 分析天数 |
|----------|----------|---------|------------|----------|----------|---------|----------|
"""
        
        for result in performance_results:
            report += f"| {result['code']} | {result['name']} | {result['similarity_score']:.4f} | {result['total_return']:.2%} | {result['max_gain']:.2%} | {result['max_loss']:.2%} | {result['volatility']:.2%} | {result['subsequent_days']} |\n"
        
        if performance_results:
            # 计算统计摘要
            returns = [r['total_return'] for r in performance_results]
            max_gains = [r['max_gain'] for r in performance_results]
            max_losses = [r['max_loss'] for r in performance_results]
            
            avg_return = np.mean(returns)
            median_return = np.median(returns)
            avg_max_gain = np.mean(max_gains)
            avg_max_loss = np.mean(max_losses)
            
            positive_count = sum(1 for r in returns if r > 0)
            win_rate = positive_count / len(returns)
            
            report += f"""

### 📈 统计摘要

- **平均后续收益率**: {avg_return:.2%}
- **中位数收益率**: {median_return:.2%}
- **平均最大涨幅**: {avg_max_gain:.2%}
- **平均最大跌幅**: {avg_max_loss:.2%}
- **胜率**: {win_rate:.1%} ({positive_count}/{len(returns)})

### 详细表现分析

"""
            
            for result in performance_results:
                report += f"""
#### {result['code']} ({result['name']})

- **行业**: {result['industry']}
- **相似度得分**: {result['similarity_score']:.4f}
- **匹配历史期间**: {result['match_period']}
- **后续分析期间**: {result['subsequent_days']} 个交易日
- **期间收益率**: {result['total_return']:.2%}
- **最大涨幅**: {result['max_gain']:.2%}
- **最大跌幅**: {result['max_loss']:.2%}
- **年化波动率**: {result['volatility']:.2%}
- **起始价格**: ¥{result['start_price']:.2f}
- **结束价格**: ¥{result['end_price']:.2f}

**算法分解**:
- Matrix Profile: {result['algorithm_details']['mp_similarity']:.4f}
- DTW: {result['algorithm_details']['dtw_similarity']:.4f}
- MASS: {result['algorithm_details']['mass_similarity']:.4f}

---
"""
        
        report += f"""

## 🔍 分析方法说明

### 相似度算法

1. **Matrix Profile**: 时间序列模式发现的主流算法，擅长发现重复模式
2. **Dynamic Time Warping (DTW)**: 处理时间轴上的非线性变形，适合不同速度的相似走势
3. **MASS (Mueen's Algorithm for Similarity Search)**: 高效的相似子序列搜索

### 多维度分析

- **价格维度**: 基于收益率序列的相似度 (权重: 70%)
- **成交量维度**: 基于成交量变化率的相似度 (权重: 30%)

### 筛选条件

- **最小日成交额**: {self.config['filters']['min_daily_volume']:,} 元
- **最小相似度阈值**: {self.config['similarity']['search']['min_similarity']}
- **分析窗口长度**: 30个交易日
- **预测期间长度**: 20个交易日

---

## ⚠️ 重要声明

1. **历史表现不代表未来结果**: 本分析基于历史相似模式，不构成投资建议
2. **市场环境差异**: 不同时期的市场环境可能影响相似模式的有效性
3. **风险提示**: 股票投资存在风险，投资需谨慎
4. **数据有效性**: 分析结果依赖于数据质量和算法准确性

---

**报告生成时间**: {current_time}
**数据来源**: 股票交易数据库
**分析引擎**: StockTradebyZ Likelihood System v1.0
"""
        
        return report


def main():
    """主函数"""
    print("=" * 60)
    print("002215 (诺普信) 股票相似度分析")
    print("=" * 60)
    
    # 创建分析器
    analyzer = Stock002215Analyzer()
    
    # 生成报告
    report_path = analyzer.generate_report('002215')
    
    if report_path:
        print(f"\n✅ 分析完成！报告已保存至: {report_path}")
        print("\n📊 报告包含内容:")
        print("  - 目标股票基本信息")
        print("  - 相似股票发现结果")  
        print("  - 多算法相似度分析")
        print("  - 历史后续走势表现")
        print("  - 统计摘要与风险提示")
    else:
        print("\n❌ 分析失败，请检查数据和配置")


if __name__ == '__main__':
    main()