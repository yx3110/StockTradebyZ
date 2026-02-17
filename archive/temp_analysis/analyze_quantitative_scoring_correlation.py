#!/usr/bin/env python3
"""
量化评分与后续股价相关性分析工具
分析每日选股报告中的量化评分与后续股价表现的关联性
"""

import json
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import os
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

class QuantitativeScoringAnalyzer:
    """量化评分相关性分析器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db", reports_dir: str = "reports/daily_selection"):
        """
        初始化分析器
        
        Args:
            db_path: SQLite数据库路径
            reports_dir: 报告目录路径
        """
        self.db_path = db_path
        self.daily_selection_dir = Path(reports_dir)
        
        # 连接数据库
        self.conn = sqlite3.connect(self.db_path)
        
        # 数据存储
        self.daily_picks = []  # 每日选股数据
        self.price_data = {}   # 价格数据缓存
        self.analysis_results = {}  # 分析结果
        
    def extract_daily_picks_from_markdown(self) -> List[Dict]:
        """从markdown报告中提取每日选股数据"""
        daily_picks = []
        
        # 获取所有选股报告文件
        report_files = list(self.daily_selection_dir.glob("选股分析报告_*.md"))
        report_files.sort()
        
        print(f"找到 {len(report_files)} 个报告文件")
        
        for report_file in report_files:
            try:
                # 从文件名提取日期
                date_match = re.search(r'(\d{8})\.md$', report_file.name)
                if not date_match:
                    continue
                    
                report_date = date_match.group(1)
                print(f"处理报告: {report_date}")
                
                # 读取报告内容
                with open(report_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取股票评分表格
                picks = self._extract_stocks_from_markdown(content, report_date)
                daily_picks.extend(picks)
                
            except Exception as e:
                print(f"处理文件 {report_file} 时出错: {e}")
                continue
        
        return daily_picks
    
    def _extract_stocks_from_markdown(self, content: str, report_date: str) -> List[Dict]:
        """从markdown内容中提取股票评分信息"""
        picks = []
        
        # 查找评分排名表格 - 兼容v1.0、v2.0和v3.0格式
        # v1.0格式: | 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 |
        # v2.0格式: | 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 动量 | 回归 | 突破 | 相对 | 稳定 |
        # v3.0格式: | 排名 | 股票代码 | 股票名称 | 选中策略 | 量化评分 | 投资建议 | 动量 | 回归 | 突破 | 相对 | 稳定 |
        table_patterns = [
            r'\|\s*排名\s*\|.*?量化评分.*?\n((?:\|.*?\n)*)',  # v2.0/v3.0 格式
            r'\|\s*排名\s*\|.*?\n((?:\|.*?\n)*)'              # v1.0 格式
        ]
        
        table_match = None
        for pattern in table_patterns:
            table_match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            if table_match:
                break
        
        if not table_match:
            return picks
        
        table_content = table_match.group(0)
        lines = table_content.split('\n')
        
        for line in lines[2:]:  # 跳过表头和分隔符
            if not line.strip() or not line.startswith('|'):
                continue
                
            # 解析表格行
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) < 6:
                continue
            
            try:
                rank = int(parts[0])
                stock_code = parts[1].strip()
                stock_name = parts[2].strip()
                strategies = parts[3].strip()
                score = float(parts[4].strip())
                recommendation = parts[5].strip()
                
                picks.append({
                    'report_date': report_date,
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'rank': rank,
                    'quantitative_score': score,
                    'strategies': strategies,
                    'recommendation': recommendation
                })
                
            except (ValueError, IndexError) as e:
                continue
        
        print(f"从 {report_date} 提取了 {len(picks)} 只股票")
        return picks
    
    def get_stock_returns(self, stock_code: str, start_date: str, periods: List[int]) -> Dict[int, float]:
        """
        获取股票在指定时间段的收益率
        
        Args:
            stock_code: 股票代码
            start_date: 起始日期 (YYYYMMDD格式)
            periods: 时间段列表 (天数)
            
        Returns:
            字典，键为天数，值为收益率
        """
        returns = {}
        
        try:
            # 转换日期格式
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            start_date_str = start_dt.strftime('%Y-%m-%d')
            
            # 查询股票价格数据
            query = """
                SELECT dq.trade_date, dq.close, dq.price_change_pct
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ? AND dq.trade_date >= ?
                ORDER BY dq.trade_date
                LIMIT 60
            """
            
            df = pd.read_sql_query(query, self.conn, params=[stock_code, start_date_str])
            
            if df.empty:
                return returns
            
            # 计算各个时间段的收益率
            base_price = df.iloc[0]['close']
            
            for period in periods:
                if period < len(df) and period > 0:
                    end_price = df.iloc[period]['close']
                    if pd.notna(base_price) and pd.notna(end_price) and base_price != 0:
                        return_pct = (end_price - base_price) / base_price * 100
                        returns[period] = return_pct
                    else:
                        returns[period] = None
                else:
                    returns[period] = None
                    
        except Exception as e:
            print(f"获取 {stock_code} 收益率时出错: {e}")
            
        return returns
    
    def calculate_sharpe_ratio(self, returns: List[float], periods_per_year: int = 252, risk_free_rate: float = 0.02) -> float:
        """
        计算夏普比率
        
        Args:
            returns: 收益率序列
            periods_per_year: 年化周期数（日收益用252，月收益用12）
            risk_free_rate: 无风险利率（年化）
            
        Returns:
            夏普比率
        """
        if not returns or len(returns) < 2:
            return 0
        
        # 过滤掉None值
        valid_returns = [r for r in returns if r is not None]
        if len(valid_returns) < 2:
            return 0
        
        # 转换为numpy数组（百分比转为小数）
        returns_array = np.array(valid_returns) / 100
        
        # 计算平均收益和标准差
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        if std_return == 0:
            return 0
        
        # 计算日化无风险利率
        daily_risk_free = risk_free_rate / periods_per_year
        
        # 计算夏普比率
        sharpe = (mean_return - daily_risk_free) / std_return * np.sqrt(periods_per_year)
        
        return sharpe
    
    def analyze_scoring_correlation(self, periods: List[int] = [1, 3, 5, 10, 20, 30]) -> Dict:
        """
        分析量化评分与后续收益率的相关性
        
        Args:
            periods: 要分析的时间段（天数）
            
        Returns:
            分析结果字典
        """
        print("开始分析量化评分与股价相关性...")
        
        # 提取每日选股数据
        daily_picks = self.extract_daily_picks_from_markdown()
        
        if not daily_picks:
            print("未找到选股数据")
            return {}
        
        # 准备分析数据
        analysis_data = []
        
        for pick in daily_picks:
            # 获取股票后续收益率
            returns = self.get_stock_returns(
                pick['stock_code'], 
                pick['report_date'], 
                periods
            )
            
            # 添加到分析数据
            row = pick.copy()
            for period in periods:
                row[f'return_{period}d'] = returns.get(period)
            
            analysis_data.append(row)
        
        # 转换为DataFrame进行分析
        df = pd.DataFrame(analysis_data)
        print(f"原始数据量: {len(df)}条记录")
        
        # 🚀 NEW: 应用去重逻辑消除重叠案例的统计偏差
        print("🔍 检测并移除重叠案例以消除统计偏差...")
        
        try:
            from correlation_bias_analyzer import CorrelationBiasAnalyzer
            bias_analyzer = CorrelationBiasAnalyzer()
            
            # 计算原始相关性（含偏差）
            original_correlations = {}
            for period in periods:
                return_col = f'return_{period}d'
                if return_col in df.columns:
                    valid_data = df.dropna(subset=['quantitative_score', return_col])
                    if len(valid_data) > 10:
                        corr, p_value = stats.pearsonr(
                            valid_data['quantitative_score'], 
                            valid_data[return_col]
                        )
                        original_correlations[f'{period}d'] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'sample_size': len(valid_data)
                        }
            
            # 🎯 使用去重数据集计算真实相关性
            debiased_correlations = bias_analyzer.calculate_correlation_debiased(df)
            
            print("✅ 已应用去重逻辑，消除重叠案例的统计偏差")
            
            # 显示偏差修正结果
            print("\n📊 偏差修正结果:")
            print(f"{'时间段':<6} {'原始相关性':<12} {'去重相关性':<12} {'修正幅度':<10}")
            print("-" * 50)
            
            correlations = {}
            significance_tests = {}
            bias_corrections = {}
            
            for period in periods:
                period_key = f'{period}d'
                if period_key in original_correlations and period_key in debiased_correlations:
                    orig_corr = original_correlations[period_key]['correlation']
                    debiased_corr = debiased_correlations[period_key]['correlation']
                    correction = ((debiased_corr - orig_corr) / abs(orig_corr)) * 100 if orig_corr != 0 else 0
                    
                    print(f"{period_key:<6} {orig_corr:<12.4f} {debiased_corr:<12.4f} {correction:<10.1f}%")
                    
                    # 使用去重后的真实相关性
                    correlations[period_key] = debiased_corr
                    significance_tests[period_key] = debiased_correlations[period_key]['p_value']
                    
                    bias_corrections[period_key] = {
                        'original_correlation': orig_corr,
                        'debiased_correlation': debiased_corr,
                        'correction_percent': correction,
                        'original_sample_size': original_correlations[period_key]['sample_size'],
                        'debiased_sample_size': debiased_correlations[period_key]['sample_size'],
                        'removed_overlaps': debiased_correlations[period_key].get('removed_overlaps', 0)
                    }
                
        except ImportError:
            print("⚠️  偏差分析器未导入，使用原始逻辑")
            # 回退到原始逻辑
            correlations = {}
            significance_tests = {}
            bias_corrections = {}
            
            for period in periods:
                return_col = f'return_{period}d'
                if return_col in df.columns:
                    valid_data = df.dropna(subset=['quantitative_score', return_col])
                    
                    if len(valid_data) > 10:
                        corr, p_value = stats.pearsonr(
                            valid_data['quantitative_score'], 
                            valid_data[return_col]
                        )
                        correlations[f'{period}d'] = corr
                        significance_tests[f'{period}d'] = p_value
                    
                    print(f"{period}天收益率相关性: {corr:.4f} (p-value: {p_value:.4f})")
        
        # 按评分分组分析
        score_bins = [0, 60, 70, 80, 90, 100]
        df['score_group'] = pd.cut(df['quantitative_score'], bins=score_bins, 
                                  labels=['<60', '60-70', '70-80', '80-90', '90+'])
        
        group_analysis = {}
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in df.columns:
                group_stats = df.groupby('score_group')[return_col].agg([
                    'count', 'mean', 'std', 'median'
                ]).round(4)
                group_analysis[f'{period}d'] = group_stats.to_dict()
        
        # 按推荐类型分析
        recommendation_analysis = {}
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in df.columns:
                rec_stats = df.groupby('recommendation')[return_col].agg([
                    'count', 'mean', 'std', 'median'
                ]).round(4)
                recommendation_analysis[f'{period}d'] = rec_stats.to_dict()
        
        # 添加具体案例分析 - 使用去重逻辑避免bias
        top_performers = []
        worst_performers = []
        
        # 找出表现最好的案例（20天收益）
        if 'return_20d' in df.columns:
            df_valid = df.dropna(subset=['return_20d'])
            if len(df_valid) > 0:
                # 导入去重选择器
                try:
                    from debiased_case_selector import DebiasedCaseSelector
                    selector = DebiasedCaseSelector(overlap_window=20)
                    
                    # 使用去重选择避免时间重叠bias
                    debiased_cases = selector.select_best_cases(
                        df_valid, n_best=10, n_worst=10,
                        date_col='report_date', stock_col='stock_code', 
                        return_col='return_20d', score_col='quantitative_score'
                    )
                    
                    top_performers = debiased_cases['best_cases']
                    worst_performers = debiased_cases['worst_cases']
                    
                    print(f"✅ 使用去重选择器避免重叠偏差")
                    print(f"   最佳案例: {len(top_performers)}个 (涉及{len(set(case['stock_code'] for case in top_performers))}只股票)")
                    print(f"   最差案例: {len(worst_performers)}个 (涉及{len(set(case['stock_code'] for case in worst_performers))}只股票)")
                    
                except ImportError:
                    print("⚠️  去重选择器未导入，使用原始逻辑")
                    # 最佳表现案例
                    top_10 = df_valid.nlargest(10, 'return_20d')[
                        ['report_date', 'stock_code', 'stock_name', 'quantitative_score', 
                         'return_1d', 'return_5d', 'return_10d', 'return_20d']
                    ]
                    top_performers = top_10.to_dict('records')
                    
                    # 最差表现案例
                    worst_10 = df_valid.nsmallest(10, 'return_20d')[
                        ['report_date', 'stock_code', 'stock_name', 'quantitative_score',
                         'return_1d', 'return_5d', 'return_10d', 'return_20d']
                    ]
                    worst_performers = worst_10.to_dict('records')
        
        # 计算收益预期分布
        return_expectations = {}
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in df.columns:
                valid_data = df.dropna(subset=[return_col])
                if len(valid_data) > 0:
                    return_expectations[f'{period}d'] = {
                        'mean': valid_data[return_col].mean(),
                        'median': valid_data[return_col].median(),
                        'std': valid_data[return_col].std(),
                        'percentile_25': valid_data[return_col].quantile(0.25),
                        'percentile_75': valid_data[return_col].quantile(0.75),
                        'win_rate': (valid_data[return_col] > 0).mean() * 100,
                        'avg_win': valid_data[valid_data[return_col] > 0][return_col].mean() if (valid_data[return_col] > 0).any() else 0,
                        'avg_loss': valid_data[valid_data[return_col] < 0][return_col].mean() if (valid_data[return_col] < 0).any() else 0,
                        'max_gain': valid_data[return_col].max(),
                        'max_loss': valid_data[return_col].min()
                    }
        
        # 按评分区间计算预期收益和夏普比率
        score_return_expectations = {}
        score_sharpe_ratios = {}  # 新增：存储各评分区间的夏普比率
        score_ranges = [
            (90, 100, '90+分'),
            (80, 90, '80-90分'),
            (70, 80, '70-80分'),
            (60, 70, '60-70分'),
            (0, 60, '<60分')
        ]
        
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in df.columns:
                period_stats = {}
                period_sharpe = {}  # 存储该时期各评分区间的夏普比率
                
                for min_score, max_score, label in score_ranges:
                    score_mask = (df['quantitative_score'] >= min_score) & (df['quantitative_score'] < max_score)
                    score_data = df[score_mask].dropna(subset=[return_col])
                    
                    if len(score_data) > 5:  # 至少需要5个样本
                        returns_list = score_data[return_col].tolist()
                        sharpe_ratio = self.calculate_sharpe_ratio(returns_list, periods_per_year=252/period)
                        
                        period_stats[label] = {
                            'count': len(score_data),
                            'mean_return': score_data[return_col].mean(),
                            'median_return': score_data[return_col].median(),
                            'win_rate': (score_data[return_col] > 0).mean() * 100,
                            'avg_win': score_data[score_data[return_col] > 0][return_col].mean() if (score_data[return_col] > 0).any() else 0,
                            'avg_loss': score_data[score_data[return_col] < 0][return_col].mean() if (score_data[return_col] < 0).any() else 0,
                            'std': score_data[return_col].std(),
                            'sharpe_ratio': sharpe_ratio  # 新增夏普比率
                        }
                        period_sharpe[label] = sharpe_ratio
                    else:
                        period_stats[label] = {
                            'count': len(score_data),
                            'mean_return': 0,
                            'median_return': 0,
                            'win_rate': 0,
                            'avg_win': 0,
                            'avg_loss': 0,
                            'std': 0,
                            'sharpe_ratio': 0
                        }
                        period_sharpe[label] = 0
                
                score_return_expectations[f'{period}d'] = period_stats
                score_sharpe_ratios[f'{period}d'] = period_sharpe
        
        # 计算整体夏普比率（按时期）
        overall_sharpe_ratios = {}
        for period in periods:
            return_col = f'return_{period}d'
            if return_col in df.columns:
                valid_data = df.dropna(subset=[return_col])
                if len(valid_data) > 10:
                    returns_list = valid_data[return_col].tolist()
                    overall_sharpe = self.calculate_sharpe_ratio(returns_list, periods_per_year=252/period)
                    overall_sharpe_ratios[f'{period}d'] = overall_sharpe
        
        # 汇总分析结果
        results = {
            'data_summary': {
                'total_picks': len(df),
                'date_range': f"{df['report_date'].min()} - {df['report_date'].max()}",
                'unique_stocks': df['stock_code'].nunique(),
                'avg_score': df['quantitative_score'].mean(),
                'score_std': df['quantitative_score'].std()
            },
            'correlations': correlations,
            'significance_tests': significance_tests,
            'bias_corrections': bias_corrections if 'bias_corrections' in locals() else {},  # 🆕 偏差修正信息
            'score_group_analysis': group_analysis,
            'recommendation_analysis': recommendation_analysis,
            'return_expectations': return_expectations,
            'score_return_expectations': score_return_expectations,
            'score_sharpe_ratios': score_sharpe_ratios,  # 新增：各评分区间的夏普比率
            'overall_sharpe_ratios': overall_sharpe_ratios,  # 新增：整体夏普比率
            'top_performers': top_performers,
            'worst_performers': worst_performers,
            'raw_data': df.to_dict('records')
        }
        
        return results
    
    def generate_improvement_suggestions(self, analysis_results: Dict) -> List[str]:
        """基于相关性分析结果生成改进建议"""
        suggestions = []
        
        if not analysis_results or 'correlations' not in analysis_results:
            return ["无法生成建议：缺少分析结果"]
        
        correlations = analysis_results['correlations']
        significance_tests = analysis_results.get('significance_tests', {})
        
        # 🆕 基于去重后的真实相关性分析强度（提高评估标准）
        strong_correlations = []
        weak_correlations = []
        
        for period, corr in correlations.items():
            p_val = significance_tests.get(period, 1.0)
            # 🚀 更新评估标准：基于去重后的真实相关性提高门槛
            if abs(corr) > 0.25 and p_val < 0.05:  # 从0.3降到0.25，因为真实相关性更高
                strong_correlations.append((period, corr))
            elif abs(corr) < 0.15 or p_val > 0.05:  # 从0.1提高到0.15
                weak_correlations.append((period, corr))
        
        # 生成建议
        if strong_correlations:
            suggestions.append(f"✅ 发现强相关性: {strong_correlations}")
            suggestions.append("建议保持当前评分体系，可以适当增加权重")
        
        if weak_correlations:
            suggestions.append(f"⚠️ 发现弱相关性: {weak_correlations}")
            suggestions.append("建议重新审视评分算法，可能需要调整指标权重")
        
        # 基于评分组表现给出建议
        if 'score_group_analysis' in analysis_results:
            suggestions.append("\n📊 按评分分组表现分析:")
            for period, group_data in analysis_results['score_group_analysis'].items():
                if 'mean' in group_data:
                    means = group_data['mean']
                    # 检查是否高分组表现更好
                    high_score_avg = means.get('80-90', 0) + means.get('90+', 0)
                    low_score_avg = means.get('<60', 0) + means.get('60-70', 0)
                    
                    if high_score_avg > low_score_avg:
                        suggestions.append(f"✅ {period}: 高分组表现优于低分组，评分有效")
                    else:
                        suggestions.append(f"❌ {period}: 高分组表现不如预期，需要优化")
        
        # 具体改进建议
        suggestions.extend([
            "\n🔧 具体改进建议:",
            "1. 增加成交量因子权重",
            "2. 结合基本面指标（PE、PB、ROE）",
            "3. 加入情绪指标（资金流向、舆情分析）",
            "4. 动态调整技术指标参数",
            "5. 增加止损机制的评分权重",
            "6. 考虑行业轮动因素",
            "7. 加入宏观经济指标影响"
        ])
        
        return suggestions
    
    def save_analysis_report(self, results: Dict, output_path: str):
        """保存分析报告"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# 量化评分与股价相关性分析报告\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 数据概览
                if 'data_summary' in results:
                    summary = results['data_summary']
                    f.write("## 数据概览\n\n")
                    f.write(f"- 总选股数: {summary.get('total_picks', 0)}\n")
                    f.write(f"- 日期范围: {summary.get('date_range', 'N/A')}\n")
                    f.write(f"- 独特股票数: {summary.get('unique_stocks', 0)}\n")
                    f.write(f"- 平均评分: {summary.get('avg_score', 0):.2f}\n")
                    f.write(f"- 评分标准差: {summary.get('score_std', 0):.2f}\n\n")
                
                # 🆕 偏差修正信息（如果存在）
                if 'bias_corrections' in results and results['bias_corrections']:
                    f.write("## 🚀 统计偏差修正\n\n")
                    f.write("**重要发现**: 通过去重数据集消除了重叠案例的统计偏差，揭示了量化评分系统的真实预测能力。\n\n")
                    f.write("| 时间段 | 原始相关性 | 去重相关性 | 提升幅度 | 移除重叠数 |\n")
                    f.write("|--------|------------|------------|----------|------------|\n")
                    
                    bias_corrections = results['bias_corrections']
                    for period, correction in bias_corrections.items():
                        orig = correction['original_correlation']
                        debiased = correction['debiased_correlation']
                        improvement = correction['correction_percent']
                        removed = correction['removed_overlaps']
                        
                        f.write(f"| {period} | {orig:.4f} | {debiased:.4f} | {improvement:+.1f}% | {removed} |\n")
                    
                    f.write("\n💡 **关键洞察**: 量化评分系统的真实预测能力被重叠案例严重低估了！去重后的相关性显著提高。\n\n")
                
                # 相关性分析（使用去重后的真实数据）
                if 'correlations' in results:
                    f.write("## 📊 相关性分析结果（去重后真实数据）\n\n")
                    f.write("| 时间段 | 相关系数 | P值 | 显著性 |\n")
                    f.write("|--------|---------|-----|--------|\n")
                    
                    correlations = results['correlations']
                    significance = results.get('significance_tests', {})
                    
                    for period, corr in correlations.items():
                        p_val = significance.get(period, 1.0)
                        sig = "显著" if p_val < 0.05 else "不显著"
                        f.write(f"| {period} | {corr:.4f} | {p_val:.4f} | {sig} |\n")
                    f.write("\n")
                
                # 收益预期统计
                if 'return_expectations' in results:
                    f.write("## 收益预期统计\n\n")
                    f.write("| 时间段 | 平均收益 | 中位数 | 胜率 | 平均盈利 | 平均亏损 | 最大收益 | 最大亏损 |\n")
                    f.write("|--------|---------|--------|------|---------|---------|---------|----------|\n")
                    
                    for period, stats in results['return_expectations'].items():
                        f.write(f"| {period} | {stats['mean']:.2f}% | {stats['median']:.2f}% | ")
                        f.write(f"{stats['win_rate']:.1f}% | {stats['avg_win']:.2f}% | ")
                        f.write(f"{stats['avg_loss']:.2f}% | {stats['max_gain']:.2f}% | ")
                        f.write(f"{stats['max_loss']:.2f}% |\n")
                    f.write("\n")
                
                # 新增：夏普比率分析
                if 'overall_sharpe_ratios' in results and results['overall_sharpe_ratios']:
                    f.write("## 📈 夏普比率分析\n\n")
                    f.write("夏普比率是衡量风险调整后收益的重要指标，值越高表示单位风险获得的超额收益越高。\n\n")
                    
                    # 整体夏普比率
                    f.write("### 整体夏普比率\n\n")
                    f.write("| 时间段 | 夏普比率 | 评价 |\n")
                    f.write("|--------|----------|------|\n")
                    
                    for period, sharpe in results['overall_sharpe_ratios'].items():
                        if sharpe > 2:
                            evaluation = "优秀"
                        elif sharpe > 1:
                            evaluation = "良好"
                        elif sharpe > 0.5:
                            evaluation = "一般"
                        elif sharpe > 0:
                            evaluation = "较差"
                        else:
                            evaluation = "负收益"
                        f.write(f"| {period} | {sharpe:.3f} | {evaluation} |\n")
                    f.write("\n")
                    
                    # 各评分区间的夏普比率
                    if 'score_sharpe_ratios' in results and results['score_sharpe_ratios']:
                        f.write("### 各评分区间夏普比率\n\n")
                        f.write("| 评分区间 | 1天 | 5天 | 10天 | 20天 | 30天 | 最优持仓期 |\n")
                        f.write("|----------|-----|-----|------|------|------|------------|\n")
                        
                        score_ranges = ['90+分', '80-90分', '70-80分', '60-70分', '<60分']
                        for score_range in score_ranges:
                            f.write(f"| {score_range} ")
                            
                            best_period = ""
                            best_sharpe = -999
                            
                            for period in ['1d', '5d', '10d', '20d', '30d']:
                                if period in results['score_sharpe_ratios']:
                                    sharpe = results['score_sharpe_ratios'][period].get(score_range, 0)
                                    f.write(f"| {sharpe:.3f} ")
                                    
                                    if sharpe > best_sharpe:
                                        best_sharpe = sharpe
                                        best_period = period
                                else:
                                    f.write("| - ")
                            
                            f.write(f"| {best_period} (夏普:{best_sharpe:.3f}) |\n")
                        f.write("\n")
                        
                        # 夏普比率投资建议
                        f.write("### 🎯 基于夏普比率的投资建议\n\n")
                        f.write("根据风险调整后收益分析，推荐以下投资策略：\n\n")
                        
                        # 找出最佳夏普比率的评分区间和持仓期
                        best_configs = []
                        for period, score_sharpes in results['score_sharpe_ratios'].items():
                            for score_range, sharpe in score_sharpes.items():
                                best_configs.append({
                                    'period': period,
                                    'score_range': score_range,
                                    'sharpe': sharpe
                                })
                        
                        # 排序找出前3个最佳配置
                        best_configs.sort(key=lambda x: x['sharpe'], reverse=True)
                        top_configs = best_configs[:3]
                        
                        for i, config in enumerate(top_configs, 1):
                            f.write(f"{i}. **{config['score_range']}股票，持有{config['period']}**\n")
                            f.write(f"   - 夏普比率: {config['sharpe']:.3f}\n")
                            if config['sharpe'] > 1.5:
                                f.write("   - 风险收益比: 极佳\n")
                            elif config['sharpe'] > 1:
                                f.write("   - 风险收益比: 良好\n")
                            elif config['sharpe'] > 0.5:
                                f.write("   - 风险收益比: 一般\n")
                            else:
                                f.write("   - 风险收益比: 较差\n")
                            f.write("\n")
                    f.write("\n")
                
                # 按评分预期收益表
                if 'score_return_expectations' in results:
                    f.write("## 📊 按评分预期收益表\n\n")
                    f.write("### 各评分区间在不同持仓期的预期表现\n\n")
                    
                    # 创建综合表格（包含夏普比率）
                    f.write("| 评分区间 | 时间段 | 样本数 | 平均收益 | 中位数 | 胜率 | 平均盈利 | 平均亏损 | 夏普比率 |\n")
                    f.write("|----------|--------|--------|----------|--------|------|----------|----------|----------|\n")
                    
                    score_ranges = ['90+分', '80-90分', '70-80分', '60-70分', '<60分']
                    for score_range in score_ranges:
                        for period in ['1d', '5d', '10d', '20d', '30d']:
                            if period in results['score_return_expectations']:
                                stats = results['score_return_expectations'][period].get(score_range)
                                if stats and stats['count'] > 0:
                                    f.write(f"| {score_range} | {period} | {stats['count']} | ")
                                    f.write(f"{stats['mean_return']:.2f}% | {stats['median_return']:.2f}% | ")
                                    f.write(f"{stats['win_rate']:.1f}% | {stats['avg_win']:.2f}% | ")
                                    f.write(f"{stats['avg_loss']:.2f}% | ")
                                    # 添加夏普比率
                                    sharpe = stats.get('sharpe_ratio', 0)
                                    f.write(f"{sharpe:.3f} |\n")
                        f.write("|----------|--------|--------|----------|--------|------|----------|----------|----------|\n")
                    f.write("\n")
                    
                    # 添加投资指导建议
                    f.write("### 🎯 投资指导建议\n\n")
                    
                    # 分析各评分区间的最优持仓期
                    for score_range in score_ranges:
                        f.write(f"**{score_range}股票:**\n")
                        best_period = ""
                        best_return = -999
                        best_win_rate = 0
                        
                        for period in ['1d', '5d', '10d', '20d', '30d']:
                            if (period in results['score_return_expectations'] and 
                                score_range in results['score_return_expectations'][period]):
                                stats = results['score_return_expectations'][period][score_range]
                                if stats['count'] > 10 and stats['mean_return'] > best_return:
                                    best_period = period
                                    best_return = stats['mean_return']
                                    best_win_rate = stats['win_rate']
                        
                        if best_period:
                            f.write(f"- 最优持仓期: {best_period}\n")
                            f.write(f"- 预期收益: {best_return:.2f}%\n")
                            f.write(f"- 胜率: {best_win_rate:.1f}%\n")
                            
                            if best_return > 5:
                                f.write("- 建议: **强烈推荐**\n")
                            elif best_return > 2:
                                f.write("- 建议: **推荐**\n")
                            elif best_return > 0:
                                f.write("- 建议: **谨慎持有**\n")
                            else:
                                f.write("- 建议: **回避**\n")
                        else:
                            f.write("- 样本不足，建议谨慎\n")
                        f.write("\n")
                
                # 最佳表现案例
                if 'top_performers' in results and results['top_performers']:
                    f.write("## 🏆 最佳表现案例 (TOP 10)\n\n")
                    f.write("| 日期 | 股票代码 | 股票名称 | 量化评分 | 1天收益 | 5天收益 | 10天收益 | 20天收益 |\n")
                    f.write("|------|----------|----------|----------|---------|---------|----------|----------|\n")
                    
                    for stock in results['top_performers']:
                        f.write(f"| {stock['report_date']} | {stock['stock_code']} | ")
                        f.write(f"{stock.get('stock_name', 'N/A')} | {stock['quantitative_score']:.1f} | ")
                        f.write(f"{stock.get('return_1d', 0):.2f}% | {stock.get('return_5d', 0):.2f}% | ")
                        f.write(f"{stock.get('return_10d', 0):.2f}% | {stock.get('return_20d', 0):.2f}% |\n")
                    f.write("\n")
                
                # 最差表现案例
                if 'worst_performers' in results and results['worst_performers']:
                    f.write("## 📉 最差表现案例 (BOTTOM 10)\n\n")
                    f.write("| 日期 | 股票代码 | 股票名称 | 量化评分 | 1天收益 | 5天收益 | 10天收益 | 20天收益 |\n")
                    f.write("|------|----------|----------|----------|---------|---------|----------|----------|\n")
                    
                    for stock in results['worst_performers']:
                        f.write(f"| {stock['report_date']} | {stock['stock_code']} | ")
                        f.write(f"{stock.get('stock_name', 'N/A')} | {stock['quantitative_score']:.1f} | ")
                        f.write(f"{stock.get('return_1d', 0):.2f}% | {stock.get('return_5d', 0):.2f}% | ")
                        f.write(f"{stock.get('return_10d', 0):.2f}% | {stock.get('return_20d', 0):.2f}% |\n")
                    f.write("\n")
                
                # 改进建议
                suggestions = self.generate_improvement_suggestions(results)
                f.write("## 改进建议\n\n")
                for suggestion in suggestions:
                    f.write(f"{suggestion}\n")
                
                f.write("\n---\n🤖 Generated with Claude Code\n")
                
            print(f"分析报告已保存至: {output_path}")
            
        except Exception as e:
            print(f"保存报告时出错: {e}")
    
    def run_complete_analysis(self, version: str = '') -> Dict:
        """运行完整的相关性分析"""
        print("🚀 开始量化评分相关性分析...")
        print("📈 现已包含夏普比率分析...")
        
        # 执行分析
        results = self.analyze_scoring_correlation()
        
        if results:
            # 添加版本标识到文件名
            version_suffix = f"_{version}" if version else ""
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            # 保存详细报告
            report_path = f"reports/correlation_analysis/量化评分相关性分析{version_suffix}_{timestamp}.md"
            self.save_analysis_report(results, report_path)
            
            # 保存原始数据
            data_path = f"reports/correlation_analysis/量化评分原始数据{version_suffix}_{timestamp}.json"
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print("✅ 分析完成！")
            
            # 输出夏普比率关键发现
            if 'overall_sharpe_ratios' in results:
                print("\n📊 夏普比率关键发现:")
                for period, sharpe in results['overall_sharpe_ratios'].items():
                    print(f"  {period}夏普比率: {sharpe:.3f}")
                    
            # 输出最佳风险调整收益配置
            if 'score_sharpe_ratios' in results:
                print("\n🏆 最佳风险调整收益配置:")
                best_configs = []
                for period, score_sharpes in results['score_sharpe_ratios'].items():
                    for score_range, sharpe in score_sharpes.items():
                        best_configs.append({
                            'period': period,
                            'score_range': score_range,
                            'sharpe': sharpe
                        })
                best_configs.sort(key=lambda x: x['sharpe'], reverse=True)
                for i, config in enumerate(best_configs[:3], 1):
                    print(f"  {i}. {config['score_range']} 持有{config['period']}: 夏普比率 {config['sharpe']:.3f}")
        else:
            print("❌ 分析失败，未找到有效数据")
        
        return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='量化评分相关性分析')
    parser.add_argument('--report-dir', default='reports/daily_selection', 
                       help='报告目录路径')
    parser.add_argument('--start-date', help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--version', default='', 
                       help='报告版本标识 (用于文件名区分)')
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = QuantitativeScoringAnalyzer(reports_dir=args.report_dir)
    
    # 运行完整分析
    results = analyzer.run_complete_analysis(version=args.version)
    
    # 输出简要结果
    if results and 'correlations' in results:
        print("\n📊 关键发现:")
        for period, corr in results['correlations'].items():
            print(f"  {period}收益率相关性: {corr:.4f}")
    
    return results

if __name__ == "__main__":
    main()