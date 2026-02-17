#!/usr/bin/env python3
"""
数据质量检查机制
专门用于检查优化训练数据的质量和完整性
"""

import os
import sys
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import json

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(project_root, 'data_adapter/stock_data.db')
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 质量标准
        self.quality_standards = {
            'min_trading_days_per_stock': 200,  # 每只股票最少交易天数
            'max_missing_ratio': 0.05,          # 最大缺失比例5%
            'max_zero_volume_ratio': 0.01,      # 最大零成交量比例1%
            'max_extreme_change_ratio': 0.02,   # 最大极端涨跌比例2%
            'min_avg_volume': 10000,             # 最小平均成交量
            'completeness_threshold': 90.0       # 数据完整性阈值90%
        }
    
    def check_stock_quality(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        检查指定股票的数据质量
        
        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含质量评分的DataFrame
        """
        self.logger.info(f"📋 检查 {len(ts_codes)} 只股票的数据质量...")
        
        conn = sqlite3.connect(self.db_path)
        
        quality_results = []
        
        for i, ts_code in enumerate(ts_codes):
            if (i + 1) % 100 == 0:
                self.logger.info(f"已检查 {i + 1}/{len(ts_codes)} 只股票")
            
            # 获取股票基本信息
            stock_info_query = """
            SELECT security_id, fullname FROM stock_basic_info 
            WHERE ts_code = ?
            """
            stock_info = pd.read_sql_query(stock_info_query, conn, params=[ts_code])
            
            if stock_info.empty:
                continue
                
            security_id = stock_info.iloc[0]['security_id']
            stock_name = stock_info.iloc[0]['fullname']
            
            # 获取交易数据 (处理文本类型的日期)
            trading_query = """
            SELECT trade_date, close, volume, price_change_pct, is_suspend
            FROM daily_quotes
            WHERE security_id = ? AND trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date
            """
            
            df_stock = pd.read_sql_query(trading_query, conn, 
                                       params=[security_id, start_date, end_date])
            
            if df_stock.empty:
                quality_results.append({
                    'ts_code': ts_code,
                    'stock_name': stock_name,
                    'quality_score': 0.0,
                    'issues': ['no_data'],
                    'trading_days': 0,
                    'recommendation': 'exclude'
                })
                continue
            
            # 质量检查
            issues = []
            quality_score = 100.0
            
            # 1. 交易天数检查
            trading_days = len(df_stock)
            if trading_days < self.quality_standards['min_trading_days_per_stock']:
                issues.append('insufficient_trading_days')
                quality_score -= 20
            
            # 2. 缺失值检查
            missing_ratio = df_stock.isna().sum().sum() / (len(df_stock) * len(df_stock.columns))
            if missing_ratio > self.quality_standards['max_missing_ratio']:
                issues.append('excessive_missing_data')
                quality_score -= 15
            
            # 3. 零成交量检查
            zero_volume_ratio = (df_stock['volume'] == 0).sum() / len(df_stock)
            if zero_volume_ratio > self.quality_standards['max_zero_volume_ratio']:
                issues.append('frequent_zero_volume')
                quality_score -= 10
            
            # 4. 极端价格变动检查
            if 'price_change_pct' in df_stock.columns:
                price_changes = df_stock['price_change_pct'].dropna()
                if len(price_changes) > 0:
                    extreme_ratio = (abs(price_changes) > 0.2).sum() / len(price_changes)
                    if extreme_ratio > self.quality_standards['max_extreme_change_ratio']:
                        issues.append('frequent_extreme_changes')
                        quality_score -= 10
            
            # 5. 平均成交量检查
            avg_volume = df_stock['volume'].mean()
            if avg_volume < self.quality_standards['min_avg_volume']:
                issues.append('low_liquidity')
                quality_score -= 15
            
            # 6. 停牌检查
            if 'is_suspend' in df_stock.columns:
                suspend_ratio = df_stock['is_suspend'].sum() / len(df_stock)
                if suspend_ratio > 0.1:  # 超过10%时间停牌
                    issues.append('frequent_suspension')
                    quality_score -= 20
            
            # 7. 价格连续性检查
            price_gaps = df_stock['close'].diff().abs() / df_stock['close'].shift(1)
            large_gaps = (price_gaps > 0.5).sum()  # 超过50%的价格跳空
            if large_gaps > 5:
                issues.append('price_discontinuity')
                quality_score -= 10
            
            # 确保质量评分不为负
            quality_score = max(0.0, quality_score)
            
            # 推荐决策
            if quality_score >= 80:
                recommendation = 'include'
            elif quality_score >= 60:
                recommendation = 'conditional'
            else:
                recommendation = 'exclude'
            
            quality_results.append({
                'ts_code': ts_code,
                'stock_name': stock_name,
                'quality_score': quality_score,
                'issues': issues,
                'trading_days': trading_days,
                'missing_ratio': missing_ratio,
                'zero_volume_ratio': zero_volume_ratio,
                'avg_volume': avg_volume,
                'recommendation': recommendation
            })
        
        conn.close()
        
        df_quality = pd.DataFrame(quality_results)
        self.logger.info(f"✅ 数据质量检查完成")
        
        return df_quality
    
    def generate_quality_report(self, quality_df: pd.DataFrame) -> Dict:
        """生成质量报告"""
        
        report = {
            'summary': {
                'total_stocks': len(quality_df),
                'high_quality_stocks': len(quality_df[quality_df['quality_score'] >= 80]),
                'medium_quality_stocks': len(quality_df[(quality_df['quality_score'] >= 60) & 
                                                      (quality_df['quality_score'] < 80)]),
                'low_quality_stocks': len(quality_df[quality_df['quality_score'] < 60]),
                'recommended_stocks': len(quality_df[quality_df['recommendation'] == 'include']),
                'avg_quality_score': quality_df['quality_score'].mean()
            },
            'quality_distribution': {
                'score_90_100': len(quality_df[quality_df['quality_score'] >= 90]),
                'score_80_90': len(quality_df[(quality_df['quality_score'] >= 80) & 
                                            (quality_df['quality_score'] < 90)]),
                'score_70_80': len(quality_df[(quality_df['quality_score'] >= 70) & 
                                            (quality_df['quality_score'] < 80)]),
                'score_60_70': len(quality_df[(quality_df['quality_score'] >= 60) & 
                                            (quality_df['quality_score'] < 70)]),
                'score_below_60': len(quality_df[quality_df['quality_score'] < 60])
            },
            'common_issues': {},
            'recommendations': {
                'include': quality_df[quality_df['recommendation'] == 'include']['ts_code'].tolist(),
                'conditional': quality_df[quality_df['recommendation'] == 'conditional']['ts_code'].tolist(),
                'exclude': quality_df[quality_df['recommendation'] == 'exclude']['ts_code'].tolist()
            }
        }
        
        # 统计常见问题
        all_issues = []
        for issues_list in quality_df['issues']:
            all_issues.extend(issues_list)
        
        issue_counts = pd.Series(all_issues).value_counts()
        report['common_issues'] = issue_counts.to_dict()
        
        return report
    
    def filter_high_quality_stocks(self, ts_codes: List[str], 
                                 start_date: str, end_date: str,
                                 min_quality_score: float = 70.0) -> List[str]:
        """
        筛选高质量股票
        
        Args:
            ts_codes: 候选股票列表
            start_date: 检查开始日期
            end_date: 检查结束日期  
            min_quality_score: 最低质量分数阈值
            
        Returns:
            高质量股票代码列表
        """
        self.logger.info(f"🎯 筛选高质量股票 (最低分数: {min_quality_score})")
        
        quality_df = self.check_stock_quality(ts_codes, start_date, end_date)
        
        high_quality_stocks = quality_df[quality_df['quality_score'] >= min_quality_score]
        
        self.logger.info(f"✅ 从 {len(ts_codes)} 只候选股票中筛选出 {len(high_quality_stocks)} 只高质量股票")
        
        return high_quality_stocks['ts_code'].tolist()
    
    def save_quality_report(self, quality_df: pd.DataFrame, 
                          output_dir: str = None) -> None:
        """保存质量报告"""
        if output_dir is None:
            output_dir = Path(__file__).parent / 'quality_reports'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存详细数据
        detail_file = output_dir / f'data_quality_details_{timestamp}.csv'
        quality_df.to_csv(detail_file, index=False, encoding='utf-8-sig')
        
        # 生成并保存报告
        report = self.generate_quality_report(quality_df)
        report_file = output_dir / f'data_quality_report_{timestamp}.json'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        md_file = output_dir / f'数据质量报告_{timestamp}.md'
        self._generate_markdown_report(report, md_file)
        
        self.logger.info(f"✅ 质量报告已保存:")
        self.logger.info(f"  详细数据: {detail_file}")
        self.logger.info(f"  JSON报告: {report_file}")
        self.logger.info(f"  Markdown报告: {md_file}")
    
    def _generate_markdown_report(self, report: Dict, output_file: Path) -> None:
        """生成Markdown格式的质量报告"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 数据质量检查报告\\n\\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # 概览
            f.write("## 📊 质量概览\\n\\n")
            summary = report['summary']
            f.write(f"- **总股票数**: {summary['total_stocks']}\\n")
            f.write(f"- **高质量股票**: {summary['high_quality_stocks']} ({summary['high_quality_stocks']/summary['total_stocks']*100:.1f}%)\\n")
            f.write(f"- **中等质量股票**: {summary['medium_quality_stocks']} ({summary['medium_quality_stocks']/summary['total_stocks']*100:.1f}%)\\n")
            f.write(f"- **低质量股票**: {summary['low_quality_stocks']} ({summary['low_quality_stocks']/summary['total_stocks']*100:.1f}%)\\n")
            f.write(f"- **推荐使用股票**: {summary['recommended_stocks']}\\n")
            f.write(f"- **平均质量评分**: {summary['avg_quality_score']:.1f}\\n\\n")
            
            # 质量分布
            f.write("## 📈 质量分布\\n\\n")
            dist = report['quality_distribution']
            f.write("| 评分区间 | 股票数量 | 占比 |\\n")
            f.write("|---------|---------|------|\\n")
            total = summary['total_stocks']
            f.write(f"| 90-100 | {dist['score_90_100']} | {dist['score_90_100']/total*100:.1f}% |\\n")
            f.write(f"| 80-90 | {dist['score_80_90']} | {dist['score_80_90']/total*100:.1f}% |\\n")
            f.write(f"| 70-80 | {dist['score_70_80']} | {dist['score_70_80']/total*100:.1f}% |\\n")
            f.write(f"| 60-70 | {dist['score_60_70']} | {dist['score_60_70']/total*100:.1f}% |\\n")
            f.write(f"| <60 | {dist['score_below_60']} | {dist['score_below_60']/total*100:.1f}% |\\n\\n")
            
            # 常见问题
            f.write("## ⚠️ 常见质量问题\\n\\n")
            issues = report['common_issues']
            if issues:
                for issue, count in issues.items():
                    f.write(f"- **{issue}**: {count} 只股票\\n")
            else:
                f.write("暂无发现普遍性质量问题。\\n")
            
            f.write("\\n---\\n")
            f.write("🤖 *Generated by Data Quality Checker*\\n")

def main():
    """测试数据质量检查器"""
    checker = DataQualityChecker()
    
    # 读取验证过的活跃股票列表进行测试
    active_stocks_file = Path(__file__).parent / 'verified_active_stocks_1500.csv'
    if active_stocks_file.exists():
        df_stocks = pd.read_csv(active_stocks_file)
        test_stocks = df_stocks['ts_code'].head(100).tolist()  # 测试前100只
        
        print(f"开始测试前100只股票的数据质量...")
        quality_df = checker.check_stock_quality(test_stocks, '2024-01-01', '2024-12-31')
        
        # 生成报告
        report = checker.generate_quality_report(quality_df)
        
        print("\\n=== 数据质量报告 ===")
        print(f"总股票数: {report['summary']['total_stocks']}")
        print(f"高质量股票: {report['summary']['high_quality_stocks']}")
        print(f"推荐股票数: {report['summary']['recommended_stocks']}")
        print(f"平均质量评分: {report['summary']['avg_quality_score']:.1f}")
        
        print("\\n质量分布:")
        for score_range, count in report['quality_distribution'].items():
            print(f"  {score_range}: {count} 只")
        
        print("\\n常见问题:")
        for issue, count in report['common_issues'].items():
            print(f"  {issue}: {count} 只股票")
        
        # 保存报告
        checker.save_quality_report(quality_df)
        
        # 筛选高质量股票
        high_quality = checker.filter_high_quality_stocks(test_stocks, '2024-01-01', '2024-12-31', 70.0)
        print(f"\\n✅ 筛选出 {len(high_quality)} 只高质量股票")
        
    else:
        print("❌ 未找到活跃股票列表文件")

if __name__ == "__main__":
    main()