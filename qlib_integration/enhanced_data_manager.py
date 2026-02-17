#!/usr/bin/env python3
"""
增强版数据管理器
实现多时间段数据分割、数据质量检查和样本扩展功能
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

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

class EnhancedDataManager:
    """增强版数据管理器 - 支持大规模数据和多时间段分割"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(project_root, 'data_adapter/stock_data.db')
        self.active_stocks_file = Path(__file__).parent / 'active_stocks_1500.csv'
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 时间段定义
        self.periods = {
            'training': ('2022-01-01', '2024-06-30'),
            'validation': ('2024-07-01', '2024-12-31'), 
            'testing': ('2025-01-01', '2025-09-08')
        }
        
        # 加载活跃股票列表
        self._load_active_stocks()
    
    def _load_active_stocks(self) -> None:
        """加载活跃股票列表"""
        if self.active_stocks_file.exists():
            self.active_stocks = pd.read_csv(self.active_stocks_file)
            self.logger.info(f"✅ 加载了 {len(self.active_stocks)} 只活跃股票")
        else:
            self.logger.error(f"❌ 活跃股票文件不存在: {self.active_stocks_file}")
            self.active_stocks = pd.DataFrame()
    
    def get_period_data(self, period: str, features: List[str] = None) -> pd.DataFrame:
        """
        获取指定时间段的数据
        
        Args:
            period: 'training', 'validation', 'testing'
            features: 需要的特征列表，None表示获取所有可用特征
        """
        if period not in self.periods:
            raise ValueError(f"期间必须是: {list(self.periods.keys())}")
        
        start_date, end_date = self.periods[period]
        
        conn = sqlite3.connect(self.db_path)
        
        # 构建基础查询
        ts_codes = "', '".join(self.active_stocks['ts_code'].tolist())
        
        base_query = f"""
        SELECT 
            sbi.ts_code,
            dq.trade_date,
            dq.open, dq.high, dq.low, dq.close, dq.volume,
            dq.price_change_pct,
            dq.is_limit_up, dq.is_limit_down
        FROM daily_quotes dq
        JOIN stock_basic_info sbi ON dq.security_id = sbi.security_id
        WHERE sbi.ts_code IN ('{ts_codes}')
        AND dq.trade_date >= '{start_date}'
        AND dq.trade_date <= '{end_date}'
        AND dq.volume > 0 
        AND dq.is_suspend = 0
        ORDER BY sbi.ts_code, dq.trade_date
        """
        
        self.logger.info(f"📊 获取{period}期间数据: {start_date} 到 {end_date}")
        df = pd.read_sql_query(base_query, conn)
        
        # 如果需要技术指标，添加技术指标查询
        if features and any('kdj' in f or 'rsi' in f or 'bbi' in f for f in features):
            tech_query = f"""
            SELECT 
                sbi.ts_code,
                ti.trade_date,
                ti.kdj_k, ti.kdj_d, ti.kdj_j,
                ti.rsi6, ti.rsi12, ti.rsi24,
                ti.bbi
            FROM technical_indicators ti
            JOIN stock_basic_info sbi ON ti.security_id = sbi.security_id
            WHERE sbi.ts_code IN ('{ts_codes}')
            AND ti.trade_date >= '{start_date}'
            AND ti.trade_date <= '{end_date}'
            """
            
            df_tech = pd.read_sql_query(tech_query, conn)
            df = df.merge(df_tech, on=['ts_code', 'trade_date'], how='left')
        
        # 如果需要基本面数据，添加基本面查询
        if features and any('pe' in f or 'pb' in f or 'market_cap' in f for f in features):
            fundamental_query = f"""
            SELECT 
                sbi.ts_code,
                db.trade_date,
                db.pe_ttm, db.pb, db.total_mv as market_cap,
                db.turnover_rate
            FROM daily_basic db
            JOIN stock_basic_info sbi ON db.security_id = sbi.security_id
            WHERE sbi.ts_code IN ('{ts_codes}')
            AND db.trade_date >= '{start_date}'
            AND db.trade_date <= '{end_date}'
            """
            
            df_fundamental = pd.read_sql_query(fundamental_query, conn)
            df = df.merge(df_fundamental, on=['ts_code', 'trade_date'], how='left')
        
        conn.close()
        
        self.logger.info(f"✅ {period}期间数据获取完成: {len(df)} 条记录, {len(df['ts_code'].unique())} 只股票")
        return df
    
    def calculate_future_returns(self, df: pd.DataFrame, periods: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        """
        计算未来收益率
        
        Args:
            df: 包含价格数据的DataFrame
            periods: 收益率计算周期列表
        """
        self.logger.info(f"📈 计算未来收益率: {periods}日")
        
        df_returns = df.copy()
        df_returns = df_returns.sort_values(['ts_code', 'trade_date'])
        
        for period in periods:
            returns = []
            for ts_code in df_returns['ts_code'].unique():
                stock_data = df_returns[df_returns['ts_code'] == ts_code].copy()
                stock_data[f'future_return_{period}d'] = stock_data['close'].shift(-period) / stock_data['close'] - 1
                returns.append(stock_data)
            
            if returns:
                df_returns = pd.concat(returns, ignore_index=True)
        
        # 移除最后几天没有未来收益率的数据
        max_period = max(periods)
        df_returns = df_returns.dropna(subset=[f'future_return_{max_period}d'])
        
        self.logger.info(f"✅ 未来收益率计算完成: {len(df_returns)} 条有效记录")
        return df_returns
    
    def check_data_quality(self, period: str = None) -> Dict:
        """
        数据质量检查
        
        Args:
            period: 检查的时间段，None表示检查所有时间段
        """
        self.logger.info("🔍 开始数据质量检查...")
        
        quality_report = {}
        periods_to_check = [period] if period else list(self.periods.keys())
        
        for p in periods_to_check:
            self.logger.info(f"检查{p}期间数据质量...")
            
            df = self.get_period_data(p)
            
            # 基础统计
            stats = {
                'total_records': len(df),
                'unique_stocks': len(df['ts_code'].unique()),
                'unique_dates': len(df['trade_date'].unique()),
                'date_range': f"{df['trade_date'].min()} 到 {df['trade_date'].max()}",
            }
            
            # 缺失值检查
            missing_stats = {}
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    missing_stats[col] = {
                        'count': missing_count,
                        'percentage': missing_count / len(df) * 100
                    }
            
            # 数据完整性检查
            expected_records = stats['unique_stocks'] * stats['unique_dates']
            completeness = stats['total_records'] / expected_records * 100
            
            # 异常值检查
            anomalies = {}
            if 'volume' in df.columns:
                zero_volume = (df['volume'] == 0).sum()
                anomalies['zero_volume_days'] = zero_volume
            
            if 'price_change_pct' in df.columns:
                # 处理空值
                price_changes = df['price_change_pct'].dropna()
                if len(price_changes) > 0:
                    extreme_changes = (abs(price_changes) > 0.2).sum()
                    anomalies['extreme_price_changes'] = extreme_changes
                else:
                    anomalies['extreme_price_changes'] = 0
            
            quality_report[p] = {
                'basic_stats': stats,
                'missing_values': missing_stats,
                'completeness_percentage': round(completeness, 2),
                'anomalies': anomalies
            }
        
        self.logger.info("✅ 数据质量检查完成")
        return quality_report
    
    def get_balanced_sample(self, df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
        """
        获取行业平衡的样本
        
        Args:
            df: 原始数据
            sample_size: 目标样本大小
        """
        if len(df['ts_code'].unique()) <= sample_size:
            return df
        
        # 获取行业分布
        conn = sqlite3.connect(self.db_path)
        industry_query = """
        SELECT sbi.ts_code, sbi.market
        FROM stock_basic_info sbi
        """
        df_industry = pd.read_sql_query(industry_query, conn)
        conn.close()
        
        # 按市场分配样本
        market_counts = df_industry['market'].value_counts()
        market_ratios = market_counts / market_counts.sum()
        
        selected_stocks = []
        for market, ratio in market_ratios.items():
            market_stocks = df_industry[df_industry['market'] == market]['ts_code'].tolist()
            market_sample_size = int(sample_size * ratio)
            
            # 确保不超过可用股票数
            market_sample_size = min(market_sample_size, len(market_stocks))
            selected_market_stocks = np.random.choice(market_stocks, market_sample_size, replace=False)
            selected_stocks.extend(selected_market_stocks)
        
        # 如果样本不足，随机补充
        if len(selected_stocks) < sample_size:
            remaining_stocks = list(set(df['ts_code'].unique()) - set(selected_stocks))
            additional_needed = sample_size - len(selected_stocks)
            additional_stocks = np.random.choice(remaining_stocks, 
                                               min(additional_needed, len(remaining_stocks)), 
                                               replace=False)
            selected_stocks.extend(additional_stocks)
        
        balanced_df = df[df['ts_code'].isin(selected_stocks)]
        
        self.logger.info(f"✅ 获得平衡样本: {len(balanced_df['ts_code'].unique())} 只股票")
        return balanced_df
    
    def prepare_training_data(self, features: List[str], 
                            target_periods: List[int] = [1, 3, 5, 10],
                            balance_sample: bool = False,
                            sample_size: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        准备训练数据
        
        Args:
            features: 特征列表
            target_periods: 目标收益率周期
            balance_sample: 是否平衡样本
            sample_size: 样本大小（如果balance_sample=True）
        """
        self.logger.info("🚀 开始准备训练数据...")
        
        data_splits = {}
        
        for period in ['training', 'validation', 'testing']:
            self.logger.info(f"处理{period}期间数据...")
            
            # 获取数据
            df = self.get_period_data(period, features)
            
            # 计算未来收益率
            df = self.calculate_future_returns(df, target_periods)
            
            # 平衡样本（如果需要）
            if balance_sample and period == 'training':
                df = self.get_balanced_sample(df, sample_size)
            
            data_splits[period] = df
        
        self.logger.info("✅ 训练数据准备完成")
        return data_splits
    
    def save_data_splits(self, data_splits: Dict[str, pd.DataFrame], output_dir: str = None) -> None:
        """保存数据分割结果"""
        if output_dir is None:
            output_dir = Path(__file__).parent / 'data_splits'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for period, df in data_splits.items():
            output_file = output_dir / f'{period}_data.csv'
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"✅ {period}数据已保存: {output_file}")

def main():
    """测试数据管理器"""
    dm = EnhancedDataManager()
    
    # 数据质量检查
    quality_report = dm.check_data_quality()
    print("\n=== 数据质量报告 ===")
    for period, report in quality_report.items():
        print(f"\n{period.upper()}期间:")
        print(f"  记录总数: {report['basic_stats']['total_records']:,}")
        print(f"  股票数量: {report['basic_stats']['unique_stocks']}")
        print(f"  交易日数: {report['basic_stats']['unique_dates']}")
        print(f"  数据完整性: {report['completeness_percentage']}%")
        
        if report['missing_values']:
            print("  缺失值:")
            for col, stats in report['missing_values'].items():
                print(f"    {col}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    # 准备训练数据示例
    features = ['kdj_k', 'kdj_d', 'rsi6', 'bbi', 'pe_ttm', 'pb', 'market_cap']
    data_splits = dm.prepare_training_data(
        features=features,
        target_periods=[1, 3, 5, 10],
        balance_sample=True,
        sample_size=500  # 测试用较小样本
    )
    
    print("\n=== 数据分割结果 ===")
    for period, df in data_splits.items():
        print(f"{period}: {len(df):,} 记录, {len(df['ts_code'].unique())} 只股票")
    
    # 保存数据
    dm.save_data_splits(data_splits)

if __name__ == "__main__":
    main()