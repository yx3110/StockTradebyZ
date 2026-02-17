#!/usr/bin/env python3
"""
因子计算器 - 批量预计算和存储各类因子
支持增量更新和历史回填
"""

import numpy as np
import pandas as pd
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from data_adapter.database_manager import DatabaseManager
from scoring_improvements.squeeze_momentum_calculator import SqueezeMomentumCalculator

class FactorCalculator:
    """统一的因子计算器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        """初始化因子计算器"""
        self.db_manager = DatabaseManager(db_path)
        self.db_path = db_path
        self.squeeze_calculator = SqueezeMomentumCalculator()
        self.logger = self._setup_logger()
        
        # 初始化因子数据库表
        self._init_factor_tables()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("FactorCalculator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _init_factor_tables(self):
        """初始化因子表结构"""
        schema_path = Path(__file__).parent / "factor_database_schema.sql"
        
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # 执行schema创建
                for statement in schema_sql.split(';'):
                    if statement.strip():
                        try:
                            cursor.execute(statement)
                        except sqlite3.OperationalError as e:
                            if "already exists" not in str(e):
                                self.logger.error(f"执行SQL失败: {e}")
                conn.commit()
                
            self.logger.info("因子表结构初始化完成")
    
    def calculate_technical_factors(self, security_id: int, 
                                   start_date: str, 
                                   end_date: str) -> pd.DataFrame:
        """计算技术因子"""
        
        # 获取原始数据
        query = """
        SELECT 
            dq.trade_date, dq.open, dq.high, dq.low, dq.close, dq.volume,
            dq.price_change_pct, dq.ma5, dq.ma10, dq.ma20, dq.ma60,
            ti.kdj_k, ti.kdj_d, ti.kdj_j, ti.rsi12, ti.bbi,
            ti.macd, ti.macd_signal, ti.macd_histogram
        FROM daily_quotes dq
        LEFT JOIN technical_indicators ti 
            ON dq.security_id = ti.security_id 
            AND dq.trade_date = ti.trade_date
        WHERE dq.security_id = ?
        AND dq.trade_date BETWEEN ? AND ?
        ORDER BY dq.trade_date
        """
        
        with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[security_id, start_date, end_date])
        
        if df.empty:
            return pd.DataFrame()
        
        # 计算动量因子
        df['momentum_5d'] = df['close'].pct_change(5) * 100
        df['momentum_10d'] = df['close'].pct_change(10) * 100
        df['momentum_20d'] = df['close'].pct_change(20) * 100
        df['momentum_60d'] = df['close'].pct_change(60) * 100
        
        # 动量加速度
        df['momentum_acceleration'] = df['momentum_5d'] - df['momentum_10d'].shift(5)
        
        # 均值回归因子
        df['price_to_ma5'] = df['close'] / df['ma5']
        df['price_to_ma20'] = df['close'] / df['ma20']
        df['price_to_ma60'] = df['close'] / df['ma60']
        
        # 计算均值回归得分
        df['mean_reversion_score'] = self._calculate_mean_reversion_score(df)
        
        # 波动率因子
        df['volatility_5d'] = df['price_change_pct'].rolling(5).std()
        df['volatility_20d'] = df['price_change_pct'].rolling(20).std()
        df['volatility_60d'] = df['price_change_pct'].rolling(60).std()
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']
        
        # 成交量因子
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio_5d'] = df['volume'] / df['volume_ma5']
        df['volume_ratio_20d'] = df['volume'] / df['volume_ma20']
        df['volume_momentum'] = df['volume'].pct_change(5)
        df['volume_volatility'] = df['volume'].rolling(20).std() / df['volume_ma20']
        
        # 价格形态因子
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['price_position'] = (df['close'] - df['support_level']) / (df['resistance_level'] - df['support_level'])
        df['breakout_strength'] = np.where(
            df['close'] > df['resistance_level'].shift(1),
            (df['close'] - df['resistance_level'].shift(1)) / df['resistance_level'].shift(1),
            0
        )
        
        # 技术指标衍生因子
        df['rsi_divergence'] = self._calculate_rsi_divergence(df)
        df['macd_histogram_slope'] = df['macd_histogram'].diff()
        df['kdj_golden_cross'] = (df['kdj_k'] > df['kdj_d']) & (df['kdj_k'].shift(1) <= df['kdj_d'].shift(1))
        df['kdj_death_cross'] = (df['kdj_k'] < df['kdj_d']) & (df['kdj_k'].shift(1) >= df['kdj_d'].shift(1))
        df['bbi_trend_strength'] = (df['close'] - df['bbi']) / df['bbi']
        
        # 计算挤压动量因子
        if len(df) >= 30:
            squeeze_factors = self._calculate_squeeze_factors(df)
            df = pd.concat([df, squeeze_factors], axis=1)
        else:
            df['squeeze_state'] = 0
            df['squeeze_duration'] = 0
            df['squeeze_momentum'] = 0
            df['squeeze_momentum_change'] = 0
            df['squeeze_release_signal'] = False
        
        # 添加security_id
        df['security_id'] = security_id
        
        return df
    
    def _calculate_mean_reversion_score(self, df: pd.DataFrame) -> pd.Series:
        """计算均值回归得分"""
        score = pd.Series(index=df.index, dtype=float)
        
        # RSI超卖超买
        rsi_score = np.where(df['rsi12'] < 30, 100 - df['rsi12'], 
                            np.where(df['rsi12'] > 70, 100 - df['rsi12'], 50))
        
        # KDJ超卖超买
        kdj_avg = (df['kdj_k'] + df['kdj_d']) / 2
        kdj_score = np.where(kdj_avg < 20, 100 - kdj_avg,
                            np.where(kdj_avg > 80, 100 - kdj_avg, 50))
        
        # BBI偏离
        bbi_deviation = ((df['close'] - df['bbi']) / df['bbi']) * 100
        bbi_score = 50 - abs(bbi_deviation) * 2
        
        # 综合得分
        score = (rsi_score * 0.4 + kdj_score * 0.4 + bbi_score * 0.2)
        
        return score
    
    def _calculate_rsi_divergence(self, df: pd.DataFrame) -> pd.Series:
        """计算RSI背离信号"""
        divergence = pd.Series(0, index=df.index, dtype=float)
        
        # 简化的背离检测
        price_trend = df['close'].diff(5)
        rsi_trend = df['rsi12'].diff(5)
        
        # 底背离：价格创新低但RSI没有
        bullish_divergence = (price_trend < 0) & (rsi_trend > 0)
        # 顶背离：价格创新高但RSI没有
        bearish_divergence = (price_trend > 0) & (rsi_trend < 0)
        
        divergence[bullish_divergence] = 1
        divergence[bearish_divergence] = -1
        
        return divergence
    
    def _calculate_squeeze_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算挤压动量因子"""
        squeeze_df = pd.DataFrame(index=df.index)
        
        # 使用挤压动量计算器
        indicators = self.squeeze_calculator.calculate_squeeze_momentum_indicators(
            df['high'].values,
            df['low'].values, 
            df['close'].values
        )
        
        # 提取挤压状态
        squeeze_df['squeeze_state'] = indicators['is_squeezed'].astype(int)
        
        # 计算挤压持续时间
        squeeze_duration = []
        duration = 0
        for is_squeezed in indicators['is_squeezed']:
            if is_squeezed:
                duration += 1
            else:
                duration = 0
            squeeze_duration.append(duration)
        squeeze_df['squeeze_duration'] = squeeze_duration
        
        # 动量值和变化
        squeeze_df['squeeze_momentum'] = indicators['momentum']
        squeeze_df['squeeze_momentum_change'] = indicators['momentum'].diff()
        
        # 释放信号
        squeeze_df['squeeze_release_signal'] = (
            (squeeze_df['squeeze_state'] == 0) & 
            (squeeze_df['squeeze_state'].shift(1) == 1)
        )
        
        return squeeze_df
    
    def calculate_market_factors(self, security_id: int,
                                start_date: str,
                                end_date: str) -> pd.DataFrame:
        """计算市场因子"""
        
        # 获取个股和大盘数据
        query = """
        SELECT 
            dq.trade_date,
            dq.close as stock_close,
            dq.price_change_pct as stock_return,
            idx.close as market_close,
            idx.price_change_pct as market_return
        FROM daily_quotes dq
        LEFT JOIN (
            SELECT trade_date, close, price_change_pct
            FROM daily_quotes 
            WHERE security_id = (SELECT id FROM securities WHERE code = '000001.SH')
        ) idx ON dq.trade_date = idx.trade_date
        WHERE dq.security_id = ?
        AND dq.trade_date BETWEEN ? AND ?
        ORDER BY dq.trade_date
        """
        
        with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=[security_id, start_date, end_date])
        
        if df.empty:
            return pd.DataFrame()
        
        # 计算相对强度
        df['relative_strength_index'] = (
            df['stock_return'].rolling(20).mean() - 
            df['market_return'].rolling(20).mean()
        ) * 100
        
        # 计算阿尔法和贝塔（60日窗口）
        window = 60
        df['alpha_60d'] = np.nan
        df['beta_60d'] = np.nan
        
        for i in range(window, len(df)):
            stock_returns = df['stock_return'].iloc[i-window:i]
            market_returns = df['market_return'].iloc[i-window:i]
            
            if len(stock_returns) == window:
                # 计算贝塔
                covariance = np.cov(stock_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance if market_variance != 0 else 1
                
                # 计算阿尔法
                alpha = stock_returns.mean() - beta * market_returns.mean()
                
                df.loc[df.index[i], 'beta_60d'] = beta
                df.loc[df.index[i], 'alpha_60d'] = alpha * 252  # 年化
        
        # 计算夏普比率
        df['sharpe_ratio_60d'] = (
            df['stock_return'].rolling(60).mean() / 
            df['stock_return'].rolling(60).std()
        ) * np.sqrt(252)
        
        # 与市场的相关性
        df['correlation_with_market'] = (
            df['stock_return'].rolling(60).corr(df['market_return'])
        )
        
        # 特质波动率
        df['idiosyncratic_volatility'] = (
            df['stock_return'].rolling(20).std() * 
            np.sqrt(1 - df['correlation_with_market']**2)
        )
        
        # 添加security_id
        df['security_id'] = security_id
        
        return df
    
    def backfill_factors(self, start_date: str, end_date: str, 
                        security_codes: Optional[List[str]] = None,
                        batch_size: int = 100):
        """批量回填历史因子数据"""
        
        self.logger.info(f"开始回填因子数据: {start_date} 到 {end_date}")
        
        # 获取需要处理的股票列表
        if security_codes:
            placeholders = ','.join(['?'] * len(security_codes))
            query = f"""
            SELECT id, code, name FROM securities 
            WHERE code IN ({placeholders}) AND type = 'A股'
            """
            params = security_codes
        else:
            query = "SELECT id, code, name FROM securities WHERE type = 'A股'"
            params = []
        
        with self.db_manager.get_connection() as conn:
            securities = pd.read_sql_query(query, conn, params=params)
        
        self.logger.info(f"共需处理 {len(securities)} 只股票")
        
        # 批量处理
        total_processed = 0
        errors = []
        
        for batch_start in range(0, len(securities), batch_size):
            batch_end = min(batch_start + batch_size, len(securities))
            batch = securities.iloc[batch_start:batch_end]
            
            self.logger.info(f"处理批次 {batch_start//batch_size + 1}: "
                           f"股票 {batch_start+1} 到 {batch_end}")
            
            for _, security in tqdm(batch.iterrows(), total=len(batch)):
                try:
                    # 计算技术因子
                    tech_factors = self.calculate_technical_factors(
                        security['id'], start_date, end_date
                    )
                    
                    # 计算市场因子
                    market_factors = self.calculate_market_factors(
                        security['id'], start_date, end_date
                    )
                    
                    # 保存到数据库
                    if not tech_factors.empty:
                        self._save_technical_factors(tech_factors)
                    
                    if not market_factors.empty:
                        self._save_market_factors(market_factors)
                    
                    total_processed += 1
                    
                except Exception as e:
                    error_msg = f"处理股票 {security['code']} 失败: {e}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
        
        # 记录日志
        self._log_calculation(
            calculation_date=datetime.now().date(),
            factor_category='all',
            securities_processed=total_processed,
            error_count=len(errors),
            error_details='\n'.join(errors) if errors else None
        )
        
        self.logger.info(f"回填完成: 成功处理 {total_processed} 只股票, "
                        f"失败 {len(errors)} 只")
        
        return total_processed, errors
    
    def _save_technical_factors(self, df: pd.DataFrame):
        """保存技术因子到数据库"""
        columns = [
            'security_id', 'trade_date', 'momentum_5d', 'momentum_10d',
            'momentum_20d', 'momentum_60d', 'momentum_acceleration',
            'mean_reversion_score', 'price_to_ma5', 'price_to_ma20',
            'price_to_ma60', 'volatility_5d', 'volatility_20d',
            'volatility_60d', 'volatility_ratio', 'volume_ratio_5d',
            'volume_ratio_20d', 'volume_momentum', 'volume_volatility',
            'support_level', 'resistance_level', 'price_position',
            'breakout_strength', 'rsi_divergence', 'macd_histogram_slope',
            'kdj_golden_cross', 'kdj_death_cross', 'bbi_trend_strength',
            'squeeze_state', 'squeeze_duration', 'squeeze_momentum',
            'squeeze_momentum_change', 'squeeze_release_signal'
        ]
        
        # 只保留需要的列
        save_df = df[columns].copy()
        
        # 替换或插入数据
        with self.db_manager.get_connection() as conn:
            save_df.to_sql('technical_factors', conn, if_exists='replace', index=False)
    
    def _save_market_factors(self, df: pd.DataFrame):
        """保存市场因子到数据库"""
        columns = [
            'security_id', 'trade_date', 'relative_strength_index',
            'alpha_60d', 'beta_60d', 'sharpe_ratio_60d',
            'correlation_with_market', 'idiosyncratic_volatility'
        ]
        
        # 只保留需要的列
        save_df = df[columns].copy()
        
        # 替换或插入数据
        with self.db_manager.get_connection() as conn:
            save_df.to_sql('market_factors', conn, if_exists='replace', index=False)
    
    def _log_calculation(self, **kwargs):
        """记录计算日志"""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO factor_calculation_log 
                (calculation_date, factor_category, securities_processed, 
                 error_count, error_details, status)
                VALUES (?, ?, ?, ?, ?, 'completed')
            """, (
                kwargs.get('calculation_date'),
                kwargs.get('factor_category'),
                kwargs.get('securities_processed'),
                kwargs.get('error_count'),
                kwargs.get('error_details')
            ))
            conn.commit()
    
    def update_daily_factors(self, date: str = None):
        """每日更新因子（增量更新）"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"更新 {date} 的因子数据")
        
        # 只更新当天的数据
        processed, errors = self.backfill_factors(date, date)
        
        return processed, errors


def main():
    """主函数：演示因子计算和回填"""
    import argparse
    
    parser = argparse.ArgumentParser(description="因子计算和回填工具")
    parser.add_argument('--mode', choices=['backfill', 'update'], 
                       default='update', help='运行模式')
    parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--stocks', nargs='+', help='指定股票代码列表')
    parser.add_argument('--batch-size', type=int, default=100, 
                       help='批处理大小')
    
    args = parser.parse_args()
    
    # 初始化计算器
    calculator = FactorCalculator()
    
    if args.mode == 'backfill':
        # 回填历史数据
        if not args.start_date or not args.end_date:
            print("回填模式需要指定 --start-date 和 --end-date")
            return
        
        processed, errors = calculator.backfill_factors(
            args.start_date, 
            args.end_date,
            args.stocks,
            args.batch_size
        )
        
        print(f"回填完成: 处理 {processed} 只股票, 错误 {len(errors)} 个")
        
    else:
        # 更新当日数据
        date = args.end_date if args.end_date else None
        processed, errors = calculator.update_daily_factors(date)
        
        print(f"更新完成: 处理 {processed} 只股票, 错误 {len(errors)} 个")


if __name__ == "__main__":
    main()