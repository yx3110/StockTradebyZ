"""
数据适配器：将StockTradebyZ的SQLite数据库适配到Qlib格式

主要功能：
1. 从SQLite数据库读取市场数据（daily_quotes, technical_indicators等）
2. 转换为Qlib DataHandler兼容格式
3. 支持股票池过滤和时间范围选择
4. 处理中国A股特有数据（涨跌停、停牌等）
"""

import os
import sys
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from data_adapter.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class StockTradebyzDataAdapter:
    """
    StockTradebyZ数据适配器
    
    将SQLite数据库中的股票数据转换为Qlib回测框架可用的格式
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        初始化数据适配器
        
        Args:
            db_path: 数据库路径，默认使用data_adapter目录下的stock_data.db
        """
        if db_path is None:
            db_path = os.path.join(project_root, "data_adapter", "stock_data.db")
        
        self.db_path = db_path
        self.db_manager = DatabaseManager()
        
        # 验证数据库连接
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
        
        logger.info(f"数据适配器初始化完成，数据库路径: {self.db_path}")
    
    def get_stock_list(self, 
                      start_date: str,
                      end_date: str,
                      min_trading_days: int = 100) -> List[str]:
        """
        获取在指定时间范围内有足够交易数据的股票列表
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            min_trading_days: 最少交易天数
            
        Returns:
            股票代码列表
        """
        query = """
        SELECT s.code, COUNT(dq.trade_date) as trading_days
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.type = 'A股' 
        AND dq.trade_date BETWEEN ? AND ?
        AND dq.close IS NOT NULL
        GROUP BY s.code
        HAVING trading_days >= ?
        ORDER BY s.code
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[start_date, end_date, min_trading_days])
        
        stock_list = df['code'].tolist()
        logger.info(f"找到 {len(stock_list)} 只股票，交易天数 >= {min_trading_days}")
        
        return stock_list
    
    def get_market_data(self,
                       instruments: List[str],
                       start_date: str,
                       end_date: str,
                       fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取市场数据
        
        Args:
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期  
            fields: 需要的字段列表，默认包含OHLCV
            
        Returns:
            MultiIndex DataFrame (datetime, instrument) -> 市场数据
        """
        if fields is None:
            fields = ['open', 'high', 'low', 'close', 'volume', 'price_change_pct']
        
        # 构建SQL查询
        field_str = ', '.join([f"dq.{field}" for field in fields])
        instruments_str = "','".join(instruments)
        
        query = f"""
        SELECT 
            s.code as instrument,
            dq.trade_date,
            {field_str}
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id  
        WHERE s.code IN ('{instruments_str}')
        AND dq.trade_date BETWEEN ? AND ?
        AND dq.close IS NOT NULL
        ORDER BY s.code, dq.trade_date
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        
        if df.empty:
            logger.warning("未找到符合条件的市场数据")
            return pd.DataFrame()
        
        # 转换日期格式
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # 设置MultiIndex
        df = df.set_index(['trade_date', 'instrument'])
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(0)
        
        logger.info(f"获取市场数据: {len(instruments)}只股票, "
                   f"{df.index.get_level_values(0).min()} - {df.index.get_level_values(0).max()}")
        
        return df
    
    def get_technical_indicators(self,
                               instruments: List[str], 
                               start_date: str,
                               end_date: str,
                               indicators: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取技术指标数据
        
        Args:
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            indicators: 技术指标列表，默认获取常用指标
            
        Returns:
            MultiIndex DataFrame 包含技术指标
        """
        if indicators is None:
            indicators = ['ma_5', 'ma_10', 'ma_20', 'rsi_14', 'macd', 'kdj_k', 'kdj_d', 'bbi']
        
        # 检查数据库中可用的指标
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(technical_indicators)")
            available_columns = [col[1] for col in cursor.fetchall()]
        
        # 过滤出实际存在的指标
        valid_indicators = [ind for ind in indicators if ind in available_columns]
        if not valid_indicators:
            logger.warning("未找到有效的技术指标列")
            return pd.DataFrame()
        
        field_str = ', '.join([f"ti.{ind}" for ind in valid_indicators])
        instruments_str = "','".join(instruments)
        
        query = f"""
        SELECT 
            s.code as instrument,
            ti.trade_date,
            {field_str}
        FROM securities s
        JOIN technical_indicators ti ON s.id = ti.security_id
        WHERE s.code IN ('{instruments_str}')
        AND ti.trade_date BETWEEN ? AND ?
        ORDER BY s.code, ti.trade_date
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        
        if df.empty:
            logger.warning("未找到技术指标数据")
            return pd.DataFrame()
        
        # 转换日期格式和设置索引
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.set_index(['trade_date', 'instrument'])
        
        # 处理缺失值
        df = df.fillna(method='ffill').fillna(0)
        
        logger.info(f"获取技术指标: {len(valid_indicators)}个指标, {len(instruments)}只股票")
        
        return df
    
    def get_fundamental_data(self,
                           instruments: List[str],
                           start_date: str, 
                           end_date: str) -> pd.DataFrame:
        """
        获取基本面数据
        
        Args:
            instruments: 股票代码列表  
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            MultiIndex DataFrame 包含PE/PB/市值等基本面数据
        """
        instruments_str = "','".join(instruments)
        
        query = f"""
        SELECT 
            s.code as instrument,
            db.trade_date,
            db.pe_ttm,
            db.pb,
            db.ps_ttm,
            db.total_mv as market_cap,
            db.turnover_rate,
            db.total_share,
            db.float_share,
            db.free_share
        FROM securities s
        JOIN daily_basic db ON s.id = db.security_id
        WHERE s.code IN ('{instruments_str}')
        AND db.trade_date BETWEEN ? AND ?
        ORDER BY s.code, db.trade_date
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        
        if df.empty:
            logger.warning("未找到基本面数据")
            return pd.DataFrame()
        
        # 转换日期格式和设置索引
        df['trade_date'] = pd.to_datetime(df['trade_date'])  
        df = df.set_index(['trade_date', 'instrument'])
        
        # 处理缺失值和异常值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        logger.info(f"获取基本面数据: {len(instruments)}只股票")
        
        return df
    
    def create_qlib_dataset(self,
                           instruments: List[str],
                           start_date: str,
                           end_date: str,
                           include_technical: bool = True,
                           include_fundamental: bool = True) -> pd.DataFrame:
        """
        创建Qlib兼容的数据集
        
        Args:
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            include_technical: 是否包含技术指标
            include_fundamental: 是否包含基本面数据
            
        Returns:
            完整的MultiIndex DataFrame适用于Qlib
        """
        logger.info(f"创建Qlib数据集: {len(instruments)}只股票, {start_date} - {end_date}")
        
        # 获取基础市场数据  
        market_data = self.get_market_data(instruments, start_date, end_date)
        
        if market_data.empty:
            logger.error("基础市场数据为空，无法创建数据集")
            return pd.DataFrame()
        
        result = market_data.copy()
        
        # 添加技术指标
        if include_technical:
            tech_data = self.get_technical_indicators(instruments, start_date, end_date)
            if not tech_data.empty:
                result = pd.concat([result, tech_data], axis=1)
                logger.info("已添加技术指标数据")
        
        # 添加基本面数据
        if include_fundamental:
            fund_data = self.get_fundamental_data(instruments, start_date, end_date)
            if not fund_data.empty:
                result = pd.concat([result, fund_data], axis=1)
                logger.info("已添加基本面数据")
        
        # 计算收益率（Qlib回测需要）
        if 'close' in result.columns:
            # 按股票分组计算收益率
            close_prices = result['close'].unstack(level=1)  # 转为股票为列的格式
            returns = close_prices.pct_change().stack()  # 计算收益率并还原多重索引
            returns.name = 'ret'
            result = pd.concat([result, returns], axis=1)
        
        # 删除包含NaN的行
        result = result.dropna(how='all')
        
        logger.info(f"数据集创建完成: {result.shape[0]}行 x {result.shape[1]}列")
        
        return result
    
    def get_tradable_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日列表
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日期列表
        """
        query = """
        SELECT DISTINCT trade_date 
        FROM daily_quotes 
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        
        return df['trade_date'].tolist()
    
    def get_stock_info(self, instruments: List[str]) -> pd.DataFrame:
        """
        获取股票基本信息
        
        Args:
            instruments: 股票代码列表
            
        Returns:
            股票信息DataFrame
        """
        instruments_str = "','".join(instruments)
        
        query = f"""
        SELECT 
            code,
            name,
            industry, 
            area,
            market,
            list_date
        FROM securities 
        WHERE code IN ('{instruments_str}')
        AND type = 'A股'
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
        
        return df