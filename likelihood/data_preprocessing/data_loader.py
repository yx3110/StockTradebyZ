"""
数据加载器模块
负责从SQLite数据库加载股票数据并进行预处理
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, db_path: str = None, config: Dict[str, Any] = None):
        """
        初始化数据加载器
        
        Args:
            db_path: 数据库路径
            config: 配置字典
        """
        self.config = config or {}
        
        # 设置数据库路径
        if db_path:
            self.db_path = Path(db_path)
        else:
            # 尝试从配置或默认路径加载
            db_path = self.config.get('data', {}).get('database_path', '../data_adapter/stock_data.db')
            self.db_path = Path(db_path)
            if not self.db_path.exists():
                self.db_path = Path(__file__).parent.parent.parent / 'data_adapter' / 'stock_data.db'
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")
        
        logger.info(f"使用数据库: {self.db_path}")
        
    def get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        return sqlite3.connect(str(self.db_path))
    
    def load_stock_data(self, 
                       stock_code: str,
                       start_date: str,
                       end_date: str,
                       data_types: List[str] = None) -> pd.DataFrame:
        """
        加载股票数据
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            data_types: 数据类型列表 ['daily_quotes', 'daily_basic', 'technical_indicators']
        
        Returns:
            包含所有数据的DataFrame
        """
        if data_types is None:
            data_types = self.config.get('data', {}).get('data_types', 
                                        ['daily_quotes', 'daily_basic', 'technical_indicators'])
        
        logger.info(f"加载股票 {stock_code} 从 {start_date} 到 {end_date} 的数据")
        
        with self.get_connection() as conn:
            # 首先获取证券ID
            security_id = self._get_security_id(conn, stock_code)
            if security_id is None:
                raise ValueError(f"找不到股票代码: {stock_code}")
            
            # 构建查询SQL
            query = self._build_query(data_types)
            
            # 执行查询
            params = {
                'security_id': security_id,
                'start_date': start_date,
                'end_date': end_date
            }
            
            df = pd.read_sql_query(query, conn, params=params)
            
            # 转换日期格式
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 设置索引
            df.set_index('trade_date', inplace=True)
            
            # 按日期排序
            df.sort_index(inplace=True)
            
            logger.info(f"成功加载 {len(df)} 条数据记录")
            
            return df
    
    def _get_security_id(self, conn: sqlite3.Connection, stock_code: str) -> Optional[int]:
        """获取证券ID"""
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM securities WHERE code = ?", (stock_code,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _build_query(self, data_types: List[str]) -> str:
        """构建查询SQL"""
        base_query = """
        SELECT 
            dq.trade_date,
            dq.open, dq.high, dq.low, dq.close,
            dq.volume, dq.price_change_pct,
            dq.is_limit_up, dq.is_limit_down
        """
        
        join_clauses = []
        select_clauses = []
        
        if 'daily_basic' in data_types:
            select_clauses.append("""
                db.turnover_rate, db.pe_ttm, db.pb,
                db.ps_ttm, db.total_mv, db.circ_mv
            """)
            join_clauses.append("""
                LEFT JOIN daily_basic db 
                ON dq.security_id = db.security_id 
                AND dq.trade_date = db.trade_date
            """)
        
        if 'technical_indicators' in data_types:
            select_clauses.append("""
                ti.kdj_k, ti.kdj_d, ti.kdj_j,
                ti.macd_dif, ti.macd_dea, ti.macd_macd,
                ti.rsi6, ti.rsi12, ti.rsi24,
                ti.boll_upper, ti.boll_middle, ti.boll_lower,
                ti.bbi, ti.volume_ma5, ti.volume_ma10, ti.volume_ratio
            """)
            join_clauses.append("""
                LEFT JOIN technical_indicators ti 
                ON dq.security_id = ti.security_id 
                AND dq.trade_date = ti.trade_date
            """)
        
        # 组合查询
        if select_clauses:
            base_query += ", " + ", ".join(select_clauses)
        
        base_query += """
        FROM daily_quotes dq
        """
        
        if join_clauses:
            base_query += "\n".join(join_clauses)
        
        base_query += """
        WHERE dq.security_id = :security_id
            AND dq.trade_date BETWEEN :start_date AND :end_date
        ORDER BY dq.trade_date
        """
        
        return base_query
    
    def load_multiple_stocks(self,
                           stock_codes: List[str],
                           start_date: str,
                           end_date: str) -> Dict[str, pd.DataFrame]:
        """
        批量加载多只股票数据
        
        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            字典，key为股票代码，value为DataFrame
        """
        results = {}
        
        for code in stock_codes:
            try:
                df = self.load_stock_data(code, start_date, end_date)
                results[code] = df
            except Exception as e:
                logger.warning(f"加载股票 {code} 失败: {str(e)}")
                continue
        
        return results
    
    def normalize_prices(self, df: pd.DataFrame, method: str = 'log_return') -> pd.DataFrame:
        """
        价格标准化
        
        Args:
            df: 原始数据
            method: 标准化方法 ('log_return', 'simple_return', 'z_score')
        
        Returns:
            标准化后的数据
        """
        df = df.copy()
        
        price_cols = ['open', 'high', 'low', 'close']
        existing_cols = [col for col in price_cols if col in df.columns]
        
        if method == 'log_return':
            # 对数收益率
            for col in existing_cols:
                df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
        
        elif method == 'simple_return':
            # 简单收益率
            for col in existing_cols:
                df[f'{col}_return'] = df[col].pct_change()
        
        elif method == 'z_score':
            # Z-score标准化
            for col in existing_cols:
                mean = df[col].mean()
                std = df[col].std()
                df[f'{col}_zscore'] = (df[col] - mean) / std
        
        # 删除第一行NaN
        df = df.iloc[1:]
        
        return df
    
    def normalize_volume(self, df: pd.DataFrame, method: str = 'turnover_rate') -> pd.DataFrame:
        """
        成交量标准化
        
        Args:
            df: 原始数据
            method: 标准化方法 ('turnover_rate', 'log_change', 'relative')
        
        Returns:
            标准化后的数据
        """
        df = df.copy()
        
        if 'volume' not in df.columns:
            return df
        
        if method == 'turnover_rate':
            # 使用换手率（如果有）
            if 'turnover_rate' not in df.columns:
                logger.warning("缺少换手率数据，使用对数变化代替")
                method = 'log_change'
        
        if method == 'log_change':
            # 对数变化
            df['volume_log_change'] = np.log(df['volume'] / df['volume'].shift(1))
        
        elif method == 'relative':
            # 相对成交量（与移动平均比较）
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_relative'] = df['volume'] / df['volume_ma20']
        
        return df
    
    def handle_missing_data(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        处理缺失数据
        
        Args:
            df: 原始数据
            method: 处理方法 ('forward_fill', 'backward_fill', 'interpolate', 'drop')
        
        Returns:
            处理后的数据
        """
        df = df.copy()
        
        if method == 'forward_fill':
            df.fillna(method='ffill', inplace=True)
        elif method == 'backward_fill':
            df.fillna(method='bfill', inplace=True)
        elif method == 'interpolate':
            df.interpolate(method='linear', inplace=True)
        elif method == 'drop':
            df.dropna(inplace=True)
        
        return df
    
    def create_sliding_windows(self, 
                             df: pd.DataFrame,
                             window_length: int,
                             step_size: int = 1) -> List[pd.DataFrame]:
        """
        创建滑动窗口
        
        Args:
            df: 原始数据
            window_length: 窗口长度
            step_size: 步进大小
        
        Returns:
            窗口列表
        """
        windows = []
        
        for i in range(0, len(df) - window_length + 1, step_size):
            window = df.iloc[i:i + window_length]
            windows.append(window)
        
        return windows
    
    def get_market_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            交易日列表
        """
        with self.get_connection() as conn:
            query = """
            SELECT DISTINCT trade_date 
            FROM daily_quotes 
            WHERE trade_date BETWEEN ? AND ?
            ORDER BY trade_date
            """
            
            start = start_date.replace('-', '')
            end = end_date.replace('-', '')
            
            cursor = conn.cursor()
            cursor.execute(query, (start, end))
            
            dates = [row[0] for row in cursor.fetchall()]
            
            return dates
    
    def filter_stocks_by_criteria(self,
                                 min_volume: float = None,
                                 min_market_cap: float = None,
                                 industries: List[str] = None,
                                 date: str = None) -> List[str]:
        """
        根据条件筛选股票
        
        Args:
            min_volume: 最小成交额
            min_market_cap: 最小市值
            industries: 行业列表
            date: 筛选日期
        
        Returns:
            符合条件的股票代码列表
        """
        with self.get_connection() as conn:
            query = """
            SELECT DISTINCT s.code
            FROM securities s
            """
            
            conditions = ["s.type = 'A股'"]
            params = []
            
            if min_volume or min_market_cap:
                query += """
                JOIN daily_quotes dq ON s.id = dq.security_id
                JOIN daily_basic db ON dq.security_id = db.security_id 
                    AND dq.trade_date = db.trade_date
                """
                
                if date:
                    conditions.append("dq.trade_date = ?")
                    params.append(date.replace('-', ''))
                
                if min_volume:
                    conditions.append("dq.volume * dq.close > ?")
                    params.append(min_volume)
                
                if min_market_cap:
                    conditions.append("db.total_mv > ?")
                    params.append(min_market_cap)
            
            if industries:
                query += """
                LEFT JOIN stock_basic_info sbi ON s.code = sbi.code
                """
                placeholders = ','.join(['?' for _ in industries])
                conditions.append(f"sbi.industry IN ({placeholders})")
                params.extend(industries)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            stock_codes = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"筛选出 {len(stock_codes)} 只符合条件的股票")
            
            return stock_codes


if __name__ == '__main__':
    # 测试代码
    loader = DataLoader()
    
    # 测试加载单只股票数据
    df = loader.load_stock_data('000001', '2025-07-01', '2025-08-08')
    print(f"加载数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    print(f"数据示例:\n{df.head()}")
    
    # 测试价格标准化
    df_normalized = loader.normalize_prices(df)
    print(f"\n标准化后的数据列: {df_normalized.columns.tolist()}")
    
    # 测试筛选股票
    stocks = loader.filter_stocks_by_criteria(
        min_volume=100000000,
        date='2025-08-08'
    )
    print(f"\n符合条件的股票数量: {len(stocks)}")
    print(f"前10只股票: {stocks[:10]}")