#!/usr/bin/env python3
"""
数据访问层 (Data Access Object)
提供高级的数据查询和操作接口
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

try:
    from .database_manager import DatabaseManager
except ImportError:
    from database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class StockDataDAO:
    """股票数据访问对象"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        初始化数据访问对象
        
        Args:
            db_manager: 数据库管理器
        """
        self.db = db_manager
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历（有数据的日期）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日期列表
        """
        query = """
        SELECT DISTINCT trade_date
        FROM daily_quotes
        WHERE trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
        """
        
        result = self.db.execute_query(query, (start_date, end_date))
        return [row['trade_date'] for row in result]
    
    def get_stock_list(self, security_type: Optional[str] = None, 
                      exchange: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            security_type: 证券类型筛选
            exchange: 交易所筛选
            
        Returns:
            股票列表DataFrame
        """
        query = "SELECT code, name, type, exchange FROM securities WHERE is_active = 1"
        params = []
        
        if security_type:
            query += " AND type = ?"
            params.append(security_type)
        
        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)
        
        query += " ORDER BY code"
        
        with self.db.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params if params else None)
    
    def get_stock_data(self, code: str, start_date: str, end_date: str,
                      fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取单个股票的历史数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要的字段列表，None表示所有字段
            
        Returns:
            股票数据DataFrame
        """
        # 默认字段
        if fields is None:
            fields = ['trade_date', 'open', 'high', 'low', 'close', 'volume',
                     'price_change_pct', 'is_limit_up', 'is_limit_down']

        # 白名单校验防止SQL注入
        ALLOWED_FIELDS = {
            'trade_date', 'open', 'high', 'low', 'close', 'volume',
            'price_change_pct', 'is_limit_up', 'is_limit_down', 'amount',
            'turnover_rate', 'pe_ttm', 'pb', 'ps_ttm', 'market_cap',
            'total_mv', 'circ_mv', 'security_id',
        }
        for f in fields:
            if f not in ALLOWED_FIELDS:
                raise ValueError(f"非法字段名: {f}")

        field_str = ', '.join([f"q.{field}" for field in fields])
        
        query = f"""
        SELECT {field_str}
        FROM securities s
        JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.code = ?
            AND q.trade_date >= ?
            AND q.trade_date <= ?
        ORDER BY q.trade_date
        """
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=(code, start_date, end_date))
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
            return df
    
    def get_multiple_stocks_data(self, codes: List[str], start_date: str, end_date: str,
                                fields: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        批量获取多个股票的历史数据
        
        Args:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要的字段列表
            
        Returns:
            {stock_code: DataFrame} 字典
        """
        result = {}
        
        # 使用批量查询优化性能
        if fields is None:
            fields = ['trade_date', 'open', 'high', 'low', 'close', 'volume']
        
        field_str = ', '.join([f"q.{field}" for field in fields])
        code_placeholders = ', '.join(['?' for _ in codes])
        
        query = f"""
        SELECT s.code, {field_str}
        FROM securities s
        JOIN daily_quotes q ON s.id = q.security_id
        WHERE s.code IN ({code_placeholders})
            AND q.trade_date >= ?
            AND q.trade_date <= ?
        ORDER BY s.code, q.trade_date
        """
        
        params = codes + [start_date, end_date]
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 按股票代码分组
            for code in codes:
                stock_data = df[df['code'] == code].copy()
                if not stock_data.empty:
                    stock_data.set_index('trade_date', inplace=True)
                    stock_data.drop('code', axis=1, inplace=True)
                    result[code] = stock_data
        
        return result
    
    def get_market_data_by_date(self, trade_date: str, 
                               security_type: str = "A股") -> pd.DataFrame:
        """
        获取指定日期的全市场数据
        
        Args:
            trade_date: 交易日期
            security_type: 证券类型
            
        Returns:
            当日全市场数据DataFrame
        """
        query = """
        SELECT 
            s.code,
            s.name,
            q.close,
            q.volume,
            q.price_change_pct,
            q.is_limit_up,
            q.is_limit_down
        FROM securities s
        JOIN daily_quotes q ON s.id = q.security_id
        WHERE q.trade_date = ?
            AND s.type = ?
            AND s.is_active = 1
        ORDER BY q.volume DESC
        """
        
        with self.db.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(trade_date, security_type))
    
    def get_stock_signals(self, signal_date: str, strategy_name: Optional[str] = None,
                         top_n: Optional[int] = None) -> pd.DataFrame:
        """
        获取选股信号
        
        Args:
            signal_date: 信号日期
            strategy_name: 策略名称筛选
            top_n: 返回前N个信号
            
        Returns:
            选股信号DataFrame
        """
        query = """
        SELECT 
            s.code,
            s.name,
            sig.strategy_name,
            sig.signal_type,
            sig.comprehensive_score,
            sig.suggested_buy_price,
            sig.stop_loss_price,
            sig.take_profit_price,
            sig.risk_reward_ratio
        FROM stock_signals sig
        JOIN securities s ON sig.security_id = s.id
        WHERE sig.signal_date = ?
        """
        
        params = [signal_date]
        
        if strategy_name:
            query += " AND sig.strategy_name = ?"
            params.append(strategy_name)
        
        query += " ORDER BY sig.comprehensive_score DESC"
        
        if top_n:
            top_n = int(top_n)  # 防止SQL注入
            if top_n < 1 or top_n > 10000:
                raise ValueError(f"top_n 必须在 1-10000 之间: {top_n}")
            query += f" LIMIT {top_n}"
        
        with self.db.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def calculate_technical_indicators(self, code: str, start_date: str, 
                                     end_date: str) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含技术指标的DataFrame
        """
        # 获取基础价格数据
        price_data = self.get_stock_data(code, start_date, end_date)
        
        if price_data.empty:
            return price_data
        
        # 计算移动平均线
        price_data['ma5'] = price_data['close'].rolling(window=5).mean()
        price_data['ma10'] = price_data['close'].rolling(window=10).mean()
        price_data['ma20'] = price_data['close'].rolling(window=20).mean()
        price_data['ma60'] = price_data['close'].rolling(window=60).mean()
        
        # 计算RSI
        price_data['rsi12'] = self._calculate_rsi(price_data['close'], 12)
        
        # 计算MACD
        macd_data = self._calculate_macd(price_data['close'])
        price_data = pd.concat([price_data, macd_data], axis=1)
        
        # 计算KDJ
        kdj_data = self._calculate_kdj(price_data['high'], price_data['low'], price_data['close'])
        price_data = pd.concat([price_data, kdj_data], axis=1)
        
        return price_data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal).mean()
        macd = (dif - dea) * 2
        
        return pd.DataFrame({
            'macd_dif': dif,
            'macd_dea': dea,
            'macd_macd': macd
        })
    
    def _calculate_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9) -> pd.DataFrame:
        """计算KDJ指标"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        rsv = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        rsv = rsv.fillna(0)
        
        k = rsv.ewm(alpha=1/3).mean()
        d = k.ewm(alpha=1/3).mean()
        j = 3 * k - 2 * d
        
        return pd.DataFrame({
            'kdj_k': k,
            'kdj_d': d,
            'kdj_j': j
        })
    
    def save_stock_signals(self, signals: List[Dict]) -> int:
        """
        保存选股信号到数据库
        
        Args:
            signals: 选股信号列表
            
        Returns:
            保存的记录数
        """
        if not signals:
            return 0
        
        query = """
        INSERT OR REPLACE INTO stock_signals (
            signal_date, security_id, strategy_name, signal_type,
            comprehensive_score, suggested_buy_price, stop_loss_price,
            take_profit_price, risk_reward_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        rows_data = []
        for signal in signals:
            # 获取security_id
            security_query = "SELECT id FROM securities WHERE code = ?"
            result = self.db.execute_query(security_query, (signal['stock_code'],))
            
            if not result:
                logger.warning(f"未找到股票代码: {signal['stock_code']}")
                continue
            
            security_id = result[0]['id']
            
            rows_data.append((
                signal['signal_date'],
                security_id,
                signal['strategy_name'],
                signal['signal_type'],
                signal.get('comprehensive_score'),
                signal.get('suggested_buy_price'),
                signal.get('stop_loss_price'),
                signal.get('take_profit_price'),
                signal.get('risk_reward_ratio')
            ))
        
        return self.db.execute_many(query, rows_data)
    
    def get_backtest_summary(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取回测结果汇总
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            回测结果汇总DataFrame
        """
        query = """
        SELECT 
            backtest_id,
            strategy_name,
            start_date,
            end_date,
            initial_capital,
            final_capital,
            total_return,
            annual_return,
            max_drawdown,
            sharpe_ratio,
            total_trades,
            win_rate,
            created_at
        FROM backtest_results
        WHERE start_date >= ? AND end_date <= ?
        ORDER BY created_at DESC
        """
        
        with self.db.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=(start_date, end_date))


class BacktraderDataAdapter:
    """Backtrader数据适配器"""
    
    def __init__(self, dao: StockDataDAO):
        """
        初始化适配器
        
        Args:
            dao: 股票数据访问对象
        """
        self.dao = dao
    
    def get_data_for_backtrader(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取适配backtrader格式的数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Backtrader格式的DataFrame
        """
        data = self.dao.get_stock_data(code, start_date, end_date)
        
        if data.empty:
            return data
        
        # 重命名列以符合backtrader期望
        bt_data = data.rename(columns={
            'trade_date': 'datetime',
            'volume': 'volume'
        })
        
        # 确保包含backtrader需要的列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in bt_data.columns:
                logger.warning(f"缺少必要列: {col}")
                return pd.DataFrame()
        
        # 添加backtrader需要的额外字段
        if 'openinterest' not in bt_data.columns:
            bt_data['openinterest'] = 0
        
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume']:
            bt_data[col] = pd.to_numeric(bt_data[col], errors='coerce')
        
        return bt_data[['open', 'high', 'low', 'close', 'volume', 'openinterest']]


if __name__ == "__main__":
    # 测试数据访问层
    from database_manager import DatabaseManager
    
    db = DatabaseManager()
    dao = StockDataDAO(db)
    
    # 测试获取股票列表
    stocks = dao.get_stock_list("A股")
    print(f"A股数量: {len(stocks)}")
    
    # 测试获取交易日历
    calendar = dao.get_trading_calendar("2024-01-01", "2024-12-31")
    print(f"2024年交易日数: {len(calendar)}")
    
    # 测试获取单个股票数据
    if not stocks.empty:
        test_code = stocks.iloc[0]['code']
        data = dao.get_stock_data(test_code, "2024-01-01", "2024-12-31")
        print(f"股票 {test_code} 数据量: {len(data)}")
        
        # 测试技术指标计算
        tech_data = dao.calculate_technical_indicators(test_code, "2024-01-01", "2024-12-31")
        print(f"技术指标数据量: {len(tech_data)}")
        print(f"可用指标: {list(tech_data.columns)}")