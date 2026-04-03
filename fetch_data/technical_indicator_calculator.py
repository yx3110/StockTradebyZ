#!/usr/bin/env python3
"""
技术指标计算器
填充缺失的技术指标数据到数据库中
解决Claude分析时"缺乏技术指标"的问题
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechnicalIndicatorCalculator:
    """技术指标计算器"""
    
    def __init__(self, db_path: str = "../data_adapter/stock_data.db", start_date: str = None, end_date: str = None, max_workers: int = 4):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        self.start_date = start_date
        self.end_date = end_date
        self.max_workers = max_workers  # 降低并发数
        self.conn = None
        self.thread_local = threading.local()
        self.stats = {
            'processed_stocks': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'start_time': None,
            'end_time': None
        }
        self.stats_lock = threading.Lock()
        self.db_lock = threading.Lock()  # 数据库全局锁
    
    def connect_db(self):
        """连接数据库"""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
    
    def close_db(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager 支持"""
        self.connect_db()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 退出时清理所有连接"""
        self.close_all_connections()
        return False

    def get_thread_connection(self):
        """获取线程本地数据库连接"""
        if not hasattr(self.thread_local, 'conn'):
            self.thread_local.conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self.thread_local.conn.execute("PRAGMA foreign_keys = ON")
            self.thread_local.conn.execute("PRAGMA journal_mode = WAL")
            self.thread_local.conn.execute("PRAGMA synchronous = NORMAL")
            self.thread_local.conn.execute("PRAGMA temp_store = memory")
            self.thread_local.conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
        return self.thread_local.conn

    def close_thread_connection(self):
        """关闭当前线程的数据库连接"""
        if hasattr(self.thread_local, 'conn'):
            try:
                self.thread_local.conn.close()
            except Exception:
                pass
            delattr(self.thread_local, 'conn')

    def close_all_connections(self):
        """关闭主连接和所有已知的线程连接"""
        self.close_db()
        self.close_thread_connection()
    
    def get_stock_list(self, limit: Optional[int] = None) -> List[Tuple]:
        """获取股票列表"""
        query = """
        SELECT id, code, name 
        FROM securities 
        WHERE is_active = 1 AND type IN ('A股', '科创板', '创业板')
        ORDER BY code
        """
        params = []
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
    
    def get_stock_price_data(self, security_id: int, days: int = 250, conn=None) -> Optional[pd.DataFrame]:
        """获取股票价格数据"""
        if conn is None:
            conn = self.get_thread_connection()
            
        if self.end_date:
            # 如果指定了结束日期，获取该日期之前的数据
            query = """
            SELECT trade_date, open, high, low, close, volume
            FROM daily_quotes 
            WHERE security_id = ? AND trade_date <= ?
            ORDER BY trade_date DESC
            LIMIT ?
            """
            params = (security_id, self.end_date, days)
        else:
            query = """
            SELECT trade_date, open, high, low, close, volume
            FROM daily_quotes 
            WHERE security_id = ? 
            ORDER BY trade_date DESC
            LIMIT ?
            """
            params = (security_id, days)
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            if df.empty:
                return None
            
            # 反向排序，确保日期从早到晚
            df = df.sort_values('trade_date')
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        
        except Exception as e:
            logger.error(f"获取股票价格数据失败 (security_id: {security_id}): {e}")
            return None
    
    def sma(self, data: np.array, period: int) -> np.array:
        """计算简单移动平均线"""
        return pd.Series(data).rolling(window=period).mean().values
    
    def ema(self, data: np.array, period: int) -> np.array:
        """计算指数移动平均线"""
        return pd.Series(data).ewm(span=period).mean().values
    
    def calculate_kdj(self, high: np.array, low: np.array, close: np.array, period: int = 9) -> tuple:
        """计算KDJ指标"""
        try:
            # 计算最低价和最高价的滚动窗口
            df = pd.DataFrame({'high': high, 'low': low, 'close': close})
            
            # 计算RSV值 (Raw Stochastic Value)
            df['lowest_low'] = df['low'].rolling(window=period).min()
            df['highest_high'] = df['high'].rolling(window=period).max()
            df['rsv'] = (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']) * 100
            
            # 计算K值 (RSV的移动平均)
            df['k'] = df['rsv'].ewm(alpha=1/3).mean()
            
            # 计算D值 (K值的移动平均)
            df['d'] = df['k'].ewm(alpha=1/3).mean()
            
            # 计算J值
            df['j'] = 3 * df['k'] - 2 * df['d']
            
            return df['k'].values, df['d'].values, df['j'].values
        except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
            logger.debug(f"计算异常: {e}")
            return np.full(len(high), np.nan), np.full(len(high), np.nan), np.full(len(high), np.nan)

    def calculate_macd(self, close: np.array, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """计算MACD指标"""
        try:
            # 计算快慢EMA
            ema_fast = self.ema(close, fast)
            ema_slow = self.ema(close, slow)
            
            # 计算DIF (MACD线)
            dif = ema_fast - ema_slow
            
            # 计算DEA (信号线)
            dea = self.ema(dif, signal)
            
            # 计算MACD柱状图
            macd = (dif - dea) * 2
            
            return dif, dea, macd
        except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
            logger.debug(f"计算异常: {e}")
            return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)

    def calculate_rsi(self, close: np.array, period: int = 14) -> np.array:
        """计算RSI指标"""
        try:
            df = pd.DataFrame({'close': close})
            df['change'] = df['close'].diff()
            df['gain'] = df['change'].where(df['change'] > 0, 0)
            df['loss'] = -df['change'].where(df['change'] < 0, 0)
            
            # 计算平均涨幅和跌幅
            df['avg_gain'] = df['gain'].rolling(window=period).mean()
            df['avg_loss'] = df['loss'].rolling(window=period).mean()
            
            # 计算RS和RSI
            df['rs'] = df['avg_gain'] / df['avg_loss']
            df['rsi'] = 100 - (100 / (1 + df['rs']))
            
            return df['rsi'].values
        except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
            logger.debug(f"计算异常: {e}")
            return np.full(len(close), np.nan)

    def calculate_bollinger_bands(self, close: np.array, period: int = 20, std_dev: float = 2) -> tuple:
        """计算布林带"""
        try:
            df = pd.DataFrame({'close': close})
            df['middle'] = df['close'].rolling(window=period).mean()
            df['std'] = df['close'].rolling(window=period).std()
            df['upper'] = df['middle'] + (df['std'] * std_dev)
            df['lower'] = df['middle'] - (df['std'] * std_dev)
            
            return df['upper'].values, df['middle'].values, df['lower'].values
        except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
            logger.debug(f"计算异常: {e}")
            return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        if df.empty or len(df) < 26:  # 需要至少26天数据计算MACD
            return None
        
        try:
            indicators = {}
            
            # 价格数据
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # 1. 计算KDJ指标
            try:
                k, d, j = self.calculate_kdj(high, low, close, 9)
                
                indicators['kdj_k'] = k[-1] if not np.isnan(k[-1]) else None
                indicators['kdj_d'] = d[-1] if not np.isnan(d[-1]) else None  
                indicators['kdj_j'] = j[-1] if not np.isnan(j[-1]) else None
            except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
                logger.debug(f"计算异常: {e}")
                indicators['kdj_k'] = indicators['kdj_d'] = indicators['kdj_j'] = None
            
            # 2. 计算MACD指标
            try:
                macd_dif, macd_dea, macd_hist = self.calculate_macd(close, 12, 26, 9)
                
                indicators['macd_dif'] = macd_dif[-1] if not np.isnan(macd_dif[-1]) else None
                indicators['macd_dea'] = macd_dea[-1] if not np.isnan(macd_dea[-1]) else None
                indicators['macd_macd'] = macd_hist[-1] if not np.isnan(macd_hist[-1]) else None
            except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
                logger.debug(f"计算异常: {e}")
                indicators['macd_dif'] = indicators['macd_dea'] = indicators['macd_macd'] = None
            
            # 3. 计算RSI指标
            try:
                rsi6 = self.calculate_rsi(close, 6)
                rsi12 = self.calculate_rsi(close, 12)
                rsi24 = self.calculate_rsi(close, 24)
                
                indicators['rsi6'] = rsi6[-1] if not np.isnan(rsi6[-1]) else None
                indicators['rsi12'] = rsi12[-1] if not np.isnan(rsi12[-1]) else None
                indicators['rsi24'] = rsi24[-1] if not np.isnan(rsi24[-1]) else None
            except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
                logger.debug(f"计算异常: {e}")
                indicators['rsi6'] = indicators['rsi12'] = indicators['rsi24'] = None
            
            # 4. 计算布林带
            try:
                upper, middle, lower = self.calculate_bollinger_bands(close, 20, 2)
                
                indicators['boll_upper'] = upper[-1] if not np.isnan(upper[-1]) else None
                indicators['boll_middle'] = middle[-1] if not np.isnan(middle[-1]) else None
                indicators['boll_lower'] = lower[-1] if not np.isnan(lower[-1]) else None
            except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
                logger.debug(f"计算异常: {e}")
                indicators['boll_upper'] = indicators['boll_middle'] = indicators['boll_lower'] = None
            
            # 5. 计算BBI指标 (Bull and Bear Index)
            try:
                ma3 = self.sma(close, 3)
                ma6 = self.sma(close, 6)
                ma12 = self.sma(close, 12)
                ma24 = self.sma(close, 24)
                
                bbi = (ma3 + ma6 + ma12 + ma24) / 4
                indicators['bbi'] = bbi[-1] if not np.isnan(bbi[-1]) else None
            except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
                logger.debug(f"计算异常: {e}")
                indicators['bbi'] = None
            
            # 6. 计算成交量指标
            try:
                vol_ma5 = self.sma(volume, 5)
                vol_ma10 = self.sma(volume, 10)
                
                indicators['volume_ma5'] = int(vol_ma5[-1]) if not np.isnan(vol_ma5[-1]) else None
                indicators['volume_ma10'] = int(vol_ma10[-1]) if not np.isnan(vol_ma10[-1]) else None
                
                # 量比 = 当日成交量 / 5日平均成交量
                if indicators['volume_ma5'] and indicators['volume_ma5'] > 0:
                    indicators['volume_ratio'] = round(volume[-1] / indicators['volume_ma5'], 3)
                else:
                    indicators['volume_ratio'] = None
            except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
                logger.debug(f"计算异常: {e}")
                indicators['volume_ma5'] = indicators['volume_ma10'] = indicators['volume_ratio'] = None
            
            # 7. 计算CCI指标 (Commodity Channel Index, 14日)
            try:
                if len(close) >= 14:
                    typical_price = (high + low + close) / 3
                    tp_series = pd.Series(typical_price)
                    tp_sma = tp_series.rolling(window=14).mean().values
                    tp_mad = tp_series.rolling(window=14).apply(
                        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
                    ).values
                    if tp_mad[-1] != 0:
                        indicators['cci_14'] = float((typical_price[-1] - tp_sma[-1]) / (0.015 * tp_mad[-1]))
                    else:
                        indicators['cci_14'] = 0.0
                else:
                    indicators['cci_14'] = None
            except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
                logger.debug(f"计算异常: {e}")
                indicators['cci_14'] = None

            # 8. 计算ATR指标 (Average True Range, 14日)
            try:
                if len(close) >= 15:  # 需要prev_close，所以至少15天
                    prev_close = np.roll(close, 1)
                    prev_close[0] = close[0]
                    tr = np.maximum(
                        high - low,
                        np.maximum(
                            np.abs(high - prev_close),
                            np.abs(low - prev_close)
                        )
                    )
                    # 用EMA平滑
                    atr_series = pd.Series(tr).ewm(span=14, adjust=False).mean().values
                    indicators['atr_14'] = float(atr_series[-1])
                else:
                    indicators['atr_14'] = None
            except (ValueError, ZeroDivisionError, IndexError, TypeError) as e:
                logger.debug(f"计算异常: {e}")
                indicators['atr_14'] = None

            # 9. 计算知行策略所需指标
            try:
                # 知行短期趋势线：EMA(EMA(CLOSE, 10), 10)
                ema10 = self.ema(close, 10)
                zhixing_short_trend = self.ema(ema10, 10)
                indicators['zhixing_short_trend'] = zhixing_short_trend[-1] if not np.isnan(zhixing_short_trend[-1]) else None
                
                # 知行多空线：(MA14 + MA28 + MA57 + MA114) / 4
                if len(close) >= 114:  # 确保有足够数据计算MA114
                    ma14 = self.sma(close, 14)
                    ma28 = self.sma(close, 28)
                    ma57 = self.sma(close, 57)
                    ma114 = self.sma(close, 114)
                    
                    zhixing_multi_kong = (ma14 + ma28 + ma57 + ma114) / 4
                    indicators['zhixing_multi_kong'] = zhixing_multi_kong[-1] if not np.isnan(zhixing_multi_kong[-1]) else None
                    
                    # 单独保存这些MA值以便其他策略使用
                    indicators['ma14'] = ma14[-1] if not np.isnan(ma14[-1]) else None
                    indicators['ma28'] = ma28[-1] if not np.isnan(ma28[-1]) else None
                    indicators['ma57'] = ma57[-1] if not np.isnan(ma57[-1]) else None
                    indicators['ma114'] = ma114[-1] if not np.isnan(ma114[-1]) else None
                else:
                    indicators['zhixing_multi_kong'] = None
                    indicators['ma14'] = indicators['ma28'] = indicators['ma57'] = indicators['ma114'] = None
                    
            except Exception as e:
                logger.warning(f"计算知行指标失败: {e}")
                indicators['zhixing_short_trend'] = indicators['zhixing_multi_kong'] = None
                indicators['ma14'] = indicators['ma28'] = indicators['ma57'] = indicators['ma114'] = None
            
            return indicators
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return None
    
    def update_ma_in_daily_quotes(self, security_id: int, df: pd.DataFrame, conn):
        """更新daily_quotes表中的移动平均线"""
        try:
            close = df['close'].values
            
            # 计算移动平均线
            ma5 = self.sma(close, 5)
            ma10 = self.sma(close, 10)
            ma20 = self.sma(close, 20)
            ma60 = self.sma(close, 60)
            
            # 更新最新的MA值到daily_quotes表
            latest_date = df['trade_date'].iloc[-1].strftime('%Y-%m-%d')
            
            update_query = """
            UPDATE daily_quotes 
            SET ma5 = ?, ma10 = ?, ma20 = ?, ma60 = ?
            WHERE security_id = ? AND trade_date = ?
            """
            
            ma_values = [
                ma5[-1] if not np.isnan(ma5[-1]) else None,
                ma10[-1] if not np.isnan(ma10[-1]) else None,
                ma20[-1] if not np.isnan(ma20[-1]) else None,
                ma60[-1] if not np.isnan(ma60[-1]) else None
            ]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with self.db_lock:  # 使用全局锁
                        cursor = conn.cursor()
                        cursor.execute(update_query, ma_values + [security_id, latest_date])
                        conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        import time
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        logger.error(f"更新MA指标失败 (security_id: {security_id}): {e}")
                        break
                except Exception as e:
                    logger.error(f"更新MA指标失败 (security_id: {security_id}): {e}")
                    break
            
        except Exception as e:
            logger.error(f"更新MA指标失败 (security_id: {security_id}): {e}")
    
    def update_ma_for_date(self, security_id: int, df: pd.DataFrame, target_date: str, conn):
        """更新指定日期的MA数据"""
        try:
            close = df['close'].values
            
            # 计算移动平均线
            ma5 = self.sma(close, 5)
            ma10 = self.sma(close, 10)
            ma20 = self.sma(close, 20)
            ma60 = self.sma(close, 60)
            
            update_query = """
            UPDATE daily_quotes 
            SET ma5 = ?, ma10 = ?, ma20 = ?, ma60 = ?
            WHERE security_id = ? AND trade_date = ?
            """
            
            ma_values = [
                ma5[-1] if not np.isnan(ma5[-1]) else None,
                ma10[-1] if not np.isnan(ma10[-1]) else None,
                ma20[-1] if not np.isnan(ma20[-1]) else None,
                ma60[-1] if not np.isnan(ma60[-1]) else None
            ]
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    with self.db_lock:  # 使用全局锁
                        cursor = conn.cursor()
                        cursor.execute(update_query, ma_values + [security_id, target_date])
                        conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        import time
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        logger.error(f"更新MA指标失败 (security_id: {security_id}, date: {target_date}): {e}")
                        break
                except Exception as e:
                    logger.error(f"更新MA指标失败 (security_id: {security_id}, date: {target_date}): {e}")
                    break
            
        except Exception as e:
            logger.error(f"更新MA指标失败 (security_id: {security_id}, date: {target_date}): {e}")
    
    def _make_ti_values(self, security_id: int, trade_date: str, indicators: Dict) -> tuple:
        """构造技术指标插入值元组"""
        return (
            security_id, trade_date,
            indicators.get('kdj_k'), indicators.get('kdj_d'), indicators.get('kdj_j'),
            indicators.get('macd_dif'), indicators.get('macd_dea'), indicators.get('macd_macd'),
            indicators.get('rsi6'), indicators.get('rsi12'), indicators.get('rsi24'),
            indicators.get('boll_upper'), indicators.get('boll_middle'), indicators.get('boll_lower'),
            indicators.get('bbi'), indicators.get('volume_ma5'), indicators.get('volume_ma10'), indicators.get('volume_ratio'),
            indicators.get('zhixing_short_trend'), indicators.get('zhixing_multi_kong'),
            indicators.get('ma14'), indicators.get('ma28'), indicators.get('ma57'), indicators.get('ma114'),
            indicators.get('cci_14'), indicators.get('atr_14'),
            datetime.now()
        )

    _TI_INSERT_QUERY = """
        INSERT OR REPLACE INTO technical_indicators (
            security_id, trade_date, kdj_k, kdj_d, kdj_j,
            macd_dif, macd_dea, macd_macd,
            rsi6, rsi12, rsi24,
            boll_upper, boll_middle, boll_lower,
            bbi, volume_ma5, volume_ma10, volume_ratio,
            zhixing_short_trend, zhixing_multi_kong, ma14, ma28, ma57, ma114,
            cci_14, atr_14,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    _MA_UPDATE_QUERY = """
        UPDATE daily_quotes
        SET ma5 = ?, ma10 = ?, ma20 = ?, ma60 = ?
        WHERE security_id = ? AND trade_date = ?
    """

    def batch_commit(self, ti_batch: List[tuple], ma_batch: List[tuple]):
        """批量写入技术指标和MA数据 (单次commit)"""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            cursor = conn.cursor()
            if ti_batch:
                cursor.executemany(self._TI_INSERT_QUERY, ti_batch)
            if ma_batch:
                cursor.executemany(self._MA_UPDATE_QUERY, ma_batch)
            conn.commit()
            logger.info(f"批量写入完成: {len(ti_batch)} 条技术指标, {len(ma_batch)} 条MA更新")
        except Exception as e:
            logger.error(f"批量写入失败: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def insert_technical_indicators(self, security_id: int, trade_date: str, indicators: Dict, conn=None):
        """插入技术指标到数据库"""
        if conn is None:
            conn = self.get_thread_connection()

        values = list(self._make_ti_values(security_id, trade_date, indicators))

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.db_lock:  # 使用全局锁
                    cursor = conn.cursor()
                    cursor.execute(self._TI_INSERT_QUERY, values)
                    conn.commit()
                break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))  # 指数退避
                    continue
                else:
                    logger.error(f"插入技术指标失败 (security_id: {security_id}): {e}")
                    raise
            except Exception as e:
                logger.error(f"插入技术指标失败 (security_id: {security_id}): {e}")
                raise
    
    def process_single_stock(self, security_id: int, code: str, name: str) -> bool:
        """处理单只股票（线程安全版本）"""
        try:
            # 获取线程本地连接
            conn = self.get_thread_connection()
            
            # 获取价格数据
            df = self.get_stock_price_data(security_id, days=250, conn=conn)
            if df is None or len(df) < 26:
                logger.warning(f"跳过股票 {code} - 数据不足")
                return False
            
            # 如果指定了日期范围，需要为范围内每一天计算指标
            if self.start_date and self.end_date:
                result = self.process_date_range(security_id, code, name, df, conn)
            else:
                # 原有逻辑：只计算最新日期的指标
                result = self.process_single_date(security_id, code, name, df, conn)
            
            # 更新统计信息
            with self.stats_lock:
                self.stats['processed_stocks'] += 1
                if result:
                    self.stats['successful_calculations'] += 1
                else:
                    self.stats['failed_calculations'] += 1
                
                # 进度报告
                if self.stats['processed_stocks'] % 100 == 0:
                    logger.info(f"进度: 已处理 {self.stats['processed_stocks']} 只股票")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 处理股票失败 {code}: {e}")
            with self.stats_lock:
                self.stats['processed_stocks'] += 1
                self.stats['failed_calculations'] += 1
            return False
    
    def process_single_date(self, security_id: int, code: str, name: str, df: pd.DataFrame, conn) -> bool:
        """处理单个日期的技术指标"""
        # 计算技术指标
        indicators = self.calculate_technical_indicators(df)
        if indicators is None:
            logger.warning(f"跳过股票 {code} - 指标计算失败")
            return False
        
        # 获取最新交易日期
        latest_date = df['trade_date'].iloc[-1].strftime('%Y-%m-%d')
        
        # 插入技术指标数据
        self.insert_technical_indicators(security_id, latest_date, indicators, conn)
        
        # 更新daily_quotes表中的MA数据
        self.update_ma_in_daily_quotes(security_id, df, conn)
        
        logger.info(f"✅ 完成 {code} ({name}) - KDJ: {indicators.get('kdj_k', 0):.2f}")
        return True
    
    def process_date_range(self, security_id: int, code: str, name: str, df: pd.DataFrame, conn) -> bool:
        """处理日期范围内的技术指标"""
        # 筛选日期范围内的数据
        start_dt = pd.to_datetime(self.start_date)
        end_dt = pd.to_datetime(self.end_date)
        
        # 获取范围内的交易日期
        target_dates = df[(df['trade_date'] >= start_dt) & (df['trade_date'] <= end_dt)]['trade_date'].tolist()
        
        if not target_dates:
            logger.warning(f"跳过股票 {code} - 日期范围内无数据")
            return False
        
        success_count = 0
        for target_date in target_dates:
            # 获取到目标日期为止的历史数据
            historical_data = df[df['trade_date'] <= target_date].copy()
            
            if len(historical_data) < 26:
                continue
            
            # 计算该日期的技术指标
            indicators = self.calculate_technical_indicators(historical_data)
            if indicators is None:
                continue
            
            # 插入技术指标数据
            date_str = target_date.strftime('%Y-%m-%d')
            self.insert_technical_indicators(security_id, date_str, indicators, conn)
            
            # 更新该日期的MA数据
            self.update_ma_for_date(security_id, historical_data, date_str, conn)
            
            success_count += 1
        
        if success_count > 0:
            logger.info(f"✅ 完成 {code} ({name}) - 计算了 {success_count} 个日期的指标")
            return True
        else:
            logger.warning(f"跳过股票 {code} - 日期范围内无法计算指标")
            return False
    
    def calculate_all_indicators(self, limit: Optional[int] = None, batch_size: int = 100):
        """并行计算所有股票的技术指标"""
        self.stats['start_time'] = datetime.now()
        logger.info(f"开始并行计算技术指标... (线程数: {self.max_workers})")
        
        try:
            self.connect_db()
            
            # 获取股票列表
            stock_list = self.get_stock_list(limit)
            total_stocks = len(stock_list)
            logger.info(f"总计需要处理 {total_stocks} 只股票")
            
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_stock = {
                    executor.submit(self.process_single_stock, security_id, code, name): (security_id, code, name)
                    for security_id, code, name in stock_list
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_stock):
                    security_id, code, name = future_to_stock[future]
                    try:
                        success = future.result()
                        # 统计已在process_single_stock中更新
                    except Exception as exc:
                        logger.error(f"股票 {code} 处理异常: {exc}")
                        with self.stats_lock:
                            self.stats['processed_stocks'] += 1
                            self.stats['failed_calculations'] += 1
            
            self.stats['end_time'] = datetime.now()
            
            # 输出统计信息
            self._print_statistics()
            
        except Exception as e:
            logger.error(f"计算技术指标过程失败: {e}")
        finally:
            self.close_all_connections()
    
    def calculate_all_indicators_batch(self, limit: Optional[int] = None) -> int:
        """单线程计算所有股票的技术指标，最后批量写入 (适合单日更新)

        Returns:
            成功计算的股票数
        """
        self.stats['start_time'] = datetime.now()
        logger.info("开始计算技术指标 (批量模式)...")

        try:
            self.connect_db()

            stock_list = self.get_stock_list(limit)
            total_stocks = len(stock_list)
            logger.info(f"总计需要处理 {total_stocks} 只股票")

            ti_batch = []
            ma_batch = []
            success_count = 0

            for security_id, code, name in stock_list:
                try:
                    df = self.get_stock_price_data(security_id, days=250, conn=self.conn)
                    if df is None or len(df) < 26:
                        continue

                    indicators = self.calculate_technical_indicators(df)
                    if indicators is None:
                        continue

                    latest_date = df['trade_date'].iloc[-1].strftime('%Y-%m-%d')

                    # 收集技术指标
                    ti_batch.append(self._make_ti_values(security_id, latest_date, indicators))

                    # 收集MA数据
                    close = df['close'].values
                    ma5 = self.sma(close, 5)
                    ma10 = self.sma(close, 10)
                    ma20 = self.sma(close, 20)
                    ma60 = self.sma(close, 60)
                    ma_batch.append((
                        ma5[-1] if not np.isnan(ma5[-1]) else None,
                        ma10[-1] if not np.isnan(ma10[-1]) else None,
                        ma20[-1] if not np.isnan(ma20[-1]) else None,
                        ma60[-1] if not np.isnan(ma60[-1]) else None,
                        security_id, latest_date
                    ))

                    success_count += 1
                    if success_count % 500 == 0:
                        logger.info(f"进度: 已计算 {success_count}/{total_stocks} 只股票")

                except Exception as e:
                    logger.debug(f"处理 {code} 失败: {e}")
                    continue

            # 批量写入
            self.batch_commit(ti_batch, ma_batch)

            self.stats['successful_calculations'] = success_count
            self.stats['processed_stocks'] = total_stocks
            self.stats['end_time'] = datetime.now()
            self._print_statistics()

            return success_count

        except Exception as e:
            logger.error(f"批量计算技术指标失败: {e}")
            return 0
        finally:
            self.close_db()

    def _print_statistics(self):
        """打印统计信息"""
        processing_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info("=" * 60)
        logger.info("技术指标计算完成！")
        logger.info(f"📊 总计处理: {self.stats['processed_stocks']} 只股票")
        logger.info(f"✅ 成功计算: {self.stats['successful_calculations']} 只")
        logger.info(f"❌ 失败计算: {self.stats['failed_calculations']} 只")
        logger.info(f"⏱️ 总耗时: {processing_time:.1f} 秒")
        logger.info(f"🚄 处理速度: {self.stats['processed_stocks'] / processing_time:.1f} 只/秒")
        logger.info(f"✅ 成功率: {self.stats['successful_calculations'] / max(self.stats['processed_stocks'], 1):.1%}")
        logger.info("=" * 60)
    
    def verify_data(self):
        """验证计算结果"""
        self.connect_db()
        try:
            cursor = self.conn.cursor()
            
            # 检查technical_indicators表记录数
            cursor.execute("SELECT COUNT(*) FROM technical_indicators")
            ti_count = cursor.fetchone()[0]
            
            # 检查有MA数据的daily_quotes记录数
            cursor.execute("SELECT COUNT(*) FROM daily_quotes WHERE ma5 IS NOT NULL")
            ma_count = cursor.fetchone()[0]
            
            # 随机查看几条技术指标数据
            cursor.execute("""
                SELECT s.code, ti.trade_date, ti.kdj_k, ti.kdj_d, ti.macd_dif, ti.rsi12, ti.bbi
                FROM technical_indicators ti
                JOIN securities s ON ti.security_id = s.id
                ORDER BY ti.trade_date DESC
                LIMIT 5
            """)
            sample_data = cursor.fetchall()
            
            logger.info("数据验证结果:")
            logger.info(f"technical_indicators表记录数: {ti_count}")
            logger.info(f"有MA数据的daily_quotes记录数: {ma_count}")
            logger.info("技术指标样本数据:")
            for row in sample_data:
                logger.info(f"  {row[0]} ({row[1]}): KDJ_K={row[2]:.2f}, KDJ_D={row[3]:.2f}, MACD={row[4]:.4f}, RSI12={row[5]:.2f}, BBI={row[6]:.2f}")
                
        except Exception as e:
            logger.error(f"验证数据失败: {e}")
        finally:
            self.close_db()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="技术指标计算器")
    parser.add_argument("--limit", type=int, default=None,
                       help="限制处理的股票数量（测试用）")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="批处理大小")
    parser.add_argument("--verify", action="store_true",
                       help="仅验证数据，不进行计算")
    parser.add_argument("--db-path", type=str, default="data_adapter/stock_data.db",
                       help="数据库路径")
    parser.add_argument("--start-date", type=str, default=None,
                       help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                       help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="最大并行线程数")
    
    args = parser.parse_args()
    
    try:
        calculator = TechnicalIndicatorCalculator(args.db_path, args.start_date, args.end_date, args.max_workers)
        
        if args.verify:
            calculator.verify_data()
        else:
            calculator.calculate_all_indicators(args.limit, args.batch_size)
            calculator.verify_data()
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()