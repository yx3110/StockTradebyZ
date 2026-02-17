#!/usr/bin/env python3
"""
中国股票数据适配器
将本地中国股票数据转换为TradingAgents格式
"""

import pandas as pd
import numpy as np
import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChinaStockAdapter:
    """中国股票数据适配器"""
    
    def __init__(self, local_data_dir: str = "full_securities_data", ta_data_dir: str = "TA_integration/data", db_path: str = None):
        # 获取项目根目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # 从 TA_integration/adapters/ 回到项目根目录
        
        self.local_data_dir = project_root / local_data_dir
        self.ta_data_dir = project_root / ta_data_dir
        self.market_data_dir = self.ta_data_dir / "market_data" / "price_data"
        self.cache_dir = self.ta_data_dir / "cache"
        
        # 数据库路径
        if db_path is None:
            # 尝试多个可能的数据库位置
            possible_db_paths = [
                project_root / "data_adapter" / "stock_data.db",
                project_root / "TA_integration" / "stock_data.db",
                project_root / "stock_data.db"
            ]
            
            self.db_path = None
            for path in possible_db_paths:
                if path.exists():
                    self.db_path = str(path)
                    break
            
            if self.db_path is None:
                logger.warning("未找到股票数据库，将使用CSV文件数据")
        else:
            self.db_path = db_path
        
        # 创建必要的目录
        self.market_data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 股票信息缓存
        self.stock_info_cache = {}
        self._load_stock_info_cache()
    
    def _load_stock_info_cache(self):
        """加载股票信息缓存"""
        cache_file = self.cache_dir / "stock_info_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self.stock_info_cache = json.load(f)
            except Exception as e:
                logger.warning(f"加载股票信息缓存失败: {e}")
    
    def _save_stock_info_cache(self):
        """保存股票信息缓存"""
        cache_file = self.cache_dir / "stock_info_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.stock_info_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存股票信息缓存失败: {e}")
    
    def get_stock_data_from_db(self, stock_code: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """从SQLite数据库读取股票数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 构建SQL查询
                sql = """
                SELECT 
                    dq.trade_date as date,
                    dq.open,
                    dq.high,
                    dq.low,
                    dq.close,
                    dq.volume,
                    dq.price_change,
                    dq.price_change_pct,
                    s.name,
                    s.industry,
                    s.area
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code = ?
                """
                
                params = [stock_code]
                
                if start_date:
                    sql += " AND dq.trade_date >= ?"
                    params.append(start_date)
                
                if end_date:
                    sql += " AND dq.trade_date <= ?"
                    params.append(end_date)
                
                sql += " ORDER BY dq.trade_date ASC"
                
                df = pd.read_sql_query(sql, conn, params=params)
                
                if df.empty:
                    logger.warning(f"未找到股票 {stock_code} 的数据")
                    return None
                
                # 转换日期格式
                df['date'] = pd.to_datetime(df['date'])
                
                return df
                
        except Exception as e:
            logger.error(f"从数据库读取股票 {stock_code} 数据失败: {e}")
            return None
    
    def _convert_df_to_ta_format(self, df: pd.DataFrame, stock_code: str, stock_name: str = "") -> pd.DataFrame:
        """将数据库数据转换为TradingAgents格式"""
        try:
            # 确保数据按日期排序
            df = df.sort_values('date').copy()
            
            # 转换为TradingAgents需要的格式
            ta_df = pd.DataFrame({
                'Date': df['date'].dt.strftime('%Y-%m-%d'),
                'Open': df['open'].round(2),
                'High': df['high'].round(2), 
                'Low': df['low'].round(2),
                'Close': df['close'].round(2),
                'Volume': df['volume'].astype(int),
                'Adj Close': df['close'].round(2)  # 暂时使用收盘价作为调整后价格
            })
            
            return ta_df
            
        except Exception as e:
            logger.error(f"转换股票 {stock_code} 数据格式失败: {e}")
            return pd.DataFrame()
    
    def _save_ta_data(self, ta_df: pd.DataFrame, stock_code: str) -> bool:
        """保存转换后的TradingAgents格式数据"""
        try:
            if ta_df.empty:
                return False
                
            # 生成文件名
            ta_file = os.path.join(
                self.market_data_dir,
                f"{stock_code}-YFin-data-2015-01-01-2025-12-31.csv"
            )
            
            # 保存数据
            ta_df.to_csv(ta_file, index=False)
            logger.info(f"保存股票 {stock_code} 转换数据到 {ta_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"保存股票 {stock_code} 转换数据失败: {e}")
            return False
    
    def convert_stock_data(self, stock_code: str, stock_name: str = "") -> bool:
        """将单只股票数据转换为TradingAgents格式，优先使用数据库"""
        try:
            # 首先尝试从数据库读取数据
            df = self.get_stock_data_from_db(stock_code)
            
            if df is not None and not df.empty:
                logger.info(f"从数据库获取股票 {stock_code} 数据，共 {len(df)} 条记录")
                
                # 转换为TradingAgents格式
                ta_data = self._convert_df_to_ta_format(df, stock_code, stock_name)
                
                # 保存转换后的数据
                return self._save_ta_data(ta_data, stock_code)
            
            # 回退到CSV文件读取方法
            logger.warning(f"数据库中未找到股票 {stock_code}，尝试CSV文件...")
            
            # 寻找本地数据文件
            local_file = self._find_local_file(stock_code)
            if not local_file:
                logger.warning(f"未找到股票 {stock_code} 的本地数据文件")
                return False
            
            # 读取本地数据
            df = pd.read_csv(local_file)
            
            # 转换为TradingAgents格式
            ta_df = self._convert_to_ta_format(df)
            
            # 保存为TradingAgents格式
            output_file = os.path.join(
                self.market_data_dir, 
                f"{stock_code}-YFin-data-2015-01-01-2025-12-31.csv"
            )
            ta_df.to_csv(output_file, index=False)
            
            # 更新股票信息缓存
            if stock_name:
                self.stock_info_cache[stock_code] = {
                    'name': stock_name,
                    'market': 'A股',
                    'currency': 'CNY',
                    'last_update': datetime.now().isoformat()
                }
                self._save_stock_info_cache()
            
            logger.info(f"成功转换股票 {stock_code} 数据")
            return True
            
        except Exception as e:
            logger.error(f"转换股票 {stock_code} 数据失败: {e}")
            return False
    
    def _find_local_file(self, stock_code: str) -> Optional[str]:
        """寻找本地股票数据文件"""
        possible_files = [
            f"{stock_code}_A股.csv",
            f"{stock_code}_ETF.csv",
            f"{stock_code}_基金.csv"
        ]
        
        for filename in possible_files:
            filepath = os.path.join(self.local_data_dir, filename)
            if os.path.exists(filepath):
                return filepath
        
        return None
    
    def _convert_to_ta_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换数据格式为TradingAgents兼容格式"""
        # 检查必要的列
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要列: {col}")
        
        # 转换格式
        ta_df = pd.DataFrame({
            'Date': pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d'),
            'Open': df['open'].astype(float),
            'High': df['high'].astype(float),
            'Low': df['low'].astype(float),
            'Close': df['close'].astype(float),
            'Adj Close': df['close'].astype(float),  # A股没有调整收盘价，使用收盘价
            'Volume': df['volume'].astype(float)
        })
        
        # 排序并去重
        ta_df = ta_df.sort_values('Date').drop_duplicates(subset=['Date'])
        ta_df = ta_df.reset_index(drop=True)
        
        return ta_df
    
    def convert_multiple_stocks(self, stock_list: List[Tuple[str, str]]) -> Dict[str, bool]:
        """批量转换多只股票数据"""
        results = {}
        
        for stock_code, stock_name in stock_list:
            success = self.convert_stock_data(stock_code, stock_name)
            results[stock_code] = success
        
        return results
    
    def get_stock_info(self, stock_code: str) -> Dict[str, str]:
        """获取股票基本信息"""
        if stock_code in self.stock_info_cache:
            return self.stock_info_cache[stock_code]
        
        # 默认信息
        return {
            'name': f'股票{stock_code}',
            'market': 'A股',
            'currency': 'CNY',
            'last_update': datetime.now().isoformat()
        }
    
    def create_chinese_news_data(self, stock_code: str, date: str) -> Dict[str, List[Dict]]:
        """创建中文新闻数据模拟"""
        # 这里可以集成真实的中文新闻源
        # 目前返回模拟数据
        return {
            date: [
                {
                    'title': f'{stock_code} 业绩预告发布',
                    'summary': '公司发布年度业绩预告，预计净利润同比增长',
                    'sentiment': 'positive',
                    'source': '财经网站'
                }
            ]
        }
    
    def create_technical_indicators_data(self, stock_code: str) -> Dict[str, float]:
        """创建技术指标数据"""
        # 从本地数据计算技术指标
        local_file = self._find_local_file(stock_code)
        if not local_file:
            return {}
        
        try:
            df = pd.read_csv(local_file)
            if len(df) < 20:  # 数据不足
                return {}
            
            # 计算常用技术指标
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 返回最新值
            latest = df.iloc[-1]
            return {
                'MA5': float(latest['MA5']) if not pd.isna(latest['MA5']) else 0,
                'MA10': float(latest['MA10']) if not pd.isna(latest['MA10']) else 0,
                'MA20': float(latest['MA20']) if not pd.isna(latest['MA20']) else 0,
                'RSI': float(latest['RSI']) if not pd.isna(latest['RSI']) else 0,
                'current_price': float(latest['close'])
            }
            
        except Exception as e:
            logger.error(f"计算 {stock_code} 技术指标失败: {e}")
            return {}
    
    def check_data_availability(self, stock_code: str) -> Dict[str, bool]:
        """检查股票数据可用性"""
        local_file = self._find_local_file(stock_code)
        ta_file = os.path.join(
            self.market_data_dir, 
            f"{stock_code}-YFin-data-2015-01-01-2025-12-31.csv"
        )
        
        return {
            'local_data_exists': local_file is not None,
            'ta_data_exists': os.path.exists(ta_file),
            'local_file_path': local_file,
            'ta_file_path': ta_file if os.path.exists(ta_file) else None
        }
    
    def get_data_statistics(self) -> Dict[str, int]:
        """获取数据统计信息"""
        local_files = len([f for f in os.listdir(self.local_data_dir) 
                          if f.endswith('.csv')])
        
        ta_files = len([f for f in os.listdir(self.market_data_dir) 
                       if f.endswith('.csv')]) if os.path.exists(self.market_data_dir) else 0
        
        return {
            'local_stock_files': local_files,
            'converted_ta_files': ta_files,
            'cached_stock_info': len(self.stock_info_cache)
        }

class ChinaMarketDataProvider:
    """中国市场数据提供器"""
    
    def __init__(self, adapter: ChinaStockAdapter):
        self.adapter = adapter
    
    def get_stock_data_for_ta(self, stock_code: str, start_date: str, end_date: str) -> Optional[str]:
        """为TradingAgents提供股票数据"""
        # 确保数据已转换
        if not self.adapter.check_data_availability(stock_code)['ta_data_exists']:
            stock_info = self.adapter.get_stock_info(stock_code)
            success = self.adapter.convert_stock_data(stock_code, stock_info.get('name', ''))
            if not success:
                return None
        
        # 读取转换后的数据
        ta_file = os.path.join(
            self.adapter.market_data_dir,
            f"{stock_code}-YFin-data-2015-01-01-2025-12-31.csv"
        )
        
        try:
            df = pd.read_csv(ta_file)
            # 过滤日期范围
            mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
            filtered_df = df[mask]
            
            if filtered_df.empty:
                return f"没有找到 {stock_code} 在 {start_date} 到 {end_date} 期间的数据"
            
            # 格式化输出
            header = f"# 股票数据 {stock_code} 从 {start_date} 到 {end_date}\n"
            header += f"# 总记录数: {len(filtered_df)}\n"
            header += f"# 数据获取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            return header + filtered_df.to_csv(index=False)
            
        except Exception as e:
            logger.error(f"读取 {stock_code} 数据失败: {e}")
            return None

if __name__ == "__main__":
    # 测试适配器
    adapter = ChinaStockAdapter()
    
    # 测试转换单只股票
    success = adapter.convert_stock_data("000001", "平安银行")
    print(f"转换000001: {'成功' if success else '失败'}")
    
    # 检查数据可用性
    availability = adapter.check_data_availability("000001")
    print(f"数据可用性: {availability}")
    
    # 获取统计信息
    stats = adapter.get_data_statistics()
    print(f"数据统计: {stats}")