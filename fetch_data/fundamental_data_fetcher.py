#!/usr/bin/env python3
"""
基本面数据获取器
从Tushare Pro API获取股票基本面数据和大盘指数数据
"""

import os
import json
import pandas as pd
import tushare as ts
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

class FundamentalDataFetcher:
    """基本面数据获取器"""
    
    def __init__(self, config_path: str = "../config.json", db_path: str = "../data_adapter/stock_data.db"):
        """初始化获取器"""
        self.db_path = db_path
        self.config = self._load_config(config_path)
        
        # 初始化Tushare
        token = self.config.get('tushare', {}).get('token')
        if not token:
            raise ValueError("请在config.json中配置Tushare token")
        
        ts.set_token(token)
        self.pro = ts.pro_api()
        
        # API调用限制配置（根据Tushare Pro等级调整）
        self.api_limits = {
            'calls_per_minute': 120,  # 每分钟调用次数限制
            'points_per_day': 5000,   # 每日积分限制
            'delay_between_calls': 0.5  # 调用间隔秒数
        }
        
        self.last_call_time = 0
        
        # 并行处理配置
        self.max_workers = self.config.get('system', {}).get('max_workers', 5)
        self.api_lock = threading.Lock()  # API调用锁
        
        # 批量查询配置
        self.batch_size = 20  # 每批查询的股票数量（Tushare建议不超过30）
        
        # 智能缓存设置
        self.cache_settings = {
            'stock_basic_info_days': 7,  # 股票基本信息缓存7天
            'market_indices_days': 30,   # 指数信息缓存30天
            'financial_indicator_days': 30,  # 财务指标缓存30天
            'enable_incremental': True   # 启用增量更新
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def _rate_limit(self):
        """API调用速率限制（线程安全）"""
        with self.api_lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            
            if time_since_last_call < self.api_limits['delay_between_calls']:
                sleep_time = self.api_limits['delay_between_calls'] - time_since_last_call
                time.sleep(sleep_time)
            
            self.last_call_time = time.time()
    
    def _is_data_fresh(self, table_name: str, cache_days: int) -> bool:
        """检查数据是否足够新鲜，不需要更新"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 根据表类型检查最新数据时间
                if table_name == 'stock_basic_info':
                    query = "SELECT MAX(updated_at) FROM stock_basic_info"
                elif table_name == 'market_indices':
                    query = "SELECT MAX(created_at) FROM market_indices"
                elif table_name == 'financial_indicator':
                    query = "SELECT MAX(created_at) FROM financial_indicator"
                else:
                    return False
                
                result = conn.execute(query).fetchone()
                if not result or not result[0]:
                    return False
                
                last_update = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
                days_since_update = (datetime.now() - last_update).days
                
                return days_since_update < cache_days
                
        except Exception as e:
            logger.debug(f"检查数据新鲜度失败: {e}")
            return False
    
    def _get_missing_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取缺失的交易日期列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 获取已有的交易日期
                existing_dates = pd.read_sql_query(
                    "SELECT DISTINCT trade_date FROM daily_basic WHERE trade_date BETWEEN ? AND ?",
                    conn, params=[start_date, end_date]
                )
                
                existing_set = set(existing_dates['trade_date'].tolist())
                
                # 生成完整的日期范围（工作日）
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                all_dates = []
                current_date = start_dt
                while current_date <= end_dt:
                    # 跳过周末
                    if current_date.weekday() < 5:  # 0-4 是周一到周五
                        date_str = current_date.strftime('%Y-%m-%d')
                        if date_str not in existing_set:
                            all_dates.append(date_str)
                    current_date += timedelta(days=1)
                
                return all_dates
                
        except Exception as e:
            logger.error(f"获取缺失日期失败: {e}")
            return []
    
    def init_database_tables(self):
        """初始化数据库表结构"""
        try:
            # 读取SQL文件并执行
            sql_file = Path(__file__).parent / "fundamental_data_schema.sql"
            
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            
            with sqlite3.connect(self.db_path) as conn:
                # 分割SQL语句并执行
                statements = sql_script.split(';')
                for statement in statements:
                    statement = statement.strip()
                    if statement:
                        conn.execute(statement)
                
                conn.commit()
                logger.info("基本面数据库表结构初始化完成")
                
        except Exception as e:
            logger.error(f"初始化数据库表结构失败: {e}")
            raise
    
    def fetch_stock_basic_info(self, update_existing: bool = False) -> int:
        """获取股票基本信息 - 带智能缓存"""
        logger.info("开始获取股票基本信息...")
        
        # 检查缓存是否新鲜
        if not update_existing and self._is_data_fresh('stock_basic_info', self.cache_settings['stock_basic_info_days']):
            logger.info(f"股票基本信息缓存仍然新鲜（{self.cache_settings['stock_basic_info_days']}天内），跳过获取")
            
            # 返回现有记录数
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM stock_basic_info").fetchone()[0]
                return count
        
        try:
            self._rate_limit()
            
            # 获取A股基本信息
            stock_basic = self.pro.stock_basic(exchange='', 
                                             list_status='L',  # 只获取上市的股票
                                             fields='ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,curr_type,list_status,list_date,delist_date,is_hs')
            
            if stock_basic.empty:
                logger.warning("未获取到股票基本信息")
                return 0
            
            logger.info(f"获取到 {len(stock_basic)} 只股票的基本信息")
            
            # 存储到数据库
            saved_count = self._save_stock_basic_info(stock_basic, update_existing)
            
            return saved_count
            
        except Exception as e:
            logger.error(f"获取股票基本信息失败: {e}")
            return 0
    
    def _save_stock_basic_info(self, stock_basic: pd.DataFrame, update_existing: bool) -> int:
        """保存股票基本信息到数据库 - 优化批量版本"""
        saved_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 获取所有securities的映射
                securities_df = pd.read_sql_query(
                    "SELECT code, id FROM securities", 
                    conn
                )
                securities_map = dict(zip(securities_df['code'], securities_df['id']))
                
                # 如果不更新现有数据，获取已存在的记录
                existing_ids = set()
                if not update_existing:
                    existing_df = pd.read_sql_query(
                        "SELECT security_id FROM stock_basic_info",
                        conn
                    )
                    existing_ids = set(existing_df['security_id'].tolist())
                
                # 预处理数据，批量准备
                data_to_insert = []
                for _, row in stock_basic.iterrows():
                    try:
                        ts_code = row['ts_code']
                        stock_code = ts_code.split('.')[0]
                        
                        # 查找security_id
                        security_id = securities_map.get(stock_code)
                        if not security_id:
                            continue
                        
                        # 检查是否跳过已存在的记录
                        if not update_existing and security_id in existing_ids:
                            continue
                        
                        data_to_insert.append((
                            security_id,
                            ts_code,
                            row.get('market', ''),
                            row.get('list_status', 'L'),
                            row.get('fullname', ''),
                            row.get('enname', ''),
                            row.get('list_date', None),
                            '',  # main_business 需要额外获取
                            '',  # business_scope 需要额外获取
                        ))
                        
                    except Exception as e:
                        logger.warning(f"处理股票 {row.get('ts_code', 'Unknown')} 基本信息失败: {e}")
                        continue
                
                # 批量插入数据
                if data_to_insert:
                    insert_sql = """
                    INSERT OR REPLACE INTO stock_basic_info 
                    (security_id, ts_code, market, list_status, fullname, enname, 
                     setup_date, main_business, business_scope, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """
                    
                    conn.executemany(insert_sql, data_to_insert)
                    conn.commit()
                    saved_count = len(data_to_insert)
                    
                    logger.info(f"批量保存 {saved_count} 只股票的基本信息")
                
        except Exception as e:
            logger.error(f"批量保存股票基本信息失败: {e}")
        
        return saved_count
    
    def _fetch_single_stock_financial(self, stock_info: Dict, periods: int = 8) -> Dict:
        """获取单只股票的财务指标数据（用于并行处理）"""
        security_id = stock_info['id']
        code = stock_info['code']
        name = stock_info['name']
        
        result = {
            'security_id': security_id,
            'code': code,
            'name': name,
            'success': False,
            'records_saved': 0,
            'error': None
        }
        
        try:
            # 转换股票代码为Tushare格式
            ts_code = self._convert_stock_code(code)
            
            # API调用限制
            self._rate_limit()
            
            # 获取财务指标数据
            df = self.pro.fina_indicator(
                ts_code=ts_code,
                fields='ts_code,ann_date,end_date,eps,gross_margin,current_ratio,roe,roa,netprofit_margin,debt_to_assets,basic_eps_yoy,netprofit_yoy,or_yoy',
                limit=periods
            )
            
            if df.empty:
                result['error'] = "无财务指标数据"
                return result
            
            # 保存数据
            saved_count = self._save_financial_indicator_data(df, security_id)
            result['success'] = True
            result['records_saved'] = saved_count
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _fetch_batch_financial_indicators(self, stock_codes: List[str], periods: int = 8) -> pd.DataFrame:
        """批量获取多只股票的财务指标数据（单次API调用）"""
        try:
            # API调用限制
            self._rate_limit()
            
            # 使用逗号分隔的方式批量查询
            ts_codes_str = ','.join(stock_codes)
            
            df = self.pro.fina_indicator(
                ts_code=ts_codes_str,
                fields='ts_code,ann_date,end_date,eps,gross_margin,current_ratio,roe,roa,netprofit_margin,debt_to_assets,basic_eps_yoy,netprofit_yoy,or_yoy',
                limit=periods * len(stock_codes)  # 每只股票periods条记录
            )
            
            return df if df is not None else pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"批量获取财务指标失败: {e}")
            return pd.DataFrame()
    
    def _process_batch_list(self, batch_list: List[List[str]], stock_code_map: Dict, periods: int) -> tuple:
        """并行处理批量获取任务"""
        success_count = 0
        fail_count = 0
        total_records = 0
        
        def process_single_batch(batch_codes):
            """处理单个批次"""
            batch_result = {
                'success_count': 0,
                'fail_count': 0,
                'total_records': 0,
                'batch_codes': batch_codes
            }
            
            # 批量获取数据
            df = self._fetch_batch_financial_indicators(batch_codes, periods)
            
            if df.empty:
                batch_result['fail_count'] = len(batch_codes)
                return batch_result
            
            # 按股票分组保存数据
            for ts_code, group_df in df.groupby('ts_code'):
                if ts_code in stock_code_map:
                    security_id, stock_name = stock_code_map[ts_code]
                    saved_count = self._save_financial_indicator_data(group_df, security_id)
                    
                    if saved_count > 0:
                        batch_result['total_records'] += saved_count
                    
                    batch_result['success_count'] += 1
                else:
                    batch_result['fail_count'] += 1
            
            return batch_result
        
        # 如果批次数量足够多，使用并行处理批次
        if len(batch_list) > 3:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch_list))) as executor:
                futures = [executor.submit(process_single_batch, batch_codes) for batch_codes in batch_list]
                
                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        result = future.result()
                        success_count += result['success_count']
                        fail_count += result['fail_count']
                        total_records += result['total_records']
                        
                        logger.debug(f"批次 {i} 完成: 成功 {result['success_count']}, 失败 {result['fail_count']}, 记录 {result['total_records']}")
                        
                    except Exception as e:
                        logger.warning(f"批次 {i} 处理异常: {e}")
                        fail_count += len(batch_list[i-1]) if i-1 < len(batch_list) else 0
        else:
            # 批次较少时，串行处理
            for batch_codes in batch_list:
                result = process_single_batch(batch_codes)
                success_count += result['success_count']
                fail_count += result['fail_count'] 
                total_records += result['total_records']
        
        return success_count, fail_count, total_records

    def fetch_financial_indicators(self, periods: int = 8, start_idx: int = 0, limit: int = None, force: bool = False) -> int:
        """批量获取财务指标数据（优化版：批量API + 并行处理）"""
        if not force and self._is_data_fresh('financial_indicator', self.cache_settings.get('financial_indicator_days', 30)):
            logger.info("财务指标数据足够新鲜，跳过更新")
            return 0
            
        logger.info("开始获取财务指标数据...(批量API + 并行处理模式)")
        
        try:
            # 获取股票代码列表
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT id, code, name, type 
                    FROM securities 
                    WHERE type LIKE '%股%' 
                    ORDER BY code
                """
                stock_df = pd.read_sql_query(query, conn)
            
            if stock_df.empty:
                logger.warning("未找到股票数据")
                return 0
            
            total_stocks = len(stock_df)
            if limit:
                stock_df = stock_df[start_idx:start_idx + limit]
                logger.info(f"准备处理第{start_idx + 1}-{start_idx + len(stock_df)}只股票 (共{total_stocks}只)")
            else:
                stock_df = stock_df[start_idx:]
                logger.info(f"准备处理第{start_idx + 1}-{total_stocks}只股票")
            
            start_time = time.time()
            
            # 创建股票代码映射
            stock_code_map = {}  # ts_code -> (security_id, name)
            ts_codes = []
            
            for _, row in stock_df.iterrows():
                ts_code = self._convert_stock_code(row['code'])
                stock_code_map[ts_code] = (row['id'], row['name'])
                ts_codes.append(ts_code)
            
            # 分批处理 - 创建批次列表
            batch_list = []
            for batch_idx in range(0, len(ts_codes), self.batch_size):
                batch_codes = ts_codes[batch_idx:batch_idx + self.batch_size]
                batch_list.append(batch_codes)
            
            total_batches = len(batch_list)
            logger.info(f"使用批量API模式，批次大小: {self.batch_size}, 总批次: {total_batches}")
            
            if total_batches > 3:
                logger.info(f"批次较多，启用并行批处理，并发数: {min(self.max_workers, total_batches)}")
            
            # 处理所有批次（可能并行）
            success_count, fail_count, total_records = self._process_batch_list(batch_list, stock_code_map, periods)
            
            # 最终统计
            elapsed = time.time() - start_time
            logger.info(f"🎉 财务指标数据获取完成！")
            logger.info(f"📊 处理股票: {len(stock_df)}只 ({total_batches}个批次)")
            logger.info(f"✅ 成功: {success_count}只")
            logger.info(f"❌ 失败: {fail_count}只")
            logger.info(f"📝 总记录: {total_records}条")
            logger.info(f"⏱️ 用时: {elapsed/60:.1f}分钟")
            logger.info(f"⚡ 处理速度: {len(stock_df)/(elapsed/60):.1f}只/分钟")
            
            return total_records
            
        except Exception as e:
            logger.error(f"批量获取财务指标数据失败: {e}")
            return 0
    
    def _convert_stock_code(self, code: str) -> str:
        """转换股票代码为Tushare格式"""
        if len(code) != 6:
            return code
        
        # A股代码转换规则
        if code.startswith('6'):
            return f"{code}.SH"  # 上海
        elif code.startswith('0') or code.startswith('3'):
            return f"{code}.SZ"  # 深圳
        elif code.startswith('8'):
            return f"{code}.BJ"  # 北交所
        else:
            return f"{code}.SH"  # 默认上海
    
    def _save_financial_indicator_data(self, df: pd.DataFrame, security_id: int) -> int:
        """保存财务指标数据到数据库"""
        if df is None or df.empty:
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                saved_count = 0
                
                for _, row in df.iterrows():
                    # 使用核心字段的INSERT语句
                    cursor.execute("""
                        INSERT OR IGNORE INTO financial_indicator (
                            security_id, ann_date, end_date, eps, gross_margin, 
                            current_ratio, roe, roa, netprofit_margin, debt_to_assets, 
                            basic_eps_yoy, netprofit_yoy, or_yoy
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        security_id,
                        row.get('ann_date'),
                        row.get('end_date'), 
                        row.get('eps'),
                        row.get('gross_margin'),
                        row.get('current_ratio'),
                        row.get('roe'),
                        row.get('roa'),
                        row.get('netprofit_margin'),
                        row.get('debt_to_assets'),
                        row.get('basic_eps_yoy'),
                        row.get('netprofit_yoy'),
                        row.get('or_yoy')
                    ))
                    
                    if cursor.rowcount > 0:
                        saved_count += 1
                
                conn.commit()
                return saved_count
                
        except Exception as e:
            logger.error(f"保存财务指标数据失败: {e}")
            return 0

    def fetch_daily_basic_data(self, trade_date: str = None) -> int:
        """获取每日基本面数据 - 优化版本：批量获取所有股票数据"""
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
        else:
            # 转换日期格式 YYYY-MM-DD -> YYYYMMDD
            trade_date = trade_date.replace('-', '')
        
        logger.info(f"开始批量获取 {trade_date} 的每日基本面数据...")
        
        try:
            # 检查是否已有当日数据
            with sqlite3.connect(self.db_path) as conn:
                existing_count = conn.execute(
                    "SELECT COUNT(*) FROM daily_basic WHERE trade_date = ?",
                    (pd.to_datetime(trade_date).strftime('%Y-%m-%d'),)
                ).fetchone()[0]
                
                if existing_count > 0:
                    logger.info(f"已存在 {existing_count} 条 {trade_date} 的数据，跳过获取")
                    return existing_count
            
            # 使用批量API调用 - 一次获取所有股票的数据
            self._rate_limit()
            
            logger.info(f"调用批量API获取所有股票的 {trade_date} 数据...")
            
            # 批量获取所有股票的每日基本面数据（不指定ts_code即获取全部）
            daily_basic = self.pro.daily_basic(
                trade_date=trade_date,
                fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv'
            )
            
            if daily_basic.empty:
                logger.warning(f"未获取到 {trade_date} 的每日基本面数据")
                return 0
            
            total_records = len(daily_basic)
            logger.info(f"获取到 {total_records} 条每日基本面数据，开始批量保存...")
            
            # 批量保存数据
            saved_count = self._save_daily_basic_data_batch(daily_basic)
            
            logger.info(f"成功获取并保存 {saved_count} 条每日基本面数据")
            return saved_count
            
        except Exception as e:
            logger.error(f"获取每日基本面数据失败: {e}")
            return 0
    
    def _save_daily_basic_data(self, daily_basic: pd.DataFrame) -> int:
        """保存每日基本面数据"""
        saved_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for _, row in daily_basic.iterrows():
                try:
                    ts_code = row['ts_code']
                    stock_code = ts_code.split('.')[0]
                    
                    # 获取security_id
                    cursor = conn.execute(
                        "SELECT id FROM securities WHERE code = ?",
                        (stock_code,)
                    )
                    result = cursor.fetchone()
                    
                    if not result:
                        continue
                    
                    security_id = result[0]
                    trade_date = pd.to_datetime(row['trade_date']).strftime('%Y-%m-%d')
                    
                    # 插入数据
                    insert_sql = """
                    INSERT OR REPLACE INTO daily_basic 
                    (security_id, trade_date, close, turnover_rate, turnover_rate_f, 
                     volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                     total_share, float_share, free_share, total_mv, circ_mv)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    conn.execute(insert_sql, (
                        security_id,
                        trade_date,
                        row.get('close'),
                        row.get('turnover_rate'),
                        row.get('turnover_rate_f'),
                        row.get('volume_ratio'),
                        row.get('pe'),
                        row.get('pe_ttm'),
                        row.get('pb'),
                        row.get('ps'),
                        row.get('ps_ttm'),
                        row.get('dv_ratio'),
                        row.get('dv_ttm'),
                        row.get('total_share'),
                        row.get('float_share'),
                        row.get('free_share'),
                        row.get('total_mv'),
                        row.get('circ_mv')
                    ))
                    
                    saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"保存每日基本面数据失败: {e}")
                    continue
            
            conn.commit()
        
        return saved_count
    
    def _save_daily_basic_data_batch(self, daily_basic: pd.DataFrame) -> int:
        """批量保存每日基本面数据 - 性能优化版本"""
        saved_count = 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 首先获取所有securities的映射，避免重复查询
                securities_df = pd.read_sql_query(
                    "SELECT code, id FROM securities", 
                    conn
                )
                securities_map = dict(zip(securities_df['code'], securities_df['id']))
                
                # 预处理数据，批量转换
                data_to_insert = []
                for _, row in daily_basic.iterrows():
                    try:
                        ts_code = row['ts_code']
                        stock_code = ts_code.split('.')[0]
                        
                        # 查找security_id
                        security_id = securities_map.get(stock_code)
                        if not security_id:
                            continue
                        
                        trade_date = pd.to_datetime(row['trade_date']).strftime('%Y-%m-%d')
                        
                        data_to_insert.append((
                            security_id,
                            trade_date,
                            row.get('close'),
                            row.get('turnover_rate'),
                            row.get('turnover_rate_f'),
                            row.get('volume_ratio'),
                            row.get('pe'),
                            row.get('pe_ttm'),
                            row.get('pb'),
                            row.get('ps'),
                            row.get('ps_ttm'),
                            row.get('dv_ratio'),
                            row.get('dv_ttm'),
                            row.get('total_share'),
                            row.get('float_share'),
                            row.get('free_share'),
                            row.get('total_mv'),
                            row.get('circ_mv')
                        ))
                        
                    except Exception as e:
                        logger.warning(f"处理数据行失败: {e}")
                        continue
                
                # 批量插入数据
                if data_to_insert:
                    insert_sql = """
                    INSERT OR REPLACE INTO daily_basic 
                    (security_id, trade_date, close, turnover_rate, turnover_rate_f, 
                     volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                     total_share, float_share, free_share, total_mv, circ_mv)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    # 使用executemany进行批量插入
                    conn.executemany(insert_sql, data_to_insert)
                    conn.commit()
                    saved_count = len(data_to_insert)
                    
                    logger.info(f"批量保存 {saved_count} 条每日基本面数据")
                
        except Exception as e:
            logger.error(f"批量保存每日基本面数据失败: {e}")
            
        return saved_count
    
    def fetch_market_indices(self) -> int:
        """获取大盘指数信息 - 带智能缓存"""
        logger.info("开始获取大盘指数信息...")
        
        # 检查缓存是否新鲜
        if self._is_data_fresh('market_indices', self.cache_settings['market_indices_days']):
            logger.info(f"指数信息缓存仍然新鲜（{self.cache_settings['market_indices_days']}天内），跳过获取")
            
            # 返回现有记录数
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM market_indices").fetchone()[0]
                return count
        
        try:
            self._rate_limit()
            
            # 获取指数基本信息
            index_basic = self.pro.index_basic(market='SSE,SZSE,CSI')
            
            if index_basic.empty:
                logger.warning("未获取到指数信息")
                return 0
            
            # 过滤主要指数
            major_indices = [
                '000001.SH',  # 上证指数
                '399001.SZ',  # 深证成指
                '399006.SZ',  # 创业板指
                '000300.SH',  # 沪深300
                '000905.SH',  # 中证500
                '000016.SH',  # 上证50
                '399005.SZ',  # 中小板指
                '000002.SH',  # A股指数
                '000003.SH',  # B股指数
                '000688.SH',  # 科创50
            ]
            
            # 筛选主要指数
            index_basic = index_basic[index_basic['ts_code'].isin(major_indices)]
            
            logger.info(f"获取到 {len(index_basic)} 个主要指数信息")
            
            # 保存指数基本信息
            saved_count = self._save_market_indices(index_basic)
            
            return saved_count
            
        except Exception as e:
            logger.error(f"获取大盘指数信息失败: {e}")
            return 0
    
    def _save_market_indices(self, index_basic: pd.DataFrame) -> int:
        """保存指数基本信息"""
        saved_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for _, row in index_basic.iterrows():
                try:
                    insert_sql = """
                    INSERT OR REPLACE INTO market_indices 
                    (ts_code, name, fullname, market, publisher, index_type, 
                     category, base_date, base_point, list_date, weight_rule, desc_detail)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    conn.execute(insert_sql, (
                        row.get('ts_code'),
                        row.get('name'),
                        row.get('fullname'),
                        row.get('market'),
                        row.get('publisher'),
                        row.get('index_type'),
                        row.get('category'),
                        row.get('base_date'),
                        row.get('base_point'),
                        row.get('list_date'),
                        row.get('weight_rule'),
                        row.get('desc')
                    ))
                    
                    saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"保存指数 {row.get('ts_code', 'Unknown')} 信息失败: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"成功保存 {saved_count} 个指数信息")
        return saved_count
    
    def fetch_index_daily_data(self, trade_date: str = None, days_back: int = 30) -> int:
        """获取指数日线数据"""
        if trade_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        else:
            end_date = trade_date.replace('-', '')
        
        start_date = (datetime.strptime(end_date, '%Y%m%d') - timedelta(days=days_back)).strftime('%Y%m%d')
        
        logger.info(f"开始获取指数日线数据: {start_date} 到 {end_date}")
        
        try:
            # 获取已存储的指数
            with sqlite3.connect(self.db_path) as conn:
                indices = pd.read_sql_query(
                    "SELECT id, ts_code, name FROM market_indices", 
                    conn
                )
            
            if indices.empty:
                logger.warning("未找到已存储的指数信息，请先运行 fetch_market_indices()")
                return 0
            
            total_saved = 0
            
            for _, index_row in indices.iterrows():
                try:
                    self._rate_limit()
                    
                    # 获取指数日线数据
                    index_daily = self.pro.index_daily(
                        ts_code=index_row['ts_code'],
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if not index_daily.empty:
                        saved = self._save_index_daily_data(index_daily, index_row['id'])
                        total_saved += saved
                        logger.info(f"保存指数 {index_row['name']} 数据 {saved} 条")
                    
                except Exception as e:
                    logger.warning(f"获取指数 {index_row['name']} 日线数据失败: {e}")
                    continue
            
            logger.info(f"总共保存指数日线数据 {total_saved} 条")
            return total_saved
            
        except Exception as e:
            logger.error(f"获取指数日线数据失败: {e}")
            return 0
    
    def _save_index_daily_data(self, index_daily: pd.DataFrame, index_id: int) -> int:
        """保存指数日线数据"""
        saved_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            for _, row in index_daily.iterrows():
                try:
                    trade_date = pd.to_datetime(row['trade_date']).strftime('%Y-%m-%d')
                    
                    insert_sql = """
                    INSERT OR REPLACE INTO index_daily 
                    (index_id, trade_date, close, open, high, low, pre_close,
                     change, pct_chg, vol, amount)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    conn.execute(insert_sql, (
                        index_id,
                        trade_date,
                        row.get('close'),
                        row.get('open'),
                        row.get('high'),
                        row.get('low'),
                        row.get('pre_close'),
                        row.get('change'),
                        row.get('pct_chg'),
                        row.get('vol'),
                        row.get('amount')
                    ))
                    
                    saved_count += 1
                    
                except Exception as e:
                    logger.warning(f"保存指数日线数据失败: {e}")
                    continue
            
            conn.commit()
        
        return saved_count
    
    def fetch_daily_basic_data_range(self, start_date: str, end_date: str = None, incremental: bool = True) -> int:
        """获取指定日期范围的每日基本面数据 - 支持增量更新"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"开始获取 {start_date} 到 {end_date} 的每日基本面数据...")
        
        try:
            # 如果启用增量更新，只获取缺失的日期
            if incremental and self.cache_settings['enable_incremental']:
                missing_dates = self._get_missing_dates(start_date, end_date)
                
                if not missing_dates:
                    logger.info("没有缺失的日期数据，跳过获取")
                    
                    # 返回现有数据量
                    with sqlite3.connect(self.db_path) as conn:
                        count = conn.execute(
                            "SELECT COUNT(*) FROM daily_basic WHERE trade_date BETWEEN ? AND ?",
                            (start_date, end_date)
                        ).fetchone()[0]
                        return count
                
                logger.info(f"发现 {len(missing_dates)} 个缺失的交易日，将逐个获取")
                dates_to_fetch = missing_dates[:10]  # 限制每次最多获取10个日期
            else:
                # 生成完整日期列表
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                
                dates_to_fetch = []
                current_date = start_dt
                while current_date <= end_dt:
                    if current_date.weekday() < 5:  # 工作日
                        dates_to_fetch.append(current_date.strftime('%Y-%m-%d'))
                    current_date += timedelta(days=1)
            
            total_saved = 0
            
            # 逐个日期获取数据
            for date_str in dates_to_fetch:
                try:
                    saved = self.fetch_daily_basic_data(date_str)
                    total_saved += saved
                    logger.info(f"获取 {date_str} 数据: {saved} 条")
                    
                    # 避免过快调用API
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"获取 {date_str} 数据失败: {e}")
                    continue
            
            logger.info(f"总共获取 {total_saved} 条每日基本面数据")
            return total_saved
            
        except Exception as e:
            logger.error(f"获取日期范围数据失败: {e}")
            return 0
    
    def run_full_update(self, trade_date: str = None) -> Dict[str, int]:
        """运行完整的基本面数据更新"""
        logger.info("开始运行完整的基本面数据更新...")
        
        results = {
            'stock_basic_info': 0,
            'daily_basic': 0,
            'market_indices': 0,
            'index_daily': 0,
            'financial_indicators': 0
        }
        
        try:
            # 1. 初始化数据库表
            self.init_database_tables()
            
            # 2. 获取股票基本信息（只需要运行一次或定期更新）
            results['stock_basic_info'] = self.fetch_stock_basic_info()
            
            # 3. 获取每日基本面数据
            results['daily_basic'] = self.fetch_daily_basic_data(trade_date)
            
            # 4. 获取指数基本信息（只需要运行一次或定期更新）
            results['market_indices'] = self.fetch_market_indices()
            
            # 5. 获取指数日线数据
            results['index_daily'] = self.fetch_index_daily_data(trade_date)
            
            # 6. 获取财务指标数据（限制数量避免API限制）
            logger.info("开始获取财务指标数据（批量处理）...")
            results['financial_indicators'] = self.fetch_financial_indicators(
                periods=6,  # 获取最近6期财报数据
                start_idx=0,
                limit=200,  # 限制为200只股票避免API超限
                force=False
            )
            
            logger.info("完整的基本面数据更新完成")
            logger.info(f"更新结果: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"完整更新失败: {e}")
            return results


def main():
    """主函数 - 用于测试和命令行执行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="基本面数据获取器 - 优化版本")
    parser.add_argument("--mode", choices=['full', 'basic', 'daily', 'indices', 'range', 'financial'], 
                       default='daily', help="运行模式")
    parser.add_argument("--date", type=str, help="交易日期 (YYYY-MM-DD)")
    parser.add_argument("--start-date", type=str, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--config", type=str, default="config.json", help="配置文件路径")
    parser.add_argument("--force", action='store_true', help="强制更新，忽略缓存")
    parser.add_argument("--no-incremental", action='store_true', help="禁用增量更新")
    parser.add_argument("--start", type=int, default=0, help="起始索引（仅financial模式）")
    parser.add_argument("--limit", type=int, default=None, help="处理数量限制（仅financial模式）")
    parser.add_argument("--periods", type=int, default=8, help="财务指标获取期数（仅financial模式）")
    parser.add_argument("--batch-size", type=int, default=20, help="批量API每批处理股票数量（仅financial模式）")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        fetcher = FundamentalDataFetcher(args.config)
        
        # 如果强制更新，禁用缓存
        if args.force:
            fetcher.cache_settings['stock_basic_info_days'] = 0
            fetcher.cache_settings['market_indices_days'] = 0
            fetcher.cache_settings['financial_indicator_days'] = 0
        
        # 如果禁用增量更新
        if args.no_incremental:
            fetcher.cache_settings['enable_incremental'] = False
        
        if args.mode == 'full':
            results = fetcher.run_full_update(args.date)
            print(f"完整更新结果: {results}")
            
        elif args.mode == 'basic':
            count = fetcher.fetch_stock_basic_info(update_existing=args.force)
            print(f"更新股票基本信息: {count} 条")
            
        elif args.mode == 'daily':
            count = fetcher.fetch_daily_basic_data(args.date)
            print(f"获取每日基本面数据: {count} 条")
            
        elif args.mode == 'range':
            if not args.start_date:
                args.start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            count = fetcher.fetch_daily_basic_data_range(
                args.start_date, 
                args.end_date, 
                incremental=not args.no_incremental
            )
            print(f"获取日期范围数据: {count} 条")
            
        elif args.mode == 'indices':
            fetcher.fetch_market_indices()
            count = fetcher.fetch_index_daily_data(args.date)
            print(f"获取指数数据: {count} 条")
            
        elif args.mode == 'financial':
            # 设置批次大小
            if args.batch_size != 20:
                fetcher.batch_size = args.batch_size
                
            count = fetcher.fetch_financial_indicators(
                periods=args.periods,
                start_idx=args.start,
                limit=args.limit,
                force=args.force
            )
            print(f"获取财务指标数据: {count} 条")
            
    except Exception as e:
        logger.error(f"执行失败: {e}")


if __name__ == "__main__":
    main()