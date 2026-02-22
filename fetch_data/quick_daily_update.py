#!/usr/bin/env python3
"""
快速每日数据更新 - 使用批量API调用并写入数据库
"""

import pandas as pd
import tushare as ts
import time
import logging
from pathlib import Path
from datetime import datetime
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_adapter.database_manager import DatabaseManager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pure_tushare_news_fetcher import PureTushareNewsFetcher

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# 读取配置
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)
    ts_token = config['tushare']['token']

ts.set_token(ts_token)
pro = ts.pro_api()

# 初始化数据库管理器
db_manager = DatabaseManager()

def batch_update_stocks(date_str: str, batch_size: int = 500):
    """批量更新A股数据到数据库"""
    logger.info(f"开始批量更新A股数据: {date_str}")
    
    try:
        # 一次API调用获取所有A股数据
        df = pro.daily(
            trade_date=date_str,
            fields='ts_code,trade_date,open,close,high,low,vol,pct_chg,limit'
        )
        
        if df.empty:
            logger.warning(f"没有找到 {date_str} 的A股数据")
            return 0
        
        logger.info(f"获取到 {len(df)} 只A股的数据")
        
        # 准备批量插入数据
        data_to_insert = []
        success_count = 0
        
        for _, row in df.iterrows():
            try:
                code = row['ts_code'].split('.')[0]
                exchange = row['ts_code'].split('.')[1]
                
                # 插入或获取证券ID
                security_id = db_manager.insert_security(
                    code=code,
                    name=code,  # 名称需要单独获取，这里先用代码代替
                    security_type='A股',
                    exchange=exchange
                )
                
                if security_id:
                    # 准备日线数据
                    trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                    data_to_insert.append({
                        'security_id': security_id,
                        'trade_date': trade_date,
                        'open': row['open'],
                        'close': row['close'],
                        'high': row['high'],
                        'low': row['low'],
                        'volume': row['vol'],
                        'price_change_pct': row.get('pct_chg', 0) / 100 if row.get('pct_chg') else 0,
                        'is_limit_up': row.get('limit') == 'U' if row.get('limit') else False,
                        'is_limit_down': row.get('limit') == 'D' if row.get('limit') else False
                    })

                    success_count += 1
                    
            except Exception as e:
                logger.error(f"处理 {row['ts_code']} 失败: {e}")
                continue
        
        # 批量插入数据库
        if data_to_insert:
            db_rows = db_manager.insert_daily_quotes(data_to_insert)
            logger.info(f"数据库插入 {db_rows} 条记录")
        
        logger.info(f"A股更新完成: {success_count} 只")
        return success_count
        
    except Exception as e:
        logger.error(f"批量获取A股数据失败: {e}")
        return 0

def batch_update_funds(date_str: str):
    """批量更新ETF/基金数据到数据库"""
    logger.info(f"开始批量更新ETF/基金数据: {date_str}")
    
    try:
        # 一次API调用获取所有基金数据
        df = pro.fund_daily(
            trade_date=date_str,
            fields='ts_code,trade_date,open,close,high,low,vol,pct_chg'
        )
        
        if df.empty:
            logger.warning(f"没有找到 {date_str} 的基金数据")
            return 0
        
        logger.info(f"获取到 {len(df)} 只基金的数据")
        
        # 准备批量插入数据
        data_to_insert = []
        success_count = 0
        
        for _, row in df.iterrows():
            try:
                code = row['ts_code'].split('.')[0]
                exchange = row['ts_code'].split('.')[1]
                
                # 插入或获取证券ID
                security_id = db_manager.insert_security(
                    code=code,
                    name=code,  # 名称需要单独获取，这里先用代码代替
                    security_type='ETF_基金',
                    exchange=exchange
                )
                
                if security_id:
                    # 准备日线数据
                    trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                    data_to_insert.append({
                        'security_id': security_id,
                        'trade_date': trade_date,
                        'open': row['open'],
                        'close': row['close'],
                        'high': row['high'],
                        'low': row['low'],
                        'volume': row['vol'],
                        'price_change_pct': row.get('pct_chg', 0) / 100 if row.get('pct_chg') else 0,
                        'is_limit_up': False,
                        'is_limit_down': False
                    })

                    success_count += 1
                    
            except Exception as e:
                logger.error(f"处理 {row['ts_code']} 失败: {e}")
                continue
        
        # 批量插入数据库
        if data_to_insert:
            db_rows = db_manager.insert_daily_quotes(data_to_insert)
            logger.info(f"数据库插入 {db_rows} 条记录")
        
        logger.info(f"基金更新完成: {success_count} 只")
        return success_count
        
    except Exception as e:
        logger.error(f"批量获取基金数据失败: {e}")
        return 0

def update_market_indices(date_str: str):
    """更新大盘指数数据"""
    logger.info(f"开始更新 {date_str} 的大盘指数数据...")
    
    try:
        # 初始化数据获取器
        fetcher = PureTushareNewsFetcher(config_path)
        
        # 重要A股指数（包含中证2000和中证全指）
        important_indices = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指', 
            '399006.SZ': '创业板指',
            '000688.SH': '科创50',
            '000016.SH': '上证50',
            '000300.SH': '沪深300',
            '000905.SH': '中证500',
            '000852.SH': '中证1000',
            '932000.CSI': '中证2000',
            '000985.SH': '中证全指'
        }
        
        success_count = 0
        
        for ts_code, name in important_indices.items():
            try:
                time.sleep(0.3)  # API调用间隔
                
                df = pro.index_daily(
                    ts_code=ts_code,
                    trade_date=date_str,
                    fields='ts_code,trade_date,open,close,high,low,vol,amount,pct_chg'
                )
                
                if not df.empty:
                    row = df.iloc[0]
                    
                    # 插入或获取证券ID
                    code = ts_code
                    security_id = db_manager.insert_security(
                        code=code,
                        name=name,
                        security_type='指数',
                        exchange=ts_code.split('.')[1] if '.' in ts_code else 'CSI'
                    )
                    
                    if security_id:
                        # 准备指数数据
                        trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                        data_to_insert = {
                            'security_id': security_id,
                            'trade_date': trade_date,
                            'open': row['open'],
                            'close': row['close'],
                            'high': row['high'],
                            'low': row['low'],
                            'volume': row.get('vol', 0),
                            'price_change_pct': row.get('pct_chg', 0) / 100 if row.get('pct_chg') else 0,
                            'is_limit_up': False,
                            'is_limit_down': False
                        }

                        # 插入数据库
                        db_manager.insert_daily_quotes([data_to_insert])

                        success_count += 1
                        logger.info(f"更新{name}成功: {row['close']:.2f} ({row.get('pct_chg', 0):+.2f}%)")
                    
            except Exception as e:
                logger.warning(f"更新{name}({ts_code})失败: {e}")
                continue
        
        logger.info(f"大盘指数更新完成: {success_count} 个")
        return success_count
        
    except Exception as e:
        logger.error(f"更新大盘指数失败: {e}")
        return 0

def update_daily_basic(date_str: str):
    """更新每日基本面指标（PE、PB、市值等）"""
    logger.info(f"开始更新 {date_str} 的基本面指标...")
    
    try:
        # 获取daily_basic数据
        df = pro.daily_basic(trade_date=date_str)
        
        if df.empty:
            logger.warning(f"未获取到 {date_str} 的基本面数据")
            return 0
        
        logger.info(f"获取到 {len(df)} 条基本面指标")
        
        # 连接数据库
        import sqlite3
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        
        # 获取证券ID映射
        cursor.execute('SELECT code, id FROM securities')
        security_map = {row[0]: row[1] for row in cursor.fetchall()}
        
        count = 0
        for _, row in df.iterrows():
            code = row['ts_code'][:6]
            if code in security_map:
                try:
                    # 统一日期格式为 YYYY-MM-DD
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_basic 
                        (security_id, trade_date, close, turnover_rate, turnover_rate_f,
                         volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                         total_share, float_share, free_share, total_mv, circ_mv)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        security_map[code], formatted_date, row.get('close'),
                        row.get('turnover_rate'), row.get('turnover_rate_f'),
                        row.get('volume_ratio'), row.get('pe'), row.get('pe_ttm'),
                        row.get('pb'), row.get('ps'), row.get('ps_ttm'),
                        row.get('dv_ratio'), row.get('dv_ttm'),
                        row.get('total_share'), row.get('float_share'), row.get('free_share'),
                        row.get('total_mv'), row.get('circ_mv')
                    ))
                    count += 1
                except Exception as e:
                    logger.debug(f"插入基本面数据失败 {code}: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ 基本面指标更新完成: {count} 条")
        return count
        
    except Exception as e:
        logger.error(f"更新基本面指标失败: {e}")
        return 0

def update_financial_indicators(date_str: str):
    """检查并更新当天或前一天发布财务数据的公司"""
    logger.info(f"检查 {date_str} 及前一天发布财务数据的公司...")
    
    try:
        # 获取今天和前一天的日期
        today = int(date_str)
        yesterday = today - 1
        target_dates = [str(today), str(yesterday)]
        
        # 从数据库获取活跃股票列表（限制数量避免API超限）
        import sqlite3
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT s.code || '.' || 
                   CASE WHEN s.exchange = 'SH' THEN 'SH' 
                        WHEN s.exchange = 'SZ' THEN 'SZ'
                        ELSE s.exchange END as ts_code
            FROM securities s 
            WHERE s.type = 'A股'
            ORDER BY s.id
        """)
        stock_codes = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not stock_codes:
            logger.info("未找到活跃股票，跳过财务指标更新")
            return 0
            
        logger.info(f"并行检查 {len(stock_codes)} 只A股的财务数据发布情况")
        
        # 并行检查哪些股票在目标日期发布了财务数据
        import concurrent.futures
        import threading
        
        companies_with_new_reports = []
        checked_count = 0
        lock = threading.Lock()
        
        def check_single_stock(ts_code):
            """检查单只股票的财务数据发布情况"""
            try:
                # 获取该股票最新的财务指标，检查公告日期
                # v3.9扩展：添加更多财务指标字段
                fina_df = pro.fina_indicator(
                    ts_code=ts_code,
                    fields='ts_code,ann_date,end_date,eps,dt_eps,roe,roe_waa,roe_dt,roa,'
                           'grossprofit_margin,netprofit_margin,profit_to_gr,'
                           'ocf_to_profit,debt_to_assets,current_ratio,quick_ratio,'
                           'ar_turn,ca_turn,fa_turn,assets_turn,'
                           'netprofit_yoy,or_yoy'
                )
                
                if not fina_df.empty:
                    latest = fina_df.iloc[0]
                    ann_date = str(latest['ann_date']) if latest['ann_date'] else ""
                    
                    # 检查是否在目标日期发布
                    if ann_date in target_dates:
                        with lock:
                            companies_with_new_reports.append({
                                'ts_code': ts_code,
                                'ann_date': ann_date,
                                'end_date': latest['end_date'],
                                'data': latest
                            })
                        logger.info(f"发现 {ts_code} 在 {ann_date} 发布财务数据")
                
                # 更新进度
                nonlocal checked_count
                with lock:
                    checked_count += 1
                    if checked_count % 100 == 0:
                        logger.info(f"已检查 {checked_count}/{len(stock_codes)} 只股票")
                        
                return True
                    
            except Exception as e:
                logger.debug(f"检查 {ts_code} 财务数据失败: {e}")
                return False
        
        # 使用线程池并行处理，控制并发数避免API限制
        max_workers = 5  # 控制并发数量
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(check_single_stock, ts_code) for ts_code in stock_codes]
            
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        
        if not companies_with_new_reports:
            logger.info("未发现当天或前一天发布财务数据的公司")
            return 0
        
        logger.info(f"发现 {len(companies_with_new_reports)} 家公司发布了新的财务数据")
        
        # 保存这些新的财务数据
        count = 0
        for company in companies_with_new_reports:
            try:
                # 构造DataFrame格式以复用现有保存逻辑
                df = pd.DataFrame([company['data']])
                saved = save_financial_data_to_db(df)
                count += saved
                
                if saved > 0:
                    logger.info(f"更新 {company['ts_code']} 财务数据成功 (公告日期: {company['ann_date']})")
                    
            except Exception as e:
                logger.debug(f"保存 {company['ts_code']} 财务数据失败: {e}")
        
        logger.info(f"✅ 新财报公司财务指标更新完成: {count} 条")
        return count
        
    except Exception as e:
        logger.error(f"更新财务指标失败: {e}")
        return 0


def save_financial_data_to_db(df):
    """保存财务数据到数据库"""
    try:
        import sqlite3
        conn = sqlite3.connect('data_adapter/stock_data.db')
        cursor = conn.cursor()
        
        # 获取证券ID映射
        cursor.execute('SELECT code, id FROM securities')
        security_map = {row[0]: row[1] for row in cursor.fetchall()}
        
        count = 0
        for _, row in df.iterrows():
            code = row['ts_code'][:6]  # 获取6位代码
            if code in security_map:
                try:
                    # 插入或更新财务数据 (v3.9扩展：包含更多字段)
                    cursor.execute('''
                        INSERT OR REPLACE INTO financial_indicator
                        (security_id, ann_date, end_date, eps, dt_eps, roe, roe_waa, roe_dt, roa,
                         grossprofit_margin, netprofit_margin, profit_to_gr, ocf_to_profit,
                         debt_to_assets, current_ratio, quick_ratio, ar_turn, ca_turn, fa_turn, assets_turn,
                         netprofit_yoy, or_yoy)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        security_map[code],
                        row.get('ann_date'),
                        row.get('end_date'),
                        row.get('eps'),
                        row.get('dt_eps'),
                        row.get('roe'),
                        row.get('roe_waa'),
                        row.get('roe_dt'),
                        row.get('roa'),
                        row.get('grossprofit_margin'),
                        row.get('netprofit_margin'),
                        row.get('profit_to_gr'),
                        row.get('ocf_to_profit'),
                        row.get('debt_to_assets'),
                        row.get('current_ratio'),
                        row.get('quick_ratio'),
                        row.get('ar_turn'),
                        row.get('ca_turn'),
                        row.get('fa_turn'),
                        row.get('assets_turn'),
                        row.get('netprofit_yoy'),
                        row.get('or_yoy')
                    ))
                    count += 1
                except Exception as e:
                    logger.debug(f"插入财务数据失败 {code}: {e}")
        
        conn.commit()
        conn.close()
        return count
        
    except Exception as e:
        logger.error(f"保存财务数据失败: {e}")
        return 0

def calculate_technical_indicators(date_str: str):
    """计算技术指标"""
    logger.info(f"开始计算 {date_str} 的技术指标...")

    try:
        import subprocess
        # 转换日期格式
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        result = subprocess.run([
            "python3", "fetch_data/technical_indicator_calculator.py",
            "--start-date", date_dash,
            "--end-date", date_dash
        ], capture_output=True, text=True)

        if result.returncode == 0:
            # 从输出中提取计算的记录数
            import re
            match = re.search(r'成功计算 (\d+) 只股票', result.stdout)
            if match:
                count = int(match.group(1))
                logger.info(f"✅ 技术指标计算成功: {count} 条")
                return count
            else:
                logger.info("✅ 技术指标计算完成")
                return 0
        else:
            logger.error(f"❌ 技术指标计算失败: {result.stderr}")
            return 0
    except Exception as e:
        logger.error(f"❌ 技术指标计算异常: {e}")
        return 0


def update_active_mv_features(date_str: str):
    """更新V3.9.4活跃市值特征 (用于ML评分) - 优化版"""
    logger.info(f"开始更新 {date_str} 的活跃市值特征...")

    try:
        # 转换日期格式
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # 直接调用预计算模块，避免subprocess开销
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from precompute_active_mv_features import precompute_features, verify_cache

        # 只处理当天日期
        precompute_features(date_dash, date_dash, batch_size=1)

        # 验证结果
        stats = verify_cache(date_dash, date_dash)
        if stats is not None and len(stats) > 0:
            count = int(stats['stock_count'].sum())
            logger.info(f"✅ 活跃市值特征更新成功: {count} 条")
            return count
        else:
            logger.info("✅ 活跃市值特征更新完成 (可能已缓存)")
            return 0

    except Exception as e:
        logger.warning(f"⚠️ 活跃市值特征更新异常: {e}")
        # 出错时不阻塞主流程
        return 0

def update_v39_feature_cache(date_str: str):
    """更新V3.9/V3.95特征缓存 (支持ML评分) - 优化版"""
    logger.info(f"开始更新 {date_str} 的V3.9特征缓存 (优化版)...")

    try:
        # 直接调用优化版更新器，避免subprocess开销
        from fetch_data.v39_feature_cache_updater import V39FeatureCacheUpdaterOptimized

        # 转换日期格式
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        updater = V39FeatureCacheUpdaterOptimized()
        count = updater.update_single_date(date_dash)

        if count > 0:
            logger.info(f"✅ V3.9特征缓存更新成功: {count} 条")
        else:
            logger.info("✅ V3.9特征缓存更新完成 (无新数据)")

        return count

    except Exception as e:
        logger.warning(f"⚠️ V3.9特征缓存更新异常: {e}")
        # 出错时不阻塞主流程
        return 0


def update_squeeze_momentum_indicators(date_str: str):
    """更新挤压动量指标 (v3.2新增)"""
    logger.info(f"开始更新 {date_str} 的挤压动量指标...")
    
    try:
        import subprocess
        # 转换日期格式
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        
        # 计算需要的历史数据范围 (挤压动量需要60天历史数据)
        from datetime import datetime, timedelta
        end_date = datetime.strptime(date_dash, '%Y-%m-%d')
        start_date = end_date - timedelta(days=90)  # 多留一些余量确保有足够数据
        start_date_str = start_date.strftime('%Y-%m-%d')
        
        result = subprocess.run([
            "python3", "data_adapter/squeeze_momentum_updater.py",
            "--start-date", start_date_str,
            "--end-date", date_dash,
            "--max-securities", "50"  # 限制数量以控制更新时间
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # 从输出中提取成功更新的股票数
            import re
            match = re.search(r'成功: (\d+) 只', result.stdout)
            if match:
                count = int(match.group(1))
                logger.info(f"✅ 挤压动量指标更新成功: {count} 只股票")
                return count
            else:
                logger.info("✅ 挤压动量指标更新完成")
                return 0
        else:
            logger.warning(f"⚠️ 挤压动量指标更新遇到问题: {result.stderr}")
            return 0
    except Exception as e:
        logger.warning(f"⚠️ 挤压动量指标更新异常: {e}")
        return 0

def update_sw_index_daily(date_str: str):
    """更新申万行业指数日线数据"""
    logger.info(f"开始更新 {date_str} 的申万行业指数数据...")

    try:
        from fetch_data.sw_daily_fetcher import SWDailyFetcher

        fetcher = SWDailyFetcher()
        count = fetcher.fetch_single_date(date_str)
        if count > 0:
            logger.info(f"申万行业指数更新完成: {count} 条")
        else:
            logger.info("申万行业指数: 无数据 (可能非交易日)")
        return count

    except Exception as e:
        logger.warning(f"申万行业指数更新异常: {e}")
        return 0


def update_hsgt_daily(date_str: str):
    """更新沪深港通资金流向数据"""
    logger.info(f"开始更新 {date_str} 的北向资金数据...")

    try:
        from fetch_data.hsgt_daily_fetcher import HSGTDailyFetcher

        fetcher = HSGTDailyFetcher()
        count = fetcher.fetch_single_date(date_str)
        if count > 0:
            logger.info(f"北向资金数据更新完成: {count} 条")
        else:
            logger.info("北向资金: 无数据 (可能非交易日)")
        return count

    except Exception as e:
        logger.warning(f"北向资金更新异常: {e}")
        return 0


def check_sw_industry(date_str: str):
    """检查并更新申万行业分类 (月度更新)"""
    logger.info("检查申万行业分类数据...")

    try:
        from fetch_data.sw_industry_fetcher import SWIndustryFetcher

        fetcher = SWIndustryFetcher()
        if fetcher.is_stale():
            logger.info("申万行业数据已过期，开始更新...")
            count = fetcher.update_all(force=True)
            logger.info(f"✅ 申万行业分类更新完成: {count} 条")
            return count
        else:
            logger.info("✅ 申万行业分类数据有效，跳过更新")
            return 0

    except Exception as e:
        logger.warning(f"⚠️ 申万行业分类检查异常: {e}")
        return 0


def quick_daily_update(date: str = None, skip_financial: bool = True):
    """快速每日更新 - 包含市场行情、基本面、财务和技术指标"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    logger.info(f"开始完整数据更新 {date}")
    logger.info("="*60)
    start_time = time.time()

    # 统计信息
    stats = {
        'quotes': 0,
        'indices': 0,
        'basic': 0,
        'sw_industry': 0,
        'sw_index': 0,      # 申万行业指数日线
        'hsgt': 0,           # 北向资金
        'financial': 0,
        'technical': 0,
        'squeeze_momentum': 0,
        'active_mv': 0,  # V3.9.4 活跃市值特征
        'v39_cache': 0   # V3.9/V3.95 特征缓存
    }

    # 1. 批量更新市场行情（A股）
    logger.info("【步骤1/12】更新A股市场行情...")
    stats['quotes'] += batch_update_stocks(date)

    # 短暂休息避免API限制
    time.sleep(2)

    # 2. 批量更新ETF/基金
    logger.info("【步骤2/12】更新ETF/基金行情...")
    stats['quotes'] += batch_update_funds(date)

    # 3. 更新大盘指数数据
    time.sleep(2)
    logger.info("【步骤3/12】更新大盘指数数据...")
    stats['indices'] = update_market_indices(date)

    # 4. 更新基本面指标
    time.sleep(2)
    logger.info("【步骤4/12】更新基本面指标...")
    stats['basic'] = update_daily_basic(date)

    # 5. 检查申万行业分类 (月度更新)
    logger.info("【步骤5/12】检查申万行业分类 (月度更新)...")
    stats['sw_industry'] = check_sw_industry(date)

    # 6. 更新申万行业指数日线
    time.sleep(1)
    logger.info("【步骤6/12】更新申万行业指数日线...")
    stats['sw_index'] = update_sw_index_daily(date)

    # 7. 更新北向资金数据
    time.sleep(1)
    logger.info("【步骤7/12】更新北向资金数据...")
    stats['hsgt'] = update_hsgt_daily(date)

    # 8. 更新财务指标（如果有）
    if not skip_financial:
        time.sleep(2)
        logger.info("【步骤8/12】检查财务指标更新...")
        stats['financial'] = update_financial_indicators(date)
    else:
        logger.info("【步骤8/12】跳过财务指标更新...")
        stats['financial'] = 0

    # 9. 计算技术指标
    logger.info("【步骤9/12】计算技术指标...")
    stats['technical'] = calculate_technical_indicators(date)

    # 10. 更新挤压动量指标 (v3.2新增)
    time.sleep(1)
    logger.info("【步骤10/12】更新挤压动量指标...")
    stats['squeeze_momentum'] = update_squeeze_momentum_indicators(date)

    # 11. 更新V3.9.4活跃市值特征
    time.sleep(1)
    logger.info("【步骤11/12】更新V3.9.4活跃市值特征...")
    stats['active_mv'] = update_active_mv_features(date)

    # 12. 更新V3.9/V3.95特征缓存 (ML评分系统)
    time.sleep(1)
    logger.info("【步骤12/12】更新V3.9/V3.95特征缓存...")
    stats['v39_cache'] = update_v39_feature_cache(date)

    end_time = time.time()
    duration = end_time - start_time

    # 记录更新日志到数据库
    db_manager.log_data_update(
        update_type='DAILY_COMPLETE',
        securities_updated=stats['quotes'],
        records_added=sum(stats.values()),
        status='SUCCESS' if stats['quotes'] > 0 else 'FAILED',
        duration=duration
    )

    # 输出统计报告
    logger.info("="*60)
    logger.info("✅ 完整数据更新完成!")
    logger.info(f"更新日期: {date}")
    logger.info(f"市场行情: {stats['quotes']:,} 条")
    logger.info(f"大盘指数: {stats['indices']:,} 条")
    logger.info(f"基本面指标: {stats['basic']:,} 条")
    logger.info(f"申万行业分类: {stats['sw_industry']:,} 条")
    logger.info(f"申万行业指数: {stats['sw_index']:,} 条")
    logger.info(f"北向资金: {stats['hsgt']:,} 条")
    logger.info(f"财务指标: {stats['financial']:,} 条")
    logger.info(f"技术指标: {stats['technical']:,} 条")
    logger.info(f"挤压动量指标: {stats['squeeze_momentum']:,} 只股票")
    logger.info(f"V3.9.4活跃市值特征: {stats['active_mv']:,} 条")
    logger.info(f"V3.9/V3.95特征缓存: {stats['v39_cache']:,} 条")
    logger.info(f"总耗时: {duration:.1f} 秒")
    if stats['quotes'] > 0:
        logger.info(f"平均速度: {stats['quotes']/duration:.1f} 只/秒")
    logger.info("="*60)
    
    # 显示数据库统计信息
    db_stats = db_manager.get_database_stats()
    logger.info("数据库状态:")
    logger.info(f"  证券总数: {db_stats['total_securities']:,}")
    logger.info(f"  数据记录: {db_stats['total_quotes']:,}")
    logger.info(f"  数据库大小: {db_stats['db_size_mb']:.2f} MB")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="快速每日数据更新")
    parser.add_argument("--date", help="更新日期 YYYYMMDD (默认: 今天)")
    parser.add_argument("--include-financial", action="store_true", help="包含财务数据检查（默认跳过以加快速度）")
    
    args = parser.parse_args()
    
    quick_daily_update(args.date, skip_financial=not args.include_financial)