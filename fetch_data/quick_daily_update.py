#!/usr/bin/env python3
"""
快速每日数据更新 - 使用批量API调用并写入数据库

性能优化版:
- 移除不必要的 sleep (本地计算步骤间无需等待)
- 批量 security_map 替代 N+1 insert_security
- 批量指数更新 (收集后一次 insert_daily_quotes)
- executemany 替代逐行插入 (daily_basic)
- 技术指标进程内调用 + 批量 commit
- v39/v40 共享股票数据缓存
- 移除已失效的挤压动量和活跃市值步骤
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

try:
    from core.config import PROJECT_ROOT, get_tushare_token, get_db_path, MARKET_INDICES, load_config
    _config_path = str(PROJECT_ROOT / 'config.json')
    _ts_token = get_tushare_token()
    _db_path = str(get_db_path())
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent
    _config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    with open(_config_path, 'r') as _f:
        _cfg = json.load(_f)
        _ts_token = _cfg['tushare']['token']
    _db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_adapter', 'stock_data.db')
    MARKET_INDICES = {
        '000001.SH': '上证指数', '399001.SZ': '深证成指', '399006.SZ': '创业板指',
        '000688.SH': '科创50', '000016.SH': '上证50', '000300.SH': '沪深300',
        '000905.SH': '中证500', '000852.SH': '中证1000', '932000.CSI': '中证2000',
        '000985.SH': '中证全指',
    }

from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

ts.set_token(_ts_token)
pro = ts.pro_api()

# 初始化数据库管理器
db_manager = DatabaseManager()


def _get_security_map_cached():
    """获取并缓存 security_map (一次查询)"""
    if not hasattr(_get_security_map_cached, '_cache'):
        _get_security_map_cached._cache = db_manager.get_security_map()
    return _get_security_map_cached._cache


def _refresh_security_map():
    """刷新 security_map 缓存"""
    _get_security_map_cached._cache = db_manager.get_security_map()
    return _get_security_map_cached._cache


def batch_update_stocks(date_str: str, batch_size: int = 500):
    """批量更新A股数据到数据库 (优化: 批量security_map)"""
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

        # 一次加载 security_map
        security_map = _get_security_map_cached()

        # 找出新股票，批量插入
        new_securities = []
        for _, row in df.iterrows():
            code = row['ts_code'].split('.')[0]
            if code not in security_map:
                exchange = row['ts_code'].split('.')[1]
                new_securities.append({
                    'code': code, 'name': code,
                    'type': 'A股', 'exchange': exchange
                })

        if new_securities:
            db_manager.batch_insert_securities(new_securities)
            security_map = _refresh_security_map()
            logger.info(f"新增 {len(new_securities)} 只A股证券")

        # 准备批量插入数据
        data_to_insert = []
        success_count = 0

        for _, row in df.iterrows():
            try:
                code = row['ts_code'].split('.')[0]
                security_id = security_map.get(code)

                if security_id:
                    trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')
                    pct_val = row.get('pct_chg')
                    data_to_insert.append({
                        'security_id': security_id,
                        'trade_date': trade_date,
                        'open': row['open'],
                        'close': row['close'],
                        'high': row['high'],
                        'low': row['low'],
                        'volume': row['vol'],
                        'price_change_pct': pct_val / 100 if pd.notna(pct_val) else 0,
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
    """批量更新ETF/基金数据到数据库 (优化: 批量security_map)"""
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

        # 一次加载 security_map
        security_map = _get_security_map_cached()

        # 找出新基金，批量插入
        new_securities = []
        for _, row in df.iterrows():
            code = row['ts_code'].split('.')[0]
            if code not in security_map:
                exchange = row['ts_code'].split('.')[1]
                new_securities.append({
                    'code': code, 'name': code,
                    'type': 'ETF_基金', 'exchange': exchange
                })

        if new_securities:
            db_manager.batch_insert_securities(new_securities)
            security_map = _refresh_security_map()
            logger.info(f"新增 {len(new_securities)} 只基金证券")

        # 准备批量插入数据
        data_to_insert = []
        success_count = 0

        for _, row in df.iterrows():
            try:
                code = row['ts_code'].split('.')[0]
                security_id = security_map.get(code)

                if security_id:
                    trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')
                    pct_val = row.get('pct_chg')
                    data_to_insert.append({
                        'security_id': security_id,
                        'trade_date': trade_date,
                        'open': row['open'],
                        'close': row['close'],
                        'high': row['high'],
                        'low': row['low'],
                        'volume': row['vol'],
                        'price_change_pct': pct_val / 100 if pd.notna(pct_val) else 0,
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
    """更新大盘指数数据 (优化: 去掉sleep, 批量插入)"""
    logger.info(f"开始更新 {date_str} 的大盘指数数据...")

    try:
        security_map = _get_security_map_cached()

        # 确保指数证券已注册
        new_securities = []
        for ts_code, name in MARKET_INDICES.items():
            if ts_code not in security_map:
                exchange = ts_code.split('.')[1] if '.' in ts_code else 'CSI'
                new_securities.append({
                    'code': ts_code, 'name': name,
                    'type': '指数', 'exchange': exchange
                })
        if new_securities:
            db_manager.batch_insert_securities(new_securities)
            security_map = _refresh_security_map()

        # 按交易所分组批量获取
        all_data = []
        success_count = 0

        for ts_code, name in MARKET_INDICES.items():
            try:
                df = pro.index_daily(
                    ts_code=ts_code,
                    trade_date=date_str,
                    fields='ts_code,trade_date,open,close,high,low,vol,amount,pct_chg'
                )

                if not df.empty:
                    row = df.iloc[0]
                    security_id = security_map.get(ts_code)

                    if security_id:
                        trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')
                        pct_val = row.get('pct_chg')
                        all_data.append({
                            'security_id': security_id,
                            'trade_date': trade_date,
                            'open': row['open'],
                            'close': row['close'],
                            'high': row['high'],
                            'low': row['low'],
                            'volume': row.get('vol', 0),
                            'price_change_pct': pct_val / 100 if pd.notna(pct_val) else 0,
                            'is_limit_up': False,
                            'is_limit_down': False
                        })
                        success_count += 1
                        logger.info(f"更新{name}成功: {row['close']:.2f} ({row.get('pct_chg', 0):+.2f}%)")

            except Exception as e:
                logger.warning(f"更新{name}({ts_code})失败: {e}")
                continue

        # 一次批量插入所有指数数据
        if all_data:
            db_manager.insert_daily_quotes(all_data)

        logger.info(f"大盘指数更新完成: {success_count} 个")
        return success_count

    except Exception as e:
        logger.error(f"更新大盘指数失败: {e}")
        return 0

def update_daily_basic(date_str: str):
    """更新每日基本面指标 (优化: executemany批量插入)"""
    logger.info(f"开始更新 {date_str} 的基本面指标...")

    try:
        df = pro.daily_basic(trade_date=date_str)

        if df.empty:
            logger.warning(f"未获取到 {date_str} 的基本面数据")
            return 0

        logger.info(f"获取到 {len(df)} 条基本面指标")

        import sqlite3
        db_path = _db_path

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # 获取证券ID映射
            cursor.execute('SELECT code, id FROM securities')
            security_map = {row[0]: row[1] for row in cursor.fetchall()}

            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            # 构建批量数据
            batch_data = []
            for _, row in df.iterrows():
                code = row['ts_code'][:6]
                if code in security_map:
                    batch_data.append((
                        security_map[code], formatted_date, row.get('close'),
                        row.get('turnover_rate'), row.get('turnover_rate_f'),
                        row.get('volume_ratio'), row.get('pe'), row.get('pe_ttm'),
                        row.get('pb'), row.get('ps'), row.get('ps_ttm'),
                        row.get('dv_ratio'), row.get('dv_ttm'),
                        row.get('total_share'), row.get('float_share'), row.get('free_share'),
                        row.get('total_mv'), row.get('circ_mv')
                    ))

            # executemany 一次插入
            if batch_data:
                cursor.executemany('''
                    INSERT OR REPLACE INTO daily_basic
                    (security_id, trade_date, close, turnover_rate, turnover_rate_f,
                     volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                     total_share, float_share, free_share, total_mv, circ_mv)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch_data)
                conn.commit()

        count = len(batch_data)
        logger.info(f"✅ 基本面指标更新完成: {count} 条")
        return count

    except Exception as e:
        logger.error(f"更新基本面指标失败: {e}")
        return 0

def update_financial_indicators(date_str: str):
    """检查并更新当天或前一天发布财务数据的公司"""
    logger.info(f"检查 {date_str} 及前一天发布财务数据的公司...")

    try:
        from datetime import datetime, timedelta
        today_dt = datetime.strptime(date_str, '%Y%m%d')
        yesterday_dt = today_dt - timedelta(days=1)
        target_dates = [date_str, yesterday_dt.strftime('%Y%m%d')]

        import sqlite3
        db_path = _db_path
        with sqlite3.connect(db_path) as conn:
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

        if not stock_codes:
            logger.info("未找到活跃股票，跳过财务指标更新")
            return 0

        logger.info(f"并行检查 {len(stock_codes)} 只A股的财务数据发布情况")

        import concurrent.futures
        import threading

        companies_with_new_reports = []
        checked_count = 0
        lock = threading.Lock()

        def check_single_stock(ts_code):
            try:
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

                    if ann_date in target_dates:
                        with lock:
                            companies_with_new_reports.append({
                                'ts_code': ts_code,
                                'ann_date': ann_date,
                                'end_date': latest['end_date'],
                                'data': latest
                            })
                        logger.info(f"发现 {ts_code} 在 {ann_date} 发布财务数据")

                nonlocal checked_count
                with lock:
                    checked_count += 1
                    if checked_count % 100 == 0:
                        logger.info(f"已检查 {checked_count}/{len(stock_codes)} 只股票")

                return True

            except Exception as e:
                logger.debug(f"检查 {ts_code} 财务数据失败: {e}")
                return False

        max_workers = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(check_single_stock, ts_code) for ts_code in stock_codes]
            concurrent.futures.wait(futures)

        if not companies_with_new_reports:
            logger.info("未发现当天或前一天发布财务数据的公司")
            return 0

        logger.info(f"发现 {len(companies_with_new_reports)} 家公司发布了新的财务数据")

        count = 0
        for company in companies_with_new_reports:
            try:
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
        db_path = _db_path

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT code, id FROM securities')
            security_map = {row[0]: row[1] for row in cursor.fetchall()}

            count = 0
            for _, row in df.iterrows():
                code = row['ts_code'][:6]
                if code in security_map:
                    try:
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
        return count

    except Exception as e:
        logger.error(f"保存财务数据失败: {e}")
        return 0

def calculate_technical_indicators(date_str: str):
    """计算技术指标 (优化: 进程内调用 + 批量commit)"""
    logger.info(f"开始计算 {date_str} 的技术指标...")

    try:
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        from fetch_data.technical_indicator_calculator import TechnicalIndicatorCalculator
        calculator = TechnicalIndicatorCalculator(
            db_path=_db_path,
            start_date=date_dash,
            end_date=date_dash
        )
        count = calculator.calculate_all_indicators_batch()
        logger.info(f"✅ 技术指标计算成功: {count} 条")
        return count

    except Exception as e:
        logger.error(f"❌ 技术指标计算异常: {e}")
        return 0


def update_v39_feature_cache(date_str: str):
    """更新V3.9/V3.95特征缓存，返回 (count, stock_data_map) 用于v40共享"""
    logger.info(f"开始更新 {date_str} 的V3.9特征缓存 (优化版)...")

    try:
        from fetch_data.v39_feature_cache_updater import V39FeatureCacheUpdaterOptimized

        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        updater = V39FeatureCacheUpdaterOptimized()
        count = updater.update_single_date(date_dash)

        # 获取已加载的股票数据用于v40共享
        stock_data_cache = getattr(updater, '_batch_stock_data', None)

        if count > 0:
            logger.info(f"✅ V3.9特征缓存更新成功: {count} 条")
        else:
            logger.info("✅ V3.9特征缓存更新完成 (无新数据)")

        return count, stock_data_cache

    except Exception as e:
        logger.warning(f"⚠️ V3.9特征缓存更新异常: {e}")
        return 0, None


def update_v40_feature_cache(date_str: str, stock_data_cache: dict = None):
    """更新V4.0 Cross-Sectional特征缓存 (支持复用股票数据缓存)"""
    logger.info(f"开始更新 {date_str} 的V4.0特征缓存...")

    try:
        from fetch_data.v40_feature_cache_updater import update_v40_feature_cache as _update_v40

        count = _update_v40(date_str, stock_data_cache=stock_data_cache)

        if count and count > 0:
            logger.info(f"✅ V4.0特征缓存更新成功: {count} 条")
        else:
            logger.info("✅ V4.0特征缓存更新完成 (无新数据)")

        return count or 0

    except Exception as e:
        logger.warning(f"⚠️ V4.0特征缓存更新异常: {e}")
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


def update_neural_embeddings(date_str: str):
    """更新GRU神经网络嵌入缓存 (V5.0用)"""
    logger.info(f"开始更新 {date_str} 的GRU嵌入缓存...")

    try:
        from ml_models.neural.backfill_neural_embeddings import daily_update_embeddings

        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        count = daily_update_embeddings(date_dash) or 0

        if count > 0:
            logger.info(f"✅ GRU嵌入缓存更新成功: {count} 条")
        else:
            logger.info("✅ GRU嵌入缓存更新完成 (无模型或无新数据)")

        return count

    except ImportError:
        logger.info("⏭️ GRU嵌入模块未安装 (torch未安装)，跳过")
        return 0
    except FileNotFoundError:
        logger.info("⏭️ GRU模型未训练，跳过嵌入更新")
        return 0
    except Exception as e:
        logger.warning(f"⚠️ GRU嵌入更新异常: {e}")
        return 0


def quick_daily_update(date: str = None, skip_financial: bool = True):
    """快速每日更新 - 包含市场行情、基本面、财务和技术指标

    优化后步骤: 12步 (移除了已失效的挤压动量和活跃市值步骤)
    - API调用之间保留 sleep(2) 避免限流
    - 本地计算步骤之间不再 sleep
    """
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
        'sw_index': 0,
        'hsgt': 0,
        'financial': 0,
        'technical': 0,
        'v39_cache': 0,
        'v40_cache': 0,
        'neural_embed': 0
    }

    # 1. 批量更新市场行情（A股）
    logger.info("【步骤1/12】更新A股市场行情...")
    stats['quotes'] += batch_update_stocks(date)

    # API限流等待
    time.sleep(2)

    # 2. 批量更新ETF/基金
    logger.info("【步骤2/12】更新ETF/基金行情...")
    stats['quotes'] += batch_update_funds(date)

    # API限流等待
    time.sleep(2)

    # 3. 更新大盘指数数据
    logger.info("【步骤3/12】更新大盘指数数据...")
    stats['indices'] = update_market_indices(date)

    # API限流等待
    time.sleep(2)

    # 4. 更新基本面指标
    logger.info("【步骤4/12】更新基本面指标...")
    stats['basic'] = update_daily_basic(date)

    # 5. 检查申万行业分类 (月度更新) — 本地检查，无需sleep
    logger.info("【步骤5/12】检查申万行业分类 (月度更新)...")
    stats['sw_industry'] = check_sw_industry(date)

    # 6. 更新申万行业指数日线 — API调用
    logger.info("【步骤6/12】更新申万行业指数日线...")
    stats['sw_index'] = update_sw_index_daily(date)

    # 7. 更新北向资金数据 — API调用
    logger.info("【步骤7/12】更新北向资金数据...")
    stats['hsgt'] = update_hsgt_daily(date)

    # 8. 更新财务指标（如果有）
    if not skip_financial:
        logger.info("【步骤8/12】检查财务指标更新...")
        stats['financial'] = update_financial_indicators(date)
    else:
        logger.info("【步骤8/12】跳过财务指标更新...")
        stats['financial'] = 0

    # 9. 计算技术指标 — 本地计算，无需sleep
    logger.info("【步骤9/12】计算技术指标...")
    stats['technical'] = calculate_technical_indicators(date)

    # 10. 更新V3.9/V3.95特征缓存 — 本地计算，返回stock_data_cache供v40共享
    logger.info("【步骤10/12】更新V3.9/V3.95特征缓存...")
    v39_count, stock_data_cache = update_v39_feature_cache(date)
    stats['v39_cache'] = v39_count

    # 11. 更新V4.0特征缓存 — 复用v39的股票数据缓存
    logger.info("【步骤11/12】更新V4.0 Cross-Sectional特征缓存...")
    stats['v40_cache'] = update_v40_feature_cache(date, stock_data_cache=stock_data_cache)

    # 释放缓存
    stock_data_cache = None

    # 12. 更新GRU神经网络嵌入缓存 (V5.0)
    logger.info("【步骤12/12】更新GRU神经网络嵌入缓存...")
    stats['neural_embed'] = update_neural_embeddings(date)

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
    logger.info(f"V3.9/V3.95特征缓存: {stats['v39_cache']:,} 条")
    logger.info(f"V4.0特征缓存: {stats['v40_cache']:,} 条")
    logger.info(f"GRU神经网络嵌入: {stats['neural_embed']:,} 条")
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
