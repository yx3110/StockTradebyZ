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
            fields='ts_code,trade_date,open,close,high,low,vol,amount,pct_chg,limit'
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
                    vol = row['vol']

                    # 停牌检测: volume=0 且 pct_chg 缺失
                    is_suspended = (vol == 0 or pd.isna(vol)) and pd.isna(pct_val)

                    data_to_insert.append({
                        'security_id': security_id,
                        'trade_date': trade_date,
                        'open': row['open'],
                        'close': row['close'],
                        'high': row['high'],
                        'low': row['low'],
                        'volume': vol,
                        'amount': row.get('amount'),
                        'price_change_pct': None if is_suspended else (pct_val / 100 if pd.notna(pct_val) else 0),
                        'is_limit_up': row.get('limit') == 'U' if row.get('limit') else False,
                        'is_limit_down': row.get('limit') == 'D' if row.get('limit') else False,
                        'is_suspend': 1 if is_suspended else 0
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
        # 注意: fields 必须覆盖 daily_quotes schema 的非空列 (open/close/high/low/vol/amount)
        # 若漏字段, insert 时会写 NULL (amount 字段曾因此漏了多年, 2026-04-13 修复)
        df = pro.fund_daily(
            trade_date=date_str,
            fields='ts_code,trade_date,open,close,high,low,vol,amount,pct_chg'
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
                        'amount': row.get('amount'),
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
                            'amount': row.get('amount'),
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

        with sqlite3.connect(db_path, timeout=30) as conn:
            cursor = conn.cursor()

            security_map = _get_security_map_cached()

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
    """检查并更新当天或前一天发布财务数据的公司

    使用 ann_date 参数批量查询，仅需1-2次API调用（替代原来5000+次逐股查询）
    """
    logger.info(f"检查 {date_str} 及前一天发布财务数据的公司...")

    try:
        from datetime import datetime, timedelta
        today_dt = datetime.strptime(date_str, '%Y%m%d')
        yesterday_dt = today_dt - timedelta(days=1)
        target_dates = [date_str, yesterday_dt.strftime('%Y%m%d')]

        fields = ('ts_code,ann_date,end_date,eps,dt_eps,roe,roe_waa,roe_dt,roa,'
                  'grossprofit_margin,netprofit_margin,profit_to_gr,'
                  'ocf_to_profit,debt_to_assets,current_ratio,quick_ratio,'
                  'ar_turn,ca_turn,fa_turn,assets_turn,'
                  'netprofit_yoy,or_yoy')

        all_results = []
        for ann_date in target_dates:
            try:
                df = pro.fina_indicator(ann_date=ann_date, fields=fields)
                if df is not None and not df.empty:
                    logger.info(f"ann_date={ann_date} 查询到 {len(df)} 条财务数据")
                    all_results.append(df)
                else:
                    logger.info(f"ann_date={ann_date} 无新发布财务数据")
            except Exception as e:
                logger.warning(f"查询 ann_date={ann_date} 财务数据失败: {e}")

        if not all_results:
            logger.info("未发现当天或前一天发布财务数据的公司")
            return 0

        combined_df = pd.concat(all_results, ignore_index=True)

        # 同一股票可能在today和yesterday都有记录，按ts_code+end_date去重，保留最新ann_date
        combined_df = combined_df.sort_values('ann_date', ascending=False)
        combined_df = combined_df.drop_duplicates(subset=['ts_code', 'end_date'], keep='first')

        logger.info(f"发现 {len(combined_df)} 条新发布的财务数据，开始保存...")

        count = save_financial_data_to_db(combined_df)

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

        with sqlite3.connect(db_path, timeout=30) as conn:
            cursor = conn.cursor()

            security_map = _get_security_map_cached()

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


def update_ng_feature_cache(date_str: str):
    """更新NG特征缓存 (生产版本 + ng1.0.6 regime-switch依赖版本)"""
    logger.info(f"开始更新 {date_str} 的NG特征缓存...")

    total_count = 0
    try:
        from ml_models.ng.ng_cache_updater import NGCacheUpdater
        from ml_models.ng.ng_schema import PRODUCTION_VERSION, get_table_name

        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # 生产版本优先; ng1.0.3/ng1.0.4 供 ng1.0.6 牛熊切换回退使用
        # 多版本共表时 (如 ng1.1.0/ng1.0.1 同写 ng101_feature_cache) 仅跑一次避免重复I/O
        versions = [PRODUCTION_VERSION, 'ng1.0.3', 'ng1.0.4']
        seen_tables: set[str] = set()
        for ver in versions:
            table = get_table_name(ver)
            if table in seen_tables:
                continue
            seen_tables.add(table)
            try:
                updater = NGCacheUpdater(version=ver)
                count = updater.update_single_date(date_dash)
                if count and count > 0:
                    logger.info(f"  {ver} 缓存更新: {count} 条")
                    total_count += count
            except Exception as e:
                logger.warning(f"  {ver} 缓存更新失败: {e}")

        if total_count > 0:
            logger.info(f"✅ NG特征缓存更新成功: {total_count} 条 ({len(seen_tables)}张表)")
        else:
            logger.info("✅ NG特征缓存更新完成 (无新数据)")

        return total_count

    except Exception as e:
        logger.warning(f"⚠️ NG特征缓存更新异常: {e}")
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

    优化后步骤: 14步 (含v39/v40/NG/BRAIN/GRU缓存更新)
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
    logger.info("【步骤1/14】更新A股市场行情...")
    quotes_count = batch_update_stocks(date)
    stats['quotes'] += quotes_count

    # 非交易日检测: 如果A股行情返回0条，说明今天非交易日，跳过后续步骤
    if quotes_count == 0:
        logger.info(f"⚠️ {date} 非交易日或无数据，跳过后续更新步骤")
        elapsed = time.time() - start_time
        logger.info(f"数据更新完成 (非交易日)，耗时: {elapsed:.1f}秒")
        return stats

    # API限流等待
    time.sleep(2)

    # 2. 批量更新ETF/基金
    logger.info("【步骤2/14】更新ETF/基金行情...")
    stats['quotes'] += batch_update_funds(date)

    # API限流等待
    time.sleep(2)

    # 3. 更新大盘指数数据
    logger.info("【步骤3/14】更新大盘指数数据...")
    stats['indices'] = update_market_indices(date)

    # API限流等待
    time.sleep(2)

    # 4. 更新基本面指标
    logger.info("【步骤4/14】更新基本面指标...")
    stats['basic'] = update_daily_basic(date)

    # 5. 检查申万行业分类 (月度更新) — 本地检查，无需sleep
    logger.info("【步骤5/14】检查申万行业分类 (月度更新)...")
    stats['sw_industry'] = check_sw_industry(date)

    # 6. 更新申万行业指数日线 — API调用
    logger.info("【步骤6/14】更新申万行业指数日线...")
    stats['sw_index'] = update_sw_index_daily(date)

    # 7. 更新北向资金数据 — API调用
    logger.info("【步骤7/14】更新北向资金数据...")
    stats['hsgt'] = update_hsgt_daily(date)

    # 8. 更新财务指标（如果有）
    if not skip_financial:
        logger.info("【步骤8/14】检查财务指标更新...")
        stats['financial'] = update_financial_indicators(date)
    else:
        logger.info("【步骤8/14】跳过财务指标更新...")
        stats['financial'] = 0

    # 9. 计算技术指标 — 本地计算，无需sleep
    logger.info("【步骤9/14】计算技术指标...")
    stats['technical'] = calculate_technical_indicators(date)

    # 10. 更新V3.9/V3.95特征缓存 — 本地计算，返回stock_data_cache供v40共享
    logger.info("【步骤10/14】更新V3.9/V3.95特征缓存...")
    v39_count, stock_data_cache = update_v39_feature_cache(date)
    stats['v39_cache'] = v39_count

    # 11. 更新V4.0特征缓存 — 复用v39的股票数据缓存
    logger.info("【步骤11/14】更新V4.0 Cross-Sectional特征缓存...")
    stats['v40_cache'] = update_v40_feature_cache(date, stock_data_cache=stock_data_cache)

    # 释放缓存
    stock_data_cache = None

    # 12. 更新NG特征缓存 (Daily Selection NG因子 — 内部自动获取moneyflow)
    logger.info("【步骤12/14】更新NG特征缓存...")
    stats['ng_cache'] = update_ng_feature_cache(date)

    # 13. 更新BRAIN因子缓存 (V4.8.4 brain_roll_spread等)
    logger.info("【步骤13/14】更新BRAIN因子缓存...")
    try:
        from wqbrain_integration.cache_brain_features import batch_compute
        brain_count = batch_compute(_db_path, date, date)
        stats['brain_cache'] = brain_count
        logger.info(f"  BRAIN缓存更新完成: {brain_count} 条")
    except Exception as e:
        logger.warning(f"  BRAIN缓存更新失败: {e}")
        stats['brain_cache'] = 0

    # 14. 更新GRU神经网络嵌入缓存 (V5.0)
    logger.info("【步骤14/14】更新GRU神经网络嵌入缓存...")
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
    logger.info(f"NG特征缓存: {stats.get('ng_cache', 0):,} 条")
    logger.info(f"GRU神经网络嵌入: {stats['neural_embed']:,} 条")
    logger.info(f"总耗时: {duration:.1f} 秒")
    if stats['quotes'] > 0:
        logger.info(f"平均速度: {stats['quotes']/duration:.1f} 只/秒")
    logger.info("="*60)

    # 15. 更新0AMV市场活跃市值指标
    logger.info("【步骤15/15】更新0AMV活跃市值指标...")
    try:
        from indicators.market_amv import compute_and_save
        compute_and_save()
        logger.info("  0AMV更新完成")
    except Exception as e:
        logger.warning(f"  0AMV更新失败(非关键): {e}")

    # 16. 刷新 ST 戴帽 / 改名 (securities.name)
    logger.info("【步骤16/16】刷新股票名称 (ST namechange)...")
    try:
        from scripts.refresh_stock_names import fetch_latest_names, update_db_names
        latest = fetch_latest_names(lookback_days=540)
        changed, _ = update_db_names(latest)
        logger.info(f"  更新 {changed} 只股票名称 (共 {len(latest)} 条 namechange)")
    except Exception as e:
        logger.warning(f"  名称刷新失败(非关键): {e}")

    logger.info("="*60)

    # 显示数据库统计信息
    db_stats = db_manager.get_database_stats()
    logger.info("数据库状态:")
    logger.info(f"  证券总数: {db_stats['total_securities']:,}")
    logger.info(f"  数据记录: {db_stats['total_quotes']:,}")
    logger.info(f"  数据库大小: {db_stats['db_size_mb']:.2f} MB")

    # 关键字段 NULL 防复发 smoke-test
    # 历史上 amount 字段因 fields 漏字段多年写入 NULL, 2026-04-13 修复.
    # 每次更新后验证当天写入的关键列非 NULL 比例, 异常时 WARN.
    _daily_update_smoke_check(date)


def _daily_update_smoke_check(trade_date_str: str):
    """Verify key columns (amount, volume, close) non-NULL for today's data.

    If any column has > 5% NULL for A股, log WARN. This catches fetch-pipeline
    field drops (e.g., fields string missing 'amount') before they silently
    corrupt months of data.
    """
    import sqlite3
    td = f"{trade_date_str[:4]}-{trade_date_str[4:6]}-{trade_date_str[6:8]}" if len(trade_date_str) == 8 else trade_date_str
    try:
        with sqlite3.connect(str(db_manager.db_path), timeout=30) as conn:
            row = conn.execute("""
                SELECT COUNT(*) AS n,
                       SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS null_amount,
                       SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS null_volume,
                       SUM(CASE WHEN close  IS NULL THEN 1 ELSE 0 END) AS null_close
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE dq.trade_date = ? AND s.type IN ('A股', 'ETF_基金')
            """, (td,)).fetchone()
            if not row or row[0] == 0:
                logger.warning(f"[smoke-check] {td} 无 A股/ETF 数据写入, 跳过验证")
                return
            n, null_amt, null_vol, null_close = row
            pct_amt = null_amt / n * 100
            pct_vol = null_vol / n * 100
            pct_close = null_close / n * 100
            msg = f"[smoke-check] {td} n={n}, NULL: amount={pct_amt:.1f}%, volume={pct_vol:.1f}%, close={pct_close:.1f}%"
            if pct_amt > 5 or pct_vol > 5 or pct_close > 5:
                logger.warning(f"⚠️  {msg} — 有字段 NULL 比例>5%, 检查 fetch fields 字符串是否漏字段")
            else:
                logger.info(f"✅ {msg}")
    except Exception as e:
        logger.warning(f"[smoke-check] 失败(非关键): {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="快速每日数据更新")
    parser.add_argument("--date", help="更新日期 YYYYMMDD (默认: 今天)")
    parser.add_argument("--include-financial", action="store_true", help="包含财务数据检查（默认跳过以加快速度）")

    args = parser.parse_args()

    quick_daily_update(args.date, skip_financial=not args.include_financial)
