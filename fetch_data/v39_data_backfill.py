#!/usr/bin/env python3
"""
v3.9数据初始化/回填工具 (合并版)

合并自:
- v39_data_initializer.py    (证券初始化 + 按股票逐个抓取)
- v39_data_parallel_fetcher.py (线程安全并行抓取)
- v39_daily_update_enhanced.py (按日期批量API增量更新)

两种模式:
  batch: 按股票并行抓取历史数据 (适合首次初始化/大范围回填)
  daily: 按日期批量API获取当天数据 (适合每日增量更新)

用法:
  python3 fetch_data/v39_data_backfill.py --mode batch --steps 1,2,3 --start-date 20240101
  python3 fetch_data/v39_data_backfill.py --mode daily --date 20260221
  python3 fetch_data/v39_data_backfill.py --mode daily  # 默认今天
"""

import os
import sys
import json
import time
import logging
import argparse
import sqlite3
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认路径
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, 'config.json')
DEFAULT_DB = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')


class RateLimiter:
    """全局API速率限制器（线程安全）"""

    def __init__(self, calls_per_second: float = 1.0):
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.last_call_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call_time = time.time()


class V39DataBackfill:
    """v3.9数据初始化/回填工具"""

    def __init__(self, config_path: str = None, db_path: str = None, max_workers: int = 2):
        self.config_path = config_path or DEFAULT_CONFIG
        self.db_path = db_path or DEFAULT_DB
        self.max_workers = max_workers
        self.config = self._load_config()

        # 初始化Tushare (token 优先从 core.config/.env 获取)
        try:
            from core.config import get_tushare_token
            token = get_tushare_token()
        except ImportError:
            token = self.config.get('tushare', {}).get('token')
        if not token:
            raise ValueError("请在 .env 中配置 TUSHARE_TOKEN 或在config.json中配置Tushare token")

        ts.set_token(token)
        self.pro = ts.pro_api()

        # 全局速率限制器
        self.rate_limiter = RateLimiter(calls_per_second=1.0)

        # 线程安全统计
        self.stats_lock = threading.Lock()
        self.stats = {
            'securities_added': 0,
            'daily_basic_added': 0,
            'financial_indicator_added': 0,
            'success_count': 0,
            'error_count': 0,
            'errors': []
        }

        logger.info(f"初始化完成 - DB: {self.db_path}, Workers: {max_workers}")

    def _load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return {}

    def _get_security_id(self, ts_code: str) -> Optional[int]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM securities WHERE code = ?", (ts_code,))
            result = cursor.fetchone()
        finally:
            conn.close()
        return result[0] if result else None

    def _get_stock_list(self) -> List[str]:
        """从数据库读取活跃A股列表"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT code FROM securities WHERE type = 'A股' AND is_active = 1")
            stock_list = [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()
        logger.info(f"从数据库读取到 {len(stock_list)} 只股票")
        return stock_list

    # ========== Step 1: 证券初始化 (来自 initializer) ==========

    def step1_initialize_securities(self) -> List[str]:
        """获取A股列表并初始化到数据库"""
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: 获取股票列表")
        logger.info("=" * 60)

        self.rate_limiter.wait()
        df = self.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,market,list_date'
        )

        if df.empty:
            logger.error("未获取到股票列表")
            return []

        logger.info(f"获取到 {len(df)} 只A股")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            for _, row in df.iterrows():
                try:
                    ts_code = row['ts_code']
                    exchange = ts_code.split('.')[1]

                    cursor.execute("SELECT id FROM securities WHERE code = ?", (ts_code,))
                    if cursor.fetchone():
                        continue

                    cursor.execute("""
                        INSERT INTO securities (code, name, type, exchange, industry, list_date, is_active)
                        VALUES (?, ?, 'A股', ?, ?, ?, 1)
                    """, (ts_code, row['name'], exchange, row.get('industry'), row.get('list_date')))
                    self.stats['securities_added'] += 1

                except Exception as e:
                    logger.error(f"插入证券失败 ({row['ts_code']}): {e}")

            conn.commit()
        finally:
            conn.close()
        logger.info(f"新增 {self.stats['securities_added']} 只股票到数据库")
        return df['ts_code'].tolist()

    # ========== Batch模式: 并行按股票抓取 (来自 parallel_fetcher) ==========

    def _fetch_daily_basic_single(self, ts_code: str, start_date: str, end_date: str) -> Dict:
        """抓取单只股票的daily_basic数据 (check-before-insert安全模式)"""
        result = {'success': False, 'count': 0, 'error': None}

        try:
            security_id = self._get_security_id(ts_code)
            if not security_id:
                result['error'] = "Security ID not found"
                return result

            self.rate_limiter.wait()

            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,'
                       'pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,'
                       'total_share,float_share,free_share,total_mv,circ_mv'
            )

            if df.empty:
                result['success'] = True
                return result

            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()

                for _, row in df.iterrows():
                    try:
                        trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                        cursor.execute("""
                            SELECT COUNT(*) FROM daily_basic
                            WHERE security_id = ? AND trade_date = ?
                        """, (security_id, trade_date))

                        if cursor.fetchone()[0] > 0:
                            continue

                        cursor.execute("""
                            INSERT INTO daily_basic
                            (security_id, trade_date, close, turnover_rate, turnover_rate_f, volume_ratio,
                             pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                             total_share, float_share, free_share, total_mv, circ_mv)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            security_id, trade_date,
                            row.get('close'), row.get('turnover_rate'), row.get('turnover_rate_f'),
                            row.get('volume_ratio'), row.get('pe'), row.get('pe_ttm'),
                            row.get('pb'), row.get('ps'), row.get('ps_ttm'),
                            row.get('dv_ratio'), row.get('dv_ttm'),
                            row.get('total_share'), row.get('float_share'), row.get('free_share'),
                            row.get('total_mv'), row.get('circ_mv')
                        ))
                        result['count'] += 1
                    except Exception as e:
                        logger.debug(f"插入daily_basic失败 ({ts_code}, {trade_date}): {e}")

                conn.commit()
            finally:
                conn.close()
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"抓取daily_basic失败 ({ts_code}): {e}")

        return result

    def _fetch_financial_indicator_single(self, ts_code: str, start_date: str, end_date: str) -> Dict:
        """抓取单只股票的financial_indicator数据 (check-before-insert安全模式)"""
        result = {'success': False, 'count': 0, 'error': None}

        try:
            security_id = self._get_security_id(ts_code)
            if not security_id:
                result['error'] = "Security ID not found"
                return result

            self.rate_limiter.wait()

            df = self.pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,ann_date,end_date,eps,dt_eps,roe,roe_waa,roe_dt,roa,'
                       'grossprofit_margin,netprofit_margin,profit_to_gr,ocf_to_profit,'
                       'debt_to_assets,current_ratio,quick_ratio,ar_turn,ca_turn,fa_turn,assets_turn'
            )

            if df.empty:
                result['success'] = True
                return result

            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.cursor()

                for _, row in df.iterrows():
                    try:
                        ann_date = pd.to_datetime(row['ann_date'], format='%Y%m%d').strftime('%Y-%m-%d')
                        end_date_fmt = pd.to_datetime(row['end_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                        cursor.execute("""
                            SELECT COUNT(*) FROM financial_indicator
                            WHERE security_id = ? AND end_date = ?
                        """, (security_id, end_date_fmt))

                        if cursor.fetchone()[0] > 0:
                            continue

                        cursor.execute("""
                            INSERT INTO financial_indicator
                            (security_id, ann_date, end_date, eps, dt_eps, roe, roe_waa, roe_dt, roa,
                             grossprofit_margin, netprofit_margin, profit_to_gr, ocf_to_profit,
                             debt_to_assets, current_ratio, quick_ratio, ar_turn, ca_turn, fa_turn, assets_turn)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            security_id, ann_date, end_date_fmt,
                            row.get('eps'), row.get('dt_eps'), row.get('roe'),
                            row.get('roe_waa'), row.get('roe_dt'), row.get('roa'),
                            row.get('grossprofit_margin'), row.get('netprofit_margin'),
                            row.get('profit_to_gr'), row.get('ocf_to_profit'),
                            row.get('debt_to_assets'), row.get('current_ratio'),
                            row.get('quick_ratio'), row.get('ar_turn'), row.get('ca_turn'),
                            row.get('fa_turn'), row.get('assets_turn')
                        ))
                        result['count'] += 1
                    except Exception as e:
                        logger.debug(f"插入financial_indicator失败 ({ts_code}): {e}")

                conn.commit()
            finally:
                conn.close()
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"抓取financial_indicator失败 ({ts_code}): {e}")

        return result

    def _run_parallel(self, fetch_fn, stock_list: List[str], start_date: str, end_date: str, label: str):
        """通用并行抓取执行器"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"并行抓取 {label}")
        logger.info(f"  时间范围: {start_date} ~ {end_date}")
        logger.info(f"  股票数量: {len(stock_list)}, Workers: {self.max_workers}")
        logger.info("=" * 60)

        start_time = time.time()
        processed = 0
        total = len(stock_list)
        local_success = 0
        local_error = 0
        local_added = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_code = {
                executor.submit(fetch_fn, code, start_date, end_date): code
                for code in stock_list
            }

            for future in as_completed(future_to_code):
                ts_code = future_to_code[future]
                processed += 1

                try:
                    result = future.result()
                    if result['success']:
                        local_success += 1
                        local_added += result['count']
                    else:
                        local_error += 1
                except Exception as e:
                    logger.error(f"处理{ts_code}失败: {e}")
                    local_error += 1

                if processed % 100 == 0 or processed == total:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / rate if rate > 0 else 0
                    logger.info(
                        f"进度: {processed}/{total} ({processed / total * 100:.1f}%) "
                        f"| {rate:.1f}股/秒 | ETA: {eta / 60:.1f}分"
                    )

        elapsed_time = time.time() - start_time
        logger.info(f"\n{label}完成: {local_success}/{total} 成功, +{local_added}条, {elapsed_time / 60:.1f}分钟")

        with self.stats_lock:
            if 'daily_basic' in label.lower():
                self.stats['daily_basic_added'] += local_added
            else:
                self.stats['financial_indicator_added'] += local_added
            self.stats['success_count'] += local_success
            self.stats['error_count'] += local_error

    def run_batch(self, start_date: str, end_date: str, steps: List[int]):
        """Batch模式: 按股票并行抓取历史数据"""
        logger.info(f"\n[Batch模式] 时间范围: {start_date} ~ {end_date}, Steps: {steps}")
        start_time = time.time()

        stock_list = []
        if 1 in steps:
            stock_list = self.step1_initialize_securities()
            if not stock_list:
                logger.error("未能获取股票列表，终止执行")
                return

        if not stock_list:
            stock_list = self._get_stock_list()

        if 2 in steps:
            self._run_parallel(
                self._fetch_daily_basic_single, stock_list, start_date, end_date, "daily_basic"
            )

        if 3 in steps:
            self._run_parallel(
                self._fetch_financial_indicator_single, stock_list, start_date, end_date, "financial_indicator"
            )

        elapsed = time.time() - start_time
        self._print_summary(elapsed)

    # ========== Daily模式: 按日期批量API (来自 daily_update_enhanced) ==========

    def _daily_update_basic(self, date_str: str) -> int:
        """按日期批量更新daily_basic (单次API获取所有股票)"""
        logger.info(f"[Daily] 更新 {date_str} 的daily_basic数据...")

        self.rate_limiter.wait()
        df = self.pro.daily_basic(trade_date=date_str)

        if df.empty:
            logger.warning(f"未获取到 {date_str} 的daily_basic数据")
            return 0

        logger.info(f"获取到 {len(df)} 条daily_basic数据")

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute('SELECT code, id FROM securities')
            security_map = {row[0]: row[1] for row in cursor.fetchall()}

            insert_count = 0
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            for _, row in df.iterrows():
                ts_code = row['ts_code']
                if ts_code not in security_map:
                    continue

                security_id = security_map[ts_code]
                try:
                    cursor.execute("""
                        SELECT COUNT(*) FROM daily_basic
                        WHERE security_id = ? AND trade_date = ?
                    """, (security_id, formatted_date))

                    if cursor.fetchone()[0] > 0:
                        continue

                    cursor.execute("""
                        INSERT INTO daily_basic
                        (security_id, trade_date, close, turnover_rate, turnover_rate_f,
                         volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                         total_share, float_share, free_share, total_mv, circ_mv)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        security_id, formatted_date, row.get('close'),
                        row.get('turnover_rate'), row.get('turnover_rate_f'),
                        row.get('volume_ratio'), row.get('pe'), row.get('pe_ttm'),
                        row.get('pb'), row.get('ps'), row.get('ps_ttm'),
                        row.get('dv_ratio'), row.get('dv_ttm'),
                        row.get('total_share'), row.get('float_share'), row.get('free_share'),
                        row.get('total_mv'), row.get('circ_mv')
                    ))
                    insert_count += 1
                except Exception as e:
                    logger.debug(f"插入daily_basic失败 {ts_code}: {e}")

            conn.commit()
        finally:
            conn.close()
        logger.info(f"daily_basic更新完成: 新增 {insert_count}")
        return insert_count

    def _daily_update_financial(self, date_str: str, check_days: int = 3) -> int:
        """按日期检查最近发布的财务数据"""
        logger.info(f"[Daily] 检查 {date_str} 前{check_days}天内发布的财务数据...")

        end_date = datetime.strptime(date_str, '%Y%m%d')
        start_date = end_date - timedelta(days=check_days)
        start_date_str = start_date.strftime('%Y%m%d')

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute("SELECT code FROM securities WHERE type = 'A股' AND is_active = 1")
            stock_codes = [row[0] for row in cursor.fetchall()]

            cursor.execute('SELECT code, id FROM securities')
            security_map = {row[0]: row[1] for row in cursor.fetchall()}

            insert_count = 0
            error_count = 0

            batch_size = 50
            for i in range(0, len(stock_codes), batch_size):
                batch_codes = stock_codes[i:i + batch_size]

                for ts_code in batch_codes:
                    try:
                        time.sleep(0.2)

                        df = self.pro.fina_indicator(
                            ts_code=ts_code,
                            start_date=start_date_str,
                            end_date=date_str,
                            fields='ts_code,ann_date,end_date,eps,dt_eps,roe,roe_waa,roe_dt,roa,'
                                   'grossprofit_margin,netprofit_margin,profit_to_gr,ocf_to_profit,'
                                   'debt_to_assets,current_ratio,quick_ratio,ar_turn,ca_turn,fa_turn,assets_turn'
                        )

                        if df.empty:
                            continue

                        security_id = security_map.get(ts_code)
                        if not security_id:
                            continue

                        for _, row in df.iterrows():
                            try:
                                ann_date = pd.to_datetime(row['ann_date'], format='%Y%m%d').strftime('%Y-%m-%d')
                                end_date_fmt = pd.to_datetime(row['end_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                                cursor.execute("""
                                    SELECT COUNT(*) FROM financial_indicator
                                    WHERE security_id = ? AND end_date = ?
                                """, (security_id, end_date_fmt))

                                if cursor.fetchone()[0] > 0:
                                    continue

                                cursor.execute("""
                                    INSERT INTO financial_indicator
                                    (security_id, ann_date, end_date, eps, dt_eps, roe, roe_waa, roe_dt, roa,
                                     grossprofit_margin, netprofit_margin, profit_to_gr, ocf_to_profit,
                                     debt_to_assets, current_ratio, quick_ratio, ar_turn, ca_turn, fa_turn, assets_turn)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    security_id, ann_date, end_date_fmt,
                                    row.get('eps'), row.get('dt_eps'), row.get('roe'),
                                    row.get('roe_waa'), row.get('roe_dt'), row.get('roa'),
                                    row.get('grossprofit_margin'), row.get('netprofit_margin'),
                                    row.get('profit_to_gr'), row.get('ocf_to_profit'),
                                    row.get('debt_to_assets'), row.get('current_ratio'),
                                    row.get('quick_ratio'), row.get('ar_turn'), row.get('ca_turn'),
                                    row.get('fa_turn'), row.get('assets_turn')
                                ))
                                insert_count += 1
                            except Exception as e:
                                logger.debug(f"插入financial_indicator失败 {ts_code}: {e}")

                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:
                            logger.error(f"获取{ts_code}财务数据失败: {e}")

                conn.commit()
                if (i + batch_size) % 500 == 0:
                    logger.info(f"进度: {min(i + batch_size, len(stock_codes))}/{len(stock_codes)}")
        finally:
            conn.close()
        logger.info(f"financial_indicator更新完成: 新增 {insert_count}, 错误 {error_count}")
        return insert_count

    def run_daily(self, date_str: str = None, update_financial: bool = True) -> Dict:
        """Daily模式: 按日期批量更新"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')

        logger.info(f"\n[Daily模式] 日期: {date_str}")
        start_time = time.time()

        stats = {'date': date_str, 'daily_basic_added': 0, 'financial_indicator_added': 0, 'success': True}

        try:
            logger.info("\n[1/2] 更新daily_basic数据...")
            stats['daily_basic_added'] = self._daily_update_basic(date_str)

            if update_financial:
                logger.info("\n[2/2] 更新financial_indicator数据...")
                stats['financial_indicator_added'] = self._daily_update_financial(date_str)
            else:
                logger.info("\n[2/2] 跳过financial_indicator更新")

            elapsed = time.time() - start_time
            logger.info(f"\n[Daily] 完成: daily_basic +{stats['daily_basic_added']}, "
                        f"financial +{stats['financial_indicator_added']}, {elapsed:.1f}秒")

        except Exception as e:
            logger.error(f"每日数据更新失败: {e}")
            stats['success'] = False

        return stats

    # ========== 公共方法 ==========

    def _print_summary(self, elapsed: float):
        logger.info(f"\n{'=' * 60}")
        logger.info("数据抓取完成统计")
        logger.info("=" * 60)
        logger.info(f"  新增股票: {self.stats['securities_added']}")
        logger.info(f"  daily_basic: +{self.stats['daily_basic_added']}条")
        logger.info(f"  financial_indicator: +{self.stats['financial_indicator_added']}条")
        logger.info(f"  成功/失败: {self.stats['success_count']}/{self.stats['error_count']}")
        logger.info(f"  总耗时: {elapsed / 60:.1f}分钟")

        if self.stats['errors']:
            logger.info(f"\n错误详情（前10个）:")
            for error in self.stats['errors'][:10]:
                logger.info(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(
        description='v3.9数据初始化/回填工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Batch模式: 并行抓取历史数据 (首次初始化)
  python3 fetch_data/v39_data_backfill.py --mode batch --steps 1,2,3 --start-date 20240101

  # Batch模式: 只回填daily_basic (跳过证券初始化)
  python3 fetch_data/v39_data_backfill.py --mode batch --steps 2 --start-date 20250101 --workers 3

  # Daily模式: 更新今天的数据
  python3 fetch_data/v39_data_backfill.py --mode daily

  # Daily模式: 更新指定日期
  python3 fetch_data/v39_data_backfill.py --mode daily --date 20260221
        """
    )

    parser.add_argument('--mode', type=str, choices=['batch', 'daily'], default='batch',
                        help='运行模式: batch(历史回填) 或 daily(每日更新)')
    parser.add_argument('--steps', type=str, default='1,2,3',
                        help='[batch模式] 执行步骤: 1=证券初始化, 2=daily_basic, 3=financial_indicator')
    parser.add_argument('--start-date', type=str, default='20240101',
                        help='[batch模式] 开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y%m%d'),
                        help='[batch模式] 结束日期 (YYYYMMDD)')
    parser.add_argument('--workers', type=int, default=2,
                        help='[batch模式] 并行Worker数量 (建议2-3)')
    parser.add_argument('--date', type=str, default=None,
                        help='[daily模式] 交易日期 (YYYYMMDD)，默认今天')
    parser.add_argument('--no-financial', action='store_true',
                        help='[daily模式] 跳过财务指标更新')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')
    parser.add_argument('--db', type=str, default=None,
                        help='数据库路径')

    args = parser.parse_args()

    Path(os.path.join(PROJECT_ROOT, "logs")).mkdir(exist_ok=True)

    backfill = V39DataBackfill(
        config_path=args.config,
        db_path=args.db,
        max_workers=args.workers
    )

    if args.mode == 'batch':
        steps = [int(s.strip()) for s in args.steps.split(',')]
        backfill.run_batch(start_date=args.start_date, end_date=args.end_date, steps=steps)
    else:
        result = backfill.run_daily(date_str=args.date, update_financial=not args.no_financial)
        sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
