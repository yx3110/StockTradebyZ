#!/usr/bin/env python3
"""
v3.9并行数据抓取器

支持多线程并行抓取，同时严格控制API速率：
- 默认2个worker (Tushare普通用户: 120次/分钟)
- 全局速率限制: 1次API调用/秒 (安全值)
- 预计速度提升: 1.5-2倍
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
import queue

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/v39_parallel_fetch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """全局API速率限制器（线程安全）"""

    def __init__(self, calls_per_second: float = 1.0):
        """
        Args:
            calls_per_second: 每秒允许的API调用次数
                             Tushare普通: 120次/分钟 = 2次/秒
                             保守设置: 1次/秒 (确保安全)
        """
        self.min_interval = 1.0 / calls_per_second
        self.last_call_time = 0
        self.lock = threading.Lock()

    def wait(self):
        """等待直到可以进行下一次API调用"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time

            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)

            self.last_call_time = time.time()


class V39ParallelFetcher:
    """v3.9并行数据抓取器"""

    def __init__(self, config_path: str = "config.json", db_path: str = "data_adapter/stock_data.db", max_workers: int = 2):
        """
        Args:
            max_workers: 并行worker数量 (建议2-3个，避免API限流)
        """
        self.db_path = db_path
        self.config = self._load_config(config_path)
        self.max_workers = max_workers

        # 初始化Tushare
        token = self.config.get('tushare', {}).get('token')
        if not token:
            raise ValueError("❌ 请在config.json中配置Tushare token")

        ts.set_token(token)
        self.pro = ts.pro_api()

        # 全局速率限制器（1次/秒，确保安全）
        self.rate_limiter = RateLimiter(calls_per_second=1.0)

        # 统计信息（线程安全）
        self.stats_lock = threading.Lock()
        self.stats = {
            'daily_basic_added': 0,
            'financial_indicator_added': 0,
            'success_count': 0,
            'error_count': 0,
            'errors': []
        }

        logger.info(f"✅ 初始化完成 - 并行Worker数: {max_workers}")
        logger.info(f"   API速率限制: 1次/秒 (安全模式)")

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            return {}

    def _get_security_id(self, ts_code: str) -> Optional[int]:
        """获取证券ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM securities WHERE code = ?", (ts_code,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"❌ 获取证券ID失败 ({ts_code}): {e}")
            return None

    def fetch_daily_basic_single(self, ts_code: str, start_date: str, end_date: str) -> Dict:
        """抓取单只股票的daily_basic数据"""
        result = {'success': False, 'count': 0, 'error': None}

        try:
            # 获取security_id
            security_id = self._get_security_id(ts_code)
            if not security_id:
                result['error'] = f"Security ID not found"
                return result

            # 全局速率限制
            self.rate_limiter.wait()

            # API调用
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

            # 插入数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for _, row in df.iterrows():
                try:
                    trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                    # 安全写入：先检查是否存在，避免覆盖已有数据
                    cursor.execute("""
                        SELECT COUNT(*) FROM daily_basic
                        WHERE security_id = ? AND trade_date = ?
                    """, (security_id, trade_date))

                    if cursor.fetchone()[0] > 0:
                        # 数据已存在，跳过
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
            conn.close()
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ 抓取daily_basic失败 ({ts_code}): {e}")

        return result

    def fetch_financial_indicator_single(self, ts_code: str, start_date: str, end_date: str) -> Dict:
        """抓取单只股票的financial_indicator数据"""
        result = {'success': False, 'count': 0, 'error': None}

        try:
            # 获取security_id
            security_id = self._get_security_id(ts_code)
            if not security_id:
                result['error'] = f"Security ID not found"
                return result

            # 全局速率限制
            self.rate_limiter.wait()

            # API调用
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

            # 插入数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for _, row in df.iterrows():
                try:
                    ann_date = pd.to_datetime(row['ann_date'], format='%Y%m%d').strftime('%Y-%m-%d')
                    end_date_fmt = pd.to_datetime(row['end_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                    # 安全写入：先检查是否存在，避免覆盖已有数据
                    cursor.execute("""
                        SELECT COUNT(*) FROM financial_indicator
                        WHERE security_id = ? AND end_date = ?
                    """, (security_id, end_date_fmt))

                    if cursor.fetchone()[0] > 0:
                        # 数据已存在，跳过
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
                    logger.debug(f"插入financial_indicator失败 ({ts_code}, {end_date_fmt}): {e}")

            conn.commit()
            conn.close()
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ 抓取financial_indicator失败 ({ts_code}): {e}")

        return result

    def fetch_daily_basic_parallel(self, stock_list: List[str], start_date: str, end_date: str):
        """并行抓取daily_basic数据"""
        logger.info("\n" + "="*60)
        logger.info("📊 并行抓取daily_basic数据")
        logger.info(f"   时间范围: {start_date} ~ {end_date}")
        logger.info(f"   股票数量: {len(stock_list)}")
        logger.info(f"   并行Worker: {self.max_workers}")
        logger.info("="*60)

        start_time = time.time()
        processed_count = 0
        total_stocks = len(stock_list)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.fetch_daily_basic_single, code, start_date, end_date): code
                for code in stock_list
            }

            # 处理完成的任务
            for future in as_completed(future_to_code):
                ts_code = future_to_code[future]
                processed_count += 1

                try:
                    result = future.result()

                    with self.stats_lock:
                        if result['success']:
                            self.stats['success_count'] += 1
                            self.stats['daily_basic_added'] += result['count']
                        else:
                            self.stats['error_count'] += 1
                            if result['error']:
                                self.stats['errors'].append(f"daily_basic {ts_code}: {result['error']}")

                    # 进度显示
                    if processed_count % 100 == 0 or processed_count == total_stocks:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta = (total_stocks - processed_count) / rate if rate > 0 else 0

                        logger.info(
                            f"进度: {processed_count}/{total_stocks} ({processed_count/total_stocks*100:.1f}%) "
                            f"| 速率: {rate:.1f}股/秒 | ETA: {eta/60:.1f}分钟"
                        )

                except Exception as e:
                    logger.error(f"❌ 处理{ts_code}失败: {e}")
                    with self.stats_lock:
                        self.stats['error_count'] += 1

        elapsed_time = time.time() - start_time
        logger.info(f"\n✅ daily_basic抓取完成:")
        logger.info(f"   耗时: {elapsed_time/60:.1f}分钟")
        logger.info(f"   成功: {self.stats['success_count']}/{total_stocks}")
        logger.info(f"   失败: {self.stats['error_count']}")
        logger.info(f"   数据条数: {self.stats['daily_basic_added']}")

    def fetch_financial_indicator_parallel(self, stock_list: List[str], start_date: str, end_date: str):
        """并行抓取financial_indicator数据"""
        logger.info("\n" + "="*60)
        logger.info("💰 并行抓取financial_indicator数据")
        logger.info(f"   时间范围: {start_date} ~ {end_date}")
        logger.info(f"   股票数量: {len(stock_list)}")
        logger.info(f"   并行Worker: {self.max_workers}")
        logger.info("="*60)

        start_time = time.time()
        processed_count = 0
        total_stocks = len(stock_list)
        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_code = {
                executor.submit(self.fetch_financial_indicator_single, code, start_date, end_date): code
                for code in stock_list
            }

            # 处理完成的任务
            for future in as_completed(future_to_code):
                ts_code = future_to_code[future]
                processed_count += 1

                try:
                    result = future.result()

                    with self.stats_lock:
                        if result['success']:
                            success_count += 1
                            self.stats['financial_indicator_added'] += result['count']
                        else:
                            error_count += 1
                            if result['error']:
                                self.stats['errors'].append(f"financial_indicator {ts_code}: {result['error']}")

                    # 进度显示
                    if processed_count % 100 == 0 or processed_count == total_stocks:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta = (total_stocks - processed_count) / rate if rate > 0 else 0

                        logger.info(
                            f"进度: {processed_count}/{total_stocks} ({processed_count/total_stocks*100:.1f}%) "
                            f"| 速率: {rate:.1f}股/秒 | ETA: {eta/60:.1f}分钟"
                        )

                except Exception as e:
                    logger.error(f"❌ 处理{ts_code}失败: {e}")
                    error_count += 1

        elapsed_time = time.time() - start_time
        logger.info(f"\n✅ financial_indicator抓取完成:")
        logger.info(f"   耗时: {elapsed_time/60:.1f}分钟")
        logger.info(f"   成功: {success_count}/{total_stocks}")
        logger.info(f"   失败: {error_count}")
        logger.info(f"   数据条数: {self.stats['financial_indicator_added']}")

    def print_summary(self):
        """打印统计摘要"""
        logger.info("\n" + "="*60)
        logger.info("📊 数据抓取完成统计")
        logger.info("="*60)
        logger.info(f"✅ daily_basic数据: {self.stats['daily_basic_added']}条")
        logger.info(f"✅ financial_indicator数据: {self.stats['financial_indicator_added']}条")
        logger.info(f"❌ 错误数: {len(self.stats['errors'])}")

        if self.stats['errors']:
            logger.info("\n错误详情（前10个）:")
            for error in self.stats['errors'][:10]:
                logger.info(f"  - {error}")

    def run(self, start_date: str, end_date: str):
        """运行并行数据抓取"""
        logger.info("\n" + "🚀 "*30)
        logger.info("v3.9并行数据抓取开始")
        logger.info(f"时间范围: {start_date} ~ {end_date}")
        logger.info(f"并行Worker数: {self.max_workers}")
        logger.info("🚀 "*30 + "\n")

        start_time = time.time()

        # 从数据库读取股票列表
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT code FROM securities WHERE type = 'A股' AND is_active = 1")
        stock_list = [row[0] for row in cursor.fetchall()]
        conn.close()
        logger.info(f"✅ 从数据库读取到 {len(stock_list)} 只股票\n")

        # Step 2: 并行抓取daily_basic
        self.fetch_daily_basic_parallel(stock_list, start_date, end_date)

        # Step 3: 并行抓取financial_indicator
        self.fetch_financial_indicator_parallel(stock_list, start_date, end_date)

        elapsed_time = time.time() - start_time

        # 打印统计摘要
        self.print_summary()

        logger.info(f"\n⏱️  总耗时: {elapsed_time/60:.1f}分钟")
        logger.info("✅ v3.9并行数据抓取完成！\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='v3.9并行数据抓取脚本')
    parser.add_argument('--start-date', type=str, default='20240101',
                       help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y%m%d'),
                       help='结束日期 (YYYYMMDD)')
    parser.add_argument('--workers', type=int, default=2,
                       help='并行Worker数量 (建议2-3，默认2)')
    parser.add_argument('--db', type=str, default='data_adapter/stock_data.db',
                       help='数据库路径')

    args = parser.parse_args()

    # 创建logs目录
    Path("logs").mkdir(exist_ok=True)

    # 运行并行抓取
    fetcher = V39ParallelFetcher(db_path=args.db, max_workers=args.workers)
    fetcher.run(start_date=args.start_date, end_date=args.end_date)


if __name__ == "__main__":
    main()
