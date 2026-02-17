#!/usr/bin/env python3
"""
v3.9数据初始化脚本

填充数据库以支持v3.9的42个新特征：
- Step 1: 获取股票列表
- Step 2: 批量抓取daily_basic数据（PE/PB/PS/换手率）
- Step 3: 批量抓取financial_indicator数据（财务指标）
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

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/v39_data_init.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class V39DataInitializer:
    """v3.9数据初始化器"""

    def __init__(self, config_path: str = "config.json", db_path: str = "data_adapter/stock_data.db"):
        """初始化"""
        self.db_path = db_path
        self.config = self._load_config(config_path)

        # 初始化Tushare
        token = self.config.get('tushare', {}).get('token')
        if not token:
            raise ValueError("❌ 请在config.json中配置Tushare token")

        ts.set_token(token)
        self.pro = ts.pro_api()
        logger.info("✅ Tushare API初始化成功")

        # API调用限制
        self.api_delay = 0.5  # 每次调用间隔0.5秒 (500次/分钟限制)
        self.last_call_time = 0

        # 统计信息
        self.stats = {
            'securities_added': 0,
            'daily_basic_added': 0,
            'financial_indicator_added': 0,
            'errors': []
        }

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            return {}

    def _rate_limit(self):
        """API调用速率限制"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.api_delay:
            sleep_time = self.api_delay - time_since_last_call
            time.sleep(sleep_time)

        self.last_call_time = time.time()

    def _get_or_create_security_id(self, ts_code: str, name: str = None) -> Optional[int]:
        """获取或创建证券ID"""
        try:
            code = ts_code.split('.')[0]
            exchange = ts_code.split('.')[1]

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 检查是否已存在
            cursor.execute("SELECT id FROM securities WHERE code = ?", (ts_code,))
            result = cursor.fetchone()

            if result:
                security_id = result[0]
            else:
                # 插入新证券
                cursor.execute("""
                    INSERT INTO securities (code, name, type, exchange, is_active)
                    VALUES (?, ?, 'A股', ?, 1)
                """, (ts_code, name or code, exchange))
                security_id = cursor.lastrowid
                self.stats['securities_added'] += 1

            conn.commit()
            conn.close()
            return security_id

        except Exception as e:
            logger.error(f"❌ 获取证券ID失败 ({ts_code}): {e}")
            return None

    def step1_initialize_securities(self) -> List[str]:
        """Step 1: 获取并初始化股票列表"""
        logger.info("\n" + "="*60)
        logger.info("📋 Step 1: 获取股票列表")
        logger.info("="*60)

        try:
            self._rate_limit()

            # 获取A股列表
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )

            if df.empty:
                logger.error("❌ 未获取到股票列表")
                return []

            logger.info(f"✅ 获取到 {len(df)} 只A股")

            # 插入到数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for _, row in df.iterrows():
                try:
                    ts_code = row['ts_code']
                    code = ts_code.split('.')[0]
                    exchange = ts_code.split('.')[1]

                    # 检查是否已存在
                    cursor.execute("SELECT id FROM securities WHERE code = ?", (ts_code,))
                    if cursor.fetchone():
                        continue

                    # 插入新证券
                    cursor.execute("""
                        INSERT INTO securities (code, name, type, exchange, industry, list_date, is_active)
                        VALUES (?, ?, 'A股', ?, ?, ?, 1)
                    """, (
                        ts_code,
                        row['name'],
                        exchange,
                        row.get('industry'),
                        row.get('list_date')
                    ))

                    self.stats['securities_added'] += 1

                except Exception as e:
                    logger.error(f"❌ 插入证券失败 ({row['ts_code']}): {e}")
                    self.stats['errors'].append(f"Insert security {row['ts_code']}: {e}")

            conn.commit()
            conn.close()

            logger.info(f"✅ 新增 {self.stats['securities_added']} 只股票到数据库")

            stock_list = df['ts_code'].tolist()
            return stock_list

        except Exception as e:
            logger.error(f"❌ Step 1失败: {e}")
            self.stats['errors'].append(f"Step 1: {e}")
            return []

    def step2_fetch_daily_basic(self, stock_list: List[str], start_date: str, end_date: str):
        """Step 2: 批量抓取daily_basic数据"""
        logger.info("\n" + "="*60)
        logger.info("📊 Step 2: 抓取每日指标数据 (daily_basic)")
        logger.info(f"    时间范围: {start_date} ~ {end_date}")
        logger.info(f"    股票数量: {len(stock_list)}")
        logger.info("="*60)

        total_stocks = len(stock_list)
        success_count = 0
        error_count = 0

        for idx, ts_code in enumerate(stock_list, 1):
            try:
                # 进度显示
                if idx % 50 == 0 or idx == total_stocks:
                    logger.info(f"进度: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%)")

                # 获取security_id
                security_id = self._get_or_create_security_id(ts_code)
                if not security_id:
                    error_count += 1
                    continue

                # API调用速率限制
                self._rate_limit()

                # 抓取daily_basic数据
                df = self.pro.daily_basic(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,'
                           'pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,'
                           'total_share,float_share,free_share,total_mv,circ_mv'
                )

                if df.empty:
                    continue

                # 插入数据库
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for _, row in df.iterrows():
                    try:
                        trade_date = pd.to_datetime(row['trade_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                        cursor.execute("""
                            INSERT OR REPLACE INTO daily_basic
                            (security_id, trade_date, close, turnover_rate, turnover_rate_f, volume_ratio,
                             pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm,
                             total_share, float_share, free_share, total_mv, circ_mv)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
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

                        self.stats['daily_basic_added'] += 1

                    except Exception as e:
                        logger.debug(f"插入daily_basic失败 ({ts_code}, {trade_date}): {e}")

                conn.commit()
                conn.close()
                success_count += 1

            except Exception as e:
                logger.error(f"❌ 处理{ts_code}失败: {e}")
                error_count += 1
                self.stats['errors'].append(f"daily_basic {ts_code}: {e}")

        logger.info(f"\n✅ Step 2完成:")
        logger.info(f"   成功: {success_count}/{total_stocks}")
        logger.info(f"   失败: {error_count}")
        logger.info(f"   数据条数: {self.stats['daily_basic_added']}")

    def step3_fetch_financial_indicator(self, stock_list: List[str], start_date: str, end_date: str):
        """Step 3: 批量抓取financial_indicator数据"""
        logger.info("\n" + "="*60)
        logger.info("💰 Step 3: 抓取财务指标数据 (financial_indicator)")
        logger.info(f"    时间范围: {start_date} ~ {end_date}")
        logger.info(f"    股票数量: {len(stock_list)}")
        logger.info("="*60)

        total_stocks = len(stock_list)
        success_count = 0
        error_count = 0

        for idx, ts_code in enumerate(stock_list, 1):
            try:
                # 进度显示
                if idx % 50 == 0 or idx == total_stocks:
                    logger.info(f"进度: {idx}/{total_stocks} ({idx/total_stocks*100:.1f}%)")

                # 获取security_id
                security_id = self._get_or_create_security_id(ts_code)
                if not security_id:
                    error_count += 1
                    continue

                # API调用速率限制
                self._rate_limit()

                # 抓取财务指标数据
                df = self.pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    fields='ts_code,ann_date,end_date,eps,dt_eps,roe,roe_waa,roe_dt,roa,'
                           'grossprofit_margin,netprofit_margin,profit_to_gr,'
                           'ocf_to_profit,debt_to_assets,current_ratio,quick_ratio,'
                           'ar_turn,ca_turn,fa_turn,assets_turn'
                )

                if df.empty:
                    continue

                # 插入数据库
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                for _, row in df.iterrows():
                    try:
                        ann_date = pd.to_datetime(row['ann_date'], format='%Y%m%d').strftime('%Y-%m-%d')
                        end_date_fmt = pd.to_datetime(row['end_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                        cursor.execute("""
                            INSERT OR REPLACE INTO financial_indicator
                            (security_id, ann_date, end_date, eps, dt_eps, roe, roe_waa, roe_dt, roa,
                             grossprofit_margin, netprofit_margin, profit_to_gr, ocf_to_profit,
                             debt_to_assets, current_ratio, quick_ratio, ar_turn, ca_turn, fa_turn, assets_turn)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            security_id,
                            ann_date,
                            end_date_fmt,
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
                            row.get('assets_turn')
                        ))

                        self.stats['financial_indicator_added'] += 1

                    except Exception as e:
                        logger.debug(f"插入financial_indicator失败 ({ts_code}, {end_date_fmt}): {e}")

                conn.commit()
                conn.close()
                success_count += 1

            except Exception as e:
                logger.error(f"❌ 处理{ts_code}失败: {e}")
                error_count += 1
                self.stats['errors'].append(f"financial_indicator {ts_code}: {e}")

        logger.info(f"\n✅ Step 3完成:")
        logger.info(f"   成功: {success_count}/{total_stocks}")
        logger.info(f"   失败: {error_count}")
        logger.info(f"   数据条数: {self.stats['financial_indicator_added']}")

    def print_summary(self):
        """打印统计摘要"""
        logger.info("\n" + "="*60)
        logger.info("📊 数据填充完成统计")
        logger.info("="*60)
        logger.info(f"✅ 新增股票: {self.stats['securities_added']}")
        logger.info(f"✅ daily_basic数据: {self.stats['daily_basic_added']}条")
        logger.info(f"✅ financial_indicator数据: {self.stats['financial_indicator_added']}条")
        logger.info(f"❌ 错误数: {len(self.stats['errors'])}")

        if self.stats['errors']:
            logger.info("\n错误详情（前10个）:")
            for error in self.stats['errors'][:10]:
                logger.info(f"  - {error}")

    def run(self, start_date: str, end_date: str, steps: List[int] = [1, 2, 3]):
        """运行数据初始化"""
        logger.info("\n" + "🚀 "*30)
        logger.info("v3.9数据初始化开始")
        logger.info(f"时间范围: {start_date} ~ {end_date}")
        logger.info("🚀 "*30 + "\n")

        start_time = time.time()

        # Step 1: 获取股票列表
        stock_list = []
        if 1 in steps:
            stock_list = self.step1_initialize_securities()
            if not stock_list:
                logger.error("❌ 未能获取股票列表，终止执行")
                return

        # 如果跳过Step 1，从数据库读取股票列表
        if not stock_list:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT code FROM securities WHERE type = 'A股' AND is_active = 1")
            stock_list = [row[0] for row in cursor.fetchall()]
            conn.close()
            logger.info(f"✅ 从数据库读取到 {len(stock_list)} 只股票")

        # Step 2: 抓取daily_basic
        if 2 in steps:
            self.step2_fetch_daily_basic(stock_list, start_date, end_date)

        # Step 3: 抓取financial_indicator
        if 3 in steps:
            self.step3_fetch_financial_indicator(stock_list, start_date, end_date)

        elapsed_time = time.time() - start_time

        # 打印统计摘要
        self.print_summary()

        logger.info(f"\n⏱️  总耗时: {elapsed_time/60:.1f}分钟")
        logger.info("✅ v3.9数据初始化完成！\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='v3.9数据初始化脚本')
    parser.add_argument('--start-date', type=str, default='20240101',
                       help='开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y%m%d'),
                       help='结束日期 (YYYYMMDD)')
    parser.add_argument('--steps', type=str, default='1,2,3',
                       help='执行的步骤，逗号分隔 (例如: 1,2,3 或 2,3)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--db', type=str, default='data_adapter/stock_data.db',
                       help='数据库路径')

    args = parser.parse_args()

    # 解析步骤
    steps = [int(s.strip()) for s in args.steps.split(',')]

    # 创建logs目录
    Path("logs").mkdir(exist_ok=True)

    # 运行初始化
    initializer = V39DataInitializer(
        config_path=args.config,
        db_path=args.db
    )

    initializer.run(
        start_date=args.start_date,
        end_date=args.end_date,
        steps=steps
    )


if __name__ == "__main__":
    main()
