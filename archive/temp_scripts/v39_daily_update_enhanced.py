#!/usr/bin/env python3
"""
v3.9增强版每日数据更新

整合v3.9所需的数据抓取逻辑：
1. daily_basic（PE/PB/PS/换手率等）- 每日更新
2. financial_indicator（财务指标）- 增量更新（仅当有新发布时）

特点：
- 安全的增量写入（不覆盖已有数据）
- 兼容v3.7/v3.8/v3.81评分器
- 支持批量并行处理
"""

import pandas as pd
import tushare as ts
import time
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os
from typing import List, Dict

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def update_daily_basic_safe(date_str: str, db_path: str = "data_adapter/stock_data.db") -> int:
    """
    安全更新daily_basic数据（v3.9增强版）

    特点：
    - 使用INSERT OR IGNORE，不覆盖已有数据
    - 支持增量更新
    - 确保数据一致性

    Args:
        date_str: 交易日期 (YYYYMMDD)
        db_path: 数据库路径

    Returns:
        成功插入的记录数
    """
    logger.info(f"[v3.9] 开始更新 {date_str} 的daily_basic数据...")

    try:
        # 获取daily_basic数据（批量一次性获取所有股票）
        df = pro.daily_basic(trade_date=date_str)

        if df.empty:
            logger.warning(f"未获取到 {date_str} 的daily_basic数据")
            return 0

        logger.info(f"获取到 {len(df)} 条daily_basic数据")

        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取证券ID映射
        cursor.execute('SELECT code, id FROM securities')
        security_map = {row[0]: row[1] for row in cursor.fetchall()}

        insert_count = 0
        skip_count = 0

        # 格式化日期
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        for _, row in df.iterrows():
            ts_code = row['ts_code']

            if ts_code not in security_map:
                continue

            security_id = security_map[ts_code]

            try:
                # 安全检查：先查询是否已存在
                cursor.execute("""
                    SELECT COUNT(*) FROM daily_basic
                    WHERE security_id = ? AND trade_date = ?
                """, (security_id, formatted_date))

                if cursor.fetchone()[0] > 0:
                    skip_count += 1
                    continue

                # 插入新数据
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
        conn.close()

        logger.info(f"✅ daily_basic更新完成: 新增 {insert_count}, 跳过 {skip_count}")
        return insert_count

    except Exception as e:
        logger.error(f"❌ 更新daily_basic失败: {e}")
        return 0


def update_financial_indicators_safe(date_str: str, db_path: str = "data_adapter/stock_data.db",
                                     check_days: int = 3) -> int:
    """
    安全更新financial_indicator数据（v3.9增强版）

    特点：
    - 仅更新最近check_days天内发布的财报
    - 使用INSERT OR IGNORE，不覆盖已有数据
    - 支持增量更新

    Args:
        date_str: 交易日期 (YYYYMMDD)
        db_path: 数据库路径
        check_days: 检查最近几天的财报发布

    Returns:
        成功插入的记录数
    """
    logger.info(f"[v3.9] 检查 {date_str} 前{check_days}天内发布的财务数据...")

    try:
        # 计算日期范围
        end_date = datetime.strptime(date_str, '%Y%m%d')
        start_date = end_date - timedelta(days=check_days)
        start_date_str = start_date.strftime('%Y%m%d')

        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取所有A股代码
        cursor.execute("""
            SELECT code FROM securities
            WHERE type = 'A股' AND is_active = 1
        """)
        stock_codes = [row[0] for row in cursor.fetchall()]

        if not stock_codes:
            logger.warning("未找到活跃股票")
            conn.close()
            return 0

        logger.info(f"检查 {len(stock_codes)} 只A股在 {start_date_str}~{date_str} 期间发布的财报")

        # 获取证券ID映射
        cursor.execute('SELECT code, id FROM securities')
        security_map = {row[0]: row[1] for row in cursor.fetchall()}

        insert_count = 0
        skip_count = 0
        error_count = 0

        # 分批处理（避免API超限）
        batch_size = 50
        for i in range(0, len(stock_codes), batch_size):
            batch_codes = stock_codes[i:i+batch_size]

            for ts_code in batch_codes:
                try:
                    # API速率限制
                    time.sleep(0.2)

                    # 获取该股票最近发布的财务指标
                    df = pro.fina_indicator(
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

                    # 插入数据
                    for _, row in df.iterrows():
                        try:
                            ann_date = pd.to_datetime(row['ann_date'], format='%Y%m%d').strftime('%Y-%m-%d')
                            end_date_fmt = pd.to_datetime(row['end_date'], format='%Y%m%d').strftime('%Y-%m-%d')

                            # 安全检查：先查询是否已存在
                            cursor.execute("""
                                SELECT COUNT(*) FROM financial_indicator
                                WHERE security_id = ? AND end_date = ?
                            """, (security_id, end_date_fmt))

                            if cursor.fetchone()[0] > 0:
                                skip_count += 1
                                continue

                            # 插入新数据
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
                            logger.debug(f"插入financial_indicator失败 {ts_code}/{end_date_fmt}: {e}")

                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # 只显示前5个错误
                        logger.error(f"获取{ts_code}财务数据失败: {e}")

            # 每批次提交一次
            conn.commit()

            if (i + batch_size) % 500 == 0:
                logger.info(f"进度: {min(i+batch_size, len(stock_codes))}/{len(stock_codes)}")

        conn.close()

        logger.info(f"✅ financial_indicator更新完成: 新增 {insert_count}, 跳过 {skip_count}, 错误 {error_count}")
        return insert_count

    except Exception as e:
        logger.error(f"❌ 更新financial_indicator失败: {e}")
        return 0


def v39_daily_update(date_str: str = None, update_financial: bool = True) -> Dict:
    """
    v3.9每日数据更新主函数

    Args:
        date_str: 交易日期 (YYYYMMDD)，默认为今天
        update_financial: 是否更新财务指标（默认True，但周末可设为False）

    Returns:
        更新统计信息
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')

    logger.info("\n" + "="*60)
    logger.info(f"[v3.9] 开始每日数据更新: {date_str}")
    logger.info("="*60)

    stats = {
        'date': date_str,
        'daily_basic_added': 0,
        'financial_indicator_added': 0,
        'success': True
    }

    start_time = time.time()

    try:
        # 1. 更新daily_basic（必需，每日更新）
        logger.info("\n[1/2] 更新daily_basic数据...")
        stats['daily_basic_added'] = update_daily_basic_safe(date_str)

        # 2. 更新financial_indicator（可选，检查最近3天的财报发布）
        if update_financial:
            logger.info("\n[2/2] 更新financial_indicator数据...")
            stats['financial_indicator_added'] = update_financial_indicators_safe(date_str, check_days=3)
        else:
            logger.info("\n[2/2] 跳过financial_indicator更新")

        elapsed = time.time() - start_time

        logger.info("\n" + "="*60)
        logger.info("[v3.9] 每日数据更新完成")
        logger.info(f"  daily_basic: +{stats['daily_basic_added']} 条")
        logger.info(f"  financial_indicator: +{stats['financial_indicator_added']} 条")
        logger.info(f"  耗时: {elapsed:.1f}秒")
        logger.info("="*60 + "\n")

    except Exception as e:
        logger.error(f"❌ 每日数据更新失败: {e}")
        stats['success'] = False

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='v3.9每日数据更新')
    parser.add_argument('--date', type=str, default=None,
                       help='交易日期 (YYYYMMDD)，默认为今天')
    parser.add_argument('--no-financial', action='store_true',
                       help='跳过财务指标更新（适用于周末）')

    args = parser.parse_args()

    # 执行更新
    result = v39_daily_update(
        date_str=args.date,
        update_financial=not args.no_financial
    )

    # 返回状态码
    sys.exit(0 if result['success'] else 1)
