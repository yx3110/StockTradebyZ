"""
交易日判断工具

从 tomorrow_stock_selector.py 提取, 统一交易日判断逻辑。
优先级: Tushare API > 数据库查询 > 简单规则(周一至周五)
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def is_trading_day(date_str: str) -> bool:
    """检查指定日期是否为交易日

    Args:
        date_str: 日期字符串, 格式 'YYYY-MM-DD'

    Returns:
        True 如果是交易日
    """
    try:
        # 1. 尝试 Tushare API (单次调用，缓存结果)
        tushare_result = _check_via_tushare(date_str)
        if tushare_result is not None:
            return tushare_result
    except Exception:
        pass

    # 2. 检查数据库
    try:
        result = _check_via_database(date_str)
        if result is not None:
            return result
    except Exception:
        pass

    # 3. 简单规则: 周一到周五
    try:
        weekday = datetime.strptime(date_str, '%Y-%m-%d').weekday()
        return weekday < 5
    except Exception:
        return True


def check_trading_day_via_tushare(date_str: str):
    """公开的 tri-state 交易日查询: True/False/None(API 查询失败).

    与 is_trading_day 不同, 不做 DB/星期几 fallback — 调用方需要区分
    「明确非交易日」和「日历也查不到 = API 全线故障」时用这个
    (如 quick_daily_update 判定行情 0 条是节假日还是数据源故障)。
    接受 'YYYY-MM-DD' 或 'YYYYMMDD'。
    """
    if '-' not in date_str and len(date_str) == 8:
        date_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    res = _check_via_tushare(date_str)
    # pandas 比较产出 np.bool_, 而调用方常用 `is False` 身份判断 — 必须转 Python bool
    return None if res is None else bool(res)


def _check_via_tushare(date_str: str):
    """通过 Tushare API 查询交易日历, 返回 True/False/None(查询失败)"""
    try:
        import tushare as ts
        from core.config import get_tushare_token

        token = get_tushare_token()
        ts.set_token(token)
        pro = ts.pro_api()

        check_date = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y%m%d')
        cal_df = pro.trade_cal(
            exchange='SSE',
            start_date=check_date,
            end_date=check_date,
            fields='cal_date,is_open'
        )
        if not cal_df.empty:
            return cal_df.iloc[0]['is_open'] == 1
    except Exception as e:
        logger.debug(f"Tushare交易日查询失败: {e}")
    return None


def _check_via_database(date_str: str):
    """通过数据库查询是否有行情数据, 返回 True/False/None"""
    import sqlite3
    from core.config import get_db_path

    db_path = get_db_path()
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path), timeout=5)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM daily_quotes WHERE trade_date = ?", (date_str,))
        count = cursor.fetchone()[0]
        if count > 100:
            return True
        elif count == 0:
            # 可能是非交易日, 也可能是数据缺失, 返回 None 让 fallback 处理
            return None
        return True
    except Exception:
        return None
    finally:
        conn.close()
