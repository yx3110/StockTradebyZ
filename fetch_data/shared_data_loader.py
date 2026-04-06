#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared data loading utilities for feature cache updaters.

Extracted from v39_feature_cache_updater.py and v40_feature_cache_updater.py
to eliminate code duplication.
"""

import sqlite3
import logging
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Sequence

DB_PATH = str(Path(__file__).parent.parent / 'data_adapter' / 'stock_data.db')

logger = logging.getLogger(__name__)


def batch_load_stock_data(
    date: str,
    lookback: int = 60,
    db_path: str = None,
    stock_types: Sequence[str] = ('A股', 'ETF_基金'),
) -> Dict[str, pd.DataFrame]:
    """Load all stock OHLCV + price_change_pct from daily_quotes, grouped by code.

    Args:
        date:        End date in 'YYYY-MM-DD' format.
        lookback:    Number of trading days to look back.
        db_path:     Path to SQLite database (default: stock_data.db).
        stock_types: Security types to include, e.g. ('A股',) or ('A股', 'ETF_基金').

    Returns:
        {code: DataFrame} where each DataFrame has columns
        [code, trade_date, open, high, low, close, volume, price_change_pct].
    """
    if db_path is None:
        db_path = DB_PATH

    logger.info(f"批量预加载股票数据 (lookback={lookback}天, types={stock_types})...")
    start_time = time.time()

    end_dt = datetime.strptime(date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=lookback + 30)  # 多加余量

    conn = sqlite3.connect(db_path)

    placeholders = ','.join('?' for _ in stock_types)
    query = f"""
    SELECT s.code, q.trade_date, q.open, q.high, q.low, q.close, q.volume, q.price_change_pct
    FROM daily_quotes q
    JOIN securities s ON q.security_id = s.id
    WHERE s.type IN ({placeholders})
    AND q.trade_date >= ?
    AND q.trade_date <= ?
    ORDER BY s.code, q.trade_date
    """

    params = list(stock_types) + [start_dt.strftime('%Y-%m-%d'), date]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    stock_data: Dict[str, pd.DataFrame] = {}
    for code, group in df.groupby('code'):
        stock_data[code] = group.reset_index(drop=True)

    elapsed = time.time() - start_time
    logger.info(f"批量加载完成: {len(stock_data)} 只股票, 耗时 {elapsed:.1f}秒")

    return stock_data


def load_sw_industry_mapping(
    db_path: str = None,
) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, str]]:
    """Load Shenwan L1 industry mapping from sw_industry table.

    Args:
        db_path: Path to SQLite database (default: stock_data.db).

    Returns:
        Tuple of three dicts:
          - code_to_industry:  {stock_code: l1_name}
          - industry_to_label: {l1_name: int}  (sorted alphabetical encoding)
          - industry_to_code:  {l1_name: l1_code}  (申万一级行业代码)
    """
    if db_path is None:
        db_path = DB_PATH

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sw_industry'")
    if not cursor.fetchone():
        logger.warning("sw_industry表不存在，行业特征将使用默认值")
        conn.close()
        return {}, {}, {}

    # code -> l1_name
    cursor.execute("SELECT code, l1_name FROM sw_industry WHERE is_new = 'Y'")
    code_to_industry = {row[0]: row[1] for row in cursor.fetchall()}

    # l1_name -> label (排序后编码)
    cursor.execute("SELECT DISTINCT l1_name FROM sw_industry WHERE is_new = 'Y' ORDER BY l1_name")
    names = [row[0] for row in cursor.fetchall()]
    industry_to_label = {name: i for i, name in enumerate(names)}

    # l1_name -> l1_code (申万一级行业代码)
    cursor.execute("SELECT DISTINCT l1_code, l1_name FROM sw_industry WHERE is_new = 'Y'")
    industry_to_code = {row[1]: row[0] for row in cursor.fetchall()}

    conn.close()
    logger.info(f"加载申万行业映射: {len(code_to_industry)} 只股票, "
                f"{len(industry_to_label)} 个行业")

    return code_to_industry, industry_to_label, industry_to_code
