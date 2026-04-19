"""
工具函数
"""
import sqlite3
import logging
from contextlib import closing
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def format_date(date_str: str, input_format: str = '%Y-%m-%d', output_format: str = '%Y-%m-%d') -> str:
    """格式化日期字符串"""
    try:
        return datetime.strptime(date_str, input_format).strftime(output_format)
    except Exception:
        return date_str


def ensure_directory(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_market_regime(stock_db_path) -> str:
    """从沪深300指数判断市场状态 (bull/bear/neutral)。

    20日收益>5%且收盘>60日均线 → bull；<-5%且<60日均线 → bear；其他 neutral。
    """
    try:
        with closing(sqlite3.connect(stock_db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            rows = conn.execute("""
                SELECT dq.close FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code IN ('000300', '399300')
                ORDER BY dq.trade_date DESC LIMIT 60
            """).fetchall()
        if len(rows) < 20:
            return 'neutral'
        prices = [r[0] for r in reversed(rows)]
        ret_20d = (prices[-1] - prices[-20]) / prices[-20]
        ma60 = sum(prices) / len(prices)
        if ret_20d > 0.05 and prices[-1] > ma60:
            return 'bull'
        if ret_20d < -0.05 and prices[-1] < ma60:
            return 'bear'
        return 'neutral'
    except Exception as e:
        logger.warning("detect_market_regime failed: %s", e)
        return 'neutral'
