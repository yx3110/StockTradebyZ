from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import os

import pandas as pd
import tushare as ts
from tqdm import tqdm

from db_manager import DatabaseManager

warnings.filterwarnings("ignore")

# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("quick_daily_update.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("quick_daily_update")

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
BAN_PATTERNS = (
    "访问频繁", "请稍后", "超过频率", "频繁访问",
    "too many requests", "429",
    "forbidden", "403",
    "max retries exceeded"
)

DEFAULT_START = "20190101"  # 若数据库无记录时的默认起始日期


def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in BAN_PATTERNS)


class RateLimitError(RuntimeError):
    """表示命中限流/封禁，需要长时间冷却后重试。"""
    pass


def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)


# --------------------------- 历史K线（Tushare 日线，固定qfq） --------------------------- #
pro: Optional[ts.pro_api] = None  # 模块级会话


def set_api(session) -> None:
    """由外部(比如GUI)注入已创建好的 ts.pro_api() 会话"""
    global pro
    pro = session


def _to_ts_code(code: str) -> str:
    """把6位code映射到标准 ts_code 后缀。"""
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"{code}.SH"
    elif code.startswith(("4", "8")):
        return f"{code}.BJ"
    else:
        return f"{code}.SZ"


def _get_kline_tushare(code: str, start: str, end: str) -> pd.DataFrame:
    ts_code = _to_ts_code(code)
    try:
        df = ts.pro_bar(
            ts_code=ts_code,
            adj="qfq",
            start_date=start,
            end_date=end,
            freq="D",
            api=pro
        )
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={"trade_date": "date", "vol": "volume"})[
        ["date", "open", "close", "high", "low", "volume"]
    ].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df


# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #

def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    """
    exclude_boards 子集：{'gem','star','bj'}
    - gem  : 创业板 300/301（.SZ）
    - star : 科创板 688（.SH）
    - bj   : 北交所（.BJ 或 4/8 开头）
    """
    code = df["symbol"].astype(str)
    ts_code = df["ts_code"].astype(str).str.upper()
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~code.str.startswith(("300", "301"))
    if "star" in exclude_boards:
        mask &= ~code.str.startswith(("688",))
    if "bj" in exclude_boards:
        mask &= ~(ts_code.str.endswith(".BJ") | code.str.startswith(("4", "8")))

    return df[mask].copy()


def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    df = pd.read_csv(stocklist_csv)
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info("从 %s 读取到 %d 只股票（排除板块：%s）",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes


# --------------------------- 单只增量更新（仅写 SQLite） --------------------------- #

def update_one(
    code: str,
    end: str,
    db: DatabaseManager,
    default_start: str = DEFAULT_START,
) -> None:
    """从 SQLite 读取最新日期，抓取增量数据，仅写入 SQLite（不写 CSV）。

    Parameters
    ----------
    code:
        6 位股票代码。
    end:
        抓取截止日期，格式 YYYYMMDD。
    db:
        DatabaseManager 实例，线程共享（内部使用线程本地连接）。
    default_start:
        数据库中无记录时的默认起始日期，格式 YYYYMMDD。
    """
    # 从 SQLite 读取该股票的最新已存储日期
    last_date_str = db.get_last_date(code)

    if last_date_str is None:
        # 数据库无记录，从 default_start 开始全量抓取
        start = default_start
    else:
        # 从最后存储日期的下一天开始抓取，避免重复
        last_date = dt.datetime.strptime(last_date_str, "%Y-%m-%d").date()
        next_date = last_date + dt.timedelta(days=1)
        start = next_date.strftime("%Y%m%d")

    # 若起始日期已晚于截止日期，该股票已是最新，跳过
    end_date = dt.datetime.strptime(end, "%Y%m%d").date()
    start_date = dt.datetime.strptime(start, "%Y%m%d").date()
    if start_date > end_date:
        logger.debug("%s 已是最新（last=%s），跳过。", code, last_date_str)
        return

    for attempt in range(1, 4):
        try:
            new_df = _get_kline_tushare(code, start, end)
            if new_df.empty:
                logger.debug("%s 增量区间 [%s, %s] 无新数据。", code, start, end)
                return
            new_df = validate(new_df)
            # 仅写入 SQLite，不写 CSV
            db.upsert_klines(code, new_df)
            logger.debug("%s 增量写入 %d 行 → SQLite。", code, len(new_df))
            return
        except Exception as e:
            if _looks_like_ip_ban(e):
                logger.error("%s 第 %d 次抓取疑似被封禁，沉睡 %d 秒", code, attempt, COOLDOWN_SECS)
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 15 * attempt
                logger.info("%s 第 %d 次抓取失败，%d 秒后重试：%s", code, attempt, silent_seconds, e)
                time.sleep(silent_seconds)

    logger.error("%s 三次抓取均失败，已跳过！", code)


# --------------------------- 主入口 --------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="从 SQLite 读取最新日期，增量抓取 Tushare 日线K线（固定qfq），仅写入 SQLite（不写 CSV）"
    )
    # 抓取范围
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'")
    parser.add_argument(
        "--default-start", default=DEFAULT_START,
        help="数据库无记录时的默认起始日期 YYYYMMDD（默认 20190101）"
    )
    # 股票清单与板块过滤
    parser.add_argument(
        "--stocklist", type=Path, default=Path("./stocklist.csv"),
        help="股票清单CSV路径（需含 ts_code 与 symbol 列）"
    )
    parser.add_argument(
        "--exclude-boards",
        nargs="*",
        default=[],
        choices=["gem", "star", "bj"],
        help="排除板块，可多选：gem(创业板300/301) star(科创板688) bj(北交所.BJ/4/8)"
    )
    # 数据库
    parser.add_argument("--db", default="./stock_data.db", help="SQLite 数据库路径")
    # 其它
    parser.add_argument("--workers", type=int, default=6, help="并发线程数")
    args = parser.parse_args()

    # ---------- Tushare Token ---------- #
    os.environ["NO_PROXY"] = "api.waditu.com,.waditu.com,waditu.com"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    ts_token = os.environ.get("TUSHARE_TOKEN")
    if not ts_token:
        raise ValueError("请先设置环境变量 TUSHARE_TOKEN，例如：export TUSHARE_TOKEN=你的token")
    ts.set_token(ts_token)
    global pro
    pro = ts.pro_api()

    # ---------- 日期解析 ---------- #
    end = dt.date.today().strftime("%Y%m%d") if str(args.end).lower() == "today" else args.end

    # ---------- 初始化数据库 ---------- #
    db = DatabaseManager(db_path=args.db)

    # ---------- 从 stocklist.csv 读取股票池 ---------- #
    exclude_boards = set(args.exclude_boards or [])
    codes = load_codes_from_stocklist(args.stocklist, exclude_boards)

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    logger.info(
        "开始增量更新 %d 支股票 → SQLite(%s) | 截止日期:%s | 排除:%s",
        len(codes), args.db, end, ",".join(sorted(exclude_boards)) or "无",
    )

    # ---------- 多线程增量更新（仅写 SQLite） ---------- #
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                update_one,
                code,
                end,
                db,
                args.default_start,
            )
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="增量更新进度"):
            pass

    db.close_all()
    logger.info("全部增量更新完成，数据已写入 SQLite: %s", Path(args.db).resolve())


if __name__ == "__main__":
    main()
