#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板块/成分数据抓取器 (webapp 市场行情三页面数据源)

抓取内容:
1. sw_industry 表扩展 L2/L3 列并全量刷新 (index_member_all, 申万2021三级成分)
2. dc_index_daily 表: 东财板块指数日行情 (概念/行业/地域, 含 pct_change, 2024-12-20 起)
3. dc_member 表: 东财概念板块成分股 (当前快照, 每次运行刷新)
4. index_weight_snapshot 表: 沪深300 + 中证2000 月度成分快照 (index_weight)
5. limit_list_daily 表: 每日涨停/跌停/炸板列表 (limit_list_d, 2024-01-02 起)

用法:
  python3 fetch_data/market_board_fetcher.py --backfill              # 首次全量回填
  python3 fetch_data/market_board_fetcher.py --daily                 # 日常增量 (接入 quick_daily_update)
  python3 fetch_data/market_board_fetcher.py --daily --date 20260710
"""

import json
import logging
import argparse
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

import tushare as ts
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_adapter.sqlite_utils import connect as sqlite_connect, write_retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DC_INDEX_EARLIEST = '20241220'      # dc_index 数据最早可用日期
LIMIT_LIST_EARLIEST = '20240102'    # limit_list_d 回填起点
INDEX_WEIGHT_EARLIEST = '202401'    # 指数成分快照回填起点 (月)
TRACKED_INDICES = ['000300.SH', '932000.CSI']  # 沪深300 / 中证2000


class MarketBoardFetcher:

    def __init__(self, db_path: str = None, config_path: str = None):
        self.db_path = db_path or str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
        # token 优先从 core.config/.env 获取
        try:
            from core.config import get_tushare_token
            token = get_tushare_token()
        except ImportError:
            config_path = config_path or str(PROJECT_ROOT / 'config.json')
            with open(config_path) as f:
                token = json.load(f)['tushare']['token']
        ts.set_token(token)
        self.pro = ts.pro_api()
        self._ensure_tables()

    def _conn(self):
        return sqlite_connect(self.db_path)

    def _call(self, api_name: str, **kw) -> pd.DataFrame:
        """Tushare 调用: 3 次重试 + 限频 (~400次/分)"""
        for attempt in range(3):
            try:
                df = getattr(self.pro, api_name)(**kw)
                time.sleep(0.25)  # 限频保守值: 与其他抓取任务共享 500次/分配额
                return df
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning("%s%s 失败 (%s), 重试 %d/2", api_name, kw, e, attempt + 1)
                time.sleep(2 ** attempt * 5)

    def _write_retry(self, fn):
        return write_retry(fn)

    def _call_paged(self, api_name: str, page_size: int = 5000, **kw) -> pd.DataFrame:
        """带 offset 分页的 Tushare 调用 (结果可能超单次上限的接口)"""
        frames, offset = [], 0
        while True:
            df = self._call(api_name, limit=page_size, offset=offset, **kw)
            if df is None or df.empty:
                break
            frames.append(df)
            if len(df) < page_size:
                break
            offset += page_size
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _ensure_tables(self):
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dc_index_daily (
                ts_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,          -- YYYYMMDD
                name TEXT,
                idx_type TEXT,                     -- 概念板块/行业板块/地域板块
                pct_change REAL,
                leading TEXT,
                leading_code TEXT,
                leading_pct REAL,
                total_mv REAL,
                turnover_rate REAL,
                up_num INTEGER,
                down_num INTEGER,
                PRIMARY KEY (ts_code, trade_date)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dc_index_date ON dc_index_daily(trade_date)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dc_member (
                ts_code TEXT NOT NULL,             -- 板块代码 BKxxxx.DC
                name TEXT,                         -- 板块名
                con_code TEXT NOT NULL,            -- 成分股 000001.SZ
                con_name TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ts_code, con_code)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dc_member_con ON dc_member(con_code)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_weight_snapshot (
                index_code TEXT NOT NULL,
                con_code TEXT NOT NULL,
                trade_date TEXT NOT NULL,          -- YYYYMMDD (月末快照日)
                weight REAL,
                PRIMARY KEY (index_code, con_code, trade_date)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS limit_list_daily (
                trade_date TEXT NOT NULL,          -- YYYYMMDD
                ts_code TEXT NOT NULL,
                name TEXT,
                limit_type TEXT,                   -- U涨停 D跌停 Z炸板
                open_times INTEGER,                -- 炸板次数
                up_stat TEXT,                      -- N连板中的位置 如 2/3
                limit_times INTEGER,               -- 连板数
                PRIMARY KEY (trade_date, ts_code)
            )
        """)
        # sw_industry 表扩展 L2/L3 列
        sw_cols = {r[1] for r in conn.execute("PRAGMA table_info(sw_industry)").fetchall()}
        for col in ['l2_code', 'l2_name', 'l3_code', 'l3_name']:
            if sw_cols and col not in sw_cols:
                conn.execute(f"ALTER TABLE sw_industry ADD COLUMN {col} TEXT")
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # 1. 申万三级成分
    # ------------------------------------------------------------------
    def fetch_sw_members(self, max_age_days: int = 7):
        """全量抓取申万 L1/L2/L3 成分, 更新 sw_industry (含新增 L2/L3 列)"""
        conn = self._conn()
        last = conn.execute(
            "SELECT MAX(updated_at) FROM sw_industry WHERE l2_code IS NOT NULL").fetchone()[0]
        if last and (datetime.now() - datetime.fromisoformat(last)).days < max_age_days:
            conn.close()
            logger.info("sw_industry L2/L3 快照 %s 仍新鲜, 跳过", last[:10])
            return
        logger.info("抓取申万三级行业成分 (index_member_all 分页)...")
        all_df = self._call_paged('index_member_all')
        if all_df.empty:
            conn.close()
            logger.warning("index_member_all 无数据返回")
            return
        # (ts_code, l1_code) 表内唯一; 同键多条历史记录时让 is_new='Y' 最后应用
        all_df = (all_df.drop_duplicates(subset=['ts_code', 'l3_code', 'in_date'])
                        .sort_values('is_new'))
        logger.info("获取 %d 条成分记录 (%d 只股票)", len(all_df), all_df['ts_code'].nunique())

        existing = {(r[0], r[1]) for r in conn.execute("SELECT ts_code, l1_code FROM sw_industry")}
        upd, ins = [], []
        for row in all_df.itertuples(index=False):
            if (row.ts_code, row.l1_code) in existing:
                upd.append((row.l2_code, row.l2_name, row.l3_code, row.l3_name,
                            row.name, row.out_date, row.is_new, row.ts_code, row.l1_code))
            else:
                ins.append((row.ts_code.split('.')[0], row.ts_code, row.name,
                            row.l1_code, row.l1_name, row.l2_code, row.l2_name,
                            row.l3_code, row.l3_name, row.in_date, row.out_date, row.is_new))

        def _write():
            conn.executemany(
                "UPDATE sw_industry SET l2_code=?, l2_name=?, l3_code=?, l3_name=?, "
                "stock_name=?, out_date=?, is_new=?, updated_at=CURRENT_TIMESTAMP "
                "WHERE ts_code=? AND l1_code=?", upd)
            conn.executemany(
                "INSERT OR IGNORE INTO sw_industry "
                "(code, ts_code, stock_name, l1_code, l1_name, l2_code, l2_name, "
                " l3_code, l3_name, in_date, out_date, is_new, src) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'SW2021')", ins)
            conn.commit()
        self._write_retry(_write)
        n_l2 = conn.execute(
            "SELECT COUNT(DISTINCT l2_name) FROM sw_industry WHERE l2_name IS NOT NULL").fetchone()[0]
        conn.close()
        logger.info("✅ 申万成分: 更新 %d, 新增 %d, L2 行业数 %d", len(upd), len(ins), n_l2)

    # ------------------------------------------------------------------
    # 2. 东财板块指数日行情
    # ------------------------------------------------------------------
    def fetch_dc_index(self, dates: list):
        """按日抓取 dc_index 全板块行情"""
        conn = self._conn()
        done = {r[0] for r in conn.execute("SELECT DISTINCT trade_date FROM dc_index_daily")}
        todo = [d for d in dates if d not in done]
        logger.info("dc_index: 需抓取 %d 天 (已有 %d 天)", len(todo), len(done))
        for i, d in enumerate(todo):
            df = self._call('dc_index', trade_date=d)
            if df is None or df.empty:
                continue
            rows = df[['ts_code', 'trade_date', 'name', 'idx_type', 'pct_change', 'leading',
                       'leading_code', 'leading_pct', 'total_mv', 'turnover_rate',
                       'up_num', 'down_num']].itertuples(index=False, name=None)
            rows = list(rows)
            self._write_retry(lambda: (conn.executemany(
                "INSERT OR REPLACE INTO dc_index_daily "
                "(ts_code, trade_date, name, idx_type, pct_change, leading, leading_code, "
                " leading_pct, total_mv, turnover_rate, up_num, down_num) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows), conn.commit()))
            if (i + 1) % 50 == 0:
                logger.info("  dc_index 进度 %d/%d", i + 1, len(todo))
        conn.close()
        logger.info("✅ dc_index_daily 完成")

    # ------------------------------------------------------------------
    # 3. 东财概念成分 (当前快照)
    # ------------------------------------------------------------------
    def fetch_dc_members(self, max_age_days: int = 7):
        """刷新概念板块成分股. 按板块粒度跳过新鲜快照 (支持中断续跑)"""
        conn = self._conn()
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat(sep=' ')
        fresh = {r[0] for r in conn.execute(
            "SELECT ts_code FROM dc_member GROUP BY ts_code HAVING MAX(updated_at) >= ?",
            (cutoff,))}
        boards = [(bk, name) for bk, name in conn.execute(
            "SELECT ts_code, name FROM dc_index_daily WHERE idx_type='概念板块' "
            "AND trade_date=(SELECT MAX(trade_date) FROM dc_index_daily)")
            if bk not in fresh]
        logger.info("刷新 %d 个概念板块成分 (%d 个仍新鲜跳过)...", len(boards), len(fresh))
        for i, (bk, name) in enumerate(boards):
            # dc_member 按板块返回逐日快照 (大板块可达10万+行, 深度翻页会被拒),
            # 只取首页 (接口按日期倒序, 8000 行足够覆盖最大板块单日成分) 并过滤最新快照日
            df = self._call('dc_member', ts_code=bk, limit=8000)
            if df is None or df.empty:
                continue
            df = (df[df['trade_date'] == df['trade_date'].max()]
                  .drop_duplicates(subset=['con_code']))

            def _write(bk=bk, name=name, df=df):
                conn.execute("DELETE FROM dc_member WHERE ts_code=?", (bk,))
                conn.executemany(
                    "INSERT OR REPLACE INTO dc_member (ts_code, name, con_code, con_name, updated_at) "
                    "VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
                    [(bk, name, r.con_code, r.name) for r in df.itertuples(index=False)])
                conn.commit()
            self._write_retry(_write)
            if (i + 1) % 50 == 0:
                logger.info("  dc_member 进度 %d/%d", i + 1, len(boards))
        conn.close()
        logger.info("✅ dc_member 完成")

    # ------------------------------------------------------------------
    # 4. 指数成分月度快照
    # ------------------------------------------------------------------
    def fetch_index_weights(self):
        """按月抓取沪深300/中证2000成分快照, 增量补缺"""
        conn = self._conn()
        months = []
        cur = datetime.strptime(INDEX_WEIGHT_EARLIEST, '%Y%m')
        while cur.strftime('%Y%m') <= datetime.now().strftime('%Y%m'):
            months.append(cur.strftime('%Y%m'))
            cur = (cur + timedelta(days=32)).replace(day=1)
        for idx_code in TRACKED_INDICES:
            done = {r[0][:6] for r in conn.execute(
                "SELECT DISTINCT trade_date FROM index_weight_snapshot WHERE index_code=?",
                (idx_code,))}
            todo = [m for m in months if m not in done]
            logger.info("index_weight %s: 补 %d 个月", idx_code, len(todo))
            for m in todo:
                df = self._call_paged('index_weight', index_code=idx_code,
                                      start_date=m + '01', end_date=m + '31')
                if df is None or df.empty:
                    continue
                # 一个月内可能有多个快照日, 只留最新一天
                latest = df['trade_date'].max()
                df = df[df['trade_date'] == latest]
                rows = list(df[['index_code', 'con_code', 'trade_date', 'weight']]
                            .itertuples(index=False, name=None))
                self._write_retry(lambda: (conn.executemany(
                    "INSERT OR REPLACE INTO index_weight_snapshot "
                    "(index_code, con_code, trade_date, weight) VALUES (?,?,?,?)",
                    rows), conn.commit()))
        conn.close()
        logger.info("✅ index_weight_snapshot 完成")

    # ------------------------------------------------------------------
    # 5. 涨跌停列表
    # ------------------------------------------------------------------
    def fetch_limit_list(self, dates: list):
        conn = self._conn()
        done = {r[0] for r in conn.execute("SELECT DISTINCT trade_date FROM limit_list_daily")}
        todo = [d for d in dates if d not in done]
        logger.info("limit_list_d: 需抓取 %d 天", len(todo))
        for i, d in enumerate(todo):
            df = self._call('limit_list_d', trade_date=d)
            if df is None or df.empty:
                continue
            rows = list(df[['trade_date', 'ts_code', 'name', 'limit', 'open_times', 'up_stat',
                            'limit_times']].itertuples(index=False, name=None))
            self._write_retry(lambda: (conn.executemany(
                "INSERT OR REPLACE INTO limit_list_daily "
                "(trade_date, ts_code, name, limit_type, open_times, up_stat, limit_times) "
                "VALUES (?,?,?,?,?,?,?)", rows), conn.commit()))
            if (i + 1) % 50 == 0:
                logger.info("  limit_list 进度 %d/%d", i + 1, len(todo))
        conn.close()
        logger.info("✅ limit_list_daily 完成")

    # ------------------------------------------------------------------
    def local_trade_dates(self, start_yyyymmdd: str, end_yyyymmdd: str) -> list:
        """从本地 daily_quotes 取交易日历 (YYYYMMDD)"""
        conn = self._conn()
        s = f"{start_yyyymmdd[:4]}-{start_yyyymmdd[4:6]}-{start_yyyymmdd[6:]}"
        e = f"{end_yyyymmdd[:4]}-{end_yyyymmdd[4:6]}-{end_yyyymmdd[6:]}"
        rows = conn.execute(
            "SELECT DISTINCT trade_date FROM daily_quotes "
            "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date", (s, e)).fetchall()
        conn.close()
        return [r[0].replace('-', '') for r in rows]

    def _run_sections(self, sections):
        """逐节执行, 单节失败记录后继续 (与其他回填任务并发时局部锁死不拖垮全局)"""
        failed = []
        for name, fn in sections:
            try:
                fn()
            except Exception as e:
                logger.error("❌ %s 失败: %s", name, e)
                failed.append(name)
        if failed:
            raise RuntimeError(f"以下环节失败, 请重跑: {failed}")

    def run_backfill(self):
        today = datetime.now().strftime('%Y%m%d')
        self._run_sections([
            ('sw_members', self.fetch_sw_members),
            ('dc_index', lambda: self.fetch_dc_index(
                self.local_trade_dates(DC_INDEX_EARLIEST, today))),
            ('dc_members', self.fetch_dc_members),
            ('index_weights', self.fetch_index_weights),
            ('limit_list', lambda: self.fetch_limit_list(
                self.local_trade_dates(LIMIT_LIST_EARLIEST, today))),
        ])

    def run_daily(self, date_yyyymmdd: str = None):
        d = date_yyyymmdd or datetime.now().strftime('%Y%m%d')
        # 增量: 从已有最大日期补到 d (漏跑几天也能补齐)
        conn = self._conn()
        last_dc = conn.execute("SELECT MAX(trade_date) FROM dc_index_daily").fetchone()[0]
        last_ll = conn.execute("SELECT MAX(trade_date) FROM limit_list_daily").fetchone()[0]
        conn.close()
        self._run_sections([
            ('dc_index', lambda: self.fetch_dc_index(
                self.local_trade_dates(last_dc or DC_INDEX_EARLIEST, d))),
            ('limit_list', lambda: self.fetch_limit_list(
                self.local_trade_dates(last_ll or LIMIT_LIST_EARLIEST, d))),
            ('dc_members', self.fetch_dc_members),
            ('sw_members', self.fetch_sw_members),
            ('index_weights', self.fetch_index_weights),
        ])


def main():
    ap = argparse.ArgumentParser(description='板块/成分数据抓取器')
    ap.add_argument('--backfill', action='store_true', help='首次全量回填')
    ap.add_argument('--daily', action='store_true', help='日常增量更新')
    ap.add_argument('--date', help='指定日期 YYYYMMDD (配合 --daily)')
    ap.add_argument('--db-path', help='数据库路径')
    args = ap.parse_args()

    fetcher = MarketBoardFetcher(db_path=args.db_path)
    if args.backfill:
        fetcher.run_backfill()
    elif args.daily:
        fetcher.run_daily(args.date)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
