#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场脉搏预计算: 板块日统计 + 全市场情绪日统计 (webapp 市场行情三页面数据层)

产出表 (stock_data.db):
1. sector_daily_stats  — 每日 × 板块 (taxonomy: sw_l1/sw_l2/concept):
   等权涨幅 / 主力净流入(亿) / 20日新高成分股数 / 涨跌家数
   - sw_l1/sw_l2: 申万点时成分 (in_date/out_date), 涨幅=成分等权平均
   - concept: 东财概念 (dc_member 当前快照), 涨幅优先用 dc_index_daily 官方值
2. market_sentiment_daily — 每日: 涨停/跌停/炸板数 (limit_list_daily),
   上涨/下跌家数, 沪深300 & 中证2000 成分 20日新高数

依赖: market_board_fetcher.py 先回填 dc_index_daily / dc_member /
      index_weight_snapshot / limit_list_daily / sw_industry(L2)

用法:
  python3 scripts/build_market_pulse.py                       # 增量 (接 quick_daily_update)
  python3 scripts/build_market_pulse.py --start-date 2024-01-01 --end-date 2026-07-10
  python3 scripts/build_market_pulse.py --rebuild             # 从 DEFAULT_START 全量重算
"""

import logging
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_adapter.sqlite_utils import connect as sqlite_connect, write_retry as _write_retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = str(PROJECT_ROOT / 'data_adapter' / 'stock_data.db')
DEFAULT_START = '2024-01-02'   # limit_list_d 数据起点, 也是页面默认可回看范围
NEWHIGH_WINDOW = 20            # N日新高窗口


def _conn():
    return sqlite_connect(DB_PATH)


def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sector_daily_stats (
            taxonomy TEXT NOT NULL,        -- sw_l1 / sw_l2 / concept
            trade_date TEXT NOT NULL,      -- YYYY-MM-DD
            sector_code TEXT NOT NULL,
            sector_name TEXT NOT NULL,
            pct_change REAL,               -- 板块涨幅 %
            main_net_inflow REAL,          -- 主力净流入 亿元 (大单+超大单)
            newhigh_cnt INTEGER,           -- 20日新高成分股数
            up_cnt INTEGER,
            down_cnt INTEGER,
            stock_cnt INTEGER,
            PRIMARY KEY (taxonomy, trade_date, sector_code)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sds_tax_date ON sector_daily_stats(taxonomy, trade_date)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_sentiment_daily (
            trade_date TEXT PRIMARY KEY,   -- YYYY-MM-DD
            limit_up_cnt INTEGER,
            limit_down_cnt INTEGER,
            broken_cnt INTEGER,            -- 炸板数
            up_cnt INTEGER,
            down_cnt INTEGER,
            hs300_newhigh20 INTEGER,
            zz2000_newhigh20 INTEGER
        )
    """)
    conn.commit()


def _dash(d8: str) -> str:
    return f"{d8[:4]}-{d8[4:6]}-{d8[6:]}"


def load_stock_days(conn, start: str, end: str) -> pd.DataFrame:
    """加载 [start, end] 全 A 股日行情 + 20日新高标记 + 个股主力净流入.

    返回列: code6, trade_date(YYYY-MM-DD), pct(%), is_newhigh, main_net(亿), is_up, is_down
    """
    # 多取 ~NEWHIGH_WINDOW*2 个自然日做 rolling 缓冲
    buf_start = (pd.Timestamp(start) - pd.Timedelta(days=NEWHIGH_WINDOW * 2 + 10)).strftime('%Y-%m-%d')
    q = """
        SELECT s.code AS code6, dq.trade_date, dq.close,
               COALESCE(dq.adj_close, dq.close) AS px,
               dq.price_change_pct AS pct
        FROM daily_quotes dq
        JOIN securities s ON s.id = dq.security_id AND s.type = 'A股'
        WHERE dq.trade_date BETWEEN ? AND ?
    """
    df = pd.read_sql_query(q, conn, params=(buf_start, end))
    logger.info("加载行情 %s ~ %s: %d 行", buf_start, end, len(df))
    df = df.sort_values(['code6', 'trade_date']).reset_index(drop=True)
    # 20日新高: 收盘价 > 前19日最高收盘 (复权价); 原生 groupby.rolling 避免逐组 lambda
    shifted = df.groupby('code6')['px'].shift(1)
    prev_max = (shifted.groupby(df['code6'])
                       .rolling(NEWHIGH_WINDOW - 1, min_periods=NEWHIGH_WINDOW - 1)
                       .max().reset_index(level=0, drop=True))
    df['is_newhigh'] = (df['px'] > prev_max).fillna(False)
    df = df[df['trade_date'] >= start].copy()
    df['pct'] = df['pct'] * 100  # 小数 → 百分比
    df['is_up'] = df['pct'] > 0
    df['is_down'] = df['pct'] < 0

    # code_6 列近期未回填 (NULL), 从 code ('000001.SZ') 截取
    mf = pd.read_sql_query(
        """SELECT substr(code, 1, 6) AS code6, trade_date,
                  (COALESCE(buy_lg_amount,0) + COALESCE(buy_elg_amount,0)
                   - COALESCE(sell_lg_amount,0) - COALESCE(sell_elg_amount,0)) / 10000.0 AS main_net
           FROM moneyflow_daily WHERE trade_date BETWEEN ? AND ?""",
        conn, params=(start, end))
    df = df.merge(mf, on=['code6', 'trade_date'], how='left')
    df['main_net'] = df['main_net'].fillna(0.0)
    return df[['code6', 'trade_date', 'pct', 'is_newhigh', 'main_net', 'is_up', 'is_down']]


def _agg_sectors(members: pd.DataFrame, stock_days: pd.DataFrame,
                 taxonomy: str) -> pd.DataFrame:
    """成分表 (code6, sector_code, sector_name[, in_date, out_date]) × 个股日频 → 板块日频"""
    m = stock_days.merge(members, on='code6', how='inner')
    if 'in_date' in m.columns:  # 申万点时成分过滤
        d8 = m['trade_date'].str.replace('-', '', regex=False)
        m = m[(m['in_date'].isna() | (m['in_date'] <= d8))
              & (m['out_date'].isna() | (m['out_date'] > d8))]
    g = m.groupby(['trade_date', 'sector_code', 'sector_name']).agg(
        pct_change=('pct', 'mean'),
        main_net_inflow=('main_net', 'sum'),
        newhigh_cnt=('is_newhigh', 'sum'),
        up_cnt=('is_up', 'sum'),
        down_cnt=('is_down', 'sum'),
        stock_cnt=('pct', 'size'),
    ).reset_index()
    g['taxonomy'] = taxonomy
    return g


def load_static_tables(conn):
    """成分/快照表 (与日期无关, 全量加载一次供各月分块复用)"""
    sw = pd.read_sql_query(
        """SELECT code AS code6, l1_code, l1_name, l2_code, l2_name, in_date, out_date
           FROM sw_industry""", conn)
    dc = pd.read_sql_query(
        "SELECT ts_code AS sector_code, name AS sector_name, con_code FROM dc_member", conn)
    if len(dc):
        dc['code6'] = dc['con_code'].str[:6]
    snap = pd.read_sql_query(
        "SELECT index_code, con_code, trade_date FROM index_weight_snapshot", conn)
    snap['code6'] = snap['con_code'].str[:6]
    return sw, dc, snap


def build_sector_stats(conn, start: str, end: str, stock_days: pd.DataFrame,
                       sw: pd.DataFrame, dc: pd.DataFrame):
    frames = []

    l1 = sw.rename(columns={'l1_code': 'sector_code', 'l1_name': 'sector_name'})
    frames.append(_agg_sectors(l1[['code6', 'sector_code', 'sector_name', 'in_date', 'out_date']],
                               stock_days, 'sw_l1'))

    l2 = sw[sw['l2_code'].notna()].rename(columns={'l2_code': 'sector_code', 'l2_name': 'sector_name'})
    if len(l2):
        frames.append(_agg_sectors(l2[['code6', 'sector_code', 'sector_name', 'in_date', 'out_date']],
                                   stock_days, 'sw_l2'))
    else:
        logger.warning("sw_industry 无 L2 数据 (回填未完成?), 跳过 sw_l2")

    if len(dc):
        cg = _agg_sectors(dc[['code6', 'sector_code', 'sector_name']], stock_days, 'concept')
        # 概念涨幅优先用东财官方板块指数值
        idx = pd.read_sql_query(
            """SELECT ts_code AS sector_code, trade_date, pct_change AS dc_pct,
                      up_num, down_num
               FROM dc_index_daily WHERE idx_type='概念板块' AND trade_date BETWEEN ? AND ?""",
            conn, params=(start.replace('-', ''), end.replace('-', '')))
        idx['trade_date'] = idx['trade_date'].map(_dash)
        cg = cg.merge(idx, on=['sector_code', 'trade_date'], how='left')
        for ours, official in [('pct_change', 'dc_pct'), ('up_cnt', 'up_num'), ('down_cnt', 'down_num')]:
            cg[ours] = cg[official].fillna(cg[ours])
        frames.append(cg[['trade_date', 'sector_code', 'sector_name', 'pct_change',
                          'main_net_inflow', 'newhigh_cnt', 'up_cnt', 'down_cnt',
                          'stock_cnt', 'taxonomy']])
    else:
        logger.warning("dc_member 为空 (回填未完成?), 跳过 concept")

    out = pd.concat(frames, ignore_index=True)
    rows = [(r.taxonomy, r.trade_date, r.sector_code, r.sector_name,
             round(r.pct_change, 3) if pd.notna(r.pct_change) else None,
             round(r.main_net_inflow, 3), int(r.newhigh_cnt),
             int(r.up_cnt), int(r.down_cnt), int(r.stock_cnt))
            for r in out.itertuples(index=False)]
    _write_retry(lambda: (conn.executemany(
        """INSERT OR REPLACE INTO sector_daily_stats
           (taxonomy, trade_date, sector_code, sector_name, pct_change,
            main_net_inflow, newhigh_cnt, up_cnt, down_cnt, stock_cnt)
           VALUES (?,?,?,?,?,?,?,?,?,?)""", rows), conn.commit()))
    logger.info("✅ sector_daily_stats: 写入 %d 行 (%s ~ %s)", len(rows), start, end)


def build_sentiment(conn, start: str, end: str, stock_days: pd.DataFrame,
                    snap: pd.DataFrame):
    dates = sorted(stock_days['trade_date'].unique())

    # 涨跌停/炸板 (limit_list_daily, YYYYMMDD)
    ll = pd.read_sql_query(
        """SELECT trade_date, limit_type, COUNT(*) AS n FROM limit_list_daily
           WHERE trade_date BETWEEN ? AND ? GROUP BY trade_date, limit_type""",
        conn, params=(start.replace('-', ''), end.replace('-', '')))
    ll['trade_date'] = ll['trade_date'].map(_dash)
    lim = (ll.pivot(index='trade_date', columns='limit_type', values='n')
             .reindex(index=dates, columns=['U', 'D', 'Z'], fill_value=0).fillna(0))

    # 全市场涨跌家数
    ud = stock_days.groupby('trade_date').agg(
        up_cnt=('is_up', 'sum'), down_cnt=('is_down', 'sum')).reindex(dates, fill_value=0)

    # 指数成分 20 日新高: 每日用 ≤ 当日的最近一期成分快照
    nh_by_date = (stock_days[stock_days['is_newhigh']]
                  .groupby('trade_date')['code6'].agg(set).to_dict())
    idx_nh = {}
    for idx_code, col in [('000300.SH', 'hs300_newhigh20'), ('932000.CSI', 'zz2000_newhigh20')]:
        s = snap[snap['index_code'] == idx_code]
        snap_dates = sorted(s['trade_date'].unique())  # YYYYMMDD
        if not snap_dates:
            logger.warning("index_weight_snapshot 无 %s 数据, 该列置 0", idx_code)
            idx_nh[col] = {}
            continue
        members_by_snap = {d: set(g) for d, g in s.groupby('trade_date')['code6']}
        counts = {}
        for d in dates:
            d8 = d.replace('-', '')
            eff = max((sd for sd in snap_dates if sd <= d8), default=snap_dates[0])
            counts[d] = len(nh_by_date.get(d, set()) & members_by_snap[eff])
        idx_nh[col] = counts

    rows = [(d, int(lim.at[d, 'U']), int(lim.at[d, 'D']), int(lim.at[d, 'Z']),
             int(ud.at[d, 'up_cnt']), int(ud.at[d, 'down_cnt']),
             idx_nh['hs300_newhigh20'].get(d, 0), idx_nh['zz2000_newhigh20'].get(d, 0))
            for d in dates]
    _write_retry(lambda: (conn.executemany(
        """INSERT OR REPLACE INTO market_sentiment_daily
           (trade_date, limit_up_cnt, limit_down_cnt, broken_cnt,
            up_cnt, down_cnt, hs300_newhigh20, zz2000_newhigh20)
           VALUES (?,?,?,?,?,?,?,?)""", rows), conn.commit()))
    logger.info("✅ market_sentiment_daily: 写入 %d 天", len(rows))


def main():
    ap = argparse.ArgumentParser(description='市场脉搏预计算')
    ap.add_argument('--start-date', help='YYYY-MM-DD')
    ap.add_argument('--end-date', help='YYYY-MM-DD')
    ap.add_argument('--rebuild', action='store_true', help='从默认起点全量重算')
    args = ap.parse_args()

    conn = _conn()
    _write_retry(lambda: _ensure_tables(conn))

    end = args.end_date or conn.execute(
        "SELECT MAX(trade_date) FROM daily_quotes").fetchone()[0]
    if args.start_date:
        start = args.start_date
    elif args.rebuild:
        start = DEFAULT_START
    else:  # 增量: 已算最新日 (重算最后一天, 容忍当日数据晚到)
        last = conn.execute("SELECT MAX(trade_date) FROM sector_daily_stats").fetchone()[0]
        start = last or DEFAULT_START
    if start > end:
        logger.info("无需更新 (start %s > end %s)", start, end)
        return

    logger.info("计算范围: %s ~ %s", start, end)
    sw, dc, snap = load_static_tables(conn)
    # 按月分块, 控制内存
    months = pd.period_range(start[:7], end[:7], freq='M')
    for m in months:
        m_start = max(start, f"{m}-01")
        m_end = min(end, str(m.end_time.date()))
        stock_days = load_stock_days(conn, m_start, m_end)
        if stock_days.empty:
            continue
        build_sector_stats(conn, m_start, m_end, stock_days, sw, dc)
        build_sentiment(conn, m_start, m_end, stock_days, snap)
    conn.close()
    logger.info("🎉 完成")


if __name__ == '__main__':
    main()
