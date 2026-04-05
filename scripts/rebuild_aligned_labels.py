#!/usr/bin/env python3
"""批量重建 feature cache 标签 — 对齐回测执行定义

旧标签: label_Nd = close[T+N] / close[T] - 1  (close-to-close, 报告日起算)
新标签: label_Nd = close[T+1+N] / open[T+1] - 1  (open-to-close, 买入日起算)

算法:
  1. 一次性加载全部 daily_quotes (open, close, volume) 到内存 (~2GB)
  2. 按股票分组, 构建 {code: {trade_date: (open, close, volume)}} 查找字典
  3. 遍历 cache 表, 用查找字典高速计算新标签
  4. 批量 UPDATE (每 5000 条 commit 一次)

预计耗时: v39 ~5-8分钟, v40 ~3-5分钟, alpha158 ~4-6分钟
"""

import sqlite3
import time
import sys
import math
import argparse
from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parent.parent / 'data_adapter' / 'stock_data.db')


def load_quotes_lookup(conn):
    """一次性加载全部行情到内存, 构建 {code: [(date, open, close, volume), ...]} 有序列表"""
    print("Phase 1: 加载全部 daily_quotes 到内存...")
    t0 = time.time()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.code, q.trade_date, q.open, q.close, q.volume
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.type IN ('A股', 'ETF_基金')
        ORDER BY s.code, q.trade_date
    """)

    stock_data = {}  # code -> [(date, open, close, volume), ...]
    count = 0
    for code, trade_date, open_p, close_p, volume in cursor:
        if code not in stock_data:
            stock_data[code] = []
        stock_data[code].append((trade_date, open_p, close_p, volume))
        count += 1

    elapsed = time.time() - t0
    print(f"  加载 {count:,} 条行情, {len(stock_data):,} 只股票, 耗时 {elapsed:.1f}秒")
    return stock_data


def build_date_index(records):
    """构建 date -> index 映射"""
    return {r[0]: i for i, r in enumerate(records)}


def compute_aligned_labels_v39(records, date_idx, date):
    """计算对齐回测的标签 (v39/alpha158: 无超额收益)

    label_Nd = close[buy+N] / open[buy] - 1
    buy = date 的下一个交易日 (index + 1)
    """
    pos = date_idx.get(date)
    if pos is None:
        return None, None, None, None

    n = len(records)

    # 报告日停牌检测
    if records[pos][3] == 0:  # volume
        return None, None, None, None

    # 买入日 = pos + 1
    buy_pos = pos + 1
    if buy_pos >= n:
        return None, None, None, None

    buy_open = records[buy_pos][1]   # open
    buy_vol = records[buy_pos][3]    # volume

    if not buy_open or buy_open <= 0:
        return None, None, None, None
    if buy_vol == 0:
        return None, None, None, None

    label_3d = label_5d = label_10d = label_15d = None

    # label_Nd = close[buy_pos + N] / buy_open - 1
    if buy_pos + 3 < n:
        c = records[buy_pos + 3][2]  # close
        if c and c > 0:
            label_3d = c / buy_open - 1
    if buy_pos + 5 < n:
        c = records[buy_pos + 5][2]
        if c and c > 0:
            label_5d = c / buy_open - 1
    if buy_pos + 10 < n:
        c = records[buy_pos + 10][2]
        if c and c > 0:
            label_10d = c / buy_open - 1
    if buy_pos + 15 < n:
        c = records[buy_pos + 15][2]
        if c and c > 0:
            label_15d = c / buy_open - 1

    return label_3d, label_5d, label_10d, label_15d


def rebuild_v39_labels(conn, stock_data):
    """重建 v39_feature_cache 标签"""
    print("\n" + "=" * 70)
    print("重建 v39_feature_cache 标签")
    print("=" * 70)

    cursor = conn.cursor()

    # 读取所有需要更新的记录
    print("  读取缓存记录...")
    t0 = time.time()
    cursor.execute("SELECT id, code, trade_date FROM v39_feature_cache ORDER BY code, trade_date")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"  共 {total:,} 条记录, 耗时 {time.time()-t0:.1f}秒")

    # 预构建每只股票的 date_index
    date_indices = {}
    for code, records in stock_data.items():
        date_indices[code] = build_date_index(records)

    # 批量计算并更新
    print("  计算新标签并批量更新...")
    t0 = time.time()
    updated = 0
    skipped = 0
    batch = []
    batch_size = 5000

    for i, (cache_id, code, trade_date) in enumerate(rows):
        records = stock_data.get(code)
        if not records:
            skipped += 1
            continue

        date_idx = date_indices.get(code)
        if not date_idx:
            skipped += 1
            continue

        l3, l5, l10, l15 = compute_aligned_labels_v39(records, date_idx, trade_date)
        batch.append((l3, l5, l10, l15, cache_id))

        if len(batch) >= batch_size:
            cursor.executemany("""
                UPDATE v39_feature_cache
                SET label_3d = ?, label_5d = ?, label_10d = ?, label_15d = ?
                WHERE id = ?
            """, batch)
            conn.commit()
            updated += len(batch)
            batch = []

            if (updated % 50000) == 0:
                elapsed = time.time() - t0
                pct = updated / total * 100
                speed = updated / elapsed
                eta = (total - updated) / speed if speed > 0 else 0
                print(f"    进度: {updated:,}/{total:,} ({pct:.1f}%), "
                      f"速度: {speed:,.0f}/秒, ETA: {eta:.0f}秒")

    # 最后一批
    if batch:
        cursor.executemany("""
            UPDATE v39_feature_cache
            SET label_3d = ?, label_5d = ?, label_10d = ?, label_15d = ?
            WHERE id = ?
        """, batch)
        conn.commit()
        updated += len(batch)

    elapsed = time.time() - t0
    print(f"  ✅ v39 完成: {updated:,} 条更新, {skipped:,} 条跳过, 耗时 {elapsed:.1f}秒")
    return updated


def rebuild_v40_labels(conn, stock_data):
    """重建 v40_feature_cache 超额收益标签"""
    print("\n" + "=" * 70)
    print("重建 v40_feature_cache 超额收益标签")
    print("=" * 70)

    cursor = conn.cursor()

    # 加载沪深300数据
    cursor.execute("""
        SELECT q.trade_date, q.close
        FROM daily_quotes q
        JOIN securities s ON q.security_id = s.id
        WHERE s.code = '000300.SH' OR s.name = '沪深300'
        ORDER BY q.trade_date
    """)
    hs300_rows = cursor.fetchall()
    hs300_idx = {r[0]: i for i, r in enumerate(hs300_rows)}
    hs300_closes = [r[1] for r in hs300_rows]
    print(f"  沪深300: {len(hs300_rows):,} 条 ({hs300_rows[0][0]} ~ {hs300_rows[-1][0]})")

    # 读取缓存记录
    print("  读取缓存记录...")
    t0 = time.time()
    cursor.execute("SELECT id, code, trade_date FROM v40_feature_cache ORDER BY code, trade_date")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"  共 {total:,} 条记录, 耗时 {time.time()-t0:.1f}秒")

    # 预构建 date_index
    date_indices = {}
    for code, records in stock_data.items():
        date_indices[code] = build_date_index(records)

    # 批量计算
    print("  计算新超额收益标签...")
    t0 = time.time()
    updated = 0
    skipped = 0
    batch = []
    batch_size = 5000

    for i, (cache_id, code, trade_date) in enumerate(rows):
        records = stock_data.get(code)
        if not records:
            skipped += 1
            continue

        date_idx = date_indices.get(code)
        if not date_idx:
            skipped += 1
            continue

        pos = date_idx.get(trade_date)
        if pos is None:
            skipped += 1
            continue

        n = len(records)

        # 报告日停牌
        if records[pos][3] == 0:
            batch.append((None, None, None, cache_id))
            continue

        buy_pos = pos + 1
        if buy_pos >= n:
            batch.append((None, None, None, cache_id))
            continue

        buy_open = records[buy_pos][1]
        buy_vol = records[buy_pos][3]

        if not buy_open or buy_open <= 0 or buy_vol == 0:
            batch.append((None, None, None, cache_id))
            continue

        # 沪深300基准: 同持仓期 close[T+1] → close[T+1+N]
        hs_pos = hs300_idx.get(trade_date)
        if hs_pos is None:
            batch.append((None, None, None, cache_id))
            continue

        hs_buy_pos = hs_pos + 1  # 基准也从 T+1 起算
        if hs_buy_pos >= len(hs300_closes):
            batch.append((None, None, None, cache_id))
            continue

        hs_base = hs300_closes[hs_buy_pos]
        if not hs_base or hs_base <= 0:
            batch.append((None, None, None, cache_id))
            continue

        l3 = l5 = l10 = None
        for nd, setter in [(3, 'l3'), (5, 'l5'), (10, 'l10')]:
            if buy_pos + nd < n and hs_buy_pos + nd < len(hs300_closes):
                c = records[buy_pos + nd][2]  # close
                hc = hs300_closes[hs_buy_pos + nd]
                if c and c > 0 and hc and hc > 0:
                    stock_ret = c / buy_open - 1
                    mkt_ret = hc / hs_base - 1
                    val = stock_ret - mkt_ret
                    if nd == 3: l3 = val
                    elif nd == 5: l5 = val
                    elif nd == 10: l10 = val

        batch.append((l3, l5, l10, cache_id))

        if len(batch) >= batch_size:
            cursor.executemany("""
                UPDATE v40_feature_cache
                SET label_3d_excess = ?, label_5d_excess = ?, label_10d_excess = ?
                WHERE id = ?
            """, batch)
            conn.commit()
            updated += len(batch)
            batch = []

            if (updated % 50000) == 0:
                elapsed = time.time() - t0
                pct = updated / total * 100
                speed = updated / elapsed
                eta = (total - updated) / speed if speed > 0 else 0
                print(f"    进度: {updated:,}/{total:,} ({pct:.1f}%), "
                      f"速度: {speed:,.0f}/秒, ETA: {eta:.0f}秒")

    if batch:
        cursor.executemany("""
            UPDATE v40_feature_cache
            SET label_3d_excess = ?, label_5d_excess = ?, label_10d_excess = ?
            WHERE id = ?
        """, batch)
        conn.commit()
        updated += len(batch)

    elapsed = time.time() - t0
    print(f"  ✅ v40 完成: {updated:,} 条更新, {skipped:,} 条跳过, 耗时 {elapsed:.1f}秒")
    return updated


def rebuild_alpha158_labels(conn, stock_data):
    """重建 alpha158_feature_cache 标签 (log-return)"""
    print("\n" + "=" * 70)
    print("重建 alpha158_feature_cache 标签")
    print("=" * 70)

    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='alpha158_feature_cache'")
    if cursor.fetchone()[0] == 0:
        print("  ⚠️ alpha158_feature_cache 表不存在, 跳过")
        return 0

    print("  读取缓存记录...")
    t0 = time.time()
    cursor.execute("SELECT rowid, code, trade_date FROM alpha158_feature_cache ORDER BY code, trade_date")
    rows = cursor.fetchall()
    total = len(rows)
    print(f"  共 {total:,} 条记录, 耗时 {time.time()-t0:.1f}秒")

    if total == 0:
        print("  无记录, 跳过")
        return 0

    # 预构建 date_index
    date_indices = {}
    for code, records in stock_data.items():
        date_indices[code] = build_date_index(records)

    print("  计算新标签 (log-return)...")
    t0 = time.time()
    updated = 0
    skipped = 0
    batch = []
    batch_size = 5000

    for i, (row_id, code, trade_date) in enumerate(rows):
        records = stock_data.get(code)
        if not records:
            skipped += 1
            continue

        date_idx = date_indices.get(code)
        if not date_idx:
            skipped += 1
            continue

        pos = date_idx.get(trade_date)
        if pos is None:
            skipped += 1
            continue

        n = len(records)

        # 报告日停牌
        if records[pos][3] == 0:
            batch.append((None, None, None, row_id))
            continue

        buy_pos = pos + 1
        if buy_pos >= n:
            batch.append((None, None, None, row_id))
            continue

        buy_open = records[buy_pos][1]
        buy_vol = records[buy_pos][3]

        if not buy_open or buy_open <= 0 or buy_vol == 0:
            batch.append((None, None, None, row_id))
            continue

        l3 = l5 = l10 = None
        # alpha158 用 log-return
        if buy_pos + 3 < n:
            c = records[buy_pos + 3][2]
            if c and c > 0:
                l3 = math.log(c / buy_open)
        if buy_pos + 5 < n:
            c = records[buy_pos + 5][2]
            if c and c > 0:
                l5 = math.log(c / buy_open)
        if buy_pos + 10 < n:
            c = records[buy_pos + 10][2]
            if c and c > 0:
                l10 = math.log(c / buy_open)

        batch.append((l3, l5, l10, row_id))

        if len(batch) >= batch_size:
            cursor.executemany("""
                UPDATE alpha158_feature_cache
                SET label_3d = ?, label_5d = ?, label_10d = ?
                WHERE rowid = ?
            """, batch)
            conn.commit()
            updated += len(batch)
            batch = []

            if (updated % 50000) == 0:
                elapsed = time.time() - t0
                pct = updated / total * 100
                speed = updated / elapsed
                eta = (total - updated) / speed if speed > 0 else 0
                print(f"    进度: {updated:,}/{total:,} ({pct:.1f}%), "
                      f"速度: {speed:,.0f}/秒, ETA: {eta:.0f}秒")

    if batch:
        cursor.executemany("""
            UPDATE alpha158_feature_cache
            SET label_3d = ?, label_5d = ?, label_10d = ?
            WHERE rowid = ?
        """, batch)
        conn.commit()
        updated += len(batch)

    elapsed = time.time() - t0
    print(f"  ✅ alpha158 完成: {updated:,} 条更新, {skipped:,} 条跳过, 耗时 {elapsed:.1f}秒")
    return updated


def verify_labels(conn):
    """验证标签重建结果"""
    print("\n" + "=" * 70)
    print("标签验证")
    print("=" * 70)

    cursor = conn.cursor()

    # v39: 抽样对比新旧标签与回测定义
    cursor.execute("""
        SELECT fc.code, fc.trade_date, fc.label_10d,
               dq_buy.open as buy_open,
               dq_sell.close as sell_close
        FROM v39_feature_cache fc
        JOIN securities s ON fc.code = s.code
        JOIN daily_quotes dq_buy ON dq_buy.security_id = s.id
        JOIN daily_quotes dq_sell ON dq_sell.security_id = s.id
        WHERE fc.label_10d IS NOT NULL
          AND fc.trade_date >= '2024-01-01'
        LIMIT 5
    """)
    # Simple verification: check a few samples
    cursor.execute("""
        SELECT code, trade_date, label_3d, label_5d, label_10d, label_15d
        FROM v39_feature_cache
        WHERE label_10d IS NOT NULL AND trade_date >= '2025-01-01'
        ORDER BY RANDOM()
        LIMIT 10
    """)
    samples = cursor.fetchall()

    print("\n  v39 抽样验证 (10条):")
    print(f"  {'Code':<12} {'Date':<12} {'3d':>8} {'5d':>8} {'10d':>8} {'15d':>8}")
    for code, date, l3, l5, l10, l15 in samples:
        l3s = f"{l3*100:+.2f}%" if l3 else "NULL"
        l5s = f"{l5*100:+.2f}%" if l5 else "NULL"
        l10s = f"{l10*100:+.2f}%" if l10 else "NULL"
        l15s = f"{l15*100:+.2f}%" if l15 else "NULL"
        print(f"  {code:<12} {date:<12} {l3s:>8} {l5s:>8} {l10s:>8} {l15s:>8}")

    # 统计
    for table, label_col in [
        ('v39_feature_cache', 'label_10d'),
        ('v40_feature_cache', 'label_10d_excess'),
    ]:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {label_col} IS NOT NULL")
            labeled = cursor.fetchone()[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            total = cursor.fetchone()[0]
            cursor.execute(f"SELECT AVG({label_col}), STDEV({label_col}) FROM {table} WHERE {label_col} IS NOT NULL")
            row = cursor.fetchone()
            avg_val = row[0] if row[0] else 0
            print(f"\n  {table}: {labeled:,}/{total:,} 有标签, 均值={avg_val*100:+.3f}%")
        except Exception as e:
            print(f"\n  {table}: 跳过 ({e})")

    # 与回测定义交叉验证: 选一只股票, 手动计算
    print("\n  交叉验证 (手动计算 vs 标签):")
    cursor.execute("""
        SELECT fc.code, fc.trade_date, fc.label_10d
        FROM v39_feature_cache fc
        WHERE fc.label_10d IS NOT NULL AND fc.trade_date >= '2025-06-01'
        ORDER BY fc.trade_date DESC
        LIMIT 3
    """)
    for code, date, label_val in cursor.fetchall():
        # 手动从 daily_quotes 计算
        cursor.execute("""
            SELECT q.trade_date, q.open, q.close
            FROM daily_quotes q JOIN securities s ON q.security_id = s.id
            WHERE s.code = ? AND q.trade_date >= ?
            ORDER BY q.trade_date LIMIT 13
        """, (code, date))
        rows = cursor.fetchall()
        if len(rows) > 11:
            buy_open = rows[1][1]  # T+1 open
            sell_close = rows[11][2]  # T+1+10 close
            manual = sell_close / buy_open - 1
            diff = abs(label_val - manual)
            match = "✅" if diff < 1e-8 else f"❌ diff={diff:.10f}"
            print(f"  {code} {date}: label={label_val:+.6f}, manual={manual:+.6f} {match}")
        else:
            print(f"  {code} {date}: 数据不足, 跳过验证")


def main():
    parser = argparse.ArgumentParser(description='重建对齐回测的标签')
    parser.add_argument('--v39-only', action='store_true', help='仅重建v39')
    parser.add_argument('--v40-only', action='store_true', help='仅重建v40')
    parser.add_argument('--alpha158-only', action='store_true', help='仅重建alpha158')
    parser.add_argument('--verify-only', action='store_true', help='仅验证, 不重建')
    parser.add_argument('--db', default=DB_PATH, help='数据库路径')
    args = parser.parse_args()

    do_all = not (args.v39_only or args.v40_only or args.alpha158_only)

    print("=" * 70)
    print("标签重建: 对齐回测执行定义")
    print("  旧: close[T] → close[T+N]  (close-to-close)")
    print("  新: open[T+1] → close[T+1+N]  (与回测买卖一致)")
    print("=" * 70)

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")

    if args.verify_only:
        verify_labels(conn)
        conn.close()
        return

    # Phase 1: 加载行情
    stock_data = load_quotes_lookup(conn)

    total_updated = 0
    t_total = time.time()

    # Phase 2: 重建各表
    if do_all or args.v39_only:
        total_updated += rebuild_v39_labels(conn, stock_data)

    if do_all or args.v40_only:
        total_updated += rebuild_v40_labels(conn, stock_data)

    if do_all or args.alpha158_only:
        total_updated += rebuild_alpha158_labels(conn, stock_data)

    # Phase 3: 验证
    verify_labels(conn)

    conn.close()

    elapsed = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"全部完成! 共更新 {total_updated:,} 条标签, 总耗时 {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
