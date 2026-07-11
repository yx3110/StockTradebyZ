#!/usr/bin/env python3
"""Backfill daily_quotes.amount for 北交所 stocks using pro.daily(ts_code='XXXXXX.BJ').

Strategy: iterate over each 北交所 stock code, one API call per stock covering full
history. Upsert amount by (security_id, trade_date).
Checkpoint-able: skip stocks already fully populated.
"""
import sqlite3, json, logging, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data_adapter" / "stock_data.db"
CONFIG_PATH = PROJECT_ROOT / "config.json"
CHECKPOINT = PROJECT_ROOT / "scripts" / "backfill" / "bj_amount_checkpoint.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("amount_fill_bj")


def get_pending_stocks(conn):
    """北交所 A股 with any NULL amount row."""
    q = """
        SELECT DISTINCT s.code FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股' AND dq.amount IS NULL
          AND (s.code LIKE '43%' OR s.code LIKE '83%' OR s.code LIKE '87%' OR s.code LIKE '88%' OR s.code LIKE '92%')
        ORDER BY s.code
    """
    return [r[0] for r in conn.execute(q).fetchall()]


def load_checkpoint():
    if not CHECKPOINT.exists():
        return set()
    return set(CHECKPOINT.read_text().strip().split("\n")) if CHECKPOINT.stat().st_size > 0 else set()


def save_checkpoint(done: set):
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text("\n".join(sorted(done)))


def backfill_one_stock(pro, conn, code: str) -> tuple:
    ts_code = f"{code}.BJ"
    # pro.daily() does NOT return 北交所; use stk_factor which covers BJ exchange
    df = pro.stk_factor(ts_code=ts_code, fields="ts_code,trade_date,amount")
    if df is None or len(df) == 0:
        return 0, "tushare returned empty"
    # Tushare format YYYYMMDD -> YYYY-MM-DD for DB
    df["trade_date"] = df["trade_date"].apply(lambda d: f"{d[:4]}-{d[4:6]}-{d[6:8]}")
    rows = [(r["amount"], code, r["trade_date"]) for _, r in df.iterrows() if r["amount"] is not None]

    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        cur.executemany("""
            UPDATE daily_quotes
            SET amount = ?
            WHERE security_id = (SELECT id FROM securities WHERE code = ? AND type = 'A股')
              AND trade_date = ?
              AND amount IS NULL
        """, rows)
        updated = cur.rowcount
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return updated, f"tushare_rows={len(df)}, updated={updated}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-stocks", type=int)
    args = ap.parse_args()

    import tushare as ts
    # token 优先从 core.config/.env 获取
    try:
        from core.config import get_tushare_token
        token = get_tushare_token()
    except ImportError:
        token = json.loads(CONFIG_PATH.read_text())["tushare"]["token"]
    ts.set_token(token)
    pro = ts.pro_api()

    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.execute("PRAGMA busy_timeout = 30000")

    stocks = get_pending_stocks(conn)
    logger.info(f"北交所 pending stocks: {len(stocks)}")
    done = load_checkpoint()
    stocks = [s for s in stocks if s not in done]
    if args.max_stocks:
        stocks = stocks[:args.max_stocks]

    if args.dry_run:
        if not stocks:
            logger.info("nothing to do")
            return 0
        s = stocks[0]
        t0 = time.time()
        updated, info = backfill_one_stock(pro, conn, s)
        logger.info(f"DRY-RUN {s}: {info} in {time.time()-t0:.1f}s")
        return 0

    total_updated = 0
    t0 = time.time()
    for i, s in enumerate(stocks, start=1):
        try:
            updated, info = backfill_one_stock(pro, conn, s)
            total_updated += updated
            done.add(s)
            if i % 20 == 0:
                save_checkpoint(done)
                el = time.time() - t0
                rate = i / el if el > 0 else 0
                eta = (len(stocks) - i) / rate if rate > 0 else 0
                logger.info(
                    f"[{i}/{len(stocks)}] {s}: {info} | el {el/60:.1f}min ETA {eta/60:.1f}min, "
                    f"total_upd={total_updated:,}"
                )
            else:
                logger.info(f"[{i}/{len(stocks)}] {s}: {info}")
            time.sleep(0.15)
        except Exception as e:
            logger.error(f"{s} FAILED: {e}")
            save_checkpoint(done)
            time.sleep(3)
            try:
                updated, info = backfill_one_stock(pro, conn, s)
                total_updated += updated
                done.add(s)
                logger.info(f"  retry OK: {s}: {info}")
            except Exception as e2:
                logger.error(f"  retry failed: {e2}")

    save_checkpoint(done)
    logger.info(f"DONE. {total_updated:,} rows updated in {(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
