#!/bin/bash
# 2026-07-11 重构 session 接力链: 等财报补数结束 → VACUUM → ng101 缓存重建 pre-flight → 全量重建
# 用法: nohup bash scripts/post_backfill_chain_20260711.sh > logs/post_backfill_chain_20260711.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[chain] 等待 backfill_financial_indicator 退出..."
while pgrep -f "backfill_financial_indicator" > /dev/null; do sleep 60; done
echo "[chain] 补数已结束 $(date)"

python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db'); c.execute('PRAGMA busy_timeout=60000')
print('rows:', c.execute('SELECT COUNT(*) FROM financial_indicator').fetchone()[0])
print('max ann_date:', c.execute('SELECT MAX(ann_date) FROM financial_indicator').fetchone()[0])
print('typeof:', dict(c.execute('SELECT typeof(ann_date), COUNT(*) FROM financial_indicator GROUP BY 1').fetchall()))
"

echo "[chain] VACUUM 开始 (释放 5 张已删缓存表空间, 预计 30-90min)..."
python3 -c "
import sqlite3, time
t0=time.time()
c = sqlite3.connect('data_adapter/stock_data.db'); c.execute('PRAGMA busy_timeout=120000')
c.execute('VACUUM')
print(f'VACUUM done in {(time.time()-t0)/60:.0f}min')
"
df -h / | awk 'NR==2{print "[chain] VACUUM 后磁盘可用:",$4}'
ls -lh data_adapter/stock_data.db | awk '{print "[chain] DB 大小:",$5}'

echo "[chain] Pre-flight: ng101 缓存单日重建试跑 (2026-07-10)..."
python3 ml_models/ng/ng_cache_updater.py --start-date 2026-07-10 --end-date 2026-07-10 --version ng1.0.1
N=$(python3 -c "
import sqlite3
c = sqlite3.connect('data_adapter/stock_data.db'); c.execute('PRAGMA busy_timeout=60000')
print(c.execute(\"SELECT COUNT(*) FROM ng101_feature_cache WHERE trade_date='2026-07-10'\").fetchone()[0])
")
echo "[chain] pre-flight 行数: $N (预期 ~3000)"
if [ "$N" -lt 2500 ]; then
  echo "[chain] ❌ pre-flight 行数不足, 中止全量重建 — 人工排查"
  exit 1
fi

echo "[chain] ✅ pre-flight 通过, kickoff 全量重建 2018-01-01 → 2026-07-10 (caffeinate 防睡眠)..."
caffeinate -i python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2018-01-01 --end-date 2026-07-10 --version ng1.0.1 \
  2>&1 | tee logs/ng101_cache_rebuild_20260711.log
echo "[chain] 全量重建结束 $(date) — 完成后需重训评估 (见 docs/code_review_and_refactor_plan.md 第六节 P0-5)"
