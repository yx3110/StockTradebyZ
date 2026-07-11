#!/bin/zsh
# 新口径基线重跑 (2026-07-11 北极星 P0 修复后, Check 5 公平对比)
# 口径 tag: 2026-07-11-p0fix (基准回填/惩罚拆三/PSR/L8/L9修复/复权pct回填)
# 统一参数: --top-n 10 --focus-days 10 --rank-field composite (与历史 headline 对齐)
# 前置: backfill_adj_factor.py + backfill_price_change_pct.py 已完成
set -e
cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M)

run_eval() {
  local dir=$1 label=$2
  echo "=== $label ($dir) $(date +%H:%M:%S) ==="
  caffeinate -i python3 backtest/run_north_star_eval.py --backtest \
    --report-dir "$dir" --label "$label" \
    --top-n 10 --focus-days 10 --rank-field composite \
    2>&1 | tee "logs/baseline_newcal_${label}_${TS}.log" | tail -5
}

run_eval reports/daily_selection_ng101          NG101-NEWCAL
run_eval reports/daily_selection_ng106          NG106-NEWCAL
run_eval reports/daily_selection_ng101_pre2020  NG101-PRE2020-NEWCAL
run_eval reports/daily_selection_ng106_pre2020  NG106-PRE2020-NEWCAL

echo "=== 全部完成 $(date +%H:%M:%S) ==="
