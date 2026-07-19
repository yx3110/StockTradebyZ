#!/bin/zsh
# 新口径基线重跑 (2026-07-11 北极星 P0 修复后, Check 5 公平对比)
# 口径 tag: 2026-07-11-p0fix (基准回填/惩罚拆三/PSR/L8/L9修复/复权pct回填)
# 统一参数: --top-n 10 --focus-days 10 --rank-field composite (与历史 headline 对齐)
# 前置: backfill_adj_factor.py + backfill_price_change_pct.py 已完成
#
# 2026-07-19 T02-a 评估-生产同构化:
#   每个 ng101 基线同时出两行 — RAW (裸信号) + PRODOVERLAY (生产 L1/L2/L4 重放,
#   scripts/ng_production_overlay_replay.py 直调生产代码生成)。
#   PRODOVERLAY 线只看 NAV 链指标 (Sharpe/MaxDD/CVaR/换手), 宇宙级指标 (L1 IC/L6)
#   建立在幸存者宇宙上不可与 RAW 混比。
# 2026-07-19 T02-e: ng101 线注入 --wf-summary (ng1.0.1 7-13 训练摘要) 修 L4 两项
#   N/A 仪器盲区 (WFER + OOS IC 半衰期)。注意: wf_summary 来自 seed42 单 seed 训练,
#   用于 3-seed 报告线是近似 (WF 训练效率指标近似 seed 不变)。
set -e
cd "$(dirname "$0")/.."
TS=$(date +%Y%m%d_%H%M)
NG_WFS=ml_models/trained_models/ng/wf_summary.json

run_eval() {
  local dir=$1 label=$2
  shift 2
  echo "=== $label ($dir) $(date +%H:%M:%S) ==="
  caffeinate -i python3 backtest/run_north_star_eval.py --backtest \
    --report-dir "$dir" --label "$label" \
    --top-n 10 --focus-days 10 --rank-field composite "$@" \
    2>&1 | tee "logs/baseline_newcal_${label}_${TS}.log" | tail -5
}

# 生产 overlay 重放目录刷新 (幂等; regime/L4 直调生产代码)
python3 scripts/ng_production_overlay_replay.py \
  --src reports/daily_selection_ng101_3seed \
  --dst reports/daily_selection_ng101_3seed_prodoverlay

run_eval reports/daily_selection_ng101_3seed             NG101-3SEED-RAW      --wf-summary "$NG_WFS"
run_eval reports/daily_selection_ng101_3seed_prodoverlay NG101-3SEED-PRODOVLY --wf-summary "$NG_WFS"
run_eval reports/daily_selection_ng101                   NG101-NEWCAL
run_eval reports/daily_selection_ng106                   NG106-NEWCAL
run_eval reports/daily_selection_ng101_pre2020           NG101-PRE2020-NEWCAL
run_eval reports/daily_selection_ng106_pre2020           NG106-PRE2020-NEWCAL

echo "=== 全部完成 $(date +%H:%M:%S) ==="
