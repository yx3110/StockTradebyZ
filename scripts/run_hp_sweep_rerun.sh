#!/bin/bash
# HP sweep 补跑: 5 个被 Tier0 fast-check 滑窗改动污染的 profile
# --fast-check-train-days 0 = 不截断 train (还原 7-12 基线 expanding 条件)
set -u
cd /Users/yangxu/StockTradebyZ
LOGDIR=logs/hp_sweep_20260712
mkdir -p "$LOGDIR"

for p in lr005 leaves63 ff085 loose lr001; do
    echo "===== [$(date '+%H:%M:%S')] rerun hp-profile=$p 开始 ====="
    caffeinate -i python3 ml_models/ng/ng_trainer.py \
        --version ng1.0.1 \
        --start-date 2020-01-01 \
        --purge-days 15 \
        --seed 42 \
        --fast-check \
        --fast-check-train-days 0 \
        --target-parallel 4 \
        --hp-profile "$p" \
        > "$LOGDIR/fc_${p}_expanding.log" 2>&1
    echo "===== [$(date '+%H:%M:%S')] rerun hp-profile=$p 结束 (rc=$?) ====="
done
echo "RERUN ALL DONE"
