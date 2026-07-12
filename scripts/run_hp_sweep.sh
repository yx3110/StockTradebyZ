#!/bin/bash
# HP sweep: 7 个 profile 顺序 fast-check (ng1.0.1, seed42, 与 7-12 基线完全对齐)
# 基线 (无 profile, logs/train_ng101_fastcheck_20260712.log):
#   10d IC=0.0662 ICIR=0.8930 | 15d IC=0.0714 ICIR=1.0005
set -u
cd /Users/yangxu/StockTradebyZ
LOGDIR=logs/hp_sweep_20260712
mkdir -p "$LOGDIR"

for p in mdil500 tight lr005 leaves63 ff085 loose lr001; do
    echo "===== [$(date '+%H:%M:%S')] hp-profile=$p 开始 ====="
    caffeinate -i python3 ml_models/ng/ng_trainer.py \
        --version ng1.0.1 \
        --start-date 2020-01-01 \
        --purge-days 15 \
        --seed 42 \
        --fast-check \
        --target-parallel 4 \
        --hp-profile "$p" \
        > "$LOGDIR/fc_${p}.log" 2>&1
    rc=$?
    tail -8 "$LOGDIR/fc_${p}.log" | grep -E "IC=|完成" || true
    echo "===== [$(date '+%H:%M:%S')] hp-profile=$p 结束 (rc=$rc) ====="
done
echo "ALL SWEEP DONE"
