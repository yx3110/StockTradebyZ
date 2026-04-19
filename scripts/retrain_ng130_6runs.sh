#!/usr/bin/env bash
# Sequential training for 6 ng1.3.0 runs (3 seeds × 2 heads).
# Uses --target-parallel 4 (1.4x per-window speedup on M5 Max).
# First run per (version, head) is cache-miss (~45min), rest are cache-hit (~25min).

set -euo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/ng130_retrain_${TS}
mkdir -p "${LOG_DIR}"

for SEED in 42 123 456; do
  for HEAD in excess downside; do
    TAG="seed${SEED}_${HEAD}"
    LOG="${LOG_DIR}/ng130_${TAG}.log"
    echo "=== [$(date '+%H:%M:%S')] Training ng1.3.0 ${TAG} ==="
    python3 ml_models/ng/ng_trainer.py \
      --version ng1.3.0 \
      --head "${HEAD}" \
      --seed "${SEED}" \
      --target-parallel 4 \
      --purge-days 15 \
      > "${LOG}" 2>&1 || {
        echo "❌ ${TAG} FAILED — log: ${LOG}"
        exit 1
      }
    echo "✅ ${TAG} done — log: ${LOG}"
  done
done
echo "All 6 runs complete. Logs in ${LOG_DIR}"
