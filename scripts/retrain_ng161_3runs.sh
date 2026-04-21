#!/usr/bin/env bash
# ng1.6.1 = ng1.0.1 features + cross-sectional factor-residual labels (F2).
# Target: Stage 4a V5.2 >= 76% with beta_UMD < 1.0.

set -euo pipefail
set -o pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/ng161_retrain_${TS}
mkdir -p "${LOG_DIR}"

for SEED in 42 123 456; do
  TAG="seed${SEED}"
  LOG="${LOG_DIR}/ng161_${TAG}.log"
  echo "=== [$(date '+%H:%M:%S')] Training ng1.6.1 ${TAG} ==="
  caffeinate -i python3 ml_models/ng/ng_trainer.py \
    --version ng1.6.1 \
    --seed "${SEED}" \
    --target-parallel 4 \
    --purge-days 15 \
    2>&1 | tee "${LOG}" || {
      echo "❌ ${TAG} FAILED — log: ${LOG}"
      exit 1
    }
  echo "✅ ${TAG} done — log: ${LOG}"
done
echo "All 3 runs complete. Logs in ${LOG_DIR}"
