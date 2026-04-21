#!/usr/bin/env bash
# Sequential training for 3 ng1.4.0 runs (3 seeds, single-head, no downside pair).
# ng1.4.0 = ng1.0.1 stable base + 4 Tier A downside + 3 AMV = 70 features.
# Cache hits reuse ng130_feature_cache (shared with ng1.3.0). First seed ~70min,
# subsequent ~30-45min.

set -euo pipefail
set -o pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/ng140_retrain_${TS}
mkdir -p "${LOG_DIR}"

for SEED in 42 123 456; do
  TAG="seed${SEED}"
  LOG="${LOG_DIR}/ng140_${TAG}.log"
  echo "=== [$(date '+%H:%M:%S')] Training ng1.4.0 ${TAG} ==="
  caffeinate -i python3 ml_models/ng/ng_trainer.py \
    --version ng1.4.0 \
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
