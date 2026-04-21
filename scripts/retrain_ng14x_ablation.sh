#!/usr/bin/env bash
# Phase C ablation: retrain ng1.4.1 and ng1.4.2 (seed 42 only for speed).
# ng1.4.1 = ng1.4.0 - 4 downside stock (66 features)
# ng1.4.2 = ng1.4.0 - 3 AMV market   (67 features)
# Single seed = ~70 min each (cache miss on new feature hash).

set -euo pipefail
cd "$(dirname "$0")/.."

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/ng14x_ablation_${TS}
mkdir -p "${LOG_DIR}"

for VERSION in ng1.4.1 ng1.4.2; do
  TAG="${VERSION}_seed42"
  LOG="${LOG_DIR}/${TAG}.log"
  echo "=== [$(date '+%H:%M:%S')] Training ${VERSION} seed42 ==="
  python3 ml_models/ng/ng_trainer.py \
    --version "${VERSION}" \
    --seed 42 \
    --target-parallel 4 \
    --purge-days 15 \
    > "${LOG}" 2>&1 || {
      echo "❌ ${TAG} FAILED — log: ${LOG}"
      exit 1
    }
  echo "✅ ${TAG} done — log: ${LOG}"
done
echo "Ablation complete. Logs in ${LOG_DIR}"
