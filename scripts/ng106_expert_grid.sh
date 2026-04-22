#!/usr/bin/env bash
# Grid-search regime_switch sub-experts. Target window 2024-2026 V5.2.
# Regime = AMV bull/bear from market_amv (existing code).
# For each (bull, bear) combo, run regime_switch_backtest then north_star V5.2.

set -euo pipefail
cd "$(dirname "$0")/.."

BULLS=(
  "reports/daily_selection_ng1.0.1_fast"
  "reports/daily_selection_ng1.0.7_fast"
  "reports/daily_selection_ng1.4.0_stage4a"
  "reports/daily_selection_ng1.4.1_stage4a"
)
BEARS=(
  "reports/daily_selection_ng104_ensemble_3seed"
  "reports/daily_selection_ng104_ensemble_5seed"
)

TS=$(date +%Y%m%d_%H%M%S)
OUT=reports/ng106_grid_${TS}
mkdir -p "${OUT}"
RESULTS="${OUT}/grid_results.tsv"
printf "bull\tbear\tV5.2\n" > "${RESULTS}"

for BULL in "${BULLS[@]}"; do
  BULL_TAG=$(basename "${BULL}")
  for BEAR in "${BEARS[@]}"; do
    BEAR_TAG=$(basename "${BEAR}")
    TAG="${BULL_TAG}__${BEAR_TAG}"
    MERGED="${OUT}/merged_${TAG}"
    LOG="${OUT}/${TAG}.log"
    echo "=== [$(date '+%H:%M:%S')] ${BULL_TAG} + ${BEAR_TAG} ==="

    # Produce merged report dir
    python3 backtest/regime_switch_backtest.py \
      --bull-dir "${BULL}" --bear-dir "${BEAR}" \
      --top-n 10 --focus-days 10 --rank-field score > "${LOG}" 2>&1 || true

    # regime_switch_backtest writes to reports/daily_selection_regime_switch;
    # move it so the next iteration doesn't clobber.
    if [ -d reports/daily_selection_regime_switch ]; then
      rm -rf "${MERGED}"
      mv reports/daily_selection_regime_switch "${MERGED}"
    else
      echo "  [skip] merge failed"
      continue
    fi

    # North-star V5.2 on 2024-2026
    NS_LOG="${OUT}/${TAG}_north_star.log"
    python3 backtest/run_north_star_eval.py \
      --backtest --report-dir "${MERGED}" \
      --label "ng106-grid-${TAG}" \
      --top-n 10 --focus-days 10 --rank-field composite --score-version v52 \
      --start-date 2024-01-01 --end-date 2026-04-17 > "${NS_LOG}" 2>&1 || true

    V52=$(grep -oE "原始总分:[^(]*\(未加权[0-9]+%\)" "${NS_LOG}" | tail -1 | grep -oE "[0-9]+%" | tail -1 | tr -d '%' || echo "?")
    printf "%s\t%s\t%s\n" "${BULL_TAG}" "${BEAR_TAG}" "${V52}" | tee -a "${RESULTS}"
  done
done

echo ""
echo "Grid complete. Results: ${RESULTS}"
sort -t$'\t' -k3 -rn "${RESULTS}" | head
