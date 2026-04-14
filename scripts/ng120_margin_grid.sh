#!/usr/bin/env bash
# Run ng1.2.0 fast-check WF training across margin grid sequentially.
# Parallel is tempting but each run uses 12GB RAM + full CPU — serial is safer.
#
# Usage:
#   bash scripts/ng120_margin_grid.sh              # default: fast-check only
#   bash scripts/ng120_margin_grid.sh --full       # full WF (hours)
#
# Output logs: logs/ng120_fastcheck_m{tag}_{timestamp}.log
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:---fast-check}"
MARGINS=(0.03 0.05 0.08 0.10)
ts=$(date +%Y%m%d_%H%M%S)

# Exclude the m005 run we already launched for smoke — if its model exists, skip it.
# (LightGBM custom-objective .pkl files are named with _m005 tag.)
existing_m005=$(ls ml_models/trained_models/ng/ng120_*_m005_*.pkl 2>/dev/null | head -1 || true)

for m in "${MARGINS[@]}"; do
    tag=$(printf "m%03d" "$(echo "$m*100/1" | bc)")
    if [[ "$m" == "0.05" && -n "$existing_m005" ]]; then
        echo "[$(date '+%H:%M:%S')] skip m=$m (found existing $existing_m005)"
        continue
    fi
    logfile="logs/ng120_${MODE#--}_${tag}_${ts}.log"
    echo "[$(date '+%H:%M:%S')] launching margin=$m → $logfile"
    python3 ml_models/ng/ng_trainer.py \
        --version ng1.2.0 \
        --margin "$m" \
        $MODE \
        --purge-days 15 \
        2>&1 | tee "$logfile"
    echo "[$(date '+%H:%M:%S')] done margin=$m"
done

echo ""
echo "Grid complete. Inspect results:"
echo "  grep -E 'ICIR|IC\\s*=' logs/ng120_${MODE#--}_m*_${ts}.log"
