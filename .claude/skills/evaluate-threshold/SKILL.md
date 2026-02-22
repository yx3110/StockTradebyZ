---
name: evaluate-threshold
description: "Evaluate ML model score thresholds to find optimal stock filtering cutoffs based on 3d/5d/10d expected returns. Supports any model version (v3.9, v3.95, v4.0, etc.)"
disable-model-invocation: true
argument-hint: "[--models v3.9 v3.95 v4.0] [--target 10] [--csv model=path ...]"
allowed-tools: Bash(python3 *), Read, Glob, Grep
---

# Evaluate ML Score Thresholds

Analyze any ML model version's scores to find optimal thresholds that filter stocks for next-day watchlist. The threshold is calibrated so that stocks scoring above it have the best 3d/5d/10d expected returns while keeping the daily count manageable.

## Arguments

- `$ARGUMENTS`: Passed directly to the evaluation script. Key options:
  - `--models v3.9 v3.95 v4.0`: Model versions to analyze (default: auto-discover all available)
  - `--target N`: Target daily stock count (default: 10)
  - `--csv model=path`: Manually specify CSV file per model (e.g., `v4.0=reports/backtest/some_picks.csv`)
  - `--dirs model=dir`: Report directory per model, generates picks first
  - `--regenerate`: Regenerate all picks data before analysis
  - `--no-cross`: Skip dual-model cross analysis

## Execution Steps

### Step 1: Check for existing picks data

```bash
python3 -c "
from glob import glob
import os
from datetime import datetime
search_dir = '/Users/yangxu/StockTradebyZ/reports/backtest'
for pattern in ['report_backtest_*_picks.csv', 'ml_backtest_*_picks.csv']:
    files = sorted(glob(os.path.join(search_dir, pattern)), key=os.path.getmtime, reverse=True)
    for f in files[:5]:
        dt = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
        size = os.path.getsize(f) / 1024
        print(f'{dt} ({size:.0f}KB) {os.path.basename(f)}')
"
```

If no picks CSV files exist for the requested models, inform the user they need to generate them first:
```bash
python3 /Users/yangxu/StockTradebyZ/backtest/backtest_report_based.py --report-dir <报告目录> --label <标签>
```

### Step 2: Run threshold evaluation

```bash
python3 /Users/yangxu/StockTradebyZ/backtest/evaluate_score_threshold.py $ARGUMENTS
```

If `$ARGUMENTS` is empty, runs with defaults (auto-discover all available model CSVs, target <=10 stocks/day).

### Step 3: Read and present the report

The report is saved to `reports/backtest/阈值评估报告_YYYYMMDD.md`.

Read the generated report and present a concise summary:
- Recommended thresholds per model
- Key metrics: daily count, 5d/10d average return and win rate
- Score distribution overview
- Cross-model comparison if multiple models analyzed

### Step 4: Practical interpretation

After presenting the analysis, provide actionable advice:
- Which threshold to use for daily screening per model
- Expected number of stocks per day at the recommended threshold
- Confidence level based on win rate and sample size
- Which model performs best for the user's target holding period

## Examples

```bash
# Analyze all available models
/evaluate-threshold

# Only v4.0
/evaluate-threshold --models v4.0

# Compare v3.95 and v4.0, target 5 stocks/day
/evaluate-threshold --models v3.95 v4.0 --target 5

# Use specific CSV file for a new model
/evaluate-threshold --csv v4.1=reports/backtest/v41_picks.csv
```

## Notes

- Auto-discovery searches `reports/backtest/` for `report_backtest_*_picks.csv` and `ml_backtest_*_picks.csv`
- Known models with auto-discovery: v3.9, v3.95, v4.0 (extensible via MODEL_SEARCH_PATTERNS dict)
- For unknown model versions, use `--csv model=path` to specify the CSV file manually
- The picks CSV must have columns: date, code, score, return_3d, return_5d, return_10d (return_1d, return_15d optional)
- Returns are calculated as: buy at open on next trading day, sell at close N days later
