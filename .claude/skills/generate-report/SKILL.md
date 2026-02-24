---
name: generate-report
description: "Generate daily stock selection report using 8 quantitative strategies and ML scoring (v3.9/v3.95)"
disable-model-invocation: true
argument-hint: "[date in YYYY-MM-DD] [scoring version: v3.9|v3.95]"
allowed-tools: Bash(python3 *), Read, Glob, Grep
---

# Generate Daily Stock Selection Report

Run the quantitative stock selector with ML scoring and output a selection report.

## Arguments

- `$0`: Trading date in YYYY-MM-DD format (e.g., 2026-02-21). Required.
- `$1`: Scoring version. Optional, defaults to `v3.9`. Accepted values: `v3.9`, `v3.95`.

## Execution Steps

### Step 1: Verify data availability

Before running the selector, check that data exists for the requested date:

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/Users/yangxu/StockTradebyZ/data_adapter/stock_data.db')
c = conn.cursor()
date_str = '$0'.replace('-', '')
c.execute('SELECT COUNT(*) FROM daily_quotes WHERE trade_date = ?', (date_str,))
count = c.fetchone()[0]
c.execute('SELECT MAX(trade_date) FROM daily_quotes')
latest = c.fetchone()[0]
print(f'Quotes for {date_str}: {count}')
print(f'Latest date in DB: {latest}')
if count == 0:
    print('WARNING: No data for this date. Run /update-data first.')
conn.close()
"
```

If no data exists for the requested date, warn the user and suggest running `/update-data` first. Do NOT proceed without data.

### Step 2: Run stock selector

Default (v3.9):
```bash
python3 /Users/yangxu/StockTradebyZ/tomorrow_stock_selector.py $0
```

With explicit version:
```bash
python3 /Users/yangxu/StockTradebyZ/tomorrow_stock_selector.py $0 --scoring-version $1
```

If `$1` is empty or not provided, omit `--scoring-version` to use the default v3.9.

This runs:
- 8 quantitative strategies (BBI-KDJ, BBI Short/Long, Breakout Volume KDJ, Peak KDJ, SuperB1, ZhiXing, MA60 Cross Volume Wave, Big Bullish Volume)
- ML ensemble scoring (LightGBM + XGBoost + CatBoost + RandomForest)
- Report generation with ranked stock picks

### Step 3: Read and summarize the report

The report is saved to:
- v3.9: `reports/daily_selection_v3.9/选股分析报告_YYYYMMDD.md`
- v3.95: `reports/daily_selection_v3.95/选股分析报告_YYYYMMDD.md`

Read the generated report and present a concise summary to the user:
- Total number of stocks selected
- Top 10 stocks with scores and strategies
- Score distribution overview
- Any notable patterns (e.g., multiple strategies flagging the same stock)

### Step 4: Optional follow-up suggestions

After presenting the summary, mention:
- For AI-enhanced analysis: `python3 /Users/yangxu/StockTradebyZ/ai_enhanced_daily_report.py --date $0`
- For v3.95 multi-target comparison (if v3.9 was used): suggest running with `v3.95`
- Report location for user reference

## Notes

- v3.9 is the production-recommended version (42 features + 17 extended financials)
- v3.95 provides multi-target predictions (3d/5d/10d returns) with rolling training windows
- If v3.95 shows uniform scores of 45.0, the v39_feature_cache may be missing data for the date
