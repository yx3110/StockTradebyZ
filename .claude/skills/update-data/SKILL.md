---
name: update-data
description: "Fetch and update daily market data from Tushare API including quotes, fundamentals, technical indicators, and v39 feature cache"
disable-model-invocation: true
argument-hint: "[date in YYYYMMDD, default today]"
allowed-tools: Bash(python3 *), Read, Glob, Grep
---

# Daily Data Update

Update all market data for the specified trading date.

## Arguments

- `$ARGUMENTS`: Trading date in YYYYMMDD format (e.g., 20260221). If empty, defaults to today.

## Execution Steps

### Step 1: Run complete data update

```bash
python3 /Users/yangxu/StockTradebyZ/fetch_data/quick_daily_update.py --date $ARGUMENTS
```

If no date argument was provided, run without `--date`:

```bash
python3 /Users/yangxu/StockTradebyZ/fetch_data/quick_daily_update.py
```

This script updates:
- Market quotes (7000+ stocks/ETFs OHLCV)
- Market indices (10 major indices: SSE, SZSE, ChiNext, STAR50, SSE50, CSI300, CSI500, CSI1000, CSI2000, CSI Total)
- Daily basic data (PE, PB, PS, market cap, turnover rate for 5400+ stocks)
- Financial indicators (EPS, ROE, ROA + 17 extended v3.9 fields, quarterly)
- Technical indicators (MA, EMA, RSI, MACD, KDJ, BBI)
- v39_feature_cache (automatic sync)

### Step 2: Verify the update

After the script completes, run a quick verification:

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/Users/yangxu/StockTradebyZ/data_adapter/stock_data.db')
c = conn.cursor()
c.execute('SELECT MAX(trade_date), COUNT(*) FROM daily_quotes WHERE trade_date = (SELECT MAX(trade_date) FROM daily_quotes)')
date, count = c.fetchone()
c.execute('SELECT COUNT(*) FROM v39_feature_cache WHERE trade_date = (SELECT MAX(trade_date) FROM v39_feature_cache)')
cache_count = c.fetchone()[0]
print(f'Latest date: {date}')
print(f'Quotes on latest date: {count}')
print(f'v39 cache on latest date: {cache_count}')
conn.close()
"
```

### Step 3: Report results

Summarize to the user:
- Whether the update succeeded or failed
- Latest date in the database
- Number of quotes updated
- Number of v39 feature cache entries
- Any errors or warnings encountered

## Error Handling

- If the script fails due to Tushare API limits, report the error and suggest retrying later
- If `--date` is a non-trading day (weekend/holiday), the script will report no data available - this is normal
- If v39_feature_cache update fails, mention it can be manually updated with:
  ```bash
  python3 /Users/yangxu/StockTradebyZ/fetch_data/v39_feature_cache_updater.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD
  ```
