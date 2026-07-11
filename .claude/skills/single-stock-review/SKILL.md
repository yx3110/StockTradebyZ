---
name: single-stock-review
description: "Per-stock historical backtest + tech/fundamental deep-dive for a day's top picks. Takes (date, version) and emits a markdown report Claude can narrate on top of."
disable-model-invocation: true
argument-hint: "[date YYYY-MM-DD] [version ng1.0.1|ng1.0.6|...] [top-n default 10]"
allowed-tools: Bash(python3 *), Read, Glob, Grep
---

# Single-Stock Review

For a given date + NG version, scan the day's Top-N picks and produce:
1. **Historical backtest** — scan all past reports for dates where each pick was in Top-10 / Top-50, compute forward 3d/5d/10d/15d adj_close returns and aggregate
2. **Tech + fundamental snapshot** — MA/KDJ/RSI/MACD/BOLL + PE/PB/PS/market cap + 20-day range position

Intended as a daily complement to the selection report: the ML model tells you *what to buy*, this tells you *how each pick has performed historically and where it stands technically*.

## Arguments

- `$0`: Trading date in YYYY-MM-DD format. Required.
- `$1`: NG scoring version (e.g. `ng1.0.1`, `ng1.0.6`). Required.
- `$2`: Top-N to review. Optional, default 10.

## Execution Steps

### Step 1: Verify today's report exists

```bash
ls -la /Users/yangxu/StockTradebyZ/reports/daily_selection_$(echo $1 | tr -d '.')*/analysis_data_$(echo $0 | tr -d '-').json 2>/dev/null | head -3
```

If nothing found, tell user to run `/generate-report $0 $1` first. Do NOT proceed.

### Step 2: Run the review script

```bash
python3 /Users/yangxu/StockTradebyZ/scripts/single_stock_review.py --date $0 --version $1 --top-n ${2:-10}
```

This writes:
- `reports/single_stock_review/{version}_{date}.md` — human-readable report with 3 parts (Top picks / Historical backtest / Snapshot)
- `reports/single_stock_review/{version}_{date}.json` — structured data for downstream analysis

Runtime: ~15–20s (scans ~1800 historical JSONs, one DB query for prices, 10 DB queries for snapshots).

### Step 3: Read and enhance with narrative

Read the generated markdown. Then **enhance it with Claude's interpretation**:

1. **Business descriptions** — for each of the Top-N stocks, add a 1–2 sentence business description (industry, main products, key customer type, cyclicality). Use your knowledge of Chinese A-share companies.
2. **Technical verdict per stock** — combine the auto-generated metrics with your read on:
   - Position vs MA20/MA60 (trend)
   - KDJ/RSI overbought-oversold (short-term timing)
   - MACD hist sign change (momentum turn)
   - 20-day range position (pullback vs breakout)
   - ATR relative to price (risk budget)
3. **Historical-backtest callouts** — flag stocks with ≥60% win rate at 10d (strong signal) or ≤40% win rate (avoid). Note sample size — anything <10 is "fragile".
4. **Cross-reference the top-10 vs top-50 backtest tables** — if a stock's top-10 sample is tiny but top-50 looks healthy, mention it as "new to Top-10, broader-signal supportive".
5. **Portfolio-level recommendation** — synthesize into 3 buckets:
   - 🟢 **首选** (strong history + healthy tech + reasonable R:R)
   - 🟡 **次选 / 等回踩** (good fundamentals but short-term overbought, or R:R<1)
   - 🔴 **跳过** (historical <40% win rate, or broken tech structure)

Present the enhanced narrative to the user. Reference the raw report file path so they can revisit.

### Step 4: Optional clipboard copy

If the user asks to copy the analysis:
```bash
cat reports/single_stock_review/{version}_{date}.md | pbcopy
```

## Integration With Daily Flow

This skill is called as **Phase 1.5** in the `/daily-rebalance` workflow, right after `generate-report` and before `live_portfolio`. It doesn't affect trade execution — it's a sanity-check layer to help the user understand why the model picked each stock and whether the signal has historical teeth.

## Notes

- **Uses `adj_close` for forward returns** — survives dividends/splits. Falls back to `close` when `adj_close` is NULL.
- **T+1 entry convention** — signal emitted at `date_str` close, entry at next trading day's close, exit at close+N.
- **Today's date is excluded** from historical scan (can't peek at future).
- Supported versions: anything with a `reports/daily_selection_ng{tag}/` or `reports/daily_selection_ng{tag}_fullmarket/` folder.
- The `**自动点评**` line per stock is rule-based (coarse signals like "MA60下X%", "KDJ超买", "MACD翻红"). The `🟢/🟡/🔴` priority tags are added by Claude during Step 3 narrative enhancement. Don't skip Step 3 — that's where the report becomes useful.
