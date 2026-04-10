# NG v1.0.8 Low-Turnover Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 fixed-parameter portfolio rules to backtest engine to reduce turnover from 44x to 15-20x while preserving ng1.0.1's signal quality.

**Architecture:** All changes in `backtest/backtest_report_based.py` (portfolio construction logic) and `backtest/run_north_star_eval.py` (CLI parameters). No model/scorer/trainer changes. New parameters: `buy_threshold`, `sell_threshold`, `n_groups`, `min_hold_days`, `cost_penalty`.

**Tech Stack:** Python, numpy, existing backtest infrastructure.

---

### Task 1: Add CLI parameters to run_north_star_eval.py

**Files:**
- Modify: `backtest/run_north_star_eval.py`

- [ ] **Step 1: Add 5 new CLI arguments**

In the argument parser section (near other `--retention-bonus`, `--score-floor` args), add:

```python
parser.add_argument('--buy-threshold', type=int, default=0,
                    help='NG1.0.8 持仓缓冲买入门槛 (0=关闭, 推荐8: 新股排进Top-8才买入)')
parser.add_argument('--sell-threshold', type=int, default=0,
                    help='NG1.0.8 持仓缓冲卖出门槛 (0=关闭, 推荐20: 跌出Top-20才卖出)')
parser.add_argument('--n-groups', type=int, default=1,
                    help='NG1.0.8 分批调仓组数 (1=不分批, 2=两组错开5天调仓)')
parser.add_argument('--min-hold-days', type=int, default=0,
                    help='NG1.0.8 最小持有天数 (0=关闭, 推荐5: 买入后至少持有5天)')
parser.add_argument('--cost-penalty', type=float, default=0.0,
                    help='NG1.0.8 新股成本惩罚 (0=关闭, 推荐0.003: 扣0.3%%交易成本)')
```

- [ ] **Step 2: Pass new args to run_single_backtest**

Find where `run_single_backtest` is called (search for `run_single_backtest(`) and add the new kwargs:

```python
buy_threshold=args.buy_threshold,
sell_threshold=args.sell_threshold,
n_groups=args.n_groups,
min_hold_days=args.min_hold_days,
cost_penalty=args.cost_penalty,
```

- [ ] **Step 3: Compile check and commit**

```bash
python3 -c "import py_compile; py_compile.compile('backtest/run_north_star_eval.py', doraise=True)"
git add backtest/run_north_star_eval.py
git commit -m "feat(ng108): CLI参数 — buy/sell_threshold, n_groups, min_hold_days, cost_penalty"
```

---

### Task 2: Add parameters to run_single_backtest signature

**Files:**
- Modify: `backtest/backtest_report_based.py`

- [ ] **Step 1: Add 5 new parameters to function signature**

Add after `regime_gate_aggressive=False):` in the `run_single_backtest` definition (line ~942):

```python
                        # NG1.0.8: Low-turnover portfolio rules
                        buy_threshold=0,
                        sell_threshold=0,
                        n_groups=1,
                        min_hold_days=0,
                        cost_penalty=0.0):
```

- [ ] **Step 2: Add parameter logging**

After the existing parameter logging block (after `regime_gate_aggressive` log, around line 1009), add:

```python
    # NG1.0.8: Low-turnover portfolio rules
    ng108_active = buy_threshold > 0 or n_groups > 1 or min_hold_days > 0 or cost_penalty > 0
    if ng108_active:
        parts = []
        if buy_threshold > 0:
            parts.append(f"买入≤Top{buy_threshold}/卖出>Top{sell_threshold}")
        if n_groups > 1:
            parts.append(f"{n_groups}组分批调仓")
        if min_hold_days > 0:
            parts.append(f"最小持有{min_hold_days}天")
        if cost_penalty > 0:
            parts.append(f"新股成本惩罚{cost_penalty:.1%}")
        print(f"  NG1.0.8 低换手: {' + '.join(parts)}")
```

- [ ] **Step 3: Initialize portfolio tracking state**

Before the main date loop (before `daily_results = []`, around line 1146), add:

```python
    # NG1.0.8: Portfolio state tracking
    ng108_portfolio = {}  # {code: {'entry_date': str, 'group': int, 'rank_score': float}}
    ng108_rebal_counter = 0  # tracks which group to rebalance
```

- [ ] **Step 4: Commit**

```bash
python3 -c "import py_compile; py_compile.compile('backtest/backtest_report_based.py', doraise=True)"
git add backtest/backtest_report_based.py
git commit -m "feat(ng108): run_single_backtest签名+日志+状态初始化"
```

---

### Task 3: Implement the 4 portfolio rules in rebalancing logic

**Files:**
- Modify: `backtest/backtest_report_based.py`

This is the core change. The current logic (around line 1237-1271) does: retention_bonus → hold_buffer → replace_threshold → top_stocks. We add ng1.0.8 logic as an **alternative path** when `ng108_active` is True.

- [ ] **Step 1: Add ng1.0.8 rebalancing block**

After the existing `if hold_buffer > 0` block (around line 1271) and before the score_floor/risk_control blocks, add the ng1.0.8 path. Insert this code right after the `replace_threshold` block ends (around line 1300):

```python
        # NG1.0.8: Low-turnover portfolio construction (4 rules)
        if ng108_active:
            # Rule 4: Cost-aware ranking — penalize new positions
            if cost_penalty > 0:
                current_codes = set(ng108_portfolio.keys())
                adjusted_stocks = []
                for s in stocks:
                    s_copy = dict(s)
                    if s_copy['code'] not in current_codes:
                        s_copy['rank_score'] = s_copy.get('rank_score', 0) - cost_penalty
                    adjusted_stocks.append(s_copy)
                adjusted_stocks.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
                stocks_for_selection = adjusted_stocks
            else:
                stocks_for_selection = stocks

            # Build rank lookup: code -> rank (1-based)
            rank_map = {s['code']: idx + 1 for idx, s in enumerate(stocks_for_selection)}

            # Rule 2: Staggered rebalancing — determine active group
            if n_groups > 1:
                active_group = ng108_rebal_counter % n_groups
                # Only rebalance every focus_days/n_groups days
                should_rebal = (i % (focus_days // n_groups) == 0) if focus_days > 1 else True
            else:
                active_group = -1  # all slots active
                should_rebal = (i % focus_days == 0) if focus_days > 1 else True

            if should_rebal:
                if n_groups > 1:
                    ng108_rebal_counter += 1

                # Determine which positions to evaluate this cycle
                sell_candidates = []
                hold_positions = []
                for code, info in list(ng108_portfolio.items()):
                    if active_group >= 0 and info['group'] != active_group:
                        hold_positions.append(code)  # not this group's turn
                        continue

                    rank = rank_map.get(code, 9999)
                    days_held = i - info.get('entry_idx', 0)

                    # Rule 3: Min holding period
                    if min_hold_days > 0 and days_held < min_hold_days:
                        hold_positions.append(code)
                        continue

                    # Rule 1: Sell threshold
                    sell_thresh = sell_threshold if sell_threshold > 0 else effective_top_n
                    if rank > sell_thresh:
                        sell_candidates.append(code)
                    else:
                        hold_positions.append(code)

                # Remove sold positions
                for code in sell_candidates:
                    del ng108_portfolio[code]

                # Rule 1: Buy threshold — fill empty slots
                buy_thresh = buy_threshold if buy_threshold > 0 else effective_top_n
                slots_per_group = effective_top_n // max(n_groups, 1)
                current_group_count = sum(1 for c in ng108_portfolio
                                         if active_group < 0 or ng108_portfolio[c]['group'] == active_group)
                slots_available = slots_per_group - current_group_count if n_groups > 1 else effective_top_n - len(ng108_portfolio)

                if slots_available > 0:
                    for s in stocks_for_selection:
                        if slots_available <= 0:
                            break
                        code = s['code']
                        if code in ng108_portfolio:
                            continue
                        rank = rank_map.get(code, 9999)
                        if rank <= buy_thresh:
                            ng108_portfolio[code] = {
                                'entry_idx': i,
                                'group': active_group if active_group >= 0 else len(ng108_portfolio) % max(n_groups, 1),
                                'rank_score': s.get('rank_score', 0),
                            }
                            slots_available -= 1

            # Build top_stocks from current portfolio
            portfolio_codes = set(ng108_portfolio.keys())
            top_stocks = [s for s in stocks_for_selection if s['code'] in portfolio_codes]
            if not top_stocks:
                # Fallback: use top_n if portfolio empty
                top_stocks = stocks_for_selection[:effective_top_n]
            top_stocks.sort(key=lambda x: x.get('rank_score', 0), reverse=True)
```

- [ ] **Step 2: Compile check**

```bash
python3 -c "import py_compile; py_compile.compile('backtest/backtest_report_based.py', doraise=True)"
```

- [ ] **Step 3: Quick smoke test**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "ng108-test" --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 20 --n-groups 2 --min-hold-days 5 --cost-penalty 0.003 \
    --start-date 2025-01-01 --end-date 2025-06-30 2>&1 | grep -E "换手|Sharpe|年化"
```

Expected: turnover significantly lower than baseline 44x.

- [ ] **Step 4: Commit**

```bash
git add backtest/backtest_report_based.py
git commit -m "feat(ng108): 4规则低换手组合(持仓缓冲+分批调仓+最小持有+成本感知)"
```

---

### Task 4: Run ablation experiments

**Files:** None (evaluation only)

- [ ] **Step 1: Baseline**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "baseline" --top-n 10 --focus-days 10 --rank-field composite \
    2>&1 | grep -E "🎯 10日|年化|Sharpe|换手"
```

- [ ] **Step 2: +Hysteresis only**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "+hysteresis" --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 20 \
    2>&1 | grep -E "🎯 10日|年化|Sharpe|换手"
```

- [ ] **Step 3: +Hysteresis +Cost-Aware**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "+hyst+cost" --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 20 --cost-penalty 0.003 \
    2>&1 | grep -E "🎯 10日|年化|Sharpe|换手"
```

- [ ] **Step 4: +Hysteresis +Cost-Aware +Min-Hold**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "+hyst+cost+minhold" --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 20 --cost-penalty 0.003 --min-hold-days 5 \
    2>&1 | grep -E "🎯 10日|年化|Sharpe|换手"
```

- [ ] **Step 5: Full ng1.0.8 (all 4 rules)**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "ng108-full" --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 20 --n-groups 2 --min-hold-days 5 --cost-penalty 0.003 \
    2>&1 | grep -E "🎯 10日|年化|Sharpe|换手"
```

- [ ] **Step 6: Collect and compare results**

Create a comparison table of all 5 configs: baseline, +hyst, +hyst+cost, +hyst+cost+minhold, full.
Focus on: turnover, Sharpe, net annual return, V5.2 score.

---

### Task 5: Pre-2020 validation

**Files:** None (evaluation only)

- [ ] **Step 1: Run Pre-2020 with best config**

Use the full ng1.0.8 config on ng1.0.1 pre-2020 reports:

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "ng108-Pre2020" --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 20 --n-groups 2 --min-hold-days 5 --cost-penalty 0.003 \
    --start-date 2018-01-01 --end-date 2019-12-31 \
    2>&1 | grep -E "🎯 10日|年化|Sharpe|换手|加权评分|等级"
```

Expected: V5.2 should remain A+ (>= 70%).

- [ ] **Step 2: Compare with ng1.0.1 baseline Pre-2020**

ng1.0.1 baseline Pre-2020: V5.2=73.7% A+, Sharpe=2.37.
ng1.0.8 should have similar or better Sharpe (lower turnover = lower cost drag).

- [ ] **Step 3: Commit evaluation results to wiki**

Update `docs/wiki/models/ng-series.md` and `docs/wiki/log.md` with ng1.0.8 results.

```bash
git add docs/wiki/
git commit -m "eval(ng108): 消融实验+Pre-2020验证结果"
```

---

### Task 6: Test with ng1.0.5 risk overlay

**Files:** None (evaluation only)

- [ ] **Step 1: Full ng1.0.8 + ng1.0.5 overlay**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "ng108+ng105" --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 20 --n-groups 2 --min-hold-days 5 --cost-penalty 0.003 \
    --stop-loss 0.06 --regime-gate-aggressive --vol-target 0.20 \
    --cppi-floor 0.08 --cppi-multiplier 20 \
    2>&1 | grep -E "🎯 10日|年化|Sharpe|换手|加权评分|等级|MaxDD"
```

- [ ] **Step 2: Pre-2020 with overlay**

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label "ng108+ng105-Pre2020" --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 20 --n-groups 2 --min-hold-days 5 --cost-penalty 0.003 \
    --stop-loss 0.06 --regime-gate-aggressive --vol-target 0.20 \
    --cppi-floor 0.08 --cppi-multiplier 20 \
    --start-date 2018-01-01 --end-date 2019-12-31 \
    2>&1 | grep -E "🎯 10日|年化|Sharpe|换手|加权评分|等级"
```

- [ ] **Step 3: Final summary and wiki update**

Compare all combinations and update wiki with final recommendation.

```bash
git add docs/wiki/
git commit -m "eval(ng108): 最终评估 — ng1.0.8+ng1.0.5风控组合"
```
