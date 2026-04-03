# North Star V5.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade North Star scoring from V5.1 (7L/46指标) to V5.2 (9L/58指标) — fix bugs, rebalance weights, add L8 execution quality + L9 条件评分, multi-benchmark.

**Architecture:** All changes in `backtest/north_star_metrics.py` (targets + compute functions + scoring) and `backtest/backtest_report_based.py` (scorecard printing), with CLI integration in `backtest/run_north_star_eval.py`. V5.2 is additive — V5/V5.1 code untouched.

**Tech Stack:** Python, NumPy, Pandas, scipy.stats, statsmodels (existing deps)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `backtest/north_star_metrics.py` | Modify | Bug fixes + V5.2 targets + new compute functions + compute_v52_score |
| `backtest/backtest_report_based.py` | Modify | `_print_scorecard_v52()` + metric_value_map assembly |
| `backtest/run_north_star_eval.py` | Modify | `--score-version v52` CLI + V5.2 integration in run_backtest |

---

### Task 1: Bug fixes in score_metric_v5 and compute functions

**Files:**
- Modify: `backtest/north_star_metrics.py:2597-2604` (higher extrapolation)
- Modify: `backtest/north_star_metrics.py:3144` (regime DD floor)
- Modify: `backtest/north_star_metrics.py:2701` (OOS IC half-life)
- Modify: `backtest/north_star_metrics.py:3175` (CSCV PBO variant length)

- [ ] **Step 1: Fix score_metric_v5 higher-direction extrapolation asymmetry**

Line 2600-2604: Add `|pass|*0.5` safety to match lower-direction logic:

```python
# OLD (line 2601-2604):
range_span = max(abs(bp_raw[-1] - bp_raw[0]), 1e-8)
floor = bp_raw[0] - range_span

# NEW:
range_span = max(abs(bp_raw[-1] - bp_raw[0]), abs(bp_raw[0]) * 0.5 + 1e-8)
floor = bp_raw[0] - range_span
```

- [ ] **Step 2: Fix regime_transition_dd denominator floor**

Line 3144: raise floor from 0.001 to 0.02:

```python
# OLD:
denominator = max(normal_dd_median, bm_normal_dd_median, 0.001)

# NEW:
denominator = max(normal_dd_median, bm_normal_dd_median, 0.02)
```

- [ ] **Step 3: Fix OOS IC half-life when slope >= 0**

Line 2701-2702: non-decaying IC should get capped score, not perfect:

```python
# OLD:
if slope >= 0:
    return 12.0

# NEW:
if slope >= 0:
    return 6.0  # good but not perfect — non-decay could be overfitting
```

- [ ] **Step 4: Fix CSCV PBO variant length mismatch**

Line 3175: trim first variant to match bootstrap length:

```python
# OLD:
variants = [returns]

# NEW:
trimmed_len = n_subperiods * period_len
variants = [returns[:trimmed_len]]
```

- [ ] **Step 5: Commit bug fixes**

```bash
git add backtest/north_star_metrics.py
git commit -m "fix: V5.2 score_metric外推对称 + regime DD floor + OOS半衰期 + CSCV长度"
```

---

### Task 2: V5.2 targets, weights, new L7 metrics, style profiles

**Files:**
- Modify: `backtest/north_star_metrics.py` (add after line 3410)

- [ ] **Step 1: Add 4 new L7 compute functions**

```python
def compute_adv_coverage(picks_with_volume, n_positions=10, threshold=5.0):
    """ADV覆盖率: 持仓中ADV > threshold倍仓位金额的比例."""

def compute_sector_hhi(sector_series):
    """行业Herfindahl指数. 0=完全分散, 1=单一行业."""

def compute_avg_impact_cost(picks_with_volume, avg_turnover, n_positions=10, eta=0.15):
    """平均冲击成本 (Almgren-Chriss, 年化%)."""

def compute_micro_cap_ratio(market_caps, threshold_bn=3.0):
    """微盘股比例: 市值<threshold的持仓占比."""
```

- [ ] **Step 2: Add L8 execution quality compute functions (5 metrics)**

```python
def compute_delay_cost(daily_results, holding_days=10):
    """延迟成本: 信号日vs次日执行的收益差."""

def compute_execution_fill_rate(limit_up_fail_count, total_trades):
    """执行成功率: 非涨停不可买比例."""

def compute_realized_vs_theoretical(net_return, gross_return):
    """实现/理论收益比: 扣成本后vs毛收益."""

def compute_turnover_efficiency(excess_return, annual_turnover):
    """换手效率: 每单位换手产生的超额收益."

def compute_implementation_shortfall(signal_returns, actual_returns):
    """实施缺口: 信号理论收益vs实际收益差."
```

- [ ] **Step 3: Add L9 regime-conditional scoring functions (4 metrics)**

```python
def compute_regime_ic_consistency(ic_series, benchmark_returns, lookback=60):
    """Regime间IC一致性: min(牛市ICIR, 熊市ICIR, 震荡ICIR) / max(...)."""

def compute_regime_sharpe_floor(daily_returns, benchmark_returns, lookback=60):
    """最差regime Sharpe: min(牛市Sharpe, 熊市Sharpe, 震荡Sharpe)."""

def compute_multi_benchmark_excess(daily_returns, primary_bm, secondary_bm):
    """多基准超额: min(vs primary, vs secondary) 年化."""

def compute_regime_drawdown_ratio(daily_returns, benchmark_returns, lookback=60):
    """Regime DD比: 最差regime MaxDD / 全期MaxDD."""
```

- [ ] **Step 4: Define V5.2 targets + weights + style profiles**

```python
V52_LAYER_NAMES = {
    1: '信号质量', 2: '组合效率', 3: '风险控制',
    4: 'OOS鲁棒性', 5: '超额收益', 6: '因子归因',
    7: '容量可扩展', 8: '执行质量', 9: '条件稳健性',
}

V52_LAYER_WEIGHTS = {
    1: 0.22, 2: 0.10, 3: 0.15, 4: 0.12,
    5: 0.08, 6: 0.08, 7: 0.08, 8: 0.08, 9: 0.09,
}
# Sum = 1.00, L7 from 14% to 8%, new L8=8%, L9=9%

# Style profiles: adjust targets for small-cap vs default
V52_STYLE_PROFILES = {
    'default': {},  # use standard targets
    'small_cap': {  # median_cap < 15bn
        'strategy_capacity_mn': {'target': 1000},  # vs 5000
        'participation_rate_p90': {'target': 0.03},  # vs 0.01
        'effective_n_corr': {'target': 6.0},  # vs 8.0
    },
    'large_cap': {  # median_cap >= 80bn
        'daily_ic': {'pass': 0.02, 'target': 0.06},  # harder alpha
        'icir': {'pass': 0.20, 'target': 0.55},
    },
}
```

- [ ] **Step 5: Implement compute_v52_score with style adaptation**

- [ ] **Step 6: Commit**

```bash
git add backtest/north_star_metrics.py
git commit -m "feat: V5.2 9层58指标 + 执行质量L8 + 条件稳健L9 + 风格适配"
```

---

### Task 3: Scorecard printing + CLI integration

**Files:**
- Modify: `backtest/backtest_report_based.py` (add `_print_scorecard_v52`)
- Modify: `backtest/run_north_star_eval.py` (add v52 to CLI choices + integration)

- [ ] **Step 1: Add _print_scorecard_v52 in backtest_report_based.py**

Based on `_print_scorecard_v51` pattern, add:
- L8/L9 metric value map entries
- V5.2 new metrics format keys
- Multi-benchmark injection
- Style profile display

- [ ] **Step 2: Add v52 to CLI choices in run_north_star_eval.py**

```python
# choices: add 'v52'
choices=['v2', 'v3', 'v4', 'v5', 'v51', 'v52', 'both', 'all']

# In _inject_wf_summary: add _print_scorecard_v52 call
# In production config: update default to v52
```

- [ ] **Step 3: Wire up V5.2 metrics computation in backtest engine**

Ensure backtest_report_based.py computes and passes:
- L7 new: adv_coverage, sector_hhi, avg_impact_cost, micro_cap_ratio
- L8: delay_cost, execution_fill_rate, realized_vs_theoretical, turnover_efficiency, implementation_shortfall
- L9: regime_ic_consistency, regime_sharpe_floor, multi_benchmark_excess, regime_drawdown_ratio

- [ ] **Step 4: Run production evaluation to verify**

```bash
python3 backtest/run_north_star_eval.py --production --score-version v52
```

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/backtest_report_based.py backtest/run_north_star_eval.py
git commit -m "feat: V5.2评分卡打印 + CLI集成 + 生产验证"
```
