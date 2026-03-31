# North Star V5.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend North Star V5 (39 metrics, 6 layers) to V5.1 (46 metrics, 7 layers) with capacity estimation, CSCV overfitting detection, Hurst exponent, regime transition drawdown, and effective N.

**Architecture:** Append 7 new computation functions + NORTH_STAR_TARGETS_V51 + compute_v51_score to `north_star_metrics.py`. Add `_print_scorecard_v51` to backtest integration. Follow V5's exact patterns.

**Tech Stack:** Python 3, NumPy, Pandas, SciPy, SQLite

**Spec:** `docs/superpowers/specs/2026-03-31-north-star-v51-design.md`

---

### Task 1: Add L3 Stability Metrics — Hurst Exponent + Regime Transition DD

**Files:**
- Modify: `backtest/north_star_metrics.py` (append after V5.1 placeholder comments, ~line 3046)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestV51L3Stability:
    """V5.1 L3: Hurst + Regime Transition DD"""

    def test_hurst_random_walk(self):
        """随机游走 → H≈0.5"""
        from backtest.north_star_metrics import compute_hurst_exponent
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 1000))
        h = compute_hurst_exponent(returns)
        assert 0.35 < h < 0.65, f"Random walk H={h}, expected ~0.5"

    def test_hurst_trending(self):
        """趋势序列 → H>0.6"""
        from backtest.north_star_metrics import compute_hurst_exponent
        np.random.seed(42)
        # 构造trending: 累积随机游走的差分有正自相关
        rw = np.cumsum(np.random.normal(0.001, 0.005, 1000))
        returns = pd.Series(np.diff(rw))
        h = compute_hurst_exponent(returns)
        assert h > 0.55, f"Trending H={h}, expected >0.55"

    def test_hurst_short_series(self):
        """短序列 → 默认0.5"""
        from backtest.north_star_metrics import compute_hurst_exponent
        returns = pd.Series(np.random.normal(0, 0.01, 50))
        assert compute_hurst_exponent(returns) == 0.5

    def test_regime_transition_dd_stable(self):
        """稳定策略, 转换DD放大倍数应<2"""
        from backtest.north_star_metrics import compute_regime_transition_dd
        np.random.seed(42)
        n = 500
        # 稳定正收益, 不受市场regime影响
        strategy_ret = pd.Series(np.random.normal(0.001, 0.01, n))
        # 构造有regime变化的benchmark
        benchmark_ret = pd.Series(np.concatenate([
            np.random.normal(0.002, 0.01, 200),  # bull
            np.random.normal(-0.003, 0.015, 100),  # bear
            np.random.normal(0.001, 0.01, 200),  # bull
        ]))
        ratio = compute_regime_transition_dd(strategy_ret, benchmark_ret)
        assert ratio is not None
        # 稳定策略应该放大倍数不大
        assert ratio < 5.0

    def test_regime_transition_dd_short(self):
        """短序列 → None"""
        from backtest.north_star_metrics import compute_regime_transition_dd
        ret = pd.Series(np.random.normal(0, 0.01, 50))
        bench = pd.Series(np.random.normal(0, 0.01, 50))
        result = compute_regime_transition_dd(ret, bench)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV51L3Stability -v 2>&1 | head -10`

- [ ] **Step 3: Implement**

Append to `backtest/north_star_metrics.py` (replace the V5.1 placeholder comments at the end):

```python
# ═══════════════════════════════════════════════════════
# V5.1 新增指标
# ═══════════════════════════════════════════════════════

def compute_hurst_exponent(returns: pd.Series, min_window: int = 20) -> float:
    """
    R/S法计算Hurst指数. H=0.5随机游走, H>0.5趋势持续, H<0.5均值回复.
    理想区间0.55-0.65 (mild persistence).
    """
    if returns is None or len(returns) < 100:
        return 0.5

    windows = [20, 40, 60, 80, 100, 150, 200]
    windows = [w for w in windows if w < len(returns) // 2]
    if len(windows) < 3:
        return 0.5

    rs_values = []
    for w in windows:
        rs_list = []
        for start in range(0, len(returns) - w, w):
            chunk = returns.iloc[start:start + w]
            mean_r = chunk.mean()
            deviations = (chunk - mean_r).cumsum()
            R = deviations.max() - deviations.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(w), np.log(np.mean(rs_list))))

    if len(rs_values) < 3:
        return 0.5

    x = [v[0] for v in rs_values]
    y = [v[1] for v in rs_values]
    H = np.polyfit(x, y, 1)[0]
    return float(np.clip(H, 0.0, 1.0))


def compute_regime_transition_dd(daily_returns: pd.Series,
                                  benchmark_returns: pd.Series,
                                  lookback: int = 60,
                                  pre_window: int = 10,
                                  post_window: int = 20) -> Optional[float]:
    """
    Regime转换期间的DD放大倍数.
    = max(DD在转换窗口) / 正常期间中位DD
    """
    if daily_returns is None or benchmark_returns is None:
        return None
    n = min(len(daily_returns), len(benchmark_returns))
    if n < 200:
        return None

    dr = daily_returns.values[:n]
    br = benchmark_returns.values[:n]

    # 判定regime: 60日滚动收益
    rolling_ret = pd.Series(br).rolling(lookback).sum()
    regimes = []
    for r in rolling_ret:
        if np.isnan(r):
            regimes.append('neutral')
        elif r > 0.05:
            regimes.append('bull')
        elif r < -0.05:
            regimes.append('bear')
        else:
            regimes.append('neutral')

    # 找regime变化点
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            transitions.append(i)

    if not transitions:
        return 1.0  # 无转换 → 无放大

    # 计算转换窗口内的DD
    cum_ret = (1 + pd.Series(dr)).cumprod()
    transition_dds = []
    for t in transitions:
        start = max(0, t - pre_window)
        end = min(n, t + post_window)
        if end - start < 5:
            continue
        window_cum = cum_ret.iloc[start:end]
        peak = window_cum.expanding().max()
        dd = (window_cum / peak - 1).min()
        transition_dds.append(abs(dd))

    if not transition_dds:
        return 1.0

    # 正常期间DD (排除转换窗口)
    is_transition = np.zeros(n, dtype=bool)
    for t in transitions:
        start = max(0, t - pre_window)
        end = min(n, t + post_window)
        is_transition[start:end] = True

    normal_mask = ~is_transition
    if normal_mask.sum() < 60:
        return 1.0

    # 用60日滚动窗口计算正常期间DD
    normal_dds = []
    normal_cum = cum_ret.copy()
    normal_cum[is_transition] = np.nan
    # 简化: 用全期DD作为baseline
    full_peak = cum_ret.expanding().max()
    full_dd_series = abs(cum_ret / full_peak - 1)
    normal_dd_median = full_dd_series[normal_mask].median()

    if normal_dd_median <= 0.001:
        normal_dd_median = 0.001

    max_transition_dd = max(transition_dds)
    return float(max_transition_dd / normal_dd_median)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV51L3Stability -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5.1 L3稳定性指标 — Hurst指数 + Regime转换DD"
```

---

### Task 2: Add L4 Metrics — CSCV PBO + Effective N

**Files:**
- Modify: `backtest/north_star_metrics.py` (append)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestV51L4Advanced:
    """V5.1 L4: CSCV PBO + Effective N"""

    def test_cscv_pbo_random_strategy(self):
        """随机策略 → PBO≈0.5"""
        from backtest.north_star_metrics import compute_cscv_pbo
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.02, 500))
        pbo = compute_cscv_pbo(returns, n_subperiods=16, max_combinations=200)
        assert 0.2 < pbo < 0.8, f"Random PBO={pbo}, expected ~0.5"

    def test_cscv_pbo_strong_strategy(self):
        """强策略 (稳定正收益) → PBO较低"""
        from backtest.north_star_metrics import compute_cscv_pbo
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.003, 0.01, 500))
        pbo = compute_cscv_pbo(returns, n_subperiods=16, max_combinations=200)
        assert pbo < 0.6  # 稳定正收益, PBO不应太高

    def test_cscv_pbo_short_series(self):
        """短序列 → None"""
        from backtest.north_star_metrics import compute_cscv_pbo
        returns = pd.Series(np.random.normal(0, 0.02, 100))
        assert compute_cscv_pbo(returns) is None

    def test_effective_n_uncorrelated(self):
        """不相关持仓 → N_eff ≈ N"""
        from backtest.north_star_metrics import compute_effective_n_corr
        np.random.seed(42)
        holdings = pd.DataFrame({
            f'stock_{i}': np.random.normal(0, 0.02, 100) for i in range(10)
        })
        n_eff = compute_effective_n_corr(holdings)
        assert n_eff > 5.0, f"Uncorrelated N_eff={n_eff}, expected >5"

    def test_effective_n_highly_correlated(self):
        """高相关持仓 → N_eff << N"""
        from backtest.north_star_metrics import compute_effective_n_corr
        np.random.seed(42)
        base = np.random.normal(0, 0.02, 100)
        holdings = pd.DataFrame({
            f'stock_{i}': base + np.random.normal(0, 0.003, 100) for i in range(10)
        })
        n_eff = compute_effective_n_corr(holdings)
        assert n_eff < 3.0, f"Correlated N_eff={n_eff}, expected <3"

    def test_effective_n_single_stock(self):
        """单只股票 → N_eff=1"""
        from backtest.north_star_metrics import compute_effective_n_corr
        holdings = pd.DataFrame({'stock_0': np.random.normal(0, 0.02, 100)})
        assert compute_effective_n_corr(holdings) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV51L4Advanced -v 2>&1 | head -10`

- [ ] **Step 3: Implement**

Append to `backtest/north_star_metrics.py`:

```python
def compute_cscv_pbo(daily_returns: pd.Series,
                      n_subperiods: int = 16,
                      n_variants: int = 10,
                      max_combinations: int = 1000) -> Optional[float]:
    """
    CSCV过拟合概率 (PBO). Lopez de Prado (2014).

    将回测期切为n_subperiods个子期, 随机抽样IS/OOS组合,
    用block bootstrap生成n_variants个策略变体, 计算PBO.
    """
    if daily_returns is None or len(daily_returns) < n_subperiods * 20:
        return None

    returns = daily_returns.values
    n = len(returns)
    period_len = n // n_subperiods
    if period_len < 10:
        return None

    # 切分为子期
    sub_returns = []
    for i in range(n_subperiods):
        start = i * period_len
        end = start + period_len if i < n_subperiods - 1 else n
        sub_returns.append(returns[start:end])

    # 预计算每个子期的Sharpe组件 (sum, sum_sq, count)
    sub_stats = []
    for sr in sub_returns:
        sub_stats.append({
            'sum': sr.sum(),
            'sum_sq': (sr ** 2).sum(),
            'count': len(sr),
        })

    # 生成策略变体: 原始 + block bootstrap (不同起点)
    variant_sub_stats = [sub_stats]  # 第0个是原始
    rng = np.random.RandomState(42)
    for v in range(1, n_variants):
        shift = rng.randint(1, period_len)
        shifted = np.roll(returns, shift)
        v_stats = []
        for i in range(n_subperiods):
            start = i * period_len
            end = start + period_len if i < n_subperiods - 1 else n
            chunk = shifted[start:end]
            v_stats.append({
                'sum': chunk.sum(),
                'sum_sq': (chunk ** 2).sum(),
                'count': len(chunk),
            })
        variant_sub_stats.append(v_stats)

    def sharpe_from_stats(stats_list, indices):
        total_sum = sum(stats_list[i]['sum'] for i in indices)
        total_sq = sum(stats_list[i]['sum_sq'] for i in indices)
        total_n = sum(stats_list[i]['count'] for i in indices)
        if total_n < 2:
            return 0.0
        mean = total_sum / total_n
        var = total_sq / total_n - mean ** 2
        if var <= 0:
            return 0.0
        return mean / np.sqrt(var) * np.sqrt(252)

    # 抽样IS/OOS组合
    from itertools import combinations
    import random
    half = n_subperiods // 2
    all_combos = list(combinations(range(n_subperiods), half))
    rng2 = random.Random(42)
    sampled = rng2.sample(all_combos, min(max_combinations, len(all_combos)))

    overfit_count = 0
    for is_indices in sampled:
        oos_indices = tuple(i for i in range(n_subperiods) if i not in is_indices)

        # 找IS最优变体
        best_is_variant = 0
        best_is_sharpe = -999
        for v in range(n_variants):
            s = sharpe_from_stats(variant_sub_stats[v], is_indices)
            if s > best_is_sharpe:
                best_is_sharpe = s
                best_is_variant = v

        # IS最优变体在OOS的排名
        oos_sharpes = []
        for v in range(n_variants):
            oos_sharpes.append(sharpe_from_stats(variant_sub_stats[v], oos_indices))

        best_oos = oos_sharpes[best_is_variant]
        rank = sum(1 for s in oos_sharpes if s > best_oos)  # 0-indexed rank
        if rank >= n_variants // 2:  # 排名在下半 → 过拟合
            overfit_count += 1

    return float(overfit_count / len(sampled))


def compute_effective_n_corr(holdings_returns: pd.DataFrame) -> float:
    """
    相关性调整有效N. N_eff = N / (1 + (N-1) × avg_pairwise_corr)
    """
    if holdings_returns is None or holdings_returns.empty:
        return 1.0

    N = holdings_returns.shape[1]
    if N < 2:
        return float(N)

    corr_matrix = holdings_returns.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper_corrs = corr_matrix.values[mask]
    upper_corrs = upper_corrs[~np.isnan(upper_corrs)]

    if len(upper_corrs) == 0:
        return float(N)

    avg_corr = float(np.mean(upper_corrs))
    denominator = 1 + (N - 1) * max(avg_corr, 0)
    if denominator <= 0:
        return float(N)
    return float(N / denominator)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV51L4Advanced -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5.1 L4高级OOS — CSCV过拟合概率 + 有效N"
```

---

### Task 3: Add L7 Capacity Metrics — Capacity + Participation + LA Sharpe

**Files:**
- Modify: `backtest/north_star_metrics.py` (append)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestV51L7Capacity:
    """V5.1 L7: 容量评估"""

    def test_capacity_high_liquidity(self):
        """高流动性组合 → capacity > 200M"""
        from backtest.north_star_metrics import compute_strategy_capacity
        picks = pd.DataFrame({
            'code': ['000001.SZ', '600519.SH', '000858.SZ'],
            'adv_20d_value': [5e8, 3e8, 2e8],  # 日均成交额(元)
            'daily_vol': [0.02, 0.015, 0.025],
        })
        cap = compute_strategy_capacity(picks, gross_annual_return=0.30, avg_turnover=0.3)
        assert cap > 200, f"High liquidity capacity={cap}M, expected >200"

    def test_capacity_low_liquidity(self):
        """低流动性组合 → capacity < 200M"""
        from backtest.north_star_metrics import compute_strategy_capacity
        picks = pd.DataFrame({
            'code': ['000001.SZ', '600519.SH', '000858.SZ'],
            'adv_20d_value': [5e6, 3e6, 2e6],  # 很小的日均成交额
            'daily_vol': [0.03, 0.035, 0.04],
        })
        cap = compute_strategy_capacity(picks, gross_annual_return=0.30, avg_turnover=0.3)
        assert cap < 200, f"Low liquidity capacity={cap}M, expected <200"

    def test_participation_rate(self):
        """参与率计算"""
        from backtest.north_star_metrics import compute_participation_rate_p90
        picks = pd.DataFrame({
            'code': ['A', 'B', 'C'],
            'adv_20d_value': [1e8, 5e7, 2e7],  # 日均成交额(元)
        })
        # 1亿AUM, 3只等权 → 每只3333万
        p90 = compute_participation_rate_p90(picks, assumed_aum_mn=100, n_positions=3)
        assert 0 < p90 < 1.0

    def test_liquidity_adj_sharpe(self):
        """流动性调整Sharpe应 <= 原始Sharpe"""
        from backtest.north_star_metrics import compute_liquidity_adj_sharpe
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.015, 252))
        raw_sharpe = returns.mean() / returns.std() * np.sqrt(252)
        la_sharpe = compute_liquidity_adj_sharpe(
            returns, impact_cost_annual=0.02)
        assert la_sharpe < raw_sharpe + 0.1
        assert la_sharpe > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV51L7Capacity -v 2>&1 | head -10`

- [ ] **Step 3: Implement**

Append to `backtest/north_star_metrics.py`:

```python
def compute_strategy_capacity(picks_with_volume: pd.DataFrame,
                               gross_annual_return: float,
                               avg_turnover: float,
                               eta: float = 0.15,
                               n_positions: int = 10) -> float:
    """
    Almgren-Chriss策略容量估计 (百万RMB).

    二分搜索AUM使得 total_impact_cost ≈ gross_alpha.
    impact_per_stock = σ × η × sqrt(trade_value / ADV)
    """
    if picks_with_volume is None or picks_with_volume.empty:
        return 0.0
    if gross_annual_return <= 0:
        return 0.0

    adv_values = picks_with_volume['adv_20d_value'].values  # 日均成交额(元)
    daily_vols = picks_with_volume.get('daily_vol')
    if daily_vols is None:
        daily_vols = np.full(len(adv_values), 0.025)
    else:
        daily_vols = daily_vols.values

    # 日均交易天数
    trades_per_year = 252 * avg_turnover  # 年换手×252天
    daily_trade_frac = avg_turnover  # 简化: 日均换仓比例

    def total_impact_cost(aum_yuan):
        """给定AUM, 计算年化impact cost"""
        position_value = aum_yuan / max(n_positions, len(adv_values))
        total_impact = 0.0
        for i in range(len(adv_values)):
            adv = adv_values[i]
            vol = daily_vols[i]
            if adv <= 0:
                continue
            trade_value = position_value * daily_trade_frac
            participation = trade_value / adv
            impact = vol * eta * np.sqrt(max(participation, 0))
            total_impact += impact
        # 年化: 每次调仓都有impact, 年252天
        return total_impact * 252 / max(len(adv_values), 1)

    # 二分搜索: 找AUM使impact_cost = gross_return的50%
    # (超过50%的alpha被impact吃掉 → 容量极限)
    lo, hi = 1e6, 1e11  # 100万 ~ 1000亿
    target_cost = gross_annual_return * 0.5

    for _ in range(50):
        mid = (lo + hi) / 2
        cost = total_impact_cost(mid)
        if cost < target_cost:
            lo = mid
        else:
            hi = mid

    return float(lo / 1e6)  # 转为百万


def compute_participation_rate_p90(picks_with_volume: pd.DataFrame,
                                    assumed_aum_mn: float = 100,
                                    n_positions: int = 10) -> float:
    """
    持仓参与率P90. 假设AUM等权分配到各持仓.
    participation = position_value / adv_20d_value
    """
    if picks_with_volume is None or picks_with_volume.empty:
        return 0.0

    aum_yuan = assumed_aum_mn * 1e6
    position_value = aum_yuan / max(n_positions, len(picks_with_volume))

    adv_values = picks_with_volume['adv_20d_value'].values
    participations = []
    for adv in adv_values:
        if adv > 0:
            participations.append(position_value / adv)
        else:
            participations.append(1.0)

    if not participations:
        return 0.0
    return float(np.percentile(participations, 90))


def compute_liquidity_adj_sharpe(daily_returns: pd.Series,
                                  impact_cost_annual: float = 0.02,
                                  risk_free_rate: float = 0.02) -> float:
    """
    流动性调整Sharpe. 扣除market impact后的Sharpe.
    """
    if daily_returns is None or len(daily_returns) < 20:
        return 0.0

    daily_impact = impact_cost_annual / 252
    adj_returns = daily_returns - daily_impact

    mean_r = adj_returns.mean() - risk_free_rate / 252
    std_r = adj_returns.std()
    if std_r <= 0:
        return 0.0
    return float(mean_r / std_r * np.sqrt(252))
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV51L7Capacity -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5.1 L7容量评估 — 策略容量+参与率+流动性调整Sharpe"
```

---

### Task 4: Add TARGETS_V51 + compute_v51_score

**Files:**
- Modify: `backtest/north_star_metrics.py` (append)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestV51Score:
    """V5.1评分主函数"""

    def test_46_metrics(self):
        from backtest.north_star_metrics import NORTH_STAR_TARGETS_V51
        assert len(NORTH_STAR_TARGETS_V51) == 46
        layer_counts = {}
        for info in NORTH_STAR_TARGETS_V51.values():
            l = info['layer']
            layer_counts[l] = layer_counts.get(l, 0) + 1
        assert layer_counts == {1: 10, 2: 5, 3: 9, 4: 8, 5: 5, 6: 6, 7: 3}

    def test_max_score_230(self):
        from backtest.north_star_metrics import compute_v51_score, NORTH_STAR_TARGETS_V51
        metric_values = {name: 0.01 for name in NORTH_STAR_TARGETS_V51}
        result = compute_v51_score(metric_values, n_trading_days=600)
        assert result['max_score'] == 230.0

    def test_seven_layers(self):
        from backtest.north_star_metrics import compute_v51_score, NORTH_STAR_TARGETS_V51
        metric_values = {name: 0.01 for name in NORTH_STAR_TARGETS_V51}
        result = compute_v51_score(metric_values, n_trading_days=600)
        assert len(result['layer_details']) == 7
        for lid in [1, 2, 3, 4, 5, 6, 7]:
            assert lid in result['layer_details']

    def test_perfect_scores(self):
        from backtest.north_star_metrics import compute_v51_score, NORTH_STAR_TARGETS_V51
        metric_values = {}
        for name, info in NORTH_STAR_TARGETS_V51.items():
            t = info['target']
            if info['direction'] == 'higher':
                metric_values[name] = t * 1.1 if t >= 0 else t * 0.5
            else:
                metric_values[name] = t * 0.9 if t > 0 else t * 1.1
        result = compute_v51_score(metric_values, n_trading_days=600)
        assert result['final_pct'] >= 99.0
        assert result['grade'] == 'S'

    def test_weights_sum_to_100(self):
        from backtest.north_star_metrics import V51_LAYER_WEIGHTS
        assert abs(sum(V51_LAYER_WEIGHTS.values()) - 1.0) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV51Score -v 2>&1 | head -10`

- [ ] **Step 3: Implement**

Append to `backtest/north_star_metrics.py`:

```python
# ── V5.1 目标定义 ──

V51_LAYER_NAMES = {
    1: '信号质量', 2: '组合效率', 3: '风险控制',
    4: 'OOS鲁棒性', 5: '超额收益', 6: '因子归因', 7: '容量可扩展',
}

V51_LAYER_WEIGHTS = {
    1: 0.25, 2: 0.12, 3: 0.18, 4: 0.15, 5: 0.08, 6: 0.08, 7: 0.14,
}

NORTH_STAR_TARGETS_V51 = dict(NORTH_STAR_TARGETS_V5)  # 继承V5全部39个
NORTH_STAR_TARGETS_V51.update({
    # ── L3 新增: 稳定性 (2项) ──
    'hurst_deviation': {
        'pass': 0.15, 'ok': 0.10, 'good': 0.07, 'great': 0.05, 'target': 0.02,
        'direction': 'lower', 'layer': 3, 'display': 'Hurst偏差',
        'min_days': 200,
    },
    'regime_transition_dd': {
        'pass': 3.0, 'ok': 2.5, 'good': 2.0, 'great': 1.5, 'target': 1.0,
        'direction': 'lower', 'layer': 3, 'display': 'Regime转换DD',
        'min_days': 200,
    },

    # ── L4 新增: 高级OOS (2项) ──
    'cscv_pbo': {
        'pass': 0.50, 'ok': 0.40, 'good': 0.25, 'great': 0.15, 'target': 0.05,
        'direction': 'lower', 'layer': 4, 'display': 'CSCV PBO',
        'min_days': 320,
    },
    'effective_n_corr': {
        'pass': 2.0, 'ok': 3.0, 'good': 4.0, 'great': 6.0, 'target': 8.0,
        'direction': 'higher', 'layer': 4, 'display': '有效N(相关调整)',
    },

    # ── L7 容量可扩展 (3项) ──
    'strategy_capacity_mn': {
        'pass': 50, 'ok': 200, 'good': 500, 'great': 1000, 'target': 5000,
        'direction': 'higher', 'layer': 7, 'display': '策略容量(百万)',
    },
    'participation_rate_p90': {
        'pass': 0.10, 'ok': 0.05, 'good': 0.03, 'great': 0.02, 'target': 0.01,
        'direction': 'lower', 'layer': 7, 'display': '参与率P90',
    },
    'liquidity_adj_sharpe': {
        'pass': 0.5, 'ok': 0.8, 'good': 1.0, 'great': 1.5, 'target': 2.0,
        'direction': 'higher', 'layer': 7, 'display': '流动性调整Sharpe',
    },
})


def compute_v51_score(metric_values: Dict[str, float],
                       n_trading_days: int = 500,
                       n_trials: int = 10) -> Dict:
    """
    V5.1评分: 7层46指标, 满分230, 连续插值.
    """
    layer_scores = {i: 0.0 for i in range(1, 8)}
    layer_maxes = {i: 0.0 for i in range(1, 8)}
    metric_scores = {}

    for metric_name, target_info in NORTH_STAR_TARGETS_V51.items():
        layer = target_info['layer']
        layer_maxes[layer] += 5.0

        value = metric_values.get(metric_name)
        if value is None:
            metric_scores[metric_name] = (0.0, '░' * 20, None)
            continue

        score = score_metric_v5(value, target_info)
        layer_scores[layer] += score

        filled = int(score / 5.0 * 20)
        bar = '█' * filled + '░' * (20 - filled)
        metric_scores[metric_name] = (score, bar, value)

    weighted_pct = 0.0
    layer_details = {}
    for layer in range(1, 8):
        lmax = layer_maxes[layer]
        lscore = layer_scores[layer]
        lpct = lscore / lmax * 100 if lmax > 0 else 0
        weight = V51_LAYER_WEIGHTS[layer]
        weighted_pct += lpct * weight
        layer_details[layer] = {
            'score': lscore, 'max': lmax,
            'pct': lpct, 'weight': weight,
        }

    length_factor = compute_backtest_length_factor_v5(n_trading_days)
    final_pct = weighted_pct * length_factor

    total_score = sum(layer_scores.values())
    max_score = sum(layer_maxes.values())

    grade = compute_v5_grade(final_pct)

    return {
        'total_score': total_score,
        'max_score': max_score,
        'raw_pct': weighted_pct,
        'length_factor': length_factor,
        'final_pct': final_pct,
        'grade': grade,
        'layer_details': layer_details,
        'metric_scores': metric_scores,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV51Score -v`
Expected: 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5.1目标定义(46指标/7层) + compute_v51_score"
```

---

### Task 5: Integrate V5.1 into backtest pipeline + CLI

**Files:**
- Modify: `backtest/backtest_report_based.py` (add imports, _print_scorecard_v51, V51 metrics computation)
- Modify: `backtest/run_north_star_eval.py` (add v51 to choices, add --assumed-aum)

- [ ] **Step 1: Add V5.1 imports to backtest_report_based.py**

Add to the existing V5 import block (around lines 76-84):

```python
from backtest.north_star_metrics import (
    compute_v51_score, NORTH_STAR_TARGETS_V51, V51_LAYER_NAMES, V51_LAYER_WEIGHTS,
    compute_hurst_exponent, compute_regime_transition_dd,
    compute_cscv_pbo, compute_effective_n_corr,
    compute_strategy_capacity, compute_participation_rate_p90,
    compute_liquidity_adj_sharpe,
)
```

- [ ] **Step 2: Add V5.1 metric computation in run_single_backtest**

After the V5 metric computation section, add V5.1 metrics:

```python
# ── V5.1 新增指标计算 ──

# L3: Hurst
if daily_returns is not None and len(daily_returns) >= 200:
    hurst = compute_hurst_exponent(daily_returns)
    s['hurst_deviation'] = abs(hurst - 0.60)

# L3: Regime Transition DD
if daily_returns is not None and benchmark_returns is not None and len(daily_returns) >= 200:
    rtdd = compute_regime_transition_dd(daily_returns, benchmark_returns)
    if rtdd is not None:
        s['regime_transition_dd'] = rtdd

# L4: CSCV PBO
if daily_returns is not None and len(daily_returns) >= 320:
    pbo = compute_cscv_pbo(daily_returns, max_combinations=500)
    if pbo is not None:
        s['cscv_pbo'] = pbo

# L4: Effective N (from holdings returns if available)
# Populated from trade records in backtest loop
s.setdefault('effective_n_corr', top_n * 1.0)  # 默认=N (无相关数据时)

# L7: Capacity metrics (need volume data)
try:
    import sqlite3
    conn = sqlite3.connect(DB_PATH, timeout=30)
    # 获取持仓股票的日均成交额和波动率
    if 'portfolio_codes' in s and s['portfolio_codes']:
        codes = s['portfolio_codes']
        placeholders = ','.join('?' * len(codes))
        vol_df = pd.read_sql(f"""
            SELECT s.code,
                   AVG(dq.volume * dq.close) as adv_20d_value,
                   STDEV(dq.price_change_pct) as daily_vol
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.code IN ({placeholders})
              AND dq.trade_date >= ?
            GROUP BY s.code
        """, conn, params=list(codes) + [start_d])
        if not vol_df.empty:
            s['strategy_capacity_mn'] = compute_strategy_capacity(
                vol_df, s.get('annual_return', 0.2), s.get('annual_turnover', 30) / 100,
                n_positions=top_n)
            s['participation_rate_p90'] = compute_participation_rate_p90(
                vol_df, assumed_aum_mn=100, n_positions=top_n)
    conn.close()
except Exception as e:
    logger.warning(f"Capacity metrics failed: {e}")

# L7: Liquidity-adjusted Sharpe
if daily_returns is not None and 'strategy_capacity_mn' in s:
    # 粗略估算impact cost: capacity越小, impact越大
    cap = max(s.get('strategy_capacity_mn', 500), 1)
    impact_pct = min(0.10, 100 / cap * 0.01)  # 简化: 100M/capacity * 1%
    s['liquidity_adj_sharpe'] = compute_liquidity_adj_sharpe(
        daily_returns, impact_cost_annual=impact_pct)
```

- [ ] **Step 3: Add _print_scorecard_v51 function**

Add after `_print_scorecard_v5`. Follow the exact same pattern as V5 but use `NORTH_STAR_TARGETS_V51`, `V51_LAYER_NAMES`, `V51_LAYER_WEIGHTS`, `compute_v51_score`.

Key differences from V5:
- Title: "北极星评分卡 V5.1"
- 7 layers instead of 6
- V5.1 new metrics marked with "V5.1"
- L7 section header includes "(AUM=100M假设)"

- [ ] **Step 4: Wire V5.1 scorecard call**

After the V5 scorecard call, add:

```python
if score_version in ('v51', 'all'):
    _print_scorecard_v51(s, label, focus_days, n_trading_days, n_trials=n_trials)
```

- [ ] **Step 5: Update CLI**

In `backtest/run_north_star_eval.py` line 611-612, update choices:

```python
parser.add_argument('--score-version', type=str, default='both',
                    choices=['v2', 'v3', 'v4', 'v5', 'v51', 'both', 'all'],
                    help='评分卡版本: v2/v3/v4/v5/v51/both(v2+v4)/all(全部)')
```

Add after --wf-summary:

```python
parser.add_argument('--assumed-aum', type=float, default=100,
                    help='V5.1容量评估假设AUM (百万RMB, default: 100)')
```

- [ ] **Step 6: Test with real data**

Run: `cd /Users/yangxu/StockTradebyZ && python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v4.7.5 --score-version v51 --top-n 10 --focus-days 10 2>&1 | grep -A 5 'V5.1'`
Expected: V5.1 scorecard with 7 layers including L7

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py -v --tb=short 2>&1 | tail -20`
Expected: All tests pass (previous 40 + new ~16 = ~56)

- [ ] **Step 8: Commit**

```bash
git add backtest/backtest_report_based.py backtest/run_north_star_eval.py
git commit -m "feat: V5.1评分卡集成 — 7层46指标 + CLI --score-version v51"
```

---

### Task 6: Validation — V5.1 against real reports

**Files:** No new files

- [ ] **Step 1: V5 regression test (V5 scores should be unchanged)**

Run: `cd /Users/yangxu/StockTradebyZ && python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v4.7.5 --score-version v5 --top-n 10 --focus-days 10 2>&1 | grep '加权评分'`
Expected: Same V5 score as before (64.7%)

- [ ] **Step 2: Run V5.1 full evaluation**

Run: `cd /Users/yangxu/StockTradebyZ && python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v4.7.5 --score-version v51 --top-n 10 --focus-days 10 2>&1 | grep -A 80 '北极星评分卡 V5.1'`

- [ ] **Step 3: Run all-version comparison**

Run: `cd /Users/yangxu/StockTradebyZ && python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v4.7.5 --score-version all --top-n 10 --focus-days 10 2>&1 | grep '加权评分\|最终'`

- [ ] **Step 4: Full test suite**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py -v --tb=short`
Expected: All ~56 tests pass

- [ ] **Step 5: Final commit**

```bash
git add -A && git commit -m "feat: 北极星V5.1完整实现 — 7层46指标(容量+CSCV+Hurst)"
```
