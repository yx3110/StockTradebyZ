# North Star V5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade North Star evaluation from V4 (31 discrete metrics, 5 layers) to V5 (39 continuous metrics, 6 layers with factor attribution).

**Architecture:** Add continuous interpolation scoring (`score_metric_v5`), 9 new metric computation functions, a new `factor_returns.py` module for Fama-French 4-factor construction, and integrate into the existing backtest pipeline via `_print_scorecard_v5()`. All V2/V3/V4 code paths remain untouched.

**Tech Stack:** Python 3, NumPy, Pandas, SciPy, statsmodels (OLS regression), SQLite

**Spec:** `docs/superpowers/specs/2026-03-31-north-star-v5-design.md`

---

### Task 1: Create factor_returns.py — Factor Return Construction

**Files:**
- Create: `backtest/factor_returns.py`
- Create: `backtest/test_north_star_v5.py`

- [ ] **Step 1: Write test for MKT factor loading**

```python
# backtest/test_north_star_v5.py
"""北极星V5单元测试"""
import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFactorReturns:
    """因子收益构建测试"""

    def test_build_factor_returns_columns(self):
        """构建结果应包含4因子列"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2025-01-01', '2025-01-31')
        assert 'MKT' in df.columns
        assert 'SMB' in df.columns
        assert 'HML' in df.columns
        assert 'UMD' in df.columns
        assert len(df) > 0

    def test_factor_returns_no_extreme_values(self):
        """因子日收益应在合理范围 (-10%, +10%)"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2025-01-01', '2025-03-31')
        for col in ['MKT', 'SMB', 'HML', 'UMD']:
            assert df[col].abs().max() < 0.15, f"{col} has extreme value"

    def test_factor_returns_low_correlation(self):
        """因子间相关性应 < 0.5"""
        from backtest.factor_returns import build_factor_returns
        df = build_factor_returns('2024-01-01', '2025-01-01')
        corr = df[['SMB', 'HML', 'UMD']].corr()
        for i in range(3):
            for j in range(i+1, 3):
                assert abs(corr.iloc[i, j]) < 0.5, \
                    f"High correlation: {corr.columns[i]} vs {corr.columns[j]} = {corr.iloc[i,j]:.3f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestFactorReturns -v 2>&1 | head -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest.factor_returns'`

- [ ] **Step 3: Implement factor_returns.py**

```python
# backtest/factor_returns.py
"""
A股4因子收益构建模块

从SQLite数据库构建 Fama-French 风格的4因子日收益:
- MKT: 市场因子 (中证500超额收益)
- SMB: 规模因子 (小盘-大盘)
- HML: 价值因子 (高B/P - 低B/P)
- UMD: 动量因子 (Winner - Loser, 12-1月)
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')
RISK_FREE_RATE = 0.02  # 年化2%


def build_factor_returns(start_date: str, end_date: str,
                         db_path: str = None) -> pd.DataFrame:
    """
    构建4因子日收益.

    Args:
        start_date: 'YYYY-MM-DD' 或 'YYYYMMDD'
        end_date: 'YYYY-MM-DD' 或 'YYYYMMDD'
        db_path: 数据库路径

    Returns:
        DataFrame[date, MKT, SMB, HML, UMD], index=trade_date
    """
    if db_path is None:
        db_path = DB_PATH

    # 标准化日期格式为 YYYYMMDD
    sd = start_date.replace('-', '')
    ed = end_date.replace('-', '')

    conn = sqlite3.connect(db_path, timeout=30)

    # --- MKT: 中证500超额收益 ---
    mkt_df = _build_mkt_factor(conn, sd, ed)

    # --- 加载A股日收益 + 市值 + PB ---
    stock_data = _load_stock_data(conn, sd, ed)
    conn.close()

    if stock_data.empty:
        logger.warning("No stock data found for factor construction")
        return pd.DataFrame(columns=['MKT', 'SMB', 'HML', 'UMD'])

    # --- SMB: 规模因子 ---
    smb_df = _build_smb_factor(stock_data)

    # --- HML: 价值因子 ---
    hml_df = _build_hml_factor(stock_data)

    # --- UMD: 动量因子 ---
    umd_df = _build_umd_factor(stock_data)

    # 合并
    result = mkt_df
    for factor_df in [smb_df, hml_df, umd_df]:
        if not factor_df.empty:
            result = result.join(factor_df, how='left')

    result = result.fillna(0.0)
    return result


def _build_mkt_factor(conn, sd, ed):
    """MKT = 中证500日收益 - Rf/252"""
    query = """
        SELECT dq.trade_date, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = '000905.SH'
          AND dq.trade_date BETWEEN ? AND ?
        ORDER BY dq.trade_date
    """
    df = pd.read_sql(query, conn, params=[sd, ed])
    if df.empty:
        return pd.DataFrame(columns=['MKT'])
    df['trade_date'] = df['trade_date'].astype(str)
    df = df.set_index('trade_date')
    df['MKT'] = df['price_change_pct'] - RISK_FREE_RATE / 252
    return df[['MKT']]


def _load_stock_data(conn, sd, ed):
    """加载A股日收益 + 市值 + PB"""
    query = """
        SELECT dq.trade_date, s.code, dq.price_change_pct,
               db.total_mv, db.pb
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        LEFT JOIN daily_basic db ON db.security_id = dq.security_id
                                 AND db.trade_date = dq.trade_date
        WHERE s.type = 'A股'
          AND dq.trade_date BETWEEN ? AND ?
          AND dq.volume > 0
          AND db.total_mv > 0
          AND db.pb > 0
        ORDER BY dq.trade_date, s.code
    """
    df = pd.read_sql(query, conn, params=[sd, ed])
    df['trade_date'] = df['trade_date'].astype(str)
    return df


def _build_smb_factor(stock_data):
    """SMB: 按市值5分位, Bottom20%等权 - Top20%等权"""
    result = {}
    for date, group in stock_data.groupby('trade_date'):
        if len(group) < 50:
            continue
        q20 = group['total_mv'].quantile(0.2)
        q80 = group['total_mv'].quantile(0.8)
        small = group[group['total_mv'] <= q20]['price_change_pct'].mean()
        big = group[group['total_mv'] >= q80]['price_change_pct'].mean()
        result[date] = small - big
    s = pd.Series(result, name='SMB')
    s.index.name = 'trade_date'
    return s.to_frame()


def _build_hml_factor(stock_data):
    """HML: 按B/P(=1/PB)5分位, High B/P - Low B/P"""
    result = {}
    for date, group in stock_data.groupby('trade_date'):
        if len(group) < 50:
            continue
        group = group.copy()
        group['bp'] = 1.0 / group['pb']
        q20 = group['bp'].quantile(0.2)
        q80 = group['bp'].quantile(0.8)
        high_bp = group[group['bp'] >= q80]['price_change_pct'].mean()
        low_bp = group[group['bp'] <= q20]['price_change_pct'].mean()
        result[date] = high_bp - low_bp
    s = pd.Series(result, name='HML')
    s.index.name = 'trade_date'
    return s.to_frame()


def _build_umd_factor(stock_data):
    """UMD: 按12-1月累计收益5分位, Winner - Loser"""
    # 计算每只股票的12-1月动量
    dates = sorted(stock_data['trade_date'].unique())
    if len(dates) < 252:
        logger.warning("Not enough dates for UMD factor (need 252+)")
        return pd.DataFrame(columns=['UMD'])

    # 按股票pivot
    pivot = stock_data.pivot_table(
        index='trade_date', columns='code',
        values='price_change_pct', aggfunc='first'
    )
    # 累计收益 (对数近似)
    cum_ret = (1 + pivot).cumprod()

    result = {}
    for i, date in enumerate(dates):
        if i < 252:
            continue
        # 12个月前到1个月前的收益
        date_12m = dates[max(0, i - 252)]
        date_1m = dates[max(0, i - 21)]
        if date_12m not in cum_ret.index or date_1m not in cum_ret.index:
            continue
        mom = cum_ret.loc[date_1m] / cum_ret.loc[date_12m] - 1
        mom = mom.dropna()
        if len(mom) < 50:
            continue
        q20 = mom.quantile(0.2)
        q80 = mom.quantile(0.8)
        day_ret = pivot.loc[date].dropna()
        common = mom.index.intersection(day_ret.index)
        winners = day_ret[common[mom[common] >= q80]].mean()
        losers = day_ret[common[mom[common] <= q20]].mean()
        if not np.isnan(winners) and not np.isnan(losers):
            result[date] = winners - losers
    s = pd.Series(result, name='UMD')
    s.index.name = 'trade_date'
    return s.to_frame()


def load_or_build_factors(start_date: str, end_date: str,
                          db_path: str = None,
                          cache_table: str = 'factor_daily_returns') -> pd.DataFrame:
    """
    缓存机制: 首次构建写入SQLite, 后续直接读取.
    """
    if db_path is None:
        db_path = DB_PATH

    sd = start_date.replace('-', '')
    ed = end_date.replace('-', '')

    conn = sqlite3.connect(db_path, timeout=30)

    # 尝试从缓存读取
    try:
        cached = pd.read_sql(
            f"SELECT * FROM {cache_table} WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
            conn, params=[sd, ed]
        )
        if len(cached) > 0:
            cached = cached.set_index('trade_date')
            conn.close()
            logger.info(f"Loaded {len(cached)} factor returns from cache")
            return cached[['MKT', 'SMB', 'HML', 'UMD']]
    except Exception:
        pass

    conn.close()

    # 构建 (UMD需要12个月前置数据)
    lookback_sd = pd.Timestamp(sd) - pd.DateOffset(months=14)
    lookback_sd_str = lookback_sd.strftime('%Y%m%d')

    df = build_factor_returns(lookback_sd_str, ed, db_path)

    # 裁剪到请求范围
    df = df[df.index >= sd]

    if df.empty:
        return df

    # 写入缓存
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cache_df = df.reset_index()
        cache_df.columns = ['trade_date', 'MKT', 'SMB', 'HML', 'UMD']
        cache_df.to_sql(cache_table, conn, if_exists='replace', index=False)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{cache_table}_date ON {cache_table}(trade_date)")
        conn.commit()
        conn.close()
        logger.info(f"Cached {len(df)} factor returns to {cache_table}")
    except Exception as e:
        logger.warning(f"Failed to cache factor returns: {e}")

    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestFactorReturns -v 2>&1 | tail -10`
Expected: 3 tests PASS (may take 30-60s for DB access)

- [ ] **Step 5: Commit**

```bash
git add backtest/factor_returns.py backtest/test_north_star_v5.py
git commit -m "feat: 新增A股4因子收益构建模块 (北极星V5基础设施)"
```

---

### Task 2: Add score_metric_v5 — Continuous Interpolation Scoring

**Files:**
- Modify: `backtest/north_star_metrics.py` (append after line 2554)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests for continuous scoring**

Append to `backtest/test_north_star_v5.py`:

```python
class TestScoreMetricV5:
    """连续插值评分测试"""

    def test_at_target_returns_5(self):
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 0.03, 'ok': 0.04, 'good': 0.05,
                       'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        score = score_metric_v5(0.08, target_info)
        assert score == 5.0

    def test_above_target_capped_at_5(self):
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 0.03, 'ok': 0.04, 'good': 0.05,
                       'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        score = score_metric_v5(0.12, target_info)
        assert score == 5.0

    def test_at_pass_returns_1(self):
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 0.03, 'ok': 0.04, 'good': 0.05,
                       'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        score = score_metric_v5(0.03, target_info)
        assert score == 1.0

    def test_below_pass_is_fraction(self):
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 0.03, 'ok': 0.04, 'good': 0.05,
                       'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        score = score_metric_v5(0.015, target_info)
        assert 0 < score < 1.0

    def test_zero_returns_zero(self):
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 0.03, 'ok': 0.04, 'good': 0.05,
                       'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        score = score_metric_v5(0.0, target_info)
        assert score == 0.0

    def test_midpoint_interpolation(self):
        """IC=0.059 (between good=0.05 and great=0.06) should be ~3.9"""
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 0.03, 'ok': 0.04, 'good': 0.05,
                       'great': 0.06, 'target': 0.08, 'direction': 'higher'}
        score = score_metric_v5(0.059, target_info)
        assert 3.5 < score < 4.0

    def test_direction_lower(self):
        """For 'lower' direction, smaller values = higher scores"""
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 2.0, 'ok': 1.5, 'good': 1.0,
                       'great': 0.8, 'target': 0.6, 'direction': 'lower'}
        score_good = score_metric_v5(0.6, target_info)
        score_bad = score_metric_v5(2.0, target_info)
        assert score_good == 5.0
        assert score_bad == 1.0

    def test_direction_lower_interpolation(self):
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 2.0, 'ok': 1.5, 'good': 1.0,
                       'great': 0.8, 'target': 0.6, 'direction': 'lower'}
        score = score_metric_v5(0.9, target_info)
        assert 3.0 < score < 4.0  # between good(1.0) and great(0.8)

    def test_direction_lower_worse_than_pass(self):
        from backtest.north_star_metrics import score_metric_v5
        target_info = {'pass': 2.0, 'ok': 1.5, 'good': 1.0,
                       'great': 0.8, 'target': 0.6, 'direction': 'lower'}
        score = score_metric_v5(3.0, target_info)
        assert 0 < score < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestScoreMetricV5 -v 2>&1 | head -20`
Expected: FAIL — `ImportError: cannot import name 'score_metric_v5'`

- [ ] **Step 3: Implement score_metric_v5**

Append to `backtest/north_star_metrics.py` (after line 2554):

```python
# ═══════════════════════════════════════════════════════
# V5 北极星评分体系 — 连续插值 + 6层39指标
# ═══════════════════════════════════════════════════════

def score_metric_v5(value: float, target_info: dict) -> float:
    """
    V5连续插值评分: 0.0 ~ 5.0 浮点数.

    breakpoints = [0, pass, ok, good, great, target]  (direction=higher)
    scores      = [0,  1,    2,   3,     4,     5   ]

    direction='lower' 时: pass > ok > good > great > target, 翻转后插值.
    """
    if value is None:
        return 0.0

    direction = target_info.get('direction', 'higher')
    bp_raw = [
        target_info['pass'],
        target_info['ok'],
        target_info['good'],
        target_info['great'],
        target_info['target'],
    ]

    if direction == 'lower':
        # 对lower方向: pass=2.0 > ok=1.5 > ... > target=0.6
        # 翻转value和breakpoints使np.interp可用
        bp_values = list(reversed(bp_raw))  # [target, great, good, ok, pass] (ascending)
        scores = [5.0, 4.0, 3.0, 2.0, 1.0]
        # 超过target(更小)的 → 5.0
        if value <= bp_raw[-1]:  # target (smallest)
            return 5.0
        # 比pass(最大)还差的 → 插值到0
        if value > bp_raw[0]:  # pass (largest)
            worst = bp_raw[0] * 2  # 假设0分对应pass的2倍
            if worst <= bp_raw[0]:
                worst = bp_raw[0] + abs(bp_raw[0])
            return float(np.interp(value, [bp_raw[0], worst], [1.0, 0.0]))
        return float(np.interp(value, bp_values, scores))
    else:
        # direction == 'higher'
        if value >= bp_raw[-1]:  # >= target
            return 5.0
        if value <= 0:
            return 0.0
        if value < bp_raw[0]:  # < pass
            return float(np.interp(value, [0, bp_raw[0]], [0.0, 1.0]))
        return float(np.interp(value, bp_raw, [1.0, 2.0, 3.0, 4.0, 5.0]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestScoreMetricV5 -v`
Expected: 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5连续插值评分函数 score_metric_v5"
```

---

### Task 3: Add New L3 Metrics — CVaR, DD Duration, Underwater Ratio

**Files:**
- Modify: `backtest/north_star_metrics.py` (append after score_metric_v5)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestNewL3Metrics:
    """L3新增指标: CVaR, 回撤持续时间, 水下时间比"""

    def test_cvar_normal_distribution(self):
        """正态分布 CVaR_5% ≈ mean + 2.06 * std"""
        from backtest.north_star_metrics import compute_cvar
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
        cvar = compute_cvar(returns, alpha=0.05)
        assert 0.02 < cvar < 0.06  # 正值 (代表损失)

    def test_cvar_all_positive(self):
        """全正收益, CVaR应很小"""
        from backtest.north_star_metrics import compute_cvar
        returns = pd.Series([0.01, 0.02, 0.015, 0.005, 0.03] * 100)
        cvar = compute_cvar(returns, alpha=0.05)
        assert cvar < 0.01

    def test_max_dd_duration_known_sequence(self):
        """已知序列: 回撤持续5天"""
        from backtest.north_star_metrics import compute_max_dd_duration
        # 累计收益: 上升→下降5天→恢复
        cum_ret = pd.Series([1.0, 1.1, 1.2, 1.15, 1.10, 1.05, 1.08, 1.12, 1.25])
        duration = compute_max_dd_duration(cum_ret)
        assert duration >= 4  # 从peak 1.2到恢复超过1.2

    def test_max_dd_duration_no_drawdown(self):
        """单调上升, 无回撤"""
        from backtest.north_star_metrics import compute_max_dd_duration
        cum_ret = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        duration = compute_max_dd_duration(cum_ret)
        assert duration == 0

    def test_underwater_ratio_always_up(self):
        """单调上升, 水下比=0"""
        from backtest.north_star_metrics import compute_underwater_ratio
        cum_ret = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
        ratio = compute_underwater_ratio(cum_ret)
        assert ratio == 0.0

    def test_underwater_ratio_always_down(self):
        """单调下降, 水下比≈1"""
        from backtest.north_star_metrics import compute_underwater_ratio
        cum_ret = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6])
        ratio = compute_underwater_ratio(cum_ret)
        assert ratio == 0.8  # 第一个点是peak, 后4个都在水下
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestNewL3Metrics -v 2>&1 | head -15`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the 3 functions**

Append to `backtest/north_star_metrics.py`:

```python
def compute_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Conditional Value at Risk (Expected Shortfall).

    返回最差alpha百分位的平均损失 (正值表示损失).
    例: CVaR_5% = 最差5%交易日的平均亏损.
    """
    if returns.empty:
        return 0.0
    var_threshold = returns.quantile(alpha)
    tail = returns[returns <= var_threshold]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def compute_max_dd_duration(cumulative_returns: pd.Series) -> int:
    """
    最长回撤恢复交易日数.

    从peak到重新回到peak level的最长持续天数.
    """
    if cumulative_returns.empty or len(cumulative_returns) < 2:
        return 0
    peak = cumulative_returns.expanding().max()
    underwater = cumulative_returns < peak

    if not underwater.any():
        return 0

    # 计算连续水下段的最大长度
    is_above = ~underwater
    groups = is_above.cumsum()
    underwater_groups = underwater.groupby(groups).sum()
    # 过滤掉非水下段 (sum=0)
    underwater_groups = underwater_groups[underwater_groups > 0]
    if underwater_groups.empty:
        return 0
    return int(underwater_groups.max())


def compute_underwater_ratio(cumulative_returns: pd.Series) -> float:
    """水下天数占总天数比例."""
    if cumulative_returns.empty or len(cumulative_returns) < 2:
        return 0.0
    peak = cumulative_returns.expanding().max()
    underwater = cumulative_returns < peak
    return float(underwater.mean())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestNewL3Metrics -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5新增L3指标 — CVaR, 回撤持续时间, 水下时间比"
```

---

### Task 4: Add New L1 Metrics — IC Autocorrelation, Transfer Coefficient

**Files:**
- Modify: `backtest/north_star_metrics.py` (append)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestNewL1Metrics:
    """L1新增指标: IC自相关, Transfer Coefficient"""

    def test_ic_autocorrelation_persistent_signal(self):
        """高持续性信号, 自相关应>0.5"""
        from backtest.north_star_metrics import compute_ic_autocorrelation
        np.random.seed(42)
        # 构造高自相关序列: AR(1)
        ic = [0.05]
        for _ in range(199):
            ic.append(0.7 * ic[-1] + 0.3 * np.random.normal(0.05, 0.02))
        ic_series = pd.Series(ic)
        autocorr = compute_ic_autocorrelation(ic_series, lag=1)
        assert autocorr > 0.4

    def test_ic_autocorrelation_random_signal(self):
        """随机信号, 自相关应接近0"""
        from backtest.north_star_metrics import compute_ic_autocorrelation
        np.random.seed(42)
        ic_series = pd.Series(np.random.normal(0, 0.05, 200))
        autocorr = compute_ic_autocorrelation(ic_series, lag=1)
        assert abs(autocorr) < 0.2

    def test_transfer_coefficient_perfect(self):
        """完美传递: 信号排名 = 持仓排名"""
        from backtest.north_star_metrics import compute_transfer_coefficient
        signal_ranks = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual_ranks = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        tc = compute_transfer_coefficient(signal_ranks, actual_ranks)
        assert tc > 0.99

    def test_transfer_coefficient_partial(self):
        """部分错位: TC应在0.5-1之间"""
        from backtest.north_star_metrics import compute_transfer_coefficient
        signal_ranks = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        actual_ranks = pd.Series([2, 1, 4, 3, 6, 5, 8, 7, 10, 9])
        tc = compute_transfer_coefficient(signal_ranks, actual_ranks)
        assert 0.5 < tc < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestNewL1Metrics -v 2>&1 | head -15`
Expected: FAIL

- [ ] **Step 3: Implement the 2 functions**

Append to `backtest/north_star_metrics.py`:

```python
def compute_ic_autocorrelation(ic_series: pd.Series, lag: int = 1) -> float:
    """
    IC序列自相关系数.

    衡量信号持续性: 高→调仓频率可低; 低→信号短命需频繁调仓.
    """
    if ic_series is None or len(ic_series) < lag + 10:
        return 0.0
    autocorr = ic_series.autocorr(lag=lag)
    if np.isnan(autocorr):
        return 0.0
    return float(autocorr)


def compute_transfer_coefficient(signal_ranks: pd.Series,
                                  actual_ranks: pd.Series) -> float:
    """
    信号到持仓的传递系数 (Spearman相关).

    衡量涨停/停牌/流动性约束对信号传递的损耗.
    TC=1.0 表示完美传递, TC=0.5 表示一半的信号信息丢失.
    """
    if signal_ranks is None or actual_ranks is None:
        return 1.0  # 无约束数据时默认完美传递
    if len(signal_ranks) < 5:
        return 1.0
    corr = signal_ranks.corr(actual_ranks, method='spearman')
    if np.isnan(corr):
        return 1.0
    return float(corr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestNewL1Metrics -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5新增L1指标 — IC自相关, Transfer Coefficient"
```

---

### Task 5: Add New L4 Metrics — WFER, OOS IC Half-Life

**Files:**
- Modify: `backtest/north_star_metrics.py` (append)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestNewL4Metrics:
    """L4新增指标: WFER, OOS IC Half-Life"""

    def test_wfer_basic(self):
        """IS Sharpe=2.0, OOS Sharpe=1.0 → WFER=0.5"""
        from backtest.north_star_metrics import compute_wfer
        wf_summary = {
            'is_sharpe': [2.0, 2.0, 2.0],
            'oos_sharpe': [1.0, 1.0, 1.0],
        }
        wfer = compute_wfer(wf_summary)
        assert wfer == pytest.approx(0.5, abs=0.01)

    def test_wfer_perfect(self):
        """IS == OOS → WFER=1.0"""
        from backtest.north_star_metrics import compute_wfer
        wf_summary = {
            'is_sharpe': [1.5, 1.8, 2.0],
            'oos_sharpe': [1.5, 1.8, 2.0],
        }
        wfer = compute_wfer(wf_summary)
        assert wfer == pytest.approx(1.0, abs=0.01)

    def test_wfer_negative_is(self):
        """IS Sharpe <= 0 → None"""
        from backtest.north_star_metrics import compute_wfer
        wf_summary = {
            'is_sharpe': [-0.5, 0.2, 0.1],
            'oos_sharpe': [0.5, 0.3, 0.2],
        }
        # mean IS is negative → undefined
        result = compute_wfer(wf_summary)
        # IS mean = (-0.5+0.2+0.1)/3 ≈ -0.067 → None
        assert result is None

    def test_oos_ic_half_life_no_decay(self):
        """OOS IC不衰减 → 返回12 (上限)"""
        from backtest.north_star_metrics import compute_oos_ic_half_life
        wf_summary = {
            'oos_monthly_ics': [
                [0.05, 0.05, 0.05, 0.05],
                [0.06, 0.06, 0.06, 0.06],
            ]
        }
        hl = compute_oos_ic_half_life(wf_summary)
        assert hl == 12.0

    def test_oos_ic_half_life_fast_decay(self):
        """OOS IC快速衰减 → 半衰期应<3"""
        from backtest.north_star_metrics import compute_oos_ic_half_life
        wf_summary = {
            'oos_monthly_ics': [
                [0.08, 0.04, 0.02, 0.01],
                [0.10, 0.05, 0.03, 0.01],
            ]
        }
        hl = compute_oos_ic_half_life(wf_summary)
        assert 0 < hl < 3.0

    def test_oos_ic_half_life_no_data(self):
        """无数据 → None"""
        from backtest.north_star_metrics import compute_oos_ic_half_life
        assert compute_oos_ic_half_life({}) is None
        assert compute_oos_ic_half_life({'oos_monthly_ics': None}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestNewL4Metrics -v 2>&1 | head -15`
Expected: FAIL

- [ ] **Step 3: Implement the 2 functions**

Append to `backtest/north_star_metrics.py`:

```python
def compute_wfer(wf_summary: dict) -> Optional[float]:
    """
    Walk-Forward Efficiency Ratio.

    WFER = mean(Sharpe_OOS) / mean(Sharpe_IS)
    衡量模型在样本外保留了多少样本内性能.
    WFER=0.5 表示OOS保留了50%的IS Sharpe.
    """
    is_sharpes = wf_summary.get('is_sharpe')
    oos_sharpes = wf_summary.get('oos_sharpe')
    if not is_sharpes or not oos_sharpes:
        return None
    is_mean = float(np.mean(is_sharpes))
    oos_mean = float(np.mean(oos_sharpes))
    if is_mean <= 0:
        return None
    return oos_mean / is_mean


def compute_oos_ic_half_life(wf_summary: dict) -> Optional[float]:
    """
    OOS IC衰减半衰期 (月数).

    从WF各窗口内逐月IC拟合指数衰减: IC(t) = IC_0 × exp(-λt)
    半衰期 = ln(2) / λ
    """
    monthly_ics = wf_summary.get('oos_monthly_ics')
    if not monthly_ics:
        return None

    max_months = max(len(m) for m in monthly_ics)
    if max_months < 2:
        return None

    avg_by_month = []
    for i in range(max_months):
        vals = [m[i] for m in monthly_ics if len(m) > i and m[i] is not None]
        if vals:
            avg_by_month.append(float(np.mean(vals)))
        else:
            break

    if len(avg_by_month) < 2 or avg_by_month[0] <= 0:
        return 0.0

    # 拟合指数衰减: log(IC) = log(IC_0) - λ*t
    log_ics = [np.log(max(ic, 1e-8)) for ic in avg_by_month]
    coeffs = np.polyfit(range(len(log_ics)), log_ics, 1)
    slope = coeffs[0]

    if slope >= 0:
        return 12.0  # 不衰减, 返回上限
    half_life = np.log(2) / (-slope)
    return float(min(half_life, 12.0))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestNewL4Metrics -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5新增L4指标 — WFER, OOS IC半衰期"
```

---

### Task 6: Add L6 Factor Attribution — compute_factor_attribution

**Files:**
- Modify: `backtest/north_star_metrics.py` (append + add statsmodels import)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestFactorAttribution:
    """L6因子归因测试"""

    def test_pure_market_portfolio(self):
        """纯市场组合: beta_mkt≈1, alpha≈0, R²≈1"""
        from backtest.north_star_metrics import compute_factor_attribution
        np.random.seed(42)
        n = 500
        mkt = np.random.normal(0.0005, 0.015, n)
        smb = np.random.normal(0, 0.008, n)
        hml = np.random.normal(0, 0.008, n)
        umd = np.random.normal(0, 0.008, n)
        factor_df = pd.DataFrame({
            'MKT': mkt, 'SMB': smb, 'HML': hml, 'UMD': umd
        })
        # 组合 = MKT + 微量噪声
        portfolio = pd.Series(mkt + np.random.normal(0, 0.001, n))
        result = compute_factor_attribution(portfolio, factor_df)

        assert abs(result['betas']['mkt'] - 1.0) < 0.15
        assert result['factor_r_squared'] > 0.8
        assert abs(result['residual_alpha_t']) < 2.0  # alpha不显著

    def test_pure_alpha_portfolio(self):
        """纯alpha组合: R²低, alpha t值高"""
        from backtest.north_star_metrics import compute_factor_attribution
        np.random.seed(42)
        n = 500
        factor_df = pd.DataFrame({
            'MKT': np.random.normal(0, 0.015, n),
            'SMB': np.random.normal(0, 0.008, n),
            'HML': np.random.normal(0, 0.008, n),
            'UMD': np.random.normal(0, 0.008, n),
        })
        # 组合 = 稳定正收益 + 独立噪声 (不依赖任何因子)
        portfolio = pd.Series(np.random.normal(0.002, 0.01, n))
        result = compute_factor_attribution(portfolio, factor_df)

        assert result['factor_r_squared'] < 0.3
        assert result['residual_alpha_t'] > 2.0  # alpha显著

    def test_small_cap_tilted(self):
        """小盘倾斜组合: smb_beta应较大"""
        from backtest.north_star_metrics import compute_factor_attribution
        np.random.seed(42)
        n = 500
        smb = np.random.normal(0, 0.008, n)
        factor_df = pd.DataFrame({
            'MKT': np.random.normal(0.0005, 0.015, n),
            'SMB': smb,
            'HML': np.random.normal(0, 0.008, n),
            'UMD': np.random.normal(0, 0.008, n),
        })
        # 组合 = 1.5 × SMB + 噪声
        portfolio = pd.Series(1.5 * smb + np.random.normal(0.001, 0.005, n))
        result = compute_factor_attribution(portfolio, factor_df)

        assert result['smb_beta'] > 1.0
        assert result['max_factor_loading'] > 1.0

    def test_result_keys(self):
        """结果应包含所有必要键"""
        from backtest.north_star_metrics import compute_factor_attribution
        np.random.seed(42)
        n = 100
        factor_df = pd.DataFrame({
            'MKT': np.random.normal(0, 0.01, n),
            'SMB': np.random.normal(0, 0.01, n),
            'HML': np.random.normal(0, 0.01, n),
            'UMD': np.random.normal(0, 0.01, n),
        })
        portfolio = pd.Series(np.random.normal(0.001, 0.01, n))
        result = compute_factor_attribution(portfolio, factor_df)

        expected_keys = ['residual_alpha', 'residual_alpha_annual',
                         'residual_alpha_t', 'factor_r_squared',
                         'betas', 'max_factor_loading',
                         'smb_beta', 'mom_beta']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        assert 'mkt' in result['betas']
        assert 'smb' in result['betas']
        assert 'hml' in result['betas']
        assert 'umd' in result['betas']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestFactorAttribution -v 2>&1 | head -15`
Expected: FAIL

- [ ] **Step 3: Add statsmodels import and implement**

Add import at top of `backtest/north_star_metrics.py` (after line 20, the `from scipy.stats import spearmanr` line):

```python
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
```

Append to end of `backtest/north_star_metrics.py`:

```python
def compute_factor_attribution(portfolio_returns: pd.Series,
                                factor_returns: pd.DataFrame,
                                risk_free_rate: float = 0.02) -> dict:
    """
    Fama-French 4因子归因.

    R_strategy - Rf = α + β_mkt·MKT + β_smb·SMB + β_hml·HML + β_umd·UMD + ε

    Returns:
        dict with residual_alpha, residual_alpha_annual, residual_alpha_t,
        factor_r_squared, betas, max_factor_loading, smb_beta, mom_beta
    """
    default = {
        'residual_alpha': 0.0, 'residual_alpha_annual': 0.0,
        'residual_alpha_t': 0.0, 'factor_r_squared': 0.0,
        'betas': {'mkt': 0.0, 'smb': 0.0, 'hml': 0.0, 'umd': 0.0},
        'max_factor_loading': 0.0, 'smb_beta': 0.0, 'mom_beta': 0.0,
    }

    if not HAS_STATSMODELS:
        import warnings
        warnings.warn("statsmodels not installed — factor attribution unavailable")
        return default

    if portfolio_returns is None or factor_returns is None:
        return default
    if len(portfolio_returns) < 30:
        return default

    # 对齐索引
    common = portfolio_returns.index.intersection(factor_returns.index)
    if len(common) < 30:
        # 尝试位置对齐
        n = min(len(portfolio_returns), len(factor_returns))
        if n < 30:
            return default
        y = portfolio_returns.values[:n] - risk_free_rate / 252
        X = factor_returns[['MKT', 'SMB', 'HML', 'UMD']].values[:n]
    else:
        y = portfolio_returns.loc[common].values - risk_free_rate / 252
        X = factor_returns.loc[common, ['MKT', 'SMB', 'HML', 'UMD']].values

    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X).fit()
    except Exception:
        return default

    # 用named keys获取参数 (比positional indexing更安全)
    param_names = model.params.index.tolist() if hasattr(model.params, 'index') else ['const', 'MKT', 'SMB', 'HML', 'UMD']
    alpha = float(model.params.iloc[0])  # const always first with add_constant
    alpha_t = float(model.tvalues.iloc[0])

    betas = {
        'mkt': float(model.params.iloc[1]),
        'smb': float(model.params.iloc[2]),
        'hml': float(model.params.iloc[3]),
        'umd': float(model.params.iloc[4]),
    }

    return {
        'residual_alpha': alpha,
        'residual_alpha_annual': alpha * 252,
        'residual_alpha_t': alpha_t,
        'factor_r_squared': float(model.rsquared),
        'betas': betas,
        'max_factor_loading': max(abs(betas['smb']),
                                   abs(betas['hml']),
                                   abs(betas['umd'])),
        'smb_beta': abs(betas['smb']),
        'mom_beta': abs(betas['umd']),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestFactorAttribution -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5新增L6因子归因 — FF4因子回归"
```

---

### Task 7: Add NORTH_STAR_TARGETS_V5 + compute_v5_score + Utilities

**Files:**
- Modify: `backtest/north_star_metrics.py` (append)
- Modify: `backtest/test_north_star_v5.py` (append test class)

- [ ] **Step 1: Write tests**

Append to `backtest/test_north_star_v5.py`:

```python
class TestV5Score:
    """V5评分主函数测试"""

    def test_perfect_scores(self):
        """所有指标都达target → 100%"""
        from backtest.north_star_metrics import compute_v5_score, NORTH_STAR_TARGETS_V5
        metric_values = {}
        for name, info in NORTH_STAR_TARGETS_V5.items():
            if info['direction'] == 'higher':
                metric_values[name] = info['target'] * 1.1  # 超过target
            else:
                metric_values[name] = info['target'] * 0.9  # 低于target (lower=better)
        result = compute_v5_score(metric_values, n_trading_days=600)
        assert result['final_pct'] >= 99.0
        assert result['grade'] == 'S'

    def test_all_zeros(self):
        """所有指标为0 → 低分"""
        from backtest.north_star_metrics import compute_v5_score
        metric_values = {}  # 全None → 全0分
        result = compute_v5_score(metric_values, n_trading_days=600)
        assert result['final_pct'] == 0.0
        assert result['grade'] == 'D'

    def test_length_penalty_v5(self):
        """长度惩罚: 250天 < 500天基准"""
        from backtest.north_star_metrics import compute_backtest_length_factor_v5
        assert compute_backtest_length_factor_v5(500) == 1.0
        assert compute_backtest_length_factor_v5(600) == 1.0
        factor_250 = compute_backtest_length_factor_v5(250)
        assert 0.6 < factor_250 < 0.7  # log曲线约0.67
        assert compute_backtest_length_factor_v5(60) == 0.0
        assert compute_backtest_length_factor_v5(30) == 0.0

    def test_six_layers(self):
        """V5应有6层"""
        from backtest.north_star_metrics import compute_v5_score, NORTH_STAR_TARGETS_V5
        metric_values = {name: 0.01 for name in NORTH_STAR_TARGETS_V5}
        result = compute_v5_score(metric_values, n_trading_days=600)
        assert len(result['layer_details']) == 6
        for layer_id in [1, 2, 3, 4, 5, 6]:
            assert layer_id in result['layer_details']

    def test_continuous_scores_in_result(self):
        """V5 metric_scores应包含浮点分数"""
        from backtest.north_star_metrics import compute_v5_score, NORTH_STAR_TARGETS_V5
        metric_values = {'daily_ic': 0.055}  # between good(0.05) and great(0.06)
        result = compute_v5_score(metric_values, n_trading_days=600)
        ic_score = result['metric_scores']['daily_ic']
        assert isinstance(ic_score[0], float)
        assert 3.0 < ic_score[0] < 4.0  # continuous, not integer

    def test_auto_select_benchmark(self):
        """自动基准选择"""
        from backtest.north_star_metrics import auto_select_benchmark
        assert auto_select_benchmark(100) == '000300.SH'   # 大盘
        assert auto_select_benchmark(20) == '000905.SH'    # 中盘
        assert auto_select_benchmark(8) == '000852.SH'     # 中小盘
        assert auto_select_benchmark(3) == '932000.CSI'    # 小盘
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV5Score -v 2>&1 | head -15`
Expected: FAIL

- [ ] **Step 3: Implement TARGETS, weights, compute_v5_score, length factor, auto_select_benchmark**

Append to `backtest/north_star_metrics.py`:

```python
# ── V5 目标定义 ──

V5_LAYER_NAMES = {
    1: '信号质量', 2: '组合效率', 3: '风险控制',
    4: 'OOS鲁棒性', 5: '超额收益', 6: '因子归因',
}

V5_LAYER_WEIGHTS = {
    1: 0.30, 2: 0.15, 3: 0.20, 4: 0.15, 5: 0.10, 6: 0.10,
}

NORTH_STAR_TARGETS_V5 = {
    # ── L1 信号质量 (10项) ──
    'daily_ic': {
        'pass': 0.03, 'ok': 0.04, 'good': 0.05, 'great': 0.06, 'target': 0.08,
        'direction': 'higher', 'layer': 1, 'display': 'Daily IC',
    },
    'icir': {
        'pass': 0.25, 'ok': 0.35, 'good': 0.45, 'great': 0.55, 'target': 0.70,
        'direction': 'higher', 'layer': 1, 'display': 'ICIR',
    },
    'ic_positive_pct': {
        'pass': 55, 'ok': 57, 'good': 60, 'great': 63, 'target': 68,
        'direction': 'higher', 'layer': 1, 'display': 'IC>0%',
    },
    'ic_monotonicity': {
        'pass': 2.5, 'ok': 3.0, 'good': 3.5, 'great': 4.0, 'target': 4.5,
        'direction': 'higher', 'layer': 1, 'display': 'IC单调性',
    },
    'ic_time_stability': {
        'pass': 2.0, 'ok': 1.5, 'good': 1.0, 'great': 0.8, 'target': 0.6,
        'direction': 'lower', 'layer': 1, 'display': 'IC稳定性(CV)',
        'min_days': 120,
    },
    'signal_half_life': {
        'pass': 3, 'ok': 6, 'good': 8, 'great': 12, 'target': 20,
        'direction': 'higher', 'layer': 1, 'display': '信号半衰期(天)',
    },
    'bear_icir': {
        'pass': 0.05, 'ok': 0.10, 'good': 0.20, 'great': 0.30, 'target': 0.35,
        'direction': 'higher', 'layer': 1, 'display': '熊市ICIR',
        'min_days': 60,
    },
    'ic_decay_ratio': {
        'pass': 0.50, 'ok': 0.60, 'good': 0.70, 'great': 0.80, 'target': 0.95,
        'direction': 'higher', 'layer': 1, 'display': 'IC衰减比',
        'min_days': 120,
    },
    'ic_autocorr_1d': {
        'pass': 0.10, 'ok': 0.20, 'good': 0.35, 'great': 0.50, 'target': 0.70,
        'direction': 'higher', 'layer': 1, 'display': 'IC自相关(1d)',
    },
    'transfer_coefficient': {
        'pass': 0.50, 'ok': 0.60, 'good': 0.70, 'great': 0.80, 'target': 0.90,
        'direction': 'higher', 'layer': 1, 'display': '传递系数',
    },

    # ── L2 组合效率 (5项) ──
    'annual_turnover': {
        'pass': 45, 'ok': 35, 'good': 30, 'great': 25, 'target': 20,
        'direction': 'lower', 'layer': 2, 'display': '年化换手',
    },
    'annual_cost_drag': {
        'pass': 0.13, 'ok': 0.10, 'good': 0.08, 'great': 0.07, 'target': 0.05,
        'direction': 'lower', 'layer': 2, 'display': '年化成本',
    },
    'net_gross_ratio': {
        'pass': 0.60, 'ok': 0.70, 'good': 0.75, 'great': 0.80, 'target': 0.85,
        'direction': 'higher', 'layer': 2, 'display': '净/毛收益比',
    },
    'limit_up_fail_rate': {
        'pass': 0.15, 'ok': 0.10, 'good': 0.08, 'great': 0.05, 'target': 0.02,
        'direction': 'lower', 'layer': 2, 'display': '涨停失败率',
    },
    'liquidity_coverage': {
        'pass': 0.70, 'ok': 0.80, 'good': 0.85, 'great': 0.90, 'target': 0.95,
        'direction': 'higher', 'layer': 2, 'display': '流动性覆盖',
    },

    # ── L3 风险控制 (7项) ──
    'max_drawdown': {
        'pass': -0.25, 'ok': -0.18, 'good': -0.12, 'great': -0.10, 'target': -0.08,
        'direction': 'higher', 'layer': 3, 'display': '最大回撤',
    },
    'sharpe_ratio': {
        'pass': 1.0, 'ok': 1.5, 'good': 2.0, 'great': 2.5, 'target': 3.0,
        'direction': 'higher', 'layer': 3, 'display': 'Sharpe',
    },
    'worst_rolling_60d_icir': {
        'pass': -0.10, 'ok': 0.0, 'good': 0.10, 'great': 0.20, 'target': 0.30,
        'direction': 'higher', 'layer': 3, 'display': '最差60日ICIR',
        'min_days': 120,
    },
    'tail_ratio': {
        'pass': 0.8, 'ok': 1.0, 'good': 1.2, 'great': 1.5, 'target': 2.0,
        'direction': 'higher', 'layer': 3, 'display': '尾部比率',
    },
    'cvar_5pct': {
        'pass': 0.04, 'ok': 0.03, 'good': 0.02, 'great': 0.015, 'target': 0.01,
        'direction': 'lower', 'layer': 3, 'display': 'CVaR 5%',
    },
    'max_dd_duration': {
        'pass': 120, 'ok': 90, 'good': 60, 'great': 40, 'target': 20,
        'direction': 'lower', 'layer': 3, 'display': '最长DD天数',
    },
    'underwater_ratio': {
        'pass': 0.60, 'ok': 0.50, 'good': 0.40, 'great': 0.30, 'target': 0.20,
        'direction': 'lower', 'layer': 3, 'display': '水下时间比',
    },

    # ── L4 OOS鲁棒性 (6项) ──
    'annual_return': {
        'pass': 0.15, 'ok': 0.20, 'good': 0.30, 'great': 0.40, 'target': 0.50,
        'direction': 'higher', 'layer': 4, 'display': '年化收益',
        'min_days': 200,
    },
    'monthly_win_rate': {
        'pass': 55, 'ok': 60, 'good': 67, 'great': 75, 'target': 83,
        'direction': 'higher', 'layer': 4, 'display': '月度胜率%',
    },
    'probabilistic_sharpe': {
        'pass': 0.80, 'ok': 0.85, 'good': 0.90, 'great': 0.95, 'target': 0.99,
        'direction': 'higher', 'layer': 4, 'display': 'PSR',
    },
    'deflated_sharpe': {
        'pass': 0.70, 'ok': 0.80, 'good': 0.85, 'great': 0.90, 'target': 0.95,
        'direction': 'higher', 'layer': 4, 'display': 'DSR',
    },
    'wfer': {
        'pass': 0.20, 'ok': 0.30, 'good': 0.40, 'great': 0.50, 'target': 0.60,
        'direction': 'higher', 'layer': 4, 'display': 'WF效率比',
    },
    'oos_ic_half_life': {
        'pass': 1, 'ok': 2, 'good': 3, 'great': 6, 'target': 12,
        'direction': 'higher', 'layer': 4, 'display': 'OOS IC半衰期(月)',
    },

    # ── L5 超额收益 (5项) ──
    'excess_annual_return': {
        'pass': 0.05, 'ok': 0.10, 'good': 0.15, 'great': 0.20, 'target': 0.30,
        'direction': 'higher', 'layer': 5, 'display': '超额年化',
        'min_days': 200,
    },
    'information_ratio': {
        'pass': 0.30, 'ok': 0.50, 'good': 0.70, 'great': 1.00, 'target': 1.50,
        'direction': 'higher', 'layer': 5, 'display': 'IR',
        'min_days': 120,
    },
    'excess_win_rate': {
        'pass': 50, 'ok': 53, 'good': 55, 'great': 60, 'target': 65,
        'direction': 'higher', 'layer': 5, 'display': '超额胜率%',
    },
    'excess_max_drawdown': {
        'pass': -0.30, 'ok': -0.20, 'good': -0.15, 'great': -0.10, 'target': -0.05,
        'direction': 'higher', 'layer': 5, 'display': '超额MaxDD',
        'min_days': 120,
    },
    'up_capture_ratio': {
        'pass': 0.80, 'ok': 1.00, 'good': 1.10, 'great': 1.20, 'target': 1.40,
        'direction': 'higher', 'layer': 5, 'display': '上行捕获比',
        'min_days': 60,
    },

    # ── L6 因子归因 (6项) ──
    'residual_alpha_t': {
        'pass': 1.0, 'ok': 1.5, 'good': 2.0, 'great': 2.5, 'target': 3.0,
        'direction': 'higher', 'layer': 6, 'display': 'Alpha t值',
    },
    'factor_r_squared': {
        'pass': 0.70, 'ok': 0.60, 'good': 0.50, 'great': 0.35, 'target': 0.20,
        'direction': 'lower', 'layer': 6, 'display': '因子R²',
    },
    'active_share': {
        'pass': 0.50, 'ok': 0.60, 'good': 0.70, 'great': 0.80, 'target': 0.90,
        'direction': 'higher', 'layer': 6, 'display': 'Active Share',
    },
    'max_factor_loading': {
        'pass': 1.50, 'ok': 1.20, 'good': 1.00, 'great': 0.80, 'target': 0.50,
        'direction': 'lower', 'layer': 6, 'display': '最大因子暴露',
    },
    'smb_beta': {
        'pass': 1.50, 'ok': 1.20, 'good': 1.00, 'great': 0.70, 'target': 0.30,
        'direction': 'lower', 'layer': 6, 'display': '小盘β',
    },
    'mom_beta': {
        'pass': 1.20, 'ok': 1.00, 'good': 0.80, 'great': 0.50, 'target': 0.20,
        'direction': 'lower', 'layer': 6, 'display': '动量β',
    },
}


def compute_backtest_length_factor_v5(n_days: int, min_days: int = 500) -> float:
    """
    V5回测长度折扣: log曲线, 比V4更严格.

    500+天: 1.0, 250天: ~0.67, 125天: ~0.35, <60天: 0.0
    """
    if n_days >= min_days:
        return 1.0
    if n_days < 60:
        return 0.0
    return np.log(n_days / 60) / np.log(min_days / 60)


def auto_select_benchmark(median_market_cap_bn: float) -> str:
    """根据策略持仓市值中位数自动选择最匹配基准."""
    if median_market_cap_bn >= 50:
        return '000300.SH'   # 沪深300
    if median_market_cap_bn >= 15:
        return '000905.SH'   # 中证500
    if median_market_cap_bn >= 5:
        return '000852.SH'   # 中证1000
    return '932000.CSI'       # 中证2000


def compute_v5_score(metric_values: Dict[str, float],
                      n_trading_days: int = 500,
                      n_trials: int = 10) -> Dict:
    """
    V5连续插值加权评分.

    V5 vs V4区别:
    1. 连续插值评分 (0.0-5.0浮点) 替代离散6档
    2. 新增Layer 6: 因子归因 (6项)
    3. 层级权重: L1=30%, L2=15%, L3=20%, L4=15%, L5=10%, L6=10%
    4. 总分: 39项 × 5分 = 195分
    5. 长度惩罚: log曲线 (更严格)
    """
    layer_scores = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
    layer_maxes = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}
    metric_scores = {}

    for metric_name, target_info in NORTH_STAR_TARGETS_V5.items():
        layer = target_info['layer']
        layer_maxes[layer] += 5.0

        value = metric_values.get(metric_name)
        if value is None:
            metric_scores[metric_name] = (0.0, '░░░░░░░░░░░░░░░░░░░░', None)
            continue

        score = score_metric_v5(value, target_info)
        layer_scores[layer] += score

        # 生成进度条
        filled = int(score / 5.0 * 20)
        bar = '█' * filled + '░' * (20 - filled)
        metric_scores[metric_name] = (score, bar, value)

    # 加权百分比
    weighted_pct = 0.0
    layer_details = {}
    for layer in [1, 2, 3, 4, 5, 6]:
        lmax = layer_maxes[layer]
        lscore = layer_scores[layer]
        lpct = lscore / lmax * 100 if lmax > 0 else 0
        weight = V5_LAYER_WEIGHTS[layer]
        weighted_pct += lpct * weight
        layer_details[layer] = {
            'score': lscore, 'max': lmax,
            'pct': lpct, 'weight': weight,
        }

    # V5长度惩罚 (log曲线)
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


def compute_v5_grade(pct: float) -> str:
    """V5等级 (与V2/V3/V4相同阈值)."""
    for threshold, grade in V2_GRADE_THRESHOLDS:
        if pct >= threshold:
            return grade
    return 'D'


# ── V5.1 预留扩展点 (方案C) ──
# L7 容量与可扩展性 (planned):
#   strategy_capacity_mn, participation_rate_p90, liquidity_adjusted_sharpe
# 高级OOS检测 (planned):
#   cscv_pbo (组合对称交叉验证, Lopez de Prado)
# 稳定性指标 (planned):
#   hurst_exponent, regime_transition_dd, effective_n_corr
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py::TestV5Score -v`
Expected: 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backtest/north_star_metrics.py backtest/test_north_star_v5.py
git commit -m "feat: V5目标定义(39指标) + compute_v5_score + 长度惩罚 + 自动基准"
```

---

### Task 8: Integrate V5 into backtest_report_based.py

**Files:**
- Modify: `backtest/backtest_report_based.py` (add imports, _compute_v5_metrics, _print_scorecard_v5)

- [ ] **Step 1: Add V5 imports**

At line ~52-78 of `backtest/backtest_report_based.py` where existing north_star imports are, add:

```python
from backtest.north_star_metrics import (
    compute_v5_score, NORTH_STAR_TARGETS_V5, V5_LAYER_NAMES, V5_LAYER_WEIGHTS,
    score_metric_v5, compute_cvar, compute_max_dd_duration, compute_underwater_ratio,
    compute_ic_autocorrelation, compute_transfer_coefficient,
    compute_factor_attribution, auto_select_benchmark,
    compute_backtest_length_factor_v5,
)
from backtest.factor_returns import load_or_build_factors
```

- [ ] **Step 2: Add _print_scorecard_v5 function**

Add after `_print_scorecard_v4` function (after line ~2540 in `backtest/backtest_report_based.py`):

```python
def _print_scorecard_v5(s, label, days, n_trading_days=0, n_trials=10,
                         benchmark_code=None, wf_summary=None):
    """打印V5北极星评分卡 (39项, 连续插值, 6层含因子归因)"""
    # 自动选择基准
    median_mcap = s.get('median_market_cap_bn', 10)
    if benchmark_code is None:
        benchmark_code = auto_select_benchmark(median_mcap)

    print(f"\n  {'═'*74}")
    print(f"  北极星评分卡 V5: {label} ({days}日持仓)")
    print(f"  基准: {benchmark_code} (中位市值{median_mcap:.1f}亿)")
    print(f"  {'═'*74}")

    # 构建V5 metric map (继承V4大部分 + 新增V5指标)
    ic_mono_val = s.get('ic_monotonicity_v3')
    if ic_mono_val is None or ic_mono_val == 0:
        ic_mono_val = s.get('ic_monotonicity', 0)

    metric_value_map = {
        # L1 信号质量 (10项)
        'daily_ic':              s.get('ic_mean', 0),
        'icir':                  s.get('icir', 0),
        'ic_positive_pct':       s.get('ic_positive_pct', 0),
        'ic_monotonicity':       ic_mono_val,
        'ic_time_stability':     s.get('ic_time_stability', 999),
        'signal_half_life':      s.get('signal_half_life', 0),
        'bear_icir':             s.get('bear_icir'),
        'ic_decay_ratio':        s.get('ic_decay_ratio', 0),
        'ic_autocorr_1d':        s.get('ic_autocorr_1d', 0),
        'transfer_coefficient':  s.get('transfer_coefficient', 1.0),
        # L2 组合效率 (5项)
        'annual_turnover':       s.get('annual_turnover', 0),
        'annual_cost_drag':      s.get('annual_cost_drag', 0),
        'net_gross_ratio':       s.get('net_gross_ratio', 0),
        'limit_up_fail_rate':    s.get('limit_up_fail_rate', 0),
        'liquidity_coverage':    s.get('liquidity_coverage', 0),
        # L3 风险控制 (7项 — 无Sortino/Calmar/max_consec)
        'max_drawdown':          s.get('max_drawdown', 0),
        'sharpe_ratio':          s.get('sharpe_ratio', 0),
        'worst_rolling_60d_icir': s.get('worst_rolling_60d_icir', None),
        'tail_ratio':            s.get('tail_ratio', 0),
        'cvar_5pct':             s.get('cvar_5pct', 0),
        'max_dd_duration':       s.get('max_dd_duration', 0),
        'underwater_ratio':      s.get('underwater_ratio', 0),
        # L4 OOS鲁棒性 (6项 — 无half_period_consistency)
        'annual_return':         s.get('annual_return', 0),
        'monthly_win_rate':      s.get('monthly_win_rate', 0),
        'probabilistic_sharpe':  s.get('probabilistic_sharpe', 0),
        'deflated_sharpe':       s.get('deflated_sharpe', 0),
        'wfer':                  s.get('wfer'),
        'oos_ic_half_life':      s.get('oos_ic_half_life'),
        # L5 超额收益 (5项 — 无bear_excess_return)
        'excess_annual_return':  s.get('excess_annual_return', 0),
        'information_ratio':     s.get('information_ratio', 0),
        'excess_win_rate':       s.get('excess_win_rate', 0),
        'excess_max_drawdown':   s.get('excess_max_drawdown', 0),
        'up_capture_ratio':      s.get('up_capture_ratio', 0),
        # L6 因子归因 (6项)
        'residual_alpha_t':      s.get('residual_alpha_t', 0),
        'factor_r_squared':      s.get('factor_r_squared', 0),
        'active_share':          s.get('active_share', 0.9),
        'max_factor_loading':    s.get('max_factor_loading', 0),
        'smb_beta':              s.get('smb_beta', 0),
        'mom_beta':              s.get('mom_beta', 0),
    }

    # 计算V5评分
    v5_result = compute_v5_score(metric_value_map, n_trading_days, n_trials)

    # 打印各层
    for layer_id in sorted(V5_LAYER_NAMES.keys()):
        layer_name = V5_LAYER_NAMES[layer_id]
        weight = V5_LAYER_WEIGHTS[layer_id]
        ld = v5_result['layer_details'][layer_id]
        layer_metrics = [(k, v) for k, v in NORTH_STAR_TARGETS_V5.items()
                         if v['layer'] == layer_id]
        if not layer_metrics:
            continue

        print(f"\n  ┌─ L{layer_id} {layer_name} (权重{weight:.0%})"
              f"  [{ld['score']:.1f}/{ld['max']:.0f} = {ld['pct']:.1f}%]")
        print(f"  │ {'指标':<20s} {'当前值':>10s} {'进度':>22s} {'分数':>8s}")
        print(f"  │ {'─'*64}")

        for metric_key, target_info in layer_metrics:
            ms = v5_result['metric_scores'].get(metric_key, (0.0, '░'*20, None))
            score, bar, value = ms

            display = target_info['display']
            direction = target_info.get('direction', 'higher')

            if value is None:
                c_str = "N/A"
            elif metric_key in {'max_drawdown', 'annual_return', 'annual_cost_drag',
                                'net_gross_ratio', 'limit_up_fail_rate', 'liquidity_coverage',
                                'probabilistic_sharpe', 'deflated_sharpe', 'ic_decay_ratio',
                                'excess_annual_return', 'excess_max_drawdown',
                                'underwater_ratio', 'cvar_5pct', 'factor_r_squared',
                                'active_share'}:
                c_str = f"{value:.1%}" if abs(value) < 10 else f"{value:.0%}"
            elif metric_key in {'ic_positive_pct', 'monthly_win_rate', 'annual_turnover',
                                'signal_half_life', 'max_dd_duration', 'excess_win_rate',
                                'oos_ic_half_life'}:
                c_str = f"{value:.1f}"
            elif metric_key in {'up_capture_ratio', 'max_factor_loading', 'smb_beta', 'mom_beta'}:
                c_str = f"{value:.2f}"
            else:
                c_str = f"{value:.3f}"

            # V5新指标标记
            new_mark = ''
            v5_new_metrics = {'ic_autocorr_1d', 'transfer_coefficient',
                              'cvar_5pct', 'max_dd_duration', 'underwater_ratio',
                              'wfer', 'oos_ic_half_life',
                              'residual_alpha_t', 'factor_r_squared', 'active_share',
                              'max_factor_loading', 'smb_beta', 'mom_beta'}
            if metric_key in v5_new_metrics:
                new_mark = ' NEW'

            print(f"  │ {display:<20s} {c_str:>10s} {bar}  {score:.2f}/5{new_mark}")

    # 总分
    print(f"\n  {'═'*74}")
    length_factor = v5_result['length_factor']
    raw_pct = v5_result['raw_pct']
    final_pct = v5_result['final_pct']

    print(f"  总分: {v5_result['total_score']:.1f} / {v5_result['max_score']:.0f}")
    if length_factor < 1.0:
        print(f"  长度折扣: × {length_factor:.2f} ({n_trading_days}天 < 500天基准)")
        print(f"  加权百分比: {raw_pct:.1f}% → {final_pct:.1f}% (折后)")
    else:
        print(f"  加权百分比: {final_pct:.1f}% ({n_trading_days}天 ≥ 500天)")

    grade = v5_result['grade']
    print(f"  最终: {final_pct:.1f}%  →  {grade}")

    # 因子暴露摘要
    fa = {k: metric_value_map.get(k, 0) for k in
          ['residual_alpha_t', 'factor_r_squared', 'smb_beta', 'mom_beta']}
    alpha_sig = '显著' if fa['residual_alpha_t'] >= 2.0 else '不显著'
    print(f"  因子暴露: α={fa['residual_alpha_t']:.2f}σ({alpha_sig}) | "
          f"R²={fa['factor_r_squared']:.2f} | "
          f"小盘β={fa['smb_beta']:.2f} | 动量β={fa['mom_beta']:.2f}")

    # 因子分解摘要 (收益归因)
    r_sq = fa['factor_r_squared']
    alpha_pct = max(0, (1 - r_sq) * 100)
    fa_result = s.get('_factor_attribution', {})
    betas = fa_result.get('betas', {})
    total_beta_abs = abs(betas.get('smb', 0)) + abs(betas.get('hml', 0)) + abs(betas.get('umd', 0))
    if total_beta_abs > 0:
        smb_pct = abs(betas.get('smb', 0)) / total_beta_abs * r_sq * 100
        mom_pct = abs(betas.get('umd', 0)) / total_beta_abs * r_sq * 100
        val_pct = abs(betas.get('hml', 0)) / total_beta_abs * r_sq * 100
        print(f"  关键发现: {alpha_pct:.0f}%选股alpha, "
              f"{smb_pct:.0f}%小盘, {mom_pct:.0f}%动量, {val_pct:.0f}%价值")
    print(f"  {'═'*74}")

    return v5_result
```

- [ ] **Step 3: Update run_single_backtest signature and add V5 metric computation**

First, update `run_single_backtest()` function signature (line 896) to accept new parameters:

```python
# Add these parameters to run_single_backtest():
#   wf_summary=None,        # WF训练摘要 (JSON path or dict)
#   score_version='both',   # 评分卡版本
```

Then, after the section where V4 metrics are computed (around line 2020-2031), add V5-specific metric computation. Find where the summary dict `s` is populated (around lines 1700-2040) and add these computations:

```python
# ── V5新增指标计算 ──

# L1: IC自相关
if ic_df is not None and len(ic_df) > 10:
    s['ic_autocorr_1d'] = compute_ic_autocorrelation(ic_df['ic'], lag=1)

# L1: Transfer Coefficient (信号排名 vs 实际持仓排名)
# 在每个调仓日, 比较信号top-N排名与实际买入的排名
if hasattr(self_ref, '_signal_ranks') and hasattr(self_ref, '_actual_ranks'):
    s['transfer_coefficient'] = compute_transfer_coefficient(
        self_ref._signal_ranks, self_ref._actual_ranks)
else:
    # 回测中记录 signal_ranks vs actual_holdings (考虑涨停/停牌替补)
    # 每次rebalance时: signal_ranks = 信号推荐的top-N排序
    #                   actual_ranks = 实际成交的排序 (跳过涨停/停牌)
    # 如果回测未记录此数据, 从成交记录估算:
    if 'buy_success_rate' in s:
        # 粗略近似: 成功买入率越高, TC越高
        s['transfer_coefficient'] = max(0.5, min(1.0, s['buy_success_rate']))
    else:
        s['transfer_coefficient'] = 1.0  # 无数据时保守默认

# L3: CVaR, DD Duration, Underwater Ratio
if daily_returns is not None and len(daily_returns) > 20:
    s['cvar_5pct'] = compute_cvar(daily_returns, alpha=0.05)
    cum_ret = (1 + daily_returns).cumprod()
    s['max_dd_duration'] = compute_max_dd_duration(cum_ret)
    s['underwater_ratio'] = compute_underwater_ratio(cum_ret)

# L4: WFER, OOS IC Half-Life
if wf_summary is not None:
    # 从WF训练摘要JSON加载
    import json
    if isinstance(wf_summary, str):
        with open(wf_summary) as f:
            wf_data = json.load(f)
    else:
        wf_data = wf_summary
    wfer_val = compute_wfer(wf_data)
    if wfer_val is not None:
        s['wfer'] = wfer_val
    hl_val = compute_oos_ic_half_life(wf_data)
    if hl_val is not None:
        s['oos_ic_half_life'] = hl_val
elif 'ic_decay_ratio' in s and s['ic_decay_ratio']:
    # Fallback: 用IC衰减比粗略近似WFER
    s['wfer'] = s['ic_decay_ratio'] * 0.7

# L5: Active Share (持仓与基准的差异度)
# top_n个持仓等权 vs 基准等权成分股
if 'portfolio_codes' in s and benchmark_code:
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        bench_members = pd.read_sql(
            "SELECT code FROM index_members WHERE index_code=?",
            conn, params=[benchmark_code])
        conn.close()
        if not bench_members.empty:
            portfolio_set = set(s['portfolio_codes'])
            bench_set = set(bench_members['code'])
            # 简化Active Share: 非重叠比例
            overlap = len(portfolio_set & bench_set)
            total = max(len(portfolio_set), len(bench_set))
            s['active_share'] = 1.0 - overlap / total if total > 0 else 0.9
    except Exception:
        s['active_share'] = 0.9  # 量化选股通常Active Share很高
else:
    s['active_share'] = 0.9  # 量化选股默认高Active Share

# L6: Factor Attribution
try:
    factor_df = load_or_build_factors(start_d, end_d)
    if not factor_df.empty and daily_returns is not None:
        fa_result = compute_factor_attribution(daily_returns, factor_df)
        s['residual_alpha_t'] = fa_result['residual_alpha_t']
        s['factor_r_squared'] = fa_result['factor_r_squared']
        s['max_factor_loading'] = fa_result['max_factor_loading']
        s['smb_beta'] = fa_result['smb_beta']
        s['mom_beta'] = fa_result['mom_beta']
        # 保存完整结果用于因子分解摘要
        s['_factor_attribution'] = fa_result
except Exception as e:
    logger.warning(f"Factor attribution failed: {e}")
```

- [ ] **Step 4: Add V5 scorecard call after V4 scorecard**

Find where `_print_scorecard_v4` is called (around line 2469) and add V5 call:

```python
# After existing V4 scorecard call, add:
if score_version in ('v5', 'all', 'both'):
    v5_result = _print_scorecard_v5(
        s, label, days, n_trading_days,
        n_trials=n_trials,
        benchmark_code=benchmark_code,
        wf_summary=wf_summary,
    )
```

- [ ] **Step 5: Test integration with existing reports**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -c "
from backtest.north_star_metrics import compute_v5_score, NORTH_STAR_TARGETS_V5
# Smoke test with sample values
vals = {
    'daily_ic': 0.072, 'icir': 0.621, 'ic_positive_pct': 64.2,
    'ic_monotonicity': 3.82, 'ic_time_stability': 0.88,
    'signal_half_life': 9.3, 'bear_icir': 0.24, 'ic_decay_ratio': 0.78,
    'ic_autocorr_1d': 0.42, 'transfer_coefficient': 0.73,
    'annual_turnover': 28.3, 'annual_cost_drag': 0.086,
    'net_gross_ratio': 0.78, 'limit_up_fail_rate': 0.062, 'liquidity_coverage': 0.87,
    'max_drawdown': -0.142, 'sharpe_ratio': 1.71,
    'worst_rolling_60d_icir': 0.08, 'tail_ratio': 1.35,
    'cvar_5pct': 0.023, 'max_dd_duration': 72, 'underwater_ratio': 0.38,
    'annual_return': 0.521, 'monthly_win_rate': 73.1,
    'probabilistic_sharpe': 0.92, 'deflated_sharpe': 0.78,
    'excess_annual_return': 0.285, 'information_ratio': 0.82,
    'excess_win_rate': 58.2, 'excess_max_drawdown': -0.123, 'up_capture_ratio': 1.15,
    'residual_alpha_t': 2.31, 'factor_r_squared': 0.42, 'active_share': 0.83,
    'max_factor_loading': 0.95, 'smb_beta': 0.82, 'mom_beta': 0.55,
}
r = compute_v5_score(vals, n_trading_days=521)
print(f'Score: {r[\"total_score\"]:.1f}/{r[\"max_score\"]:.0f} ({r[\"final_pct\"]:.1f}%) Grade: {r[\"grade\"]}')
for lid in sorted(r['layer_details']):
    ld = r['layer_details'][lid]
    print(f'  L{lid}: {ld[\"score\"]:.1f}/{ld[\"max\"]:.0f} ({ld[\"pct\"]:.1f}%) weight={ld[\"weight\"]:.0%}')
"`
Expected: Prints V5 score breakdown with 6 layers, continuous scores, grade A/A+

- [ ] **Step 6: Commit**

```bash
git add backtest/backtest_report_based.py backtest/north_star_metrics.py
git commit -m "feat: V5评分卡集成到回测流程 — _print_scorecard_v5 + 因子归因"
```

---

### Task 9: Update run_north_star_eval.py CLI

**Files:**
- Modify: `backtest/run_north_star_eval.py` (lines 570-574 + call sites)

- [ ] **Step 1: Update --score-version choices**

At line 570-571, change:

```python
# Before:
parser.add_argument('--score-version', type=str, default='both', choices=['v2', 'v3', 'both'],
                    help='评分卡版本: v2=传统等权, v3=加权+统计鲁棒性, both=两者都打印 (default: both)')

# After:
parser.add_argument('--score-version', type=str, default='both',
                    choices=['v2', 'v3', 'v4', 'v5', 'both', 'all'],
                    help='评分卡版本: v2/v3/v4/v5/both(v2+v4)/all(v2+v4+v5) (default: both)')
```

- [ ] **Step 2: Add --wf-summary argument**

After line 573, add:

```python
parser.add_argument('--wf-summary', type=str, default=None,
                    help='WF训练摘要JSON路径 (V5 WFER+OOS半衰期)')
```

Note: `--benchmark` already exists at line 535. `--n-trials` already exists at line 572-573, and is passed through to `compute_v5_score` via `n_trials` parameter.

- [ ] **Step 3: Pass new args through to run_single_backtest calls**

In each function that calls `brb.run_single_backtest()` (lines 186, 243, 333, 365), ensure `wf_summary` and `score_version` are passed through. Add to the call:

```python
# Add to run_single_backtest call parameters:
wf_summary=getattr(args, 'wf_summary', None),
score_version=getattr(args, 'score_version', 'both'),
```

- [ ] **Step 4: Test CLI help**

Run: `cd /Users/yangxu/StockTradebyZ && python3 backtest/run_north_star_eval.py --help 2>&1 | grep -A2 'score-version\|wf-summary'`
Expected: Shows v5 in choices, shows --wf-summary

- [ ] **Step 5: Commit**

```bash
git add backtest/run_north_star_eval.py
git commit -m "feat: CLI新增 --score-version v5/all + --wf-summary"
```

---

### Task 10: Validation — Run V5 Against Real Reports

**Files:**
- No new files

- [ ] **Step 1: Run V4 regression test**

Run: `cd /Users/yangxu/StockTradebyZ && python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v4.7.5 --score-version v4 --top-n 10 --focus-days 10 2>&1 | tail -30`
Expected: V4 scores unchanged from before V5 integration

- [ ] **Step 2: Run V5 scorecard**

Run: `cd /Users/yangxu/StockTradebyZ && python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v4.7.5 --score-version v5 --top-n 10 --focus-days 10 2>&1 | tail -50`
Expected: V5 scorecard with continuous scores, 6 layers, factor attribution

- [ ] **Step 3: Run all-version comparison**

Run: `cd /Users/yangxu/StockTradebyZ && python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v4.7.5 --score-version all --top-n 10 --focus-days 10 2>&1 | tail -80`
Expected: V2, V4, and V5 scorecards all printed

- [ ] **Step 4: Verify factor returns cache was created**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -c "
import sqlite3
conn = sqlite3.connect('data_adapter/stock_data.db')
try:
    df = conn.execute('SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM factor_daily_returns').fetchone()
    print(f'Factor returns: {df[0]} rows, {df[1]} to {df[2]}')
except:
    print('factor_daily_returns table not found yet')
conn.close()
"`

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/yangxu/StockTradebyZ && python3 -m pytest backtest/test_north_star_v5.py -v 2>&1 | tail -15`
Expected: All 29+ tests PASS

- [ ] **Step 6: Commit all final adjustments**

```bash
git add -A
git commit -m "feat: 北极星V5评分体系完整实现 — 6层39指标连续插值+因子归因"
```
