# 北极星 V5 评分体系设计规格

> 日期: 2026-03-31
> 状态: 已批准, 待实施
> 前序: V4 (2026-03-20, 31指标/155分/5层)

## 1. 目标

全面升级北极星评分体系，解决 V4 的四个致命缺陷和六个重要缺陷：

**致命缺陷:**
1. 无因子归因 — 无法区分 alpha 和 factor beta
2. 无 OOS 衰减度量 — 缺 Walk-Forward 效率比
3. 5 档离散评分 — IC 0.06 vs 0.079 同分
4. 风险指标高度冗余 — Sharpe/Sortino/Calmar r>0.7

**重要缺陷:**
5. 无回撤持续时间 (深度≠痛苦)
6. 无 CVaR/Expected Shortfall (尾部风险维度缺失)
7. 无 IC 自相关分析 (不知道最优调仓频率)
8. 长度惩罚过于宽松 (250天仅30%折扣)
9. 基准硬编码 CSI500 (小盘策略被错误惩罚)
10. 无容量/可扩展性评估 (预留 V5.1)

## 2. 评分机制: 离散→连续

### V4 (离散)
```python
# 每个指标映射到 {0, 1, 2, 3, 4, 5} 整数
# 31 指标 × 6 档 = 理论 186 个可能分值, 实际集中在 60-80 区间
```

### V5 (连续插值)
```python
def score_metric_v5(value, thresholds, direction='higher'):
    """
    连续插值: 0.0 ~ 5.0 浮点数

    breakpoints = [0, pass, ok, good, great, target]
    scores      = [0,  1,    2,   3,     4,     5   ]

    - value < pass → np.interp(value, [0, pass], [0, 1])
    - pass ≤ value < target → np.interp(value, breakpoints, scores)
    - value ≥ target → 5.0 (封顶, 不超额奖励)
    - direction='lower' 时翻转轴
    """
```

效果: IC=0.059→3.80, IC=0.079→4.95 (差距清晰可见)

等级阈值不变: S≥80%, A+≥70%, A≥60%, B≥45%, C≥30%, D<30%

## 3. 层级结构: 5层31指标 → 6层38指标

### 总览

| 层 | 名称 | V4指标数 | V5指标数 | V4权重 | V5权重 |
|----|------|----------|----------|--------|--------|
| L1 | 信号质量 | 8 | 10 | 35% | 30% |
| L2 | 组合效率 | 5 | 5 | 15% | 15% |
| L3 | 风险控制 | 7 | 7 | 20% | 20% |
| L4 | OOS鲁棒性 | 5 | 6 | 15% | 15% |
| L5 | 超额收益 | 6 | 5 | 15% | 10% |
| L6 | 因子归因 | - | 6 | - | 10% |
| **合计** | | **31** | **39** (净+8) | **100%** | **100%** |

满分: 39 × 5 = 195 分

### 变更摘要

| 层 | 删除 | 新增 | 理由 |
|----|------|------|------|
| L1 | - | ic_autocorr_1d, transfer_coefficient | 信号持续性+传递效率 |
| L2 | - | - | 不变 |
| L3 | Sortino, Calmar, max_consecutive_loss_periods | cvar_5pct, max_dd_duration, underwater_ratio | 去冗余+补尾部风险维度 (max_dd_duration+underwater_ratio覆盖连续亏损信息) |
| L4 | half_period_consistency | wfer, oos_ic_half_life | 强化OOS检测 |
| L5 | bear_excess_return | - | 与L1 bear_icir + L6因子归因重叠 |
| L6 | (全新) | residual_alpha_t, factor_r_squared, active_share, max_factor_loading, smb_beta, mom_beta | 因子归因 |

## 4. 各层指标详细定义

### L1 信号质量 (10指标, 权重30%, 满分50)

| # | 指标 | 含义 | pass | ok | good | great | target | 方向 |
|---|------|------|------|----|----|------|--------|------|
| 1 | daily_ic | rank_corr(score, return) | 0.03 | 0.04 | 0.05 | 0.06 | 0.08 | ↑ |
| 2 | icir | mean(IC)/std(IC) | 0.25 | 0.35 | 0.45 | 0.55 | 0.70 | ↑ |
| 3 | ic_positive_pct | %days IC>0 | 55% | 57% | 60% | 63% | 68% | ↑ |
| 4 | ic_monotonicity | 累积分位收益单调性 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | ↑ |
| 5 | ic_time_stability | rolling IC的CV | 2.0 | 1.5 | 1.0 | 0.8 | 0.6 | ↓ |
| 6 | signal_half_life | IC自相关衰减到0.5的天数 | 3 | 6 | 8 | 12 | 20 | ↑ |
| 7 | bear_icir | 熊市期间ICIR | 0.05 | 0.10 | 0.20 | 0.30 | 0.35 | ↑ |
| 8 | ic_decay_ratio | IC后半/前半比 | 0.50 | 0.60 | 0.70 | 0.80 | 0.95 | ↑ |
| 9 | **ic_autocorr_1d** | corr(IC_t, IC_{t-1}) | 0.10 | 0.20 | 0.35 | 0.50 | 0.70 | ↑ |
| 10 | **transfer_coefficient** | corr(信号排名, 实际持仓排名) | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | ↑ |

**ic_autocorr_1d**: 信号持续性。高→调仓频率可低→成本低。低→信号短命需频繁调仓。
```python
def compute_ic_autocorrelation(ic_series: pd.Series, lag: int = 1) -> float:
    return ic_series.autocorr(lag=lag)
```

**transfer_coefficient**: 信号到持仓的传递效率。受涨停/停牌/流动性约束影响。
```python
def compute_transfer_coefficient(signal_ranks: pd.Series,
                                  actual_holdings: pd.Series) -> float:
    # signal_ranks: 每日top-N的信号排名
    # actual_holdings: 实际买入的排名 (涨停/停牌后替补)
    return signal_ranks.corr(actual_holdings, method='spearman')
```

### L2 组合效率 (5指标, 权重15%, 满分25) — 与V4完全相同

| # | 指标 | pass | ok | good | great | target | 方向 |
|---|------|------|----|----|------|--------|------|
| 1 | annual_turnover | 45% | 35% | 30% | 25% | 20% | ↓ |
| 2 | annual_cost_drag | 13% | 10% | 8% | 7% | 5% | ↓ |
| 3 | net_gross_ratio | 0.60 | 0.70 | 0.75 | 0.80 | 0.85 | ↑ |
| 4 | limit_up_fail_rate | 15% | 10% | 8% | 5% | 2% | ↓ |
| 5 | liquidity_coverage | 0.70 | 0.80 | 0.85 | 0.90 | 0.95 | ↑ |

### L3 风险控制 (7指标, 权重20%, 满分35)

| # | 指标 | 含义 | pass | ok | good | great | target | 方向 | 变更 |
|---|------|------|------|----|----|------|--------|------|------|
| 1 | max_drawdown | 最大回撤 | -25% | -18% | -12% | -10% | -8% | ↑ | 不变 |
| 2 | sharpe_ratio | (R-Rf)/σ | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | ↑ | 不变 |
| 3 | worst_rolling_60d_icir | 最差60日滚动ICIR | -0.10 | 0.0 | 0.10 | 0.20 | 0.30 | ↑ | 不变 |
| 4 | tail_ratio | P95/|P5| | 0.8 | 1.0 | 1.2 | 1.5 | 2.0 | ↑ | 不变 |
| 5 | **cvar_5pct** | -mean(R \| R≤P5) 日级 | 4.0% | 3.0% | 2.0% | 1.5% | 1.0% | ↓ | 替Sortino |
| 6 | **max_dd_duration** | 最长回撤恢复交易日 | 120 | 90 | 60 | 40 | 20 | ↓ | 替Calmar |
| 7 | **underwater_ratio** | 水下天数/总天数 | 0.60 | 0.50 | 0.40 | 0.30 | 0.20 | ↓ | 新增 |

删除 Sortino/Calmar 理由:
- Sortino 与 Sharpe 相关系数通常 >0.85 (仅下行波动率替换全波动率)
- Calmar = 年化收益/MaxDD, 与 Sharpe + MaxDD 线性冗余
- 腾出槽位给 CVaR 和回撤持续时间, 信息增量远大于冗余指标

```python
def compute_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (Expected Shortfall) at alpha level."""
    var_threshold = returns.quantile(alpha)
    return -returns[returns <= var_threshold].mean()

def compute_max_dd_duration(cumulative_returns: pd.Series) -> int:
    """最长回撤恢复交易日数."""
    peak = cumulative_returns.expanding().max()
    underwater = cumulative_returns < peak
    # 计算连续水下段的最大长度
    groups = (~underwater).cumsum()
    durations = underwater.groupby(groups).sum()
    return int(durations.max()) if len(durations) > 0 else 0

def compute_underwater_ratio(cumulative_returns: pd.Series) -> float:
    """水下天数占总天数比例."""
    peak = cumulative_returns.expanding().max()
    underwater = cumulative_returns < peak
    return underwater.mean()
```

### L4 OOS鲁棒性 (6指标, 权重15%, 满分30)

| # | 指标 | 含义 | pass | ok | good | great | target | 方向 | 变更 |
|---|------|------|------|----|----|------|--------|------|------|
| 1 | annual_return | 年化收益率 | 15% | 20% | 30% | 40% | 50% | ↑ | 不变 |
| 2 | monthly_win_rate | 月度胜率 | 55% | 60% | 67% | 75% | 83% | ↑ | 不变 |
| 3 | probabilistic_sharpe | PSR | 0.80 | 0.85 | 0.90 | 0.95 | 0.99 | ↑ | 不变 |
| 4 | deflated_sharpe | DSR(真实N) | 0.70 | 0.80 | 0.85 | 0.90 | 0.95 | ↑ | 改进: n_trials传入实际试验次数 |
| 5 | **wfer** | mean(Sharpe_OOS)/mean(Sharpe_IS) | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | ↑ | 替half_period_consistency |
| 6 | **oos_ic_half_life** | OOS IC衰减到IS一半的月数 | 1 | 2 | 3 | 6 | 12 | ↑ | 新增 |

```python
def compute_wfer(wf_summary: dict) -> Optional[float]:
    """
    Walk-Forward Efficiency Ratio.
    wf_summary: {'is_sharpe': [w1, w2, ...], 'oos_sharpe': [w1, w2, ...]}
    """
    is_mean = np.mean(wf_summary['is_sharpe'])
    oos_mean = np.mean(wf_summary['oos_sharpe'])
    if is_mean <= 0:
        return None
    return oos_mean / is_mean

def compute_oos_ic_half_life(wf_summary: dict) -> Optional[float]:
    """
    OOS IC衰减半衰期 (月数).
    从WF各窗口内逐月IC拟合指数衰减: IC(t) = IC_0 * exp(-λt)
    半衰期 = ln(2) / λ
    """
    monthly_ics = wf_summary.get('oos_monthly_ics')  # List[List[float]]
    if not monthly_ics:
        return None
    # 按月offset聚合
    max_months = max(len(m) for m in monthly_ics)
    avg_by_month = []
    for i in range(max_months):
        vals = [m[i] for m in monthly_ics if len(m) > i]
        avg_by_month.append(np.mean(vals))
    # 拟合指数衰减
    if avg_by_month[0] <= 0:
        return 0.0
    log_ics = [np.log(max(ic, 1e-6)) for ic in avg_by_month]
    slope = np.polyfit(range(len(log_ics)), log_ics, 1)[0]
    if slope >= 0:
        return 12.0  # 不衰减, 返回上限
    return min(np.log(2) / (-slope), 12.0)
```

WFER Fallback (无WF训练日志时):
```python
def get_wfer(wf_summary_path=None, ic_decay_ratio=None):
    if wf_summary_path and os.path.exists(wf_summary_path):
        data = json.load(open(wf_summary_path))
        return compute_wfer(data)
    elif ic_decay_ratio is not None:
        return ic_decay_ratio * 0.7  # 经验缩放fallback
    else:
        return None  # 不参与评分, 权重重分配到L4其他指标
```

### L5 超额收益 (5指标, 权重10%, 满分25)

| # | 指标 | pass | ok | good | great | target | 方向 | 变更 |
|---|------|------|----|----|------|--------|------|------|
| 1 | excess_annual_return | 5% | 10% | 15% | 20% | 30% | ↑ | 不变 |
| 2 | information_ratio | 0.30 | 0.50 | 0.70 | 1.00 | 1.50 | ↑ | 不变 |
| 3 | excess_win_rate | 50% | 53% | 55% | 60% | 65% | ↑ | 不变 |
| 4 | excess_max_drawdown | -30% | -20% | -15% | -10% | -5% | ↑ | 不变 |
| 5 | up_capture_ratio | 0.80 | 1.00 | 1.10 | 1.20 | 1.40 | ↑ | 不变 |

删除 bear_excess_return: 与 L1 bear_icir + L6 因子归因重叠。

### L6 因子归因 (6指标, 权重10%, 满分30) — 全新

| # | 指标 | 含义 | pass | ok | good | great | target | 方向 |
|---|------|------|------|----|----|------|--------|------|
| 1 | residual_alpha_t | FF回归截距t值 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | ↑ |
| 2 | factor_r_squared | FF回归R² (越低越独特) | 0.70 | 0.60 | 0.50 | 0.35 | 0.20 | ↓ |
| 3 | active_share | Σ\|w_i - w_bench_i\|/2 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | ↑ |
| 4 | max_factor_loading | max(\|β_k\|) across factors | 1.50 | 1.20 | 1.00 | 0.80 | 0.50 | ↓ |
| 5 | smb_beta | 小盘因子暴露\|β_smb\| | 1.50 | 1.20 | 1.00 | 0.70 | 0.30 | ↓ |
| 6 | mom_beta | 动量因子暴露\|β_umd\| | 1.20 | 1.00 | 0.80 | 0.50 | 0.20 | ↓ |

**因子回归模型** (中国A股4因子):
```
R_strategy - Rf = α + β_mkt(R_mkt - Rf) + β_smb(SMB) + β_hml(HML) + β_umd(UMD) + ε
```

```python
def compute_factor_attribution(portfolio_returns: pd.Series,
                                factor_returns: pd.DataFrame,
                                risk_free_rate: float = 0.02) -> dict:
    """
    Fama-French 4因子归因.

    Args:
        portfolio_returns: 策略日收益率
        factor_returns: DataFrame with columns ['MKT', 'SMB', 'HML', 'UMD']
        risk_free_rate: 年化无风险利率

    Returns:
        {
            'residual_alpha': float,      # 日化alpha
            'residual_alpha_annual': float, # 年化alpha
            'residual_alpha_t': float,     # t统计量
            'factor_r_squared': float,     # R²
            'betas': {'mkt': float, 'smb': float, 'hml': float, 'umd': float},
            'max_factor_loading': float,   # max(|β_k|) 不含mkt
            'smb_beta': float,             # |β_smb|
            'mom_beta': float,             # |β_umd|
        }
    """
    rf_daily = risk_free_rate / 252
    y = portfolio_returns - rf_daily
    X = factor_returns[['MKT', 'SMB', 'HML', 'UMD']]
    X = sm.add_constant(X)  # statsmodels OLS
    model = sm.OLS(y, X).fit()

    return {
        'residual_alpha': model.params['const'],
        'residual_alpha_annual': model.params['const'] * 252,
        'residual_alpha_t': model.tvalues['const'],
        'factor_r_squared': model.rsquared,
        'betas': {
            'mkt': model.params['MKT'],
            'smb': model.params['SMB'],
            'hml': model.params['HML'],
            'umd': model.params['UMD'],
        },
        'max_factor_loading': max(abs(model.params['SMB']),
                                   abs(model.params['HML']),
                                   abs(model.params['UMD'])),
        'smb_beta': abs(model.params['SMB']),
        'mom_beta': abs(model.params['UMD']),
    }
```

## 5. 因子收益构建

新文件: `backtest/factor_returns.py`

```python
def build_factor_returns(start_date: str, end_date: str,
                         db_path: str = 'data_adapter/stock_data.db') -> pd.DataFrame:
    """
    每个交易日构建4因子收益:

    1. MKT: 中证500(000905.SH)日收益 - Rf/252
       来源: daily_quotes WHERE code='000905.SH'

    2. SMB: 按市值5分位, Bottom20%等权 - Top20%等权
       来源: daily_basic(total_mv) + daily_quotes(price_change_pct)

    3. HML: 按B/P(=1/PB)5分位, High B/P - Low B/P
       来源: daily_basic(pb) + daily_quotes(price_change_pct)

    4. UMD: 按12-1月累计收益5分位, Winner - Loser
       来源: daily_quotes(close) → 12-1月动量 → 分位排序

    Returns: DataFrame[date, MKT, SMB, HML, UMD]
    """

def load_or_build_factors(start_date, end_date, db_path,
                          cache_table='factor_daily_returns'):
    """缓存机制: 首次构建写入SQLite, 后续直接读取."""
```

性能预估:
- 首次构建: ~3-5分钟
- 缓存后: <1秒
- 增量更新: 每日追加1行

因子验证标准:
- SMB/HML/UMD 日均值应接近 0 (无趋势)
- 日标准差约 0.5%-1.5%
- 因子间相关性应 < 0.3

## 6. 长度惩罚升级

```python
def compute_backtest_length_factor_v5(n_days: int, min_days: int = 500) -> float:
    """
    V4: sqrt(n/500) — 250天仅折30%
    V5: log曲线, 更严格, <60天直接拒绝
    """
    if n_days >= min_days:
        return 1.0
    if n_days < 60:
        return 0.0
    return np.log(n_days / 60) / np.log(min_days / 60)
```

| 天数 | V4折扣 | V5折扣 |
|------|--------|--------|
| 500+ | 1.00 | 1.00 |
| 250 | 0.71 | 0.67 |
| 125 | 0.50 | 0.35 |
| 60 | 0.35 | 0.00 |

## 7. 动态基准选择

```python
def auto_select_benchmark(median_market_cap_bn: float) -> str:
    """根据策略持仓市值中位数自动选择最匹配基准."""
    if median_market_cap_bn >= 50:   return '000300.SH'   # 沪深300 (大盘)
    if median_market_cap_bn >= 15:   return '000905.SH'   # 中证500 (中盘)
    if median_market_cap_bn >= 5:    return '000852.SH'   # 中证1000 (中小盘)
    return '932000.CSI'                                     # 中证2000 (小盘)
```

支持 `--benchmark CODE` CLI 参数覆盖自动选择。

## 8. 实现架构

### 文件结构

```
backtest/
├── north_star_metrics.py          # 现有, 新增V5函数
│   ├── NORTH_STAR_TARGETS_V5      # 38指标阈值
│   ├── score_metric_v5()          # 连续插值评分
│   ├── compute_v5_score()         # V5加权评分主函数
│   ├── compute_backtest_length_factor_v5()
│   ├── auto_select_benchmark()
│   ├── compute_ic_autocorrelation()
│   ├── compute_transfer_coefficient()
│   ├── compute_cvar()
│   ├── compute_max_dd_duration()
│   ├── compute_underwater_ratio()
│   ├── compute_wfer()
│   ├── compute_oos_ic_half_life()
│   └── compute_factor_attribution()
│
├── factor_returns.py              # 新文件: A股4因子收益构建
│   ├── build_factor_returns()
│   ├── load_or_build_factors()
│   └── FACTOR_CACHE_TABLE
│
├── backtest_report_based.py       # 现有, 新增V5集成
│   ├── _compute_v5_metrics()
│   ├── _print_scorecard_v5()
│   └── run_single_backtest()      # 增加 --score-version v5
│
├── run_north_star_eval.py         # 现有, 新增V5 CLI
│   └── --score-version v5 | all
│
└── test_north_star_v5.py          # 新文件: V5单元测试
```

### 向后兼容

- V2/V3/V4 评分函数全部保留, 不修改
- `--score-version` 新增 `v5` 和 `all` 选项
- V5 可单独调用, 也可与旧版本同时打印对比

## 9. CLI 接口

```bash
# V5评分 (自动选基准)
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4.7.5 \
    --score-version v5 --top-n 10 --focus-days 10

# V5评分 (手动指定基准)
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4.7.5 \
    --score-version v5 --benchmark 000905.SH

# V5评分 (指定试验次数)
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4.7.5 \
    --score-version v5 --n-trials 50

# 全版本对比
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4.7.5 \
    --score-version all

# 提供WF训练摘要 (WFER + OOS半衰期)
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4.7.5 \
    --score-version v5 \
    --wf-summary ml_models/trained_models/v475_wf_summary.json
```

新增CLI参数:
- `--score-version v5 | all`
- `--benchmark CODE` (覆盖自动基准)
- `--wf-summary PATH` (WF训练摘要JSON)

## 10. 迁移策略

### 4阶段实施

**阶段1: 基础设施** — factor_returns.py
- 构建4因子日收益, 写入SQLite
- 验证: 因子统计特征符合预期

**阶段2: 新增计算函数** — north_star_metrics.py
- 9个新函数 + TARGETS_V5 + compute_v5_score + score_metric_v5
- 单元测试: 合成数据边界条件验证

**阶段3: 集成** — backtest_report_based.py + run_north_star_eval.py
- V5指标计算 + 评分卡打印 + CLI参数
- 回归测试: V2/V3/V4分数与升级前完全一致

**阶段4: 验证与校准**
- 对比V4.7.5 (好) vs V4.7.4 (差), V5应拉开更大差距
- 阈值微调: 如果所有策略L6都<2分, 放宽阈值

## 11. 测试计划

```bash
# 1. 因子收益验证
python3 -c "
from backtest.factor_returns import build_factor_returns
df = build_factor_returns('2020-01-01', '2026-03-31')
print(df.describe())
print(df.corr())
"

# 2. 单指标函数测试
python3 -m pytest backtest/test_north_star_v5.py -v

# 3. V4回归测试
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4.7.5 \
    --score-version all --top-n 10 --focus-days 10
# 验证: V2/V3/V4分数不变

# 4. 区分度验证
# V4.7.5 vs V4.7.4: V4差8分, V5目标差15+分
```

## 12. V5.1 预留 (方案C)

V5 代码预留扩展点:

```python
# 预留 L7 容量与可扩展性:
# strategy_capacity_mn, participation_rate_p90, liquidity_adjusted_sharpe

# 预留高级OOS:
# cscv_pbo (组合对称交叉验证)

# 预留稳定性:
# hurst_exponent, regime_transition_dd, effective_n_corr
```

## 13. 评分卡输出格式

每个指标用连续进度条 (████░░░░) 替代离散星星 (★★★☆☆):
```
daily_ic          0.072   ████████████████░░░░  4.40 / 5
```

底部增加因子暴露摘要:
```
因子暴露摘要: α=2.31σ(显著) | R²=0.42 | 小盘β=0.82 | 动量β=0.55
关键发现: 52%收益来自选股alpha, 31%来自小盘暴露, 17%来自动量
```

NEW 标记新增指标, 方便与 V4 对比。
