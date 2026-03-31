# 北极星 V5.1 评分体系设计规格

> 日期: 2026-03-31
> 状态: 已批准, 待实施
> 前序: V5 (2026-03-31, 39指标/195分/6层)

## 1. 目标

在 V5 基础上追加方案C的 7 个指标: 容量评估 + CSCV过拟合检测 + 稳定性指标，升级为 7层46指标/230分。

## 2. 层级结构: 6层39指标 → 7层46指标

| 层 | 名称 | V5指标数 | V5.1指标数 | V5权重 | V5.1权重 |
|----|------|----------|-----------|--------|---------|
| L1 | 信号质量 | 10 | 10 | 30% | 25% |
| L2 | 组合效率 | 5 | 5 | 15% | 12% |
| L3 | 风险控制 | 7 | 9 | 20% | 18% |
| L4 | OOS鲁棒性 | 6 | 8 | 15% | 15% |
| L5 | 超额收益 | 5 | 5 | 10% | 8% |
| L6 | 因子归因 | 6 | 6 | 10% | 8% |
| L7 | 容量可扩展 | - | 3 | - | 14% |
| **合计** | | **39** | **46** | **100%** | **100%** |

满分: 46 × 5 = 230 分

## 3. 新增指标详细定义

### L3 新增: 稳定性指标 (2项)

#### hurst_exponent

Hurst指数, 衡量收益序列的持续性/均值回复特征.

- **公式**: R/S法 (Rescaled Range). H=0.5=随机游走, H>0.5=趋势持续, H<0.5=均值回复
- **评分特殊**: 理想区间0.55-0.65 (mild persistence), 偏离0.60越远越扣分
- **阈值** (基于距离0.60的偏差 |H-0.60|):
  - pass: 0.15 (H在0.45-0.75), ok: 0.10, good: 0.07, great: 0.05, target: 0.02
  - direction: 'lower' (偏差越小越好)
- **layer**: 3
- **min_days**: 200

```python
def compute_hurst_exponent(returns: pd.Series, min_window: int = 20) -> float:
    """
    R/S法计算Hurst指数.

    对多个窗口n计算R/S, 拟合 log(R/S) = H * log(n) + c
    """
    if len(returns) < 100:
        return 0.5  # 默认随机游走

    windows = [20, 40, 60, 80, 100, 150, 200]
    windows = [w for w in windows if w < len(returns) // 2]
    if len(windows) < 3:
        return 0.5

    rs_values = []
    for w in windows:
        rs_list = []
        for start in range(0, len(returns) - w, w):
            chunk = returns.iloc[start:start+w]
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
```

**评分转换**: 先算 `deviation = abs(hurst - 0.60)`, 然后用 `score_metric_v5(deviation, hurst_target_info)` (direction='lower')

#### regime_transition_dd

Regime转换期间的最大回撤放大倍数.

- **公式**: max(DD在转换窗口内) / 正常期间平均DD深度
- **转换窗口**: regime切换点前10天~后20天
- **阈值**: pass=3.0, ok=2.5, good=2.0, great=1.5, target=1.0
- **direction**: 'lower' (放大倍数越小=转换期越稳健)
- **layer**: 3
- **min_days**: 200

```python
def compute_regime_transition_dd(daily_returns: pd.Series,
                                  benchmark_returns: pd.Series,
                                  lookback: int = 60,
                                  pre_window: int = 10,
                                  post_window: int = 20) -> float:
    """
    Regime转换期间的DD放大倍数.

    1. 用benchmark 60日滚动收益判定regime (>5%=bull, <-5%=bear, else=neutral)
    2. 找到所有regime变化点
    3. 计算变化点[-10d, +20d]窗口内的策略maxDD
    4. 与正常期间平均DD对比
    """
```

### L4 新增: 高级OOS检测 (2项)

#### cscv_pbo

Combinatorially Symmetric Cross-Validation 过拟合概率.

- **公式**: Lopez de Prado (2014). 将回测期切为S=16子期, 对C(16,8)=12870种IS/OOS组合, 计算IS最优在OOS排名低于中位数的比例.
- **适应单策略**: 用block bootstrap生成N=10个"策略变体"(不同起点+微量噪声), 在每个IS/OOS split上比较10个变体
- **阈值**: pass=0.50, ok=0.40, good=0.25, great=0.15, target=0.05
- **direction**: 'lower' (PBO越低=过拟合概率越低)
- **layer**: 4
- **min_days**: 320 (需至少16×20=320天)

```python
def compute_cscv_pbo(daily_returns: pd.Series,
                      n_subperiods: int = 16,
                      n_variants: int = 10,
                      max_combinations: int = 1000) -> float:
    """
    CSCV过拟合概率 (PBO).

    优化: C(16,8)=12870太多, 随机抽样max_combinations=1000个组合.
    对每个组合:
      1. 选8个子期做IS, 剩余8个做OOS
      2. 对10个策略变体计算IS和OOS的Sharpe
      3. IS最优变体在OOS的排名
      4. 如果OOS排名 <= 中位数(5) → 算过拟合
    PBO = 过拟合组合数 / 总组合数
    """
```

#### effective_n_corr

相关性调整后的有效持仓数.

- **公式**: N_eff = N / (1 + (N-1) × avg_pairwise_corr)
- **含义**: 10只高相关股票(r=0.7)的有效N≈2.5, 10只低相关股票(r=0.1)的有效N≈5.3
- **阈值**: pass=2.0, ok=3.0, good=4.0, great=6.0, target=8.0
- **direction**: 'higher'
- **layer**: 4

```python
def compute_effective_n_corr(holdings_returns: pd.DataFrame) -> float:
    """
    相关性调整有效N.

    holdings_returns: DataFrame, columns=各持仓股票的日收益, rows=交易日
    N_eff = N / (1 + (N-1) * avg_pairwise_corr)
    """
    if holdings_returns is None or holdings_returns.shape[1] < 2:
        return holdings_returns.shape[1] if holdings_returns is not None else 1.0

    N = holdings_returns.shape[1]
    corr_matrix = holdings_returns.corr()
    # 取上三角(不含对角线)的平均相关系数
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    avg_corr = corr_matrix.values[mask].mean()

    if np.isnan(avg_corr):
        return float(N)

    denominator = 1 + (N - 1) * max(avg_corr, 0)
    if denominator <= 0:
        return float(N)
    return float(N / denominator)
```

### L7 容量可扩展 (3项, 全新层)

假设目标AUM = 1亿 RMB.

#### strategy_capacity_mn

策略容量估计 (百万RMB).

- **公式**: Almgren-Chriss简化模型. 找到AUM使 market_impact_cost = gross_alpha
  - impact = σ_daily × η × sqrt(V_trade / ADV_20d)
  - η = 0.15 (A股经验系数, 考虑T+1)
  - 对每只持仓股票计算impact, 累加
- **阈值**: pass=50, ok=200, good=500, great=1000, target=5000 (百万RMB)
- **direction**: 'higher'
- **layer**: 7

```python
def compute_strategy_capacity(picks_with_volume: pd.DataFrame,
                               gross_annual_return: float,
                               avg_turnover: float,
                               eta: float = 0.15) -> float:
    """
    Almgren-Chriss策略容量估计 (百万RMB).

    picks_with_volume: DataFrame with columns [code, adv_20d, daily_vol]
      adv_20d = 20日均成交量(手)
      daily_vol = 日收益标准差

    二分搜索AUM使 total_impact_cost ≈ gross_alpha × AUM
    """
```

#### participation_rate_p90

持仓参与率第90百分位 (假设1亿AUM).

- **公式**: 对每只持仓, participation = position_value / (ADV_20d × avg_price). 取P90
- **阈值**: pass=0.10, ok=0.05, good=0.03, great=0.02, target=0.01
- **direction**: 'lower' (占比越低=对市场冲击越小)
- **layer**: 7

#### liquidity_adj_sharpe

流动性调整Sharpe (假设1亿AUM).

- **公式**: 扣除market_impact后的Sharpe
  - LA_Return = Gross_Return - Commission - Slippage - Market_Impact
  - LA_Sharpe = mean(LA_Return) / std(LA_Return) × sqrt(252)
- **阈值**: pass=0.5, ok=0.8, good=1.0, great=1.5, target=2.0
- **direction**: 'higher'
- **layer**: 7
- **与L3 Sharpe对比**: L3的Sharpe只扣佣金/滑点, L7的LA_Sharpe额外扣market impact

## 4. 数据依赖

### 已有数据 (无需新增)
- daily_returns: 策略日收益 (所有新指标都需要)
- benchmark_returns: 基准日收益 (regime_transition_dd需要)
- daily_quotes(volume): 20日均量ADV (容量层需要)

### 需要新增获取的
- **持仓股票的日收益矩阵**: effective_n_corr需要每个调仓期内各持仓的单独收益. 从backtest的trade记录+daily_quotes可构建.
- **持仓股票的ADV和波动率**: 容量层需要. 从daily_quotes聚合.

## 5. 实现架构

### 文件变更

```
backtest/
├── north_star_metrics.py          # 修改: +7个新函数 + TARGETS_V51 + compute_v51_score
│   ├── compute_hurst_exponent()
│   ├── compute_regime_transition_dd()
│   ├── compute_cscv_pbo()
│   ├── compute_effective_n_corr()
│   ├── compute_strategy_capacity()
│   ├── compute_participation_rate_p90()
│   ├── compute_liquidity_adj_sharpe()
│   ├── NORTH_STAR_TARGETS_V51     # 46指标
│   ├── V51_LAYER_WEIGHTS          # 7层权重
│   ├── compute_v51_score()
│   └── compute_backtest_length_factor_v5()  # 复用V5的
│
├── backtest_report_based.py       # 修改: +_print_scorecard_v51 + V51指标计算
│
├── run_north_star_eval.py         # 修改: --score-version v51 选项
│
└── test_north_star_v5.py          # 修改: +V51测试
```

### 向后兼容

- V5 代码全部保留, V5.1 新增而非替换
- `--score-version v5` 仍然可用, `v51` 是新选项
- `--score-version all` 打印 V2 + V4 + V5 + V5.1

## 6. CSCV 性能优化

完整C(16,8)=12870个组合计算量大. 优化方案:

```python
# 随机抽样1000个组合 (统计上足够)
from itertools import combinations
import random

all_combos = list(combinations(range(16), 8))
sampled = random.sample(all_combos, min(1000, len(all_combos)))
# 向量化Sharpe计算: 预计算每个子期的returns sum和sum_sq
# 组合IS/OOS只需索引相加, 不需重新遍历原始数据
```

预估耗时: ~30秒 (1000组合 × 10变体 × 向量化Sharpe)

## 7. Hurst 特殊评分

Hurst指数不是单调递增/递减, 而是有最优区间(0.55-0.65). 评分方式:

```python
# 先转换为偏差
hurst_deviation = abs(hurst - 0.60)
# 然后用direction='lower'评分 (偏差越小越好)
# TARGETS中存储偏差阈值:
'hurst_deviation': {
    'pass': 0.15, 'ok': 0.10, 'good': 0.07, 'great': 0.05, 'target': 0.02,
    'direction': 'lower', 'layer': 3, 'display': 'Hurst偏差',
}
```

## 8. 容量层AUM假设

默认假设AUM = 1亿 RMB. 支持CLI参数覆盖:

```bash
python3 backtest/run_north_star_eval.py --backtest \
    --score-version v51 --assumed-aum 100  # 百万RMB, 默认100
```

## 9. 测试计划

```bash
# 1. Hurst指数验证 (已知序列)
# 纯随机序列 → H≈0.5, 趋势序列 → H>0.6

# 2. CSCV验证
# 完美策略(无过拟合) → PBO<0.2
# 纯噪声策略 → PBO≈0.5

# 3. 容量估计验证
# 高流动性大盘股组合 → capacity>1000M
# 低流动性小盘股组合 → capacity<100M

# 4. V5回归测试
# V5分数不受V5.1代码影响

# 5. 全版本对比
python3 backtest/run_north_star_eval.py --backtest \
    --score-version all --top-n 10 --focus-days 10
```
