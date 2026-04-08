# NG v1.0.5 Design: Low Turnover + MaxDD < 10%

**Date**: 2026-04-08
**Status**: Approved
**Base**: ng1.0.1 (生产版, V5.2=78.9% A+, Sharpe=2.339, MaxDD=-12.6%)
**Goal**: 换手率 < 15x, MaxDD < 10% (含CPPI), 保持Sharpe > 1.5

---

## 1. Problem Statement

ng1.0.1 + CPPI(F0.08, M20) 当前表现:
- MaxDD = -12.6% (目标 < 10%, 差 2.6pp)
- 换手率 = 24x/年 (目标 < 15x)
- 磨损率高: 换手×交易成本侵蚀净收益

## 2. Design: 三层组合优化

全部在组合构建层，不改模型层。基于ng1.0.1现有报告。

### 2A. EMA评分平滑

**原理**: 当日评分 = alpha × 原始评分 + (1-alpha) × 前一日平滑评分

**参数**: alpha ∈ [0.2, 0.3, 0.5]
- alpha=0.2: 强平滑，换手降~60%，信号延迟大
- alpha=0.3: 中等平滑，换手降~50%，推荐起点
- alpha=0.5: 弱平滑，换手降~30%

**实现**: 在 `run_north_star_eval.py` 的回测引擎中，加载报告后对每只股票的score序列做EMA:
```python
smoothed_scores[stock][t] = alpha * raw_score[t] + (1-alpha) * smoothed_scores[stock][t-1]
```

### 2B. 持仓缓冲 (Holding Buffer)

**原理**: 当前持有的股票在下次排名时获得bonus分加成，降低因微小排名变动导致的换入换出。

**参数**: buffer_bonus ∈ [3, 5, 8] (加到0-100的score上)

**实现**: 每次调仓时，已持有的股票score += buffer_bonus，再排名选Top-N。

### 2C. 激进CPPI参数

当前 CPPI(F0.08, M20): MaxDD=-12.6%

网格搜索:
- floor ∈ [0.08, 0.10, 0.12, 0.15]
- multiplier ∈ [10, 15, 20]

更大floor = 更早减仓 = 更低MaxDD，但牺牲收益弹性。

### 2D. 调仓频率

当前 focus_days=10 (每10天调仓)。尝试:
- focus_days=15: 换手再降33%
- focus_days=20: 换手再降50%

## 3. 网格搜索空间

```
alpha    ∈ [0.2, 0.3, 0.5, 1.0(无平滑)]
buffer   ∈ [0, 3, 5, 8]
floor    ∈ [0.08, 0.10, 0.12, 0.15]
mult     ∈ [10, 15, 20]
focus    ∈ [10, 15, 20]
```

总组合: 4 × 4 × 4 × 3 × 3 = 576 种

每种评估约30秒，总计约5小时。可用并行加速。

## 4. Success Criteria

| 指标 | 当前(ng1.0.1+CPPI) | 目标 | 优先级 |
|:-----|:---:|:---:|:---:|
| MaxDD | -12.6% | **< -10%** | P0 硬约束 |
| 换手率 | 24x | **< 15x** | P1 |
| Sharpe | 2.339 | **> 1.5** | P1 保持 |
| 年化(毛) | 72.2% | **> 40%** | P2 可trade-off |

## 5. Implementation

### Step 1: 修改 run_north_star_eval.py 支持 EMA + buffer

添加CLI参数:
- `--ema-alpha`: EMA平滑系数 (默认1.0=无平滑)
- `--holding-buffer`: 持仓缓冲bonus (默认0)

在回测循环中:
1. 加载当日所有股票score
2. 对score做EMA平滑
3. 已持有股票score += buffer
4. 排名选Top-N
5. CPPI计算仓位

### Step 2: 网格搜索脚本

新建 `scripts/ng105_grid_search.py`:
- 遍历所有参数组合
- 调用回测引擎
- 记录关键指标
- 输出CSV + 最优配置

### Step 3: 验证最优配置

用最优参数重新评估，确认满足所有目标。

### Step 4: 更新 production_config.json

## 6. Files to Modify

| File | Changes |
|------|---------|
| `backtest/run_north_star_eval.py` | 添加 --ema-alpha, --holding-buffer 参数 |
| `scripts/ng105_grid_search.py` | **新建**: 网格搜索脚本 |
| `production_config.json` | 更新为最优配置 |
| `docs/wiki/models/ng-series.md` | 添加 ng1.0.5 章节 |
