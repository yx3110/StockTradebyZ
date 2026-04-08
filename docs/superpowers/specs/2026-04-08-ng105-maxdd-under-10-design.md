# NG v1.0.5 Design: MaxDD < 10% via Risk Overlays

**Date**: 2026-04-08
**Status**: Approved
**Base**: ng1.0.1 + CPPI(F0.08, M20) — MaxDD=-12.6%, Sharpe=2.339
**Goal**: MaxDD < 10%, Sharpe > 1.5, 换手 < 20x

---

## 1. Problem Statement

CPPI参数网格搜索证明: F0.08/M20是当前MaxDD天花板(-12.6%)。
更激进的floor反而更差(CPPI trap)，EMA/buffer延迟CPPI响应。
需要额外风控层来跨越-12.6%→-10%的gap。

## 2. Root Cause Analysis

-12.6% MaxDD的构成:
- 基础选股MaxDD (无CPPI): ~-20%
- CPPI减仓: 降约8pp → -12.6%
- 剩余2.6pp来自: 单日跳空损失(gap risk) + CPPI反应延迟(隔日生效)

## 3. Three Complementary Risk Overlays

### 3A. 个股止损 (Stock-Level Stop-Loss)

**原理**: 单只持仓股跌破X%阈值时，下一调仓日强制卖出，不等排名自然轮换。

**参数**: stop_loss_pct = -8% (从买入价计)

**实现**: 在回测引擎的持仓循环中:
```python
for stock in current_holdings:
    if stock.return_since_entry < stop_loss_pct:
        force_sell(stock)
        replace_with_next_ranked()
```

**预期**: 阻止单只股票拖累整体组合超过-8%贡献，砍断左尾。

### 3B. 市场Regime门控 (Market Gate Enhancement)

**现有**: 回测引擎已有gate逻辑(bear → reduce top_n)
**增强**: 更激进的熊市减仓
- market_return_20d < -5%: 仓位上限50%
- market_return_20d < -10%: 仓位上限20%  
- VIX_proxy > P90: 仓位上限60%

**实现**: 修改 `_compute_overlay_exposure()` 中的regime damping逻辑。

### 3C. 波动率目标化 (Volatility Targeting)

**原理**: 根据近期组合波动率动态调整仓位，保持年化波动率在目标水平。

**参数**: vol_target = 20% (年化)

**公式**:
```python
realized_vol = std(portfolio_daily_returns[-20:]) * sqrt(252)
vol_exposure = min(1.0, vol_target / realized_vol)
final_exposure = min(cppi_exposure, vol_exposure, regime_exposure)
```

**预期**: 高波动期自动减仓，低波动期满仓。与CPPI形成双重保护。

## 4. Implementation Plan

**Phase 1: 在回测引擎中添加3个overlay参数**
- `--stop-loss`: 个股止损百分比 (默认0=关闭)
- `--regime-gate-aggressive`: 使用增强版regime门控
- `--vol-target`: 波动率目标(默认0=关闭)

**Phase 2: 网格搜索最优组合**
- stop_loss ∈ [-6%, -8%, -10%, 0(off)]
- regime_gate ∈ [off, standard, aggressive]
- vol_target ∈ [0(off), 15%, 20%, 25%]
- CPPI ∈ [F0.07/M20, F0.08/M20]
- 总组合: 4 × 3 × 4 × 2 = 96种

**Phase 3: 验证最优配置**

## 5. Success Criteria

| 指标 | 当前最优 | 目标 |
|:-----|:---:|:---:|
| MaxDD | -12.6% | **< -10%** |
| Sharpe | 2.339 | **> 1.5** |
| 年化(毛) | 72.2% | **> 30%** |
| 换手率 | 24x | **< 20x** |

## 6. Files to Modify

| File | Changes |
|------|---------|
| `backtest/backtest_report_based.py` | 添加stop-loss + vol-target + aggressive regime gate |
| `backtest/run_north_star_eval.py` | 添加对应CLI参数 |
| `scripts/ng105_overlay_grid_search.py` | 新建: overlay组合网格搜索 |
| `production_config.json` | 更新为最优配置 |

## 7. Risk

- 个股止损可能在震荡市频繁触发，增加换手
- Vol-targeting + CPPI叠加可能过度去风险化，压低收益
- 需要回测验证不会出现"总是空仓"的极端情况
