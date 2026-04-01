# V4.9.2 迭代计划 — 从82.8% S级推向85%+ S级

## 现状诊断

**基线**: V4.9.0.1 comp + Ret0.2 + CPPI(8,20) = **82.8% S级 (128/155)**

### 丢分清单 (27分)

| 指标 | 当前 | 得分 | 丢分 | 目标 | 优先级 |
|------|------|------|------|------|--------|
| 年化换手 | 47.3x | 0/5 | **5** | 20x | P0 |
| 熊市超额 | N/A | 0/5 | **5** | 30% | P3(数据限制) |
| MaxDD | -13.5% | 2/5 | **3** | -8% | P1 |
| 超额MDD | -15.2% | 2/5 | **3** | -5% | P2 |
| IC衰减比 | 77.4% | 2/5 | **3** | 95% | P1 |
| 流动性覆盖 | 80.8% | 2/5 | **3** | 95% | P2 |
| 年化成本 | 7.1% | 3/5 | **2** | 5% | 随换手降 |
| Sharpe | 2.37 | 3/5 | **2** | 3.0 | P1 |
| 月度胜率 | 74.1% | 3/5 | **2** | 83% | P2 |
| 超额胜率 | 62.3% | 4/5 | **1** | 65% | P3 |
| IR | 1.65 | 4/5 | **0** | ✅满分 | — |
| 前后半段一致 | 88.0% | 5/5 | **0** | ✅满分 | — |

### 可攻方向分析

1. **换手 (5分)**: 最大单项丢分，但V4.9.0.1每次调仓94.6%换手说明模型预测排名极不稳定。retention_bonus=0.2只降了0.4x，几乎无效。**需要模型层面干预**。

2. **IC衰减 (3分)**: 后半段IC均值0.14 vs 前半段0.20。2026年1-2月IC跌至0.03-0.08。**模型对近期市场regime适应慢**。

3. **MaxDD/Sharpe (5分合计)**: CPPI已把-18.8%压到-13.5%，但离-8%目标还远。需要**更激进的尾部保护或波动率控制**。

---

## V4.9.2 改进方案 (3层)

### 层1: 预测平滑 (目标: 换手47→30x, +2~3分)

**原理**: 当前模型每天独立预测，两天之间pred_10d可能完全翻转。但真正的alpha不会一天内消失——用EMA平滑消除噪声翻转。

**方法: Exponential Moving Average Prediction Smoothing**

参考: [arXiv:2506.06356](https://arxiv.org/abs/2506.06356) — 多日换仓量化算法，通过参数平滑（0.7×当前 + 0.3×前期）减少不必要的换仓。

```python
# 在scorer层, 对每只股票的pred_10d做时序EMA
# 需要维护一个 {code: last_pred_10d} 的状态缓存

class PredictionSmoother:
    def __init__(self, alpha=0.6):
        """alpha: 当日权重。0.6=较快响应, 0.3=很平滑"""
        self.alpha = alpha
        self.cache = {}  # {code: smoothed_pred_10d}

    def smooth(self, code, raw_pred):
        if code in self.cache:
            smoothed = self.alpha * raw_pred + (1 - self.alpha) * self.cache[code]
        else:
            smoothed = raw_pred
        self.cache[code] = smoothed
        return smoothed
```

**实现位置**: V4.9.2 ProductionScorer, 在predict_scores返回前对pred_10d应用EMA。需要在batch_generate中按时间顺序调用（当前已是按日期排序）。

**预期效果**:
- alpha=0.6: 换手降30-40% (47→28-33x), 成本降~2%
- alpha=0.4: 换手降50%+ (47→23x), 但可能损失IC响应速度

**风险**: 过度平滑会降低IC（信号变迟钝）。需要在IC和换手之间找平衡点。

---

### 层2: Walk-Forward步长缩短 + 窗口加权 (目标: IC衰减77→88%+, +2分)

**原理**: 当前WF用step_days=90（每90天更新一次模型），最后一个窗口到回测末期已经过时6个月。后半段IC衰减根因是**模型对市场regime变化反应太慢**。

**方法A: 缩短WF步长**

```python
# 当前: step_days=90, 4个窗口
# 改为: step_days=60, 6个窗口
# 每个窗口的模型更"新鲜"
```

训练时间从~6h增加到~9h（6个窗口×1.5h），但模型在后半段更适应。

**方法B: 时间加权集成 (Temporal Ensemble Weighting)**

参考: 量化基金常用做法——多个WF窗口的模型按时间衰减加权，而非等权平均。

```python
# 当前: 所有WF窗口的模型等权集成
# 改为: 指数衰减加权, 最近的窗口权重更大
# weight_i = exp(-lambda * (n_windows - i))

window_weights = [exp(-0.3 * (4-i)) for i in range(5)]
# W1=0.30, W2=0.37, W3=0.45, W4=0.55, W5=0.67 → 归一化
```

**实现位置**: train_v395_multi_target.py 的 walk_forward_train 中，最终模型集成改用时间加权。

**预期效果**: IC衰减比从77%→85-90%（后半段模型更新鲜，预测更准）

---

### 层3: 自适应波动率目标 (目标: MaxDD -13.5→-10%, Sharpe 2.37→2.8+, +2分)

**原理**: 当前CPPI用固定floor/multiplier，但市场波动率本身在变化。高波动期应该更激进地减仓。

**方法: Moreira-Muir Vol Scaling + CPPI双层叠加**

参考: Moreira & Muir (2017) "Volatility-Managed Portfolios" — 在已有CPPI基础上，按波动率反比缩放仓位。

```python
# 当前: CPPI alone
# 改为: Vol-Scaling × CPPI

realized_vol_20d = std(daily_returns[-20:]) * sqrt(252)
vol_target = 0.15  # 目标年化波动率15%
vol_scale = min(1.0, vol_target / realized_vol_20d)

# 最终仓位 = vol_scale × cppi_exposure
final_exposure = vol_scale * cppi_exposure
```

**实现位置**: backtest_report_based.py 中已有 `vol_target` 参数，只需与CPPI叠加。

**预期效果**: MaxDD进一步压缩2-3pp, Sharpe提升0.3-0.5

---

## 实施计划

### Phase 1: 预测平滑 (scorer层, ~2小时)

1. 创建 `V492ProductionScorer(V490ProductionScorer)`
2. 添加 `PredictionSmoother` 类
3. 在 `predict_scores` 返回前对 pred_10d/pred_15d 应用EMA
4. 修改 `batch_generate` 确保按时间顺序调用（已满足）
5. 参数搜索: alpha ∈ {0.4, 0.5, 0.6, 0.7}
6. 评估: 对每个alpha跑北极星, 选最优

**验证指标**: 换手降到30x以下, IC不低于0.16

### Phase 2: WF优化 (训练层, ~10小时训练)

1. 创建 `V492Trainer(V4901Trainer)` — 实际是复用V485Trainer
2. 修改 `step_days=60`, 增加到6个窗口
3. 添加时间加权集成逻辑
4. 训练新模型
5. 评估IC衰减比变化

**验证指标**: IC衰减比提升到85%+, WF平均10d ICIR不降

### Phase 3: Vol-CPPI叠加 (回测层, ~30分钟)

1. 在 `run_single_backtest` 中添加 `vol_target` + `cppi` 联合模式
2. 参数搜索: vol_target ∈ {0.12, 0.15, 0.18} × CPPI(5-10, 15-25)
3. 选最优组合

**验证指标**: MaxDD < -10%, Sharpe > 2.5

### Phase 4: 综合评估

1. 最优Phase 1 + Phase 2 + Phase 3 叠加
2. 北极星V4评分
3. 目标: **85%+ S级 (132+/155)**

---

## 预期效果汇总

| 改进 | 目标指标 | 预期提分 | 实现成本 |
|------|---------|---------|---------|
| 预测EMA平滑 | 换手47→30 + 成本降 | +2~3分 | 2小时 |
| WF步长60d + 时间加权 | IC衰减77→88% | +1~2分 | 10小时训练 |
| Vol+CPPI叠加 | MaxDD -13.5→-10% | +1~2分 | 30分钟 |
| **合计** | | **+4~7分 (82.8→87%)** | |

---

## 风险与备选

1. **EMA过度平滑**: 如果alpha=0.5导致IC<0.15, 回退到alpha=0.7
2. **WF 6窗口过拟合**: 如果更多窗口反而ICIR降低, 保持4窗口+时间加权
3. **Vol-Scaling + CPPI冲突**: 两者可能过度减仓, 需要设置最低仓位 min_exposure=0.3

## 执行顺序

Phase 1 (平滑) 和 Phase 3 (Vol-CPPI) 可以**并行**——前者改scorer，后者改回测参数，互不影响。Phase 2 (WF) 需要训练，耗时最长，**先启动后台训练**。

推荐: **先做Phase 1+3（快速验证）, 然后Phase 2（后台训练）, 最后综合叠加**。
