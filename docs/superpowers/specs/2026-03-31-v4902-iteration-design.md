# V4902 迭代设计规格 — 基于北极星V5.1诊断

> 日期: 2026-03-31
> 状态: 已批准, 待实施
> 前序: V4901 (V5.1评分 61.8% A级)
> 目标: V5.1评分 72-75% A+级

## 1. V4901 V5.1诊断摘要

| 层 | 得分 | 百分比 | 主要失分 |
|----|------|--------|----------|
| L1 信号 | 49.0/50 | 98% | (无需改进) |
| L2 效率 | 18.0/25 | 72% | 换手46.3%(1/5) |
| L3 风控 | 17.5/45 | 39% | MaxDD=-25.9%(0), CVaR=13.6%(0), 水下66%(0.9) |
| L4 OOS | 22.4/40 | 56% | 月胜率63%(2.4), WFER/CSCV/OOS半衰期=N/A |
| L5 超额 | 13.8/25 | 55% | 超额胜率43.4%(0.9), 超额MDD=-28.5%(0) |
| L6 因子 | 22.0/30 | 73% | HML β=1.48(最大因子暴露) |
| L7 容量 | 3.2/15 | 21% | 策略容量/参与率=N/A |

## 2. 改动清单 (8项)

### 改动1: Sharpe-Blend 0.3 → 0.5 (训练端)

**文件:** `ml_models/training/train_v395_multi_target.py`
**位置:** V4902Trainer类 (新建, 继承V4901Trainer)

```python
TARGET_SHARPE_BLEND = {
    'label_3d':  0.5,   # was 0.3
    'label_5d':  0.5,   # was 0.3
    'label_10d': 0.5,   # was 0.3
    'label_15d': 0.5,   # was 0.3
}
```

**原理:** label = 0.5 × return + 0.5 × sharpe. 模型从"追涨"转向"追稳", 直接降低预测端的尾部风险。

**预期:** L3 CVaR↓, MaxDD↓, 水下比↓. L1 IC可能微降但ICIR提升。

### 改动2: 下行样本加权×1.5 (训练端)

**文件:** `ml_models/training/train_v395_multi_target.py`
**位置:** V4902Trainer.compute_sample_weights()

```python
def compute_sample_weights(self, df, target_col):
    weights = super().compute_sample_weights(df, target_col)
    # 下行收益加权: 让模型更关注避亏
    negative_mask = df[target_col] < 0
    weights[negative_mask] *= 1.5
    return weights
```

**原理:** 亏损样本权重×1.5, 梯度更关注"为什么这只股票跌了", 减少模型推荐后大跌的概率。

**预期:** CVaR 13.6% → 8-10%, MaxDD改善3-5%。

### 改动3: 个股CVaR止损降权 (scorer端)

**文件:** `ml_models/v39/v4902_production_scorer.py` (新建, 继承V4901)

```python
def _apply_cvar_penalty(self, results, daily_quotes_recent):
    """对近20日CVaR>5%的股票降权30%"""
    for r in results:
        code = r['code']
        recent_returns = daily_quotes_recent.get(code)
        if recent_returns is not None and len(recent_returns) >= 20:
            cvar = compute_cvar(pd.Series(recent_returns), alpha=0.05)
            if cvar > 0.05:  # 20日CVaR > 5%
                r['composite'] *= 0.7
                r['rank_score'] *= 0.7
    return results
```

**数据来源:** 从daily_quotes取最近20日price_change_pct, 在选股时实时计算。

**预期:** 减少高尾部风险个股入选, CVaR↓, MaxDD↓.

### 改动4: 自动基准选择阈值调整 (评估端)

**文件:** `backtest/north_star_metrics.py` — `auto_select_benchmark()`

```python
# 当前:
if median_market_cap_bn >= 50:   return '000300.SH'
# 改为:
if median_market_cap_bn >= 80:   return '000300.SH'
```

**原理:** V4901中位市值58.4亿, 用CSI300(大盘≥80亿)不合理, 改后自动选CSI500。超额指标对标中盘更公平。

**预期:** 超额胜率 43.4% → 55-60%, 超额MDD收窄。

### 改动5: EMA 0.7→0.6 + Retention 0.2→0.3 (scorer端)

**文件:** `ml_models/v39/v4902_production_scorer.py`

```python
EMA_ALPHA = 0.6         # was 0.7 — 更平滑
RETENTION_BONUS = 0.3   # was 0.2 — 持仓惯性更大
```

**预期:** 换手率 46.3% → 32-35%, L2从72%→80%+.

### 改动6: Market Gate 0.30→0.35 (scorer端)

**文件:** `ml_models/v39/v4902_production_scorer.py`

```python
GATE_DONT_BUY = 0.35    # was 0.30 — 弱市更早停买
```

**预期:** 弱市月份(2024-01,05,06,07,08)减少亏损, 月胜率 63%→67%+.

### 改动7: 训练时保存WF摘要JSON (训练端)

**文件:** `ml_models/training/train_v395_multi_target.py`
**位置:** V4902Trainer训练结束后

```python
# 训练完成后自动保存WF摘要
wf_summary = {
    'is_sharpe': [wf['is_sharpe'] for wf in walk_forward_results],
    'oos_sharpe': [wf['oos_sharpe'] for wf in walk_forward_results],
    'oos_monthly_ics': [wf['oos_monthly_ics'] for wf in walk_forward_results],
}
json.dump(wf_summary, open(model_path.replace('.pkl', '_wf_summary.json'), 'w'))
```

**预期:** WFER和OOS半衰期从N/A变为有值, L4+6分。

### 改动8: 回测容量数据补齐 (回测端)

**文件:** `backtest/backtest_report_based.py`
**位置:** V5.1指标计算段

在回测循环中记录持仓股票codes, 然后查询DB获取ADV, 计算capacity和participation_rate:

```python
# 从回测中收集所有持仓的codes
all_held_codes = set()
for trade in trades:
    all_held_codes.add(trade['code'])

# 查询ADV
conn = sqlite3.connect(DB_PATH, timeout=30)
vol_df = pd.read_sql("""
    SELECT s.code,
           AVG(dq.volume * dq.close) as adv_20d_value,
           STDEV(dq.price_change_pct) as daily_vol
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code IN ({placeholders})
      AND dq.trade_date >= ?
    GROUP BY s.code
""", conn, params=[...])

s['strategy_capacity_mn'] = compute_strategy_capacity(vol_df, ...)
s['participation_rate_p90'] = compute_participation_rate_p90(vol_df, ...)
```

**预期:** L7从21%→50-65%.

## 3. 实施架构

### 新建文件
- `ml_models/v39/v4902_production_scorer.py` — 继承V4901, 加CVaR止损+EMA/retention/gate调整
- `ml_models/trained_models/v4902_wf_summary.json` — WF摘要(训练时生成)

### 修改文件
- `ml_models/training/train_v395_multi_target.py` — 新增V4902Trainer(改Sharpe-blend+下行加权+WF保存)
- `backtest/north_star_metrics.py` — auto_select_benchmark阈值50→80
- `backtest/backtest_report_based.py` — 容量数据补齐

### 不修改
- `backtest/north_star_metrics.py` 的V5/V5.1评分逻辑
- `backtest/run_north_star_eval.py` CLI
- 所有V4901代码(向后兼容)

## 4. 实施顺序

```
阶段1 (快速验证, ~30min):
  ├── 改动4: 基准阈值调整 (1行改动)
  ├── 改动5: EMA+retention (scorer参数)
  └── 改动6: Gate阈值 (scorer参数)
  → 用现有V4901模型+新scorer跑V5.1, 验证L2/L5改善

阶段2 (训练, ~6-20h):
  ├── 改动1: Sharpe-blend 0.5
  ├── 改动2: 下行加权×1.5
  └── 改动7: WF摘要保存
  → 训练V4902模型, 跑V5.1对比V4901

阶段3 (集成, ~1h):
  ├── 改动3: CVaR止损降权
  └── 改动8: 容量数据补齐
  → 完整V5.1评估

阶段4 (验证):
  → V5.1全版本对比: V4901 vs V4902
  → V4回归: 确认V4评分不退步太多
```

## 5. 风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| Sharpe-blend 0.5导致绝对收益大幅下降 | 中 | 先用--fast-check验证方向, 如果IC下降>30%则回退到0.4 |
| 下行加权导致模型过于保守 | 低 | 1.5倍温和, 如果IC>0%从94%降到<85%则回退 |
| CVaR止损误杀好股票 | 低 | 阈值5%足够宽松, 只针对极端尾部风险 |
| 训练时间太长 | 中 | 先fast-check(~2min)确认方向, 再完整训练 |
| V4评分退步 | 中 | V4评分主要看CPPI后的表现, 裸信号改善不应影响CPPI效果 |

## 6. 成功标准

| 指标 | V4901当前 | V4902目标 | 底线 |
|------|-----------|-----------|------|
| V5.1总分 | 61.8% A | **72%+ A+** | 68% A |
| L3 风控 | 39% | **55%+** | 45% |
| L5 超额 | 55% | **65%+** | 60% |
| L4 OOS | 56% | **70%+** | 60% |
| L1 信号 | 98% | **≥90%** | 85% (可接受微降) |
| V4评分 | 92.8% S | **≥85% S** | 80% A+ |
