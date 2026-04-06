# 回测方法论

回测系统的设计原则、交易模拟规则、成本建模。

## 核心原则：无泄露

**训练数据不可用于回测**。所有模型评估必须使用无泄露方法：

1. **Walk-Forward OOS**: 训练窗口和测试窗口严格分离，中间有 purge gap
2. **Pre-2020**: 用 2020+ 训练的模型预测 2018-2019 数据
3. **Purge gap**: 15 天，防止标签跨期泄露

```
|--- Training ---|-- Purge(15d) --|--- Test (OOS) ---|
                  ← 不可回测 →
```

> ⚠️ `--production` 模式训练/回测数据有重叠，评分虚高，仅供内部参考。

## 交易模拟规则

### A股特殊规则
- **T+1**: 当日买入次日才能卖出
- **涨停检测**: 买入当天涨停则交易失败
  - 主板: price_change_pct ≥ 9.5%
  - 创业板/科创板: ≥ 19.5%
  - 北交所: ≥ 29.5%
  - 注意：不用 `is_limit_up` 字段（不可靠），用 `price_change_pct` 判断

### 交易成本模型
| 成本项 | 比率 | 说明 |
|---|---|---|
| 佣金 | 0.025% × 2 | 双边 |
| 印花税 | 0.05% | 卖出单边 |
| 滑点 | 0.1% × 2 | 双边 |
| 过户费 | 0.002% × 2 | 双边 |
| **总计** | **0.302%** | 单次调仓 |

CPPI 调仓时，exposure 变化 >1% 则扣减 0.302% × 变化量。

## CPPI 风控框架

Constant Proportion Portfolio Insurance（固定比例组合保险）：

```python
cushion = portfolio_value - floor_value          # 安全垫
risky_allocation = min(multiplier * cushion, 1.0) # 风险资产比例
```

| 参数 | 当前生产值 | 含义 |
|---|---|---|
| floor | 5%~8% | 最大可承受亏损 |
| multiplier | 20 | 杠杆倍数 |
| decaying_peak | 0.995/day | 峰值衰减（避免永远锁仓） |

### CPPI 效果实证
- MaxDD: -28.5% → -11.5%（V4.6 base → +CPPI）
- Sharpe: 0.41 → 1.28
- **关键认知**: CPPI 是风控工具，不提升信号质量

### CPPI 参数实验结论 (2026-03-01)
- 天花板: 85/105 S级，无法超越
- 小 floor(0.05) + 大 multiplier(20) 最优
- retention_bonus / 流动性过滤 / 替换门槛均损害 alpha

## 回测引擎

### 基于报告的回测 (推荐)
```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101 \
    --label NG101 --top-n 10 --focus-days 10
```

从已生成的日报 JSON 中读取选股结果，模拟交易：
1. 每 focus_days 天调仓一次
2. 按 rank_field 排名取 top_n
3. 等权分配持仓
4. 扣除交易成本
5. 计算北极星指标

### 批量报告生成
```bash
python3 backtest/batch_generate_v395_reports.py \
    --version ng1.0.1 \
    --start-date 2020-01-01 --end-date 2026-04-03
```

### 关键参数

| 参数 | 说明 | 推荐值 |
|---|---|---|
| --top-n | 持仓数 | 10 |
| --focus-days | 调仓周期(天) | 10-15 |
| --rank-field | 排名方式 | composite 或 pred_10d |
| --report-dir | 报告目录 | reports/daily_selection_* |

## 常见错误

1. **用 --production 评模型** → 数据泄露，评分虚高
2. **忽略交易成本** → 0.302% × 频繁换手 = 大量 alpha 被吃掉
3. **用 is_limit_up 判涨停** → 字段不可靠，用 price_change_pct
4. **CPPI 调仓不算成本** → 每次 exposure 变化都有摩擦成本

## 实现文件

- 北极星指标: `backtest/north_star_metrics.py`
- 评估 CLI: `backtest/run_north_star_eval.py`
- 报告回测: `backtest/backtest_report_based.py`
- 交易环境: `backtest/backtest_trading_environment.py`
- 批量报告: `backtest/batch_generate_v395_reports.py`

## 相关页面

- [北极星评估体系](north-star.md)
- [已知陷阱 — 回测类](../lessons/known-pitfalls.md#回测类)
- [模型世代总览](../models/evolution.md)
