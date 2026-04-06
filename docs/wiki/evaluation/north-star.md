# 北极星评估体系

北极星（North Star）是项目的统一模型评估框架，从 V1 演化到 V5.2，用于无泄露地衡量选股模型的综合表现。

## 版本演化

| 版本 | 指标数 | 满分 | 分档 | 时间 |
|---|---|---|---|---|
| V1 | 11 | 33 | A+/A/B/C | 2026-02 |
| V2 | 21 | 105 | S/A+/A/B/C/D | 2026-02 |
| V5 | ~21 | 百分比 | S/A+/A/B/C/D | 2026-04 |
| V5.2 | ~21 | 百分比 | S/A+/A/B/C/D | 2026-04 (当前) |

## V5.2 评分体系（当前生产）

### 四层结构

**L1 信号检测 (30分, 权重最高)**
- IC均值: Information Coefficient 均值
- ICIR: IC 的 Sharpe Ratio（IC均值/IC标准差）
- IC>0 比例: 有多少天模型预测方向正确
- 年化超额: 模型选股 vs 基准的年化超额收益
- 月度胜率: 每月跑赢基准的比例

**L2 组合效率 (25分)**
- Sharpe Ratio: 风险调整收益
- Sortino Ratio: 下行风险调整收益
- 最大回撤: MaxDD
- 平均回撤: 回撤均值
- 波动率: 日收益标准差年化

**L3 风险控制 (25分)**
- 单仓占比: 最大单只持仓比例
- 行业集中度: HHI 指数
- 涨停失败率: 买入当天涨停被撤的比例
- 前后半段一致性: 前半段和后半段评分一致
- 最差60日 ICIR: 最差连续60日的信号质量

**L4 执行纪律 (25分)**
- 年化收益: 毛收益和净收益
- 月度胜率: 绝对收益月度胜率
- 一致性: 滚动窗口评分稳定性
- 市值均衡: 不过度偏向大盘/小盘
- 中位市值: 选股的中位市值

### 分档标准

| 档次 | 百分比 | 含义 |
|---|---|---|
| S | ≥80% | 卓越 |
| A+ | ≥70% | 优秀 |
| A | ≥60% | 良好 |
| B | ≥45% | 合格 |
| C | ≥30% | 勉强 |
| D | <30% | 不合格 |

## 评估模式

### 1. WF-OOS（向前泛化）— 标准模式
```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101_wf_oos \
    --label WF-OOS --top-n 10 --focus-days 10 --rank-field composite
```
- 使用 Walk-Forward 的 Out-of-Sample 窗口
- 评估模型能否预测未来
- 合理预期：B ~ A 级

### 2. Pre-2020（向后泛化）
```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng101_pre2020 \
    --label PRE-2020 --top-n 10 --focus-days 10 --rank-field composite
```
- 用生产模型预测 2018-2019 数据（完全未见过的时期）
- 评估模型学到的是通用规律还是过拟合
- A 级说明信号真实

### 3. --production（含泄露，仅参考）
```bash
python3 backtest/run_north_star_eval.py --production
```
- ⚠️ 训练数据和回测数据有重叠
- 评分会虚高（通常 S 级）
- **不可作为模型评估依据**

## 评估解读

| WF-OOS | Pre-2020 | 结论 |
|---|---|---|
| A+ | A+ | 高置信，模型学到真实 alpha |
| A | B | 模型有效但可能有时期依赖 |
| B | A | 模型泛化能力好但近期信号弱 |
| B/C | B/C | 模型有问题，需重新审视 |

## 关键参数

| 参数 | 推荐值 | 说明 |
|---|---|---|
| --top-n | 10 | Top-10 持仓 |
| --focus-days | 10 | 10日持仓周期 |
| --rank-field | composite | 排名方式（也可用 pred_10d） |
| --extended | - | 扩展窗口评估 |
| --regime-analysis | - | 市况分析（牛/熊/震荡） |

## 实现文件

- 指标计算: `backtest/north_star_metrics.py`
- CLI 工具: `backtest/run_north_star_eval.py`
- 回测引擎: `backtest/backtest_report_based.py`

## 相关页面

- [回测方法论](backtesting.md)
- [模型世代总览](../models/evolution.md)
- [已知陷阱 — 数据泄露](../lessons/known-pitfalls.md#数据泄露--production-回测)
