# 北极星评估体系

北极星（North Star）是项目的统一模型评估框架，从 V1 演化到 V5.2，用于无泄露地衡量选股模型的综合表现。

> **⚠️ 口径分界 (2026-07-11-p0fix)**: 修复了基准 2022-24 NULL / 未复权价格 /
> -10% 惩罚三合一 / PSR-DSR 饱和 / L8 恒满分 / L9 重叠序列膨胀 六项缺陷
> (详见 `reports/system_evaluation/选股系统与北极星系统评估与风控内化可行性研究_20260711.md`)。
> **此日期之前产出的所有北极星数字与新跑数字不可混比**; 新基线见
> `reports/system_evaluation/新口径基线重跑_20260711.md`。EvalCache 键已含
> 数据/引擎指纹, 引擎改动自动失效缓存。

## 版本演化

| 版本 | 指标数 | 满分 | 评分方式 | 时间 |
|---|---|---|---|---|
| V1 | 11 | 33 | 离散(0-3) | 2026-02 |
| V2 | 21 | 105 | 离散(0-5) | 2026-02 |
| V3 | 25 | 125 | 离散(0-5) | 2026-03 |
| V4 | 31 | 155 | 离散(0-5) | 2026-03 |
| V5 | 39 | 195 | **连续插值(0.0-5.0)** | 2026-03 |
| V5.1 | 48 | 240 | 连续插值 | 2026-03 |
| V5.2 | 59 | 295 | 连续插值 + 风格自适应 | 2026-04 (当前) |

**V5+ 关键突破**: 从离散5档改为连续浮点插值 `np.interp`，0.01分精度，消除同分聚集效应。

## V5.2 评分体系（当前生产）

### 九层结构（59指标，295满分）

| 层 | 名称 | 权重 | 指标数 | 满分 |
|---|---|---|---|---|
| L1 | 信号质量 Signal Quality | 22% | 10 | 50 |
| L2 | 组合效率 Portfolio Efficiency | 10% | 5 | 25 |
| L3 | 风险控制 Risk Control | 15% | 9 | 45 |
| L4 | OOS鲁棒性 OOS Robustness | 12% | 8 | 40 |
| L5 | 超额收益 Excess Returns | 8% | 5 | 25 |
| L6 | 因子归因 Factor Attribution | 8% | 6 | 30 |
| L7 | 容量限制 Strategy Capacity | 8% | 7 | 35 |
| L8 | 执行质量 Execution Quality | 8% | 5 | 25 |
| L9 | 条件稳健 Regime Robustness | 9% | 4 | 20 |

**L1 信号质量 (22%, 10指标)** — 权重最高
- daily_ic, icir, ic_positive_pct, ic_monotonicity, ic_time_stability
- signal_half_life, bear_icir, ic_decay_ratio, ic_autocorr_1d, transfer_coefficient

**L2 组合效率 (10%, 5指标)**
- annual_turnover, annual_cost_drag, net_gross_ratio, limit_up_fail_rate, liquidity_coverage

**L3 风险控制 (15%, 9指标)**
- max_drawdown, sharpe_ratio, worst_rolling_60d_icir, tail_ratio, cvar_5pct
- max_dd_duration, underwater_ratio, monthly_win_rate, volatility

**L4 OOS 鲁棒性 (12%, 8指标)**
- annual_return, monthly_win_rate, probabilistic_sharpe(PSR), deflated_sharpe(DSR)
- wfer(WF效率比), oos_ic_half_life, monthly_consistency ×2

**L5 超额收益 (8%, 5指标)**
- excess_annual_return, information_ratio, excess_win_rate, excess_max_drawdown, up_capture_ratio

**L6 因子归因 (8%, 6指标)** — Fama-French 4因子日回归
- residual_alpha_t, factor_r_squared, active_share, max_factor_loading, smb_beta, mom_beta

**L7 容量限制 (8%, 7指标)**
- strategy_capacity_mn, participation_rate_p90, effective_n_corr, hurst_exponent
- regime_transition_dd, cscv_pbo, adv_coverage

**L8 执行质量 (8%, 5指标)** — V5.2新增
- delay_cost, execution_fill_rate, realized_vs_theoretical, turnover_efficiency, implementation_shortfall

**L9 条件稳健 (9%, 4指标)** — V5.2新增，需≥200交易日
- regime_ic_consistency, regime_sharpe_floor, multi_benchmark_excess, regime_drawdown_ratio

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
