# StockTradebyZ Wiki

项目知识库 — 记录系统演化决策、模型实验结论、踩坑教训。

## 架构
- [系统架构总览](architecture/system-overview.md) — 整体架构、数据流、组件关系图
- [数据管线](architecture/data-pipeline.md) — Tushare API → SQLite → Feature Cache 全链路
- [ML 管线](architecture/ml-pipeline.md) — 特征工程 → 训练 → 推理 → 报告生成
- [自动调仓系统](architecture/auto-rebalancing.md) — EastMoneyTrader联动、NG1.0.5风控、执行策略回测
- [EMT 量化分析框架](architecture/emt-analysis-framework.md) — Level 1 完整覆盖: IC/中性化/组合优化/回测验证/信号驱动调仓/退市股修复
- [Signal Trust 信号可信度系统](architecture/signal-trust.md) — 给选股贴 🟢🟡🔴 可信度标签, 识别假信号
- [0AMV Regime Classifier v1](architecture/regime-classifier-v1.md) — 牛熊分辨器 V11 上线 (位置+水上/上升+3日平滑), 三窗口击败旧 V3 strict
- [风控管线](architecture/risk-control-pipeline.md) — 选股报告 4 层风控 (post_filters → booster → ng21 overlay → P0.1 sizing) + score-scale 自适应约定

## 模型演化
- [模型世代总览](models/evolution.md) — V3.8 → V3.9 → V4.x → NG 系列演化路径与关键转折
- [NG 系列详解](models/ng-series.md) — NG 1.0.0 → 1.0.9 每版本改动与性能（ng1.0.8最优:Sharpe=2.52）
- [NG 因子质量与权重分布](models/ng-factor-quality.md) — 每版本因子重要性排名、组占比、近零因子、跨版本对比
- [V4.x 实验总结](models/v4x-series.md) — V4.4 → V4.9.0.1 实验迭代、成功与失败
- [V3.9/3.95 旧版参考](models/v39-series.md) — 旧版 ensemble 系统说明（保留参考）

## 评估体系
- [北极星评估体系](evaluation/north-star.md) — V1 → V5.2 演化、评分规则、如何解读
- [回测方法论](evaluation/backtesting.md) — 无泄露原则、交易成本建模、CPPI 风控
- [8策略长期回测(2018-2026)](evaluation/quant-strategies-2018-2026.md) — 40K信号: 暴力K最强(+1.59% alpha), 知行是噪音, 策略有强regime分化

## 教训
- [已知陷阱](lessons/known-pitfalls.md) — 数据泄露、过拟合、pipe 死锁等踩坑汇总

## 特征工程
- [特征指南](features/feature-guide.md) — 69 个特征的含义、来源、选择逻辑

---
*维护规则见 [schema.md](schema.md) | 里程碑时间线见 [log.md](log.md)*
