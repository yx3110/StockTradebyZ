# V4.x 实验总结

V4.x 系列是密集实验迭代期（2026-02-24 ~ 2026-03-07），探索了多个方向，最终经验沉淀为 NG 系列的设计基础。

## 版本一览

| 版本 | 核心实验 | 结果 | 关键教训 |
|---|---|---|---|
| V4.0 | Cross-Sectional Alpha | ❌ 失败 | 纯截面模型丢失择时信号 |
| V4.3 | WF + 强正则化 | ✅ 有效 | WF 验证成为标配 |
| V4.4 | 数据对齐修复 | ✅ **88/105 S级** | 数据对齐比模型复杂度重要 |
| V4.5 | CPPI Overlay | ✅ **84/105 S级** | CPPI 大幅改善风控 |
| V4.6 | ICIR权重+Meta-Learner | ⚠️ 需CPPI | 后处理可能损害信号 |
| V4.7.2 | composite 排名 | ✅ 71/105 A | 连续值排名优于离散化 |
| V4.7.3 | 去 Meta-Learner | ✅ **75/105 A+** | 裸信号质量更优 |
| V4.7.4 | 6项同时改动 | ❌ 失败 | 同时改太多无法归因 |
| V4.7.5 | 特征裁剪+连续评分 | ✅ **77/105 A+** | pred_10d > composite |
| V4.8.x | 因子挖掘 pipeline | ❌ 过拟合 | fast=86.59, WF全量=58.44 |
| V4.9.0.1 | EMA平滑+市场门控 | ✅ 生产版 | 含泄露S级，真实B-A级 |

## 详细记录

### V4.0 Cross-Sectional Alpha (2026-02-23) ❌
- 完全去除时序信号，只用截面排名
- 5d 持仓 IC=0.010, ICIR=0.062（近随机）
- **根因**: 纯 cross-sectional 移除了市场择时信号，stock-picking 能力有限

### V4.3 Walk-Forward + 强正则化 (2026-02-24) ✅
- 59 特征(+10技术指标)，强正则化，Walk-Forward 验证
- WF OOS ICIR: 3d=0.930, 5d=0.926, 10d=0.750, 15d=0.804
- **贡献**: 确立 WF 作为标准评估方法

### V4.4 数据对齐修复 (2026-02-25) ✅ 关键里程碑
- 修复 4 大训练/推理不一致问题（详见 [已知陷阱](../lessons/known-pitfalls.md)）
- V4.4-aligned: 88/105 S级（WF ICIR: 3d=0.964, 5d=1.052, 10d=0.882）
- **教训**: 仅修复数据对齐就让模型从 B 级飞到 S 级

### V4.6 ICIR 权重 + Meta-Learner (2026-02-28~03-01) ⚠️
- 新增: ICIR 权重优化、Meta-Learner(Ridge)、Combined Isotonic、小盘加权(1.5x)
- Base: 64/105 A级（MaxDD=-28.5%）
- +CPPI: 84/105 S级（MaxDD=-11.5%）
- **关键发现**: 小盘加权 ×2.5 扭曲学习，重训练后 WF IC 近随机

### V4.7.3 裸信号提纯 (2026-03-05) ✅
- 去掉 Meta-Learner + Combined Isotonic，num_leaves 20→31
- 75/105 A+, Sharpe=1.230, MaxDD=-18.4%
- **教训**: 后处理层（Meta-Learner、Isotonic、离散化）都在压缩头部区分度

### V4.7.5 特征裁剪+连续评分 (2026-03-07) ✅
- 特征 70→50，np.interp 连续评分，自适应权重
- pred_10d 排名: 77/105 A+（优于 composite 69/105）
- **教训**: composite 不总优于单目标排名；自适应权重有害（短周期权重过高）

### V4.8.x 因子挖掘 (2026-03-24~26) ❌
- 自动因子挖掘 pipeline: 算子×操作数×窗口 + IC 筛选
- Fast-check 最优 86.59（69特征+RRF+LambdaRank）
- WF 全量验证: 58.44（严重过拟合）
- **教训**: fast-check 和全量之间差距大说明 overfit，不能信 fast 结果

### V4.9.0.1 EMA + 市场门控 (生产版)
- V4.9.0 底座 + EMA 平滑(alpha=0.7) + 市场门控(GateV2, AUC=0.714)
- --production 回测: 92.8% S级（含泄露）
- WF-OOS 真实: 54.1% B级
- **根因**: β_UMD=3.029 隐性动量暴露

## 从 V4.x 沉淀到 NG 的设计原则

1. **Walk-Forward 必须** — V4.3 起成为标配
2. **裸信号优先** — 去掉 Meta-Learner/Isotonic 等后处理
3. **版本分表** — 避免特征缓存混用
4. **fast-check first** — 2分钟验证方向再全量训练
5. **数据对齐** — 训练和推理必须用完全相同的特征计算逻辑
6. **CPPI 是风控层** — 不依赖它提升信号质量

## 相关页面

- [模型世代总览](evolution.md)
- [NG 系列详解](ng-series.md)
- [已知陷阱](../lessons/known-pitfalls.md)
