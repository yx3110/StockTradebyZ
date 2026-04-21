# ng 下一迭代路线图 — 2026-04-21 (post Tier A)

**前提**: Tier A 特征路线 (ng1.3.x / ng1.4.x / ng1.5.0) 全部失败. 所有"加新特征"尝试都被 ng1.0.1 (76% Stage 4a) 击败.

**教训**: 特征维度已 saturated. 加特征 = 噪音 (跨 regime 不泛化).

**新方向**: 不动特征集 (沿用 ng1.0.1 或 ng1.0.6 的 66 特征), 改训练/标签/后处理.

**生产基线**: ng1.0.1 (76% Stage 4a, 73.4% WF-OOS) 或切换到 ng1.0.6 (78.9% WF-OOS, +AMV regime switch, 需 ng1.0.5 风控压制 MaxDD).

---

## 候选方向 (按实施成本 × 预期收益排序)

### F1: LambdaRank 主力 (中成本, 中收益)

**动机**: 6-algo MSE ensemble 打数值准度战. Top-K 选股只需排序准. LambdaRank 直接优化排序.

**证据**: ng1.0.1 的 lgb_rank 在 pkl 里占 8% 权重, 独立 ICIR 只有 0.2 (噪音). 但这是训练目标错配: MSE loss + LambdaRank predict. 如果用 LambdaRank 作为 PRIMARY loss (不是 auxiliary), 可能强.

**执行**:
1. 新版本 ng1.6.0: 同 ng1.0.1 特征 (66), 训练 loss 改为 LambdaRank (truncation=10, relevance grades=10)
2. 保留 MSE 作为 auxiliary (50/50 blend or gate)
3. Fast-check → 3-seed train (~3h) → Stage 3.5 + 4a

**成本**: ~4h
**预期**: +2-4pp V5.2 Stage 4a (如果排序损失确实比 MSE 好)
**风险**: LambdaRank 易过拟合 top-K, 可能牺牲中段排序

---

### F2: 风格中性化标签 (中成本, 高收益)

**动机**: 当前 labels 是 "stock return - industry_median". 仍含 size/value/momentum factor 暴露. 模型学到的 alpha 部分是 factor beta, 跨 regime 不稳.

**证据**: ng1.0.6 β_UMD=+0.005 (几乎零) 跨 regime 稳定 (Pre-2020 唯一正年化). ng1.0.1 β_UMD=+0.38 (5.4 σ) 在 2024 H1 动量反转时受伤.

**设计**:
1. 计算每日 SMB/HML/UMD factor returns (用 existing `factor_returns.py`)
2. 每个 label_Nd 对 factors 回归, 取 residual 作为"纯 alpha label"
3. 训练目标改为该 residual (同特征集, 同架构)

**执行**:
1. 新版本 ng1.6.1: `factor_neutralized_label_Nd = label_Nd - sum(β_k × factor_k_Nd)`
2. 回填 ng106_feature_cache 或创建 ng161 子表 (加 factor_neutralized 列)
3. 训练 3 seeds (~3h)
4. Stage 3.5 + 4a

**成本**: ~5-6h (labels 重新计算 + 回填)
**预期**: +3-6pp V5.2 跨 regime (去除 factor drag)
**风险**: factor residuals 信号更弱 (SNR 更低), 可能短期表现下降

---

### F3: 后处理校准 (低成本, 低-中收益)

**动机**: 预测分数分布可能与实际 return 分布错位. 保序校准 (Isotonic Regression) 可修正.

**证据**: V5.2 L5 超额得分 (ng1.0.1=~89%) 高但绝对 Alpha 不总是高. 排序好但分数刻度可能错.

**设计**:
1. 在 WF test 段拟合 isotonic: X=model_score, y=actual_return
2. 推断时把 raw score 通过 isotonic 映射
3. 只动分数标度, 不动排序

**执行**:
1. 写 `ml_models/ng/calibrators.py` — 读 WF history, 拟合 isotonic
2. NGProductionScorer 加 `--calibrate` 路径
3. Stage 3.5 + 4a 对比

**成本**: ~3h
**预期**: +0-2pp V5.2 (不一定, 排序不变则 IC 不变, 但阈值/composite 会变)
**风险**: 低 (不破坏现有模型)

---

### F4: 多 horizon 学习权重 (低-中成本)

**动机**: 当前 target_weights 是 ICIR 自适应, 但是 **训练窗口内** 优化的. 可能用 OOS performance 直接优化.

**设计**: 训练完整模型后, 用 last-6month OOS 窗口拟合 Stacking LGB: features = [pred_3d, pred_5d, pred_10d, pred_15d], target = label_10d. 输出: 学习出的 composite weight.

**执行**:
1. 在 WF 最后几个窗口做 stacking
2. 存权重到 pkl
3. 推断时用 stacked weights 代替 ICIR weights

**成本**: ~3h
**预期**: +1-3pp

---

### F5: Sample weighting 优化 (低成本)

**动机**: 现在 sample weights 是 regime-weighted + decay, 但粗糙.

**可尝试**:
- Drawdown-aware weighting: 高 drawdown 日 (下跌行情) 样本降权 (避免学太多"避险模式")
- Top-K focused: 每日 top-20% 和 bottom-20% 样本加权 (V4.9 思路, 已有代码)

**成本**: ~2h
**预期**: +1-2pp

---

## 推荐执行顺序

```
立即: D 收尾 (commit + memory) ✅ 已做

下一步: F2 (风格中性化标签, 高收益)
├── 成功 (V5.2 Stage 4a >= 78%) → 上生产 ng1.6.1
├── 持平 (74-78%) → 和 ng1.0.6 并列, 选 MaxDD 好的
└── 失败 (<74%) → 试 F1 (LambdaRank)

F1 / F3 / F4 / F5 视 F2 结果而定.
```

**为什么优先 F2 (风格中性化)**: ng1.0.6 的零 β_UMD 是 跨 regime 赢家的最大特征. 复刻这个特征到 ng1.0.1 架构, 理论上有空间叠加 ng1.0.1 的强 signal.

---

## 全流程成本估算

| Phase | 实施 | 训练 | 评估 | 小计 |
|:---|:---:|:---:|:---:|:---:|
| F2 labels + 回填 | 2h | 3h | 1h | ~6h |
| F1 LambdaRank | 1h | 3h | 1h | ~5h |
| F3 校准 | 1h | - | 1h | ~2h |
| F4 stacking | 1h | - | 1h | ~2h |
| F5 weighting | 1h | 3h | 1h | ~5h |
| **总** | | | | **~20h** |

可分几个 session 做. F2 优先.

---

## 接受准则 (不再给 wiggle room)

| Gate | 阈值 | 对比 |
|:---|:---:|:---|
| Stage 3.5 (2025) | ≥ 76% | ng1.0.1 gold standard |
| Stage 4a (2024-2026) | ≥ 76% | ng1.0.1 gold standard |
| Pre-2020 | ≥ 50% | ng1.0.1 45.5% (低门槛, 但别比 baseline 差) |
| β_UMD | < 1.0 | ng1.0.1 +0.38, ng1.0.6 +0.005 |
| MaxDD | < 15% | ng1.0.1 -11.7%, ng1.0.6 -22% |
| Sharpe | ≥ 2.5 | ng1.0.1 2.37, ng1.0.6 2.81 |

**任一关键 gate (Stage 4a) fail → ABORT, 不再挽救.**

---

## Pre-flight Checklist (每个 F 实验必做)

遵循 CLAUDE.md 的 10 项:
1. Schema 一致性
2. Backfill 逻辑
3. 高效执行路径 (fast-check + target-parallel)
4. Acceptance criteria + ABORT gate
5. Baseline 公平对比 (对齐 ng1.0.1 配置)
6. Checkpointing + 落盘日志
7. 数据泄露扫描 (新 labels 必看)
8. 资源预算
9. 可重现性元数据
10. /simplify 3 轮

---

**End of plan.**
