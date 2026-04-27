# P1.1 ng1.0.6+overlay 评估 (2026-04-27)

## 改动

`tomorrow_stock_selector.py` 新增 `ng1.0.6+overlay` / `ng1.0.62+overlay` 两个 scoring_version,
复用 ng2.1 的 L1-L5 风控 (`stock_selctor/ng21_risk_overlay.py`):
- L1: score floor 30
- L2: industry_cap (bull=3 / bear=2)
- L3: VT 波动率目标
- L4: crisis hard-stop (B2_RV_pct ≥ 90% + hs300 ≤ -3% → top_n×0.5)
- L5: stop_loss 6% / trailing -6%

实现路径: `ng106_overlay_mode` flag → 设置 `_ng21_mode=True` → 复用现有 overlay apply 逻辑.

## 实证: 2024-2026 (in-sample, focus_days=10, top_n=10)

| 指标 | ng1.0.6 RAW | ng1.0.6+overlay | Δ |
|---|---|---|---|
| 净年化 | 100.9% | 78.7% | **-22.2pp** ❌ |
| Sharpe | 2.791 | 2.345 | -0.45 ❌ |
| MaxDD | -21.3% | -23.0% | -1.7pp ❌ |
| 超额年化 | 167.7% | 133.5% | -34.2pp ❌ |
| 超额 MaxDD | -18.4% | -19.9% | -1.5pp ❌ |

**结论: overlay 单独应用到 ng1.0.6 上, in-sample 段全维度退步.**

## 与 ng2.1 实证的差异

ng2.1 实证 overlay 改善 MaxDD 5.3pp / 超额年化 2.3×, 但那是 **specialist+overlay 联合**:
- ng2.1-bull / ng2.1-bear 是用 V11 regime-filtered 数据训出的子专家
- overlay 在 specialist 输出上叠加才有改善

ng1.0.6 (= ng1.0.1 bull + ng1.0.4 bear) 不是 regime-filtered 训练 (子模型用全期), 所以:
- bull 子模型 ng1.0.1 在 bull regime 已经 alpha 充足, overlay 的 industry_cap=3 反而剔除了集中持仓的赢家
- bear 子模型 ng1.0.4 受 RF 权重失衡影响倾向银行集中, industry_cap=2 强制分散到次优 → score floor 30 进一步过滤
- L4 crisis 触发太少 (5 个 crisis 天数 / 1606 总), 不够带来防御红利

## 决策

- **不切生产**: ng1.0.6+overlay 留作灰度选项, 不替代 ng1.0.6 v1 默认
- **CLI 已支持**: `python3 tomorrow_stock_selector.py YYYY-MM-DD --scoring-version ng1.0.6+overlay`
- **未来验证**: 需通过 P0.1 forward test 框架累积 ≥ 20 个交易日真 forward 数据再决策
- **教训**: overlay 不是 universal improver, 必须配合训练侧的 regime-aware 才能放大价值. 与 ng2.0b sample-weighted 失败 (alpha 提升但 MaxDD -14pp) + ng2.1 specialist 持平 V5.2 的教训一致 — **alpha 在 67 特征上已饱和, 价值杠杆要么从风控配套训练, 要么从新数据源**.

## Pre-2020 OOS 测试

未完成: 单次 eval 跑 1606 报告 ~3-5min, 完整对比要 20+min. P0.1 框架做这事更高效 (CSV 内已有 forward 收益).

后续: 用 forward_test_tracker 直接对比 raw ng1.0.6 vs overlay-tagged samples.

## 文件

- `tomorrow_stock_selector.py`: 新增 ng106_overlay_mode 路由 + +overlay 后缀解析
- `reports/daily_selection_ng106_overlay/` (本地): 1606 个 overlay-tagged 报告 (post-hoc applied)
- `docs/superpowers/plans/2026-04-27-system-improvement-roadmap.md`: 进度追踪
