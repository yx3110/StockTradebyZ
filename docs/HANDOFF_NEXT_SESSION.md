# 下 Session 入口 (Handoff Doc)

> 状态: 2026-04-28 EOS. 3 sessions 完成 P0/P1/P2 16 子任务 + 2 regression 修复.
> 下 session 直接从这里开始, 不需重读全部历史.

---

## 🎯 项目目标 (用户原话)

> "把风控系统糅合进模型里 — 这能做到么?做出可行性研究 + 提改进方案"

3-session 已交付:
1. 系统性评估 (选股 + 北极星框架)
2. 改进方案 P0/P1/P2 16 子任务
3. **可行性研究结论**: path B (auxiliary head + meta-learner) 可行, 但简单线性 utility 不提升 alpha;
   真 value-add 在 L1-L5 hard rules (生产已 active)

## 📁 关键产物索引 (依此查阅)

| 文件 | 内容 |
|---|---|
| `docs/superpowers/plans/2026-04-27-risk-control-roadmap.md` | 总 plan, P0/P1/P2 详细步骤 |
| `docs/wiki/architecture/risk-control-pipeline.md` | 4 层风控数据流 + score-scale 协议 |
| `docs/wiki/lessons/known-pitfalls.md` (#score-scale 量纲混淆) | 2 个 regression 教训 |
| `~/.claude/projects/-Users-yangxu-StockTradebyZ/memory/risk_control_roadmap_2026_04_28.md` | 完整 commit 表 + 实测 |
| `reports/forward_test/dashboard.md` | 90d 滚动 forward IC ALERT (Δ=-0.0307 触发) |
| `reports/diagnostics/{pre2020_factor_decay,booster_ab,soft_moe_smoke,hard_vs_soft_moe}.md` | 四个诊断报告 |
| `reports/capacity/ng106_capacity_curve.md` | 容量曲线 (1亿/3亿/10亿) |
| `reports/ng22/layer2_oos.csv` | Meta-learner grid search 输出 |

## 🚀 19 个 commits (按时间倒序)

```
d11671ec feat(P2.7c): soft-MOE production batch — dual scorer + EMA P_bull blend
62d8c271 docs(P2.9): wiki — score-scale bug 教训 + 风控管线文档
6a8ec051 fix(P2.8c regression): booster bonus 量纲 + 跳过 rank_score=0 picks
f87a7671 fix(P0.1 regression): apply_overlay_to_picks 自适应 score scale
eb9f569c feat(P2.8c): wire post-rank booster into selector behind --enable-booster
fbddd911 fix(P1.6c): maxdd_60d head 强制 min_iter=50 防早停退化为 1 棵树
42a77ca3 feat(P2.8b): booster A/B 验证 — trust filter +33 bps alpha 提升
6760bb2d feat(P2.7b): soft-MOE EMA 平滑 + 真数据 smoke
d032964e feat(P1.6b): ng2.2 Layer 2 meta-learner — utility-aware grid search
f0830865 feat(P1.3 Step A): risk-adjusted label generator (Calmar / Sortino)
3dc33bf0 feat(P1.5): 容量诊断 — ADV 5% cap × 1/3/10亿资金规模
925089bd feat(P2.8): post-rank booster — 8策略 regime-conditional + signal trust
3c316037 feat(P2.7): soft-MOE — compute_bull_proba + blend_scores
252eabdc feat(P1.6): ng2.2 Layer 1 — risk auxiliary heads (maxdd_60d / vol_10d)
8f5f2024 feat(P1.4): Pre-2020 因子风格衰减诊断脚本
37295fce feat(P0.2): forward OOS 90 日滚动 dashboard + daily wire
b337934a feat(P0.1): 生产化 L3 vol-target + L5 SL — ng1.0.6 默认启用 overlay
```

## ✅ 已 LIVE 的 4 层 production 风控管线

```
[ML 评分 7376 票]
   ↓
[post_filters]  ── trust 🔴 drop / 🟡 penalize / industry cap (默认 ON)
   ↓
[P2.8 booster]  ── strategy bonus (regime-conditional)  (--enable-booster opt-in, default OFF)
   ↓
[ng21 overlay]  ── L1 percentile floor + L2 industry cap → top_n 截断 (默认 ON)
   ↓
[P0.1 sizing]   ── L3 vol target + L5 stop-loss → position_size 字段 (默认 ON)
   ↓
[JSON: position_size + stop_loss + regime + crisis_active]
```

**用 `--scoring-version ng1.0.6` 默认启用 overlay+sizing**;
**加 `--enable-booster` 启用 P2.8 灰度**.

## ⚠️ 关键 GOTCHA — score-scale 量纲

NG 模型 `rank_score` ∈ [-0.05, +0.02] (预测收益), V3 时代 `composite` ∈ [0, 100].
任何"阈值 / 加权"代码必须 auto-detect, **绝不假设量纲**. 详见 wiki/lessons.

## 🚧 待办 (下 session 优先级排序)

### P0 (本周该做)

1. **P1.3 Step B** — trainer Calmar label 接入 + ng1.6.2 重训 (3-5h, 单 session 跑得完)
   - 入口: `ml_models/ng/risk_adjusted_labels.py` 已生成 csv (P1.3 A)
   - 改 `ml_models/ng/ng_trainer.py` 加 `--label-mode {industry_excess,calmar,sortino}` 选项
   - 训完用 `backtest/run_north_star_eval.py` 北极星对比 ng1.0.6
   - **接受准则** (写死, 不要事后改): WF-OOS V5.2 ≥ 70% AND MaxDD 比 ng1.0.6 (-21.4%) 改善 ≥ 3pp
   - **ABORT 线**: 第 1 个 WF 窗口 10d ICIR < 0.6 立即 kill

2. **forward IC 真闭环 alpha 验证** — 等几天 forward returns 累积后:
   - 跑 `forward_test_tracker scan` 拉新数据
   - 跑 `forward_test_dashboard` 看 ALERT 是否回稳
   - 跑 `booster_ab_compare` + `compare_hard_vs_soft_moe` 看 +33bp 在新数据是否复现

### P1 (本月该做)

3. **P2.7 真生产接入 selector** — 现 batch 已验证 6.3s/day, 把它搬进 selector:
   - 加 `+soft` scoring_version suffix 到 `stock_selctor/scoring_router.py`
   - 改 `tomorrow_stock_selector.py` ng106 分支同时载 bull+bear scorer
   - 注意接 post_filters / overlay / sizing 全栈 (batch 没接, 输出含 ST 票)
   - 内存预算 +60%

4. **P0.2 ALERT 自动化** — Forward IC Δ<-0.02 触发邮件:
   - dashboard.md 写出后 grep 是否含 "🚨 ALERT", `if grep -q triggered; then mail ...`
   - 接 daily_update.sh 末尾

5. **P1.6b Layer 2 重设计** — 简单线性 utility 不 work, 试:
   - non-linear interaction (跨 alpha 和 risk 头的乘法/分箱)
   - 或换成"position size" 决策而非 "ranking score" 决策 (更 native to risk control)
   - 或等 P1.3 Step B 完成后用 Calmar 标签的主模型 + risk heads 联合再试

### P2 (长期, 多 session)

6. **ng2.2 端到端联合训练** — 主 alpha 模型 + maxdd / vol heads 同 epoch 联合:
   - shared backbone + multi-head GBDT 不可行, 需要 NN 重构
   - Differentiable Sharpe loss (path C) — 6-10 周, ng3.x 长期实验

7. **真实容量回测** — P1.5 是诊断, 真生产 backtest engine 改造接 ADV cap 没做.

## 🔥 上一 session 最深刻的两个发现

1. **生产 pipeline 自 2026-04-09 起静默 0 picks** — P0.1 default-on 带来的 score_floor=30 vs NG 0.003 量纲冲突.
   两 session 后才被启用 booster 的诊断流程意外发现. 教训: production smoke 必须每次 production 改动都跑.

2. **简单线性 utility meta-learner 无 alpha 提升** — 风控糅合进模型最直观的做法
   (final = alpha - λ_dd × pred_dd - λ_vol × pred_vol) OOS 不 PASS gate. 真正 value-add
   在 hard rules (overlay/sizing), 不在 ranking-time penalty. 这印证 P0.1 推 overlay
   到生产是正确选择, 不要执着于 ML-内化路径.

## 🛠️ 立即可跑的 sanity checks

```bash
# 1. 生产 selector 还工作? (4-09 之后曾 0 picks)
python3 tomorrow_stock_selector.py 2026-04-24 --scoring-version ng1.0.6
# 期望: "全市场总股票: 10只", JSON 含 position_size 字段

# 2. Booster wiring 工作?
python3 tomorrow_stock_selector.py 2026-04-24 --scoring-version ng1.0.6 --enable-booster
# 期望: log 含 "[P2.8 booster] regime=bear, top-10 swapped X/10, avg bonus=Ypts"

# 3. Soft-MOE batch 工作?
python3 scripts/batch_generate_ng_soft.py --start-date 2026-04-24 --end-date 2026-04-24
# 期望: 5496 stocks/day, P_bull 输出, ~6s

# 4. 全测试 (3 fails 是 pre-existing, 不管)
python3 -m pytest stock_selctor/test/
# 期望: 112 passed, 3 failed (pre-existing BBI / select_stock)

# 5. Forward dashboard
python3 scripts/forward_test_dashboard.py --scoring-version ng1.0.6 --window-days 90
# 期望: 输出 reports/forward_test/dashboard.md, ALERT 是否仍触发
```

## 💬 给下 session 的开场提示词

复制这一段给下 session:

```
项目: StockTradebyZ. 上 session 完成了 3-session 风控糅合进模型路线图
(P0/P1/P2 16 子任务 + 2 regression 修复, 19 commits). 状态详见
docs/HANDOFF_NEXT_SESSION.md.

下一步优先级 (按 ROI):
1. P1.3 Step B: trainer 接入 Calmar label + ng1.6.2 重训 (3-5h, 此 session 完成)
2. forward IC 闭环 alpha 验证 (等数据)
3. P2.7 真生产接入 selector (架构改动)

请先读 docs/HANDOFF_NEXT_SESSION.md, 跑一下 sanity checks 确认 pipeline 没回归,
然后告诉我你想先做哪个. 我倾向先做 P1.3 Step B — Calmar label 验证是否能改善
ng1.0.6 v1 的 MaxDD (-21.4%) 短板.
```
