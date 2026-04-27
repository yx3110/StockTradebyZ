# 系统改进路线图 (2026-04-27)

基于选股系统 + 北极星评估系统的全面诊断，按 ROI/风险排序的改进措施清单。

## 当前生产基线

- **生产模型**: ng1.0.6 v1 (MOE: V11 regime → ng1.0.1 bull / ng1.0.4 bear)
- **回滚来源**: 4-25 真零泄漏 OOS Pre-2020 (N=300) 验证 v1 > v2 78%
- **灰度并存**: ng2.0a (multi-beta vote regime), ng2.1 (bull/bear specialist + L1-L5)
- **评估系统**: 北极星 V5.2 (59 指标, 9 层, 295 满分)

## 核心诊断（待改进的根因）

1. **In-sample inflation 3-8x**: V5.2 跑训练区间 = in-sample memorization, 跨模型排名失真
2. **V5.2 ≠ alpha**: 22% L1 信号权重, 主要是 risk-adjusted, 误用作 alpha 排名
3. **Sparse-trading 年化 bug**: `n_dates` 算年化, cash 50%+ 天数让 Sharpe 虚高到 7+
4. **选股层 overlay 缺失**: ng2.1 实证 L1-L5 是真杠杆 (5pp MaxDD), 生产没用
5. **模型迭代 ROI 断崖**: 67 特征底座 alpha 饱和, ng1.4-2.1 全部未切生产
6. **Forward test 闭环缺失**: 切版本只能靠 in-sample backtest

---

## P0 — 真 OOS 闭环（解决评估真实性）

### P0.1 Forward test 自动化框架 ⭐ 第一优先级

**问题**: 每次切版本只能靠回测决策, 但回测被 in-sample inflation 污染. 4-25 v1→v2→v1 来回切已经证明评估不可靠.

**方案**:
- 每日生产报告留底（已经在做, `reports/daily_selection_*/`）
- 新增 `scripts/forward_test_tracker.py`：
  - 扫描 N 日前的报告 → 用真实日线收益打分
  - 维护累积 forward sample 库 (`reports/forward_test/forward_samples.parquet`)
  - 每周输出 forward IC vs in-sample IC 对照表
  - 累积 N=20/40/60 后输出 paper-trade vs backtest 显著性检验
- Gate 写死: 任何模型生产切换前, paper trade ≥ 20 个交易日, forward IC ≥ in-sample IC × 0.6

**成本**: ~3-5 天建框架, 之后自动跑

**交付物**:
- `scripts/forward_test_tracker.py`
- `reports/forward_test/{forward_samples.parquet, weekly_report.md}`
- CLAUDE.md 新增"生产切换 gate"小节

### P0.2 Annualization bug 修复

**问题**: `backtest_report_based.run_single_backtest` 用 `n_dates` (报告字典 size) 算年化. cash filter 跳过的天从分母消失, 等价"模型每年只工作 X% 时间", 单位时间收益放大 3-5 倍. V15 实例: cash 65% 天数 → Sharpe 7.285 假象.

**方案**:
- 改用 calendar days: `(1+cumret)^(252/total_calendar_days) - 1`
- 保留 `n_dates` 计算结果作为 `annual_return_active`, 标注"仅 active days"
- Sharpe > 4 时自动 print warning + 输出 cash 比例

**成本**: ~半天

**交付物**:
- `backtest/backtest_report_based.py` 修改
- 单测: 同一份报告字典, cash 50%/100% 两种 case, calendar 年化应一致

---

## P1 — 选股层 overlay 入生产（解决 alpha 饱和）

### P1.1 L1-L5 overlay 移植到 ng1.0.6 生产路径 ⭐

**问题**: ng2.1 实证 L1-L5 风控让 MaxDD -23.7% → -18.4% (改善 5.3pp), 超额年化 80% → 188% (2.3×). 但生产仍是 ng1.0.6 raw output.

**方案**:
- `stock_selctor/ng21_risk_overlay.py` 已存在, 改成通用模块 `stock_selctor/post_score_overlay.py`
- 为 ng1.0.6 增加 `--with-overlay` flag, 默认 off (保护现状), 提供 `--scoring-version ng1.0.6+overlay` 灰度
- 五个规则:
  - L1: score floor 30
  - L2: industry_cap (bull=3 / bear=2, regime-aware)
  - L3: VT 波动率目标 20%
  - L4: crisis hard-stop (V11 bear + 大盘 5d ≤ -5% → 减仓)
  - L5: SL 止损 6%
- Stage 4 验证: WF-OOS 2024-2026 + Pre-2020 + 真 forward 20 日

**Gate**: MaxDD 改善 ≥ 3pp 且 V5.2 退步 ≤ 2pp

**成本**: ~3-5 天

**交付物**:
- `stock_selctor/post_score_overlay.py` (从 ng21 重构)
- `tomorrow_stock_selector.py` 新增 `+overlay` 后缀路由
- `reports/overlay_evaluation_2026_04_27.md`

### P1.2 行业分散硬约束独立化

**问题**: ng1.0.4 时代 RF 权重失衡导致 Top-10 全银行. 当前 ng1.0.6 v1 没有 industry_cap, bear regime 风险敞口集中.

**方案**: P1.1 的子集, 但作为独立可单独启用的过滤器, 验证单一规则贡献.

**成本**: 包含在 P1.1 内

### P1.3 Signal Trust 从"标签"升级到"过滤"

**问题**: 当前 trust_tag 只贴在 JSON 供人参考, 🔴 假信号股票仍能进 Top-10.

**方案**:
- 选股阶段: 🔴 trust 直接剔除 (硬过滤)
- 🟡 trust: score × 0.7 (软扣分)
- 灰度对比: 启用前后 30 日 forward IC 对比

**Gate**: 启用后 forward 30 日 IC 不退步, MaxDD 改善

**成本**: ~2-3 天

**交付物**:
- `tomorrow_stock_selector.py` 接入 `signal_trust` 模块的过滤接口
- `reports/signal_trust_filter_evaluation.md`

---

## P2 — 北极星精简（解决评估信噪比）

### P2.1 V_ALPHA 评分卡（独立于 V5.2）

**问题**: V5.2 是 risk-adjusted, 22% L1 信号权重不够代表 alpha 强度. 历史多次出现"V5.2 高的模型 raw alpha 反而弱"(ng1.0.62 vs ng1.0.1).

**方案**:
- 不替换 V5.2 (保留作"综合健康度")
- 新增 `north_star_metrics.py:compute_v_alpha` (8 指标):
  - daily_ic, icir, ic_positive_pct (L1 信号 3 个)
  - top10_excess_return, information_ratio, excess_win_rate (L5 超额 3 个)
  - regime_ic_consistency, oos_ic_half_life (鲁棒性 2 个)
- CLI: `run_north_star_eval.py --v-alpha` 单独输出
- 决策时双卡并存, V_ALPHA 看 alpha 排名, V5.2 看 risk profile

**注意**: 4-25 V_ALPHA 提案曾被 reviewer 判 sunk cost trap, 当时论据是"用 in-sample alpha 校验 in-sample V5.2 = 循环论证". 本次方案接受这个批评 — V_ALPHA 必须搭配 P0.1 forward test 才有意义, 单独跑 batch_generate 仍是 in-sample.

**成本**: ~3 天 (代码已有原料, 主要是组合 + CLI)

**交付物**:
- `backtest/north_star_metrics.py:compute_v_alpha`
- `backtest/run_north_star_eval.py --v-alpha` flag

### P2.2 In-sample / OOS 标签强制化

**问题**: 跨模型 V5.2 对比表常无视数据来源差异. ng1.0.1 (训于 4-13, 评估 2024-2026) vs ng1.7 (训于 4-24) 都是 in-sample 但 inflation 程度不同.

**方案**:
- `run_north_star_eval.py` 输出强制带 banner:
  - `[IN-SAMPLE]` 评估区间与训练区间重叠
  - `[WF-OOS]` Walk-Forward 测试段
  - `[PRE-2020]` 训练区前
  - `[FORWARD]` 训练完成后真 forward
- Markdown 报告头部强制写明
- 跨模型对比表自动加 inflation caveat (粗估 ÷3)

**成本**: ~1-2 天

**交付物**:
- `backtest/run_north_star_eval.py` 修改
- 模板: `reports/templates/north_star_report.md`

---

## P3 — 架构债清理（解决迭代速度）

### P3.1 tomorrow_stock_selector.py 拆分

**问题**: 6381 行单体, 每加 scoring_version (ng2.0a/ng2.1) 都在加分支, 模块边界模糊. 后续 P1/P2 修改风险大.

**方案**:
- 不重写逻辑, 只切边界:
  - `stock_selctor/scoring_router.py` (scoring_version → scorer 选择)
  - `stock_selctor/quant_filter.py` (8 策略量化过滤)
  - `stock_selctor/post_score_overlay.py` (P1.1 已建)
  - `stock_selctor/report_writer.py` (JSON/MD 输出)
  - `tomorrow_stock_selector.py` 保留作 orchestrator (~1000 行)
- 拆分前后行为不变, 加 snapshot test 保证

**成本**: ~1-2 周 (必须 P1 完成后做, 不然冲突)

**交付物**:
- 4 个新模块
- `tests/test_selector_snapshot.py`

### P3.2 reports/ 清理 + naming convention

**问题**: 50+ 个 daily_selection_*_stage* 目录 (已 REJECTED 的 ng1.2/1.3/1.4/1.5). git status 噪声大, 占盘.

**方案**:
- 移到 `reports/archive/{version}/`
- 建立命名规范: `daily_selection_{ver}_{purpose}` (purpose ∈ {fast, fullmarket, wf_oos, pre2020, stage35, stage4a, forward})
- README 记录哪些 version 已 deprecated

**成本**: ~半天

**交付物**:
- `reports/archive/` + README
- 移动 / 删除脚本

---

## P4 — 新数据源（突破 alpha 天花板）

### P4.1 Alt-data 4 因子接入 ng1.0.6 v1 的 ng1.0.1 bull 子模型

**问题**: 龙虎榜/融资融券 4 因子 (F_margin_1/2 + F_top_1/2) 已抓到 5.47M 行, IC=-0.10 ICIR=-0.72 (散户追高反向). 但只在 ng1.7 用过, ng1.7 因 risk profile 不切生产.

**方案**:
- 把 4 因子 backfill 到 `ng101_feature_cache.features_json` (新增 4 列)
- ng1.0.1 schema 升级到 70 特征 (66+4)
- 重训 ng1.0.1' (`ng1.0.1+alt`)
- 灰度对比: ng1.0.6 v1 (ng1.0.1 bull) vs ng1.0.6 v1' (ng1.0.1+alt bull)
- 通过 P0.1 forward test 验证

**Gate**: forward 20 日 IC ≥ baseline + 0.05

**成本**: ~3-5 天

**交付物**:
- `ml_models/ng/ng_schema.py` 新增 alt-factor 字段
- `ml_models/trained_models/ng/ng101_alt_*.pkl`

### P4.2 行业 momentum / 板块轮动因子

**问题**: 当前 ng_schema 10 market features 都是大盘级 (沪深300/0AMV). 缺行业相对动量、板块拥挤度、概念热度.

**方案**: (本期不展开, 列入 P4 后续)

**成本**: ~1 周

---

## 不做的事（明确避免）

- ❌ 再训 ng2.x specialist (alpha saturated, ng2.0b/2.1 已证)
- ❌ 再加 V5.3/V5.4 评分指标 (信号稀释, 295 分卡已过载)
- ❌ portfolio-level CPPI/VT/SL (用户明说不需要, 价值点是选股列表非组合)
- ❌ regime classifier 再调 (V11 三窗口最优, panic-drop 已证伪)

---

## 执行顺序

按"确定性收益 → 中等不确定性 → 高不确定性"排序:

1. **P0.2** Annualization bug (~半天, 确定性收益)
2. **P0.1** Forward test 框架 (~3-5 天, 后续所有 gate 的基础)
3. **P1.1** L1-L5 overlay 入生产 (~3-5 天, 已有 ng2.1 实证)
4. **P2.2** In-sample/OOS 标签强制化 (~1-2 天, 防御性改进)
5. **P1.3** Signal Trust 升级到过滤 (~2-3 天)
6. **P2.1** V_ALPHA 评分卡 (~3 天, 依赖 P0.1)
7. **P3.2** reports/ 清理 (~半天)
8. **P4.1** Alt-data 4 因子重训 ng1.0.1' (~3-5 天, 依赖 P0.1 验证)
9. **P3.1** tomorrow_stock_selector.py 拆分 (~1-2 周, 最后做避免冲突)

每完成一项, 输出汇报 + git commit, 用户 review 后再进下一项.

---

## 进度追踪

| # | 项目 | 状态 | Commit | 备注 |
|---|---|---|---|---|
| P0.2 | Annualization bug | ✅ 完成 | 79e23650 | calendar-time 校正 + cash_ratio + Sharpe>4 warning, 5 单测 PASS |
| P0.1 | Forward test 框架 | ⏳ 待办 | — | — |
| P1.1 | L1-L5 overlay 入生产 | ⏳ 待办 | — | — |
| P2.2 | In-sample/OOS 标签 | ⏳ 待办 | — | — |
| P1.3 | Signal Trust 过滤 | ⏳ 待办 | — | — |
| P2.1 | V_ALPHA 评分卡 | ⏳ 待办 | — | — |
| P3.2 | reports/ 清理 | ⏳ 待办 | — | — |
| P4.1 | Alt-data 重训 ng1.0.1' | ⏳ 待办 | — | — |
| P3.1 | selector 拆分 | ⏳ 待办 | — | — |
