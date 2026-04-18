# ng1.3.0 下一代模型设计 — Multi-task 双头 + β Composite

**日期**: 2026-04-18
**版本**: ng1.3.0 (架构级变更, 延续 ng1.x)
**基线**: ng1.0.1 bugfix 重训版 (V5.2=72.1% A+ / 年化 165.7% / MaxDD=-11.7% / Sharpe=2.753)

---

## 1. 目标 (Goal)

相对 ng1.0.1 bugfix 基线：

| 指标 | ng1.0.1 基线 | ng1.3.0 目标 |
|---|---:|---:|
| MaxDD | -11.7% | **<-8%** (压 4pp) |
| 年化(毛) | 165.7% | **≥130%** (容忍 -20% 下滑) |
| Sharpe | 2.753 | **≥3.5** (提升 +0.75) |
| V5.2 (WF-OOS) | 72.1% A+ | **≥72%** 保住水平 |
| V5.2 (Pre-2020) | 73.7% A+ | **≥65%** A (放宽 8pp 接受 downside head 拖累) |

---

## 2. 动机与短板 (Motivation)

### 2.1 ng1.0.1 短板 (数据支持)

| 短板 | 证据 | 影响 |
|---|---|---|
| 市场依赖高 | 14 核心因子中 8 个是 market 组 (38% 权重) | 市场反转时信号失效 |
| 个股 alpha 弱 | 除 roe_ttm / cs_rank_turnover / turnover_rate 外少有个股信号 | 行业中性后排名粗糙 |
| 熊市反转 | worst 60d ICIR = -0.249 | 熊市可能产生反向选股 |
| 换手高 | 年化 45x，净收益被交易成本侵蚀 | Sharpe 天花板受限 |
| 基本面空白 | moneyflow / accrual / quality 因子历史多次尝试失败 | 无法区分"高质量"和"炒作" |

### 2.2 ng1.0.6 短板

| 短板 | 证据 |
|---|---|
| 模型切换架构复杂 | 牛熊切换 18 次，缓跌 10 天阈值固定 |
| MaxDD 仍-20.2% | 单纯切换没有内生风控 |
| Pre-2020 未验证 | 风险未知 |
| 底模依赖 | ng1.0.1 和 ng1.0.4 天花板固定，切换只能折中 |

### 2.3 过去失败铁律 (必须避开)

1. **只改 loss/label 不改特征** (ng1.2.0/1/2 全败)
2. **Contrarian 因子叠加** (ng1.2.3 WF-OOS=45%/Pre-2020=31% 双崩)
3. **2 个 mf 因子太边际** (ng1.2.4 Stage 3.5 V5.2=48.5%)
4. **22 特征太薄** (ng1.1.1 Phase2 0/14 ACCEPT)
5. **Fast-check ICIR 虚高骗人** (ng1.0.9 Sharpe 崩塌至 0.79)
6. **条件/RA 标签不泛化** (ng1.0.4/0.7 Pre-2020 C 级)

---

## 3. 设计决策 (Design Decisions)

### 3.1 核心决策链 (已与用户对齐)

| 决策点 | 选择 | 拒绝的替代 | 理由 |
|---|---|---|---|
| 目标基准 | vs ng1.0.1 裸模型 | vs ng1.0.6 / vs ng1.0.5 overlay | ng1.0.1 是唯一跨 regime A+ 底座 |
| Label 方案 | L3 Multi-task 双头 | L1 软惩罚 / L2 非对称权重 | L1 历史两次崩 Pre-2020; L2 与 MaxDD 目标错配 |
| 特征范围 | Tier A + B + C (78 特征) | 仅 Tier A / 仅 AB | amount 已 100% 回填, Tier C 边际成本低 |
| Seed 策略 | 3-seed ensemble | 1-seed / 5-seed | ng1.0.2 实测 Sharpe +97% (1.95→3.72); 5-seed 边际<10% 且 RF 风险 |
| β 优化 | WF 网格 {0.1-0.5}, 最大化 Sharpe | V5.2 / 固定 0.3 | Sharpe 与 V5.2 相关性>0.9 但计算快 10x |
| 版本号 | ng1.3.0 | ng2.0.0 | 延续 ng1.x 系列, 1.3.x 代表 multi-task 架构分支 |

### 3.2 不做的事

- **不做** hard regime switching (ng1.0.6 式) — AMV 信号内嵌为连续特征
- **不做** CPPI/止损/vol target 等 overlay — 目标是**裸模型**达标
- **不做** fast-check 作 go/no-go 决策 — 只用 full WF + Stage 3.5 spot check
- **不做** 因子/label 联合 contrarian 叠加 (ng1.2.3 前车)

---

## 4. 架构 (Architecture)

### 4.1 全景图

```
数据层
  ├─ daily_quotes.amount           ✅ 已 100% 回填 (10.1M 行)
  ├─ moneyflow_daily (code 修复)   Tier B 先做, ~2h
  ├─ financial_indicator           已有 roe/pe/ocf_to_profit
  └─ ng130_feature_cache           ★ 新表, 不改 ng101

特征层 (78 个)
  ├─ ng1.0.1 base                  66 个, bugfix 重训版
  ├─ Tier A (downside + AMV)       7 新 (零数据依赖)
  ├─ Tier B (moneyflow)            3 新
  └─ Tier C (amount-based)         候选 4 砍 2 (EMT 四关)

训练层 (Multi-task 双头)
  for seed in {42, 123, 456}:
    Head A (excess):  6-algo ensemble × 4 horizons → pred_excess_Nd
    Head B (downside): 6-algo ensemble × 4 horizons → pred_downside_Nd

预测聚合层 (3-seed average)
  pred_excess_avg_Nd   = mean([seed_42, seed_123, seed_456])
  pred_downside_avg_Nd = mean([seed_42, seed_123, seed_456])

打分层 (β-composite)
  Z_ex = rank_pct(pred_excess_avg_10d)    (截面排名 0-1)
  Z_dn = rank_pct(pred_downside_avg_10d)  (截面排名 0-1, 低=跌幅大)
  score = Z_ex - β · Z_dn
  β* = argmax WF-OOS Sharpe over {0.1, 0.2, 0.3, 0.4, 0.5}
```

### 4.2 与 ng1.0.1 / ng1.0.6 的关系

- **特征层**：ng1.0.1 的 66 特征 100% 保留 + 加 12 新特征
- **标签层**：ng1.0.1 的 industry excess label 保留 + 新加 downside label
- **市场信号**：ng1.0.6 的 AMV 作为**连续特征**内嵌 (amv_var1/macd/regime_days)，**不做硬切换**
- **架构层**：**新**——multi-task 双头是全新结构，ng1.0.1/1.0.6 都没有

---

## 5. 特征集详表 (78 个)

### 5.1 ng1.0.1 base 66 个

按 2026-04-12 EMT 独立审计：22 个全 horizon STRONG，0 个 WEAK。核心 14 个（按稳定性排）：

vix_proxy, cs_rank_turnover, market_volatility_20d, industry_return_20d, northbound_flow_5d, roe_ttm, market_return_20d, market_drawdown, turnover_rate, idiosyncratic_volatility, market_volume_ratio, market_new_high_ratio, relative_strength_vs_peers, market_return_5d

### 5.2 Tier A 新增 7 个 (零数据依赖)

| 因子 | 来源 | 定义 | 预期角色 |
|---|---|---|---|
| `current_drawdown` | ng1.0.4 | 当前收盘距 60d 高点的距离 | downside head 核心 |
| `downside_vol_20d` | ng1.0.4 | 下行波动率 (仅计算负收益日) | downside head 核心 |
| `recovery_speed_20d` | ng1.0.4 | 回撤触底后的反弹速率 | excess head 辅助 |
| `gap_risk_20d` | ng1.0.4 | 20d 内跳空幅度累计 | downside head 辅助 |
| `amv_var1` | ng1.0.6 | 0AMV 活跃筹码指数连续值 | 两个 head 共用 (regime proxy) |
| `amv_macd` | ng1.0.6 | 0AMV MACD 强度 | 两个 head 共用 |
| `amv_regime_days` | ng1.0.6 | 当前 regime 持续天数 | 两个 head 共用 |

### 5.3 Tier B 新增 3 个 (修 moneyflow code 后)

| 因子 | 公式 | 动机 |
|---|---|---|
| `elg_net_inflow_20d_z` | Z-score of ∑(buy_elg - sell_elg) 20d | 特大单净流入 = 机构资金 |
| `mf_main_ratio_20d` | ∑(net_lg + net_elg) / ∑(amount) 20d | 主力资金参与度 |
| `mf_concentration_20d` | std(daily_net_mf) / mean(\|daily_net_mf\|) | 资金流稳定性 |

### 5.4 Tier C 候选 4 个 → 留 2 (EMT 四关)

| 因子 | 公式 | Gate (10d ICIR) |
|---|---|---|
| `amihud_illiq_20d` | mean(\|ret\| / amount × 1e8) | ≥ 0.15 |
| `vwap_close_ratio_20d` | mean((close - vwap) / vwap) | ≥ 0.15 |
| `amount_acceleration_5d` | amount_ma5 / amount_ma20 - 1 | ≥ 0.10 |
| `tail_beta_60d` | β on bottom-30% market days | ≥ 0.10 |

**EMT 四关规则**：
1. ICIR ≥ 阈值
2. 与现有 78 特征 |corr| < 0.7
3. IC 方向稳定 (6 regime 一致)
4. Gain-importance > 1% (训练后验证)

预期留 1-2 个 (Amihud 最有希望, 学术验证充分)。

### 5.5 最终特征总数

- 上限: 66 + 7 + 3 + 4 = **80**
- 预期: 66 + 7 + 3 + 2 = **78** (Tier C 砍 2)
- 下限: 66 + 7 + 3 + 0 = **76** (Tier C 全砍)

---

## 6. 标签设计 (Labels)

### 6.1 已有 (沿用 ng101 cache)

- `label_3d, label_5d, label_10d, label_15d` = industry excess return
- `label_raw_3d, ..., label_raw_15d` = 绝对收益 (ablation 用)

### 6.2 新增 downside label (ng130 cache 新列)

```python
# For each trade_date t and horizon N:
closes = get_close(code, t+1, t+N)  # N 个交易日后的收盘
downside_Nd = min(closes) / close[t] - 1  # 负数, 越小越差
```

- 4 个 horizon: `downside_3d, downside_5d, downside_10d, downside_15d`
- 用 **min-cumret** 而不是 intra-window maxdd (计算简单、噪声小、与持仓收益直接挂钩)

### 6.3 为什么不污染 excess label

- L3 方案的**核心优势**: excess label 完全不变，downside 独立学
- 即使 downside head 在 Pre-2020 弱，β 调节也能让 composite 退化到接近 ng1.0.1
- β=0 时 score ≡ ng1.0.1 基线

---

## 7. 训练 Pipeline

### 7.1 双头训练 (每 seed)

```python
for seed in [42, 123, 456]:
    for head in ['excess', 'downside']:
        for horizon in [3, 5, 10, 15]:
            label_col = f'label_{horizon}d' if head=='excess' else f'downside_{horizon}d'

            # 6 个 booster, 每个用同 seed 训练
            for algo in [lgb, xgb, cb, rf, hgb, lgb_rank]:
                model = train(X_78, y=label_col, algo=algo, seed=seed)

            # ICIR 加权 ensemble (per WF window)
            weights = icir_weighted(oos_preds, floor=5%)
            # downside head 额外 cap RF 权重 ≤ 40% (防 ng1.0.4 式失衡)
```

**产物**: 3 seeds × 2 heads × 4 horizons × 6 algos = **144 个 booster**
**存储**: `ml_models/trained_models/ng/ng130_seed{42,123,456}_{head}_{horizon}d.pkl` (打包按 seed)

### 7.2 Auto-WF 模式选择

训练前 turbo-check 3 种 WF 配置 (~6min)，按 **Head A (excess) 10d ICIR** 选最优：

- `expanding` (ng1.0.1 原配置)
- `sliding-720d`
- `sliding-500d + time-decay halflife=730`

**理由**: excess head 是主信号, WF 模式选择不依赖 downside head (降低过拟合维度)。

### 7.3 Sanity test (Seed 传播防御)

训练完成后:
```python
# 吸取 ng1.0.2 seed 传播 bug 教训
corr_42_123 = corr(pred_excess[42], pred_excess[123])
corr_42_456 = corr(pred_excess[42], pred_excess[456])
corr_123_456 = corr(pred_excess[123], pred_excess[456])

assert all(0.85 <= c <= 0.95 for c in [corr_42_123, corr_42_456, corr_123_456]), \
    "Seed 传播 bug! 过高 = 种子未传进去; 过低 = 训练不稳定"
# 失败动作: ABORT 训练流程, 检查 ng_trainer.py 的 _GLOBAL_RANDOM_SEED 传播是否覆盖所有 booster (lgb/xgb/cb/rf/hgb/lgb_rank)
```

### 7.4 训练资源估算

| 项 | 数量 | 单项时间 | 总计 |
|---|---|---|---|
| Booster 总数 | 144 | 6-12min | 14-29h (单机单线程) |
| 实际 (并行) | — | — | 20-30h (3-4 线程) |

---

## 8. β 搜索协议

### 8.1 搜索流程

```python
β_grid = [0.1, 0.2, 0.3, 0.4, 0.5]

results = {}
for β in β_grid:
    sharpe_per_window = []
    for wf_window in [window_1, window_2, window_3]:
        Z_ex = rank_pct(pred_excess_avg_10d[wf_window])
        Z_dn = rank_pct(pred_downside_avg_10d[wf_window])
        score = Z_ex - β * Z_dn

        # Top-10 持仓, 10d focus, 等权
        sharpe = backtest(score, top_n=10, focus_days=10)
        sharpe_per_window.append(sharpe)

    results[β] = {
        'mean': mean(sharpe_per_window),
        'cv': std(sharpe_per_window) / max(abs(mean(sharpe_per_window)), 1e-6),
    }

# 每个 window 独立评出 β 的 Sharpe 排名
# top2_count[β] = ∑ indicator(rank_in_window(β) ≤ 2)
top2_count = {}
for β in β_grid:
    c = 0
    for w in wf_windows:
        sharpes_this_window = {β: backtest_sharpe(β, w) for β in β_grid}
        β_sorted = sorted(β_grid, key=lambda b: -sharpes_this_window[b])
        if β in β_sorted[:2]:
            c += 1
    top2_count[β] = c

# β* 选择约束 (防过拟合)
β_star = argmax(results[β]['mean'])
assert results[β_star]['cv'] < 0.5, "β* 跨 window 不稳定, 退回 β=0.2 默认值"
assert top2_count[β_star] >= 2, "β* 只靠 1 window 拉高均值 (至少 2/3 window 要进 Top-2), 退回 β=0.2"
```

### 8.2 β* 落点解读

| β* | 含义 | 行动 |
|---|---|---|
| 0.1 | downside head 信号弱 | 如仍通过其他 gate, 可接受但警示 |
| **0.2-0.3** | **理想区间** | 正常推进 |
| 0.4 | downside 较主导 | 正常推进, Pre-2020 gate 加强审视 |
| 0.5 | downside 过于主导 | 警告; 如 Pre-2020 勉强过 gate, 可疑 |

---

## 9. Gate 阈值与 Stage 流程

### 9.1 四级 Gate (严格版)

| Stage | 检查点 | 通过标准 | 失败动作 |
|---|---|---|---|
| **3.5** 训练后立即 | 2025 全年 spot-check (200+ 交易日), Top-10 composite, 10d 持仓 | V5.2 ≥ **65%** | ABORT 回退 ng1.0.1, 生成 diagnostic report |
| **4a** WF-OOS (2020-2026) | 3 WF windows | V5.2 ≥ **72%** **且** MaxDD < **-10%** **且** Sharpe ≥ **3.0** | ABORT |
| **4b** Pre-2020 (2018-2019) | 独立 backfill 评估 | V5.2 ≥ **65%** **且** 年化 ≥ **0%** | ABORT |
| **5** 生产 smoke | 最近 30 日 ng1.3.0 vs ng1.0.1 Top-10 重合度 | 30-70% | 警告但不 ABORT |

### 9.2 Stage 3.5 的关键意义

吸取 **ng1.2.4 教训**: 61 天 spot check gate 通过 (IC=0.73)，扩展到 370 天后 IC 降至 0.28，V5.2=48.5%。**必须用 200+ 天**。

2025 全年有约 240 交易日，天然满足 200+ 要求。

### 9.3 Pre-2020 Gate 的意义

过去失败版本在 Pre-2020 的表现:

| 版本 | Pre-2020 V5.2 | 结局 |
|---|---:|---|
| ng1.0.1 | 73.7% A+ | ✅ 基线 |
| ng1.0.3 | 55.5% B | 勉强 |
| ng1.0.4 (RA label) | 45.5% **C** | 失败 |
| ng1.0.7 (条件 label) | 41.0% **C** | 失败 |
| ng1.2.3 (contrarian) | 31% **D** | 双崩 |

**Pre-2020 是最严格的 gate**。65% 阈值是 "不低于 ng1.0.3 的勉强水平"。

---

## 10. 实施时间表

| Day | Phase | 任务 | 预估 |
|---|---|---|---|
| **Day 1** | 0 | ng130 schema 设计 + ng_schema.py 注册 ng1.3.0 | 2h |
| | 1a | moneyflow_daily code 格式修复 (加 code_6 列或 JOIN 时 split) | 2h |
| | 1b | Tier A 7 特征实现 + 集成到 ng_feature_calculator | 3h |
| **Day 2** | 1c | Tier B 3 moneyflow 因子实现 | 2h |
| | 1d | Tier C 4 候选 + **EMT 四关验证** | 4h |
| | 2 | downside label 计算 + ng130 cache backfill 2020-2026 | 3h |
| **Day 3** | 3a | 3-seed × 2-head 训练启动 (auto-WF) | 20-30h (overnight) |
| **Day 4** | 3b | 训练完成 + sanity test (seed 传播验证) | 1h |
| | 3c | **Stage 3.5 spot check** (2025 全年) | 2h |
| | — | **Gate: V5.2 ≥ 65%** → continue, else ABORT | — |
| | 4 | β 网格搜索 (5 β × 3 windows) | 2h |
| | 5a | **Stage 4a WF-OOS 完整评估** | 2h |
| | 5b | Pre-2020 cache 回填 (2018-01 to 2019-12) | 3h |
| **Day 5** | 5c | **Stage 4b Pre-2020 评估** | 2h |
| | — | **Gate: V5.2 ≥ 65% 且 年化 ≥ 0%** → continue, else ABORT | — |
| | 6 | **Stage 5 生产 smoke** + 观察期启动 | 1h |
| | 7 | 1-2 周观察后切换 `PRODUCTION_VERSION='ng1.3.0'` | 手动 |

**总计**: ~50-60h 实际工作, 跨 **4-5 个日历天**。

---

## 11. 风险缓解与 Rollback

### 11.1 Known 陷阱防御

| 陷阱 | 历史教训 | ng1.3.0 防御 |
|---|---|---|
| Fast-check ICIR 虚高 | ng1.0.9 ICIR=1.29 但 Sharpe=0.79 | 禁用 fast-check 作 go/no-go, 只用 full WF |
| Seed 传播 bug | ng1.0.2 种子未传入 booster, ensemble 无效 | Sanity test: 3-seed pred_excess 互相 corr ∈ [0.85, 0.95] |
| Cache 版本 bug | ng1.2.x schema 混用 AMV 列 | `_is_1_3_branch()` guard + ng130 独立表 |
| RF 权重失衡 | ng1.0.4 RF=94% 导致银行股垄断 | Downside head RF cap ≤ 40% |
| Contrarian 因子反向 | ng1.2.3 mined+mf 叠加双崩 | 每个新因子单独 IC 方向验证 (6 regime 一致) |
| Pre-2020 过拟合 | ng1.0.4 (45.5%) / ng1.0.7 (41%) | Stage 4b 硬 gate 65%; β* 落在 0.4+ 时加强审视 |
| downside head 不泛化 | ng1.0.2 Pre-2020 Sharpe=-0.2 | β=0 可退化到 ng1.0.1, 永远有下限 |

### 11.2 Rollback 策略

1. **全程保持** `PRODUCTION_VERSION='ng1.0.1'`, ng1.3.0 通过 `--scoring-version ng1.3.0` 测试
2. Stage 3.5/4a/4b 任何 FAIL → 立即 ABORT, ng1.0.1 continues serving
3. 全 gate 通过后，**观察 1-2 周日常表现** 再切换 `PRODUCTION_VERSION`
4. ng1.3.0 所有产物 (ng130 cache / 144 个 booster / 报告) **永久保留** 作为后续迭代参考

### 11.3 部分成功场景处理

| 场景 | 动作 |
|---|---|
| Stage 4a 通过, 4b 失败 | Pre-2020 不泛化, ABORT 不切生产; 但保留产物作为"牛市专用"候选 |
| Stage 4a 失败, 4b 通过 | 反常场景, 检查是否 label 计算错误或 cache 污染 |
| β*=0.1 且全 gate 通过 | downside head 信号弱, 等于 ng1.0.1 + 3seed, 仍可切生产 (Sharpe 预期 +50%) |

---

## 12. 成功判据 (Success Criteria)

### 12.1 最小成功 (Minimum Viable Success)

- ✅ Stage 3.5 V5.2 ≥ 65%
- ✅ Stage 4a V5.2 ≥ 72%, MaxDD < -10%, Sharpe ≥ 3.0
- ✅ Stage 4b V5.2 ≥ 65%, 年化 ≥ 0%

→ 允许切换生产, Sharpe 目标可能未完全达到 3.5, 但优于 ng1.0.1。

### 12.2 目标成功 (Target Success)

- ✅ Stage 4a: MaxDD < **-8%**, 年化 ≥ **130%**, Sharpe ≥ **3.5**, V5.2 ≥ **76%** A+
- ✅ Stage 4b: V5.2 ≥ **70%** A

→ 全面超越 ng1.0.1 基线, 实现用户原始目标。

### 12.3 理想成功 (Stretch)

- ✅ Stage 4a: MaxDD < -6%, Sharpe ≥ 4.0, V5.2 ≥ 80%
- ✅ Pre-2020 V5.2 ≥ 73.7% (持平 ng1.0.1)

→ 建立新的跨 regime A+ 底座, ng1.0.1 退役。

---

## 13. 附录 - 命令参考

> ⚠️ 下列命令中标注 `(待实现)` 的在当前代码库**尚未存在**, 需作为 writing-plans 阶段的 implementation task 开发。
> - `ng_trainer.py --multi-task` 参数 (双头训练开关)
> - `ng_cache_updater.py --version ng1.3.0` 分支 (ng130 schema 处理)
> - `scripts/ng130_beta_search.py` (全新脚本)

### 13.1 Schema 创建

```bash
python3 -c "from ml_models.ng.ng_schema import create_table; create_table(version='ng1.3.0')"
```

### 13.2 Cache 回填

```bash
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2020-01-01 --end-date 2026-04-17 --version ng1.3.0
```

### 13.3 训练 (3-seed)

```bash
for seed in 42 123 456; do
  python3 ml_models/ng/ng_trainer.py \
    --version ng1.3.0 --seed $seed --purge-days 15 \
    --auto-wf --multi-task
done
```

### 13.4 Pre-2020 cache 回填

```bash
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2018-01-01 --end-date 2019-12-31 --version ng1.3.0
```

### 13.5 Stage 3.5 spot check

```bash
python3 backtest/batch_generate_v395_reports.py \
  --version ng1.3.0 --start-date 2025-01-01 --end-date 2025-12-31

python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_ng130 \
  --label STAGE-3.5 --top-n 10 --focus-days 10 --rank-field composite
```

### 13.6 β 搜索

```bash
python3 scripts/ng130_beta_search.py \
  --grid 0.1,0.2,0.3,0.4,0.5 --metric sharpe --output reports/ng130_beta_search.md
```

### 13.7 Stage 4a WF-OOS

```bash
python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_ng130_wf_oos \
  --label WF-OOS --top-n 10 --focus-days 10 --rank-field composite
```

### 13.8 Stage 4b Pre-2020

```bash
python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_ng130_pre2020 \
  --label PRE-2020 --top-n 10 --focus-days 10 --rank-field composite
```

### 13.9 生产切换 (gate 全通过后, 观察 1-2 周)

```python
# ml_models/ng/ng_schema.py
PRODUCTION_VERSION = 'ng1.3.0'  # was 'ng1.0.1'
```

---

## 14. 相关文档

- [ng series 详解](../../wiki/models/ng-series.md)
- [ng factor quality 审计](../../wiki/models/ng-factor-quality.md)
- [北极星评估方法](../../wiki/evaluation/north-star.md)
- [ng1.2.4 REJECTED 教训](../../../.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng124_plan.md)
- [ng1.2.3 REJECTED 教训](../../../.claude/projects/-Users-yangxu-StockTradebyZ/memory/ng123_rejected.md)
- [3-seed ensemble 经验](../../../.claude/projects/-Users-yangxu-StockTradebyZ/memory/ensemble_iteration_2026_04_08.md)

---

**本设计文档的状态**: ✅ 已与用户对齐全部决策链, 待用户最终 review 后进入 writing-plans 阶段。
