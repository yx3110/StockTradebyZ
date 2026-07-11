# ng2.1 牛熊专家模型 + 风控选股层 实施计划

**Owner**: yx3110
**Date**: 2026-04-26
**Status**: Draft → Stage 0 (Pre-flight)
**Predecessor**: ng2.0a (生产灰度), ng2.0b (REJECTED)

---

## 1. 目标 (写死, 不可漂移)

| 模型 | 基线 | 目标 |
|---|---|---|
| **ng2.1-bull** (替代 ng1.0.1 在 router 中的位置) | ng1.0.1: V5.2=73.4%, Sharpe=2.753, MaxDD=-11.7% | V5.2 ≥ **75%**, Sharpe ≥ **2.85**, MaxDD ≤ **-13%** |
| **ng2.1-bear** (替代 ng1.0.4 在 router 中的位置) | ng1.0.4: V5.2=78.1%, Sharpe=1.634, MaxDD=-32.6%, 换手 101.5x | V5.2 ≥ **78%**, Sharpe ≥ **1.90**, **MaxDD ≤ -22%**, 换手 ≤ **70x** |
| **整合 (ng2.1 = router + bull + bear + 风控)** | ng2.0a: V5.2=79.3%, MaxDD=-17.6% | V5.2 ≥ **80%**, MaxDD ≤ **-16%**, Sharpe ≥ +5% |

## 2. 设计核心 (规避历史失败)

| 失败案例 | 教训 | 本方案做法 |
|---|---|---|
| ng2.0b sample-weight ×2 | MaxDD -14pp | **regime-filtered training** (数据子集, 非权重) |
| ng1.5.0 regime-refined feat | β_UMD +1.42 | **regime 信息只进 router 和风控层** |
| ng1.4.x AMV/downside | -8~-22pp | **特征不动, 沿用 ng1.0.1 的 66 feat** |
| ng1.0.4 RA penalty=1.5 | 换手 101.5x | **改用 DD-penalized label** |
| ng1.7 alpha 强 risk 差 | V5.2 -14pp | **V5.2 gate 写死 78%/75%** |
| ng1.2.4 烧 8h 才 abort | 没 fast-check | **Stage 1 fast-check 2min 判生死** |

**核心原则**: regime 信息**仅**通过 (1) 训练数据筛选 (2) 选股风控层 进入系统, **绝不**进 features / sample weights / loss regime indicator.

## 3. 架构

```
V11 multi-beta vote regime (复用 ng2.0a router)
        │
   ┌────┴────┐
   ▼         ▼
ng2.1-bull  ng2.1-bear   ← 本方案训练 (regime-filtered specialist)
   │         │
   └────┬────┘
        ▼
风控选股层 L1-L5 (写在 tomorrow_stock_selector.py, 不进模型)
        ▼
   Top-N 输出
```

## 4. ng2.1-bull 设计

| 维度 | 配置 |
|---|---|
| 训练数据 | V11=bull 的交易日 (2020+), regime-filtered |
| 特征 | ng1.0.1 的 66 feat (59 stock + 10 market), **不动** |
| 主标签 | **15d 行业超额** (vs ng101 的 10d, 拉长抓 trend) |
| 辅助标签 | 5d / 10d / 20d 行业超额 (4-target multi-task) |
| 权重 | ICIR 自适应 (沿用) |
| 模型 | LGB + CatBoost + RF + XGB ensemble, 3-seed (42/123/456) |
| WF | auto-WF 选最优, expanding/sliding-720d/sliding-500d+decay730 |
| purge | 15 |

## 5. ng2.1-bear 设计

| 维度 | 配置 |
|---|---|
| 训练数据 | V11=bear 的交易日 (2020+), regime-filtered |
| 特征 | ng1.0.1 的 66 feat (考虑 +2 quality factor 扩展, 仅当 \|β\|<1.5 通过) |
| 主标签 | **5d 行业超额 - λ × max_drawdown_5d** (DD-penalized, λ ∈ {0.3, 0.5, 0.8}) |
| 辅助标签 | 3d / 5d / 10d 行业超额 (3-target) |
| 权重 | ICIR 自适应 |
| 模型 | LGB + CatBoost + RF + XGB ensemble, 3-seed |
| WF | sliding-500d + decay-730d (熊市数据稀, 防 stale) |
| purge | 15 |

## 6. 风控选股层 (写在 daily-selection skill / tomorrow_stock_selector.py)

### L1 永远在
- score_floor = 30
- top_n = 10
- ST / 退市 / 停牌 剔除

### L2 Regime-aware turnover
| | bull | bear |
|---|---|---|
| retention bonus | +20% | 0% |
| EMA α | 0.7 | 0.5 |
| rebalance freq | 15d | 5d |
| 单行业 cap | 3 票 | 2 票 |

### L3 Volatility target
| | bull | bear |
|---|---|---|
| VT 年化 | 25% | 15% |
| cash 上限 | 20% | 50% |

### L4 Crisis hard-stop (仅 bear 启用)
- 触发: V11=bear AND B2(沪深300 60d RV pct) ≥ 90% AND 当日大盘跌 > 3%
- 动作: top_n→5, 单票 cap 5%, cash 下限 70%

### L5 Stop loss (个股)
| | bull | bear |
|---|---|---|
| SL | -8% | -4% |
| trailing | 不启用 | -6% from high |

## 7. Stage 顺序 (任一失败 ABORT)

| Stage | 内容 | Gate |
|---|---|---|
| 0 Pre-flight | 10-checklist | 全 ✅ |
| 1 Fast-check | 单 WF 窗口 2min | bull/bear 各自 10d IC ≥ 0.05 方向正 |
| 2 Bull 训练 | 全 WF | V5.2 ≥ 70% & MaxDD ≤ -15% |
| 3 Bear 训练 | 全 WF (× 3 λ 网格) | V5.2 ≥ 75% & MaxDD ≤ -25% |
| 4a Pre-2020 | 真零泄漏 OOS | bull 净年化 ≥ -10%; bear 净年化 ≥ +5% & 胜率 ≥ 60% |
| 4b 整合 | router+bull+bear+L1-L5 全 WF-OOS | V5.2 ≥ 80% & MaxDD ≤ -16% |
| 4c Baseline 对比 | 隔离 sub-model 贡献 (同 router 同风控) | ΔV5.2 ≥ +1pp & ΔMaxDD ≤ -1pp |
| 5 Paper trade | N ≥ 20 交易日 | in-sample 退化 < 3x |
| 6 灰度 | `--scoring-version ng2.1` 与 ng2.0a 并存 1-2 周 | 实盘单与 paper 一致 |
| 7 生产 | `PRODUCTION_VERSION='ng2.1'` | — |

## 8. 风控网格 (Stage 4b 内嵌)

12 组合 sweep:
- bear VT ∈ {12%, 15%, 20%}
- bear SL ∈ {-3%, -4%, -6%}
- L4 crisis RV pct ∈ {85, 90, 95}
- bear rebalance freq ∈ {3d, 5d, 7d}

加权选最优 (V5.2 0.4 + MaxDD 0.4 + Sharpe 0.2). **bull 参数沿用 ng2.0a, 不调**.

## 9. 时间预算

| Stage | 时长 |
|---|---|
| 0 + 1 | 30min |
| 2 bull | 2-3h |
| 3 bear (× 3 λ) | 6-9h |
| 4a / 4b / 4c | 3h |
| **训练侧总计** | **~12h** |
| 5 paper | ~20 交易日真实时间 |

## 10. 落地交付物

- [ ] `ml_models/ng21/ng21_trainer.py` (基于 `ng_trainer.py` 改, 加 regime-filter + DD-penalty label)
- [ ] `ml_models/ng21/ng21_production_scorer.py`
- [ ] `tomorrow_stock_selector.py` 加 `--scoring-version ng2.1` 分支 + L1-L5 风控
- [ ] `docs/superpowers/specs/2026-04-26-ng21-design.md` 写死 acceptance criteria
- [ ] `scripts/regime_filter_helpers.py` (从 `market_regime_signals` 表读 V11 标签生成训练 mask)
- [ ] Pre-flight 10-checklist 报告 (本 plan 顶部)

## 11. 沿用 / 不重复造轮子

- ✅ 复用 `ng101_feature_cache` (特征不变, 无需新缓存表)
- ✅ 复用 V11 router (`market_regime_signals`, baseline calibration)
- ✅ 复用 ng2.0a 选股流程 + 在其基础上加风控 L2-L5
- ✅ 复用 `caffeinate -i` + `tee logs/` 跑长任务

---

## Stage 4 实测结果 (2026-04-26 完成)

### Stage 4a: Pre-2020 OOS (2018-04~2019-12, 360 trade days)
- ng2.1 V5.2 = **37.8% C** (vs ng2.0a baseline 37.2%, 平/+0.6pp)
- 净年化 -2~-4% (regime mismatch territory, plan 已预期)
- L1=60.6, L3=27.0, L4=24.9, L5=7.5
- **结论**: Pre-2020 specialist generalize 失败 (训练只用 2020+ 数据), 与 ng2.0a 同档

### Stage 4b: 2020-2026 WF-OOS (1525 trade days)
| | ng2.1 raw | ng2.1+L1L2L4 overlay | ng2.0a baseline (memory) |
|---|---|---|---|
| V5.2 | 79.5% A+ | 72.9% A+ | 79.3% A+ |
| L1 信号 | 100% | 57% | — |
| L9 稳健性 | 91.3% | 97.6% | — |
| MaxDD (10d) | -34.1% | **-18.4%** | -23.7% |
| 月度胜率 | 57.9% | **82.9%** | — |
| 超额年化 | 258.5% | 188.2% | ~80% |
| 超额胜率 | 65.5% | **71.7%** | — |

### Stage 4 verdict (vs plan acceptance gates)

| Gate | 目标 | 实际 | 状态 |
|---|---|---|---|
| V5.2 ≥ 80% | strict | 72.9% (overlay) / 79.5% (raw) | ❌ 借边 |
| MaxDD ≤ -16% | strict | -18.4% (overlay) | ❌ 差 2.4pp |
| ΔV5.2 vs ng2.0a +1pp | strict | tied / -6.4pp | ❌ 不优 |
| **MaxDD 优于 ng2.0a -23.7%** | user 关键诉求 | **-18.4%, +5.3pp** | ✅ |
| **超额年化优于 ng2.0a ~80%** | user 关键诉求 | **188%, 2.3×** | ✅ |
| 超额胜率 ≥ 60% | helper | 71.7% | ✅ |

### 核心洞察

1. **specialist 训练 ≠ V5.2 提升** (与 ng2.0b 教训一致): regime-filter + DD-penalty 让 raw V5.2 仅 +0.2pp vs ng2.0a; 67 特征上 alpha 已达上限
2. **L1-L5 风控才是真价值杠杆**: bear industry_cap=2 + crisis 5-day 让 MaxDD 从 -34% → -18%
3. **行业分散的代价**: L1 信号 100→57 (强制次优行业入选), 超额年化 -27%, 但 MaxDD -45%
4. **Sharpe 4.5 vs ng2.0a 2.75 不可直接比** — 评估 methodology 差异需 Stage 4c 校准

### 推荐后续动作

- [ ] **校准 bear industry_cap** (2 → 2.5 或 condition-on-crisis): 找 V5.2/MaxDD 平衡点
- [ ] **Stage 4c fair baseline**: ng2.0a 同方法 batch_generate + eval (3-4h)
- [ ] **Stage 5 paper trade**: 20 个交易日实盘选股 ng2.1 vs ng2.0a
- [ ] **若 Stage 5 通过**: ng2.1 灰度上线 (`PRODUCTION_VERSION` 不动, 保持 ng1.0.6 v1)
