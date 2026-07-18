# NG 系列详解

Next Generation（NG）系列是从 V4.x 代码中独立重构的新一代模型。独立 trainer/scorer/cache，版本分表管理。

## 版本命名规则

格式：`ng{major}.{minor}.{patch}`（如 `ng1.0.0`），不用 `ng` 或 `v` 前缀做版本号。

## NG 1.0.0 — 重构基线 (2026-04-04)

**目标**: 从头构建干净的训练/推理管线，摆脱 V4.x 实验代码积累

**核心设计**:
- 62 特征（59 股票 + 3 市场）
- 绝对收益标签
- 独立 `ng_trainer.py` / `ng_production_scorer.py` / `ng_cache_updater.py`
- 缓存表：`ng_feature_cache`（永久保留）

**性能**: WF 10d ICIR ~0.51

**关键文件**:
- Trainer: `ml_models/ng/ng_trainer.py`
- Scorer: `ml_models/ng/ng_production_scorer.py`
- Cache: `ml_models/ng/ng_cache_updater.py`

---

## NG 1.0.1 — 行业超额 + ICIR 权重 (2026-04-05)

**升级动机**: 绝对收益标签包含行业β噪声，ICIR 固定权重不够自适应

**核心改动**:
- 69 特征（59 股票 + 10 市场）— 增加 7 个市场宽度/动量指标
- 行业超额收益标签（收益率 - 所属行业平均收益）
- ICIR 自适应权重（每个 WF 窗口根据 OOS ICIR 动态计算）
- WF summary 输出（方便快速评估）

**性能**:
- WF 10d ICIR: 0.515 → 0.931（+81%）
- V5: 61.1% A → 70.1% A+（+9pp）
- 缓存表：`ng101_feature_cache`

**模型**: `ng101_multi_target_20260405_013038.pkl`（67.7MB）

**分表规则**: 每版本独立表，不共用 `ng_feature_cache`。原因：features_json 内容不同（62 vs 69 因子），label 语义不同。详见 [已知陷阱](../lessons/known-pitfalls.md)。

### 🆕 2026-07-12 完整财报缓存重训 (当前生产 pkl)

7-11 财报数据修复 (81% 不可见→100%) + `ng101_feature_cache` 全量重建 (label 改后复权) 后按 handoff P0 重训。
**双 gate 通过并已切生产** (`PINNED_PRODUCTION_MODELS['ng1.0.1']`):

| Gate (新口径 2026-07-11-p0fix) | 4-12 pkl | 7-12 重训 pkl |
|---|---|---|
| 对齐窗口 V5.2 (2018-11~2026-04) | 81.3% S | **83.8% S** |
| 对齐窗口 Sharpe / MaxDD | 2.550 / -17.7% | 3.530 / -23.0% |
| Pre-2020 净年化 / V5.2 | +17.2% / 49.7% B | **+19.4% / 55.4% B** |
| WF 10d ICIR 均值 | 0.854 (3窗) | 0.851 (4窗) |

- 完整财报最大兑现在 Pre-2020 后向泛化 (全维度提升); MaxDD 退化 5.3pp 是主要代价
- 前向 WF-OOS 匹配窗 (2024-05~2025-11, vs 4-28 旧缓存 fold preds): 全持仓期矩阵 mixed-parity
  (1d/3d/7d-Sharpe 重训胜, 5d/10d 基线胜); **10d 非重叠口径重训明显偏弱 (9.4% vs 27.1% 净年化, 36 期相位敏感)
  — paper trade ≥20 交易日重点盯这一格**
- 首窗 ABORT 线插曲: w1 ICIR 0.464 < 0.6 (门槛源自 4-12 基线 w1=0.613), 但 label 口径已变不可直比,
  经用户确认跑完; w2/w3 反超 (1.072/1.137 vs 0.933/1.016), 事后证明 w1 是 924 行情 regime 噪声
- 重建缓存不含 3 个重复特征 (volume_contraction/sw_index_return_5d/industry_relative_strength),
  训练全 NaN → 树不使用, canonical 孪生在场, 信息零损失
- 详见 `reports/system_evaluation/ng101重训评估_20260712.md`

**模型**: `ng101_seed42_multi_target_20260712_213343.pkl` (77MB, 含 Check 9 元数据: git `73a36ce4`, seed 42, expanding, purge 15)

### 🆕 2026-07-18 3-Seed Ensemble (当前生产)

seed 123/456 同配置补训 (hp sweep 确认 V4.7.3 参数最优后执行), 三 pkl 平均:

| 指标 (对齐窗口 10d) | 单 seed 42 | **3-seed** | 旧 4-12 基线 |
|---|---|---|---|
| V5.2 | 83.8% S | **84.2% S** | 81.3% S |
| Sharpe / Calmar | 3.530 / 6.47 | **3.648 / 10.09** | 2.550 / 5.85 |
| **MaxDD** | -23.0% | **-14.7%** | -17.7% |
| Pre-2020 净年化 | +19.4% | **+21.2%** | +17.2% |
| 匹配窗 WF-OOS 10d 净年化 | 9.4% | **22.9%** | 27.1% (旧缓存 4-28) |
| WF-OOS 全窗 V5.2 | 59.1% B | **62.3% A** | — |

- 单 seed 切换的两大遗留问题**全部被 ensemble 修复**: MaxDD 退化 (-23.0→-14.7, 反超基线) 与 10d 前向弱格 (9.4→22.9)
- 三 seed w1 (2024-05~11) ICIR 全部 0.46-0.48 — 该窗口难度是结构性的, 与 seed 无关
- scorer ensemble 判定重构: PINNED 清单长度 >1 即 ensemble (唯一权威), version_ge 只作未注册版本 fallback
- fold-preds 平均工具: `scripts/avg_seed_fold_preds.py`; 评估详情: `reports/system_evaluation/ng101重训评估_20260712.md` 附录二

**模型**: `ng101_seed{42,123,456}_multi_target_202607*.pkl` × 3 (PINNED_PRODUCTION_MODELS)

---

## NG 1.0.2 — 下行风险模型 (2026-04-05)

**升级动机**: 1.0.1 只预测收益，不评估下行风险

**核心改动**:
- 新增 `downside_10d` 预测目标（10日最大回撤）
- Risk-discounted composite scoring（收益预测 × 风险折扣）
- CPPI(floor=5%, multiplier=20) 生产配置

**性能**: V5.2 = 74.0% A+

**生产配置**: `production_config.json`

**缓存表**: `ng102_feature_cache`

---

## NG 1.0.3 — 去翻转因子 (2026-04-07)

**升级动机**: 1.0.2 在 2018-2020 OOS 评估中存在 cache 不匹配 bug（pred_10d 全0）。修复后发现去掉 3 个 IC 方向翻转的因子可以大幅提升跨周期泛化能力。

**核心改动**:
- 66 特征（56 股票 + 10 市场）— 从 69 中去掉 3 个翻转因子
- 去掉的因子：`log_market_cap`, `cs_rank_market_cap`, `pullback_from_high`
  - 这些因子在 2020 年前后 IC 方向翻转（大盘→小盘偏好切换）
  - GBDT 学到了训练期的方向，但在 OOS 上方向错误
- 共享缓存表 `ng103_feature_cache`（features 是 ng1.1.0 的严格子集）

**2018-2020 OOS 评估 (score ranking, Top-5, 10d持仓)**:
| 指标 | ng1.0.2 baseline | **ng1.0.3** |
|------|:---:|:---:|
| 年化(毛) | +1.3% | **+18.1%** |
| 超额年化 | +6.8% | **+24.8%** |
| Sharpe | -0.02 | **0.45** |
| Alpha | +7.6% | **+25.0%** |
| IR | 0.28 | **0.89** |
| V5.2 | 49.1% B | **55.5% B** |

**6 配置鲁棒性验证**：所有 Top-N × hold-days 组合一致提升 12-39pp。

**模型**: `ng103_multi_target_20260407_005245.pkl`

**训练命令**:
```bash
python3 ml_models/ng/ng_trainer.py --start-date 2020-01-01 --purge-days 15
```

**选股命令**:
```bash
python3 tomorrow_stock_selector.py 2026-04-07 --scoring-version ng1.0.3
```

---

## NG 1.0.4 — 风险调整标签 + 多种子Ensemble (2026-04-08)

**升级动机**: ng1.0.3 只优化原始收益，不考虑回撤；换手率~43x过高

**核心改动**:
- 75 特征（65 股票 + 10 市场）— 在 ng1.0.3 基础上新增 9 个信号平滑特征
- 风险调整标签: `ra_label = excess × (1 + maxDD)^1.5`（惩罚高回撤股票）
- 5-seed Ensemble（seed=42/123/456/789/2024，predictions平均）
- IC 稳定性分析器: `scripts/ic_stability_analyzer.py`（6-regime 自动筛选）
- `version_ge()` 安全版本比较函数（解决 ng1.0.10 字符串比较问题）
- 新增 9 特征: trend_strength_60d, ma60_distance, price_channel_pos_40d, vol_ratio_5d_60d, vol_regime, downside_vol_20d, current_drawdown, recovery_speed_20d, gap_risk_20d

**缓存表**: `ng104_feature_cache`（3.18M 行, 1514 天）

**模型**: `ng104_seed{42,123,456,789,2024}_multi_target_*.pkl`（各 68MB）

**WF IC (seed 42)**: 3d=0.056/0.75, 5d=0.059/0.79, 10d=0.066/0.93, 15d=0.073/1.07

**/simplify 审查修复**: version_ge向量化+vol_regime优化+gap_risk向量化+NaN安全+无用参数清理

---

## NG 版本综合排名 (2026-04-11, 2026-04-20 补充复核)

### 裸信号对比 (Top-10, focus_days=10, 无CPPI, WF-OOS 2020-2026)

> **⚠️ 2026-04-20 全量复核后的排名倒置**: 用统一口径 (1606-1877 天完整样本, 当前评分卡 V5.2) 重测, **ng1.0.6 综合实际最优** — 详见 "WF-OOS 完整复核" 小节。下表是 2026-04-11 旧数字, 个别行 (ICIR/熊市ICIR) 可能基于不同评估窗口, 仅作方向参考。

| 排名 | 版本 | V5.2 | 年化(毛) | 年化(净) | Sharpe | ICIR | 换手 | 熊市ICIR | 特点 |
|:---:|:-----|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | ng1.0.6 (0AMV) | **78.9% A+** (4-20 实测) | — | 115.7% (10d) | **2.808** (10d) | — | — | — | ⭐ **综合最优**, β_UMD=+0.005 最干净 |
| 2 | **ng1.0.1** (4-12 bugfix) | 73.4% A+ (4-20 实测) | — | 91.5% | 2.367 (10d) | 0.93 | 45x | 0.13 | WF-OOS 单边强, MaxDD 最小 (-11.7%) |
| 3 | (旧排名) **ng1.0.7** | 76.8% A+ | 84.4% | 76.9% | 1.93 | **0.656** | 44x | **+0.205** | ★信号质量L1=91.3%最高 (老数字) |
| 4 | (旧排名) ng1.0.4-3s | 79.5% A+ | 64.5% | — | 1.63 | — | 24x | +0.143 | 多种子稳定 (老数字) |
| 5 | ng1.0.3 | 69.9% A | 30.4% | — | 1.16 | — | 24x | — | Pre-2020最优 |

### 带组合优化对比 (Top-10, focus_days=10)

| 排名 | 版本 | V5.2 | 年化(净) | Sharpe | MaxDD | 换手 | 优化方式 |
|:---:|:-----|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **ng1.0.8** | **A+** | **94.6%** | **2.52** | **-12.6%** | **36x** | sell50+cost0.3%+分批调仓 |
| 2 | ng1.0.1+CPPI | A+ (78.9%) | 72.2% | 2.339 | -12.6% | 24x | CPPI F0.08/M20 |
| 3 | ng1.0.4-5s+CPPI | A+ (75.9%) | 47.4% | 1.611 | -16.6% | 24x | CPPI F0.08/M20 |
| 4 | ng1.0.2-3s+CPPI | A+ (76.2%) | 82.5% | 1.501 | -15.8% | — | CPPI F0.05/M20 |

### Pre-2020 OOS (2018-2019, 无泄露) — **2026-04-20 全量订正**

**⚠️ 之前版本排行榜的 V5.2 A+ 数字 (ng1.0.1=73.7%, ng1.0.5=A+, ng1.0.3=55.5%, ng1.0.4=45.5% C) 全部已证实是 4-10 评估 bug 修复前的 ghost numbers**。以下是统一口径 (10d hold, composite rank, V5.2, 当前评分卡) 实测值：

| 排名 | 版本 | V5.2 | 年化(净) | Sharpe | MaxDD | β_UMD | 老声称 |
|:---:|:-----|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | **ng1.0.1 (4-10 pkl + --production CPPI)** | **55.2% B** | +18.3% | +0.89 | -21.4% | -2.52 | — |
| 2 | ng1.0.1 (4-10 pkl, 裸 10d) | 49.5% B | -5.0% | -0.21 | -36.9% | -4.78 | ~~73.7% A+~~ |
| 3 | ng1.0.1 (4-12 bugfix pkl, 10d) | 45.5% B | -19.0% | -0.33 | -25%+ | -4.78 | — |
| 4 | ng1.0.1 (4-12 bugfix pkl, 5d) | 45.6% B | — | — | — | +3.06 | — |
| 5 | ng1.0.3 (pkl 20260407) | 42.5% C | -33.4% | -0.96 | **-73.4%** | -4.26 | ~~55.5% B~~ |
| 6 | ng1.0.6 (0AMV 牛熊切换) | 41.1% C | **+0.7%** ⭐ | **+0.18** ⭐ | -27.1% | -5.17 | — |
| 7 | ng1.0.4 (pkl 20260408) | 38.8% C | -9.2% | -0.14 | -28.5% | -1.65 | ~~45.5% C~~ |
| 8 | ng1.0.7 (条件 label) | 34.7% C | -35.7% | -1.06 | -49.3% | -2.57 | ~~41.0% C~~ |

**关键 Pre-2020 观察** (2026-04-20):
1. ng1.0.1 4-10 pkl + CPPI 是 V5.2 最高 (55.2% B), 但 ng1.0.1 裸 4-12 bugfix pkl 已经掉到 45.5% B
2. **ng1.0.6 是唯一 Pre-2020 年化/Sharpe 为正的模型** (+0.7%, +0.18), 虽然 V5.2 只有 41.1% C (因 MaxDD -27.1%/换手拖分)
3. **ng1.0.3 MaxDD -73.4% 是灾难** — 老 "55.5% B" 声称和实际不是一个世界
4. 所有 β_UMD 都在 -5.2 到 +3.1 之间摇摆, t<1.7, 纯小样本噪声, 不反映真动量暴露

**关键教训**:
- 4-12 `revenue_growth` bugfix 提升了 post-2020 (WF-OOS A+), 但 **拖累了 Pre-2020** (ng1.0.1 净年化 +9%→-19%) — 真正的 regime tradeoff, 不是 bug
- 覆盖率只有 ~30% (CIRC_MV_MIN=50亿 + MIN_DATA_DAYS=60 过滤), 回填缓存也只增 0.3% 行数, 不能改变结构
- β 归因: Pre-2020 样本太小 (24-72 非重叠点) 所有 t<1.7 统计不显著, 符号摇摆纯噪声不可参考

### WF-OOS 完整复核 (2026-04-20, 15d/10d composite)

| 版本 | V5.2 | 10d Sharpe | 10d 年化(净) | MaxDD | β_UMD | β_SMB | Alpha t | 样本 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ng1.0.6 (0AMV)** | **78.9% A+** ⭐ | **2.808** ⭐ | **115.7%** ⭐ | -21.4% | **+0.005** ⭐ | +1.54 | 4.54 | 1606 天 |
| ng1.0.1 (4-12 bugfix) | 73.4% A+ | 2.367 | 91.5% | **-11.7%** ⭐ | +0.38 | +0.74 | **5.39** | 1877 天 |

**ng1.0.6 综合最优**: V5.2 / 10d Sharpe / 10d 年化 / β_UMD 清洁度 全部胜 ng1.0.1, **唯一劣势是 MaxDD -21.4% 比 ng1.0.1 -11.7% 差一倍**。

### 核心结论 (2026-04-20 订正, 继续迭代中)

1. **WF-OOS + Pre-2020 综合最优: ng1.0.6 (0AMV 牛熊切换)**, 不是 ng1.0.1
   - WF-OOS: V5.2=78.9% A+, Sharpe=2.808 (10d) / 2.081 (15d), β_UMD=+0.005 最干净
   - Pre-2020: 唯一正年化/正 Sharpe 的版本 (+0.7% / +0.18)
   - 唯一痛点: MaxDD=-21.4%~-22.9%, 约 ng1.0.1 两倍, 应叠加 ng1.0.5 三层风控
2. **ng1.0.1 仍是 "MaxDD 友好型" 最优** (4-12 bugfix, -11.7% MaxDD), WF-OOS 单边 A+ 强信号, 但跨 regime 泛化弱 (Pre-2020 负年化)
3. **ng1.0.8 是当前最优"低换手"产品 overlay** (ng1.0.1 基座 + sell50 规则, Sharpe=2.52, 换手 36x)
4. ng1.0.5(三层风控) Pre-2020 "A+ Sharpe=3.09" 是 ghost number, 待用新 pkl 复核
5. ng1.0.7 的 regime 改进在 Pre-2020 退化 (C级), 方向放弃 (结论成立)
6. ng1.0.9 持久特征方向失败 — fast-check ICIR 虚高，生产 Sharpe 仅 0.79
7. **后续迭代的 Pre-2020 Gate 口径调整**: 不再要求 ≥ 65%/70% A+, 改为 "同向 alpha 验证" (净年化 ≥ 0 且超额胜率 ≥ 60%) 即可通过
8. **生产推荐待定** (2026-04-20 后): ng1.0.6+ng1.0.5 风控叠加 是当前最有潜力的候选, 待跑完 MaxDD 压制测试后可能替代 ng1.0.8 作为生产主配置

### ng1.0.4 CPPI 参数网格 (WF-OOS)

| CPPI配置 | V5.2 | 年化(毛) | Sharpe | MaxDD |
|:---------|:---:|:---:|:---:|:---:|
| F0.05,M15 | 78.0% | 34.1% | 1.474 | -19.7% |
| F0.05,M20 | 79.0% | 44.9% | 1.621 | -16.6% |
| **F0.08,M15** | **78.5%** | **47.3%** | **1.626** | **-16.3%** |
| F0.08,M20 | 78.3% | 47.4% | 1.611 | -16.6% |
| F0.10,M20 | 78.2% | 47.9% | 1.567 | -15.7% |
| F0.12,M15 | 78.4% | 48.7% | 1.549 | -16.9% |

### ng1.0.8 sell_threshold 参数搜索 (WF-OOS)

| sell_threshold | 换手 | Sharpe | 年化(净) | 评价 |
|:-:|:-:|:-:|:-:|:---|
| 20 | 40x | — | — | 几乎无效(A股10d内82%跌出Top-20) |
| **50** | **36x** | **2.52** | **94.6%** | **甜蜜点** |
| 100 | 31x | — | 下降 | 过度宽松 |
| 200 | 27x | — | 下降 | Sharpe恶化 |

---

## NG 1.0.6 — 0AMV牛熊切换模型 (2026-04-09)

**升级动机**: ng1.0.1裸模型年化129.7%/Sharpe3.17最强但MaxDD=-25.4%；ng1.0.4在熊市更稳（最差60dICIR=+0.14）。用市场级指标自动切换可以取两者之长。

**核心设计**:
- 复刻指南针(Compass)0AMV活筹指数，改造为全市场版本
- 数据源: 上证指数+深证成指 每日成交额之和
- 算法: 通达信SMA/DMA + 标准MACD(12/26/9) + MA60
- 牛熊状态机:
  - 转牛(急涨): var1涨≥4.3% AND var1>ma60 AND macd>0
  - 转熊(急跌): var1跌≤-2.3% AND var1<ma60 AND macd<0
  - 转熊(缓跌): 连续10天 var1<ma60且macd<0 → 强制转熊
- 牛市→ng1.0.1, 熊市→ng1.0.4-3seed

**性能 (Top-10, 10d持仓, 无CPPI, 2020-2026)**:

| 指标 | ng1.0.1(纯) | ng1.0.4-3s(纯) | **ng1.0.6** |
|------|:---:|:---:|:---:|
| V5.2 | 79.4% A+ | 79.5% A+ | **78.3% A+** |
| 年化(毛) | 129.7% | 64.5% | **92.0%** |
| Sharpe | 3.166 | 1.626 | **2.160** |
| MaxDD | -25.4% | -23.1% | **-20.2%** |
| 最差60dICIR | -0.249 | +0.143 | **+0.337** |
| 超额胜率 | 70.1% | 57.6% | **65.6%** |

**牛熊分布**: 1931天, 牛468天(24%), 熊1463天(76%), 18次切换

**关键切换时点**:
- 2020-03-31 →熊 (新冠), 2020-07-02 →牛 (夏季牛)
- 2024-09-25 →牛 (政策刺激), 2025-01-06 →熊
- 2026-01-12 →牛, 2026-03-31 →熊 (缓跌10天强制转熊)

**关键文件**:
- 0AMV引擎: `indicators/market_amv.py`
- 切换回测: `backtest/regime_switch_backtest.py`
- 数据回填: `fetch_data/backfill_index_amount.py`
- DB表: `market_amv` (trade_date, var1, amv_c5/c13/c34/inf, amv_ma60, amv_dif/dea/macd, amv_regime)

**选股命令**:
```bash
python3 tomorrow_stock_selector.py 2026-04-09 --scoring-version ng1.0.6
```

**已知限制**:
- 缓跌转熊用固定10天阈值，未来可能需要根据市场环境自适应
- 0AMV数据依赖上证+深证指数amount，北交所未纳入（占比<1%可忽略）

### factor_returns.py bug修复 (2026-04-09)

在NG模型裸信号公平对比中发现 `load_or_build_factors()` 的Build路径返回string类型index，与datetime的portfolio_returns无法交集，导致因子归因全零。已修复（一行 `df.index = pd.to_datetime(df.index)`）。此bug曾导致ng1.0.4的L6因子归因虚高（25.5/30→修复后29.5/30）。

### ng1.0.4 RF权重失衡问题 (2026-04-08 发现)

ng1.0.4 的 Top-10 几乎全是银行股（42只银行占训练集仅1.1%，但平均得分是全市场6.7倍）。

**根因分析**:
1. **RF主导10d/15d**: ICIR优化后 Random Forest 权重高达 94-95%，其他模型（LGB/XGB/CB）被压到1%
2. **Composite 70%给长周期**: 10d=35% + 15d=35% 合计70%，而这两个目标几乎100%由RF决定
3. **RF天然偏好低波动**: 多棵树取平均→预测趋向均值，银行股波动小、特征稳定→RF给出更一致的正面预测
4. **downside_model + liquidity_discount**: 进一步惩罚高波动中小盘，利好银行

**对比**: ng1.0.1 没有此问题，选股行业分散（机器人、医疗、军工、半导体等），预测收益也更高（1.4-2.0% vs 0.9-1.4%）

**潜在修复方向**:
- 限制RF在ensemble中的最大权重（如cap 50%）
- 加入行业分散约束（每行业最多N只）
- 用行业中性化排名代替绝对排名

### Daily Update 缓存修复 (2026-04-09)

**问题**: `quick_daily_update.py` 只更新 `ng103_feature_cache`（默认版本），ng1.0.1/ng1.0.4 缓存不会自动更新，导致 ng1.0.6 切换后找不到当天数据。

**修复**: `update_ng_feature_cache()` 现在同时更新 3 个版本: ng1.0.3(默认) + ng1.0.1(牛市) + ng1.0.4(熊市)

### SCORER_REGISTRY model_path 修复 (2026-04-08)

**问题**: `NGProductionScorer()` 不指定 model_path 时自动选最新 .pkl，导致指定 ng1.0.1/ng1.0.2 时实际加载了 ng1.1.0 模型。

**修复**: 在 `SCORER_REGISTRY` 中为每个 NG 版本显式指定 `model_path` 或 `version` 参数:
- ng1.0.1: `model_path='...ng101_multi_target_20260405_013038.pkl'`
- ng1.0.2: `model_path='...ng_multi_target_20260405_194751.pkl'`
- ng1.0.4: `version='ng1.0.4'`（自动发现5个seed模型做ensemble）

### 待解决问题

1. ng1.0.1 / ng1.0.4 缺少 Pre-2020 独立验证
2. 缓跌转熊阈值(10天)是否需要自适应调整
3. 牛市切换后是否需要延迟N天确认（避免假突破）
4. ng1.0.4 RF权重失衡导致银行垄断Top-10，需考虑权重cap或行业分散

---

## NG 1.1.0 — 已废弃

ng1.1.0 的三方向实验（资金流因子、残差标签、WF框架升级）评估后发现仅"去翻转因子"方向有效，已合并为 ng1.0.3。其余方向（残差标签=无效、WF8+regime=无效、moneyflow=10d退步）废弃。

**数据资产保留**：`ng103_feature_cache`（3.6M行）、`moneyflow_daily`（8.8M行）由 ng1.0.3 继续使用。

---

## NG 1.0.7 — 条件化单模型 + Pareto回撤过滤 (2026-04-10)

**升级动机**: NG1.0.1裸信号最强(129.7%年化)但MaxDD=-25.4%，熊市信号反转(worst 60d ICIR=-0.249)。需要模型本身感知市场环境，而非依赖后置风控。

**核心设计 (Part A: 条件化模型)**:
- 56 stock + 18 market (10基础 + 8扩展) = 74 特征 (交叉特征7个全被IC筛选淘汰)
- 新增8个市场状态连续特征: 0AMV连续值/MACD/regime天数/60d收益/波动比/涨跌面动量/偏度/流动性压力
- **条件化标签**: 熊市(mkt_ret_20d<-5%)用截面排名blend，牛市用行业超额，连续插值不硬切换
- 增强样本加权: bull=0.7, sideways=1.0, bear=1.5, crisis(mkt_ret<-10%)=2.0

**核心设计 (Part B: Pareto回撤过滤)**:
- 独立downside模型预测maxdd_10d
- 硬过滤最差20%风险标的后按alpha排序
- 参数: `--risk-filter-quantile 0.20`

**性能** (V5.2评分卡, Top-10, 10日持仓):
- V5.2: **76.8% A+**
- 年化(毛): **84.4%** (净: 77.8%)
- Sharpe: 1.96 (受换手率44x拖累)
- ICIR: **0.656** (vs NG1.0.1的0.51, **+29%**)
- IC>0%: **73.5%** (vs 70.1%, +3.4pp)
- L1信号质量: **91.3%** (全系列最高)
- L9条件稳健性: **95.0%** (regime自适应)
- 熊市ICIR: **0.205 > 0** (解决了NG1.0.1的-0.249反转)

**关键发现**:
1. 7个交叉特征全被IC筛选淘汰(与已有特征corr>0.7或IC<0.015)，说明GBDT已能自动学到交叉效应
2. 条件化标签+扩展市场特征是核心改进来源(L1从78.9%→91.3%)
3. 换手率44x过高，需focus_days=15或EMA平滑降低

**Production Config 网格搜索 (2026-04-10)**:

| Config | V5.2 | 年化(净) | Sharpe | 换手 |
|--------|:----:|:--------:|:------:|:----:|
| **baseline(裸)** | 76.7% | **76.9%** | **1.93** | 44x |
| ema0.7 | 78.0% | 71.7% | 1.76 | 42x |
| ema0.7+sf30+ret0.2 | 77.6% | 76.3% | 1.81 | 42x |
| ema0.7+sf30+ret0.3 | 78.2% | 77.6% | 1.67 | 41x |
| sf30+ret0.2(no EMA) | 78.3% | 40.3% | 1.30 | 24x |

**结论**: 裸baseline最优(Sharpe=1.93)。EMA/retention在ng1.0.7上反而降Sharpe — 条件化标签已自带regime自适应，EMA延迟了信号。

**Pareto过滤效果**: IC=0.137的downside模型训练成功，但Pareto硬过滤效果微小(年化-0.9pp)——risk discount已在composite中软惩罚，硬过滤只排除极少量边缘股票。

**ng1.0.7 vs ng1.0.6 对比**:
- ng1.0.7信号质量更高(ICIR 0.656 vs ng1.0.6无直接对比, L1=91.3%)
- ng1.0.6年化更高(92.0% vs 84.4%)但靠牛熊切换的ng1.0.1贡献
- ng1.0.7是**单一模型**，ng1.0.6是两个模型硬切换，维护更简单
- ng1.0.7解决了熊市信号反转(ICIR=+0.205)，ng1.0.6靠切换回避
- Sharpe: ng1.0.6=2.16 > ng1.0.7=1.93（ng1.0.6换手更低24x vs 44x）

**缓存表**: `ng107_feature_cache`（318万行, 1515天）
**模型文件**: `ng107_seed42_multi_target_20260410_012502.pkl`（含downside模型, IC=0.137）

---

## NG 1.0.8 — 低换手组合构建 (2026-04-10)

**升级动机**: ng1.0.1 裸信号年化129.7%但换手45x，交易成本侵蚀严重。需要在不改变模型的前提下通过组合构建规则降低换手。

**核心设计**: 4条固定参数规则（零过拟合风险），不修改模型本身，仅改变持仓选择和调仓逻辑。

**四条规则**:
1. **Hysteresis 持仓缓冲** — 买入需进Top-8，卖出跌出Top-50(sell_threshold=50)
2. **Staggered Rebalancing 分批调仓** — 10仓分2组，交替调仓(组A: day0/10/20, 组B: day5/15/25)
3. **Minimum Holding 最小持有期** — min_hold_days=5，防止频繁进出
4. **Cost-Aware Ranking 成本感知** — 新股惩罚0.3%(单次交易成本)，需显著优于现有持仓才替换

**性能 (Top-10, 10d持仓, WF-OOS)**:

| 指标 | ng1.0.1(裸) | **ng1.0.8** | 变化 |
|------|:---:|:---:|:---:|
| 换手 | 45x | **36x** | **-20%** |
| Sharpe | 2.37 | **2.52** | **+6%** |
| 年化(净) | 129.7% | **94.6%** | -27%(交易成本节省) |
| MaxDD | -25.4% | **-12.6%** | 改善 |
| V5.2 | A+ | **A+** | 维持 |

**关键发现**: sell_threshold 必须远大于 top_n。A股信号衰减快，10天后Top-10中8.2只跌出Top-20。buy=8/sell=20几乎无效(45→40x)，需sell=50才有效(→36x)。详见 [已知陷阱](../lessons/known-pitfalls.md)。

**生产命令**:
```bash
python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_ng101 \
  --top-n 10 --focus-days 10 --rank-field composite \
  --buy-threshold 8 --sell-threshold 50 --n-groups 2 \
  --min-hold-days 5 --cost-penalty 0.003
```

**关键文件**:
- 回测引擎: `backtest/backtest_report_based.py`（hysteresis/staggered逻辑）
- CLI: `backtest/run_north_star_eval.py`（--buy-threshold/--sell-threshold等参数）
- 设计文档: `docs/superpowers/specs/2026-04-10-ng108-low-turnover-portfolio-design.md`

**Bug修复 (2026-04-10)**: 初版缺少 `is_rebal_day` 检查，导致每天执行调仓逻辑→换手未降(45→44x)。加 `i % rebal_interval == 0` 后正常(→36x)。

---

## NG 1.0.9 — 持久特征实验 (2026-04-11) ❌ 失败

**升级动机**: ng1.0.8通过组合规则降换手到36x，但能否从信号端根本降低排名波动？过滤掉快变特征，只保留慢变/持久特征，应该产生更稳定的排名。

**核心设计**:
- **Part A: 特征半衰期过滤** — 计算每个特征的10日rank autocorrelation(Spearman)，只保留autocorr≥0.5的"持久特征"
- **Part B: 平滑标签** — `smooth_label_10d = mean(return(t+k, t+k+10) for k in range(5))`，奖励持续机会而非精确择时
- `--min-autocorr` 可调阈值(默认0.5)，预计算autocorr查找表加速

**实验结果**:

| 配置 | 特征数 | fast-check ICIR | 生产Sharpe | 换手 | 结论 |
|------|:---:|:---:|:---:|:---:|:---|
| ng1.0.1 baseline | 69 | 0.93 | 2.37 | 45x | 基线 |
| **ng1.0.9 (ac≥0.5)** | **22** | **1.29 (+39%)** | **0.79** | **14.7x** | ❌ Sharpe崩塌 |
| ng1.0.9 折中(ac≥0.4) | 31 | 1.38 | 1.69 | 41x | ❌ 仍不如ng1.0.8 |
| ng1.0.9 折中+sell50 | 31 | — | 1.54 | 28x | ❌ 仍不如ng1.0.8 |

**失败根因**:
1. **Fast-check ICIR虚高**: 特征减少→IC方差小→ICIR高(WF窗口内)，但IC绝对值在生产推理时大幅下降
2. **A股短期alpha来源错位**: 10天alpha主要来自动量/技术特征(autocorr<0.3)，而非基本面特征(autocorr>0.5)。过滤掉快变特征=过滤掉alpha来源
3. 换手虽然达标(14.7x)，但Sharpe=0.79完全无法接受

**最终结论**: ng1.0.9方向 **ABANDONED**。持久特征不能产生足够的10天短期alpha。**ng1.0.8(组合规则降换手)仍是最优方案**。

详见 [已知陷阱 — Fast-check ICIR高≠生产ICIR高](../lessons/known-pitfalls.md)。

---

## NG 1.5.0 — Tier B Regime-Refined 特征 (2026-04-21) ❌ REJECTED

**升级动机**: ng1.4.0 Stage 4a V5.2=67.6% A (< 70%), 需要继续加强跨 regime 稳健性. spec 设计加 5 个 regime-refined 特征覆盖:
- Stock (4): `industry_regime_agreement` (60d 行业-大盘方向一致性), `recent_maxdd_60d` (path-dep 60d 最大回撤), `volatility_skew_20d` (下行/上行波动比), `upside_capture_60d` (牛市跟涨能力)
- Market (1): `amv_regime_bull_prob` (0AMV 连续牛市概率, 替代硬 0/1 regime)

**核心设计**:
- 75 特征 (61 stock + 14 market), 自有 schema `ng150_feature_cache` (3.7M rows 2018-04..2026-04)
- ng1.4.0 底座 + 5 Tier B 增量, 单头 MSE, 3-seed ensemble (42/123/456)
- Fast-check PASS: WF-OOS 10d IC = 0.06-0.08 (方向正)
- 完整训练 3h35m, WF-OOS 10d ICIR 0.97-1.72 (三窗口均强)

**Gate 结果**:

| Stage | 窗口 | V5.2 raw | V5.2 weighted | Sharpe | MaxDD | β_UMD | Verdict |
|---|---|---|---|---|---|---|---|
| 3.5 | 2025 only (242d) | 73% | 48% B | 2.605 | - | 0.00* | PASS |
| **4a** | 2024-2026 (552d) | 63% | **63% A** | **0.89** | -14.4% | **+1.42** | **REJECTED** |

*Stage 3.5 β zeros = FF 样本不足 (<阈值), 非真"无暴露".

**基线对比** (10d hold, composite, top-10):

| 模型 | V5.2 A+ | Sharpe | MaxDD | β_UMD | Alpha t |
|---|---|---|---|---|---|
| ng1.0.1 (生产) | 73.4% | 2.75 | **-11.7%** | +0.38 | 5.39 |
| ng1.0.6 (regime switch) | **78.9%** | **2.81** | -22.9% | **+0.005** | 4.54 |
| ng1.4.0 | 67.6% | 1.04 | -13.7% | +0.56 | 1.55 |
| **ng1.5.0** | **63.0%** | **0.89** | -14.4% | **+1.42** | 1.49 |

**失败根因**:

5 个 Tier B 特征都是 **动量相邻信号**. 模型在 in-sample WF 里学习它们的权重 (ICIR 看起来强), 但实际暴露形式是大 β_UMD=+1.42 (约 ng1.4.0 的 3×). 2024 熊市 regime:
- β_MKT 反转到 **-0.94** (大 negative market exposure, 对冲成本)
- Alpha t 降到 1.49 (not statistically significant)
- Sharpe 从 ng1.4.0 的 1.04 腰斩到 0.89

**2025-only vs 2024-2026 的分化** 揭示核心问题: 新特征在 2025 温和牛 helped (Stage 3.5 raw 73%), 但在 2024 熊市 **害了模型** (Stage 4a 63%). 跨 regime 失败.

同 **ng1.0.7** (conditional label + AMV, Pre-2020 34.7% C) 和 **ng1.2.x** (loss-layer regime, all V5.2 41-53% C) 的失败模式. spec I3/I5 洞察再验证:

> **把 regime 内化作模型输入, 不能替代外部 regime switching (ng1.0.6 路径).**

**决策**: Phase C fallback (ng1.0.1 + ng1.0.6 组合 overlay) 被拒 — 相当于 ng1.0.6 的 soft 版, 不值得 3h 工作量. ng1.5.0 停在 Stage 4a 失败点. 生产保持 ng1.0.1.

**Infra 永久保留** (未来版本可复用):
- ng_schema / ng_trainer / ng_cache_updater / ng_production_scorer 的 ng1.5.x 分支
- Check 9 pkl reproducibility metadata (`git_commit_hash / host / training_duration_sec / seed / schema_version`) — 所有未来 NG 训练都写入
- `_load_industry_5d_ret_history` LAG-based industry 1d→5d 返回 helper (可复用)
- `recent_maxdd_60d` refactor: path-dependent 最大回撤 vs snapshot `current_drawdown`, 值得保留作未来特征

**Artefacts** (归档, 不进生产):
- 3 × 70MB pkls: `ng150_seed{42,123,456}_multi_target_20260421_*.pkl`
- 552 reports: `reports/daily_selection_ng1.5.0_stage4a/`
- Postmortem: `reports/ng150/stage4a_rejected.md`
- Memory: `memory/ng150_rejected.md`

**执行时间总结**: 2026-04-20 晚 P0 开始 → 2026-04-21 凌晨 P6 REJECTED. 共 ~8h (含 2.8h backfill + 3.6h 训练 + 25min Stage 4a 报告生成 + 评估)

---

## 真·零泄漏 forward OOS — ng1.0.1 单模 vs ng1.0.6 生产 MOE (2026-07-11)

迄今最干净的 OOS 检验。窗口 **2026-04-28 → 06-26 (N=40 交易日, 10d 前向)**，这段数据在模型训练时**物理上不存在**（production ng101 pkl 训于 2026-04-28 @ commit a19866b8，DB 数据当时止于 04-27），本次补齐 + 重建 0AMV/regime 后才可评估。属 forward paper-trade 口径，置信度最高。

**大环境**：A股全市场 10d 前向中位收益 = **−5.06%**（40天均值），是个下跌市。

| 指标 (Top-10, 统一A股中位基准, 覆盖100%) | ng1.0.1 裸信号 | ng1.0.6 生产MOE |
|---|---|---|
| Top-10 平均 10d 收益 | **−0.57%** | −2.97% |
| 超额 vs 中位 | **+4.49%** (85%天赢) | +2.09% (78%天赢) |
| 非重叠 10d 累计 | −8.96% | −16.13% |

**三条结论**:
1. **信号泛化成立**：两模型在未见数据上都大幅跑赢大盘（超额 +4.49% / +2.09%），未过拟合，alpha 真实。
2. **绝对收益亏**：大盘跌 5%，模型只是跌得少。熊市不该满仓做多，模型价值在"选相对最强票"非"预测大盘方向"。
3. **ng1.0.1 单模全面胜生产 MOE**：绝对/超额/胜率各维度都赢。用最新鲜 forward OOS 独立印证 in-sample（`memory/moe_failure_2026_04_25`）+ Pre-2020（`memory/pre2020_real_oos_2026_04_25`）+ 牛熊拆解（`memory/ng101_alpha_dominant_2026_04_25`）的既有结论，是第 4 个独立证据，支持"生产切 ng1.0.1 + 风控 overlay"。期间 MOE regime 探测器把 26/40 天判"牛市"（下跌市里），0AMV regime 反应偏慢。

**Caveat**: 40天短样本、只一个下跌 regime，非全周期定论。V5.2 评分卡两个都给 D 是**短窗口伪影**（需≥200天，年化/Sharpe/regime 指标全 ⚠短 归零），不可采信。close[D]→close[D+10] 口径与 selector "D+1买入" 差1天，但两模型同口径对比公平。

**复现**: `run_north_star_eval.py --backtest --report-dir <dir> --start-date 2026-04-28 --end-date 2026-06-26 --rank-field composite --top-n 10 --focus-days 10`；直接 A/B 脚本见 `memory/oos_fresh_2mo_2026_07_10.md`。

---

## 回填命令参考

```bash
# NG 1.0.1
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2020-01-01 --end-date 2026-04-03 --version ng1.0.1

# 训练
python3 ml_models/ng/ng_trainer.py --start-date 2020-01-01 --purge-days 15

# 报告
python3 backtest/batch_generate_v395_reports.py --version ng1.0.1
```

## 🆕 2026-07-11 新口径基线重跑 (口径 tag: 2026-07-11-p0fix)

北极星 3 个 P0 数据缺陷修复后 (基准 2022-24 NULL / 未复权 / -10% 惩罚不对称,
见 `reports/system_evaluation/选股系统与北极星系统评估与风控内化可行性研究_20260711.md`),
**本页以上所有旧口径数字不可与新跑数字混比**。新口径对齐窗口对比
(2018-11-02~2026-04-08, Top-10/10d/composite, 完整表见
`reports/system_evaluation/新口径基线重跑_20260711.md`):

| | ng1.0.1 单模 | ng1.0.6 生产MOE |
|---|---|---|
| V5.2 | **81.3% S** | 80.3% S |
| Sharpe / MaxDD | **2.550 / -17.7%** | 2.477 / -21.2% |
| Calmar / L6归因 | **5.85 / 100%** | 4.90 / 77% |
| Pre-2020 V5.2 / 净年化 | **49.7% B / +17.2%** | 42.9% C / +14.2% |

**重大订正**: 旧口径 "ng101 Pre-2020 净年化 -19%" 与 "ng106 唯一 Pre-2020 正年化"
双双失效 — 均为 -10% 惩罚跨期不对称的伪影。新口径下两者 Pre-2020 都正, ng101 更高。
这是第 5 个 (首个新口径) 证据支持生产切 ng1.0.1 单模; 切换前置: 评估-生产同构化复核
+ PINNED 注册表 + paper trade ≥20 日。

## 超参数 Sweep — 现行参数确认近似最优 (2026-07-13) ❌ 无免费收益

6 成员 ensemble 超参数 (V4.7.3 一套, v3.95 沿用至今) 首次系统性扫描, 三层验证:
1. 7 profile × fast-check (与 7-12 基线严格 paired): 仅 `lr001` (lr0.01+2000轮) 压线过
   预注册线 (10d ICIR +0.0503), 其余 6 个全灭; leaves63 near-miss (+0.044)
2. seed 123 配对复检: lr001 优势精确复现 (+0.0509) — 通过
3. **全量 4 窗口双 gate: lr001 与基线完全打平 (10d IC +0.0002/ICIR -0.0005) — FAIL, 不切换**

**方法论教训 (重要)**: fast-check 小窗口优势 (+0.05 ICIR, 跨 seed 复现!) 仍会在全量上
归零 — fast-check 只能杀方向, 不能定胜负 (与 ng1.2.3 教训同源)。
**同 schema 下不要再扫超参数**; 复扫时机 = 特征数 ±30% / 标签口径变更 / 训练数据翻倍。
机制保留: `ng_trainer --hp-profile X --fast-check` (~20-40min/配置,
`ml_models/training/hp_profiles.py`, pkl 元数据含 hp_profile)。
详见 `reports/system_evaluation/超参数sweep_20260713.md`。

## 相关页面

- [模型世代总览](evolution.md)
- [ML 管线](../architecture/ml-pipeline.md)
- [特征指南](../features/feature-guide.md)
- [北极星评估](../evaluation/north-star.md)
