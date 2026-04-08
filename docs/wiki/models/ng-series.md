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

## NG 版本综合排名 (2026-04-08)

统一评估配置：Top-10, focus_days=10, CPPI(F0.08, M20)

### WF-OOS (2020-2026, 含训练期数据泄露)

| 排名 | 版本 | V5.2 | 年化(毛) | Sharpe | MaxDD | 换手 |
|:---:|:-----|:---:|:---:|:---:|:---:|:---:|
| 1 | **ng1.0.1** | **78.9% A+** | **72.2%** | **2.339** | **-12.6%** | 24x |
| 2 | ng1.0.4-5seed | 75.9% A+ | 47.4% | 1.611 | -16.6% | 24x |
| 3 | ng1.0.2-3seed | 76.2% A+ | 82.5% | 1.501 | -15.8% | — |
| 4 | ng1.0.3 | 69.9% A | 30.4% | 1.163 | -21.9% | 24x |

### Pre-2020 OOS (2018-2020, 无泄露)

| 排名 | 版本 | 年化(毛) | Sharpe | MaxDD |
|:---:|:-----|:---:|:---:|:---:|
| 1 | **ng1.0.3** | **+39.7%** | **1.190** | **-15.5%** |
| 2 | ng1.0.2 | -3.1% | -0.198 | -42.3% |
| — | ng1.0.1 | 未评估 | — | — |
| — | ng1.0.4 | 未评估 | — | — |

### ng1.0.4 CPPI 参数网格 (WF-OOS)

| CPPI配置 | V5.2 | 年化(毛) | Sharpe | MaxDD |
|:---------|:---:|:---:|:---:|:---:|
| F0.05,M15 | 78.0% | 34.1% | 1.474 | -19.7% |
| F0.05,M20 | 79.0% | 44.9% | 1.621 | -16.6% |
| **F0.08,M15** | **78.5%** | **47.3%** | **1.626** | **-16.3%** |
| F0.08,M20 | 78.3% | 47.4% | 1.611 | -16.6% |
| F0.10,M20 | 78.2% | 47.9% | 1.567 | -15.7% |
| F0.12,M15 | 78.4% | 48.7% | 1.549 | -16.9% |

### 核心矛盾

- **WF-OOS 最优**(ng1.0.1): 含 3 个 IC 翻转因子，Pre-2020 可能退化
- **Pre-2020 最优**(ng1.0.3): WF-OOS 表现最差，信号质量弱于 1.0.1
- **当前生产**: ng1.0.1 + CPPI(F0.08, M20)（V5.2=78.9%）

### 待解决问题

1. MaxDD -12.6% 仍未达 <10% 目标
2. 换手率 24x 仍然较高（目标 <15x）
3. ng1.0.1 / ng1.0.4 缺少 Pre-2020 验证
4. 需要找到 WF-OOS 和 Pre-2020 双向都强的模型配置

---

## NG 1.1.0 — 已废弃

ng1.1.0 的三方向实验（资金流因子、残差标签、WF框架升级）评估后发现仅"去翻转因子"方向有效，已合并为 ng1.0.3。其余方向（残差标签=无效、WF8+regime=无效、moneyflow=10d退步）废弃。

**数据资产保留**：`ng103_feature_cache`（3.6M行）、`moneyflow_daily`（8.8M行）由 ng1.0.3 继续使用。

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

## 相关页面

- [模型世代总览](evolution.md)
- [ML 管线](../architecture/ml-pipeline.md)
- [特征指南](../features/feature-guide.md)
- [北极星评估](../evaluation/north-star.md)
