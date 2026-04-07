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
