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

## NG 1.1.0 — 资金流 + 残差标签 (2026-04-05, 开发中)

**升级动机**: 1.0.2 的 V5.2=74.0%，目标突破 76%+

**三方向联合升级**:

### 方向1：信号质量
- 8 资金流因子（moneyflow）: 主力净流入比、大单占比等
- 8 交互因子: 通过 IC 筛选保留有效交互
- 新表 `moneyflow_daily` 存储日资金流数据

### 方向2：标签工程
- 风格因子残差标签：去除 size / momentum / volatility 暴露
- 目标：让模型学习纯 alpha 信号而非风格β

### 方向3：训练框架
- 6-8 个 WF 窗口（vs 之前 4 个）
- 市况样本加权：牛市 0.8 / 震荡 1.0 / 熊市 1.2
- CLI 开关独立控制每个方向

**CLI 开关**:
```bash
--enable-moneyflow      # 启用资金流因子
--enable-interaction    # 启用交互因子
--residual-label        # 启用风格残差标签
--wf-windows N          # WF窗口数
--regime-weight         # 启用市况加权
```

**当前状态**: 代码完成（Tasks 1-7），moneyflow 回填中，待 fast-check 验证

**初步结果**: V5.2 = 69.6% A（残差标签 ICIR +14.6%，但整体低于 1.0.2）

**缓存表**: `ng110_feature_cache`

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
