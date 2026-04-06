# ML 管线

从特征工程到每日推荐的完整机器学习管线。

## 管线总览

```
Feature Cache ──→ 训练管线 ──→ 模型文件 ──→ 推理管线 ──→ 日报
      │               │            │            │           │
  69+特征/股     Walk-Forward    .pkl文件    Scorer评分    Top-10
  7000+股/天    LGB+XGB+CB+RF   ~60-100MB   EMA平滑     JSON报告
```

## 训练管线

### 入口
```bash
# NG 系列（推荐）
python3 ml_models/ng/ng_trainer.py --start-date 2020-01-01 --purge-days 15

# V4.9.0.1
python3 ml_models/training/train_v395_multi_target.py --v4901 --purge-days 15
```

### Walk-Forward 验证

训练采用 Walk-Forward（WF）方式避免数据泄露：

```
|-------- Train Window --------|-- Purge --|-- Test --|
|                               |  15 days  |  ~180d  |
      W1: 2020-01 → 2022-06     purge       2022-07 → 2022-12
      W2: 2020-01 → 2023-06     purge       2023-07 → 2023-12
      W3: 2020-01 → 2024-06     purge       2024-07 → 2024-12
      W4: 2020-01 → 2025-06     purge       2025-07 → 最新
```

- **Purge gap**: 15天，防止标签泄露到训练集
- **Auto-WF**: 自动 turbo-check 3种配置（expanding / sliding-720d / sliding-500d+decay730），按 10d ICIR 选最优
- **CLI**: `--no-auto-wf` 跳过自动选择，`--max-train-days N` 手动指定滑动窗口

### Ensemble 集成

每个 WF 窗口独立训练 5-6 个基模型：

| 模型 | 用途 |
|---|---|
| LightGBM | 主力模型，速度快 |
| XGBoost | 补充 LightGBM 的弱点 |
| CatBoost | 类别特征处理，鲁棒性好 |
| RandomForest | 降低过拟合风险 |
| HistGBM | sklearn 原生梯度提升 |
| LambdaRank | 排序学习（trunc=10），优化 top-10 排名 |

集成方式：ICIR 自适应加权（每个目标周期独立权重）

### 多目标预测

同时预测 4 个时间窗口的收益：
- `pred_3d`: 3日收益
- `pred_5d`: 5日收益
- `pred_10d`: 10日收益（主要排名依据）
- `pred_15d`: 15日收益

### 标签工程

| 版本 | 标签类型 | 说明 |
|---|---|---|
| NG 1.0.0 | 绝对收益 | 未来 N 日收益率 |
| NG 1.0.1 | 行业超额收益 | 收益率 - 所属行业平均收益 |
| NG 1.0.2 | 行业超额 + 下行风险 | 新增 downside_10d 目标 |
| NG 1.1.0 | 风格残差收益 | 去除 size/momentum/volatility 暴露 |
| V5.0 | 因子残差 | Rank-Transform + 因子残差，消除 β_UMD |

## 推理管线

### Scorer 流程
```python
# 1. 加载模型
scorer = NGProductionScorer(model_path="ng101_multi_target_*.pkl")

# 2. 从缓存读取当天特征
features = load_from_cache("ng101_feature_cache", date)

# 3. 多模型集成预测
pred_3d, pred_5d, pred_10d, pred_15d = ensemble_predict(features)

# 4. Composite 评分
composite = weighted_average(pred_3d, pred_5d, pred_10d, pred_15d)

# 5. EMA 平滑（降低换手）
smoothed = ema_smooth(composite, alpha=0.7)

# 6. 市场门控
final = market_gate(smoothed, gate_version="GateV2", auc=0.714)

# 7. 排名输出 Top-10
top10 = rank_and_filter(final, top_n=10, score_floor=30)
```

### 关键推理参数（V4.9.0.1 生产配置）

| 参数 | 值 | 说明 |
|---|---|---|
| top_n | 10 | Top-10 持仓 |
| focus_days | 15 | 15日调仓周期 |
| score_floor | 30 | 30分以下不入选 |
| ema_alpha | 0.7 | EMA 平滑系数 |
| retention_bonus | 0.2 | 20% 持仓保留加分 |
| CPPI floor | 8% | 风控底线 |
| CPPI multiplier | 20 | 杠杆倍数 |

## 模型文件管理

- 存放路径：`ml_models/trained_models/`
- 命名格式：`{version}_multi_target_{datetime}.pkl`
- 大小：60-100MB，**禁止 git add**
- 每个版本保留最新生产模型，旧版本可删除

## 双向评估（无泄露）

训练完成后自动执行两项评估：

1. **WF OOS（向前泛化）**: 模型能否预测未来
2. **Pre-2020（向后泛化）**: 模型学到的是通用规律还是过拟合

解读：两个都有 alpha → 高置信；只有一个 → 谨慎；都没有 → 模型有问题

详见 [北极星评估体系](../evaluation/north-star.md) 和 [回测方法论](../evaluation/backtesting.md)

## 相关页面

- [数据管线](data-pipeline.md)
- [模型演化史](../models/evolution.md)
- [特征指南](../features/feature-guide.md)
- [已知陷阱 — 模型训练类](../lessons/known-pitfalls.md#模型训练类)
