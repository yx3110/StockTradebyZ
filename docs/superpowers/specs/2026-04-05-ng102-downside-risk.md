# NG v1.0.2 设计：多目标下行风险预测

**日期**: 2026-04-05
**目标**: L3风控从47%提升至65%+，V5从70.1%→75%+

## 1. 问题分析

ng1.0.1 的V5=70.1% A+，最大短板是L3风控(47%)：

| L3子指标 | 当前值 | 目标 | 得分 | 差距 |
|----------|--------|------|------|------|
| Max Drawdown | -27.2% | -8.0% | 0.9/5 | -19pp |
| Worst 60-Day ICIR | -0.360 | +0.300 | 0.3/5 | +0.66 |
| CVaR 5% | 10.3% | 1.0% | 0.0/5 | -9.3pp |
| Underwater Time | 54.3% | 20% | 1.6/5 | -34pp |

根因：模型只预测收益方向，不识别下行风险。高分股票可能同时有高崩盘概率。

## 2. 方案：多目标下行风险预测

### 2.1 新增预测目标

在现有4个收益目标(3d/5d/10d/15d)基础上，新增1个下行风险目标：

```python
downside_10d = max(0, -excess_return_10d)
```

- 股票跌了 → downside > 0（模型学习"这只股票容易崩"）
- 股票涨了 → downside = 0（无惩罚信号）
- 使用行业超额收益（与v1.0.1一致），不是绝对收益

为什么选10d而非其他horizon：
- 10d是v1.0.1的最强ICIR目标(0.931)，与之对齐
- 下行风险在中期(10d)比短期(3d)更有预测性（噪声更低）
- 与持仓周期(focus_days=10)匹配

### 2.2 训练架构

```
现有V485 Ensemble (6模型 × 4目标 = 24个子模型)
  ↓ 不改动
新增: 6模型 × 1目标(downside_10d) = 6个子模型
  ↓
总计: 30个子模型
```

downside_10d使用与其他目标相同的训练pipeline：
- LightGBM + XGBoost + CatBoost + RF + HGB + LambdaRank
- Walk-Forward窗口、purge-days、ICIR裁剪全部沿用
- 特征集不变（69个因子）

### 2.3 风险折扣评分

Scorer的composite评分从：
```python
# v1.0.1
combined = w_3d * pred_3d + w_5d * pred_5d + w_10d * pred_10d + w_15d * pred_15d
```

改为：
```python
# v1.0.2
return_score = w_3d * pred_3d + w_5d * pred_5d + w_10d * pred_10d + w_15d * pred_15d
risk_discount = lambda_risk * pred_downside_10d
combined = return_score - risk_discount
```

- `lambda_risk` 初始值0.5，通过fast-check调参(0.3/0.5/0.7/1.0)
- 效果：高收益但高崩盘概率的股票被降权，低风险的稳健股票升权

### 2.4 可选叠加：Asymmetric Loss

对收益预测目标(3d/5d/10d/15d)的LightGBM子模型，使用不对称MSE损失：

```python
def asymmetric_mse(pred, actual):
    error = pred - actual
    if pred > 0 and actual < 0:  # false positive: 预测涨但实际跌
        return 2.5 * error^2     # 2.5倍惩罚
    else:
        return error^2           # 正常MSE
```

- 只应用于LightGBM（XGBoost/CatBoost保持默认loss，保持多样性）
- 不改变downside_10d目标的loss（它本身就是下行预测）

## 3. 文件变更清单

| 文件 | 改动 |
|------|------|
| `ml_models/ng/ng_schema.py` | 添加ng102_feature_cache到VERSION_TABLE_MAP |
| `ml_models/ng/ng_cache_updater.py` | 计算downside_10d标签，写入新表 |
| `ml_models/ng/ng_trainer.py` | 新增downside_10d训练目标，asymmetric loss可选 |
| `ml_models/ng/ng_production_scorer.py` | 风险折扣评分逻辑 |
| `ml_models/ng/__init__.py` | 版本号更新 |
| `backtest/batch_generate_v395_reports.py` | 添加ng1.0.2 version支持 |

## 4. 不改动的部分

- 特征集（69个因子，与v1.0.1完全相同）
- 行业超额标签计算逻辑
- ICIR自适应权重机制
- 缓存表schema（列结构不变，downside标签复用label字段或加新列）
- WF训练窗口参数

## 5. 缓存策略

- 新表 `ng102_feature_cache`
- 与ng101的差异仅在标签：ng102多一列 `downside_10d`
- 特征部分(features_json)与ng101完全相同
- 可以考虑：ng102直接从ng101复制features_json，只重算labels，节省回填时间

实际实现：在schema中新增 `downside_10d REAL` 列到ng102表。

## 6. 验证计划

1. **fast-check**: 2个WF窗口，验证downside_10d的IC为正（预期0.05-0.10）
2. **lambda调参**: fast-check中用0.3/0.5/0.7/1.0，选V5最优的lambda
3. **asymmetric loss A/B**: 有/无asymmetric loss各做一次fast-check
4. **完整训练**: 最优配置全量WF训练
5. **V5对比**: ng1.0.2 vs ng1.0.1，同数据公平对比

## 7. 预期收益

| 指标 | v1.0.1 | v1.0.2预期 | 依据 |
|------|--------|-----------|------|
| L3风控 | 47% | 60-70% | MaxDD改善30-50%(文献), CVaR大幅下降 |
| V5加权 | 70.1% | 73-76% | L3×20%权重贡献+3-5pp |
| MaxDD | -27.2% | -15~-18% | 高风险股被折扣排除 |
| 10d ICIR | 0.931 | 0.85-0.95 | 可能略降(保守偏移),但ICIR仍强 |

保守估计V5达73%，乐观可达76%。
