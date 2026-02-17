# V3.9改进策略 - 时间衰减采样 + 完整评估体系

## 🎯 核心理念

**"近期数据 > 历史数据"** - 股市规律在变化，越近的数据越有价值

---

## 📊 Part 1: 时间衰减采样策略

### 1.1 采样原则

```
重要性权重：
┌─────────────────────────────────────┐
│  最近3个月   │ 100% 密集采样        │
│  3-6个月     │  70% 采样率          │
│  6-12个月    │  40% 采样率          │
│  12个月以上  │  不使用              │
└─────────────────────────────────────┘
```

**具体方案**：
- **最近3个月（2025-08 ~ 2025-11）**：每个交易日 × 1000只股票 = 约60,000样本
- **3-6个月前（2025-05 ~ 2025-07）**：每隔1天采样 × 1000只股票 = 约30,000样本
- **6-12个月前（2024-11 ~ 2025-04）**：每隔2天采样 × 1000只股票 = 约60,000样本

**总样本量：约150,000样本**

### 1.2 为什么这样设计？

| 时期 | 采样率 | 理由 |
|------|--------|------|
| **最近3个月** | 100% | 市场环境最相关，规律最稳定 |
| **3-6个月** | 70% | 仍有参考价值，但开始衰减 |
| **6-12个月** | 40% | 提供多样性，但权重降低 |
| **12个月+** | 0% | 市场环境已变，规律失效 |

### 1.3 实现代码

```python
def get_time_weighted_dates(all_dates, cutoff_recent=90, cutoff_medium=180):
    """
    时间衰减采样交易日

    Args:
        all_dates: 所有交易日列表（倒序）
        cutoff_recent: 最近N天（100%采样）
        cutoff_medium: 中期N天（70%采样）

    Returns:
        selected_dates: 采样后的交易日列表
    """
    from datetime import datetime, timedelta

    today = datetime.now()
    selected = []

    for i, date in enumerate(all_dates):
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        days_ago = (today - date_obj).days

        if days_ago <= cutoff_recent:
            # 最近3个月：100%采样
            selected.append(date)
        elif days_ago <= cutoff_medium:
            # 3-6个月：70%采样（每3天取2天）
            if i % 3 != 2:
                selected.append(date)
        elif days_ago <= 365:
            # 6-12个月：40%采样（每5天取2天）
            if i % 5 < 2:
                selected.append(date)
        # else: 超过1年不采样

    return selected
```

---

## 📈 Part 2: 完整评估指标体系

### 2.1 预测准确性指标（基础）

#### 1. 方向准确率（Direction Accuracy）⭐⭐⭐⭐⭐
**定义**：预测涨跌方向与实际方向一致的比例

```python
direction_accuracy = np.mean((y_pred > 0) == (y_true > 0))
```

**判断标准**：
- < 50%：比抛硬币还差 ❌
- 50-60%：勉强可用 🟡
- 60-70%：良好 ✅
- 70-80%：优秀 ⭐
- > 80%：卓越（或过拟合）⚠️

**重要性**：**最重要！** 交易中方向比幅度更关键

---

#### 2. IC（信息系数）Information Coefficient ⭐⭐⭐⭐⭐
**定义**：预测值与实际值的Spearman秩相关系数

```python
from scipy.stats import spearmanr
ic = spearmanr(y_pred, y_true)[0]
```

**判断标准**：
- IC < 0.03：几乎没用 ❌
- IC = 0.03-0.05：可用 🟡
- IC = 0.05-0.08：良好 ✅
- IC = 0.08-0.10：优秀 ⭐
- IC > 0.10：卓越 ⭐⭐

**重要性**：量化基金常用，衡量排序能力

---

#### 3. R²（决定系数）⭐⭐⭐
**定义**：模型解释方差的比例

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

**判断标准**（A股）：
- R² < 0.1：很弱 ❌
- R² = 0.1-0.2：勉强 🟡
- R² = 0.2-0.3：良好 ✅
- R² = 0.3-0.4：优秀 ⭐
- R² > 0.5：可疑（可能过拟合）⚠️

**注意**：A股R²很难超过0.4，不要过度追求

---

#### 4. MSE/MAE/RMSE ⭐⭐
**定义**：预测误差的度量

```python
mse = np.mean((y_pred - y_true)**2)
mae = np.mean(np.abs(y_pred - y_true))
rmse = np.sqrt(mse)
```

**判断标准**（5日收益率预测）：
- MAE < 2%：很准 ⭐
- MAE = 2-3%：良好 ✅
- MAE = 3-5%：可接受 🟡
- MAE > 5%：太差 ❌

---

### 2.2 金融专用指标（实战）

#### 5. Top N 准确率 ⭐⭐⭐⭐⭐
**定义**：预测排名前N的股票，实际表现如何

```python
def top_n_accuracy(y_pred, y_true, n=20):
    """
    Top N股票的平均收益率
    """
    top_n_idx = np.argsort(y_pred)[-n:]
    return y_true[top_n_idx].mean()
```

**判断标准**：
- Top 20平均收益 < 0%：选股失败 ❌
- Top 20平均收益 = 0-2%：勉强 🟡
- Top 20平均收益 = 2-5%：良好 ✅
- Top 20平均收益 > 5%：优秀 ⭐

**重要性**：**最接近实战！** 直接反映选股能力

---

#### 6. 分位数性能分析 ⭐⭐⭐⭐
**定义**：不同预测分位数的实际表现

```python
def quantile_performance(y_pred, y_true, n_quantiles=5):
    """
    将预测分成N个分位数，看每个分位数的实际表现
    """
    quantiles = pd.qcut(y_pred, q=n_quantiles, labels=False)
    results = []
    for q in range(n_quantiles):
        mask = (quantiles == q)
        results.append({
            'quantile': q,
            'mean_return': y_true[mask].mean(),
            'win_rate': (y_true[mask] > 0).mean()
        })
    return pd.DataFrame(results)
```

**判断标准**：
- 单调递增（Q1<Q2<Q3<Q4<Q5）：优秀 ⭐
- 基本递增但有波动：良好 ✅
- 随机分布：无效 ❌

---

#### 7. 胜率（Win Rate）⭐⭐⭐⭐
**定义**：按模型预测买入，赚钱的概率

```python
# 只买预测涨幅>阈值的股票
threshold = np.percentile(y_pred, 80)  # Top 20%
buy_mask = (y_pred > threshold)
win_rate = (y_true[buy_mask] > 0).mean()
```

**判断标准**：
- 胜率 < 50%：不如抛硬币 ❌
- 胜率 = 50-60%：勉强 🟡
- 胜率 = 60-70%：良好 ✅
- 胜率 > 70%：优秀 ⭐

---

#### 8. 盈亏比（Profit Factor）⭐⭐⭐
**定义**：总盈利 / 总亏损

```python
buy_mask = (y_pred > threshold)
profits = y_true[buy_mask & (y_true > 0)].sum()
losses = -y_true[buy_mask & (y_true < 0)].sum()
profit_factor = profits / losses if losses > 0 else np.inf
```

**判断标准**：
- 盈亏比 < 1.0：亏钱 ❌
- 盈亏比 = 1.0-1.5：勉强盈利 🟡
- 盈亏比 = 1.5-2.0：良好 ✅
- 盈亏比 > 2.0：优秀 ⭐

---

### 2.3 模型健康度指标

#### 9. Learning Curve 收敛性 ⭐⭐⭐⭐
**检查项**：
```python
def check_convergence(train_scores, val_scores):
    """
    检查学习曲线收敛性

    健康模型：
    1. 训练误差和验证误差都下降
    2. 最终两者差距小（< 20%）
    3. 验证误差稳定（不再大幅波动）
    """
    # 1. 验证集是否下降
    val_improving = val_scores[-1] < val_scores[0]

    # 2. 过拟合程度
    gap = (val_scores[-1] - train_scores[-1]) / train_scores[-1]
    overfitting = (gap < 0.2)  # 差距<20%

    # 3. 稳定性
    recent_std = np.std(val_scores[-3:])
    stable = (recent_std < 0.01)

    return {
        'val_improving': val_improving,
        'not_overfitting': overfitting,
        'stable': stable,
        'healthy': val_improving and overfitting and stable
    }
```

---

#### 10. 预测分布 vs 实际分布 ⭐⭐⭐
**KL散度检查**：

```python
from scipy.stats import entropy

def distribution_match(y_pred, y_true, bins=20):
    """
    检查预测分布是否匹配实际分布
    """
    hist_pred, _ = np.histogram(y_pred, bins=bins, density=True)
    hist_true, _ = np.histogram(y_true, bins=bins, density=True)

    # 避免0
    hist_pred = hist_pred + 1e-10
    hist_true = hist_true + 1e-10

    kl_div = entropy(hist_true, hist_pred)

    return {
        'kl_divergence': kl_div,
        'match_quality': 'good' if kl_div < 0.5 else 'poor'
    }
```

---

#### 11. 残差正态性检验 ⭐⭐
**Shapiro-Wilk检验**：

```python
from scipy.stats import shapiro

def check_residual_normality(y_pred, y_true):
    """
    检验残差是否服从正态分布
    健康模型的残差应该接近正态分布
    """
    residuals = y_true - y_pred
    stat, p_value = shapiro(residuals[:5000])  # 最多5000样本

    return {
        'statistic': stat,
        'p_value': p_value,
        'is_normal': p_value > 0.05,
        'interpretation': 'Normal' if p_value > 0.05 else 'Non-normal'
    }
```

---

### 2.4 综合评分卡

```python
def comprehensive_evaluation(y_pred, y_true):
    """
    综合评估模型

    Returns:
        dict: 包含所有指标的评估结果
    """
    from scipy.stats import spearmanr, shapiro

    # 1. 基础准确性
    direction_acc = np.mean((y_pred > 0) == (y_true > 0))
    ic = spearmanr(y_pred, y_true)[0]
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    mae = np.mean(np.abs(y_pred - y_true))

    # 2. 金融指标
    # Top 20平均收益
    top_20_idx = np.argsort(y_pred)[-20:]
    top_20_return = y_true[top_20_idx].mean()

    # 分位数分析
    quantiles = pd.qcut(y_pred, q=5, labels=False, duplicates='drop')
    q_means = [y_true[quantiles == q].mean() for q in range(5) if (quantiles == q).any()]
    monotonic = all(q_means[i] <= q_means[i+1] for i in range(len(q_means)-1))

    # 胜率
    threshold = np.percentile(y_pred, 80)
    buy_mask = (y_pred > threshold)
    win_rate = (y_true[buy_mask] > 0).mean()

    # 盈亏比
    profits = y_true[buy_mask & (y_true > 0)].sum()
    losses = -y_true[buy_mask & (y_true < 0)].sum()
    profit_factor = profits / losses if losses > 0 else np.inf

    # 3. 分数计算
    scores = {
        'direction_accuracy': direction_acc,
        'ic': ic,
        'r2': r2,
        'mae': mae,
        'top_20_return': top_20_return,
        'quantile_monotonic': monotonic,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

    # 4. 综合评分（0-100分）
    final_score = (
        direction_acc * 30 +  # 方向准确率（30分）
        min(ic / 0.10, 1.0) * 25 +  # IC（25分）
        min(r2 / 0.40, 1.0) * 15 +  # R²（15分）
        min(top_20_return / 0.05, 1.0) * 20 +  # Top20收益（20分）
        (10 if monotonic else 0)  # 分位数单调性（10分）
    )

    scores['final_score'] = final_score
    scores['grade'] = (
        'A' if final_score >= 80 else
        'B' if final_score >= 70 else
        'C' if final_score >= 60 else
        'D' if final_score >= 50 else
        'F'
    )

    return scores


def print_evaluation_report(scores):
    """
    打印评估报告
    """
    print("="*80)
    print("📊 V3.9模型综合评估报告")
    print("="*80)
    print()
    print("【基础准确性指标】")
    print(f"  方向准确率:    {scores['direction_accuracy']*100:.2f}%  " +
          ("✅" if scores['direction_accuracy'] > 0.70 else "🟡" if scores['direction_accuracy'] > 0.60 else "❌"))
    print(f"  IC (信息系数):  {scores['ic']:.4f}         " +
          ("✅" if scores['ic'] > 0.05 else "🟡" if scores['ic'] > 0.03 else "❌"))
    print(f"  R²:            {scores['r2']:.4f}         " +
          ("✅" if scores['r2'] > 0.20 else "🟡" if scores['r2'] > 0.10 else "❌"))
    print(f"  MAE:           {scores['mae']:.4f} ({scores['mae']*100:.2f}%)  " +
          ("✅" if scores['mae'] < 0.03 else "🟡" if scores['mae'] < 0.05 else "❌"))
    print()
    print("【金融实战指标】")
    print(f"  Top 20平均收益: {scores['top_20_return']*100:.2f}%    " +
          ("✅" if scores['top_20_return'] > 0.02 else "🟡" if scores['top_20_return'] > 0 else "❌"))
    print(f"  分位数单调性:   {'是' if scores['quantile_monotonic'] else '否'}           " +
          ("✅" if scores['quantile_monotonic'] else "❌"))
    print(f"  胜率:          {scores['win_rate']*100:.2f}%      " +
          ("✅" if scores['win_rate'] > 0.60 else "🟡" if scores['win_rate'] > 0.50 else "❌"))
    print(f"  盈亏比:         {scores['profit_factor']:.2f}         " +
          ("✅" if scores['profit_factor'] > 1.5 else "🟡" if scores['profit_factor'] > 1.0 else "❌"))
    print()
    print("="*80)
    print(f"📈 综合评分: {scores['final_score']:.1f}/100  等级: {scores['grade']}")
    print("="*80)
    print()
    print("【评级标准】")
    print("  A (80-100): 卓越，可以实盘")
    print("  B (70-79):  优秀，谨慎实盘")
    print("  C (60-69):  良好，模拟盘测试")
    print("  D (50-59):  及格，继续改进")
    print("  F (<50):    不及格，重新训练")
    print("="*80)
```

---

## 🎯 Part 3: 完整实施方案

### 3.1 数据准备（时间衰减采样）

```bash
# 使用改进的预计算脚本
python3 precompute_v39_features_time_weighted.py \
  --start-date 2024-11-01 \
  --end-date 2025-11-15 \
  --sample-stocks 1000 \
  --time-decay-strategy recent_dense
```

### 3.2 模型训练（含完整评估）

```bash
# 使用评估版训练脚本
python3 train_v390_with_full_evaluation.py \
  --test-size 0.2 \
  --random-state 42 \
  --enable-all-metrics
```

### 3.3 验收标准

**只有通过以下标准才能实盘**：

| 指标 | 最低标准 | 目标标准 |
|------|---------|---------|
| **方向准确率** | > 65% | > 70% |
| **IC** | > 0.03 | > 0.05 |
| **Top 20收益** | > 1% | > 3% |
| **胜率** | > 55% | > 60% |
| **盈亏比** | > 1.2 | > 1.5 |
| **综合评分** | > 60分(C) | > 70分(B) |

---

## 📋 快速检查清单

训练完成后，问自己：

- [ ] 方向准确率 > 70%？
- [ ] IC > 0.05？
- [ ] Top 20股票平均收益 > 2%？
- [ ] 分位数单调递增？
- [ ] 胜率 > 60%？
- [ ] Learning curve收敛且无过拟合？
- [ ] 综合评分 ≥ 70分（B级）？

**只有全部打勾，才能实盘！**

---

*这才是专业量化团队的评估标准*
*不要因为模型训练完成就盲目使用*
*数据决定上限，指标决定下限*
