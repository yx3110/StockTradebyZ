# NG v1.0.9 — 信号持久性优化 (特征半衰期过滤 + 平滑标签)

**Date**: 2026-04-10
**Base**: ng1.0.1 信号底座 (69特征, 行业超额标签)
**Target**: 10日调仓换手率从82%/次降到50%/次, 年化换手45x→20x以下
**Principle**: 预处理层面改进, 不改训练流程, 固定阈值无过拟合

## 背景

ng1.0.8实验发现: 持仓缓冲sell50+cost0.3%只能将换手从45x降到36x。根因是ng1.0.1的**排名不稳定** — 10天后Top-10中82%跌出Top-20。这是因为模型依赖快变特征(RSI/KDJ/volume_ratio等), 导致预测排名每天大幅变化。

## Part A: 特征半衰期过滤

### 原理

计算每个stock特征的**10日rank自相关系数**: 同一特征在日期t和t+10的cross-sectional rank的Spearman相关。

高自相关(>0.5): 特征变化慢, 排名稳定 → 保留
低自相关(<0.5): 特征变化快, 排名不稳定 → 过滤

### 实现

在ng_trainer.py的`walk_forward_train`中, load_data之后、WF训练之前:

```python
def _compute_feature_rank_autocorr(df, feature_cols, lag_days=10, min_autocorr=0.5):
    """计算每个特征的lag_days日rank自相关, 返回通过阈值的特征列表"""
    from scipy.stats import spearmanr
    dates = sorted(df['trade_date'].unique())
    passed = []
    for col in feature_cols:
        autocorrs = []
        for i in range(0, len(dates) - lag_days, lag_days):
            d1, d2 = dates[i], dates[i + lag_days]
            mask1 = df['trade_date'] == d1
            mask2 = df['trade_date'] == d2
            codes1 = set(df.loc[mask1, 'code'])
            codes2 = set(df.loc[mask2, 'code'])
            common = codes1 & codes2
            if len(common) < 100:
                continue
            v1 = df.loc[mask1].set_index('code').loc[list(common), col].values
            v2 = df.loc[mask2].set_index('code').loc[list(common), col].values
            valid = ~(np.isnan(v1) | np.isnan(v2))
            if valid.sum() < 100:
                continue
            corr, _ = spearmanr(v1[valid], v2[valid])
            if not np.isnan(corr):
                autocorrs.append(corr)
        if autocorrs:
            mean_autocorr = np.mean(autocorrs)
            if mean_autocorr >= min_autocorr:
                passed.append((col, mean_autocorr))
            else:
                logger.info(f"  FILTER {col}: rank_autocorr={mean_autocorr:.3f} < {min_autocorr}")
        else:
            passed.append((col, 0.5))  # 数据不足, 保留
    return passed
```

### 参数

| 参数 | 默认值 | 含义 |
|------|:------:|------|
| `lag_days` | 10 | 自相关滞后天数 (匹配focus_days) |
| `min_autocorr` | 0.5 | 最低rank自相关阈值 |

### 预期被过滤的特征 (待验证)

| 特征 | 预计autocorr | 过滤? |
|------|:---:|:---:|
| pe_ttm, pb, roe_ttm | 0.90+ | 保留 |
| industry_relative_strength | 0.70+ | 保留 |
| trend_strength_20d | 0.60+ | 保留 |
| volume_ratio_5d | 0.30-0.40 | 过滤 |
| rsi_14 | 0.25-0.35 | 过滤 |
| kdj_j_value | 0.15-0.25 | 过滤 |
| volume_breakout | 0.20-0.30 | 过滤 |

预计从56个stock特征过滤掉10-20个快变特征, 保留36-46个。

## Part B: 平滑标签

### 原理

当前: `label_10d = return(t → t+10)`

改为: `smoothed_label_10d = mean(return(t → t+10), return(t+1 → t+11), ..., return(t+4 → t+14))`

等效于: 预测"未来5天中任意入场的10天收益均值"。

### 为什么能提升持久性

- 当前标签只奖励精确在t日买入的收益。如果一只股票在t日是好的但t+2日不好, 模型可能学到只在t日有效的短暂pattern
- 平滑标签奖励在t到t+4任意日买入都好的股票。这种股票的alpha更持久, 预测排名也更稳定

### 实现

在ng_cache_updater.py的标签计算部分:

```python
# 现有: label_10d = future_return(t, t+10)
# 新增: smoothed_label_10d = mean([future_return(t+k, t+k+10) for k in range(smooth_window)])

SMOOTH_WINDOW = 5  # 平滑窗口 (天)

for sid in labels_all:
    for h in [3, 5, 10, 15]:
        raw_labels = []
        for k in range(SMOOTH_WINDOW):
            shifted_label = labels_shifted[sid].get(f'label_{h}d_shift{k}')
            if shifted_label is not None and not np.isnan(shifted_label):
                raw_labels.append(shifted_label)
        if raw_labels:
            labels_all[sid][f'smooth_label_{h}d'] = np.mean(raw_labels)
```

### 参数

| 参数 | 默认值 | 含义 |
|------|:------:|------|
| `smooth_window` | 5 | 标签平滑窗口 (天) |

## 文件变更

| 文件 | 改动 |
|------|------|
| `ml_models/ng/ng_trainer.py` | 添加 `_compute_feature_rank_autocorr()`, 训练前过滤快变特征 |
| `ml_models/ng/ng_cache_updater.py` | 计算shifted labels, 生成smooth_label_Xd |
| `ml_models/ng/ng_schema.py` | ng1.0.9表添加smooth_label列 |

**不改的文件**: ng_feature_calculator.py, ng_production_scorer.py

## 评估计划

### Step 1: 特征autocorr分析 (不训练)

先跑autocorr分析看哪些特征会被过滤:
```bash
python3 -c "
from ml_models.ng.ng_trainer import NGTrainer
trainer = NGTrainer(version='ng1.0.1')
df = trainer.load_data(start_date='2020-01-01')
# compute autocorr for all features...
"
```

### Step 2: Fast-check (A only)

只用特征过滤, 不改标签, 验证IC方向:
```bash
python3 ml_models/ng/ng_trainer.py --version ng1.0.9 --fast-check \
    --start-date 2020-01-01 --purge-days 15 --min-autocorr 0.5
```

### Step 3: Fast-check (A+B)

特征过滤 + 平滑标签:
```bash
python3 ml_models/ng/ng_trainer.py --version ng1.0.9 --fast-check \
    --start-date 2020-01-01 --purge-days 15 --min-autocorr 0.5 --smooth-label 5
```

### Step 4: 完整训练 + 报告 + 北极星

### Step 5: 换手率验证

生成报告后, 模拟10日调仓, 检查每次调仓的换手比例:
```bash
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_ng109 \
    --top-n 10 --focus-days 10 --rank-field composite \
    --buy-threshold 8 --sell-threshold 50 --cost-penalty 0.003
```

目标: 换手率 < 20x

### Step 6: Pre-2020验证

必须在Pre-2020-only报告上评估, 不能用全量目录(已知的--start-date bug)。

## 预期效果

| 指标 | ng1.0.1裸 | ng1.0.8(sell50) | ng1.0.9(预期) | 目标 |
|------|:---:|:---:|:---:|:---:|
| 每次调仓换手 | 82% | 72% | **50%** | <50% |
| 年化换手 | 45x | 36x | **~20x** | <20x |
| Sharpe | 2.37 | 2.52 | **2.5+** | >2.5 |
| 10d ICIR | 0.51 | 0.51 | **0.40+** | 可能略降 |

注意: 特征过滤可能降低裸IC(丢掉部分短期信号), 但排名稳定性提升带来的换手降低应该在净收益层面补回来。

## 版本号

**ng1.0.9** — 信号持久性 (特征半衰期过滤 + 平滑标签)
