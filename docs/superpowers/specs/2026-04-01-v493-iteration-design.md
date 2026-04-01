# V4.9.3 迭代设计规格 — 纯特征工程 + 浓度风险对策

> 日期: 2026-04-01
> 状态: 已批准, 待实施
> 基线: V4901 (V5.1=77.1% A+ 生产配置)
> 目标: V5.1 ≥ 82% S级

## 1. 迭代原则

V4902 失败教训: **不改训练配置(loss/权重/blend), 只做特征工程和正则化参数调整**。

## 2. 三项改动

### 改动A: 特征裁剪 (61→48特征)

**删除13个确认无效特征** (重要性<0.5%):

```python
V493_REMOVE_FEATURES = [
    'dv_ttm',               # 股息率, 与PB冗余
    'max_pct_change_5d',    # 被atr_percentile覆盖
    'cci_14',               # 被squeeze_mom覆盖
    'macd_hist',            # 被macd_dif/dea覆盖
    'brain_roll_spread',    # V4.8.4 BRAIN因子, 无贡献
    'vol_price_div',        # V4.8.1, 无贡献
    'return_skewness_proxy',
    'return_10d',           # 被return_20d覆盖
    'ma10_ratio',           # 被ma20_ratio覆盖
    'return_1d',            # 噪声太大
    'price_acceleration',
    'max_ret_20d',
    'avg_pct_change_5d',
]
```

### 改动B: 添加3个BRAIN因子 (48→51特征)

```python
V493_NEW_BRAIN_FACTORS = [
    'brain_vol_clustering',  # GARCH聚类波动率 → L3 CVaR/水下比
    'brain_tail_risk',       # 极端收益频率 → L3 CVaR/MaxDD
    'brain_ret_autocorr',    # 收益自相关 → L4 PBO/L3 Hurst
]
```

**因子构建** (在 `wqbrain_integration/brain_feature_importer.py` 或训练脚本中):

```python
# brain_vol_clustering: GARCH(1,1)条件波动率的20日滚动聚类指标
# = 条件波动率 / 无条件波动率 (>1表示波动聚集, <1表示平静期)
# 简化实现: ewm_vol_5d / rolling_vol_20d
brain_vol_clustering = close.pct_change().ewm(span=5).std() / close.pct_change().rolling(20).std()

# brain_tail_risk: 过去20日中|收益|>2σ的天数占比
returns = close.pct_change()
sigma = returns.rolling(60).std()
brain_tail_risk = (returns.abs() > 2 * sigma).rolling(20).mean()

# brain_ret_autocorr: 过去20日收益的1阶自相关
brain_ret_autocorr = returns.rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
```

### 改动C: 浓度风险对策 (训练超参数)

```python
# P0: 节点级特征采样 (最关键)
lgb_params['feature_fraction_bynode'] = 0.5     # 新增
xgb_params['colsample_bynode'] = 0.5            # 新增

# P1: 树级特征采样收紧
lgb_params['feature_fraction'] = 0.7             # was 0.8
xgb_params['colsample_bytree'] = 0.7            # was 0.8

# P2: 最小叶子样本增加
lgb_params['min_data_in_leaf'] = 400             # was 200

# P0: Ensemble权重上限 + ICIR收缩
ENSEMBLE_WEIGHT_CLIP = (0.10, 0.35)             # was (0.08, 0.50)
ICIR_SHRINKAGE = 0.3                             # 30%向等权收缩
```

## 3. 不改的东西 (V4901全部继承)

- TARGET_SHARPE_BLEND: {3d: 0.10, 5d: 0.25, 10d: 0.35, 15d: 0.35}
- 样本权重: V4.8.5标准 (涨跌停+极端+熊市+时间衰减)
- Q95 alpha=0.95 + LambdaRank trunc=10, grades=10
- WF配置: min_train=900, val=120, test=120, step=90, purge=15
- reg_alpha=0.1, reg_lambda=0.1, path_smooth=5
- 无Meta-Learner, 无Combined Isotonic

## 4. 实现架构

### 新建文件
- `ml_models/v39/v493_production_scorer.py` — 继承V4901, 模型目录指向v493

### 修改文件
- `ml_models/training/train_v395_multi_target.py` — 新增V493Trainer:
  - 覆盖特征裁剪列表 (V493_REMOVE_FEATURES)
  - 添加3个BRAIN因子计算
  - 覆盖LGB/XGB params (bynode + feature_fraction + min_data)
  - 覆盖ensemble权重逻辑 (clip_max=0.35 + shrinkage=0.3)
  - walk_forward_train保存为v493格式
- `backtest/batch_generate_v395_reports.py` — 注册v4.9.3

### 不修改
- north_star_metrics.py (V5/V5.1评分逻辑)
- backtest_report_based.py (回测流程)
- V4901的任何代码

## 5. 预期效果

| 改动 | 影响层 | 预期改善 |
|------|--------|---------|
| 删13无效特征 | L4 PBO | 减少噪声特征, 降低过拟合概率 |
| +brain_vol_clustering | L3 CVaR/水下比 | 高波动聚集期降低预测 → 减少尾部损失 |
| +brain_tail_risk | L3 CVaR/MaxDD | 识别高尾部风险个股 |
| +brain_ret_autocorr | L3 Hurst/L4 PBO | 判断趋势vs均值回复 |
| feature_fraction_bynode=0.5 | L4 PBO | atr_percentile浓度17.6%→8-12% |
| ensemble clip 0.35 | L4 PBO | 有效模型数1→3-4, 鲁棒性提升 |
| min_data_in_leaf=400 | L4 PBO | 减少细粒度过拟合 |

**预期V5.1**: 77.1% → **82-85%** (L3+5%, L4+5%, L7+2%)

## 6. 验证计划

```
阶段1: fast-check (~5min)
  python3 ml_models/training/train_v395_multi_target.py --v493 --fast-check
  验证: 10d IC仍>0.04, ICIR仍>0.5

阶段2: 完整训练 (~6h)
  python3 ml_models/training/train_v395_multi_target.py --v493 --start-date 2020-01-01

阶段3: 生成报告+V5.1评估
  python3 backtest/batch_generate_v395_reports.py --version v4.9.3
  python3 backtest/run_north_star_eval.py --backtest --report-dir reports/daily_selection_v493 \
      --score-version v51 --top-n 10 --focus-days 15 --production
```

## 7. 风险与回退

| 风险 | 概率 | 回退 |
|------|------|------|
| 新BRAIN因子IC为负 | 中 | 删除该因子, 用48特征版本 |
| feature_fraction_bynode导致IC下降>10% | 低 | 改回0.7或0.6 |
| ensemble clip过紧导致收益下降 | 中 | 放宽到0.40 |
| 3个新因子数据缺失 | 低 | 从daily_quotes实时计算, 无外部依赖 |
