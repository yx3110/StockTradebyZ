# V3.9/3.95 旧版参考

V3.9 和 V3.95 是项目早期的 ML 评分系统，目前已被 NG 系列取代，但代码保留供参考。

## V3.8 增量学习 (deprecated)

- **特点**: 增量学习引擎，支持在线更新
- **路径**: `ml_models/v38/`、`incremental_learning/`
- **状态**: deprecated，保留模型文件供参考
- **为什么放弃**: 增量学习在金融场景中漂移检测不够可靠，不如定期全量重训

## V3.9 增强特征 Ensemble

- **特征**: 42 个增强特征 + 17 个扩展财务指标
- **Ensemble**: LightGBM + XGBoost + CatBoost + RF（4模型）
- **训练**: `ml_models/training/train_v390_from_cache.py`（30-60分钟）
- **Scorer**: `ml_models/v39/v390_production_scorer.py`
- **缓存表**: `v39_feature_cache`（7.1M+ 记录）

### V3.9 重训练 (2026-02-22)
- 6年数据(7M+记录, 2020-2026)
- 模型: `ml_models/trained_models/v390_full_from_cache.pkl`

## V3.95 多目标预测

- **改进**: 多目标预测（3d/5d/10d/15d），滚动训练窗口
- **特征**: 49个（V3.9 42 + 5个 daily_basic 新特征 + 2个改进）
- **Ensemble**: 5模型
- **训练**: `ml_models/training/train_v395_multi_target.py`（40-70分钟）
- **Scorer**: `ml_models/v39/v395_production_scorer.py`

### V3.95 截面改进实验 (2026-02-23)
- 第1轮(纯 Rank): 1d/3d 好但 5d+ 退化
- 第2轮(级联 Rank): 全面失败，噪声传播
- **第3轮(RobustZScore) ✅ 最优**: Robust Z-Score + 行业超额收益标签
  - ICIR: 10d=0.445, 15d=0.564（唯一全周期 ICIR>0.2）
  - 关键：robust_zscore 保留幅度信息(vs rank 丢失)

## V3.94

- V3.9 和 V3.95 之间的过渡版本
- Scorer: `ml_models/v39/v394_production_scorer.py`

## 导入方式

```python
from ml_models.v39.v390_production_scorer import V390ProductionScorer
from ml_models.v39.v395_production_scorer import V395ProductionScorer
```

## 相关页面

- [模型世代总览](evolution.md)
- [V4.x 实验总结](v4x-series.md)
