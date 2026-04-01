# Autoresearch Program: 模型迭代优化

## Phase 1 结果 (组合构建参数优化, 已完成)
- 基线: 73.24 → 最优: **83.81** (+14.4%)
- 最优: focus_days=15, CPPI(F0.05,M20), score_floor=30, top_n=10
- 结论: 组合构建参数已达极限, 进一步突破需模型改进

## Phase 2: 快速代理训练 (当前)

### 目标
通过修改训练超参/特征/标签处理来提升模型信号质量

### 评估命令
```bash
python3 scripts/autoresearch_fast_train.py 2>/dev/null
```
- 输出一个数字 (v4_weighted_pct, 越高越好)
- 运行时间: 训练~30-45min + 报告~15min + 评估~5min = **~60min/轮**
- 使用 Phase 1 最优回测参数 (focus_days=15, CPPI, score_floor=30)

### 可修改的文件 (Scope)
```
scripts/autoresearch_train_params.json
```

### 参数搜索空间

#### 训练数据
| 参数 | 当前值 | 搜索范围 | 说明 |
|------|--------|----------|------|
| `start_date` | 2022-01-01 | 2020-01-01, 2021-01-01, 2022-01-01, 2023-01-01 | 训练数据起始 |
| `purge_days` | 15 | 10, 15, 20, 30 | 训练/验证间隔 |

#### 超参数 (影响模型复杂度)
| 参数 | 当前值 | 搜索范围 | 说明 |
|------|--------|----------|------|
| `num_leaves` | 31 | 15, 20, 25, 31, 40, 50 | 树复杂度 |
| `learning_rate` | 0.02 | 0.01, 0.02, 0.03, 0.05 | 学习率 |
| `min_data_in_leaf` | 200 | 100, 150, 200, 300, 400 | 叶子最小样本 |
| `reg_alpha` | 0.5 | 0.1, 0.3, 0.5, 1.0, 2.0 | L1正则化 |
| `reg_lambda` | 3.0 | 1.0, 2.0, 3.0, 5.0, 8.0 | L2正则化 |

#### 标签处理
| 参数 | 当前值 | 搜索范围 | 说明 |
|------|--------|----------|------|
| `sharpe_blend` | 0.30 | 0.0, 0.15, 0.25, 0.30, 0.40, 0.50 | Sharpe-Blend 比例 |

#### 特征裁剪
| 参数 | 当前值 | 搜索范围 | 说明 |
|------|--------|----------|------|
| `extra_prune_features` | [] | 从候选列表中选取 | 额外去掉的特征 |

**候选裁剪特征** (V4.7.5保留的50个中, 边缘贡献的):
```
roe, dv_ttm, turnover_rate_f, float_ratio,
return_skewness_proxy, bid_ask_spread_proxy,
industry_pe_ratio, industry_pb_ratio
```

### 约束
- 不要修改 `autoresearch_fast_train.py` (评估脚本)
- 不要修改 `autoresearch_params.json` (回测参数, Phase 1 已优化)
- 每次只改 1-2 个训练参数
- 训练耗时长, 建议先排除明显无效方向

## Phase 3: 全量训练验证

当 Phase 2 找到比 83.81 更高的配置后:
```bash
# 改 mode 为 full, start_date 为 2020-01-01
python3 scripts/autoresearch_fast_train.py --full
```
- 完整 Walk-Forward 训练 (~6h)
- 6年数据 (2020-2026)
- 如果全量分数 > 83.81, 则提升为生产模型
