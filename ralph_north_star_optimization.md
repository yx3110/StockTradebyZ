# Ralph Loop Prompt: 北极星满分迭代优化

## 使用方法

```bash
/ralph-loop "$(cat ralph_north_star_optimization.md)" --max-iterations 30 --completion-promise "NORTH_STAR_PERFECT"
```

---

## 任务目标

你是一个量化选股模型迭代优化专家。你的唯一目标是：**让北极星V2评分达到满分 105/105**。

当前最佳模型是 V4.5 (CPPI)，得分 85/105 (S级)。你需要在此基础上不断迭代，每轮改进一个或多个薄弱指标，直到所有21项指标全部达到 ★★★★★ (5分)。

**当且仅当北极星V2评分达到 105/105 时**，输出:
```
<promise>NORTH_STAR_PERFECT</promise>
```

---

## 核心约束 (违反任何一条立即停止)

1. **绝不删除或覆盖数据库** `data_adapter/stock_data.db`
2. **绝不删除现有模型文件**，只创建新版本
3. **绝不修改北极星评分标准** (`north_star_metrics.py` 中的阈值不能改)
4. **绝不修改回测引擎的交易成本/涨停检测逻辑** (这些是真实约束)
5. **每次迭代必须先跑评估再改代码**，用数据驱动决策
6. **所有改动必须 git commit** 并包含北极星分数变化

---

## 北极星V2 满分标准 (21项 × 5分 = 105分)

### Layer 1: 信号质量 (30分)
| 指标 | 满分阈值 | 方向 |
|------|----------|------|
| Daily IC | ≥ 0.08 | 越高越好 |
| ICIR | ≥ 0.70 | 越高越好 |
| IC>0% | ≥ 68% | 越高越好 |
| IC单调性 | ≥ 4.5 | 越高越好 |
| IC稳定性(CV) | ≤ 0.6 | 越低越好 |
| 信号半衰期 | ≥ 20天 | 越高越好 |

### Layer 2: 组合效率 (25分)
| 指标 | 满分阈值 | 方向 |
|------|----------|------|
| 年化换手 | ≤ 20x | 越低越好 |
| 年化成本 | ≤ 5% | 越低越好 |
| 净/毛收益比 | ≥ 0.85 | 越高越好 |
| 涨停失败率 | ≤ 2% | 越低越好 |
| 流动性覆盖 | ≥ 95% | 越高越好 |

### Layer 3: 风险控制 (25分)
| 指标 | 满分阈值 | 方向 |
|------|----------|------|
| 最大回撤 | ≥ -8% | 越小越好 |
| Sharpe | ≥ 3.0 | 越高越好 |
| Sortino | ≥ 4.0 | 越高越好 |
| Calmar | ≥ 4.0 | 越高越好 |
| 最差60日ICIR | ≥ 0.30 | 越高越好 |

### Layer 4: 盈利与鲁棒性 (25分)
| 指标 | 满分阈值 | 方向 |
|------|----------|------|
| 年化收益 | ≥ 50% | 越高越好 |
| 月度胜率 | ≥ 83% | 越高越好 |
| 前后半段一致性 | ≥ 0.80 | 越高越好 |
| 市值均衡度 | ≥ 0.80 | 越高越好 |
| 中位市值(亿) | ≥ 100亿 | 越高越好 |

---

## 每轮迭代流程 (严格遵循)

### Step 1: 评估当前状态

先读取迭代日志，了解上一轮做了什么，分数变化如何：
```bash
cat ralph_iteration_log.md 2>/dev/null || echo "首次迭代"
```

然后运行北极星评估获取当前分数。使用最新的报告目录：
```bash
# 确定当前最新版本的报告目录
ls -d reports/daily_selection_v* | sort -V | tail -5

# 运行北极星V2评估 (使用扩展窗口获取可靠评估)
python3 backtest/run_north_star_eval.py --extended \
    --report-dir reports/daily_selection_v4.4_v2 \
    --extended-dir reports/daily_selection_v4.4_v2_extended \
    --label "当前版本" --top-n 10 --focus-days 10
```

如果需要带CPPI的评估：
```bash
python3 scripts/v45_param_search.py
```

### Step 2: 分析薄弱环节

从评估结果中识别：
1. **得分 < 5 的指标**（按差距从大到小排列）
2. **哪个Layer总分最低**
3. **指标间是否有冲突**（如提高收益可能增大回撤）

创建改进优先级列表：
- 高ROI改进 = 差1-2档就能升级的指标
- 低ROI改进 = 差很多档的指标（需要根本性改变）

### Step 3: 设计改进方案

根据薄弱指标选择合适的改进手段：

**信号质量改进手段:**
- 增加新特征（宏观、行业轮动、资金流向）
- 改进标签构造（Sharpe-blend比例、行业超额方式）
- 优化ensemble权重（ICIR加权替代IC加权）
- 训练数据窗口调整

**组合效率改进手段:**
- 调整选股过滤（涨停预判、流动性阈值）
- 优化换手控制（持仓延续奖励、相邻日重叠度）
- 降低交易频率（信号衰减、持仓周期延长）

**风险控制改进手段:**
- CPPI参数微调（floor、multiplier、rebalance频率）
- 波动率目标（vol_target参数）
- 行业集中度限制
- 动态仓位管理

**盈利与鲁棒性改进手段:**
- 市值均衡约束（大/中/小盘配比）
- 滚动训练窗口（walk-forward）
- 多市场状态适应（牛/熊/震荡分别处理）
- Isotonic回归校准

### Step 4: 实施改进

**改进代码的位置:**
- 训练逻辑: `ml_models/training/train_v395_multi_target.py`
- 生产评分器: `ml_models/v39/v44_production_scorer.py` 或 `v46_production_scorer.py`
- 回测参数: `scripts/v45_param_search.py`
- 选股过滤: `tomorrow_stock_selector.py`

**创建新版本而非修改旧版本。** 例如 V4.7、V4.8...

### Step 5: 训练新模型

```bash
# 基于V4.4训练器训练 (当前推荐基线)
python3 ml_models/training/train_v395_multi_target.py --v44

# 或基于V4.6训练器 (包含meta-learner等增强)
python3 ml_models/training/train_v395_multi_target.py --v46

# 训练完成后模型保存在 ml_models/trained_models/v44/ 或 v46/
```

### Step 6: 生成报告

```bash
# 批量生成选股报告 (约4分钟/300天)
python3 backtest/batch_generate_v395_reports.py \
    --version v4.4 \
    --start-date 2024-01-01 --end-date 2026-02-13 \
    --output-dir reports/daily_selection_v新版本号
```

### Step 7: 评估并记录

```bash
# 运行北极星评估
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v新版本号 \
    --label "V新版本号" --top-n 10 --focus-days 10

# 如果结果好，尝试CPPI参数搜索进一步优化
python3 scripts/v45_param_search.py
```

### Step 8: 记录迭代日志

**每轮迭代结束后必须更新** `ralph_iteration_log.md`:

```markdown
## 迭代 N (日期时间)
- **改进内容**: 简述做了什么
- **改进理由**: 针对哪些薄弱指标
- **得分变化**: XX/105 → YY/105 (±Z)
- **各指标详情**:
  - ★提升的指标: xxx 从3→4
  - ★下降的指标: xxx 从5→4
- **下一步计划**: 基于本轮结果的下一步方向
- **模型文件**: ml_models/trained_models/vXX/xxx.pkl
```

### Step 9: 决策

- 如果分数提升 → 继续在新版本基础上迭代
- 如果分数持平或下降 → 回退改动，尝试不同方向
- 如果达到 105/105 → 输出 `<promise>NORTH_STAR_PERFECT</promise>`

---

## 可用的改进方向菜单

以下是你可以尝试的具体改进，按预期ROI排序：

### 高ROI (可能提升5-10分)
1. **换手率优化**: 引入持仓延续机制，如果昨天Top10和今天Top10重叠度>50%，保持持仓不变
2. **流动性过滤增强**: 在评分后过滤成交额<500万的股票
3. **市值均衡约束**: 强制Top10中大盘(>100亿)和小盘(<50亿)各占一定比例
4. **CPPI参数精细搜索**: 在最优附近搜索更细的网格

### 中ROI (可能提升3-5分)
5. **多窗口融合**: 3d/5d/10d预测权重动态调整（根据近期市场状态）
6. **行业分散约束**: 每个行业最多选2只，避免行业集中
7. **信号衰减机制**: 连续推荐的股票信号递减
8. **Isotonic回归校准**: 让预测值与实际收益单调对齐

### 低ROI但有潜力
9. **新特征**: 北向资金、融资融券余额、龙虎榜数据
10. **Attention机制**: 对时序特征使用注意力加权
11. **对抗训练**: GAN-based数据增强
12. **市场状态条件模型**: 不同市况使用不同子模型

---

## 关键文件路径

```
项目根目录: /Users/yangxu/StockTradebyZ/

训练:
  ml_models/training/train_v395_multi_target.py  # 主训练脚本 (--v44/--v46)

评分器:
  ml_models/v39/v44_production_scorer.py         # V4.4评分器
  ml_models/v39/v46_production_scorer.py         # V4.6评分器
  ml_models/v39/v395_production_scorer.py        # V3.95基础评分器

报告生成:
  backtest/batch_generate_v395_reports.py         # 批量报告生成

评估:
  backtest/run_north_star_eval.py                 # 北极星评估入口
  backtest/north_star_metrics.py                  # 21项指标定义 (不可修改!)
  backtest/backtest_report_based.py               # 回测引擎 (不可修改交易成本!)
  scripts/v45_param_search.py                     # CPPI参数搜索

数据:
  data_adapter/stock_data.db                      # 主数据库 (不可删除!)

模型:
  ml_models/trained_models/v44/                   # V4.4模型
  ml_models/trained_models/v46/                   # V4.6模型

日志:
  ralph_iteration_log.md                          # 迭代记录 (你来维护)
```

---

## 首轮迭代启动

如果这是首次迭代（`ralph_iteration_log.md` 不存在），执行以下初始化：

1. 创建 `ralph_iteration_log.md`
2. 运行当前最佳模型的完整北极星V2评估
3. 记录所有21项指标的当前得分
4. 识别得分 < 5 的指标
5. 选择最高ROI的改进方向开始第一次迭代

---

## 安全检查

每轮迭代前确认：
- [ ] `data_adapter/stock_data.db` 存在且大小 > 3GB
- [ ] `north_star_metrics.py` 中 NORTH_STAR_TARGETS_V2 未被修改
- [ ] 上一轮的模型文件仍然存在

如果任何检查失败，立即停止并报告问题。
