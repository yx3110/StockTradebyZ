# Signal Trust — 选股信号可信度验证系统

**日期**: 2026-04-12
**状态**: 待审核
**作者**: Claude (brainstorming with @yx3110)

---

## 背景与动机

当前选股流水线（ML 模型 + 8 策略）对每只股票产出 `pred_3d/5d/10d/15d` 等预测涨幅。但用户的核心疑虑是：

> **虽然我们可以选出某些个股，但每个个股的庄家（主力资金）习惯不同，他可能就是会针对我们的模型释放假信号。如果我选出了高分的小市值股票，我想看如果这个股票之前被我们选中的时候的涨跌幅 vs 我们预测的涨跌幅——这样能识别出虚假信号。**

本系统建立在**历史事实**之上：对每只股票，统计它过去被模型标记为"看多"时，后续 10 日实际收益与预测的偏差，输出"可信度标签"帮助用户筛选。同时在全市场层面按市值/行业/流动性分组识别系统性失效区域。

---

## 目标

**主目标**：
1. 为每日 Top-50 选股贴 🟢可信 / 🟡存疑 / 🔴高风险 / ⚪数据不足 的可信度标签
2. 每周输出按市值档 / 行业 / 流动性分组的模型失效诊断报告

**非目标**：
- 不修改 ML 模型、选股策略、回测流程
- 不做"实时"防假信号检测（这是事后质检层，不是风控层）
- 不替代人工判断（只提供数据支撑）

---

## 关键设计决策（已与用户确认）

| 决策点 | 最终选择 | 理由 |
|--------|---------|------|
| 用途 | A（日常标签）+ C（全局统计） | 用户原始需求 |
| 可信度指标 | ①方向命中率 + ③系统偏差 + ④高预测兑现率 | 贴合"虚假信号"直觉，避免小样本 IC 噪音 |
| "被选中"定义 | 样本池 `pred_10d > 0.01`，展示用 Top-N | 样本稳健 + 关注聚焦 |
| 统计时间窗口 | 全历史累计 | 用户选择；接受早期市场环境差异噪音 |
| 输出形式 | 双层：颜色标签 + 三指标原始值 | 一眼扫分级 + 可下钻 |
| 全局分组 | 市值 / 行业 / 流动性 独立三份统计 | 避免三维交叉样本稀释 |
| 技术架构 | 方案 B：SQLite 缓存表 + 增量更新 | 日报集成 <1秒；与项目缓存表惯例一致 |

**颜色标签阈值**（保守偏向识别假信号）：
- 🟢 可信：方向命中 ≥55% **且** 偏差 ≥ -2% **且** 兑现率 ≥40%
- 🔴 高风险：方向命中 <45% **或** 偏差 < -3% **或** 兑现率 <20%
- 🟡 存疑：其余情况
- ⚪ 数据不足：样本 <10 次

**三指标数学定义**（样本池已限定 `pred_10d > 0.01`，即"模型至少温和看多"）：
```
direction_hit_rate     = mean(sign(pred_10d) == sign(actual_10d))
systematic_bias        = mean(actual_10d - pred_10d)     # 负值大=模型系统高估
high_pred_realize_rate = mean(actual_10d > 0.02)         # 实际是否兑现"≥2%"真实收益
```
说明 ④：由于样本池已过滤 pred_10d>1%，若模型"诚实"，actual 应大概率达到 2%；若庄家对该股放假信号，actual 会经常接近 0 或为负，此指标会显著偏低。

---

## 架构

```
signal_trust/
├── __init__.py
├── sample_builder.py      # 样本池构建/增量更新
├── scorer.py              # 单股可信度聚合
├── report_appender.py     # 为日报追加标签
└── global_stats.py        # 周度全局统计

scripts/
├── rebuild_signal_trust.py       # 首次建库
├── update_signal_trust_daily.py  # 每日增量（集成 run_daily_update.sh）
└── weekly_signal_trust_stats.py  # 周度跑

data_adapter/stock_data.db（新增两张表）:
├── signal_trust_samples    # 样本池，主键 (code, trade_date)
└── signal_trust_scores     # 每股聚合指标 + 标签
```

**集成触点**：
1. `run_daily_update.sh` → 调用 `update_signal_trust_daily.py`（每天 ~5 秒）
2. `tomorrow_stock_selector.py` 生成 JSON 报告后 → 调用 `report_appender.py`（每天 <1 秒）
3. `weekly_signal_trust_stats.py` 独立 cron（每周日 21:00）

**不变量**：
- 样本表按 `(code, trade_date)` 唯一主键，幂等入库
- 可信度分数仅使用 `sample_end_date < today` 的样本（防数据泄露）
- 样本与分数表可重建不可改写

---

## 数据表结构

### `signal_trust_samples`
| 字段 | 类型 | 说明 |
|------|------|------|
| code | TEXT | 股票代码 带后缀（主键） |
| trade_date | TEXT | 预测日 YYYY-MM-DD（主键） |
| sample_end_date | TEXT | trade_date + 10 交易日，预计算列 |
| pred_10d | REAL | 预测涨幅 |
| actual_10d | REAL | 实际涨幅，NULL 表示未兑现/停牌 |
| version | TEXT | 来源模型版本（ng106/ng101/v4901/...） |
| market_cap_bucket | TEXT | 微盘/小盘/中盘/大盘/未知（入库时快照） |
| industry | TEXT | 申万一级行业 / 未分类 |
| liquidity_bucket | TEXT | 低/中低/中高/高（按 30 日日均成交额） |
| created_at | TEXT | 入库时间 |

索引：`(code)`, `(trade_date)`, `(sample_end_date)`, `(market_cap_bucket)`, `(industry)`, `(liquidity_bucket)`

### `signal_trust_scores`
| 字段 | 类型 | 说明 |
|------|------|------|
| code | TEXT | 主键 |
| as_of_date | TEXT | 计算截止日 |
| n_samples | INT | 参与计算的样本数 |
| direction_hit_rate | REAL | 指标 ① |
| systematic_bias | REAL | 指标 ③ |
| high_pred_realize_rate | REAL | 指标 ④ |
| trust_tag | TEXT | 🟢/🟡/🔴/⚪ |
| updated_at | TEXT | 最后更新时间 |

---

## 数据流

### 首次建库（一次性，~10 分钟）
1. 扫 `reports/daily_selection_*/analysis_data_*.json` 所有报告
2. 对每只股票抽取 (code, trade_date, pred_10d, version)
3. **跨版本去重**：同 (code, trade_date) 按版本优先级 `ng106 > ng101 > v4901 > v39` 取最新
4. 过滤 `pred_10d > 0.01`
5. 关联 `daily_quotes` 计算 `actual_10d`（T+10 未到则 NULL）
6. 关联 `daily_basic` / `stock_basic_info` 冻结入库日快照
7. `INSERT OR IGNORE` 写入 samples 表
8. 全量跑 `compute_trust_scores()` 刷新 scores 表

### 每日增量（<5 秒）
1. **(A) 入库今日新样本**：扫当日新报告，添加 actual_10d=NULL 的记录
2. **(B) 回填 T-10 的 actual_10d**：`UPDATE ... WHERE trade_date BETWEEN T-12 AND T-10 AND actual_10d IS NULL`
3. **(C) 刷新被影响股票的 scores**

### 日报贴标签（<1 秒）
1. 读 `analysis_data_YYYYMMDD.json`
2. 取 Top-50 股票 code
3. 单次 SQL 查 `signal_trust_scores`
4. 追加 `trust_tag`/`trust_samples`/`trust_details` 字段
5. 原子写回（临时文件 + rename）

### 周度全局统计（每周日）
按三维度独立 GROUP BY 聚合 → Markdown 报告 → `reports/signal_trust/global_stats_YYYYMMDD.md`

### 数据泄露防护 🚨
`compute_trust_scores(as_of_date)` 的 SQL：
```sql
SELECT * FROM signal_trust_samples
WHERE sample_end_date < :as_of_date
  AND actual_10d IS NOT NULL
```
今日新入库样本要到 **T+10 日**才参与可信度计算——这是正确行为。

---

## 错误处理与边界

| 场景 | 策略 |
|------|------|
| JSON 损坏 | 跳过 + WARN，继续下一个 |
| pred_10d 缺失 | 视为 0 自动过滤 |
| 停牌 ≥10 日 | actual_10d=NULL 永不回填 |
| 缺行业/市值 | 标 "未分类"/"未知" |
| 多版本冲突 | 版本优先级列表硬编码取最新 |
| SQLite 并发 | `busy_timeout=30000`（项目规范） |
| 入库/回填/分数刷新 | 全部幂等，可重跑 |
| 增量失败 | 次日回填自动补；缺失新样本需 `--date` 手动补跑 |

**版本优先级列表**：硬编码于 `sample_builder.VERSION_PRIORITY = ['ng106', 'ng101', 'v4901', 'v39', ...]`。新版本上线需手动加入列表顶部。

---

## 性能预算

- 首次建库 < 15 分钟
- 每日增量 < 10 秒
- 日报贴标签 < 2 秒
- 周度全局统计 < 30 秒

---

## 测试策略

### 单元测试
- `test_sample_builder.py`：阈值过滤、跨版本去重、actual_10d 计算（含跨周末）、停牌、市值快照、幂等
- `test_scorer.py`：三指标数值、min_samples 门槛、颜色阈值边界、**数据泄露防护 🚨**
- `test_report_appender.py`：字段追加、分数表缺失时优雅降级、原子写
- `test_global_stats.py`：市值分档、行业 Top/Bottom5、流动性动态阈值

### 集成测试
- `test_integration_full_cycle.py`：小数据集端到端（3股 × 30天），CI 运行
- `test_integration_real_data.py`：真实库建库验证，仅本地跑

### 数据验证脚本
- `scripts/validate_signal_trust.py`：建库后强制跑，输出样本分布 + 泄露自检 + 抽样 10 只股票人工对照

### 测试取舍
- 不 mock SQLite，用临时 .db 文件跑真实 SQL
- 性能预算不入 CI，改用耗时日志 + 阈值告警
- 重点覆盖边界：泄露、幂等、停牌、跨版本去重

---

## 使用范例

**日报贴标签后的 JSON 样例**：
```json
{
  "stock_code": "002215.SZ",
  "stock_name": "诺普信",
  "pred_10d": 0.012,
  "rank_score": 87.3,
  "trust_tag": "🔴高风险",
  "trust_samples": 34,
  "trust_details": {
    "direction_hit_rate": 0.44,
    "systematic_bias": -0.038,
    "high_pred_realize_rate": 0.15,
    "as_of_date": "2026-04-12"
  }
}
```

**周度全局统计节选**：
```
===== 按市值分组 (as_of 2026-04-12) =====
市值档        样本数   方向命中    系统偏差    兑现率
微盘(<30亿)   1,234   42.1%      -3.8%      18.5%   ⚠️高风险
小盘          3,456   48.7%      -2.1%      28.3%   🟡存疑
中盘          4,321   54.2%      -1.2%      35.1%   ✓
大盘          2,108   57.8%      -0.5%      42.6%   ✓
```

---

## 未来可能扩展（非本期范围）

1. 按模型版本分别计算可信度（目前是合并后的"综合可信度"）
2. 加入 pred_3d / pred_5d 的独立可信度（目前只看 pred_10d）
3. 相对可信度：同市值档内的相对排名，而非绝对标签
4. 可信度时间衰减：近期样本权重更高（目前全历史等权）

---

## 审核 Checklist

- [ ] 架构划分合理
- [ ] 组件接口清晰
- [ ] 数据流完备
- [ ] 泄露防护严格
- [ ] 错误处理周全
- [ ] 测试覆盖充分
