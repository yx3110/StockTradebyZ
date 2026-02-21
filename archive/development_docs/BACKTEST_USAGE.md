# 真实数据评分系统回测对比使用指南

## 📋 概述

`genuine_scoring_backtest_comparator.py` 是一个专业级的量化回测系统，用于对比不同版本评分器的历史表现。

**重要**：此脚本读取已有的评分报告数据，不需要重新生成报告。

## 🚀 基本用法

### 命令行使用

```bash
# 基本用法 - 对比默认版本
python3 genuine_scoring_backtest_comparator.py

# 指定版本对比
python3 genuine_scoring_backtest_comparator.py --versions V3.0 V3.4 V3.5

# 指定时间范围（建议至少180天）
python3 genuine_scoring_backtest_comparator.py \
    --versions V3.0 V3.4 \
    --start-date 2024-01-01 \
    --end-date 2025-09-10 \
    --min-sample-days 300

# 查看可用版本
python3 genuine_scoring_backtest_comparator.py --versions INVALID_VERSION
# 会显示所有可用版本列表
```

### Python代码调用

```python
from genuine_scoring_backtest_comparator import quick_compare

# 快速对比
report_path = quick_compare(
    versions=['V3.0', 'V3.4', 'V3.5'], 
    start_date='2024-01-01', 
    end_date='2025-09-10'
)
print(f"报告已生成: {report_path}")
```

## 📊 支持的版本

系统自动检测以下版本的数据：
- V3.0, V3.1, V3.2, V3.3, V3.4, V3.41
- V3.5, V3.51, V3.52, V3.53, V3.6
- V4.0

## ⚠️ 数据质量要求

### 样本量要求
- **最少60个交易日**（约3个月）
- **建议180个交易日**（约9个月）
- **专业级300个交易日**（约1.5年）

### 数据来源
系统读取以下数据：
- `reports/daily_selection_v*/analysis_data_YYYYMMDD.json` - 评分数据
- `data_adapter/stock_data.db` - 价格数据

## 📈 输出报告

生成的报告包含：

### 1. 数据概览
- 总记录数、覆盖股票数、交易日数
- 平均评分、评分标准差、有效评分数

### 2. IC分析（Information Coefficient）
- 1日、3日、5日、10日、20日周期
- 平均IC、IC信息比率、正IC占比
- 预测能力评级

### 3. 组合表现分析
- 前30%股票策略表现
- 年化收益率、夏普比率、胜率
- 最大/最小收益

### 4. 风险分析
- 波动率、最大回撤、VaR(95%)
- 偏度、峰度、下行偏差

### 5. 综合排名
- 基于IC、收益、风险的综合评分
- 版本对比和推荐

## 🔍 使用场景

### 日常监控
```bash
# 监控最新版本表现
python3 genuine_scoring_backtest_comparator.py --versions V3.6 V4.0
```

### 版本选择
```bash
# 对比多个版本选择最佳
python3 genuine_scoring_backtest_comparator.py \
    --versions V3.0 V3.4 V3.5 V3.52 \
    --start-date 2024-01-01 \
    --end-date 2025-09-10 \
    --min-sample-days 200
```

### 长期分析
```bash
# 使用最大样本量进行深度分析
python3 genuine_scoring_backtest_comparator.py \
    --versions V3.0 V3.4 \
    --start-date 2023-01-01 \
    --end-date 2025-09-10 \
    --min-sample-days 400
```

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--versions` | 要对比的版本列表 | V3.0 V3.52 V3.6 | --versions V3.0 V3.4 |
| `--start-date` | 开始日期 | 2025-01-01 | --start-date 2024-01-01 |
| `--end-date` | 结束日期 | 2025-09-10 | --end-date 2025-12-31 |
| `--min-sample-days` | 最少样本天数 | 60 | --min-sample-days 180 |

## 🚨 注意事项

1. **数据依赖**：确保对应版本的报告文件存在
2. **样本量**：小样本结果不可靠，建议至少180天
3. **数据质量**：某些版本可能数据不足，系统会自动警告
4. **计算时间**：大样本量分析需要较长时间

## 📁 输出位置

报告保存在：`reports/backtest/genuine_scoring_backtest_TIMESTAMP.md`

## 🔧 故障排除

### 常见问题

**Q: 版本数据不足怎么办？**
A: 选择数据更充足的版本或扩大时间范围包含更多历史数据

**Q: 某个版本显示"数据为空"？**
A: 检查 `reports/daily_selection_vX.X/` 目录是否存在相应的JSON文件

**Q: IC值都是负数？**
A: 这可能反映真实的市场表现，负IC说明该期间预测效果不佳

**Q: 样本量警告？**
A: 增加 `--min-sample-days` 或扩大日期范围

### 数据检查
```bash
# 检查某版本有多少数据文件
ls reports/daily_selection_v3.0/analysis_data_2024*.json | wc -l

# 检查时间范围
ls reports/daily_selection_v3.0/analysis_data_*.json | head -5
ls reports/daily_selection_v3.0/analysis_data_*.json | tail -5
```

---

*更新时间: 2025-09-11*
*系统版本: genuine_scoring_backtest_comparator.py v1.0*