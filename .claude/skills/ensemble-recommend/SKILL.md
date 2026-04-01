---
name: ensemble-recommend
description: "多模型综合选股: 自动选择北极星V4最优的4个模型，生成全市场报告并交叉验证，输出综合Top10推荐"
argument-hint: "[date in YYYY-MM-DD, default today] [--top-models N] [--models v4.7.3 v4.7.5 ...] [--force-refresh]"
allowed-tools: Bash(python3 *), Read, Glob, Grep
---

# 多模型综合选股推荐

基于北极星V4评分自动选择历史表现最优的模型，为指定日期生成全市场选股报告，交叉验证后输出综合推荐。

## Arguments

- `$ARGUMENTS`: 可选参数，支持以下格式：
  - 空: 使用今天日期，自动选择北极星V4最优的4个模型
  - `2026-03-18`: 指定日期
  - `2026-03-18 --top-models 3`: 指定日期 + 选择最优的3个模型
  - `--models v4.7.3 v4.7.5 v4.7.6 v4.7.7`: 直接指定模型版本，跳过北极星评估
  - `--force-refresh`: 强制重新评估所有模型（忽略缓存）

## 工作流程

整个流程分3个阶段:

1. **北极星V4评估** (首次~3分钟/模型，后续使用缓存秒级完成): 对所有候选模型运行回测，计算V4加权评分，选出Top N
2. **全市场报告生成** (~30秒/模型): 为选中模型生成当日全市场ML评分报告
3. **交叉验证推荐** (<1秒): 提取各模型Top20，按出现次数和平均排名综合排序，输出Top10

缓存说明: 北极星V4评分结果会缓存到 `reports/ensemble_recommend/north_star_v4_cache.json`。
只有当模型的回测报告数量变化时才重新评估。使用 `--force-refresh` 强制重新评估。

## Execution Steps

### Step 1: 运行综合推荐脚本

解析用户参数并构建命令:

```bash
python3 /Users/yangxu/StockTradebyZ/scripts/ensemble_daily_recommend.py $ARGUMENTS
```

如果用户没有提供日期参数，使用今天日期。例如今天是2026-03-20:

```bash
python3 /Users/yangxu/StockTradebyZ/scripts/ensemble_daily_recommend.py --date 2026-03-20
```

如果用户指定了模型:

```bash
python3 /Users/yangxu/StockTradebyZ/scripts/ensemble_daily_recommend.py --date 2026-03-20 --models v4.7.3 v4.7.5 v4.7.6 v4.7.7
```

超时设置: 此命令可能耗时较长 (北极星评估约3分钟/模型 + 报告生成约30秒/模型)。
建议设置 timeout 为 600000ms (10分钟)。

### Step 2: 展示结果

脚本会输出:
1. 北极星V4排名表 (如果执行了评估)
2. 各模型Top20交叉验证表
3. 综合推荐Top10

将关键结果以表格形式呈现给用户，重点展示:
- 选中的模型及其北极星V4分数
- 综合Top10推荐股票，标注每只股票被几个模型命中
- 被3个以上模型同时选中的股票用 ⭐ 标注

### Step 3: 补充分析 (可选)

如果用户需要，可以进一步:
- 读取 `reports/ensemble_recommend/综合推荐_YYYYMMDD.json` 获取详细数据
- 读取各版本的 `reports/daily_selection_vX.X.X_fullmarket/选股分析报告_YYYYMMDD.md` 查看个股详情

## Error Handling

- 如果北极星评估阶段某个模型失败，会跳过该模型继续评估其他模型
- 如果有效模型不足2个，脚本会报错退出
- 如果报告生成失败 (例如当天非交易日无数据)，提示用户检查数据更新状态
- 建议先运行 `/update-data` 确保数据是最新的

## Notes

- 北极星V4评分体系: 31项/155分, 5层权重 (L1信号35% + L2效率15% + L3风控20% + L4鲁棒15% + L5超额15%)
- 候选模型池来自 `reports/*_merged_extended` 目录中有500+天回测数据的版本
- 结果保存在 `reports/ensemble_recommend/综合推荐_YYYYMMDD.json`
- 全市场报告保存在 `reports/daily_selection_vX.X.X_fullmarket/`
