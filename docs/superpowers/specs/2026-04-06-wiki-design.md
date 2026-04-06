# StockTradebyZ Wiki 设计文档

**日期**: 2026-04-06
**状态**: Approved
**作者**: Claude + yangxu

## 背景

基于 [Karpathy LLM Wiki 模式](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) 为 StockTradebyZ 建立项目知识库。核心理念：LLM 维护的互联 markdown Wiki，知识持续沉淀而非消散。

## 目标

- **主要受众**: 项目作者 (yangxu) + 未来 Claude 会话
- **核心用途**: 个人知识库，记录系统演化决策、模型实验结论、踩坑教训
- **语言**: 混合（技术术语英文，解释和决策记录中文）

## 三层架构

| 层 | 内容 | 维护者 |
|---|---|---|
| Raw Sources | 代码、配置、git log、数据库 | 不可变，LLM 只读 |
| Wiki | `docs/wiki/` 下的 markdown 页面 | Claude 写入 + 用户审核 |
| Schema | `docs/wiki/schema.md` 定义规则 | 用户定义，Claude 遵循 |

## 目录结构

```
docs/wiki/
├── index.md              # 分类目录（所有页面入口，带摘要）
├── log.md                # 项目里程碑时间线（重大事件）
├── schema.md             # Wiki 维护规则（给 Claude 的指令）
│
├── architecture/
│   ├── system-overview.md        # 整体架构、数据流、组件关系
│   ├── data-pipeline.md          # 数据层：Tushare→SQLite→Feature Cache
│   └── ml-pipeline.md            # ML层：特征→训练→推理→报告
│
├── models/
│   ├── evolution.md              # 模型世代总览（V3.8→NG1.1.0）
│   ├── ng-series.md              # NG系列详解（1.0.0→1.1.0）
│   ├── v4x-series.md             # V4.x实验总结
│   └── v39-series.md             # V3.9/3.95（旧版参考）
│
├── evaluation/
│   ├── north-star.md             # 北极星V1→V5.2演化与解读
│   └── backtesting.md            # 回测方法论、无泄露原则
│
├── lessons/
│   └── known-pitfalls.md         # 已知陷阱汇总
│
└── features/
    └── feature-guide.md          # 69特征说明、来源、选择逻辑
```

## 核心操作闭环

```
开始改动 → 查 Wiki (Query) → 做改动 → 更新 Wiki (Ingest) → 更新 CLAUDE.md
```

### Query（改动前）
1. 读取 `docs/wiki/index.md` 了解有哪些页面
2. 根据任务关键词读取最相关的 1-3 个 Wiki 页面
3. 利用上下文避免踩坑、了解历史决策、保持架构一致

### Ingest（改动后）
1. 判断改动是否影响 Wiki 已有页面
2. 影响则更新相关页面 + `log.md` 追加条目
3. 全新主题则创建新页面 + 更新 `index.md` + `log.md`
4. 纯小 bug fix / 代码风格调整则跳过

### CLAUDE.md 同步（改动后）
- 新命令/入口/配置 → 更新 Quick Start / Commands
- 模型版本/性能指标 → 更新 ML Systems
- 新文件/目录结构 → 更新 Project Structure
- 新教训/陷阱 → 更新已知陷阱

### Lint（按需）
用户可要求 Claude 审计 Wiki：检查过时信息、缺失页面、矛盾内容。

## 与 Memory 系统的分工

| | Memory (.claude/memory/) | Wiki (docs/wiki/) |
|---|---|---|
| 范围 | 对话级上下文、用户偏好 | 项目级知识、架构决策 |
| 生命周期 | 短~中期，可能过时 | 长期，主动维护 |
| 维护者 | Claude 自动写入 | Claude 写入 + 用户审核 |
| 示例 | "用户偏好中文commit" | "V4.6重训练失败因为小盘加权×2.5" |

## log.md 格式

```
YYYY-MM-DD | 类别 | 描述
```

类别: `model` / `arch` / `fix` / `feature` / `data` / `eval`

## 初始内容来源

| Wiki 页面 | 数据源 |
|---|---|
| system-overview.md | CLAUDE.md + 代码结构 |
| data-pipeline.md | fetch_data/ + data_adapter/ |
| ml-pipeline.md | ml_models/ + tomorrow_stock_selector.py |
| evolution.md | MEMORY.md + git log |
| ng-series.md | ml_models/ng/ + MEMORY.md |
| v4x-series.md | MEMORY.md V4.x 记录 |
| v39-series.md | ml_models/v39/ + CLAUDE.md |
| north-star.md | backtest/north_star_metrics.py |
| backtesting.md | backtest/ + MEMORY.md |
| known-pitfalls.md | MEMORY.md + CLAUDE.md |
| feature-guide.md | ng_feature_calculator.py + ng_trainer.py |
| log.md | git log + MEMORY.md |

## 写入原则

- 从真实代码和记录中提取，不猜测不编造
- 保留关键数字（ICIR、V5.2分数、年化收益等）
- 每页聚焦一个主题，交叉引用其他页面
- 中文解释 + 英文术语的混合风格
