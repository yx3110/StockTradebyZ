# Claude API 集成指南

## 概述

本指南详细介绍了如何在TradingAgents集成系统中使用Anthropic Claude API进行股票分析。Claude具有出色的推理能力和中文支持，特别适合金融分析场景。

## 🚀 快速开始

### 1. API密钥配置（已预配置）

✅ **API密钥已配置** - 系统已自动检测到项目配置文件中的 Anthropic API 密钥

```bash
# 验证密钥设置（可选）
python3 TA_integration/adapters/claude_config.py

# 如需更换API密钥，可编辑主配置文件
# /Users/yangxu/StockTradebyZ/config.json -> anthropic.api_key
```

**密钥读取优先级：**
1. 项目根目录 `config.json` 文件中的 `anthropic.api_key` （当前使用）
2. 环境变量 `ANTHROPIC_API_KEY`
3. TA_integration 子配置文件（备用）

### 2. 运行Claude分析

```bash
# 从项目根目录运行（推荐）
cd /Users/yangxu/StockTradebyZ

# 🚀 使用默认配置（Claude 4 + 分析所有股票）
python3 TA_integration/main.py --date 2025-08-04

# 或简写为当日分析
python3 TA_integration/main.py

# 其他配置选项：
# 高质量分析（Claude 3.5）
python3 TA_integration/main.py --config claude_high_quality --date 2025-08-04

# 快速分析（批量筛选）
python3 TA_integration/main.py --config claude_fast --date 2025-08-04

# 顶级质量分析（Claude 3 Opus）
python3 TA_integration/main.py --config claude_premium --date 2025-08-04

# 只分析前10只股票（测试用）
python3 TA_integration/main.py --top-n 10
```

### 🎯 默认配置

- **默认模型**: Claude Sonnet 4 (最新版本)
- **默认行为**: 分析所有推荐股票
- **默认模式**: enhance（在量化分析基础上增加AI洞察）
- **输出位置**: `reports/ai_enhanced/` 目录

## 🎯 预设配置详解

### claude_4 (🆕 最新推荐)
- **深度思考**: Claude Sonnet 4 (2025年5月最新发布)
- **快速响应**: Claude Sonnet 4  
- **适用场景**: 顶级AI能力，混合推理模型，64K输出
- **成本**: 较高，但性能卓越
- **特点**: 最新AI技术，最强编程和推理能力

```bash
python TA_integration/main.py --config claude_4
```

### claude_high_quality (推荐)
- **深度思考**: Claude 3.5 Sonnet (最新版)
- **快速响应**: Claude 3.5 Haiku  
- **适用场景**: 日常重要决策，最佳性价比
- **成本**: 中等，性能优秀

```bash
python TA_integration/main.py --config claude_high_quality
```

### claude_balanced
- **深度思考**: Claude 3 Sonnet
- **快速响应**: Claude 3 Haiku
- **适用场景**: 日常分析，平衡成本和性能
- **成本**: 适中

```bash
python TA_integration/main.py --config claude_balanced
```

### claude_fast
- **深度思考**: Claude 3.5 Haiku
- **快速响应**: Claude 3.5 Haiku
- **适用场景**: 批量分析，快速筛选
- **成本**: 最低

```bash
python TA_integration/main.py --config claude_fast
```

### claude_premium
- **深度思考**: Claude 3 Opus
- **快速响应**: Claude 3.5 Sonnet
- **适用场景**: 关键投资决策，最高质量
- **成本**: 最高

```bash
python TA_integration/main.py --config claude_premium
```

## 💰 成本估算

### 单股分析成本（约3000输入 + 1500输出tokens）

| 模型 | 单股成本 | 10股成本 | 适用场景 |
|------|----------|----------|----------|
| Claude Sonnet 4 (🆕) | $0.032 | $0.32 | 顶级AI能力 |
| Claude 3.5 Haiku | $0.008 | $0.08 | 快速筛选 |
| Claude 3.5 Sonnet | $0.032 | $0.32 | 日常分析 |
| Claude 3 Sonnet | $0.032 | $0.32 | 平衡选择 |
| Claude 3 Opus | $0.158 | $1.58 | 重要决策 |

### 成本优化建议

1. **日常分析**: 使用 `claude_high_quality` 配置
2. **批量筛选**: 使用 `claude_fast` 配置  
3. **重要决策**: 使用 `claude_premium` 配置
4. **成本控制**: 限制 `--top-n` 参数

## 🔧 自定义配置

### 创建自定义配置文件

```json
{
  "llm_provider": "anthropic",
  "deep_think_llm": "claude-3-5-sonnet-20241022",
  "quick_think_llm": "claude-3-5-haiku-20241022",
  "max_tokens": 8192,
  "temperature": 0.1,
  "max_debate_rounds": 3,
  "custom_prompts": {
    "market_analyst": "你的自定义技术分析提示词...",
    "sentiment_analyst": "你的自定义情绪分析提示词..."
  }
}
```

### 使用自定义配置

```bash
python TA_integration/main.py --config custom --custom-config /path/to/your/config.json
```

## 🎭 中国市场专用提示词

系统内置了针对中国A股市场优化的提示词，涵盖：

### 技术分析师 (market_analyst)
- A股技术指标解读（MA、MACD、KDJ、RSI）
- T+1交易制度考虑
- 涨跌停板限制分析
- 政策敏感性评估

### 情绪分析师 (sentiment_analyst)  
- 散户情绪波动特点
- 雪球、东方财富股吧情绪
- 政策消息敏感性
- 热点题材炒作识别

### 新闻分析师 (news_analyst)
- 政策导向影响分析
- 行业发展趋势判断
- 监管变化影响评估
- 国际环境影响分析

### 基本面分析师 (fundamental_analyst)
- 国企vs民企特点分析
- 政策扶持行业识别
- 产业链上下游关系
- 区域经济发展影响

### 风险管理师 (risk_manager)
- A股特有风险评估
- 退市制度风险
- 资金面波动影响
- 政策调控风险

## 📊 分析流程

### 1. 数据准备
```python
# 系统自动转换中国股票数据格式
china_data -> yahoo_finance_format -> tradingagents_input
```

### 2. AI多智能体分析
```
技术分析师 -> 情绪分析师 -> 新闻分析师 -> 基本面分析师
    ↓
Bull研究员 vs Bear研究员 (辩论)
    ↓  
风险管理三方讨论
    ↓
最终投资决策
```

### 3. 结果整合
- AI决策建议 (BUY/SELL/HOLD)
- 置信度评分 (0-1)
- 看涨/看跌理由
- 风险评估等级
- 具体投资计划

## 🔍 调试和监控

### 启用详细日志
```bash
python TA_integration/main.py --config claude_high_quality --verbose
```

### 查看日志文件
```bash
# 实时查看日志
tail -f TA_integration/logs/TA_Integration_*.log

# 查看Claude API调用详情
grep "Claude" TA_integration/logs/*.log
```

### 成本监控
```python
from adapters.claude_config import ClaudeConfig

# 估算成本
cost = ClaudeConfig.estimate_cost("claude-3-5-sonnet-20241022", 1000, 500)
print(f"预估成本: ${cost:.4f}")
```

## ⚡ 性能优化

### 1. 模型选择优化
- **日常使用**: Sonnet 3.5 + Haiku 3.5
- **成本敏感**: Haiku 3.5 + Haiku 3.5  
- **质量优先**: Opus + Sonnet 3.5

### 2. 参数优化
```python
config = {
    "max_tokens": 4096,      # 适中的token限制
    "temperature": 0.1,      # 低温度确保一致性
    "max_debate_rounds": 2,  # 平衡质量和成本
}
```

### 3. 批量处理优化
```bash
# 分批处理，避免API限制
python TA_integration/main.py --top-n 5 --config claude_fast
python TA_integration/main.py --top-n 10 --config claude_balanced
```

## 🚨 错误处理

### 常见错误及解决方案

#### API密钥错误
```
❌ ANTHROPIC_API_KEY环境变量未设置
```
**解决**: `export ANTHROPIC_API_KEY='your_key'`

#### 模型访问错误
```
❌ Model not found or access denied
```
**解决**: 确认API密钥有效且有模型访问权限

#### Token限制错误
```
❌ Token limit exceeded
```
**解决**: 减少输入数据量或使用更高token限制的模型

#### 网络超时
```
❌ Request timeout
```
**解决**: 检查网络连接，重试分析

## 📈 最佳实践

### 1. 成本控制
- 日常分析使用 `claude_balanced`
- 重要决策使用 `claude_high_quality`
- 批量筛选使用 `claude_fast`

### 2. 质量保证
- 重要股票使用 `claude_premium`
- 对比多种配置结果
- 定期验证分析准确性

### 3. 效率提升
- 合理设置 `--top-n` 参数
- 使用缓存避免重复分析
- 批量处理相似股票

## 🔄 与量化系统集成

### 📁 输出文件规范（已更新）

所有输出文件统一保存在项目根目录的 `reports/` 文件夹中：

```
reports/
├── daily_selection/          # 每日选股报告（量化）
├── ai_enhanced/              # AI增强分析报告
├── ai_portfolio/             # AI投资组合报告  
├── trading_advice/           # 交易建议报告
└── performance/              # 绩效分析报告
```

### 增强模式（推荐）
```bash
# 在量化评分基础上增加AI洞察，输出到 reports/ai_enhanced/
python TA_integration/main.py --mode enhance --config claude_high_quality
```

### 替代模式
```bash
# 完全用AI评分替代量化评分，输出到 reports/ai_portfolio/
python TA_integration/main.py --mode replace --config claude_premium
```

### 对比模式
```bash
# 对比量化和AI评分差异，输出到 reports/performance/
python TA_integration/main.py --mode compare --config claude_balanced
```

### 🔒 安全改进

- ✅ **API密钥安全**: 不再在配置文件中硬编码API密钥
- ✅ **环境变量优先**: 优先从 `ANTHROPIC_API_KEY` 环境变量读取
- ✅ **路径兼容性**: 改进文件路径处理，支持不同工作目录
- ✅ **错误处理**: 增强异常处理和日志记录

## 📚 学习资源

### 官方文档
- [Anthropic Claude API 文档](https://docs.anthropic.com/)
- [Claude 模型对比](https://docs.anthropic.com/claude/docs/models-overview)

### 代码示例
```bash
# 运行完整示例
python TA_integration/examples/claude_usage_examples.py

# 查看配置示例
python TA_integration/adapters/claude_config.py
```

### 社区资源
- Claude API最佳实践
- 金融分析提示词工程
- 成本优化策略

## 🛠️ 开发指南

### 添加新的Claude模型
```python
# 在claude_config.py中添加新模型
CLAUDE_MODELS["new-model-name"] = {
    "max_tokens": 8192,
    "context_window": 200000,
    "cost_per_1k_input": 0.003,
    "cost_per_1k_output": 0.015,
}
```

### 自定义提示词
```python
custom_prompts = {
    "your_analyst": "你的专用分析师提示词..."
}

config = create_claude_trading_config("balanced")
config["custom_prompts"].update(custom_prompts)
```

### 扩展分析功能
- 添加新的分析维度
- 集成更多数据源
- 优化决策算法

---

## 📞 支持

如有问题，请：
1. 检查日志文件：`TA_integration/logs/`
2. 验证API密钥：`ClaudeConfig.validate_api_key()`
3. 运行示例代码：`python TA_integration/examples/claude_usage_examples.py`
4. 查看错误处理章节

**祝您使用Claude进行股票分析愉快！** 🚀📈