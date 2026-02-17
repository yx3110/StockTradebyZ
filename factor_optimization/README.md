# 📊 Factor Optimization - 可扩展因子优化系统

这个文件夹包含了v3.2及后续版本的标准化因子计算和可扩展权重优化系统，支持动态添加新因子和版本迭代。

## 🏗️ 系统架构

```
factor_optimization/
├── standard_factor_calculator.py      # 标准化因子计算器
├── weight_optimizer.py               # 🆕 可扩展权重优化器
├── standard_factors.db               # 标准化因子数据库 (0-100分)
├── optimization_results_*.json       # 优化结果数据
├── optimized_configs_*.json          # 优化后的配置文件
└── README.md                         # 本文档
```

## 📋 数据标准规范

### 标准化因子表结构 (standard_factors)

**7个维度评分** (0-100分标准化):
- `technical_score`: 技术指标维度总分
- `squeeze_momentum_score`: 挤压动量维度总分 
- `fundamental_score`: 基本面维度总分
- `performance_score`: 市场表现维度总分
- `sentiment_score`: 情绪指标维度总分
- `risk_control_score`: 风险控制维度总分
- `market_regime_score`: 市场环境维度总分

**子因子详细评分** (0-100分标准化):
- 技术指标: `kdj_strength`, `rsi_momentum`, `bbi_trend`, `volume_surge`
- 挤压动量: `squeeze_state`, `squeeze_release`, `momentum_direction`, `momentum_consistency`
- 基本面: `pe_valuation`, `pb_valuation`, `roe_profitability`, `financial_quality`, `market_cap`, `turnover_activity`
- 市场表现: `price_momentum`, `relative_strength`, `volatility_risk`
- 情绪指标: `money_flow`, `market_attention`, `investor_emotion`
- 风险控制: `stop_loss_risk`, `max_drawdown`, `risk_adjusted_return`
- 市场环境: `market_beta`, `sector_rotation`, `liquidity`

**未来收益数据** (用于优化验证):
- `return_1d`, `return_3d`, `return_5d`, `return_10d`, `return_20d`

## 🚀 使用步骤

### 🆕 一键添加新因子 (v3.3+)

```bash
# 🎯 从TradingView URL一键集成新因子
python3 factor_optimization/add_new_factor.py --interactive
# 或
python3 factor_optimization/add_new_factor.py --url https://www.tradingview.com/script/XXX/

# 📖 查看详细集成指南
cat factor_optimization/NEW_FACTOR_INTEGRATION_GUIDE.md
```

**支持的集成流程**:
- 自动数据库结构升级
- 生成因子计算器代码模板
- 更新权重优化器配置
- 执行完整的权重优化
- 生成因子影响分析报告

### 1. 计算标准化因子数据

```bash
# 计算完整历史标准化因子数据 (首次运行或更新数据)
python3 factor_optimization/standard_factor_calculator.py \
  --start-date 2024-01-01 \
  --end-date 2025-08-25 \
  --max-workers 6 \
  --batch-size 1000
```

**输出**: 在`factor_optimization/standard_factors.db`中生成标准化的0-100分因子数据

### 2. 🆕 可扩展权重优化

#### 基础优化 (v3.2)

```bash
# 基于标准化数据优化权重 - 默认v3.2配置
python3 factor_optimization/weight_optimizer.py
```

#### 高级使用 - Python API

```python
from factor_optimization.weight_optimizer import WeightOptimizer

# 1. 初始化优化器
optimizer = WeightOptimizer()

# 2. 添加新维度 (例如：v3.3版本)
new_dimension = {
    "description": "机器学习预测维度",
    "weight_range": [0.05, 0.10, 0.15, 0.20],
    "factors": ["lstm_prediction", "transformer_score"],
    "sub_weights": {
        "lstm_prediction": [0.6, 0.7, 0.8],
        "transformer_score": [0.2, 0.3, 0.4]
    }
}
optimizer.add_new_dimension("ml_prediction", new_dimension)

# 3. 向现有维度添加新因子
optimizer.add_new_factor_to_dimension("technical", "bollinger_squeeze", [0.1, 0.15, 0.2])

# 4. 执行优化
result = optimizer.optimize_weights(
    start_date='2024-01-01',
    end_date='2025-08-25'
)

# 5. 保存配置
optimizer.save_config("configs/v3_3_config.json")
```

#### 配置文件驱动

```bash
# 使用自定义配置文件
python3 -c "
from factor_optimization.weight_optimizer import WeightOptimizer
optimizer = WeightOptimizer('configs/custom_config.json')
result = optimizer.optimize_weights()
"
```

**输出**: 
- `factor_optimization/optimization_results_*.json`: 优化结果
- `factor_optimization/optimized_configs_*.json`: 优化后的配置

### 3. 应用优化结果

将优化后的权重配置应用到评分系统:
```bash
# 使用优化后的配置测试效果
python3 tomorrow_stock_selector.py --scoring-version v3.2
```

## 🔧 系统特性

### 1. 🆕 可扩展架构
- **动态维度**: 运行时添加新评分维度
- **动态因子**: 向现有维度添加新因子
- **配置驱动**: JSON/YAML格式配置文件
- **版本管理**: 支持多版本并行开发

### 2. 智能数据适配
- **自动发现**: 自动发现数据库中的可用因子 
- **智能匹配**: 验证配置与数据库的兼容性
- **缺失警告**: 自动检测和警告不存在的因子

### 3. 标准化数据格式
- **统一评分范围**: 所有因子统一为0-100分
- **缺失值处理**: 缺失数据使用50分(中性值)
- **异常值过滤**: 自动过滤超出合理范围的数据

### 4. 高级权重优化
- **多目标优化**: 6个评估维度的综合优化
- **灵活权重**: 支持维度权重和子因子权重
- **预测验证**: 基于未来收益验证优化效果

## 📈 扩展示例

### 版本演进路线图

```python
# v3.2 (当前) - 挤压动量集成
# 7个维度：技术、挤压动量、基本面、表现、情绪、风险、市场环境

# v3.3 (计划) - 技术指标扩展
optimizer.add_new_factor_to_dimension("technical", "bollinger_squeeze")
optimizer.add_new_factor_to_dimension("technical", "stochastic_rsi")

# v3.4 (计划) - 机器学习维度
ml_dimension = {
    "description": "机器学习预测维度",
    "weight_range": [0.05, 0.10, 0.15],
    "factors": ["lstm_prediction", "transformer_score", "ensemble_confidence"]
}
optimizer.add_new_dimension("ml_prediction", ml_dimension)

# v3.5 (计划) - 宏观经济维度
macro_dimension = {
    "description": "宏观经济环境维度",
    "weight_range": [0.02, 0.04, 0.06],
    "factors": ["gdp_correlation", "interest_rate_sensitivity", "inflation_hedge"]
}
optimizer.add_new_dimension("macro_economic", macro_dimension)

# v4.0 (未来) - 多市场维度
multi_market_dimension = {
    "description": "多市场联动维度",
    "weight_range": [0.03, 0.05, 0.08],
    "factors": ["us_market_correlation", "hk_market_influence", "commodity_linkage"]
}
optimizer.add_new_dimension("multi_market", multi_market_dimension)
```

### 自定义配置文件示例

```json
{
  "version": "v3.3_custom",
  "description": "自定义扩展版本",
  "dimensions": {
    "technical": {
      "description": "增强技术指标维度",
      "weight_range": [0.40, 0.45, 0.50, 0.55],
      "factors": ["kdj_strength", "rsi_momentum", "bbi_trend", "volume_surge", "bollinger_squeeze"],
      "sub_weights": {
        "bollinger_squeeze": [0.10, 0.15, 0.20]
      }
    },
    "new_custom_dimension": {
      "description": "用户自定义维度",
      "weight_range": [0.05, 0.08, 0.12],
      "factors": ["custom_factor_1", "custom_factor_2"]
    }
  }
}
```

## 📊 数据质量保证

### 评分标准化规则

**0-100分映射**:
- 0-20分: 极差表现
- 20-40分: 较差表现
- 40-60分: 一般表现 (默认值)
- 60-80分: 良好表现
- 80-100分: 优秀表现

### 数据完整性检查

```bash
# 检查数据库状态
sqlite3 factor_optimization/standard_factors.db "SELECT COUNT(*) FROM standard_factors;"

# 检查数据质量
sqlite3 factor_optimization/standard_factors.db "
SELECT 
  COUNT(DISTINCT stock_code) as stocks,
  COUNT(DISTINCT trade_date) as dates,
  MIN(trade_date) as start_date,
  MAX(trade_date) as end_date,
  AVG(technical_score) as avg_technical,
  AVG(squeeze_momentum_score) as avg_squeeze
FROM standard_factors;"
```

## 📈 优化指标说明

### 主要评估维度

1. **分布合理性** (25%权重): 评分分布是否符合预期
2. **区分度** (20%权重): 评分是否能有效区分股票质量
3. **预测能力** (25%权重): 评分与未来收益的相关性
4. **高分表现** (15%权重): 高评分股票的实际表现
5. **维度有效性** (10%权重): 新维度的独立贡献
6. **权重平衡** (5%权重): 权重分配是否合理

### 理想分布目标

- 90+分股票: <1% (极少数优质标的)
- 80-90分股票: <5% (少数值得关注的股票)
- 70-80分股票: <12% (较好的投资选择)
- 60-70分股票: <25% (一般质量股票)
- <60分股票: >65% (大部分普通股票)

## 🔄 维护和更新

### 定期更新流程

1. **数据更新** (建议每周):
   ```bash
   python3 factor_optimization/standard_factor_calculator.py \
     --start-date $(date -d "7 days ago" +%Y-%m-%d) \
     --end-date $(date +%Y-%m-%d)
   ```

2. **权重优化** (建议每月):
   ```bash
   python3 factor_optimization/weight_optimizer.py
   ```

3. **效果验证** (优化后必须):
   ```bash
   python3 tomorrow_stock_selector.py --scoring-version v3.2 --test-mode
   ```

### 版本管理最佳实践

```bash
# 1. 创建版本分支配置
cp factor_optimization/configs/v3_2_config.json factor_optimization/configs/v3_3_config.json

# 2. 开发新版本
python3 -c "
from factor_optimization.weight_optimizer import WeightOptimizer
optimizer = WeightOptimizer('factor_optimization/configs/v3_3_config.json')
# 添加新特性...
optimizer.save_config('factor_optimization/configs/v3_3_final.json')
"

# 3. 测试和验证
python3 factor_optimization/weight_optimizer.py

# 4. 应用到生产
# 更新评分系统配置...
```

## 🚨 注意事项

### 数据依赖
- **统一数据源**: 完全基于`factor_optimization/standard_factors.db`
- **数据完整性**: 确保所有配置的因子在数据库中存在
- **时间一致性**: 优化时间范围应覆盖足够的历史数据

### 计算资源
- **标准化因子计算**: CPU密集型，建议多进程并行 (6-8 workers)
- **权重优化**: 内存需求较大，建议16GB+内存
- **数据库文件**: 预计每年数据约1-2GB

### 扩展性考虑
- **因子命名**: 使用描述性命名，避免冲突
- **权重范围**: 新维度权重建议从小范围开始测试
- **配置验证**: 添加新因子前先验证数据可用性

---

## 🎯 快速开始

```bash
# 1. 生成标准化数据
python3 factor_optimization/standard_factor_calculator.py --start-date 2024-01-01 --end-date 2025-08-25

# 2. 执行权重优化  
python3 factor_optimization/weight_optimizer.py

# 3. 验证优化结果
python3 tomorrow_stock_selector.py --scoring-version v3.2
```

## 🔗 相关文档

- **项目总览**: 参考项目根目录 `CLAUDE.md`
- **评分系统**: `scoring/v3/` 目录下的具体实现
- **数据适配**: `data_adapter/` 目录下的数据库管理

**📞 技术支持**: 所有问题请参考项目根目录CLAUDE.md文档或提交Issue