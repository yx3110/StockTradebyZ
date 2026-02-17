# StockTradebyZ x Qlib 回测集成系统

## 🎯 项目概述

本项目将StockTradebyZ的A股量化选股策略与Microsoft Qlib专业回测框架深度集成，提供了工业级的回测分析能力。集成系统保留了原有策略的核心逻辑，同时获得了Qlib的高级回测功能。

### 🌟 核心特性

- **专业回测引擎**: 基于Qlib框架，支持复杂的交易模拟
- **A股市场特化**: 完整支持T+1交易、涨跌停、中国式交易成本
- **多策略集成**: 4个核心选股策略的完整实现
- **风险管理**: 内置止损止盈、仓位控制、市场风险管理
- **性能分析**: 夏普比率、最大回撤、胜率等专业指标
- **对比分析**: 支持多策略、多参数的对比回测

## 📁 系统架构

```
qlib_integration/backtest/
├── __init__.py                 # 模块入口
├── data_adapter.py            # SQLite数据库 -> Qlib格式适配器
├── strategy_adapter.py        # 策略适配器工厂
├── stocktrader_strategy.py    # 统一策略实现
├── chinese_exchange.py        # 中国A股交易所模拟
├── backtest_runner.py         # 回测运行器主入口
├── example_usage.py          # 使用示例和演示
└── config/
    └── a_share_config.yaml   # A股市场配置文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install qlib pyyaml

# 确保数据库存在
python3 -c "from data_adapter.database_manager import DatabaseManager; print(DatabaseManager().get_database_stats())"
```

### 2. 简单回测

```python
from qlib_integration.backtest.backtest_runner import run_simple_backtest

# 运行BBI+KDJ策略回测
result = run_simple_backtest(
    strategies=['bbikdj'],
    start_date='2024-01-01',
    end_date='2024-06-30',
    initial_cash=1000000
)

print(f"年化收益率: {result['annual_return']:.2%}")
print(f"夏普比率: {result['sharpe_ratio']:.2f}")
```

### 3. 高级回测

```python
from qlib_integration.backtest import BacktestRunner

runner = BacktestRunner()

# 自定义策略配置
strategy_config = {
    'strategies': ['bbikdj', 'breakout'],
    'max_positions': 10,
    'position_size': 0.1,
    'stop_loss': 0.08,
    'take_profit': 0.15
}

result = runner.run_backtest(
    strategy_config=strategy_config,
    save_results=True  # 自动生成报告
)
```

### 4. 多策略对比

```python
# 定义多个策略
strategies = [
    {'strategies': ['bbikdj'], 'max_positions': 10},
    {'strategies': ['breakout'], 'max_positions': 8},  
    {'strategies': ['bbikdj', 'breakout'], 'max_positions': 12}
]

comparison = runner.run_multi_strategy_comparison(strategies)
```

## 📊 支持的策略

### 1. 少负战法 (bbikdj)
- **核心逻辑**: BBI多空线 + KDJ超买超卖
- **适用市场**: 震荡上升市
- **特点**: 稳健型，适合中长期持有

### 2. 补票战法 (bbilongshort)  
- **核心逻辑**: BBI短期与长期趋势对比
- **适用市场**: 趋势明确市
- **特点**: 趋势跟随，止损及时

### 3. TePu战法 (breakout)
- **核心逻辑**: 价格突破 + 成交量确认
- **适用市场**: 强势突破市
- **特点**: 进攻型，追求高收益

### 4. 填坑战法 (peak)
- **核心逻辑**: 低点反弹 + 估值修复
- **适用市场**: 底部反转市  
- **特点**: 价值型，等待反弹机会

## ⚙️ 配置说明

### 主要配置项

```yaml
# 策略配置
strategy:
  strategies: ["bbikdj", "breakout"]  # 启用策略
  max_positions: 10                   # 最大持仓
  position_size: 0.1                  # 单仓占比
  stop_loss: 0.08                     # 止损线
  take_profit: 0.15                   # 止盈线

# 交易所配置  
exchange:
  limit_threshold: 0.095              # 涨跌停阈值
  open_cost: 0.0003                   # 买入成本
  close_cost: 0.0013                  # 卖出成本
  t_plus_1: true                      # T+1限制

# 股票池配置
universe:
  min_trading_days: 100               # 最少交易天数
  exclude_st: true                    # 排除ST
  max_stocks: 2000                    # 最大股票数
```

## 📈 性能指标

系统提供以下性能分析指标：

### 收益指标
- **总收益率**: 回测期间累计收益
- **年化收益率**: 按年化计算的收益率
- **超额收益**: 相对基准的超额收益

### 风险指标  
- **最大回撤**: 从高点到低点的最大跌幅
- **夏普比率**: 风险调整后收益指标
- **波动率**: 收益率的标准差

### 交易指标
- **胜率**: 盈利交易占比
- **盈亏比**: 平均盈利/平均亏损
- **换手率**: 交易活跃程度

## 🛡️ 风险管理

### 仓位控制
- **最大持仓数量**: 控制组合集中度风险
- **单仓占比限制**: 防止单股过度集中
- **现金留存**: 保证流动性需求

### 止损止盈
- **固定止损**: 基于入场价格的固定比例止损
- **动态止盈**: 根据市场情况调整止盈点
- **时间止损**: 长期无收益的强制平仓

### 市场风险控制
- **涨跌停限制**: 自动识别并避免涨跌停股票
- **T+1限制**: 严格执行A股T+1交易规则  
- **停牌处理**: 自动检测和处理停牌股票

## 📊 报告生成

系统自动生成多格式回测报告：

### JSON格式
- **详细数据**: 包含所有回测数据和指标
- **机器可读**: 适合后续程序分析
- **存储位置**: `reports/qlib_backtest/*.json`

### Markdown格式
- **可读性强**: 适合人工查看和分享
- **图表展示**: 包含性能图表和分析
- **存储位置**: `reports/qlib_backtest/*.md`

## 🔧 高级功能

### 1. 参数优化
支持网格搜索和贝叶斯优化：

```python
# 参数优化示例
param_grid = {
    'position_size': [0.08, 0.10, 0.12],
    'stop_loss': [0.06, 0.08, 0.10],
    'min_score': [60, 70, 80]
}

optimizer = ParameterOptimizer(runner)
best_params = optimizer.optimize(param_grid)
```

### 2. 滚动回测
模拟真实交易的时序性：

```python
# 滚动窗口回测
rolling_results = runner.run_rolling_backtest(
    window_size='6M',    # 6个月滚动窗口
    step_size='1M',      # 1个月步进
    min_periods=100      # 最少交易天数
)
```

### 3. Monte Carlo分析
评估策略稳定性：

```python
# 蒙特卡洛模拟
mc_results = runner.run_monte_carlo_analysis(
    n_simulations=1000,
    bootstrap_method='block'
)
```

## 🎯 使用场景

### 1. 策略研发
- **快速验证**: 新策略想法的快速回测验证
- **参数调优**: 系统性的参数优化和敏感性分析
- **风险评估**: 策略在不同市场环境下的表现

### 2. 组合管理
- **多策略配置**: 不同策略的权重分配优化
- **风险预算**: 基于风险预算的仓位分配
- **再平衡**: 动态调仓频率和条件优化

### 3. 业绩归因
- **收益来源**: 分解策略收益的来源结构
- **风险归因**: 识别主要风险来源和控制点
- **市场归因**: 分析市场环境对策略的影响

## 🚦 注意事项

### 数据质量
- **数据完整性**: 确保回测期间数据完整无缺失
- **生存者偏差**: 注意已退市股票的处理
- **前视偏差**: 避免使用未来信息

### 回测局限性
- **流动性假设**: 回测假设所有交易均能成交
- **冲击成本**: 实际交易可能面临额外的市场冲击
- **制度变化**: 监管政策变化对策略的影响

### 过拟合风险
- **样本外测试**: 使用训练期外的数据验证
- **参数稳定性**: 避免过度参数调优
- **策略简单性**: 保持策略逻辑的可解释性

## 📚 扩展开发

### 添加新策略
```python
# 1. 继承StockTraderStrategy类
class MyCustomStrategy(StockTraderStrategy):
    def _evaluate_custom_signal(self, data):
        # 实现自定义信号逻辑
        pass

# 2. 注册到策略工厂
StrategyAdapter.STRATEGY_MAP['custom'] = MyCustomStrategy
```

### 自定义交易成本
```python
# 自定义成本计算
class CustomExchange(ChineseAShareExchange):
    def calculate_trading_cost(self, stock_id, amount, price, direction):
        # 实现自定义成本计算
        return custom_cost
```

### 添加新指标
```python
# 在performance_metrics中添加新指标
def custom_metric(returns):
    # 计算自定义指标
    return metric_value
```

## 📞 技术支持

### 常见问题
1. **安装问题**: 确保Python版本 ≥ 3.8，安装所有依赖
2. **内存不足**: 减少股票池大小或调整缓存配置
3. **数据缺失**: 检查数据库完整性和数据更新

### 调试技巧
- 设置`log_level: DEBUG`获取详细日志
- 使用`save_intermediate: true`保存中间结果
- 减小回测时间范围进行快速测试

### 性能优化
- 使用SSD存储提升数据库访问速度
- 调整`parallel_workers`参数利用多核CPU
- 启用数据缓存减少重复计算

---

## 📈 系统集成完成总结

✅ **已实现功能**:
1. **数据适配器**: SQLite数据库与Qlib格式无缝转换
2. **策略集成**: 4个核心选股策略的完整Qlib适配  
3. **交易所模拟**: 中国A股特性的专业级交易模拟
4. **回测引擎**: 完整的回测运行和结果分析框架
5. **配置系统**: 灵活的YAML配置和参数优化
6. **报告生成**: 自动化的多格式专业报告

🎯 **关键优势**:
- **专业级回测**: 达到机构级别的回测精度和功能
- **A股特化**: 完美适配中国股市的特殊规则
- **高可扩展性**: 易于添加新策略和自定义功能
- **生产就绪**: 可直接用于实盘交易系统开发

🚀 **使用建议**:
1. 先运行`example_usage.py`熟悉系统功能
2. 根据需求调整`a_share_config.yaml`配置
3. 使用长期数据进行充分的样本外测试
4. 结合多种分析方法验证策略有效性

**本集成系统将StockTradebyZ提升到了专业量化平台的级别，为A股程序化交易提供了工业级的解决方案。**

🤖 *Generated with Claude Code - StockTradebyZ Team*