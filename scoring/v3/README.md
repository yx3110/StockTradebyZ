# 增强量化评分系统 v2.0

基于相关性分析结果的全面重构量化评分系统，实现多因子评分、动态权重调整和实时监控。

## 🎯 核心改进

### 评分体系升级
- **技术面 (30%)**：趋势强度 + 动量分析 + 自适应波动率
- **基本面 (35%)**：估值因子 + 盈利能力 + 财务健康度
- **资金面 (20%)**：成交量分析 + 流动性评估
- **市场面 (15%)**：行业比较 + 市场情绪

### 智能特性
- 🧠 **动态权重**：根据市场状态自动调整各因子权重
- 📊 **多周期验证**：技术指标参数自适应市场波动
- 🎯 **行业轮动**：相对行业表现的量化评估
- 📈 **情绪识别**：基于市场宽度的情绪量化

## 📁 项目结构

```
scoring_improvements/
├── __init__.py                    # 包初始化
├── README.md                      # 项目说明
├── core_framework.py              # 核心架构
├── fundamental_factors.py         # 基本面因子
├── technical_factors.py           # 技术面因子
├── capital_market_factors.py      # 资金面和市场面因子
├── backtest_framework.py          # 回测框架
├── enhanced_scoring_system.py     # 主系统
├── run_enhanced_scoring.py        # 快速启动脚本
└── sample_config.json            # 配置模板
```

## 🚀 快速开始

### 1. 环境准备

确保已安装所需依赖：
```bash
pip install pandas numpy scipy matplotlib seaborn sqlite3
```

### 2. 运行日常评分

```bash
# 使用默认参数运行最新交易日评分
python scoring_improvements/enhanced_scoring_system.py --mode score

# 指定日期和参数
python scoring_improvements/enhanced_scoring_system.py \
    --mode score \
    --date 2025-08-13 \
    --top-n 30 \
    --min-score 75
```

### 3. 运行历史回测

```bash
# 运行2年期回测
python scoring_improvements/enhanced_scoring_system.py \
    --mode backtest \
    --start-date 2023-01-01 \
    --end-date 2025-07-31
```

### 4. 监控系统表现

```bash
# 查看最近30天表现
python scoring_improvements/enhanced_scoring_system.py --mode monitor
```

## 📊 系统配置

### 默认权重配置
```json
{
  "technical_weight": 0.30,      // 技术面权重
  "fundamental_weight": 0.35,    // 基本面权重  
  "capital_weight": 0.20,        // 资金面权重
  "market_weight": 0.15,         // 市场面权重
  "volatility_threshold": 0.02,  // 波动率阈值
  "bull_market_threshold": 0.05, // 牛市判断阈值
  "bear_market_threshold": -0.05 // 熊市判断阈值
}
```

### 自定义配置
```bash
# 使用自定义配置文件
python scoring_improvements/enhanced_scoring_system.py \
    --mode score \
    --config my_config.json
```

## 🔧 核心模块说明

### 1. 基本面因子 (`fundamental_factors.py`)

**估值因子**
- PE、PB、PS相对行业分位数
- 动态估值区间判断
- 历史估值回归分析

**盈利因子**
- ROE、ROA盈利能力评估
- 净利润和营收增长率
- 盈利质量和稳定性

**财务健康度**
- 资产负债率控制
- 流动性比率分析
- 现金流健康度

### 2. 技术面因子 (`technical_factors.py`)

**趋势强度**
- ADX趋势力度指标
- 趋势一致性评估
- 多周期趋势确认

**动量分析**
- 自适应RSI周期
- MACD信号优化
- KDJ多重确认
- 均线系统评分

**波动率分析**
- 历史波动率计算
- ATR波动率评估
- 市场状态适应性调整

### 3. 资金面因子 (`capital_market_factors.py`)

**成交量分析**
- 量比计算和评分
- 量价一致性检验
- 放量突破识别

**流动性评估**
- 换手率健康度
- 冲击成本评估
- 流动性深度分析

### 4. 市场面因子 (`capital_market_factors.py`)

**行业比较**
- 相对行业强度
- 行业轮动识别
- 行业景气度评估

**市场情绪**
- 市场宽度指标
- 涨跌家数比
- 极端情绪识别

## 📈 回测系统

### 回测配置
```python
BacktestConfig(
    start_date="2023-01-01",
    end_date="2025-07-31", 
    initial_capital=1000000,    # 初始资金
    max_positions=10,           # 最大持仓数
    position_size=0.1,          # 单只仓位
    rebalance_frequency=5,      # 调仓频率(天)
    score_threshold=75.0,       # 评分阈值
    stop_loss=-0.08,            # 止损比例
    take_profit=0.15            # 止盈比例
)
```

### 回测指标
- **收益指标**：总收益率、年化收益率、夏普比率
- **风险指标**：最大回撤、波动率、风险调整收益
- **交易统计**：胜率、盈亏比、平均持仓天数
- **评分有效性**：评分与收益相关性分析

## 📊 输出报告

### 日报格式 (`enhanced_scoring_report_YYYYMMDD.md`)
- 系统配置信息
- TOP N推荐股票
- 各维度得分分析
- 市场状态判断
- 投资建议分级

### 回测报告 (`backtest_report_YYYYMMDD_HHMM.md`)
- 回测配置参数
- 收益风险指标
- 交易详细统计
- 评分有效性分析
- 策略改进建议

## ⚠️ 使用注意事项

### 数据依赖
- 确保 `data_adapter/stock_data.db` 数据完整
- 需要基本面、技术指标、成交量等多维度数据
- 建议定期更新数据以保证评分准确性

### 系统限制
- 评分系统基于历史数据，不能预测未来
- 需要结合市场判断和风险管理
- 建议分散投资，控制单只股票风险

### 性能优化
- 大批量评分时建议分批处理
- 可通过调整参数平衡准确性与计算速度
- 历史回测耗时较长，建议合理设置时间范围

## 🔄 未来改进方向

### 短期优化
- [ ] 增加更多行业因子
- [ ] 优化技术指标参数
- [ ] 加入宏观经济因子
- [ ] 完善风险控制机制

### 中期规划
- [ ] 机器学习因子挖掘
- [ ] 高频数据集成
- [ ] 情绪因子扩展
- [ ] 组合优化算法

### 长期愿景
- [ ] 实时交易系统集成
- [ ] 多资产类别扩展
- [ ] AI驱动的因子进化
- [ ] 云端部署和服务化

## 📞 技术支持

如有问题或建议，请通过以下方式联系：
- GitHub Issues: 项目问题反馈
- 技术文档: 详见各模块代码注释
- 配置调优: 参考 `sample_config.json`

---
🤖 **Enhanced Quantitative Scoring System v2.0**  
*基于Claude Code的智能量化评分解决方案*