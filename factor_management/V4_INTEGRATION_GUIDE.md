# V4挤压动量系统集成指南

## 🎯 V4系统概述

V4挤压动量增强评分系统是因子管理框架的最新创新，基于John Carter的**TTM Squeeze指标**，专门用于识别从低波动到高波动的市场转换点。

### 🔥 V4核心特色
- **挤压动量维度**(20%权重) - 全新的波动率分析维度
- **技术指标优化**(50%权重) - 从65%调整为50%，为挤压动量让出空间
- **突破预警能力** - 提前识别横盘整理后的突破时机
- **假突破过滤** - 多维度验证降低误报率

## 🏗️ V4系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    V4挤压动量增强评分系统                        │
├─────────────────────────────────────────────────────────────────┤
│  📊 技术指标(50%)        🆕 挤压动量(20%)      📈 基本面(8%)     │
│  ├─ KDJ强度(15%)        ├─ 挤压状态(5%)       ├─ PE估值(2%)     │
│  ├─ RSI动量(14%)        ├─ 挤压释放(6%)       ├─ PB估值(2%)     │
│  ├─ BBI趋势(10%)        ├─ 动量方向(5%)       ├─ 市值因子(2%)   │
│  └─ 成交量异动(11%)     └─ 动量加速度(4%)     └─ 换手率活跃度(2%)│
├─────────────────────────────────────────────────────────────────┤
│  📈 市场表现(18%)                  🌍 市场环境(4%)                │
│  ├─ 价格动量(13%)                  ├─ 市场贝塔(1%)                │
│  ├─ 相对强度(3%)                   ├─ 板块轮动(1.5%)             │
│  └─ 波动率风险(2%)                 └─ 流动性(1.5%)               │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 导入V4组件

```python
from factor_management.v4_factor_extractor import V4FactorExtractor
from factor_management.stock_selector_adapter import StockSelectorAdapter

# 初始化V4系统
v4_extractor = V4FactorExtractor()
adapter = StockSelectorAdapter()
```

### 2. 提取V4因子数据

```python
# 单只股票V4因子提取
stock_code = '000001'
v4_factors = v4_extractor.extract_factors_for_stock(
    stock_code, '2025-08-01', '2025-08-18'
)

print("V4因子列表:")
v4_factor_names = [col for col in v4_factors.columns if col.startswith('v4_')]
for factor in v4_factor_names:
    print(f"  - {factor}")

# 批量提取V4因子
stock_codes = ['000001', '000002', '000858']
batch_v4_data = v4_extractor.batch_extract_factors(
    stock_codes, '2025-08-01', '2025-08-18'
)

# 保存到数据库
v4_extractor.save_factors_to_database(batch_v4_data)
```

### 3. 运行V4选股策略

```python
# V4综合评分策略
v4_comprehensive_result = adapter.run_stock_selection(
    stock_codes, '2025-08-18', 
    strategy='v4_comprehensive', 
    top_n=20
)

print(f"V4综合评分选出 {len(v4_comprehensive_result['selected_stocks'])} 只股票")

# V4挤压动量专项策略
v4_squeeze_result = adapter.run_stock_selection(
    stock_codes, '2025-08-18',
    strategy='v4_squeeze_momentum',
    top_n=10
)

print(f"V4挤压动量选出 {len(v4_squeeze_result['selected_stocks'])} 只股票")

# 生成报告
adapter.generate_selection_report(
    v4_comprehensive_result,
    "reports/factor_management/V4综合评分报告_20250818.md"
)
```

### 4. 创建包含V4因子的统一数据集

```python
# 创建完整的统一因子数据集
unified_data = adapter.create_unified_factor_dataset(
    stock_codes, '2025-08-01', '2025-08-18',
    include_v2=True,        # V2因子
    include_v4=True,        # V4因子 🆕
    include_strategies=True  # 4个策略因子
)

print(f"统一数据集形状: {unified_data.shape}")

# 分析因子结构
v2_factors = [col for col in unified_data.columns if col.startswith('v2_')]
v4_factors = [col for col in unified_data.columns if col.startswith('v4_')]

print(f"V2因子: {len(v2_factors)} 个")
print(f"V4因子: {len(v4_factors)} 个")
print(f"总因子: {len(v2_factors) + len(v4_factors)} 个")
```

## 🔬 V4核心因子详解

### 1. 挤压动量因子(20%权重)

V4的核心创新，基于TTM Squeeze指标：

```python
# 挤压状态因子 (5%权重)
v4_squeeze_state = factor_data['v4_squeeze_state']
# 布林带在肯特纳通道内收窄，市场进入低波动蓄势期
# 评分逻辑: 挤压状态给25分奖励，长期挤压(>10天)额外10分

# 挤压释放因子 (6%权重) - 核心信号
v4_squeeze_release = factor_data['v4_squeeze_release']  
# 布林带突破肯特纳通道，预示波动率扩张
# 评分逻辑: 刚释放100分，近期释放60分，长期挤压40分

# 动量方向因子 (5%权重)
v4_momentum_direction = factor_data['v4_momentum_direction']
# 线性回归斜率判断突破方向
# 评分逻辑: 50基础分 ± 动量强度*100

# 动量加速度因子 (4%权重)
v4_momentum_acceleration = factor_data['v4_momentum_acceleration']
# 动量变化率，确保趋势持续性
# 评分逻辑: 50基础分 ± 加速度*50，一致性奖励20分
```

### 2. 技术指标因子(50%权重)

继承V3逻辑，调整权重配置：

```python
# KDJ强度因子 (15%权重) - 从V3的20%降低
v4_kdj_strength = factor_data['v4_kdj_strength']
# KDJ≤20给100分，超卖区间高分奖励

# RSI动量因子 (14%权重) - 从V3的20%降低  
v4_rsi_momentum = factor_data['v4_rsi_momentum']
# RSI≤30给100分，健康区间(30-70)高分

# BBI趋势因子 (10%权重) - 从V3的15%降低
v4_bbi_trend = factor_data['v4_bbi_trend']
# 价格/BBI比值，≥1.05给100分

# 成交量异动因子 (11%权重) - 从V3的10%提高
v4_volume_surge = factor_data['v4_volume_surge']
# 近5日/历史20日成交量比，≥3倍给100分
```

### 3. V4综合评分

```python
# V4综合评分计算
v4_comprehensive_score = factor_data['v4_comprehensive_score']

# 权重配置
V4_WEIGHTS = {
    # 技术指标 (50%)
    'kdj_strength': 0.15,
    'rsi_momentum': 0.14, 
    'bbi_trend': 0.10,
    'volume_surge': 0.11,
    
    # 🆕 挤压动量 (20%)
    'squeeze_state': 0.05,
    'squeeze_release': 0.06,    # 核心信号
    'momentum_direction': 0.05,
    'momentum_acceleration': 0.04,
    
    # 基本面 (8%)
    'pe_valuation': 0.02,
    'pb_valuation': 0.02,
    'market_cap_factor': 0.02,
    'turnover_activity': 0.02,
    
    # 市场表现 (18%)
    'price_momentum': 0.13,
    'relative_strength': 0.03,
    'volatility_risk': 0.02,
    
    # 市场环境 (4%)
    'market_beta': 0.01,
    'sector_rotation': 0.015,
    'liquidity': 0.015
}
```

## 🎯 V4策略应用

### 策略1: V4综合评分 (`v4_comprehensive`)

**使用场景**: 全面评估，平衡各维度因子
```python
result = adapter.run_stock_selection(
    stock_codes, trade_date,
    strategy='v4_comprehensive',
    top_n=50
)
```

**评分标准**:
- A+级(≥90分): 全方位优质标的
- A级(85-90分): 综合表现优秀
- B+级(75-85分): 值得关注的候选

### 策略2: V4挤压动量 (`v4_squeeze_momentum`)

**使用场景**: 专注突破时机，横盘整理后的方向性选择
```python  
result = adapter.run_stock_selection(
    stock_codes, trade_date,
    strategy='v4_squeeze_momentum',
    top_n=20
)
```

**信号强度**:
- 强信号(≥80分): 刚发生挤压释放
- 中信号(60-80分): 近期有释放信号
- 弱信号(<60分): 长期挤压，蓄势待发

## 📊 V4挤压动量实战指南

### 1. 挤压状态识别

```python
# 筛选长期挤压状态的股票
squeeze_candidates = unified_data[
    unified_data['v4_squeeze_state'] >= 70  # 高分挤压状态
]

print(f"发现 {len(squeeze_candidates)} 只长期挤压状态股票")

# 分析挤压特征
for _, row in squeeze_candidates.head().iterrows():
    print(f"{row['stock_code']}: 挤压状态 {row['v4_squeeze_state']:.1f}分")
```

### 2. 挤压释放信号捕捉

```python
# 筛选挤压释放信号
release_signals = unified_data[
    unified_data['v4_squeeze_release'] >= 80  # 强释放信号
]

print(f"发现 {len(release_signals)} 只挤压释放信号股票")

# 结合动量方向确认
bullish_releases = release_signals[
    release_signals['v4_momentum_direction'] >= 60  # 看涨动量
]

print(f"看涨方向的挤压释放: {len(bullish_releases)} 只")
```

### 3. 多时间框架分析

```python
# 分析不同时间框架的挤压动量效果
time_frames = [
    ('2025-08-01', '2025-08-05'),  # 短期
    ('2025-08-01', '2025-08-10'),  # 中期  
    ('2025-08-01', '2025-08-18')   # 长期
]

for start_date, end_date in time_frames:
    data = adapter.create_unified_factor_dataset(
        stock_codes, start_date, end_date, include_v4=True
    )
    
    avg_squeeze_release = data['v4_squeeze_release'].mean()
    print(f"{start_date}至{end_date}: 平均挤压释放 {avg_squeeze_release:.1f}分")
```

## ⚡ V4性能优化

### 1. 批量处理优化

```python
# 推荐: 使用批量处理
batch_size = 100
large_stock_pool = get_all_stocks()  # 假设4000只股票

for i in range(0, len(large_stock_pool), batch_size):
    batch_stocks = large_stock_pool[i:i+batch_size]
    
    # 批量提取V4因子
    batch_v4_data = v4_extractor.batch_extract_factors(
        batch_stocks, start_date, end_date, batch_size=batch_size
    )
    
    # 批量保存
    v4_extractor.save_factors_to_database(batch_v4_data)
    
    print(f"处理进度: {min(i+batch_size, len(large_stock_pool))}/{len(large_stock_pool)}")
```

### 2. 缓存机制

```python
from functools import lru_cache

# 缓存挤压动量计算结果
@lru_cache(maxsize=500)
def cached_squeeze_signals(stock_code, start_date, end_date):
    return v4_extractor.extract_factors_for_stock(stock_code, start_date, end_date)

# 使用缓存
cached_data = cached_squeeze_signals('000001', '2025-08-01', '2025-08-18')
```

### 3. 并行计算

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def extract_single_stock_v4(stock_code):
    return v4_extractor.extract_factors_for_stock(
        stock_code, start_date, end_date
    )

# 多线程处理
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(extract_single_stock_v4, code) 
               for code in stock_codes]
    results = [future.result() for future in futures]
```

## 📈 V4策略回测

### 1. 挤压释放策略回测

```python
def backtest_squeeze_release_strategy(start_date, end_date, stock_pool):
    """回测挤压释放策略"""
    
    results = []
    
    # 逐日回测
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    for current_date in date_range:
        if current_date.weekday() >= 5:  # 跳过周末
            continue
            
        date_str = current_date.strftime('%Y-%m-%d')
        
        # 运行V4挤压动量策略
        selection_result = adapter.run_stock_selection(
            stock_pool, date_str, strategy='v4_squeeze_momentum', top_n=10
        )
        
        selected_stocks = selection_result['selected_stocks']
        
        # 计算次日收益(这里简化处理)
        for stock in selected_stocks:
            next_day_return = calculate_next_day_return(stock['stock_code'], date_str)
            results.append({
                'date': date_str,
                'stock_code': stock['stock_code'],
                'squeeze_score': stock['score'],
                'next_day_return': next_day_return
            })
    
    return pd.DataFrame(results)

# 运行回测
backtest_results = backtest_squeeze_release_strategy(
    '2025-07-01', '2025-08-18', stock_codes
)

# 分析结果
avg_return = backtest_results['next_day_return'].mean()
win_rate = len(backtest_results[backtest_results['next_day_return'] > 0]) / len(backtest_results)

print(f"挤压释放策略回测结果:")
print(f"平均次日收益: {avg_return:.2%}")
print(f"胜率: {win_rate:.1%}")
```

### 2. 因子有效性分析

```python
# 分析V4各因子与未来收益的相关性
correlation_analysis = {}

v4_factors = ['v4_squeeze_state', 'v4_squeeze_release', 
              'v4_momentum_direction', 'v4_momentum_acceleration']

for factor in v4_factors:
    correlation = unified_data[factor].corr(
        unified_data['future_return_5d']  # 假设有5日后收益数据
    )
    correlation_analysis[factor] = correlation
    print(f"{factor}: {correlation:.3f}")

# 因子重要性排序
sorted_factors = sorted(correlation_analysis.items(), 
                       key=lambda x: abs(x[1]), reverse=True)

print("\nV4因子重要性排序:")
for factor, corr in sorted_factors:
    print(f"  {factor}: {corr:.3f}")
```

## 🔧 V4参数调优

### 挤压动量参数

```python
# 可调整的挤压动量参数
SQUEEZE_PARAMS = {
    'bb_length': 20,           # 布林带周期 (推荐: 15-25)
    'bb_multiplier': 2.0,      # 布林带倍数 (推荐: 1.8-2.2)
    'kc_length': 20,           # 肯特纳通道周期 (推荐: 15-25)
    'kc_multiplier': 1.5,      # 肯特纳通道倍数 (推荐: 1.2-2.0)
    'momentum_length': 20      # 动量计算周期 (推荐: 15-30)
}

# 创建自定义参数的V4提取器
custom_v4_extractor = V4FactorExtractor()
custom_v4_extractor.squeeze_calculator = SqueezeMomentumCalculator(
    bb_length=SQUEEZE_PARAMS['bb_length'],
    bb_multiplier=SQUEEZE_PARAMS['bb_multiplier'],
    kc_length=SQUEEZE_PARAMS['kc_length'],
    kc_multiplier=SQUEEZE_PARAMS['kc_multiplier'],
    momentum_length=SQUEEZE_PARAMS['momentum_length']
)
```

### 权重优化

```python
# A/B测试不同权重配置
WEIGHT_CONFIGS = {
    'conservative': {  # 保守配置，降低挤压动量权重
        'squeeze_momentum_total': 0.15,
        'technical_total': 0.55
    },
    'aggressive': {   # 激进配置，提高挤压动量权重
        'squeeze_momentum_total': 0.25,
        'technical_total': 0.45
    }
}

# 测试不同配置的效果
for config_name, weights in WEIGHT_CONFIGS.items():
    # 调整权重后运行选股
    results = test_weight_configuration(weights)
    print(f"{config_name}配置平均收益: {results['avg_return']:.2%}")
```

## 🧪 测试与验证

### 1. 运行完整测试

```bash
cd factor_management
python test_v4_integration.py
```

### 2. 单元测试

```python
def test_v4_squeeze_logic():
    """测试挤压动量逻辑"""
    v4_extractor = V4FactorExtractor()
    
    # 测试挤压状态识别
    factor_data = v4_extractor.extract_factors_for_stock(
        '000001', '2025-08-01', '2025-08-18'
    )
    
    assert 'v4_squeeze_state' in factor_data.columns
    assert 'v4_squeeze_release' in factor_data.columns
    assert 'v4_momentum_direction' in factor_data.columns
    assert 'v4_momentum_acceleration' in factor_data.columns
    
    # 检查评分范围
    assert factor_data['v4_squeeze_state'].min() >= 0
    assert factor_data['v4_squeeze_state'].max() <= 100
    
    print("✅ V4挤压逻辑测试通过")

# 运行测试
test_v4_squeeze_logic()
```

### 3. 数据质量检查

```python
def check_v4_data_quality():
    """检查V4数据质量"""
    
    # 检查因子数据完整性
    unified_data = adapter.create_unified_factor_dataset(
        ['000001', '000002'], '2025-08-15', '2025-08-18', include_v4=True
    )
    
    v4_factors = [col for col in unified_data.columns if col.startswith('v4_')]
    
    print(f"V4因子数量: {len(v4_factors)}")
    
    # 检查缺失值
    for factor in v4_factors:
        null_rate = unified_data[factor].isnull().sum() / len(unified_data)
        if null_rate > 0.1:  # 缺失率超过10%
            print(f"⚠️ {factor} 缺失率过高: {null_rate:.1%}")
        else:
            print(f"✅ {factor} 数据质量良好: 缺失率 {null_rate:.1%}")

# 运行数据质量检查
check_v4_data_quality()
```

## 📚 V4应用案例

### 案例1: 挤压释放捕捉大阳线

```python
# 寻找挤压释放后的大阳线机会
def find_squeeze_breakout_opportunities():
    stock_codes = ['000001', '000002', '000858', '002215']
    
    # 获取V4因子数据
    unified_data = adapter.create_unified_factor_dataset(
        stock_codes, '2025-08-10', '2025-08-18', include_v4=True
    )
    
    # 筛选条件
    opportunities = unified_data[
        (unified_data['v4_squeeze_release'] >= 80) &      # 强挤压释放
        (unified_data['v4_momentum_direction'] >= 70) &   # 看涨动量
        (unified_data['v4_volume_surge'] >= 70)           # 放量确认
    ]
    
    print(f"发现 {len(opportunities)} 个挤压突破机会:")
    for _, row in opportunities.iterrows():
        print(f"  {row['stock_code']}: 释放{row['v4_squeeze_release']:.0f}分, "
              f"动量{row['v4_momentum_direction']:.0f}分, "
              f"成交量{row['v4_volume_surge']:.0f}分")

find_squeeze_breakout_opportunities()
```

### 案例2: 多策略组合验证

```python
# 结合V2、V4和传统策略的组合选股
def multi_strategy_selection():
    stock_codes = get_stock_pool_top500()  # 获取500只活跃股票
    
    strategies = ['v2_composite', 'v4_comprehensive', 'combined']
    strategy_results = {}
    
    for strategy in strategies:
        result = adapter.run_stock_selection(
            stock_codes, '2025-08-18', strategy=strategy, top_n=20
        )
        strategy_results[strategy] = set([s['stock_code'] for s in result['selected_stocks']])
    
    # 找出多策略共同推荐的股票
    common_picks = strategy_results['v2_composite'] & strategy_results['v4_comprehensive'] & strategy_results['combined']
    
    print(f"多策略共同推荐股票 ({len(common_picks)} 只):")
    for stock_code in common_picks:
        print(f"  {stock_code}")
    
    return list(common_picks)

# 运行多策略选股
consensus_stocks = multi_strategy_selection()
```

## ⚠️ 风险提示与最佳实践

### 1. 挤压动量策略风险

- **市场环境依赖**: 震荡市场效果更佳，单边市场可能失效
- **假突破风险**: 需要结合成交量和基本面多重确认
- **参数敏感性**: 不同股票和时间段可能需要调整参数

### 2. 最佳实践建议

```python
# 建议的风险控制措施
RISK_CONTROLS = {
    'position_sizing': 0.05,        # 单股最大仓位5%
    'stop_loss': 0.08,             # 止损比例8%
    'min_squeeze_score': 70,       # 最低挤压释放分数
    'min_volume_ratio': 1.5,       # 最低成交量放大倍数
    'max_holdings': 10             # 最大持股数量
}

def apply_risk_controls(selected_stocks):
    """应用风险控制措施"""
    filtered_stocks = []
    
    for stock in selected_stocks:
        # 检查挤压释放分数
        if stock.get('squeeze_release_score', 0) < RISK_CONTROLS['min_squeeze_score']:
            continue
            
        # 检查成交量放大
        if stock.get('volume_ratio', 0) < RISK_CONTROLS['min_volume_ratio']:
            continue
            
        filtered_stocks.append(stock)
    
    # 限制持股数量
    return filtered_stocks[:RISK_CONTROLS['max_holdings']]
```

### 3. 监控指标

```python
# 建议监控的关键指标
MONITORING_METRICS = {
    'squeeze_success_rate': 0.6,     # 挤压释放成功率>60%
    'avg_holding_period': 5,         # 平均持股期5个交易日
    'max_drawdown': 0.15,            # 最大回撤<15%
    'sharpe_ratio': 1.5              # 夏普比率>1.5
}

def monitor_v4_performance(historical_trades):
    """监控V4策略表现"""
    squeeze_trades = [t for t in historical_trades if t['strategy'] == 'v4_squeeze_momentum']
    
    success_rate = len([t for t in squeeze_trades if t['return'] > 0]) / len(squeeze_trades)
    avg_return = np.mean([t['return'] for t in squeeze_trades])
    max_drawdown = calculate_max_drawdown(squeeze_trades)
    
    print(f"V4挤压动量策略监控指标:")
    print(f"  成功率: {success_rate:.1%}")
    print(f"  平均收益: {avg_return:.2%}")
    print(f"  最大回撤: {max_drawdown:.1%}")
```

## 🔮 V4未来发展方向

### 1. 技术增强

- **多时间框架融合**: 结合分钟级、小时级、日级挤压动量
- **机器学习优化**: 使用LSTM预测挤压释放时机
- **另类数据集成**: 结合情绪数据、资金流数据

### 2. 策略扩展

- **行业轮动版本**: 针对不同行业优化挤压参数
- **市值分层策略**: 大中小盘差异化挤压动量策略
- **跨市场应用**: 港股、美股挤压动量策略

### 3. 风险管理升级

- **动态止损**: 基于挤压状态调整止损位
- **仓位管理**: 根据挤压强度动态调整仓位
- **对冲策略**: 挤压动量多空组合策略

---

## 🎉 总结

V4挤压动量增强评分系统成功将TTM Squeeze指标集成到因子管理框架，为A股量化交易带来了全新的维度。通过本指南，您可以：

✅ **掌握V4系统的核心原理和技术架构**  
✅ **熟练使用V4因子提取和选股功能**  
✅ **理解挤压动量策略的适用场景和风险**  
✅ **应用最佳实践进行参数优化和风险控制**  

V4系统不仅提升了选股的准确性，更为未来的AI增强和深度学习应用奠定了坚实基础。

**🚀 开始您的V4挤压动量交易之旅吧！**