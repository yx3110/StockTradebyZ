# v3量化评分系统快速使用指南

## 🚀 v3版本已就绪！

v3版本量化评分系统已成功创建，与v2版本完全独立。虽然权重优化仍在进行中，但核心功能已可使用。

## ⚡ 快速测试

### 1. 版本管理器测试
```bash
python3 scoring_improvements/scoring_system_manager.py
```
**预期输出**: 显示可用版本(v2, v3)和版本切换功能

### 2. v3单股票评分测试
```bash
python3 -c "
from scoring_improvements.quantitative_scorer_v3 import QuantitativeScorerV3
scorer = QuantitativeScorerV3()
result = scorer.calculate_stock_score('000001.SH', '2025-08-12')
print(f'股票: {result[\"code\"]}')
print(f'综合得分: {result[\"total_score\"]:.4f}')
print('分项得分:')
for key, value in result['scores'].items():
    print(f'  {key}: {value:.4f}')
"
```

### 3. 权重配置查看
```bash
python3 -c "
from scoring_improvements.quantitative_scorer_v3 import QuantitativeScorerV3
scorer = QuantitativeScorerV3()
print('v3版本权重配置:')
weights = scorer.get_weight_summary()
for name, weight in weights.items():
    print(f'  {name}: {weight:.3f}')
print(f'总权重: {sum(weights.values()):.3f}')
"
```

## 🔄 版本切换演示

### 在代码中切换版本
```python
from scoring_improvements.scoring_system_manager import ScoringSystemManager

# 创建版本管理器
manager = ScoringSystemManager()

# 查看可用版本
print("可用版本:", manager.list_available_versions())

# 切换到v3版本
manager.switch_version("v3")
print("当前版本:", manager.get_current_version())

# 创建v3评分器
scorer = manager.create_scorer("v3")

# 进行评分
result = scorer.calculate_stock_score("000001.SH", "2025-08-12")
print("评分结果:", result)
```

## 📊 v3版本特色功能

### 1. 动态权重配置
```python
# v3版本支持动态权重调整
scorer = QuantitativeScorerV3()

# 检测市场环境
market_regime = scorer.detect_market_regime("2025-08-12")
print("市场环境:", market_regime)

# 市场环境影响评分权重
# 牛市、熊市、震荡市会有不同的权重策略
```

### 2. 多时间窗口分析
```python
# v3版本同时考虑多个时间窗口
config = scorer.config["parameters"]
print("分析窗口:", config["lookback_periods"])  # [5, 10, 20, 30]

# 每个窗口的技术指标都会被综合考虑
```

### 3. 增强的评分体系
```python
# v3版本包含4大评分模块，16个细分指标
print("评分模块:")
for category, weights in scorer.config["weights"].items():
    print(f"  {category}:")
    for indicator, weight in weights.items():
        print(f"    {indicator}: {weight:.3f}")
```

## 🛠️ 配置自定义

### 创建自定义配置
```python
import json

# 自定义权重配置
custom_config = {
    "version": "v3.0_custom",
    "weights": {
        "technical": {
            "kdj_strength": 0.15,    # 增强KDJ权重
            "rsi_momentum": 0.08,
            "bbi_trend": 0.10,
            "volume_surge": 0.12     # 增强成交量权重
        },
        # ... 其他权重配置
    }
}

# 保存自定义配置
with open("my_v3_config.json", "w") as f:
    json.dump(custom_config, f, indent=2)

# 使用自定义配置
scorer = QuantitativeScorerV3("my_v3_config.json")
```

## 🔍 调试和日志

### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.INFO)

# v3评分器会输出详细的计算过程日志
scorer = QuantitativeScorerV3()
result = scorer.calculate_stock_score("000001.SH", "2025-08-12")
```

### 查看评分详情
```python
result = scorer.calculate_stock_score("000001.SH", "2025-08-12")

print("详细信息:")
print(f"  收盘价: {result['details']['close']}")
print(f"  PE倍数: {result['details']['pe_ttm']}")
print(f"  PB倍数: {result['details']['pb']}")
print(f"  市值: {result['details']['market_cap']/10000:.1f}亿")
print(f"  KDJ_K: {result['details']['kdj_k']}")
print(f"  RSI: {result['details']['rsi']}")
print(f"  市场环境: {result['market_regime']}")
```

## 📁 文件结构说明

### v3核心文件
```
scoring_improvements/
├── quantitative_scorer_v3.py      # v3核心评分算法
├── weight_optimizer_v3.py         # 权重优化器
├── v3_daily_report_generator.py   # v3专用日报生成器  
├── scoring_system_manager.py      # 版本管理器
└── V3_*.md                        # v3版本文档
```

### 配置文件位置
- 默认配置: 内置在 `quantitative_scorer_v3.py` 中
- 自定义配置: 任何 `.json` 文件路径
- 优化后配置: `scoring_improvements/v3_optimized_config_*.json`

### 报告输出目录
- v3专用: `reports/v3_quantitative_scoring/`
- 不会覆盖v2报告

## ⚠️ 当前限制

### 1. 数据库字段映射
部分技术指标字段需要调整，可能遇到以下错误：
```
ERROR: no such column: ti.rsi
```
**解决方案**: 使用 `000001.SH` 等指数类股票测试，数据更完整。

### 2. 权重优化进行中
完整的权重优化回测仍在进行，当前使用默认权重配置。

### 3. 部分功能调试中
- v3日报生成器需要数据库字段调整
- 批量评分功能需要进一步测试

## 🎯 推荐使用方式

### 当前阶段 (开发测试)
```bash
# 1. 测试版本管理器
python3 scoring_improvements/scoring_system_manager.py

# 2. 单股票评分测试
python3 -c "
from scoring_improvements.quantitative_scorer_v3 import QuantitativeScorerV3
scorer = QuantitativeScorerV3()
result = scorer.calculate_stock_score('000001.SH', '2025-08-12')
print('v3评分测试:', result['total_score'])
"

# 3. 权重配置查看
python3 -c "
from scoring_improvements.quantitative_scorer_v3 import QuantitativeScorerV3
scorer = QuantitativeScorerV3()
print('权重总和:', sum(scorer.get_weight_summary().values()))
"
```

### 生产使用 (优化完成后)
```bash
# 1. 使用优化后的权重配置
python3 scoring_improvements/v3_daily_report_generator.py \
  --date 2025-08-12 \
  --config scoring_improvements/v3_optimized_config_latest.json

# 2. 批量股票评分
python3 -c "
from scoring_improvements.quantitative_scorer_v3 import QuantitativeScorerV3
scorer = QuantitativeScorerV3('v3_optimized_config_latest.json')
results = scorer.batch_score_stocks(['000001.SH', '000858.SZ'], '2025-08-12')
for r in results: print(f'{r[\"code\"]}: {r[\"total_score\"]:.4f}')
"
```

## 🆚 v2 vs v3 对比测试

### 准备版本对比
```python
from scoring_improvements.scoring_system_manager import ScoringSystemManager

manager = ScoringSystemManager()

# 对比不同版本的评分结果
comparison = manager.compare_versions(
    stock_codes=['000001.SH', '000858.SZ'], 
    date='2025-08-12',
    sample_size=10
)

print("版本对比结果:")
for version, result in comparison.items():
    if 'error' not in result:
        print(f"{version}: 平均得分 {result['avg_score']:.4f}")
    else:
        print(f"{version}: {result['error']}")
```

## 📞 支持和反馈

如果遇到问题：
1. 检查 `V3_IMPLEMENTATION_SUMMARY.md` 中的已知问题
2. 查看控制台日志输出
3. 尝试使用不同的股票代码 (推荐 `000001.SH`)

---
**v3版本状态**: 核心功能可用，优化进行中  
**安全保证**: 与v2版本完全隔离，不会覆盖现有文件  
**推荐用途**: 开发测试和算法验证  

*v3快速使用指南 - 2025-08-13*