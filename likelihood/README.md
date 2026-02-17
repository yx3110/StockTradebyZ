# 股票相似度回测系统 (Stock Similarity Backtest System)

## 🎯 项目简介

基于时间序列相似度算法的股票走势预测系统。通过在历史数据中查找与目标股票近期走势相似的K线模式，分析这些相似模式的后续表现，从而预测目标股票的可能走势。

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行单只股票的相似度搜索
python main.py --stock 000001.SZ --window 30 --date 2025-08-08

# 3. 批量回测
python batch_backtest.py --config configs/default_config.yaml

# 4. 生成报告
python generate_report.py --date 2025-08-08
```

## 📊 核心功能

### 1. 相似度搜索算法
- **Matrix Profile**: 快速全局搜索，O(n log n)复杂度
- **DTW (Dynamic Time Warping)**: 处理时间伸缩的相似性
- **MASS**: 单次查询优化算法
- **集成方法**: 多算法加权投票

### 2. 多维度分析
- 价格走势相似度
- 成交量模式匹配
- 价量关系一致性
- 技术指标相似度

### 3. 回测与评估
- 多时间窗口评估 (5/10/20/60日)
- 统计显著性检验
- 风险调整收益分析
- 交易成本考虑

### 4. 报告生成
- 相似K线对比图
- 收益分布统计
- 参数敏感性分析
- 风险评估报告

## 📁 项目结构

```
likelihood/
├── algorithms/           # 相似度算法实现
│   ├── matrix_profile.py
│   ├── dtw.py
│   ├── mass.py
│   └── search_engine.py
├── backtest/            # 回测框架
│   ├── backtest_engine.py
│   └── metrics.py
├── data_preprocessing/  # 数据处理
│   ├── data_loader.py
│   └── feature_engineering.py
├── reports/            # 报告生成
│   ├── analyzer.py
│   └── report_generator.py
├── configs/            # 配置文件
│   └── default_config.yaml
├── tests/              # 测试用例
│   └── test_*.py
└── utils/              # 工具函数
    └── helpers.py
```

## 🔧 配置说明

### 基础配置 (configs/default_config.yaml)
```yaml
similarity:
  window_length: 30      # 窗口长度
  methods: ['matrix_profile', 'dtw']  # 使用的算法
  top_k: 10             # 返回最相似的K个结果

backtest:
  horizons: [5, 10, 20, 60]  # 评估时间窗口
  transaction_cost: 0.001     # 交易成本
  
filters:
  min_volume: 100000000       # 最小成交额
  exclude_zone: 30            # 排除带
```

## 📈 使用示例

### 单只股票分析
```python
from likelihood import SimilaritySearchEngine

# 初始化搜索引擎
engine = SimilaritySearchEngine(config_path='configs/default_config.yaml')

# 搜索相似模式
results = engine.search_similar_patterns(
    stock_code='000001.SZ',
    query_date='2025-08-08',
    window_length=30
)

# 分析结果
for match in results['similar_patterns']:
    print(f"股票: {match['stock']}, 相似度: {match['similarity_score']:.2f}")
    print(f"预期收益 (20日): {match['future_returns']['20d']:.2%}")
```

### 批量回测
```python
from likelihood import BatchBacktest

# 运行批量回测
backtest = BatchBacktest(config='configs/backtest_config.yaml')
results = backtest.run(
    stock_list=['000001.SZ', '000002.SZ', '600000.SH'],
    start_date='2024-01-01',
    end_date='2025-08-08'
)

# 生成报告
backtest.generate_report(results, output_dir='reports/backtest/')
```

## 📊 输出示例

### 相似度搜索结果
```json
{
    "query_stock": "000001.SZ",
    "query_period": "2025-07-08 to 2025-08-08",
    "top_matches": [
        {
            "rank": 1,
            "stock": "000002.SZ",
            "period": "2023-03-15 to 2023-04-15",
            "similarity": 0.92,
            "future_performance": {
                "5d": {"return": 0.025, "win_rate": 0.68},
                "20d": {"return": 0.068, "win_rate": 0.75}
            }
        }
    ],
    "statistical_summary": {
        "expected_return_20d": 0.051,
        "confidence_interval": [0.021, 0.081],
        "success_probability": 0.71
    }
}
```

## 🔬 算法性能

| 算法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|------------|------------|----------|
| Matrix Profile | O(n log n) | O(n) | 快速全局搜索 |
| DTW | O(nm) | O(nm) | 时间伸缩匹配 |
| MASS | O(n log n) | O(n) | 单次查询 |
| FAISS | O(log n) | O(n) | 大规模检索 |

## ⚠️ 风险提示

1. **历史表现不代表未来收益**
2. **市场环境变化可能影响预测准确性**
3. **需要考虑交易成本和滑点**
4. **本系统仅供研究参考，不构成投资建议**

## 🔄 更新计划

- [ ] 实现深度学习模型 (TS2Vec)
- [ ] 添加实时数据流支持
- [ ] 优化大规模并行计算
- [ ] 增加更多技术指标
- [ ] Web界面开发

## 📝 开发指南

### 添加新的相似度算法
```python
# likelihood/algorithms/your_algorithm.py
from .base import BaseSimilarityAlgorithm

class YourAlgorithm(BaseSimilarityAlgorithm):
    def compute_similarity(self, series1, series2):
        # 实现你的算法
        pass
```

### 自定义评估指标
```python
# likelihood/backtest/custom_metrics.py
def your_metric(returns, benchmark):
    # 实现自定义指标
    pass
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目仅供学习研究使用，不得用于商业目的。

## 📮 联系方式

- 项目维护：StockTradebyZ Team
- Issue反馈：[GitHub Issues](https://github.com/your-repo/issues)

---

*基于 prompts/likelihood.md 设计规范实现*

*最后更新：2025-08-10*