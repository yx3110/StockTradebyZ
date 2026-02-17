# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 最高指示
- 不要偷懒，不要编造数据，不要随意删除辛辛苦苦计算和抓取来的真实数据！
- 不要随意删除或覆盖数据库中的内容！
- 所有报告必须保存在 `reports/` 相应子目录中

## 🏗️ System Architecture Overview

**StockTradebyZ** is a sophisticated Chinese A-share trading system combining:
- **7,111 stocks** tracked with daily updates
- **SQLite database** for high-performance data storage
- **4 quantitative strategies** for stock selection
- **4 ML scoring systems** (v3.7, v3.8, v3.81, v3.9) with ensemble learning and incremental updates
- **AI enhancement** via Claude 4 and TradingAgents
- **Chinese market specialization** (T+1, 涨跌停, sentiment analysis)

### Core Data Flow
```
[Tushare API] → [SQLite DB] → [4 Quant Strategies] → [4 ML Systems] → [AI Analysis] → [Trading Reports]
     ↓               ↓                ↓                     ↓               ↓              ↓
  5 data types   stock_data.db   tomorrow_stock    V3.7/V3.8/V3.81/V3.9  Claude + TA    reports/
  (30-45 sec)    10+ tables        selector.py        ML scoring        integration   subdirectories
```

### Daily Data Update Components (🆕 已扩展支持大盘分析 + v3.9财务指标)
1. **Market Quotes**: Open, High, Low, Close, Volume (7000+ stocks/ETFs)
2. **🆕 Market Indices**: 10个重要指数数据 (上证指数、深证成指、创业板指、科创50、上证50、沪深300、中证500、中证1000、中证2000、中证全指)
3. **Daily Basic**: PE, PB, PS, Market Cap, Turnover Rate (5400+ stocks)
4. **🆕 Financial Indicators**: EPS, ROE, ROA + v3.9扩展字段 (17个额外财务指标，quarterly updates)
5. **Technical Indicators**: MA, EMA, RSI, MACD, KDJ, BBI (calculated daily)
6. **Company Info**: Industry, Sector, Listing Date (updated weekly)

## 🚀 Quick Start Commands

### Daily Operations
```bash
# Complete daily workflow (RECOMMENDED)
./run_daily_update.sh                     # Complete data update + stock selection
./run_ai_enhanced_daily.sh               # Add AI analysis

# Individual components
python3 fetch_data/quick_daily_update.py --date 20250930  # Complete data update (30-45 sec)
  # 🆕 Includes: Market quotes, market indices, daily basic, financial indicators, technical indicators
python3 tomorrow_stock_selector.py 2025-09-30             # Stock selection with v3.0 scoring
python3 tomorrow_stock_selector.py 2025-09-30 --scoring-version v3.7  # v3.7 ML scoring
python3 tomorrow_stock_selector.py 2025-09-30 --scoring-version v3.8  # v3.8 incremental learning
python3 tomorrow_stock_selector.py 2025-09-30 --scoring-version v3.81 # v3.81 quality meta-learner
python3 tomorrow_stock_selector.py 2025-09-30 --scoring-version v3.9  # 🆕 v3.9 enhanced features
python3 ai_enhanced_daily_report.py --date 2025-09-30     # 🆕 AI enhancement with market analysis
python3 trading_advisor.py                                # Trading advice

# 🆕 New market analysis components
python3 market_comprehensive_analyzer.py --date 2025-09-30 --save  # Standalone market analysis
python3 pure_tushare_news_fetcher.py                               # Test market data fetching
```

### 🚀 Historical Data Batch Processing (优化版两阶段抓取)
```bash
# 🎯 推荐方法：分离式批量历史数据抓取 (节省30-50%时间)

# 阶段1：API数据批量抓取 (市场行情+基本面+指数，跳过技术指标)
python3 batch_fetch_historical_data.py \
  --start-date 2018-01-01 \
  --end-date 2025-07-01 \
  --max-workers 4 \
  --skip-technical

# 阶段2：技术指标高并发批量计算 (无API限制，充分利用CPU)
python3 fetch_data/technical_indicator_calculator.py \
  --start-date 2018-01-01 \
  --end-date 2025-07-01 \
  --max-workers 8

# 🔧 传统方法：一体化批量抓取 (较慢，但简单)
python3 batch_fetch_historical_data.py \
  --start-date 2018-01-01 \
  --end-date 2025-07-01 \
  --max-workers 4

# 📊 数据完整性验证
python3 fetch_data/data_quality_check_db.py
```

### 🔍 Stock Similarity Analysis (并行化优化版)
```bash
# 🚀 全A股快速扫描 (日常推荐 - 约5分钟)
python3 likelihood/stock_similarity_analyzer.py \
  --code 002215 \
  --window 15 \
  --all-stocks \
  --single-period \
  --threshold 0.10

# 🔬 全A股深度分析 (专业版 - 约25分钟)
python3 likelihood/stock_similarity_analyzer.py \
  --code 002215 \
  --window 15 \
  --all-stocks \
  --threshold 0.10 \
  --processes 8

# 📊 中等规模测试 (快速验证)
python3 likelihood/stock_similarity_analyzer.py \
  --code 002215 \
  --window 15 \
  --candidates 1000 \
  --threshold 0.12

# 🎯 自定义参数示例
python3 likelihood/stock_similarity_analyzer.py \
  --code 000001 \
  --window 20 \
  --candidates 500 \
  --threshold 0.15 \
  --processes 4 \
  --start-date 2023-01-01 \
  --end-date 2025-09-30
```

### Database Operations
```bash
# Check database status
python3 -c "from data_adapter.database_manager import DatabaseManager; db = DatabaseManager(); print(db.get_database_stats())"

# Update fundamental data (OPTIMIZED: 3000+ stocks/min!)
python3 data_adapter/fundamental_data_fetcher.py --mode daily
python3 data_adapter/financial_indicator_fetcher.py --batch-size 25

# Optimize database
python3 data_adapter/optimize_database.py
```

### Testing
```bash
# Run all tests
cd stock_selctor/test && python3 run_tests.py

# Run with coverage
python3 -m pytest stock_selctor/test/ --cov=stock_selctor --cov-report=html

# Test AI integration
python3 TA_integration/test_claude_integration.py

# Test V3.8 incremental learning
cd incremental_learning/tests && python3 test_v380_basic.py
```

## 📁 Project Structure

### Core Components

#### 1. **Data Layer** (`data_adapter/`)
- **Primary Storage**: SQLite database `stock_data.db`
- **Tables**: securities, daily_quotes, technical_indicators, daily_basic, stock_basic_info, stock_signals, backtest_trades, backtest_results
- **Performance**: Database mode is 1800x faster than CSV mode
- **Backup**: `full_securities_data/` contains 7,111 CSV files

#### 2. **Quantitative Engine** (`stock_selctor/`)
- **8 Strategies in `Selector.py`**:
  - BBIKDJSelector (少负战法)
  - BBIShortLongSelector (补票战法)
  - BreakoutVolumeKDJSelector (TePu战法)
  - PeakKDJSelector (填坑战法)
  - SuperB1Selector (SuperB1战法)
  - ZhiXingSelector (知行战法)
  - MA60CrossVolumeWaveSelector (上穿60放量战法)
  - **BigBullishVolumeSelector (暴力K战法)** - 🆕 捕捉大阳线放量但贴近短期均线的股票
- **Orchestrator**: `tomorrow_stock_selector.py`
- **Testing**: Comprehensive pytest suite

#### 3. **🆕 Machine Learning Systems** (`ml_models/`)
```
ml_models/
├── __init__.py              # 统一模型导入接口
├── v37/                     # V3.7高级机器学习系统
│   ├── __init__.py
│   ├── v370_advanced_ml_system.py           # 三层Ensemble (LightGBM+XGBoost+CatBoost+RF+MLP)
│   ├── backtest_v37_engine.py               # V3.7专用回测引擎
│   └── backtest_v37_engine_optimized.py     # V3.7优化回测引擎
├── v38/                     # V3.8增量学习系统
│   ├── __init__.py
│   └── v380_advanced_incremental_ml_system.py  # 增量学习+自适应评分+模型漂移检测
├── v381/                    # V3.81质量评分集成系统
│   ├── __init__.py
│   ├── v380_level4_integrated_system.py        # V380+Level4整合
│   ├── level4_quality_meta_learner.py          # Level4质量元学习器
│   ├── level4_quality_feature_extractor.py     # 质量特征提取器v1
│   ├── level4_quality_feature_extractor_v2.py  # 质量特征提取器v2
│   └── level4_quality_postprocessor.py         # 质量后处理器
└── v39/                     # 🆕 V3.9增强特征系统
    ├── __init__.py
    └── [待集成]             # 42个增强特征，17个扩展财务指标
```

**ML系统特点:**
- **V3.7**: 三层Ensemble架构，49个特征，5个基础模型 (LightGBM, XGBoost, CatBoost, RandomForest, MLP)
- **V3.8**: 增量学习系统，实时特征计算，自适应评分标准化，模型漂移检测
- **V3.81**: V3.8基础 + Level 4质量元学习器，解决质量评分聚集问题
- **🆕 V3.9**: 42个增强特征，包含17个扩展财务指标 (已集成到每日更新脚本，详见 V39_INTEGRATION_GUIDE.md)

#### 4. **Incremental Learning Components** (`incremental_learning/`)
```
incremental_learning/
├── engines/                 # 增量学习引擎
│   ├── incremental_engine.py        # 核心增量学习引擎
│   ├── drift_detector.py            # 模型漂移检测
│   ├── adaptive_forgetting.py       # 自适应遗忘机制
│   └── online_validation.py         # 在线验证
├── features/                # 实时特征计算
│   ├── realtime_calculator.py       # 实时特征计算器
│   ├── intraday_features.py         # 日内特征
│   ├── market_features.py           # 市场特征
│   └── sentiment_indicators.py      # 情绪指标
├── utils/                   # 工具函数
│   ├── feature_storage.py           # 特征存储管理器
│   ├── model_utils.py               # 模型工具
│   ├── data_utils.py                # 数据工具
│   └── cache_utils.py               # 缓存工具
└── tests/                   # 测试文件
    └── test_v380_basic.py           # V3.8基础测试
```

#### 5. **AI Enhancement** (`TA_integration/`)
- **Multi-Agent System**: Technical, Fundamental, News, Sentiment analysts
- **Chinese Specialization**: 雪球 + 东方财富股吧 sentiment
- **Claude Integration**: Multiple configuration presets
- **3 Modes**: enhance, replace, compare

#### 6. **TradingAgents** (`TradingAgents/`)
- **LangGraph Framework**: Multi-agent LLM system
- **Specialized Agents**: Analysts, Researchers, Risk Managers
- **Debate Mechanism**: Bull vs Bear balanced decisions
- **Memory System**: Learning from past trades

#### 7. **Similarity Analysis Engine** (`likelihood/`)
- **Core Tool**: `stock_similarity_analyzer.py` (并行化优化版)
- **算法支持**: Matrix Profile + DTW + MASS + 统计相关性
- **性能特点**: 多进程并行 + 数据预加载 + 向量化计算
- **扫描能力**: 全A股4285只股票，25分钟完成深度分析
- **技术优势**:
  - 批量数据库查询避免21,425次独立查询
  - 8进程并行提升效率10倍
  - 向量化算法早期退出优化
  - 支持任意窗口期(10-30天)和相似度阈值

#### 8. **Strategy & Backtesting**
- **`strategy/`**: Conservative, Balanced, Aggressive strategies
- **`backtest/`**: Comprehensive backtesting engine
- **`backtrader/`**: Professional framework with 122+ indicators
- **`extensible_backtest_engine.py`**: 支持V3.7/V3.8/V3.81的统一回测框架

#### 9. **Report Generation** (`reports/`)
```
reports/
├── daily_selection/       # V1.0 quantitative selection reports
├── daily_selection_v2/    # V2.0 scoring system reports
├── daily_selection_v3/    # V3.0 quantitative scoring reports
├── daily_selection_v3.7/  # 🆕 V3.7 ML ensemble reports
├── daily_selection_v3.8/  # 🆕 V3.8 incremental learning reports
├── daily_selection_v3.81/ # 🆕 V3.81 quality meta-learner reports
├── ai_enhanced/          # AI-augmented analysis
├── correlation_analysis/ # 🆕 Quantitative scoring correlation analysis & debiased case studies
├── similarity_analysis/  # Stock similarity analysis reports
├── trading_advice/       # Position-based recommendations
├── trading_plans/        # Detailed trading plans
├── backtest/            # Backtesting results and reports
├── csv/                 # Data exports
├── pdf/                 # PDF reports
└── v3_quantitative_scoring/ # V3 scoring system data
```

## 🤖 AI Integration

### Claude 4 Configuration
```bash
# Default (Claude Sonnet 4 - RECOMMENDED)
python3 TA_integration/main.py --date 2025-09-30

# High quality (Claude 3.5 Sonnet)
python3 TA_integration/main.py --config claude_high_quality

# Fast batch (Claude 3.5 Haiku)
python3 TA_integration/main.py --config claude_fast

# Premium (Claude 3 Opus)
python3 TA_integration/main.py --config claude_premium
```

### AI-Enhanced Daily Report
```bash
# Complete AI workflow
./run_ai_enhanced_daily.sh --date 2025-09-30

# Components:
# - Quantitative scoring (35%)
# - AI analysis (25%)
# - Sentiment analysis (20%)
# - Technical indicators (15%)
# - Risk assessment (5%)
```

### Chinese Sentiment Analysis
- **Data Sources**: 雪球, 东方财富股吧
- **Features**: 水军识别, 反话检测, 可信度评分
- **Integration**: Automatic in AI-enhanced reports

## 📊 Database Schema

### Core Tables
```sql
-- Securities information (🆕 扩展支持大盘指数)
securities (code, name, type, exchange, listing_date)
  -- type 现在包含: 'A股', 'ETF_基金', '指数'
  -- 🆕 大盘指数如: '000001.SH' (上证指数), '932000.CSI' (中证2000), '000985.SH' (中证全指)

-- Daily market data (🆕 统一存储股票+指数数据)
daily_quotes (security_id, trade_date, open, high, low, close, volume, price_change_pct, is_limit_up, is_limit_down)
  -- 🆕 现在同时存储A股、ETF、大盘指数的日线数据
  -- 🆕 大盘指数的is_limit_up/is_limit_down恒为false
  -- 🆕 支持10个重要指数：上证指数、深证成指、创业板指、科创50、上证50、沪深300、中证500、中证1000、中证2000、中证全指

-- Fundamental data
daily_basic (security_id, trade_date, pe_ttm, pb, ps_ttm, market_cap, turnover_rate, ...)

-- Company information
stock_basic_info (code, name, industry, area, market, list_date)

-- Trading signals
stock_signals (code, date, signal_type, strength, strategy)

-- Backtesting
backtest_trades (trade_id, strategy, code, entry_date, exit_date, pnl)
backtest_results (strategy, total_return, sharpe_ratio, max_drawdown)
```

## 🔧 Configuration Files

### `config.json`
```json
{
  "tushare": {"token": "YOUR_TOKEN"},
  "anthropic": {"api_key": "YOUR_KEY"},
  "system": {
    "max_workers": 5,
    "retry_attempts": 3,
    "database_mode": true
  }
}
```

### `strategy_configs.json`
External strategy parameters for quantitative strategies

### `TA_integration/config/config.json`
AI analysis configuration and weights

## 📈 Trading Strategies

### Quantitative Strategies
1. **少负战法** (BBIKDJSelector): BBI + KDJ combination
2. **补票战法** (BBIShortLongSelector): BBI short/long term RSV
3. **TePu战法** (BreakoutVolumeKDJSelector): Volume breakout + KDJ
4. **填坑战法** (PeakKDJSelector): Peak detection + KDJ
5. **SuperB1战法** (SuperB1Selector): Post-dip recovery after BBI condition
6. **知行战法** (ZhiXingSelector): KDJ + 知行趋势线约束
7. **上穿60放量战法** (MA60CrossVolumeWaveSelector): MA60 breakout with volume confirmation
8. **🆕 暴力K战法** (BigBullishVolumeSelector): 捕捉大阳线放量但贴近短期均线的股票
   - 四大筛选条件：涨幅>4% + 上影线控制 + 1.5倍放量 + 贴近知行短期线

### ML Scoring Systems
1. **V3.7 Advanced ML Ensemble**:
   - 三层Ensemble架构 (Layer1: 5个基础模型, Layer2: Meta模型, Layer3: 最终Ensemble)
   - 49个技术特征 + 基本面特征
   - LightGBM + XGBoost + CatBoost + RandomForest + MLP
   - 训练脚本: `train_v380_parameterized.py`

2. **V3.8 Incremental Learning System**:
   - 增量学习引擎，无需重新训练完整模型
   - 实时特征计算 (日内动量、开盘缺口、早盘表现)
   - 自适应评分标准化 (基于市场波动率和置信度)
   - 模型漂移检测和自动触发重训练
   - 训练脚本: `train_v380_parameterized.py`

3. **V3.81 Quality Meta-Learner**:
   - V3.8基础 + Level 4质量元学习器
   - 解决质量评分聚集问题 (原V3.8质量分80-90聚集)
   - 22个质量特征 + 17个元学习特征
   - 训练脚本: `train_v380_parameterized.py`

### Risk Management
- **Position Sizing**: Max 10% per stock
- **Stop Loss**: 8% default
- **Portfolio Limits**: Max 10 holdings
- **Cash Reserve**: 5% minimum

## 🛠️ Development Guidelines

### Adding New Features
1. **Data Updates**: Use `data_adapter/` for database operations
2. **New Strategies**: Add to `stock_selctor/Selector.py`
3. **ML Models**: Add to `ml_models/` with version subdirectory
4. **Reports**: Save to `reports/` with proper subdirectory
5. **Testing**: Add tests to relevant test directories

### Performance Optimization
- **Database Mode**: Always use database mode (1800x faster)
- **Batch Operations**: Use batch API calls when possible
- **Parallel Processing**: Leverage `max_workers` configuration
- **Caching**: Use built-in caching mechanisms

### File Naming Convention
- Reports: `{描述}_{YYYYMMDD}.md` in `reports/` subdirectories
- Data files: `.csv` or `.json` with date stamps
- Logs: Centralized in `logs/` directory
- ML Models: Version-based organization in `ml_models/v{major}{minor}/`

### Import Conventions
```python
# ✅ 正确的导入方式
from ml_models.v37 import V370AdvancedMLSystem
from ml_models.v38 import V380AdvancedIncrementalMLSystem
from ml_models.v381 import V380Level4IntegratedSystem

# ❌ 错误的导入方式
from v370_advanced_ml_system import V370AdvancedMLSystem  # 旧方式，已弃用
```

## 🔍 Common Workflows

### Morning Routine
```bash
# 1. Complete data update (9:00 AM)
python3 fetch_data/quick_daily_update.py
# Updates: Market quotes, Daily basic (PE/PB/Market cap),
#          Financial indicators (if new), Technical indicators

# 2. Run stock selection (9:30 AM)
python3 tomorrow_stock_selector.py 2025-09-30 --scoring-version v3.81

# 3. Generate AI analysis (9:45 AM)
python3 ai_enhanced_daily_report.py --date 2025-09-30

# 4. Get trading advice (10:00 AM)
python3 trading_advisor.py
```

### 🔍 相似度分析工作流程
```bash
# 周末深度分析: 寻找相似走势股票
# 1. 快速扫描潜在机会 (5分钟)
python3 likelihood/stock_similarity_analyzer.py \
  --code 002215 --all-stocks --single-period --threshold 0.10

# 2. 针对重点股票深度分析 (25分钟)
python3 likelihood/stock_similarity_analyzer.py \
  --code 重点股票代码 --all-stocks --threshold 0.08 --processes 8

# 3. 历史验证不同窗口期效果
python3 likelihood/stock_similarity_analyzer.py \
  --code 002215 --window 10 --candidates 1000  # 短期模式
python3 likelihood/stock_similarity_analyzer.py \
  --code 002215 --window 20 --candidates 1000  # 中期模式
python3 likelihood/stock_similarity_analyzer.py \
  --code 002215 --window 30 --candidates 1000  # 长期模式
```

### Backtesting New Strategy
```bash
# 1. Implement in stock_selctor/Selector.py
# 2. Add configuration to strategy_configs.json
# 3. Run backtest with specific ML version
python3 extensible_backtest_engine.py --ml-version v3.81 --start-date 2024-01-01 --end-date 2025-09-30

# 4. Analyze results
cat reports/backtest/回测结果_YYYYMMDD.md
```

### Training ML Models
```bash
# Train V3.7 model (三层Ensemble)
python3 train_v380_parameterized.py --model-version v3.7

# Train V3.8 model (增量学习系统)
python3 train_v380_parameterized.py --model-version v3.8

# Train V3.81 model (质量元学习器)
python3 train_v380_parameterized.py --model-version v3.81

# Train with custom parameters
python3 train_v380_parameterized.py \
  --model-version v3.81 \
  --start-date 2020-01-01 \
  --end-date 2025-09-30 \
  --lookback-days 10 \
  --lookahead-days 5
```

### 🔍 Quantitative Scoring Correlation Analysis
```bash
# 🆕 Complete correlation analysis with debiased case selection
python3 analyze_quantitative_scoring_correlation.py --report-dir reports/daily_selection_v3 --version v3

# Test debiased case selection independently
python3 test_debiased_analysis.py

# Key features:
# - Analyzes 11,479+ stock picks from 2025-01-01 to present
# - Calculates correlation between quantitative scores and future returns (1d, 3d, 5d, 10d, 20d, 30d)
# - 🚀 NEW: Eliminates overlapping time-period bias in case studies
# - Ensures each performance example represents independent stock behavior
# - Saves reports to reports/correlation_analysis/
```

### Performance Analysis
```bash
# Check data quality
python3 fetch_data/data_quality_check_db.py

# Monitor system performance
tail -f logs/daily_update.log

# Database statistics
python3 data_adapter/database_manager.py --stats

# Test ML model imports
python3 -c "from ml_models.v37 import V370AdvancedMLSystem; print('✅ v3.7 OK')"
python3 -c "from ml_models.v38 import V380AdvancedIncrementalMLSystem; print('✅ v3.8 OK')"
python3 -c "from ml_models.v381 import V380Level4IntegratedSystem; print('✅ v3.81 OK')"
```

## 🚨 Important Notes

### Data Integrity
- **NEVER** delete data from SQLite database without backup
- **ALWAYS** use database transactions for updates
- **MAINTAIN** CSV backups in `full_securities_data/`

### API Rate Limits
- Tushare: Respect daily quota limits
- Anthropic: Monitor token usage and costs
- Use caching to reduce redundant API calls

### Production Deployment
- Set up cron jobs with `setup_daily_cron.sh`
- Monitor logs in `logs/` directory
- Regular database optimization with `optimize_database.py`

### ML Model Management
- **训练数据**: 使用历史数据训练，确保有足够的样本量 (建议5年以上)
- **模型文件**: 存储在 `models/` 目录，按版本管理
- **增量更新**: V3.8/V3.81支持增量学习，无需每日重训练
- **模型评估**: 定期评估模型性能，监控漂移情况

## 📞 Support & Feedback

- **Issues**: Report at https://github.com/anthropics/claude-code/issues
- **Help**: Use `/help` command in Claude Code
- **Documentation**: See individual module README files

## 🎯 Key Performance Metrics

### Daily Operations
- **Complete Data Update**: 30-45 seconds for all data types
- **Market Quotes**: 7000+ stocks/ETFs in 18 seconds (2 API calls)
- **Basic/Financial**: 5400+ stocks PE/PB/market cap in 10 seconds
- **Technical Indicators**: 5400+ stocks calculated in 15 seconds
- **Database Performance**: 1800x faster than CSV mode
- **Storage**: ~4GB database + CSV backups
- **Coverage**: 10,073 securities (A-shares, ETFs, indices)
- **相似度分析**: 4285只A股，25分钟完成全市场扫描

### ML Model Performance
- **V3.7 Training**: ~30-60分钟 (5年历史数据，5个基础模型+元学习)
- **V3.8 Training**: ~40-70分钟 (增量学习引擎初始化)
- **V3.81 Training**: ~50-80分钟 (V3.8基础+Level4质量学习器)
- **Daily Prediction**: <1秒/股 (所有版本)
- **Incremental Update**: <10秒 (V3.8/V3.81增量更新)

### 🚀 Historical Batch Processing Performance
#### 优化版两阶段方法 (推荐)
- **阶段1 (API数据)**: 约1-2分钟/交易日 × 并发数
- **阶段2 (技术指标)**: 5600只股票约15-30分钟 (8线程)
- **总时间节省**: 30-50% vs 传统方法
- **API调用优化**: 避免技术指标计算期间的API等待
- **CPU利用率**: 技术指标阶段可达100%多核利用

#### 性能对比 (以抓取7年历史数据为例)
| 方法 | API阶段 | 技术指标阶段 | 总时间 | 优势 |
|------|---------|-------------|---------|------|
| **优化版两阶段** | 60-90分钟 | 15-30分钟 | **75-120分钟** | 📈 API专注,高并发计算 |
| 传统一体化 | - | - | **120-180分钟** | 🔧 简单但较慢 |
| 手工分批 | 数小时 | 数小时 | **半天** | ❌ 低效 |

#### API限制优化
- **Tushare频率**: 500次/分钟限制
- **推荐并发数**: 4个进程 (约52次API调用/分钟)
- **技术指标并发**: 8-16线程无API限制

## 📋 相似度分析参数说明

### 基本参数
- `--code`: 目标股票代码 (默认: 002215)
- `--window`: 分析窗口天数 (默认: 15，推荐: 10-30)
- `--threshold`: 相似度阈值 (默认: 0.12，范围: 0.05-0.3)

### 扫描范围
- `--all-stocks`: 扫描全A股4285只股票
- `--candidates N`: 限制候选股票数量 (默认: None)
- `--single-period`: 单时期分析 (默认: 多时期)

### 性能优化
- `--processes N`: 并行进程数 (默认: auto，推荐: 4-8)

### 时间范围
- `--start-date`: 开始日期 (默认: 2020-01-01)
- `--end-date`: 结束日期 (默认: 2025-09-30)

### 使用建议
| 用途 | 推荐参数 | 预计时间 |
|------|----------|----------|
| 日常快速扫描 | `--all-stocks --single-period --threshold 0.10` | 5分钟 |
| 专业深度分析 | `--all-stocks --threshold 0.08 --processes 8` | 25分钟 |
| 测试验证 | `--candidates 1000 --threshold 0.12` | 1分钟 |
| 短线交易 | `--window 10 --threshold 0.15` | 根据候选数 |
| 中长线投资 | `--window 25 --threshold 0.10` | 根据候选数 |

This system represents state-of-the-art quantitative trading infrastructure with advanced ML ensemble learning, incremental updates, and AI enhancement, specifically optimized for Chinese A-share markets.