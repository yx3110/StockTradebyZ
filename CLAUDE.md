# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 最高指示
- 不要偷懒，不要编造数据，不要随意删除辛辛苦苦计算和抓取来的真实数据！
- 不要随意删除或覆盖数据库中的内容！
- 所有报告必须保存在 `reports/` 相应子目录中

## 🎯 工作风格 (必须遵守)

### 执行优先，禁止空转
- **收到任务后立即开始执行**。最多花2分钟理解上下文，然后动手改代码
- 如果用户给了编号计划，逐步执行，每步完成后简要确认并继续下一步
- **禁止**花整个session只读代码和规划而不产出任何修改
- 如果范围太大无法一次完成，先执行最高优先级的部分，而不是继续规划

### 代码质量
- **所有新写的代码在运行前必须先用 `/simplify` 检查修复**，包括脚本、训练代码、回测框架等
- OOS评估前必须验证报告pred_10d有非零值: `sum(1 for s in data['all_stocks_with_scores'] if float(s.get('pred_10d',0) or 0)!=0)`

### 已知陷阱 (避免反复踩坑)
- 股票代码必须带交易所后缀，如 `000001.SZ`、`600519.SH`
- SQLite并发操作必须设置 `busy_timeout=30000`（至少30秒）
- `.pkl` / `.joblib` 模型文件**绝对不要** git add，必须在 .gitignore 中
- Tushare API 使用 `pro.index_member_all()` 做分页查询，不要用 per-industry 调用
- 性能优化前**先 profile**（`python -m cProfile`），不要猜瓶颈在哪里

### Git 操作
- 除非用户明确要求拆分commit，否则用 `git add -A` 提交所有变更
- 每次任务完成后主动 commit（除非用户说不要）
- commit message 用中文简要描述改动内容

### 📚 Wiki 维护（必须遵守）
- Wiki 位于 `docs/wiki/`，维护规则详见 `docs/wiki/schema.md`
- **开始改动前**：读取 `docs/wiki/index.md`，查阅与当前任务相关的 Wiki 页面，了解历史决策和已知陷阱
- **完成改动后**：评估是否需要更新 Wiki（模型/架构/教训/特征相关则必须更新），同时更新 CLAUDE.md 中受影响的章节
- 闭环：**查 Wiki → 做改动 → 更新 Wiki → 更新 CLAUDE.md**

## 🏗️ System Architecture Overview

**StockTradebyZ** is a sophisticated Chinese A-share trading system combining:
- **7,111 stocks** tracked with daily updates
- **SQLite database** for high-performance data storage
- **8 quantitative strategies** for stock selection
- **2 active ML scoring systems** (v3.9, v3.95) with ensemble learning
- **AI enhancement** via Claude 4 and TradingAgents
- **Chinese market specialization** (T+1, 涨跌停, sentiment analysis)

### Core Data Flow
```
[Tushare API] → [SQLite DB] → [8 Quant Strategies] → [2 ML Systems] → [AI Analysis] → [Trading Reports]
     ↓               ↓                ↓                     ↓               ↓              ↓
  5 data types   stock_data.db   tomorrow_stock        V3.9/V3.95     Claude + TA    reports/
  (30-45 sec)    10+ tables        selector.py        ML scoring      integration   subdirectories
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
python3 tomorrow_stock_selector.py 2025-09-30                             # Stock selection (默认v4.9.0.1)
python3 tomorrow_stock_selector.py 2025-09-30 --scoring-version v4.9.0.1  # 🏆 生产推荐 V4=92.8% S级
python3 tomorrow_stock_selector.py 2025-09-30 --scoring-version v3.9      # v3.9 旧版
# 生产回测评估:
python3 backtest/run_north_star_eval.py --production
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

# Backfill v39 data (daily_basic / financial_indicator)
python3 fetch_data/v39_data_backfill.py --mode daily
python3 fetch_data/backfill_historical_data.py --mode all

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
- **Storage**: Database-only mode (no CSV backups)

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

#### 3. **Machine Learning Systems** (`ml_models/`)
```
ml_models/
├── __init__.py              # 统一模型导入接口
├── v38/                     # V3.8增量学习系统 (deprecated, 保留)
│   ├── __init__.py
│   └── v380_advanced_incremental_ml_system.py
├── v39/                     # V3.9增强特征系统 (活跃)
│   ├── __init__.py
│   ├── v390_production_scorer.py      # V3.9生产评分器
│   ├── v390_enhanced_feature_ml_system.py
│   ├── v394_production_scorer.py      # V3.94生产评分器
│   └── v395_production_scorer.py      # V3.95生产评分器
├── training/                # 训练脚本 (从根目录迁入)
│   ├── train_v390_from_cache.py       # v3.9 日常推荐训练
│   ├── train_v395_multi_target.py     # v3.95 最新生产版
│   ├── train_v380_parameterized.py    # v3.8 (deprecated)
│   └── backfill/                      # 数据回填脚本
└── trained_models/          # 训练好的模型文件 (.gitignore)
    ├── v390_full_from_cache.pkl       # 生产 v3.9 模型
    ├── v39/                           # v3.9 训练组件
    ├── v380/                          # v3.8 deprecated 模型
    ├── v394/                          # v3.94 模型
    └── v395/                          # v3.95 rolling ensemble
```

**ML系统特点:**
- **🏆 NG v1.0.1** (最新): 69特征(59股票+10市场), 行业超额标签, ICIR自适应权重, V5=70.1% A+
  - 缓存表: `ng101_feature_cache` (分表存储, backward compatible)
  - 训练: `python3 ml_models/ng/ng_trainer.py --start-date 2020-01-01 --purge-days 15`
  - 回填: `python3 ml_models/ng/ng_cache_updater.py --start-date 2020-01-01 --end-date 2026-04-03 --version ng1.0.1`
  - 报告: `python3 backtest/batch_generate_v395_reports.py --version ng1.0.1`
- **NG v1.0.0**: 62特征, 绝对收益标签, 缓存表: `ng_feature_cache` (永久保留)
- **V3.9** (旧版): 42个增强特征 + 17个扩展财务指标，LightGBM+XGBoost+CatBoost+RF Ensemble
- **V3.95** (旧版): 多目标预测（3d/5d/10d），滚动训练窗口
- **V3.8** (deprecated): 增量学习系统，保留模型文件供参考

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
- **`backtest/`**: Comprehensive backtesting engine + all backtest scripts
- **`backtest/extensible_backtest_engine.py`**: 支持多版本的统一回测框架
- **`backtest/backtest_report_based.py`**: 基于报告的多模型对比回测 + 北极星V1/V2双评分卡
- **`backtest/north_star_metrics.py`**: 北极星指标计算模块 (V1: 11项/33分, V2: 21项/105分/6档)
- **`backtest/run_north_star_eval.py`**: CLI评估工具 (--extended扩展窗口, --regime-analysis市况分析)
- **`backtest/batch_generate_v395_reports.py`**: 快速批量报告生成器 (v3.9/v3.95/v4.3/v5.0)

#### 9. **Archive** (`archive/`)
```
archive/
├── debug_scripts/          # 调试脚本 (debug_*.py, diagnose_*.py)
├── temp_analysis/          # 一次性分析脚本 (analyze_*.py)
├── temp_scripts/           # 一次性工具脚本 (fill_gap_*.py, migrate_*.py, validate_*.py, *.sh)
├── training_experiments/   # 训练实验脚本 (quick_train_*.py, precompute_*.py)
├── experimental/           # 实验性脚本 (trading_strategy*.py, run_3strategy_*.py)
├── development_docs/       # 开发文档 (BACKTEST_*.md, REBALANCE_*.md)
├── old_versions/           # 旧版本代码
├── tests/                  # 归档测试
├── logs/                   # 旧日志
└── weight_optimization/    # 权重优化实验
```

#### 10. **Report Generation** (`reports/`)
```
reports/
├── daily_selection/       # V1.0 quantitative selection reports
├── daily_selection_v2/    # V2.0 scoring system reports
├── daily_selection_v3/    # V3.0 quantitative scoring reports
├── daily_selection_v3.7/  # V3.7 ML ensemble reports (archived)
├── daily_selection_v3.8/  # V3.8 incremental learning reports (archived)
├── daily_selection_v3.81/ # V3.81 quality meta-learner reports (archived)
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

**长期回测 (2018-2026)** — 见 [docs/wiki/evaluation/quant-strategies-2018-2026.md](docs/wiki/evaluation/quant-strategies-2018-2026.md)
- 暴力K = 最强(+1.59% 10d alpha, Sharpe=1.96), 熊市特化 Sharpe=2.05
- 少负 = 牛市特化 (Sharpe=1.75 in bull regime)
- SuperB1/补票 = 跨regime稳定基线
- 知行/上穿60放量 = 长期无alpha, 建议退役
- ⚠️ 不要用8策略做ML的pre-filter — 已实证让v4.7.5从A+降到B

### ML Scoring Systems (活跃版本)
1. **🏆 V4.9.0.1 Production** (生产推荐, 北极星V4=92.8% S级):
   - 61特征, Q95 Widen-then-Concentrate头部精筛
   - LightGBM + XGBoost + CatBoost + RF + HGB + LambdaRank(trunc=10) Ensemble
   - 推理: EMA平滑(alpha=0.7) + 市场门控(GateV2, AUC=0.714)
   - 组合: Top10 + Focus15d + Ret0.2 + CPPI(8,20) + SF30
   - 性能: 年化124.5%, Sharpe=3.14, MaxDD=-6.4%, 超额胜率68.6%
   - Scorer: `ml_models/v39/v492_production_scorer.py` (V490+EMA) 或 `v490_production_scorer.py`
   - 配置: `production_config.json`
   - 回测: `python3 backtest/run_north_star_eval.py --production`

2. **V3.9 Production Scorer** (旧版):
   - 42个增强特征 + 17个扩展财务指标
   - 训练脚本: `ml_models/training/train_v390_from_cache.py`

### Risk Management (生产配置)
- **Position Sizing**: Max 10% per stock, Top 10持仓
- **Holding Period**: 15日调仓 (focus_days=15)
- **CPPI**: floor=8%, multiplier=20 (MaxDD=-6.4%)
- **Score Floor**: 30分以下不入选
- **Retention**: 20%持仓保留加分 (降低换手)
- **EMA Smoothing**: alpha=0.7 (换手32x)

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
# ✅ 活跃版本 (推荐)
from ml_models.v39.v390_production_scorer import V390ProductionScorer
from ml_models.v39.v395_production_scorer import V395ProductionScorer
```

## 🔍 Common Workflows

### Morning Routine
```bash
# 1. Complete data update (9:00 AM)
python3 fetch_data/quick_daily_update.py
# Updates: Market quotes, Daily basic (PE/PB/Market cap),
#          Financial indicators (if new), Technical indicators

# 2. Run stock selection (9:30 AM) - 默认使用v3.9
python3 tomorrow_stock_selector.py 2025-09-30

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
python3 backtest/extensible_backtest_engine.py --ml-version v3.9 --start-date 2024-01-01 --end-date 2025-09-30

# 4. Analyze results
cat reports/backtest/回测结果_YYYYMMDD.md
```

### 🎯 北极星评估 (无泄露双向评估 — 正式方法)

**⚠️ 重要**: 训练数据不可用于回测。所有模型评估必须使用以下无泄露方法：

```bash
# 🏆 推荐方式: 训练时自动执行双向评估 (默认开启, 无需手动操作)
python3 ml_models/training/train_v395_multi_target.py --v4901 --purge-days 15
# 训练完成后自动输出:
#   1. WF OOS 北极星评估 (向前泛化, WF test period报告)
#   2. Pre-2020 北极星评估 (向后泛化, 生产模型对2018-2019的预测)

# 手动评估WF OOS报告 (训练后已自动生成在 reports/daily_selection_{ver}_wf_oos/)
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4901_wf_oos \
    --label WF-OOS --top-n 10 --focus-days 10 --rank-field composite

# 手动评估Pre-2020报告 (训练后已自动生成在 reports/daily_selection_{ver}_pre2020/)
python3 backtest/run_north_star_eval.py --backtest \
    --report-dir reports/daily_selection_v4901_pre2020 \
    --label PRE-2020 --top-n 10 --focus-days 10 --rank-field composite

# 回填2018-2020特征缓存 (首次使用Pre-2020评估前需执行一次)
python3 fetch_data/v39_feature_cache_updater.py --start-date 2018-01-01 --end-date 2019-12-31
```

**评估解读:**
- WF OOS (向前): 模型能否预测未来 — B~A级为合理预期
- Pre-2020 (向后): 模型学到的是通用规律还是过拟合 — A级说明信号真实
- 两个都有alpha → 高置信; 只有一个有 → 谨慎; 都没有 → 模型有问题
- **--production 回测(S级)包含数据泄露，仅供内部参考，不可作为模型评估依据**

### Training ML Models
```bash
# Train V4901 (生产推荐, 默认流程: auto-WF模式选择 + 3进程并行WF + 自动双向评估)
# 训练前自动turbo-check 3种WF配置(expanding/sliding-720d/sliding-500d+decay730)
# 选择10d ICIR最优的配置后再全量训练 (~6min额外开销)
python3 ml_models/training/train_v395_multi_target.py --v4901 --purge-days 15

# 跳过auto-WF, 直接用默认expanding模式训练
python3 ml_models/training/train_v395_multi_target.py --v4901 --purge-days 15 --no-auto-wf

# 手动指定滑动窗口模式 (跳过auto-WF)
python3 ml_models/training/train_v395_multi_target.py --v4901 --purge-days 15 --max-train-days 720

# 手动调整时间衰减半衰期 (默认365天)
python3 ml_models/training/train_v395_multi_target.py --v4901 --purge-days 15 --time-decay-halflife 730

# Train V3.9 model (旧版)
python3 ml_models/training/train_v390_from_cache.py
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

# Test ML model imports (active versions)
python3 -c "from ml_models.v39.v390_production_scorer import V390ProductionScorer; print('✅ v3.9 OK')"
python3 -c "from ml_models.v39.v395_production_scorer import V395ProductionScorer; print('✅ v3.95 OK')"

# Test training scripts
python3 ml_models/training/train_v390_from_cache.py --help
python3 ml_models/training/train_v395_multi_target.py --help
```

## 🚨 Important Notes

### Data Integrity
- **NEVER** delete data from SQLite database without backup
- **ALWAYS** use database transactions for updates
- **MAINTAIN** regular database backups

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
- **模型文件**: 存储在 `ml_models/trained_models/` 目录，按版本管理
- **增量更新**: V3.8支持增量学习，无需每日重训练
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
- **Storage**: ~4GB database
- **Coverage**: 10,073 securities (A-shares, ETFs, indices)
- **相似度分析**: 4285只A股，25分钟完成全市场扫描

### ML Model Performance
- **V3.9 Training**: ~30-60分钟 (基于缓存特征，4模型Ensemble)
- **V3.95 Training**: ~40-70分钟 (多目标预测，滚动窗口)
- **Daily Prediction**: <1秒/股 (所有版本)

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