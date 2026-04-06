# 系统架构总览

StockTradebyZ 是一套 A 股量化交易系统，覆盖数据采集、特征工程、ML 建模、回测评估、AI 增强分析全链路。

## 核心数据流

```
[Tushare API] → [SQLite DB] → [Feature Cache] → [ML Scoring] → [Stock Selection] → [Reports]
     │              │              │                  │                │               │
  5类数据       stock_data.db   ng*_feature_cache   NG/V3.9模型    tomorrow_stock    reports/
  (30-45sec)    10+表, 141GB    69特征/股/天        Ensemble评分    selector.py      JSON+MD
```

## 组件关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         外部数据源                                   │
│   Tushare API │ WQBrain API │ 新闻源 │ 市场指数                      │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   fetch_data/ 数据层  │
              │ • quick_daily_update  │  ← 每日 9:00 AM 运行
              │ • technical_indicator │  ← MA/RSI/MACD/KDJ/BBI
              │ • ng_cache_updater   │  ← 特征缓存同步
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  data_adapter/ 存储层 │
              │ • SQLite stock_data.db│  ← 主数据库
              │ • 10+ 核心表          │
              │ • *_feature_cache    │  ← 模型特征缓存
              └──────────┬───────────┘
                         │
          ┌──────────────┼──────────────┬─────────────┐
          │              │              │             │
          ▼              ▼              ▼             ▼
    ┌───────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐
    │ ml_models/│ │ Selector.py│ │ backtest/│ │ webapp/  │
    │ NG系列    │ │ 8个量化策略 │ │ 回测引擎  │ │ Web界面  │
    │ V3.9/3.95 │ │ 技术选股   │ │ 北极星评估│ │ 仓位管理 │
    └─────┬─────┘ └─────┬──────┘ └──────────┘ └──────────┘
          │             │
          └──────┬──────┘
                 ▼
    ┌─────────────────────────────┐
    │ tomorrow_stock_selector.py  │  ← 主编排器
    │ • 加载 7000+ 股票数据        │
    │ • 应用量化策略过滤           │
    │ • ML 模型评分排名            │
    │ • 输出 Top-10 推荐           │
    └─────────────┬───────────────┘
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
┌─────────┐ ┌──────────┐ ┌──────────────┐
│ reports/ │ │ AI增强报告│ │ trading_     │
│ JSON/MD  │ │ Claude 4 │ │ advisor.py   │
└─────────┘ └──────────┘ └──────────────┘
```

## 核心模块一览

| 模块 | 路径 | 职责 |
|---|---|---|
| 数据采集 | `fetch_data/` | Tushare API 数据抓取、技术指标计算、特征缓存更新 |
| 数据存储 | `data_adapter/` | SQLite 数据库管理、数据加载、Schema 维护 |
| 量化策略 | `Selector.py` | 8个技术面选股策略（BBI/KDJ/MA60等） |
| ML 模型 | `ml_models/` | NG 系列 + V3.9/3.95 ML 评分系统 |
| 主编排器 | `tomorrow_stock_selector.py` | 日常选股流程编排（6246行） |
| 回测评估 | `backtest/` | 回测引擎 + 北极星评分体系 |
| AI 增强 | `TA_integration/` | Claude 4 多 Agent 分析系统 |
| Web 界面 | `webapp/` | Flask 仓位管理和交易界面 |
| 配置 | `core/` + `*.json` | 系统配置、交易日历、策略参数 |

## 技术栈

- **语言**: Python 3.10+
- **数据库**: SQLite（141GB，busy_timeout=30000）
- **ML**: LightGBM + XGBoost + CatBoost + RandomForest + HistGBM Ensemble
- **数据源**: Tushare Pro API
- **AI**: Anthropic Claude 4 API
- **Web**: Flask + Jinja2
- **调度**: Shell scripts + cron

## 关键配置文件

| 文件 | 用途 |
|---|---|
| `config.json` | Tushare/Anthropic API 密钥、系统参数 |
| `production_config.json` | 生产模型配置（当前 NG1.0.2） |
| `strategy_configs.json` | 8个量化策略参数 |
| `CLAUDE.md` | Claude Code 工作指引 |

## 日常运行命令

```bash
# 完整日常流程
./run_daily_update.sh

# 分步执行
python3 fetch_data/quick_daily_update.py --date 20260406    # 数据更新
python3 tomorrow_stock_selector.py 2026-04-06               # 选股（默认v4.9.0.1）
python3 ai_enhanced_daily_report.py --date 2026-04-06       # AI增强
```

## 相关页面

- [数据管线详解](data-pipeline.md)
- [ML 管线详解](ml-pipeline.md)
- [模型演化史](../models/evolution.md)
