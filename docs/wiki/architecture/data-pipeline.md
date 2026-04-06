# 数据管线

从外部 API 到模型特征缓存的完整数据链路。

## 数据流总览

```
Tushare API ──→ quick_daily_update.py ──→ SQLite DB ──→ Feature Cache
                     │                      │               │
                     ├─ 市场行情             ├─ daily_quotes  ├─ ng*_feature_cache
                     ├─ 大盘指数             ├─ securities    ├─ v39_feature_cache
                     ├─ 日线基本面           ├─ daily_basic   └─ v40_feature_cache
                     ├─ 财务指标(季度)       ├─ technical_indicators
                     └─ 技术指标(计算)       └─ stock_basic_info
```

## 每日更新流程

`fetch_data/quick_daily_update.py` 是日常数据更新的编排器，耗时 30-45 秒：

| 步骤 | 数据类型 | 覆盖范围 | 耗时 |
|---|---|---|---|
| 1 | 市场行情 (OHLCV) | 7000+ 股票/ETF | ~18秒 (2次API) |
| 2 | 大盘指数 | 10个重要指数 | ~2秒 |
| 3 | 日线基本面 (PE/PB/PS/市值/换手率) | 5400+ 股票 | ~5秒 |
| 4 | 财务指标 (EPS/ROE/ROA + 17个扩展) | 季度更新 | ~5秒 |
| 5 | 技术指标 (MA/EMA/RSI/MACD/KDJ/BBI) | 5400+ 股票 | ~15秒 |
| 6 | 特征缓存同步 | 当天特征写入 *_feature_cache | ~3秒 |

### 10个大盘指数

上证指数(000001.SH)、深证成指(399001.SZ)、创业板指(399006.SZ)、科创50(000688.SH)、上证50(000016.SH)、沪深300(000300.SH)、中证500(000905.SH)、中证1000(000852.SH)、中证2000(932000.CSI)、中证全指(000985.SH)

## 数据库表结构

### securities — 证券主数据
```sql
securities (code, name, type, exchange, listing_date, industry)
-- type: 'A股', 'ETF_基金', '指数'
-- 7111+ 条记录
```

### daily_quotes — 日线行情
```sql
daily_quotes (security_id, trade_date, open, high, low, close, volume,
              price_change_pct, is_limit_up, is_limit_down)
-- 同时存储 A 股、ETF、大盘指数数据
-- 指数的 is_limit_up/is_limit_down 恒为 false
```

### daily_basic — 日线基本面
```sql
daily_basic (security_id, trade_date, pe_ttm, pb, ps_ttm,
             market_cap, turnover_rate, ...)
-- 5400+ 股票/天
```

### technical_indicators — 技术指标
```sql
technical_indicators (security_id, trade_date,
    ma5, ma10, ma20, ma60, ma120, ma250,
    ema12, ema26, rsi6, rsi12, rsi24,
    macd, macd_signal, macd_hist,
    kdj_k, kdj_d, kdj_j,
    bbi)
-- 从 close/volume 计算，无 API 调用
```

### *_feature_cache — 模型特征缓存
```sql
-- 每个模型版本独立表（见 lessons/known-pitfalls.md 分表规则）
ng_feature_cache      -- NG v1.0.0 (62特征, 永久保留)
ng101_feature_cache   -- NG v1.0.1 (69特征)
ng102_feature_cache   -- NG v1.0.2 (69特征 + downside_10d)
ng110_feature_cache   -- NG v1.1.0 (69+资金流+交互因子)
v39_feature_cache     -- V3.9 (42特征)
v40_feature_cache     -- V4.0 (cross-sectional)
```

## 历史批量回填

### 推荐：两阶段方法（节省 30-50% 时间）

```bash
# 阶段1：API 数据抓取（受频率限制）
python3 batch_fetch_historical_data.py \
  --start-date 2018-01-01 --end-date 2025-07-01 \
  --max-workers 4 --skip-technical

# 阶段2：技术指标计算（纯CPU，无API限制）
python3 fetch_data/technical_indicator_calculator.py \
  --start-date 2018-01-01 --end-date 2025-07-01 \
  --max-workers 8
```

### 特征缓存回填

```bash
# NG 系列
python3 ml_models/ng/ng_cache_updater.py \
  --start-date 2020-01-01 --end-date 2026-04-03 --version ng1.0.1

# V3.9
python3 fetch_data/v39_feature_cache_updater.py \
  --start-date 2020-01-01 --end-date 2026-04-03
```

## 数据质量检查

```bash
python3 fetch_data/data_quality_check_db.py
```

检查项：缺失交易日、异常价格、空值比例、索引覆盖率。

## 当前数据状态（持续更新）

| 表 | 记录数 | 时间范围 |
|---|---|---|
| daily_quotes | 11.5M+ | 2018-01-02 → 最新 |
| v39_feature_cache | 7.1M+ | 2020-01-02 → 最新 |
| v40_feature_cache | 5.1M+ | 2022-01-04 → 最新 |

## API 限制

- **Tushare**: 500次/分钟，推荐并发 4 进程（~52次API/分钟）
- **技术指标计算**: 无 API 限制，可用 8-16 线程

## 相关页面

- [系统架构总览](system-overview.md)
- [ML 管线](ml-pipeline.md)
- [已知陷阱 — 数据类](../lessons/known-pitfalls.md#数据类)
