# 特征指南

NG 系列模型使用的特征体系。当前 v1.0.3 共 66 个基础特征（56 股票级 + 10 市场级）。v1.0.3 去掉了 3 个 IC 方向翻转的因子（log_market_cap, cs_rank_market_cap, pullback_from_high）。

所有特征计算在 `ml_models/ng/ng_feature_calculator.py` 中实现。

## 特征分组总览

| 组 | 数量 | 类别 | 说明 |
|---|---|---|---|
| Group 1 | 5 | Trend State | 趋势强度、突破、ADX |
| Group 2 | 6 | Pullback Entry | RSI、KDJ、量缩、下影线 |
| Group 3 | 7 | Volume Confirmation | OBV、量比、换手率 |
| Group 4 | 14 | Fundamental | ROE、利润率、估值、市值、流动性 |
| Group 5 | 10 | Market Environment | 大盘趋势、宽度、北向资金 |
| Group 6 | 11 | Industry Rotation | 行业强度、宽度、HHI + 行业活跃度 |
| Group 7 | 10 | Cross-Sectional Rank | 行业内百分位排名（v1.1.0新增） |
| Group 8 | 5 | Residual Factors | 市场/行业中性化 alpha（v1.1.0新增） |
| 合计 | 68 | | 58 股票级 + 10 市场级 |

## Group 1: Trend State (5 因子)

| 因子名 | 计算方式 | 含义 |
|---|---|---|
| `trend_strength_20d` | 20日收盘价OLS斜率 / 标准差 | 趋势强度，越大趋势越明确 |
| `days_since_breakout` | 连续收盘价>前20日最高价的天数 | 突破持续时间 |
| `adx_proxy` | \|MA5-MA20\| / ATR14 | 趋势方向性近似值 |
| `pullback_from_high` | 1 - close/5日最高 | 从近期高点的回撤幅度 |
| `volume_contraction` | 5日均量 / 20日均量 | 量缩比率，<1 说明缩量 |

**v1.1.0 移除的 v1.0.0 因子**: price_above_ma20/60, ma_alignment, new_high_20d/60d, macd_histogram/acceleration, price_channel_position, cumulative_return_60d, bollinger_position, consecutive_down_days — 移除原因：跨截面区分度低。

## Group 2: Pullback Entry (6 因子)

| 因子名 | 计算方式 | 含义 |
|---|---|---|
| `pullback_to_ma10` | close/MA10 - 1 | 相对MA10的偏离 |
| `pullback_to_ma20` | close/MA20 - 1 | 相对MA20的偏离 |
| `rsi_14` | 0.6×RSI12 + 0.4×RSI24 | 超买超卖指标近似 |
| `kdj_j_value` | KDJ的J值 | 短期超买超卖 |
| `lower_shadow_ratio` | (close-low)/(high-low) | 下影线比例，越大买盘越强 |
| `intraday_recovery` | 5日平均(close-low)/(high-low) | 日内恢复能力 |

## Group 3: Volume Confirmation (7 因子)

| 因子名 | 计算方式 | 含义 |
|---|---|---|
| `volume_ratio_5d` | 5日均量 / 20日均量 | 短期放量/缩量 |
| `volume_price_corr` | 20日量价相关系数 | 量价齐升/背离 |
| `obv_trend` | OBV 20日斜率 / 均量 | 资金流入趋势 |
| `volume_breakout` | 当日成交量 / 20日均量 | 当日放量程度 |
| `log_amount_ma5` | log(5日平均成交额) | 流动性（对数尺度） |
| `up_volume_ratio` | 20日阳线成交量占比 | 买盘力度 |
| `volume_cv` | 20日成交量变异系数 | 成交量稳定性 |

注：`turnover_rate` 在此占位，实际由 Group 4 覆盖。

## Group 4: Fundamental (14 因子)

| 因子名 | 来源 | 含义 |
|---|---|---|
| `roe_ttm` | 财务指标 | 净资产收益率(TTM) |
| `roe_change` | 财务指标 | ROE同比变化 |
| `revenue_growth` | 财务指标 | 营收增长率 |
| `net_profit_margin` | 财务指标 | 净利润率 |
| `ocf_quality` | 财务指标 | 经营现金流/净利润 |
| `pe_ttm` | daily_basic | 市盈率(TTM) |
| `pb` | daily_basic | 市净率 |
| `pe_percentile_60d` | 计算 | PE在60日历史中的百分位 |
| `debt_to_assets` | 财务指标 | 资产负债率 |
| `current_ratio` | 财务指标 | 流动比率 |
| `log_market_cap` | daily_basic | 对数流通市值 |
| `log_adv_20d` | 计算 | 对数20日平均成交额 |
| `free_float_ratio` | 基本信息 | 自由流通股比例 |
| `dv_ratio` | daily_basic | 股息率 |
| `turnover_rate` | daily_basic | 换手率 |

## Group 5: Market Environment (10 因子)

全市场级别指标，所有股票当天值相同。

| 因子名 | 计算方式 | 含义 |
|---|---|---|
| `market_return_5d` | 基准指数5日收益 | 短期市场趋势 |
| `market_return_20d` | 基准指数20日收益 | 中期市场趋势 |
| `market_volatility_20d` | 20日log收益std×√252 | 年化波动率 |
| `market_breadth` | 全市场上涨股占比 | 市场宽度 |
| `market_new_high_ratio` | close/20日高>0.98占比 | 创新高比例 |
| `northbound_flow_5d` | 5日北向净买入/标准差 | 标准化北向资金 |
| `market_volume_ratio` | 当日成交额/20日均 | 市场量能 |
| `market_drawdown` | close/60日最高 - 1 | 市场回撤 |
| `vix_proxy` | 20日波动率/60日波动率 | 短期恐慌指标 |
| `market_momentum_diff` | 5日收益-20日收益 | 动量加速度 |

## Group 6: Industry Rotation (11 因子)

| 因子名 | 含义 |
|---|---|
| `industry_return_5d` | 行业5日平均收益 |
| `industry_return_20d` | 行业20日平均收益 |
| `industry_relative_strength` | 个股20日收益 - 行业20日收益 |
| `industry_breadth` | 行业内上涨股占比 |
| `industry_volume_change` | 行业5日/20日成交额比 |
| `industry_rank_return_5d` | 行业5日收益在全行业的百分位 |
| `sw_index_return_5d` | 申万行业指数5日收益 |
| `industry_hhi` | 行业收益集中度(HHI) |
| `sector_breadth_vs_market` | 行业宽度/市场宽度 (v1.1.0新增) |
| `sector_volume_vs_market` | 行业量变/市场量变 (v1.1.0新增) |
| `n_sectors_strong` | 5日收益>2%的行业数 (v1.1.0新增) |

## Group 7: Cross-Sectional Rank (10 因子, v1.1.0 新增)

行业内百分位排名，范围 [0,1]。消除行业共同运动的影响。

| 因子名 | 排名维度 |
|---|---|
| `cs_rank_return_5d` | 5日收益在行业内排名 |
| `cs_rank_return_20d` | 20日收益在行业内排名 |
| `cs_rank_volume_surge` | 放量程度在行业内排名 |
| `cs_rank_turnover` | 换手率在行业内排名 |
| `cs_rank_rsi` | RSI在行业内排名 |
| `cs_rank_new_high` | 近新高距离在行业内排名 |
| `cs_rank_pullback` | 回撤幅度在行业内排名 |
| `cs_rank_volatility` | 波动率在行业内排名 |
| `cs_rank_market_cap` | 市值在行业内排名 |
| `cs_rank_pe` | PE在行业内排名 |

**设计动机**: v1.0.0 的 β_UMD=3.7 说明模型学到了行业动量，cross-sectional rank 将行业效应中性化。

## Group 8: Residual Factors (5 因子, v1.1.0 新增)

去除市场和行业效应后的纯 alpha 信号。

| 因子名 | 计算方式 | 含义 |
|---|---|---|
| `residual_return_20d` | 个股收益 - 行业均值 | 行业中性化超额收益 |
| `residual_volume` | log(个股量) - log(行业均量) | 行业中性化成交量 |
| `idiosyncratic_volatility` | (个股-市场)残差收益std×√252 | 个股特异波动率 |
| `residual_skewness` | 残差收益的偏度 | 尾部风险特征 |
| `relative_strength_vs_peers` | (1+R_stock)/(1+R_industry)-1 | 相对行业的相对强弱 |

## v1.1.0 附加因子（可选）

### 资金流因子 (8个, `--enable-moneyflow`)
主力净流入比、大单占比等，从 `moneyflow_daily` 表获取。

### 交互因子 (8个, `--enable-interaction`)
通过 IC 筛选保留有效的因子交互项。

## 特征演化历史

| 版本 | 特征数 | 主要变化 |
|---|---|---|
| V3.9 | 42 | 基础技术面+基本面 |
| V3.95 | 49 | +daily_basic 5个特征 |
| V4.3 | 59 | +10技术指标 |
| V4.7.5 | 50 | 裁剪20个低重要性特征 |
| NG 1.0.0 | 62 | 重构，59股票+3市场 |
| NG 1.0.1 | 69 | +7市场指标，59+10 |
| NG 1.1.0 | 68+ | 重构分组，+15新因子，-11低区分度因子 |

## 相关页面

- [ML 管线](../architecture/ml-pipeline.md)
- [NG 系列详解](../models/ng-series.md)
- [已知陷阱 — 模型训练类](../lessons/known-pitfalls.md#模型训练类)
