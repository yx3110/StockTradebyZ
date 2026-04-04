# Daily Selection NG — 右侧趋势波段因子体系设计

**日期**: 2026-04-04
**目标**: 从零设计一个基于经济直觉先验的因子体系，作为新一代选股模型的训练起点
**代号**: Daily Selection NG (Next Generation)

## 1. 交易逻辑

**右侧趋势波段**：确认主升浪已启动 → 在回调到支撑位时入场 → 持有5-10天 → 趋势衰竭前退出

三个核心问题：
1. **趋势确认**：股票是否处于主升浪中？（不买左侧）
2. **入场时机**：主升浪中是否出现了好的回调买点？
3. **风险过滤**：这只股票值不值得买？（基本面/流动性/情绪）

## 2. 设计约束

- **持仓周期**: 5-10天
- **因子数量**: 62个，全面覆盖但每个因子有清晰经济逻辑
- **市值约束**: 硬性下限50亿 + log_market_cap/log_adv作为特征让模型自适应
- **Alpha来源**: 趋势动量+回调入场（GARP+技术确认风格）
- **复用**: 沿用V4901训练pipeline（WF、Ensemble、LambdaRank）
- **评估**: WF OOS报告 + V5.2无泄漏评分

## 3. 因子体系：6大类 62个因子

### 3.1 趋势状态（12个）— "是否在主升浪中"

主升浪特征：价格站稳中期均线之上，短中长均线多头排列，创新高。

| # | 因子名 | 计算 | 经济直觉 | 数据源 |
|---|--------|------|----------|--------|
| 1 | `price_above_ma20` | close/ma20 - 1 | 站稳20日均线=中期趋势向上 | daily_quotes.ma20 |
| 2 | `price_above_ma60` | close/ma60 - 1 | 站稳60日均线=长期趋势确认 | daily_quotes.ma60 |
| 3 | `ma_alignment` | 多头排列程度: mean(ma5>ma10, ma10>ma20, ma20>ma60) × (ma5-ma60)/close | 均线多头排列=主升浪核心信号 | daily_quotes.ma5/10/20/60 |
| 4 | `trend_strength_20d` | 20日close线性回归斜率 / close的20日std | 趋势强度（斜率除以噪声） | daily_quotes.close |
| 5 | `new_high_20d` | close / max(high, 20d) | 接近20日新高=趋势延续 | daily_quotes.close/high |
| 6 | `new_high_60d` | close / max(high, 60d) | 中期高点突破 | daily_quotes.close/high |
| 7 | `days_since_breakout` | 突破20日最高价后的天数（未突破=0） | 主升浪"年龄"——太老的趋势衰竭 | daily_quotes.close/high |
| 8 | `adx_proxy` | abs(ma5 - ma20) / atr_14 | 趋势vs震荡（ADX简化版） | daily_quotes.ma5/20 + tech.atr_14 |
| 9 | `macd_histogram` | macd_macd值 | MACD柱状图>0=中期动能向上 | tech.macd_macd |
| 10 | `macd_acceleration` | macd_macd - macd_macd[5日前] | MACD加速=趋势加速阶段 | tech.macd_macd |
| 11 | `price_channel_position` | (close - min(low,20d)) / (max(high,20d) - min(low,20d)) | 价格在20日通道中的位置 | daily_quotes |
| 12 | `cumulative_return_60d` | close/close[60d前] - 1 | 中期动量——主升浪的前提条件 | daily_quotes.close |

### 3.2 回调入场（10个）— "主升浪中的好买点"

主升浪中回调到支撑位+缩量+超卖=最佳波段入场点。

| # | 因子名 | 计算 | 经济直觉 | 数据源 |
|---|--------|------|----------|--------|
| 13 | `pullback_from_high` | 1 - close/max(close, 5d) | 从近期高点回撤幅度 | daily_quotes.close |
| 14 | `pullback_to_ma10` | close/ma10 - 1 | 回调到10日线（短期支撑） | daily_quotes |
| 15 | `pullback_to_ma20` | close/ma20 - 1 | 回调到20日线（中期支撑） | daily_quotes |
| 16 | `rsi_14` | RSI(14) | RSI<40在趋势中=超卖回调 | tech.rsi_14 (需从rsi12/rsi24推算或重算) |
| 17 | `kdj_j_value` | KDJ的J值 | J<20=短期极度超卖，J>80=超买 | tech.kdj_j |
| 18 | `volume_contraction` | mean(volume, 5d) / mean(volume, 20d) | 缩量回调=洗盘；放量下跌=出货 | daily_quotes.volume |
| 19 | `lower_shadow_ratio` | (close - low) / (high - low + 1e-8) | 下影线长=下方有买盘支撑 | daily_quotes |
| 20 | `consecutive_down_days` | 连续收跌天数（正数） | 连跌后在趋势中=回调到位概率高 | daily_quotes.close |
| 21 | `bollinger_position` | (close - boll_lower) / (boll_upper - boll_lower + 1e-8) | 接近布林下轨=统计超卖 | tech.boll_* |
| 22 | `intraday_recovery` | mean((close-low)/(high-low+1e-8), 5d) | 持续收复日内跌幅=买盘强 | daily_quotes |

### 3.3 成交量确认（8个）— "聪明钱在进场吗"

主升浪需量价配合：放量突破+缩量回调是经典健康模式。

| # | 因子名 | 计算 | 经济直觉 | 数据源 |
|---|--------|------|----------|--------|
| 23 | `volume_ratio_5d` | mean(volume,5d) / mean(volume,20d) | 近期相对放量程度 | daily_quotes.volume |
| 24 | `volume_price_corr` | corr(close, volume, 20d) | 量价正相关=健康上涨趋势 | daily_quotes |
| 25 | `obv_trend` | OBV的20日线性回归斜率（归一化） | 累积量能方向——资金持续流入or流出 | daily_quotes |
| 26 | `volume_breakout` | 今日volume / mean(volume, 20d) | 单日放量信号（突破确认） | daily_quotes.volume |
| 27 | `log_amount_ma5` | log(mean(amount, 5d)) | 绝对流动性（大资金可进出） | daily_quotes.amount |
| 28 | `turnover_rate` | 换手率 | 市场关注度+流动性 | daily_basic.turnover_rate |
| 29 | `up_volume_ratio` | sum(volume where close>open, 20d) / sum(volume, 20d) | 上涨日放量vs下跌日放量=资金做多意愿 | daily_quotes |
| 30 | `volume_cv` | std(volume,20d) / mean(volume,20d) | 成交量稳定性——突然放量预警 | daily_quotes.volume |

### 3.4 基本面质量（14个）— "值不值得买"

好趋势+好基本面=更安全的主升浪。纯投机驱动的趋势更容易崩。

| # | 因子名 | 计算 | 经济直觉 | 数据源 |
|---|--------|------|----------|--------|
| 31 | `roe_ttm` | ROE(TTM) | 盈利能力——核心质量指标 | financial_indicator.roe |
| 32 | `roe_change` | 本季ROE - 去年同期ROE | 盈利改善（趋势的基本面支撑） | financial_indicator |
| 33 | `revenue_growth` | 营收同比增速（需从eps/利润率推算或用profit_to_gr） | 成长性——主升浪最常见催化剂 | financial_indicator.profit_to_gr |
| 34 | `net_profit_margin` | 净利润率 | 赚钱效率 | financial_indicator.netprofit_margin |
| 35 | `ocf_quality` | 经营现金流/净利润 | 盈利质量——现金流支撑的利润更可靠 | financial_indicator.ocf_to_profit |
| 36 | `pe_ttm` | PE(TTM) | 估值水平 | daily_basic.pe_ttm |
| 37 | `pb` | PB | 估值安全边际 | daily_basic.pb |
| 38 | `pe_percentile_60d` | PE_TTM在自身60日内的百分位 | 相对自身估值位置（高=贵了） | daily_basic.pe_ttm |
| 39 | `debt_to_assets` | 资产负债率 | 财务杠杆风险 | financial_indicator.debt_to_assets |
| 40 | `current_ratio` | 流动比率 | 短期偿债能力——排除财务危机 | financial_indicator.current_ratio |
| 41 | `log_market_cap` | log(流通市值) | 规模因子（让模型学习大小偏好） | daily_basic.circ_mv |
| 42 | `log_adv_20d` | log(20日均成交额) | 流动性/可交易性 | daily_quotes.amount |
| 43 | `free_float_ratio` | free_share / total_share | 筹码结构——自由流通少=筹码稀缺 | daily_basic |
| 44 | `dv_ratio` | 股息率 | 价值锚——提供安全边际 | daily_basic.dv_ratio |

### 3.5 市场环境（10个）— "大盘支持做多吗"

大盘环境决定个股趋势交易成功率。牛市做多胜率远高于熊市。

| # | 因子名 | 计算 | 经济直觉 | 数据源 |
|---|--------|------|----------|--------|
| 45 | `market_return_5d` | 沪深300 5日收益 | 短期大盘方向 | daily_quotes(000300.SH) |
| 46 | `market_return_20d` | 沪深300 20日收益 | 中期大盘趋势 | daily_quotes(000300.SH) |
| 47 | `market_volatility_20d` | 沪深300日收益的20日标准差×sqrt(252) | 低波=稳定做多环境 | daily_quotes(000300.SH) |
| 48 | `market_breadth` | 全市场上涨家数占比的20日均值 | 市场广度——赚钱效应 | daily_quotes全市场 |
| 49 | `market_new_high_ratio` | 创20日新高股票占全市场比例 | 市场热度指标 | daily_quotes全市场 |
| 50 | `northbound_flow_5d` | 北向资金5日净流入z-score | 外资方向——聪明钱风向标 | hsgt_daily |
| 51 | `market_volume_ratio` | 全市场成交额 / 20日均值 | 整体量能——放量=资金活跃 | daily_quotes全市场 |
| 52 | `market_drawdown` | 沪深300距60日高点的回撤 | 系统性风险水平 | daily_quotes(000300.SH) |
| 53 | `vix_proxy` | 市场20日波动率 / 60日波动率 | 波动率放大=恐慌/不确定性上升 | daily_quotes(000300.SH) |
| 54 | `market_momentum_diff` | market_return_5d - market_return_20d | 大盘加速or减速 | 衍生计算 |

### 3.6 行业动量与轮动（8个）— "风口上的行业"

主升浪常伴随行业轮动，行业整体走强时个股趋势更可靠。

| # | 因子名 | 计算 | 经济直觉 | 数据源 |
|---|--------|------|----------|--------|
| 55 | `industry_return_5d` | 所在申万一级行业5日均收益 | 行业短期动量 | daily_quotes(行业成分) |
| 56 | `industry_return_20d` | 所在行业20日均收益 | 行业中期趋势 | daily_quotes(行业成分) |
| 57 | `industry_relative_strength` | 个股20d收益 - 行业20d收益 | 行业内相对强弱 | 衍生 |
| 58 | `industry_breadth` | 行业内上涨股票占比 | 行业广度——普涨vs龙头独涨 | daily_quotes(行业成分) |
| 59 | `industry_volume_change` | 行业总成交额5d/20d | 资金涌入行业 | daily_quotes(行业成分) |
| 60 | `industry_rank_return_5d` | 行业5日收益在31个行业中的排名(0-1) | 行业轮动位置 | 衍生 |
| 61 | `sw_index_return_5d` | 申万行业指数5日收益 | 精确行业指数动量 | daily_quotes(申万指数) |
| 62 | `industry_hhi` | 行业内涨幅的HHI集中度 | 龙头集中vs普涨——普涨更健康 | daily_quotes(行业成分) |

## 4. 与V4901核心差异

| 维度 | V4901（现有） | NG（新） |
|------|-------------|---------|
| 设计逻辑 | 数据挖掘驱动，因子来源杂 | 交易逻辑驱动：趋势→回调→确认 |
| 趋势因子 | 零散(return_Xd, rsi_14) | 12个系统化趋势状态因子 |
| 入场因子 | 无 | 10个回调入场因子（核心创新） |
| 量价关系 | 2个(volume_ratio, volume_trend) | 8个系统化量价因子 |
| 基本面 | 3个行业排名(pe/pb/ps) | 14个质量+估值+安全边际 |
| 市场环境 | 13个(偏散) | 10个围绕"做多环境"组织 |
| 行业 | 6个行业日度统计 | 8个围绕"行业轮动"组织 |
| 经济直觉 | 弱 | 强（每个因子有交易逻辑） |

## 5. 标签设计

保持多目标训练，调整重心：
- **label_5d**（主标签，权重0.50）：5-10天持仓的核心信号
- **label_10d**（辅助，权重0.35）：趋势延续性验证
- **label_3d**（辅助，权重0.15）：短期入场方向确认
- Composite排名权重：5d=0.50, 10d=0.35, 3d=0.15

## 6. 训练框架

复用V4901训练pipeline，关键参数：
- **Walk-Forward**: 3窗口, min_train=900d, val=120d, test=120d, step=120d, purge=10d
- **Ensemble**: LightGBM + XGBoost + CatBoost + RandomForest + HistGradientBoosting
- **LambdaRank**: truncation=10, 10档
- **Q95**: Widen-then-Concentrate (Top-30 → Top-10)
- **Sharpe-blend**: 0.3（默认）
- **Winsorization**: 1%/99% (train-only bounds, 无泄漏)

## 7. 选股约束

- **市值硬性下限**: 50亿流通市值
- **涨停/ST/停牌过滤**: 排除不可交易的股票
- **Top-N**: 10只
- **Focus days**: 10天（对齐5-10天持仓逻辑）

## 8. 评估方式

- **训练验证**: WF OOS IC/ICIR (每窗口每目标)
- **无泄漏回测**: 仅用WF OOS报告 + V5.2评分
- **快速验证**: --fast-check (2个紧凑WF窗口, ~3-5分钟)
- **成功标准**: V5.2无泄漏评分 > 64.0% A级（超过当前V4901基线）

## 9. 数据依赖

所有因子均可从现有数据库表计算，无需额外API调用：
- daily_quotes: OHLCV + MA + 涨跌停
- technical_indicators: KDJ, MACD, RSI, Bollinger, ATR
- daily_basic: PE, PB, 换手率, 市值, 股息率, 股本结构
- financial_indicator: ROE, ROA, 利润率, 现金流, 负债率（季度更新）
- hsgt_daily: 北向资金（仅1个因子）
- securities: 行业分类

## 10. 实现计划

1. **特征计算器**: `ml_models/ng/ng_feature_calculator.py` — 62个因子的计算逻辑
2. **特征缓存**: `ng_feature_cache` 表 — 预计算并缓存到SQLite
3. **训练器**: `ml_models/ng/ng_trainer.py` — 继承V485Trainer，覆盖特征加载
4. **评分器**: `ml_models/ng/ng_production_scorer.py` — 推理+选股
5. **报告生成**: 复用 `batch_generate_v395_reports.py` 添加NG版本
