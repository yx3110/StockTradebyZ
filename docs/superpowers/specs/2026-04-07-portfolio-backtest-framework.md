# 组合回测框架设计

**日期**: 2026-04-07
**目标**: 不动模型，通过组合构建规则优化提升超额收益、Sharpe和降低回撤
**基线**: ng1.0.2 等权Top-5, 10日调仓, 无止损 → IS年化+117%/Sharpe 2.62/MaxDD-7.7%, OOS年化+16.4%/超额+2.2%/MaxDD-20.3%

## 架构

单文件模块化引擎 `scripts/portfolio_backtest.py`，读取现有报告JSON + 行情数据，输出P&L和风控指标。支持参数网格搜索。

```
reports/*.json (模型输出) + daily_quotes (行情)
        ↓
  PortfolioBacktester (组合规则引擎)
        ↓
  NAV曲线 + Sharpe/MaxDD/超额/换手率
```

## 组合规则 (全部可配置)

### 1. 选股规则
- `top_n`: 持仓数量 (默认5, 可选5/10/15/20)
- `score_floor`: 最低composite阈值, 低于不入选 (默认0, 即不过滤)
- `holding_buffer`: 持仓缓冲数 (默认0)。已持有股票排名掉到top_n+buffer内仍保留, 超出才卖。减少换手

### 2. 仓位分配
- `equal`: 等权 (默认)
- `score_weighted`: 按composite值比例分配, Top-1权重 > Top-5
- `inv_volatility`: 按14日ATR倒数分配, 波动大的给少仓位

### 3. 调仓机制
- `rebal_days`: 固定调仓周期天数 (默认10)
- 调仓日: 检查最新报告排名, 应用holding_buffer后决定买卖

### 4. 止损
- `atr_stop_mult`: ATR止损倍数 (默认0=不止损, 推荐试2.0)
  - 止损价 = 入场价 - atr_stop_mult × ATR14(入场日)
  - **每个交易日检查**, 不等调仓日, 触发立即卖出
  - 空出仓位等下次调仓补入
- `max_loss_pct`: 备用固定百分比止损 (默认0=不启用, 推荐试0.08)
  - 止损价 = 入场价 × (1 - max_loss_pct)
  - 与ATR止损取更紧的那个

### 5. CPPI风控
- `cppi_floor`: 净值保护底线 (默认0=不启用, 推荐试0.05)
- `cppi_multiplier`: 杠杆因子 (默认20)
- 计算: `floor_nav = peak_nav × (1 - cppi_floor)`
- `exposure = min(1.0, cppi_multiplier × (nav - floor_nav) / nav)`
- 实际持仓 = 目标持仓 × exposure, 剩余转为现金

### 6. 成本模型
- `cost_per_side`: 单边交易成本 (默认0.15%, 含佣金+滑点)
- 每笔买卖按实际金额扣成本
- 涨停(price_change_pct >= 9.5%/19.5%)不可买入
- 跌停(price_change_pct <= -9.5%/-19.5%)不可卖出(止损延迟到可卖日)

## 数据加载

### 报告数据
从 `analysis_data_YYYYMMDD.json` 读取:
- `stock_code`, `pred_10d`, `composite`/`rank_score`
- 按 composite (优先) 或 pred_10d 排序

### 行情数据
从 SQLite `daily_quotes` 读取:
- `close`, `open`, `high`, `low`, `volume`, `price_change_pct`
- 用于计算ATR、止损判断、涨跌停检测
- 预加载整个回测期行情到内存

### 基准
沪深300 (`000300.SH`) 收盘价序列

## 输出指标

### 核心指标
- 年化收益(毛/净), 基准年化, 超额年化
- Sharpe (年化, 基于调仓期收益)
- MaxDD, MaxDD持续天数
- Calmar (年化收益/MaxDD)
- 胜率 (调仓期正收益占比)
- 换手率 (年化)

### 辅助指标
- 止损触发次数/占比
- CPPI减仓天数/占比
- 月度收益分解
- 分年收益分解

## 参数网格搜索

```python
grid = {
    'top_n': [5, 10],
    'holding_buffer': [0, 3, 5],
    'weighting': ['equal', 'score_weighted'],
    'atr_stop_mult': [0, 1.5, 2.0],
    'max_loss_pct': [0, 0.08],
    'cppi_floor': [0, 0.05, 0.08],
    'cppi_multiplier': [15, 20],
    'rebal_days': [5, 10, 15],
}
```

对每个参数组合:
1. 跑2024-2026 IS回测
2. 跑2018-2020 OOS回测
3. **两个时期都要 Sharpe>0 且 超额>0 才算有效**
4. 按 `0.5×IS_Sharpe + 0.5×OOS_Sharpe` 排序
5. 输出Top-10配置

## CLI接口

```bash
# 单配置回测
python3 scripts/portfolio_backtest.py \
  --report-dir reports/daily_selection_ng102 \
  --top-n 5 --rebal-days 10 --atr-stop 2.0 --cppi-floor 0.05

# 网格搜索
python3 scripts/portfolio_backtest.py \
  --is-dir reports/daily_selection_ng102 \
  --oos-dir reports/daily_selection_ng102_pre2020_v2 \
  --grid

# 对比多配置
python3 scripts/portfolio_backtest.py \
  --report-dir reports/daily_selection_ng102 \
  --compare "equal_top5,score_weighted_top5,equal_top10"
```

## 不做的事
- 不改模型、特征、标签
- 不做日内交易模拟(只用收盘价)
- 不做融资融券/做空
- 不做行业约束(那是模型层面的事)
