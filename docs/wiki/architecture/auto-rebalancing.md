# 自动调仓系统

> EastMoneyTrader + StockTradebyZ 联合实现信用账户半自动仓位管理

## 系统架构

```
StockTradebyZ (大脑)                    EastMoneyTrader (执行)
├── NG1.0.1 ML评分                      ├── GUI自动化 (PyAutoGUI + Vision OCR)
├── 69特征 + 行业超额标签                  ├── 15步交易执行流水线
├── 报告: daily_selection_ng101/         ├── 风控: 止损6% + Regime门控
└── 北极星V5.2评估                       └── 信用账户100%管理
```

两个系统通过**报告JSON文件**解耦：StockTradebyZ 生成报告，EastMoneyTrader 读取并执行。

## 生产配置: NG1.0.5

NG1.0.5 = NG1.0.1 模型 + 三层风控叠加：

| 参数 | 值 | 作用 |
|------|-----|------|
| 模型 | NG1.0.1 (69特征, 行业超额标签) | 选股评分 |
| 止损 | 6% | 亏损超阈值强制卖出 |
| Regime门控 | aggressive | 熊市减半买入 |
| Vol目标 | 20% | 限制组合波动率 |
| CPPI | floor=8%, multiplier=20 | 动态敞口控制 |
| Score floor | 30分 | 过滤低质量信号 |
| Top-N | 10 | 最大持仓数 |
| Focus days | 15 | 调仓周期 |

**回测性能 (WF-OOS 2020-2026):** V5.2=78.9% A+, Sharpe=2.339, MaxDD=-12.6%

## 执行策略

经过 2024-2026 回测验证（48种参数组合扫描 + 290万条gap分析）:

| 策略 | 年化 | MaxDD | Sharpe |
|------|------|-------|--------|
| **Baseline Open Exec (采用)** | **71.4%** | **-6.9%** | **3.00** |
| Gap 3%/1% 过滤 | 60.7% | -7.0% | 2.86 |
| Smart V2 保守 | 63.8% | -5.4% | 3.08 |

**结论: 不做任何 gap 过滤，开盘价直接买卖，是收益最大化的策略。**

### 回测关键发现

1. **开盘价执行远优于收盘价** — Sharpe 2.44 → 3.00 (+23%)
2. **买入gap过滤反而损害收益** — 高开股往往是强势股，跳过=错失alpha
3. **高分股大跌是最强买入信号** — Score 85-100 + gap<-3% → 10日收益6.7%
4. **全卖优于减仓** — 模型说该卖就果断卖，犹豫拖累收益

## 每日操作流程

```
开盘前:
  1. StockTradebyZ 数据更新 + NG1.0.1 评分
  2. EastMoneyTrader calibrate + 生成调仓计划
  3. 用户审阅计划

9:25 集合竞价结束:
  4. tushare 读取开盘价
  5. 用户确认最终计划

9:30 开盘:
  6. 执行交易（先卖后买）
  7. 同步持仓确认结果
```

### 命令参考

```bash
# StockTradebyZ 数据更新
cd /Users/yangxu/StockTradebyZ
python3 fetch_data/quick_daily_update.py
python3 backtest/batch_generate_v395_reports.py --version ng1.0.1 \
    --start-date YYYY-MM-DD --end-date YYYY-MM-DD
cp reports/daily_selection_ng1.0.1_fast/analysis_data_YYYYMMDD.json \
    reports/daily_selection_ng101/

# EastMoneyTrader 调仓
cd /Users/yangxu/EastMoneyTrader
python3 trade.py calibrate                          # 校准UI坐标
python3 trade.py --account credit migrate           # 预览调仓计划
python3 trade.py --account credit migrate --confirm # 执行
python3 trade.py --account credit sync-positions --accounts credit  # 确认
```

## EastMoneyTrader 关键文件

| 文件 | 用途 |
|------|------|
| `trade.py` | CLI入口 |
| `daily_pipeline.py` | 每日编排器 (6-Phase) |
| `core/trade_executor.py` | 交易执行 (15步流水线 + 熔断) |
| `core/rebalancer.py` | 调仓计划生成 (NG1.0.5风控) |
| `core/strategy_bridge.py` | 连接StockTradebyZ报告 |
| `core/risk_guard.py` | 9层风控检查 |
| `core/cppi_manager.py` | CPPI敞口控制 |
| `core/screen_reader.py` | OCR持仓读取 |
| `config/settings.json` | 全局配置 |
| `config/calibration.json` | UI坐标 (窗口移动后需重新校准) |

## 已知问题与注意事项

1. **OCR 精度**: 个别低置信度字段（如小数价格）可能读错，用 market_value/qty 交叉验证修复
2. **窗口位置**: calibration 是绝对坐标，窗口移动后必须重新校准
3. **弹窗处理**: 确认弹窗(是/否)和成功弹窗(好)用坐标定位，不依赖OCR
4. **报告目录**: `batch_generate` 生成到 `ng1.0.1_fast/`，需手动 cp 到 `ng101/`
5. **非交易时段**: 可以提交委托但不会立即成交

## 回测基础设施

| 脚本 | 用途 |
|------|------|
| `backtest/execution_backtest.py` | 执行策略基线对比 + gap扫描 |
| `backtest/execution_backtest_v2.py` | Smart V2 动态仓位回测 |
| `backtest/gap_signal_analysis.py` | Gap×Score交叉收益分析 (290万条) |

## EMT 侧量化分析扩展 (2026-04-11/12)

EastMoneyTrader 新增独立的 `analysis/` 模块，不依赖 StockTradebyZ 内部工具：

- **因子诊断**: IC/IR、IC 衰减、分层回测、中性化（行业/市值）、时间趋势、拥挤度
- **组合优化**: Markowitz、Black-Litterman、风险平价、换手率约束、流动性约束
- **回测验证**: Walk-Forward、蒙特卡洛、多重检验校正
- **信号驱动调仓**: `HoldingsStateManager` 替代日频调仓，IC 决定持有期
- **退市股导入**: 修复幸存者偏差（322 只 + 257K 日线）

详见 [EMT 量化分析框架](emt-analysis-framework.md)。

关键诊断结果（ng1.0.1）:
- 双重中性后 IC 保留 53% → 模型有约一半 Alpha, 一半行业/市值 Beta
- IC 衰减峰值在 10 日 → 日频调仓过度, 最优持有 10-20 日
- 2024 IC 时间趋势 ratio=0.58 → 轻微衰减
