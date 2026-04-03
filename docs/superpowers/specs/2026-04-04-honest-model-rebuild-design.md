# 无泄露诚实模型重建 — 分阶段设计

> **日期**: 2026-04-04
> **目标**: 将V4.9.0.1模型从"3倍杠杆动量赌注"重建为"真实alpha选股模型"
> **评估体系**: 北极星V5.2 无泄露双向评估 (WF-OOS + PRE-2020)
> **当前基线**: WF-OOS B级 54.1% | PRE-2020 A级 66.7%

## 问题诊断

| 问题 | 证据 | 根因 |
|------|------|------|
| 隐性动量暴露 β_UMD=3.029 | L6=51.1%, 目标≤0.2 | 标签是原始收益，特征含纯动量指标 |
| Max DD -21.5% | L3=41.9%, 目标-8% | 动量反转时组合崩溃 |
| DSR仅1.1% | L4=50.9%, 目标95% | 过拟合+因子暴露=虚假Sharpe |
| 冲击成本8.1% | L7=70%, 目标0.5% | 小盘股无流动性约束 |
| 年化换手32x | L2=75.6% | retention_bonus=0.2不足以抑制换手 |

## 架构：三Phase + Fast-Check门控

```
Phase A: 标签净化 ──fast-check──→ Phase B: 特征强化 ──fast-check──→ Phase C: 组合优化 ──完整评估──→ 生产部署
   │                                  │                                  │
   ↓ 失败                              ↓ 失败                              ↓ 失败
 无真alpha,                          逐个回退改造                        逐个放松约束
 需全面重新设计特征                    找到有害改动                        (行业→流动性→换手)
```

---

## Phase A: 标签净化（因子残差标签）

### A1. 目标
消除训练标签中的系统性因子暴露，使模型学习纯alpha而非因子beta。

### A2. 当前标签
```python
# v39_feature_cache_updater.py:252
label_Nd = close[T+1+N] / open[T+1] - 1      # 简单前向收益
label_Nd -= industry_median(label_Nd, date)     # 行业中性化
```
行业中性化去除了行业beta，但跨行业的SMB/HML/UMD因子暴露仍然保留。

### A3. 新标签：因子残差收益

```python
# 复用已有infrastructure
from backtest.factor_returns import load_or_build_factors

def _residualize_labels(self, df, train_mask, val_mask, test_mask):
    """在每个WF窗口内独立计算因子残差标签（防止泄露）"""
    
    factors = load_or_build_factors(
        start_date=df['trade_date'].min(),
        end_date=df['trade_date'].max(),
        db_path=self.db_path
    )  # columns: MKT, SMB, HML, UMD
    
    for label_col in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
        N = int(label_col.split('_')[1].replace('d', ''))
        
        # Step 1: 用rolling 120天历史估计每只股票的因子beta
        # 只用train期数据估计beta（防止val/test泄露）
        betas = _estimate_rolling_betas(
            stock_returns=df[train_mask],
            factors=factors,
            window=120  # 120个交易日 ≈ 6个月
        )  # DataFrame: (code, date) -> (β_mkt, β_smb, β_hml, β_umd)
        
        # Step 2: 对val/test用最近已知的beta（不重新估计）
        betas_all = _extend_betas_to_valtest(betas, df, val_mask, test_mask)
        
        # Step 3: 计算N天期因子预期收益
        expected = _compute_expected_return(betas_all, factors, N)
        # expected_Nd = β_mkt·ΣR_mkt[T+1..T+N] + β_smb·ΣR_smb + ...
        
        # Step 4: 残差标签
        df[label_col] = df[label_col] - expected
```

### A4. 关键设计参数

| 参数 | 值 | 理由 |
|------|-----|------|
| Beta估计窗口 | 120天 | 平衡稳定性和时效性 |
| 最小有效天数 | 60天 | 不足60天用截面中位beta填充 |
| Beta估计频率 | 每日滚动 | 跟踪因子暴露变化 |
| 因子模型 | Fama-French 4因子 | MKT+SMB+HML+UMD，复用factor_returns.py |
| WF内执行 | 是 | 每个WF窗口独立估计，防止未来因子泄露 |

### A5. 实现文件

| 文件 | 改动 |
|------|------|
| `ml_models/training/train_v395_multi_target.py` | 新增 `_residualize_labels()` 方法，在 `split_data()` 后、模型训练前调用 |
| `backtest/factor_returns.py` | 无改动，直接复用 `load_or_build_factors()` |

### A6. Fast-Check门控

```
通过条件（任意满足即可继续）:
  ✓ 任意目标(3d/5d/10d/15d)的OOS IC > 0.02 且 ICIR > 0.15
  ✓ 至少2个目标IC为正

失败处理:
  ✗ 全部IC ≤ 0 → 当前特征集无真alpha，立即进入Phase B重新设计特征
  ✗ IC > 0 但 ICIR < 0.10 → 信号存在但极不稳定，仍进Phase B
```

### A7. 命令

```bash
# Fast-check验证 (~2分钟)
python3 ml_models/training/train_v395_multi_target.py \
  --v5 --fast-check --residual-labels --purge-days 15

# 如通过，完整训练 (~3-5小时)
python3 ml_models/training/train_v395_multi_target.py \
  --v5 --residual-labels --purge-days 15
```

---

## Phase B: 特征强化

### B1. 目标
移除动量代理特征，加入防御性特征，rank-transform消除尾部敏感。

### B2. 特征处理策略

**删除（2个纯动量）：**
- `market_momentum_20d`
- `market_momentum_5d`

**替换为残差动量（3个）：**
| 原特征 | 新特征 | 定义 |
|--------|--------|------|
| `return_5d` | `residual_mom_5d` | 5日收益 - β×市场5日收益 |
| `return_10d` | `residual_mom_10d` | 10日收益 - β×市场10日收益 |
| `return_20d` | `residual_mom_20d` | 20日收益 - β×市场20日收益 |

β用rolling 60天估计（短窗口跟踪快）。

**正交化（RSI/MACD）：**
```python
# 对RSI_14, RSI_6, MACD, MACD_signal, MACD_hist:
# 在每个截面日期，回归动量代理(原始20日收益)，取残差
# 注意：正交化在替换return_20d之前执行，使用原始return_20d作为动量代理
# 执行顺序: 1)正交化RSI/MACD → 2)替换return_Nd为residual_mom_Nd

for date in trade_dates:
    mask = df['trade_date'] == date
    for col in ['RSI_14', 'RSI_6', 'MACD', 'MACD_signal', 'MACD_hist']:
        raw = df.loc[mask, col]
        mom_proxy = df.loc[mask, 'raw_return_20d']  # 保留的原始动量（仅用于正交化）
        # 简单线性回归取残差
        beta = np.cov(raw, mom_proxy)[0,1] / (np.var(mom_proxy) + 1e-8)
        df.loc[mask, col] = raw - beta * mom_proxy
```

**新增防御性特征（6个）：**

| 特征 | 定义 | 改善目标 |
|------|------|----------|
| `idio_vol_20d` | 20日残差波动率(去市场beta后的std) | L3 风险控制 |
| `quality_score` | 4季度ROE的变异系数取反(-CV) | L3 尾部风险 |
| `earnings_momentum` | 最近2季EPS环比变化率 | L6 非价格动量 |
| `vol_of_vol_20d` | 波动率的波动率(std of rolling std) | L9 条件稳健 |
| `max_drawdown_20d` | 20日滚动最大回撤 | L3 下行信号 |
| `amihud_illiq_20d` | Amihud非流动性=mean(\|ret\|/volume, 20d) | L7 容量 |

**特征总数变化**: 61 - 2(删除) + 6(新增) = 65

### B3. 全特征Cross-Sectional Rank Transform

```python
# 在prepare_features()中，winsorization之后添加:

if self.use_rank_transform:  # --rank-transform 开关
    for date in df['trade_date'].unique():
        mask = df['trade_date'] == date
        for col in feature_cols:
            vals = df.loc[mask, col]
            df.loc[mask, col] = vals.rank(pct=True)  # [0, 1]
```

**位置**: `prepare_features()` 内，winsorization之后。rank-transform自然消除异常值，使winsorization效果更彻底。

### B4. 实现文件

| 文件 | 改动 |
|------|------|
| `fetch_data/v39_feature_cache_updater.py` | 新增6个防御性特征计算 + 残差动量计算 |
| `ml_models/training/train_v395_multi_target.py` | 特征列表更新；`prepare_features()` 加 rank-transform；正交化逻辑 |
| `data_adapter/stock_data.db` | v39_feature_cache表新增6列 |

### B5. Fast-Check门控

```
通过条件（对比Phase A基线）:
  ✓ IC不下降（允许±0.005波动）
  ✓ ICIR改善或持平
  ✓ 如果IC↑ > 0.005 → 明确成功
  ✓ 如果IC平 + ICIR↑ → 成功（信号更稳定）

失败处理:
  ✗ IC和ICIR都下降 → 逐个回退改造:
    1. 先回退rank-transform（可能损失量级信息）
    2. 再回退正交化（可能过度去噪）
    3. 最后回退新特征（可能引入噪声）
```

### B6. 命令

```bash
# Fast-check验证 (~2分钟)
python3 ml_models/training/train_v395_multi_target.py \
  --v5 --fast-check --residual-labels --rank-transform --purge-days 15

# 回填新特征到缓存 (需在fast-check前执行)
python3 fetch_data/v39_feature_cache_updater.py \
  --start-date 2018-01-01 --end-date 2026-04-04 --v5-features
```

---

## Phase C: 组合优化

### C1. 目标
在不损失alpha的前提下，降低冲击成本、控制行业集中度、减少换手。

### C2. 流动性加权评分

```python
def _apply_liquidity_penalty(self, stock_scores, date):
    """Soft penalty: ADV不足的股票降权"""
    target_position = self.portfolio_value / self.top_n  # 单只目标仓位
    max_participation = 0.02  # 单日不超过ADV的2%
    
    for code, score_data in stock_scores.items():
        adv_20d = get_adv_20d(code, date)  # 20日日均成交额
        if adv_20d > 0:
            participation = target_position / adv_20d
            penalty = np.clip(1.0 - participation / max_participation, 0.1, 1.0)
            # penalty下限0.1: 即使流动性极差也不完全排除
        else:
            penalty = 0.1
        score_data['adjusted_score'] = score_data['rank_score'] * penalty
```

### C3. 行业集中度限制

```python
def _build_portfolio_with_sector_limit(self, sorted_stocks, max_per_sector=3):
    """单行业最多3只（Top10中30%上限）"""
    sector_count = defaultdict(int)
    portfolio = []
    
    for stock in sorted_stocks:
        sector = get_sw_l1(stock['code'])
        if sector_count[sector] >= max_per_sector:
            continue
        portfolio.append(stock)
        sector_count[sector] += 1
        if len(portfolio) >= self.top_n:
            break
    return portfolio
```

### C4. 动态换手门槛

```python
def _apply_turnover_penalty(self, new_scores, current_holdings):
    """替换门槛: 新股score必须高出持仓stock至少round_trip_cost等效"""
    round_trip_cost = 0.006  # 双边交易成本0.60%
    
    # 转换为score单位: 如score是10日预期收益率
    # 需要新股alpha在1个调仓期内覆盖交易成本
    replace_threshold = round_trip_cost  # 直接在收益率空间比较
    
    for code in current_holdings:
        if code in new_scores:
            new_scores[code]['adjusted_score'] += replace_threshold
    
    # 效果: 持仓股有0.6%的"惯性加分"
    # 只有新股预期收益高出0.6%以上才会替换
```

### C5. 实现文件

| 文件 | 改动 |
|------|------|
| `ml_models/v39/v5_production_scorer.py` | **新建**，继承V4901，加入流动性惩罚 |
| `backtest/backtest_report_based.py` | `_rebalance()` 中实现行业限制和换手门槛 |
| `ml_models/training/train_v395_multi_target.py` | 训练后自动评估调用新scorer |

### C6. 完整验证

```bash
# 完整训练（Phase A+B标签和特征）
python3 ml_models/training/train_v395_multi_target.py \
  --v5 --residual-labels --rank-transform --purge-days 15 \
  2>&1 | tee logs/v5_honest_rebuild_$(date +%Y%m%d_%H%M%S).log

# 双向V5.2评估
python3 backtest/run_north_star_eval.py --production --score-version v52
```

### C7. 成功标准

| 指标 | 当前(WF-OOS) | 目标 | Phase负责 |
|------|-------------|------|-----------|
| **V5.2总分** | B 54.1% | **≥A 70%** | 综合 |
| L1 信号质量 | 71.7% | ≥65% (允许小降) | A |
| L2 组合效率 | 75.6% | **≥80%** | C |
| L3 风险控制 | **41.9%** | **≥55%** | A+B |
| L4 OOS鲁棒性 | 50.9% | **≥60%** | A+B |
| L5 超额收益 | 73.4% | ≥65% (允许小降) | — |
| L6 因子归因 | **51.1%** | **≥80%** | A |
| L7 容量 | 70.0% | **≥75%** | C |
| L8 执行质量 | 79.9% | ≥80% | C |
| L9 条件稳健 | 70.0% | **≥75%** | B |

**注意**: L1和L5允许小幅下降。因为消除因子暴露后"真实"信号会弱于"伪信号+因子beta"，但这是诚实的表现。

---

## 执行顺序与时间估计

```
Step 0: 回填2018-2020特征缓存（如未完成）         ~15min
Step 1: Phase A — 实现因子残差标签                  ~1h 编码
Step 2: Phase A — Fast-check验证                   ~2min
Step 3: Phase B — 实现特征改造                      ~2h 编码
Step 4: Phase B — 回填新特征缓存                    ~15min
Step 5: Phase B — Fast-check验证                   ~2min
Step 6: Phase C — 实现组合优化                      ~1.5h 编码
Step 7: 完整训练 + 双向评估                         ~3-5h
Step 8: 结果分析 + 参数微调                         ~1h
                                        总计: ~9-11h (含训练)
```

## 新增CLI参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--v5` | — | 使用V5训练器（因子残差+新特征） |
| `--residual-labels` | False | 启用因子残差标签 |
| `--rank-transform` | False | 启用cross-sectional rank transform |
| `--v5-features` | False | v39_feature_cache_updater中计算新防御性特征 |
| `--beta-window` | 120 | 因子beta估计滚动窗口天数 |
| `--max-participation` | 0.02 | 流动性惩罚:单日最大ADV占比 |
| `--max-per-sector` | 3 | 行业集中度:单行业最多N只 |
| `--replace-threshold` | 0.006 | 换手门槛:替换需要的最低alpha差 |
