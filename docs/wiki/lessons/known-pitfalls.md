# 已知陷阱

项目历史中踩过的坑，避免重复犯错。按类别组织，每个陷阱记录：现象、根因、解决方案。

---

## 数据类

### 股票代码必须带交易所后缀
- **现象**: 查询/匹配失败，数据关联错误
- **根因**: A股代码不唯一，需要 `.SZ`/`.SH` 区分
- **解决**: 全系统统一使用 `000001.SZ`、`600519.SH` 格式

### price_change_pct 单位不一致
- **现象**: 波动率计算异常，旧数据和新数据结果差异巨大
- **根因**: 旧数据(2025年前)是百分比格式(2.16=2.16%)，新数据(2026年)是小数格式(0.012=1.2%)
- **解决**: 波动率从 close 价格直接计算，避免依赖 price_change_pct 字段

### 行业数据来源
- **现象**: 用 `stock_basic_info.industry` 查不到行业或数据不全
- **根因**: 行业分类在 `securities` 表的 `industry` 字段，不在 `stock_basic_info`
- **解决**: 统一从 `securities` 表读取行业信息

### Tushare API 分页查询
- **现象**: 指数成分股查询返回不全或超时
- **根因**: 部分 API 有单次返回行数限制
- **解决**: 使用 `pro.index_member_all()` 做分页查询，不要用 per-industry 调用

---

## 特征工程类

### 特征重复 / 同义重命名
- **现象**: 两个特征名不同但计算完全相同 (相关系数 1.000), 变成"训练数据的噪声副本", 抢夺 LGB 每棵树的 feature_fraction 采样槽位, 稀释真正有效的特征.
- **历史案例** (2026-04-12 EMT feature_audit 发现, ng1.0.1 有 4 对):
  1. `volume_contraction` (`:183`) 和 `volume_ratio_5d` (`:224`) 都是 `vol5_mean / vol20_mean`
  2. `industry_relative_strength` (`:585`) 和 `residual_return_20d` (`:738`) 都是 `stock_20d - industry_20d`
  3. `sw_index_return_5d` (`:616`) 参数名暗示 SW 指数收益, 实际接的是 `ind_agg['mean_return_5d']` (行业均值), 和 `industry_return_5d` 等同
  4. `revenue_growth` (`:426`) 被赋值 `profit_to_gr` (利润总额/营业总收入, 是 margin 指标), 和 `net_profit_margin` 相关 0.999. 真正的营收增长率字段是 `or_yoy`
- **解决**:
  1. 每次新增特征后, 跑 EMT 的 `scripts/audit_ng_features.py`, 检查冗余对报告
  2. 相关 > 0.95 的特征对强制人工 review
  3. 命名要匹配实际计算 (e.g. `profit_to_gr_ratio` 而不是 `revenue_growth`)

### 僵尸特征 — 声明但训练时全 NaN
- **现象**: 模型元信息声明 N 特征, 但部分特征的 LGB gain importance = 0 + SHAP = 0, 实际 OOS 不起作用.
- **历史案例** (2026-04-12 ng1.1.0): 声明 77 特征 (67 stock + 10 market), 但 7 个新增的 `cx_*` 交互特征 (`cx_beta_mkt_vol` 等) 在 `ng101_feature_cache.features_json` 里根本不存在. 训练时这些列全 NaN, LGB 从未分裂, 实际有效特征只有 70 个. 模型部署了一个"纸面规模"虚高的版本.
- **根因**: feature engineering 升级 (新增 cx_*) 只改了 calculator 和模型代码, **没有同步回填 cache**. 训练时不会报错因为 LGB 容忍 NaN, 但特征的 predictive power 为零.
- **解决**:
  1. 新特征上线前, 必须先 backfill `*_feature_cache.features_json` 再重训
  2. 训练后立即跑 EMT `audit_ng_features.py` 验证: **每个声明的特征都应该有非零 gain 和非零 SHAP**
  3. 模型签收条件: gain=0 的特征数量应为 0 (或 < 3%)

### `factor_quality_*.json` 只能反映训练数据分布, 不是 OOS
- **现象**: 训练日志里 ICIR 很高, 但上线后 IC 衰减快
- **根因**: train_v395 的 factor_quality 字段是训练集内部 IC, 不是 WF-OOS IC
- **解决**: 始终以 `wf_summary.json` 为准; `factor_quality_*.json` 只能做训练方向性参考

---

## 数据库类

### SQLite 并发死锁
- **现象**: 多进程同时写数据库时报 `database is locked`
- **根因**: SQLite 默认 busy_timeout 太短
- **解决**: 设置 `busy_timeout=30000`（至少30秒）

### 多进程 Pool pipe 死锁
- **现象**: `multiprocessing.Pool` 传递大数据时卡死不返回
- **根因**: Pool 通过 OS pipe 传递 `computer_data`（5700个 DataFrame），超出 pipe buffer 64KB
- **解决**: `backfill_v39_complete.py` 添加 `--num-workers 0` 顺序模式；推荐使用 `fetch_data/v39_feature_cache_updater.py` 代替（每天1.8秒 vs 数分钟）

### 数据库恢复流程
- **现象**: auto-claude 自动任务做 git merge/checkout 时删除未 commit 的文件
- **根因**: 数据库文件太大无法 git commit，被 .gitignore 忽略
- **解决**: 所有核心代码已 commit；数据库需从备份恢复
- **备份路径**: `~/StockTradebyZ_backup_20260216/data_adapter/stock_data.db.backup_20251103_231057`

---

## 模型训练类

### 数据泄露 — --production 回测
- **现象**: --production 回测显示 S 级评分，但实际 OOS 表现远低于预期
- **根因**: 训练数据和回测数据有重叠，评分虚高
- **解决**: 必须使用 WF-OOS + Pre-2020 双向无泄露评估；`--production` 仅供内部参考
- **参考**: [回测方法论](../evaluation/backtesting.md)

### 四分位小盘加权扭曲学习
- **现象**: V4.6 重训练后 WF IC 近随机（W3 10d=0.002），年化仅 39.5%
- **根因**: bottom 25% 市值股票样本权重 ×2.5，过度加权噪声样本，损害预测能力
- **解决**: 保留旧模型 `v46_multi_target_20260301_050112.pkl` 为生产版；移除极端小盘加权

### Composite 排名不总是最优
- **现象**: 多周期 composite 评分有时反而低于单周期 pred_10d
- **根因**: 特征裁剪可能提升某个预测周期但降低其他周期，混合后稀释最优信号
- **解决**: 不能假设 composite 总是好的，需要实测对比
  - V4.7.3: composite=75 > pred_10d=63（composite 有益）
  - V4.7.5: pred_10d=77 > composite=69（composite 有害）

### Meta-Learner + Combined Isotonic 压缩头部区分度
- **现象**: V4.7.4/V4.8 评分退化
- **根因**: Meta-Learner + Combined Isotonic 校准把头部预测值压缩到窄区间，top-10 严重同分
- **解决**: V4.7.3 去掉这两层，直接用裸信号；V4.7.5 进一步裁剪特征

### 全局百分位离散化导致同分
- **现象**: `_to_global_score()` 输出 0-100 整数，平均 67 只股票同分竞争 top-10
- **根因**: 离散化丢失细粒度排名信息
- **解决**: 使用 `rank_field=composite` 或 `rank_field=pred_10d` 连续值排名

### 隐性动量暴露
- **现象**: V4.9.0.1 WF-OOS 仅 B 级 54.1%，MaxDD=-21.5%
- **根因**: β_UMD=3.029，模型学到了动量因子但没有显式控制
- **解决**: V5.0 诚实重建，使用因子残差标签 + Rank-Transform 消除动量暴露

---

## 回测类

### 涨停检测方法
- **现象**: `is_limit_up` 字段不可靠
- **根因**: 数据源的 `is_limit_up` 标记不准确
- **解决**: 用 `price_change_pct` 判断：主板≥9.5%、创/科≥19.5%、北交≥29.5%

### 交易成本低估
- **现象**: 回测收益显著高于实盘
- **根因**: 初始只算了 0.15% 单边手续费
- **解决**: 总成本 0.302%（含双边滑点 0.1% + 过户费）；CPPI 调仓 exposure 变化 >1% 时扣减

### 稀疏 trading 年化膨胀 (sparse-trading annualization)
- **现象**: 加 cash filter 跳过 50%+ 天数的策略报告 Sharpe>4 / 年化 200-500%, 看似 SOTA 但与历史结论矛盾
- **根因**: `backtest_report_based.run_single_backtest` 用 `n_dates`(=报告字典 size) 算年化, cash 跳过的日子从分母里消失, 等价"模型每年只工作 X% 时间", 单位时间收益被放大 3-5 倍
- **解决**:
  1. Sharpe>4 立即怀疑 — 检查 cash 比例
  2. cash >20% 的策略, 必须算 calendar-time 年化 `(1+cumret)^(252/total_calendar_days)-1` 做对照
  3. 决策依据用 robust 指标: MaxDD (天然 calendar-based), cumulative return over fixed span, monthly win rate
  4. 跨多次实验对比时, 同口径 (相同 cash 比例 / 相同 framework 计算方式) 才公允
- **历史**: 2026-04-25 V15 三重危机 cash filter pre-2020 表面 +74% 净 / Sharpe 7.285, 实际 calendar-time 估打 3-5 折; 与 4-22 "crisis overlay -6pp 全败" 矛盾

---

## 工程类

### .pkl/.joblib 模型文件不能 git add
- **现象**: git repo 膨胀，push 失败
- **根因**: 训练好的模型文件动辄几十 MB~几百 MB
- **解决**: 确保在 `.gitignore` 中排除

### 性能优化前先 profile
- **现象**: 凭直觉优化错误的瓶颈
- **根因**: 数据管线的瓶颈往往不在直觉判断的位置
- **解决**: `python -m cProfile` 先定位，再优化

### Fast-check 再完整训练
- **现象**: 花 5-19 小时完整训练后发现方向错误
- **根因**: 没有快速验证就投入全量训练
- **解决**: `--fast-check` 模式（2个WF窗口，~2min）先验证 IC/ICIR 方向，通过后再完整训练

### 条件化标签/市场regime特征过拟合 (2026-04-10, ng1.0.7)
- **现象**: ng1.0.7训练期内信号质量L1=91.3%(全系列最高), 但Pre-2020=C级(年化-36%)
- **根因**: 条件化标签(熊市用rank_pct blend)和8个扩展市场特征(0AMV连续值等)过拟合了2020+的市场regime模式。2018-2019的regime分布与2020+显著不同，模型学到的"熊市偏好"在旧数据上不适用
- **教训**: 
  1. WF-OOS好(A+)不代表模型泛化好——Pre-2020双向评估是检验过拟合的金标准
  2. 市场regime特征本质上是非平稳的，GBDT容易记住训练期regime模式
  3. 交叉特征(stock×market)全被IC筛选淘汰，说明GBDT自身已能学到交叉效应
  4. ng1.0.1(纯行业超额标签+69特征)反而是跨regime泛化最强的——简单模型胜过复杂模型
- **解决**: 不使用regime-conditional标签；如需regime感知，用后置风控(ng1.0.5)而非改变模型标签

### Fast-check ICIR高 ≠ 生产ICIR高 (2026-04-11, ng1.0.9)
- **现象**: 22/31个持久特征fast-check 10d ICIR=1.29-1.38(+48%), 但生产评估ICIR仅0.26-0.30
- **根因**: WF训练窗口内(IS+OOS)的IC高估了模型在全新数据上的表现。特征减少后模型更稳定(IC方差小→ICIR高), 但IC绝对值在生产推理时下降
- **教训**: fast-check验证方向, 但不能作为生产性能的可靠预测。**必须生成报告+北极星评估**才能确认真实表现
- **A股特殊性**: 短期alpha主要来自动量/技术特征(autocorr<0.3), 基本面特征(autocorr>0.5)的10天选股能力不足

### 持仓缓冲sell_threshold必须远大于top_n (2026-04-10, ng1.0.8)
- **现象**: sell_threshold=20配合top_n=10, 换手率仅从45x降到40x(几乎无效)
- **根因**: A股信号衰减快，10天后Top-10中8.2只跌出Top-20。buy=8/sell=20的区间太窄
- **解决**: sell_threshold=50才有效(45→36x), sell=100降到31x但收益开始下降
- **教训**: 持仓缓冲的sell_threshold应至少是top_n的5倍(50/10), 而非2倍(20/10)

### RF权重失衡导致行业集中 (2026-04-08, ng1.0.4)
- **现象**: Top-10几乎全是银行股
- **根因**: ICIR优化后RandomForest权重94-95%，RF天然偏好低波动→银行垄断
- **解决**: 限制单模型在ensemble中最大权重；或加行业分散约束

### 短期回测窗口得出错误策略排名 (2026-04-12, 8策略评估)
- **现象**: 6个月窗口(2025-09~2026-02)显示 SuperB1 最强(+2.68%), 暴力K 最差(+0.82%); 2018-2026 长窗口排名完全反转: 暴力K(+1.59% alpha) > SuperB1(+0.12%)
- **根因**: 策略表现有强 regime 依赖 — 少负=牛市特化(Sharpe=1.75), 暴力K=熊市特化(Sharpe=2.05), 上穿60放量在牛市反而 -1.19%。单一 regime 窗口会把"regime 幸运"伪装成"策略优劣"
- **教训**: 评价任何选股策略的 alpha 必须跨越至少一个完整市场周期(>=3年, 含牛熊转换)
- **解决**: 长期回测脚本 `backtest/backtest_strategy_metrics.py` + regime 拆解 `backtest/analyze_strategy_longhorizon.py`; 详见 [8策略长期回测](../evaluation/quant-strategies-2018-2026.md)

### Score-scale 量纲混淆: 0-100 阈值 × NG 预测收益 score (2026-04-28, P0.1+P2.8 双 regression)
- **现象**: 2026-04-09 起生产 `reports/daily_selection_ng106` 不再有新报告; 手动跑 selector 报告 "全市场总股票: 0只" 但日志显示 ML 评分了 7376 只. 启用 `--enable-booster` 后 Top-10 全是 rank_score=0 的策略候选股 (rsb=8.0), 真 ML picks (rank_score=0.003) 反被排到底部.
- **根因**: 跨世代代码做了"阈值 / 加权"操作, 假设 score 在 0-100 区间, 但 NG 模型 (ng1.0.1/ng1.0.4) 的 `rank_score` 是预测收益小数, 区间 ≈ [-0.05, +0.02].
  1. **P0.1 overlay**: `apply_overlay_to_picks` 用 `L1_SCORE_FLOOR=30` 做硬阈值, NG max(rank_score)≈0.016 全部 < 30 → 全 drop → 0 picks
  2. **P2.8 booster**: `_strategy_bonus` 加 8 pts (V3 0-100 量纲) 到 NG 0.003 量级 score, 量级差 2700×, 让 ML 未评分的策略候选股 (rank_score=0+bonus 8) 完全压过真 picks
- **解决** (commit `f87a7671` + `6a8ec051`):
  - overlay: 检测 `max(scores) < 1.0` 时切到 percentile floor (默认底 10%), 否则保持绝对 floor=30
  - booster: 同样 NG-scale 检测, `bonus_scale = pos_max/100` 让 1pt bonus ≈ 1% lift; 同时 `skip_when_score_zero=True` 防 ML 未评分的 strategy 候选股错误上位
- **教训**: 跨版本代码若涉及"score 阈值/加权", 永远先 `df.score.describe()` 看分布, 不要假设量纲. 写测试时既测 0-100 也测 [0, 1) 两种 scale.
- **未来防御**: 任何 score-comparison 函数应该: (a) auto-detect scale, OR (b) 文档明确入参 scale + 在外层 normalize. 全栈静态约定不可靠.

### sqlite3 不接受 pd.Timestamp 绑定 — L4 熔断静默失效 2.5 个月 (2026-07-11)
- **现象**: 生产日志每天出现 `L4 crisis check skipped (DB error: Error binding parameter 1: type 'Timestamp' is not supported)`, 从 2026-04-28 上线起熊市危机熔断从未真正评估过
- **根因**: 双重叠加 — (1) selector 把 `pd.Timestamp` 原样传给 `WHERE trade_date = ?` 绑定; (2) 风控层 except 全部只 `logger.warning` 后继续, 失败被淹没在日志里
- **解决** (commit `977ff401`): 日期入口统一 `_norm_date()` 归一化为 'YYYY-MM-DD' 字符串; 风控层 (post_filters/booster/overlay/L4/est_vol) 异常升 ERROR + `analysis['risk_layer_degraded']` 报告头部盖戳
- **教训**: **风控层的失败模式必须"高可见 fail-open"而不是"静默 fail-open"**。任何 `except → warning → 继续` 的风控代码都等价于没有风控。同类事故: est_vol 因 `'code'/'stock_code'` 键错配恒走 0.25 fallback (仓位系统性超配 50%), 也是靠 warning 埋了 2.5 个月

### "选股管道过滤" 不能截断全市场数据产物 (2026-07-11)
- **现象**: 全市场报告/JSON 从 7343 只 → 440 只 (04-23 post_filters industry_cap) → 10 只 (04-28 overlay top_n); 下游 Signal Trust 贴标签、forward tracker、OOS `pred_10d` 非零校验、webapp 全部失真
- **根因**: 给 Top-N 选股设计的过滤器 (行业限流/信任过滤/overlay 截断) 被直接赋值回 `all_stocks_with_scores`
- **解决** (commit `977ff401`): 风控筛选链在 `pick_pipeline` 副本上执行; 全市场列表保持完整, 淘汰股打 `_post_filter_drop`/`_drop_reason` 标记; overlay 最终持仓单独存 `analysis['risk_overlay']`
- **教训**: 区分两种数据产物 — "全市场排名表" (信息完整性优先) 和 "最终持仓清单" (风控优先)。过滤器只能作用于后者

### 模型加载 mtime-glob = 生产模型可被任何重训静默顶替 (2026-07-11)
- **现象**: ng1.0.6 生产 bull 专家自 04-28 起实际是未验收的重训 pkl; bear ensemble glob 到 7 个 pkl (seed42×3 + 5-seed 实验遗留), 偏离已验证 3-seed 配置
- **根因**: scorer 按 `glob + mtime 最新` 选模型, 任何实验性重训/rename 失败的 specialist pkl 都会自动"上生产"
- **解决** (commit `00f79e53`): `ng_schema.PINNED_PRODUCTION_MODELS` 固定生产 pkl 文件名; ensemble 按 seed 去重; ng21 specialist pkl 打 `ng21_variant` 标记 + scorer 拒载
- **教训**: 生产模型必须显式 pin (文件名/checksum), 换模型 = 改注册表 + review, 绝不能靠文件系统 mtime

### SQLite 类型排序: INTEGER 恒 < TEXT — 日期双格式劈表 (2026-07-11)
- **现象**: financial_indicator 81% 行 (TEXT 'YYYY-MM-DD') 对 NG 特征的整数边界查询 (`ann_date BETWEEN 20250606 AND 20260710`) 完全不可见; roe/or_yoy 等生产特征大面积静默 NaN; 同一财报双格式共存 37,556 对击穿 UNIQUE 约束
- **根因**: 两个写入方约定不同 — backfill 写 TEXT 带杠, quick_daily 写 Tushare 原始整数; SQLite 动态类型允许同列混存, 比较时按类型排序 (INTEGER < TEXT), 不做隐式转换
- **解决** (commit `0f1d09d6`): `scripts/migrate_financial_indicator_date_format.py` 一次性迁移 (备份→去重→统一 TEXT); 读写两侧统一 'YYYY-MM-DD'
- **教训**: SQLite 日期列必须全仓约定单一格式 ('YYYY-MM-DD' TEXT); 新写入方上线前 `SELECT typeof(col), COUNT(*) GROUP BY 1` 验证与存量一致
