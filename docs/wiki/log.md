# 项目里程碑时间线

重大事件记录。格式：`YYYY-MM-DD | 类别 | 描述`。最新在前。

类别：`model` / `arch` / `fix` / `feature` / `data` / `eval`

---

2026-07-18 | data   | 官方0AMV(指南针活跃市值)入库+锚定外推 — CSV (1993-01~2026-07, 8165行) 导入 `market_amv_official` 表 (`scripts/import_amv_official.py`, is_simulated=0). 现有模拟 var1 与官方日度动态相关仅 0.74; 新递归模型 `Y_t = 0.93437·Y_{t-1}·(1+1.25·r_中证全指) + 5.802e-9·成交额(元)` (活跃筹码=随大盘漂移+日衰减6.56%+成交注入, `scripts/fit_amv_official.py` 可复现拟合) 日度相关 0.98, 一步MAPE 0.60%, 锚定外推 20天~5%/60天~7% 有界不发散. `indicators/market_amv.extend_amv_official` 挂 compute_and_save 每日自动外推 (is_simulated=1), 重导官方CSV自动覆盖刷新锚点. 官方序列 33 年可扩 regime 回测到 2005/2008/2015 大周期 (待做). 详见 `reports/system_evaluation/0AMV官方数据导入与拟合_20260718.md`

2026-07-13 | arch   | 训练提速 Tier0 全量实施 (bit-exact) — 75-agent 审计 (`reports/system_evaluation/训练提速审计_20260712.md`) 后落地: (1) `_group_indices_by_date` 一遍分组替换 4 处 O(日期×样本) `dates==d` object 掩码热点 (sharpe blend/LambdaRank relevance/日度+月度 IC), 分组本身 862x, 合成数据 bit-exact 等价测试全绿, NaN 日期响亮 ValueError; (2) V473 LGB/XGB/LambdaRank valid_sets 移除训练集 (三库 parity: best_iteration+预测 bit-exact); (3) 分位数+推荐阈值共享一遍 ensemble 预测; (4) NG downside 头复用 joblib 缓存 (原每 run 二次全量 SQL+JSON ~2min); (5) fast-check 复用滑窗机制截 train 窗 500d (`--fast-check-train-days`), 实测 36.7→25.8min, 剩余大头 RF/HGB/LambdaRank; (6) orjson 落环境. 默认关 gate 开关: `--is-ic-day-cap` (IS 诊断每日子采样, WFER 进北极星 L4 需档位对照) + `--hgb-early-stop` (HGB 唯一不早停占窗口 1/3, 改数值需验收); fast-check 只判方向建议直接带两开关. batchgen 与 ng_cache_updater 均新增 `--shards N` 日期分片并行 (报告输出/特征行均验证与串行字节一致 — 回填侧 97,837 行零差异; 回填 24 日基准 8 分片 2.8x, 全量预计 78→~25min; 部分失败 exit 1). fast-check 三级实测 36.7→25.8→17.5min (双开关). `scripts/prune_training_cache.py` 清理 4 月残留缓存释放 74GB. 硬件真相修正: 18 核/48GiB (非 16/64), 并发 seed 上限 2. ⚠️ ng_trainer 不传 --version 默认解析到 ng1.0.3 (非生产 ng1.0.1). 未做: float32/HGB+cap 全量 gate 验收 (挂计划内重训)/线程限制 (两次实验负收益, 永久驳回)

2026-04-25 | eval   | 急跌 panic-drop overlay 全 REJECTED — 用户提议 var1 单日 ≤ -2.3% 作牛→熊触发. 实现 5 种 variant (V16 immediate / V17 cooldown 3d / V18 cash 3d / V19 AND-position / V20 streak 2d) 在 V11 base 上叠加, 三窗口 × 7 variant 完整回测. **结果**: V18 直接破 2020 净年化 ≥ 92.5% gate (91.0% < 92.5%); V16/V17/V19/V20 与 V11 财报指标**完全一致到三位小数** (Sharpe 1.989/2.386/-0.498, 净年化 108.9/97.5/-17.8%). 76 个 panic 事件中 V16 仅在 8 天上覆盖 V11 的牛判定, 都落入 bull/bear 模型 forward return 接近的 2018-2020. 根因: V11 = `位置 AND (水上 OR 上升) + 3 日平滑`, 急跌日通常已破水上 + 加速条件, panic overlay 是结构性冗余. 与 4-22 "crisis overlay -6pp 全败" 一致. **不要再 grid search panic 阈值**, 想急跌防御应改 bear 子专家 (ng104 系列). DEFAULT_PRESET 保持 v11_loose_smooth3. V16-V20 代码永久保留作 reference. 详见 architecture/regime-classifier-v1.md §10
2026-04-25 | arch   | 0AMV regime classifier v1 (V11) 上线 — 替换旧 V3 strict (急涨/急跌+缓跌), 新规则: 位置 (var1>ma60) AND (MACD水上 OR 上升), 3 日 streak 才切换. 三窗口击败 V3: 2024-2026 +7.8pp 年化 (108→116, Sharpe 1.86→1.99), 2020-2026 +4.7pp, pre-2020 +3.5pp (-14.3→-10.8). MaxDD 同步收窄 ~2pp. 新建 `indicators/regime_classifier.py` 可组合框架 (原子信号+组合器+平滑器+11预设). market_amv 表 + ng106v2 历史已用 V11 重算/重生成. 实验留底未上线: bull=ng101 替代 ng107 在 V11 regime 下表面 +30pp 2024 改进, 与 4-22 v2 sub-expert swap 结论矛盾, 需同口径复测; V15 三重危机 cash filter pre-2020 表面 +74% 净, 但 Sharpe 7.285 是 framework sparse-trading annualization 膨胀 (cash 65% 天数), 与 4-22 "crisis overlay -6pp 全败" 矛盾. commit `b2a93580`. 详见 architecture/regime-classifier-v1.md
2026-04-21 | model  | ng1.5.0 REJECTED — ng1.4.0 + 5 Tier B regime-refined 特征 (4 stock + 1 market). 3-seed ensemble 完整训练 3h35m, Stage 4a (2024-2026 552d) V5.2=63.0% A < 70% 门槛, β_UMD=+1.42 (3× ng1.4.0 的 +0.56) 动量暴露爆掉, β_MKT 反转 -0.94, Sharpe=0.89 是 ng1.4.0 (1.04) 的 85%. Stage 3.5 (2025 only 242d) V5.2 raw 73% PASS 但 Stage 4a 跨 regime 失败, 2024 熊市暴露问题. 同 ng1.0.7 / ng1.2.x 失败模式 (regime 内化作特征 ≠ 外部 switch). 生产保持 ng1.0.1. Infra 永久保留 (schema/trainer ng1.5.x + Check 9 pkl metadata). 详见 memory/ng150_rejected.md
2026-04-20 | eval   | ng1.0.6 (0AMV) 综合复核胜出 — 1606 天完整 WF-OOS: V5.2=78.9% A+, 10d Sharpe=2.808/年化115.7%, 15d Sharpe=2.081/年化81%, β_UMD=+0.005 (1/76 ng1.0.1 的 +0.38, 动量暴露几乎为零). Pre-2020 复核是唯一正年化 (+0.7%) + 正 Sharpe (+0.18) 的版本. 唯一痛点: MaxDD=-21.4%~-22.9%, 约 ng1.0.1 (-11.7%) 的两倍, 需叠加 ng1.0.5 三层风控压制. 结论: 可能应切换生产 ng1.0.1→ng1.0.6+ng1.0.5
2026-04-20 | eval   | 老版本 Pre-2020 全量复核 — 用当前 V5.2 评分卡统一口径实测: ng1.0.3=42.5% C (vs 老 55.5% B) / ng1.0.4=38.8% C (老 45.5% C) / ng1.0.6=41.1% C / ng1.0.7=34.7% C (老 41.0% C). 全部老 Pre-2020 A+/B+ 数字都是 4-10 bug 遗留 ghost numbers. 但按实际年化+Sharpe, ng1.0.6 才是 Pre-2020 最强 (+0.7%/+0.18 唯一为正)
2026-04-20 | fix    | Pre-2020 73.7% A+ 文档 ghost number 订正 — 审计全仓发现 CLAUDE.md / ng-series.md / ng-factor-quality.md / 5 个 specs 引用的 "ng1.0.1 Pre-2020 V5.2=73.7% A+" 无法复现. 实测 4-12 bugfix pkl + 新报告: 10d V5.2=45.5% B, 5d V5.2=45.6% B; 老 4-10 pkl 最高 --production 55.2% B; 均远低 73.7%. 本质: 4-10 fix 后数字没刷新, 然后被后续 specs 反复引用. ng1.0.1 **WF-OOS A+ 仍然可信** (V5.2=73.4% A+, β_UMD=+0.38 t=5.4), 但不是 "双向 A+ 基座" — 是单边 WF-OOS A+ + Pre-2020 弱 alpha 模型. 详见 memory/ng101_pre2020_audit_2026_04_20.md
2026-04-20 | eval   | ng1.0.1 β 暴露审计 — WF-OOS 1874天 OLS: β_UMD=+0.38 (10d) / +0.22 (15d) 均 t>5 显著, R²<3%, 杜绝 v4.9.0.1 式隐性动量(3.029)泄露. Pre-2020 β 因小样本(24-72点) 统计不显著, 符号在 +3.06 (5d) 到 -4.78 (10d) 之间摇摆, 纯噪声. 因子归因 artefact 写入 logs/ng101_wfoos_factor_*.log

2026-04-12 | fix    | ng110 僵尸特征修复 — EMT 审计发现 ng110 声称 77 特征, 7 个新 cx_* 交互特征 (cx_beta_mkt_vol/cx_drawdown_regime/cx_ind_mkt_dir/cx_momentum_trend/cx_quality_stress/cx_value_bear/cx_vol_stress) gain=0/SHAP=0, cache 未回填导致训练时全 NaN. 实际有效特征 70 个. 修复: 回填 ng101_feature_cache.features_json + 重训 ng111
2026-04-12 | fix    | ng101 4 个特征定义 bug — EMT feature_audit 工具审计发现 4 对特征相关 >= 0.999, 全部定位为代码 bug. (1) volume_contraction ≡ volume_ratio_5d 同公式; (2) sw_index_return_5d 参数名误导, 实际是行业均值; (3) industry_relative_strength ≡ residual_return_20d 同公式; (4) revenue_growth 实际是 profit_to_gr (margin 指标), 非营收增长率 (应用 or_yoy). 修复: ng_feature_calculator.py 改名/合并/去重 + 拉 or_yoy + 重训
2026-04-12 | feature| EMT 特征审计 + 候选验证工具链 — analysis/feature_audit.py (gain/SHAP/单因子IC/相关性矩阵四维评估) + scripts/audit_ng_features.py + analysis/feature_validator.py (候选新特征四关验证: IC/分组/冗余/LGB增量训练) + scripts/validate_feature.py. 全量跑 ng101+ng110 四 target 20s, 输出 logs/feature_audit/. 审计本身就是头部量化日常工作的 70%
2026-04-12 | eval   | 8策略长期回测(2018-2026) — 40,290信号/376采样日. 暴力K最强(+1.59% 10d alpha, Sharpe=1.96), 知行-0.05%纯噪音. Regime分化: 少负=牛市特化(Sharpe=1.75), 暴力K=熊市特化(Sharpe=2.05). 6月窗口结论被长样本推翻. 详见 evaluation/quant-strategies-2018-2026.md
2026-04-12 | eval   | 退市股对 IC 无影响验证 — 2024 全年 IC 导入前后均 0.1902, 真 Alpha 占比均 47%. 因 StockTradebyZ 生成报告时已过滤退市股, 历史 IC 本无幸存者偏差. 数据价值留给未来重训练
2026-04-12 | data   | 退市股数据导入 — tushare pro.stock_basic(list_status='D') + pro.daily, 322 只退市股 + 257,416 条日线, 修复 EMT 侧 IC 分析的幸存者偏差
2026-04-12 | arch   | EMT 组合优化扩展 — markowitz/BL 加换手率约束(LP+trust-constr)+流动性约束(max_position_value/budget), 回退到 current_weights 保证安全
2026-04-12 | feature| EMT IC 时间趋势监测 — compute_ic_time_trend 滚动60日, 前后半段对比判定因子衰减. 实测 ng1.0.1 在 2024 下半年比上半年 IC 降 42%(轻微下降)
2026-04-11 | arch   | EMT 信号驱动调仓 — HoldingsStateManager(entry_date/entry_score/cooldowns) 替代日频调仓; 触发: 止损6%|min_hold7天|评分地板25|评分衰减30%|超持有20天. 冷却期7天
2026-04-11 | feature| EMT 因子中性化诊断 — neutralize_factor(行业/市值 OLS残差), ng1.0.1 双重中性后 IC 保留 53%, 说明约一半收益是 Beta
2026-04-11 | arch   | EMT analysis/ 模块构建 — 完整 Level 1 量化框架: data_loader/ic_analyzer/portfolio_optimizer/backtest_framework 四个子模块, CLI `trade.py analyze --mode full`. 见 wiki/architecture/emt-analysis-framework.md
2026-04-11 | eval   | NG v1.0.9 折中方案(31feat,ac≥0.4) — 换手41x/28x(+sell50), Sharpe=1.69/1.54. fast-check ICIR=1.38但生产仅0.3. 最终结论: ng1.0.8(sell50 on 1.0.1)=最优(Sharpe=2.52,换手36x,A+)
2026-04-11 | eval   | NG v1.0.9 完整评估(22feat) — 换手14.7x达标但Sharpe=0.79. 慢变特征短期alpha不足
2026-04-11 | model  | NG v1.0.9 持久特征 — 22个慢变特征(autocorr≥0.5), fast-check 10d ICIR=1.29(+39% vs ng1.0.1)
2026-04-10 | model  | NG v1.0.8 低换手组合 — sell50+cost0.3%最优: 换手45x→36x(-20%), Sharpe=2.52(+6%), 净收益94.6%. 初版bug(每天调仓)已修复
2026-04-10 | fix    | ng108 is_rebal_day缺失 — 每天执行ng108逻辑导致换手率未降(45→44x), 加i%rebal_interval检查后正常(→36x)
2026-04-10 | fix    | Pre-2020评估bug — --start-date不过滤backtest日期, 之前"Pre-2020 A+"是全量数据. 真实Pre-2020: ng1.0.1=B(0.40)
2026-04-10 | eval   | ng108参数搜索: sell20→sell50→sell100→sell200, sell50+cost0.3%是甜蜜点(Sharpe/换手/收益平衡)
2026-04-10 | eval   | Pre-2020完整排行 — ng1.0.5=A+(Sharpe3.09)风控最强, ng1.0.1=A+(2.37)信号最强, ng1.0.6=C(0.18), ng1.0.7=C(-1.06)过拟合. 结论: ng1.0.1+1.0.5风控是最可靠组合
2026-04-10 | model  | NG v1.0.7 发布 — 条件化标签+18市场特征+Pareto过滤, V5.2=76.8% A+, L1信号91.3%(全系列最高), ICIR=0.656(+29%), 熊市ICIR=0.205(解决反转)
2026-04-10 | fix    | /simplify 3轮审查 — N+1查询(3.7x提速)+4个bug(IX筛选/scorer加载/train-serve skew/Pareto)+features_json列名冲突
2026-04-09 | fix    | daily update缓存修复 — update_ng_feature_cache()同时更新ng1.0.3/1.0.1/1.0.4三版本缓存
2026-04-09 | fix    | SCORER_REGISTRY修复 — ng1.0.1/1.0.2显式指定model_path，避免误加载最新pkl
2026-04-09 | eval   | ng1.0.4 RF权重失衡 — 10d/15d RF占94-95%, Top-10全银行, 银行得分=全市场6.7x
2026-04-09 | model  | NG v1.0.6 发布 — 0AMV牛熊切换(ng1.0.1牛+ng1.0.4熊), 年化92.0%, MaxDD=-20.2%, 18次切换
2026-04-09 | feature| 0AMV全市场活跃市值指标 — 复刻指南针活筹指数, indicators/market_amv.py
2026-04-09 | fix    | factor_returns.py Build路径string index bug修复(因子归因全零)
2026-04-09 | eval   | NG裸模型公平对比 — ng1.0.1最强(年化129.7%, Sharpe3.17), ng1.0.4熊市最稳
2026-04-08 | eval   | NG版本综合排名 — ng1.0.1+CPPI(F0.08,M20) WF-OOS最优(V5.2=78.9% A+, Sharpe=2.339, MaxDD=-12.6%)
2026-04-08 | model  | 生产切换到ng1.0.1+CPPI(F0.08,M20), 替代ng1.0.2
2026-04-08 | model  | NG v1.0.4 发布 — RA标签+5-seed ensemble+9新特征+IC分析器, V5.2=75.9% A+
2026-04-08 | fix    | /simplify审查 — version_ge安全比较+vol_regime向量化+NaN处理+8项修复
2026-04-07 | model  | NG v1.0.3 发布 — 去3翻转因子(66feat), 2018-2020 OOS年化+18.1%/超额+24.8%
2026-04-07 | fix    | 发现Pre-2020 OOS评估bug(composite=0→随机选股), 修复后ng1.1.0实验重新评估
2026-04-07 | model  | NG v1.1.0 废弃 — 三方向仅"去翻转因子"有效, 合并为ng1.0.3
2026-04-06 | arch   | 建立项目 Wiki 知识库（docs/wiki/），Karpathy LLM Wiki 模式
2026-04-05 | model  | NG v1.1.0 实验 — 残差标签/资金流/WF升级三方向(后废弃, 见ng1.0.3)
2026-04-05 | model  | NG v1.0.2 生产配置 — 下行风险模型 + CPPI(5,20), V5.2=74.0% A+
2026-04-05 | model  | NG v1.0.1 发布 — 行业超额标签+ICIR权重, 10d ICIR 0.515→0.931(+81%)
2026-04-04 | model  | V5.0 诚实重建 — 因子残差标签+Rank-Transform, 消除 β_UMD=3.029
2026-04-04 | eval   | V4.9.0.1 无泄露真实评分: WF-OOS B级 54.1%, PRE-2020 A级 66.7%
2026-04-04 | model  | NG v1.0.0 首版 — 62因子独立 trainer/scorer/cache, 从V4.x独立重构
2026-03-24 | model  | V4.8.6 因子挖掘 — fast最优86.59, WF全量58.44(严重过拟合)
2026-03-07 | model  | V4.7.5 特征裁剪+连续评分 — pred_10d 77/105 A+, 50特征
2026-03-05 | model  | V4.7.3 裸信号提纯 — 去Meta-Learner, 75/105 A+, Sharpe=1.230
2026-03-04 | model  | V4.7.2 composite排名 — 解决全局百分位同分问题, 71/105 A
2026-03-03 | eval   | L4优化实验 — 旧模型+CPPI 75/105 A+, 新模型(四分位权重)71/105 A
2026-03-01 | fix    | 回测12-bug修复 — 涨停检测pct化, 交易成本0.15%→0.302%, CPPI调仓成本
2026-03-01 | eval   | V4.6.2 CPPI网格搜索 — 天花板85/105 S, 无法超越
2026-02-28 | model  | V4.6 ICIR权重+Meta-Learner — base 64/105 A, +CPPI 84/105 S
2026-02-26 | feature| V4.5 CPPI Overlay — 84/105 S级, MaxDD -19.7%→-8.1%
2026-02-26 | feature| Portfolio Pilot Score仓位领航评分 — 4层/20指标/100分
2026-02-25 | fix    | 训练/推理数据一致性修复 — 4大问题(市场公式/缓存缺失/Winsorize/fallback)
2026-02-25 | model  | V4.4-aligned 88/105 S级 — 仅修复数据对齐就从B到S
2026-02-24 | model  | V4.3 Walk-Forward+强正则化 — WF成为标准评估方法
2026-02-24 | model  | V3.95 Phase3 组合优化 — 31/33 A+(V1评分), Sharpe=2.94
2026-02-23 | model  | V3.95 RobustZScore 最优 — 行业超额标签, 全周期ICIR>0.2
2026-02-23 | model  | V4.0 Cross-Sectional Alpha — 失败, 纯截面丢失择时信号
2026-02-22 | model  | V3.9 6年数据重训练
2026-02-17 | fix    | auto-claude反复清空仓库 — 核心代码commit(661 files), 数据库备份方案
2026-02-16 | data   | 完整备份 ~/StockTradebyZ_backup_20260216/
