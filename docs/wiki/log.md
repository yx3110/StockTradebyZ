# 项目里程碑时间线

重大事件记录。格式：`YYYY-MM-DD | 类别 | 描述`。最新在前。

类别：`model` / `arch` / `fix` / `feature` / `data` / `eval`

---

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
