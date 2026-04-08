# 项目里程碑时间线

重大事件记录。格式：`YYYY-MM-DD | 类别 | 描述`。最新在前。

类别：`model` / `arch` / `fix` / `feature` / `data` / `eval`

---

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
