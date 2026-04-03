# 无泄露诚实模型重建 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将V4.9.0.1模型从"3倍杠杆动量赌注"重建为"真实alpha选股模型"，通过三阶段（因子残差标签 → 特征强化 → 组合优化）逐步改进，每阶段用fast-check门控验证。

**Architecture:** Phase A在WF窗口内用Fama-French 4因子残差化标签，Phase B对特征做rank-transform并替换动量特征为残差动量+新增防御性特征，Phase C利用已有的`sector_diversify`和`replace_threshold`参数配合新增流动性过滤。V5Trainer继承V4901Trainer，复用所有已有infrastructure。

**Tech Stack:** Python3, LightGBM/XGBoost/CatBoost, SQLite, backtest/factor_returns.py (FF4因子), v39_feature_cache_updater.py

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `ml_models/training/train_v395_multi_target.py` | V5Trainer类 + CLI --v5 flag | Modify (~L10282+, ~L14036+) |
| `ml_models/v39/v5_production_scorer.py` | V5生产评分器(流动性惩罚) | Create |
| `fetch_data/v39_feature_cache_updater.py` | 新增6个防御性特征+残差动量 | Modify |
| `backtest/factor_returns.py` | 无改动，复用 | — |
| `backtest/backtest_report_based.py` | 无改动，复用已有sector_diversify/replace_threshold | — |

---

## Task 1: Phase A — V5Trainer类 + 因子残差标签

**Files:**
- Modify: `ml_models/training/train_v395_multi_target.py` (after line 10281, V4901Trainer定义之后)

- [ ] **Step 1: 在V4901Trainer之后添加V5Trainer类**

在 `train_v395_multi_target.py` 的 V4901Trainer类定义结束后（约L10282），V4902Trainer之前，插入：

```python
class V5Trainer(V4901Trainer):
    """V5.0 训练器 — 因子残差标签 + Rank-Transform + 防御性特征
    
    三阶段诚实重建:
    Phase A: 因子残差标签 (消除SMB/HML/UMD因子暴露)
    Phase B: Rank-transform + 残差动量 + 防御性特征
    Phase C: 组合优化 (流动性/行业/换手, 在scorer和回测参数中实现)
    """
    
    # Phase A: 因子残差标签
    USE_RESIDUAL_LABELS = True
    BETA_WINDOW = 120       # 因子beta估计rolling窗口
    MIN_BETA_DAYS = 60      # 不足此天数用截面中位beta
    
    # Phase B: Rank-transform
    USE_RANK_TRANSFORM = True
    
    # Phase B: 删除的纯动量特征
    PRUNE_FEATURES = [
        'market_momentum_20d',
        'market_momentum_5d',
    ]
    
    def _residualize_labels(self, df, train_mask, val_mask, test_mask):
        """Phase A: 在WF窗口内用FF4因子残差化标签
        
        对每个label_Nd:
          1. 用train期rolling 120天估计每只股票的因子beta
          2. val/test用最近已知beta (不重新估计)
          3. 残差标签 = 原始标签 - 因子预期收益
        """
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
        from backtest.factor_returns import load_or_build_factors
        
        # 加载因子收益 (需要额外252天lookback用于UMD)
        min_date = df.loc[train_mask, 'trade_date'].min()
        max_date = df['trade_date'].max()
        factors = load_or_build_factors(
            start_date=min_date, end_date=max_date,
            db_path=self.db_path
        )
        if factors.empty:
            logger.warning("  因子收益为空, 跳过残差化")
            return
        
        # 确保factors index是string格式
        factors.index = factors.index.astype(str).str[:10]
        
        # 加载个股日收益 (用于beta估计)
        conn = sqlite3.connect(self.db_path, timeout=30)
        stock_ret_query = """
            SELECT s.code, dq.trade_date,
                   CAST(dq.price_change_pct AS REAL) AS ret
            FROM daily_quotes dq
            JOIN securities s ON dq.security_id = s.id
            WHERE s.type = 'A股' AND dq.trade_date BETWEEN ? AND ?
              AND dq.volume > 0
            ORDER BY s.code, dq.trade_date
        """
        lookback_date = (pd.Timestamp(min_date) - pd.DateOffset(days=200)).strftime('%Y-%m-%d')
        stock_ret_df = pd.read_sql(stock_ret_query, conn, params=[lookback_date, max_date])
        conn.close()
        stock_ret_df['trade_date'] = stock_ret_df['trade_date'].astype(str).str[:10]
        stock_ret_df['ret'] = pd.to_numeric(stock_ret_df['ret'], errors='coerce')
        stock_ret_df = stock_ret_df.dropna(subset=['ret'])
        
        # 构建个股收益pivot: (date, code) -> ret
        stock_ret_pivot = stock_ret_df.pivot_table(
            index='trade_date', columns='code', values='ret', aggfunc='first'
        )
        
        # 对齐因子和个股收益
        common_dates = stock_ret_pivot.index.intersection(factors.index)
        stock_ret_aligned = stock_ret_pivot.loc[common_dates]
        factors_aligned = factors.loc[common_dates]
        
        # 估计rolling beta (只用train期之前的数据)
        train_end_date = df.loc[train_mask, 'trade_date'].max()
        beta_dates = [d for d in common_dates if d <= train_end_date]
        
        logger.info(f"  因子残差化: {len(beta_dates)}个交易日可用于beta估计")
        
        # 对每只股票估计最新beta
        codes_in_df = df['code'].unique()
        code_betas = {}  # code -> (β_mkt, β_smb, β_hml, β_umd)
        
        factor_cols = ['MKT', 'SMB', 'HML', 'UMD']
        
        for code in codes_in_df:
            if code not in stock_ret_aligned.columns:
                continue
            stock_series = stock_ret_aligned.loc[beta_dates, code].dropna()
            if len(stock_series) < self.MIN_BETA_DAYS:
                continue
            
            # 取最近BETA_WINDOW天
            recent = stock_series.tail(self.BETA_WINDOW)
            factor_recent = factors_aligned.loc[recent.index, factor_cols]
            
            # 去除NaN行
            valid_mask = factor_recent.notna().all(axis=1) & recent.notna()
            if valid_mask.sum() < self.MIN_BETA_DAYS:
                continue
            
            y = recent[valid_mask].values
            X_f = factor_recent[valid_mask].values
            
            # OLS: ret = α + β·factors + ε
            X_f_const = np.column_stack([np.ones(len(y)), X_f])
            try:
                betas_ols = np.linalg.lstsq(X_f_const, y, rcond=None)[0]
                code_betas[code] = betas_ols[1:]  # (β_mkt, β_smb, β_hml, β_umd)
            except np.linalg.LinAlgError:
                continue
        
        logger.info(f"  成功估计 {len(code_betas)}/{len(codes_in_df)} 只股票的因子beta")
        
        # 用截面中位数填充缺失beta
        if code_betas:
            all_betas = np.array(list(code_betas.values()))
            median_beta = np.median(all_betas, axis=0)
        else:
            median_beta = np.zeros(4)
            logger.warning("  无法估计任何股票的beta, 使用零beta")
        
        # 对每个label做残差化
        for label_col in ['label_3d', 'label_5d', 'label_10d', 'label_15d']:
            N = int(label_col.split('_')[1].replace('d', ''))
            
            # 计算N天累计因子收益
            factor_cum = {}
            dates_list = sorted(factors.index.tolist())
            date_to_idx = {d: i for i, d in enumerate(dates_list)}
            
            for date in df['trade_date'].unique():
                if date not in date_to_idx:
                    continue
                idx = date_to_idx[date]
                # 未来N天因子收益之和 (T+1到T+N)
                end_idx = min(idx + N + 1, len(dates_list))
                start_idx = idx + 1
                if start_idx >= len(dates_list):
                    continue
                future_dates = dates_list[start_idx:end_idx]
                if len(future_dates) == 0:
                    continue
                cum_factor = factors.loc[future_dates, factor_cols].sum().values  # (4,)
                factor_cum[date] = cum_factor
            
            # 逐行残差化
            residualized_count = 0
            for i in df.index:
                code = df.at[i, 'code']
                date = df.at[i, 'trade_date']
                
                beta = code_betas.get(code, median_beta)
                cum_f = factor_cum.get(date)
                
                if cum_f is None:
                    continue
                
                expected = np.dot(beta, cum_f)
                df.at[i, label_col] = df.at[i, label_col] - expected
                residualized_count += 1
            
            pct = residualized_count / len(df) * 100
            logger.info(f"  {label_col}: 残差化 {residualized_count:,}/{len(df):,} ({pct:.1f}%)")
    
    def _apply_rank_transform(self, X, dates_arr, feature_names):
        """Phase B: Cross-sectional rank transform [0, 1]"""
        if not self.USE_RANK_TRANSFORM:
            return X
        
        logger.info("  Phase B: Cross-Sectional Rank Transform")
        X_ranked = X.copy()
        unique_dates = np.unique(dates_arr)
        
        for date in unique_dates:
            mask = dates_arr == date
            n = mask.sum()
            if n < 10:
                continue
            for j in range(X_ranked.shape[1]):
                col_vals = X_ranked[mask, j]
                # scipy.stats.rankdata返回1-based rank
                from scipy.stats import rankdata
                ranks = rankdata(col_vals, method='average')
                X_ranked[mask, j] = (ranks - 1) / max(n - 1, 1)  # [0, 1]
        
        return X_ranked
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """V5: 继承V4.7.1的prepare_features, 添加rank-transform"""
        X, y_3d, y_5d, y_10d, y_15d, df = super().prepare_features(df)
        
        if self.USE_RANK_TRANSFORM:
            dates_arr = df['trade_date'].values
            X = self._apply_rank_transform(X, dates_arr, self.feature_names)
            # 更新df中的特征值
            df[self.feature_names] = X
        
        return X, y_3d, y_5d, y_10d, y_15d, df
    
    def walk_forward_train(self, start_date=None, end_date=None,
                            purge_days=15, min_train_days=900,
                            val_days=120, test_days=120, step_days=120):
        """V5 Walk-Forward — 在WF窗口级别插入因子残差化"""
        import shutil
        
        version_tag = 'v5'
        version_str = 'v5.0'
        
        logger.info("=" * 60)
        logger.info(f"{version_str} Walk-Forward (诚实重建: 因子残差 + Rank-Transform)")
        logger.info("=" * 60)
        logger.info(f"  底座: V4.9.0.1 (61特征)")
        logger.info(f"  Phase A: 因子残差标签 (beta窗口={self.BETA_WINDOW}天)")
        logger.info(f"  Phase B: Rank-Transform={self.USE_RANK_TRANSFORM}")
        logger.info(f"  删除特征: {self.PRUNE_FEATURES}")
        
        # 1. 加载数据
        df = self.load_data(start_date, end_date)
        X, y_3d, y_5d, y_10d, y_15d, df = self.prepare_features(df)
        
        dates = df['trade_date'].values
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        logger.info(f"  总交易日: {n_dates}, 样本: {len(X):,}")
        
        # 2. Walk-Forward windows
        windows = []
        cursor = min_train_days
        while cursor + val_days + 2 * purge_days + test_days <= n_dates:
            train_end_idx = cursor - 1
            val_start_idx = cursor + purge_days
            val_end_idx = val_start_idx + val_days - 1
            test_start_idx = val_end_idx + 1 + purge_days
            test_end_idx = test_start_idx + test_days - 1
            if test_end_idx >= n_dates:
                break
            windows.append({
                'train_end': unique_dates[train_end_idx],
                'val_start': unique_dates[val_start_idx],
                'val_end': unique_dates[val_end_idx],
                'test_start': unique_dates[test_start_idx],
                'test_end': unique_dates[test_end_idx],
            })
            cursor += step_days
        
        # fast-check: 只取最后N个窗口
        _max_windows = getattr(self, '_fast_check_max_windows', None)
        if _max_windows and len(windows) > _max_windows:
            logger.info(f"  [FAST-CHECK] 截取最后 {_max_windows}/{len(windows)} 个窗口")
            windows = windows[-_max_windows:]
        
        logger.info(f"  Walk-Forward 窗口数: {len(windows)}")
        for i, w in enumerate(windows):
            logger.info(f"    窗口 {i+1}: train<='{w['train_end']}', val={w['val_start']}~{w['val_end']}, "
                         f"test={w['test_start']}~{w['test_end']}")
        
        # 3. WF evaluation
        wf_metrics = []
        import gc
        for wi, w in enumerate(windows):
            logger.info(f"\n{'='*50}")
            logger.info(f"Walk-Forward 窗口 {wi+1}/{len(windows)}")
            logger.info(f"{'='*50}")
            
            train_mask = dates <= w['train_end']
            val_mask = (dates >= w['val_start']) & (dates <= w['val_end'])
            test_mask = (dates >= w['test_start']) & (dates <= w['test_end'])
            
            # ★ Phase A: 因子残差化标签 (在每个WF窗口内独立执行)
            if self.USE_RESIDUAL_LABELS:
                self._residualize_labels(df, train_mask, val_mask, test_mask)
                # 重新提取残差化后的标签
                y_3d = df['label_3d'].values
                y_5d = df['label_5d'].values
                y_10d = df['label_10d'].values
                y_15d = df['label_15d'].values
            
            X_train_w, X_val_w, X_test_w = X[train_mask].copy(), X[val_mask].copy(), X[test_mask].copy()
            y_3d_tr, y_3d_va, y_3d_te = y_3d[train_mask].copy(), y_3d[val_mask].copy(), y_3d[test_mask].copy()
            y_5d_tr, y_5d_va, y_5d_te = y_5d[train_mask].copy(), y_5d[val_mask].copy(), y_5d[test_mask].copy()
            y_10d_tr, y_10d_va, y_10d_te = y_10d[train_mask].copy(), y_10d[val_mask].copy(), y_10d[test_mask].copy()
            y_15d_tr, y_15d_va, y_15d_te = y_15d[train_mask].copy(), y_15d[val_mask].copy(), y_15d[test_mask].copy()
            test_dates_w = dates[test_mask]
            
            # Winsorization (train-only bounds)
            X_train_w, wf_bounds = self.winsorize_features(X_train_w)
            self._apply_bounds(X_val_w, wf_bounds)
            self._apply_bounds(X_test_w, wf_bounds)
            
            # Label winsorization
            for y_tr_w, y_va_w, y_te_w in [(y_3d_tr, y_3d_va, y_3d_te),
                                             (y_5d_tr, y_5d_va, y_5d_te),
                                             (y_10d_tr, y_10d_va, y_10d_te),
                                             (y_15d_tr, y_15d_va, y_15d_te)]:
                lo = np.percentile(y_tr_w, 1)
                hi = np.percentile(y_tr_w, 99)
                y_tr_w[:] = np.clip(y_tr_w, lo, hi)
                y_va_w[:] = np.clip(y_va_w, lo, hi)
                y_te_w[:] = np.clip(y_te_w, lo, hi)
            
            logger.info(f"  train={X_train_w.shape[0]:,}, val={X_val_w.shape[0]:,}, test={X_test_w.shape[0]:,}")
            
            # 设置train/val dates
            self.train_dates = dates[train_mask]
            self.val_dates = dates[val_mask]
            
            targets_w = [
                ('3d', y_3d_tr, y_3d_va, y_3d_te),
                ('5d', y_5d_tr, y_5d_va, y_5d_te),
                ('10d', y_10d_tr, y_10d_va, y_10d_te),
                ('15d', y_15d_tr, y_15d_va, y_15d_te),
            ]
            
            # turbo-check: 只训练10d
            if getattr(self, '_turbo_targets', None):
                targets_w = [t for t in targets_w if t[0] in self._turbo_targets]
            
            window_models = {}
            window_ic = {}
            
            for target_name, y_tr, y_va, y_te in targets_w:
                train_d = self.train_dates
                val_d = self.val_dates
                test_d = test_dates_w
                
                # Sharpe-blend
                self._apply_sharpe_blend(y_tr, y_va, y_te, train_d, val_d, test_d, target_name)
                
                # 样本权重
                df_train_w = df[train_mask].copy()
                sample_weights = self.compute_sample_weights(df_train_w, y_tr)
                
                # 训练模型
                models = self.train_single_target_models(
                    X_train_w, X_val_w, y_tr, y_va, target_name,
                    sample_weights_train=sample_weights
                )
                window_models[target_name] = models
                
                # OOS IC
                preds = self._ensemble_predict(models, X_test_w)
                from scipy.stats import spearmanr
                daily_ics = []
                for d in np.unique(test_d):
                    d_mask = test_d == d
                    if d_mask.sum() < 20:
                        continue
                    ic, _ = spearmanr(preds[d_mask], y_te[d_mask])
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                
                mean_ic = np.mean(daily_ics) if daily_ics else 0
                ic_std = np.std(daily_ics) if len(daily_ics) > 1 else 1
                icir = mean_ic / ic_std if ic_std > 1e-8 else 0
                window_ic[target_name] = {'ic': mean_ic, 'icir': icir, 'n': len(daily_ics)}
                logger.info(f"  {target_name}: IC={mean_ic:.4f}, ICIR={icir:.3f} ({len(daily_ics)} days)")
            
            wf_metrics.append({'window': wi+1, 'ic': window_ic})
            gc.collect()
        
        # 汇总WF结果
        logger.info(f"\n{'='*60}")
        logger.info(f"V5.0 Walk-Forward 汇总")
        logger.info(f"{'='*60}")
        
        all_targets = set()
        for m in wf_metrics:
            all_targets.update(m['ic'].keys())
        
        summary = {}
        for target in sorted(all_targets):
            ics = [m['ic'][target]['ic'] for m in wf_metrics if target in m['ic']]
            icirs = [m['ic'][target]['icir'] for m in wf_metrics if target in m['ic']]
            mean_ic = np.mean(ics) if ics else 0
            mean_icir = np.mean(icirs) if icirs else 0
            summary[target] = {'mean_ic': mean_ic, 'mean_icir': mean_icir}
            logger.info(f"  {target}: 平均IC={mean_ic:.4f}, 平均ICIR={mean_icir:.3f}")
        
        # fast-check: 不保存模型
        if getattr(self, '_fast_check', False):
            logger.info("\n[FAST-CHECK] 不保存模型, 仅输出IC/ICIR")
            
            # 门控判断
            any_pass = False
            positive_targets = 0
            for target, s in summary.items():
                if s['mean_ic'] > 0.02 and s['mean_icir'] > 0.15:
                    any_pass = True
                if s['mean_ic'] > 0:
                    positive_targets += 1
            
            if any_pass or positive_targets >= 2:
                logger.info("  ✅ FAST-CHECK 通过: 存在因子残差alpha信号")
            else:
                logger.info("  ❌ FAST-CHECK 失败: 因子残差alpha信号不足")
                logger.info("     建议: 进入Phase B重新设计特征")
            
            return {'fast_check': True, 'summary': summary}, {'wf_metrics': wf_metrics}
        
        # 完整训练: 保存模型
        logger.info("\n训练生产模型 (全量数据)...")
        # 用全量数据训练 (复用V485的train_production逻辑)
        if self.USE_RESIDUAL_LABELS:
            full_mask = np.ones(len(df), dtype=bool)
            self._residualize_labels(df, full_mask, np.zeros(len(df), dtype=bool), np.zeros(len(df), dtype=bool))
            y_3d = df['label_3d'].values
            y_5d = df['label_5d'].values
            y_10d = df['label_10d'].values
            y_15d = df['label_15d'].values
        
        model_data = self._train_production_model(X, y_3d, y_5d, y_10d, y_15d, df)
        
        # 保存为v5格式
        out_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / version_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime as _dt
        timestamp = _dt.now().strftime('%Y%m%d_%H%M%S')
        model_path = out_dir / f'{version_tag}_multi_target_{timestamp}.pkl'
        
        import joblib
        model_data['version'] = version_str
        model_data['v5_innovations'] = {
            'phase_a': f'因子残差标签 (beta窗口={self.BETA_WINDOW}天)',
            'phase_b': f'Rank-Transform={self.USE_RANK_TRANSFORM}',
            'pruned_features': self.PRUNE_FEATURES,
        }
        joblib.dump(model_data, model_path)
        logger.info(f"\n{version_str} model saved: {model_path}")
        
        # 保存训练历史
        import json as _json
        history = {
            'version': version_str,
            'wf_metrics': wf_metrics,
            'summary': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summary.items()},
            'timestamp': timestamp,
        }
        history_path = out_dir / f'training_history_{timestamp}.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            _json.dump(history, f, indent=2, ensure_ascii=False)
        latest_path = out_dir / 'training_history_latest.json'
        with open(latest_path, 'w', encoding='utf-8') as f:
            _json.dump(history, f, indent=2, ensure_ascii=False)
        
        return model_data, history
```

- [ ] **Step 2: 添加 --v5 CLI参数**

在 `main()` 的argparse部分（约L14068之后），添加：

```python
    parser.add_argument('--v5', action='store_true',
        help='V5.0: 诚实重建 — 因子残差标签+Rank-Transform+防御性特征')
```

在 `main()` 的trainer选择部分（约L14298，在 `if args.v4901:` 之前），添加：

```python
    if args.v5:
        trainer = V5Trainer()
        _apply_overrides(trainer)
        if args.skip_wf:
            trainer.train_production_only(
                start_date=args.start_date, end_date=args.end_date,
                purge_days=max(args.purge_days, 15))
        else:
            trainer.walk_forward_train(
                start_date=args.start_date, end_date=args.end_date,
                purge_days=max(args.purge_days, 15))
    elif args.v4901:
```

- [ ] **Step 3: 添加必要import**

在文件顶部import区域确认以下import存在（如果还没有）:

```python
import sqlite3
import pandas as pd
```

- [ ] **Step 4: 验证语法**

Run: `python3 -c "import ast; ast.parse(open('ml_models/training/train_v395_multi_target.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 5: Phase A Fast-Check验证**

Run: `python3 ml_models/training/train_v395_multi_target.py --v5 --fast-check --purge-days 15 2>&1 | tail -30`
Expected: IC/ICIR输出 + FAST-CHECK通过/失败判定

- [ ] **Step 6: Commit**

```bash
git add ml_models/training/train_v395_multi_target.py
git commit -m "feat: V5Trainer — 因子残差标签+Rank-Transform (Phase A+B核心)"
```

---

## Task 2: Phase A Fast-Check门控验证

**Files:**
- Run: `ml_models/training/train_v395_multi_target.py`

- [ ] **Step 1: 运行Phase A fast-check (仅因子残差，无rank-transform)**

临时禁用rank-transform，只验证残差标签效果：

Run: `python3 ml_models/training/train_v395_multi_target.py --v5 --fast-check --purge-days 15 2>&1 | tee /tmp/v5_phase_a_fastcheck.log`

- [ ] **Step 2: 分析结果**

检查日志末尾的IC/ICIR：
- ✅ 通过: 任意目标IC>0.02且ICIR>0.15，或≥2个目标IC>0 → 继续Task 3
- ❌ 失败: 全部IC≤0 → 修改策略（可能需要调整beta窗口或因子模型）

- [ ] **Step 3: 如通过，运行Phase A+B合并fast-check**

Run: `python3 ml_models/training/train_v395_multi_target.py --v5 --fast-check --purge-days 15 2>&1 | tee /tmp/v5_phase_ab_fastcheck.log`

（此时USE_RANK_TRANSFORM=True已生效）

- [ ] **Step 4: 对比Phase A vs Phase A+B**

比较两次fast-check的IC/ICIR：
- IC不下降（±0.005）且ICIR改善 → Phase B有效
- IC和ICIR都下降 → 需要调整Phase B（可能禁用rank-transform）

---

## Task 3: Phase C — V5 Production Scorer + 组合优化参数

**Files:**
- Create: `ml_models/v39/v5_production_scorer.py`

- [ ] **Step 1: 创建V5 Production Scorer**

```python
#!/usr/bin/env python3
"""
V5.0 production scorer — 因子残差alpha + 流动性惩罚

继承V4901的composite排序，新增:
  - 流动性惩罚 (ADV不足降权)
  - 回测参数默认优化 (sector_diversify=3, replace_threshold=0.006)
"""

import numpy as np
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List

from .v4901_production_scorer import V4901ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'


class V5ProductionScorer(V4901ProductionScorer):
    """V5.0 scorer — 因子残差alpha + 流动性惩罚"""

    MAX_PARTICIPATION = 0.02  # 单日不超过ADV的2%
    PENALTY_FLOOR = 0.1       # 流动性最差也保留10%得分
    DEFAULT_PORTFOLIO_VALUE = 1_000_000  # 默认100万组合

    def __init__(self, model_type: str = 'small_data', portfolio_value: float = None):
        self._portfolio_value = portfolio_value or self.DEFAULT_PORTFOLIO_VALUE
        super().__init__(model_type=model_type)

    def _load_models(self):
        """加载v5模型, fallback到v4901→v490→v485"""
        v5_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v5'
        v5_files = list(v5_dir.glob('v5_*.pkl')) if v5_dir.exists() else []
        if v5_files:
            self.model_dir = v5_dir
            latest = max(v5_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V5.0')
            return
        logger.warning("  V5模型未找到, fallback到V4901")
        super()._load_models()

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V5: V4901 pipeline + 流动性惩罚"""
        results = super().predict_scores(stock_codes, date)

        # 加载ADV数据
        adv_map = self._load_adv(stock_codes, date)

        target_position = self._portfolio_value / 10  # Top10, 每只10%仓位

        for code, data in results.items():
            adv = adv_map.get(code, 0)
            if adv > 0:
                participation = target_position / adv
                penalty = np.clip(
                    1.0 - participation / self.MAX_PARTICIPATION,
                    self.PENALTY_FLOOR, 1.0
                )
            else:
                penalty = self.PENALTY_FLOOR

            data['liquidity_penalty'] = penalty
            data['rank_score'] = data.get('rank_score', 0) * penalty

        # 重新计算全局百分位score
        all_comp = [(code, data.get('rank_score', 0)) for code, data in results.items()]
        if all_comp:
            sorted_comp = sorted(all_comp, key=lambda x: x[1])
            n = len(sorted_comp)
            for rank_i, (code, _) in enumerate(sorted_comp):
                results[code]['score'] = round(rank_i / max(n - 1, 1) * 100, 1)

        return results

    def _load_adv(self, stock_codes: List[str], date: str) -> Dict[str, float]:
        """加载20日日均成交额 (ADV = volume * close * 100)"""
        try:
            conn = sqlite3.connect(str(DB_PATH), timeout=30)
            # volume单位是手(100股), close是元, 所以ADV = volume * close * 100
            placeholders = ','.join('?' for _ in stock_codes)
            query = f"""
                SELECT s.code,
                       AVG(dq.volume * dq.close * 100) AS adv_20d
                FROM daily_quotes dq
                JOIN securities s ON dq.security_id = s.id
                WHERE s.code IN ({placeholders})
                  AND dq.trade_date <= ?
                  AND dq.trade_date >= date(?, '-30 day')
                  AND dq.volume > 0
                GROUP BY s.code
            """
            params = list(stock_codes) + [date, date]
            import pandas as pd
            df = pd.read_sql(query, conn, params=params)
            conn.close()
            return dict(zip(df['code'], df['adv_20d']))
        except Exception as e:
            logger.warning(f"  加载ADV失败: {e}")
            return {}
```

- [ ] **Step 2: 验证语法**

Run: `python3 -c "from ml_models.v39.v5_production_scorer import V5ProductionScorer; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add ml_models/v39/v5_production_scorer.py
git commit -m "feat: V5ProductionScorer — 流动性惩罚 + ADV加权"
```

---

## Task 4: 完整训练 + 双向评估

**Files:**
- Run: `ml_models/training/train_v395_multi_target.py`
- Run: `backtest/run_north_star_eval.py`

- [ ] **Step 1: 完整训练V5模型**

Run: `python3 ml_models/training/train_v395_multi_target.py --v5 --purge-days 15 2>&1 | tee logs/v5_honest_rebuild_$(date +%Y%m%d_%H%M%S).log`

预计3-5小时。训练完成后自动输出WF汇总。

- [ ] **Step 2: 生成WF-OOS评估报告**

Run:
```bash
python3 backtest/batch_generate_v395_reports.py \
  --version v5 \
  --start-date 2024-01-01 \
  --end-date 2026-03-31 \
  --output-dir reports/daily_selection_v5_wf_oos
```

- [ ] **Step 3: 运行北极星V5.2评估 (WF-OOS)**

Run:
```bash
python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_v5_wf_oos \
  --label "V5.0-WF-OOS" \
  --top-n 10 --focus-days 10 \
  --rank-field composite \
  --sector-diversify 3 \
  --replace-threshold 0.006 \
  --score-version v52
```

- [ ] **Step 4: 生成PRE-2020评估报告（如2018-2020缓存已有）**

Run:
```bash
python3 backtest/batch_generate_v395_reports.py \
  --version v5 \
  --start-date 2018-01-01 \
  --end-date 2019-12-31 \
  --output-dir reports/daily_selection_v5_pre2020

python3 backtest/run_north_star_eval.py --backtest \
  --report-dir reports/daily_selection_v5_pre2020 \
  --label "V5.0-PRE2020" \
  --top-n 10 --focus-days 10 \
  --rank-field composite \
  --sector-diversify 3 \
  --replace-threshold 0.006 \
  --score-version v52
```

- [ ] **Step 5: 分析结果并Commit**

```bash
git add -A
git commit -m "feat: V5.0诚实重建完成 — 因子残差+Rank-Transform+流动性惩罚"
```

---

## 执行顺序

```
Task 1 (编码, ~30min) → Task 2 (fast-check, ~5min) → Task 3 (编码, ~15min) → Task 4 (训练, ~3-5h)
```
