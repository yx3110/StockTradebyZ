#!/usr/bin/env python3
"""
V4.7.3 生产评分器 — 简化管线 + V4.7.1精简特征 + ICIR权重 + 无压缩

继承V4.4: Bear Specialist + Per-target Isotonic + 可执行性过滤 + 市况自适应
新增V4.7.1(精简版): roe + daily_basic扩展 + 微观结构 + 反转 + idio_volatility
保留V4.6: ICIR权重裁剪 + 增强流动性折扣
去除V4.6: Meta-Learner + Combined Isotonic (两层压缩破坏预测区分度)
"""

import numpy as np
import pandas as pd
import sqlite3
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional

from .v44_production_scorer import V44ProductionScorer

import logging
logger = logging.getLogger(__name__)


class V473ProductionScorer(V44ProductionScorer):
    """V4.7.3 生产评分器 — V4.4管线 + V4.7.1精简特征 + ICIR权重 + 无压缩"""

    def __init__(self, model_type: str = 'small_data'):
        self._v473_model_dir = Path(__file__).parent.parent.parent / 'ml_models' / 'trained_models' / 'v473'
        self._financial_cache = {}  # date -> DataFrame
        self._micro_cache = {}  # date -> DataFrame
        super().__init__(model_type=model_type)

    def _load_models(self):
        """覆盖加载方法, 使用 v473 模型目录"""
        self.model_dir = self._v473_model_dir
        if self.model_type == 'rolling':
            self._load_rolling_models()
        else:
            self._load_v473_model()

    def _load_v473_model(self):
        """加载v4.7.3模型 — V4.4结构 + V4.7.1精简特征 + ICIR权重, 无Meta-Learner/Combined Isotonic"""
        model_files = list(self.model_dir.glob('v473_*.pkl'))
        if not model_files:
            print(f"V4.7.3 未找到模型文件: {self.model_dir}/v473_*.pkl")
            return

        latest = max(model_files, key=lambda f: f.stat().st_mtime)
        try:
            model_data = joblib.load(latest)
        except Exception:
            with open(latest, 'rb') as f:
                model_data = pickle.load(f)

        raw_models = model_data.get('models', {})
        self.models = {}
        self.weights = model_data.get('ensemble_weights', {})
        for target, target_data in raw_models.items():
            if isinstance(target_data, dict) and 'models' in target_data:
                self.models[target] = target_data['models']
                if not self.weights:
                    self.weights[f'label_{target}'] = target_data.get('weights', {})
            else:
                self.models[target] = target_data

        self.scaler = model_data.get('scaler')
        self.feature_cols = model_data.get('feature_names', model_data.get('feature_cols', []))
        self.market_feature_cols = model_data.get('market_features', model_data.get('market_feature_cols', []))
        # V4.7.3 optimal: 10d+15d after ablation (511 days, 2024-01~2026-02)
        # 10d+15d: AnnRet +97.3%, Sharpe 1.576 vs default: +83.8%, 1.499
        self.target_weights = {
            'label_3d': 0.00, 'label_5d': 0.00, 'label_10d': 0.60, 'label_15d': 0.40
        }

        # 元数据
        self.cascade = False
        self.cascade_feature_names = None
        self.dual_stream = False
        self.rank_normalized = False
        self.robust_zscore = model_data.get('robust_zscore', True)
        self.extra_features_from_daily_basic = model_data.get('extra_features_from_daily_basic', None)
        self.extra_tech_features = model_data.get('extra_features_from_tech_indicators', None)
        self.stock_rank_cols = model_data.get('stock_feature_cols', None)

        # V4.7.1 精简特征列表 (V4.7.3: roe only + 无downside_deviation)
        self.extra_features_financial = model_data.get('extra_features_financial', [])
        self.extra_features_microstructure = model_data.get('extra_features_microstructure', [])
        self.extra_features_reversal = model_data.get('extra_features_reversal', [])
        self.extra_features_risk = model_data.get('extra_features_risk', [])

        # Winsorize bounds
        raw_bounds = model_data.get('winsorize_bounds')
        if raw_bounds:
            self.winsorize_bounds = {k: tuple(v) for k, v in raw_bounds.items()} if isinstance(raw_bounds, dict) else raw_bounds

        # V4.4 组件 (继承)
        self.bear_models = model_data.get('bear_models', {})
        # V4.7.3: 使用isotonic校准用于评分/推荐, 原始值通过raw_pred_Xd保留给报告展示
        self.isotonic_calibration = model_data.get('isotonic_calibration', {})

        # V4.7.3: 无Meta-Learner, 无Combined Isotonic (核心设计)
        # 不加载这些组件, 即使模型文件中存在也忽略

        # 全局分位数
        raw_quantiles = model_data.get('global_quantiles')
        if raw_quantiles is not None:
            self.global_quantiles = np.array(raw_quantiles)
        else:
            quantiles_path = self.model_dir / 'global_quantiles.npy'
            if quantiles_path.exists():
                self.global_quantiles = np.load(quantiles_path)

        # 投资建议阈值
        self.recommendation_thresholds = model_data.get('recommendation_thresholds')
        if not self.recommendation_thresholds:
            rec_path = self.model_dir / 'recommendation_thresholds.json'
            if rec_path.exists():
                import json as _json
                with open(rec_path, 'r') as f:
                    self.recommendation_thresholds = _json.load(f)

        # ICIR权重: clip到[0.08, 0.50]后重归一化 (从V4.6继承)
        self.weights = self._clip_icir_weights(self.weights)

        wf = model_data.get('walk_forward_metrics', {})
        gq_status = "全局评分" if self.global_quantiles is not None else "截面评分"
        print(f"V4.7.3 模型加载完成: {list(self.models.keys())} [简化管线+精简特征+{gq_status}]")
        print(f"  模型文件: {latest.name}")
        print(f"  特征数: {len(self.feature_cols)}")
        print(f"  精简特征: 财务{len(self.extra_features_financial)}+微观{len(self.extra_features_microstructure)}"
              f"+反转{len(self.extra_features_reversal)}+风险{len(self.extra_features_risk)}")
        print(f"  熊市专家: {list(self.bear_models.keys()) if self.bear_models else '无'}")
        print(f"  保序校准: {list(self.isotonic_calibration.keys()) if self.isotonic_calibration else '无'}")
        print(f"  Meta-Learner: 无 (V4.7.3设计: 去除压缩)")
        print(f"  Combined Isotonic: 无 (V4.7.3设计: 去除压缩)")
        if wf:
            for t, m in wf.items():
                print(f"  WF {t}: ICIR={m.get('mean_icir', 0):.4f}±{m.get('std_icir', 0):.4f}")

    # ========== ICIR权重裁剪 (从V4.6继承) ==========

    def _clip_icir_weights(self, weights: Dict) -> Dict:
        """V4.6.1: clip ICIR权重到[0.08, 0.50]后重归一化, 保持集成多样性"""
        clipped = {}
        for target_key, model_weights in weights.items():
            if not isinstance(model_weights, dict):
                clipped[target_key] = model_weights
                continue
            new_w = {}
            for name, w in model_weights.items():
                new_w[name] = np.clip(w, 0.08, 0.50)
            total = sum(new_w.values())
            if total > 0:
                new_w = {k: v / total for k, v in new_w.items()}
            clipped[target_key] = new_w
        return clipped

    # ========== Bug 3修复: 市场指数统一为000300.SH ==========

    def _get_market_return_20d(self, date: str) -> Optional[float]:
        """Bug 3修复: 统一使用000300.SH (沪深300), 与训练侧一致"""
        if date in self._market_return_cache:
            return self._market_return_cache[date]

        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT q.close
            FROM daily_quotes q
            JOIN securities s ON q.security_id = s.id
            WHERE s.code = '000300.SH' AND q.trade_date <= ?
            ORDER BY q.trade_date DESC
            LIMIT 21
            """
            df = pd.read_sql_query(query, conn, params=[date])
        finally:
            conn.close()

        if len(df) < 21:
            self._market_return_cache[date] = None
            return None

        ret = (df['close'].iloc[0] / df['close'].iloc[20]) - 1
        self._market_return_cache[date] = float(ret)
        return float(ret)

    # ========== V4.7.1精简特征加载方法 ==========

    def _load_financial_features(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """加载财务质量因子 — V4.7.3仅加载roe (其余5个96%+缺失)"""
        if not self.extra_features_financial:
            return features_df

        if date in self._financial_cache:
            df_fi = self._financial_cache[date]
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                # V4.7.3: 只查询roe (不查gross_margin/current_ratio/assets_turn/netprofit_yoy/or_yoy)
                select_cols = ', '.join([f'fi.{col}' for col in self.extra_features_financial])
                query = f"""
                SELECT s.code, {select_cols}
                FROM financial_indicator fi
                JOIN securities s ON fi.security_id = s.id
                WHERE fi.ann_date <= ? AND fi.ann_date IS NOT NULL AND fi.ann_date != ''
                AND fi.id IN (
                    SELECT MAX(fi2.id) FROM financial_indicator fi2
                    WHERE fi2.security_id = fi.security_id AND fi2.ann_date <= ?
                    AND fi2.ann_date IS NOT NULL AND fi2.ann_date != ''
                )
                """
                df_fi = pd.read_sql_query(query, conn, params=[date, date])
            finally:
                conn.close()
            self._financial_cache[date] = df_fi

        if len(df_fi) > 0:
            features_df = features_df.merge(df_fi, on='code', how='left')
        else:
            for col in self.extra_features_financial:
                if col not in features_df.columns:
                    features_df[col] = 0.0

        for col in self.extra_features_financial:
            if col in features_df.columns:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)

        return features_df

    def _load_daily_basic_extra(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """加载daily_basic扩展特征 (dv_ttm, turnover_rate_f, float_ratio)"""
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT s.code, db.dv_ttm, db.turnover_rate_f, db.circ_mv, db.total_mv
            FROM daily_basic db
            JOIN securities s ON db.security_id = s.id
            WHERE db.trade_date = ?
            """
            df_extra = pd.read_sql_query(query, conn, params=[date])
        finally:
            conn.close()

        if len(df_extra) > 0:
            df_extra['float_ratio'] = df_extra['circ_mv'] / df_extra['total_mv'].clip(lower=1e-8)
            df_extra.drop(columns=['circ_mv', 'total_mv'], inplace=True)
            features_df = features_df.merge(df_extra, on='code', how='left')

        for col in ['dv_ttm', 'turnover_rate_f', 'float_ratio']:
            if col in features_df.columns:
                median_val = features_df[col].median()
                features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            else:
                features_df[col] = 0.0

        return features_df

    def _compute_microstructure_features(self, features_df: pd.DataFrame, date: str) -> pd.DataFrame:
        """计算微观结构/反转/风险因子 — V4.7.3: 不计算downside_deviation_20d"""
        all_new_cols = (self.extra_features_microstructure +
                        self.extra_features_reversal +
                        self.extra_features_risk)
        if not all_new_cols:
            return features_df

        if date in self._micro_cache:
            df_micro = self._micro_cache[date]
        else:
            conn = sqlite3.connect(self.db_path)
            try:
                query = """
                SELECT s.code, q.trade_date, q.close, q.volume, q.price_change_pct
                FROM daily_quotes q
                JOIN securities s ON q.security_id = s.id
                WHERE s.type = 'A股' AND q.trade_date <= ?
                AND q.trade_date >= date(?, '-40 days')
                ORDER BY s.code, q.trade_date
                """
                df_ohlcv = pd.read_sql_query(query, conn, params=[date, date])
            finally:
                conn.close()

            if len(df_ohlcv) == 0:
                for col in all_new_cols:
                    features_df[col] = 0.0
                return features_df

            results_list = []
            for code, grp in df_ohlcv.groupby('code'):
                grp = grp.sort_values('trade_date')
                if len(grp) < 5:
                    continue

                close = grp['close'].values.astype(float)
                volume = grp['volume'].values.astype(float)
                pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)

                row = {'code': code}

                # 微观结构
                if self.extra_features_microstructure:
                    abs_ret = np.abs(pct[-20:]) if len(pct) >= 20 else np.abs(pct)
                    vol_safe = np.where(volume[-20:] > 0, volume[-20:], 1e-8) if len(volume) >= 20 else np.where(volume > 0, volume, 1e-8)
                    row['amihud_illiquidity'] = float(np.mean(abs_ret / vol_safe))

                    n = min(10, len(close))
                    if n >= 5:
                        corr = np.corrcoef(close[-n:], volume[-n:])[0, 1]
                        row['volume_price_corr_10d'] = float(corr) if not np.isnan(corr) else 0.0
                    else:
                        row['volume_price_corr_10d'] = 0.0

                    n_dd = min(20, len(close))
                    window = close[-n_dd:]
                    running_max = np.maximum.accumulate(window)
                    dd = (window - running_max) / np.where(running_max > 0, running_max, 1e-8)
                    row['max_drawdown_20d'] = float(np.min(dd))

                    n_ud = min(10, len(pct))
                    up_vol = np.sum(volume[-n_ud:][pct[-n_ud:] > 0])
                    dn_vol = np.sum(volume[-n_ud:][pct[-n_ud:] < 0])
                    row['updown_volume_asymmetry'] = float(up_vol / max(dn_vol, 1e-8))

                # 反转因子
                if self.extra_features_reversal:
                    row['return_1d'] = float(close[-1] / close[-2] - 1) if len(close) >= 2 else 0.0
                    row['return_3d'] = float(close[-1] / close[-4] - 1) if len(close) >= 4 else 0.0

                # 风险因子 (V4.7.3: 仅idio_volatility_20d, 不计算downside_deviation_20d)
                if self.extra_features_risk:
                    n_risk = min(20, len(close))
                    daily_ret = np.diff(close[-n_risk:]) / close[-n_risk:-1]
                    if len(daily_ret) >= 5:
                        demeaned = daily_ret - np.mean(daily_ret)
                        row['idio_volatility_20d'] = float(np.std(demeaned))
                    else:
                        row['idio_volatility_20d'] = 0.0

                results_list.append(row)

            df_micro = pd.DataFrame(results_list)
            self._micro_cache[date] = df_micro

        if len(df_micro) > 0:
            features_df = features_df.merge(df_micro, on='code', how='left')

        for col in all_new_cols:
            if col in features_df.columns:
                features_df[col] = features_df[col].fillna(0.0)
            else:
                features_df[col] = 0.0

        return features_df

    # ========== 覆写: 不应用小盘加成 ==========

    def _apply_small_cap_bonus(self, results, date, codes):
        """V4.7.3: 不应用小盘加成"""
        return results

    # ========== V4.6增强可执行性过滤 (连续流动性折扣) ==========

    def _apply_enhanced_executability_filters(self, results, date):
        """V4.6.1风格增强可执行性过滤 — 连续流动性折扣(1.5%阈值)"""
        exec_df = self._load_executability_data(date)
        if exec_df is None or len(exec_df) == 0:
            return results

        # T+1日期
        next_date = self._get_next_trade_date(date)

        for _, row in exec_df.iterrows():
            code = row.get('code', '')
            if code not in results:
                continue

            pct_t = row.get('pct_t', 0) or 0
            pct_t1 = row.get('pct_t1', 0) or 0
            turnover = row.get('turnover_rate', 0) or 0

            # 涨停不可买 (T+1)
            board_type = self._get_board_type(code)
            limit_pct = 0.095 if board_type == 'main' else (0.195 if board_type in ('gem', 'star') else 0.295)

            if pct_t1 >= limit_pct:
                results[code]['score'] = 0
                results[code]['exec_filter'] = 'T+1_limit_up'
                continue

            # T日涨停
            if pct_t >= limit_pct:
                results[code]['score'] = 0
                results[code]['exec_filter'] = 'T_limit_up'
                continue

            # T+1日大涨 (>5%)
            if pct_t1 > 0.05:
                results[code]['score'] *= 0.2
                results[code]['exec_filter'] = 'T+1_high_gain'
                continue

            # T日大涨
            if pct_t > 0.05:
                results[code]['score'] *= 0.3
                results[code]['exec_filter'] = 'T_high_gain'
            elif pct_t > 0.03:
                results[code]['score'] *= 0.7
                results[code]['exec_filter'] = 'T_medium_gain'

            # 连续流动性折扣 (V4.6.1风格: 1.5%阈值)
            if turnover > 0 and turnover < 1.5:
                discount = max(0.2, turnover / 1.5)
                results[code]['score'] *= discount
                if 'exec_filter' not in results[code]:
                    results[code]['exec_filter'] = f'low_liquidity_{turnover:.1f}%'

        return results

    def _load_executability_data(self, date: str) -> Optional[pd.DataFrame]:
        """加载可执行性数据 (T日和T+1日涨跌幅 + 换手率)"""
        cache_key = date
        if cache_key in self._exec_cache:
            return self._exec_cache[cache_key]

        conn = sqlite3.connect(self.db_path)
        try:
            next_date = self._get_next_trade_date(date)
            if next_date is None:
                next_date = date

            query = """
            SELECT s.code,
                   q1.price_change_pct as pct_t,
                   q2.price_change_pct as pct_t1,
                   db.turnover_rate
            FROM securities s
            LEFT JOIN daily_quotes q1 ON q1.security_id = s.id AND q1.trade_date = ?
            LEFT JOIN daily_quotes q2 ON q2.security_id = s.id AND q2.trade_date = ?
            LEFT JOIN daily_basic db ON db.security_id = s.id AND db.trade_date = ?
            WHERE s.type = 'A股'
            """
            df = pd.read_sql_query(query, conn, params=[date, next_date, date])
        finally:
            conn.close()

        self._exec_cache[cache_key] = df
        return df

    def _get_next_trade_date(self, date: str) -> Optional[str]:
        """获取下一个交易日"""
        if date in self._next_trade_date_cache:
            return self._next_trade_date_cache[date]

        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT DISTINCT trade_date FROM daily_quotes
            WHERE trade_date > ? ORDER BY trade_date LIMIT 1
            """
            result = pd.read_sql_query(query, conn, params=[date])
        finally:
            conn.close()

        next_date = result['trade_date'].iloc[0] if len(result) > 0 else None
        self._next_trade_date_cache[date] = next_date
        return next_date

    def _get_board_type(self, code: str) -> str:
        """判断股票板块类型 (用于涨停阈值判断)"""
        if code.startswith('30'):
            return 'gem'  # 创业板 20%
        elif code.startswith('68'):
            return 'star'  # 科创板 20%
        elif code.startswith(('8', '4')):
            return 'bse'  # 北交所 30%
        else:
            return 'main'  # 主板 10%

    # ========== 覆写: predict_scores — 7步简化管线 ==========

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.7.3 评分管线 — 7步简化版 (无Meta-Learner, 无Combined Isotonic):
        Step 1: V4.3 基础特征 (robust z-score + daily_basic + tech)
        Step 1.5: V4.7.1 精简特征 (roe + daily_basic_extra + 微观+反转+idio_vol)
        Step 2: 5/6模型×4目标 base predictions
        Step 3: ICIR加权融合 (直接用per-target weights, 无Meta-Learner)
        Step 4: 全局百分位评分 (直接_to_global_score, 无Combined Isotonic)
        Step 5: 熊市专家混合 + Per-target Isotonic + 重算分数
        Step 6: 增强可执行性过滤 + 连续流动性折扣
        """
        # 日期格式标准化
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        results = {}

        # Step 1: V4.3 基础特征
        features_df = self._get_features(stock_codes, date, load_full_cross_section=True)
        if features_df is not None and len(features_df) > 0:
            features_df = self._robust_zscore_normalize_features(features_df)
            features_df = self._load_daily_basic_features(features_df, date)
            features_df = self._load_technical_features(features_df, date)

            # Step 1.5: V4.7.1 精简特征
            features_df = self._load_financial_features(features_df, date)
            features_df = self._load_daily_basic_extra(features_df, date)
            features_df = self._compute_microstructure_features(features_df, date)

            features_df = features_df[features_df['code'].isin(stock_codes)].copy()

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # 准备特征矩阵
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in features_df.columns]
            if missing:
                if len(missing) > len(self.feature_cols) * 0.3:
                    logger.warning(f"⚠️ V4.7.3: {len(missing)}/{len(self.feature_cols)} 特征缺失: {missing[:5]}...")
                for col in missing:
                    features_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in features_df.columns if c not in exclude_cols]

        X = features_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = features_df['code'].tolist()

        # Step 2: 5/6模型×4目标 base predictions
        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue
            target_pred = np.zeros(len(X))
            total_weight = 0
            success_count = 0

            # 收集所有子模型预测
            preds = {}
            for name, model in self.models[target].items():
                try:
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        preds[name] = model.predict(xgb_lib.DMatrix(X))
                    else:
                        preds[name] = model.predict(X)
                except Exception:
                    continue

            # Rescale rank模型(lgb_rank/lgb_listnet)到回归模型尺度
            regression_names = [n for n in preds if n not in ('lgb_rank', 'lgb_listnet')]
            rank_names = [n for n in preds if n in ('lgb_rank', 'lgb_listnet')]
            if regression_names and rank_names:
                reg_means = [np.mean(preds[n]) for n in regression_names]
                reg_stds = [max(np.std(preds[n]), 1e-8) for n in regression_names]
                t_mean, t_std = np.mean(reg_means), np.mean(reg_stds)
                for rn in rank_names:
                    rp = preds[rn]
                    rp_std = max(np.std(rp), 1e-8)
                    preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

            target_w = self.weights.get(f'label_{target}', {})
            success_count = len(preds)
            for name, pred in preds.items():
                weight = target_w.get(name, 0.2)
                target_pred += weight * pred
                total_weight += weight

            if total_weight > 0:
                target_pred /= total_weight
                predictions[target] = target_pred
                if success_count > 0:
                    model_predictions_success = True

        # Step 3: ICIR加权融合 (无Meta-Learner — V4.7.3核心变更)
        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            combined_pred = (
                regime_weights.get('label_3d', 0.20) * predictions['3d'] +
                regime_weights.get('label_5d', 0.25) * predictions['5d'] +
                regime_weights.get('label_10d', 0.35) * predictions['10d'] +
                regime_weights.get('label_15d', 0.20) * predictions['15d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(features_df, available_cols)
            predictions = self._estimate_predictions_from_features(features_df, available_cols)

        # Step 4: 全局百分位评分 (无Combined Isotonic — V4.7.3核心变更)
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Step 5: Bear specialist + Per-target Isotonic + 重算分数
        results = self._blend_bear_specialist(results, date, X, codes)
        results = self._apply_isotonic_calibration(results, codes)

        # 校准后重新计算综合分数 (全局百分位, 无Combined Isotonic)
        if model_predictions_success and self.isotonic_calibration:
            new_combined = np.zeros(len(codes))
            for i, code in enumerate(codes):
                if code in results:
                    r = results[code]
                    new_combined[i] = (
                        regime_weights.get('label_3d', 0.20) * r.get('pred_3d', 0) +
                        regime_weights.get('label_5d', 0.25) * r.get('pred_5d', 0) +
                        regime_weights.get('label_10d', 0.35) * r.get('pred_10d', 0) +
                        regime_weights.get('label_15d', 0.20) * r.get('pred_15d', 0)
                    )
            new_scores = self._to_global_score(new_combined)
            if len(new_scores) > 0:
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        # Step 6: 增强可执行性过滤 + 连续流动性折扣
        results = self._apply_enhanced_executability_filters(results, date)

        # 补全缺失code
        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        # Module F: 附加市况信息 (不影响评分)
        regime_info = self._get_regime_info(date)
        for code in results:
            results[code]['regime_info'] = regime_info

        return results

    def predict_scores_from_preloaded(self, stock_codes: List[str], date: str,
                                       features_df: Optional[pd.DataFrame]) -> Dict[str, Dict]:
        """使用预加载特征评分 — V4.7.3版本 (批量评分用), 简化管线"""
        results = {}

        if features_df is None or len(features_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # robust z-score + daily_basic + tech features
        features_df = self._robust_zscore_normalize_features(features_df.copy())
        features_df = self._load_daily_basic_features(features_df, date)
        features_df = self._load_technical_features(features_df, date)

        # V4.7.1 精简特征
        features_df = self._load_financial_features(features_df, date)
        features_df = self._load_daily_basic_extra(features_df, date)
        features_df = self._compute_microstructure_features(features_df, date)

        mask = features_df['code'].isin(stock_codes)
        filtered_df = features_df[mask].copy()

        if len(filtered_df) == 0:
            for code in stock_codes:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}
            return results

        # 准备特征矩阵
        exclude_cols = {'code', 'trade_date'}
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in filtered_df.columns:
                    filtered_df[col] = 0
            available_cols = self.feature_cols
        else:
            available_cols = [c for c in filtered_df.columns if c not in exclude_cols]

        X = filtered_df[available_cols].fillna(0).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        codes = filtered_df['code'].tolist()

        model_predictions_success = False
        predictions = {
            '3d': np.zeros(len(X)), '5d': np.zeros(len(X)),
            '10d': np.zeros(len(X)), '15d': np.zeros(len(X))
        }

        for target in ['3d', '5d', '10d', '15d']:
            if target not in self.models or not self.models[target]:
                continue
            target_pred = np.zeros(len(X))
            total_weight = 0
            success_count = 0

            # 收集所有子模型预测
            preds = {}
            for name, model in self.models[target].items():
                try:
                    if name == 'xgb':
                        import xgboost as xgb_lib
                        preds[name] = model.predict(xgb_lib.DMatrix(X))
                    else:
                        preds[name] = model.predict(X)
                except Exception:
                    continue

            # Rescale rank模型(lgb_rank/lgb_listnet)到回归模型尺度
            regression_names = [n for n in preds if n not in ('lgb_rank', 'lgb_listnet')]
            rank_names = [n for n in preds if n in ('lgb_rank', 'lgb_listnet')]
            if regression_names and rank_names:
                reg_means = [np.mean(preds[n]) for n in regression_names]
                reg_stds = [max(np.std(preds[n]), 1e-8) for n in regression_names]
                t_mean, t_std = np.mean(reg_means), np.mean(reg_stds)
                for rn in rank_names:
                    rp = preds[rn]
                    rp_std = max(np.std(rp), 1e-8)
                    preds[rn] = (rp - np.mean(rp)) / rp_std * t_std + t_mean

            target_w = self.weights.get(f'label_{target}', {})
            success_count = len(preds)
            for name, pred in preds.items():
                weight = target_w.get(name, 0.2)
                target_pred += weight * pred
                total_weight += weight

            if total_weight > 0:
                target_pred /= total_weight
                predictions[target] = target_pred
                if success_count > 0:
                    model_predictions_success = True

        # ICIR加权融合 (无Meta-Learner)
        regime_weights = self._get_regime_target_weights(date)
        if model_predictions_success:
            combined_pred = (
                regime_weights.get('label_3d', 0.20) * predictions['3d'] +
                regime_weights.get('label_5d', 0.25) * predictions['5d'] +
                regime_weights.get('label_10d', 0.35) * predictions['10d'] +
                regime_weights.get('label_15d', 0.20) * predictions['15d']
            )
        else:
            combined_pred = self._calculate_fallback_scores(filtered_df, available_cols)
            predictions = self._estimate_predictions_from_features(filtered_df, available_cols)

        # 全局百分位 (无Combined Isotonic)
        scores = self._to_global_score(combined_pred)

        for i, code in enumerate(codes):
            results[code] = {
                'score': float(scores[i]),
                'pred_3d': float(predictions['3d'][i]) if i < len(predictions['3d']) else 0,
                'pred_5d': float(predictions['5d'][i]) if i < len(predictions['5d']) else 0,
                'pred_10d': float(predictions['10d'][i]) if i < len(predictions['10d']) else 0,
                'pred_15d': float(predictions.get('15d', np.zeros(1))[min(i, len(predictions.get('15d', [0]))-1)]) if '15d' in predictions else 0,
            }

        # Bear specialist + Isotonic + re-score
        results = self._blend_bear_specialist(results, date, X, codes)
        results = self._apply_isotonic_calibration(results, codes)

        if model_predictions_success and self.isotonic_calibration:
            new_combined = np.zeros(len(codes))
            for i, code in enumerate(codes):
                if code in results:
                    r = results[code]
                    new_combined[i] = (
                        regime_weights.get('label_3d', 0.20) * r.get('pred_3d', 0) +
                        regime_weights.get('label_5d', 0.25) * r.get('pred_5d', 0) +
                        regime_weights.get('label_10d', 0.35) * r.get('pred_10d', 0) +
                        regime_weights.get('label_15d', 0.20) * r.get('pred_15d', 0)
                    )
            new_scores = self._to_global_score(new_combined)
            if len(new_scores) > 0:
                for i, code in enumerate(codes):
                    if code in results:
                        results[code]['score'] = float(new_scores[i])

        # 增强可执行性过滤
        results = self._apply_enhanced_executability_filters(results, date)

        # 补全缺失code
        for code in stock_codes:
            if code not in results:
                results[code] = {'score': 50.0, 'pred_3d': 0, 'pred_5d': 0, 'pred_10d': 0, 'pred_15d': 0,
                                 'exec_filter': 'no_data'}

        return results

    # ========== 10d+15d recommendation overrides ==========

    def _recommendation_from_composite(self, pred_3d: float, pred_5d: float,
                                        pred_10d: float, pred_15d: float = 0.0) -> str:
        """投资建议 -- 基于 0.6*10d + 0.4*15d composite"""
        composite = 0.6 * pred_10d + 0.4 * pred_15d

        t = self.recommendation_thresholds
        if t:
            if composite >= t['strong_buy']:
                return '强烈买入'
            elif composite >= t['buy']:
                return '买入'
            elif composite >= t['cautious']:
                return '谨慎买入'
            elif composite >= t['hold']:
                return '观望'
            else:
                return '回避'

        if composite >= 0.008:
            return '强烈买入'
        elif composite >= 0.005:
            return '买入'
        elif composite >= 0.002:
            return '谨慎买入'
        elif composite >= -0.001:
            return '观望'
        return '回避'

    def _risk_level_from_composite(self, pred_3d: float, pred_5d: float,
                                    pred_10d: float, pred_15d: float = 0.0) -> str:
        """风险等级 -- 基于 0.6*10d + 0.4*15d composite"""
        composite = 0.6 * pred_10d + 0.4 * pred_15d

        t = self.recommendation_thresholds
        if t:
            if composite >= t['buy']:
                return 'low'
            elif composite >= t['hold']:
                return 'medium'
            else:
                return 'high'

        if composite >= 0.005:
            return 'low'
        elif composite >= -0.001:
            return 'medium'
        return 'high'
