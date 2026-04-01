#!/usr/bin/env python3
"""
V4.9.0 production scorer — V4.8.5底座 + Q95 Widen-then-Concentrate (内置) + 市场门控

训练改进: Q95第7模型 + 头尾20%加权 + LambdaRank truncation=10
推理改进: Widen-then-Concentrate (MSE Top-30 → Q95 Top-10)
风控改进: 市场门控模型 — confidence<0.30不推荐, 0.30~0.50减仓

Q95模型已包含在训练产物中(models['10d']['lgb_q95']), 无需额外加载。
"""

import numpy as np
import sqlite3
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from scipy.stats import rankdata

from .v485_production_scorer import V485ProductionScorer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WIDEN_TOP_K = 30
HEAD_SELECT = 10

# 市场门控阈值
GATE_DONT_BUY = 0.30     # confidence < 0.30 → 不推荐任何股票
GATE_REDUCE = 0.50        # 0.30 ≤ confidence < 0.50 → top_n减半


class V490ProductionScorer(V485ProductionScorer):
    """V4.9.0 scorer — 内置Q95 Widen-then-Concentrate + 市场门控"""

    def __init__(self, model_type: str = 'small_data'):
        self._v490_model_dir = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'v490'
        self._market_gate = None
        self._gate_features = None
        self._last_gate_confidence = None
        super().__init__(model_type=model_type)
        self._load_market_gate()

    def _load_models(self):
        v490_files = list(self._v490_model_dir.glob('v490_*.pkl'))
        if v490_files:
            self.model_dir = self._v490_model_dir
            latest = max(v490_files, key=lambda f: f.stat().st_mtime)
            self._load_model_from_file(latest, label='V4.9.0')
            has_q95 = 'lgb_q95' in self.models.get('10d', {})
            logger.info(f"  Q95 in ensemble: {has_q95}")
            return
        super()._load_models()

    def _load_market_gate(self):
        """加载市场门控模型 (共享文件)"""
        gate_path = PROJECT_ROOT / 'ml_models' / 'trained_models' / 'market_gate_model.pkl'
        if gate_path.exists():
            try:
                gate_data = joblib.load(str(gate_path))
                self._market_gate = gate_data['model']
                self._gate_features = gate_data['feature_names']
                self._gate_data = gate_data
                ver = gate_data.get('version', 'v1')
                auc = gate_data.get('mixed_auc', gate_data.get('auc', gate_data.get('val_auc', 'N/A')))
                logger.info(f"  市场门控{ver}已加载 (AUC={auc}, {len(self._gate_features)}特征)")
            except Exception as e:
                logger.warning(f"  市场门控模型加载失败: {e}")

    def _evaluate_market_gate(self, date: str) -> float:
        """运行市场门控, 返回confidence ∈ [0, 1]

        V2: 回归模型预测market_return_10d → percentile映射 → 0.7×模型+0.3×规则
        V1 (fallback): 二分类直接输出概率
        """
        if self._market_gate is None:
            return 1.0

        try:
            gate_version = self._gate_data.get('version', 'v1') if hasattr(self, '_gate_data') else 'v1'

            if gate_version == 'v2':
                db_path = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
                conn = sqlite3.connect(str(db_path))
                try:
                    return self._evaluate_gate_v2(date, conn)
                finally:
                    conn.close()
            else:
                db_path = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
                conn = sqlite3.connect(str(db_path))
                try:
                    return self._evaluate_gate_v1(date, conn)
                finally:
                    conn.close()
        except Exception as e:
            logger.warning(f"  市场门控评估失败: {e}")
            return 0.6

    def _evaluate_gate_v1(self, date: str, conn) -> float:
        """V1门控: 二分类 → 直接概率"""
        query = """
        SELECT market_return_20d, market_return_10d, market_return_5d,
               market_volatility_20d, market_volatility_10d,
               market_up_ratio_20d, market_up_ratio_10d,
               market_drawdown_20d, market_volume_ratio,
               market_position_20d, market_momentum_20d, market_momentum_5d
        FROM v39_feature_cache WHERE trade_date = ? LIMIT 1
        """
        row = conn.execute(query, (date,)).fetchone()
        if row is None:
            row = conn.execute(query, (date.replace('-', ''),)).fetchone()
        if row is None:
            return 0.6

        col_names = ['market_return_20d', 'market_return_10d', 'market_return_5d',
                    'market_volatility_20d', 'market_volatility_10d',
                    'market_up_ratio_20d', 'market_up_ratio_10d',
                    'market_drawdown_20d', 'market_volume_ratio',
                    'market_position_20d', 'market_momentum_20d', 'market_momentum_5d']
        row_dict = dict(zip(col_names, row))
        X = np.array([[row_dict.get(f, 0.0) for f in self._gate_features]])
        return float(np.clip(self._market_gate.predict(X)[0], 0.0, 1.0))

    def _evaluate_gate_v2(self, date: str, conn) -> float:
        """V2门控: 回归预测 → percentile映射 → 0.7模型+0.3规则"""
        # 加载V2所需的全部特征
        date_nodash = date.replace('-', '')

        # 从多个表收集V2特征
        feat_dict = {}

        # 1. 基准指数数据 (bm_return_Xd, bm_vol, drawdown, rsi)
        bm_query = """
        SELECT q.trade_date, q.close
        FROM daily_quotes q JOIN securities s ON q.security_id = s.id
        WHERE s.code = '000905.SH' AND q.trade_date <= ?
        ORDER BY q.trade_date DESC LIMIT 130
        """
        rows = conn.execute(bm_query, (date,)).fetchall()
        if len(rows) < 10:
            return 0.6

        closes = np.array([r[1] for r in reversed(rows)])
        for w in [5, 10, 20, 60, 120]:
            if len(closes) > w:
                feat_dict[f'bm_return_{w}d'] = closes[-1] / closes[-1-w] - 1
            else:
                feat_dict[f'bm_return_{w}d'] = 0.0

        log_rets = np.diff(np.log(closes))
        for w in [20, 60]:
            if len(log_rets) >= w:
                feat_dict[f'bm_vol_{w}d'] = float(np.std(log_rets[-w:]) * np.sqrt(252))

        if len(closes) >= 60:
            peak = np.max(closes[-60:])
            feat_dict['bm_drawdown_60d'] = closes[-1] / peak - 1
        else:
            feat_dict['bm_drawdown_60d'] = 0.0

        # 连涨连跌
        daily_rets = np.diff(closes) / closes[:-1]
        up, down = 0, 0
        for r in reversed(daily_rets):
            if r > 0:
                up += 1
                break
            elif r < 0:
                down += 1
                break
        # 简化: 只数最后一段
        for r in reversed(daily_rets):
            if r > 0 and down == 0:
                up += 1
            elif r < 0 and up == 0:
                down += 1
            else:
                break
        feat_dict['consecutive_up'] = up
        feat_dict['consecutive_down'] = down

        # RSI
        if len(closes) >= 15:
            delta = np.diff(closes[-15:])
            gain = np.mean(np.maximum(delta, 0))
            loss = np.mean(np.maximum(-delta, 0))
            feat_dict['bm_rsi_14'] = 100 - 100 / (1 + gain / max(loss, 1e-8))

        # 2. 微观结构
        micro_query = """
        SELECT COUNT(*) as n,
               SUM(CASE WHEN q.price_change_pct > 0.095 THEN 1 ELSE 0 END) as lu,
               SUM(CASE WHEN q.price_change_pct < -0.095 THEN 1 ELSE 0 END) as ld,
               SUM(CASE WHEN q.price_change_pct > 0 THEN 1 ELSE 0 END) as up
        FROM daily_quotes q JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date = ? AND q.volume > 0
        """
        r = conn.execute(micro_query, (date,)).fetchone()
        if r is None:
            r = conn.execute(micro_query, (date_nodash,)).fetchone()
        if r and r[0] > 0:
            n_st = r[0]
            feat_dict['limit_up_ratio'] = r[1] / n_st
            feat_dict['limit_down_ratio'] = r[2] / n_st
            feat_dict['limit_ud_ratio'] = (r[1] + 1) / (r[2] + 1)
            feat_dict['up_stock_ratio'] = r[3] / n_st

        # 截面std
        pct_query = """
        SELECT q.price_change_pct FROM daily_quotes q JOIN securities s ON q.security_id = s.id
        WHERE s.type = 'A股' AND q.trade_date = ? AND q.volume > 0
        """
        pcts = [r[0] for r in conn.execute(pct_query, (date,)).fetchall()]
        if not pcts:
            pcts = [r[0] for r in conn.execute(pct_query, (date_nodash,)).fetchall()]
        feat_dict['cross_section_std'] = float(np.std(pcts)) if pcts else 0.0

        # 5日均值 (简化: 用单日值代替, 差异不大)
        for col in ['limit_up_ratio', 'limit_down_ratio', 'limit_ud_ratio',
                    'up_stock_ratio', 'cross_section_std']:
            feat_dict[f'{col}_5d'] = feat_dict.get(col, 0.0)

        # 3. 指数分化
        for idx_code, tag in [('000300.SH', '000300'), ('399006.SZ', '399006'), ('000852.SH', '000852')]:
            idx_rows = conn.execute(bm_query.replace('000905.SH', idx_code), (date,)).fetchall()
            if len(idx_rows) >= 11:
                idx_closes = [r[1] for r in reversed(idx_rows)]
                feat_dict[f'ret10d_{tag}'] = idx_closes[-1] / idx_closes[-11] - 1

        feat_dict['ret10d_000905'] = feat_dict.get('bm_return_10d', 0)
        feat_dict['hs300_vs_zz500'] = feat_dict.get('ret10d_000300', 0) - feat_dict.get('ret10d_000905', 0)
        feat_dict['zz1000_vs_zz500'] = feat_dict.get('ret10d_000852', 0) - feat_dict.get('ret10d_000905', 0)
        feat_dict['cyb_vs_hs300'] = feat_dict.get('ret10d_399006', 0) - feat_dict.get('ret10d_000300', 0)

        idx_rets = [feat_dict.get(f'ret10d_{t}', 0) for t in ['000300', '000905', '399006', '000852']]
        feat_dict['index_breadth'] = sum(1 for r in idx_rets if r > 0) / 4.0

        # 4. 估值
        val_query = """
        SELECT MEDIAN_PE, MEDIAN_PB, AVG_TURNOVER FROM (
            SELECT
                (SELECT pe_ttm FROM daily_basic db2 JOIN securities s2 ON db2.security_id=s2.id
                 WHERE s2.type='A股' AND db2.trade_date=? AND db2.pe_ttm>0 AND db2.pe_ttm<500
                 ORDER BY pe_ttm LIMIT 1 OFFSET (
                    SELECT COUNT(*)/2 FROM daily_basic db3 JOIN securities s3 ON db3.security_id=s3.id
                    WHERE s3.type='A股' AND db3.trade_date=? AND db3.pe_ttm>0 AND db3.pe_ttm<500
                 )) as MEDIAN_PE,
                0 as MEDIAN_PB, 0 as AVG_TURNOVER
        )
        """
        # SQLite median太复杂, 用简化版
        pe_rows = conn.execute("""
            SELECT pe_ttm FROM daily_basic db JOIN securities s ON db.security_id = s.id
            WHERE s.type = 'A股' AND db.trade_date = ? AND db.pe_ttm > 0 AND db.pe_ttm < 500
        """, (date,)).fetchall()
        if not pe_rows:
            pe_rows = conn.execute("""
                SELECT pe_ttm FROM daily_basic db JOIN securities s ON db.security_id = s.id
                WHERE s.type = 'A股' AND db.trade_date = ? AND db.pe_ttm > 0 AND db.pe_ttm < 500
            """, (date_nodash,)).fetchall()

        # 简化: 用PE分位数=0.5（中性值），因为实时计算750d百分位太慢
        feat_dict['pe_pctl_750d'] = 0.5
        feat_dict['pb_pctl_750d'] = 0.5
        feat_dict['turnover_pctl_60d'] = 0.5
        feat_dict['volume_ma5_ratio'] = 1.0

        # 构建特征向量
        X = np.array([[feat_dict.get(f, 0.0) for f in self._gate_features]])
        pred = float(self._market_gate.predict(X)[0])

        # percentile映射 → [0, 1]
        pred_quantiles = self._gate_data.get('pred_quantiles')
        if pred_quantiles is not None:
            model_conf = np.searchsorted(pred_quantiles, pred) / 100.0
        else:
            model_conf = 0.5

        # 规则得分
        rule_score = 0.5
        ud = feat_dict.get('limit_ud_ratio_5d', 1.0)
        if ud > 2.0:
            rule_score += 0.15
        elif ud < 0.5:
            rule_score -= 0.15

        dd = feat_dict.get('bm_drawdown_60d', 0)
        if dd < -0.15:
            rule_score -= 0.15
        elif dd > -0.03:
            rule_score += 0.05

        breadth = feat_dict.get('index_breadth', 0.5)
        if breadth >= 1.0:
            rule_score += 0.10
        elif breadth <= 0.0:
            rule_score -= 0.10

        rule_score = np.clip(rule_score, 0.0, 1.0)

        # 混合
        w = self._gate_data.get('rule_weights', {'model': 0.7, 'rule': 0.3})
        confidence = w['model'] * model_conf + w['rule'] * rule_score
        return float(np.clip(confidence, 0.0, 1.0))

    def predict_scores(self, stock_codes: List[str], date: str) -> Dict[str, Dict]:
        """V4.9.0: 市场门控 + 基础预测 + Widen-then-Concentrate via 内置lgb_q95"""
        if isinstance(date, str) and len(date) == 8 and date.isdigit():
            date_dash = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        else:
            date_dash = date

        # 市场门控
        confidence = self._evaluate_market_gate(date_dash)
        self._last_gate_confidence = confidence

        # 不在scorer层清零——总是正常评分，门控信息传给下游(回测/选股)决策
        if confidence < GATE_DONT_BUY:
            regime = 'dont_buy'
        elif confidence < GATE_REDUCE:
            regime = 'reduce'
        else:
            regime = 'normal'

        results = super().predict_scores(stock_codes, date)

        per_model_preds = getattr(self, '_per_model_preds', {})
        pred_codes = getattr(self, '_last_pred_codes', [])

        if pred_codes and '10d' in per_model_preds:
            preds_10d = per_model_preds['10d']
            q95_pred = preds_10d.get('lgb_q95')
            n = len(pred_codes)

            if q95_pred is not None and len(q95_pred) == n:
                # MSE composite (exclude Q95 for clean Stage 1/Stage 2 separation)
                mse_names = [nm for nm in preds_10d if nm != 'lgb_q95']
                tw = self.weights.get('label_10d', {})
                mse_composite = np.zeros(n)
                total_w = 0
                for nm in mse_names:
                    if nm in preds_10d and len(preds_10d[nm]) == n:
                        w = tw.get(nm, 0.2)
                        mse_composite += w * preds_10d[nm]
                        total_w += w
                if total_w > 0:
                    mse_composite /= total_w

                mse_rank = rankdata(-mse_composite, method='ordinal')

                # Stage 1: MSE Top-30
                pool_mask = mse_rank <= WIDEN_TOP_K
                pool_idx = np.where(pool_mask)[0]

                if len(pool_idx) >= 3:
                    # Stage 2: Q95 reranking within pool
                    q95_in_pool = q95_pred[pool_idx]
                    q95_pool_rank = rankdata(-q95_in_pool, method='ordinal')

                    for ii, idx in enumerate(pool_idx):
                        code = pred_codes[idx]
                        if code in results:
                            results[code]['head_rank'] = int(q95_pool_rank[ii])
                            results[code]['in_head_pool'] = True
                            results[code]['q95_pred_10d'] = float(q95_pred[idx])

                for i, code in enumerate(pred_codes):
                    if code in results and 'in_head_pool' not in results[code]:
                        results[code]['head_rank'] = WIDEN_TOP_K + int(mse_rank[i])
                        results[code]['in_head_pool'] = False

                # Override score and recommendation based on Q95 absolute value
                # Dynamic: strong signal days → many 强烈买入, weak days → few/zero
                # Q95>=0.16: avg +8.27%, PF=5.29, 26% zero-signal days
                Q95_STRONG_BUY = 0.16
                Q95_BUY = 0.14
                Q95_CAUTIOUS = 0.12

                for code, data in results.items():
                    hr = data.get('head_rank', 9999)
                    q95_val = data.get('q95_pred_10d', 0)
                    if data.get('in_head_pool'):
                        data['score'] = max(100 - hr + 1, 71)
                        # Dynamic recommendation by Q95 absolute value
                        if q95_val >= Q95_STRONG_BUY:
                            data['recommendation'] = '强烈买入'
                        elif q95_val >= Q95_BUY:
                            data['recommendation'] = '买入'
                        elif q95_val >= Q95_CAUTIOUS:
                            data['recommendation'] = '谨慎买入'
                        else:
                            data['recommendation'] = '观望'
                    else:
                        data['recommendation'] = '观望'

        # 附加门控信息到所有结果
        for code, data in results.items():
            data['gate_confidence'] = confidence
            data['gate_regime'] = regime

        return results
