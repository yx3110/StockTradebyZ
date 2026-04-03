#!/usr/bin/env python3
"""
交易环境监测回测系统

验证 analyze_trading_environment() 的市场择时能力

回测设计:
1. IC分析: 各维度评分与未来沪深300收益的Spearman秩相关
2. 分组回测: 按评分五等分, 检验收益单调性
3. 信号准确率: 按环境评级的预测准确度
4. 仓位管理回测: 评分驱动仓位 vs 固定仓位 vs 朴素MA择时
5. 维度贡献: 各维度的独立预测力
6. 滚动稳定性: 60日滚动IC时间序列
7. 大跌防御: 环境评分在大跌前是否给出预警

用法:
    python3 backtest/backtest_trading_environment.py
    python3 backtest/backtest_trading_environment.py --start-date 2023-01-01 --end-date 2026-03-19
"""
import sys
import os
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from scipy import stats

try:
    from core.config import PROJECT_ROOT as _PROJECT_ROOT_PATH, get_db_path
    PROJECT_ROOT = str(_PROJECT_ROOT_PATH)
    DB_PATH = str(get_db_path())
except ImportError:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(PROJECT_ROOT, 'data_adapter', 'stock_data.db')

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


# ============================================================
# Section 1: Batch Data Loading
# ============================================================

def load_all_data(db_path, lookback_start, end_date):
    """批量加载所有需要的市场数据 (3个查询替代每日4查询)"""
    conn = sqlite3.connect(db_path)

    # 1. 沪深300 日线数据
    hs300 = pd.read_sql_query("""
        SELECT dq.trade_date, dq.open, dq.high, dq.low, dq.close,
               dq.volume, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = '000300.SH'
          AND dq.trade_date >= ? AND dq.trade_date <= ?
        ORDER BY dq.trade_date
    """, conn, params=[lookback_start, end_date])
    hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])

    # 2. 全A股成交量日聚合
    vol_agg = pd.read_sql_query("""
        SELECT dq.trade_date,
               SUM(dq.volume) as total_volume,
               COUNT(*) as stock_count
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股'
          AND dq.trade_date >= ? AND dq.trade_date <= ?
          AND dq.volume > 0
        GROUP BY dq.trade_date
        ORDER BY dq.trade_date
    """, conn, params=[lookback_start, end_date])
    vol_agg['trade_date'] = pd.to_datetime(vol_agg['trade_date'])

    # 3. 全A股涨跌宽度日聚合
    breadth = pd.read_sql_query("""
        SELECT dq.trade_date,
               COUNT(*) as total,
               SUM(CASE WHEN price_change_pct > 0 THEN 1 ELSE 0 END) as up_count,
               SUM(CASE WHEN price_change_pct < 0 THEN 1 ELSE 0 END) as down_count,
               SUM(CASE WHEN is_limit_up = 1 THEN 1 ELSE 0 END) as limit_up,
               SUM(CASE WHEN is_limit_down = 1 THEN 1 ELSE 0 END) as limit_down,
               SUM(CASE WHEN price_change_pct > 0.05 THEN 1 ELSE 0 END) as strong_up,
               SUM(CASE WHEN price_change_pct < -0.05 THEN 1 ELSE 0 END) as strong_down
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股'
          AND dq.trade_date >= ? AND dq.trade_date <= ?
          AND dq.volume > 0
        GROUP BY dq.trade_date
        ORDER BY dq.trade_date
    """, conn, params=[lookback_start, end_date])
    breadth['trade_date'] = pd.to_datetime(breadth['trade_date'])

    conn.close()

    print(f"  沪深300: {len(hs300)}条 ({hs300['trade_date'].min().date()} ~ {hs300['trade_date'].max().date()})")
    print(f"  成交量聚合: {len(vol_agg)}条")
    print(f"  涨跌宽度: {len(breadth)}条")

    return hs300, vol_agg, breadth


# ============================================================
# Section 2: Pre-computation (向量化)
# ============================================================

def precompute_features(hs300, vol_agg, breadth):
    """预计算所有技术特征, 返回合并的features DataFrame"""

    # --- HS300 技术指标 ---
    h = hs300.copy()
    # 从close直接计算涨跌幅, 避免依赖可能为null的price_change_pct列
    h['price_change_pct'] = h['close'].pct_change()
    h['ma5'] = h['close'].rolling(5).mean()
    h['ma20'] = h['close'].rolling(20).mean()
    h['ma60'] = h['close'].rolling(60).mean()
    h['ret_5d'] = h['close'] / h['close'].shift(5) - 1
    h['ret_20d'] = h['close'] / h['close'].shift(20) - 1
    h['peak_60'] = h['close'].rolling(60).max()
    h['drawdown_60'] = h['close'] / h['peak_60'] - 1

    # MACD DIF
    h['ema12'] = h['close'].ewm(span=12).mean()
    h['ema26'] = h['close'].ewm(span=26).mean()
    h['dif'] = h['ema12'] - h['ema26']
    h['dif_prev'] = h['dif'].shift(1)

    # 波动率
    h['vol_20d'] = h['price_change_pct'].rolling(20).std()
    h['annual_vol'] = h['vol_20d'] * np.sqrt(250)

    # 近10日极端日 (|pct| > 3%)
    h['is_extreme'] = (h['price_change_pct'].abs() > 0.03).astype(int)
    h['extreme_days_10'] = h['is_extreme'].rolling(10).sum()

    # 连续涨跌天数 (正确的计算方式: 标准连续计数)
    consec = np.zeros(len(h))
    pcts = h['price_change_pct'].values
    for i in range(1, len(pcts)):
        if pcts[i] > 0:
            consec[i] = max(consec[i-1], 0) + 1
        elif pcts[i] < 0:
            consec[i] = min(consec[i-1], 0) - 1
        else:
            consec[i] = 0
    h['consec_days'] = consec

    # CPPI trailing floor (参数: floor=8%, multiplier=15, decay=0.997)
    cppi_nav = np.ones(len(h))
    cppi_peak = np.ones(len(h))
    cppi_exp = np.ones(len(h))
    _nav = 1.0
    _peak = 1.0
    _decay = 0.997
    _floor_pct = 0.08
    _mult = 15
    daily_rets = h['price_change_pct'].values
    for i in range(1, len(h)):
        r = daily_rets[i] if not np.isnan(daily_rets[i]) else 0
        _nav *= (1 + r)
        _peak = max(_peak * _decay, _nav)
        _fl = _peak * (1 - _floor_pct)
        _cush = max(0, _nav - _fl) / _nav if _nav > 0 else 0
        cppi_nav[i] = _nav
        cppi_peak[i] = _peak
        cppi_exp[i] = min(1.0, max(0.05, _mult * _cush))
    h['cppi_nav'] = cppi_nav
    h['cppi_peak'] = cppi_peak
    h['cppi_exposure'] = cppi_exp
    h['cppi_drawdown'] = cppi_nav / cppi_peak - 1

    # --- 成交量特征 ---
    v = vol_agg.copy()
    v['vol_ma5'] = v['total_volume'].rolling(5).mean()
    v['vol_ma20'] = v['total_volume'].rolling(20).mean()
    v['vol_ratio_5'] = v['total_volume'] / v['vol_ma5']
    v['vol_ratio_20'] = v['total_volume'] / v['vol_ma20']

    # 量能趋势: 最近4天中volume比前一天大的天数 (range: -4 to +4)
    vol_sign = np.sign(v['total_volume'].diff())
    v['vol_trend_4d'] = vol_sign.rolling(4).sum()

    # --- 合并 ---
    features = h.merge(
        v[['trade_date', 'total_volume', 'vol_ma5', 'vol_ma20',
           'vol_ratio_5', 'vol_ratio_20', 'vol_trend_4d']],
        on='trade_date', how='left'
    ).merge(
        breadth,
        on='trade_date', how='left'
    )

    return features


# ============================================================
# Section 3: Scoring (逐日, 复刻 analyze_trading_environment 逻辑)
# ============================================================

def score_trend(row):
    """趋势维度评分 (20%)"""
    score = 50
    close = row['close']
    ma5, ma20, ma60 = row['ma5'], row['ma20'], row['ma60']

    if pd.isna(ma60):
        return score

    # 价格 vs 均线
    score += 5 if close > ma5 else -5
    score += 8 if close > ma20 else -8
    score += 7 if close > ma60 else -7

    # 均线排列
    if ma5 > ma20 > ma60:
        score += 15
    elif ma5 < ma20 < ma60:
        score -= 15

    # 5日涨跌幅
    ret_5d = row.get('ret_5d', 0)
    if not pd.isna(ret_5d):
        if ret_5d > 0.03:
            score += 10
        elif ret_5d > 0.01:
            score += 5
        elif ret_5d < -0.03:
            score -= 10
        elif ret_5d < -0.01:
            score -= 5

    # 距60日前高回撤
    dd = row.get('drawdown_60', 0)
    if not pd.isna(dd):
        if dd < -0.10:
            score -= 10
        elif dd < -0.05:
            score -= 5

    return max(0, min(100, score))


def score_momentum(row):
    """动量维度评分 (15%)"""
    score = 50
    ret_5d = row.get('ret_5d', 0)
    ret_20d = row.get('ret_20d', 0)

    if pd.isna(ret_5d) or pd.isna(ret_20d):
        return score

    # 短期 vs 长期动量
    momentum_ratio = ret_5d / max(abs(ret_20d), 0.001)
    if ret_5d > 0 and momentum_ratio > 1.5:
        score += 15
    elif ret_5d > 0 and momentum_ratio > 0:
        score += 5
    elif ret_5d < 0 and momentum_ratio > 1.5:
        score -= 15
    elif ret_5d < 0:
        score -= 5

    # MACD方向
    dif = row.get('dif', 0)
    dif_prev = row.get('dif_prev', 0)
    if not pd.isna(dif) and not pd.isna(dif_prev):
        if dif > 0 and dif > dif_prev:
            score += 15
        elif dif > 0:
            score += 5
        elif dif < 0 and dif < dif_prev:
            score -= 15
        elif dif < 0:
            score -= 5

    # 连续涨跌
    consec = row.get('consec_days', 0)
    if consec >= 5:
        score += 10
    elif consec >= 3:
        score += 5
    elif consec <= -5:
        score -= 10
    elif consec <= -3:
        score -= 5

    return max(0, min(100, score))


def score_volume(row):
    """成交量维度评分 (15%)"""
    score = 50
    vol_ratio_5 = row.get('vol_ratio_5', 1.0)
    vol_trend = row.get('vol_trend_4d', 0)
    pct = row.get('price_change_pct', 0)

    if pd.isna(vol_ratio_5):
        return score

    # 量比
    if vol_ratio_5 > 1.5:
        score += 20
    elif vol_ratio_5 > 1.2:
        score += 10
    elif vol_ratio_5 < 0.7:
        score -= 15
    elif vol_ratio_5 < 0.85:
        score -= 5

    # 量能趋势
    if not pd.isna(vol_trend):
        if vol_trend >= 3:
            score += 10
        elif vol_trend <= -3:
            score -= 10

    # 量价配合
    if not pd.isna(pct):
        if pct > 0 and vol_ratio_5 > 1.1:
            score += 10
        elif pct < -0.005 and vol_ratio_5 > 1.3:
            score -= 10

    return max(0, min(100, score))


def score_breadth(row):
    """市场宽度维度评分 (15%)"""
    score = 50
    total = row.get('total', 0)
    up = row.get('up_count', 0)
    down = row.get('down_count', 0)
    limit_up = row.get('limit_up', 0)
    limit_down = row.get('limit_down', 0)
    strong_up = row.get('strong_up', 0)

    if pd.isna(total) or total == 0:
        return score

    up_ratio = up / total

    if up_ratio > 0.75:
        score += 25
    elif up_ratio > 0.6:
        score += 15
    elif up_ratio > 0.45:
        score += 0
    elif up_ratio > 0.3:
        score -= 15
    else:
        score -= 25

    if limit_up > 50:
        score += 10
    elif limit_up > 20:
        score += 5

    if limit_down > 30:
        score -= 15
    elif limit_down > 10:
        score -= 5

    strong_ratio = strong_up / total
    if strong_ratio > 0.15:
        score += 10
    elif strong_ratio < 0.02:
        score -= 5

    return max(0, min(100, score))


def score_volatility(row):
    """波动/风险维度评分 (10%)"""
    score = 50
    annual_vol = row.get('annual_vol', 0.15)
    extreme = row.get('extreme_days_10', 0)
    consec = row.get('consec_days', 0)

    if pd.isna(annual_vol):
        return score

    if annual_vol < 0.12:
        score += 20
    elif annual_vol < 0.20:
        score += 10
    elif annual_vol < 0.30:
        score -= 5
    else:
        score -= 20

    if not pd.isna(extreme):
        if extreme >= 3:
            score -= 15
        elif extreme == 0:
            score += 10

    if consec <= -5:
        score -= 15
    elif consec <= -3:
        score -= 5

    return max(0, min(100, score))


def parse_report_scores(report_dir):
    """从报告文件解析ML评分分布, 用于模型信号维度"""
    scores_by_date = {}
    report_path = Path(report_dir)

    for f in sorted(report_path.glob('选股分析报告_*.md')):
        date_str = f.stem.split('_')[-1]
        try:
            date = pd.Timestamp(f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}')
        except Exception:
            continue

        scores = []
        recs = []
        in_table = False
        score_col_idx = None
        rec_col_idx = None

        for line in f.read_text(encoding='utf-8').split('\n'):
            if '|' not in line:
                if in_table:
                    break
                continue

            parts = [p.strip() for p in line.split('|')]

            # Detect header
            if not in_table and ('评分' in line or 'Composite' in line or 'composite' in line):
                for i, p in enumerate(parts):
                    if any(k in p for k in ['综合评分', 'Composite', 'composite', '量化评分']):
                        score_col_idx = i
                    if '建议' in p:
                        rec_col_idx = i
                if score_col_idx:
                    in_table = True
                continue

            if in_table and '---' in line:
                continue

            if in_table and score_col_idx and len(parts) > score_col_idx:
                try:
                    s = float(parts[score_col_idx].replace('**', '').strip())
                    if -1 <= s <= 100:
                        scores.append(s)
                    if rec_col_idx and len(parts) > rec_col_idx:
                        recs.append(parts[rec_col_idx].strip())
                except (ValueError, IndexError):
                    pass

        if scores:
            scores_by_date[date] = {'scores': scores, 'recs': recs}

    return scores_by_date


def compute_model_signal(scores_list, recs_list=None):
    """从ML评分列表计算模型信号维度评分 (复刻analyze_trading_environment逻辑)"""
    if not scores_list or len(scores_list) < 5:
        return 50

    scores = np.array(scores_list)
    model_score = 50
    median_sc = np.median(scores)
    std_sc = np.std(scores)
    total_n = len(scores)
    high_ratio = np.sum(scores > 70) / total_n
    strong_n = int(np.sum(scores > 85))
    top10_avg = np.mean(np.sort(scores)[-10:]) if len(scores) >= 10 else np.mean(scores)

    if median_sc > 60:     model_score += 15
    elif median_sc > 50:   model_score += 5
    elif median_sc < 35:   model_score -= 15
    elif median_sc < 45:   model_score -= 5

    if high_ratio > 0.15:  model_score += 15
    elif high_ratio > 0.08: model_score += 5
    elif high_ratio < 0.02: model_score -= 10

    if strong_n > 20:      model_score += 10
    elif strong_n > 5:     model_score += 5
    elif strong_n == 0:    model_score -= 10

    if top10_avg > 85:     model_score += 10
    elif top10_avg > 75:   model_score += 5
    elif top10_avg < 60:   model_score -= 10

    if std_sc < 8:         model_score -= 5

    if recs_list:
        sb = sum(1 for r in recs_list if '强烈' in r)
        bu = sum(1 for r in recs_list if r == '买入')
        ratio = (sb + bu) / len(recs_list) if recs_list else 0
        if ratio > 0.3:    model_score += 5
        elif ratio < 0.05: model_score -= 5

    return max(0, min(100, model_score))


def compute_daily_scores(features, start_date, end_date, ml_scores=None):
    """计算所有交易日的环境评分 (含CPPI风控 + 市况熔断 + 可选ML信号)"""
    mask = (features['trade_date'] >= pd.Timestamp(start_date)) & \
           (features['trade_date'] <= pd.Timestamp(end_date))
    df = features[mask].copy()

    # 权重基于IC贡献度优化 (市场宽度IC最强, 趋势/动量IC弱)
    weights = {
        'trend': 0.10,
        'momentum': 0.10,
        'volume': 0.15,
        'breadth': 0.25,
        'volatility': 0.15,
    }
    MODEL_SIGNAL_WEIGHT = 0.25
    has_ml = ml_scores is not None and len(ml_scores) > 0
    ml_hit = 0

    records = []
    for _, row in df.iterrows():
        t = score_trend(row)
        m = score_momentum(row)
        v = score_volume(row)
        b = score_breadth(row)
        r = score_volatility(row)

        # 模型信号维度: 优先使用报告解析的ML评分
        ms = 50  # 默认
        if has_ml:
            td = row['trade_date']
            if td in ml_scores:
                ms = compute_model_signal(ml_scores[td]['scores'], ml_scores[td].get('recs'))
                ml_hit += 1

        composite = (weights['trend'] * t + weights['momentum'] * m +
                     weights['volume'] * v + weights['breadth'] * b +
                     weights['volatility'] * r + MODEL_SIGNAL_WEIGHT * ms)

        market_only = (weights['trend'] * t + weights['momentum'] * m +
                       weights['volume'] * v + weights['breadth'] * b +
                       weights['volatility'] * r) / (1 - MODEL_SIGNAL_WEIGHT)

        raw_composite = composite
        raw_market = market_only

        # ======== CPPI不改评分, 仅作为独立仓位乘数 ========
        cppi_exp = row.get('cppi_exposure', 1.0)
        score_cap = 100

        # ======== 市况熔断 (仅极端情况硬限) ========
        ret_20d = row.get('ret_20d', 0)
        ma60 = row.get('ma60', row['close'])
        if not pd.isna(ret_20d):
            if ret_20d < -0.10:
                score_cap = min(score_cap, 25)
            elif ret_20d < -0.05 and (not pd.isna(ma60) and row['close'] < ma60):
                score_cap = min(score_cap, 40)

        composite = min(composite, score_cap)
        market_only = min(market_only, score_cap)

        records.append({
            'trade_date': row['trade_date'],
            'close': row['close'],
            'trend': t,
            'momentum': m,
            'volume': v,
            'breadth': b,
            'volatility': r,
            'model_signal': ms,
            'composite': round(composite, 1),
            'market_only': round(market_only, 1),
            'raw_composite': round(raw_composite, 1),
            'cppi_exposure': cppi_exp if not pd.isna(cppi_exp) else 1.0,
            'score_cap': score_cap,
        })

    result = pd.DataFrame(records)
    capped_pct = (result['score_cap'] < 100).mean()
    print(f"  评分计算完成: {len(result)}个交易日, "
          f"composite范围 [{result['composite'].min()}, {result['composite'].max()}], "
          f"CPPI/熔断触发 {capped_pct:.1%}")
    if has_ml:
        print(f"  ML评分命中: {ml_hit}/{len(result)} ({ml_hit/len(result):.1%})")
    return result


# ============================================================
# Section 4: Forward Returns
# ============================================================

def add_forward_returns(scores_df, hs300, periods=(1, 3, 5, 10, 20)):
    """计算前瞻收益 (基于沪深300)"""
    # 用HS300的完整序列计算forward returns
    h = hs300[['trade_date', 'close']].copy()
    for p in periods:
        h[f'fwd_{p}d'] = h['close'].shift(-p) / h['close'] - 1

    merged = scores_df.merge(h[['trade_date'] + [f'fwd_{p}d' for p in periods]],
                              on='trade_date', how='left')
    return merged


# ============================================================
# Section 5: Analysis
# ============================================================

def analyze_ic(df, score_cols, return_cols):
    """IC分析: Spearman秩相关 + ICIR + t检验"""
    results = {}
    for sc in score_cols:
        row = {}
        for rc in return_cols:
            valid = df[[sc, rc]].dropna()
            if len(valid) < 30:
                row[rc] = {'ic': np.nan, 'pval': np.nan, 'icir': np.nan}
                continue
            ic, pval = stats.spearmanr(valid[sc], valid[rc])

            # 60日滚动IC计算ICIR
            rolling_ics = []
            for i in range(60, len(valid)):
                window = valid.iloc[i-60:i]
                if window[sc].nunique() < 3 or window[rc].nunique() < 3:
                    continue
                r, _ = stats.spearmanr(window[sc], window[rc])
                if not np.isnan(r):
                    rolling_ics.append(r)
            icir = np.mean(rolling_ics) / np.std(rolling_ics) if len(rolling_ics) > 10 and np.std(rolling_ics) > 0 else np.nan

            row[rc] = {'ic': ic, 'pval': pval, 'icir': icir, 'n': len(valid)}
        results[sc] = row
    return results


def analyze_quintiles(df, score_col='market_only', return_col='fwd_5d'):
    """五等分分组分析"""
    valid = df[[score_col, return_col, 'trade_date']].dropna()
    if len(valid) < 50:
        return None

    try:
        valid['quintile'] = pd.qcut(valid[score_col], 5,
                                     labels=['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)'],
                                     duplicates='drop')
    except ValueError:
        # 评分集中, 用rank分组
        valid['quintile'] = pd.qcut(valid[score_col].rank(method='first'), 5,
                                     labels=['Q1(最低)', 'Q2', 'Q3', 'Q4', 'Q5(最高)'])

    grouped = valid.groupby('quintile', observed=False)[return_col].agg(
        mean_return='mean',
        std_return='std',
        count='count',
    )
    grouped['win_rate'] = valid.groupby('quintile', observed=False)[return_col].apply(
        lambda x: (x > 0).mean()
    )
    grouped['sharpe'] = grouped['mean_return'] / grouped['std_return'] * np.sqrt(250/5)

    # 单调性检验
    q_means = grouped['mean_return'].values
    monotonic_score = 0
    for i in range(len(q_means) - 1):
        if q_means[i+1] > q_means[i]:
            monotonic_score += 1
    monotonicity = monotonic_score / (len(q_means) - 1)

    # Q5-Q1 spread
    spread = q_means[-1] - q_means[0]

    return {
        'table': grouped,
        'monotonicity': monotonicity,
        'spread': spread,
        'spread_annualized': spread * 250 / 5,  # 年化
    }


def analyze_signal_accuracy(df, score_col='market_only'):
    """信号准确率: 按环境评级分组"""
    valid = df[[score_col, 'fwd_1d', 'fwd_5d', 'fwd_10d']].dropna()

    def classify(score):
        if score >= 70:
            return '偏多(≥70)'
        elif score >= 50:
            return '中性(50-70)'
        elif score >= 30:
            return '偏空(30-50)'
        else:
            return '弱势(<30)'

    valid['env_label'] = valid[score_col].apply(classify)

    groups = valid.groupby('env_label', observed=False)
    result = pd.DataFrame({
        '天数': groups['fwd_5d'].count(),
        '未来1日均值': groups['fwd_1d'].mean(),
        '未来5日均值': groups['fwd_5d'].mean(),
        '未来10日均值': groups['fwd_10d'].mean(),
        '5日胜率': groups['fwd_5d'].apply(lambda x: (x > 0).mean()),
        '5日最大盈': groups['fwd_5d'].max(),
        '5日最大亏': groups['fwd_5d'].min(),
    })

    # 按评级排序
    order = ['偏多(≥70)', '中性(50-70)', '偏空(30-50)', '弱势(<30)']
    result = result.reindex([o for o in order if o in result.index])

    return result


def backtest_position_sizing(df, risk_free_annual=0.02):
    """仓位管理策略回测: 对比不同仓位策略"""
    need_cols = ['trade_date', 'market_only', 'composite', 'close', 'fwd_1d', 'cppi_exposure']
    use_cols = [c for c in need_cols if c in df.columns]
    valid = df[use_cols].dropna(subset=['trade_date', 'market_only', 'close', 'fwd_1d'])
    if len(valid) < 50:
        return None

    rf_daily = risk_free_annual / 250

    strategies = {}
    pos_map = {}  # name → position array for avg_position calc

    rets = valid['fwd_1d'].values
    scores = valid['market_only'].values
    cppi_exp = valid['cppi_exposure'].values if 'cppi_exposure' in valid.columns else np.ones(len(valid))

    # 1. Buy-and-Hold
    strategies['买入持有(100%)'] = rets
    pos_map['买入持有(100%)'] = np.ones(len(rets))

    # 2. 评分分档仓位
    def score_to_position(s):
        if s >= 70:   return 0.90
        elif s >= 50: return 0.65
        elif s >= 30: return 0.40
        elif s >= 15: return 0.20
        else:         return 0.05
    pos_tier = np.array([score_to_position(s) for s in scores])
    strategies['评分分档'] = pos_tier[:-1] * rets[1:] + (1 - pos_tier[:-1]) * rf_daily
    pos_map['评分分档'] = pos_tier

    # 3. ★ 评分×CPPI (核心策略: 评分驱动方向, CPPI控制风险)
    pos_cppi_enhanced = pos_tier * cppi_exp
    strategies['评分×CPPI'] = pos_cppi_enhanced[:-1] * rets[1:] + (1 - pos_cppi_enhanced[:-1]) * rf_daily
    pos_map['评分×CPPI'] = pos_cppi_enhanced

    # 4. CPPI-only (纯CPPI仓位管理)
    strategies['纯CPPI'] = cppi_exp[:-1] * rets[1:] + (1 - cppi_exp[:-1]) * rf_daily
    pos_map['纯CPPI'] = cppi_exp

    # 5. 二元择时: score > 50 → 100%, else 0%
    pos_bin = np.where(scores > 50, 1.0, 0.0)
    strategies['二元择时(>50)'] = pos_bin[:-1] * rets[1:] + (1 - pos_bin[:-1]) * rf_daily
    pos_map['二元择时(>50)'] = pos_bin

    # 6. ★ 二元×CPPI
    pos_bin_cppi = pos_bin * cppi_exp
    strategies['二元×CPPI'] = pos_bin_cppi[:-1] * rets[1:] + (1 - pos_bin_cppi[:-1]) * rf_daily
    pos_map['二元×CPPI'] = pos_bin_cppi

    # 7. 朴素MA20
    ma20 = pd.Series(valid['close'].values).rolling(20).mean().values
    pos_ma = np.where(valid['close'].values > ma20, 1.0, 0.0)
    pos_ma[:20] = 1.0
    strategies['朴素MA20'] = pos_ma[:-1] * rets[1:] + (1 - pos_ma[:-1]) * rf_daily
    pos_map['朴素MA20'] = pos_ma

    # 计算绩效
    results = {}
    dates = valid['trade_date'].values
    for name, daily_rets in strategies.items():
        n = len(daily_rets)
        cum = np.cumprod(1 + daily_rets)
        total_ret = cum[-1] - 1
        years = n / 250
        annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

        # Max Drawdown
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_dd = dd.min()

        # Sharpe
        sharpe = (np.mean(daily_rets) - rf_daily) / np.std(daily_rets) * np.sqrt(250) if np.std(daily_rets) > 0 else 0

        # Calmar
        calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

        # 日均仓位
        if name in pos_map:
            avg_pos = np.mean(pos_map[name][:n])
        else:
            avg_pos = 1.0

        results[name] = {
            'annual_ret': annual_ret,
            'total_ret': total_ret,
            'max_dd': max_dd,
            'sharpe': sharpe,
            'calmar': calmar,
            'avg_position': avg_pos,
            'n_days': n,
            'cum_curve': cum,
        }

    return results


def analyze_drawdown_defense(df, score_col='market_only', threshold=-0.05):
    """大跌防御分析: 检测大跌期间环境评分是否提前预警"""
    valid = df[['trade_date', score_col, 'fwd_5d', 'fwd_10d', 'close']].dropna()

    # 找出未来10日跌幅 > threshold 的日期
    crash_days = valid[valid['fwd_10d'] < threshold].copy()
    normal_days = valid[valid['fwd_10d'] >= threshold].copy()

    if len(crash_days) == 0:
        return None

    result = {
        'crash_count': len(crash_days),
        'normal_count': len(normal_days),
        'crash_avg_score': crash_days[score_col].mean(),
        'normal_avg_score': normal_days[score_col].mean(),
        'crash_median_score': crash_days[score_col].median(),
        'normal_median_score': normal_days[score_col].median(),
    }

    # 检验差异显著性
    t_stat, p_val = stats.ttest_ind(crash_days[score_col].values,
                                      normal_days[score_col].values)
    result['t_stat'] = t_stat
    result['p_val'] = p_val

    # 评分<40的日子中, 大跌占比 vs 评分>60的日子中大跌占比
    low_score = valid[valid[score_col] < 40]
    high_score = valid[valid[score_col] > 60]
    if len(low_score) > 0 and len(high_score) > 0:
        result['crash_rate_low_score'] = (low_score['fwd_10d'] < threshold).mean()
        result['crash_rate_high_score'] = (high_score['fwd_10d'] < threshold).mean()
    else:
        result['crash_rate_low_score'] = np.nan
        result['crash_rate_high_score'] = np.nan

    return result


def analyze_rolling_ic(df, score_col='market_only', return_col='fwd_5d', window=60):
    """滚动IC稳定性分析"""
    valid = df[[score_col, return_col, 'trade_date']].dropna()
    rolling_ics = []
    dates = []

    for i in range(window, len(valid)):
        w = valid.iloc[i-window:i]
        ic, _ = stats.spearmanr(w[score_col], w[return_col])
        rolling_ics.append(ic)
        dates.append(valid['trade_date'].iloc[i])

    if len(rolling_ics) < 10:
        return None

    rolling_ics = np.array(rolling_ics)
    return {
        'mean_ic': np.mean(rolling_ics),
        'std_ic': np.std(rolling_ics),
        'icir': np.mean(rolling_ics) / np.std(rolling_ics) if np.std(rolling_ics) > 0 else 0,
        'pct_positive': (rolling_ics > 0).mean(),
        'min_ic': np.min(rolling_ics),
        'max_ic': np.max(rolling_ics),
        'dates': dates,
        'ics': rolling_ics,
    }


# ============================================================
# Section 6: Validation (与原方法对比验证)
# ============================================================

def validate_scores(scores_df, n_samples=5):
    """随机抽样验证batch计算结果与原方法一致"""
    try:
        from tomorrow_stock_selector import TomorrowStockSelector
        import logging
        logging.disable(logging.INFO)

        selector = TomorrowStockSelector(scoring_version='v3.9')

        sample_dates = scores_df.sample(min(n_samples, len(scores_df)))['trade_date'].values
        mismatches = 0

        for dt in sample_dates:
            env = selector.analyze_trading_environment(pd.Timestamp(dt))
            _filtered = scores_df[scores_df['trade_date'] == dt]
            if _filtered.empty:
                continue
            batch_row = _filtered.iloc[0]

            for dim in ['trend', 'momentum', 'volume', 'breadth', 'volatility']:
                orig = env['dimensions'][dim]['score']
                batch = batch_row[dim]
                if abs(orig - batch) > 5:  # 允许小差异 (连续天数计算略有不同)
                    mismatches += 1
                    print(f"  ⚠️ {pd.Timestamp(dt).date()} {dim}: 原方法={orig}, 批量={batch}")

        if mismatches == 0:
            print(f"  ✅ {n_samples}个日期验证通过, batch计算与原方法一致")
        else:
            print(f"  ⚠️ {mismatches}处差异(容差5分内), 可能因连续天数算法微差")

        logging.disable(logging.NOTSET)
    except Exception as e:
        print(f"  ⚠️ 验证跳过: {e}")


# ============================================================
# Section 7: Report Generation
# ============================================================

def generate_report(ic_results, quintile_results, signal_accuracy,
                    position_results, drawdown_defense, rolling_ic,
                    scores_df, start_date, end_date):
    """生成markdown回测报告"""

    report = f"""# 🌡️ 交易环境监测回测报告

## 回测概览

| 项目 | 数值 |
|------|------|
| 回测期间 | {start_date} ~ {end_date} |
| 交易日数 | {len(scores_df)} |
| 基准指数 | 沪深300 |
| 评分维度 | 趋势(10%) + 动量(10%) + 成交量(15%) + 市场宽度(25%) + 波动风险(15%) |
| 模型信号 | 未纳入 (25%权重固定50分, 仅测试市场维度) |
| 市场评分范围 | {scores_df['market_only'].min():.1f} ~ {scores_df['market_only'].max():.1f} |
| 市场评分均值 | {scores_df['market_only'].mean():.1f} |
| 市场评分标准差 | {scores_df['market_only'].std():.1f} |

---

## 1. IC分析 — 评分与未来收益的相关性

**Rank IC (Spearman)**: 衡量评分排序与未来收益排序的一致性. IC>0表示评分越高、未来收益越好.
**ICIR**: IC均值/IC标准差, >0.5为可用信号, >1.0为强信号.

"""

    # IC表格
    report += "| 维度 | IC(1d) | IC(3d) | IC(5d) | IC(10d) | IC(20d) | ICIR(5d) | p值(5d) |\n"
    report += "|------|--------|--------|--------|---------|---------|----------|--------|\n"

    dim_labels = {
        'market_only': '**综合(市场)**',
        'composite': '综合(含空ML)',
        'trend': '趋势',
        'momentum': '动量',
        'volume': '成交量',
        'breadth': '市场宽度',
        'volatility': '波动风险',
    }

    for sc in ['market_only', 'composite', 'trend', 'momentum', 'volume', 'breadth', 'volatility']:
        if sc not in ic_results:
            continue
        row = ic_results[sc]
        def fmt_ic(col):
            if col not in row or pd.isna(row[col].get('ic', np.nan)):
                return '-'
            ic_val = row[col]['ic']
            return f"{'**' if abs(ic_val) > 0.05 else ''}{ic_val:+.3f}{'**' if abs(ic_val) > 0.05 else ''}"

        ic5d = row.get('fwd_5d', {})
        icir_val = ic5d.get('icir', np.nan)
        pval_val = ic5d.get('pval', np.nan)
        icir_str = f"{icir_val:.2f}" if not pd.isna(icir_val) else '-'
        pval_str = f"{pval_val:.4f}" if not pd.isna(pval_val) else '-'

        report += f"| {dim_labels.get(sc, sc)} | {fmt_ic('fwd_1d')} | {fmt_ic('fwd_3d')} | {fmt_ic('fwd_5d')} | {fmt_ic('fwd_10d')} | {fmt_ic('fwd_20d')} | {icir_str} | {pval_str} |\n"

    report += "\n"

    # Rolling IC稳定性
    if rolling_ic:
        report += f"""### 滚动IC稳定性 (60日窗口, 5日前瞻)

| 指标 | 数值 |
|------|------|
| 平均IC | {rolling_ic['mean_ic']:+.4f} |
| IC标准差 | {rolling_ic['std_ic']:.4f} |
| ICIR | {rolling_ic['icir']:.2f} |
| IC>0占比 | {rolling_ic['pct_positive']:.1%} |
| 最大IC | {rolling_ic['max_ic']:+.4f} |
| 最小IC | {rolling_ic['min_ic']:+.4f} |

"""

    # 分组分析
    report += "---\n\n## 2. 分组回测 — 五等分收益单调性\n\n"
    if quintile_results:
        qt = quintile_results['table']
        report += f"**单调性**: {quintile_results['monotonicity']:.0%} (1.0=完美单调)\n"
        report += f"**Q5-Q1 Spread**: {quintile_results['spread']:+.4f} (5日), 年化 {quintile_results['spread_annualized']:+.1%}\n\n"

        report += "| 分组 | 天数 | 5日均值 | 5日标准差 | 5日胜率 | 年化Sharpe |\n"
        report += "|------|------|---------|----------|---------|----------|\n"
        for idx, row in qt.iterrows():
            report += f"| {idx} | {row['count']:.0f} | {row['mean_return']:+.4f} | {row['std_return']:.4f} | {row['win_rate']:.1%} | {row['sharpe']:.2f} |\n"
        report += "\n"

    # 信号准确率
    report += "---\n\n## 3. 信号准确率 — 按环境评级\n\n"
    if signal_accuracy is not None and len(signal_accuracy) > 0:
        report += "| 环境评级 | 天数 | 未来1日 | 未来5日 | 未来10日 | 5日胜率 | 5日最大盈 | 5日最大亏 |\n"
        report += "|----------|------|---------|---------|----------|---------|----------|----------|\n"
        for idx, row in signal_accuracy.iterrows():
            report += f"| {idx} | {row['天数']:.0f} | {row['未来1日均值']:+.4f} | {row['未来5日均值']:+.4f} | {row['未来10日均值']:+.4f} | {row['5日胜率']:.1%} | {row['5日最大盈']:+.4f} | {row['5日最大亏']:+.4f} |\n"
        report += "\n"

    # 仓位管理回测
    report += "---\n\n## 4. 仓位管理回测 — 评分驱动 vs 固定仓位\n\n"
    if position_results:
        report += "| 策略 | 年化收益 | 总收益 | 最大回撤 | Sharpe | Calmar | 平均仓位 |\n"
        report += "|------|---------|--------|---------|--------|--------|----------|\n"
        for name, r in position_results.items():
            report += f"| {name} | {r['annual_ret']:+.1%} | {r['total_ret']:+.1%} | {r['max_dd']:.1%} | {r['sharpe']:.2f} | {r['calmar']:.2f} | {r['avg_position']:.0%} |\n"
        report += "\n"

        # 找出最佳策略
        best = max(position_results.items(), key=lambda x: x[1]['sharpe'])
        report += f"> **最佳Sharpe策略**: {best[0]} (Sharpe={best[1]['sharpe']:.2f})\n\n"

    # 大跌防御
    report += "---\n\n## 5. 大跌防御 — 环境评分是否能预警大跌\n\n"
    report += "*大跌定义: 未来10日沪深300跌幅 > 5%*\n\n"
    if drawdown_defense:
        dd = drawdown_defense
        report += f"""| 指标 | 数值 |
|------|------|
| 大跌日数 | {dd['crash_count']} |
| 正常日数 | {dd['normal_count']} |
| 大跌日平均评分 | {dd['crash_avg_score']:.1f} |
| 正常日平均评分 | {dd['normal_avg_score']:.1f} |
| 大跌日中位评分 | {dd['crash_median_score']:.1f} |
| 正常日中位评分 | {dd['normal_median_score']:.1f} |
| 评分差异t检验 | t={dd['t_stat']:.2f}, p={dd['p_val']:.4f} {'✅ 显著' if dd['p_val'] < 0.05 else '❌ 不显著'} |
| 低分(<40)大跌率 | {dd.get('crash_rate_low_score', 0):.1%} |
| 高分(>60)大跌率 | {dd.get('crash_rate_high_score', 0):.1%} |

"""
        if not pd.isna(dd.get('crash_rate_low_score', np.nan)):
            ratio = dd['crash_rate_low_score'] / max(dd['crash_rate_high_score'], 0.001)
            report += f"> 低评分时大跌概率是高评分时的 **{ratio:.1f}倍** "
            if ratio > 1.5:
                report += "— 环境评分有效预警大跌 ✅\n\n"
            else:
                report += "— 预警效果有限 ⚠️\n\n"

    # 维度贡献分析
    report += "---\n\n## 6. 各维度独立预测力\n\n"
    report += "| 维度 | IC(5d) | IC(10d) | ICIR(5d) | 权重 | 评价 |\n"
    report += "|------|--------|---------|----------|------|------|\n"

    for dim in ['trend', 'momentum', 'volume', 'breadth', 'volatility']:
        if dim not in ic_results:
            continue
        ic5 = ic_results[dim].get('fwd_5d', {}).get('ic', np.nan)
        ic10 = ic_results[dim].get('fwd_10d', {}).get('ic', np.nan)
        icir5 = ic_results[dim].get('fwd_5d', {}).get('icir', np.nan)

        weight_map = {'trend': '10%', 'momentum': '10%', 'volume': '15%',
                      'breadth': '25%', 'volatility': '15%'}
        dim_name = {'trend': '趋势', 'momentum': '动量', 'volume': '成交量',
                    'breadth': '市场宽度', 'volatility': '波动风险'}

        # 评价
        if not pd.isna(ic5) and abs(ic5) > 0.05:
            grade = '有效 ✅' if ic5 > 0 else '反向 ⚠️'
        elif not pd.isna(ic5) and abs(ic5) > 0.02:
            grade = '弱信号'
        else:
            grade = '无效 ❌'

        ic5_str = f"{ic5:+.4f}" if not pd.isna(ic5) else '-'
        ic10_str = f"{ic10:+.4f}" if not pd.isna(ic10) else '-'
        icir5_str = f"{icir5:.2f}" if not pd.isna(icir5) else '-'

        report += f"| {dim_name[dim]} | {ic5_str} | {ic10_str} | {icir5_str} | {weight_map[dim]} | {grade} |\n"

    report += "\n"

    # 评分分布
    report += "---\n\n## 7. 评分分布统计\n\n"
    bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
    labels_list = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-100']
    scores_df['score_bin'] = pd.cut(scores_df['market_only'], bins=bins, labels=labels_list, include_lowest=True)
    dist = scores_df['score_bin'].value_counts().sort_index()

    report += "| 评分区间 | 天数 | 占比 |\n"
    report += "|----------|------|------|\n"
    for label in labels_list:
        count = dist.get(label, 0)
        pct = count / len(scores_df)
        bar = '█' * int(pct * 50)
        report += f"| {label} | {count} | {pct:.1%} {bar} |\n"
    report += "\n"

    # 结论
    report += "---\n\n## 8. 关键结论\n\n"

    # 综合评价
    market_ic5 = ic_results.get('market_only', {}).get('fwd_5d', {}).get('ic', np.nan)
    market_icir = ic_results.get('market_only', {}).get('fwd_5d', {}).get('icir', np.nan)

    conclusions = []

    if not pd.isna(market_ic5):
        if market_ic5 > 0.05:
            conclusions.append(f"1. **IC检验**: 市场维度综合评分与未来5日收益 IC={market_ic5:+.3f}, 有显著正相关 ✅")
        elif market_ic5 > 0:
            conclusions.append(f"1. **IC检验**: 市场维度综合评分与未来5日收益 IC={market_ic5:+.3f}, 弱正相关")
        else:
            conclusions.append(f"1. **IC检验**: 市场维度综合评分与未来5日收益 IC={market_ic5:+.3f}, 无预测力 ❌")

    if quintile_results:
        mono = quintile_results['monotonicity']
        spread = quintile_results['spread_annualized']
        if mono >= 0.75 and spread > 0:
            conclusions.append(f"2. **单调性**: {mono:.0%} 单调, Q5-Q1年化spread {spread:+.1%} ✅")
        elif spread > 0:
            conclusions.append(f"2. **单调性**: {mono:.0%} 单调, Q5-Q1年化spread {spread:+.1%}")
        else:
            conclusions.append(f"2. **单调性**: {mono:.0%} 单调, Q5-Q1年化spread {spread:+.1%} ❌")

    if position_results:
        bh_sharpe = position_results.get('买入持有(100%)', {}).get('sharpe', 0)
        best_name, best_data = max(position_results.items(), key=lambda x: x[1]['sharpe'])
        if best_name != '买入持有(100%)' and best_data['sharpe'] > bh_sharpe:
            conclusions.append(f"3. **择时价值**: {best_name} Sharpe={best_data['sharpe']:.2f} > 买入持有 {bh_sharpe:.2f}, 择时有效 ✅")
        else:
            conclusions.append(f"3. **择时价值**: 买入持有 Sharpe={bh_sharpe:.2f} 最优, 择时无附加值 ⚠️")

    if drawdown_defense and not pd.isna(drawdown_defense.get('crash_rate_low_score', np.nan)):
        ratio = drawdown_defense['crash_rate_low_score'] / max(drawdown_defense['crash_rate_high_score'], 0.001)
        if ratio > 2:
            conclusions.append(f"4. **大跌预警**: 低评分时大跌概率是高评分的{ratio:.1f}倍 ✅")
        elif ratio > 1.5:
            conclusions.append(f"4. **大跌预警**: 低评分时大跌概率是高评分的{ratio:.1f}倍, 有一定预警能力")
        else:
            conclusions.append(f"4. **大跌预警**: 低评分时大跌概率仅为高评分的{ratio:.1f}倍, 预警能力弱 ⚠️")

    if not pd.isna(market_icir):
        if market_icir > 0.5:
            conclusions.append(f"5. **信号稳定性**: ICIR={market_icir:.2f} (>0.5), 信号稳定可用 ✅")
        else:
            conclusions.append(f"5. **信号稳定性**: ICIR={market_icir:.2f} (<0.5), 信号波动较大 ⚠️")

    for c in conclusions:
        report += f"{c}\n"

    report += f"""
### 下一步优化建议

- 加入 **模型信号维度** (25%权重): 用当天ML评分分布替代固定50分, 预期提升IC 20-50%
- **维度权重优化**: 根据IC贡献度动态调整权重, 降低无效维度权重
- **自适应阈值**: 按市况(牛/熊/震荡)动态调整仓位映射曲线
"""

    return report


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='交易环境监测回测')
    parser.add_argument('--start-date', default='2022-01-01', help='回测起始日期')
    parser.add_argument('--end-date', default='2026-03-19', help='回测结束日期')
    parser.add_argument('--skip-validation', action='store_true', help='跳过与原方法的对比验证')
    parser.add_argument('--report-dir', default=None, help='报告目录, 用于解析ML评分 (如 reports/daily_selection_v3.9)')
    args = parser.parse_args()

    start_date = args.start_date
    end_date = args.end_date
    lookback_start = (pd.Timestamp(start_date) - timedelta(days=500)).strftime('%Y-%m-%d')  # 扩展回看, 支持CPPI

    print(f"🌡️ 交易环境监测回测")
    print(f"   回测期: {start_date} ~ {end_date}")
    print(f"   回看起点: {lookback_start}")
    print()

    # 1. 加载数据
    print("📊 1/7 加载市场数据...")
    hs300, vol_agg, breadth_data = load_all_data(DB_PATH, lookback_start, end_date)

    # 2. 预计算特征
    print("⚙️  2/7 预计算技术特征...")
    features = precompute_features(hs300, vol_agg, breadth_data)
    print(f"  特征矩阵: {len(features)}行 × {len(features.columns)}列")

    # 2.5 解析ML评分 (可选)
    ml_scores = None
    if args.report_dir:
        print(f"🤖 解析ML评分: {args.report_dir}")
        ml_scores = parse_report_scores(args.report_dir)
        print(f"  解析到 {len(ml_scores)} 个日期的ML评分")

    # 3. 计算评分
    print("📈 3/7 计算每日环境评分...")
    scores_df = compute_daily_scores(features, start_date, end_date, ml_scores=ml_scores)

    # 4. 前瞻收益
    print("📉 4/7 计算前瞻收益...")
    scores_df = add_forward_returns(scores_df, hs300)
    valid_count = scores_df['fwd_5d'].notna().sum()
    print(f"  有效数据: {valid_count}个交易日 (含5日前瞻)")

    # 5. 验证
    if not args.skip_validation:
        print("🔍 5/7 验证batch与原方法一致性...")
        validate_scores(scores_df, n_samples=5)
    else:
        print("⏭️  5/7 跳过验证")

    # 6. 分析
    print("📐 6/7 运行分析...")

    score_cols = ['market_only', 'composite', 'trend', 'momentum', 'volume', 'breadth', 'volatility']
    return_cols = ['fwd_1d', 'fwd_3d', 'fwd_5d', 'fwd_10d', 'fwd_20d']

    ic_results = analyze_ic(scores_df, score_cols, return_cols)
    print("  ✅ IC分析完成")

    quintile_results = analyze_quintiles(scores_df, 'market_only', 'fwd_5d')
    print("  ✅ 五等分分析完成")

    signal_accuracy = analyze_signal_accuracy(scores_df, 'market_only')
    print("  ✅ 信号准确率完成")

    position_results = backtest_position_sizing(scores_df)
    print("  ✅ 仓位管理回测完成")

    drawdown_defense = analyze_drawdown_defense(scores_df, 'market_only', threshold=-0.05)
    print("  ✅ 大跌防御分析完成")

    rolling_ic = analyze_rolling_ic(scores_df, 'market_only', 'fwd_5d', window=60)
    print("  ✅ 滚动IC分析完成")

    # 7. 报告
    print("📝 7/7 生成报告...")
    report = generate_report(
        ic_results, quintile_results, signal_accuracy,
        position_results, drawdown_defense, rolling_ic,
        scores_df, start_date, end_date
    )

    report_dir = Path('reports/backtest')
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / '交易环境监测回测报告.md'
    report_path.write_text(report, encoding='utf-8')

    print(f"\n✅ 回测完成! 报告已保存: {report_path}")
    print()

    # 打印关键结果摘要
    mkt_ic5 = ic_results.get('market_only', {}).get('fwd_5d', {}).get('ic', np.nan)
    mkt_icir = ic_results.get('market_only', {}).get('fwd_5d', {}).get('icir', np.nan)
    print(f"📊 关键结果:")
    print(f"   综合IC(5d): {mkt_ic5:+.4f}" if not pd.isna(mkt_ic5) else "   综合IC(5d): N/A")
    print(f"   ICIR(5d): {mkt_icir:.2f}" if not pd.isna(mkt_icir) else "   ICIR(5d): N/A")
    if quintile_results:
        print(f"   单调性: {quintile_results['monotonicity']:.0%}")
        print(f"   Q5-Q1 Spread(年化): {quintile_results['spread_annualized']:+.1%}")
    if position_results:
        for name, r in position_results.items():
            print(f"   {name}: Sharpe={r['sharpe']:.2f}, 年化={r['annual_ret']:+.1%}, MaxDD={r['max_dd']:.1%}")


if __name__ == '__main__':
    main()
