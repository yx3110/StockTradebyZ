#!/usr/bin/env python3
"""
交易环境6维度评分质量验证脚本
输出composite metric: 各维度(分布合理度 + 区分度 + 预测力)
"""
import sqlite3, numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = 'data_adapter/stock_data.db'

def load_indices():
    conn = sqlite3.connect(DB_PATH)
    indices = {}
    for code in ['000300.SH', '000001.SH', '399001.SZ', '399006.SZ', '932000.CSI', '000985.SH']:
        df = pd.read_sql(f'''
            SELECT dq.trade_date, dq.close, dq.price_change_pct, dq.volume
            FROM daily_quotes dq JOIN securities s ON s.id=dq.security_id
            WHERE s.code="{code}" AND dq.trade_date >= "2019-06-01"
            ORDER BY dq.trade_date
        ''', conn)
        df['price_change_pct'] = pd.to_numeric(df['price_change_pct'], errors='coerce').fillna(0)
        df['close'] = pd.to_numeric(df['close'], errors='coerce').ffill()
        indices[code] = df

    # Breadth data
    breadth_df = pd.read_sql('''
        SELECT dq.trade_date,
            COUNT(*) as total,
            SUM(CASE WHEN dq.price_change_pct > 0 THEN 1 ELSE 0 END) as up_count,
            SUM(CASE WHEN dq.price_change_pct < 0 THEN 1 ELSE 0 END) as down_count,
            SUM(CASE WHEN (s.code LIKE '6%' OR s.code LIKE '0%') AND dq.price_change_pct >= 0.095 THEN 1
                     WHEN (s.code LIKE '3%' OR s.code LIKE '688%') AND dq.price_change_pct >= 0.195 THEN 1
                     WHEN (s.code LIKE '8%' OR s.code LIKE '920%') AND dq.price_change_pct >= 0.295 THEN 1
                     ELSE 0 END) as limit_up,
            SUM(CASE WHEN (s.code LIKE '6%' OR s.code LIKE '0%') AND dq.price_change_pct <= -0.095 THEN 1
                     WHEN (s.code LIKE '3%' OR s.code LIKE '688%') AND dq.price_change_pct <= -0.195 THEN 1
                     WHEN (s.code LIKE '8%' OR s.code LIKE '920%') AND dq.price_change_pct <= -0.295 THEN 1
                     ELSE 0 END) as limit_down,
            SUM(CASE WHEN dq.price_change_pct > 0.05 THEN 1 ELSE 0 END) as strong_up
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股' AND dq.trade_date >= '2019-06-01' AND dq.volume > 0
        GROUP BY dq.trade_date ORDER BY dq.trade_date
    ''', conn)

    # Volume data
    vol_df = pd.read_sql('''
        SELECT dq.trade_date, SUM(dq.volume) as total_volume
        FROM daily_quotes dq JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股' AND dq.trade_date >= '2019-06-01' AND dq.volume > 0
        GROUP BY dq.trade_date ORDER BY dq.trade_date
    ''', conn)

    # Turnover data for new factor
    turnover_df = pd.read_sql('''
        SELECT db.trade_date, AVG(db.turnover_rate) as avg_tr
        FROM daily_basic db JOIN securities s ON db.security_id=s.id
        WHERE s.type='A股' AND db.trade_date >= '2019-06-01'
          AND db.turnover_rate > 0 AND db.turnover_rate < 50
        GROUP BY db.trade_date ORDER BY db.trade_date
    ''', conn)

    conn.close()
    return indices, breadth_df, vol_df, turnover_df


def simulate_trend(indices, i):
    trend_raw = []
    for t_code, t_w in [('000300.SH', 0.5), ('000985.SH', 0.5)]:
        if t_code not in indices: continue
        tc = indices[t_code]['close'].values[:i+1].astype(float)
        if len(tc) < 20: continue
        latest = tc[-1]
        ma20 = np.mean(tc[-20:])
        ma60 = np.mean(tc[-60:]) if len(tc) >= 60 else ma20
        t_s = 50
        t_s += np.clip((latest / ma20 - 1) * 500, -15, 15)
        t_s += np.clip((latest / ma60 - 1) * 200, -10, 10)
        if len(tc) >= 25:
            ma20_5ago = np.mean(tc[-25:-5])
            t_s += np.clip((ma20 / ma20_5ago - 1) * 500, -10, 10)
        peak = np.max(tc[-60:]) if len(tc) >= 60 else np.max(tc)
        t_s += np.clip((latest / peak - 1) * 100, -15, 0)
        trend_raw.append((t_s, t_w))
    if not trend_raw: return 50
    return max(0, min(100, sum(s*w for s,w in trend_raw) / sum(w for _,w in trend_raw)))


def simulate_momentum(hs300, i):
    c = hs300['close'].values[:i+1].astype(float)
    if len(c) < 20: return 50
    s = 50
    ret_5d = c[-1] / c[-5] - 1
    ret_20d = c[-1] / c[-20] - 1
    s += np.clip(ret_5d * 500, -15, 15)
    s += np.clip(ret_20d * 125, -10, 10)
    if abs(ret_20d) > 0.001:
        accel = (ret_5d / abs(ret_20d)) - 1
        s += np.clip(accel * 5, -8, 8)
    if len(c) >= 26:
        ema12 = pd.Series(c).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(c).ewm(span=26).mean().iloc[-1]
        dif = ema12 - ema26
        dif_norm = dif / c[-1] * 10000
        s += np.clip(dif_norm * 0.3, -10, 10)
        dif_prev = pd.Series(c[:-1]).ewm(span=12).mean().iloc[-1] - pd.Series(c[:-1]).ewm(span=26).mean().iloc[-1]
        dif_delta = (dif - dif_prev) / c[-1] * 10000
        s += np.clip(dif_delta * 2, -7, 7)
    return max(0, min(100, s))


def simulate_volume(vol_df, hs300, i):
    if i < 20: return 50
    vols = vol_df['total_volume'].values[:i+1].astype(float)
    if len(vols) < 5: return 50
    v_today = vols[-1]
    v_ma5 = np.mean(vols[-5:])
    v_ma20 = np.mean(vols[-20:]) if len(vols) >= 20 else v_ma5
    vr5 = v_today / v_ma5 if v_ma5 > 0 else 1.0
    vr20 = v_today / v_ma20 if v_ma20 > 0 else 1.0
    s = 50
    s += np.clip((vr5 - 1.0) * 60, -25, 30)
    s += np.clip((vr20 - 1.0) * 30, -15, 20)
    if len(vols) >= 5:
        vt = sum(1 if vols[-j] > vols[-j-1] else -1 for j in range(1, 5))
        s += vt * 2.5
    pct_today = hs300['price_change_pct'].values[i] if i < len(hs300) else 0
    if pct_today > 0.005 and vr5 > 1.1: s += 8
    elif pct_today < -0.005 and vr5 > 1.2: s -= 8
    elif pct_today > 0.005 and vr5 < 0.85: s -= 5
    return max(0, min(100, s))


def simulate_breadth(breadth_row):
    total = breadth_row['total']
    if total == 0: return 50
    up_ratio = breadth_row['up_count'] / total
    s = 50
    if up_ratio > 0.75: s += 25
    elif up_ratio > 0.6: s += 15
    elif up_ratio > 0.45: s += 0
    elif up_ratio > 0.3: s -= 15
    else: s -= 25
    lu = breadth_row['limit_up']
    ld = breadth_row['limit_down']
    if lu > 50: s += 10
    elif lu > 20: s += 5
    if ld > 30: s -= 15
    elif ld > 10: s -= 5
    sr = breadth_row['strong_up'] / total
    if sr > 0.15: s += 10
    elif sr < 0.02: s -= 5
    return max(0, min(100, s))


def simulate_volatility(indices, i):
    idx_vols = {}
    for code in ['000300.SH', '932000.CSI', '000985.SH']:
        if code not in indices: continue
        p = indices[code]['price_change_pct'].values[:i+1].astype(float)
        if len(p) < 20: continue
        idx_vols[code] = np.std(p[-20:]) * np.sqrt(250)
    if not idx_vols: return 50
    wv = sum(idx_vols.get(c, 0) * w for c, w in [('000300.SH', 0.3), ('932000.CSI', 0.4), ('000985.SH', 0.3)])
    tw = sum(w for c, w in [('000300.SH', 0.3), ('932000.CSI', 0.4), ('000985.SH', 0.3)] if c in idx_vols)
    if tw > 0: wv /= tw
    s = 50
    if wv < 0.10: s += 22
    elif wv < 0.13: s += 12
    elif wv < 0.155: s += 2
    elif wv < 0.20: s -= 5
    elif wv < 0.265: s -= 15
    else: s -= 25
    if '000300.SH' in idx_vols and '932000.CSI' in idx_vols:
        spread = idx_vols['932000.CSI'] - idx_vols['000300.SH']
        if spread > 0.15: s -= 5
    ext_code = '000985.SH' if '000985.SH' in indices else '000300.SH'
    ep = indices[ext_code]['price_change_pct'].values[:i+1].astype(float)
    ext2 = sum(1 for x in ep[-10:] if abs(x) > 0.02)
    if ext2 >= 5: s -= 12
    elif ext2 >= 3: s -= 5
    elif ext2 == 0: s += 8
    if len(ep) >= 60:
        v20 = np.std(ep[-20:]) * np.sqrt(250)
        v60 = np.std(ep[-60:]) * np.sqrt(250)
        if v60 > 0:
            vr = v20 / v60
            if vr > 1.3: s -= 8
            elif vr < 0.75: s += 5
    p300 = indices['000300.SH']['price_change_pct'].values[:i+1].astype(float)
    cd = 0
    for x in reversed(p300):
        if x < 0: cd += 1
        else: break
    if cd >= 5: s -= 12
    elif cd >= 3: s -= 5
    return max(0, min(100, s))


def simulate_turnover_heat(turnover_df, i):
    """全市场换手率热度 (高换手=资金活跃=利好, 回测验证diff=+0.58%)"""
    if i < 20: return 50
    vals = turnover_df['avg_tr'].values[:i+1].astype(float)
    if len(vals) < 20: return 50
    z = (vals[-1] - np.mean(vals[-20:])) / max(np.std(vals[-20:]), 0.001)
    return max(0, min(100, 50 + np.clip(z * 15, -30, 30)))  # 正向: 高换手→高分


def simulate_style_momentum(indices, i):
    """大小盘风格动量 (大盘领先=利好, 小盘领先=见顶信号, 回测验证diff=+0.60%)"""
    if '000300.SH' not in indices or '932000.CSI' not in indices:
        return 50
    c300 = indices['000300.SH']['close'].values[:i+1].astype(float)
    c2000 = indices['932000.CSI']['close'].values[:i+1].astype(float)
    if len(c300) < 5 or len(c2000) < 5: return 50
    ret300 = c300[-1] / c300[-5] - 1
    ret2000 = c2000[-1] / c2000[-5] - 1
    spread = ret300 - ret2000  # 大盘-小盘, 正=大盘领先=利好
    return max(0, min(100, 50 + np.clip(spread * 500, -30, 30)))


def evaluate_dimension(name, scores, future_returns):
    """Evaluate a single dimension: distribution + uniqueness + predictive power"""
    arr = np.array(scores)
    n = len(arr)
    mean_s = arr.mean()
    std_s = arr.std()
    median_s = np.median(arr)
    unique = len(np.unique(arr))

    # Distribution score (0-30): penalize if median far from 50 or std too low/high
    dist_score = 30
    dist_score -= min(15, abs(median_s - 50) * 0.5)  # median far from 50
    if std_s < 10: dist_score -= 10
    elif std_s > 35: dist_score -= 5
    dist_score = max(0, dist_score)

    # Uniqueness score (0-20)
    uniq_score = min(20, unique)

    # Predictive power (0-50): high_score bucket should have better returns than low_score
    buckets = {'high': (70, 101), 'mid': (40, 70), 'low': (0, 40)}
    bucket_rets = {}
    for bname, (lo, hi) in buckets.items():
        rets = [future_returns[j] for j in range(len(arr))
                if arr[j] >= lo and arr[j] < hi and j < len(future_returns) and future_returns[j] is not None]
        bucket_rets[bname] = np.mean(rets) if rets else 0

    # Predictive power = high_return - low_return (should be positive)
    pred_diff = bucket_rets.get('high', 0) - bucket_rets.get('low', 0)
    pred_score = max(0, min(50, pred_diff * 5000))  # scale: 1% diff = 50 points

    total = dist_score + uniq_score + pred_score

    print(f"  {name:>12}: mean={mean_s:5.1f} std={std_s:5.1f} med={median_s:4.0f} uniq={unique:3d} | "
          f"dist={dist_score:4.1f} uniq={uniq_score:4.1f} pred={pred_score:4.1f} | "
          f"high={bucket_rets.get('high',0):+.2%} low={bucket_rets.get('low',0):+.2%} diff={pred_diff:+.2%} | "
          f"TOTAL={total:5.1f}/100")
    return total


def main():
    print("Loading data...")
    indices, breadth_df, vol_df, turnover_df = load_indices()

    hs300 = indices['000300.SH']
    dates = hs300['trade_date'].values
    closes = hs300['close'].values.astype(float)

    breadth_lookup = {row['trade_date']: row for _, row in breadth_df.iterrows()}
    turnover_lookup = {row['trade_date']: idx for idx, (_, row) in enumerate(turnover_df.iterrows())}

    START = 60

    trend_s, momentum_s, volume_s, breadth_s, volatility_s = [], [], [], [], []
    turnover_heat_s, style_mom_s = [], []
    future_rets = []

    print(f"Simulating {len(dates)-START} days...")
    for i in range(START, len(dates)):
        date = dates[i]
        trend_s.append(simulate_trend(indices, i))
        momentum_s.append(simulate_momentum(hs300, i))

        vi = vol_df[vol_df['trade_date'] == date].index
        volume_s.append(simulate_volume(vol_df, hs300, vi[0]) if len(vi) > 0 else 50)

        breadth_s.append(simulate_breadth(breadth_lookup[date]) if date in breadth_lookup else 50)
        volatility_s.append(simulate_volatility(indices, i))

        # New factors
        ti = turnover_lookup.get(date)
        turnover_heat_s.append(simulate_turnover_heat(turnover_df, ti) if ti is not None else 50)
        style_mom_s.append(simulate_style_momentum(indices, i))

        if i + 10 < len(closes):
            future_rets.append(closes[i+10] / closes[i] - 1)
        else:
            future_rets.append(None)

    print(f"\n{'='*120}")
    print(f"  交易环境8维度评分质量报告 ({len(trend_s)} 天)")
    print(f"{'='*120}")

    # 8维度权重 (新增2个有预测力因子, 替代部分趋势/动量/波动权重)
    weights = {
        'trend': 0.05, 'momentum': 0.05, 'volume': 0.18,
        'breadth': 0.22, 'volatility': 0.05,
        'turnover_heat': 0.10, 'style_momentum': 0.10,
        'model_signal': 0.25,
    }

    composite = 0
    for name, scores in [('trend', trend_s), ('momentum', momentum_s),
                          ('volume', volume_s), ('breadth', breadth_s),
                          ('volatility', volatility_s),
                          ('turnover_heat', turnover_heat_s),
                          ('style_momentum', style_mom_s)]:
        score = evaluate_dimension(name, scores, future_rets)
        composite += score * weights[name] / (1 - weights['model_signal'])

    print(f"\n  COMPOSITE METRIC (excl. model_signal): {composite:.1f}/100")
    print(f"  (目标: >60, dist>15 + uniq>15 + pred>30 per dimension)")


if __name__ == '__main__':
    main()
