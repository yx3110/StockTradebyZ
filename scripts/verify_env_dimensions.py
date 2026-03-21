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

    # Up ratio time series for breadth_momentum
    up_ratio_df = pd.read_sql('''
        SELECT dq.trade_date,
            CAST(SUM(CASE WHEN dq.price_change_pct > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as up_ratio
        FROM daily_quotes dq JOIN securities s ON dq.security_id=s.id
        WHERE s.type='A股' AND dq.trade_date >= '2019-06-01' AND dq.volume > 0
        GROUP BY dq.trade_date ORDER BY dq.trade_date
    ''', conn)

    conn.close()
    return indices, breadth_df, vol_df, up_ratio_df


def simulate_mean_reversion(indices, i):
    """20日收益率反转: 跌多→利好(均值回归), diff=+1.94%"""
    raw = []
    for code, w in [('000300.SH', 0.4), ('000985.SH', 0.3), ('932000.CSI', 0.3)]:
        if code not in indices: continue
        c = indices[code]['close'].values[:i+1].astype(float)
        if len(c) < 20: continue
        ret20 = c[-1] / c[-20] - 1
        # 反向: 20日跌幅越大→分数越高 (均值回归机会)
        s = 50 - np.clip(ret20 * 350, -35, 35)
        raw.append((s, w))
    if not raw: return 50
    return max(0, min(100, sum(s*w for s,w in raw) / sum(w for _,w in raw)))


def simulate_growth_value(indices, i):
    """创业板vs沪深300: 创业板领先=风险偏好上升=利好, diff=+1.24%"""
    if '000300.SH' not in indices or '399006.SZ' not in indices:
        return 50
    c300 = indices['000300.SH']['close'].values[:i+1].astype(float)
    cgem = indices['399006.SZ']['close'].values[:i+1].astype(float)
    if len(c300) < 5 or len(cgem) < 5: return 50
    ret300 = c300[-1] / c300[-5] - 1
    retgem = cgem[-1] / cgem[-5] - 1
    spread = retgem - ret300  # 正=创业板领先
    return max(0, min(100, 50 + np.clip(spread * 800, -35, 35)))


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


def simulate_breadth(breadth_row, prev_up_ratio=None):
    total = breadth_row['total']
    if total == 0: return 50
    up_ratio = breadth_row['up_count'] / total
    s = 50
    # 连续化: ±20%偏离50%中位映射到±25分
    s += np.clip((up_ratio - 0.50) * 100, -25, 25)
    lu = breadth_row['limit_up']
    ld = breadth_row['limit_down']
    s += np.clip((lu - 40) * 0.15, -3, 10)
    s -= np.clip((ld - 10) * 0.3, 0, 10)
    sr = breadth_row['strong_up'] / total if total > 0 else 0
    s += np.clip((sr - 0.04) * 150, -3, 8)
    # 涨家数动量 (vs 前一天, 如果可用)
    if prev_up_ratio is not None and prev_up_ratio > 0:
        delta = up_ratio - prev_up_ratio
        s += np.clip(delta * 80, -7, 7)
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


def simulate_breadth_momentum(breadth_df, i):
    """涨家数MA5变化: 市场宽度趋势改善=利好 (回测diff=+1.47%)"""
    if i < 10: return 50
    ur = breadth_df['up_ratio'].values[:i+1].astype(float)
    if len(ur) < 10: return 50
    ma5_now = np.mean(ur[max(0,len(ur)-5):])
    ma5_prev = np.mean(ur[max(0,len(ur)-10):max(0,len(ur)-5)])
    delta = ma5_now - ma5_prev
    return max(0, min(100, 50 + np.clip(delta * 350, -35, 35)))


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
    indices, breadth_df, vol_df, up_ratio_df = load_indices()

    hs300 = indices['000300.SH']
    dates = hs300['trade_date'].values
    closes = hs300['close'].values.astype(float)

    breadth_lookup = {row['trade_date']: row for _, row in breadth_df.iterrows()}
    ur_lookup = {row['trade_date']: idx for idx, (_, row) in enumerate(up_ratio_df.iterrows())}

    START = 60

    volume_s, breadth_s = [], []
    breadth_mom_s, style_mom_s = [], []
    mean_rev_s, growth_val_s = [], []
    future_rets = []

    print(f"Simulating {len(dates)-START} days...")
    for i in range(START, len(dates)):
        date = dates[i]

        vi = vol_df[vol_df['trade_date'] == date].index
        volume_s.append(simulate_volume(vol_df, hs300, vi[0]) if len(vi) > 0 else 50)

        if date in breadth_lookup:
            br_row = breadth_lookup[date]
            prev_date = dates[i-1] if i > 0 else None
            prev_ur = None
            if prev_date and prev_date in breadth_lookup:
                prev_total = breadth_lookup[prev_date]['total']
                if prev_total > 0:
                    prev_ur = breadth_lookup[prev_date]['up_count'] / prev_total
            breadth_s.append(simulate_breadth(br_row, prev_ur))
        else:
            breadth_s.append(50)

        ui = ur_lookup.get(date)
        breadth_mom_s.append(simulate_breadth_momentum(up_ratio_df, ui) if ui is not None else 50)
        style_mom_s.append(simulate_style_momentum(indices, i))
        mean_rev_s.append(simulate_mean_reversion(indices, i))
        growth_val_s.append(simulate_growth_value(indices, i))

        if i + 10 < len(closes):
            future_rets.append(closes[i+10] / closes[i] - 1)
        else:
            future_rets.append(None)

    print(f"\n{'='*120}")
    print(f"  交易环境评分质量报告 ({len(volume_s)} 天)")
    print(f"{'='*120}")

    weights = {
        'volume': 0.18, 'breadth': 0.18,
        'breadth_momentum': 0.12, 'style_momentum': 0.10,
        'mean_reversion': 0.09, 'growth_value': 0.08,
        'model_signal': 0.25,
    }

    composite = 0
    for name, scores in [('volume', volume_s), ('breadth', breadth_s),
                          ('breadth_momentum', breadth_mom_s),
                          ('style_momentum', style_mom_s),
                          ('mean_reversion', mean_rev_s),
                          ('growth_value', growth_val_s)]:
        score = evaluate_dimension(name, scores, future_rets)
        composite += score * weights[name] / (1 - weights['model_signal'])

    print(f"\n  COMPOSITE METRIC (excl. model_signal): {composite:.1f}/100")
    print(f"  (目标: 90+)")


if __name__ == '__main__':
    main()
