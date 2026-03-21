#!/usr/bin/env python3
"""
交易环境维度评分质量验证 V2
预测目标: V4.81 ML Top10选股10天收益 (而非沪深300)
"""
import sqlite3, json, os, sys, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_PATH = 'data_adapter/stock_data.db'
REPORT_DIR = 'reports/daily_selection_v4.8.1'


def load_top10_returns():
    """加载V4.81 Top10选股的10天实际收益"""
    top10_by_date = {}
    for f in sorted(os.listdir(REPORT_DIR)):
        if not f.startswith('analysis_data_'): continue
        date_str = f.replace('analysis_data_', '').replace('.json', '')
        date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
        with open(os.path.join(REPORT_DIR, f)) as fh:
            data = json.load(fh)
        stocks = data.get('all_stocks_with_scores', [])
        ranked = sorted(stocks, key=lambda s: -(s.get('rank_score', 0) or 0))[:10]
        codes = [s['stock_code'] for s in ranked if s.get('stock_code')]
        if codes:
            top10_by_date[date] = codes

    conn = sqlite3.connect(DB_PATH)
    all_dates = [r[0] for r in conn.execute(
        'SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date').fetchall()]
    date_idx = {d: i for i, d in enumerate(all_dates)}

    all_codes = set()
    for codes in top10_by_date.values():
        all_codes.update(codes)

    code_str = ','.join(f"'{c}'" for c in all_codes)
    min_date = min(top10_by_date.keys())
    prices_df = pd.read_sql(f'''
        SELECT s.code, dq.trade_date, dq.close
        FROM daily_quotes dq JOIN securities s ON s.id=dq.security_id
        WHERE s.code IN ({code_str}) AND dq.trade_date >= '{min_date}'
    ''', conn)
    conn.close()

    price_map = defaultdict(dict)
    for _, row in prices_df.iterrows():
        price_map[row['code']][row['trade_date']] = row['close']

    top10_returns = {}
    for date, codes in top10_by_date.items():
        if date not in date_idx: continue
        idx = date_idx[date]
        if idx + 10 >= len(all_dates): continue
        future_date = all_dates[idx + 10]
        rets = []
        for code in codes:
            bp = price_map.get(code, {}).get(date)
            sp = price_map.get(code, {}).get(future_date)
            if bp and sp and bp > 0:
                rets.append(sp / bp - 1)
        if rets:
            top10_returns[date] = np.mean(rets)

    return top10_returns


def load_all_data():
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

    vol_df = pd.read_sql('''
        SELECT dq.trade_date, SUM(dq.volume) as total_volume
        FROM daily_quotes dq JOIN securities s ON dq.security_id=s.id
        WHERE s.type='A股' AND dq.trade_date >= '2019-06-01' AND dq.volume > 0
        GROUP BY dq.trade_date ORDER BY dq.trade_date
    ''', conn)

    # Use 2023+ for breadth to avoid old data quality issues
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
            SUM(CASE WHEN dq.price_change_pct > 0.05 THEN 1 ELSE 0 END) as strong_up,
            AVG(dq.price_change_pct) as avg_ret
        FROM daily_quotes dq JOIN securities s ON dq.security_id=s.id
        WHERE s.type='A股' AND dq.trade_date >= '2023-06-01' AND dq.volume > 0
        GROUP BY dq.trade_date ORDER BY dq.trade_date
    ''', conn)

    up_ratio_df = pd.read_sql('''
        SELECT dq.trade_date,
            CAST(SUM(CASE WHEN dq.price_change_pct > 0 THEN 1 ELSE 0 END) AS REAL) / COUNT(*) as up_ratio
        FROM daily_quotes dq JOIN securities s ON dq.security_id=s.id
        WHERE s.type='A股' AND dq.trade_date >= '2023-06-01' AND dq.volume > 0
        GROUP BY dq.trade_date ORDER BY dq.trade_date
    ''', conn)

    conn.close()
    return indices, vol_df, breadth_df, up_ratio_df


def evaluate_dimension(name, date_scores, top10_returns, weight):
    """Evaluate dimension against Top10 returns. Wider buckets for 522-day sample."""
    dates = sorted(date_scores.keys())
    scores = np.array([date_scores[d] for d in dates])

    mean_s, std_s = scores.mean(), scores.std()
    median_s = np.median(scores)
    unique = len(np.unique(np.round(scores, 1)))

    # Distribution (0-30)
    dist = 30
    dist -= min(15, abs(median_s - 50) * 0.5)
    if std_s < 10: dist -= 10
    elif std_s > 35: dist -= 5
    dist = max(0, dist)

    # Uniqueness (0-20)
    uniq = min(20, unique)

    # Predictive power vs Top10 returns (0-50)
    # Use wider buckets: top tercile vs bottom tercile
    p33 = np.percentile(scores, 33)
    p67 = np.percentile(scores, 67)
    high_rets, low_rets = [], []
    for d in dates:
        if d not in top10_returns: continue
        ret = top10_returns[d]
        if date_scores[d] >= p67:
            high_rets.append(ret)
        elif date_scores[d] <= p33:
            low_rets.append(ret)

    if high_rets and low_rets:
        diff = np.mean(high_rets) - np.mean(low_rets)
        pred = max(0, min(50, diff * 2500))  # 2% diff = 50 points
    else:
        diff = 0
        pred = 0

    total = dist + uniq + pred

    print(f"  {name:>18} ({weight:.0%}): mean={mean_s:5.1f} std={std_s:5.1f} med={median_s:4.0f} uniq={unique:>4} | "
          f"dist={dist:4.1f} uniq={uniq:4.1f} pred={pred:4.1f} | "
          f"T1={np.mean(high_rets):+.2%}(N={len(high_rets)}) T3={np.mean(low_rets):+.2%}(N={len(low_rets)}) diff={diff:+.2%} | "
          f"TOTAL={total:5.1f}")
    return total


# ============ Factor Functions ============

def calc_volume(vol_df, hs300, dates):
    vols = vol_df.set_index('trade_date')['total_volume'].to_dict()
    hs_pct = hs300.set_index('trade_date')['price_change_pct'].to_dict()
    vol_dates = sorted(vols.keys())
    out = {}
    for date in dates:
        if date not in vols: continue
        i = vol_dates.index(date) if date in vol_dates else -1
        if i < 20: continue
        v = [vols[vol_dates[j]] for j in range(max(0, i-20), i+1)]
        vr5 = v[-1] / np.mean(v[-5:]) if len(v) >= 5 else 1.0
        vr20 = v[-1] / np.mean(v[-20:]) if len(v) >= 20 else 1.0
        s = 50
        s += np.clip((vr5 - 1.0) * 60, -25, 30)
        s += np.clip((vr20 - 1.0) * 30, -15, 20)
        if len(v) >= 5:
            vt = sum(1 if v[-j] > v[-j-1] else -1 for j in range(1, min(5, len(v))))
            s += vt * 2.5
        pct = hs_pct.get(date, 0)
        if pct > 0.005 and vr5 > 1.1: s += 8
        elif pct < -0.005 and vr5 > 1.2: s -= 8
        elif pct > 0.005 and vr5 < 0.85: s -= 5
        out[date] = max(0, min(100, s))
    return out


def calc_breadth(breadth_df, dates):
    bl = {r['trade_date']: r for _, r in breadth_df.iterrows()}
    prev_ur = None
    out = {}
    for date in dates:
        if date not in bl:
            prev_ur = None
            continue
        r = bl[date]
        total = r['total']
        if total == 0:
            prev_ur = None
            continue
        up_ratio = r['up_count'] / total
        s = 50
        s += np.clip((up_ratio - 0.50) * 100, -25, 25)
        s += np.clip((r['limit_up'] - 40) * 0.15, -3, 10)
        s -= np.clip((r['limit_down'] - 10) * 0.3, 0, 10)
        sr = r['strong_up'] / total
        s += np.clip((sr - 0.04) * 150, -3, 8)
        if prev_ur is not None:
            s += np.clip((up_ratio - prev_ur) * 80, -7, 7)
        out[date] = max(0, min(100, s))
        prev_ur = up_ratio
    return out


def calc_breadth_momentum(up_ratio_df, dates):
    ur = up_ratio_df.set_index('trade_date')['up_ratio'].to_dict()
    ur_dates = sorted(ur.keys())
    out = {}
    for date in dates:
        if date not in ur: continue
        i = ur_dates.index(date) if date in ur_dates else -1
        if i < 10: continue
        vals = [ur[ur_dates[j]] for j in range(max(0, i-10), i+1)]
        if len(vals) < 8: continue
        ma5_now = np.mean(vals[-5:])
        ma5_prev = np.mean(vals[-10:-5])
        delta = ma5_now - ma5_prev
        out[date] = max(0, min(100, 50 + np.clip(delta * 350, -35, 35)))
    return out


def calc_style_momentum(indices, dates):
    if '000300.SH' not in indices or '932000.CSI' not in indices:
        return {}
    c300 = indices['000300.SH'].set_index('trade_date')['close'].to_dict()
    c2000 = indices['932000.CSI'].set_index('trade_date')['close'].to_dict()
    d300 = sorted(c300.keys())
    out = {}
    for date in dates:
        i = d300.index(date) if date in d300 else -1
        if i < 5: continue
        r300 = c300[date] / c300[d300[i-5]] - 1
        if date in c2000 and d300[i-5] in c2000:
            r2000 = c2000[date] / c2000[d300[i-5]] - 1
            spread = r300 - r2000
            out[date] = max(0, min(100, 50 + np.clip(spread * 500, -30, 30)))
    return out


def calc_mean_reversion(indices, dates):
    out = {}
    for date in dates:
        raw = []
        for code, w in [('000300.SH', 0.4), ('000985.SH', 0.3), ('932000.CSI', 0.3)]:
            if code not in indices: continue
            df = indices[code]
            d_list = df['trade_date'].values
            c = df['close'].values.astype(float)
            if date not in d_list: continue
            idx = list(d_list).index(date)
            if idx < 20: continue
            ret20 = c[idx] / c[idx-20] - 1
            raw.append((50 - np.clip(ret20 * 350, -35, 35), w))
        if raw:
            out[date] = max(0, min(100, sum(s*w for s, w in raw) / sum(w for _, w in raw)))
    return out


def calc_growth_value(indices, dates):
    if '000300.SH' not in indices or '399006.SZ' not in indices:
        return {}
    c300 = indices['000300.SH'].set_index('trade_date')['close'].to_dict()
    cgem = indices['399006.SZ'].set_index('trade_date')['close'].to_dict()
    d300 = sorted(c300.keys())
    out = {}
    for date in dates:
        i = d300.index(date) if date in d300 else -1
        if i < 5: continue
        r300 = c300[date] / c300[d300[i-5]] - 1
        if date in cgem and d300[i-5] in cgem:
            rgem = cgem[date] / cgem[d300[i-5]] - 1
            spread = rgem - r300
            out[date] = max(0, min(100, 50 + np.clip(spread * 800, -35, 35)))
    return out


def calc_short_reversal(indices, dates):
    """5日中证全指收益反转: 近5天跌→ML选股alpha高 (均值回归)"""
    code = '000985.SH'
    if code not in indices: return {}
    df = indices[code]
    d_idx = df.set_index('trade_date')['close'].to_dict()
    d_list = sorted(d_idx.keys())
    d2i = {d: i for i, d in enumerate(d_list)}
    out = {}
    for date in dates:
        if date not in d2i: continue
        i = d2i[date]
        if i < 5: continue
        ret5 = d_idx[date] / d_idx[d_list[i-5]] - 1
        out[date] = max(0, min(100, 50 - np.clip(ret5 * 600, -35, 35)))
    return out


def main():
    print("Loading Top10 returns...")
    top10_returns = load_top10_returns()
    print(f"  {len(top10_returns)} days, avg={np.mean(list(top10_returns.values())):+.2%}")

    print("Loading market data...")
    indices, vol_df, breadth_df, up_ratio_df = load_all_data()

    # Only evaluate on dates where we have Top10 returns
    eval_dates = sorted(top10_returns.keys())
    print(f"  Evaluating on {len(eval_dates)} dates")

    # Top10 optimized weights (drop breadth/breadth_momentum/style_momentum)
    weights = {
        'volume': 0.22, 'growth_value': 0.18,
        'short_reversal': 0.20, 'mean_reversion': 0.15,
    }
    model_signal_weight = 0.25

    print(f"\n{'='*130}")
    print(f"  交易环境维度评分 V2 — 预测目标: ML Top10选股10天收益 ({len(eval_dates)}天)")
    print(f"{'='*130}")

    dims = {
        'volume': calc_volume(vol_df, indices['000300.SH'], eval_dates),
        'growth_value': calc_growth_value(indices, eval_dates),
        'short_reversal': calc_short_reversal(indices, eval_dates),
        'mean_reversion': calc_mean_reversion(indices, eval_dates),
    }

    composite = 0
    for name, scores_dict in dims.items():
        w = weights[name]
        score = evaluate_dimension(name, scores_dict, top10_returns, w)
        composite += score * w / (1 - model_signal_weight)

    print(f"\n  COMPOSITE (vs Top10 returns): {composite:.1f}/100")


if __name__ == '__main__':
    main()
