#!/usr/bin/env python3
"""
环境评分验证 V3 — 同时评估方向准确性 + Top10预测力
目标: 市场跌时分数低, 市场涨时分数高, 且高分时Top10收益更好
"""
import sqlite3, json, os, sys, numpy as np, pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = 'data_adapter/stock_data.db'
REPORT_DIR = 'reports/daily_selection_v4.8.1'


def load_data():
    """Load everything needed"""
    # Top10 returns
    top10_by_date = {}
    for f in sorted(os.listdir(REPORT_DIR)):
        if not f.startswith('analysis_data_'): continue
        ds = f.replace('analysis_data_', '').replace('.json', '')
        date = f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}'
        with open(os.path.join(REPORT_DIR, f)) as fh:
            data = json.load(fh)
        stocks = data.get('all_stocks_with_scores', [])
        ranked = sorted(stocks, key=lambda s: -(s.get('rank_score', 0) or 0))[:10]
        codes = [s['stock_code'] for s in ranked if s.get('stock_code')]
        if codes: top10_by_date[date] = codes

    conn = sqlite3.connect(DB_PATH)
    all_dates = [r[0] for r in conn.execute('SELECT DISTINCT trade_date FROM daily_quotes ORDER BY trade_date').fetchall()]
    date_idx = {d: i for i, d in enumerate(all_dates)}

    all_codes = set()
    for codes in top10_by_date.values(): all_codes.update(codes)
    code_str = ','.join(f"'{c}'" for c in all_codes)
    min_date = min(top10_by_date.keys())
    prices = pd.read_sql(f"SELECT s.code, dq.trade_date, dq.close FROM daily_quotes dq JOIN securities s ON s.id=dq.security_id WHERE s.code IN ({code_str}) AND dq.trade_date>='{min_date}'", conn)
    price_map = defaultdict(dict)
    for _, row in prices.iterrows(): price_map[row['code']][row['trade_date']] = row['close']

    top10_ret = {}
    for date, codes in top10_by_date.items():
        if date not in date_idx: continue
        idx = date_idx[date]
        if idx + 10 >= len(all_dates): continue
        fd = all_dates[idx + 10]
        rets = [price_map[c][fd] / price_map[c][date] - 1 for c in codes if c in price_map and date in price_map[c] and fd in price_map[c]]
        if rets: top10_ret[date] = np.mean(rets)

    # HS300 daily returns (for "market direction")
    hs300 = pd.read_sql("SELECT dq.trade_date, dq.close, dq.price_change_pct FROM daily_quotes dq JOIN securities s ON s.id=dq.security_id WHERE s.code='000300.SH' AND dq.trade_date>='2023-06-01' ORDER BY dq.trade_date", conn)
    hs300['price_change_pct'] = pd.to_numeric(hs300['price_change_pct'], errors='coerce').fillna(0)

    # Breadth data
    breadth = pd.read_sql('''SELECT dq.trade_date,
        COUNT(*) as total,
        SUM(CASE WHEN dq.price_change_pct > 0 THEN 1 ELSE 0 END) as up_count
        FROM daily_quotes dq JOIN securities s ON dq.security_id=s.id
        WHERE s.type='A股' AND dq.trade_date>='2023-06-01' AND dq.volume>0
        GROUP BY dq.trade_date ORDER BY dq.trade_date''', conn)

    conn.close()
    return top10_ret, hs300, breadth


def simulate_env_score(hs300, breadth, all_stocks_scores, date, weights):
    """Simulate the full environment score for one day"""
    hs_idx = hs300[hs300['trade_date'] == date].index
    if len(hs_idx) == 0: return None

    idx = hs_idx[0]
    closes = hs300['close'].values[:idx + 1].astype(float)
    if idx < 20: return None
    scores = {}

    # MA Position (92.4% direction accuracy)
    ma5 = np.mean(closes[-5:])
    ma20 = np.mean(closes[-20:])
    s = 50
    s += np.clip((closes[-1] / ma5 - 1) * 500, -15, 15)
    s += np.clip((closes[-1] / ma20 - 1) * 300, -15, 15)
    scores['ma_position'] = max(0, min(100, s))

    # Breadth
    br_row = breadth[breadth['trade_date'] == date]
    if len(br_row) > 0:
        total = br_row['total'].iloc[0]
        up = br_row['up_count'].iloc[0]
        if total > 0:
            up_ratio = up / total
            scores['breadth'] = max(0, min(100, 50 + np.clip((up_ratio - 0.50) * 100, -25, 25)))

    # Volume (simplified)
    ret5 = closes[-1] / closes[-5] - 1
    scores['volume'] = max(0, min(100, 50 + np.clip(ret5 * 300, -20, 20)))

    # Growth value
    scores['growth_value'] = 50

    # Model signal
    if all_stocks_scores:
        top10_avg = np.mean(sorted([s.get('rank_score', 0) or 0 for s in all_stocks_scores])[-10:])
        pos_ratio = np.mean(np.array([s.get('rank_score', 0) or 0 for s in all_stocks_scores]) > 0)
        ms = 50
        if top10_avg > 0.016: ms += 20
        elif top10_avg > 0.012: ms += 12
        elif top10_avg > 0.009: ms += 5
        elif top10_avg > 0.006: ms -= 3
        else: ms -= 10
        if pos_ratio > 0.65: ms += 8
        elif pos_ratio < 0.30: ms -= 10
        scores['model_signal'] = max(0, min(100, ms))

    total = sum(scores.get(k, 50) * w for k, w in weights.items())
    return total


def main():
    print("Loading data...")
    top10_ret, hs300, breadth = load_data()
    eval_dates = sorted(top10_ret.keys())
    print(f"  {len(eval_dates)} dates with Top10 returns")

    # Load report data for model quality metrics
    report_data = {}
    for f in sorted(os.listdir(REPORT_DIR)):
        if not f.startswith('analysis_data_'): continue
        ds = f.replace('analysis_data_', '').replace('.json', '')
        date = f'{ds[:4]}-{ds[4:6]}-{ds[6:8]}'
        if date not in top10_ret: continue
        with open(os.path.join(REPORT_DIR, f)) as fh:
            report_data[date] = json.load(fh).get('all_stocks_with_scores', [])

    # Test different weight configurations
    configs = [
        ('MA方向30+成交量20+成长15+模型信号35',
         {'ma_position': 0.30, 'volume': 0.20, 'growth_value': 0.15, 'model_signal': 0.35}),
        ('MA方向25+涨跌比15+成交量15+成长10+模型信号35',
         {'ma_position': 0.25, 'breadth': 0.15, 'volume': 0.15, 'growth_value': 0.10, 'model_signal': 0.35}),
        ('MA方向35+成交量20+成长15+信号30',
         {'ma_position': 0.35, 'volume': 0.20, 'growth_value': 0.15, 'model_signal': 0.30}),
        ('MA方向20+涨跌比20+成交量15+成长10+信号35',
         {'ma_position': 0.20, 'breadth': 0.20, 'volume': 0.15, 'growth_value': 0.10, 'model_signal': 0.35}),
    ]

    # HS300 5-day return as "market direction" ground truth
    hs_closes = hs300['close'].values.astype(float)
    hs_dates = hs300['trade_date'].values
    hs_d2i = {d: i for i, d in enumerate(hs_dates)}

    print(f"\n{'='*120}")
    print(f"  环境评分 V3 验证 (方向准确性 + Top10预测力)")
    print(f"{'='*120}")

    for config_name, weights in configs:
        env_scores = {}
        for date in eval_dates:
            stocks = report_data.get(date, [])
            score = simulate_env_score(hs300, breadth, stocks, date, weights)
            if score is not None:
                env_scores[date] = score

        if len(env_scores) < 100:
            print(f"\n  {config_name}: insufficient data ({len(env_scores)})")
            continue

        dates_with_scores = sorted(env_scores.keys())
        s_arr = np.array([env_scores[d] for d in dates_with_scores])

        # 1. Direction accuracy: when market drops (HS300 5d ret < 0), score should < 50
        correct_direction = 0
        total_direction = 0
        for d in dates_with_scores:
            if d not in hs_d2i: continue
            idx = hs_d2i[d]
            if idx < 5: continue
            ret5 = hs_closes[idx] / hs_closes[idx - 5] - 1
            score = env_scores[d]
            if (ret5 > 0.01 and score > 50) or (ret5 < -0.01 and score < 50):
                correct_direction += 1
            total_direction += 1
        direction_acc = correct_direction / total_direction if total_direction > 0 else 0

        # 2. Top10 predictive power: tercile split
        p33, p67 = np.percentile(s_arr, 33), np.percentile(s_arr, 67)
        high_rets = [top10_ret[d] for d in dates_with_scores if env_scores[d] >= p67 and d in top10_ret]
        low_rets = [top10_ret[d] for d in dates_with_scores if env_scores[d] <= p33 and d in top10_ret]
        pred_diff = np.mean(high_rets) - np.mean(low_rets) if high_rets and low_rets else 0

        # 3. Distribution
        mean_s, std_s, med_s = s_arr.mean(), s_arr.std(), np.median(s_arr)

        # Composite metric: 40% direction + 40% prediction + 20% distribution
        dir_score = min(50, direction_acc * 100)  # 0-50
        pred_score = min(30, max(0, pred_diff * 1500))  # 0-30
        dist_score = 20 - min(10, abs(med_s - 50) * 0.4) - (5 if std_s < 10 else 0)  # 0-20
        composite = dir_score + pred_score + dist_score

        print(f"\n  {config_name}:")
        print(f"    方向准确率: {direction_acc:.1%} ({correct_direction}/{total_direction})")
        print(f"    Top10预测: T1={np.mean(high_rets):+.2%} T3={np.mean(low_rets):+.2%} diff={pred_diff:+.2%}")
        print(f"    分布: mean={mean_s:.0f} std={std_s:.0f} med={med_s:.0f}")
        print(f"    COMPOSITE: 方向{dir_score:.0f} + 预测{pred_score:.0f} + 分布{dist_score:.0f} = {composite:.0f}/100")


if __name__ == '__main__':
    main()
