#!/usr/bin/env python3
"""
V4.82 高效批量报告生成 — 单次DB IO + 内存因子计算

基于 fast_batch_v481.py 扩展:
- V4.8.1 的 15 因子 (20d lookback)
- V4.8.2 的 13 因子 (260d lookback for 52w high)

用法:
    python3 scripts/fast_batch_v482.py --start-date 2024-01-01 --end-date 2026-03-20
"""
import sys, os, json, time, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_v4.8.2'


def bulk_load_all_data(dates: list):
    """单次连接，批量加载（V4.8.2 需要370天lookback for 52w high）"""
    t0 = time.time()
    conn = sqlite3.connect(str(DB_PATH))

    min_date = min(dates)
    max_date = max(dates)
    # V4.8.2 needs 370 days lookback for 52w high + residual momentum
    lookback_date = pd.Timestamp(min_date) - pd.Timedelta(days=400)
    lookback_str = lookback_date.strftime('%Y-%m-%d')
    dates_str = ','.join(f"'{d}'" for d in dates)

    print(f"[DB] 批量加载 {min_date} ~ {max_date} (lookback: {lookback_str}) ...")

    # 1. v39_feature_cache
    t1 = time.time()
    df_features = pd.read_sql(f"SELECT * FROM v39_feature_cache WHERE trade_date IN ({dates_str})", conn)
    print(f"  [1/7] features: {len(df_features)} rows ({time.time()-t1:.1f}s)")

    # 2. daily_basic (current dates + lookback for turnover)
    t1 = time.time()
    df_daily_basic = pd.read_sql(f"""
        SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm,
               db.turnover_rate, db.circ_mv, db.total_mv,
               db.turnover_rate_f, db.volume_ratio
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE s.type = 'A股'
          AND db.trade_date >= '{lookback_str}' AND db.trade_date <= '{max_date}'
    """, conn)
    print(f"  [2/7] daily_basic: {len(df_daily_basic)} rows ({time.time()-t1:.1f}s)")

    # 3. technical_indicators
    t1 = time.time()
    df_tech = pd.read_sql(f"""
        SELECT s.code, ti.trade_date, ti.rsi6, ti.rsi12, ti.rsi24,
               ti.macd_dif, ti.macd_dea, ti.macd_macd,
               ti.kdj_k, ti.kdj_d, ti.kdj_j,
               ti.cci_14, ti.atr_14,
               ti.boll_upper, ti.boll_middle, ti.boll_lower
        FROM technical_indicators ti
        JOIN securities s ON ti.security_id = s.id
        WHERE ti.trade_date IN ({dates_str}) AND s.type = 'A股'
    """, conn)
    print(f"  [3/7] tech: {len(df_tech)} rows ({time.time()-t1:.1f}s)")

    # 4. financial_indicator (for delta_roe_yoy)
    t1 = time.time()
    df_fi = pd.read_sql("""
        SELECT s.code, fi.end_date, fi.roe
        FROM financial_indicator fi
        JOIN securities s ON fi.security_id = s.id
        WHERE fi.end_date >= '20180101'
        ORDER BY s.code, fi.end_date
    """, conn)
    print(f"  [4/7] financial_indicator: {len(df_fi)} rows ({time.time()-t1:.1f}s)")

    # 5. daily_quotes OHLCV (370d lookback)
    t1 = time.time()
    df_ohlcv = pd.read_sql(f"""
        SELECT s.code, dq.trade_date, dq.open, dq.high, dq.low, dq.close,
               dq.volume, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股'
          AND dq.trade_date >= '{lookback_str}'
          AND dq.trade_date <= '{max_date}'
          AND dq.volume > 0
        ORDER BY s.code, dq.trade_date
    """, conn)
    print(f"  [5/7] OHLCV: {len(df_ohlcv)} rows ({time.time()-t1:.1f}s)")

    # 6. securities
    t1 = time.time()
    df_sec = pd.read_sql("SELECT code, name, industry FROM securities WHERE type = 'A股'", conn)
    print(f"  [6/7] securities: {len(df_sec)} rows ({time.time()-t1:.1f}s)")

    # 7. market index for residual momentum
    t1 = time.time()
    df_market = pd.read_sql(f"""
        SELECT dq.trade_date, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.code = '000300.SH'
          AND dq.trade_date >= '{lookback_str}' AND dq.trade_date <= '{max_date}'
        ORDER BY dq.trade_date
    """, conn)
    print(f"  [7/7] market_index: {len(df_market)} rows ({time.time()-t1:.1f}s)")

    conn.close()
    print(f"[DB] 完成 {time.time()-t0:.1f}s\n")

    return {
        'features': df_features,
        'daily_basic': df_daily_basic,
        'tech': df_tech,
        'fi': df_fi,
        'ohlcv': df_ohlcv,
        'securities': df_sec,
        'market': df_market,
    }


def compute_all_factors_batch(data: dict, dates: list):
    """批量计算 V4.8.1 (15因子) + V4.8.2 (13因子) = 28因子"""
    from scipy.stats import rankdata
    from ml_models.v39.v482_production_scorer import V482_NEW_FACTORS

    df_ohlcv = data['ohlcv']
    df_tech = data['tech']
    df_daily_basic = data['daily_basic']
    df_sec = data['securities']
    df_fi = data['fi']

    industry_map = dict(zip(df_sec['code'], df_sec['industry']))

    # Pre-group by code
    ohlcv_by_code = {code: grp.sort_values('trade_date') for code, grp in df_ohlcv.groupby('code')}

    # Turnover by code
    turn_by_code = {}
    for code, grp in df_daily_basic.groupby('code'):
        g = grp.sort_values('trade_date')
        turn_by_code[code] = {
            'dates': g['trade_date'].values,
            'values': pd.to_numeric(g['turnover_rate'], errors='coerce').fillna(0).values,
            'total_mv': pd.to_numeric(g['total_mv'], errors='coerce').fillna(0).values,
        }

    # Tech by date
    tech_by_date = {}
    for date, grp in df_tech.groupby('trade_date'):
        tech_by_date[date] = dict(zip(grp['code'], grp[['cci_14', 'atr_14']].to_dict('records')))

    # Financial by code
    fi_by_code = {}
    if len(df_fi) > 0:
        df_fi['end_date'] = df_fi['end_date'].str.replace('-', '')
        for code, grp in df_fi.groupby('code'):
            fi_by_code[code] = grp.sort_values('end_date').reset_index(drop=True)

    # Market daily return
    market_ret = {}
    if len(data['market']) > 0:
        for _, r in data['market'].iterrows():
            market_ret[r['trade_date']] = float(r['price_change_pct']) if pd.notna(r['price_change_pct']) else 0.0

    v481_factors_by_date = {}
    v482_factors_by_date = {}
    t0 = time.time()

    for di, date in enumerate(dates):
        if (di + 1) % 50 == 0 or di == 0:
            print(f"  [Factor] {di+1}/{len(dates)} {date} ({time.time()-t0:.0f}s)")

        v481_rows = []
        v482_rows = []
        all_atrs = {}

        tech_lookup = tech_by_date.get(date, {})

        # Per-code 5d returns for industry_adj_str
        code_ret5d = {}
        for code, grp_full in ohlcv_by_code.items():
            mask = grp_full['trade_date'] <= date
            g = grp_full[mask]
            if len(g) >= 6:
                c = g['close'].values
                code_ret5d[code] = float(c[-1] / c[-6] - 1) if c[-6] > 0 else 0.0

        # Industry median 5d return
        ind_rets = {}
        for code, ret in code_ret5d.items():
            ind = industry_map.get(code, '')
            if ind:
                ind_rets.setdefault(ind, []).append(ret)
        ind_med_ret5d = {ind: float(np.median(rets)) for ind, rets in ind_rets.items()}

        for code, grp_full in ohlcv_by_code.items():
            mask = grp_full['trade_date'] <= date
            grp = grp_full[mask]
            n = len(grp)
            if n < 5:
                continue

            c = grp['close'].values.astype(float)
            o = grp['open'].values.astype(float)
            h = grp['high'].values.astype(float)
            lo = grp['low'].values.astype(float)
            v = grp['volume'].values.astype(float)
            pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)
            dates_arr = grp['trade_date'].values

            n20 = min(20, n)
            c20, o20, h20, lo20, v20, pct20 = c[-n20:], o[-n20:], h[-n20:], lo[-n20:], v[-n20:], pct[-n20:]
            industry = industry_map.get(code, '')

            # ========== V4.8.1 factors ==========
            r481 = {'code': code}

            tech_info = tech_lookup.get(code, {})
            all_atrs[code] = tech_info.get('atr_14', 0.0) or 0.0

            total_vol = np.sum(v20)
            r481['vol_concentration'] = float(np.sum((v20/max(total_vol, 1e-8))**2)) if len(v20) >= 5 else 0.0

            safe_open = np.where(o20 > 0, o20, 1e-8)
            r481['intraday_ret_20d'] = float(np.sum((c20 - o20) / safe_open))

            r481['_mom_20d'] = float(c[-1] / c[-21] - 1) if n >= 21 else 0.0

            vwap_den = np.sum(v20)
            if vwap_den > 0 and c20[-1] > 0:
                vwap = np.sum(c20 * v20) / vwap_den
                r481['vwap_dev_20d'] = float((c20[-1] - vwap) / vwap)
            else:
                r481['vwap_dev_20d'] = 0.0

            r481['max_ret_20d'] = float(np.max(pct20))
            if len(h20) >= 5:
                log_hl = np.log(h20 / np.where(lo20 > 0, lo20, 1e-8))
                log_co = np.log(c20 / np.where(o20 > 0, o20, 1e-8))
                gk = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
                r481['gk_vol_20d'] = float(np.sqrt(np.mean(gk)))
            else:
                r481['gk_vol_20d'] = 0.0

            t_data = turn_by_code.get(code)
            if t_data is not None:
                tmask = t_data['dates'] <= date
                t_vals = t_data['values'][tmask][-20:]
                if len(t_vals) >= 5:
                    avg_t = np.mean(t_vals)
                    r481['abnormal_turnover'] = float(t_vals[-1] / max(avg_t, 1e-8))
                    r481['turnover_vol_20d'] = float(np.std(t_vals))
                else:
                    r481['abnormal_turnover'] = 0.0
                    r481['turnover_vol_20d'] = 0.0
            else:
                r481['abnormal_turnover'] = 0.0
                r481['turnover_vol_20d'] = 0.0

            if len(c20) >= 2:
                prev_c, next_o = c20[:-1], o20[1:]
                safe_prev = np.where(prev_c > 0, prev_c, 1e-8)
                r481['overnight_ret_20d'] = float(np.sum((next_o - prev_c) / safe_prev))
            else:
                r481['overnight_ret_20d'] = 0.0

            r481['cci_14'] = tech_info.get('cci_14', 0.0) or 0.0

            if len(c20) >= 5:
                std_c = np.std(c20)
                r481['squeeze_mom_calc'] = float((c20[-1] - np.mean(c20)) / max(std_c, 1e-8))
            else:
                r481['squeeze_mom_calc'] = 0.0

            if len(pct20) >= 5 and len(v20) >= 5:
                v_pct = np.diff(v20) / np.where(v20[:-1] > 0, v20[:-1], 1e-8)
                p_pct = pct20[1:]
                ml = min(len(v_pct), len(p_pct))
                if ml >= 5:
                    corr = np.corrcoef(p_pct[:ml], v_pct[:ml])[0, 1]
                    r481['vol_price_div'] = float(corr) if not np.isnan(corr) else 0.0
                else:
                    r481['vol_price_div'] = 0.0
            else:
                r481['vol_price_div'] = 0.0

            if len(pct) >= 10:
                r481['price_acceleration'] = float(np.mean(pct[-5:]) - np.mean(pct[-10:-5]))
            else:
                r481['price_acceleration'] = 0.0

            if len(c20) >= 5:
                band = np.max(h20) - np.min(lo20)
                r481['price_pos_volatility'] = float((c20[-1] - np.min(lo20)) / max(band, 1e-8))
            else:
                r481['price_pos_volatility'] = 0.5

            v481_rows.append(r481)

            # ========== V4.8.2 factors ==========
            r482 = {'code': code}

            # industry_adj_str
            stock_ret5d = code_ret5d.get(code, 0.0)
            ind_med = ind_med_ret5d.get(industry, 0.0)
            r482['industry_adj_str'] = -(stock_ret5d - ind_med)

            # turnover_reversal (rank later)
            r482['_avg_turn_20d'] = r481.get('abnormal_turnover', 0.0)

            # max5_lottery
            if len(pct20) >= 10:
                r482['max5_lottery'] = -float(np.mean(np.sort(pct20)[-5:]))
            else:
                r482['max5_lottery'] = 0.0

            # retail_crowding (rank later)
            if t_data is not None:
                tmask2 = t_data['dates'] <= date
                r482['_turnover'] = float(t_data['values'][tmask2][-1]) if np.any(tmask2) else 0.0
                r482['_mcap'] = float(t_data['total_mv'][tmask2][-1]) if np.any(tmask2) else 0.0
            else:
                r482['_turnover'] = 0.0
                r482['_mcap'] = 0.0

            # residual_momentum
            if n >= 25:
                stock_r = pct[-25:]
                mkt_r = np.array([market_ret.get(d, 0.0) for d in dates_arr[-25:]])
                mkt_var = np.var(mkt_r)
                beta = np.cov(stock_r, mkt_r)[0, 1] / max(mkt_var, 1e-12) if mkt_var > 1e-12 else 0.0
                residual = stock_r - beta * mkt_r
                r482['residual_momentum'] = float(np.sum(residual[:20]))
            else:
                r482['residual_momentum'] = 0.0

            # sumd_20d
            gains = np.sum(pct20[pct20 > 0])
            losses = np.sum(-pct20[pct20 < 0])
            r482['sumd_20d'] = float((gains - losses) / max(gains + losses, 1e-8))

            # obv_price_div
            if n >= 20:
                obv_sign = np.where(pct > 0, 1, np.where(pct < 0, -1, 0))
                obv = np.cumsum(obv_sign * v)
                obv_20ago = obv[-(min(20, n) + 1)] if n > 20 else obv[0]
                obv_ret = float(obv[-1] / max(abs(obv_20ago), 1e-8) - 1) if abs(obv_20ago) > 1e-8 else 0.0
                price_ret = float(c[-1] / c[-(min(20, n) + 1)] - 1) if n > 20 else 0.0
                r482['obv_price_div'] = obv_ret - price_ret
            else:
                r482['obv_price_div'] = 0.0

            # limit_proximity_5d
            if code.startswith('3') or code.startswith('688'):
                lim = 0.20
            elif code.startswith(('4', '8')):
                lim = 0.30
            else:
                lim = 0.10
            r482['limit_proximity_5d'] = float(np.mean(np.abs(pct[-min(5, n):]) / lim))

            # delta_roe_yoy
            fi_data = fi_by_code.get(code)
            if fi_data is not None and len(fi_data) > 0:
                date_str = date.replace('-', '')
                valid = fi_data[fi_data['end_date'] <= date_str]
                if len(valid) > 0:
                    pos = fi_data.index.get_loc(valid.index[-1])
                    if pos >= 4:
                        roe_now = fi_data.iloc[pos]['roe']
                        roe_prev = fi_data.iloc[pos - 4]['roe']
                        r482['delta_roe_yoy'] = float(roe_now - roe_prev) if pd.notna(roe_now) and pd.notna(roe_prev) else 0.0
                    else:
                        r482['delta_roe_yoy'] = 0.0
                else:
                    r482['delta_roe_yoy'] = 0.0
            else:
                r482['delta_roe_yoy'] = 0.0

            # high_52w_ratio
            n252 = min(252, n)
            r482['high_52w_ratio'] = float(c[-1] / max(np.max(h[-n252:]), 1e-8))

            # imxd_20d
            if len(h20) >= 20:
                r482['imxd_20d'] = float(np.argmax(h20) / 19.0 - np.argmin(lo20) / 19.0)
            else:
                r482['imxd_20d'] = 0.0

            # realized_skew_20d
            if len(pct20) >= 10:
                mu, sigma = np.mean(pct20), np.std(pct20)
                r482['realized_skew_20d'] = float(np.mean(((pct20 - mu) / max(sigma, 1e-8)) ** 3))
            else:
                r482['realized_skew_20d'] = 0.0

            # trend_strength_60d
            n60 = min(60, n)
            if n60 >= 20:
                ret_60d = c[-1] / c[-n60] - 1
                vol_60d = np.std(pct[-n60:])
                r482['trend_strength_60d'] = float(ret_60d / (max(vol_60d, 1e-8) * np.sqrt(n60)))
            else:
                r482['trend_strength_60d'] = 0.0

            v482_rows.append(r482)

        # Build DataFrames
        if not v481_rows:
            v481_factors_by_date[date] = pd.DataFrame()
            v482_factors_by_date[date] = pd.DataFrame()
            continue

        df481 = pd.DataFrame(v481_rows)
        df482 = pd.DataFrame(v482_rows)

        # V4.8.1 cross-sectional factors
        codes = df481['code'].tolist()
        atr_vals = np.array([all_atrs.get(c, 0.0) for c in codes])
        if len(atr_vals) > 1 and np.any(atr_vals != 0):
            df481['atr_percentile'] = (rankdata(atr_vals, method='average') - 1) / max(len(atr_vals) - 1, 1)
        else:
            df481['atr_percentile'] = 0.5

        if '_mom_20d' in df481.columns:
            df481['_industry'] = df481['code'].map(industry_map)
            ind_ranks = {}
            for _, grp in df481.groupby('_industry'):
                if len(grp) < 2:
                    for idx in grp.index:
                        ind_ranks[idx] = 0.5
                    continue
                r = rankdata(grp['_mom_20d'].values, method='average')
                nr = (r - 1) / max(len(r) - 1, 1)
                for idx, rv in zip(grp.index, nr):
                    ind_ranks[idx] = rv
            df481['industry_mom_rank'] = df481.index.map(ind_ranks).fillna(0.5)
            df481.drop(columns=['_mom_20d', '_industry'], inplace=True, errors='ignore')

        v481_cols = ['code', 'atr_percentile', 'vol_concentration', 'intraday_ret_20d',
                     'industry_mom_rank', 'vwap_dev_20d', 'max_ret_20d', 'gk_vol_20d',
                     'abnormal_turnover', 'overnight_ret_20d', 'turnover_vol_20d',
                     'cci_14', 'squeeze_mom_calc', 'vol_price_div',
                     'price_acceleration', 'price_pos_volatility']
        v481_factors_by_date[date] = df481[[c for c in v481_cols if c in df481.columns]]

        # V4.8.2 cross-sectional: turnover_reversal + retail_crowding
        if '_avg_turn_20d' in df482.columns and len(df482) > 1:
            vals = df482['_avg_turn_20d'].values
            df482['turnover_reversal'] = 1.0 - rankdata(vals, method='average') / len(vals)

        if '_turnover' in df482.columns and '_mcap' in df482.columns and len(df482) > 1:
            turn_ranks = rankdata(df482['_turnover'].values, method='average') / len(df482)
            mcap_ranks = rankdata(-df482['_mcap'].values, method='average') / len(df482)
            df482['retail_crowding'] = 1.0 - (turn_ranks * mcap_ranks)

        drop_cols = [c for c in df482.columns if c.startswith('_')]
        df482.drop(columns=drop_cols, inplace=True, errors='ignore')
        keep = ['code'] + [c for c in V482_NEW_FACTORS if c in df482.columns]
        v482_factors_by_date[date] = df482[keep]

    print(f"[Factor] V481+V482因子计算完成 ({time.time()-t0:.0f}s)\n")
    return v481_factors_by_date, v482_factors_by_date


def score_all_dates(scorer, data, v481_factors, v482_factors, dates):
    """评分：注入预计算的因子缓存"""
    from backtest.batch_generate_v395_reports import build_analysis_json

    securities_info = dict(zip(data['securities']['code'],
        data['securities'][['name', 'industry']].to_dict('records')))

    feat_by_date = {date: grp for date, grp in data['features'].groupby('trade_date')}

    # Pre-populate factor caches
    for date, df in v481_factors.items():
        scorer._v481_factor_cache[date] = df
    for date, df in v482_factors.items():
        scorer._v482_factor_cache[date] = df

    scorer._industry_cache = dict(zip(data['securities']['code'], data['securities']['industry']))
    scorer._fi_loaded = True  # prevent re-loading

    results = {}
    for i, date in enumerate(dates):
        t0 = time.time()
        features_df = feat_by_date.get(date)
        if features_df is None or len(features_df) == 0:
            continue

        all_codes = features_df['code'].tolist()
        scored = scorer.predict_scores_from_preloaded(all_codes, date, features_df.copy())

        if not scored:
            continue

        analysis = build_analysis_json(scored, date, securities_info, version='v4.8.2')

        date_str = date.replace('-', '')
        json_file = OUTPUT_DIR / f'analysis_data_{date_str}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, default=str)

        if (i + 1) % 50 == 0 or i == 0:
            n_stocks = len(scored)
            top_score = max(s['score'] for s in scored.values())
            print(f"  [{i+1}/{len(dates)}] {date}: {n_stocks}只, top={top_score:.1f}, {time.time()-t0:.1f}s")

        results[date] = True

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--end-date', default='2026-03-20')
    args = parser.parse_args()

    t_total = time.time()

    conn = sqlite3.connect(str(DB_PATH))
    dates = [r[0] for r in conn.execute(
        f"SELECT DISTINCT trade_date FROM v39_feature_cache "
        f"WHERE trade_date >= '{args.start_date}' AND trade_date <= '{args.end_date}' "
        f"ORDER BY trade_date"
    ).fetchall()]
    conn.close()

    print(f"V4.82 高效批量报告 ({args.start_date} ~ {args.end_date})")
    print(f"交易日: {len(dates)} 天")
    print(f"输出: {OUTPUT_DIR}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Skip existing
    existing = {f.stem.replace('analysis_data_', '') for f in OUTPUT_DIR.glob('analysis_data_*.json')}
    dates_todo = [d for d in dates if d.replace('-', '') not in existing]
    print(f"已有: {len(existing)}, 需生成: {len(dates_todo)}\n")

    if not dates_todo:
        print("全部已有，跳过")
        return

    # Phase 1: Bulk load
    data = bulk_load_all_data(dates_todo)

    # Phase 2: Compute factors
    v481_factors, v482_factors = compute_all_factors_batch(data, dates_todo)

    # Phase 3: Score
    print("[Model] 加载V4.82模型...")
    from ml_models.v39.v482_production_scorer import V482ProductionScorer
    scorer = V482ProductionScorer()

    print(f"\n[Score] 评分 ({len(dates_todo)} 天)...")
    results = score_all_dates(scorer, data, v481_factors, v482_factors, dates_todo)

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"完成! {len(results)}/{len(dates_todo)} 天, {total:.0f}s ({total/60:.1f}min)")
    if results:
        print(f"平均 {total/len(results):.1f}s/天")


if __name__ == '__main__':
    main()
