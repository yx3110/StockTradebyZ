#!/usr/bin/env python3
"""
V4.81 高效批量报告生成 — 单次DB IO
一次性预加载所有所需数据，零per-day查询
"""
import sys, os, json, time, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'
OUTPUT_DIR = PROJECT_ROOT / 'reports' / 'daily_selection_v4.8.1'

# ============ Phase 1: Single DB Connection, Bulk Load ============

def bulk_load_all_data(dates: list[str]):
    """一次连接，批量加载19天所有数据"""
    t0 = time.time()
    conn = sqlite3.connect(str(DB_PATH))

    min_date = min(dates)
    max_date = max(dates)
    # v481 new factors need 40 days lookback
    lookback_date = pd.Timestamp(min_date) - pd.Timedelta(days=50)
    lookback_str = lookback_date.strftime('%Y-%m-%d')
    dates_str = ','.join(f"'{d}'" for d in dates)

    print(f"[DB] 单次连接批量加载 {min_date} ~ {max_date} ...")

    # 1. v39_feature_cache (base features)
    t1 = time.time()
    df_features = pd.read_sql(f"""
        SELECT * FROM v39_feature_cache
        WHERE trade_date IN ({dates_str})
    """, conn)
    print(f"  [1/7] v39_feature_cache: {len(df_features)} rows ({time.time()-t1:.1f}s)")

    # 2. daily_basic (PE/PB/PS/turnover/market_cap + extras)
    t1 = time.time()
    df_daily_basic = pd.read_sql(f"""
        SELECT s.code, db.trade_date, db.pe_ttm, db.pb, db.ps_ttm,
               db.turnover_rate, db.circ_mv, db.total_mv,
               db.turnover_rate_f, db.volume_ratio
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE db.trade_date IN ({dates_str}) AND s.type = 'A股'
    """, conn)
    print(f"  [2/7] daily_basic: {len(df_daily_basic)} rows ({time.time()-t1:.1f}s)")

    # 3. technical_indicators (for tech features + CCI/ATR for v481)
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
    print(f"  [3/7] technical_indicators: {len(df_tech)} rows ({time.time()-t1:.1f}s)")

    # 4. (financial_indicator loaded by scorer internally, skip here)
    print(f"  [4/7] financial_indicator: scorer内部加载 (skip)")

    # 5. daily_quotes OHLCV (for v481 new factors, need 40-day lookback)
    t1 = time.time()
    df_ohlcv = pd.read_sql(f"""
        SELECT s.code, dq.trade_date, dq.open, dq.high, dq.low, dq.close,
               dq.volume, dq.price_change_pct
        FROM daily_quotes dq
        JOIN securities s ON dq.security_id = s.id
        WHERE s.type = 'A股'
          AND dq.trade_date >= '{lookback_str}'
          AND dq.trade_date <= '{max_date}'
        ORDER BY s.code, dq.trade_date
    """, conn)
    print(f"  [5/7] daily_quotes (OHLCV+lookback): {len(df_ohlcv)} rows ({time.time()-t1:.1f}s)")

    # 6. turnover from daily_basic (for v481 abnormal_turnover, need lookback)
    t1 = time.time()
    df_turnover = pd.read_sql(f"""
        SELECT s.code, db.trade_date, db.turnover_rate
        FROM daily_basic db
        JOIN securities s ON db.security_id = s.id
        WHERE s.type = 'A股'
          AND db.trade_date >= '{lookback_str}'
          AND db.trade_date <= '{max_date}'
        ORDER BY s.code, db.trade_date
    """, conn)
    print(f"  [6/7] turnover lookback: {len(df_turnover)} rows ({time.time()-t1:.1f}s)")

    # 7. securities info (industry, name)
    t1 = time.time()
    df_sec = pd.read_sql("""
        SELECT code, name, industry FROM securities WHERE type = 'A股'
    """, conn)
    print(f"  [7/7] securities: {len(df_sec)} rows ({time.time()-t1:.1f}s)")

    conn.close()
    print(f"[DB] 全部加载完成, 总耗时 {time.time()-t0:.1f}s, 连接已关闭\n")

    return {
        'features': df_features,
        'daily_basic': df_daily_basic,
        'tech': df_tech,
        'ohlcv': df_ohlcv,
        'turnover': df_turnover,
        'securities': df_sec,
    }


# ============ Phase 2: Compute V481 Factors In-Memory ============

def compute_v481_factors_batch(data: dict, dates: list[str]) -> dict:
    """批量计算所有日期的V481 15个新因子，纯内存计算"""
    from scipy.stats import rankdata

    df_ohlcv = data['ohlcv']
    df_tech = data['tech']
    df_turnover = data['turnover']
    df_sec = data['securities']
    industry_map = dict(zip(df_sec['code'], df_sec['industry']))

    # Pre-group OHLCV and turnover by code
    ohlcv_by_code = {}
    for code, grp in df_ohlcv.groupby('code'):
        ohlcv_by_code[code] = grp.sort_values('trade_date')

    turnover_by_code = {}
    for code, grp in df_turnover.groupby('code'):
        sorted_grp = grp.sort_values('trade_date')
        turnover_by_code[code] = {
            'dates': sorted_grp['trade_date'].values,
            'values': sorted_grp['turnover_rate'].values.astype(float),
        }

    # Pre-group tech by date
    tech_by_date = {}
    for date, grp in df_tech.groupby('trade_date'):
        tech_by_date[date] = dict(zip(grp['code'],
            grp[['cci_14', 'atr_14']].to_dict('records')))

    factors_by_date = {}
    t0 = time.time()

    for date in dates:
        results_list = []
        all_atrs = {}

        tech_lookup = tech_by_date.get(date, {})

        for code, grp_full in ohlcv_by_code.items():
            # Filter to <= date, last 25 rows
            mask = grp_full['trade_date'] <= date
            grp = grp_full[mask].tail(25)
            if len(grp) < 5:
                continue

            o = grp['open'].values.astype(float)
            h = grp['high'].values.astype(float)
            lo = grp['low'].values.astype(float)
            c = grp['close'].values.astype(float)
            v = grp['volume'].values.astype(float)
            pct = pd.to_numeric(grp['price_change_pct'], errors='coerce').fillna(0).values.astype(float)

            n = min(20, len(c))
            c20, o20, h20, lo20, v20, pct20 = c[-n:], o[-n:], h[-n:], lo[-n:], v[-n:], pct[-n:]

            row = {'code': code}

            # ATR
            tech_info = tech_lookup.get(code, {})
            all_atrs[code] = tech_info.get('atr_14', 0.0) or 0.0

            # vol_concentration
            total_vol = np.sum(v20)
            row['vol_concentration'] = float(np.sum((v20/total_vol)**2)) if total_vol > 0 and len(v20) >= 5 else 0.0

            # intraday_ret_20d
            safe_open = np.where(o20 > 0, o20, 1e-8)
            row['intraday_ret_20d'] = float(np.sum((c20 - o20) / safe_open))

            # momentum for industry_mom_rank
            if len(c) >= 21:
                row['_mom_20d'] = float(c[-1] / c[-21] - 1)
            elif len(c) >= 2:
                row['_mom_20d'] = float(c[-1] / c[0] - 1)
            else:
                row['_mom_20d'] = 0.0

            # vwap_dev_20d
            vwap_den = np.sum(v20)
            if vwap_den > 0 and c20[-1] > 0:
                vwap = np.sum(c20 * v20) / vwap_den
                row['vwap_dev_20d'] = float((c20[-1] - vwap) / vwap)
            else:
                row['vwap_dev_20d'] = 0.0

            # max_ret_20d
            row['max_ret_20d'] = float(np.max(pct20)) if len(pct20) > 0 else 0.0

            # gk_vol_20d
            if len(h20) >= 5:
                log_hl = np.log(h20 / np.where(lo20 > 0, lo20, 1e-8))
                log_co = np.log(c20 / np.where(o20 > 0, o20, 1e-8))
                gk = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
                row['gk_vol_20d'] = float(np.sqrt(np.mean(gk)))
            else:
                row['gk_vol_20d'] = 0.0

            # abnormal_turnover + turnover_vol_20d
            t_data = turnover_by_code.get(code)
            if t_data is not None:
                mask = t_data['dates'] <= date
                t_vals = t_data['values'][mask][-20:]
                if len(t_vals) >= 5:
                    avg_t = np.mean(t_vals)
                    row['abnormal_turnover'] = float(t_vals[-1] / avg_t) if avg_t > 0 else 0.0
                    row['turnover_vol_20d'] = float(np.std(t_vals))
                else:
                    row['abnormal_turnover'] = 0.0
                    row['turnover_vol_20d'] = 0.0
            else:
                row['abnormal_turnover'] = 0.0
                row['turnover_vol_20d'] = 0.0

            # overnight_ret_20d
            if len(c20) >= 2:
                prev_c = c20[:-1]
                next_o = o20[1:]
                safe_prev = np.where(prev_c > 0, prev_c, 1e-8)
                row['overnight_ret_20d'] = float(np.sum((next_o - prev_c) / safe_prev))
            else:
                row['overnight_ret_20d'] = 0.0

            # cci_14
            row['cci_14'] = tech_info.get('cci_14', 0.0) or 0.0

            # squeeze_mom_calc
            if len(c20) >= 5:
                std_c = np.std(c20)
                row['squeeze_mom_calc'] = float((c20[-1] - np.mean(c20)) / std_c) if std_c > 1e-8 else 0.0
            else:
                row['squeeze_mom_calc'] = 0.0

            # vol_price_div
            if len(pct20) >= 5 and len(v20) >= 5:
                v_pct = np.diff(v20) / np.where(v20[:-1] > 0, v20[:-1], 1e-8)
                p_pct = pct20[1:]
                ml = min(len(v_pct), len(p_pct))
                if ml >= 5:
                    corr = np.corrcoef(p_pct[:ml], v_pct[:ml])[0, 1]
                    row['vol_price_div'] = float(corr) if not np.isnan(corr) else 0.0
                else:
                    row['vol_price_div'] = 0.0
            else:
                row['vol_price_div'] = 0.0

            # price_acceleration
            if len(pct) >= 10:
                row['price_acceleration'] = float(np.mean(pct[-5:]) - np.mean(pct[-10:-5]))
            elif len(pct) >= 5:
                row['price_acceleration'] = float(np.mean(pct[-5:]))
            else:
                row['price_acceleration'] = 0.0

            # price_pos_volatility
            if len(c20) >= 5:
                band = np.max(h20) - np.min(lo20)
                row['price_pos_volatility'] = float((c20[-1] - np.min(lo20)) / band) if band > 1e-8 else 0.5
            else:
                row['price_pos_volatility'] = 0.5

            results_list.append(row)

        if not results_list:
            factors_by_date[date] = pd.DataFrame()
            continue

        df = pd.DataFrame(results_list)

        # Cross-sectional: atr_percentile
        codes = df['code'].tolist()
        atr_vals = np.array([all_atrs.get(c, 0.0) for c in codes])
        if len(atr_vals) > 1 and np.any(atr_vals != 0):
            ranks = rankdata(atr_vals, method='average')
            df['atr_percentile'] = (ranks - 1) / max(len(ranks) - 1, 1)
        else:
            df['atr_percentile'] = 0.5

        # Cross-sectional: industry_mom_rank
        if '_mom_20d' in df.columns:
            df['_industry'] = df['code'].map(industry_map)
            ind_ranks = {}
            for _, grp in df.groupby('_industry'):
                if len(grp) < 2:
                    for idx in grp.index:
                        ind_ranks[idx] = 0.5
                    continue
                r = rankdata(grp['_mom_20d'].values, method='average')
                nr = (r - 1) / max(len(r) - 1, 1)
                for idx, rv in zip(grp.index, nr):
                    ind_ranks[idx] = rv
            df['industry_mom_rank'] = df.index.map(ind_ranks).fillna(0.5)
            df.drop(columns=['_mom_20d', '_industry'], inplace=True, errors='ignore')

        factor_cols = ['code', 'atr_percentile', 'vol_concentration', 'intraday_ret_20d',
                       'industry_mom_rank', 'vwap_dev_20d', 'max_ret_20d', 'gk_vol_20d',
                       'abnormal_turnover', 'overnight_ret_20d', 'turnover_vol_20d',
                       'cci_14', 'squeeze_mom_calc', 'vol_price_div',
                       'price_acceleration', 'price_pos_volatility']
        factors_by_date[date] = df[[c for c in factor_cols if c in df.columns]]

    print(f"[Factor] V481因子批量计算完成 ({time.time()-t0:.1f}s)")
    return factors_by_date


# ============ Phase 3: Score with Pre-populated Caches ============

def score_all_dates(scorer, data: dict, factors_by_date: dict, dates: list[str]):
    """用预加载数据评分，scorer内部不再查DB"""
    from backtest.batch_generate_v395_reports import (
        build_analysis_json, _merge_daily_basic_features
    )

    securities_info = dict(zip(data['securities']['code'],
        data['securities'][['name', 'industry']].to_dict('records')))

    # Pre-group daily_basic by date
    db_by_date = {}
    for date, grp in data['daily_basic'].groupby('trade_date'):
        db_by_date[date] = grp

    # Pre-group features by date
    feat_by_date = {}
    for date, grp in data['features'].groupby('trade_date'):
        feat_by_date[date] = grp

    # Pre-group tech by date
    tech_by_date = {}
    for date, grp in data['tech'].groupby('trade_date'):
        tech_by_date[date] = grp

    # Pre-populate scorer's v481 factor cache to avoid DB calls
    for date, df_factors in factors_by_date.items():
        scorer._v481_factor_cache[date] = df_factors

    # Pre-populate industry cache
    scorer._industry_cache = dict(zip(data['securities']['code'], data['securities']['industry']))

    results = {}
    for i, date in enumerate(dates):
        t0 = time.time()

        features_df = feat_by_date.get(date)
        if features_df is None or len(features_df) == 0:
            print(f"  [{i+1}/{len(dates)}] {date}: 无特征数据")
            continue

        # Call scorer - v481 factors already in cache, but daily_basic/tech still queried per-day
        # We pre-populated _v481_factor_cache so that's the biggest win
        all_codes = features_df['code'].tolist()
        scored = scorer.predict_scores_from_preloaded(all_codes, date, features_df.copy())

        if not scored:
            print(f"  [{i+1}/{len(dates)}] {date}: 评分失败")
            continue

        # Build JSON
        analysis = build_analysis_json(scored, date, securities_info, version='v4.8.1')

        # Save
        date_str = date.replace('-', '')
        json_file = OUTPUT_DIR / f'analysis_data_{date_str}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, default=str)

        n_stocks = len(scored)
        top_score = max(s['score'] for s in scored.values())
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(dates)}] {date}: {n_stocks}只, top={top_score:.1f}, {elapsed:.1f}s")

        results[date] = analysis

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', default='2026-02-24')
    parser.add_argument('--end-date', default='2026-03-20')
    args = parser.parse_args()

    t_total = time.time()

    # Get trading dates
    conn = sqlite3.connect(str(DB_PATH))
    dates = [r[0] for r in conn.execute(
        f"SELECT DISTINCT trade_date FROM daily_quotes "
        f"WHERE trade_date >= '{args.start_date}' AND trade_date <= '{args.end_date}' "
        f"ORDER BY trade_date"
    ).fetchall()]
    conn.close()

    print(f"V4.81 高效批量报告 ({args.start_date} ~ {args.end_date})")
    print(f"交易日: {len(dates)} 天")
    print(f"输出: {OUTPUT_DIR}\n")

    # Phase 1: Single DB IO
    data = bulk_load_all_data(dates)

    # Phase 2: Compute V481 factors in memory
    factors = compute_v481_factors_batch(data, dates)

    # Phase 3: Load model + score
    print("[Model] 加载V4.81模型...")
    from ml_models.v39.v481_production_scorer import V481ProductionScorer
    scorer = V481ProductionScorer()

    print(f"\n[Score] 开始评分 ({len(dates)} 天)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = score_all_dates(scorer, data, factors, dates)

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"完成! {len(results)}/{len(dates)} 天, 总耗时 {total:.1f}s ({total/60:.1f}min)")
    if results:
        print(f"平均 {total/len(results):.1f}s/天")


if __name__ == '__main__':
    main()
