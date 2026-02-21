#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3.9 大规模评分分布采样分析"""
import sys
sys.path.insert(0, '.')
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.WARNING)
import time
from datetime import datetime
import json

print('=' * 70, flush=True)
print('V3.9 大规模采样分析', flush=True)
print('=' * 70, flush=True)
print(f'启动时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

from ml_models.v39.v390_production_scorer import V390ProductionScorer
scorer = V390ProductionScorer()

conn = sqlite3.connect('data_adapter/stock_data.db')
stocks = pd.read_sql_query('''
    SELECT s.code FROM securities s
    JOIN daily_quotes dq ON s.id = dq.security_id
    WHERE s.type = 'A股' AND dq.trade_date >= '2024-01-01'
    GROUP BY s.code HAVING COUNT(*) >= 200
''', conn)
conn.close()

print(f'找到 {len(stocks)} 只A股', flush=True)

trade_dates = pd.date_range('2024-06-01', '2025-11-22', freq='B')
sample_dates = [d.strftime('%Y-%m-%d') for d in trade_dates][::5]
print(f'采样 {len(sample_dates)} 个交易日, 预计 {len(stocks)*len(sample_dates):,} 条', flush=True)

all_scores, all_codes, all_dates = [], [], []
start = time.time()
for i, code in enumerate(stocks['code']):
    if i % 100 == 0:
        elapsed = time.time() - start
        speed = len(all_scores)/elapsed if elapsed > 0 else 0
        eta = (len(stocks)-i)*len(sample_dates)/speed/60 if speed > 0 else 0
        print(f'[{datetime.now().strftime("%H:%M:%S")}] {i}/{len(stocks)} | {len(all_scores):,}条 | {speed:.0f}/秒 | ETA:{eta:.0f}分', flush=True)
    for date in sample_dates:
        try:
            r = scorer.predict_score(code, date)
            if r and 'score' in r:
                all_scores.append(r['score'])
                all_codes.append(code)
                all_dates.append(date)
        except: pass

print(f'\n完成! {len(all_scores):,}条, 耗时{(time.time()-start)/60:.1f}分钟', flush=True)

if all_scores:
    scores = np.array(all_scores)
    print(f'\n【统计】min:{scores.min():.1f} max:{scores.max():.1f} mean:{scores.mean():.1f} std:{scores.std():.1f}', flush=True)
    print('\n【分位数】', flush=True)
    for p in [1,5,10,25,50,75,90,95,99,99.5,99.9,99.99]:
        print(f'  {p}%: {np.percentile(scores,p):.2f}', flush=True)
    print('\n【高分概率】', flush=True)
    for t in [55,57,58,59,60,62,65,68,70,75,80]:
        c = np.sum(scores>=t)
        p = c/len(scores)*100
        r = f'每{int(100/p)}只中1只' if p>=1 else (f'每{int(10000/p)}只中1只' if p>=0.01 else '极罕见')
        print(f'  ≥{t}: {c:,}条 ({p:.4f}%) {r}', flush=True)
    print('\n【TOP50】', flush=True)
    df = pd.DataFrame({'code':all_codes,'date':all_dates,'score':all_scores})
    for i,r in enumerate(df.nlargest(50,'score').itertuples(),1):
        print(f'  {i}. {r.score:.2f} - {r.code} ({r.date})', flush=True)
    df.to_csv('reports/v39_score_samples.csv', index=False)
    print(f'\n样本已保存到 reports/v39_score_samples.csv', flush=True)
