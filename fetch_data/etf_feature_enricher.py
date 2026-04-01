#!/usr/bin/env python3
"""
ETF 特征补充器: 为 v39_feature_cache 中的 ETF 补充缺失特征

补充内容:
1. sw_l1_code: 从ETF名称/benchmark映射到申万一级行业
2. 行业特征: industry_return_5d/20d, industry_relative_strength, industry_breadth 等
3. pe/pb估值: 从跟踪指数的 index_dailybasic 获取
4. pe/pb/ps_industry_rank: 在所属行业内排名
5. turnover_rate: 从 fund_share 计算
6. log_market_cap: 从 fund_share × close 计算

用法:
    python3 fetch_data/etf_feature_enricher.py [--start-date 2020-01-01] [--end-date 2026-03-23]
"""

import sys
import json
import sqlite3
import re
import logging
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / 'data_adapter' / 'stock_data.db'

# ============================================================
# Step 1: ETF → 申万一级行业映射
# ============================================================

# 关键词 → 申万一级行业
KEYWORD_TO_SW_L1 = {
    # 金融
    '银行': '银行', '证券': '非银金融', '保险': '非银金融', '金融': '非银金融', '券商': '非银金融',
    # 医药
    '医药': '医药生物', '医疗': '医药生物', '生物': '医药生物', '创新药': '医药生物', '疫苗': '医药生物',
    # 消费
    '白酒': '食品饮料', '食品': '食品饮料', '饮料': '食品饮料', '酒': '食品饮料',
    '消费': '食品饮料',  # 大多数消费ETF跟踪食品饮料为主
    '家电': '家用电器', '家居': '家用电器',
    '美容': '美容护理', '护理': '美容护理',
    '服装': '纺织服饰', '纺织': '纺织服饰',
    '农业': '农林牧渔', '畜牧': '农林牧渔', '养殖': '农林牧渔',
    '旅游': '社会服务', '酒店': '社会服务',
    '商贸': '商贸零售', '零售': '商贸零售',
    # 科技
    '半导体': '电子', '芯片': '电子', '电子': '电子', '集成电路': '电子',
    '计算机': '计算机', '软件': '计算机', '人工智能': '计算机', 'AI': '计算机',
    '云计算': '计算机', '大数据': '计算机', '信息技术': '计算机', '信息安全': '计算机',
    '通信': '通信', '5G': '通信',
    '传媒': '传媒', '游戏': '传媒', '影视': '传媒', '动漫': '传媒',
    # 制造
    '新能源': '电力设备', '光伏': '电力设备', '锂电': '电力设备', '储能': '电力设备',
    '电池': '电力设备', '风电': '电力设备', '充电': '电力设备',
    '军工': '国防军工', '国防': '国防军工', '航天': '国防军工', '航空航天': '国防军工',
    '卫星': '国防军工',
    '机械': '机械设备', '机器人': '机械设备', '工程机械': '机械设备', '工业母机': '机械设备',
    '汽车': '汽车', '智能驾驶': '汽车', '无人驾驶': '汽车',
    '电力': '公用事业', '水务': '公用事业', '燃气': '公用事业', '公用': '公用事业',
    '环保': '环保',
    # 资源
    '有色': '有色金属', '黄金': '有色金属', '白银': '有色金属', '铜': '有色金属',
    '稀土': '有色金属', '矿业': '有色金属',
    '钢铁': '钢铁', '煤炭': '煤炭', '石油': '石油石化', '化工': '基础化工',
    # 建筑
    '建筑': '建筑装饰', '建材': '建筑材料', '水泥': '建筑材料',
    '地产': '房地产', '房地': '房地产',
    # 交通
    '交运': '交通运输', '物流': '交通运输', '港口': '交通运输', '航运': '交通运输',
    '铁路': '交通运输', '高铁': '交通运输',
    # 轻工
    '轻工': '轻工制造', '造纸': '轻工制造', '包装': '轻工制造',
}

# benchmark 中的指数名称 → 申万一级行业
BENCHMARK_TO_SW_L1 = {
    '银行': '银行', '证券': '非银金融', '保险': '非银金融',
    '医药': '医药生物', '医疗': '医药生物', '生物科技': '医药生物',
    '食品': '食品饮料', '白酒': '食品饮料', '饮料': '食品饮料',
    '消费': '食品饮料',
    '半导体': '电子', '芯片': '电子', '电子': '电子',
    '计算机': '计算机', '软件': '计算机', '人工智能': '计算机', '信息技术': '计算机',
    '通信': '通信', '传媒': '传媒',
    '新能源': '电力设备', '光伏': '电力设备', '锂电': '电力设备', '电力设备': '电力设备',
    '军工': '国防军工', '国防': '国防军工', '航天': '国防军工', '航空航天': '国防军工',
    '机械': '机械设备', '机器人': '机械设备',
    '汽车': '汽车',
    '有色': '有色金属', '黄金': '有色金属', '稀土': '有色金属',
    '钢铁': '钢铁', '煤炭': '煤炭', '石油': '石油石化', '化工': '基础化工',
    '建筑': '建筑装饰', '建材': '建筑材料',
    '房地产': '房地产', '地产': '房地产',
    '家电': '家用电器', '家用电器': '家用电器',
    '农': '农林牧渔', '畜牧': '农林牧渔',
    '环保': '环保', '公用事业': '公用事业', '电力': '公用事业',
    '交通运输': '交通运输', '物流': '交通运输',
    '纺织': '纺织服饰',
}


def build_etf_industry_mapping() -> Dict[str, str]:
    """构建 ETF code(无后缀) → 申万一级行业名称 映射"""
    import tushare as ts
    with open(PROJECT_ROOT / 'config.json') as f:
        cfg = json.load(f)
    pro = ts.pro_api(cfg['tushare']['token'])

    df = pro.fund_basic(market='E', status='L')
    logger.info(f"  fund_basic 获取 {len(df)} 只ETF")

    # 只处理股票型ETF
    df = df[df['fund_type'].isin(['股票型'])].copy()
    logger.info(f"  股票型ETF: {len(df)} 只")

    mapping = {}  # code(无后缀) -> l1_name

    for _, row in df.iterrows():
        ts_code = row['ts_code']  # e.g. 510300.SH
        code = ts_code.split('.')[0]  # 510300
        name = row.get('name', '') or ''
        bench = row.get('benchmark', '') or ''

        # 先用ETF名称匹配
        matched = None
        for kw, industry in KEYWORD_TO_SW_L1.items():
            if kw in name:
                matched = industry
                break

        # 名称没匹配上，用benchmark匹配
        if not matched:
            for kw, industry in BENCHMARK_TO_SW_L1.items():
                if kw in bench:
                    matched = industry
                    break

        if matched:
            mapping[code] = matched

    logger.info(f"  行业映射完成: {len(mapping)}/{len(df)} 只ETF ({len(mapping)/len(df)*100:.0f}%)")
    return mapping


def get_sw_l1_encoding(conn: sqlite3.Connection) -> Dict[str, int]:
    """获取 l1_name → sw_l1_code 编码 (与 feature_cache_updater 一致)"""
    cursor = conn.execute(
        "SELECT DISTINCT l1_name FROM sw_industry WHERE is_new = 'Y' ORDER BY l1_name")
    names = [row[0] for row in cursor.fetchall()]
    return {name: i for i, name in enumerate(names)}


# ============================================================
# Step 2: 抓取ETF估值/份额数据
# ============================================================

def fetch_etf_fundamentals(etf_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
    """从 Tushare 抓取 ETF 份额 + 日线数据，计算 turnover_rate 和 market_cap"""
    import tushare as ts
    with open(PROJECT_ROOT / 'config.json') as f:
        cfg = json.load(f)
    pro = ts.pro_api(cfg['tushare']['token'])

    start_fmt = start_date.replace('-', '')
    end_fmt = end_date.replace('-', '')

    all_parts = []
    total = len(etf_codes)

    for i, code in enumerate(etf_codes):
        ts_code = f"{code}.SH" if code.startswith(('5', '6')) else f"{code}.SZ"

        try:
            # fund_share: 份额
            df_share = pro.fund_share(ts_code=ts_code, start_date=start_fmt, end_date=end_fmt)
            if df_share.empty:
                continue

            # fund_daily: 日线 (含vol/amount)
            df_daily = pro.fund_daily(ts_code=ts_code, start_date=start_fmt, end_date=end_fmt)
            if df_daily.empty:
                continue

            # 合并
            df_share['trade_date'] = df_share['trade_date'].astype(str)
            df_daily['trade_date'] = df_daily['trade_date'].astype(str)

            merged = df_daily.merge(df_share[['trade_date', 'fd_share']], on='trade_date', how='left')
            merged['fd_share'] = merged['fd_share'].ffill()  # 份额不是每天更新

            # 计算
            merged['code'] = code
            merged['trade_date_fmt'] = merged['trade_date'].apply(
                lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")
            merged['turnover_rate'] = np.where(
                merged['fd_share'] > 0,
                merged['vol'] / merged['fd_share'] * 100,  # 百分比
                0.0)
            merged['market_cap'] = merged['close'] * merged['fd_share'] / 10000  # 万元→亿元
            merged['log_market_cap'] = np.log1p(merged['market_cap'])

            all_parts.append(merged[['code', 'trade_date_fmt', 'close',
                                      'turnover_rate', 'market_cap', 'log_market_cap']]
                             .rename(columns={'trade_date_fmt': 'trade_date'}))

        except Exception as e:
            if 'freq' in str(e).lower() or '每分钟' in str(e):
                time.sleep(1)
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"    fund数据: {i+1}/{total}")
            time.sleep(0.5)  # rate limit

    if not all_parts:
        return pd.DataFrame()

    result = pd.concat(all_parts, ignore_index=True)
    logger.info(f"  ETF基本面数据: {len(result):,} 条, {result['code'].nunique()} 只ETF")
    return result


def fetch_index_valuation(start_date: str, end_date: str) -> pd.DataFrame:
    """从 Tushare 抓取主要指数的 PE/PB 数据"""
    import tushare as ts
    with open(PROJECT_ROOT / 'config.json') as f:
        cfg = json.load(f)
    pro = ts.pro_api(cfg['tushare']['token'])

    # 申万行业指数代码 (用于估值)
    # 但更实际的是用主要宽基指数
    indices = {
        '000300.SH': '沪深300',
        '000905.SH': '中证500',
        '000852.SH': '中证1000',
        '399006.SZ': '创业板指',
        '000688.SH': '科创50',
    }

    start_fmt = start_date.replace('-', '')
    end_fmt = end_date.replace('-', '')

    parts = []
    for idx_code, name in indices.items():
        try:
            df = pro.index_dailybasic(ts_code=idx_code, start_date=start_fmt, end_date=end_fmt,
                                       fields='ts_code,trade_date,pe,pe_ttm,pb,turnover_rate,total_mv')
            if not df.empty:
                df['trade_date'] = df['trade_date'].apply(
                    lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")
                parts.append(df)
                time.sleep(0.3)
        except Exception as e:
            logger.warning(f"  index_dailybasic {idx_code} 失败: {e}")

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    logger.info(f"  指数估值数据: {len(result):,} 条")
    return result


# ============================================================
# Step 3: 更新 v39_feature_cache
# ============================================================

def load_industry_daily_stats(conn: sqlite3.Connection) -> Dict[str, Dict]:
    """加载 A股 行业日统计数据 (用于给 ETF 填充行业特征)

    Returns: {(trade_date, l1_name): {industry_return_5d, industry_return_20d, ...}}
    """
    logger.info("  加载A股行业日统计...")

    # 从已有A股的feature_cache中提取行业统计
    query = """
    SELECT v.trade_date,
           json_extract(v.features_json, '$.sw_l1_code') as sw_code,
           AVG(json_extract(v.features_json, '$.industry_return_5d')) as ind_ret5,
           AVG(json_extract(v.features_json, '$.industry_return_20d')) as ind_ret20,
           AVG(json_extract(v.features_json, '$.industry_breadth')) as ind_breadth,
           AVG(json_extract(v.features_json, '$.industry_volume_change')) as ind_vol,
           AVG(json_extract(v.features_json, '$.industry_limit_up_ratio')) as ind_limit,
           AVG(json_extract(v.features_json, '$.industry_kdj_avg')) as ind_kdj,
           AVG(json_extract(v.features_json, '$.industry_macd_bullish_pct')) as ind_macd,
           AVG(json_extract(v.features_json, '$.industry_concentration')) as ind_conc
    FROM v39_feature_cache v
    JOIN securities s ON v.code = s.code
    WHERE s.type = 'A股' AND json_extract(v.features_json, '$.sw_l1_code') >= 0
    GROUP BY v.trade_date, sw_code
    """

    df = pd.read_sql(query, conn)
    logger.info(f"    行业统计: {len(df):,} 条")

    stats = {}
    for _, row in df.iterrows():
        key = (row['trade_date'], int(row['sw_code']))
        stats[key] = {
            'industry_return_5d': row['ind_ret5'] or 0,
            'industry_return_20d': row['ind_ret20'] or 0,
            'industry_breadth': row['ind_breadth'] or 0.5,
            'industry_volume_change': row['ind_vol'] or 1.0,
            'industry_limit_up_ratio': row['ind_limit'] or 0,
            'industry_kdj_avg': row['ind_kdj'] or 50,
            'industry_macd_bullish_pct': row['ind_macd'] or 0.5,
            'industry_concentration': row['ind_conc'] or 0.03,
        }
    return stats


def load_industry_valuation_ranks(conn: sqlite3.Connection) -> Dict:
    """加载 A股 行业内估值排名分布 (用于给 ETF 分配合理的 rank)

    Returns: {(trade_date, sw_code): {pe_median, pb_median, ps_median}}
    """
    logger.info("  加载行业估值中位数...")
    query = """
    SELECT v.trade_date,
           json_extract(v.features_json, '$.sw_l1_code') as sw_code,
           AVG(json_extract(v.features_json, '$.pe_industry_rank')) as pe_rank,
           AVG(json_extract(v.features_json, '$.pb_industry_rank')) as pb_rank,
           AVG(json_extract(v.features_json, '$.ps_industry_rank')) as ps_rank
    FROM v39_feature_cache v
    JOIN securities s ON v.code = s.code
    WHERE s.type = 'A股' AND json_extract(v.features_json, '$.sw_l1_code') >= 0
    GROUP BY v.trade_date, sw_code
    """
    df = pd.read_sql(query, conn)
    result = {}
    for _, row in df.iterrows():
        key = (row['trade_date'], int(row['sw_code']))
        result[key] = {
            'pe_industry_rank': row['pe_rank'] or 0.5,
            'pb_industry_rank': row['pb_rank'] or 0.5,
            'ps_industry_rank': row['ps_rank'] or 0.5,
        }
    return result


def update_etf_features(etf_industry_map: Dict[str, str],
                         etf_fundamentals: pd.DataFrame,
                         start_date: str, end_date: str):
    """更新 v39_feature_cache 中 ETF 的 features_json"""

    conn = sqlite3.connect(str(DB_PATH), timeout=30)

    # 获取 sw_l1 编码
    sw_encoding = get_sw_l1_encoding(conn)
    logger.info(f"  申万编码: {len(sw_encoding)} 个行业")

    # 加载行业统计和估值
    industry_stats = load_industry_daily_stats(conn)
    industry_vals = load_industry_valuation_ranks(conn)

    # 构建 ETF code → sw_l1_code 映射
    etf_sw_code = {}
    for code, l1_name in etf_industry_map.items():
        if l1_name in sw_encoding:
            etf_sw_code[code] = sw_encoding[l1_name]

    logger.info(f"  ETF sw_l1_code 映射: {len(etf_sw_code)} 只")

    # 构建 fundamentals 查找表
    fund_lookup = {}
    if not etf_fundamentals.empty:
        for _, row in etf_fundamentals.iterrows():
            fund_lookup[(row['code'], row['trade_date'])] = {
                'turnover_rate': row.get('turnover_rate', 0),
                'log_market_cap': row.get('log_market_cap', 0),
            }

    # 批量更新 ETF features_json
    cursor = conn.execute("""
        SELECT v.id, v.code, v.trade_date, v.features_json
        FROM v39_feature_cache v
        JOIN securities s ON v.code = s.code
        WHERE s.type = 'ETF_基金'
          AND v.trade_date >= ? AND v.trade_date <= ?
    """, (start_date, end_date))

    batch = []
    updated = 0
    total = 0

    for row_id, code, trade_date, fj in cursor:
        total += 1
        try:
            features = json.loads(fj)
        except Exception:
            continue

        changed = False
        sw_code = etf_sw_code.get(code)

        # 1. 更新 sw_l1_code
        if sw_code is not None and features.get('sw_l1_code', -1) == -1:
            features['sw_l1_code'] = sw_code
            changed = True

            # 2. 更新行业特征
            key = (trade_date, sw_code)
            stats = industry_stats.get(key)
            if stats:
                for feat_name, value in stats.items():
                    features[feat_name] = value
                changed = True

                # industry_relative_strength = 个股5d收益 - 行业5d收益
                ret_5d = features.get('return_5d', 0)
                features['industry_relative_strength'] = ret_5d - stats.get('industry_return_5d', 0)

            # 3. 更新估值排名
            vals = industry_vals.get(key)
            if vals:
                features['pe_industry_rank'] = vals['pe_industry_rank']
                features['pb_industry_rank'] = vals['pb_industry_rank']
                features['ps_industry_rank'] = vals['ps_industry_rank']
                changed = True

        # 4. 更新 turnover_rate / log_market_cap (从 fund_share)
        fund_data = fund_lookup.get((code, trade_date))
        if fund_data:
            if fund_data['turnover_rate'] > 0:
                features['_turnover_rate'] = fund_data['turnover_rate']
            if fund_data['log_market_cap'] > 0:
                features['_log_market_cap'] = fund_data['log_market_cap']
            changed = True

        if changed:
            batch.append((json.dumps(features, ensure_ascii=False), row_id))
            updated += 1

        if len(batch) >= 5000:
            conn.executemany("UPDATE v39_feature_cache SET features_json = ? WHERE id = ?", batch)
            conn.commit()
            logger.info(f"    已更新 {updated}/{total}")
            batch = []

    if batch:
        conn.executemany("UPDATE v39_feature_cache SET features_json = ? WHERE id = ?", batch)
        conn.commit()

    conn.close()
    logger.info(f"  特征更新完成: {updated}/{total} 条ETF记录")


def main():
    parser = argparse.ArgumentParser(description='ETF特征补充器')
    parser.add_argument('--start-date', default='2020-01-01')
    parser.add_argument('--end-date', default='2026-03-23')
    parser.add_argument('--skip-fetch', action='store_true', help='跳过Tushare数据抓取')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ETF 特征补充器")
    logger.info("=" * 60)

    # Step 1: 构建行业映射
    logger.info("\n[Step 1] 构建ETF→申万行业映射...")
    etf_industry_map = build_etf_industry_mapping()

    # 统计映射分布
    industry_counts = {}
    for ind in etf_industry_map.values():
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
    logger.info("  行业分布:")
    for ind, cnt in sorted(industry_counts.items(), key=lambda x: -x[1])[:15]:
        logger.info(f"    {ind}: {cnt} 只ETF")

    # Step 2: 抓取ETF基本面数据
    etf_fundamentals = pd.DataFrame()
    if not args.skip_fetch:
        logger.info("\n[Step 2] 抓取ETF份额/市值数据...")
        etf_codes = list(etf_industry_map.keys())
        etf_fundamentals = fetch_etf_fundamentals(
            etf_codes, args.start_date, args.end_date)
    else:
        logger.info("\n[Step 2] 跳过数据抓取 (--skip-fetch)")

    # Step 3: 更新特征缓存
    logger.info("\n[Step 3] 更新 v39_feature_cache 中ETF特征...")
    update_etf_features(etf_industry_map, etf_fundamentals,
                        args.start_date, args.end_date)

    logger.info("\n完成!")


if __name__ == '__main__':
    main()
