#!/usr/bin/env python3
"""
增强型相似度分析报告生成器 - 包含后续走势跟踪
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import json
import requests
import os

# 添加项目根目录到路径
sys.path.append('/Users/yangxu/StockTradebyZ')

from data_adapter.database_manager import DatabaseManager

def track_future_performance(db, stock_code, query_date, days_forward=[5, 10]):
    """
    跟踪股票在特定日期后的走势表现
    """
    results = {}
    
    # 获取基准日期的收盘价
    base_query = """
    SELECT close 
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date = ?
    """
    
    base_price = db.execute_query(base_query, (stock_code, query_date))
    if not base_price:
        return results
    
    base_price = base_price[0][0]
    
    # 获取后续N天的数据
    future_query = """
    SELECT trade_date, high, low, close
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date > ?
    ORDER BY trade_date ASC
    LIMIT ?
    """
    
    for days in days_forward:
        future_data = db.execute_query(future_query, (stock_code, query_date, days))
        
        if future_data and len(future_data) > 0:
            highs = [row[1] for row in future_data]
            lows = [row[2] for row in future_data]
            closes = [row[3] for row in future_data]
            
            # 计算最大浮盈和浮亏
            max_gain = (max(highs) / base_price - 1) * 100
            max_loss = (min(lows) / base_price - 1) * 100
            
            # 计算最终收益
            final_return = (closes[-1] / base_price - 1) * 100 if len(closes) > 0 else None
            
            results[f'{days}_days'] = {
                'max_gain': round(max_gain, 2),
                'max_loss': round(max_loss, 2),
                'final_return': round(final_return, 2) if final_return else None,
                'actual_days': len(future_data)
            }
        else:
            results[f'{days}_days'] = {
                'max_gain': None,
                'max_loss': None,
                'final_return': None,
                'actual_days': 0
            }
    
    return results

def get_stock_fundamentals(db, stock_code, trade_date=None):
    """获取股票的基本面、技术面、财务面数据"""
    
    if not trade_date:
        trade_date = datetime.now().strftime('%Y-%m-%d')
    
    # 基本信息
    basic_query = """
    SELECT s.name, sbi.market, s.type, s.exchange
    FROM securities s
    LEFT JOIN stock_basic_info sbi ON s.id = sbi.security_id
    WHERE s.code = ?
    """
    
    # 最新财务数据
    financial_query = """
    SELECT 
        db.pe_ttm, db.pb, db.ps_ttm, db.dv_ratio, db.dv_ttm,
        db.total_mv, db.circ_mv, db.turnover_rate, db.volume_ratio
    FROM daily_basic db
    JOIN securities s ON db.security_id = s.id
    WHERE s.code = ? AND db.trade_date <= ?
    ORDER BY db.trade_date DESC
    LIMIT 1
    """
    
    # 技术指标
    technical_query = """
    SELECT 
        dq.ma5, dq.ma10, dq.ma20, dq.ma60,
        ti.rsi6, ti.rsi12, ti.macd_dif, ti.macd_dea, ti.kdj_k, ti.kdj_d, ti.kdj_j,
        ti.bbi
    FROM technical_indicators ti
    JOIN securities s ON ti.security_id = s.id
    JOIN daily_quotes dq ON ti.security_id = dq.security_id AND ti.trade_date = dq.trade_date
    WHERE s.code = ? AND ti.trade_date <= ?
    ORDER BY ti.trade_date DESC
    LIMIT 1
    """
    
    # 最新价格数据
    price_query = """
    SELECT 
        dq.close, dq.volume, dq.price_change_pct,
        dq.high, dq.low, dq.open
    FROM daily_quotes dq
    JOIN securities s ON dq.security_id = s.id
    WHERE s.code = ? AND dq.trade_date <= ?
    ORDER BY dq.trade_date DESC
    LIMIT 1
    """
    
    # 执行查询
    basic_info = db.execute_query(basic_query, (stock_code,))
    financial_data = db.execute_query(financial_query, (stock_code, trade_date))
    technical_data = db.execute_query(technical_query, (stock_code, trade_date))
    price_data = db.execute_query(price_query, (stock_code, trade_date))
    
    result = {
        'code': stock_code,
        'basic': {},
        'financial': {},
        'technical': {},
        'price': {}
    }
    
    if basic_info:
        result['basic'] = {
            'name': basic_info[0][0],
            'market': basic_info[0][1] or 'N/A',
            'type': basic_info[0][2] or 'N/A',
            'exchange': basic_info[0][3] or 'N/A',
            'industry': 'N/A',  # 暂不可用
            'area': 'N/A',      # 暂不可用
            'list_date': 'N/A'  # 暂不可用
        }
    
    if financial_data:
        result['financial'] = {
            'pe_ttm': financial_data[0][0],
            'pb': financial_data[0][1],
            'ps_ttm': financial_data[0][2],
            'dv_ratio': financial_data[0][3],
            'dv_ttm': financial_data[0][4],
            'total_mv': financial_data[0][5],
            'circ_mv': financial_data[0][6],
            'turnover_rate': financial_data[0][7],
            'volume_ratio': financial_data[0][8]
        }
    
    if technical_data:
        result['technical'] = {
            'ma_5': technical_data[0][0],
            'ma_10': technical_data[0][1],
            'ma_20': technical_data[0][2],
            'ma_60': technical_data[0][3],
            'rsi_6': technical_data[0][4],
            'rsi_12': technical_data[0][5],
            'macd': technical_data[0][6],
            'signal': technical_data[0][7],
            'kdj_k': technical_data[0][8],
            'kdj_d': technical_data[0][9],
            'kdj_j': technical_data[0][10],
            'bbi': technical_data[0][11]
        }
    
    if price_data:
        result['price'] = {
            'close': price_data[0][0],
            'volume': price_data[0][1],
            'change_pct': price_data[0][2],
            'high': price_data[0][3],
            'low': price_data[0][4],
            'open': price_data[0][5]
        }
    
    return result

def enhance_similarity_report(target_code='002215', report_date='2025-08-11'):
    """
    增强现有的相似度分析报告，添加后续走势跟踪
    """
    
    print(f"🚀 开始增强 {target_code} 的相似度分析报告...")
    
    # 初始化数据库
    db = DatabaseManager()
    
    # 读取最新的相似度分析结果
    report_path = Path('/Users/yangxu/StockTradebyZ/reports/similarity_analysis')
    
    # 查找报告文件，优先选择parallel_analysis（包含相似股票数据）
    parallel_reports = list(report_path.glob(f'{target_code}_parallel_analysis_*.md'))
    enhanced_reports = list(report_path.glob(f'{target_code}_enhanced_report_*.md'))
    
    # 优先选择parallel_analysis报告，因为它包含相似股票数据
    if parallel_reports:
        latest_report = max(parallel_reports, key=lambda x: x.stat().st_mtime)
        print(f"📄 读取parallel分析报告: {latest_report}")
    elif enhanced_reports:
        latest_report = max(enhanced_reports, key=lambda x: x.stat().st_mtime)
        print(f"📄 读取enhanced报告: {latest_report}")
    else:
        print(f"❌ 未找到 {target_code} 的相似度分析报告")
        return
    
    # 解析报告获取相似股票列表
    similar_stocks = []
    with open(latest_report, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 查找表格数据
    in_table = False
    for line in lines:
        if '| 排名 | 代码 | 名称 | 行业 | 相似度 | 算法数 |' in line:
            in_table = True
            continue
        if in_table and line.startswith('|') and '---' not in line:
            parts = line.split('|')
            if len(parts) >= 7:
                try:
                    rank = int(parts[1].strip())
                    code = parts[2].strip()
                    name = parts[3].strip()
                    industry = parts[4].strip()
                    similarity = float(parts[5].strip())
                    
                    similar_stocks.append({
                        'rank': rank,
                        'code': code,
                        'name': name,
                        'industry': industry,
                        'similarity': similarity
                    })
                except:
                    continue
    
    print(f"✅ 找到 {len(similar_stocks)} 只相似股票")
    
    # 为每只股票添加后续走势数据
    # 使用2025-07-25作为基准日期（15个交易日前）
    base_date = '2025-07-25'
    
    print(f"📊 分析后续走势（基准日期: {base_date}）...")
    
    for stock in similar_stocks:
        perf = track_future_performance(db, stock['code'], base_date)
        stock['future_performance'] = perf
    
    # 获取目标股票的基本面数据
    print(f"📈 获取 {target_code} 的基本面数据...")
    fundamentals = get_stock_fundamentals(db, target_code)
    
    # 生成增强报告 - 使用版本号 + 时间日期命名
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_time_short = datetime.now().strftime('%Y%m%d_%H%M')
    
    # 检查现有版本
    existing_reports = list(report_path.glob(f'{target_code}_enhanced_report_v*.md'))
    if existing_reports:
        # 获取最新版本号
        version_numbers = []
        for report in existing_reports:
            match = report.name
            if 'v' in match:
                try:
                    v_part = match.split('v')[1].split('_')[0]
                    version_numbers.append(int(v_part))
                except:
                    pass
        next_version = max(version_numbers) + 1 if version_numbers else 1
    else:
        next_version = 1
    
    enhanced_report_file = report_path / f'{target_code}_enhanced_report_v{next_version}_{current_time_short}.md'
    
    report_content = f"""# {target_code} 增强型投资分析报告 v{next_version}

**生成时间**: {current_time}  
**分析类型**: 相似度分析 + 后续走势跟踪  
**基准日期**: {base_date} （用于后续走势分析）  
**版本号**: v{next_version}  

---

## 📊 目标股票基本信息

### 基础信息
- **股票代码**: {target_code}
- **股票名称**: {fundamentals['basic'].get('name', 'N/A')}
- **所属行业**: {fundamentals['basic'].get('industry', 'N/A')}
- **地区**: {fundamentals['basic'].get('area', 'N/A')}
- **上市日期**: {fundamentals['basic'].get('list_date', 'N/A')}

### 财务指标
- **市盈率(TTM)**: {fundamentals['financial'].get('pe_ttm', 'N/A')}
- **市净率**: {fundamentals['financial'].get('pb', 'N/A')}
- **市销率(TTM)**: {fundamentals['financial'].get('ps_ttm', 'N/A')}
- **总市值**: {fundamentals['financial'].get('total_mv', 'N/A')} 万元
- **流通市值**: {fundamentals['financial'].get('circ_mv', 'N/A')} 万元
- **换手率**: {fundamentals['financial'].get('turnover_rate', 'N/A')}%

### 技术指标
- **当前价格**: {fundamentals['price'].get('close', 'N/A')}
- **涨跌幅**: {fundamentals['price'].get('change_pct', 'N/A')}%
- **MA5**: {fundamentals['technical'].get('ma_5', 'N/A')}
- **MA20**: {fundamentals['technical'].get('ma_20', 'N/A')}
- **RSI(6)**: {fundamentals['technical'].get('rsi_6', 'N/A')}
- **KDJ_K**: {fundamentals['technical'].get('kdj_k', 'N/A')}
- **MACD**: {fundamentals['technical'].get('macd', 'N/A')}

---

## 📈 相似股票及后续走势分析

基于历史走势相似度分析，找到以下高度相似的股票，并跟踪其在相似形态后的表现：

| 排名 | 代码 | 名称 | 行业 | 相似度 | 5日最大浮盈 | 5日最大浮亏 | 5日收益 | 10日最大浮盈 | 10日最大浮亏 | 10日收益 |
|------|------|------|------|--------|------------|------------|---------|-------------|-------------|----------|
"""
    
    # 添加股票数据
    for stock in similar_stocks[:20]:  # 只显示前20只
        perf_5 = stock['future_performance'].get('5_days', {})
        perf_10 = stock['future_performance'].get('10_days', {})
        
        report_content += (
            f"| {stock['rank']} | {stock['code']} | {stock['name']} | "
            f"{stock['industry']} | {stock['similarity']:.4f} | "
            f"{perf_5.get('max_gain', 'N/A')}% | "
            f"{perf_5.get('max_loss', 'N/A')}% | "
            f"{perf_5.get('final_return', 'N/A')}% | "
            f"{perf_10.get('max_gain', 'N/A')}% | "
            f"{perf_10.get('max_loss', 'N/A')}% | "
            f"{perf_10.get('final_return', 'N/A')}% |\n"
        )
    
    # 统计分析
    valid_5d = [s['future_performance']['5_days']['final_return'] 
               for s in similar_stocks 
               if s['future_performance'].get('5_days', {}).get('final_return') is not None]
    
    valid_10d = [s['future_performance']['10_days']['final_return'] 
                for s in similar_stocks 
                if s['future_performance'].get('10_days', {}).get('final_return') is not None]
    
    report_content += f"""

---

## 📊 后续走势统计分析

### 5日走势统计
- **平均收益率**: {np.mean(valid_5d) if valid_5d else 0:.2f}%
- **中位数收益**: {np.median(valid_5d) if valid_5d else 0:.2f}%
- **上涨概率**: {sum(1 for x in valid_5d if x > 0)/len(valid_5d)*100 if valid_5d else 0:.1f}% ({sum(1 for x in valid_5d if x > 0) if valid_5d else 0}/{len(valid_5d) if valid_5d else 0})
- **最大收益**: {max(valid_5d) if valid_5d else 0:.2f}%
- **最大亏损**: {min(valid_5d) if valid_5d else 0:.2f}%

### 10日走势统计
- **平均收益率**: {np.mean(valid_10d) if valid_10d else 0:.2f}%
- **中位数收益**: {np.median(valid_10d) if valid_10d else 0:.2f}%
- **上涨概率**: {sum(1 for x in valid_10d if x > 0)/len(valid_10d)*100 if valid_10d else 0:.1f}% ({sum(1 for x in valid_10d if x > 0) if valid_10d else 0}/{len(valid_10d) if valid_10d else 0})
- **最大收益**: {max(valid_10d) if valid_10d else 0:.2f}%
- **最大亏损**: {min(valid_10d) if valid_10d else 0:.2f}%

---

## 🎯 投资建议参考

基于相似股票的历史表现，可以得出以下参考信息：

"""
    
    # 根据统计数据给出建议
    if valid_5d and valid_10d:
        avg_5d = np.mean(valid_5d)
        avg_10d = np.mean(valid_10d)
        win_rate_5d = sum(1 for x in valid_5d if x > 0) / len(valid_5d) * 100
        win_rate_10d = sum(1 for x in valid_10d if x > 0) / len(valid_10d) * 100
        
        if avg_5d > 2 and win_rate_5d > 60:
            report_content += "### 短期（5日）展望：🟢 积极\n"
            report_content += f"- 历史相似走势后5日平均收益 {avg_5d:.2f}%，胜率 {win_rate_5d:.1f}%\n"
        elif avg_5d > 0 and win_rate_5d > 50:
            report_content += "### 短期（5日）展望：🟡 中性偏多\n"
            report_content += f"- 历史相似走势后5日平均收益 {avg_5d:.2f}%，胜率 {win_rate_5d:.1f}%\n"
        else:
            report_content += "### 短期（5日）展望：🔴 谨慎\n"
            report_content += f"- 历史相似走势后5日平均收益 {avg_5d:.2f}%，胜率 {win_rate_5d:.1f}%\n"
        
        if avg_10d > 3 and win_rate_10d > 60:
            report_content += "\n### 中期（10日）展望：🟢 积极\n"
            report_content += f"- 历史相似走势后10日平均收益 {avg_10d:.2f}%，胜率 {win_rate_10d:.1f}%\n"
        elif avg_10d > 0 and win_rate_10d > 50:
            report_content += "\n### 中期（10日）展望：🟡 中性偏多\n"
            report_content += f"- 历史相似走势后10日平均收益 {avg_10d:.2f}%，胜率 {win_rate_10d:.1f}%\n"
        else:
            report_content += "\n### 中期（10日）展望：🔴 谨慎\n"
            report_content += f"- 历史相似走势后10日平均收益 {avg_10d:.2f}%，胜率 {win_rate_10d:.1f}%\n"
    
    report_content += f"""

---

---

## 🤖 Claude AI 专业投资分析

CLAUDE_ANALYSIS_PLACEHOLDER

---

**生成时间**: {current_time}  
**分析引擎**: StockTradebyZ Enhanced Analysis System + Claude AI  
**声明**: 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。
"""
    
    # 生成Claude投资分析
    print(f"🤖 开始Claude投资分析...")
    claude_analysis = generate_claude_analysis(target_code, fundamentals, similar_stocks, valid_5d, valid_10d)
    
    # 将Claude分析结果添加到报告中
    if claude_analysis:
        claude_section = claude_analysis
        print("✅ Claude分析已完成，正在整合到报告中")
    else:
        claude_section = """**Claude分析暂不可用**

由于API调用失败或配置问题，Claude AI分析功能暂时不可用。
请检查：
1. config.json中的anthropic.api_key配置
2. 网络连接状态
3. API配额和权限

您可以手动使用以下数据进行分析：
- 基本面数据：PE {pe_ttm}, PB {pb}, 市值 {total_mv}万元
- 技术面数据：RSI {rsi6}, KDJ_K {kdj_k}, MACD {macd}
- 历史相似走势：5日平均收益{avg_5d}%, 10日平均收益{avg_10d}%
""".format(
            pe_ttm=fundamentals['financial'].get('pe_ttm', 'N/A'),
            pb=fundamentals['financial'].get('pb', 'N/A'),
            total_mv=fundamentals['financial'].get('total_mv', 'N/A'),
            rsi6=fundamentals['technical'].get('rsi_6', 'N/A'),
            kdj_k=fundamentals['technical'].get('kdj_k', 'N/A'),
            macd=fundamentals['technical'].get('macd', 'N/A'),
            avg_5d=f"{np.mean(valid_5d):.2f}" if valid_5d else "0.00",
            avg_10d=f"{np.mean(valid_10d):.2f}" if valid_10d else "0.00"
        )
        print("⚠️ Claude分析失败，将显示默认提示信息")
    
    # 完成报告内容
    final_report_content = report_content.replace('CLAUDE_ANALYSIS_PLACEHOLDER', claude_section)
    
    # 保存增强报告
    with open(enhanced_report_file, 'w', encoding='utf-8') as f:
        f.write(final_report_content)
    
    print(f"✅ 增强报告已生成: {enhanced_report_file}")

def call_claude_api(prompt):
    """调用Claude API进行投资分析"""
    
    # 读取配置文件获取API密钥
    config_file = Path('/Users/yangxu/StockTradebyZ/config.json')
    
    if not config_file.exists():
        print("❌ 配置文件 config.json 不存在")
        return None
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    api_key = config.get('anthropic', {}).get('api_key')
    if not api_key:
        print("❌ 配置文件中缺少 Anthropic API 密钥")
        return None
    
    # Claude API 调用
    headers = {
        'Content-Type': 'application/json',
        'x-api-key': api_key,
        'anthropic-version': '2023-06-01'
    }
    
    # 使用配置文件中的模型参数，优先使用Sonnet 4.0
    model = config.get('anthropic', {}).get('default_model', 'claude-sonnet-4-20250514')  # 默认使用Sonnet 4.0
    max_tokens = config.get('anthropic', {}).get('max_tokens', 32000)  # Sonnet 4.0支持更大token数
    temperature = config.get('anthropic', {}).get('temperature', 0.1)
    
    data = {
        'model': model,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }
    
    print(f"🔧 使用模型: {model}, 最大tokens: {max_tokens}")
    
    try:
        print("🤖 正在调用Claude API进行投资分析...")
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json=data,
            timeout=60  # 增加超时时间到60秒
        )
        
        if response.status_code == 200:
            result = response.json()
            analysis = result['content'][0]['text']
            print("✅ Claude分析完成")
            return analysis
        else:
            print(f"❌ API调用失败: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ API调用出错: {str(e)}")
        return None

def generate_claude_analysis(target_code, fundamentals, similar_stocks, valid_5d, valid_10d):
    """生成Claude投资分析（直接调用API，不保存prompt文件）"""
    
    # 读取通用模板
    template_file = Path('/Users/yangxu/StockTradebyZ/prompts/investment_analysis_template.md')
    
    with open(template_file, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # 准备数据字典
    data = {
        'stock_code': target_code,
        'stock_name': fundamentals['basic'].get('name', 'N/A'),
        'industry': fundamentals['basic'].get('industry', 'N/A'),
        'area': fundamentals['basic'].get('area', 'N/A'),
        'pe_ttm': fundamentals['financial'].get('pe_ttm', 'N/A'),
        'pb': fundamentals['financial'].get('pb', 'N/A'),
        'ps_ttm': fundamentals['financial'].get('ps_ttm', 'N/A'),
        'total_mv': fundamentals['financial'].get('total_mv', 'N/A'),
        'circ_mv': fundamentals['financial'].get('circ_mv', 'N/A'),
        'turnover_rate': fundamentals['financial'].get('turnover_rate', 'N/A'),
        'current_price': fundamentals['price'].get('close', 'N/A'),
        'change_pct': fundamentals['price'].get('change_pct', 'N/A'),
        'ma5': fundamentals['technical'].get('ma_5', 'N/A'),
        'ma20': fundamentals['technical'].get('ma_20', 'N/A'),
        'ma60': fundamentals['technical'].get('ma_60', 'N/A'),
        'rsi6': fundamentals['technical'].get('rsi_6', 'N/A'),
        'rsi12': fundamentals['technical'].get('rsi_12', 'N/A'),
        'kdj_k': fundamentals['technical'].get('kdj_k', 'N/A'),
        'kdj_d': fundamentals['technical'].get('kdj_d', 'N/A'),
        'kdj_j': fundamentals['technical'].get('kdj_j', 'N/A'),
        'macd': fundamentals['technical'].get('macd', 'N/A'),
        'signal': fundamentals['technical'].get('signal', 'N/A'),
        'bbi': fundamentals['technical'].get('bbi', 'N/A'),
        'similar_count': len(similar_stocks),
        'avg_return_5d': f"{np.mean(valid_5d):.2f}" if valid_5d else "0.00",
        'median_return_5d': f"{np.median(valid_5d):.2f}" if valid_5d else "0.00",
        'win_rate_5d': f"{sum(1 for x in valid_5d if x > 0)/len(valid_5d)*100:.1f}" if valid_5d else "0.0",
        'max_return_5d': f"{max(valid_5d):.2f}" if valid_5d else "0.00",
        'max_loss_5d': f"{min(valid_5d):.2f}" if valid_5d else "0.00",
        'sample_count_5d': len(valid_5d),
        'avg_return_10d': f"{np.mean(valid_10d):.2f}" if valid_10d else "0.00",
        'median_return_10d': f"{np.median(valid_10d):.2f}" if valid_10d else "0.00",
        'win_rate_10d': f"{sum(1 for x in valid_10d if x > 0)/len(valid_10d)*100:.1f}" if valid_10d else "0.0",
        'max_return_10d': f"{max(valid_10d):.2f}" if valid_10d else "0.00",
        'max_loss_10d': f"{min(valid_10d):.2f}" if valid_10d else "0.00",
        'sample_count_10d': len(valid_10d)
    }
    
    # 使用模板格式化
    prompt = template.format(**data)
    
    # 直接调用Claude API
    claude_analysis = call_claude_api(prompt)
    
    return claude_analysis

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='增强型相似度报告生成器')
    parser.add_argument('--code', default='002215', help='目标股票代码')
    parser.add_argument('--date', default='2025-08-11', help='报告日期')
    
    args = parser.parse_args()
    
    enhance_similarity_report(args.code, args.date)