#!/usr/bin/env python3
"""
财务指标数据获取器
从Tushare获取股票财务指标数据并存储到本地SQLite数据库
"""

import sqlite3
import tushare as ts
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from pathlib import Path
import argparse

# 设置日志
logger = logging.getLogger(__name__)

class FinancialIndicatorFetcher:
    """财务指标数据获取器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db", config_path: str = "config.json"):
        self.db_path = db_path
        self.config_path = config_path
        self.pro = None
        
        # 初始化Tushare
        self._init_tushare()
        
        # API调用间隔（秒）
        self.api_delay = 0.5
        
    def _init_tushare(self):
        """初始化Tushare"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            token = config['tushare']['token']
            if not token:
                raise ValueError("未配置Tushare token")
            
            ts.set_token(token)
            self.pro = ts.pro_api()
            logger.info("✅ Tushare API初始化成功")
            
        except Exception as e:
            logger.error(f"❌ Tushare初始化失败: {e}")
            raise
    
    def get_stock_codes(self) -> List[tuple]:
        """获取数据库中的股票代码列表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, code, name, type 
                    FROM securities 
                    WHERE type LIKE '%股%' 
                    ORDER BY code
                """)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"获取股票代码失败: {e}")
            return []
    
    def convert_stock_code(self, code: str) -> str:
        """转换股票代码为Tushare格式"""
        if len(code) != 6:
            return code
        
        # A股代码转换规则
        if code.startswith('6'):
            return f"{code}.SH"  # 上海
        elif code.startswith('0') or code.startswith('3'):
            return f"{code}.SZ"  # 深圳
        elif code.startswith('8'):
            return f"{code}.BJ"  # 北交所
        else:
            return f"{code}.SH"  # 默认上海
    
    def fetch_financial_indicator(self, ts_code: str, periods: int = 8) -> Optional[pd.DataFrame]:
        """获取单只股票的财务指标数据"""
        try:
            # API调用间隔
            time.sleep(self.api_delay)
            
            # 获取财务指标数据
            df = self.pro.fina_indicator(
                ts_code=ts_code,
                fields='''ts_code,ann_date,end_date,eps,dt_eps,total_revenue_ps,revenue_ps,
                        capital_rese_ps,surplus_rese_ps,undist_profit_ps,extra_item,profit_dedt,
                        gross_margin,current_ratio,quick_ratio,cash_ratio,invturn_days,arturn_days,
                        inv_turn,ar_turn,ca_turn,fa_turn,assets_turn,op_income,valuechange_income,
                        interst_income,daa,ebit,ebitda,fcff,fcfe,current_exint,noncurrent_exint,
                        interestdebt,netdebt,tangible_asset,working_capital,networking_capital,
                        invest_capital,retained_earnings,diluted2_eps,bps,ocfps,retainedps,cfps,
                        ebit_ps,fcff_ps,fcfe_ps,netprofit_margin,grossprofit_margin,cogs_of_sales,
                        expense_of_sales,profit_to_gr,saleexp_to_gr,adminexp_of_gr,finaexp_of_gr,
                        impai_ttm,gc_of_gr,op_of_gr,ebit_of_gr,roe,roe_waa,roe_dt,roa,npta,roic,
                        roe_yearly,roa2_yearly,roe_avg,opincome_of_ebt,investincome_of_ebt,
                        n_op_profit_of_ebt,tax_to_ebt,dtprofit_to_profit,salescash_to_or,ocf_to_or,
                        ocf_to_opincome,capitalized_to_da,debt_to_assets,assets_to_eqt,
                        dp_assets_to_eqt,ca_to_assets,nca_to_assets,tbassets_to_totalassets,
                        int_to_talcap,eqt_to_talcapital,currentdebt_to_debt,longdeb_to_debt,
                        ocf_to_shortdebt,debt_to_eqt,eqt_to_debt,eqt_to_interestdebt,
                        tangibleasset_to_debt,tangasset_to_intdebt,tangibleasset_to_netdebt,
                        ocf_to_debt,ocf_to_interestdebt,ocf_to_netdebt,ebit_to_interest,
                        longdebt_to_workingcapital,ebitda_to_debt,turn_days,roa_yearly,roa_dp,
                        fixed_assets,profit_prefin_exp,non_op_profit,op_to_ebt,nop_to_ebt,
                        ocf_to_profit,cash_to_liqdebt,cash_to_liqdebt_withinterest,op_to_liqdebt,
                        op_to_debt,roic_yearly,total_fa_trun,profit_to_op,q_opincome,q_investincome,
                        q_dtprofit,q_eps,q_netprofit_margin,q_gsprofit_margin,q_exp_to_sales,
                        q_profit_to_gr,q_saleexp_to_gr,q_adminexp_to_gr,q_finaexp_to_gr,q_gc_to_gr,
                        q_op_to_gr,q_roe,q_dt_roe,q_npta,q_opincome_to_ebt,q_investincome_to_ebt,
                        q_dtprofit_to_profit,q_salescash_to_or,q_ocf_to_sales,q_ocf_to_or,
                        basic_eps_yoy,dt_eps_yoy,cfps_yoy,op_yoy,ebt_yoy,netprofit_yoy,
                        dt_netprofit_yoy,ocf_yoy,roe_yoy,bps_yoy,assets_yoy,eqt_yoy,tr_yoy,or_yoy,
                        q_gr_yoy,q_gr_qoq,q_sales_yoy,q_sales_qoq,q_op_yoy,q_op_qoq,q_profit_yoy,
                        q_profit_qoq,q_netprofit_yoy,q_netprofit_qoq,equity_yoy,rd_exp''',
                limit=periods
            )
            
            if df.empty:
                logger.warning(f"  ⚠️ {ts_code} 无财务指标数据")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"  ❌ 获取{ts_code}财务指标失败: {e}")
            return None
    
    def save_financial_indicator(self, df: pd.DataFrame, security_id: int) -> int:
        """保存财务指标数据到数据库"""
        if df is None or df.empty:
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                saved_count = 0
                
                for _, row in df.iterrows():
                    # 构建插入SQL（使用ON CONFLICT IGNORE避免重复）
                    cursor.execute("""
                        INSERT OR IGNORE INTO financial_indicator (
                            security_id, ann_date, end_date, eps, dt_eps, total_revenue_ps,
                            revenue_ps, capital_rese_ps, surplus_rese_ps, undist_profit_ps,
                            extra_item, profit_dedt, gross_margin, current_ratio, quick_ratio,
                            cash_ratio, invturn_days, arturn_days, inv_turn, ar_turn, ca_turn,
                            fa_turn, assets_turn, op_income, valuechange_income, interst_income,
                            daa, ebit, ebitda, fcff, fcfe, current_exint, noncurrent_exint,
                            interestdebt, netdebt, tangible_asset, working_capital,
                            networking_capital, invest_capital, retained_earnings, diluted2_eps,
                            bps, ocfps, retainedps, cfps, ebit_ps, fcff_ps, fcfe_ps,
                            netprofit_margin, grossprofit_margin, cogs_of_sales, expense_of_sales,
                            profit_to_gr, saleexp_to_gr, adminexp_of_gr, finaexp_of_gr,
                            impai_ttm, gc_of_gr, op_of_gr, ebit_of_gr, roe, roe_waa, roe_dt,
                            roa, npta, roic, roe_yearly, roa2_yearly, roe_avg, opincome_of_ebt,
                            investincome_of_ebt, n_op_profit_of_ebt, tax_to_ebt,
                            dtprofit_to_profit, salescash_to_or, ocf_to_or, ocf_to_opincome,
                            capitalized_to_da, debt_to_assets, assets_to_eqt, dp_assets_to_eqt,
                            ca_to_assets, nca_to_assets, tbassets_to_totalassets, int_to_talcap,
                            eqt_to_talcapital, currentdebt_to_debt, longdeb_to_debt,
                            ocf_to_shortdebt, debt_to_eqt, eqt_to_debt, eqt_to_interestdebt,
                            tangibleasset_to_debt, tangasset_to_intdebt, tangibleasset_to_netdebt,
                            ocf_to_debt, ocf_to_interestdebt, ocf_to_netdebt, ebit_to_interest,
                            longdebt_to_workingcapital, ebitda_to_debt, turn_days, roa_yearly,
                            roa_dp, fixed_assets, profit_prefin_exp, non_op_profit, op_to_ebt,
                            nop_to_ebt, ocf_to_profit, cash_to_liqdebt,
                            cash_to_liqdebt_withinterest, op_to_liqdebt, op_to_debt,
                            roic_yearly, total_fa_trun, profit_to_op, q_opincome, q_investincome,
                            q_dtprofit, q_eps, q_netprofit_margin, q_gsprofit_margin,
                            q_exp_to_sales, q_profit_to_gr, q_saleexp_to_gr, q_adminexp_to_gr,
                            q_finaexp_to_gr, q_gc_to_gr, q_op_to_gr, q_roe, q_dt_roe, q_npta,
                            q_opincome_to_ebt, q_investincome_to_ebt, q_dtprofit_to_profit,
                            q_salescash_to_or, q_ocf_to_sales, q_ocf_to_or, basic_eps_yoy,
                            dt_eps_yoy, cfps_yoy, op_yoy, ebt_yoy, netprofit_yoy,
                            dt_netprofit_yoy, ocf_yoy, roe_yoy, bps_yoy, assets_yoy, eqt_yoy,
                            tr_yoy, or_yoy, q_gr_yoy, q_gr_qoq, q_sales_yoy, q_sales_qoq,
                            q_op_yoy, q_op_qoq, q_profit_yoy, q_profit_qoq, q_netprofit_yoy,
                            q_netprofit_qoq, equity_yoy, rd_exp
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        security_id,
                        row.get('ann_date'),
                        row.get('end_date'),
                        row.get('eps'),
                        row.get('dt_eps'),
                        row.get('total_revenue_ps'),
                        row.get('revenue_ps'),
                        row.get('capital_rese_ps'),
                        row.get('surplus_rese_ps'),
                        row.get('undist_profit_ps'),
                        row.get('extra_item'),
                        row.get('profit_dedt'),
                        row.get('gross_margin'),
                        row.get('current_ratio'),
                        row.get('quick_ratio'),
                        row.get('cash_ratio'),
                        row.get('invturn_days'),
                        row.get('arturn_days'),
                        row.get('inv_turn'),
                        row.get('ar_turn'),
                        row.get('ca_turn'),
                        row.get('fa_turn'),
                        row.get('assets_turn'),
                        row.get('op_income'),
                        row.get('valuechange_income'),
                        row.get('interst_income'),
                        row.get('daa'),
                        row.get('ebit'),
                        row.get('ebitda'),
                        row.get('fcff'),
                        row.get('fcfe'),
                        row.get('current_exint'),
                        row.get('noncurrent_exint'),
                        row.get('interestdebt'),
                        row.get('netdebt'),
                        row.get('tangible_asset'),
                        row.get('working_capital'),
                        row.get('networking_capital'),
                        row.get('invest_capital'),
                        row.get('retained_earnings'),
                        row.get('diluted2_eps'),
                        row.get('bps'),
                        row.get('ocfps'),
                        row.get('retainedps'),
                        row.get('cfps'),
                        row.get('ebit_ps'),
                        row.get('fcff_ps'),
                        row.get('fcfe_ps'),
                        row.get('netprofit_margin'),
                        row.get('grossprofit_margin'),
                        row.get('cogs_of_sales'),
                        row.get('expense_of_sales'),
                        row.get('profit_to_gr'),
                        row.get('saleexp_to_gr'),
                        row.get('adminexp_of_gr'),
                        row.get('finaexp_of_gr'),
                        row.get('impai_ttm'),
                        row.get('gc_of_gr'),
                        row.get('op_of_gr'),
                        row.get('ebit_of_gr'),
                        row.get('roe'),
                        row.get('roe_waa'),
                        row.get('roe_dt'),
                        row.get('roa'),
                        row.get('npta'),
                        row.get('roic'),
                        row.get('roe_yearly'),
                        row.get('roa2_yearly'),
                        row.get('roe_avg'),
                        row.get('opincome_of_ebt'),
                        row.get('investincome_of_ebt'),
                        row.get('n_op_profit_of_ebt'),
                        row.get('tax_to_ebt'),
                        row.get('dtprofit_to_profit'),
                        row.get('salescash_to_or'),
                        row.get('ocf_to_or'),
                        row.get('ocf_to_opincome'),
                        row.get('capitalized_to_da'),
                        row.get('debt_to_assets'),
                        row.get('assets_to_eqt'),
                        row.get('dp_assets_to_eqt'),
                        row.get('ca_to_assets'),
                        row.get('nca_to_assets'),
                        row.get('tbassets_to_totalassets'),
                        row.get('int_to_talcap'),
                        row.get('eqt_to_talcapital'),
                        row.get('currentdebt_to_debt'),
                        row.get('longdeb_to_debt'),
                        row.get('ocf_to_shortdebt'),
                        row.get('debt_to_eqt'),
                        row.get('eqt_to_debt'),
                        row.get('eqt_to_interestdebt'),
                        row.get('tangibleasset_to_debt'),
                        row.get('tangasset_to_intdebt'),
                        row.get('tangibleasset_to_netdebt'),
                        row.get('ocf_to_debt'),
                        row.get('ocf_to_interestdebt'),
                        row.get('ocf_to_netdebt'),
                        row.get('ebit_to_interest'),
                        row.get('longdebt_to_workingcapital'),
                        row.get('ebitda_to_debt'),
                        row.get('turn_days'),
                        row.get('roa_yearly'),
                        row.get('roa_dp'),
                        row.get('fixed_assets'),
                        row.get('profit_prefin_exp'),
                        row.get('non_op_profit'),
                        row.get('op_to_ebt'),
                        row.get('nop_to_ebt'),
                        row.get('ocf_to_profit'),
                        row.get('cash_to_liqdebt'),
                        row.get('cash_to_liqdebt_withinterest'),
                        row.get('op_to_liqdebt'),
                        row.get('op_to_debt'),
                        row.get('roic_yearly'),
                        row.get('total_fa_trun'),
                        row.get('profit_to_op'),
                        row.get('q_opincome'),
                        row.get('q_investincome'),
                        row.get('q_dtprofit'),
                        row.get('q_eps'),
                        row.get('q_netprofit_margin'),
                        row.get('q_gsprofit_margin'),
                        row.get('q_exp_to_sales'),
                        row.get('q_profit_to_gr'),
                        row.get('q_saleexp_to_gr'),
                        row.get('q_adminexp_to_gr'),
                        row.get('q_finaexp_to_gr'),
                        row.get('q_gc_to_gr'),
                        row.get('q_op_to_gr'),
                        row.get('q_roe'),
                        row.get('q_dt_roe'),
                        row.get('q_npta'),
                        row.get('q_opincome_to_ebt'),
                        row.get('q_investincome_to_ebt'),
                        row.get('q_dtprofit_to_profit'),
                        row.get('q_salescash_to_or'),
                        row.get('q_ocf_to_sales'),
                        row.get('q_ocf_to_or'),
                        row.get('basic_eps_yoy'),
                        row.get('dt_eps_yoy'),
                        row.get('cfps_yoy'),
                        row.get('op_yoy'),
                        row.get('ebt_yoy'),
                        row.get('netprofit_yoy'),
                        row.get('dt_netprofit_yoy'),
                        row.get('ocf_yoy'),
                        row.get('roe_yoy'),
                        row.get('bps_yoy'),
                        row.get('assets_yoy'),
                        row.get('eqt_yoy'),
                        row.get('tr_yoy'),
                        row.get('or_yoy'),
                        row.get('q_gr_yoy'),
                        row.get('q_gr_qoq'),
                        row.get('q_sales_yoy'),
                        row.get('q_sales_qoq'),
                        row.get('q_op_yoy'),
                        row.get('q_op_qoq'),
                        row.get('q_profit_yoy'),
                        row.get('q_profit_qoq'),
                        row.get('q_netprofit_yoy'),
                        row.get('q_netprofit_qoq'),
                        row.get('equity_yoy'),
                        row.get('rd_exp')
                    ))
                    
                    if cursor.rowcount > 0:
                        saved_count += 1
                
                conn.commit()
                return saved_count
                
        except Exception as e:
            logger.error(f"保存财务指标数据失败: {e}")
            return 0
    
    def fetch_all_financial_indicators(self, start_idx: int = 0, limit: int = None):
        """批量获取所有股票的财务指标数据"""
        stock_codes = self.get_stock_codes()
        if not stock_codes:
            logger.error("❌ 未获取到股票代码列表")
            return
        
        total_stocks = len(stock_codes)
        if limit:
            stock_codes = stock_codes[start_idx:start_idx + limit]
            logger.info(f"📊 准备处理第{start_idx + 1}-{start_idx + len(stock_codes)}只股票 (共{total_stocks}只)")
        else:
            stock_codes = stock_codes[start_idx:]
            logger.info(f"📊 准备处理第{start_idx + 1}-{total_stocks}只股票")
        
        success_count = 0
        fail_count = 0
        total_records = 0
        
        start_time = time.time()
        
        for i, (security_id, code, name, stock_type) in enumerate(stock_codes, start_idx + 1):
            logger.info(f"[{i}/{total_stocks}] 处理 {code} - {name}")
            
            # 转换股票代码
            ts_code = self.convert_stock_code(code)
            
            # 获取财务指标数据
            df = self.fetch_financial_indicator(ts_code)
            
            if df is not None:
                # 保存数据
                saved_count = self.save_financial_indicator(df, security_id)
                if saved_count > 0:
                    logger.info(f"  ✅ 保存{saved_count}条财务指标记录")
                    success_count += 1
                    total_records += saved_count
                else:
                    logger.info(f"  ⚠️ 数据已存在，跳过")
            else:
                fail_count += 1
            
            # 每100只股票显示进度
            if i % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"📈 进度: {i}/{total_stocks} ({i/total_stocks*100:.1f}%), "
                          f"成功: {success_count}, 失败: {fail_count}, "
                          f"总记录: {total_records}, 用时: {elapsed/60:.1f}分钟")
        
        # 最终统计
        elapsed = time.time() - start_time
        logger.info(f"🎉 财务指标数据获取完成！")
        logger.info(f"📊 处理股票: {len(stock_codes)}只")
        logger.info(f"✅ 成功: {success_count}只")
        logger.info(f"❌ 失败: {fail_count}只")
        logger.info(f"📝 总记录: {total_records}条")
        logger.info(f"⏱️ 用时: {elapsed/60:.1f}分钟")
    
    def update_recent_financial_indicators(self, months: int = 6):
        """更新近期财务指标（增量更新）"""
        logger.info(f"📊 开始更新最近{months}个月的财务指标数据...")
        
        # 获取有业务最新财务数据日期的股票
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT s.id, s.code, s.name, 
                           COALESCE(MAX(fi.end_date), '2020-01-01') as last_date
                    FROM securities s
                    LEFT JOIN financial_indicator fi ON s.id = fi.security_id
                    WHERE s.type LIKE '%股%'
                    GROUP BY s.id, s.code, s.name
                    ORDER BY s.code
                """)
                stock_data = cursor.fetchall()
        except Exception as e:
            logger.error(f"获取股票财务数据状态失败: {e}")
            return
        
        if not stock_data:
            logger.error("❌ 未获取到股票数据")
            return
        
        logger.info(f"📊 准备更新{len(stock_data)}只股票的财务指标")
        
        success_count = 0
        fail_count = 0
        total_records = 0
        start_time = time.time()
        
        for i, (security_id, code, name, last_date) in enumerate(stock_data, 1):
            logger.info(f"[{i}/{len(stock_data)}] 更新 {code} - {name} (最新: {last_date})")
            
            # 转换股票代码
            ts_code = self.convert_stock_code(code)
            
            # 获取财务指标数据（最近几期）
            df = self.fetch_financial_indicator(ts_code, periods=4)
            
            if df is not None:
                # 保存数据
                saved_count = self.save_financial_indicator(df, security_id)
                if saved_count > 0:
                    logger.info(f"  ✅ 新增{saved_count}条财务指标记录")
                    total_records += saved_count
                success_count += 1
            else:
                fail_count += 1
            
            # 每50只股票显示进度
            if i % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(f"📈 进度: {i}/{len(stock_data)} ({i/len(stock_data)*100:.1f}%), "
                          f"成功: {success_count}, 失败: {fail_count}, "
                          f"新增记录: {total_records}, 用时: {elapsed/60:.1f}分钟")
        
        # 最终统计
        elapsed = time.time() - start_time
        logger.info(f"🎉 财务指标数据更新完成！")
        logger.info(f"📊 处理股票: {len(stock_data)}只")
        logger.info(f"✅ 成功: {success_count}只")
        logger.info(f"❌ 失败: {fail_count}只")
        logger.info(f"📝 新增记录: {total_records}条")
        logger.info(f"⏱️ 用时: {elapsed/60:.1f}分钟")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="财务指标数据获取器")
    parser.add_argument("--mode", choices=['full', 'update', 'test'], default='update',
                       help="运行模式: full=完整获取, update=增量更新, test=测试")
    parser.add_argument("--start", type=int, default=0,
                       help="起始索引（仅full模式）")
    parser.add_argument("--limit", type=int, default=None,
                       help="处理数量限制（仅full模式）")
    parser.add_argument("--months", type=int, default=6,
                       help="更新最近N个月数据（仅update模式）")
    
    args = parser.parse_args()
    
    # 创建获取器
    fetcher = FinancialIndicatorFetcher()
    
    if args.mode == 'test':
        # 测试模式 - 获取几只股票的数据
        logger.info("🧪 测试模式：获取前5只股票的财务指标")
        fetcher.fetch_all_financial_indicators(start_idx=0, limit=5)
        
    elif args.mode == 'full':
        # 完整获取模式
        logger.info("📊 完整获取模式：获取所有股票的财务指标")
        fetcher.fetch_all_financial_indicators(start_idx=args.start, limit=args.limit)
        
    elif args.mode == 'update':
        # 增量更新模式
        logger.info("🔄 增量更新模式：更新最近财务指标")
        fetcher.update_recent_financial_indicators(months=args.months)
    
    logger.info("✅ 财务指标数据获取任务完成")


if __name__ == "__main__":
    main()