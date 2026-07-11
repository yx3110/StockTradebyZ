#!/usr/bin/env python3
"""
简化版财务指标数据获取器
获取核心财务指标数据并存储到本地SQLite数据库
"""

import sqlite3
import sys
import tushare as ts
import pandas as pd
import json
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from pathlib import Path
import argparse

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 设置日志
logger = logging.getLogger(__name__)

class SimpleFinancialIndicatorFetcher:
    """简化版财务指标数据获取器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db", config_path: str = "config.json"):
        self.db_path = db_path
        self.config_path = config_path
        self.pro = None
        
        # 初始化Tushare
        self._init_tushare()
        
        # API调用间隔（秒）
        self.api_delay = 0.5
        
    def _init_tushare(self):
        """初始化Tushare (token 优先从 core.config/.env 获取)"""
        try:
            try:
                from core.config import get_tushare_token
                token = get_tushare_token()
            except ImportError:
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
        """获取单只股票的核心财务指标数据"""
        try:
            # API调用间隔
            time.sleep(self.api_delay)
            
            # 获取核心财务指标数据
            df = self.pro.fina_indicator(
                ts_code=ts_code,
                fields='ts_code,ann_date,end_date,eps,gross_margin,current_ratio,roe,roa,netprofit_margin,debt_to_assets,basic_eps_yoy,netprofit_yoy,or_yoy',
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
        """保存核心财务指标数据到数据库"""
        if df is None or df.empty:
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                saved_count = 0
                
                for _, row in df.iterrows():
                    # 使用核心字段的INSERT语句
                    cursor.execute("""
                        INSERT OR IGNORE INTO financial_indicator (
                            security_id, ann_date, end_date, eps, gross_margin, 
                            current_ratio, roe, roa, netprofit_margin, debt_to_assets, 
                            basic_eps_yoy, netprofit_yoy, or_yoy
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        security_id,
                        row.get('ann_date'),
                        row.get('end_date'), 
                        row.get('eps'),
                        row.get('gross_margin'),
                        row.get('current_ratio'),
                        row.get('roe'),
                        row.get('roa'),
                        row.get('netprofit_margin'),
                        row.get('debt_to_assets'),
                        row.get('basic_eps_yoy'),
                        row.get('netprofit_yoy'),
                        row.get('or_yoy')
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
                    success_count += 1  # 数据已存在也算成功
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
    
    def check_data_status(self):
        """检查财务指标数据状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 统计总数据量
                cursor.execute("SELECT COUNT(*) FROM financial_indicator")
                total_records = cursor.fetchone()[0]
                
                # 统计有数据的股票数量
                cursor.execute("""
                    SELECT COUNT(DISTINCT security_id) FROM financial_indicator
                """)
                stocks_with_data = cursor.fetchone()[0]
                
                # 统计总股票数量
                cursor.execute("SELECT COUNT(*) FROM securities WHERE type LIKE '%股%'")
                total_stocks = cursor.fetchone()[0]
                
                # 获取最新数据日期
                cursor.execute("SELECT MAX(end_date) FROM financial_indicator")
                latest_date = cursor.fetchone()[0]
                
                logger.info(f"📊 财务指标数据统计：")
                logger.info(f"  总记录数: {total_records}")
                logger.info(f"  有数据股票: {stocks_with_data}/{total_stocks} ({stocks_with_data/total_stocks*100:.1f}%)")
                logger.info(f"  最新数据: {latest_date}")
                
                # 样本数据预览
                cursor.execute("""
                    SELECT s.code, s.name, fi.end_date, fi.roe, fi.roa, fi.gross_margin
                    FROM financial_indicator fi
                    JOIN securities s ON fi.security_id = s.id
                    ORDER BY fi.end_date DESC
                    LIMIT 5
                """)
                
                samples = cursor.fetchall()
                if samples:
                    logger.info(f"📋 最新财务数据样本：")
                    for code, name, end_date, roe, roa, gross_margin in samples:
                        logger.info(f"  {code} {name} ({end_date}): ROE={roe}%, ROA={roa}%, 毛利率={gross_margin}%")
                
        except Exception as e:
            logger.error(f"检查数据状态失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化版财务指标数据获取器")
    parser.add_argument("--mode", choices=['test', 'batch', 'status'], default='test',
                       help="运行模式: test=测试5只, batch=批量获取, status=检查状态")
    parser.add_argument("--start", type=int, default=0,
                       help="起始索引")
    parser.add_argument("--limit", type=int, default=None,
                       help="处理数量限制")
    
    args = parser.parse_args()
    
    # 创建获取器
    fetcher = SimpleFinancialIndicatorFetcher()
    
    if args.mode == 'test':
        # 测试模式 - 获取前5只股票的数据
        logger.info("🧪 测试模式：获取前5只股票的财务指标")
        fetcher.fetch_all_financial_indicators(start_idx=0, limit=5)
        
    elif args.mode == 'batch':
        # 批量获取模式
        logger.info("📊 批量获取模式")
        fetcher.fetch_all_financial_indicators(start_idx=args.start, limit=args.limit)
        
    elif args.mode == 'status':
        # 检查数据状态
        logger.info("📊 检查财务指标数据状态")
        fetcher.check_data_status()
    
    logger.info("✅ 财务指标数据获取任务完成")


if __name__ == "__main__":
    main()