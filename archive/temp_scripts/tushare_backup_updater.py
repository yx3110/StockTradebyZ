#!/usr/bin/env python3
"""
Tushare备份数据下载和更新脚本
用于在Tushare API维护期间从备份网页下载数据
完全兼容现有数据库结构和日常更新流程
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
from typing import List, Optional
import re
from bs4 import BeautifulSoup

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TushareBackupUpdater:
    """Tushare备份数据更新器 - 完全兼容现有数据库结构"""
    
    def __init__(self):
        self.base_url = "http://tushare.org/bak/"
        self.db_manager = DatabaseManager()
        self.temp_dir = Path("temp_scripts/tushare_backup_data")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 重要指数代码映射
        self.index_mapping = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指',
            '000688.SH': '科创50',
            '000016.SH': '上证50',
            '000300.SH': '沪深300',
            '000905.SH': '中证500',
            '000852.SH': '中证1000',
            '932000.CSI': '中证2000',
            '000985.SH': '中证全指'
        }
    
    def get_available_dates(self) -> List[str]:
        """获取可用的数据日期"""
        try:
            response = requests.get("http://tushare.org/bak.html", timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            dates = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                date_match = re.search(r'(\d{8})', href)
                if date_match:
                    dates.add(date_match.group(1))
            
            dates = sorted(list(dates), reverse=True)
            if dates:
                logger.info(f"找到可用日期: {dates[:5]}")
                return dates[:5]
            
        except Exception as e:
            logger.error(f"获取可用日期失败: {e}")
        
        # 返回最近的交易日
        today = datetime.now()
        dates = []
        for i in range(10):
            date = today - timedelta(days=i)
            if date.weekday() < 5:  # 排除周末
                dates.append(date.strftime('%Y%m%d'))
                if len(dates) >= 3:
                    break
        return dates
    
    def download_file(self, filename: str, save_path: Path) -> bool:
        """下载单个文件"""
        url = f"{self.base_url}{filename}"
        try:
            logger.info(f"下载: {filename}")
            response = requests.get(url, timeout=30, stream=True)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"✓ 成功下载: {filename}")
                return True
            else:
                logger.warning(f"✗ 文件不存在: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"✗ 下载失败 {filename}: {e}")
            return False
    
    def update_daily_quotes(self, date: str) -> bool:
        """更新日线数据 - 兼容quick_daily_update.py的结构"""
        filename = f"daily_{date}.csv"
        save_path = self.temp_dir / filename
        
        if not self.download_file(filename, save_path):
            return False
        
        try:
            df = pd.read_csv(save_path)
            logger.info(f"读取到 {len(df)} 条日线数据")
            
            # 批量插入数据
            data_to_insert = []
            
            for _, row in df.iterrows():
                try:
                    # 解析股票代码和交易所
                    code = row['ts_code'].split('.')[0]
                    exchange = row['ts_code'].split('.')[1]
                    
                    # 插入或获取证券ID
                    security_id = self.db_manager.insert_security(
                        code=code,
                        name=code,  # 后续会更新名称
                        security_type='A股',
                        exchange=exchange
                    )
                    
                    if security_id:
                        # 转换日期格式
                        trade_date = pd.to_datetime(str(row['trade_date']), format='%Y%m%d').strftime('%Y-%m-%d')
                        
                        # 准备日线数据 - 与quick_daily_update.py完全一致
                        data_to_insert.append({
                            'security_id': security_id,
                            'trade_date': trade_date,
                            'open': row.get('open', 0),
                            'high': row.get('high', 0),
                            'low': row.get('low', 0),
                            'close': row.get('close', 0),
                            'volume': row.get('vol', 0) if 'vol' in row else row.get('volume', 0),
                            'price_change_pct': row.get('pct_chg', 0) / 100 if row.get('pct_chg') else 0,
                            'is_limit_up': row.get('limit') == 'U' if 'limit' in row else False,
                            'is_limit_down': row.get('limit') == 'D' if 'limit' in row else False
                        })
                        
                        # 同时保存到CSV文件（保持向后兼容）
                        data_dir = Path("full_securities_data")
                        data_dir.mkdir(exist_ok=True)
                        file_path = data_dir / f"{code}_A股.csv"
                        
                        new_row = pd.DataFrame([{
                            'date': trade_date,
                            'open': row.get('open', 0),
                            'close': row.get('close', 0),
                            'high': row.get('high', 0),
                            'low': row.get('low', 0),
                            'volume': row.get('vol', 0) if 'vol' in row else row.get('volume', 0)
                        }])
                        
                        if file_path.exists():
                            existing_df = pd.read_csv(file_path)
                            existing_df = existing_df[existing_df['date'] != trade_date]
                            combined_df = pd.concat([existing_df, new_row], ignore_index=True)
                            combined_df.sort_values('date', inplace=True)
                            combined_df.to_csv(file_path, index=False)
                        else:
                            new_row.to_csv(file_path, index=False)
                            
                except Exception as e:
                    logger.debug(f"处理股票 {row.get('ts_code', 'unknown')} 失败: {e}")
                    continue
            
            # 批量插入到数据库
            if data_to_insert:
                count = self.db_manager.insert_daily_quotes(data_to_insert)
                logger.info(f"✓ 成功更新 {len(data_to_insert)} 条日线数据")
            
            return True
            
        except Exception as e:
            logger.error(f"处理日线数据失败: {e}")
            return False
    
    def update_daily_basic(self, date: str) -> bool:
        """更新每日基本面数据"""
        filename = f"daily_basic_{date}.csv"
        save_path = self.temp_dir / filename
        
        if not self.download_file(filename, save_path):
            return False
        
        try:
            df = pd.read_csv(save_path)
            logger.info(f"读取到 {len(df)} 条基本面数据")
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                success_count = 0
                for _, row in df.iterrows():
                    try:
                        code = row['ts_code'].split('.')[0]
                        
                        # 获取security_id
                        cursor.execute("SELECT id FROM securities WHERE code = ?", (code,))
                        result = cursor.fetchone()
                        if not result:
                            continue
                        
                        security_id = result[0]
                        trade_date = pd.to_datetime(str(row['trade_date']), format='%Y%m%d').strftime('%Y-%m-%d')
                        
                        # 插入或更新基本面数据
                        cursor.execute("""
                            INSERT OR REPLACE INTO daily_basic 
                            (security_id, trade_date, pe_ttm, pb, ps_ttm, dv_ttm, 
                             total_mv, circ_mv, turnover_rate, turnover_rate_f, 
                             volume_ratio, pe, dv_ratio)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            security_id, trade_date,
                            row.get('pe_ttm'), row.get('pb'), row.get('ps_ttm'), row.get('dv_ttm'),
                            row.get('total_mv'), row.get('circ_mv'), row.get('turnover_rate'),
                            row.get('turnover_rate_f'), row.get('volume_ratio'),
                            row.get('pe'), row.get('dv_ratio')
                        ))
                        success_count += 1
                        
                    except Exception as e:
                        logger.debug(f"处理基本面数据失败 {row.get('ts_code', 'unknown')}: {e}")
                        continue
                
                conn.commit()
            
            logger.info(f"✓ 成功更新 {success_count} 条基本面数据")
            return True
            
        except Exception as e:
            logger.error(f"处理基本面数据失败: {e}")
            return False
    
    def update_index_daily(self, date: str) -> bool:
        """更新指数日线数据 - 支持10个重要指数"""
        filename = f"index_daily_{date}.csv"
        save_path = self.temp_dir / filename
        
        if not self.download_file(filename, save_path):
            return False
        
        try:
            df = pd.read_csv(save_path)
            
            # 只处理重要指数
            important_indices = list(self.index_mapping.keys())
            df = df[df['ts_code'].isin(important_indices)]
            
            if df.empty:
                logger.warning("没有找到重要指数数据")
                return False
            
            logger.info(f"读取到 {len(df)} 条重要指数数据")
            
            data_to_insert = []
            
            for _, row in df.iterrows():
                try:
                    code = row['ts_code']
                    name = self.index_mapping.get(code, code)
                    exchange = code.split('.')[1] if '.' in code else 'SH'
                    
                    # 插入或获取指数的security_id
                    security_id = self.db_manager.insert_security(
                        code=code,
                        name=name,
                        security_type='指数',
                        exchange=exchange
                    )
                    
                    if security_id:
                        trade_date = pd.to_datetime(str(row['trade_date']), format='%Y%m%d').strftime('%Y-%m-%d')
                        
                        data_to_insert.append({
                            'security_id': security_id,
                            'trade_date': trade_date,
                            'open': row.get('open', 0),
                            'high': row.get('high', 0),
                            'low': row.get('low', 0),
                            'close': row.get('close', 0),
                            'volume': row.get('vol', 0) if 'vol' in row else row.get('volume', 0),
                            'price_change_pct': row.get('pct_chg', 0) / 100 if row.get('pct_chg') else 0,
                            'is_limit_up': False,  # 指数没有涨跌停
                            'is_limit_down': False
                        })
                        
                except Exception as e:
                    logger.debug(f"处理指数 {row.get('ts_code', 'unknown')} 失败: {e}")
                    continue
            
            # 批量插入到数据库
            if data_to_insert:
                count = self.db_manager.insert_daily_quotes(data_to_insert)
                logger.info(f"✓ 成功更新 {len(data_to_insert)} 条指数数据")
            
            return True
            
        except Exception as e:
            logger.error(f"处理指数数据失败: {e}")
            return False
    
    def update_stock_basic(self) -> bool:
        """更新股票基本信息"""
        filename = "stock_basic.csv"
        save_path = self.temp_dir / filename
        
        if not self.download_file(filename, save_path):
            return False
        
        try:
            df = pd.read_csv(save_path)
            logger.info(f"读取到 {len(df)} 条股票基本信息")
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                success_count = 0
                for _, row in df.iterrows():
                    try:
                        code = row['ts_code'].split('.')[0]
                        
                        # 更新股票基本信息表
                        cursor.execute("""
                            INSERT OR REPLACE INTO stock_basic_info 
                            (code, name, industry, area, market, list_date)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            code,
                            row.get('name', code),
                            row.get('industry', ''),
                            row.get('area', ''),
                            row.get('market', ''),
                            pd.to_datetime(str(row.get('list_date', '')), format='%Y%m%d').strftime('%Y-%m-%d') 
                            if row.get('list_date') else None
                        ))
                        
                        # 同时更新securities表的名称
                        cursor.execute("""
                            UPDATE securities SET name = ? WHERE code = ?
                        """, (row.get('name', code), code))
                        
                        success_count += 1
                        
                    except Exception as e:
                        logger.debug(f"处理股票信息失败 {row.get('ts_code', 'unknown')}: {e}")
                        continue
                
                conn.commit()
            
            logger.info(f"✓ 成功更新 {success_count} 条股票基本信息")
            return True
            
        except Exception as e:
            logger.error(f"处理股票基本信息失败: {e}")
            return False
    
    def calculate_technical_indicators(self, date: str):
        """计算技术指标 - 与日常更新保持一致"""
        logger.info("计算技术指标...")
        
        try:
            # 调用现有的技术指标计算脚本
            from data_adapter.technical_indicator_calculator import TechnicalIndicatorCalculator
            
            calculator = TechnicalIndicatorCalculator()
            calculator.calculate_all_indicators(date)
            logger.info("✓ 技术指标计算完成")
            
        except ImportError:
            logger.warning("技术指标计算模块未找到，跳过")
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
    
    def run_full_update(self, date: Optional[str] = None):
        """运行完整更新流程 - 与run_daily_update.sh保持一致"""
        logger.info("=" * 60)
        logger.info("Tushare备份数据更新工具")
        logger.info("=" * 60)
        
        # 获取要更新的日期
        if date:
            dates = [date]
        else:
            dates = self.get_available_dates()[:1]  # 只更新最新日期
        
        if not dates:
            logger.error("无法获取有效日期")
            return
        
        for update_date in dates:
            logger.info(f"\n📅 更新日期: {update_date}")
            logger.info("-" * 40)
            
            start_time = time.time()
            
            # 1. 更新股票基本信息（如果需要）
            logger.info("\n[1/5] 更新股票基本信息...")
            self.update_stock_basic()
            
            # 2. 更新日线数据
            logger.info("\n[2/5] 更新日线行情数据...")
            self.update_daily_quotes(update_date)
            
            # 3. 更新基本面数据
            logger.info("\n[3/5] 更新每日基本面数据...")
            self.update_daily_basic(update_date)
            
            # 4. 更新指数数据
            logger.info("\n[4/5] 更新大盘指数数据...")
            self.update_index_daily(update_date)
            
            # 5. 计算技术指标
            logger.info("\n[5/5] 计算技术指标...")
            self.calculate_technical_indicators(update_date)
            
            elapsed_time = time.time() - start_time
            logger.info(f"\n⏱  更新完成，耗时: {elapsed_time:.2f} 秒")
        
        # 清理临时文件
        logger.info("\n🧹 清理临时文件...")
        for file in self.temp_dir.glob("*.csv"):
            try:
                file.unlink()
            except:
                pass
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Tushare备份数据更新完成！")
        logger.info("=" * 60)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tushare备份数据更新工具')
    parser.add_argument('--date', type=str, help='指定日期 (YYYYMMDD格式，如: 20250818)')
    parser.add_argument('--type', type=str, 
                       choices=['daily', 'basic', 'index', 'stock_info', 'all'], 
                       default='all', help='更新类型')
    
    args = parser.parse_args()
    
    updater = TushareBackupUpdater()
    
    if args.type == 'all':
        updater.run_full_update(args.date)
    else:
        if not args.date:
            args.date = updater.get_available_dates()[0]
            
        if args.type == 'daily':
            updater.update_daily_quotes(args.date)
        elif args.type == 'basic':
            updater.update_daily_basic(args.date)
        elif args.type == 'index':
            updater.update_index_daily(args.date)
        elif args.type == 'stock_info':
            updater.update_stock_basic()
        
        logger.info(f"✅ {args.type} 数据更新完成")

if __name__ == "__main__":
    main()