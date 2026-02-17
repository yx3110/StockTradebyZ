#!/usr/bin/env python3
"""
CSV数据迁移工具
将现有的CSV格式股票数据迁移到SQLite数据库
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_adapter/migration.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class CSVMigrationTool:
    """CSV数据迁移工具"""
    
    def __init__(self, csv_dir: str, db_manager: DatabaseManager):
        """
        初始化迁移工具
        
        Args:
            csv_dir: CSV文件目录
            db_manager: 数据库管理器
        """
        self.csv_dir = Path(csv_dir)
        self.db = db_manager
        self.security_id_cache = {}  # 缓存security_id映射
        
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """
        解析文件名，提取股票代码和类型
        
        Args:
            filename: 文件名，如 "000001_A股.csv"
            
        Returns:
            解析结果: {'code': '000001', 'type': 'A股', 'exchange': 'SZ'}
        """
        # 排除非股票数据文件
        if filename.startswith('securities_list'):
            return None
            
        # 匹配模式：代码_类型.csv
        pattern = r'(\d{6})_(.+)\.csv'
        match = re.match(pattern, filename)
        
        if not match:
            logger.warning(f"无法解析文件名: {filename}")
            return None
        
        code, security_type = match.groups()
        
        # 推断交易所
        exchange = 'SH' if code.startswith(('60', '68', '51')) else 'SZ'
        
        return {
            'code': code,
            'type': security_type,
            'exchange': exchange
        }
    
    def get_or_create_security(self, code: str, name: str, security_type: str, exchange: str) -> int:
        """获取或创建证券记录，返回security_id"""
        if code in self.security_id_cache:
            return self.security_id_cache[code]
        
        # 先尝试查询现有记录
        query = "SELECT id FROM securities WHERE code = ?"
        result = self.db.execute_query(query, (code,))
        
        if result:
            security_id = result[0]['id']
        else:
            # 创建新记录
            security_id = self.db.insert_security(code, name, security_type, exchange)
        
        self.security_id_cache[code] = security_id
        return security_id
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗价格数据"""
        # 确保数据类型正确
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除无效数据
        df = df.dropna(subset=['close'])
        df = df[df['close'] > 0]
        df = df[df['volume'] >= 0]
        
        # 确保high >= low
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df['close']]
        df = df[df['low'] <= df['close']]
        
        # 按日期排序
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # 计算涨跌幅
        df['prev_close'] = df['close'].shift(1)
        df['price_change_pct'] = (df['close'] - df['prev_close']) / df['prev_close']
        df['price_change_pct'] = df['price_change_pct'].fillna(0)
        
        # 标记涨跌停
        df['is_limit_up'] = df['price_change_pct'] >= 0.095
        df['is_limit_down'] = df['price_change_pct'] <= -0.095
        
        return df
    
    def process_single_file(self, csv_file: Path) -> Dict[str, int]:
        """处理单个CSV文件"""
        file_info = self.parse_filename(csv_file.name)
        if not file_info:
            return {'status': 'skipped', 'records': 0}
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            if df.empty:
                logger.warning(f"文件为空: {csv_file}")
                return {'status': 'empty', 'records': 0}
            
            # 验证必要列
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"文件缺少必要列 {missing_columns}: {csv_file}")
                return {'status': 'error', 'records': 0}
            
            # 清洗数据
            df = self.clean_price_data(df)
            if df.empty:
                logger.warning(f"清洗后数据为空: {csv_file}")
                return {'status': 'empty_after_clean', 'records': 0}
            
            # 获取或创建证券记录
            # 从数据中推断股票名称（如果有的话）
            stock_name = file_info['code']  # 默认使用代码作为名称
            security_id = self.get_or_create_security(
                file_info['code'], 
                stock_name, 
                file_info['type'], 
                file_info['exchange']
            )
            
            # 准备插入数据
            records = []
            for _, row in df.iterrows():
                record = {
                    'security_id': security_id,
                    'trade_date': row['date'].strftime('%Y-%m-%d'),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'price_change_pct': float(row['price_change_pct']),
                    'is_limit_up': bool(row['is_limit_up']),
                    'is_limit_down': bool(row['is_limit_down'])
                }
                records.append(record)
            
            # 批量插入数据库
            inserted_count = self.db.insert_daily_quotes(records)
            
            logger.info(f"处理完成 {csv_file.name}: {len(records)} 条记录")
            return {'status': 'success', 'records': len(records)}
            
        except Exception as e:
            logger.error(f"处理文件失败 {csv_file}: {e}")
            return {'status': 'error', 'records': 0, 'error': str(e)}
    
    def migrate_all_files(self, max_workers: int = 4, batch_size: int = 100) -> Dict[str, int]:
        """
        迁移所有CSV文件
        
        Args:
            max_workers: 并发线程数
            batch_size: 批次大小
            
        Returns:
            迁移统计结果
        """
        start_time = time.time()
        logger.info(f"开始迁移CSV文件到数据库...")
        
        # 获取所有CSV文件
        csv_files = list(self.csv_dir.glob("*.csv"))
        csv_files = [f for f in csv_files if not f.name.startswith('securities_list')]
        
        logger.info(f"找到 {len(csv_files)} 个CSV文件")
        
        # 统计信息
        stats = {
            'total_files': len(csv_files),
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'total_records': 0,
            'errors': []
        }
        
        # 分批处理文件
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}: {len(batch_files)} 个文件")
            
            # 并发处理批次文件
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_single_file, csv_file): csv_file
                    for csv_file in batch_files
                }
                
                for future in as_completed(future_to_file):
                    csv_file = future_to_file[future]
                    stats['processed_files'] += 1
                    
                    try:
                        result = future.result()
                        
                        if result['status'] == 'success':
                            stats['successful_files'] += 1
                            stats['total_records'] += result['records']
                        elif result['status'] == 'error':
                            stats['failed_files'] += 1
                            stats['errors'].append(f"{csv_file.name}: {result.get('error', 'Unknown error')}")
                        else:
                            stats['skipped_files'] += 1
                            
                    except Exception as e:
                        stats['failed_files'] += 1
                        stats['errors'].append(f"{csv_file.name}: {str(e)}")
                        logger.error(f"文件处理异常 {csv_file}: {e}")
                    
                    # 进度更新
                    if stats['processed_files'] % 50 == 0:
                        progress = stats['processed_files'] / stats['total_files'] * 100
                        logger.info(f"迁移进度: {progress:.1f}% ({stats['processed_files']}/{stats['total_files']})")
        
        # 记录迁移结果
        duration = time.time() - start_time
        status = 'SUCCESS' if stats['failed_files'] == 0 else 'PARTIAL'
        
        self.db.log_data_update(
            update_type='MIGRATION',
            securities_updated=stats['successful_files'],
            records_added=stats['total_records'],
            status=status,
            duration=duration
        )
        
        logger.info(f"迁移完成!")
        logger.info(f"处理文件: {stats['processed_files']}/{stats['total_files']}")
        logger.info(f"成功: {stats['successful_files']}, 失败: {stats['failed_files']}, 跳过: {stats['skipped_files']}")
        logger.info(f"总记录数: {stats['total_records']:,}")
        logger.info(f"耗时: {duration:.1f} 秒")
        
        if stats['errors']:
            logger.warning(f"错误列表 ({len(stats['errors'])} 个):")
            for error in stats['errors'][:10]:  # 只显示前10个错误
                logger.warning(f"  - {error}")
        
        return stats
    
    def update_securities_info(self, securities_list_file: Optional[str] = None):
        """
        更新证券基本信息（从securities_list.csv读取名称等）
        
        Args:
            securities_list_file: 证券列表文件路径
        """
        if securities_list_file is None:
            securities_list_file = self.csv_dir / "securities_list.csv"
        
        if not Path(securities_list_file).exists():
            logger.warning(f"证券列表文件不存在: {securities_list_file}")
            return
        
        try:
            df = pd.read_csv(securities_list_file)
            logger.info(f"更新证券信息: {len(df)} 条记录")
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    if 'ts_code' in df.columns and 'name' in df.columns:
                        code = row['ts_code'].split('.')[0]  # 提取股票代码
                        name = row['name']
                        
                        cursor.execute("""
                            UPDATE securities 
                            SET name = ?, updated_at = CURRENT_TIMESTAMP
                            WHERE code = ?
                        """, (name, code))
                
                conn.commit()
                updated_count = cursor.rowcount
                logger.info(f"更新了 {updated_count} 条证券信息")
                
        except Exception as e:
            logger.error(f"更新证券信息失败: {e}")


def main():
    """主函数：执行CSV迁移"""
    # 初始化数据库
    db = DatabaseManager()
    
    # 创建迁移工具
    migrator = CSVMigrationTool("full_securities_data", db)
    
    # 执行迁移
    stats = migrator.migrate_all_files(max_workers=4, batch_size=50)
    
    # 更新证券信息
    migrator.update_securities_info()
    
    # 优化数据库
    db.optimize_database()
    
    # 显示最终统计
    final_stats = db.get_database_stats()
    print("\n=== 迁移完成统计 ===")
    print(f"数据库文件大小: {final_stats['db_size_mb']:.2f} MB")
    print(f"证券总数: {final_stats['total_securities']}")
    print(f"行情记录总数: {final_stats['total_quotes']:,}")
    print(f"数据日期范围: {final_stats['date_range']['start']} 至 {final_stats['date_range']['end']}")


if __name__ == "__main__":
    main()