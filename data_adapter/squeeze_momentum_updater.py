#!/usr/bin/env python3
"""
挤压动量指标数据库更新器 - v3.2专用
基于v4.0的挤压动量算法，为数据库中的股票计算并更新挤压动量相关指标
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scoring_improvements'))

try:
    from .database_manager import DatabaseManager
except ImportError:
    from database_manager import DatabaseManager

try:
    from squeeze_momentum_calculator import SqueezeMomentumCalculator
except ImportError:
    SqueezeMomentumCalculator = None

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SqueezeMomentumUpdater:
    """挤压动量指标数据库更新器"""
    
    def __init__(self):
        if SqueezeMomentumCalculator is None:
            raise ImportError("squeeze_momentum_calculator 模块未找到 (scoring_improvements/ 目录已移除)")
        self.db_manager = DatabaseManager()
        self.squeeze_calculator = SqueezeMomentumCalculator()
        self.batch_size = 100  # 批处理大小
        
    def get_securities_to_update(self, start_date: str = None, end_date: str = None) -> List[Dict]:
        """获取需要更新的证券列表"""
        query = """
        SELECT DISTINCT s.id, s.code, s.name, s.type 
        FROM securities s
        JOIN daily_quotes dq ON s.id = dq.security_id
        WHERE s.is_active = 1 AND s.type = 'A股'
        """
        
        params = []
        if start_date:
            query += " AND dq.trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND dq.trade_date <= ?"
            params.append(end_date)
            
        query += " ORDER BY s.code"
        
        results = self.db_manager.execute_query(query, params if params else None)
        return [dict(row) for row in results]
    
    def get_stock_ohlc_data(self, security_id: int, start_date: str = None, 
                           end_date: str = None, min_periods: int = 60) -> Optional[pd.DataFrame]:
        """获取股票的OHLC数据"""
        query = """
        SELECT trade_date, open, high, low, close, volume
        FROM daily_quotes
        WHERE security_id = ?
        """
        
        params = [security_id]
        if start_date:
            query += " AND trade_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trade_date <= ?"
            params.append(end_date)
            
        query += " ORDER BY trade_date"
        
        with self.db_manager.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
        if len(df) < min_periods:
            logger.warning(f"股票 {security_id} 数据不足 ({len(df)} < {min_periods})，跳过")
            return None
            
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df.set_index('trade_date', inplace=True)
        
        return df
    
    def calculate_squeeze_indicators_for_stock(self, security_id: int, 
                                             start_date: str = None, 
                                             end_date: str = None) -> bool:
        """为单只股票计算挤压动量指标"""
        try:
            # 获取OHLC数据
            df = self.get_stock_ohlc_data(security_id, start_date, end_date)
            if df is None or len(df) == 0:
                return False
            
            # 计算挤压动量指标
            indicators = self.squeeze_calculator.calculate_squeeze_momentum_indicators(
                high=df['high'],
                low=df['low'], 
                close=df['close']
            )
            
            if not indicators:
                logger.warning(f"股票 {security_id} 挤压动量指标计算失败")
                return False
            
            # 准备更新数据
            update_data = []
            for i, date in enumerate(df.index):
                date_str = date.strftime('%Y-%m-%d')
                
                # 安全获取指标值的辅助函数
                def safe_get(series_or_array, index, default=None):
                    try:
                        if hasattr(series_or_array, 'iloc'):
                            value = series_or_array.iloc[index]
                        else:
                            value = series_or_array[index]
                        
                        # 处理numpy类型和NaN值
                        if pd.isna(value) or np.isnan(float(value)):
                            return default
                        return float(value)
                    except (IndexError, KeyError, TypeError, ValueError):
                        return default
                
                # 提取各个指标值
                record = {
                    'security_id': security_id,
                    'trade_date': date_str,
                    'kc_upper': safe_get(indicators['kc_upper'], i),
                    'kc_middle': safe_get(indicators['kc_middle'], i),
                    'kc_lower': safe_get(indicators['kc_lower'], i),
                    'kc_width': safe_get(indicators['kc_width'], i),
                    'squeeze_state': bool(safe_get(indicators['squeeze_state'], i, False)),
                    'squeeze_release': bool(safe_get(indicators['squeeze_release'], i, False)),
                    'squeeze_intensity': safe_get(indicators['squeeze_intensity'], i, 1.0),
                    'squeeze_days': int(safe_get(indicators['squeeze_days'], i, 0)),
                    'recent_releases': int(safe_get(indicators['recent_releases'], i, 0)),
                    'squeeze_momentum': safe_get(indicators['momentum'], i, 0.0),
                    'momentum_direction': int(safe_get(indicators['momentum_direction'], i, 0)),
                    'momentum_strength': safe_get(indicators['momentum_strength'], i, 0.0),
                    'momentum_acceleration': safe_get(indicators['momentum_acceleration'], i, 0.0),
                    'momentum_consistency': safe_get(indicators['momentum_consistency'], i, 0.0)
                }
                
                update_data.append(record)
            
            # 批量更新数据库
            self._batch_update_squeeze_indicators(update_data)
            
            return True
            
        except Exception as e:
            logger.error(f"计算股票 {security_id} 挤压动量指标时出错: {e}")
            return False
    
    def _batch_update_squeeze_indicators(self, update_data: List[Dict]):
        """批量更新挤压动量指标到数据库"""
        if not update_data:
            return
            
        # 构建更新SQL
        update_sql = """
        UPDATE technical_indicators 
        SET 
            kc_upper = ?, kc_middle = ?, kc_lower = ?, kc_width = ?,
            squeeze_state = ?, squeeze_release = ?, squeeze_intensity = ?,
            squeeze_days = ?, recent_releases = ?,
            squeeze_momentum = ?, momentum_direction = ?, momentum_strength = ?,
            momentum_acceleration = ?, momentum_consistency = ?
        WHERE security_id = ? AND trade_date = ?
        """
        
        # 准备批量更新数据
        batch_data = []
        for record in update_data:
            batch_data.append((
                record['kc_upper'], record['kc_middle'], record['kc_lower'], record['kc_width'],
                record['squeeze_state'], record['squeeze_release'], record['squeeze_intensity'],
                record['squeeze_days'], record['recent_releases'],
                record['squeeze_momentum'], record['momentum_direction'], record['momentum_strength'],
                record['momentum_acceleration'], record['momentum_consistency'],
                record['security_id'], record['trade_date']
            ))
        
        # 执行批量更新
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(update_sql, batch_data)
            updated_rows = cursor.rowcount
            conn.commit()
            
        logger.info(f"✅ 更新了 {updated_rows} 条挤压动量指标记录")
    
    def update_all_securities(self, start_date: str = None, end_date: str = None, 
                            max_securities: int = None) -> Dict:
        """更新所有证券的挤压动量指标"""
        logger.info("🚀 开始更新所有证券的挤压动量指标...")
        
        # 获取证券列表
        securities = self.get_securities_to_update(start_date, end_date)
        
        if max_securities:
            securities = securities[:max_securities]
            
        logger.info(f"📊 找到 {len(securities)} 只股票需要更新")
        
        # 统计信息
        stats = {
            'total_securities': len(securities),
            'processed_securities': 0,
            'successful_securities': 0,
            'failed_securities': 0,
            'start_time': datetime.now()
        }
        
        # 逐个处理证券
        for i, security in enumerate(securities, 1):
            try:
                logger.info(f"📈 [{i}/{len(securities)}] 处理 {security['code']} - {security['name']}")
                
                success = self.calculate_squeeze_indicators_for_stock(
                    security['id'], start_date, end_date
                )
                
                if success:
                    stats['successful_securities'] += 1
                    logger.info(f"✅ {security['code']} 处理完成")
                else:
                    stats['failed_securities'] += 1
                    logger.warning(f"⚠️ {security['code']} 处理失败")
                    
                stats['processed_securities'] += 1
                
                # 每100只股票报告一次进度
                if i % 100 == 0:
                    elapsed = datetime.now() - stats['start_time']
                    rate = i / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                    logger.info(f"📊 已处理 {i}/{len(securities)} ({i/len(securities)*100:.1f}%)，速度: {rate:.2f} 股票/秒")
                
            except Exception as e:
                logger.error(f"❌ 处理 {security['code']} 时发生错误: {e}")
                stats['failed_securities'] += 1
                stats['processed_securities'] += 1
        
        # 完成统计
        stats['end_time'] = datetime.now()
        stats['duration'] = stats['end_time'] - stats['start_time']
        
        logger.info("🎉 挤压动量指标更新完成！")
        logger.info(f"📊 处理统计:")
        logger.info(f"  - 总计: {stats['total_securities']} 只股票")
        logger.info(f"  - 成功: {stats['successful_securities']} 只")
        logger.info(f"  - 失败: {stats['failed_securities']} 只")
        logger.info(f"  - 耗时: {stats['duration']}")
        
        return stats
    
    def update_recent_data(self, days: int = 30) -> Dict:
        """更新最近N天的数据"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        logger.info(f"🔄 更新最近 {days} 天的挤压动量指标 ({start_date} 到 {end_date})")
        
        return self.update_all_securities(start_date, end_date)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='挤压动量指标数据库更新器')
    parser.add_argument('--start-date', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--recent-days', type=int, help='更新最近N天')
    parser.add_argument('--max-securities', type=int, help='最大处理股票数')
    parser.add_argument('--test-mode', action='store_true', help='测试模式(只处理前10只股票)')
    
    args = parser.parse_args()
    
    updater = SqueezeMomentumUpdater()
    
    if args.test_mode:
        logger.info("🧪 测试模式：只处理前10只股票")
        stats = updater.update_all_securities(
            args.start_date, args.end_date, max_securities=10
        )
    elif args.recent_days:
        stats = updater.update_recent_data(args.recent_days)
    else:
        stats = updater.update_all_securities(
            args.start_date, args.end_date, args.max_securities
        )
    
    logger.info("✨ 程序执行完成！")

if __name__ == "__main__":
    main()