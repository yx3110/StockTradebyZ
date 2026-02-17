#!/usr/bin/env python3
"""
股票相似度回测系统主程序
Main program for Stock Similarity Backtest System
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import json
import yaml

# 添加父目录到系统路径
sys.path.append(str(Path(__file__).parent.parent))

from data_adapter.database_manager import DatabaseManager


def setup_logging(config):
    """设置日志系统"""
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO'))
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = []
    if config.get('logging', {}).get('console_output', True):
        handlers.append(logging.StreamHandler())
    
    if config.get('logging', {}).get('file_output', True):
        log_dir = Path(config.get('logging', {}).get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / config.get('logging', {}).get('log_file', 'similarity_backtest.log')
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def run_single_stock_analysis(stock_code, query_date, window_length, config, logger):
    """运行单只股票的相似度分析"""
    logger.info(f"开始分析股票 {stock_code}，查询日期 {query_date}，窗口长度 {window_length}")
    
    try:
        from algorithms.search_engine import SimilaritySearchEngine
        engine = SimilaritySearchEngine(config)
        results = engine.search_similar_patterns(stock_code, query_date, window_length)
        return results
    except Exception as e:
        logger.error(f"相似度搜索失败: {str(e)}")
        # 返回错误信息
        return {
            "query_stock": stock_code,
            "query_date": query_date,
            "window_length": window_length,
            "status": "error",
            "message": f"相似度搜索失败: {str(e)}"
        }


def run_batch_analysis(stock_list, start_date, end_date, config, logger):
    """批量分析多只股票"""
    logger.info(f"开始批量分析 {len(stock_list)} 只股票")
    
    results = []
    for stock in stock_list:
        result = run_single_stock_analysis(stock, end_date, 
                                          config['similarity']['default_window'], 
                                          config, logger)
        results.append(result)
    
    return results


def save_results(results, output_path, format='json'):
    """保存分析结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    elif format == 'yaml':
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, allow_unicode=True)


def check_database_connection(config):
    """检查数据库连接"""
    try:
        db_path = Path(config['data']['database_path'])
        if not db_path.exists():
            # 尝试在父目录查找
            db_path = Path(__file__).parent.parent / 'stock_data.db'
        
        if db_path.exists():
            db = DatabaseManager(str(db_path))
            stats = db.get_database_stats()
            return True, f"数据库连接成功，包含 {stats.get('total_securities', 0)} 只证券"
        else:
            return False, f"数据库文件不存在: {db_path}"
    except Exception as e:
        return False, f"数据库连接失败: {str(e)}"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股票相似度回测系统')
    
    # 基础参数
    parser.add_argument('--stock', type=str, help='股票代码，如 000001')
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                       help='查询日期，格式 YYYY-MM-DD')
    parser.add_argument('--window', type=int, default=30,
                       help='窗口长度（天数），默认30')
    
    # 批量分析参数
    parser.add_argument('--batch', action='store_true', help='批量分析模式')
    parser.add_argument('--stock-list', nargs='+', help='股票列表')
    parser.add_argument('--stock-file', type=str, help='包含股票代码的文件')
    parser.add_argument('--start-date', type=str, help='开始日期')
    parser.add_argument('--end-date', type=str, help='结束日期')
    
    # 配置参数
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--format', choices=['json', 'yaml', 'html'], 
                       default='json', help='输出格式')
    
    # 其他参数
    parser.add_argument('--check-db', action='store_true', help='仅检查数据库连接')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    logger = setup_logging(config)
    
    # 检查数据库连接
    if args.check_db:
        success, message = check_database_connection(config)
        if success:
            logger.info(message)
            print(f"✅ {message}")
        else:
            logger.error(message)
            print(f"❌ {message}")
        return 0 if success else 1
    
    # 执行分析
    try:
        if args.batch:
            # 批量分析模式
            stock_list = args.stock_list or []
            
            if args.stock_file:
                with open(args.stock_file, 'r') as f:
                    stock_list.extend([line.strip() for line in f if line.strip()])
            
            if not stock_list:
                logger.error("批量模式需要提供股票列表")
                return 1
            
            end_date = args.end_date or args.date
            start_date = args.start_date or (
                datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)
            ).strftime('%Y-%m-%d')
            
            results = run_batch_analysis(stock_list, start_date, end_date, config, logger)
            
        else:
            # 单只股票分析模式
            if not args.stock:
                logger.error("请提供股票代码 (--stock)")
                return 1
            
            results = run_single_stock_analysis(
                args.stock, args.date, args.window, config, logger
            )
        
        # 保存结果
        if args.output:
            save_results(results, args.output, args.format)
            logger.info(f"结果已保存到 {args.output}")
        else:
            # 输出到控制台
            print(json.dumps(results, ensure_ascii=False, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"执行失败: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())