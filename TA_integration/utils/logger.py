#!/usr/bin/env python3
"""
日志配置工具
"""

import logging
import os
from datetime import datetime

def setup_logger(name: str, verbose: bool = False, log_file: str = None):
    """设置日志记录器"""
    
    # 创建日志目录
    log_dir = "TA_integration/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志级别
    level = logging.DEBUG if verbose else logging.INFO
    
    # 创建记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file is None:
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger