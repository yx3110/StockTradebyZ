#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick debug script to find the training error
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent))

from ml_models.v39 import V390EnhancedFeatureMLSystem

try:
    # Initialize system
    logger.info("初始化V3.9系统...")
    system = V390EnhancedFeatureMLSystem(lookback_days=10, lookahead_days=5)

    # Get sample stocks
    import sqlite3
    conn = sqlite3.connect('data_adapter/stock_data.db')  # Use correct database path
    cursor = conn.cursor()
    cursor.execute("""
        SELECT code FROM securities
        WHERE type='A股'
        ORDER BY RANDOM()
        LIMIT 5
    """)
    sample_stocks = [row[0] for row in cursor.fetchall()]
    conn.close()

    logger.info(f"测试股票: {sample_stocks}")

    # Try to prepare training data
    logger.info("准备训练数据...")
    X_train, y_train, info_list = system.prepare_training_data(
        start_date='2024-10-01',
        end_date='2024-10-31',  # Just one month for testing
        sample_stocks=sample_stocks
    )

    if X_train is None:
        logger.error("❌ X_train is None - 这就是问题所在!")
        logger.error(f"y_train: {y_train}")
        logger.error(f"info_list: {info_list}")
    else:
        logger.info(f"✅ 成功！训练样本数: {len(X_train)}, 特征数: {X_train.shape[1]}")

except Exception as e:
    logger.error(f"错误: {e}")
    import traceback
    traceback.print_exc()
