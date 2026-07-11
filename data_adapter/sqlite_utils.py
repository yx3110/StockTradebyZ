#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite 轻量共享工具: 连接工厂 + 并发写重试

背景: 多个回填/预计算任务并发写 stock_data.db 时, busy_timeout 之外仍可能抛
"database is locked" (锁升级死锁场景 busy handler 不生效)。此前各脚本各自
复制了一份重试逻辑, 统一收敛到这里。
"""

import sqlite3
import time
import logging

logger = logging.getLogger(__name__)


def connect(db_path: str, busy_timeout_ms: int = 30000) -> sqlite3.Connection:
    """打开连接并设置 busy_timeout (项目规范: 至少 30 秒)"""
    conn = sqlite3.connect(db_path)
    conn.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')
    return conn


def write_retry(fn, attempts: int = 8):
    """执行写操作, 遇 "database is locked" 指数退避重试.

    用法: write_retry(lambda: (conn.executemany(sql, rows), conn.commit()))
    """
    for a in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if 'locked' not in str(e) or a == attempts - 1:
                raise
            logger.warning("DB locked, %ds 后重试 %d/%d", 5 * (a + 1), a + 1, attempts - 1)
            time.sleep(5 * (a + 1))
