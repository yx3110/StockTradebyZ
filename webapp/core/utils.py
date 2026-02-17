"""
工具函数
"""
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any


def format_date(date_str: str, input_format: str = '%Y-%m-%d', output_format: str = '%Y-%m-%d') -> str:
    """
    格式化日期字符串

    Args:
        date_str: 输入日期字符串
        input_format: 输入格式
        output_format: 输出格式

    Returns:
        格式化后的日期字符串
    """
    try:
        dt = datetime.strptime(date_str, input_format)
        return dt.strftime(output_format)
    except Exception:
        return date_str


def parse_markdown_table(content: str) -> List[Dict[str, str]]:
    """
    解析Markdown表格

    Args:
        content: Markdown内容

    Returns:
        表格数据列表
    """
    lines = content.strip().split('\n')
    if len(lines) < 2:
        return []

    # 第一行是表头
    headers = [h.strip() for h in lines[0].split('|')[1:-1]]

    # 第二行是分隔符（跳过）
    # 后续行是数据
    data = []
    for line in lines[2:]:
        if not line.strip() or line.strip().startswith('|---'):
            continue

        cells = [c.strip() for c in line.split('|')[1:-1]]
        if len(cells) == len(headers):
            row = dict(zip(headers, cells))
            data.append(row)

    return data


def ensure_directory(path: Path) -> Path:
    """
    确保目录存在

    Args:
        path: 目录路径

    Returns:
        目录路径
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_date_range(end_date: str, days: int = 30) -> List[str]:
    """
    获取日期范围

    Args:
        end_date: 结束日期（YYYY-MM-DD）
        days: 天数

    Returns:
        日期列表
    """
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    dates = []

    for i in range(days):
        date = end_dt - timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))

    return dates


def format_number(value: float, precision: int = 2, unit: str = '') -> str:
    """
    格式化数字

    Args:
        value: 数值
        precision: 精度
        unit: 单位

    Returns:
        格式化后的字符串
    """
    if value is None:
        return 'N/A'

    if abs(value) >= 1e8:
        return f'{value / 1e8:.{precision}f}亿{unit}'
    elif abs(value) >= 1e4:
        return f'{value / 1e4:.{precision}f}万{unit}'
    else:
        return f'{value:.{precision}f}{unit}'


def format_percentage(value: float, precision: int = 2) -> str:
    """
    格式化百分比

    Args:
        value: 数值
        precision: 精度

    Returns:
        百分比字符串
    """
    if value is None:
        return 'N/A'

    return f'{value:.{precision}f}%'


def extract_stock_code(text: str) -> str:
    """
    从文本中提取股票代码

    Args:
        text: 文本

    Returns:
        股票代码，如果没找到返回空字符串
    """
    match = re.search(r'\d{6}', text)
    return match.group(0) if match else ''


def calculate_duration(start_time: datetime, end_time: datetime = None) -> str:
    """
    计算持续时间

    Args:
        start_time: 开始时间
        end_time: 结束时间（None表示当前时间）

    Returns:
        持续时间字符串（例如: "2小时30分钟"）
    """
    if end_time is None:
        end_time = datetime.now()

    delta = end_time - start_time
    seconds = delta.total_seconds()

    if seconds < 60:
        return f'{int(seconds)}秒'
    elif seconds < 3600:
        return f'{int(seconds / 60)}分钟'
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f'{hours}小时{minutes}分钟'


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全转换为浮点数

    Args:
        value: 输入值
        default: 默认值

    Returns:
        浮点数
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    安全转换为整数

    Args:
        value: 输入值
        default: 默认值

    Returns:
        整数
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
