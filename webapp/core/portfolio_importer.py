"""
CSV持仓导入器

支持东方财富/同花顺等常见券商导出的CSV格式
- 自动检测GBK/UTF-8编码
- 模糊匹配列名
- 智能合并已有持仓
"""
import csv
import io
import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 常见列名映射 (券商导出格式差异)
CODE_COLUMNS = ['证券代码', '股票代码', '代码', '证券编码']
NAME_COLUMNS = ['证券名称', '股票名称', '名称', '证券简称']
QUANTITY_COLUMNS = ['持仓数量', '股份余额', '数量', '持仓', '库存数量', '当前数量']
COST_COLUMNS = ['成本价', '参考成本价', '购入价', '买入均价', '成本', '摊薄成本价']
PRICE_COLUMNS = ['当前价', '市价', '现价', '最新价', '市场价']
MARKET_VALUE_COLUMNS = ['市值', '参考市值', '最新市值', '证券市值']


def _detect_encoding(content: bytes) -> str:
    """检测文件编码: 优先GBK(券商默认), 回退UTF-8"""
    for encoding in ['gbk', 'gb2312', 'utf-8-sig', 'utf-8']:
        try:
            content.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return 'utf-8'


def _match_column(headers: List[str], candidates: List[str]) -> Optional[int]:
    """模糊匹配列名, 返回匹配的列索引"""
    headers_lower = [h.strip().lower() for h in headers]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        # 精确匹配
        if candidate_lower in headers_lower:
            return headers_lower.index(candidate_lower)
        # 包含匹配
        for i, h in enumerate(headers_lower):
            if candidate_lower in h or h in candidate_lower:
                return i
    return None


def _clean_code(raw_code: str) -> str:
    """清洗股票代码: 去除前缀后缀, 补零到6位"""
    if not raw_code:
        return ''
    code = str(raw_code).strip()
    # 移除可能的交易所后缀 (.SZ, .SH, .BJ)
    code = re.sub(r'\.(SZ|SH|BJ|sz|sh|bj)$', '', code)
    # 移除非数字字符(某些导出有引号)
    code = re.sub(r'[^\d]', '', code)
    # 补零到6位
    if code and len(code) < 6:
        code = code.zfill(6)
    return code


def _parse_number(value: str) -> Optional[float]:
    """解析数字, 处理逗号分隔和中文字符"""
    if not value:
        return None
    value = str(value).strip()
    # 移除逗号和空格
    value = value.replace(',', '').replace(' ', '').replace('，', '')
    # 移除可能的货币符号
    value = value.replace('¥', '').replace('￥', '')
    try:
        return float(value)
    except ValueError:
        return None


def parse_csv(content: bytes) -> Tuple[List[Dict], List[str]]:
    """
    解析CSV内容

    Args:
        content: CSV文件的原始字节内容

    Returns:
        (parsed_positions, warnings)
        parsed_positions: [{code, name, quantity, avg_cost, current_price}, ...]
        warnings: 解析过程中的警告信息
    """
    warnings = []

    # 检测编码
    encoding = _detect_encoding(content)
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError:
        text = content.decode('utf-8', errors='replace')
        warnings.append(f'编码检测失败，使用UTF-8(可能有乱码)')

    # 跳过BOM
    if text.startswith('\ufeff'):
        text = text[1:]

    # 解析CSV
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return [], ['文件内容不足: 至少需要表头和一行数据']

    # 尝试不同分隔符
    delimiter = ','
    if '\t' in lines[0] and ',' not in lines[0]:
        delimiter = '\t'

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)

    if len(rows) < 2:
        return [], ['文件解析失败: 无有效数据行']

    headers = rows[0]

    # 匹配列
    code_idx = _match_column(headers, CODE_COLUMNS)
    name_idx = _match_column(headers, NAME_COLUMNS)
    qty_idx = _match_column(headers, QUANTITY_COLUMNS)
    cost_idx = _match_column(headers, COST_COLUMNS)
    price_idx = _match_column(headers, PRICE_COLUMNS)

    if code_idx is None:
        return [], [f'未找到股票代码列。检测到的列: {", ".join(headers)}']

    if qty_idx is None:
        return [], [f'未找到持仓数量列。检测到的列: {", ".join(headers)}']

    if cost_idx is None and price_idx is None:
        return [], [f'未找到成本价或现价列。检测到的列: {", ".join(headers)}']

    # 解析每一行
    positions = []
    for row_num, row in enumerate(rows[1:], start=2):
        if len(row) <= max(filter(None, [code_idx, qty_idx, cost_idx, price_idx]), default=0):
            continue

        code = _clean_code(row[code_idx] if code_idx is not None else '')
        if not code or len(code) != 6:
            continue

        # 跳过非A股代码 (ETF/债券/逆回购等)
        if not (code.startswith(('0', '3', '6')) or code.startswith('8')):
            continue

        name = row[name_idx].strip() if name_idx is not None and name_idx < len(row) else ''
        quantity = _parse_number(row[qty_idx]) if qty_idx is not None and qty_idx < len(row) else None
        avg_cost = _parse_number(row[cost_idx]) if cost_idx is not None and cost_idx < len(row) else None
        current_price = _parse_number(row[price_idx]) if price_idx is not None and price_idx < len(row) else None

        # 数量必须有效
        if quantity is None or quantity <= 0:
            warnings.append(f'第{row_num}行 {code} 数量无效，已跳过')
            continue

        # 成本价回退到现价
        if avg_cost is None or avg_cost <= 0:
            avg_cost = current_price

        if avg_cost is None or avg_cost <= 0:
            warnings.append(f'第{row_num}行 {code} 无有效价格，已跳过')
            continue

        positions.append({
            'code': code,
            'name': name,
            'quantity': int(quantity),
            'avg_cost': round(avg_cost, 3),
            'current_price': round(current_price, 3) if current_price else None,
        })

    if not positions:
        warnings.append('未解析到有效持仓数据')

    matched_cols = []
    if code_idx is not None:
        matched_cols.append(f'代码={headers[code_idx]}')
    if name_idx is not None:
        matched_cols.append(f'名称={headers[name_idx]}')
    if qty_idx is not None:
        matched_cols.append(f'数量={headers[qty_idx]}')
    if cost_idx is not None:
        matched_cols.append(f'成本={headers[cost_idx]}')
    if price_idx is not None:
        matched_cols.append(f'现价={headers[price_idx]}')
    warnings.insert(0, f'列匹配: {", ".join(matched_cols)}')

    return positions, warnings


def merge_positions(parsed: List[Dict], existing_positions: List[Dict],
                    db_manager) -> Dict:
    """
    智能合并导入持仓和现有持仓

    Args:
        parsed: parse_csv返回的解析结果
        existing_positions: 现有持仓列表
        db_manager: DatabaseManager实例

    Returns:
        {added: int, updated: int, skipped: int, details: [...]}
    """
    existing_map = {p['code']: p for p in existing_positions}
    added = 0
    updated = 0
    skipped = 0
    details = []

    for item in parsed:
        code = item['code']
        name = item['name'] or db_manager.get_stock_name(code) or ''
        current_price = item.get('current_price') or db_manager.get_stock_latest_price(code) or item['avg_cost']

        if code in existing_map:
            # 已有持仓: 更新价格和数量
            existing = existing_map[code]
            pos_id = existing['id']
            update_data = {}

            if item['quantity'] != existing['quantity']:
                update_data['quantity'] = item['quantity']
            if item['avg_cost'] and abs(item['avg_cost'] - existing['avg_cost']) > 0.01:
                update_data['avg_cost'] = item['avg_cost']
            if current_price:
                update_data['current_price'] = current_price

            if update_data:
                db_manager.update_position(pos_id, update_data)
                updated += 1
                details.append(f'{code} {name}: 更新')
            else:
                skipped += 1
                details.append(f'{code} {name}: 无变化,跳过')
        else:
            # 新持仓: 添加
            pos_data = {
                'code': code,
                'name': name,
                'quantity': item['quantity'],
                'avg_cost': item['avg_cost'],
                'current_price': current_price,
            }
            db_manager.add_position(pos_data)
            added += 1
            details.append(f'{code} {name}: 新增')

    return {
        'added': added,
        'updated': updated,
        'skipped': skipped,
        'total': len(parsed),
        'details': details
    }
