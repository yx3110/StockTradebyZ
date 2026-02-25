"""
CSV/网页粘贴/HTML 持仓导入器

支持三种输入格式:
1. CSV文件 (东方财富/同花顺导出): 标准CSV, 自动检测GBK/UTF-8编码
2. 网页粘贴文本 (从券商网页直接复制): 处理多行表头、粘连行、缺失数据
3. HTML表格 (东方财富模拟交易等网页源码): 结构化解析<table>标签

核心能力:
- 自动检测GBK/UTF-8编码
- 自动检测HTML格式并路由到HTML解析器
- 模糊匹配列名
- 用正则从每行提取6位股票代码+中文名+数字字段
- 检测粘连行 (两只股票挤在一行) 并拆分
- HTML截断修复: 成本价缺失时从盈亏推算
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
PNL_COLUMNS = ['持仓盈亏', '盈亏', '浮动盈亏']

# A股/ETF有效代码前2位
# 00:深主板 30:创业板 60:沪主板 68:科创板 8x:北交所
# 15:深ETF 50/51/52/56:沪ETF/LOF
VALID_CODE_2DIGIT = (
    '00', '30', '60', '68',
    '80', '83', '87',
    '15', '16',
    '50', '51', '52', '56',
)


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
        if candidate_lower in headers_lower:
            return headers_lower.index(candidate_lower)
        for i, h in enumerate(headers_lower):
            if candidate_lower in h or h in candidate_lower:
                return i
    return None


def _clean_code(raw_code: str) -> str:
    """清洗股票代码: 去除前缀后缀, 补零到6位"""
    if not raw_code:
        return ''
    code = str(raw_code).strip()
    code = re.sub(r'\.(SZ|SH|BJ|sz|sh|bj)$', '', code)
    code = re.sub(r'[^\d]', '', code)
    if code and len(code) < 6:
        code = code.zfill(6)
    return code


def _parse_number(value: str) -> Optional[float]:
    """解析数字, 处理逗号分隔和中文字符"""
    if not value:
        return None
    value = str(value).strip()
    value = value.replace(',', '').replace(' ', '').replace('，', '')
    value = value.replace('¥', '').replace('￥', '')
    # 移除百分号
    value = value.rstrip('%')
    try:
        return float(value)
    except ValueError:
        return None


def _is_stock_code(s: str) -> bool:
    """判断字符串是否是有效的6位股票代码"""
    s = s.strip()
    return bool(re.match(r'^\d{6}$', s)) and s[:2] in VALID_CODE_2DIGIT


def _is_chinese_name(s: str) -> bool:
    """判断字符串是否包含中文（股票名称）"""
    return bool(re.search(r'[\u4e00-\u9fff]', s))


def _detect_html(text: str) -> bool:
    """检测输入是否为HTML表格格式"""
    return bool(re.search(r'<t(?:able|body|head|r)\b', text, re.IGNORECASE))


def _extract_html_cells(row_html: str) -> List[str]:
    """
    从HTML <tr> 内容中提取单元格文本列表

    处理: </td>, </th>, 截断的 </t>, 损坏的 </ts="..."> 等各种标签
    策略: 用标签边界做分隔符, 提取所有非空文本段
    """
    text = row_html
    # 关闭标签作为分隔符: </td>, </th>, 截断的</t>, 损坏的</ts="green">
    text = re.sub(r'</t[^>]*>', '\x00', text, flags=re.IGNORECASE)
    # 开启标签作为分隔符: <td>, <td class="...">, <th>
    text = re.sub(r'<t[dh][^>]*>', '\x00', text, flags=re.IGNORECASE)
    # 清除剩余HTML标签 (如<a>, <button>, <br>等)
    text = re.sub(r'<[^>]+>', '', text)
    # HTML实体
    text = text.replace('&amp;', '&').replace('&nbsp;', ' ')
    # 按分隔符切分, 取非空段
    cells = [c.strip() for c in text.split('\x00') if c.strip()]
    return cells


# ==================== 格式3: HTML表格解析 ====================

def parse_html_table(html_text: str) -> Tuple[List[Dict], List[str]]:
    """
    解析HTML表格格式的持仓数据 (东方财富模拟交易等)

    处理:
    1. <thead>/<tbody> 结构化表格
    2. HTML标签截断/损坏 (如 </t> 代替 </td>)
    3. 单元格缺失时自动检测列偏移
    4. 成本价缺失时从市值和盈亏推算
    """
    warnings = []

    # ---- 1. 提取表头 ----
    headers = []
    thead = re.search(r'<thead[^>]*>(.*?)</thead>', html_text, re.DOTALL | re.IGNORECASE)
    if thead:
        ths = re.findall(r'<th[^>]*>(.*?)</th>', thead.group(1), re.DOTALL | re.IGNORECASE)
        headers = [re.sub(r'<[^>]+>', '', h).strip() for h in ths]

    if not headers:
        # 回退: 从第一个<tr>提取
        first_tr = re.search(r'<tr[^>]*>(.*?)</tr>', html_text, re.DOTALL | re.IGNORECASE)
        if first_tr:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', first_tr.group(1),
                               re.DOTALL | re.IGNORECASE)
            headers = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]

    n_cols = len(headers)

    # ---- 2. 列映射 ----
    code_idx = _match_column(headers, CODE_COLUMNS) if headers else None
    name_idx = _match_column(headers, NAME_COLUMNS) if headers else None
    qty_idx = _match_column(headers, QUANTITY_COLUMNS) if headers else None
    cost_idx = _match_column(headers, COST_COLUMNS) if headers else None
    price_idx = _match_column(headers, PRICE_COLUMNS) if headers else None
    mv_idx = _match_column(headers, MARKET_VALUE_COLUMNS) if headers else None
    pnl_idx = _match_column(headers, PNL_COLUMNS) if headers else None

    mapped = []
    for label, idx in [('代码', code_idx), ('名称', name_idx), ('数量', qty_idx),
                        ('成本', cost_idx), ('现价', price_idx), ('市值', mv_idx),
                        ('盈亏', pnl_idx)]:
        if idx is not None:
            mapped.append(f'{label}=#{idx}')
    if mapped:
        warnings.append(f'列映射: {", ".join(mapped)} (共{n_cols}列)')

    # ---- 3. 提取数据行 ----
    tbody = re.search(r'<tbody[^>]*>(.*?)</tbody>', html_text, re.DOTALL | re.IGNORECASE)
    body = tbody.group(1) if tbody else html_text
    # 排除thead中已处理的行
    if not tbody and thead:
        body = html_text[thead.end():]
    row_htmls = re.findall(r'<tr[^>]*>(.*?)</tr>', body, re.DOTALL | re.IGNORECASE)

    positions = []
    for row_num, row_html in enumerate(row_htmls, 1):
        cells = _extract_html_cells(row_html)

        # ---- 提取股票代码 ----
        code = ''
        if code_idx is not None and code_idx < len(cells):
            code = _clean_code(cells[code_idx])
        if not _is_stock_code(code):
            # 回退: 在行文本中搜索6位代码
            row_text = re.sub(r'<[^>]+>', ' ', row_html)
            m = re.search(r'(\d{6})', row_text)
            if m and _is_stock_code(m.group(1)):
                code = m.group(1)
        if not _is_stock_code(code):
            continue

        # ---- 提取名称 ----
        name = ''
        if name_idx is not None and name_idx < len(cells):
            name = cells[name_idx]
        if not name or not _is_chinese_name(name):
            for c in cells[1:5]:
                if _is_chinese_name(c) and '买' not in c and '卖' not in c:
                    name = c
                    break

        # ---- 提取数值 ----
        quantity = avg_cost = current_price = market_value = pnl_value = None

        def _safe_get(idx):
            if idx is not None and idx < len(cells):
                return _parse_number(cells[idx])
            return None

        if len(cells) >= n_cols:
            # 单元格数量匹配 → 直接列映射
            quantity = _safe_get(qty_idx)
            avg_cost = _safe_get(cost_idx)
            current_price = _safe_get(price_idx)
            market_value = _safe_get(mv_idx)
            pnl_value = _safe_get(pnl_idx)

        elif len(cells) == n_cols - 1 and cost_idx is not None:
            # 少1个单元格 → 先尝试直接映射(末尾列缺失), 再尝试成本价列缺失
            quantity = _safe_get(qty_idx)

            # 尝试1: 直接映射 (缺失列在末尾, 不影响核心数据)
            avg_cost = _safe_get(cost_idx)
            current_price = _safe_get(price_idx)
            market_value = _safe_get(mv_idx)
            pnl_value = _safe_get(pnl_idx)

            # 验证: price * qty ≈ market_value
            direct_ok = False
            if current_price and market_value and quantity and quantity > 0 and market_value > 0:
                if abs(current_price * quantity - market_value) / market_value < 0.02:
                    direct_ok = True

            if not direct_ok:
                # 尝试2: 成本价列缺失, 后续列左移1位
                test_price = _safe_get(cost_idx)
                test_mv = _parse_number(cells[mv_idx - 1]) if mv_idx and mv_idx - 1 < len(cells) else None

                if test_price and test_mv and quantity and quantity > 0 and test_mv > 0:
                    if abs(test_price * quantity - test_mv) / test_mv < 0.02:
                        current_price = test_price
                        market_value = test_mv
                        pnl_value = _parse_number(cells[pnl_idx - 1]) if pnl_idx and pnl_idx - 1 < len(cells) else None
                        avg_cost = None  # 后续从盈亏推算
                        warnings.append(f'{code} {name}: HTML截断，成本价列缺失')

        else:
            # 严重损坏 → 正则回退: 从行纯文本提取所有数字
            row_text = re.sub(r'<[^>]+>', ' ', row_html)
            nums = []
            for n in re.findall(r'-?\d+\.?\d*', row_text):
                if n == code:
                    continue
                try:
                    nums.append(float(n))
                except ValueError:
                    pass
            if len(nums) >= 4:
                quantity = int(nums[0])
                avg_cost = nums[2]
                current_price = nums[3]
                if len(nums) >= 5:
                    market_value = nums[4]
                if len(nums) >= 6:
                    pnl_value = nums[5]
            elif len(nums) >= 1:
                quantity = int(nums[0])
            warnings.append(f'{code} {name}: HTML严重损坏({len(cells)}列vs期望{n_cols}列)')

        # ---- 验证和修复 ----
        if quantity is not None:
            quantity = int(quantity)
        if quantity is None or quantity <= 0:
            if quantity == 0:
                warnings.append(f'{code} {name}: 持仓为0，跳过')
            continue

        # 成本价修复: 从市值和盈亏推算
        if (avg_cost is None or avg_cost <= 0) and market_value and pnl_value is not None and quantity > 0:
            total_cost = market_value - pnl_value
            if total_cost > 0:
                avg_cost = total_cost / quantity
                warnings.append(f'{code} {name}: 从盈亏推算成本={avg_cost:.3f}')

        # 负成本(分红摊薄) → 用现价替代
        if avg_cost is not None and avg_cost <= 0:
            avg_cost = current_price

        if not avg_cost and not current_price:
            warnings.append(f'{code} {name}: 无有效价格，跳过')
            continue

        # 市值交叉验证
        if current_price and market_value and quantity and market_value > 0:
            expected = quantity * current_price
            if abs(expected - market_value) / market_value > 0.02:
                inferred = market_value / quantity
                if avg_cost and 0.05 < inferred / avg_cost < 20:
                    warnings.append(f'{code} {name}: 价格校验{current_price:.3f}→{inferred:.3f}')
                    current_price = inferred

        positions.append({
            'code': code,
            'name': name,
            'quantity': quantity,
            'avg_cost': round(avg_cost, 3) if avg_cost else None,
            'current_price': round(current_price, 3) if current_price else None,
        })

    warnings.insert(0, f'HTML表格: {len(row_htmls)}行数据，解析出{len(positions)}只有效持仓')
    return positions, warnings


# ==================== 格式1: 标准CSV解析 ====================

def parse_csv(content: bytes) -> Tuple[List[Dict], List[str]]:
    """
    解析CSV内容 (标准CSV格式)

    Args:
        content: CSV文件的原始字节内容

    Returns:
        (parsed_positions, warnings)
    """
    warnings = []

    encoding = _detect_encoding(content)
    try:
        text = content.decode(encoding)
    except UnicodeDecodeError:
        text = content.decode('utf-8', errors='replace')
        warnings.append('编码检测失败，使用UTF-8(可能有乱码)')

    if text.startswith('\ufeff'):
        text = text[1:]

    # HTML表格格式 (包含<table>/<tr>/<td>标签)
    if _detect_html(text):
        return parse_html_table(text)

    # 先尝试网页粘贴格式（检测特征：多行表头、没有逗号分隔、包含"买  卖"）
    if ('买' in text and '卖' in text) or ('持仓盈亏' in text and ',' not in text.split('\n')[0]):
        return parse_web_paste(text)

    # 标准CSV解析
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return [], ['文件内容不足: 至少需要表头和一行数据']

    delimiter = ','
    if '\t' in lines[0] and ',' not in lines[0]:
        delimiter = '\t'

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = list(reader)

    if len(rows) < 2:
        return [], ['文件解析失败: 无有效数据行']

    headers = rows[0]

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

    positions = []
    for row_num, row in enumerate(rows[1:], start=2):
        if len(row) <= max(filter(None, [code_idx, qty_idx, cost_idx, price_idx]), default=0):
            continue

        code = _clean_code(row[code_idx] if code_idx is not None else '')
        if not code or len(code) != 6:
            continue
        if code[:2] not in VALID_CODE_2DIGIT:
            continue

        name = row[name_idx].strip() if name_idx is not None and name_idx < len(row) else ''
        quantity = _parse_number(row[qty_idx]) if qty_idx is not None and qty_idx < len(row) else None
        avg_cost = _parse_number(row[cost_idx]) if cost_idx is not None and cost_idx < len(row) else None
        current_price = _parse_number(row[price_idx]) if price_idx is not None and price_idx < len(row) else None

        if quantity is None or quantity <= 0:
            warnings.append(f'第{row_num}行 {code} 数量无效，已跳过')
            continue

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
    if code_idx is not None: matched_cols.append(f'代码={headers[code_idx]}')
    if name_idx is not None: matched_cols.append(f'名称={headers[name_idx]}')
    if qty_idx is not None: matched_cols.append(f'数量={headers[qty_idx]}')
    if cost_idx is not None: matched_cols.append(f'成本={headers[cost_idx]}')
    if price_idx is not None: matched_cols.append(f'现价={headers[price_idx]}')
    warnings.insert(0, f'列匹配: {", ".join(matched_cols)}')

    return positions, warnings


# ==================== 格式2: 网页粘贴解析 ====================

def parse_web_paste(text: str) -> Tuple[List[Dict], List[str]]:
    """
    解析从券商网页直接复制粘贴的持仓文本

    处理的问题:
    1. 多行表头 (证券\\n代码)
    2. Tab/空格混合分隔
    3. 粘连行 (两只股票挤在同一行，如 "33.064    323637    镇海股份")
    4. 缺失数据 (只有代码和名称，没有价格)
    5. 尾部的 "买  卖" 操作按钮文本
    6. 百分号后缀

    解析策略: 用正则逐行提取所有6位股票代码，以每个代码为锚点切分数据段
    """
    # HTML表格格式: 自动路由
    if _detect_html(text):
        return parse_html_table(text)

    warnings = []

    # 清理文本：合并连续空白为单个分隔符，去掉"买  卖"
    lines = text.strip().split('\n')

    # 跳过表头行（包含中文列名但不包含6位数字代码的行）
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 表头行特征: 包含"证券" "代码" "名称" "数量" "成本" 等但没有6位数字
        if re.search(r'\d{6}', line):
            data_lines.append(line)
        elif any(kw in line for kw in ['证券', '代码', '名称', '数量', '成本', '当前', '市值', '盈亏', '比例', '操作']):
            continue  # 跳过表头
        else:
            continue  # 跳过其他非数据行

    # 把所有数据行合并成一个长字符串，用空格分隔
    # 这样即使粘连也能处理
    full_text = '  '.join(data_lines)

    # 清理"买  卖"等操作按钮文本
    full_text = re.sub(r'买\s*卖', ' ', full_text)

    # 核心解析：找到所有6位股票代码的位置，以此为锚点切分
    # 模式: 6位数字 + 空白 + 中文名称
    code_pattern = re.compile(r'(?<!\d)(\d{6})\s+([\u4e00-\u9fffA-Za-z]+\S*)')
    raw_matches = list(code_pattern.finditer(full_text))

    if not raw_matches:
        return [], ['未找到有效股票代码，请检查粘贴内容']

    # 第一遍: 过滤有效代码, 尝试从无效代码中恢复被粘连的真实代码
    # 例如 "323637" 是价格尾部 + "603637" 粘连, 恢复为 "603637"
    # 用元组 (code, name, start, end) 替代match对象, 避免闭包问题
    anchors = []  # [(code, name, match_start, match_end), ...]
    for m in raw_matches:
        code = m.group(1)
        name = m.group(2).strip()
        if code[:2] in VALID_CODE_2DIGIT:
            anchors.append((code, name, m.start(), m.end()))
        else:
            # 尝试恢复: 用后4位 + 各种有效前缀拼出合法代码
            last4 = code[2:]
            recovered = None
            priority_prefixes = ['60', '00', '30', '68'] + [
                p for p in VALID_CODE_2DIGIT if p not in ('60', '00', '30', '68')]
            for prefix in priority_prefixes:
                candidate = prefix + last4
                if candidate != code:
                    recovered = candidate
                    break
            if recovered:
                warnings.append(f'恢复粘连代码: {code}→{recovered} ({name})')
                anchors.append((recovered, name, m.start(), m.end()))
            else:
                warnings.append(f'跳过无效代码 {code} ({name}), 无法恢复')

    if not anchors:
        return [], ['未找到有效股票代码，请检查粘贴内容']

    positions = []
    corrupt_lines = []

    for i, (code, name, _start, _end) in enumerate(anchors):
        # 提取从当前股票名称之后到下一个股票代码之前的所有数字
        if i + 1 < len(anchors):
            next_start = anchors[i + 1][2]  # next anchor's match_start
            segment = full_text[_end:next_start]
        else:
            segment = full_text[_end:]

        # 从segment中提取所有数字（可能包含负号和小数点）
        numbers = re.findall(r'-?\d+\.?\d*', segment)
        numbers = [float(n) for n in numbers]

        # 东方财富网页格式的数字顺序:
        # 持仓数量, 可用数量, 成本价, 当前价, 最新市值, 持仓盈亏, 持仓盈亏比例, 当日盈亏, 当日盈亏比例
        quantity = None
        avg_cost = None
        current_price = None
        market_value = None

        if len(numbers) >= 4:
            # 标准情况: 至少有 数量, 可用数量, 成本价, 当前价
            quantity = int(numbers[0])
            # numbers[1] = 可用数量 (跳过)
            avg_cost = numbers[2]
            current_price = numbers[3]
            if len(numbers) >= 5:
                market_value = numbers[4]
        elif len(numbers) >= 2:
            # 部分数据: 只有数量和可用数量
            quantity = int(numbers[0])
            warnings.append(f'{code} {name}: 只有{len(numbers)}个数字，缺少价格数据')
        elif len(numbers) == 1:
            quantity = int(numbers[0])
            warnings.append(f'{code} {name}: 只有数量({quantity})，缺少价格数据')
        else:
            warnings.append(f'{code} {name}: 无数字数据，已跳过')
            continue

        if quantity is None or quantity <= 0:
            warnings.append(f'{code} {name}: 数量无效，已跳过')
            continue

        # 成本价为负数时用现价替代
        if avg_cost is not None and avg_cost <= 0:
            avg_cost = current_price

        # 检测粘连: segment中包含另一个有效股票代码, 或包含中文名(如"友钴业")
        inner_codes = re.findall(r'(?<!\d)(\d{6})(?!\d)', segment)
        valid_inner = [c for c in inner_codes if c[:2] in VALID_CODE_2DIGIT and c != code]
        # 也检测中文名出现在数字序列中 (如 "1友钴业" = 截断市值+丢失的股票名)
        chinese_in_data = re.search(r'\d[\u4e00-\u9fff]', segment)
        if valid_inner or chinese_in_data:
            detail = valid_inner[0] if valid_inner else '中文名嵌入数据中'
            corrupt_lines.append(f'{code} {name}: 数据段异常({detail})，市值等后续数据不可靠')
            # 粘连时: qty/cost/price(前4个数字)通常在损坏点之前，可信
            # 但market_value及之后的数据不可靠，清除
            market_value = None
            # 如果损坏点在前4个数字之后，price仍然可信
            if chinese_in_data:
                corrupt_pos = chinese_in_data.start()
                # 找到损坏点之前有多少个数字
                pre_corrupt = segment[:corrupt_pos]
                pre_numbers = re.findall(r'-?\d+\.?\d*', pre_corrupt)
                if len(pre_numbers) < 4:
                    # 损坏发生在price之前，price不可信
                    current_price = None
        elif current_price is not None and market_value is not None and market_value > 0:
            # 检查价格是否合理 (只在非粘连时做)
            expected_mv = quantity * current_price
            if abs(expected_mv - market_value) > market_value * 0.02:
                # 市值不匹配，尝试从市值反推价格
                inferred_price = market_value / quantity
                # 反推价格必须在成本价的合理倍数范围内 (0.1x ~ 10x)
                if avg_cost and 0.1 < inferred_price / avg_cost < 10:
                    warnings.append(f'{code} {name}: 价格{current_price}与市值{market_value}不匹配，'
                                    f'用市值反推价格={inferred_price:.3f}')
                    current_price = inferred_price

        positions.append({
            'code': code,
            'name': name,
            'quantity': quantity,
            'avg_cost': round(avg_cost, 3) if avg_cost else None,
            'current_price': round(current_price, 3) if current_price else None,
        })

    # 过滤掉无价格的
    valid_positions = []
    for p in positions:
        if p['avg_cost'] and p['avg_cost'] > 0:
            valid_positions.append(p)
        else:
            warnings.append(f'{p["code"]} {p["name"]}: 无有效价格，需手动补录')

    if corrupt_lines:
        warnings = [f'⚠️ 粘连检测: {c}' for c in corrupt_lines] + warnings

    warnings.insert(0, f'网页粘贴格式: 检测到{len(anchors)}只证券，有效{len(valid_positions)}只')

    return valid_positions, warnings


# ==================== 合并逻辑 ====================

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
        current_price = item.get('current_price') or db_manager.get_stock_latest_price(code) or item.get('avg_cost')

        if code in existing_map:
            existing = existing_map[code]
            pos_id = existing['id']
            update_data = {}

            if item['quantity'] != existing['quantity']:
                update_data['quantity'] = item['quantity']
            if item.get('avg_cost') and abs(item['avg_cost'] - existing['avg_cost']) > 0.01:
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
            pos_data = {
                'code': code,
                'name': name,
                'quantity': item['quantity'],
                'avg_cost': item.get('avg_cost', 0),
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
