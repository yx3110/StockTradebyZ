"""
报告解析器 - 解析Markdown和JSON报告文件
"""
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class ReportParser:
    """
    Markdown报告解析器

    解析各种报告文件（选股报告、回测报告、AI分析报告）
    """

    @staticmethod
    def parse_selection_report(filepath: Path) -> Dict[str, Any]:
        """
        解析选股报告

        Args:
            filepath: 报告文件路径

        Returns:
            解析结果字典
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取日期
            date = ReportParser._extract_date(content)

            # 提取股票列表
            stocks = ReportParser._extract_stock_table(content)

            # 提取摘要信息
            summary = ReportParser._extract_summary(content)

            return {
                'date': date,
                'stocks': stocks,
                'summary': summary,
                'raw_content': content,
                'file_path': str(filepath)
            }

        except Exception as e:
            logger.error(f'解析选股报告失败: {filepath}, 错误: {e}')
            return {
                'error': str(e),
                'file_path': str(filepath)
            }

    @staticmethod
    def parse_backtest_report(filepath: Path) -> Dict[str, Any]:
        """
        解析回测报告

        Args:
            filepath: 报告文件路径

        Returns:
            解析结果字典
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取回测配置
            config = ReportParser._extract_backtest_config(content)

            # 提取性能指标
            metrics = ReportParser._extract_backtest_metrics(content)

            # 提取交易记录
            trades = ReportParser._extract_backtest_trades(content)

            return {
                'config': config,
                'metrics': metrics,
                'trades': trades,
                'raw_content': content,
                'file_path': str(filepath)
            }

        except Exception as e:
            logger.error(f'解析回测报告失败: {filepath}, 错误: {e}')
            return {
                'error': str(e),
                'file_path': str(filepath)
            }

    @staticmethod
    def parse_ai_analysis_report(filepath: Path) -> Dict[str, Any]:
        """
        解析AI分析报告

        Args:
            filepath: 报告文件路径

        Returns:
            解析结果字典
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取日期
            date = ReportParser._extract_date(content)

            # 提取市场分析
            market_analysis = ReportParser._extract_section(content, '市场分析')

            # 提取个股分析
            stock_analysis = ReportParser._extract_section(content, '个股分析')

            # 提取风险评估
            risk_assessment = ReportParser._extract_section(content, '风险评估')

            return {
                'date': date,
                'market_analysis': market_analysis,
                'stock_analysis': stock_analysis,
                'risk_assessment': risk_assessment,
                'raw_content': content,
                'file_path': str(filepath)
            }

        except Exception as e:
            logger.error(f'解析AI分析报告失败: {filepath}, 错误: {e}')
            return {
                'error': str(e),
                'file_path': str(filepath)
            }

    # ==================== 辅助方法 ====================

    @staticmethod
    def _extract_date(content: str) -> Optional[str]:
        """从内容中提取日期"""
        # 尝试多种日期格式
        patterns = [
            r'日期[：:]\s*(\d{4}-\d{2}-\d{2})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{8})',  # YYYYMMDD
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                date_str = match.group(1)
                # 转换YYYYMMDD格式
                if len(date_str) == 8:
                    return f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
                return date_str

        return None

    @staticmethod
    def _extract_stock_table(content: str) -> List[Dict]:
        """从内容中提取股票表格"""
        stocks = []

        # 匹配Markdown表格行
        # 格式: | 代码 | 名称 | 评分 | ... |
        table_pattern = r'\|\s*(\d{6})\s*\|\s*([^\|]+)\s*\|'

        for match in re.finditer(table_pattern, content):
            code = match.group(1)
            name = match.group(2).strip()

            # 尝试提取整行数据
            line = content[match.start():content.find('\n', match.start())]
            cells = [cell.strip() for cell in line.split('|')[1:-1]]

            stock_info = {
                'code': code,
                'name': name,
            }

            # 尝试提取更多字段（根据表格结构）
            if len(cells) >= 3:
                try:
                    stock_info['score'] = float(cells[2]) if cells[2] else None
                except (ValueError, IndexError):
                    pass

            stocks.append(stock_info)

        return stocks

    @staticmethod
    def _extract_summary(content: str) -> str:
        """提取摘要部分"""
        # 尝试提取"摘要"或"总结"部分
        patterns = [
            r'#+\s*摘要\s*\n(.*?)(?=\n#+|\Z)',
            r'#+\s*总结\s*\n(.*?)(?=\n#+|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        return ''

    @staticmethod
    def _extract_backtest_config(content: str) -> Dict[str, Any]:
        """提取回测配置"""
        config = {}

        # 提取常见配置项
        patterns = {
            'start_date': r'开始日期[：:]\s*(\d{4}-\d{2}-\d{2})',
            'end_date': r'结束日期[：:]\s*(\d{4}-\d{2}-\d{2})',
            'initial_capital': r'初始资金[：:]\s*([\d,]+)',
            'strategy': r'策略[：:]\s*([^\n]+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                config[key] = match.group(1).strip()

        return config

    @staticmethod
    def _extract_backtest_metrics(content: str) -> Dict[str, Any]:
        """提取回测性能指标"""
        metrics = {}

        # 提取常见指标
        patterns = {
            'total_return': r'总收益率[：:]\s*([-\d.]+)%?',
            'sharpe_ratio': r'夏普比率[：:]\s*([-\d.]+)',
            'max_drawdown': r'最大回撤[：:]\s*([-\d.]+)%?',
            'win_rate': r'胜率[：:]\s*([-\d.]+)%?',
            'total_trades': r'总交易次数[：:]\s*(\d+)',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                try:
                    metrics[key] = float(match.group(1))
                except ValueError:
                    metrics[key] = match.group(1)

        return metrics

    @staticmethod
    def _extract_backtest_trades(content: str) -> List[Dict]:
        """提取回测交易记录"""
        trades = []

        # 匹配交易表格
        # 格式: | 日期 | 代码 | 名称 | 买入/卖出 | 价格 | ... |
        table_pattern = r'\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*(\d{6})\s*\|'

        for match in re.finditer(table_pattern, content):
            date = match.group(1)
            code = match.group(2)

            # 提取整行
            line = content[match.start():content.find('\n', match.start())]
            cells = [cell.strip() for cell in line.split('|')[1:-1]]

            trade_info = {
                'date': date,
                'code': code,
            }

            # 根据列数提取更多信息
            if len(cells) >= 4:
                trade_info['action'] = cells[3] if len(cells) > 3 else None

            trades.append(trade_info)

        return trades

    @staticmethod
    def _extract_section(content: str, section_name: str) -> str:
        """提取指定章节内容"""
        # 匹配章节标题后的内容
        pattern = rf'#+\s*{section_name}\s*\n(.*?)(?=\n#+|\Z)'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            return match.group(1).strip()

        return ''

    @staticmethod
    def get_latest_report(
        reports_dir: Path,
        pattern: str = '*.md'
    ) -> Optional[Path]:
        """
        获取最新的报告文件

        Args:
            reports_dir: 报告目录
            pattern: 文件匹配模式

        Returns:
            最新报告文件路径，如果不存在返回None
        """
        if not reports_dir.exists():
            return None

        files = list(reports_dir.glob(pattern))
        if not files:
            return None

        # 按修改时间排序，返回最新的
        return max(files, key=lambda f: f.stat().st_mtime)

    @staticmethod
    def list_reports(
        reports_dir: Path,
        pattern: str = '*.md',
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        列出报告文件

        Args:
            reports_dir: 报告目录
            pattern: 文件匹配模式
            limit: 返回数量限制

        Returns:
            报告文件列表（包含文件信息）
        """
        if not reports_dir.exists():
            return []

        files = list(reports_dir.glob(pattern))

        # 按修改时间排序
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        results = []
        for file in files[:limit]:
            stat = file.stat()
            results.append({
                'name': file.name,
                'path': str(file),
                'size': stat.st_size,
                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        return results

    # ==================== JSON报告解析 ====================

    @staticmethod
    def parse_selection_json(filepath: Path) -> Dict[str, Any]:
        """
        解析选股JSON数据文件

        Args:
            filepath: JSON文件路径

        Returns:
            解析结果字典
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取日期（从文件名）
            date_match = re.search(r'(\d{8})', filepath.name)
            date = None
            if date_match:
                d = date_match.group(1)
                date = f'{d[:4]}-{d[4:6]}-{d[6:]}'

            # 处理推荐股票
            top_recommendations = data.get('top_recommendations', [])
            all_stocks = data.get('all_stocks_with_scores', [])

            # 格式化股票数据
            formatted_stocks = []
            for stock in top_recommendations:
                formatted_stocks.append({
                    'code': stock.get('stock_code', ''),
                    'name': stock.get('stock_name', ''),
                    'industry': stock.get('industry', ''),
                    'close_price': stock.get('close_price', 0),
                    'price_change_pct': stock.get('price_change_pct', 0),
                    'score': stock.get('score', 0),
                    'confidence': stock.get('confidence', ''),
                    'recommendation': stock.get('recommendation', ''),
                    'strategies': stock.get('strategies', []),
                    'strategy_count': stock.get('selected_by_strategies', 0),
                    'risk_reward_ratio': stock.get('risk_reward_ratio', 0),
                    'stop_loss_price': stock.get('stop_loss_price', 0),
                    'take_profit_price': stock.get('take_profit_price', 0),
                    'technical_rating': stock.get('technical_rating', ''),
                    'risk_rating': stock.get('risk_rating', ''),
                    'kdj_k': stock.get('kdj_k', 0),
                    'kdj_d': stock.get('kdj_d', 0),
                    'kdj_j': stock.get('kdj_j', 0),
                    'bbi': stock.get('bbi', 0),
                    'predicted_return_5d': stock.get('predicted_return_5d'),
                    'detailed_scoring': stock.get('detailed_scoring', {}),
                    'volume': stock.get('volume', 0),
                    'suggested_buy_price': stock.get('suggested_buy_price', 0),
                })

            return {
                'date': date,
                'total_strategies': data.get('total_strategies', 0),
                'strategy_results': data.get('strategy_results', {}),
                'total_stocks': len(all_stocks),
                'top_recommendations': formatted_stocks,
                'all_stocks_count': len(all_stocks),
                'multi_strategy_count': len(data.get('multi_strategy_stocks', [])),
                'file_path': str(filepath)
            }

        except Exception as e:
            logger.error(f'解析选股JSON失败: {filepath}, 错误: {e}')
            return {
                'error': str(e),
                'file_path': str(filepath)
            }

    @staticmethod
    def get_selection_dates(reports_dir: Path, limit: int = 100) -> List[str]:
        """
        获取可用的选股日期列表

        Args:
            reports_dir: 报告目录
            limit: 返回数量限制

        Returns:
            日期列表（YYYY-MM-DD格式）
        """
        if not reports_dir.exists():
            return []

        # 查找JSON数据文件
        json_files = list(reports_dir.glob('analysis_data_*.json'))
        json_files.sort(reverse=True)

        dates = []
        for f in json_files[:limit]:
            match = re.search(r'(\d{8})', f.name)
            if match:
                d = match.group(1)
                dates.append(f'{d[:4]}-{d[4:6]}-{d[6:]}')

        return dates

    @staticmethod
    def get_selection_by_date(reports_dir: Path, date: str) -> Optional[Dict[str, Any]]:
        """
        获取指定日期的选股数据

        Args:
            reports_dir: 报告目录
            date: 日期（YYYY-MM-DD或YYYYMMDD）

        Returns:
            选股数据字典
        """
        # 转换日期格式
        if '-' in date:
            file_date = date.replace('-', '')
        else:
            file_date = date

        # 查找JSON文件
        json_file = reports_dir / f'analysis_data_{file_date}.json'
        if json_file.exists():
            return ReportParser.parse_selection_json(json_file)

        # 备选：查找Markdown文件
        md_file = reports_dir / f'选股分析报告_{file_date}.md'
        if md_file.exists():
            return ReportParser.parse_selection_report(md_file)

        return None
