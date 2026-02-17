#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML评分系统多期收益期望分析器

计算各版本评分器（v3.7, v3.8, v3.81）在基础策略选股后，持有不同天数（5d, 10d, 15d）的收益期望。

核心功能：
- 加载历史选股数据
- 计算多期持有收益率
- 统计分析和期望收益
- 生成对比报告

Author: Claude Code
Date: 2025-11-03
"""

import os
import sys
import json
import logging
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict

# 添加项目路径
sys.path.append('.')

from data_adapter.database_manager import DatabaseManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class StockSelection:
    """股票选股记录"""
    date: str
    stock_code: str
    stock_name: str
    strategies: List[str]
    comprehensive_score: float
    quality_score: float
    short_term_score: float
    medium_term_score: float
    long_term_score: float
    confidence: float
    version: str  # v3.7, v3.8, v3.81


@dataclass
class ReturnRecord:
    """收益记录"""
    selection: StockSelection
    entry_price: float
    return_5d: Optional[float]
    return_10d: Optional[float]
    return_15d: Optional[float]
    exit_price_5d: Optional[float]
    exit_price_10d: Optional[float]
    exit_price_15d: Optional[float]
    is_valid_5d: bool
    is_valid_10d: bool
    is_valid_15d: bool


@dataclass
class StatisticsSummary:
    """统计摘要"""
    version: str
    holding_period: int  # 5, 10, 15
    total_samples: int
    valid_samples: int
    mean_return: float
    median_return: float
    std_return: float
    win_rate: float  # 盈利比例
    max_return: float
    min_return: float
    percentile_25: float
    percentile_75: float


class HistoricalSelectionLoader:
    """历史选股数据加载器"""

    def __init__(self):
        self.versions = ['v3.7', 'v3.8', 'v3.81']
        self.base_dir = Path('reports')

    def load_all_selections(self) -> Dict[str, List[StockSelection]]:
        """
        加载所有版本的历史选股数据

        Returns:
            {version: [StockSelection, ...]}
        """
        all_selections = {}

        for version in self.versions:
            version_dir = self.base_dir / f'daily_selection_{version}'
            if not version_dir.exists():
                logger.warning(f"目录不存在: {version_dir}")
                all_selections[version] = []
                continue

            selections = self._load_version_selections(version_dir, version)
            all_selections[version] = selections
            logger.info(f"✅ 加载{version}选股数据: {len(selections)}条记录")

        return all_selections

    def _load_version_selections(self, version_dir: Path, version: str) -> List[StockSelection]:
        """加载单个版本的选股数据"""
        selections = []

        # 查找所有analysis_data_*.json文件
        json_files = sorted(version_dir.glob('analysis_data_*.json'))

        for json_file in json_files:
            try:
                # 从文件名提取日期
                date_str = json_file.stem.split('_')[-1]  # analysis_data_20250930.json -> 20250930

                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 解析股票列表 - 支持多种格式
                stock_list = []
                if 'detailed_stocks' in data:
                    stock_list = data['detailed_stocks']
                elif 'stock_analysis' in data:
                    stock_list = data['stock_analysis']
                elif 'all_selected_stocks' in data:
                    stock_list = data['all_selected_stocks']
                elif 'all_stock_details' in data:
                    # v3.7格式
                    stock_list = data['all_stock_details']

                for stock_info in stock_list:
                    try:
                        # 兼容多种字段命名
                        code = stock_info.get('code', stock_info.get('stock_code', ''))
                        name = stock_info.get('name', stock_info.get('stock_name', ''))

                        # 策略列表
                        strategies = stock_info.get('traditional_strategies',
                                                   stock_info.get('strategies', []))
                        if isinstance(strategies, str):
                            strategies = [strategies]

                        # 评分字段 - 兼容v3.7、v3.8、v3.81格式
                        comprehensive_score = float(stock_info.get('final_score',
                                                    stock_info.get('comprehensive_score',
                                                    stock_info.get('score', 0))))

                        # 质量评分
                        quality_score = float(stock_info.get('quality_score',
                                             stock_info.get('overall_quality', 0)))

                        # 分项得分 - v3.7使用factor_scores，v3.8/v3.81使用直接字段
                        factor_scores = stock_info.get('factor_scores', {})
                        if factor_scores:
                            # v3.7格式
                            short_term_score = float(factor_scores.get('technical', 0))
                            medium_term_score = float(factor_scores.get('temporal', 0))
                            long_term_score = float(factor_scores.get('fundamental', 0))
                        else:
                            # v3.8/v3.81格式
                            short_term_score = float(stock_info.get('short_term_score', 0))
                            medium_term_score = float(stock_info.get('medium_term_score', 0))
                            long_term_score = float(stock_info.get('long_term_score', 0))

                        # 置信度
                        confidence_val = stock_info.get('confidence_score',
                                                       stock_info.get('confidence', ''))
                        # 将文字置信度转换为数值
                        if isinstance(confidence_val, str):
                            confidence_map = {'极高': 0.9, '高': 0.7, '中': 0.5, '低': 0.3, '极低': 0.1}
                            confidence = confidence_map.get(confidence_val, 0.5)
                        else:
                            confidence = float(confidence_val) if confidence_val else 0.5

                        if code:  # 只有当股票代码存在时才添加
                            selection = StockSelection(
                                date=date_str,
                                stock_code=code,
                                stock_name=name,
                                strategies=strategies,
                                comprehensive_score=comprehensive_score,
                                quality_score=quality_score,
                                short_term_score=short_term_score,
                                medium_term_score=medium_term_score,
                                long_term_score=long_term_score,
                                confidence=confidence,
                                version=version
                            )
                            selections.append(selection)
                    except Exception as e:
                        logger.debug(f"解析股票信息失败: {stock_info.get('code', 'unknown')}, {e}")
                        continue

            except Exception as e:
                logger.warning(f"读取文件失败 {json_file}: {e}")
                continue

        return selections


class ReturnCalculator:
    """收益率计算器"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        self.price_cache = {}  # 价格缓存 {(code, date): price}

    def calculate_returns(self, selections: List[StockSelection]) -> List[ReturnRecord]:
        """
        计算选股列表的多期收益率

        Args:
            selections: 选股列表

        Returns:
            收益记录列表
        """
        return_records = []

        for i, selection in enumerate(selections):
            if (i + 1) % 100 == 0:
                logger.info(f"处理进度: {i+1}/{len(selections)}")

            try:
                record = self._calculate_single_return(selection)
                if record:
                    return_records.append(record)
            except Exception as e:
                logger.debug(f"计算收益失败 {selection.stock_code} {selection.date}: {e}")
                continue

        return return_records

    def _calculate_single_return(self, selection: StockSelection) -> Optional[ReturnRecord]:
        """计算单只股票的收益"""
        # 获取入场价格（选股日的收盘价）
        entry_price = self._get_price(selection.stock_code, selection.date)
        if not entry_price or entry_price <= 0:
            return None

        # 计算未来日期
        entry_date = datetime.strptime(selection.date, '%Y%m%d')

        # 获取未来5天、10天、15天的价格和收益
        return_5d, exit_price_5d, is_valid_5d = self._get_future_return(
            selection.stock_code, entry_date, 5, entry_price
        )
        return_10d, exit_price_10d, is_valid_10d = self._get_future_return(
            selection.stock_code, entry_date, 10, entry_price
        )
        return_15d, exit_price_15d, is_valid_15d = self._get_future_return(
            selection.stock_code, entry_date, 15, entry_price
        )

        return ReturnRecord(
            selection=selection,
            entry_price=entry_price,
            return_5d=return_5d,
            return_10d=return_10d,
            return_15d=return_15d,
            exit_price_5d=exit_price_5d,
            exit_price_10d=exit_price_10d,
            exit_price_15d=exit_price_15d,
            is_valid_5d=is_valid_5d,
            is_valid_10d=is_valid_10d,
            is_valid_15d=is_valid_15d
        )

    def _get_price(self, stock_code: str, date_str: str) -> Optional[float]:
        """获取股票在指定日期的收盘价"""
        cache_key = (stock_code, date_str)
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        try:
            # 转换日期格式：YYYYMMDD -> YYYY-MM-DD
            if len(date_str) == 8 and date_str.isdigit():
                formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            else:
                formatted_date = date_str

            query = """
            SELECT dq.close
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.code = ? AND dq.trade_date = ?
            """
            result = self.db_manager.execute_query(query, (stock_code, formatted_date))

            if result and len(result) > 0:
                price = float(result[0]['close'])
                self.price_cache[cache_key] = price
                return price

        except Exception as e:
            logger.debug(f"查询价格失败 {stock_code} {date_str}: {e}")

        return None

    def _get_future_return(self, stock_code: str, entry_date: datetime,
                          holding_days: int, entry_price: float) -> Tuple[Optional[float], Optional[float], bool]:
        """
        获取未来N天的收益率

        Returns:
            (收益率, 退出价格, 是否有效)
        """
        try:
            # 查询未来N个交易日的价格（使用带连字符的日期格式）
            start_date = entry_date.strftime('%Y-%m-%d')
            end_date = (entry_date + timedelta(days=holding_days * 2)).strftime('%Y-%m-%d')  # 扩大查询范围

            query = """
            SELECT dq.trade_date, dq.close
            FROM securities s
            JOIN daily_quotes dq ON s.id = dq.security_id
            WHERE s.code = ?
            AND dq.trade_date > ?
            AND dq.trade_date <= ?
            ORDER BY dq.trade_date ASC
            LIMIT ?
            """

            result = self.db_manager.execute_query(query, (stock_code, start_date, end_date, holding_days))

            if result and len(result) == holding_days:
                # 取第N个交易日的收盘价
                exit_price = float(result[-1]['close'])
                return_rate = (exit_price - entry_price) / entry_price * 100
                return return_rate, exit_price, True
            else:
                return None, None, False

        except Exception as e:
            logger.debug(f"查询未来价格失败 {stock_code}: {e}")
            return None, None, False


class StatisticsAnalyzer:
    """统计分析器"""

    def analyze_returns(self, return_records: List[ReturnRecord]) -> Dict[str, List[StatisticsSummary]]:
        """
        分析收益记录，生成统计摘要

        Returns:
            {version: [StatisticsSummary for 5d, 10d, 15d]}
        """
        # 按版本分组
        records_by_version = defaultdict(list)
        for record in return_records:
            records_by_version[record.selection.version].append(record)

        # 对每个版本计算统计
        statistics = {}
        for version, records in records_by_version.items():
            version_stats = []
            for holding_period in [5, 10, 15]:
                stats = self._calculate_period_statistics(records, holding_period, version)
                version_stats.append(stats)
            statistics[version] = version_stats

        return statistics

    def _calculate_period_statistics(self, records: List[ReturnRecord],
                                    period: int, version: str) -> StatisticsSummary:
        """计算特定持有期的统计"""
        # 提取有效收益率
        returns = []
        total_samples = len(records)

        for record in records:
            if period == 5 and record.is_valid_5d and record.return_5d is not None:
                returns.append(record.return_5d)
            elif period == 10 and record.is_valid_10d and record.return_10d is not None:
                returns.append(record.return_10d)
            elif period == 15 and record.is_valid_15d and record.return_15d is not None:
                returns.append(record.return_15d)

        valid_samples = len(returns)

        if valid_samples == 0:
            return StatisticsSummary(
                version=version,
                holding_period=period,
                total_samples=total_samples,
                valid_samples=0,
                mean_return=0.0,
                median_return=0.0,
                std_return=0.0,
                win_rate=0.0,
                max_return=0.0,
                min_return=0.0,
                percentile_25=0.0,
                percentile_75=0.0
            )

        returns_array = np.array(returns)

        return StatisticsSummary(
            version=version,
            holding_period=period,
            total_samples=total_samples,
            valid_samples=valid_samples,
            mean_return=float(np.mean(returns_array)),
            median_return=float(np.median(returns_array)),
            std_return=float(np.std(returns_array)),
            win_rate=float(np.sum(returns_array > 0) / valid_samples * 100),
            max_return=float(np.max(returns_array)),
            min_return=float(np.min(returns_array)),
            percentile_25=float(np.percentile(returns_array, 25)),
            percentile_75=float(np.percentile(returns_array, 75))
        )


class ReportGenerator:
    """报告生成器"""

    def generate_report(self, statistics: Dict[str, List[StatisticsSummary]],
                       output_file: str):
        """生成Markdown格式的对比报告"""

        report_lines = [
            "# 📊 ML评分系统多期收益期望分析报告\n",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "---\n",
            "\n## 📈 分析概览\n",
            f"- **评分版本**: {', '.join(statistics.keys())}",
            f"- **持有期**: 5天、10天、15天",
            "- **数据源**: 历史选股记录 + 真实市场数据\n",
            "\n## 📊 各版本收益期望对比\n"
        ]

        # 生成对比表格
        for period in [5, 10, 15]:
            report_lines.append(f"\n### 持有{period}天收益对比\n")
            report_lines.append("| 版本 | 样本数 | 有效样本 | 平均收益(%) | 中位数(%) | 标准差(%) | 胜率(%) | 最大收益(%) | 最小收益(%) | 25分位(%) | 75分位(%) |")
            report_lines.append("|------|--------|----------|-------------|-----------|-----------|---------|-------------|-------------|-----------|-----------|")

            for version in sorted(statistics.keys()):
                stats = next((s for s in statistics[version] if s.holding_period == period), None)
                if stats:
                    report_lines.append(
                        f"| {version} | {stats.total_samples} | {stats.valid_samples} | "
                        f"{stats.mean_return:.2f} | {stats.median_return:.2f} | "
                        f"{stats.std_return:.2f} | {stats.win_rate:.2f} | "
                        f"{stats.max_return:.2f} | {stats.min_return:.2f} | "
                        f"{stats.percentile_25:.2f} | {stats.percentile_75:.2f} |"
                    )

        # 生成关键指标对比
        report_lines.append("\n## 🎯 关键指标对比\n")

        # 最佳平均收益
        report_lines.append("\n### 最佳平均收益\n")
        report_lines.append("| 持有期 | 最佳版本 | 平均收益(%) | 胜率(%) |")
        report_lines.append("|--------|----------|-------------|---------|")

        for period in [5, 10, 15]:
            best_version = None
            best_return = -float('inf')
            best_win_rate = 0

            for version in statistics.keys():
                stats = next((s for s in statistics[version] if s.holding_period == period), None)
                if stats and stats.mean_return > best_return:
                    best_return = stats.mean_return
                    best_version = version
                    best_win_rate = stats.win_rate

            if best_version:
                report_lines.append(f"| {period}天 | {best_version} | {best_return:.2f} | {best_win_rate:.2f} |")

        # 最佳胜率
        report_lines.append("\n### 最佳胜率\n")
        report_lines.append("| 持有期 | 最佳版本 | 胜率(%) | 平均收益(%) |")
        report_lines.append("|--------|----------|---------|-------------|")

        for period in [5, 10, 15]:
            best_version = None
            best_win_rate = 0
            best_return = 0

            for version in statistics.keys():
                stats = next((s for s in statistics[version] if s.holding_period == period), None)
                if stats and stats.win_rate > best_win_rate:
                    best_win_rate = stats.win_rate
                    best_version = version
                    best_return = stats.mean_return

            if best_version:
                report_lines.append(f"| {period}天 | {best_version} | {best_win_rate:.2f} | {best_return:.2f} |")

        # 风险收益比
        report_lines.append("\n### 风险收益比（夏普比率近似）\n")
        report_lines.append("| 版本 | 5天 | 10天 | 15天 |")
        report_lines.append("|------|-----|------|------|")

        for version in sorted(statistics.keys()):
            sharpe_ratios = []
            for period in [5, 10, 15]:
                stats = next((s for s in statistics[version] if s.holding_period == period), None)
                if stats and stats.std_return > 0:
                    sharpe = stats.mean_return / stats.std_return
                    sharpe_ratios.append(f"{sharpe:.2f}")
                else:
                    sharpe_ratios.append("N/A")

            report_lines.append(f"| {version} | {' | '.join(sharpe_ratios)} |")

        # 结论和建议
        report_lines.append("\n## 💡 结论与建议\n")
        report_lines.append("\n### 主要发现\n")

        # 找出整体表现最好的版本
        overall_scores = defaultdict(float)
        for version in statistics.keys():
            for stats in statistics[version]:
                # 综合评分 = 平均收益 + 胜率/2
                overall_scores[version] += stats.mean_return + stats.win_rate / 2

        if overall_scores:
            best_overall = max(overall_scores.items(), key=lambda x: x[1])
            report_lines.append(f"1. **综合表现最优**: {best_overall[0]}版本在多个持有期表现最佳")
        else:
            report_lines.append("1. **数据不足**: 无法进行版本对比分析")

        report_lines.append("2. **持有期建议**: 根据平均收益和胜率，选择最优持有期")
        report_lines.append("3. **风险控制**: 关注标准差和最大回撤，合理配置仓位\n")

        report_lines.append("\n### 使用建议\n")
        report_lines.append("- 结合评分版本的收益期望，选择合适的持有策略")
        report_lines.append("- 关注胜率和收益的平衡，避免追求极端收益")
        report_lines.append("- 建议设置止损和止盈位，控制单笔交易风险")
        report_lines.append("- 定期回顾和调整策略参数\n")

        report_lines.append("\n---\n")
        report_lines.append("🤖 Generated with Claude Code\n")

        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"✅ 报告已保存: {output_file}")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("ML评分系统多期收益期望分析器")
    logger.info("=" * 60)

    # 1. 加载历史选股数据
    logger.info("\n[1/4] 加载历史选股数据...")
    loader = HistoricalSelectionLoader()
    all_selections = loader.load_all_selections()

    total_selections = sum(len(selections) for selections in all_selections.values())
    logger.info(f"共加载 {total_selections} 条选股记录")

    # 2. 计算收益率
    logger.info("\n[2/4] 计算多期持有收益率...")
    calculator = ReturnCalculator()

    all_return_records = []
    for version, selections in all_selections.items():
        if not selections:
            continue
        logger.info(f"  处理 {version}: {len(selections)} 条记录")
        records = calculator.calculate_returns(selections)
        all_return_records.extend(records)

    logger.info(f"共计算 {len(all_return_records)} 条有效收益记录")

    # 3. 统计分析
    logger.info("\n[3/4] 进行统计分析...")
    analyzer = StatisticsAnalyzer()
    statistics = analyzer.analyze_returns(all_return_records)

    # 打印统计摘要
    for version, stats_list in statistics.items():
        logger.info(f"\n{version} 统计摘要:")
        for stats in stats_list:
            logger.info(
                f"  {stats.holding_period}天: "
                f"平均收益={stats.mean_return:.2f}%, "
                f"胜率={stats.win_rate:.2f}%, "
                f"有效样本={stats.valid_samples}/{stats.total_samples}"
            )

    # 4. 生成报告
    logger.info("\n[4/4] 生成分析报告...")
    report_dir = Path('reports/ml_scoring_analysis')
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f'收益期望分析报告_{timestamp}.md'

    generator = ReportGenerator()
    generator.generate_report(statistics, str(report_file))

    # 保存详细数据到CSV
    csv_file = report_dir / f'收益详细数据_{timestamp}.csv'
    _save_detailed_csv(all_return_records, str(csv_file))

    logger.info("\n" + "=" * 60)
    logger.info("✅ 分析完成!")
    logger.info(f"📊 报告文件: {report_file}")
    logger.info(f"📊 详细数据: {csv_file}")
    logger.info("=" * 60)


def _save_detailed_csv(return_records: List[ReturnRecord], output_file: str):
    """保存详细收益数据到CSV"""
    data = []
    for record in return_records:
        data.append({
            '日期': record.selection.date,
            '股票代码': record.selection.stock_code,
            '股票名称': record.selection.stock_name,
            '版本': record.selection.version,
            '综合评分': record.selection.comprehensive_score,
            '质量评分': record.selection.quality_score,
            '入场价格': record.entry_price,
            '5天收益率': record.return_5d if record.is_valid_5d else None,
            '10天收益率': record.return_10d if record.is_valid_10d else None,
            '15天收益率': record.return_15d if record.is_valid_15d else None,
            '5天退出价': record.exit_price_5d if record.is_valid_5d else None,
            '10天退出价': record.exit_price_10d if record.is_valid_10d else None,
            '15天退出价': record.exit_price_15d if record.is_valid_15d else None,
        })

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logger.info(f"✅ 详细数据已保存: {output_file}")


if __name__ == '__main__':
    main()
