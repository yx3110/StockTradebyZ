#!/usr/bin/env python3
"""
因子管理框架与选股系统集成测试
Integration Test for Factor Management Framework and Stock Selection System
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path

# 导入集成组件
from factor_manager import FactorManager
from v2_factor_extractor import V2FactorExtractor
from stock_selector_adapter import StockSelectorAdapter

class IntegrationTester:
    """集成测试器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.logger = self._setup_logger()
        
        # 初始化组件
        self.factor_manager = FactorManager(db_path)
        self.v2_extractor = V2FactorExtractor(db_path)
        self.adapter = StockSelectorAdapter(db_path)
        
        # 测试结果存储
        self.test_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("IntegrationTester")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def test_database_initialization(self) -> bool:
        """测试数据库初始化"""
        self.logger.info("🔧 测试数据库初始化...")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 检查必要的表是否存在
                required_tables = [
                    'securities', 'daily_quotes', 'technical_indicators',
                    'factor_definitions', 'technical_factors', 'market_factors'
                ]
                
                existing_tables = []
                for table in required_tables:
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name=?
                    """, (table,))
                    if cursor.fetchone():
                        existing_tables.append(table)
                
                self.logger.info(f"✅ 找到 {len(existing_tables)}/{len(required_tables)} 个必要表")
                
                # 检查数据量
                cursor.execute("SELECT COUNT(*) FROM securities WHERE type = 'A股'")
                stock_count = cursor.fetchone()[0]
                self.logger.info(f"✅ A股数据: {stock_count} 只")
                
                # 检查最新交易日期
                cursor.execute("SELECT MAX(trade_date) FROM daily_quotes")
                latest_date = cursor.fetchone()[0]
                self.logger.info(f"✅ 最新交易日期: {latest_date}")
                
                self.test_results['database_init'] = {
                    'status': 'success',
                    'existing_tables': len(existing_tables),
                    'required_tables': len(required_tables),
                    'stock_count': stock_count,
                    'latest_date': latest_date
                }
                
                return len(existing_tables) == len(required_tables) and stock_count > 0
                
        except Exception as e:
            self.logger.error(f"❌ 数据库初始化测试失败: {e}")
            self.test_results['database_init'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_factor_manager(self) -> bool:
        """测试因子管理器"""
        self.logger.info("🧮 测试因子管理器...")
        
        try:
            # 测试因子注册
            initial_count = len(self.factor_manager.factor_registry)
            
            # 注册测试因子
            self.factor_manager.register_factor(
                name="test_factor",
                category="test",
                description="测试因子",
                dependencies=["close"],
                calculator=lambda df: df['close'].pct_change() * 100
            )
            
            after_register_count = len(self.factor_manager.factor_registry)
            self.logger.info(f"✅ 因子注册测试通过: {initial_count} -> {after_register_count}")
            
            # 测试因子数据获取
            test_stocks = ['000001', '000002']
            factor_data = self.factor_manager.get_factor_data(
                test_stocks, 
                ['momentum_5d', 'volatility_20d'],
                '2025-08-01', '2025-08-18'
            )
            
            self.logger.info(f"✅ 因子数据获取测试: {len(factor_data)} 条记录")
            
            # 测试综合评分
            score = self.factor_manager.calculate_composite_score('000001', '2025-08-18')
            self.logger.info(f"✅ 综合评分测试: {score:.2f}分")
            
            self.test_results['factor_manager'] = {
                'status': 'success',
                'factor_count': after_register_count,
                'data_records': len(factor_data),
                'composite_score': score
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 因子管理器测试失败: {e}")
            self.test_results['factor_manager'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_v2_extractor(self) -> bool:
        """测试V2因子提取器"""
        self.logger.info("📊 测试V2因子提取器...")
        
        try:
            start_time = time.time()
            
            # 测试单只股票因子提取
            factor_data = self.v2_extractor.extract_factors_for_stock(
                '000001', '2025-08-15', '2025-08-18'
            )
            
            single_stock_time = time.time() - start_time
            self.logger.info(f"✅ 单只股票因子提取: {len(factor_data)} 条记录，耗时 {single_stock_time:.2f}秒")
            
            if not factor_data.empty:
                # 检查因子完整性
                v2_factors = [col for col in factor_data.columns if col.startswith('v2_')]
                self.logger.info(f"✅ V2因子数量: {len(v2_factors)}")
                
                # 检查数据质量
                null_counts = factor_data[v2_factors].isnull().sum()
                self.logger.info(f"✅ 数据完整性: 缺失值最多 {null_counts.max()} 个")
            
            # 测试批量提取
            start_time = time.time()
            test_stocks = ['000001', '000002', '000858']
            batch_data = self.v2_extractor.batch_extract_factors(
                test_stocks, '2025-08-15', '2025-08-18'
            )
            batch_time = time.time() - start_time
            
            self.logger.info(f"✅ 批量因子提取: {len(batch_data)} 条记录，耗时 {batch_time:.2f}秒")
            
            self.test_results['v2_extractor'] = {
                'status': 'success',
                'single_stock_records': len(factor_data),
                'single_stock_time': single_stock_time,
                'batch_records': len(batch_data),
                'batch_time': batch_time,
                'factor_count': len(v2_factors) if not factor_data.empty else 0
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ V2因子提取器测试失败: {e}")
            self.test_results['v2_extractor'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_stock_selector_adapter(self) -> bool:
        """测试选股系统适配器"""
        self.logger.info("🎯 测试选股系统适配器...")
        
        try:
            start_time = time.time()
            
            # 测试统一因子数据集创建
            test_stocks = ['000001', '000002', '000858', '002215']
            unified_data = self.adapter.create_unified_factor_dataset(
                test_stocks, '2025-08-15', '2025-08-18'
            )
            
            dataset_time = time.time() - start_time
            self.logger.info(f"✅ 统一因子数据集: {len(unified_data)} 条记录，耗时 {dataset_time:.2f}秒")
            
            if not unified_data.empty:
                # 检查因子类型
                v2_factors = [col for col in unified_data.columns if col.startswith('v2_')]
                strategy_factors = [col for col in unified_data.columns if col.endswith('_signal')]
                self.logger.info(f"✅ V2因子: {len(v2_factors)} 个")
                self.logger.info(f"✅ 策略因子: {len(strategy_factors)} 个")
            
            # 测试不同策略选股
            strategies = ['combined', 'v2_composite', 'bbi_kdj', 'breakout_volume']
            strategy_results = {}
            
            for strategy in strategies:
                start_time = time.time()
                selection_result = self.adapter.run_stock_selection(
                    test_stocks, '2025-08-18', strategy=strategy, top_n=3
                )
                selection_time = time.time() - start_time
                
                selected_count = len(selection_result['selected_stocks'])
                avg_score = np.mean([s['score'] for s in selection_result['selected_stocks']]) if selected_count > 0 else 0
                
                strategy_results[strategy] = {
                    'selected_count': selected_count,
                    'avg_score': avg_score,
                    'time': selection_time
                }
                
                self.logger.info(f"✅ {strategy} 策略: 选出 {selected_count} 只，平均分 {avg_score:.1f}，耗时 {selection_time:.2f}秒")
            
            # 测试报告生成
            test_result = self.adapter.run_stock_selection(
                test_stocks, '2025-08-18', strategy='combined', top_n=3
            )
            
            report_path = self.adapter.generate_selection_report(test_result)
            report_exists = Path(report_path).exists()
            
            self.logger.info(f"✅ 报告生成: {'成功' if report_exists else '失败'}")
            
            self.test_results['adapter'] = {
                'status': 'success',
                'dataset_records': len(unified_data),
                'dataset_time': dataset_time,
                'v2_factor_count': len(v2_factors) if not unified_data.empty else 0,
                'strategy_factor_count': len(strategy_factors) if not unified_data.empty else 0,
                'strategy_results': strategy_results,
                'report_generated': report_exists
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 选股系统适配器测试失败: {e}")
            self.test_results['adapter'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_performance_benchmark(self) -> bool:
        """性能基准测试"""
        self.logger.info("⚡ 性能基准测试...")
        
        try:
            # 测试数据量
            stock_counts = [10, 50, 100]
            performance_data = {}
            
            with sqlite3.connect(self.db_path) as conn:
                # 获取测试股票池
                cursor = conn.cursor()
                cursor.execute("SELECT code FROM securities WHERE type = 'A股' LIMIT 100")
                all_test_stocks = [row[0] for row in cursor.fetchall()]
            
            for count in stock_counts:
                if count > len(all_test_stocks):
                    continue
                    
                test_stocks = all_test_stocks[:count]
                
                # V2因子提取性能
                start_time = time.time()
                v2_data = self.v2_extractor.batch_extract_factors(
                    test_stocks, '2025-08-17', '2025-08-18'
                )
                v2_time = time.time() - start_time
                
                # 统一数据集创建性能
                start_time = time.time()
                unified_data = self.adapter.create_unified_factor_dataset(
                    test_stocks, '2025-08-17', '2025-08-18'
                )
                unified_time = time.time() - start_time
                
                # 选股性能
                start_time = time.time()
                selection_result = self.adapter.run_stock_selection(
                    test_stocks, '2025-08-18', strategy='combined'
                )
                selection_time = time.time() - start_time
                
                performance_data[count] = {
                    'v2_extraction_time': v2_time,
                    'v2_records': len(v2_data),
                    'unified_dataset_time': unified_time,
                    'unified_records': len(unified_data),
                    'selection_time': selection_time,
                    'selected_stocks': len(selection_result['selected_stocks'])
                }
                
                self.logger.info(f"✅ {count}只股票: V2提取 {v2_time:.2f}s, 数据集 {unified_time:.2f}s, 选股 {selection_time:.2f}s")
            
            # 计算平均性能
            if performance_data:
                avg_v2_per_stock = np.mean([data['v2_extraction_time']/count for count, data in performance_data.items()])
                avg_unified_per_stock = np.mean([data['unified_dataset_time']/count for count, data in performance_data.items()])
                avg_selection_per_stock = np.mean([data['selection_time']/count for count, data in performance_data.items()])
                
                self.logger.info(f"✅ 平均每只股票: V2提取 {avg_v2_per_stock:.3f}s, 数据集 {avg_unified_per_stock:.3f}s, 选股 {avg_selection_per_stock:.3f}s")
                
                # 预估全市场性能
                total_stocks = 4000  # 假设全A股4000只
                estimated_total_time = (avg_v2_per_stock + avg_unified_per_stock + avg_selection_per_stock) * total_stocks
                self.logger.info(f"📈 预估全市场处理时间: {estimated_total_time/60:.1f} 分钟")
            
            self.test_results['performance'] = {
                'status': 'success',
                'benchmark_data': performance_data,
                'avg_per_stock': {
                    'v2_extraction': avg_v2_per_stock,
                    'unified_dataset': avg_unified_per_stock,
                    'selection': avg_selection_per_stock
                } if performance_data else {},
                'estimated_full_market_minutes': estimated_total_time/60 if performance_data else 0
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 性能基准测试失败: {e}")
            self.test_results['performance'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_data_quality(self) -> bool:
        """数据质量测试"""
        self.logger.info("🔍 数据质量测试...")
        
        try:
            test_stocks = ['000001', '000002', '000858']
            
            # 创建测试数据集
            unified_data = self.adapter.create_unified_factor_dataset(
                test_stocks, '2025-08-15', '2025-08-18'
            )
            
            if unified_data.empty:
                self.logger.warning("⚠️ 统一数据集为空，跳过数据质量测试")
                return False
            
            # 检查数据完整性
            total_records = len(unified_data)
            
            # V2因子数据质量
            v2_factors = [col for col in unified_data.columns if col.startswith('v2_')]
            v2_null_rates = {}
            for factor in v2_factors:
                null_rate = unified_data[factor].isnull().sum() / total_records
                v2_null_rates[factor] = null_rate
            
            avg_v2_null_rate = np.mean(list(v2_null_rates.values())) if v2_null_rates else 0
            self.logger.info(f"✅ V2因子平均缺失率: {avg_v2_null_rate:.1%}")
            
            # 策略因子数据质量
            strategy_factors = [col for col in unified_data.columns if col.endswith('_signal')]
            strategy_null_rates = {}
            for factor in strategy_factors:
                null_rate = unified_data[factor].isnull().sum() / total_records
                strategy_null_rates[factor] = null_rate
            
            avg_strategy_null_rate = np.mean(list(strategy_null_rates.values())) if strategy_null_rates else 0
            self.logger.info(f"✅ 策略因子平均缺失率: {avg_strategy_null_rate:.1%}")
            
            # 检查数值范围合理性
            v2_composite_scores = unified_data['v2_composite_score'].dropna()
            if not v2_composite_scores.empty:
                score_min, score_max = v2_composite_scores.min(), v2_composite_scores.max()
                score_mean = v2_composite_scores.mean()
                self.logger.info(f"✅ V2综合评分范围: {score_min:.1f} - {score_max:.1f}, 均值: {score_mean:.1f}")
            
            strategy_scores = unified_data['combined_strategy_signal'].dropna()
            if not strategy_scores.empty:
                strategy_min, strategy_max = strategy_scores.min(), strategy_scores.max()
                strategy_mean = strategy_scores.mean()
                self.logger.info(f"✅ 策略信号范围: {strategy_min:.1f} - {strategy_max:.1f}, 均值: {strategy_mean:.1f}")
            
            # 检查异常值
            extreme_v2_count = len(v2_composite_scores[(v2_composite_scores < 0) | (v2_composite_scores > 100)])
            extreme_strategy_count = len(strategy_scores[(strategy_scores < 0) | (strategy_scores > 100)])
            
            self.logger.info(f"✅ V2评分异常值: {extreme_v2_count} 个")
            self.logger.info(f"✅ 策略信号异常值: {extreme_strategy_count} 个")
            
            quality_score = 100 - (avg_v2_null_rate + avg_strategy_null_rate) * 50 - (extreme_v2_count + extreme_strategy_count)
            self.logger.info(f"📊 整体数据质量评分: {quality_score:.1f}/100")
            
            self.test_results['data_quality'] = {
                'status': 'success',
                'total_records': total_records,
                'v2_null_rate': avg_v2_null_rate,
                'strategy_null_rate': avg_strategy_null_rate,
                'v2_score_range': [score_min, score_max] if not v2_composite_scores.empty else [0, 0],
                'strategy_score_range': [strategy_min, strategy_max] if not strategy_scores.empty else [0, 0],
                'extreme_values': extreme_v2_count + extreme_strategy_count,
                'quality_score': quality_score
            }
            
            return quality_score > 80
            
        except Exception as e:
            self.logger.error(f"❌ 数据质量测试失败: {e}")
            self.test_results['data_quality'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_full_test_suite(self) -> Dict:
        """运行完整测试套件"""
        self.logger.info("🚀 开始因子管理框架集成测试...")
        
        test_functions = [
            ('数据库初始化', self.test_database_initialization),
            ('因子管理器', self.test_factor_manager),
            ('V2因子提取器', self.test_v2_extractor),
            ('选股系统适配器', self.test_stock_selector_adapter),
            ('性能基准测试', self.test_performance_benchmark),
            ('数据质量测试', self.test_data_quality)
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_name, test_func in test_functions:
            try:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"开始测试: {test_name}")
                self.logger.info(f"{'='*50}")
                
                result = test_func()
                if result:
                    passed_tests += 1
                    self.logger.info(f"✅ {test_name} 测试通过")
                else:
                    self.logger.error(f"❌ {test_name} 测试失败")
                    
            except Exception as e:
                self.logger.error(f"❌ {test_name} 测试异常: {e}")
        
        # 生成测试报告
        success_rate = (passed_tests / total_tests) * 100
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"测试完成: {passed_tests}/{total_tests} 通过 ({success_rate:.1f}%)")
        self.logger.info(f"{'='*50}")
        
        # 保存测试结果
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self._save_test_report()
        
        return self.test_results
    
    def _save_test_report(self):
        """保存测试报告"""
        report_path = Path("reports/factor_management/集成测试报告.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._format_test_report())
        
        self.logger.info(f"📋 测试报告已保存: {report_path}")
    
    def _format_test_report(self) -> str:
        """格式化测试报告"""
        summary = self.test_results.get('summary', {})
        
        report = f"""# 因子管理框架集成测试报告

**测试时间**: {summary.get('timestamp', 'Unknown')}  
**通过测试**: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}  
**成功率**: {summary.get('success_rate', 0):.1f}%

## 📊 测试结果汇总

"""
        
        # 各模块测试结果
        test_modules = {
            'database_init': '数据库初始化',
            'factor_manager': '因子管理器',
            'v2_extractor': 'V2因子提取器',
            'adapter': '选股系统适配器',
            'performance': '性能基准测试',
            'data_quality': '数据质量测试'
        }
        
        for module_key, module_name in test_modules.items():
            result = self.test_results.get(module_key, {})
            status = result.get('status', 'unknown')
            status_emoji = '✅' if status == 'success' else '❌'
            
            report += f"### {status_emoji} {module_name}\n\n"
            
            if status == 'success':
                if module_key == 'database_init':
                    report += f"- 股票数据: {result.get('stock_count', 0)} 只\n"
                    report += f"- 最新日期: {result.get('latest_date', 'Unknown')}\n"
                    report += f"- 数据表: {result.get('existing_tables', 0)}/{result.get('required_tables', 0)}\n"
                
                elif module_key == 'factor_manager':
                    report += f"- 注册因子: {result.get('factor_count', 0)} 个\n"
                    report += f"- 数据记录: {result.get('data_records', 0)} 条\n"
                    report += f"- 综合评分: {result.get('composite_score', 0):.2f}分\n"
                
                elif module_key == 'v2_extractor':
                    report += f"- 单股提取: {result.get('single_stock_records', 0)} 条记录\n"
                    report += f"- 单股耗时: {result.get('single_stock_time', 0):.2f}秒\n"
                    report += f"- 批量提取: {result.get('batch_records', 0)} 条记录\n"
                    report += f"- 批量耗时: {result.get('batch_time', 0):.2f}秒\n"
                    report += f"- 因子数量: {result.get('factor_count', 0)} 个\n"
                
                elif module_key == 'adapter':
                    report += f"- 数据集记录: {result.get('dataset_records', 0)} 条\n"
                    report += f"- 数据集耗时: {result.get('dataset_time', 0):.2f}秒\n"
                    report += f"- V2因子: {result.get('v2_factor_count', 0)} 个\n"
                    report += f"- 策略因子: {result.get('strategy_factor_count', 0)} 个\n"
                    
                    strategy_results = result.get('strategy_results', {})
                    for strategy, data in strategy_results.items():
                        report += f"- {strategy}: {data.get('selected_count', 0)}只股票, 平均{data.get('avg_score', 0):.1f}分\n"
                
                elif module_key == 'performance':
                    avg_per_stock = result.get('avg_per_stock', {})
                    report += f"- V2提取: {avg_per_stock.get('v2_extraction', 0):.3f}秒/股\n"
                    report += f"- 数据集: {avg_per_stock.get('unified_dataset', 0):.3f}秒/股\n"
                    report += f"- 选股: {avg_per_stock.get('selection', 0):.3f}秒/股\n"
                    report += f"- 全市场预估: {result.get('estimated_full_market_minutes', 0):.1f}分钟\n"
                
                elif module_key == 'data_quality':
                    report += f"- 总记录数: {result.get('total_records', 0)} 条\n"
                    report += f"- V2因子缺失率: {result.get('v2_null_rate', 0):.1%}\n"
                    report += f"- 策略因子缺失率: {result.get('strategy_null_rate', 0):.1%}\n"
                    report += f"- 异常值: {result.get('extreme_values', 0)} 个\n"
                    report += f"- 质量评分: {result.get('quality_score', 0):.1f}/100\n"
            
            else:
                error = result.get('error', '未知错误')
                report += f"- 错误信息: {error}\n"
            
            report += "\n"
        
        report += """## 🎯 集成效果评估

### 优势
1. **统一管理**: 所有因子通过统一框架管理，便于维护和扩展
2. **性能优化**: 预计算因子存储，查询效率高
3. **版本控制**: 支持因子版本管理，便于对比测试
4. **标准接口**: 统一的数据获取和计算接口

### 建议优化点
1. **缓存机制**: 添加因子计算结果缓存，减少重复计算
2. **并行处理**: 增强批量处理的并行能力
3. **实时更新**: 支持因子数据的实时更新
4. **监控告警**: 添加数据质量监控和异常告警

## ⚠️ 注意事项

1. **数据依赖**: 确保基础数据完整性，定期检查数据质量
2. **性能监控**: 关注系统性能，及时优化慢查询
3. **版本兼容**: 升级因子定义时注意版本兼容性
4. **备份策略**: 定期备份因子数据和配置

---

*报告由集成测试系统自动生成*
"""
        
        return report


def main():
    """运行集成测试"""
    tester = IntegrationTester()
    results = tester.run_full_test_suite()
    
    summary = results.get('summary', {})
    print(f"\n🎉 集成测试完成！")
    print(f"通过率: {summary.get('success_rate', 0):.1f}%")
    print(f"详细报告: reports/factor_management/集成测试报告.md")


if __name__ == "__main__":
    main()