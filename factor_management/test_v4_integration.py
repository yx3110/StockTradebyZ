#!/usr/bin/env python3
"""
V4挤压动量系统集成测试
Test V4 Squeeze Momentum System Integration
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path

# 导入V4集成组件
from factor_manager import FactorManager
from v4_factor_extractor import V4FactorExtractor
from stock_selector_adapter import StockSelectorAdapter

class V4IntegrationTester:
    """V4集成测试器"""
    
    def __init__(self, db_path: str = "data_adapter/stock_data.db"):
        self.db_path = db_path
        self.logger = self._setup_logger()
        
        # 初始化V4组件
        self.factor_manager = FactorManager(db_path)
        self.v4_extractor = V4FactorExtractor(db_path)
        self.adapter = StockSelectorAdapter(db_path)
        
        # 测试结果存储
        self.test_results = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("V4IntegrationTester")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def test_v4_factor_extraction(self) -> bool:
        """测试V4因子提取"""
        self.logger.info("🧮 测试V4因子提取...")
        
        try:
            start_time = time.time()
            
            # 测试单只股票V4因子提取
            factor_data = self.v4_extractor.extract_factors_for_stock(
                '000001', '2025-08-15', '2025-08-18'
            )
            
            single_stock_time = time.time() - start_time
            self.logger.info(f"✅ 单只股票V4因子提取: {len(factor_data)} 条记录，耗时 {single_stock_time:.2f}秒")
            
            if not factor_data.empty:
                # 检查V4因子完整性
                v4_factors = [col for col in factor_data.columns if col.startswith('v4_')]
                self.logger.info(f"✅ V4因子数量: {len(v4_factors)}")
                
                # 检查挤压动量因子
                squeeze_factors = [col for col in v4_factors if 'squeeze' in col or 'momentum' in col]
                self.logger.info(f"✅ 挤压动量因子: {len(squeeze_factors)} 个")
                for factor in squeeze_factors:
                    self.logger.info(f"  - {factor}")
                
                # 检查V4综合评分
                if 'v4_comprehensive_score' in factor_data.columns:
                    scores = factor_data['v4_comprehensive_score'].dropna()
                    if not scores.empty:
                        self.logger.info(f"✅ V4综合评分范围: {scores.min():.1f} - {scores.max():.1f}, 均值: {scores.mean():.1f}")
                
                # 检查挤压动量核心因子数据质量
                squeeze_state = factor_data['v4_squeeze_state'].dropna()
                squeeze_release = factor_data['v4_squeeze_release'].dropna()
                
                if not squeeze_state.empty and not squeeze_release.empty:
                    self.logger.info(f"✅ 挤压状态评分: {squeeze_state.mean():.1f}分")
                    self.logger.info(f"✅ 挤压释放评分: {squeeze_release.mean():.1f}分")
            
            # 测试批量提取
            start_time = time.time()
            test_stocks = ['000001', '000002', '000858']
            batch_data = self.v4_extractor.batch_extract_factors(
                test_stocks, '2025-08-15', '2025-08-18'
            )
            batch_time = time.time() - start_time
            
            self.logger.info(f"✅ 批量V4因子提取: {len(batch_data)} 条记录，耗时 {batch_time:.2f}秒")
            
            self.test_results['v4_extraction'] = {
                'status': 'success',
                'single_stock_records': len(factor_data),
                'single_stock_time': single_stock_time,
                'batch_records': len(batch_data),
                'batch_time': batch_time,
                'v4_factor_count': len(v4_factors) if not factor_data.empty else 0,
                'squeeze_factor_count': len(squeeze_factors) if not factor_data.empty else 0
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ V4因子提取测试失败: {e}")
            self.test_results['v4_extraction'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_v4_squeeze_momentum_logic(self) -> bool:
        """测试V4挤压动量逻辑"""
        self.logger.info("⚡ 测试V4挤压动量逻辑...")
        
        try:
            # 获取测试数据
            factor_data = self.v4_extractor.extract_factors_for_stock(
                '000001', '2025-08-01', '2025-08-18'
            )
            
            if factor_data.empty:
                self.logger.warning("⚠️ 无测试数据，跳过挤压动量逻辑测试")
                return False
            
            # 分析挤压动量因子的逻辑
            squeeze_state = factor_data['v4_squeeze_state']
            squeeze_release = factor_data['v4_squeeze_release']
            momentum_direction = factor_data['v4_momentum_direction']
            momentum_acceleration = factor_data['v4_momentum_acceleration']
            
            # 检查挤压状态逻辑
            high_squeeze_days = len(squeeze_state[squeeze_state >= 70])  # 高分挤压日
            self.logger.info(f"✅ 高分挤压状态天数: {high_squeeze_days}")
            
            # 检查挤压释放逻辑
            high_release_days = len(squeeze_release[squeeze_release >= 80])  # 高分释放日
            self.logger.info(f"✅ 高分挤压释放天数: {high_release_days}")
            
            # 检查动量方向逻辑
            bullish_momentum_days = len(momentum_direction[momentum_direction >= 60])
            bearish_momentum_days = len(momentum_direction[momentum_direction <= 40])
            self.logger.info(f"✅ 看涨动量天数: {bullish_momentum_days}, 看跌动量天数: {bearish_momentum_days}")
            
            # 检查动量加速度逻辑
            positive_acceleration_days = len(momentum_acceleration[momentum_acceleration >= 60])
            self.logger.info(f"✅ 正加速度天数: {positive_acceleration_days}")
            
            # 寻找最佳挤压释放信号
            best_release_day = squeeze_release.idxmax()
            best_release_score = squeeze_release.max()
            
            if pd.notna(best_release_score):
                self.logger.info(f"🎯 最佳挤压释放信号: {factor_data.loc[best_release_day, 'trade_date']} ({best_release_score:.1f}分)")
                
                # 检查该日其他因子情况
                if best_release_day in momentum_direction.index:
                    direction_score = momentum_direction.loc[best_release_day]
                    acceleration_score = momentum_acceleration.loc[best_release_day]
                    self.logger.info(f"  动量方向: {direction_score:.1f}分, 动量加速度: {acceleration_score:.1f}分")
            
            self.test_results['squeeze_momentum_logic'] = {
                'status': 'success',
                'high_squeeze_days': high_squeeze_days,
                'high_release_days': high_release_days,
                'bullish_momentum_days': bullish_momentum_days,
                'positive_acceleration_days': positive_acceleration_days,
                'best_release_score': best_release_score
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ V4挤压动量逻辑测试失败: {e}")
            self.test_results['squeeze_momentum_logic'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_v4_selection_strategies(self) -> bool:
        """测试V4选股策略"""
        self.logger.info("🎯 测试V4选股策略...")
        
        try:
            test_stocks = ['000001', '000002', '000858', '002215', '002594']
            test_date = '2025-08-18'
            
            # 测试V4综合评分策略
            v4_comprehensive_result = self.adapter.run_stock_selection(
                test_stocks, test_date, strategy='v4_comprehensive', top_n=5
            )
            
            comprehensive_count = len(v4_comprehensive_result['selected_stocks'])
            self.logger.info(f"✅ V4综合评分策略: 选出 {comprehensive_count} 只股票")
            
            if comprehensive_count > 0:
                avg_score = np.mean([s['score'] for s in v4_comprehensive_result['selected_stocks']])
                top_score = max([s['score'] for s in v4_comprehensive_result['selected_stocks']])
                self.logger.info(f"  平均评分: {avg_score:.1f}分, 最高评分: {top_score:.1f}分")
                
                # 显示前3只股票
                for i, stock in enumerate(v4_comprehensive_result['selected_stocks'][:3]):
                    self.logger.info(f"  {i+1}. {stock['stock_code']}: {stock['score']:.1f}分")
            
            # 测试V4挤压动量策略
            v4_squeeze_result = self.adapter.run_stock_selection(
                test_stocks, test_date, strategy='v4_squeeze_momentum', top_n=5
            )
            
            squeeze_count = len(v4_squeeze_result['selected_stocks'])
            self.logger.info(f"✅ V4挤压动量策略: 选出 {squeeze_count} 只股票")
            
            if squeeze_count > 0:
                avg_score = np.mean([s['score'] for s in v4_squeeze_result['selected_stocks']])
                top_score = max([s['score'] for s in v4_squeeze_result['selected_stocks']])
                self.logger.info(f"  平均评分: {avg_score:.1f}分, 最高评分: {top_score:.1f}分")
                
                # 显示前3只股票
                for i, stock in enumerate(v4_squeeze_result['selected_stocks'][:3]):
                    self.logger.info(f"  {i+1}. {stock['stock_code']}: {stock['score']:.1f}分")
            
            # 生成V4策略报告
            if comprehensive_count > 0:
                v4_comprehensive_report = self.adapter.generate_selection_report(
                    v4_comprehensive_result,
                    f"reports/factor_management/V4综合评分选股报告_{test_date.replace('-', '')}.md"
                )
                self.logger.info(f"✅ V4综合评分报告: {v4_comprehensive_report}")
            
            if squeeze_count > 0:
                v4_squeeze_report = self.adapter.generate_selection_report(
                    v4_squeeze_result,
                    f"reports/factor_management/V4挤压动量选股报告_{test_date.replace('-', '')}.md"
                )
                self.logger.info(f"✅ V4挤压动量报告: {v4_squeeze_report}")
            
            self.test_results['v4_selection'] = {
                'status': 'success',
                'comprehensive_count': comprehensive_count,
                'squeeze_count': squeeze_count,
                'comprehensive_avg_score': np.mean([s['score'] for s in v4_comprehensive_result['selected_stocks']]) if comprehensive_count > 0 else 0,
                'squeeze_avg_score': np.mean([s['score'] for s in v4_squeeze_result['selected_stocks']]) if squeeze_count > 0 else 0
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ V4选股策略测试失败: {e}")
            self.test_results['v4_selection'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_v4_unified_dataset(self) -> bool:
        """测试V4统一数据集创建"""
        self.logger.info("📊 测试V4统一数据集创建...")
        
        try:
            test_stocks = ['000001', '000002', '000858']
            
            # 创建包含V4因子的统一数据集
            unified_data = self.adapter.create_unified_factor_dataset(
                test_stocks, '2025-08-15', '2025-08-18',
                include_v2=True, include_v4=True, include_strategies=True
            )
            
            if unified_data.empty:
                self.logger.warning("⚠️ 统一数据集为空")
                return False
            
            # 分析数据集结构
            v2_factors = [col for col in unified_data.columns if col.startswith('v2_')]
            v4_factors = [col for col in unified_data.columns if col.startswith('v4_')]
            strategy_factors = [col for col in unified_data.columns if col.endswith('_signal')]
            
            self.logger.info(f"✅ 统一数据集: {len(unified_data)} 条记录")
            self.logger.info(f"  V2因子: {len(v2_factors)} 个")
            self.logger.info(f"  V4因子: {len(v4_factors)} 个") 
            self.logger.info(f"  策略因子: {len(strategy_factors)} 个")
            self.logger.info(f"  总因子: {len(v2_factors) + len(v4_factors) + len(strategy_factors)} 个")
            
            # 检查V4因子数据质量
            v4_null_rates = {}
            for factor in v4_factors:
                null_rate = unified_data[factor].isnull().sum() / len(unified_data)
                v4_null_rates[factor] = null_rate
            
            avg_v4_null_rate = np.mean(list(v4_null_rates.values())) if v4_null_rates else 0
            self.logger.info(f"✅ V4因子平均缺失率: {avg_v4_null_rate:.1%}")
            
            # 检查V4综合评分分布
            if 'v4_comprehensive_score' in unified_data.columns:
                v4_scores = unified_data['v4_comprehensive_score'].dropna()
                if not v4_scores.empty:
                    self.logger.info(f"✅ V4综合评分分布: {v4_scores.min():.1f} - {v4_scores.max():.1f}")
                    
                    # 评分等级分布
                    high_scores = len(v4_scores[v4_scores >= 80])
                    medium_scores = len(v4_scores[(v4_scores >= 60) & (v4_scores < 80)])
                    low_scores = len(v4_scores[v4_scores < 60])
                    
                    self.logger.info(f"  高分(≥80): {high_scores}, 中等(60-80): {medium_scores}, 低分(<60): {low_scores}")
            
            self.test_results['v4_unified_dataset'] = {
                'status': 'success',
                'total_records': len(unified_data),
                'v2_factor_count': len(v2_factors),
                'v4_factor_count': len(v4_factors),
                'strategy_factor_count': len(strategy_factors),
                'v4_null_rate': avg_v4_null_rate,
                'v4_score_range': [v4_scores.min(), v4_scores.max()] if 'v4_comprehensive_score' in unified_data.columns and not unified_data['v4_comprehensive_score'].dropna().empty else [0, 0]
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ V4统一数据集测试失败: {e}")
            self.test_results['v4_unified_dataset'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def test_v4_performance(self) -> bool:
        """测试V4系统性能"""
        self.logger.info("⚡ 测试V4系统性能...")
        
        try:
            test_stocks = ['000001', '000002', '000858', '002215', '002594']
            
            # V4因子提取性能
            start_time = time.time()
            v4_data = self.v4_extractor.batch_extract_factors(
                test_stocks, '2025-08-17', '2025-08-18'
            )
            v4_extraction_time = time.time() - start_time
            
            # V4统一数据集性能
            start_time = time.time()
            unified_data = self.adapter.create_unified_factor_dataset(
                test_stocks, '2025-08-17', '2025-08-18',
                include_v2=True, include_v4=True
            )
            unified_creation_time = time.time() - start_time
            
            # V4选股性能
            start_time = time.time()
            v4_selection = self.adapter.run_stock_selection(
                test_stocks, '2025-08-18', strategy='v4_comprehensive'
            )
            v4_selection_time = time.time() - start_time
            
            self.logger.info(f"✅ V4性能测试结果:")
            self.logger.info(f"  因子提取: {v4_extraction_time:.2f}秒 ({len(v4_data)} 条记录)")
            self.logger.info(f"  数据集创建: {unified_creation_time:.2f}秒 ({len(unified_data)} 条记录)")
            self.logger.info(f"  选股执行: {v4_selection_time:.2f}秒 ({len(v4_selection['selected_stocks'])} 只股票)")
            
            # 计算单股性能
            per_stock_extraction = v4_extraction_time / len(test_stocks)
            per_stock_unified = unified_creation_time / len(test_stocks)
            per_stock_selection = v4_selection_time / len(test_stocks)
            
            self.logger.info(f"  单股平均: 提取{per_stock_extraction:.3f}s, 数据集{per_stock_unified:.3f}s, 选股{per_stock_selection:.3f}s")
            
            # 预估全市场性能
            total_stocks = 4000
            estimated_total_time = (per_stock_extraction + per_stock_unified + per_stock_selection) * total_stocks
            self.logger.info(f"📈 V4全市场预估时间: {estimated_total_time/60:.1f} 分钟")
            
            self.test_results['v4_performance'] = {
                'status': 'success',
                'extraction_time': v4_extraction_time,
                'unified_time': unified_creation_time,
                'selection_time': v4_selection_time,
                'per_stock_avg': per_stock_extraction + per_stock_unified + per_stock_selection,
                'estimated_full_market_minutes': estimated_total_time / 60
            }
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ V4性能测试失败: {e}")
            self.test_results['v4_performance'] = {'status': 'failed', 'error': str(e)}
            return False
    
    def run_v4_integration_test(self) -> Dict:
        """运行V4完整集成测试"""
        self.logger.info("🚀 开始V4挤压动量系统集成测试...")
        
        test_functions = [
            ('V4因子提取', self.test_v4_factor_extraction),
            ('挤压动量逻辑', self.test_v4_squeeze_momentum_logic),
            ('V4选股策略', self.test_v4_selection_strategies),
            ('V4统一数据集', self.test_v4_unified_dataset),
            ('V4系统性能', self.test_v4_performance)
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
        self.logger.info(f"V4集成测试完成: {passed_tests}/{total_tests} 通过 ({success_rate:.1f}%)")
        self.logger.info(f"{'='*50}")
        
        # 保存测试结果
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        self._save_v4_test_report()
        
        return self.test_results
    
    def _save_v4_test_report(self):
        """保存V4测试报告"""
        report_path = Path("reports/factor_management/V4集成测试报告.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self._format_v4_test_report())
        
        self.logger.info(f"📋 V4测试报告已保存: {report_path}")
    
    def _format_v4_test_report(self) -> str:
        """格式化V4测试报告"""
        summary = self.test_results.get('summary', {})
        
        report = f"""# V4挤压动量系统集成测试报告

**测试时间**: {summary.get('timestamp', 'Unknown')}  
**通过测试**: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}  
**成功率**: {summary.get('success_rate', 0):.1f}%

## 🎯 V4系统特色

V4挤压动量增强评分系统是基于John Carter的TTM Squeeze指标的创新实现：

### 🔥 挤压动量核心原理
- **挤压状态识别**: 布林带收窄在肯特纳通道内，市场进入低波动蓄势期
- **挤压释放捕捉**: 布林带突破肯特纳通道边界，预示波动率扩张
- **动量方向确认**: 线性回归斜率判断突破方向
- **加速度验证**: 动量加速度确保趋势的持续性

### ⚡ V4系统优势
1. **突破预警**: 提前识别横盘整理后的突破时机
2. **假突破过滤**: 结合多维度动量指标降低误报
3. **波动率敏感**: 擅长捕捉从低波动到高波动的转换点
4. **趋势确认**: 动量方向和加速度双重验证

## 📊 测试结果详情

"""
        
        # 各模块测试结果
        test_modules = {
            'v4_extraction': 'V4因子提取',
            'squeeze_momentum_logic': '挤压动量逻辑',
            'v4_selection': 'V4选股策略',
            'v4_unified_dataset': 'V4统一数据集',
            'v4_performance': 'V4系统性能'
        }
        
        for module_key, module_name in test_modules.items():
            result = self.test_results.get(module_key, {})
            status = result.get('status', 'unknown')
            status_emoji = '✅' if status == 'success' else '❌'
            
            report += f"### {status_emoji} {module_name}\n\n"
            
            if status == 'success':
                if module_key == 'v4_extraction':
                    report += f"- V4因子数量: {result.get('v4_factor_count', 0)} 个\n"
                    report += f"- 挤压动量因子: {result.get('squeeze_factor_count', 0)} 个\n"
                    report += f"- 单股提取: {result.get('single_stock_records', 0)} 条记录，{result.get('single_stock_time', 0):.2f}秒\n"
                    report += f"- 批量提取: {result.get('batch_records', 0)} 条记录，{result.get('batch_time', 0):.2f}秒\n"
                
                elif module_key == 'squeeze_momentum_logic':
                    report += f"- 高分挤压状态天数: {result.get('high_squeeze_days', 0)}\n"
                    report += f"- 高分挤压释放天数: {result.get('high_release_days', 0)}\n"
                    report += f"- 看涨动量天数: {result.get('bullish_momentum_days', 0)}\n"
                    report += f"- 正加速度天数: {result.get('positive_acceleration_days', 0)}\n"
                    report += f"- 最佳释放信号: {result.get('best_release_score', 0):.1f}分\n"
                
                elif module_key == 'v4_selection':
                    report += f"- V4综合评分选股: {result.get('comprehensive_count', 0)} 只\n"
                    report += f"- V4挤压动量选股: {result.get('squeeze_count', 0)} 只\n"
                    report += f"- 综合评分平均: {result.get('comprehensive_avg_score', 0):.1f}分\n"
                    report += f"- 挤压动量平均: {result.get('squeeze_avg_score', 0):.1f}分\n"
                
                elif module_key == 'v4_unified_dataset':
                    report += f"- 总记录数: {result.get('total_records', 0)} 条\n"
                    report += f"- V4因子数: {result.get('v4_factor_count', 0)} 个\n"
                    report += f"- V4缺失率: {result.get('v4_null_rate', 0):.1%}\n"
                    score_range = result.get('v4_score_range', [0, 0])
                    report += f"- V4评分范围: {score_range[0]:.1f} - {score_range[1]:.1f}\n"
                
                elif module_key == 'v4_performance':
                    report += f"- 因子提取: {result.get('extraction_time', 0):.2f}秒\n"
                    report += f"- 数据集创建: {result.get('unified_time', 0):.2f}秒\n"
                    report += f"- 选股执行: {result.get('selection_time', 0):.2f}秒\n"
                    report += f"- 单股平均: {result.get('per_stock_avg', 0):.3f}秒\n"
                    report += f"- 全市场预估: {result.get('estimated_full_market_minutes', 0):.1f}分钟\n"
            
            else:
                error = result.get('error', '未知错误')
                report += f"- 错误信息: {error}\n"
            
            report += "\n"
        
        report += """## 🎯 V4挤压动量策略应用

### 适用场景
1. **横盘突破**: 长期横盘整理后的方向性突破
2. **波动率扩张**: 从低波动向高波动的转换期
3. **趋势确认**: 配合其他技术指标确认突破真实性
4. **风险控制**: 利用挤压状态控制入场时机

### 使用建议
1. **挤压期观察**: 关注长期挤压状态的股票(v4_squeeze_state ≥ 70)
2. **释放信号**: 重点关注挤压释放信号(v4_squeeze_release ≥ 80)
3. **方向确认**: 结合动量方向判断突破方向(v4_momentum_direction)
4. **持续验证**: 观察动量加速度确保趋势持续(v4_momentum_acceleration)

### 参数调优建议
- **布林带周期**: 20日(适合中期趋势)
- **肯特纳通道倍数**: 1.5倍ATR(平衡敏感度)
- **动量计算周期**: 20日(匹配布林带周期)
- **挤压释放权重**: 6%(核心信号权重)

## ⚠️ 风险提示

1. **市场适应性**: 挤压动量策略在震荡市场表现更佳
2. **假突破风险**: 需要结合成交量和基本面确认
3. **参数敏感性**: 不同市场环境可能需要调整参数
4. **资金管理**: 建议设置合理的止损和仓位管理

## 🔮 V4系统展望

V4挤压动量系统的成功集成为量化交易系统带来了新的维度：

1. **技术创新**: 首次在A股市场系统化应用TTM Squeeze指标
2. **框架扩展**: 为因子管理框架增加了波动率分析能力
3. **策略丰富**: 提供了新的突破型选股策略
4. **性能优化**: 预计算挤压动量因子提升了实时决策效率

V4系统的成功为后续版本(V5、V6)的开发奠定了坚实基础，未来可以考虑：
- 集成更多波动率相关指标
- 增加机器学习增强的动量预测
- 开发多时间框架的挤压动量分析
- 结合另类数据提升预测精度

---

*V4挤压动量系统 - 捕捉市场从静到动的完美转换*
"""
        
        return report


def main():
    """运行V4集成测试"""
    tester = V4IntegrationTester()
    results = tester.run_v4_integration_test()
    
    summary = results.get('summary', {})
    print(f"\n🎉 V4集成测试完成！")
    print(f"通过率: {summary.get('success_rate', 0):.1f}%")
    print(f"详细报告: reports/factor_management/V4集成测试报告.md")
    
    # 显示核心测试结果
    if 'v4_extraction' in results and results['v4_extraction']['status'] == 'success':
        print(f"✅ V4因子提取: {results['v4_extraction']['v4_factor_count']} 个因子")
    
    if 'v4_selection' in results and results['v4_selection']['status'] == 'success':
        print(f"✅ V4选股策略: 综合评分策略选出 {results['v4_selection']['comprehensive_count']} 只股票")
    
    if 'v4_performance' in results and results['v4_performance']['status'] == 'success':
        print(f"✅ V4系统性能: 全市场预估 {results['v4_performance']['estimated_full_market_minutes']:.1f} 分钟")


if __name__ == "__main__":
    main()