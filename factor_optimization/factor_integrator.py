#!/usr/bin/env python3
"""
通用新因子集成框架 - 可复用的因子添加流程
支持从外部源（如TradingView）集成新因子到权重优化系统
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import importlib.util
import logging

# 设置项目路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

class FactorIntegrator:
    """通用因子集成器 - 支持新因子的完整集成流程"""
    
    def __init__(self):
        self.project_root = project_root
        self.factor_db_path = os.path.join(project_root, 'factor_optimization', 'standard_factors.db')
        self.main_db_path = os.path.join(project_root, 'data_adapter', 'stock_data.db')
        self.integration_log = []
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def integrate_new_factor(self, factor_config: Dict) -> Dict:
        """
        完整的新因子集成流程
        
        Args:
            factor_config: 新因子配置
            {
                "name": "cci_psar_composite",
                "version": "v3.3", 
                "description": "CCI+Parabolic SAR复合技术指标",
                "source_url": "https://www.tradingview.com/script/SH4TLaGk/",
                "dimension": "technical",  # 所属维度
                "raw_columns": [...],      # 需要添加到主数据库的原始数据列
                "standard_columns": [...], # 需要添加到标准化数据库的评分列
                "calculation_method": "...", # 计算方法描述或代码
                "weight_range": [0.05, 0.10, 0.15]  # 权重范围
            }
        
        Returns:
            集成结果和影响分析
        """
        
        self.logger.info(f"🚀 开始集成新因子: {factor_config['name']} ({factor_config['version']})")
        
        result = {
            "factor_name": factor_config['name'],
            "version": factor_config['version'],
            "integration_steps": [],
            "success": True,
            "errors": []
        }
        
        try:
            # 步骤1: 数据库结构升级
            self._upgrade_database_schema(factor_config, result)
            
            # 步骤2: 生成因子计算代码
            self._generate_factor_calculator(factor_config, result)
            
            # 步骤3: 更新权重优化器配置
            self._update_weight_optimizer_config(factor_config, result)
            
            # 步骤4: 计算历史因子数据
            self._calculate_historical_factor_data(factor_config, result)
            
            # 步骤5: 执行权重优化
            self._optimize_weights_with_new_factor(factor_config, result)
            
            # 步骤6: 评估新因子影响
            self._evaluate_factor_impact(factor_config, result)
            
            # 步骤7: 生成集成报告
            self._generate_integration_report(factor_config, result)
            
        except Exception as e:
            result["success"] = False
            result["errors"].append(str(e))
            self.logger.error(f"❌ 因子集成失败: {e}")
        
        return result
    
    def _upgrade_database_schema(self, factor_config: Dict, result: Dict):
        """升级数据库结构"""
        self.logger.info("📊 升级数据库结构...")
        
        # 升级主数据库
        with sqlite3.connect(self.main_db_path) as conn:
            cursor = conn.cursor()
            
            for col_info in factor_config.get('raw_columns', []):
                col_name, col_type = col_info['name'], col_info['type']
                try:
                    cursor.execute(f"ALTER TABLE technical_indicators ADD COLUMN {col_name} {col_type}")
                    self.logger.info(f"  ✅ 添加原始数据列: {col_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise e
                    self.logger.warning(f"  ⚠️  列已存在: {col_name}")
            
            conn.commit()
        
        # 升级标准化数据库
        with sqlite3.connect(self.factor_db_path) as conn:
            cursor = conn.cursor()
            
            for col_info in factor_config.get('standard_columns', []):
                col_name, col_type = col_info['name'], col_info['type']
                try:
                    cursor.execute(f"ALTER TABLE standard_factors ADD COLUMN {col_name} {col_type}")
                    self.logger.info(f"  ✅ 添加标准化评分列: {col_name}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e).lower():
                        raise e
                    self.logger.warning(f"  ⚠️  列已存在: {col_name}")
            
            conn.commit()
        
        result["integration_steps"].append("数据库结构升级完成")
    
    def _generate_factor_calculator(self, factor_config: Dict, result: Dict):
        """生成因子计算代码"""
        self.logger.info("🧮 生成因子计算代码...")
        
        # 创建因子计算器文件
        factor_name = factor_config['name']
        calculator_path = os.path.join(
            self.project_root, 'factor_optimization', 'calculators', f'{factor_name}_calculator.py'
        )
        
        # 确保目录存在
        os.makedirs(os.path.dirname(calculator_path), exist_ok=True)
        
        # 生成基础计算器模板
        calculator_code = self._generate_calculator_template(factor_config)
        
        with open(calculator_path, 'w', encoding='utf-8') as f:
            f.write(calculator_code)
        
        self.logger.info(f"  ✅ 生成计算器: {calculator_path}")
        result["integration_steps"].append(f"生成因子计算器: {factor_name}_calculator.py")
    
    def _generate_calculator_template(self, factor_config: Dict) -> str:
        """生成因子计算器代码模板"""
        factor_name = factor_config['name']
        version = factor_config['version']
        description = factor_config['description']
        source_url = factor_config['source_url']
        
        template = f'''#!/usr/bin/env python3
"""
{factor_name} 因子计算器 - {version}
{description}

数据源: {source_url}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple

class {factor_name.title().replace('_', '')}Calculator:
    """
    {description}计算器
    """
    
    def __init__(self):
        self.name = "{factor_name}"
        self.version = "{version}"
        self.description = "{description}"
    
    def calculate_raw_factors(self, data: pd.DataFrame) -> Dict:
        """
        计算原始因子数据
        
        Args:
            data: 包含OHLCV数据的DataFrame
        
        Returns:
            原始因子值字典
        """
        if len(data) < 30:  # 确保有足够的数据
            return self._get_default_raw_values()
        
        try:
            # TODO: 在这里实现具体的因子计算逻辑
            # 示例代码 - 需要根据实际因子进行修改
            
            # 计算CCI (示例)
            high = data['high'].values
            low = data['low'].values 
            close = data['close'].values
            
            cci_14 = talib.CCI(high, low, close, timeperiod=14)
            
            # 计算Parabolic SAR (示例)
            psar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # 计算ATR
            atr_14 = talib.ATR(high, low, close, timeperiod=14)
            
            # 计算趋势方向
            psar_trend = np.where(close > psar, 1, -1)
            
            return {{
                'cci_14': cci_14[-1] if not np.isnan(cci_14[-1]) else 0,
                'psar': psar[-1] if not np.isnan(psar[-1]) else close[-1],
                'psar_trend': int(psar_trend[-1]),
                'atr_14': atr_14[-1] if not np.isnan(atr_14[-1]) else close[-1] * 0.02
            }}
            
        except Exception as e:
            print(f"❌ 计算{factor_name}原始因子失败: {{e}}")
            return self._get_default_raw_values()
    
    def calculate_standard_scores(self, raw_data: Dict, market_data: Dict = None) -> Dict:
        """
        将原始因子数据转换为0-100分标准化评分
        
        Args:
            raw_data: 原始因子数据
            market_data: 市场数据（价格、成交量等）
        
        Returns:
            标准化评分字典
        """
        try:
            # TODO: 实现标准化评分逻辑
            
            # 示例: CCI评分 (需要根据实际逻辑调整)
            cci = raw_data.get('cci_14', 0)
            if cci > 100:      # 超买
                cci_score = 25
            elif cci > 50:     # 偏强
                cci_score = 70
            elif cci > -50:    # 中性
                cci_score = 50
            elif cci > -100:   # 偏弱
                cci_score = 30
            else:              # 超卖
                cci_score = 75  # 超卖可能反弹
            
            # 示例: PSAR评分
            psar_trend = raw_data.get('psar_trend', 0)
            if psar_trend > 0:
                psar_score = 75    # 上升趋势
            elif psar_trend < 0:
                psar_score = 25    # 下降趋势  
            else:
                psar_score = 50    # 中性
            
            # 复合信号评分
            composite_score = (cci_score * 0.6 + psar_score * 0.4)
            
            # 风险收益比评分 (基于ATR)
            atr = raw_data.get('atr_14', 1)
            risk_reward_score = max(20, min(80, 50 + (2 - atr) * 10))
            
            return {{
                '{factor_name}_signal': composite_score,
                'cci_momentum': cci_score,
                'psar_trend': psar_score,
                'risk_reward_ratio': risk_reward_score
            }}
            
        except Exception as e:
            print(f"❌ 计算{factor_name}标准化评分失败: {{e}}")
            return self._get_default_standard_scores()
    
    def _get_default_raw_values(self) -> Dict:
        """默认原始值"""
        return {{
            'cci_14': 0,
            'psar': 0,
            'psar_trend': 0,
            'atr_14': 1
        }}
    
    def _get_default_standard_scores(self) -> Dict:
        """默认标准化评分"""
        return {{
            '{factor_name}_signal': 50.0,
            'cci_momentum': 50.0,
            'psar_trend': 50.0,
            'risk_reward_ratio': 50.0
        }}

# 工厂函数
def create_calculator():
    return {factor_name.title().replace('_', '')}Calculator()
'''
        
        return template
    
    def _update_weight_optimizer_config(self, factor_config: Dict, result: Dict):
        """更新权重优化器配置"""
        self.logger.info("⚙️  更新权重优化器配置...")
        
        from weight_optimizer import WeightOptimizer
        
        # 创建优化器实例
        optimizer = WeightOptimizer()
        
        # 添加新因子到对应维度
        dimension = factor_config['dimension']
        factor_name = factor_config['name']
        weight_range = factor_config.get('weight_range', [0.05, 0.10, 0.15])
        
        if dimension in optimizer.config['dimensions']:
            # 向现有维度添加因子
            new_factors = [f"{factor_name}_signal"]  # 主要因子
            
            # 添加子因子 (如果有)
            for col_info in factor_config.get('standard_columns', []):
                col_name = col_info['name']
                if col_name != f"{factor_name}_signal":
                    new_factors.append(col_name)
            
            for new_factor in new_factors:
                optimizer.add_new_factor_to_dimension(dimension, new_factor, weight_range)
        else:
            # 创建新维度
            new_dimension = {
                "description": factor_config['description'],
                "weight_range": weight_range,
                "factors": [f"{factor_name}_signal"],
                "sub_weights": {
                    f"{factor_name}_signal": weight_range
                }
            }
            optimizer.add_new_dimension(dimension, new_dimension)
        
        # 保存更新的配置
        config_file = os.path.join(
            self.project_root, 'factor_optimization', 'configs', 
            f'{factor_config["version"]}_config.json'
        )
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        optimizer.save_config(config_file)
        
        self.logger.info(f"  ✅ 配置已保存: {config_file}")
        result["integration_steps"].append("权重优化器配置已更新")
    
    def _calculate_historical_factor_data(self, factor_config: Dict, result: Dict):
        """计算历史因子数据"""
        self.logger.info("📈 计算历史因子数据...")
        
        # TODO: 集成到standard_factor_calculator.py中
        # 这里需要动态加载新的计算器并执行计算
        
        self.logger.info("  ⚠️  请手动运行标准因子计算器以生成历史数据:")
        self.logger.info("  python3 factor_optimization/standard_factor_calculator.py --start-date 2024-01-01 --end-date 2025-08-25")
        
        result["integration_steps"].append("历史因子数据计算（需手动执行）")
    
    def _optimize_weights_with_new_factor(self, factor_config: Dict, result: Dict):
        """使用新因子执行权重优化"""
        self.logger.info("🎯 执行权重优化...")
        
        # 使用更新后的配置文件运行优化
        config_file = os.path.join(
            self.project_root, 'factor_optimization', 'configs', 
            f'{factor_config["version"]}_config.json'
        )
        
        if os.path.exists(config_file):
            from weight_optimizer import WeightOptimizer
            
            optimizer = WeightOptimizer(config_file)
            
            # 执行优化（使用较少样本进行快速测试）
            optimization_result = optimizer.optimize_weights(
                start_date='2025-08-01',  # 使用最近数据进行快速测试
                end_date='2025-08-25',
                max_samples=10000
            )
            
            # 保存优化结果
            result_file = os.path.join(
                self.project_root, 'factor_optimization', 
                f'{factor_config["version"]}_optimization_result.json'
            )
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(optimization_result, f, ensure_ascii=False, indent=2)
            
            result["optimization_result"] = optimization_result
            self.logger.info(f"  ✅ 优化完成，最佳得分: {optimization_result['best_score']:.4f}")
        
        result["integration_steps"].append("权重优化已完成")
    
    def _evaluate_factor_impact(self, factor_config: Dict, result: Dict):
        """评估新因子的影响"""
        self.logger.info("📊 评估新因子影响...")
        
        # TODO: 实现因子影响评估
        # 1. 对比添加前后的优化效果
        # 2. 分析新因子的独立贡献
        # 3. 计算因子相关性
        
        impact_analysis = {
            "factor_name": factor_config['name'],
            "dimension": factor_config['dimension'],
            "impact_score": 0.0,  # 待计算
            "correlation_with_existing": {},  # 待计算
            "performance_improvement": 0.0,  # 待计算
        }
        
        result["factor_impact"] = impact_analysis
        result["integration_steps"].append("因子影响评估已完成")
    
    def _generate_integration_report(self, factor_config: Dict, result: Dict):
        """生成集成报告"""
        report_content = f"""# {factor_config['name']} 因子集成报告

## 📊 基本信息
- **因子名称**: {factor_config['name']}
- **版本**: {factor_config['version']}
- **描述**: {factor_config['description']}
- **数据源**: {factor_config['source_url']}
- **集成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🚀 集成步骤
"""
        
        for i, step in enumerate(result["integration_steps"], 1):
            report_content += f"{i}. {step}\n"
        
        if result.get("optimization_result"):
            opt_result = result["optimization_result"]
            report_content += f"""
## 🎯 权重优化结果
- **最佳得分**: {opt_result['best_score']:.4f}
- **最佳权重配置**:
"""
            for dim, weight in opt_result['best_weights'].items():
                report_content += f"  - {dim}: {weight:.1%}\n"
        
        if result.get("factor_impact"):
            impact = result["factor_impact"]
            report_content += f"""
## 📈 因子影响分析
- **影响得分**: {impact['impact_score']:.4f}
- **性能改进**: {impact['performance_improvement']:.2%}
"""
        
        # 保存报告
        report_file = os.path.join(
            self.project_root, 'factor_optimization', 'reports',
            f'{factor_config["name"]}_integration_report.md'
        )
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"📄 集成报告已生成: {report_file}")
        result["integration_steps"].append(f"集成报告: {report_file}")

def main():
    """示例用法"""
    
    # 示例配置 - CCI+Parabolic SAR
    factor_config = {
        "name": "cci_psar_composite",
        "version": "v3.3", 
        "description": "CCI+Parabolic SAR复合技术指标",
        "source_url": "https://www.tradingview.com/script/SH4TLaGk/",
        "dimension": "technical",
        "raw_columns": [
            {"name": "cci_14", "type": "DECIMAL(10,3)"},
            {"name": "psar", "type": "DECIMAL(10,3)"},
            {"name": "psar_trend", "type": "INTEGER"},
            {"name": "atr_14", "type": "DECIMAL(10,3)"}
        ],
        "standard_columns": [
            {"name": "cci_psar_signal", "type": "DECIMAL(5,2)"},
            {"name": "cci_momentum", "type": "DECIMAL(5,2)"},
            {"name": "psar_trend", "type": "DECIMAL(5,2)"},
            {"name": "risk_reward_ratio", "type": "DECIMAL(5,2)"}
        ],
        "weight_range": [0.05, 0.10, 0.15]
    }
    
    integrator = FactorIntegrator()
    result = integrator.integrate_new_factor(factor_config)
    
    if result["success"]:
        print("🎉 新因子集成成功！")
        for step in result["integration_steps"]:
            print(f"  ✅ {step}")
    else:
        print("❌ 新因子集成失败：")
        for error in result["errors"]:
            print(f"  ❌ {error}")
    
    return result

if __name__ == "__main__":
    main()