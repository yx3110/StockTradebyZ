#!/usr/bin/env python3
"""
量化评分系统管理器
支持v2、v3版本切换和配置管理
"""

import json
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from real_data_optimized_scorer import OptimizedQuantitativeScorer as ScorerV2
except ImportError:
    print("警告: 无法导入v2版本评分器")
    ScorerV2 = None

try:
    from quantitative_scorer_v3 import QuantitativeScorerV3 as ScorerV3
except ImportError:
    print("警告: 无法导入v3版本评分器")
    ScorerV3 = None

class ScoringSystemManager:
    """量化评分系统管理器"""
    
    def __init__(self):
        self.current_version = "v3"  # 默认使用v3
        self.config_dir = "scoring_improvements"
        self.logger = self._setup_logging()
        
        # 版本配置映射
        self.version_configs = {
            "v2": {
                "class": ScorerV2,
                "config_file": "scoring_improvements/optimized_scorer_config.json",
                "description": "v2版本 - 实际数据优化评分器"
            },
            "v3": {
                "class": ScorerV3,
                "config_file": "scoring_improvements/optimized_config_v3_latest.json",
                "description": "v3版本 - 权重优化智能评分器"
            }
        }
        
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("ScoringSystemManager")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def list_available_versions(self) -> Dict[str, str]:
        """列出可用版本"""
        available = {}
        for version, config in self.version_configs.items():
            if config["class"] is not None:
                available[version] = config["description"]
        return available
        
    def switch_version(self, version: str) -> bool:
        """切换版本"""
        if version not in self.version_configs:
            self.logger.error(f"不支持的版本: {version}")
            return False
            
        if self.version_configs[version]["class"] is None:
            self.logger.error(f"版本 {version} 不可用（导入失败）")
            return False
            
        self.current_version = version
        self.logger.info(f"已切换到版本: {version}")
        return True
        
    def get_current_version(self) -> str:
        """获取当前版本"""
        return self.current_version
        
    def create_scorer(self, version: Optional[str] = None, 
                     config_path: Optional[str] = None) -> Any:
        """创建评分器实例"""
        if version is None:
            version = self.current_version
            
        if version not in self.version_configs:
            raise ValueError(f"不支持的版本: {version}")
            
        scorer_class = self.version_configs[version]["class"]
        if scorer_class is None:
            raise ImportError(f"版本 {version} 不可用（导入失败）")
            
        # 使用指定配置文件或默认配置文件
        if config_path is None:
            config_path = self.version_configs[version]["config_file"]
            
        # 检查配置文件是否存在
        if os.path.exists(config_path):
            self.logger.info(f"使用配置文件: {config_path}")
            return scorer_class(config_path=config_path)
        else:
            self.logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return scorer_class()
            
    def list_available_configs(self, version: Optional[str] = None) -> List[str]:
        """列出可用的配置文件"""
        if version is None:
            version = self.current_version
            
        config_files = []
        config_dir = self.config_dir
        
        if os.path.exists(config_dir):
            for filename in os.listdir(config_dir):
                if filename.endswith('.json'):
                    if version == "v2" and "v2" in filename.lower():
                        config_files.append(os.path.join(config_dir, filename))
                    elif version == "v3" and "v3" in filename.lower():
                        config_files.append(os.path.join(config_dir, filename))
                        
        return sorted(config_files)
        
    def compare_versions(self, stock_codes: List[str], date: str, 
                        sample_size: int = 10) -> Dict[str, Any]:
        """比较不同版本的评分结果"""
        results = {}
        
        for version in self.version_configs.keys():
            if self.version_configs[version]["class"] is None:
                continue
                
            try:
                scorer = self.create_scorer(version)
                version_results = []
                
                # 随机选择样本股票
                import random
                sample_codes = random.sample(stock_codes, min(sample_size, len(stock_codes)))
                
                for code in sample_codes:
                    result = scorer.calculate_stock_score(code, date) if hasattr(scorer, 'calculate_stock_score') else scorer.score_stock(code, date)
                    version_results.append({
                        "code": code,
                        "score": result.get('total_score', result.get('score', 0)),
                        "details": result
                    })
                    
                # 按得分排序
                version_results.sort(key=lambda x: x['score'], reverse=True)
                
                results[version] = {
                    "results": version_results,
                    "avg_score": sum(r['score'] for r in version_results) / len(version_results) if version_results else 0,
                    "top_stock": version_results[0]['code'] if version_results else None,
                    "top_score": version_results[0]['score'] if version_results else 0
                }
                
            except Exception as e:
                self.logger.error(f"版本 {version} 比较失败: {e}")
                results[version] = {"error": str(e)}
                
        return results
        
    def save_version_comparison(self, comparison_results: Dict[str, Any], 
                              stock_codes: List[str], date: str):
        """保存版本比较结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filepath = f"scoring_improvements/v2_v3_version_comparison_{timestamp}.json"
        
        output_data = {
            "comparison_info": {
                "date": date,
                "timestamp": timestamp,
                "stock_codes": stock_codes,
                "total_stocks": len(stock_codes)
            },
            "results": comparison_results,
            "summary": self._generate_comparison_summary(comparison_results)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"版本比较结果已保存到: {filepath}")
        return filepath
        
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成比较摘要"""
        summary = {}
        
        for version, result in results.items():
            if "error" not in result:
                summary[version] = {
                    "avg_score": result["avg_score"],
                    "top_stock": result["top_stock"],
                    "top_score": result["top_score"],
                    "total_valid_scores": len([r for r in result["results"] if r["score"] > 0])
                }
            else:
                summary[version] = {"status": "error", "error": result["error"]}
                
        return summary
        
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "current_version": self.current_version,
            "available_versions": self.list_available_versions(),
            "config_directory": self.config_dir,
            "timestamp": datetime.now().isoformat()
        }


def main():
    """主程序 - 演示版本切换功能"""
    manager = ScoringSystemManager()
    
    print("🏗️  量化评分系统管理器")
    print(f"当前版本: {manager.get_current_version()}")
    print("\n可用版本:")
    
    available = manager.list_available_versions()
    for version, desc in available.items():
        print(f"  {version}: {desc}")
        
    # 演示版本切换
    print("\n🔄 测试版本切换:")
    for version in available.keys():
        success = manager.switch_version(version)
        if success:
            print(f"  ✅ 切换到 {version} 成功")
            
            # 测试创建评分器
            try:
                scorer = manager.create_scorer()
                print(f"    📊 {version} 评分器创建成功")
            except Exception as e:
                print(f"    ❌ {version} 评分器创建失败: {e}")
        else:
            print(f"  ❌ 切换到 {version} 失败")
            
    # 显示系统信息
    print("\n📋 系统信息:")
    info = manager.get_system_info()
    for key, value in info.items():
        if key != "available_versions":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()