#!/usr/bin/env python3
"""
一键添加新因子 - 从TradingView URL到完整集成

使用方法:
python3 factor_optimization/add_new_factor.py --url https://www.tradingview.com/script/SH4TLaGk/ --version v3.3

或者交互式使用:
python3 factor_optimization/add_new_factor.py --interactive
"""

import argparse
import sys
import os
from factor_integrator import FactorIntegrator

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(project_root)

def parse_tradingview_indicator(url: str) -> dict:
    """
    从TradingView URL解析指标信息
    
    Args:
        url: TradingView指标URL
    
    Returns:
        解析出的因子配置
    """
    
    try:
        from tools.web_fetch import WebFetch
        
        # 获取指标信息
        web_fetcher = WebFetch()
        indicator_info = web_fetcher.fetch(url, 
            "分析这个TradingView指标，提取：1)指标名称 2)主要计算逻辑 3)关键参数 4)适合的技术分析维度 5)建议的数据库列结构")
        
        print("📊 指标信息解析结果:")
        print(indicator_info)
        
    except ImportError:
        print("⚠️  无法自动解析TradingView指标，请手动输入配置信息")
        indicator_info = None
    
    # 获取用户输入
    print("\n🔧 请确认或输入因子配置信息:")
    
    factor_name = input("因子名称 (如: cci_psar_composite): ").strip()
    if not factor_name:
        factor_name = "custom_indicator"
    
    version = input("目标版本 (如: v3.3): ").strip()
    if not version:
        version = "v3.3"
    
    description = input("因子描述: ").strip()
    if not description:
        description = f"{factor_name} 技术指标"
    
    dimension = input("所属维度 (technical/fundamental/performance/sentiment/risk_control/market_regime): ").strip()
    if not dimension:
        dimension = "technical"
    
    # 构建配置
    factor_config = {
        "name": factor_name,
        "version": version,
        "description": description,
        "source_url": url,
        "dimension": dimension,
        "raw_columns": [
            {"name": f"{factor_name}_raw", "type": "DECIMAL(10,3)"},
            {"name": f"{factor_name}_signal", "type": "INTEGER"},
        ],
        "standard_columns": [
            {"name": f"{factor_name}_score", "type": "DECIMAL(5,2)"},
        ],
        "weight_range": [0.05, 0.10, 0.15]
    }
    
    # 针对CCI+PSAR的特殊处理 (基于URL识别)
    if "SH4TLaGk" in url:
        factor_config.update({
            "name": "cci_psar_composite",
            "description": "CCI+Parabolic SAR复合技术指标",
            "raw_columns": [
                {"name": "cci_14", "type": "DECIMAL(10,3)"},
                {"name": "psar", "type": "DECIMAL(10,3)"},
                {"name": "psar_trend", "type": "INTEGER"},
                {"name": "atr_14", "type": "DECIMAL(10,3)"}
            ],
            "standard_columns": [
                {"name": "cci_psar_signal", "type": "DECIMAL(5,2)"},
                {"name": "cci_momentum", "type": "DECIMAL(5,2)"},
                {"name": "psar_trend_score", "type": "DECIMAL(5,2)"},
                {"name": "risk_reward_ratio", "type": "DECIMAL(5,2)"}
            ]
        })
    
    return factor_config

def interactive_mode():
    """交互式模式"""
    print("🎯 交互式新因子添加模式")
    print("=" * 50)
    
    url = input("请输入TradingView指标URL: ").strip()
    if not url:
        print("❌ 必须提供TradingView URL")
        return False
    
    factor_config = parse_tradingview_indicator(url)
    
    print("\n📋 因子配置预览:")
    print(f"  名称: {factor_config['name']}")
    print(f"  版本: {factor_config['version']}")
    print(f"  描述: {factor_config['description']}")
    print(f"  维度: {factor_config['dimension']}")
    print(f"  原始列数: {len(factor_config['raw_columns'])}")
    print(f"  评分列数: {len(factor_config['standard_columns'])}")
    
    confirm = input("\n是否继续执行集成流程？(y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ 用户取消操作")
        return False
    
    # 执行集成
    return execute_integration(factor_config)

def execute_integration(factor_config: dict) -> bool:
    """执行因子集成流程"""
    
    integrator = FactorIntegrator()
    
    print(f"\n🚀 开始集成新因子: {factor_config['name']}")
    print("=" * 50)
    
    result = integrator.integrate_new_factor(factor_config)
    
    print("\n📊 集成结果:")
    if result["success"]:
        print("🎉 新因子集成成功！")
        print("\n✅ 完成的步骤:")
        for step in result["integration_steps"]:
            print(f"  • {step}")
        
        print(f"\n📁 相关文件:")
        print(f"  • 计算器: factor_optimization/calculators/{factor_config['name']}_calculator.py")
        print(f"  • 配置: factor_optimization/configs/{factor_config['version']}_config.json") 
        print(f"  • 报告: factor_optimization/reports/{factor_config['name']}_integration_report.md")
        
        print(f"\n🔄 后续步骤:")
        print(f"  1. 完善计算器代码: factor_optimization/calculators/{factor_config['name']}_calculator.py")
        print(f"  2. 运行历史数据计算: python3 factor_optimization/standard_factor_calculator.py --start-date 2024-01-01")
        print(f"  3. 测试新版本: python3 tomorrow_stock_selector.py --scoring-version {factor_config['version']}")
        
        return True
    else:
        print("❌ 新因子集成失败")
        print("\n错误信息:")
        for error in result["errors"]:
            print(f"  ❌ {error}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='一键添加新因子到权重优化系统')
    parser.add_argument('--url', help='TradingView指标URL')
    parser.add_argument('--version', default='v3.3', help='目标版本 (默认: v3.3)')
    parser.add_argument('--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--name', help='因子名称')
    parser.add_argument('--dimension', default='technical', help='所属维度')
    
    args = parser.parse_args()
    
    if args.interactive:
        success = interactive_mode()
    elif args.url:
        # 命令行模式
        factor_config = {
            "name": args.name or "custom_indicator",
            "version": args.version,
            "description": f"{args.name or 'Custom'} 技术指标",
            "source_url": args.url,
            "dimension": args.dimension,
            "raw_columns": [
                {"name": f"{args.name or 'custom'}_raw", "type": "DECIMAL(10,3)"},
            ],
            "standard_columns": [
                {"name": f"{args.name or 'custom'}_score", "type": "DECIMAL(5,2)"},
            ],
            "weight_range": [0.05, 0.10, 0.15]
        }
        
        # 特殊处理已知指标
        if "SH4TLaGk" in args.url:
            factor_config.update({
                "name": "cci_psar_composite",
                "description": "CCI+Parabolic SAR复合技术指标",
                "raw_columns": [
                    {"name": "cci_14", "type": "DECIMAL(10,3)"},
                    {"name": "psar", "type": "DECIMAL(10,3)"},
                    {"name": "psar_trend", "type": "INTEGER"},
                    {"name": "atr_14", "type": "DECIMAL(10,3)"}
                ],
                "standard_columns": [
                    {"name": "cci_psar_signal", "type": "DECIMAL(5,2)"},
                    {"name": "cci_momentum", "type": "DECIMAL(5,2)"},
                    {"name": "psar_trend_score", "type": "DECIMAL(5,2)"},
                    {"name": "risk_reward_ratio", "type": "DECIMAL(5,2)"}
                ]
            })
        
        success = execute_integration(factor_config)
    else:
        parser.print_help()
        success = False
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())