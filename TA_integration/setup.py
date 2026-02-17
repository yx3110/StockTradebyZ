#!/usr/bin/env python3
"""
TA_integration 系统设置脚本
"""

import os
import sys
import json
from pathlib import Path

def check_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本需要3.8或更高")
        return False
    
    # 检查必要的包
    required_packages = ['pandas', 'numpy', 'openai', 'langchain', 'stockstats']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少必要的包: {', '.join(missing_packages)}")
        print("请运行: pip install pandas numpy openai langchain langgraph stockstats")
        return False
    
    print("✅ 系统要求检查通过")
    return True

def check_tradingagents():
    """检查TradingAgents是否存在"""
    print("🔍 检查TradingAgents...")
    
    ta_path = Path("TradingAgents")
    if not ta_path.exists():
        print("❌ TradingAgents目录不存在")
        print("请确保TradingAgents项目在当前目录下")
        return False
    
    # 检查关键文件
    key_files = [
        "TradingAgents/tradingagents/graph/trading_graph.py",
        "TradingAgents/tradingagents/default_config.py"
    ]
    
    for file_path in key_files:
        if not Path(file_path).exists():
            print(f"❌ 缺少关键文件: {file_path}")
            return False
    
    print("✅ TradingAgents检查通过")
    return True

def check_data_directories():
    """检查数据目录"""
    print("🔍 检查数据目录...")
    
    data_dirs = ["full_securities_data", "daily_result"]
    
    for dir_name in data_dirs:
        if not Path(dir_name).exists():
            print(f"⚠️  数据目录不存在: {dir_name}")
            print(f"   请确保已运行过量化选股系统")
    
    print("✅ 数据目录检查完成")
    return True

def setup_directories():
    """创建必要的目录"""
    print("📁 创建目录结构...")
    
    directories = [
        "TA_integration/data/market_data/price_data",
        "TA_integration/data/cache",
        "TA_integration/logs",
        "TA_integration/output"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ 目录结构创建完成")

def check_api_keys():
    """检查API密钥"""
    print("🔑 检查API密钥...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️  OPENAI_API_KEY环境变量未设置")
        print("   请设置: export OPENAI_API_KEY='your_api_key'")
        return False
    
    if openai_key.startswith("sk-"):
        print("✅ OpenAI API密钥已设置")
    else:
        print("⚠️  OpenAI API密钥格式可能不正确")
    
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if finnhub_key:
        print("✅ Finnhub API密钥已设置")
    else:
        print("ℹ️  Finnhub API密钥未设置（可选）")
    
    return True

def create_example_config():
    """创建示例配置文件"""
    print("⚙️  创建配置文件...")
    
    config_path = Path("TA_integration/config/config.json")
    if config_path.exists():
        print("ℹ️  配置文件已存在，跳过")
        return
    
    example_config = {
        "system": {
            "name": "TradingAgents Integration",
            "version": "1.0.0"
        },
        "tradingagents": {
            "llm_provider": "openai",
            "deep_think_llm": "gpt-4",
            "quick_think_llm": "gpt-4",
            "online_tools": False,
            "max_debate_rounds": 1
        },
        "analysis": {
            "default_top_n": 10,
            "min_confidence_threshold": 0.6
        }
    }
    
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, ensure_ascii=False, indent=2)
    
    print("✅ 配置文件创建完成")

def run_test():
    """运行测试"""
    print("🧪 运行基本测试...")
    
    try:
        # 测试导入
        sys.path.append('TA_integration')
        from core.report_parser import ReportParser
        from adapters.china_stock_adapter import ChinaStockAdapter
        
        # 测试基本功能
        parser = ReportParser()
        adapter = ChinaStockAdapter()
        
        print("✅ 基本测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主设置函数"""
    print("🚀 TradingAgents集成系统设置")
    print("=" * 50)
    
    checks = [
        check_requirements,
        check_tradingagents,
        check_data_directories,
        check_api_keys
    ]
    
    setup_tasks = [
        setup_directories,
        create_example_config,
        run_test
    ]
    
    # 运行检查
    for check in checks:
        if not check():
            print("\n❌ 设置检查失败，请解决上述问题后重试")
            return False
        print()
    
    # 运行设置
    for task in setup_tasks:
        task()
        print()
    
    print("🎉 设置完成！")
    print("\n📚 使用指南:")
    print("1. 运行量化选股: ./run_daily_update.sh")
    print("2. AI增强分析: python TA_integration/main.py")
    print("3. 查看详细文档: cat TA_integration/README.md")

if __name__ == "__main__":
    main()