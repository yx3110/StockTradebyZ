#!/usr/bin/env python3
"""
Claude配置使用示例
演示如何在TradingAgents集成系统中使用Claude API
"""

import os
import sys
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from adapters.claude_config import ClaudeConfig, create_claude_trading_config
from adapters.china_stock_adapter import ChinaStockAdapter
from adapters.china_trading_agents import ChinaTradingAgents, ChinaStockAnalyzer

def example_1_basic_claude_config():
    """示例1: 基础Claude配置使用"""
    print("=" * 60)
    print("示例1: 基础Claude配置")
    print("=" * 60)
    
    # 验证API密钥
    if not ClaudeConfig.validate_api_key():
        print("请先设置Claude API密钥:")
        print("export ANTHROPIC_API_KEY='your_anthropic_api_key'")
        return
    
    # 获取推荐配置
    config = ClaudeConfig.get_recommended_config("high_quality")
    print(f"推荐配置 (高质量):")
    print(f"  深度思考模型: {config['deep_think_llm']}")
    print(f"  快速响应模型: {config['quick_think_llm']}")
    print(f"  最大Token数: {config['max_tokens']}")
    print(f"  温度参数: {config['temperature']}")
    print(f"  描述: {config['description']}")
    
    # 估算成本
    input_tokens = 2000
    output_tokens = 1000
    cost = ClaudeConfig.estimate_cost(config['deep_think_llm'], input_tokens, output_tokens)
    print(f"\n成本估算 ({input_tokens}输入 + {output_tokens}输出 tokens): ${cost:.4f}")

def example_2_trading_config():
    """示例2: 创建TradingAgents配置"""
    print("\n" + "=" * 60)
    print("示例2: TradingAgents完整配置")
    print("=" * 60)
    
    # 创建不同类型的配置
    configs = {
        "高质量分析": create_claude_trading_config("high_quality"),
        "平衡配置": create_claude_trading_config("balanced"),
        "快速处理": create_claude_trading_config("fast"),
        "顶级质量": create_claude_trading_config("premium")
    }
    
    for name, config in configs.items():
        print(f"\n{name} 配置:")
        print(f"  LLM提供商: {config['llm_provider']}")
        print(f"  深度模型: {config['deep_think_llm']}")
        print(f"  快速模型: {config['quick_think_llm']}")
        print(f"  最大Token: {config['max_tokens']}")
        print(f"  温度: {config['temperature']}")
        print(f"  辩论轮数: {config['max_debate_rounds']}")

def example_3_china_market_prompts():
    """示例3: 中国市场专用提示词"""
    print("\n" + "=" * 60)
    print("示例3: 中国市场专用提示词")
    print("=" * 60)
    
    prompts = ClaudeConfig.get_china_market_prompts()
    
    for role, prompt in prompts.items():
        print(f"\n{role.upper()}:")
        print("-" * 40)
        # 只显示前200个字符作为示例
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)

def example_4_cost_comparison():
    """示例4: 不同模型成本对比"""
    print("\n" + "=" * 60)
    print("示例4: Claude模型成本对比")
    print("=" * 60)
    
    # 假设一次完整分析的token使用量
    input_tokens = 3000  # 技术指标数据 + 新闻 + 基本面数据
    output_tokens = 1500  # 分析报告 + 决策建议
    
    models = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022", 
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307"
    ]
    
    print(f"单股分析成本对比 ({input_tokens}输入 + {output_tokens}输出 tokens):")
    print("-" * 70)
    
    for model in models:
        cost = ClaudeConfig.estimate_cost(model, input_tokens, output_tokens)
        model_info = ClaudeConfig.CLAUDE_MODELS[model]
        print(f"{model:<30} ${cost:.4f} - {', '.join(model_info['recommended_for'])}")
    
    print(f"\n10只股票批量分析成本:")
    print("-" * 40)
    for model in models:
        total_cost = ClaudeConfig.estimate_cost(model, input_tokens * 10, output_tokens * 10)
        print(f"{model:<30} ${total_cost:.2f}")

def example_5_stock_analysis():
    """示例5: 实际股票分析（模拟）"""
    print("\n" + "=" * 60)
    print("示例5: 股票分析演示（需要API密钥）")
    print("=" * 60)
    
    if not ClaudeConfig.validate_api_key():
        print("❌ 需要设置ANTHROPIC_API_KEY才能运行此示例")
        return
    
    try:
        # 使用高质量配置
        config = create_claude_trading_config("high_quality")
        
        # 初始化分析器
        analyzer = ChinaStockAnalyzer(config=config)
        
        print("✅ ChinaStockAnalyzer初始化成功")
        print(f"   配置: {config['deep_think_llm']} (深度) + {config['quick_think_llm']} (快速)")
        print("   可以开始股票分析...")
        
        # 注意: 实际分析需要有效的股票数据
        print("\n📊 要运行完整分析，请使用:")
        print("   python TA_integration/main.py --config claude_high_quality --date 2025-07-31")
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")

def main():
    """运行所有示例"""
    print("🤖 Claude配置和使用示例")
    print("Claude API for TradingAgents Integration")
    
    # 显示模型推荐
    print(ClaudeConfig.get_model_recommendations())
    
    # 运行示例
    example_1_basic_claude_config()
    example_2_trading_config() 
    example_3_china_market_prompts()
    example_4_cost_comparison()
    example_5_stock_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 所有示例运行完成!")
    print("=" * 60)
    print("\n📚 下一步:")
    print("1. 设置API密钥: export ANTHROPIC_API_KEY='your_key'")
    print("2. 运行分析: python TA_integration/main.py --config claude_high_quality")
    print("3. 查看结果: ls TA_integration/output/")

if __name__ == "__main__":
    main()