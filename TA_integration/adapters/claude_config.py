#!/usr/bin/env python3
"""
Claude API配置和优化
专门为中国股票分析优化Claude模型设置
"""

import os
from typing import Dict, Any

class ClaudeConfig:
    """Claude配置管理器"""
    
    # Claude模型映射
    CLAUDE_MODELS = {
        # Claude 4 系列 (最新 - 2025年5月发布)
        "claude-sonnet-4-20250514": {
            "max_tokens": 64000,
            "context_window": 200000,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
            "recommended_for": ["顶级编程能力", "混合推理", "最新AI技术"]
        },
        
        # Claude 3.5 系列
        "claude-3-5-sonnet-20241022": {
            "max_tokens": 8192,
            "context_window": 200000,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
            "recommended_for": ["深度分析", "复杂推理", "多轮对话"]
        },
        "claude-3-5-haiku-20241022": {
            "max_tokens": 8192,
            "context_window": 200000,
            "cost_per_1k_input": 0.001,
            "cost_per_1k_output": 0.005,
            "recommended_for": ["快速响应", "简单分析", "批量处理"]
        },
        
        # Claude 3 系列
        "claude-3-opus-20240229": {
            "max_tokens": 4096,
            "context_window": 200000,
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.075,
            "recommended_for": ["最高质量分析", "复杂金融推理"]
        },
        "claude-3-sonnet-20240229": {
            "max_tokens": 4096,
            "context_window": 200000,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
            "recommended_for": ["平衡性能和成本", "常规分析"]
        },
        "claude-3-haiku-20240307": {
            "max_tokens": 4096,
            "context_window": 200000,
            "cost_per_1k_input": 0.00025,
            "cost_per_1k_output": 0.00125,
            "recommended_for": ["快速批量处理", "初步筛选"]
        }
    }
    
    @classmethod
    def get_recommended_config(cls, analysis_type: str = "standard") -> Dict[str, Any]:
        """获取推荐的Claude配置"""
        
        configs = {
            "claude_4": {
                "llm_provider": "anthropic",
                "deep_think_llm": "claude-sonnet-4-20250514",
                "quick_think_llm": "claude-sonnet-4-20250514",
                "backend_url": "https://api.anthropic.com",
                "max_tokens": 32000,
                "temperature": 0.1,
                "description": "Claude 4最新版本，顶级AI能力"
            },
            "high_quality": {
                "llm_provider": "anthropic",
                "deep_think_llm": "claude-3-5-sonnet-20241022",
                "quick_think_llm": "claude-3-5-haiku-20241022",
                "backend_url": "https://api.anthropic.com",
                "max_tokens": 8192,
                "temperature": 0.1,
                "description": "最高质量分析，适合重要决策"
            },
            "balanced": {
                "llm_provider": "anthropic", 
                "deep_think_llm": "claude-3-sonnet-20240229",
                "quick_think_llm": "claude-3-haiku-20240307",
                "backend_url": "https://api.anthropic.com",
                "max_tokens": 4096,
                "temperature": 0.2,
                "description": "平衡性能和成本，适合日常使用"
            },
            "fast": {
                "llm_provider": "anthropic",
                "deep_think_llm": "claude-3-5-haiku-20241022", 
                "quick_think_llm": "claude-3-5-haiku-20241022",
                "backend_url": "https://api.anthropic.com",
                "max_tokens": 4096,
                "temperature": 0.3,
                "description": "快速处理，适合批量分析"
            },
            "premium": {
                "llm_provider": "anthropic",
                "deep_think_llm": "claude-3-opus-20240229",
                "quick_think_llm": "claude-3-5-sonnet-20241022", 
                "backend_url": "https://api.anthropic.com",
                "max_tokens": 4096,
                "temperature": 0.1,
                "description": "顶级质量，适合关键投资决策"
            }
        }
        
        return configs.get(analysis_type, configs["balanced"])
    
    @classmethod
    def get_china_market_prompts(cls) -> Dict[str, str]:
        """获取中国市场专用提示词"""
        return {
            "market_analyst": """你是一位专业的中国A股市场技术分析师。请基于提供的技术指标数据进行深入分析：

核心分析要点：
1. 技术指标解读（MA均线、MACD、KDJ、RSI、布林带）
2. 趋势判断（上升/下降/盘整趋势）
3. 支撑位和阻力位识别
4. 买卖信号提示（结合成交量分析）
5. 风险提示（关注政策面、资金面影响）

中国市场特点考虑：
- A股T+1交易制度
- 涨跌停板限制（10%或5%）
- 政策敏感性较高
- 资金面波动影响
- 热点板块轮动特征

请用专业且易懂的中文撰写技术分析报告。""",

            "sentiment_analyst": """你是中国股市社交媒体情绪分析专家。请分析股票相关的市场情绪：

分析维度：
1. 投资者情绪倾向（乐观/悲观/中性）
2. 关注度和讨论热度
3. 主要观点和争议点
4. 机构vs散户观点差异
5. 情绪变化趋势

中国市场情绪特点：
- 散户参与度高，情绪波动大
- 政策消息敏感性强
- 热点题材炒作频繁
- 雪球、东方财富股吧等平台影响
- 短期情绪vs长期价值判断

请提供情绪分析和投资建议。""",

            "news_analyst": """你是中国财经新闻分析专家。请分析新闻对股票的潜在影响：

分析框架：
1. 新闻重要性评级（高/中/低）
2. 影响方向（正面/负面/中性）
3. 影响时间（短期/中期/长期）
4. 市场预期vs实际情况
5. 后续发展预测

关注要点：
- 政策导向和监管变化
- 行业发展趋势
- 公司基本面变化
- 市场资金流向
- 国际经济环境影响

请用中文提供专业的新闻影响分析。""",

            "fundamental_analyst": """你是中国上市公司基本面分析专家。请进行全面的基本面分析：

分析框架：
1. 公司财务健康状况
2. 盈利能力和增长性
3. 行业地位和竞争优势
4. 估值水平（PE、PB、PEG等）
5. 投资价值判断

中国市场特殊考虑：
- 国企vs民企特点
- 政策扶持行业识别
- 产业链上下游关系
- 区域经济发展影响
- 监管合规风险

请提供详细的基本面分析报告。""",

            "risk_manager": """你是专业的A股风险管理专家。请评估投资风险：

风险评估维度：
1. 市场风险（系统性风险）
2. 个股风险（特定风险）
3. 流动性风险
4. 政策风险
5. 估值风险

A股特有风险：
- 涨跌停板风险
- T+1交易限制
- 政策调控影响
- 资金面波动
- 退市制度风险

请提供风险等级评估和风控建议。"""
        }
    
    @classmethod
    def validate_api_key(cls) -> bool:
        """验证Claude API密钥"""
        # 首先尝试从环境变量读取
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # 如果环境变量没有，尝试从本地配置文件读取
        if not api_key:
            try:
                import json
                from pathlib import Path
                # 尝试多个可能的配置文件路径（优先使用项目根目录的配置）
                config_paths = [
                    Path("config.json"),  # 项目根目录配置（优先）
                    Path("../config.json"),  # 从TA_integration访问根目录
                    Path("../../config.json"),  # 从TA_integration/adapters访问根目录
                    Path("config/config.json"),
                    Path("TA_integration/config/config.json")
                ]
                
                for config_path in config_paths:
                    if config_path.exists():
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            # 优先从直接配置获取API密钥，再尝试环境变量
                            api_key = config.get("anthropic", {}).get("api_key")
                            if not api_key:
                                api_key_env = config.get("anthropic", {}).get("api_key_env")
                                if api_key_env:
                                    api_key = os.getenv(api_key_env)
                            if api_key:
                                break
                        if api_key:
                            # 设置环境变量供后续使用
                            os.environ["ANTHROPIC_API_KEY"] = api_key
                            print("✅ 从config.json读取Claude API密钥")
            except Exception as e:
                print(f"⚠️  读取配置文件失败: {e}")
        
        if not api_key:
            print("❌ Claude API密钥未找到")
            print("请在以下任一位置设置:")
            print("1. 环境变量: export ANTHROPIC_API_KEY='your_key'")
            print("2. 配置文件: config.json -> anthropic.api_key")
            return False
        
        if not api_key.startswith("sk-ant-"):
            print("⚠️  Anthropic API密钥格式可能不正确（应以sk-ant-开头）")
            return False
        
        print("✅ Claude API密钥已正确设置")
        return True
    
    @classmethod
    def estimate_cost(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """估算API调用成本"""
        if model not in cls.CLAUDE_MODELS:
            return 0.0
        
        model_info = cls.CLAUDE_MODELS[model]
        input_cost = (input_tokens / 1000) * model_info["cost_per_1k_input"]
        output_cost = (output_tokens / 1000) * model_info["cost_per_1k_output"]
        
        return input_cost + output_cost
    
    @classmethod
    def get_model_recommendations(cls) -> str:
        """获取模型推荐说明"""
        return """
Claude模型选择建议：

🆕 最新版本 (2025年5月发布)：
• Claude 4: claude-sonnet-4-20250514 - 最强AI能力，混合推理模型

🏆 推荐配置：
• Claude 4最新: claude-sonnet-4 (顶级能力，64K输出)
• 高质量分析: claude-3-5-sonnet (深度) + claude-3-5-haiku (快速)
• 平衡使用: claude-3-sonnet (深度) + claude-3-haiku (快速) 
• 快速批量: claude-3-5-haiku (深度) + claude-3-5-haiku (快速)
• 顶级质量: claude-3-opus (深度) + claude-3-5-sonnet (快速)

💰 成本考虑：
• Haiku: 最便宜，适合大量API调用
• Sonnet: 中等成本，最佳性价比
• Opus: 最贵但质量最高，适合重要决策

⚡ 性能特点：
• Claude 3.5系列: 最新版本，性能最优
• Claude 3系列: 稳定版本，广泛验证
"""

def create_claude_trading_config(analysis_type: str = "balanced") -> Dict[str, Any]:
    """创建Claude TradingAgents配置"""
    
    base_config = ClaudeConfig.get_recommended_config(analysis_type)
    prompts = ClaudeConfig.get_china_market_prompts()
    
    # 完整的TradingAgents配置
    config = {
        # LLM设置
        "llm_provider": base_config["llm_provider"],
        "deep_think_llm": base_config["deep_think_llm"],
        "quick_think_llm": base_config["quick_think_llm"], 
        "backend_url": base_config["backend_url"],
        
        # Claude特定设置
        "max_tokens": base_config["max_tokens"],
        "temperature": base_config["temperature"],
        
        # 辩论和讨论设置
        "max_debate_rounds": 2,
        "max_risk_discuss_rounds": 2,
        "max_recur_limit": 100,
        
        # 工具设置
        "online_tools": False,  # 使用离线数据
        
        # 中国市场设置
        "market": "china",
        "currency": "CNY",
        "trading_rules": {
            "price_limit": 0.10,
            "st_limit": 0.05, 
            "trading_unit": 100,
            "t_plus": 1
        },
        
        # 提示词设置
        "custom_prompts": prompts,
        
        # 数据目录
        "data_dir": "TA_integration/data",
        "data_cache_dir": "TA_integration/data/cache",
        
        # 项目目录
        "project_dir": "TA_integration",
        "results_dir": "TA_integration/output"
    }
    
    return config

if __name__ == "__main__":
    # 测试Claude配置
    print("🤖 Claude配置测试")
    print("=" * 50)
    
    # 验证API密钥
    ClaudeConfig.validate_api_key()
    
    # 显示模型推荐
    print(ClaudeConfig.get_model_recommendations())
    
    # 创建配置示例
    config = create_claude_trading_config("high_quality")
    print(f"\n配置示例（高质量模式）:")
    print(f"深度思考模型: {config['deep_think_llm']}")
    print(f"快速响应模型: {config['quick_think_llm']}")
    print(f"最大token数: {config['max_tokens']}")
    
    # 成本估算示例
    cost = ClaudeConfig.estimate_cost("claude-3-5-sonnet-20241022", 1000, 500)
    print(f"\n成本估算（1000输入+500输出token）: ${cost:.4f}")