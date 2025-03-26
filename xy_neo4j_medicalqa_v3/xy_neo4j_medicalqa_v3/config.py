"""
集中管理系统配置
"""

import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# Neo4j数据库配置
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"), 
    "password": os.getenv("NEO4J_PASSWORD", "wswy0129"),
    "max_connection_lifetime": 3600,
    "max_connection_pool_size": 50,
    "connection_acquisition_timeout": 60
}

# API密钥配置
API_KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY", "sk-e38ac2aefd1345538e35919fc794aef5"),
    "zhipu": os.getenv("ZHIPU_API_KEY", "sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"),
    "siliconflow": os.getenv("SILICONFLOW_API_KEY", "sk-zhfwnoxxdhilllqjbqciqxlhcxzhopuyuaabosgcepqwagrz")
}

# LLM配置 - 重构为支持多个模型
LLM_CONFIG = {
    # 默认使用的LLM提供商
    "default_provider": "deepseek",
    
    # 各提供商的配置
    "providers": {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "default_model": "deepseek-chat",
            "max_tokens": 4096,
            "temperature": 0.7,
            "client_type": "openai"  # 使用OpenAI客户端
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "default_model": "glm-4",
            "max_tokens": 4096,
            "temperature": 0.7,
            "client_type": "openai"  # 使用OpenAI客户端
        },
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1",
            "default_model": "Pro/deepseek-ai/DeepSeek-V3",
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "client_type": "requests"  # 使用requests库
        }
    },
    
    # 兼容旧代码的配置 - 指向默认提供商的配置
    "base_url": "https://api.deepseek.com",
    "default_model": "deepseek-chat",
    "max_tokens": 4096,
    "temperature": 0.7
}

# 更新兼容层配置函数
def update_compat_layer():
    """更新兼容层配置，使旧代码能够正常工作"""
    provider = LLM_CONFIG["default_provider"]
    provider_config = LLM_CONFIG["providers"][provider]
    
    # 更新顶层配置
    LLM_CONFIG["base_url"] = provider_config["base_url"]
    LLM_CONFIG["default_model"] = provider_config["default_model"]
    LLM_CONFIG["max_tokens"] = provider_config["max_tokens"]
    LLM_CONFIG["temperature"] = provider_config["temperature"]

# 初始化时更新兼容层
update_compat_layer()

# 缓存配置
CACHE_CONFIG = {
    "ttl": 3600,  # 缓存有效期（秒）
    "max_size": 1000  # 最大缓存条目数
}

# 实体提取配置
ENTITY_CONFIG = {
    "filter_words": [
        "SWAT模型", "SWAT", "模型", "水文模型", "水文", 
        "模拟", "系统", "方法", "研究", "分析", 
        "计算", "结果", "数据", "过程", "方案", "问题"
    ],
    "model_path": "BAAI/bge-small-zh-v1.5"
}

# 文件末尾添加测试代码
if __name__ == "__main__":
    # 测试配置是否正确加载
    print("API密钥:", API_KEYS)
    print("Neo4j配置:", NEO4J_CONFIG)
    
    # 测试API连接
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=API_KEYS["deepseek"],
            base_url=LLM_CONFIG["base_url"]
        )
        response = client.chat.completions.create(
            model=LLM_CONFIG["default_model"],
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("API连接测试成功:", response)
    except Exception as e:
        print("API连接测试失败:", e) 