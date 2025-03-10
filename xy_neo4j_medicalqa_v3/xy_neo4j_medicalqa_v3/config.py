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
    "zhipu": os.getenv("ZHIPU_API_KEY", "sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA")
}

# LLM配置
LLM_CONFIG = {
    "base_url": "https://api.deepseek.com",
    "default_model": "deepseek-chat",
    "max_tokens": 2048,
    "temperature": 0.7
}

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