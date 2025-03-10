from openai import OpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import API_KEYS, LLM_CONFIG
from llm_client_factory import LLMClientFactory

class GetDeepseekResponse:
    def __init__(self, provider=None):
        """
        初始化DeepSeek响应获取器
        
        Args:
            provider: LLM提供商名称，如果为None则使用默认提供商
        """
        # 如果未指定提供商，使用默认提供商
        self.provider = provider or LLM_CONFIG["default_provider"]
        
        # 使用LLM客户端工厂创建客户端
        self.client = LLMClientFactory.create_client(self.provider)
        
        # 保存API密钥用于错误处理
        self.api_key = API_KEYS.get(self.provider)

    def get_deepseek_response(self, prompt):
        try:
            print(f"正在调用{self.provider.capitalize()} API...")
            
            # 构建消息
            messages = [
                {
                    "role": "system", 
                    "content": """
                    你是一位耐心细致的地理建模导师，专门指导初学者完成从数据准备到模型验证的完整建模流程。
                    请按照以下要求进行回答：
                    1. 结构化输出
                    2. 针对初学者的设计
                    3. 完整方案要求
                    4. 回答时,不要使用#或*等特殊字符    
                    5. 对于建模步骤,请生成详细合理的流程图图片
                    """
                },
                {"role": "user", "content": prompt}
            ]
            
            # 使用LLM客户端工厂创建的客户端进行调用
            response = self.client.chat_completion(messages=messages)
            
            return response
            
        except Exception as e:
            print(f"{self.provider.capitalize()} API调用失败: {str(e)}")
            print(f"错误类型: {type(e)}")
            # 不要打印完整API密钥，只打印部分用于调试
            if self.api_key:
                masked_key = f"{self.api_key[:5]}...{self.api_key[-4:]}"
                print(f"API密钥: {masked_key}")
            return f"抱歉，{self.provider.capitalize()} API暂时无法访问，请稍后再试。错误: {str(e)}"
    
    def switch_provider(self, provider):
        """
        切换LLM提供商
        
        Args:
            provider: 新的LLM提供商名称
            
        Returns:
            切换结果信息
        """
        if provider not in LLM_CONFIG["providers"]:
            return {"success": False, "message": f"未知的LLM提供商: {provider}"}
        
        try:
            # 更新当前提供商
            self.provider = provider
            
            # 更新LLM客户端
            self.client = LLMClientFactory.create_client(provider)
            
            # 更新API密钥
            self.api_key = API_KEYS.get(provider)
            
            return {
                "success": True, 
                "message": f"已切换到 {provider} API",
                "provider": provider
            }
        except Exception as e:
            return {"success": False, "message": f"切换API失败: {str(e)}"} 