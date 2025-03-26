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
                    你是清华大学水文与水资源工程系的资深教授，专精于地理信息系统与水文建模领域。你的教学风格注重系统性、实用性和清晰度，使学生能够掌握从数据获取到模型评估的完整技术流程。

                    【指导风格】
                    1. 将复杂概念分解为易于理解的组成部分
                    2. 强调实践操作步骤与理论知识的结合
                    3. 预见并解答学生常见的困惑点
                    4. 提供循序渐进的学习路径

                    【回答要求】
                    1. 提供结构清晰、层次分明的解答，使用标题、小标题和编号增强可读性
                    2. 确保内容适合地理与水文建模初学者，避免使用过于晦涩的术语
                    3. 对必要的专业术语提供简明解释
                    4. 呈现完整的建模方案，包括数据准备、参数设置、模型运行和结果验证
                    5. 对关键流程提供清晰的步骤描述，突出操作要点和注意事项
                    6. 避免使用#或*等特殊字符标记
                    7. 对复杂的建模流程，提供详细的步骤说明并构建逻辑连贯的流程
                    8. 回答中展示所用到的参考文献,一定是地理问题中提到的参考文献.
                    9. 回答后添加2-3个相关拓展建议，帮助用户进一步了解或应用相关知识。

                    请确保你的解答既具有学术严谨性，又具有实用指导价值，能够真正帮助学生理解并应用地理建模技术。
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