from openai import OpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import API_KEYS, LLM_CONFIG

class GetDeepseekResponse:
    def __init__(self):
        # 使用配置文件中的API密钥和基础URL
        self.api_key = API_KEYS["deepseek"]
        self.base_url = LLM_CONFIG["base_url"]
        
        # 配置DeepSeek客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30,
            max_retries=3
        )

    def get_deepseek_response(self, prompt):
        try:
            print("正在调用DeepSeek API...")
            response = self.client.chat.completions.create(
                model=LLM_CONFIG["default_model"],  # 使用配置中的默认模型名称
                messages=[
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
                ],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
                stream=False
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"DeepSeek API调用失败: {str(e)}")
            print(f"错误类型: {type(e)}")
            # 不要打印完整API密钥，只打印部分用于调试
            masked_key = f"{self.api_key[:5]}...{self.api_key[-4:]}"
            print(f"API密钥: {masked_key}")
            return "抱歉，DeepSeek API暂时无法访问，请稍后再试。" 