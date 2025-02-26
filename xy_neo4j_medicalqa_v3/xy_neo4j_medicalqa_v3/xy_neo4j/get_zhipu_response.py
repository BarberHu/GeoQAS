from openai import OpenAI

class GetDeepseekResponse:  # 可以保留这个类名，避免大量修改
    def __init__(self):
        # 修改为ChatGPT API密钥
        self.api_key = 'sk-w9ADthmrlb2lf6EdP6kkMtxXkmXTOHnzVXhCltfxMTssYoIs'
        
        # 配置客户端，修改为ChatAnywhere的中转地址
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.chatanywhere.tech/v1",  # 修改为ChatAnywhere的API地址
            timeout=30,
            max_retries=3
        )

    def get_deepseek_response(self, prompt):  # 保留方法名，避免大量修改
        try:
            print("正在调用ChatGPT接口...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # 修改为ChatGPT的模型名称
                messages=[
                    {
                        "role": "system", 
                        "content": """
                        你是一位耐心细致的地理建模导师，专门指导初学者完成从数据准备到模型验证的完整建模流程。
                        请按照以下要求进行回答：
                        1. 结构化输出
                        2. 针对初学者的设计
                        3. 完整方案要求
                        """
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,  # 增加token限制，ChatGPT支持更长回复
                stream=False
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"ChatGPT接口调用失败: {str(e)}")
            print(f"错误类型: {type(e)}")
            print(f"API密钥: {self.api_key[:11]}...{self.api_key[-4:]}")
            return "抱歉，AI接口暂时无法访问，请稍后再试。"

