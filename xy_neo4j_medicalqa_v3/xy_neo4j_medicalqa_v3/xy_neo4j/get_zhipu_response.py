from zhipuai import ZhipuAI  # 需要先安装SDK: pip install zhipuai
from . import settings

class GetZhipuResponse:
    def __init__(self):
        self.api_key = '713e7f9978f544d4ab5eb414e88b284a.hZvSzP0rVIrO80a4'

    def get_chatglm_response(self, prompt):
        try:
            print("正在调用智谱AI接口...")
            self.client = ZhipuAI(api_key=self.api_key)
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "system", "content": "你以简明扼要著称，所有回答都用最精炼的语言表达"},
                    {"role": "user", "content": f"【请用1-2句话回答】{prompt}"},  # 双重提示
                ],
                temperature=0.3,  # 降低随机性
                max_tokens=150,  # 限制最大长度
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            print("智谱AI接口调用失败:", e)
            return "抱歉，AI接口暂时无法访问，请稍后再试。"

