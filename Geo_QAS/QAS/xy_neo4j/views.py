from .get_deepseek_response import GetDeepseekResponse

# 任何调用zhipu相关API的地方，修改为deepseek:
deepseek_instance = GetDeepseekResponse()
response = deepseek_instance.get_deepseek_response(prompt)