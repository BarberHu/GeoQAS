import json
import time
from openai import OpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import API_KEYS, LLM_CONFIG


class DeepSeekMentionRecognizer:
    """使用DeepSeek API进行实体识别的组件"""
    
    def __init__(self, api_key=None):
        """初始化DeepSeek实体识别器"""
        self.api_key = api_key or API_KEYS["deepseek"]
        self.base_url = LLM_CONFIG["base_url"]
        
        # 初始化DeepSeek客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 缓存，避免重复识别
        self.mention_cache = {}
    
    def recognize(self, text):
        """识别文本中的实体"""
        # 检查缓存
        if text in self.mention_cache:
            return self.mention_cache[text]
        
        try:
            print(f"[DeepSeekMentionRecognizer] 开始识别实体: {text}")
            start_time = time.time()
            
            prompt = f"""
            请从以下文本中识别出所有与水文学、水资源管理、环境科学相关的实体名词。格式要求：
            1. 仅返回一个JSON数组，每个元素是一个字符串
            2. 识别出的实体应该是具体的名词或专有名词
            3. 不要包含普通动词、形容词或介词
            4. 识别的实体应尽可能具体，如"SWAT模型"比"模型"更好

            文本：{text}

            回复格式示例: ["实体1", "实体2", "实体3"]
            """
            
            response = self.client.chat.completions.create(
                model=LLM_CONFIG["default_model"],
                messages=[
                    {"role": "system", "content": "你是一个专业的实体识别助手，擅长从文本中提取水文和环境科学领域的专业术语和实体。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度以获得更确定的结果
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # 尝试解析JSON结果
            try:
                # 处理可能的非JSON输出
                if not result.startswith('['):
                    # 查找第一个[和最后一个]之间的内容
                    start_idx = result.find('[')
                    end_idx = result.rfind(']')
                    if start_idx != -1 and end_idx != -1:
                        result = result[start_idx:end_idx+1]
                
                entities = json.loads(result)
                # 确保结果是字符串列表
                entities = [str(entity).strip() for entity in entities if entity]
                
                # 去除重复
                entities = list(set(entities))
                
            except json.JSONDecodeError:
                print(f"[DeepSeekMentionRecognizer] JSON解析失败: {result}")
                # 如果解析失败，尝试使用简单的方法提取实体
                import re
                entities = re.findall(r'"([^"]+)"', result)
                entities = list(set(entities))  # 去除重复
            
            end_time = time.time()
            print(f"[DeepSeekMentionRecognizer] 实体识别完成，耗时: {end_time - start_time:.2f}秒")
            print(f"[DeepSeekMentionRecognizer] 识别到的实体: {entities}")
            
            # 缓存结果
            self.mention_cache[text] = entities
            return entities
            
        except Exception as e:
            print(f"[DeepSeekMentionRecognizer] 实体识别失败: {e}")
            return []  # 出错时返回空列表 