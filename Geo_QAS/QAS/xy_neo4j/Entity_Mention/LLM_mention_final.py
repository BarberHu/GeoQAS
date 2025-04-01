# -*- coding: utf-8 -*-
import json
from typing import List, Dict
from openai import OpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from config import API_KEYS, LLM_CONFIG, ENTITY_CONFIG

class DirectMentionRecognizer:
    def __init__(self, api_key=None):
        """初始化实体识别器"""
        self.cache = {}
        self.client = OpenAI(
            api_key=api_key or API_KEYS["deepseek"],
            base_url=LLM_CONFIG["base_url"]
        )
        
        self.system_prompt = """作为水文专家，请从以下文本中直接提取地理相关实体名称，要求：
1. 只需输出实体名称列表
2. 每个实体必须是文本中出现的原始片段
3. 不要分类或解释

示例输入："我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?"
示例输出：
["淮河流域", "2015-2020年的气象数据", "全国的土壤数据", "径流模拟"]"""

        # 使用配置文件中的过滤词列表
        self.filter_words = set(ENTITY_CONFIG["filter_words"])

    def recognize(self, text: str) -> List[str]:
        """识别单个文本中的实体"""
        # 检查缓存
        if text in self.cache:
            return self.cache[text]
            
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"文本：{text}"}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            
            result = response.choices[0].message.content
            
            # 处理可能的多余字符
            result = result.replace("```json", "").replace("```", "").strip()
            
            # 直接解析为列表
            entities = json.loads(result)
            result = [ent.strip() for ent in entities if isinstance(ent, str)]
            
            # 在处理 LLM 返回的实体时进行过滤
            filtered_result = []
            for entity in result:
                # 跳过过滤词列表中的词
                if entity.strip() in self.filter_words:
                    continue
                    
                # 跳过过于简单的词（如单个字符）
                if len(entity.strip()) <= 1:
                    continue
                    
                # 跳过纯数字
                if entity.strip().isdigit():
                    continue
                    
                filtered_result.append(entity.strip())
            
            # 保存到缓存
            self.cache[text] = filtered_result
            return filtered_result
            
        except Exception as e:
            print(f"实体识别失败: {e}")
            return []
    
    # 新增批量处理方法
    def recognize_batch(self, texts: List[str]) -> Dict[str, List[str]]:
        """批量识别多个文本中的实体
        
        Args:
            texts: 文本列表
            
        Returns:
            Dict[str, List[str]]: 以文本为键，实体列表为值的字典
        """
        results = {}
        uncached_texts = []
        uncached_indices = []
        
        # 检查缓存
        for i, text in enumerate(texts):
            if text in self.cache:
                results[text] = self.cache[text]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # 如果所有文本都已缓存，直接返回
        if not uncached_texts:
            return results
        
        # 构建批量处理的prompt
        batch_prompt = "作为水文专家，请从以下多个文本中分别提取地理相关实体名称，要求：\n"
        batch_prompt += "1. 为每个文本输出一个实体名称列表\n"
        batch_prompt += "2. 每个实体必须是文本中出现的原始片段\n"
        batch_prompt += "3. 不要分类或解释\n\n"
        
        for i, text in enumerate(uncached_texts, 1):
            batch_prompt += f"文本{i}: {text}\n"
        
        batch_prompt += "\n请以JSON格式返回结果，格式为:\n"
        batch_prompt += "{\n"
        for i in range(1, len(uncached_texts) + 1):
            batch_prompt += f'  "文本{i}": ["实体1", "实体2", ...],\n'
        batch_prompt += "}\n"
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": batch_prompt}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            
            result = response.choices[0].message.content
            
            # 处理可能的多余字符
            result = result.replace("```json", "").replace("```", "").strip()
            
            # 解析JSON结果
            parsed_results = json.loads(result)
            
            # 将结果添加到结果字典和缓存
            for i, text in enumerate(uncached_texts):
                key = f"文本{i+1}"
                if key in parsed_results:
                    entities = [ent.strip() for ent in parsed_results[key] if isinstance(ent, str)]
                    # 在处理 LLM 返回的实体时进行过滤
                    filtered_entities = []
                    for entity in entities:
                        # 跳过过滤词列表中的词
                        if entity.strip() in self.filter_words:
                            continue
                            
                        # 跳过过于简单的词（如单个字符）
                        if len(entity.strip()) <= 1:
                            continue
                            
                        # 跳过纯数字
                        if entity.strip().isdigit():
                            continue
                            
                        filtered_entities.append(entity.strip())
                    
                    results[text] = filtered_entities
                    self.cache[text] = filtered_entities
            
            return results
            
        except Exception as e:
            print(f"批量处理ERROR: {str(e)}")
            # 失败时回退到单个处理
            for text in uncached_texts:
                results[text] = self.recognize(text)
            return results

    def _format_prompt(self, text):
        """格式化提示词"""
        return f"""请识别以下文本中的专业术语、概念、方法和重要名词，不包括常见的模型名称和通用词汇：

文本：{text}

要求：
1. 只返回关键的专业术语和概念
2. 不要包括"SWAT模型"等常见模型名称
3. 不要包括"研究"、"分析"等通用词汇
4. 每个术语用逗号分隔
5. 不要解释，只列出术语

专业术语："""

if __name__ == "__main__":
    recognizer = DirectMentionRecognizer(
        api_key="sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
    )
    
    test_text = "我现在有黄河流域2015-2020年的气象数据和全国的DEM数据,如何对黄河源区进行径流模拟?"
    
    print("文本分析结果：")
    entities = recognizer.recognize(test_text)
    print(entities)  # 直接打印数组

    # 验证测试案例
    test_case = "我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?"
    expected_output = ["淮河流域", "2015-2020年的气象数据", "全国的土壤数据","淮河流域","径流模拟"]
    
    print("\n测试用例验证：")
    print(f"期望输出: {expected_output}")
    actual_output = recognizer.recognize(test_case)
    print(f"实际输出: {actual_output}")
    
    # 测试批量处理
    batch_texts = [
        "我现在有黄河流域2015-2020年的气象数据和全国的DEM数据,如何对黄河源区进行径流模拟?",
        "淮河流域SWAT模型的参数敏感性如何分析?",
        "如何利用SWAT模型评估气候变化对径流的影响?"
    ]
    
    print("\n批量处理测试：")
    batch_results = recognizer.recognize_batch(batch_texts)
    for text, entities in batch_results.items():
        print(f"文本: {text}")
        print(f"实体: {entities}")