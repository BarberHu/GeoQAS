# -*- coding: utf-8 -*-
import json
from typing import List, Dict
from openai import OpenAI

class DirectMentionRecognizer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.chatanywhere.tech/v1"
        )
        # 添加缓存以避免重复查询
        self.cache = {}
    
    def recognize(self, text: str) -> List[str]:
        """识别单个文本中的实体"""
        print(f"\n[Mention Recognition] 开始识别文本中的实体: {text}")
        # 检查缓存
        if text in self.cache:
            print(f"[Mention Recognition] 使用缓存结果: {self.cache[text]}")
            return self.cache[text]
            
        prompt = f'''作为水文专家，请从以下文本中直接提取地理相关实体名称，要求：
            1. 只需输出实体名称列表
            2. 每个实体必须是文本中出现的原始片段
            3. 不要分类或解释

            示例输入："我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?"
            示例输出：
            ["淮河流域", "2015-2020年的气象数据", "全国的土壤数据", "径流模拟"]

            当前输入文本：{text}'''

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                timeout=30
            )
            
            result = response.choices[0].message.content
            
            # 处理可能的多余字符
            result = result.replace("```json", "").replace("```", "").strip()
            
            # 直接解析为列表
            entities = json.loads(result)
            result = [ent.strip() for ent in entities if isinstance(ent, str)]
            
            # 保存到缓存
            self.cache[text] = result
            print(f"[Mention Recognition] 识别到的实体: {result}")
            return result
            
        except Exception as e:
            print(f"[Mention Recognition] 实体识别失败: {e}")
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
                model="gpt-4o",
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0.1,
                max_tokens=2000
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
                    results[text] = entities
                    self.cache[text] = entities
            
            return results
            
        except Exception as e:
            print(f"批量处理ERROR: {str(e)}")
            # 失败时回退到单个处理
            for text in uncached_texts:
                results[text] = self.recognize(text)
            return results

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