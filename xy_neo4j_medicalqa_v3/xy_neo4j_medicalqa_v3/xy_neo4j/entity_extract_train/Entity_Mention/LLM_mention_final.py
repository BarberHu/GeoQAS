# -*- coding: utf-8 -*-
import json
from typing import List
from openai import OpenAI

class DirectMentionRecognizer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.chatanywhere.tech/v1"
        )

    def recognize(self, text: str) -> List[str]:
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
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            
            # 处理可能的多余字符
            result = result.replace("```json", "").replace("```", "").strip()
            
            # 直接解析为列表
            entities = json.loads(result)
            return [ent.strip() for ent in entities if isinstance(ent, str)]
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            return []

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
    
    try:
        assert recognizer.recognize(test_case) == expected_output, "测试用例验证失败"
        print("测试通过!")
    except AssertionError:
        print("测试失败: 输出结果与预期不符")
