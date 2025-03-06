# -*- coding: utf-8 -*-
import json
from typing import List
from pydantic import BaseModel
from openai import OpenAI

class Entity(BaseModel):
    name: str
    type: str
    context: str

class DirectMentionRecognizer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.chatanywhere.tech/v1"
        )
        self.entity_types = [
            "地理问题", "地理场景", "对象系统", "系统机理",
            "时空数据", "数据来源", "处理方法", "集成模型",
            "基础模型", "开发步骤", "评价方法", "评价结果",
            "模型应用", "应用结果", "总结讨论"
        ]

    def recognize(self, text: str) -> List[Entity]:
        prompt = f'''作为地理学里SWAT模型径流模拟领域的专家，请从以下文本中提取实体信息，遵循这些要求：
1. 识别以下类型实体：{", ".join(self.entity_types)}
2. 每个实体必须包含：
   - name：文本中出现的原始名称
3. 用JSON格式返回，不要解释

示例输入："我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河下游进行径流模拟?"
    
示例输出：
{{
  {
  "地理场景": [
    {
      "name": "淮河流域"
    },
    {
      "name": "淮河下游"
    }
  ],
  "时空数据": [
    {
      "name": "2015-2020年的气象数据"
    },
    {
      "name": "全国的土壤数据"
    }
  ],
  "地理问题": [
    {
      "name": "径流模拟"
    }
  ]
}
}}

待分析文本：{text}'''

        try:
            print("\n[DEBUG] 发送请求到API...")  # 调试输出
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content
            print(f"[DEBUG] API返回原始结果:\n{result}\n")  # 调试输出
            
            # 处理可能的markdown代码块
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            
            print(f"[DEBUG] 处理后的JSON:\n{result}\n")  # 调试输出
            
            data = json.loads(result)
            return [Entity(**e) for e in data.get("entities", [])]
            
        except Exception as e:
            print(f"[ERROR] API调用失败: {str(e)}")  # 错误输出
            return []

if __name__ == "__main__":
    recognizer = DirectMentionRecognizer(
        api_key="sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
    )
    
    test_text = "我现在有黄河流域2015-2020年的气象数据和全国的DEM数据,如何对黄河源区进行径流模拟?"
    
    print("文本分析结果：")
    entities = recognizer.recognize(test_text)
    for idx, ent in enumerate(entities, 1):
        print(f"{idx}. [{ent.type}] {ent.name}")
        print(f"   上下文：{ent.context}\n")
