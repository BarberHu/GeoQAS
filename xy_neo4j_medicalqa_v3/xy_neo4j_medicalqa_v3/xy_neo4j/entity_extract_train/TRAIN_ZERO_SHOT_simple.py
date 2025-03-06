# -*- coding: utf-8 -*-
import json
import requests
from typing import List, Dict
from pydantic import BaseModel

# 实体类型定义（根据您的需求定制）
class EntityTypes:
    MODEL = "模型名称"
    BASIN = "研究区域"
    PROCESS = "模拟过程"
    PARAMETER = "模型参数"
    DATA = "数据需求"
    EVALUATION = "精度指标"

class Entity(BaseModel):
    name: str
    type: str

class HydrologyEntityRecognizer:
    def __init__(self, api_key: str):
        self.api_url = "https://api.chatanywhere.tech/v1/chat/completions"
        self.api_key = api_key
        self.entity_definitions = {
            EntityTypes.MODEL: ["SWAT", "MIKE SHE", "HEC-HMS", "TOPMODEL", "MODFLOW", "WetSpa"],
            EntityTypes.BASIN: ["海河流域", "黄河流域", "淮河流域", "珠江流域"],
            EntityTypes.PROCESS: ["径流模拟", "洪水预报", "水质模拟", "地下水评价", "生态水文"],
            EntityTypes.PARAMETER: ["糙率参数", "下渗系数", "蒸散发系数", "地表径流"],
            EntityTypes.DATA: ["DEM数据", "土地利用数据", "气象数据"],
            EntityTypes.EVALUATION: ["NSE系数", "RMSE", "R²", "PBIAS"]
        }

    def _build_prompt(self, text: str) -> str:
        return f'''作为水文模型专家，请从以下文本中识别实体，直接返回JSON格式：
{{"entities": [
    {{"name": "实体名", "type": "实体类型"}}
]}}

可用的实体类型和值：
{json.dumps(self.entity_definitions, indent=2, ensure_ascii=False)}

待分析文本：{text}'''

    def _extract_json_from_response(self, content: str) -> dict:
        """从响应内容中提取JSON"""
        try:
            # 如果内容被markdown包裹，去除markdown标记
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {"entities": []}

    def recognize_entities(self, text: str) -> List[Entity]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": self._build_prompt(text)
            }],
            "temperature": 0.1,
            "max_tokens": 500
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            result = self._extract_json_from_response(content)
            validated_entities = self._validate_entities(result.get('entities', []))
            return [Entity(**e) for e in validated_entities]
            
        except Exception as e:
            print(f"API请求或处理失败: {str(e)}")
            return []

    def _validate_entities(self, entities: List[dict]) -> List[dict]:
        valid_entities = []
        valid_types = self.entity_definitions.keys()
        
        for ent in entities:
            ent_type = ent.get('type', '')
            ent_name = ent.get('name', '')
            
            if ent_type not in valid_types:
                continue
                
            allowed_values = self.entity_definitions[ent_type]
            if any(val in ent_name for val in allowed_values):
                valid_entities.append({
                    "name": next((v for v in allowed_values if v in ent_name), ent_name),
                    "type": ent_type
                })
                
        return valid_entities

# 使用示例
if __name__ == "__main__":
    # 初始化识别器
    recognizer = HydrologyEntityRecognizer(
        api_key="sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
    )
    
    # 测试问题
    test_question = "我现在有黄河流域2015-2020年的气象数据和全国的DEM数据,如何对黄河源区进行径流模拟?"
    
    print("开始识别实体...")
    # 执行识别
    entities = recognizer.recognize_entities(test_question)
    
    # 打印结果
    print("\n识别到的实体：")
    for ent in entities:
        print(f"- [{ent.type}] {ent.name}")
