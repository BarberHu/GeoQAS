from openai import OpenAI
import json

class HydrologyEntityRecognizer:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA",
            base_url="https://api.chatanywhere.tech/v1"
        )
        
        # 定义实体类型
        self.entity_types = {
            "模型名称": ["SWAT", "MIKE SHE", "HEC-HMS"],
            "研究区域": ["黄河流域", "长江流域", "海河流域"],
            "数据类型": ["DEM数据", "降水数据", "气象数据"],
            "模拟过程": ["径流模拟", "产流计算", "汇流计算"]
        }

    def recognize_entities(self, text: str):
        """识别文本中的水文实体"""
        prompt = f"""作为水文领域专家，请从以下文本中识别实体，以JSON格式返回：
实体类型和可能值：
{json.dumps(self.entity_types, ensure_ascii=False, indent=2)}

待分析文本：{text}

请返回格式如下：
{{"entities": [
    {{"name": "实体名", "type": "实体类型"}}
]}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            # 如果结果被markdown包裹，去除markdown标记
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            
            entities = json.loads(result)
            return entities["entities"]
            
        except Exception as e:
            print(f"实体识别失败: {e}")
            return []

if __name__ == "__main__":
    recognizer = HydrologyEntityRecognizer()
    text = "我现在有黄河流域2005-2010年的降水数据和全国的DEM高程数据,现在我想对黄河流域进行径流模拟,请帮我设计一下建模步骤？"
    
    print("开始识别实体...")
    results = recognizer.recognize_entities(text)
    
    print("\n识别到的实体：")
    for entity in results:
        print(f"- [{entity['type']}] {entity['name']}")
