# scripts/data_processor.py
import json
import random
from faker import Faker
import os

fake = Faker('zh_CN')

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径
project_root = os.path.dirname(current_dir)
# 确保data目录存在
data_dir = os.path.join(project_root, 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 问题类型模板库
QUESTION_TEMPLATES = {
    "建模流程": [
        ("如何构建{area}的{model}模型来研究{problem}问题？", ["数据准备", "模型构建", "参数率定"]),
        ("在{area}使用{model}模拟{process}的具体步骤是什么？", ["前处理", "模型搭建", "后处理"]),
        ("基于{data}数据建立{area}的{model}模型流程是怎样的？", ["数据处理", "模型配置", "精度验证"])
    ],
    "数据需求": [
        ("使用{model}模型研究{area}的{problem}需要哪些数据？", ["基础数据", "驱动数据", "验证数据"]),
        ("构建{area}{process}模型需要准备什么数据？数据精度要求如何？", ["数据类型", "精度要求", "来源建议"]),
        ("{model}模型在{area}应用时的数据预处理流程是什么？", ["数据格式", "预处理步骤", "质量控制"])
    ],
    "参数设置": [
        ("{model}模型在{area}模拟{process}时的关键参数有哪些？", ["参数列表", "取值范围", "敏感性"]),
        ("如何确定{model}模型在{area}的{parameter}参数？", ["参数含义", "获取方法", "建议值"]),
        ("{area}使用{model}模型时如何进行参数率定？", ["率定方法", "评价指标", "验证策略"])
    ],
    "精度评估": [
        ("如何评估{model}在{area}模拟{process}的精度？", ["评价指标", "基准数据", "阈值标准"]),
        ("{model}模型在{area}的模拟结果如何验证？", ["验证方法", "数据要求", "精度标准"]),
        ("评价{model}在{area}模拟{process}效果的指标有哪些？", ["统计指标", "评价方法", "精度要求"])
    ]
}

# 地理建模领域知识库
DOMAIN_KNOWLEDGE = {
    "models": ["SWAT", "MIKE SHE", "HEC-HMS", "MODFLOW", "WetSpa", "TOPMODEL"],
    "areas": ["黄河流域", "长江流域", "海河流域", "珠江流域", "松辽流域", "淮河流域"],
    "processes": ["径流模拟", "洪水预报", "水质模拟", "地下水评价", "生态水文"],
    "parameters": ["CN值", "曼宁系数", "下渗系数", "蒸散发系数", "水力传导度", "地表糙率"],
    "data_types": ["DEM数据", "土地利用", "土壤类型", "气象数据", "水文监测", "遥感影像"],
    "problems": ["流域产流", "河道演进", "面源污染", "地下水补给", "生态需水"],
    "mechanisms": ["水文循环", "下渗过程", "河道汇流", "地表径流", "基流补给"]
}

def generate_training_data():
    dataset = []
    
    for _ in range(1500):
        # 随机选择问题类型
        q_type, templates = random.choice(list(QUESTION_TEMPLATES.items()))
        template, aspects = random.choice(templates)
        
        # 填充模板参数
        params = {
            'model': random.choice(DOMAIN_KNOWLEDGE['models']),
            'area': random.choice(DOMAIN_KNOWLEDGE['areas']),
            'process': random.choice(DOMAIN_KNOWLEDGE['processes']),
            'parameter': random.choice(DOMAIN_KNOWLEDGE['parameters']),
            'data': random.choice(DOMAIN_KNOWLEDGE['data_types']),
            'problem': random.choice(DOMAIN_KNOWLEDGE['problems']),
            'mechanism': random.choice(DOMAIN_KNOWLEDGE['mechanisms'])
        }
        
        # 生成完整问题
        question = template.format(**params)
        
        # 构建训练样本
        dataset.append({
            "instruction": "作为地理建模专家，请基于知识图谱分析以下问题",
            "input": question,
            "output": {
                "问题类型": q_type,
                "核心实体": [
                    {"type": "模型名称", "name": params.get('model', '')},
                    {"type": "研究区域", "name": params.get('area', '')},
                    {"type": "模拟过程", "name": params.get('process', '')}
                ],
                "关键要素": aspects,
                "预期分析": [
                    "技术路线",
                    "数据需求",
                    "关键步骤",
                    "注意事项"
                ]
            }
        })
    
    # 修改保存路径为绝对路径
    save_path = os.path.join(data_dir, 'train.json')
    
    # 保存数据
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2, cls=CustomEncoder)
    print(f"数据已保存到: {save_path}")

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    generate_training_data()