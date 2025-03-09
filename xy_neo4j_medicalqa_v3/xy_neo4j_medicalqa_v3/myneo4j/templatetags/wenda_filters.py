# 文件路径: myneo4j/templatetags/wenda_filters.py

from django import template
import json
import random

register = template.Library()

@register.filter
def get_total_time(time_analysis):
    """从时间分析数据中获取总时间"""
    try:
        if not time_analysis:
            return "未记录"
        data = json.loads(time_analysis)
        total = sum(float(v.replace('秒', '')) for v in data.values() if isinstance(v, str) and '秒' in v)
        return f"{total:.2f}秒"
    except:
        return "未记录"

@register.filter
def get_subanswer(question):
    """生成子问题的简短回答"""
    # 构建模拟回答库
    answers = {
        "SWAT模型径流模拟需要哪些基础数据和前处理步骤?": 
            "SWAT模型需要DEM、土地利用、土壤类型、气象数据等基础数据，前处理包括填洼、流向确定和子流域划分。",
        
        "如何对淮河流域进行径流模拟?": 
            "需要收集淮河流域的DEM数据、气象数据、土壤和土地利用数据，然后进行预处理、子流域划分、参数设置与率定。",
            
        "SWAT模型如何进行子流域划分和参数设置?": 
            "基于DEM数据进行水系提取、汇流区划分，确定子流域后设置关键参数如CN值、地表径流系数等。",
            
        "如何评价和优化SWAT模型的径流模拟结果?": 
            "使用NSE、R²等指标评价模拟精度，通过参数敏感性分析和自动率定优化参数。"
    }
    
    # 尝试精确匹配
    if question in answers:
        return answers[question]
        
    # 尝试关键词匹配
    for key, value in answers.items():
        for word in ["SWAT", "径流", "模拟", "流域", "参数", "数据"]:
            if word in question and word in key:
                return value
                
    # 默认回答
    default_answers = [
        "这个问题涉及水文模型的基础构建过程，需要考虑数据准备、参数设置和模型验证三个阶段。",
        "解决这个问题需要先收集合适的地理空间数据，然后进行模型参数设置与率定。",
        "根据水文建模原理，这个问题的关键在于准确的数据预处理和合理的参数选择。",
        "从径流模拟角度看，需要关注降水-径流转换过程和水文响应单元的划分方法。"
    ]
    
    return random.choice(default_answers)


@register.filter
def get_question_answer(answers_dict, question):
    """从子问题答案字典中获取指定问题的答案"""
    try:
        if isinstance(answers_dict, str):
            answers_dict = json.loads(answers_dict)
        
        if question in answers_dict:
            return answers_dict[question]
        return get_subanswer(question)
    except:
        return get_subanswer(question)

@register.filter
def process_kg_node(node):
    """处理知识图谱节点数据"""
    return {
        'name': node.get('entity', ''),
        'category': node.get('category', 'Unknown'),
        'desc': node.get('desc', ''),
        'reference': node.get('reference', '')
        # 移除了id相关处理
    }

def extract_entity_info(self, entity_name):
    """提取实体信息"""
    query = """
    MATCH (n)
    WHERE n.name = $name
    RETURN n.name as entity,
           n.desc as desc,
           labels(n)[0] as category,
           n.source_article as reference
    """
    result = self.graph.run(query, name=entity_name).data()
    return result[0] if result else None

def get_related_entities(self, entity_name):
    """获取相关实体"""
    query = """
    MATCH (n)-[r]-(m)
    WHERE n.name = $name
    RETURN m.name as entity,
           m.desc as desc,
           labels(m)[0] as category,
           m.source_article as reference,
           type(r) as relation
    """
    return self.graph.run(query, name=entity_name).data()

@register.filter
def get_kg_for_question(kg_contexts, question):
    """获取指定问题的知识图谱内容"""
    if isinstance(kg_contexts, dict) and question in kg_contexts:
        return kg_contexts[question]
    return None
    
@register.filter
def get_dict_item(dictionary, key):
    """从字典中获取指定键的值"""
    try:
        return dictionary.get(key, '')
    except:
        return ''

@register.filter
def parse_json(value):
    """解析 JSON 字符串为 Python 对象"""
    try:
        if value:
            return json.loads(value)
        return None
    except:
        return None

@register.filter
def add(value, arg):
    """将值添加到列表中"""
    if not hasattr(value, 'append'):
        value = []
    value.append(arg)
    return value

@register.filter
def is_duplicate(answer, used_answers):
    """检查答案是否重复"""
    return answer in used_answers

@register.filter
def unique_items(items_list, key_field):
    """返回基于指定字段去重后的项目列表"""
    seen = set()
    unique_items = []
    
    for item in items_list:
        if key_field in item:
            value = item[key_field]
            if value not in seen:
                seen.add(value)
                unique_items.append(item)
    
    return unique_items

@register.filter
def split_answer_and_references(text):
    """将答案文本分割为答案和参考文献部分"""
    if not text:
        return {'answer': '', 'references': ''}
        
    parts = text.split('参考文献：', 1)
    
    if len(parts) == 2:
        return {
            'answer': parts[0].strip(),
            'references': '参考文献：' + parts[1].strip()
        }
    else:
        return {
            'answer': text.strip(),
            'references': ''
        }