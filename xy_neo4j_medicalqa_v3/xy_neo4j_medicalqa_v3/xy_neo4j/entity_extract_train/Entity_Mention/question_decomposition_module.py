import os
import json
from typing import List, Dict, Any, Tuple
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class HydrologicalQuestionDecomposer:
    """
    水文领域问题分解模块 - 专注于SWAT模型和径流模拟相关问题
    """
    
    def __init__(self, 
                 model_path: str = None, 
                 use_api: bool = True,
                 api_key: str = "sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"):
        """
        初始化问题分解器
        
        Args:
            model_path: 本地模型路径，如果为None则使用API
            use_api: 是否使用API
            api_key: API密钥
        """
        self.use_api = use_api
        self.api_key = api_key
        self.model = None
        self.tokenizer = None
        
        # 域知识库 - 水文模型关键概念
        self.domain_concepts = {
            "径流模拟": ["降水径流", "地表径流", "壤中流", "地下径流", "蒸散发", "入渗", "产流"],
            "SWAT模型": ["子流域划分", "水文响应单元", "HRU", "参数率定", "气象数据", "土壤数据", "地形数据"],
            "基本数据": ["DEM", "土地利用", "土壤类型", "气象数据", "降水", "温度", "太阳辐射", "相对湿度", "风速"],
            "模型评价": ["Nash", "R²", "RMSE", "偏差率", "灵敏度分析", "不确定性分析"],
        }
        
        # 缓存常用问题
        self.question_cache = {}
        
        # 加载本地模型(可选)
        if not self.use_api and model_path:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            except Exception as e:
                print(f"模型加载失败: {e}")
                print("将使用规则方法进行问题分解")
                self.model = None
                self.tokenizer = None
    
    def _extract_entities(self, question: str) -> List[str]:
        """提取问题中的实体和关键词"""
        entities = []
        
        # 简单的规则匹配
        for category, keywords in self.domain_concepts.items():
            for keyword in keywords:
                if keyword in question:
                    entities.append(keyword)
        
        # 特殊实体：流域名称
        basin_pattern = r'([\u4e00-\u9fa5]+流域)'
        basins = re.findall(basin_pattern, question)
        entities.extend(basins)
        
        # 特殊实体：年份和时间段
        year_pattern = r'(\d{4}-\d{4}年|\d{4}年)'
        years = re.findall(year_pattern, question)
        entities.extend(years)
        
        # 特殊实体：数据类型
        data_pattern = r'([\u4e00-\u9fa5]+数据)'
        data_types = re.findall(data_pattern, question)
        entities.extend(data_types)
        
        return list(set(entities))  # 去重
    
    def _classify_question_type(self, question: str) -> str:
        """判断问题类型"""
        if "如何" in question or "怎么" in question:
            return "操作步骤型"
        elif "为什么" in question:
            return "原理解释型"
        elif "什么是" in question or "定义" in question:
            return "概念定义型"
        elif "影响" in question or "关系" in question:
            return "关系分析型"
        elif "优化" in question or "提高" in question:
            return "效果优化型"
        elif "对比" in question or "区别" in question:
            return "比较对比型"
        else:
            return "一般查询型"
    
    def decompose_by_rule(self, question: str) -> List[Dict[str, Any]]:
        """使用规则方法分解问题"""
        # 检查缓存
        if question in self.question_cache:
            return self.question_cache[question]
            
        # 提取实体和确定问题类型
        entities = self._extract_entities(question)
        question_type = self._classify_question_type(question)
        
        # 针对SWAT模型径流模拟任务的问题分解模板
        if "径流模拟" in question or "SWAT" in question.upper():
            # 检测是否为建模完整流程问题
            if "如何" in question and any(basin in question for basin in ["流域", "区域", "地区"]):
                # 返回分解后的子问题列表 - 减少为4个关键问题
                result = [
                    {
                        "id": 1, 
                        "question": f"SWAT模型径流模拟需要哪些基础数据和前处理步骤?",
                        "entity": ["SWAT模型", "径流模拟", "基础数据"],
                        "priority": "高",
                        "type": "前提知识"
                    },
                    {
                        "id": 2,
                        "question": f"{question}中提到的数据如何预处理和导入SWAT模型?",
                        "entity": entities,
                        "priority": "高",
                        "type": "数据准备"
                    },
                    {
                        "id": 3,
                        "question": "SWAT模型如何进行子流域划分和参数设置?",
                        "entity": ["SWAT模型", "子流域", "参数"],
                        "priority": "高",
                        "type": "模型构建"
                    },
                    {
                        "id": 4,
                        "question": "如何评价和优化SWAT模型的径流模拟结果?",
                        "entity": ["SWAT模型", "径流模拟", "评价", "优化"],
                        "priority": "中",
                        "type": "结果评价"
                    }
                ]
            # 特定参数问题
            elif "参数" in question:
                result = [
                    {
                        "id": 1,
                        "question": "SWAT模型中影响径流模拟的关键参数有哪些?",
                        "entity": ["SWAT模型", "参数", "径流模拟"],
                        "priority": "高",
                        "type": "参数识别" 
                    },
                    {
                        "id": 2,
                        "question": "如何确定SWAT模型径流模拟中各参数的合理范围?",
                        "entity": ["SWAT模型", "参数", "范围"],
                        "priority": "中",
                        "type": "参数范围"
                    },
                    {
                        "id": 3,
                        "question": "如何对SWAT模型的参数进行自动率定?",
                        "entity": ["SWAT模型", "参数", "率定"],
                        "priority": "高",
                        "type": "参数率定"
                    }
                ]
            # 数据问题
            elif any(data in question for data in ["数据", "气象", "土壤", "DEM", "土地利用"]):
                result = [
                    {
                        "id": 1,
                        "question": f"{question}中提到的数据如何预处理成SWAT模型所需格式?", 
                        "entity": entities,
                        "priority": "高",
                        "type": "数据预处理"
                    },
                    {
                        "id": 2,
                        "question": "缺少某些数据时，如何利用已有数据进行补充或替代?", 
                        "entity": ["数据", "替代"],
                        "priority": "中",
                        "type": "数据替代"
                    },
                    {
                        "id": 3,
                        "question": "这些数据在SWAT模型中分别影响哪些水文过程?",
                        "entity": ["数据", "SWAT模型", "水文过程"],
                        "priority": "中",
                        "type": "数据影响"
                    }
                ]
            # 默认分解
            else:
                result = [
                    {
                        "id": 1,
                        "question": "使用SWAT模型进行径流模拟的基本步骤是什么?",
                        "entity": ["SWAT模型", "径流模拟", "步骤"],
                        "priority": "高",
                        "type": "流程概述"
                    },
                    {
                        "id": 2,
                        "question": f"{question}中提到的特定条件如何在SWAT模型中体现?",
                        "entity": entities,
                        "priority": "高", 
                        "type": "特定条件"
                    },
                    {
                        "id": 3,
                        "question": "SWAT模型径流模拟结果如何评价和解释?",
                        "entity": ["SWAT模型", "径流模拟", "评价"],
                        "priority": "中",
                        "type": "结果评价"
                    }
                ]
        else:
            # 非SWAT相关问题的通用分解
            result = [
                {
                    "id": 1,
                    "question": f"{question}的基本概念和背景是什么?",
                    "entity": entities,
                    "priority": "中",
                    "type": "概念背景"
                },
                {
                    "id": 2, 
                    "question": f"解决{question}需要哪些关键步骤?",
                    "entity": entities,
                    "priority": "高",
                    "type": "解决步骤"
                },
                {
                    "id": 3,
                    "question": f"{question}可能遇到哪些常见问题及如何解决?",
                    "entity": entities,
                    "priority": "低",
                    "type": "问题解决"
                }
            ]
            
        # 保存到缓存
        self.question_cache[question] = result
        return result
    
    def decompose_by_llm(self, question: str) -> List[Dict[str, Any]]:
        """使用大模型分解问题"""
        if self.use_api:
            # 使用API调用大模型
            try:
                import requests
                
                # 构建prompt
                prompt = f"""
                你是一个水文领域的专家，擅长SWAT模型和径流模拟。请将以下复杂问题分解为3-4个具体的子问题，以便进行多轮问答。
                每个子问题应该清晰明确，并且按照逻辑顺序排列，从基础问题到高级问题。
                
                问题: {question}
                
                请以JSON格式返回结果，格式为:
                {{
                  "decomposed_questions": [
                    {{"id": 1, "question": "子问题1", "priority": "高/中/低", "type": "问题类型"}},
                    {{"id": 2, "question": "子问题2", "priority": "高/中/低", "type": "问题类型"}},
                    ...
                  ]
                }}
                """
                
                # 设置API请求
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                data = {
                    "model": "gpt-4o", # 使用更快的模型
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2
                }
                
                # 发送请求
                response = requests.post(
                    "https://api.chatanywhere.tech/v1/chat/completions", # 使用实际的API端点
                    headers=headers,
                    json=data
                )
                
                # 解析响应
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 提取JSON部分
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                    json_data = json.loads(json_content)
                    return json_data["decomposed_questions"]
                
                return self.decompose_by_rule(question)  # 如果API提取失败，回退到规则方法
                
            except Exception as e:
                print(f"API调用失败: {e}")
                return self.decompose_by_rule(question)  # 回退到规则方法
                
        elif hasattr(self, 'model') and self.model and hasattr(self, 'tokenizer') and self.tokenizer:
            # 使用本地模型
            try:
                # 构建prompt
                prompt = f"""
                你是一个水文领域的专家，擅长SWAT模型和径流模拟。请将以下复杂问题分解为3-4个具体的子问题，以便进行多轮问答。
                每个子问题应该清晰明确，并且按照逻辑顺序排列，从基础问题到高级问题。
                
                问题: {question}
                
                请以JSON格式返回结果，格式为:
                {{
                  "decomposed_questions": [
                    {{"id": 1, "question": "子问题1", "priority": "高/中/低", "type": "问题类型"}},
                    {{"id": 2, "question": "子问题2", "priority": "高/中/低", "type": "问题类型"}},
                    ...
                  ]
                }}
                """
                
                # 模型推理
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95
                )
                
                # 解码输出
                content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 提取JSON部分
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                    json_data = json.loads(json_content)
                    return json_data["decomposed_questions"]
                
                return self.decompose_by_rule(question)  # 如果模型提取失败，回退到规则方法
                
            except Exception as e:
                print(f"本地模型推理失败: {e}")
                return self.decompose_by_rule(question)  # 回退到规则方法
        else:
            # 没有可用的模型，使用规则方法
            return self.decompose_by_rule(question)
    
    def decompose_question(self, question: str, use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        分解问题的主函数
        
        Args:
            question: 用户问题
            use_llm: 是否使用大模型方法，如果为False则使用规则方法
        
        Returns:
            分解后的子问题列表
        """
        # 检查缓存
        if question in self.question_cache:
            return self.question_cache[question]
            
        if use_llm and (self.use_api or (hasattr(self, 'model') and self.model and hasattr(self, 'tokenizer') and self.tokenizer)):
            result = self.decompose_by_llm(question)
        else:
            result = self.decompose_by_rule(question)
            
        # 保存到缓存
        self.question_cache[question] = result
        return result
    
    def format_questions_for_qa(self, decomposed_questions: List[Dict[str, Any]]) -> List[str]:
        """
        将分解后的问题格式化为多轮问答序列
        
        Args:
            decomposed_questions: 分解后的子问题列表
            
        Returns:
            格式化后的问题列表
        """
        # 按照优先级排序
        priority_map = {"高": 3, "中": 2, "低": 1}
        sorted_questions = sorted(
            decomposed_questions, 
            key=lambda x: (priority_map.get(x.get("priority", "中"), 0), x.get("id", 0)), 
            reverse=True
        )
        
        # 提取问题文本
        return [item["question"] for item in sorted_questions]

# 使用示例
if __name__ == "__main__":
    # 初始化问题分解器
    decomposer = HydrologicalQuestionDecomposer(use_api=False)
    
    # 测试问题
    test_question = "我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?"
    
    # 分解问题
    decomposed_questions = decomposer.decompose_question(test_question, use_llm=False)
    
    # 打印结果
    print(f"原始问题: {test_question}\n")
    print("分解后的子问题:")
    for q in decomposed_questions:
        print(f"[{q['priority']}] {q['id']}. {q['question']} (类型: {q['type']})")
    
    # 获取格式化的问答序列
    qa_sequence = decomposer.format_questions_for_qa(decomposed_questions)
    
    print("\n问答序列:")
    for i, q in enumerate(qa_sequence, 1):
        print(f"{i}. {q}")