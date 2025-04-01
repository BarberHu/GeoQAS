import os
import json
from typing import List, Dict, Any, Tuple
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

class HydrologicalQuestionDecomposer:
    """
    水文领域问题分解模块 - 基于图谱结构的问题分解，增强了实体识别和关系映射
    """
    
    def __init__(self, 
                 model_path: str = None, 
                 use_api: bool = True,
                 api_key: str = None,
                 provider: str = None):
        """
        初始化问题分解器
        
        Args:
            model_path: 本地模型路径，如果为None则使用API
            use_api: 是否使用API
            api_key: API密钥，如果为None则从配置中获取
            provider: LLM提供商名称，如果为None则使用默认提供商
        """
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        from config import API_KEYS, LLM_CONFIG
        from llm_client_factory import LLMClientFactory
        
        self.use_api = use_api
        self.provider = provider or LLM_CONFIG["default_provider"]
        self.api_key = api_key or API_KEYS.get(self.provider)
        self.model = None
        self.tokenizer = None
        self.original_question = ""
        
        # 创建LLM客户端
        if self.use_api:
            try:
                self.llm_client = LLMClientFactory.create_client(self.provider)
                print(f"[问题分解器] 成功创建LLM客户端，提供商: {self.provider}")
            except Exception as e:
                print(f"[问题分解器] 创建LLM客户端失败: {e}")
                self.llm_client = None
                self.use_api = False
        
        # 图谱结构定义 - 关系及对应实体类型
        self.graph_relations = {
            "所属场景": "地理场景",
            "研究对象": "对象系统",
            "使用数据": "时空数据", 
            "使用模型": "集成模型",
            "问题结论": "总结讨论"
        }
        
        # 各关系对应的问题模板
        self.relation_question_templates = {
            "所属场景": "这个问题涉及哪些地理场景或区域？",
            "研究对象": "在解决这个问题时，需要研究哪些对象系统？",
            "使用数据": "解决这个问题需要哪些时空数据？如何获取和处理这些数据？",
            "使用模型": "解决这个问题可以使用什么模型？这些模型如何集成？",
            "问题结论": "解决这个问题后，可能得出哪些结论？"
        }
        
        # 域知识库 - 水文模型关键概念
        self.domain_concepts = {
            "径流模拟": ["降水径流", "地表径流", "壤中流", "地下径流", "蒸散发", "入渗", "产流"],
            "SWAT模型": ["子流域划分", "水文响应单元", "HRU", "参数率定", "气象数据", "土壤数据", "地形数据"],
            "基本数据": ["DEM", "土地利用", "土壤类型", "气象数据", "降水", "温度", "太阳辐射", "相对湿度", "风速"],
            "模型评价": ["Nash", "R²", "RMSE", "偏差率", "灵敏度分析", "不确定性分析"],
            # 新增图谱实体类型相关概念
            "地理场景": ["流域", "区域", "盆地", "高原", "平原", "山地", "丘陵"],
            "对象系统": ["水资源系统", "生态系统", "气候系统", "地质系统", "土地系统"],
            "时空数据": ["遥感数据", "实测数据", "统计数据", "历史数据", "预测数据"],
            "集成模型": ["耦合模型", "嵌套模型", "级联模型", "参数优化模型"],
            "基础模型": ["水文模型", "气象模型", "土壤模型", "植被模型"],
            "评价方法": ["精度评价", "敏感性分析", "不确定性评价"]
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
                print(f"[问题分解器] 成功加载本地模型: {model_path}")
            except Exception as e:
                print(f"[问题分解器] 模型加载失败: {e}")
                print("[问题分解器] 将使用规则方法进行问题分解")
                self.model = None
                self.tokenizer = None
    
    def _extract_entities(self, question: str) -> List[str]:
        """提取问题中的实体和关键词（基础版本）"""
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
    
    def _extract_entities_enhanced(self, question: str) -> Dict[str, List[str]]:
        """增强的实体提取方法，返回按图谱实体类型分类的实体"""
        entity_by_type = {
            "地理场景": [],
            "对象系统": [],
            "时空数据": [],
            "集成模型": [],
            "基础模型": [],
            "系统机理": [],
            "处理方法": [],
            "评价方法": [],
            "总结讨论": []
        }
        
        # 地理场景识别
        basin_pattern = r'([\u4e00-\u9fa5]+(?:流域|区域|地区|盆地|山脉))'
        basins = re.findall(basin_pattern, question)
        entity_by_type["地理场景"].extend(basins)
        
        # 时空数据识别
        data_pattern = r'([\u4e00-\u9fa5]+(?:数据|观测|记录))'
        data_types = re.findall(data_pattern, question)
        entity_by_type["时空数据"].extend(data_types)
        
        # 年份识别 - 作为时空数据的时间属性
        year_pattern = r'(\d{4}-\d{4}年|\d{4}年)'
        years = re.findall(year_pattern, question)
        if years:
            entity_by_type["时空数据"].extend(years)
        
        # 模型识别
        model_pattern = r'([\u4e00-\u9fa5a-zA-Z0-9]+(?:模型|方法|算法))'
        models = re.findall(model_pattern, question)
        
        # 区分基础模型和集成模型
        for model in models:
            if "SWAT" in model.upper() or "耦合" in model or "集成" in model:
                entity_by_type["集成模型"].append(model)
            else:
                entity_by_type["基础模型"].append(model)
        
        # 对象系统识别 - 基于关键词
        system_keywords = ["水资源", "生态", "气候", "地质", "土地", "水文"]
        for keyword in system_keywords:
            if keyword in question:
                entity_by_type["对象系统"].append(f"{keyword}系统")
        
        # 处理方法识别
        process_pattern = r'([\u4e00-\u9fa5]+(?:处理|分析|计算|率定|校准))'
        processes = re.findall(process_pattern, question)
        entity_by_type["处理方法"].extend(processes)
        
        # 评价方法识别
        eval_keywords = ["评价", "验证", "精度", "误差", "敏感性", "不确定性"]
        for keyword in eval_keywords:
            if keyword in question:
                entity_by_type["评价方法"].append(f"{keyword}分析")
        
        # 从域知识库中匹配专业术语
        for category, terms in self.domain_concepts.items():
            entity_type = self._map_category_to_entity_type(category)
            if entity_type and entity_type in entity_by_type:
                for term in terms:
                    if term in question:
                        entity_by_type[entity_type].append(term)
        
        # 去重
        for entity_type in entity_by_type:
            entity_by_type[entity_type] = list(set(entity_by_type[entity_type]))
            
        return entity_by_type
    
    def _map_category_to_entity_type(self, category: str) -> str:
        """将域知识库类别映射到图谱实体类型"""
        mapping = {
            "径流模拟": "系统机理",
            "SWAT模型": "集成模型",
            "基本数据": "时空数据",
            "模型评价": "评价方法",
            "地理场景": "地理场景",
            "对象系统": "对象系统",
            "时空数据": "时空数据",
            "集成模型": "集成模型",
            "基础模型": "基础模型",
            "评价方法": "评价方法"
        }
        return mapping.get(category, None)
    
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
    
    def _get_relation_priority(self, relation: str, question_type: str) -> str:
        """确定关系在当前问题类型下的优先级"""
        # 根据问题类型和关系类型确定优先级
        priority_matrix = {
            "操作步骤型": {
                "研究对象": "高",
                "使用数据": "高",
                "使用模型": "高",
                "所属场景": "中",
                "问题结论": "低"
            },
            "原理解释型": {
                "研究对象": "高",
                "所属场景": "中",
                "使用模型": "高",
                "使用数据": "中",
                "问题结论": "中"
            },
            "概念定义型": {
                "研究对象": "高",
                "所属场景": "中",
                "使用模型": "中",
                "使用数据": "低",
                "问题结论": "低"
            },
            "关系分析型": {
                "研究对象": "高",
                "使用模型": "高",
                "所属场景": "中",
                "使用数据": "中",
                "问题结论": "高"
            },
            "效果优化型": {
                "使用模型": "高",
                "使用数据": "高",
                "问题结论": "高",
                "研究对象": "中",
                "所属场景": "低"
            },
            "比较对比型": {
                "研究对象": "高",
                "使用模型": "高",
                "使用数据": "中",
                "所属场景": "中",
                "问题结论": "高"
            },
            "一般查询型": {
                "研究对象": "高",
                "所属场景": "高",
                "使用数据": "中",
                "使用模型": "中",
                "问题结论": "中"
            }
        }
        
        return priority_matrix.get(question_type, {}).get(relation, "中")
    
    def _should_ask_about_relation(self, question: str, relation: str, question_type: str) -> bool:
        """判断是否应该针对特定关系进行提问"""
        # 基于问题内容和类型确定是否需要提问特定关系
        # 例如，如果问题明确是关于模型的，就一定要问使用模型
        
        # 默认情况，所有关系都提问
        if question_type == "一般查询型":
            return True
            
        # 特定问题类型的特定关系判断
        if "模型" in question and relation == "使用模型":
            return True
            
        if "数据" in question and relation == "使用数据":
            return True
            
        if any(loc in question for loc in ["流域", "区域", "地区"]) and relation == "所属场景":
            return True
            
        # 如果是结论或分析相关，需要问结论
        if any(word in question for word in ["结果", "结论", "影响", "效果"]) and relation == "问题结论":
            return True
            
        # 基于问题类型的默认判断
        relation_by_type = {
            "操作步骤型": ["研究对象", "使用数据", "使用模型"],
            "原理解释型": ["研究对象", "使用模型"],
            "概念定义型": ["研究对象"],
            "关系分析型": ["研究对象", "使用模型", "问题结论"],
            "效果优化型": ["使用模型", "使用数据", "问题结论"],
            "比较对比型": ["研究对象", "使用模型", "问题结论"]
        }
        
        # 检查当前关系是否在该问题类型需要提问的关系列表中
        return relation in relation_by_type.get(question_type, ["研究对象", "使用数据", "使用模型", "所属场景", "问题结论"])
    
   
    def decompose_by_graph_structure(self, question: str) -> List[Dict[str, Any]]:
        """基于图谱结构的问题分解方法"""
        # 检查缓存
        if question in self.question_cache:
            return self.question_cache[question]
            
        # 保存原始问题
        self.original_question = question
        
        # 提取实体和确定问题类型
        entities_by_type = self._extract_entities_enhanced(question)
        question_type = self._classify_question_type(question)
        
        decomposed_questions = []
        question_id = 1
        
        # 根据图谱关系生成子问题
        for relation, entity_type in self.graph_relations.items():
            # 检查问题是否适合分解为该关系对应的子问题
            if self._should_ask_about_relation(question, relation, question_type):
                # 获取问题模板并填充
                template = self.relation_question_templates[relation]
                sub_question = template.replace("这个问题", question)
                
                # 确定优先级
                priority = self._get_relation_priority(relation, question_type)
                
                # 收集该实体类型的所有实体
                entity_list = entities_by_type.get(entity_type, [])
                
                decomposed_questions.append({
                    "id": question_id,
                    "question": sub_question,
                    "entity": entity_list,
                    "priority": priority,
                    "type": relation,
                    "graph_relation": relation,
                    "entity_type": entity_type
                })
                question_id += 1
        
        # 确保至少有3个问题
        if len(decomposed_questions) < 3:
            # 添加通用问题，确保覆盖基本关系
            basic_relations = ["研究对象", "使用数据", "使用模型"]
            for relation in basic_relations:
                if not any(q["graph_relation"] == relation for q in decomposed_questions):
                    entity_type = self.graph_relations[relation]
                    template = self.relation_question_templates[relation]
                    sub_question = template.replace("这个问题", question)
                    
                    decomposed_questions.append({
                        "id": question_id,
                        "question": sub_question,
                        "entity": entities_by_type.get(entity_type, []),
                        "priority": "中",
                        "type": relation,
                        "graph_relation": relation,
                        "entity_type": entity_type
                    })
                    question_id += 1
                    
                    # 如果已有3个问题则停止
                    if len(decomposed_questions) >= 3:
                        break
        
        # 保存到缓存
        self.question_cache[question] = decomposed_questions
        return decomposed_questions
    
    def decompose_by_llm_enhanced(self, question: str) -> List[Dict[str, Any]]:
        """增强的LLM问题分解，整合图谱结构"""
        start_time = time.time()
        print(f"[问题分解器] 开始分解问题: {question[:50]}...")
        
        if self.use_api and hasattr(self, 'llm_client') and self.llm_client:
            try:
                # 构建包含图谱结构的提示词
                prompt = f"""
                【专家身份】你是中国科学院地理科学与资源研究所的地理水文建模首席专家，擅长分解复杂问题并提供系统化建模方案。你的专长是将复杂的地理建模问题分解为可实施的研究步骤，确保研究过程全面、系统且符合学术规范。

                【任务目标】请对以下地理建模问题进行专业化分解，将其转化为结构清晰、逻辑连贯的子问题集合。

                【原始问题】{question}

                【分解要求】
                1. 生成3-5个高质量子问题，每个子问题必须：
                   - 构成完整独立的研究问题，而非原问题的简单复述
                   - 直接以专业而清晰的疑问句形式提出
                   - 与其他子问题共同构成对原问题的全面解答框架

                2. 子问题应系统性地覆盖地理建模的核心维度：
                   - 研究区域特征：研究区域的地理环境、水文气象和社会经济特征
                   - 过程机理分析：需要模拟的关键地理过程和作用机制
                   - 数据需求与处理：所需时空数据类型、来源和预处理方法
                   - 模型选择与应用：适用模型的选择、参数化和集成方法
                   - 结果评估与应用：模型结果的评价标准和实际应用价值

                【示例说明】
                如对"如何利用SWAT模型对太湖流域进行水质模拟"这一问题：
                1. 太湖流域的水文地理特征及其主要污染源分布是什么？（研究区域特征）
                2. SWAT模型如何模拟流域尺度的非点源污染物迁移转化过程？（过程机理分析）
                3. 建立太湖流域SWAT水质模型需要哪些关键数据及其处理方法？（数据需求与处理）
                4. SWAT模型中的水质参数如何进行率定与验证以提高模拟精度？（模型选择与应用）
                5. 如何评价SWAT水质模拟结果并应用于太湖流域污染控制？（结果评估与应用）

                【输出格式】请以JSON格式返回，为每个子问题提供完整的专业信息：
                {{
                "decomposed_questions": [
                    {{
                    "id": 1, 
                    "question": "子问题内容", 
                    "priority": "高/中/低", 
                    "type": "问题科学分类",
                    "graph_relation": "研究区域特征/过程机理分析/数据需求与处理/模型选择与应用/结果评估与应用",
                    "entity_type": "地理场景/过程系统/时空数据/模型方法/应用评估",
                    "suggested_entities": ["相关专业实体1", "相关专业实体2", ...]
                    }},
                    ...
                ]
                }}
                """
                
                print(f"[问题分解器] 发送LLM请求...")
                # 使用LLM客户端进行API调用
                messages = [
                    {"role": "system", "content": "你是中国科学院地理科学与资源研究所的地理水文建模首席专家，擅长分解复杂问题并提供系统化建模方案。你的专长是将复杂的地理建模问题分解为可实施的研究步骤，确保研究过程全面、系统且符合学术规范。"},
                    {"role": "user", "content": prompt}
                ]
                
                content = self.llm_client.chat_completion(messages)
                
                # 检查是否获得了有效响应
                if isinstance(content, str) and "抱歉" in content and "错误" in content:
                    print(f"[问题分解器] LLM调用返回错误: {content[:100]}...")
                    raise Exception(f"LLM API调用失败: {content[:100]}...")
                
                # 提取JSON部分
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                    try:
                        json_data = json.loads(json_content)
                        
                        # 增强结果 - 为每个问题添加实体
                        enhanced_questions = json_data["decomposed_questions"]
                        entities_by_type = self._extract_entities_enhanced(question)
                        
                        for q in enhanced_questions:
                            entity_type = q.get("entity_type")
                            if entity_type and entity_type in entities_by_type:
                                q["entity"] = entities_by_type[entity_type]
                            else:
                                q["entity"] = []
                        
                        end_time = time.time()
                        print(f"[问题分解器] 分解完成，耗时: {end_time - start_time:.2f}秒，生成 {len(enhanced_questions)} 个子问题")
                        return enhanced_questions
                    except json.JSONDecodeError as e:
                        print(f"[问题分解器] JSON解析失败: {e}")
                        print(f"[问题分解器] 尝试提取的JSON内容: {json_content[:200]}...")
                        raise Exception(f"JSON解析失败: {e}")
                
                print(f"[问题分解器] 未能提取有效的JSON响应，回退到图谱结构分解")
                # 如果提取失败，使用图谱结构分解
                return self.decompose_by_graph_structure(question)
                
            except Exception as e:
                print(f"[问题分解器] 增强LLM调用失败: {e}")
                end_time = time.time()
                print(f"[问题分解器] 处理耗时: {end_time - start_time:.2f}秒，使用图谱结构备用方法")
                return self.decompose_by_graph_structure(question)
                
        elif hasattr(self, 'model') and self.model and hasattr(self, 'tokenizer') and self.tokenizer:
            # 使用本地模型与增强提示词
            try:
                # 构建包含图谱结构的提示词
                prompt = f"""
                【专家身份】你是中国科学院地理科学与资源研究所的地理水文建模首席专家，擅长分解复杂问题并提供系统化建模方案。你的专长是将复杂的地理建模问题分解为可实施的研究步骤，确保研究过程全面、系统且符合学术规范。

                【任务目标】请对以下地理建模问题进行专业化分解，将其转化为结构清晰、逻辑连贯的子问题集合。

                【原始问题】{question}

                【分解要求】
                1. 生成3-5个高质量子问题，每个子问题必须：
                   - 构成完整独立的研究问题，而非原问题的简单复述
                   - 直接以专业而清晰的疑问句形式提出
                   - 与其他子问题共同构成对原问题的全面解答框架

                2. 子问题应系统性地覆盖地理建模的核心维度：
                   - 研究区域特征：研究区域的地理环境、水文气象和社会经济特征
                   - 过程机理分析：需要模拟的关键地理过程和作用机制
                   - 数据需求与处理：所需时空数据类型、来源和预处理方法
                   - 模型选择与应用：适用模型的选择、参数化和集成方法
                   - 结果评估与应用：模型结果的评价标准和实际应用价值

                【示例说明】
                如对"如何利用SWAT模型对太湖流域进行水质模拟"这一问题：
                1. 太湖流域的水文地理特征及其主要污染源分布是什么？（研究区域特征）
                2. SWAT模型如何模拟流域尺度的非点源污染物迁移转化过程？（过程机理分析）
                3. 建立太湖流域SWAT水质模型需要哪些关键数据及其处理方法？（数据需求与处理）
                4. SWAT模型中的水质参数如何进行率定与验证以提高模拟精度？（模型选择与应用）
                5. 如何评价SWAT水质模拟结果并应用于太湖流域污染控制？（结果评估与应用）

                【输出格式】请以JSON格式返回，为每个子问题提供完整的专业信息：
                {{
                "decomposed_questions": [
                    {{
                    "id": 1, 
                    "question": "子问题内容", 
                    "priority": "高/中/低", 
                    "type": "问题科学分类",
                    "graph_relation": "研究区域特征/过程机理分析/数据需求与处理/模型选择与应用/结果评估与应用",
                    "entity_type": "地理场景/过程系统/时空数据/模型方法/应用评估",
                    "suggested_entities": ["相关专业实体1", "相关专业实体2", ...]
                    }},
                    ...
                ]
                }}
                """
                                
                # 模型推理
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=800,
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
                    
                    # 增强结果 - 为每个问题添加实体
                    enhanced_questions = json_data["decomposed_questions"]
                    entities_by_type = self._extract_entities_enhanced(question)
                    
                    for q in enhanced_questions:
                        entity_type = q.get("entity_type")
                        if entity_type and entity_type in entities_by_type:
                            q["entity"] = entities_by_type[entity_type]
                        else:
                            q["entity"] = []
                    
                    end_time = time.time()
                    print(f"[问题分解器] 本地模型分解完成，耗时: {end_time - start_time:.2f}秒")
                    return enhanced_questions
                
                # 如果提取失败，使用图谱结构分解
                return self.decompose_by_graph_structure(question)
                
            except Exception as e:
                print(f"[问题分解器] 本地增强模型推理失败: {e}")
                end_time = time.time()
                print(f"[问题分解器] 处理耗时: {end_time - start_time:.2f}秒，使用备用方法")
                return self.decompose_by_graph_structure(question)
        else:
            # 没有可用的模型，使用图谱结构分解方法
            print("[问题分解器] 没有可用的API或本地模型，使用图谱结构分解方法")
            end_time = time.time()
            print(f"[问题分解器] 决策耗时: {end_time - start_time:.2f}秒")
            return self.decompose_by_graph_structure(question)
    
    def decompose_question(self, question: str, use_llm: bool = True, use_graph_structure: bool = True) -> List[Dict[str, Any]]:
        """
        分解问题的主函数，支持多种分解方法
        
        Args:
            question: 用户问题
            use_llm: 是否使用大模型方法
            use_graph_structure: 是否使用图谱结构进行分解
        
        Returns:
            分解后的子问题列表
        """
        # 检查缓存
        if question in self.question_cache:
            return self.question_cache[question]
        
        # 保存原始问题
        self.original_question = question
            
        # 根据参数选择分解方法
        if use_graph_structure:
            if use_llm and (self.use_api or (hasattr(self, 'model') and self.model and hasattr(self, 'tokenizer') and self.tokenizer)):
                result = self.decompose_by_llm_enhanced(question)
            else:
                result = self.decompose_by_graph_structure(question)
        else:
            if use_llm and (self.use_api or (hasattr(self, 'model') and self.model and hasattr(self, 'tokenizer') and self.tokenizer)):
                result = self.decompose_by_llm_enhanced(question)
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
    
    def format_questions_for_kg(self, decomposed_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将分解后的问题格式化为符合知识图谱结构的格式
        
        Args:
            decomposed_questions: 分解后的子问题列表
            
        Returns:
            符合图谱结构的字典
        """
        # 按照图谱关系组织子问题
        kg_structured_questions = {
            "type": "地理问题",
            "name": self.original_question,
            "desc": self.original_question,
            "source_article": ""
        }
        
        # 将子问题按照关系组织
        for relation in self.graph_relations.keys():
            relation_questions = [q for q in decomposed_questions if q.get("graph_relation") == relation]
            if relation_questions:
                entity_type = self.graph_relations[relation]
                kg_structured_questions[relation] = []
                
                for q in relation_questions:
                    # 准备实体数据
                    entity_data = {
                        "type": entity_type,
                        "name": q["question"],
                        "desc": q["question"]
                    }
                    
                    # 为特定实体类型添加子关系
                    if entity_type == "对象系统" and "系统机理" in q:
                        entity_data["相关机理"] = [
                            {
                                "type": "系统机理",
                                "name": mechanism,
                                "desc": mechanism
                            }
                            for mechanism in q.get("系统机理", [])
                        ]
                    
                    elif entity_type == "时空数据" and "数据来源" in q:
                        entity_data["获取来源"] = [
                            {
                                "type": "数据来源",
                                "name": source,
                                "desc": source
                            }
                            for source in q.get("数据来源", [])
                        ]
                        
                        if "处理方法" in q:
                            entity_data["数据处理"] = [
                                {
                                    "type": "处理方法",
                                    "name": method,
                                    "desc": method
                                }
                                for method in q.get("处理方法", [])
                            ]
                    
                    elif entity_type == "集成模型" and "基础模型" in q:
                        entity_data["集成依赖"] = [
                            {
                                "type": "基础模型",
                                "name": model,
                                "desc": model
                            }
                            for model in q.get("基础模型", [])
                        ]
                    
                    # 添加到关系列表
                    kg_structured_questions[relation].append(entity_data)
        
        return kg_structured_questions

# 使用示例
if __name__ == "__main__":
    # 初始化问题分解器
    decomposer = HydrologicalQuestionDecomposer(use_api=False)
    
    # 测试问题
    test_question = "我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?"
    
    # 分解问题 - 使用图谱结构
    decomposed_questions = decomposer.decompose_question(test_question, use_llm=False, use_graph_structure=True)
    
    # 打印结果
    print(f"原始问题: {test_question}\n")
    print("分解后的子问题:")
    for q in decomposed_questions:
        relation = q.get('graph_relation', q.get('type', '未知'))
        print(f"[{q['priority']}] {q['id']}. {q['question']} (关系: {relation})")
    
    # 获取格式化的问答序列
    qa_sequence = decomposer.format_questions_for_qa(decomposed_questions)
    
    print("\n问答序列:")
    for i, q in enumerate(qa_sequence, 1):
        print(f"{i}. {q}")
        
    # 获取符合图谱结构的格式
    kg_format = decomposer.format_questions_for_kg(decomposed_questions)
    
    print("\n知识图谱格式:")
    print(json.dumps(kg_format, ensure_ascii=False, indent=2))