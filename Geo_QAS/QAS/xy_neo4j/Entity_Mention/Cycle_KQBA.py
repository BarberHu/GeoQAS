
import json
import time
from typing import List, Dict, Any, Optional
import re
from neo4j import GraphDatabase

class MultiTurnQAManager:
    """
    水文领域多轮问答管理器 - 集成问题分解和知识图谱查询
    """
    
    def __init__(self, 
                 neo4j_config: Dict[str, str],
                 entity_linker,
                 llm_service=None):
        """
        初始化多轮问答管理器
        
        Args:
            neo4j_config: Neo4j数据库配置
            entity_linker: 实体链接器对象
            llm_service: LLM服务接口(可选)
        """
        # 初始化Neo4j连接
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"])
        )
        
        # 实体链接器
        self.entity_linker = entity_linker
        
        # LLM服务
        self.llm_service = llm_service
        
        # 初始化上下文管理
        self.conversation_context = {
            "history": [],           # 历史问答对
            "entities": {},          # 已识别的实体
            "attributes": {},        # 已识别的属性
            "knowledge_triples": [], # 已检索的知识三元组
            "answered_questions": [] # 已回答的问题
        }
        
        # 从问题分解模块导入
        from question_decomposition_module import HydrologicalQuestionDecomposer
        self.question_decomposer = HydrologicalQuestionDecomposer(use_api=True)
    
    def reset_context(self):
        """重置对话上下文"""
        self.conversation_context = {
            "history": [],
            "entities": {},
            "attributes": {},
            "knowledge_triples": [],
            "answered_questions": []
        }
    
    def _add_to_context(self, question: str, answer: str, entities: Dict[str, List] = None):
        """
        添加问答对和实体到上下文
        
        Args:
            question: 用户问题
            answer: 系统回答
            entities: 新识别的实体字典
        """
        # 添加问答对
        self.conversation_context["history"].append({
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        
        # 添加实体
        if entities:
            for entity_name, entity_info in entities.items():
                if entity_name not in self.conversation_context["entities"]:
                    self.conversation_context["entities"][entity_name] = entity_info
        
        # 添加已回答问题
        self.conversation_context["answered_questions"].append(question)
    
    def _query_knowledge_graph(self, entity_names: List[str]) -> List[Dict]:
        """
        查询知识图谱获取实体相关信息
        
        Args:
            entity_names: 实体名称列表
            
        Returns:
            实体相关三元组列表
        """
        triples = []
        
        with self.driver.session() as session:
            for entity_name in entity_names:
                # 查询实体作为主语的三元组
                subject_query = """
                MATCH (s {name: $entity_name})-[r]->(o)
                RETURN s.name as subject, type(r) as predicate, o.name as object, 
                       labels(s) as subject_type, labels(o) as object_type
                LIMIT 10
                """
                
                # 查询实体作为宾语的三元组
                object_query = """
                MATCH (s)-[r]->(o {name: $entity_name})
                RETURN s.name as subject, type(r) as predicate, o.name as object,
                       labels(s) as subject_type, labels(o) as object_type
                LIMIT 10
                """
                
                # 执行查询
                subject_results = session.run(subject_query, entity_name=entity_name)
                object_results = session.run(object_query, entity_name=entity_name)
                
                # 处理结果
                for record in list(subject_results) + list(object_results):
                    triple = {
                        "subject": record["subject"],
                        "predicate": record["predicate"],
                        "object": record["object"],
                        "subject_type": record["subject_type"][0] if record["subject_type"] else None,
                        "object_type": record["object_type"][0] if record["object_type"] else None
                    }
                    
                    if triple not in triples:
                        triples.append(triple)
                        # 添加到上下文的知识三元组
                        if triple not in self.conversation_context["knowledge_triples"]:
                            self.conversation_context["knowledge_triples"].append(triple)
        
        return triples
    
    def _extract_key_entities(self, question: str) -> List[str]:
        """
        从问题中提取关键实体
        
        Args:
            question: 用户问题
            
        Returns:
            关键实体列表
        """
        # 使用实体链接器识别实体
        mentions = self._identify_mentions(question)
        
        if not mentions:
            # 简单的规则匹配
            mentions = [
                m.group() for m in re.finditer(
                    r'(SWAT|径流模拟|淮河流域|土壤数据|气象数据|水文|水量)',
                    question,
                    re.IGNORECASE
                )
            ]
        
        entities = {}
        for mention in mentions:
            linked_entities = self.entity_linker.rank_entities(question, mention)
            if linked_entities:
                entities[mention] = linked_entities
        
        # 将识别的实体添加到上下文
        for entity_name, entity_info in entities.items():
            if entity_name not in self.conversation_context["entities"]:
                self.conversation_context["entities"][entity_name] = entity_info
        
        return list(entities.keys())
    
    def _identify_mentions(self, text: str) -> List[str]:
        """
        识别文本中的实体提及
        
        Args:
            text: 输入文本
            
        Returns:
            实体提及列表
        """
        # 简单的规则识别方法
        mentions = []
        
        # 流域名称
        basin_patterns = r'([\u4e00-\u9fa5]+流域)'
        mentions.extend(re.findall(basin_patterns, text))
        
        # 数据类型
        data_patterns = r'([\u4e00-\u9fa5]+(数据|资料))'
        mentions.extend(re.findall(data_patterns, text))
        
        # 模型名称
        model_patterns = r'(SWAT模型|SWAT|径流模拟)'
        mentions.extend(re.findall(model_patterns, text, re.IGNORECASE))
        
        # 水文术语
        hydro_patterns = r'(径流|水文|降雨|DEM|土地利用|土壤类型|蒸散发|地下水)'
        mentions.extend(re.findall(hydro_patterns, text))
        
        # 去重
        return list(set(mentions))
    
    def _get_relevant_properties(self, entity_name: str) -> List[str]:
        """
        获取实体的相关属性
        
        Args:
            entity_name: 实体名称
            
        Returns:
            属性列表
        """
        properties = []
        
        with self.driver.session() as session:
            query = """
            MATCH (e {name: $entity_name})
            RETURN properties(e) as props
            """
            
            result = session.run(query, entity_name=entity_name)
            record = result.single()
            
            if record and record["props"]:
                # 移除name属性
                props = dict(record["props"])
                if "name" in props:
                    del props["name"]
                    
                # 转换成结构化属性列表
                for key, value in props.items():
                    properties.append({
                        "name": key,
                        "value": value
                    })
        
        return properties
    
    def _enrich_context_with_knowledge(self, question: str):
        """
        用知识图谱中的信息丰富上下文
        
        Args:
            question: 用户问题
        """
        # 提取实体
        entities = self._extract_key_entities(question)
        
        # 查询知识图谱
        if entities:
            triples = self._query_knowledge_graph(entities)
            
            # 获取实体属性
            for entity in entities:
                properties = self._get_relevant_properties(entity)
                if properties:
                    self.conversation_context["attributes"][entity] = properties
    
    def _format_context_for_llm(self) -> str:
        """
        将上下文格式化为LLM输入
        
        Returns:
            格式化后的上下文字符串
        """
        context = []
        
        # 添加历史问答
        if self.conversation_context["history"]:
            context.append("历史对话:")
            for i, qa in enumerate(self.conversation_context["history"][-3:], 1):  # 只取最近3轮
                context.append(f"问题{i}: {qa['question']}")
                context.append(f"回答{i}: {qa['answer']}\n")
        
        # 添加实体信息
        if self.conversation_context["entities"]:
            context.append("相关实体:")
            for entity, info in self.conversation_context["entities"].items():
                context.append(f"- {entity}")
        
        # 添加知识三元组
        if self.conversation_context["knowledge_triples"]:
            context.append("\n相关知识:")
            # 只取最相关的前10个三元组
            for triple in self.conversation_context["knowledge_triples"][-10:]:
                context.append(
                    f"- {triple['subject']} ({triple['subject_type']}) "
                    f"{triple['predicate']} "
                    f"{triple['object']} ({triple['object_type']})"
                )
        
        return "\n".join(context)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        生成答案
        
        Args:
            question: 用户问题
            context: 上下文
            
        Returns:
            生成的答案
        """
        if self.llm_service:
            # 使用外部LLM服务
            return self.llm_service.get_answer(question, context)
        else:
            # 默认返回"
            return f"对于问题'{question}'，基于已有的上下文和知识图谱"