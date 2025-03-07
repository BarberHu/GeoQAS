import json
import concurrent.futures
from typing import List, Dict, Any, Optional, Union
from neo4j import GraphDatabase

class TextAssembler:
    """
    文本拼接模块 - 整合实体识别和实体链接结果，生成结构化文本
    用于后续查询和问答的输入
    """
    
    def __init__(self, neo4j_config: Dict[str, str], entity_recognizer, entity_linker):
        """
        初始化文本拼接模块
        
        Args:
            neo4j_config: Neo4j数据库配置
            entity_recognizer: 实体识别器实例
            entity_linker: 实体链接器实例
        """
        # 连接知识库
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"])
        )
        
        # 实体识别器和链接器
        self.entity_recognizer = entity_recognizer
        self.entity_linker = entity_linker
        
        # 初始化上下文
        self.context_history = []
        
        # 实体详情缓存
        self.entity_details_cache = {}
        
        # 模板配置
        self.templates = {
            "default": {
                "system": "你是一个水文领域专家，专注于SWAT模型和径流模拟相关问题的解答。请基于提供的知识信息回答问题。",
                "context_prefix": "基于之前的对话内容：",
                "entity_prefix": "相关实体信息：",
                "reference_prefix": "参考文献："
            },
            "query_generation": {
                "system": "你是一个Cypher查询生成专家，请根据给定的内容，生成Neo4j的Cypher查询语句。",
                "query_prefix": "需要查询的问题：",
                "entity_prefix": "相关实体信息：",
                "response_format": "请生成精确的Cypher查询语句，不要包含任何解释。"
            },
            "explanation": {
                "system": "你是一个水文领域专家，擅长解释SWAT模型和径流模拟的复杂概念。请详细解释以下问题：",
                "detail_level": "请提供详细且通俗易懂的解释，包括必要的专业术语解释和实际应用示例。"
            }
        }
    
    def _get_entity_details(self, entity_name: str) -> Dict[str, Any]:
        """获取实体详细信息"""
        # 检查缓存
        if entity_name in self.entity_details_cache:
            return self.entity_details_cache[entity_name]
            
        with self.driver.session() as session:
            # 1. 获取实体属性
            property_query = """
            MATCH (n) 
            WHERE n.name = $entity_name 
            RETURN properties(n) as props
            """
            
            # 2. 获取关系信息（修改后的查询）
            relation_query = """
            // 出边关系
            MATCH (n)-[r]->(m)
            WHERE n.name = $entity_name
            RETURN 
                n.name as start_node,
                type(r) as relation_type,
                m.name as end_node,
                'outgoing' as direction
            UNION
            // 入边关系
            MATCH (m)-[r]->(n)
            WHERE n.name = $entity_name
            RETURN 
                m.name as start_node,
                type(r) as relation_type,
                n.name as end_node,
                'incoming' as direction
            """
            
            try:
                # 获取属性
                result = session.run(property_query, entity_name=entity_name)
                record = result.single()
                properties = dict(record["props"]) if record and record["props"] else {}
                
                # 获取关系
                result = session.run(relation_query, entity_name=entity_name)
                relations = []
                for record in result:
                    relations.append({
                        "source": record["start_node"],
                        "relation": record["relation_type"],
                        "target": record["end_node"],
                        "direction": record["direction"]
                    })
                
                entity_details = {
                    "name": entity_name,
                    "properties": properties,
                    "relations": relations
                }
                
                # 添加到缓存
                self.entity_details_cache[entity_name] = entity_details
                
                return entity_details
                
            except Exception as e:
                print(f"获取实体 {entity_name} 详细信息时出错: {str(e)}")
                return {
                    "name": entity_name,
                    "properties": {},
                    "relations": []
                }
    
    def _get_entity_details_batch(self, entity_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量获取多个实体的详细信息
        
        Args:
            entity_names: 实体名称列表
            
        Returns:
            Dict[str, Dict[str, Any]]: 以实体名为键，详细信息为值的字典
        """
        results = {}
        uncached_entities = []
        
        # 检查缓存
        for name in entity_names:
            if name in self.entity_details_cache:
                results[name] = self.entity_details_cache[name]
            else:
                uncached_entities.append(name)
        
        # 如果所有实体都已缓存，直接返回
        if not uncached_entities:
            return results
            
        with self.driver.session() as session:
            # 批量查询实体属性
            property_query = """
            MATCH (n) 
            WHERE n.name IN $entity_names 
            RETURN n.name as name, properties(n) as props
            """
            
            # 批量查询关系信息
            relation_query = """
            // 出边关系
            MATCH (n)-[r]->(m)
            WHERE n.name IN $entity_names
            RETURN 
                n.name as entity_name,
                n.name as start_node,
                type(r) as relation_type,
                m.name as end_node,
                'outgoing' as direction
            UNION
            // 入边关系
            MATCH (m)-[r]->(n)
            WHERE n.name IN $entity_names
            RETURN 
                n.name as entity_name,
                m.name as start_node,
                type(r) as relation_type,
                n.name as end_node,
                'incoming' as direction
            """
            
            try:
                # 获取属性
                property_result = session.run(property_query, entity_names=uncached_entities)
                properties_by_name = {}
                for record in property_result:
                    name = record["name"]
                    properties_by_name[name] = dict(record["props"]) if record["props"] else {}
                    # 初始化结果
                for name in uncached_entities:
                    results[name] = {
                        "name": name,
                        "properties": properties_by_name.get(name, {}),
                        "relations": []
                    }
                
                # 获取关系
                relation_result = session.run(relation_query, entity_names=uncached_entities)
                for record in relation_result:
                    entity_name = record["entity_name"]
                    if entity_name in results:
                        results[entity_name]["relations"].append({
                            "source": record["start_node"],
                            "relation": record["relation_type"],
                            "target": record["end_node"],
                            "direction": record["direction"]
                        })
                
                # 更新缓存
                for name, details in results.items():
                    if name not in self.entity_details_cache:
                        self.entity_details_cache[name] = details
                
                return results
                
            except Exception as e:
                print(f"批量获取实体详细信息时出错: {str(e)}")
                # 回退到单个处理
                for name in uncached_entities:
                    results[name] = self._get_entity_details(name)
                return results
    
    def assemble_text(self, question: str, template_type: str = "default", 
                     include_context: bool = True, max_entities: int = 5) -> str:
        """
        组装文本，生成结构化输入
        
        Args:
            question: 用户问题
            template_type: 模板类型
            include_context: 是否包含上下文历史
            max_entities: 最大实体数量
            
        Returns:
            结构化文本
        """
        # 1. 提取实体
        entities = self.entity_recognizer.recognize(question)
        
        # 2. 实体链接及排序
        linked_entities = self.entity_linker.rank_entities(query=question, mention=entities)
        
        # 3. 获取实体详细信息 - 使用批量处理
        top_entities = []
        for entity_name, entity_matches in linked_entities.items():
            if entity_matches:  # 确保有匹配结果
                top_entity = entity_matches[0]["entity"]
                top_entities.append(top_entity)
                
                # 限制实体数量
                if len(top_entities) >= max_entities:
                    break
        
        # 批量获取实体详情
        entity_details_map = self._get_entity_details_batch(top_entities)
        entity_details = [entity_details_map[name] for name in top_entities if name in entity_details_map]
        
        # 4. 组装文本
        template = self.templates.get(template_type, self.templates["default"])
        assembled_text = [template["system"]]
        
        # 添加上下文历史
        if include_context and self.context_history:
            context_text = template.get("context_prefix", "基于之前的对话内容：") + "\n"
            for entry in self.context_history[-3:]:  # 仅包含最近3轮对话
                context_text += f"问：{entry['question']}\n答：{entry['answer']}\n"
            assembled_text.append(context_text)
        
        # 添加当前问题
        if template_type == "query_generation":
            assembled_text.append(f"{template.get('query_prefix', '')} {question}")
        else:
            assembled_text.append(f"问题：{question}")
        
        # 添加实体信息
        if entity_details:
            entity_text = template.get("entity_prefix", "相关实体信息：") + "\n"
            for i, entity in enumerate(entity_details, 1):
                entity_text += f"[{i}] 名称: {entity['name']}\n"
                
                # 添加描述
                if "desc" in entity["properties"]:
                    entity_text += f"描述: {entity['properties']['desc']}\n"
                
                # 添加关系
                if entity["relations"]:
                    entity_text += "相关概念:\n"
                    for rel in entity["relations"][:5]:  # 限制关系数量
                        if "direction" in rel:
                            direction = "→" if rel["direction"] == "outgoing" else "←"
                            if "target" in rel:
                                entity_text += f"- {entity['name']} {direction} {rel['relation']} {direction} {rel['target']}\n"
                            else:
                                entity_text += f"- {rel['source']} {direction} {rel['relation']} {direction} {entity['name']}\n"
                
                # 分隔不同实体
                entity_text += "\n"
            
            assembled_text.append(entity_text)
        
        # 添加参考文献
        all_references = []
        for entity in entity_details:
            if "source_article" in entity["properties"]:
                all_references.append(entity["properties"]["source_article"])
        
        if all_references:
            reference_text = template.get("reference_prefix", "参考文献：") + "\n"
            for i, ref in enumerate(all_references[:5], 1):  # 限制参考文献数量
                reference_text += f"[{i}] {ref}\n"
            assembled_text.append(reference_text)
        
        # 添加特定模板的额外内容
        if template_type == "query_generation":
            assembled_text.append(template.get("response_format", ""))
        elif template_type == "explanation":
            assembled_text.append(template.get("detail_level", ""))
        
        # 返回组装好的文本
        return "\n\n".join(assembled_text)
    
    def update_context(self, question: str, answer: str):
        """
        更新上下文历史
        
        Args:
            question: 用户问题
            answer: 系统回答
        """
        self.context_history.append({
            "question": question,
            "answer": answer
        })
        
        # 保持历史记录不超过10轮
        if len(self.context_history) > 10:
            self.context_history = self.context_history[-10:]
    
    def clear_context(self):
        """清空上下文历史"""
        self.context_history = []
    
    def add_template(self, template_name: str, template_config: Dict[str, str]):
        """
        添加新的模板
        
        Args:
            template_name: 模板名称
            template_config: 模板配置
        """
        self.templates[template_name] = template_config
    
    def get_cypher_query(self, question: str) -> str:
        """
        生成Cypher查询语句的专用方法
        
        Args:
            question: 用户问题
            
        Returns:
            用于生成Cypher查询的文本
        """
        return self.assemble_text(question, template_type="query_generation", include_context=False)
    
    def get_explanation(self, question: str) -> str:
        """
        生成概念解释的专用方法
        
        Args:
            question: 用户问题
            
        Returns:
            用于生成概念解释的文本
        """
        return self.assemble_text(question, template_type="explanation", include_context=True)