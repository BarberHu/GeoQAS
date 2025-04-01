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
                "system": "你是中国水利水电科学研究院的资深研究员，专注于地理水文建模领域25年，尤其精通SWAT模型应用、径流模拟与流域管理。你擅长将复杂的水文过程和建模方法转化为清晰、实用的解决方案，帮助研究者和工程师解决实际问题。请基于提供的专业知识，为用户提供科学准确、操作性强的解答。",
                "context_prefix": "考虑以下对话背景与知识链接：",
                "entity_prefix": "相关专业概念与实体：",
                "reference_prefix": "科学参考依据："
            },
            "query_generation": {
                "system": "你是一个Cypher查询生成专家，请根据给定的内容，生成Neo4j的Cypher查询语句。，专注于地理信息系统中的知识图谱与图数据库应用。作为Neo4j认证专家，你擅长设计高效的Cypher查询来检索复杂的地理建模知识。请根据提供的信息，精准构建适用于地理建模知识图谱的查询语句。",
                "query_prefix": "需要查询的专业问题：",
                "entity_prefix": "相关地理建模实体：",
                "response_format": "请生成精确的Cypher查询语句，不要包含任何解释。查询应当针对地理建模知识网络优化。"
            },
            "explanation": {
                "system": "你是武汉大学水利水电学院的教授，专注于地理水文建模研究20年，在国际顶级期刊发表论文100余篇。你擅长将复杂的水文概念转化为简明易懂的解释，既保留专业深度又确保理解清晰。请基于严谨的科学知识，为以下概念提供权威解释。",
                "detail_level": "请提供多层次解析：(1)基础定义 (2)核心原理 (3)应用场景 (4)与其他概念的关联。解释应兼顾专业准确性与教学清晰度，适合研究生学习参考。"
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
    
    # 在TextAssembler类中添加新方法
def format_kg_results_for_context(self, question: str, kg_results: List[Dict], relevance_threshold: float = 0.3) -> str:
    """
    将知识图谱查询结果转换为结构化上下文文档
    
    Args:
        question: 用户问题或子问题
        kg_results: 知识图谱查询结果列表
        relevance_threshold: 相关性阈值，低于此值的结果将被排除
        
    Returns:
        结构化的上下文文档
    """
    if not kg_results:
        return "未找到相关知识图谱信息。"
    
    # 排序结果，相关性高的排在前面
    sorted_results = sorted(kg_results, key=lambda x: x.get('relevance', 0), reverse=True)
    
    # 过滤低相关性结果
    filtered_results = [r for r in sorted_results if r.get('relevance', 0) >= relevance_threshold]
    
    # 格式化为结构化文档
    context = ["### 知识图谱信息"]
    
    # 实体信息部分
    entity_info = []
    for i, result in enumerate(filtered_results[:5], 1):  # 最多使用前5个结果
        if 'entity' in result:
            entity_text = f"实体{i}: {result['entity']}\n"
            if 'description' in result:
                entity_text += f"描述: {result['description']}\n"
            if 'properties' in result and isinstance(result['properties'], dict):
                props = result['properties']
                entity_text += "属性:\n"
                for key, value in props.items():
                    if key not in ['name', 'description'] and value:
                        entity_text += f"- {key}: {value}\n"
            entity_info.append(entity_text)
    
    if entity_info:
        context.append("## 相关实体信息")
        context.extend(entity_info)
    
    # 关系信息部分
    relation_info = []
    for i, result in enumerate(filtered_results[:5], 1):
        if 'relations' in result and result['relations']:
            rel_text = f"关系{i}:\n"
            for j, rel in enumerate(result['relations'][:3], 1):  # 每个实体最多展示3个关系
                if 'source' in rel and 'target' in rel and 'relation' in rel:
                    rel_text += f"- {rel['source']} → {rel['relation']} → {rel['target']}\n"
            relation_info.append(rel_text)
    
    if relation_info:
        context.append("## 相关关系信息")
        context.extend(relation_info)
    
    # 文献引用部分
    references = []
    for result in filtered_results:
        if 'source_article' in result and result['source_article']:
            references.append(result['source_article'])
    
    if references:
        context.append("## 参考文献")
        for i, ref in enumerate(list(set(references))[:3], 1):  # 去重并限制数量
            context.append(f"[{i}] {ref}")
    
    return "\n\n".join(context)

# 添加一个新方法，获取不同维度的知识图谱信息
def get_comprehensive_kg_context(self, question: str, entity_names: List[str] = None) -> Dict[str, Any]:
    """
    获取全面的知识图谱上下文信息，包括实体、关系和辅助信息
    
    Args:
        question: 用户问题
        entity_names: 已知的实体名称列表（可选）
        
    Returns:
        包含不同维度知识图谱信息的字典
    """
    # 如果没有提供实体，从问题中提取
    if not entity_names:
        entity_names = self.entity_recognizer.recognize(question)
    
    # 并行获取实体详情
    entity_details = self._get_entity_details_batch(entity_names)
    
    # 获取实体之间的关系
    relations = []
    if len(entity_names) > 1:
        with self.driver.session() as session:
            query = """
            MATCH (n1)-[r]-(n2)
            WHERE n1.name IN $entity_names AND n2.name IN $entity_names
            RETURN n1.name as source, type(r) as relation, n2.name as target
            """
            result = session.run(query, entity_names=entity_names)
            for record in result:
                relations.append({
                    'source': record['source'],
                    'relation': record['relation'],
                    'target': record['target']
                })
    
    # 获取扩展关系（一阶扩展）
    extended_relations = []
    if entity_names:
        with self.driver.session() as session:
            query = """
            MATCH (n1)-[r]-(n2)
            WHERE n1.name IN $entity_names AND NOT n2.name IN $entity_names
            RETURN n1.name as source, type(r) as relation, n2.name as target
            LIMIT 10
            """
            result = session.run(query, entity_names=entity_names)
            for record in result:
                extended_relations.append({
                    'source': record['source'],
                    'relation': record['relation'],
                    'target': record['target']
                })
    
    # 构建知识图谱可视化数据
    nodes = []
    links = []
    
    # 添加主实体节点
    for name in entity_names:
        if name in entity_details:
            nodes.append({
                'id': name,
                'name': name,
                'category': entity_details[name].get('properties', {}).get('category', '概念'),
                'value': 2  # 主要实体权重更高
            })
    
    # 添加扩展实体节点
    for rel in extended_relations:
        if rel['target'] not in [node['id'] for node in nodes]:
            nodes.append({
                'id': rel['target'],
                'name': rel['target'],
                'category': '相关概念',
                'value': 1
            })
    
    # 添加关系连接
    for rel in relations + extended_relations:
        links.append({
            'source': rel['source'],
            'target': rel['target'],
            'name': rel['relation']
        })
    
    # 返回多维度知识图谱信息
    return {
        'entity_details': entity_details,
        'direct_relations': relations,
        'extended_relations': extended_relations,
        'visualization': {
            'nodes': nodes,
            'links': links
        }
    }