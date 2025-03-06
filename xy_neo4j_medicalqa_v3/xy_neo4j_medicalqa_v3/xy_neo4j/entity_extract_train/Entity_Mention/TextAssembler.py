import json
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
                
                return {
                    "name": entity_name,
                    "properties": properties,
                    "relations": relations
                }
                
            except Exception as e:
                print(f"获取实体 {entity_name} 详细信息时出错: {str(e)}")
                return {
                    "name": entity_name,
                    "properties": {},
                    "relations": []
                }
    
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
        linked_entities = self.entity_linker.rank_entities(question, entities)
        
        # 3. 获取实体详细信息
        entity_details = []
        for entity_name, entity_matches in linked_entities.items():
            if entity_matches:  # 确保有匹配结果
                # 获取排名最高的实体
                top_entity = entity_matches[0]["entity"]
                entity_details.append(self._get_entity_details(top_entity))
                
                # 限制实体数量
                if len(entity_details) >= max_entities:
                    break
        
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
                        direction = "→" if rel["direction"] == "outgoing" else "←"
                        entity_text += f"- {entity['name']} {direction} {rel['relation']} {direction} {rel['target']}\n"
                
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


# --------------------- 集成示例 ---------------------
# class IntegratedQASystem:
#     """
#     集成问答系统演示类 - 整合实体识别、实体链接、文本拼接和问答
#     """
    
#     def __init__(self, neo4j_config: Dict[str, str], llm_api_key: str):
#         """
#         初始化集成问答系统
        
#         Args:
#             neo4j_config: Neo4j数据库配置
#             llm_api_key: 大模型API密钥
#         """
#         # 初始化实体识别器
#         from LLM_mention_final import DirectMentionRecognizer
#         self.entity_recognizer = DirectMentionRecognizer(api_key=llm_api_key)
        
#         # 初始化实体链接器
#         from Entity_Order import EntityLinker
#         self.entity_linker = EntityLinker(neo4j_config=neo4j_config)
        
#         # 初始化文本拼接器
#         self.text_assembler = TextAssembler(
#             neo4j_config=neo4j_config,
#             entity_recognizer=self.entity_recognizer,
#             entity_linker=self.entity_linker
#         )
        
#         # 初始化LLM客户端
#         from openai import OpenAI
#         self.llm_client = OpenAI(
#             api_key=llm_api_key,
#             base_url="https://api.chatanywhere.tech/v1"
#         )
        
#         # 初始化问题分解器(optional)
#         try:
#             from question_decomposition_module import HydrologicalQuestionDecomposer
#             self.question_decomposer = HydrologicalQuestionDecomposer(use_api=False)
#         except ImportError:
#             self.question_decomposer = None
    
#     def answer_question(self, question: str, use_decomposition: bool = True) -> str:
#         """
#         回答问题的主方法
        
#         Args:
#             question: 用户问题
#             use_decomposition: 是否使用问题分解
            
#         Returns:
#             问题的回答
#         """
#         # 如果启用了问题分解且问题分解器可用
#         if use_decomposition and self.question_decomposer:
#             return self._answer_with_decomposition(question)
#         else:
#             return self._answer_single_question(question)
    
#     def _answer_single_question(self, question: str) -> str:
#         """
#         回答单个问题
        
#         Args:
#             question: 用户问题
            
#         Returns:
#             问题的回答
#         """
#         # 组装文本
#         prompt = self.text_assembler.assemble_text(question)
        
#         # 调用LLM
#         response = self.llm_client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=1000
#         )
        
#         answer = response.choices[0].message.content
        
#         # 更新上下文
#         self.text_assembler.update_context(question, answer)
        
#         return answer
    
#     def _answer_with_decomposition(self, question: str) -> str:
#         """
#         使用问题分解回答复杂问题
        
#         Args:
#             question: 用户问题
            
#         Returns:
#             整合后的回答
#         """
#         # 分解问题
#         decomposed_questions = self.question_decomposer.decompose_question(question)
#         formatted_questions = self.question_decomposer.format_questions_for_qa(decomposed_questions)
        
#         # 分别回答子问题
#         sub_answers = []
#         for sub_question in formatted_questions:
#             sub_answer = self._answer_single_question(sub_question)
#             sub_answers.append({"question": sub_question, "answer": sub_answer})
        
#         # 整合所有回答
#         integration_prompt = f"""
#         基于以下对复杂问题"{question}"的分解问答，提供一个整合的完整回答：
        
#         {"".join([f'问题：{qa["question"]}\n回答：{qa["answer"]}\n\n' for qa in sub_answers])}
        
#         请提供一个连贯、全面的回答，避免重复信息，并确保覆盖所有关键点。
#         """
        
#         response = self.llm_client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": integration_prompt}],
#             temperature=0.3,
#             max_tokens=1500
#         )
        
#         final_answer = response.choices[0].message.content
        
#         # 更新上下文（只保存原始问题和最终答案）
#         self.text_assembler.update_context(question, final_answer)
        
#         return final_answer
    
#     def generate_cypher_query(self, question: str) -> str:
#         """
#         生成Cypher查询语句
        
#         Args:
#             question: 用户问题
            
#         Returns:
#             Cypher查询语句
#         """
#         # 组装文本
#         prompt = self.text_assembler.get_cypher_query(question)
        
#         # 调用LLM
#         response = self.llm_client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.1,
#             max_tokens=500
#         )
        
#         return response.choices[0].message.content
    
#     def execute_query(self, query: str) -> List[Dict]:
#         """
#         执行Cypher查询
        
#         Args:
#             query: Cypher查询语句
            
#         Returns:
#             查询结果
#         """
#         with self.entity_linker.driver.session() as session:
#             result = session.run(query)
#             return [dict(record) for record in result]


# # 使用示例
# if __name__ == "__main__":
#     # Neo4j配置
#     NEO4J_CONFIG = {
#         "uri": "bolt://localhost:7687",
#         "user": "neo4j",
#         "password": "wswy0129"
#     }
    
#     # 初始化系统
#     qa_system = IntegratedQASystem(
#         neo4j_config=NEO4J_CONFIG,
#         llm_api_key="sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
#     )
    
#     # 单轮问答示例
#     question = "SWAT模型在径流模拟中如何利用气象数据?"
#     print(f"问题: {question}")
#     answer = qa_system.answer_question(question, use_decomposition=False)
#     print(f"回答: {answer}")
    
#     # 多轮问答示例
#     follow_up = "如何提高模拟精度?"
#     print(f"\n问题: {follow_up}")
#     answer = qa_system.answer_question(follow_up, use_decomposition=False)
#     print(f"回答: {answer}")
    
#     # 使用问题分解的复杂问答示例
#     complex_question = "我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?"
#     print(f"\n复杂问题: {complex_question}")
#     answer = qa_system.answer_question(complex_question, use_decomposition=True)
#     print(f"回答: {answer}")
    
#     # 生成并执行Cypher查询示例
#     query_question = "找出与SWAT模型相关的所有概念"
#     print(f"\n查询问题: {query_question}")
#     cypher_query = qa_system.generate_cypher_query(query_question)
#     print(f"生成的Cypher查询: {cypher_query}")
    
#     try:
#         results = qa_system.execute_query(cypher_query)
#         print(f"查询结果: {json.dumps(results, indent=2, ensure_ascii=False)}")
#     except Exception as e:
#         print(f"查询执行错误: {e}")