# dialogue_manager.py
import json
import time
from collections import deque
import jieba
from django.conf import settings
import concurrent.futures
from xy_neo4j.Entity_Mention.LLM_mention_final import DirectMentionRecognizer
from xy_neo4j.Entity_Mention.Entity_Order import EntityLinker
from xy_neo4j.Entity_Mention.integrated_qa_system import IntegratedQASystem

class DialogueManager:
    def __init__(self):
        self.query_stats = deque(maxlen=100)  # 保留最近100次查询数据
        try:
            print("正在初始化问答系统...")
            start_time = time.time()
            
            # Neo4j配置
            NEO4J_CONFIG = {
                "uri": "bolt://localhost:7687",
                "user": "neo4j", 
                "password": "wswy0129"
            }
            
            # API密钥
            API_KEY = "sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
            
            # 初始化实体识别器
            self.entity_recognizer = DirectMentionRecognizer(api_key=API_KEY)
            
            # 初始化实体链接器
            self.entity_linker = EntityLinker(neo4j_config=NEO4J_CONFIG)
            
            # 初始化集成问答系统
            try:
                self.qa_system = IntegratedQASystem(
                    neo4j_config=NEO4J_CONFIG,
                    llm_api_key=API_KEY
                )
            except ImportError as e:
                print(f"问答系统初始化失败: {e}")
                self.qa_system = None
            
            # 初始化其他组件
            if not hasattr(settings, 'ZHIPU'):
                raise Exception("ZHIPU not found in settings")
            self.zhipu = settings.ZHIPU
            jieba.initialize()
            
            init_time = time.time() - start_time
            print(f"系统初始化完成，耗时: {init_time:.2f}秒")
        except Exception as e:
            print("DialogueManager 初始化失败:", e)
            raise

    def get_response(self, question: str) -> dict:
        """完整处理用户问题并返回包含思考过程的回答"""
        overall_start_time = time.time()
        print(f"\n[DialogueManager] 开始处理问题: {question}")
        
        try:
            # 1. 问题分解
            decomp_start_time = time.time()
            print("[DialogueManager] 开始问题分解...")
            sub_questions = self.decompose_question(question)
            decomp_time = time.time() - decomp_start_time
            print(f"[DialogueManager] 问题分解完成，耗时: {decomp_time:.2f}秒")
            
            if sub_questions:
                print("[DialogueManager] 子问题列表:")
                for i, q in enumerate(sub_questions, 1):
                    print(f"  {i}. {q}")
            
            # 2. 知识图谱查询
            print("\n[DialogueManager] 开始知识图谱查询...")
            kg_start_time = time.time()
            
            # 为每个子问题查询知识图谱
            kg_contexts = {}
            for sub_q in sub_questions:
                entities = self.entity_recognizer.recognize(sub_q)
                kg_result = self.query_knowledge_graph(entities)
                kg_contexts[sub_q] = kg_result
            
            kg_time = time.time() - kg_start_time
            print(f"[DialogueManager] 知识图谱查询完成，耗时: {kg_time:.2f}秒")
            
            # 3. 提取知识图谱节点用于可视化
            print("[DialogueManager] 开始提取知识图谱节点...")
            kg_nodes = self._extract_kg_nodes_for_vis(kg_contexts)
            print(f"[DialogueManager] 知识图谱节点统计:")
            print(f"  - 节点数量: {len(kg_nodes.get('nodes', []))}")
            print(f"  - 关系数量: {len(kg_nodes.get('links', []))}")
            
            # 4. 渐进式回答生成
            print("\n[DialogueManager] 开始渐进式生成回答...")
            sub_answers = {}
            accumulated_context = ""
            
            for sub_q in sub_questions:
                # 构建当前子问题的上下文
                current_context = f"问题: {sub_q}\n\n"
                
                # 添加知识图谱上下文
                if sub_q in kg_contexts and kg_contexts[sub_q]:
                    current_context += f"知识图谱信息: {kg_contexts[sub_q]}\n\n"
                
                # 添加累积上下文（先前子问题的回答）
                if accumulated_context:
                    current_context += f"先前的回答: {accumulated_context}\n\n"
                
                # 生成当前子问题的回答
                print(f"[DialogueManager] 生成子问题回答: {sub_q}")
                sub_answer = self._generate_answer_with_context(sub_q, current_context)
                sub_answers[sub_q] = sub_answer
                
                # 更新累积上下文
                accumulated_context += f"\n子问题: {sub_q}\n回答: {sub_answer}"
            
            # 5. 生成最终综合回答
            gen_start_time = time.time()
            print("\n[DialogueManager] 开始生成最终回答...")
            
            final_context = f"原始问题: {question}\n\n"
            final_context += "子问题分解与回答:\n"
            for sub_q, answer in sub_answers.items():
                final_context += f"问: {sub_q}\n答: {answer}\n\n"
            
            final_answer = self._generate_final_answer(question, final_context)
            gen_time = time.time() - gen_start_time
            
            # 计算总思考时间
            overall_time = time.time() - overall_start_time
            
            # 时间统计信息
            time_analysis = {
                "问题分解": f"{decomp_time:.2f}秒",
                "知识检索": f"{kg_time:.2f}秒",
                "回答生成": f"{gen_time:.2f}秒",
                "总思考时间": f"{overall_time:.2f}秒"
            }
            
            print(f"[DialogueManager] 处理完成，总耗时: {overall_time:.2f}秒")
            
            # 返回结果
            return {
                "answer": final_answer,
                "sub_questions": sub_questions,
                "sub_answers": sub_answers,
                "kg_context": kg_contexts,
                "kg_nodes": kg_nodes,
                "time_analysis": time_analysis,
                "thinking_process": self._format_thinking_process(sub_questions, kg_contexts, sub_answers)
            }
            
        except Exception as e:
            print(f"[DialogueManager] 错误: {str(e)}")
            overall_time = time.time() - overall_start_time
            
            return {
                "answer": f"抱歉，系统处理您的问题时遇到了错误: {str(e)}",
                "sub_questions": None,
                "sub_answers": None,
                "kg_context": None,
                "kg_nodes": {"nodes": [], "links": [], "categories": []},
                "time_analysis": {"总思考时间": f"{overall_time:.2f}秒"},
                "thinking_process": None
            }
    
    def decompose_question(self, question):
        """问题分解 - 使用集成问答系统的分解能力"""
        if self.qa_system and hasattr(self.qa_system, 'question_decomposer'):
            try:
                # 使用集成系统的分解器
                decomposed = self.qa_system.question_decomposer.decompose_question(question, use_llm=False)
                formatted = self.qa_system.question_decomposer.format_questions_for_qa(decomposed)
                return formatted
            except Exception as e:
                print(f"使用集成系统分解问题失败: {e}")
                # 回退到原有方法
                return self._legacy_decompose_question(question)
        else:
            # 使用原有方法
            return self._legacy_decompose_question(question)
    
    def _legacy_decompose_question(self, question):
        """结构化问题分解器（原始方法）"""
        prompt = f"""
        【问题分析任务】
        请将水文建模问题分解为可执行的知识单元，遵循以下规则：
        
        输入问题："{question}"
        
        【分解规则】
        1. 按建模流程的5个阶段分层解析
        2. 每个子问题必须包含"问题焦点"和"预期输出"
        3. 使用水文专业术语（SWAT、DEM预处理等）
        4. 保持流域数据特性
        """
        
        return self.zhipu.get_deepseek_response(prompt)
    
    def query_knowledge_graph(self, entities):
        """根据实体查询知识图谱"""
        try:
            if not entities:
                return "未识别到相关实体。"
            
            # 实体链接
            linked_entities = []
            for entity in entities:
                results = self.entity_linker._rank_single_entity("", entity, top_k=3)
                if results:
                    linked_entities.extend(results)
            
            # 整理知识图谱结果
            kg_data = []
            
            # 如果有链接实体，查询它们的关系
            if linked_entities:
                with self.entity_linker.driver.session() as session:
                    for entity_info in linked_entities:
                        entity_name = entity_info['entity']
                        
                        # 查询实体关系
                        query = """
                        MATCH (n)-[r]-(m)
                        WHERE n.name = $entity_name
                        RETURN n.name as source, type(r) as relation, m.name as target, 
                               CASE WHEN n.desc IS NOT NULL THEN n.desc ELSE '' END as source_desc, 
                               CASE WHEN m.desc IS NOT NULL THEN m.desc ELSE '' END as target_desc,
                               CASE WHEN n.source_article IS NOT NULL THEN n.source_article ELSE '' END as reference
                        LIMIT 5
                        """
                        result = session.run(query, entity_name=entity_name)
                        
                        for record in result:
                            path = f"{record['source']} → {record['relation']} → {record['target']}"
                            summary = record['source_desc'] or record['target_desc']
                            reference = record['reference']
                            
                            kg_data.append({
                                "path": path,
                                "summary": summary,
                                "reference": reference
                            })
            
            # 如果没有找到关系，至少返回实体信息
            if not kg_data and linked_entities:
                for entity_info in linked_entities:
                    kg_data.append({
                        "path": entity_info['entity'],
                        "summary": entity_info.get('desc', ''),
                        "reference": ""
                    })
            
            if kg_data:
                return json.dumps(kg_data, ensure_ascii=False)
            
            return "知识图谱中未找到相关信息。"
        except Exception as e:
            print(f"知识图谱查询失败: {e}")
            return f"知识图谱查询失败: {str(e)}"
    
    def _extract_kg_nodes_for_vis(self, kg_contexts):
        """从知识图谱上下文中提取节点用于可视化"""
        try:
            nodes = []
            links = []
            categories = []
            category_map = {}
            
            # 处理多个子问题的知识图谱结果
            for sub_q, kg_context in kg_contexts.items():
                if not kg_context or not isinstance(kg_context, str):
                    continue
                
                try:
                    if kg_context.startswith('['):
                        kg_data = json.loads(kg_context)
                        
                        for item in kg_data:
                            if 'path' in item:
                                parts = item['path'].split('→')
                                if len(parts) >= 3:
                                    source = parts[0].strip()
                                    relation = parts[1].strip()
                                    target = parts[2].strip()
                                    
                                    # 添加节点
                                    for node in [source, target]:
                                        if not any(n['name'] == node for n in nodes):
                                            # 为节点分配类别
                                            if '模型' in node:
                                                category = '模型'
                                            elif '数据' in node:
                                                category = '数据'
                                            elif '方法' in node:
                                                category = '方法'
                                            else:
                                                category = '概念'
                                            
                                            # 确保类别存在
                                            if category not in category_map:
                                                category_map[category] = len(categories)
                                                categories.append({'name': category})
                                            
                                            nodes.append({
                                                'name': node,
                                                'category': category_map[category],
                                                'value': 20,
                                                'symbolSize': 50,
                                                'draggable': True,
                                                'desc': item.get('summary', '')
                                            })
                                    
                                    # 添加关系
                                    links.append({
                                        'source': source,
                                        'target': target,
                                        'name': relation,
                                        'value': relation
                                    })
                                elif len(parts) == 1:
                                    # 处理单个实体（没有关系）
                                    node = parts[0].strip()
                                    if not any(n['name'] == node for n in nodes):
                                        category = '概念'
                                        if category not in category_map:
                                            category_map[category] = len(categories)
                                            categories.append({'name': category})
                                        
                                        nodes.append({
                                            'name': node,
                                            'category': category_map[category],
                                            'value': 20,
                                            'symbolSize': 50,
                                            'draggable': True,
                                            'desc': item.get('summary', '')
                                        })
                except json.JSONDecodeError:
                    print(f"JSON解析失败: {kg_context[:100]}...")
            
            # 返回结果
            return {
                'nodes': nodes,
                'links': links,
                'categories': categories
            }
        except Exception as e:
            print(f"提取知识图谱节点失败: {e}")
            return {
                'nodes': [],
                'links': [],
                'categories': []
            }
    
    def _generate_answer_with_context(self, question, context):
        """基于上下文生成回答"""
        try:
            prompt = f"""
            你是一个水文领域专家，专注于SWAT模型和径流模拟相关问题的解答。
            
            {context}
            
            当前问题: {question}
            
            请给出专业、准确的回答。回答应当全面但简洁，突出重点信息。
            """
            
            if self.qa_system:
                return self.qa_system.answer_question(
                    question=prompt,
                    use_decomposition=False,
                    force_refresh=True
                )
            else:
                return self.zhipu.get_deepseek_response(prompt)
        except Exception as e:
            print(f"回答生成失败: {e}")
            return f"无法生成回答: {str(e)}"
    
    def _generate_final_answer(self, question, context):
        """生成最终综合回答"""
        try:
            prompt = f"""
            作为水文领域专家，请基于以下子问题的回答，为原始问题提供一个综合全面的回答：
            
            {context}
            
            原始问题: {question}
            
            请给出一个连贯、专业、全面的回答，确保覆盖所有关键信息，并避免重复内容。回答应当结构清晰，语言流畅。
            回答时,不要使用#或*等特殊字符 
            根据给出的内容,列出参考文献,要求必须从前文中获取,不要自己生成参考文献
            """
            
            if self.qa_system:
                return self.qa_system.answer_question(
                    question=prompt,
                    use_decomposition=False,
                    force_refresh=True
                )
            else:
                return self.zhipu.get_deepseek_response(prompt)
        except Exception as e:
            print(f"生成最终回答失败: {e}")
            return f"无法生成综合回答: {str(e)}"
    
    def _format_thinking_process(self, sub_questions, kg_contexts, sub_answers):
        """格式化思考过程以便前端展示"""
        thinking = []
        
        # 添加问题分解步骤
        if sub_questions:
            thinking_item = {
                "title": "问题分解",
                "content": sub_questions
            }
            if sub_answers:
                thinking_item["answers"] = sub_answers
            thinking.append(thinking_item)
        
        # 添加知识图谱检索步骤
        kg_step = {
            "title": "知识图谱检索",
            "content": []
        }
        
        for sub_q, kg_context in kg_contexts.items():
            if kg_context and isinstance(kg_context, str) and kg_context.startswith('['):
                try:
                    kg_data = json.loads(kg_context)
                    if kg_data:
                        kg_step["content"].extend(kg_data)
                except:
                    pass
        
        if kg_step["content"]:
            thinking.append(kg_step)
        
        return thinking