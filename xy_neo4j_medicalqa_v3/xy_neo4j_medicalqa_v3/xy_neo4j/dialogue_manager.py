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
from xy_neo4j.Entity_Mention.DeepSeek_mention import DeepSeekMentionRecognizer
from openai import OpenAI
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
print("Python路径:", sys.path)
try:
    import config
    print("配置导入成功:", config.API_KEYS)
except Exception as e:
    print("配置导入错误:", e)
from config import NEO4J_CONFIG, API_KEYS, LLM_CONFIG, ENTITY_CONFIG
from llm_client_factory import LLMClientFactory

class DialogueManager:
    def __init__(self, llm_provider=None):
        self.query_stats = deque(maxlen=100)  # 保留最近100次查询数据
        self.current_provider = llm_provider or LLM_CONFIG["default_provider"]
        
        try:
            print("正在初始化问答系统...")
            start_time = time.time()
            
            # 初始化实体识别器
            try:
                self.entity_recognizer = DeepSeekMentionRecognizer(api_key=API_KEYS.get(self.current_provider))
            except Exception as e:
                print(f"实体识别器初始化失败: {e}")
                self.entity_recognizer = None  # 或使用备选识别器
            
            # 获取EntityLinker单例实例并重置查询状态
            self.entity_linker = EntityLinker.get_instance().reset_query_state()
            
            # 初始化集成问答系统
            try:
                self.qa_system = IntegratedQASystem(
                    neo4j_config=NEO4J_CONFIG,
                    llm_api_key=API_KEYS[self.current_provider]
                )
            except ImportError as e:
                print(f"问答系统初始化失败: {e}")
                self.qa_system = None
            
            # 初始化其他组件
            if not hasattr(settings, 'DEEPSEEK'):
                print("警告: DEEPSEEK不在settings中，使用config.py中的配置")
                self.deepseek = {
                    "api_key": API_KEYS[self.current_provider],
                }
            else:
                self.deepseek = settings.DEEPSEEK
            jieba.initialize()
            
            # 使用LLM客户端工厂创建LLM客户端
            self.llm_client = LLMClientFactory.create_client(self.current_provider)
            
            end_time = time.time()
            print(f"问答系统初始化完成，耗时: {end_time - start_time:.2f}秒")
        except Exception as e:
            print(f"DialogueManager初始化失败: {e}")
            raise

    def switch_provider(self, provider):
        """
        切换LLM提供商
        
        Args:
            provider: 新的LLM提供商名称
            
        Returns:
            切换结果信息
        """
        if provider not in LLM_CONFIG["providers"]:
            return {"success": False, "message": f"未知的LLM提供商: {provider}"}
        
        try:
            # 更新当前提供商
            self.current_provider = provider
            
            # 更新LLM客户端
            self.llm_client = LLMClientFactory.create_client(provider)
            
            # 更新实体识别器
            self.entity_recognizer = DeepSeekMentionRecognizer(api_key=API_KEYS.get(provider))
            
            # 更新集成问答系统
            self.qa_system = IntegratedQASystem(
                neo4j_config=NEO4J_CONFIG,
                llm_api_key=API_KEYS[provider]
            )
            
            # 更新全局默认提供商
            LLMClientFactory.set_default_provider(provider)
            
            return {
                "success": True, 
                "message": f"已切换到 {provider} API",
                "provider": provider
            }
        except Exception as e:
            return {"success": False, "message": f"切换API失败: {str(e)}"}
    
    def get_available_providers(self):
        """
        获取可用的LLM提供商列表
        
        Returns:
            可用的LLM提供商列表
        """
        return {
            "providers": LLMClientFactory.get_available_providers(),
            "current": self.current_provider
        }

    def get_response(self, question: str) -> dict:
        """完整处理用户问题并返回包含思考过程的回答"""
        overall_start_time = time.time()
        print(f"\n[DialogueManager] 开始处理问题: {question}")
        
        try:
            # 重置EntityLinker查询状态
            self.entity_linker.reset_query_state()
            
            # 1. 问题分解
            sub_questions, decomp_time = self._perform_question_decomposition(question)
            
            # 2. 知识图谱查询
            kg_contexts, kg_time = self._perform_knowledge_graph_query(sub_questions)
            
            # 3. 提取知识图谱节点用于可视化
            kg_nodes = self._extract_kg_nodes_for_vis(kg_contexts)
            self._log_kg_node_stats(kg_nodes)
            
            # 4. 渐进式回答生成
            sub_answers = self._generate_sub_answers(sub_questions, kg_contexts)
            
            # 5. 生成最终综合回答
            final_answer, gen_time = self._generate_final_comprehensive_answer(question, sub_questions, sub_answers)
            
            # 6. 计算总时间并记录
            overall_time = time.time() - overall_start_time
            time_analysis = self._create_time_analysis(decomp_time, kg_time, gen_time, overall_time)
            
            # 7. 生成思考过程和提取参考文献
            thinking_process = self._format_thinking_process(sub_questions, kg_contexts, sub_answers)
            references = self._extract_references_from_kg(kg_contexts)
            
            # 8. 返回完整响应数据
            return self._create_response_dict(
                final_answer, kg_nodes, thinking_process, time_analysis,
                references, sub_questions, sub_answers, overall_time
            )
            
        except Exception as e:
            return self._handle_response_error(e, question, overall_start_time)
    
    def _perform_question_decomposition(self, question):
        """执行问题分解步骤"""
        decomp_start_time = time.time()
        print("[DialogueManager] 开始问题分解...")
        
        sub_questions = self.decompose_question(question)
        decomp_time = time.time() - decomp_start_time
        
        print(f"[DialogueManager] 问题分解完成，耗时: {decomp_time:.2f}秒")
        if sub_questions:
            print("[DialogueManager] 子问题列表:")
            for i, q in enumerate(sub_questions, 1):
                print(f"  {i}. {q}")
        
        return sub_questions, decomp_time
    
    def _perform_knowledge_graph_query(self, sub_questions):
        """执行知识图谱查询步骤"""
        print("\n[DialogueManager] 开始知识图谱查询...")
        kg_start_time = time.time()
        
        # 为每个子问题查询知识图谱
        kg_contexts = {}
        for sub_q in sub_questions:
            entities = self.entity_recognizer.recognize(sub_q)
            kg_result = self.query_knowledge_graph(entities)
            kg_contexts[sub_q] = kg_result
        
        # 存储kg_contexts作为实例变量，供_generate_final_answer使用
        self.kg_contexts = kg_contexts
        
        kg_time = time.time() - kg_start_time
        print(f"[DialogueManager] 知识图谱查询完成，耗时: {kg_time:.2f}秒")
        
        return kg_contexts, kg_time
    
    def _log_kg_node_stats(self, kg_nodes):
        """记录知识图谱节点统计信息"""
        print(f"[DialogueManager] 知识图谱节点统计:")
        print(f"  - 节点数量: {len(kg_nodes.get('nodes', []))}")
        print(f"  - 关系数量: {len(kg_nodes.get('links', []))}")
    
    def _generate_sub_answers(self, sub_questions, kg_contexts):
        """为每个子问题生成回答"""
        print("\n[DialogueManager] 开始渐进式生成回答...")
        sub_answers = {}
        accumulated_context = ""
        
        for sub_q in sub_questions:
            # 构建当前子问题的上下文
            current_context = self._build_sub_question_context(sub_q, kg_contexts, accumulated_context)
            
            # 生成当前子问题的回答
            print(f"[DialogueManager] 生成子问题回答: {sub_q}")
            sub_answer = self._generate_answer_with_context(sub_q, current_context)
            sub_answers[sub_q] = sub_answer
            
            # 更新累积上下文
            accumulated_context += f"\n子问题: {sub_q}\n回答: {sub_answer}"
        
        return sub_answers
    
    def _build_sub_question_context(self, sub_q, kg_contexts, accumulated_context):
        """构建子问题的上下文"""
        current_context = f"问题: {sub_q}\n\n"
        
        # 添加知识图谱上下文
        if sub_q in kg_contexts and kg_contexts[sub_q]:
            current_context += f"知识图谱信息: {kg_contexts[sub_q]}\n\n"
        
        # 添加累积上下文（先前子问题的回答）
        if accumulated_context:
            current_context += f"先前的回答: {accumulated_context}\n\n"
        
        return current_context
    
    def _generate_final_comprehensive_answer(self, question, sub_questions, sub_answers):
        """生成最终综合回答"""
        gen_start_time = time.time()
        print("\n[DialogueManager] 开始生成最终回答...")
        
        final_context = self._build_final_answer_context(question, sub_questions, sub_answers)
        final_answer = self._generate_final_answer(question, final_context)
        
        gen_time = time.time() - gen_start_time
        return final_answer, gen_time
    
    def _build_final_answer_context(self, question, sub_questions, sub_answers):
        """构建最终回答的上下文"""
        final_context = f"原始问题: {question}\n\n"
        final_context += "子问题分解与回答:\n"
        
        for sub_q, answer in sub_answers.items():
            final_context += f"问: {sub_q}\n答: {answer}\n\n"
        
        return final_context
    
    def _create_time_analysis(self, decomp_time, kg_time, gen_time, overall_time):
        """创建时间分析记录"""
        return {
            "问题分解": f"{decomp_time:.2f}秒",
            "知识检索": f"{kg_time:.2f}秒",
            "回答生成": f"{gen_time:.2f}秒",
            "总思考时间": f"{overall_time:.2f}秒"
        }
    
    def _extract_references_from_kg(self, kg_contexts):
        """从知识图谱结果中提取参考文献"""
        references = []
        
        # 使用当前对话管理器中的kg_contexts变量
        for sub_q, kg_data in getattr(self, 'kg_contexts', {}).items():
            if kg_data and isinstance(kg_data, str) and kg_data.startswith('['):
                try:
                    kg_items = json.loads(kg_data)
                    for item in kg_items:
                        if 'reference' in item and item['reference'] and item['reference'] not in references:
                            references.append(item['reference'])
                except Exception as e:
                    print(f"解析参考文献错误: {e}")
                    continue
        
        return references
    
    def _create_response_dict(self, final_answer, kg_nodes, thinking_process, time_analysis, 
                             references, sub_questions, sub_answers, overall_time):
        """创建完整的响应字典"""
        return {
            'answer': final_answer,
            'kg_nodes': kg_nodes,
            'thinking_process': thinking_process,
            'time_analysis': {
                'total': f"{overall_time:.2f}秒",
                'decompose': time_analysis["问题分解"],
                'kg_query': time_analysis["知识检索"],
                'answer_gen': time_analysis["回答生成"]
            },
            'references': references,
            'sub_questions': sub_questions,
            'sub_answers': sub_answers
        }
    
    def _handle_response_error(self, error, question, start_time):
        """处理响应过程中的错误"""
        print(f"[DialogueManager] 错误: {str(error)}")
        overall_time = time.time() - start_time
        
        return {
            "answer": f"抱歉，系统处理您的问题时遇到了错误: {str(error)}",
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
                decomposed = self.qa_system.question_decomposer.decompose_question(question, use_llm=True)
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
        
        return self.deepseek["api_key"].get_deepseek_response(prompt)
    
    def query_knowledge_graph(self, entities):
        """根据实体查询知识图谱"""
        try:
            if not entities:
                print("[DialogueManager] 未识别到相关实体")
                return "未识别到相关实体。"
            
            print(f"[DialogueManager] 开始查询知识图谱，实体列表: {entities}")
            
            # 使用GPU加速的批量处理方法
            linked_entities_dict = self.entity_linker.rank_entities_batch_gpu("", entities, top_k=3)
            print(f"[DialogueManager] 实体链接结果: {linked_entities_dict}")
            
            # 检查第一个实体的节点标签（调试用）
            if linked_entities_dict and len(linked_entities_dict) > 0:
                first_entity = next(iter(linked_entities_dict))
                if linked_entities_dict[first_entity] and len(linked_entities_dict[first_entity]) > 0:
                    first_linked_entity = linked_entities_dict[first_entity][0]['name']
                    print(f"[DialogueManager] 检查Neo4j节点 '{first_linked_entity}' 的标签")
                    node_info = self.entity_linker.check_node_labels(first_linked_entity)
                    print(f"[DialogueManager] 节点标签检查结果: {node_info}")
            
            # 整理知识图谱结果时考虑source_article只存在于地理问题类型
            kg_data = []
            processed_entities = set()
            
            for source_entity, entity_results in linked_entities_dict.items():
                for entity_info in entity_results:
                    entity_name = entity_info['name']
                    
                    # 避免重复处理
                    if entity_name in processed_entities:
                        continue
                    processed_entities.add(entity_name)
                    
                    # 获取相关实体
                    related_entities = self.entity_linker.get_related_entities(entity_name, limit=3)
                    
                    # 确定实体类别
                    category = entity_info.get('category', 'Unknown')
                    
                    # 只有地理问题类型才有reference（来自source_article）
                    reference = ""
                    if category == "地理问题":
                        reference = entity_info.get('reference', '')
                    
                    # 构建知识项
                    kg_item = {
                        'path': f"{source_entity} → {entity_name}",
                        'summary': entity_info.get('desc', ''),
                        'source': source_entity,
                        'target': entity_name,
                        'score': entity_info.get('score', 0),
                        'category': category,
                        'reference': reference,
                        'related_entities': []
                    }
                    
                    # 添加相关实体信息
                    for rel in related_entities:
                        rel_category = rel.get('category', 'Unknown')
                        kg_item['related_entities'].append({
                            'name': rel['entity'],
                            'relation': rel['relation'],
                            'direction': rel['direction'],
                            'desc': rel.get('desc', ''),
                            'category': rel_category
                        })
                    
                    kg_data.append(kg_item)
            
            if kg_data:
                print(f"[DialogueManager] 成功构建知识图谱，共 {len(kg_data)} 个节点")
                return json.dumps(kg_data, ensure_ascii=False)
            
            print("[DialogueManager] 知识图谱中未找到相关信息")
            return "知识图谱中未找到相关信息。"
            
        except Exception as e:
            print(f"[DialogueManager] 知识图谱查询失败: {e}")
            return f"知识图谱查询失败: {str(e)}"
        
    def _extract_kg_nodes_for_vis(self, kg_contexts):
        """从知识图谱上下文中提取节点用于可视化"""
        try:
            nodes = []
            links = []
            categories = []
            category_map = {}
            
            print(f"\n[DialogueManager] 开始提取知识图谱节点用于可视化...")
            
            # 处理多个子问题的知识图谱结果
            for sub_q, kg_context in kg_contexts.items():
                if not kg_context or not isinstance(kg_context, str):
                    continue
                
                try:
                    if kg_context.startswith('['):
                        kg_data = json.loads(kg_context)
                        print(f"[DialogueManager] 解析到 {len(kg_data)} 个知识图谱项")
                        
                        for item in kg_data:
                            # 检查必需字段
                            if 'source' not in item or 'target' not in item:
                                continue
                                
                            source = item['source']
                            target = item['target']
                            relation = "相关联"  # 默认关系
                            
                            # 提取路径中的关系（如果存在）
                            if 'path' in item and '→' in item['path']:
                                parts = item['path'].split('→')
                                if len(parts) >= 3:
                                    relation = parts[1].strip()
                            
                            # 获取源节点和目标节点的类别
                            source_category = item.get('category', '概念')
                            print(f"[DialogueManager] 源节点 '{source}' 的类别: {source_category}")
                            
                            # 尝试从相关实体中获取目标节点的类别
                            target_category = '概念'  # 默认值
                            if 'related_entities' in item:
                                for rel_entity in item['related_entities']:
                                    if rel_entity.get('name') == target:
                                        target_category = rel_entity.get('category', '概念')
                                        break
                            print(f"[DialogueManager] 目标节点 '{target}' 的类别: {target_category}")
                            
                            # 添加节点
                            for node_name, node_desc, node_category in [
                                (source, item.get('summary', ''), source_category),
                                (target, item.get('summary', ''), target_category)
                            ]:
                                if not any(n['name'] == node_name for n in nodes):
                                    # 使用节点的实际类别
                                    category = node_category if node_category not in ['Unknown', 'unknown', 'unknown'] else '地理概念'
                                    print(f"[DialogueManager] 节点 '{node_name}' 将使用类别: {category}")
                                    
                                    # 确保类别存在于分类中
                                    if category not in category_map:
                                        category_map[category] = len(categories)
                                        categories.append({'name': category})
                                        print(f"[DialogueManager] 添加新的类别: {category}")
                                    
                                    nodes.append({
                                        'name': node_name,
                                        'category': category_map[category],
                                        'value': item.get('score', 0.5) * 20,
                                        'symbolSize': 40 + (item.get('score', 0.5) * 20),
                                        'draggable': True,
                                        'desc': node_desc
                                    })
                            
                            # 添加关系
                            if source != target:  # 避免自环
                                links.append({
                                    'source': source,
                                    'target': target,
                                    'name': relation,
                                    'value': relation
                                })
                except json.JSONDecodeError as je:
                    print(f"JSON解析失败: {kg_context[:100]}... - {je}")
                except Exception as e:
                    print(f"处理知识图谱上下文时出错: {e}")
            
            print(f"[DialogueManager] 知识图谱可视化提取完成:")
            print(f"  - 节点数量: {len(nodes)}")
            print(f"  - 关系数量: {len(links)}")
            print(f"  - 类别数量: {len(categories)}")
            print(f"  - 类别列表: {[c['name'] for c in categories]}")
            
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
            # 生成缓存键
            import hashlib
            cache_key = f"answer_{hashlib.md5((question + context).encode()).hexdigest()}"
            
            # 检查缓存
            if hasattr(self, 'answer_cache') and cache_key in self.answer_cache:
                print(f"[缓存命中] 使用缓存的子问题回答: {question[:30]}...")
                return self.answer_cache[cache_key]
            
            print(f"[缓存未命中] 生成子问题回答: {question[:30]}...")
            
            prompt = f"""
            请基于以下知识库信息回答问题。如果无法从知识库中找到答案，请说明无法回答。

            问题: {question}

            知识库信息:
            {context}
            """
            
            # 使用LLM客户端工厂创建的客户端
            messages = [{"role": "user", "content": prompt}]
            answer = self.llm_client.chat_completion(messages)
            
            # 存入缓存
            if not hasattr(self, 'answer_cache'):
                self.answer_cache = {}
            self.answer_cache[cache_key] = answer
            
            return answer
            
        except Exception as e:
            print(f"生成回答时出错: {e}")
            return f"抱歉，生成回答时出错: {str(e)}"
    
    def _generate_final_answer(self, question, context):
        """生成最终回答"""
        try:
            # 生成缓存键
            import hashlib
            cache_key = f"final_{hashlib.md5((question + context).encode()).hexdigest()}"
            
            # 检查缓存
            if hasattr(self, 'answer_cache') and cache_key in self.answer_cache:
                print(f"[缓存命中] 使用缓存的最终回答: {question[:30]}...")
                return self.answer_cache[cache_key]
            
            print(f"[缓存未命中] 生成最终回答: {question[:30]}...")
            
            # 构建提示
            prompt = f"""
            请基于以下信息生成对问题的综合回答。回答应该全面、准确、连贯。

            问题: {question}

            相关信息:
            {context}
            """
            
            # 使用LLM客户端工厂创建的客户端
            messages = [
                {"role": "system", "content": "你是一个专业的水文学和环境科学助手，擅长提供准确、全面的回答。"},
                {"role": "user", "content": prompt}
            ]
            answer = self.llm_client.chat_completion(messages)
            
            # 存入缓存
            if not hasattr(self, 'answer_cache'):
                self.answer_cache = {}
            self.answer_cache[cache_key] = answer
            
            return answer
            
        except Exception as e:
            print(f"生成最终回答时出错: {e}")
            return f"抱歉，生成最终回答时出错: {str(e)}"
    
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

    def get_node_answer(self, node_info, question):
        """获取节点的回答，避免重复"""
        # 如果已经有这个节点的回答，直接返回
        if node_info['name'] in self.used_descriptions:
            return None
        
        # 记录已使用的描述
        self.used_descriptions.add(node_info['name'])
        
        # 组合节点信息生成回答
        answer = ""
        if node_info.get('desc'):
            answer = node_info['desc']
        
        # 获取相关步骤或方法
        steps = self.get_related_steps(node_info['name'])
        if steps:
            answer += "\n具体步骤：\n" + "\n".join(f"- {step}" for step in steps)
        
        return answer

    def process_sub_questions(self, sub_questions):
        """处理子问题，避免重复答案"""
        self.used_descriptions = set()  # 记录已使用的描述
        
        answers = {}
        for question in sub_questions:
            # 获取相关节点
            nodes = self.get_relevant_nodes(question)
            
            # 生成答案
            answer_parts = []
            for node in nodes:
                node_answer = self.get_node_answer(node, question)
                if node_answer:
                    answer_parts.append(node_answer)
            
            # 如果没有找到新的答案，生成概括性回答
            if not answer_parts:
                answer_parts.append(self.generate_summary_answer(question))
            
            answers[question] = "\n".join(answer_parts)
        
        return answers

    def generate_summary_answer(self, question):
        """生成概括性回答，避免重复"""
        # 使用 LLM 生成不依赖于节点描述的回答
        prompt = f"请简要回答这个问题，不要重复已经提到过的内容：{question}"
        response = self.llm_client.get_completion(prompt)
        return response