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

class DialogueManager:
    def __init__(self):
        self.query_stats = deque(maxlen=100)  # 保留最近100次查询数据
        try:
            print("正在初始化问答系统...")
            start_time = time.time()
            
            # 使用集中配置
            # Neo4j配置
            # 不再硬编码: NEO4J_CONFIG = {"uri": "bolt://localhost:7687", ...}
            
            # API密钥
            # 不再硬编码: API_KEY = "sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
            
            # 初始化实体识别器
            try:
                self.entity_recognizer = DeepSeekMentionRecognizer(api_key=API_KEYS.get("deepseek"))
            except Exception as e:
                print(f"实体识别器初始化失败: {e}")
                self.entity_recognizer = None  # 或使用备选识别器
            
            # 获取EntityLinker单例实例并重置查询状态
            self.entity_linker = EntityLinker.get_instance().reset_query_state()
            
            # 初始化集成问答系统
            try:
                self.qa_system = IntegratedQASystem(
                    neo4j_config=NEO4J_CONFIG,
                    llm_api_key=API_KEYS["deepseek"]
                )
            except ImportError as e:
                print(f"问答系统初始化失败: {e}")
                self.qa_system = None
            
            # 初始化其他组件
            if not hasattr(settings, 'DEEPSEEK'):
                print("警告: DEEPSEEK不在settings中，使用config.py中的配置")
                self.deepseek = {
                    "api_key": API_KEYS["deepseek"],
                }
            else:
                self.deepseek = settings.DEEPSEEK
            jieba.initialize()
            
            # 修改 LLM 客户端初始化
            self.llm_client = OpenAI(
                api_key=API_KEYS["deepseek"],
                base_url=LLM_CONFIG["base_url"]
            )
            
            end_time = time.time()
            print(f"问答系统初始化完成，耗时: {end_time - start_time:.2f}秒")
        except Exception as e:
            print(f"DialogueManager初始化失败: {e}")
            raise

    def get_response(self, question: str) -> dict:
        """完整处理用户问题并返回包含思考过程的回答"""
        overall_start_time = time.time()
        print(f"\n[DialogueManager] 开始处理问题: {question}")
        
        try:
            # 重置EntityLinker查询状态
            self.entity_linker.reset_query_state()
            
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
            
            # 存储kg_contexts作为实例变量，供_generate_final_answer使用
            self.kg_contexts = kg_contexts
            
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
            thinking_process = self._format_thinking_process(sub_questions, kg_contexts, sub_answers)
            
            # 从知识图谱结果中提取参考文献
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
            
            # 返回完整响应数据
            return {
                'answer': final_answer,
                'kg_nodes': kg_nodes,
                'thinking_process': thinking_process,
                'time_analysis': {
                    'total': f"{overall_time:.2f}秒",
                    'decompose': f"{decomp_time:.2f}秒",
                    'kg_query': f"{kg_time:.2f}秒",
                    'answer_gen': f"{gen_time:.2f}秒"
                },
                'references': references,
                'sub_questions': sub_questions,
                'sub_answers': sub_answers
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
            
            # 整理知识图谱结果
            kg_data = []
            processed_entities = set()  # 用于跟踪已处理的实体
            
            for source_entity, entity_results in linked_entities_dict.items():
                print(f"[DialogueManager] 处理源实体: {source_entity}")
                
                for entity_info in entity_results:
                    try:
                        entity_name = entity_info['name']  # 使用 'name' 而不是 'entity'
                        
                        # 避免重复处理相同的实体
                        if entity_name in processed_entities:
                            continue
                        processed_entities.add(entity_name)
                        
                        print(f"[DialogueManager] 获取实体 {entity_name} 的相关实体")
                        
                        # 获取相关实体（最多3个）
                        try:
                            related_entities = self.entity_linker.get_related_entities(entity_name, limit=3)
                            print(f"[DialogueManager] 找到 {len(related_entities)} 个相关实体")
                        except Exception as rel_e:
                            print(f"[DialogueManager] 获取相关实体失败: {rel_e}")
                            related_entities = []
                        
                        # 构建知识路径
                        path = f"{source_entity} → {entity_name}"
                        if related_entities:
                            # 添加相关实体到路径
                            related_names = [rel['entity'] for rel in related_entities]
                            path += f" → [{', '.join(related_names)}]"
                        
                        # 添加到结果中
                        kg_item = {
                            'path': path,
                            'summary': entity_info.get('desc', ''),
                            'source': source_entity,
                            'target': entity_name,
                            'score': entity_info.get('score', 0),
                            'category': entity_info.get('category', 'Unknown'),
                            'reference': entity_info.get('reference', ''),
                            'related_entities': [
                                {
                                    'name': rel['entity'],
                                    'relation': rel['relation'],
                                    'direction': rel['direction'],
                                    'desc': rel.get('desc', ''),
                                    'category': rel.get('category', 'Unknown')
                                }
                                for rel in related_entities
                            ]
                        }
                        
                        # 确保所有必要字段都有值
                        for key in ['summary', 'category', 'reference']:
                            if not kg_item[key]:
                                kg_item[key] = '未知'
                        
                        kg_data.append(kg_item)
                        print(f"[DialogueManager] 成功添加实体 {entity_name} 的知识图谱项")
                        
                    except Exception as inner_e:
                        print(f"[DialogueManager] 处理实体信息时出错: {inner_e}")
                        continue
            
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
            
            # 处理多个子问题的知识图谱结果
            for sub_q, kg_context in kg_contexts.items():
                if not kg_context or not isinstance(kg_context, str):
                    continue
                
                try:
                    if kg_context.startswith('['):
                        kg_data = json.loads(kg_context)
                        
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
                            
                            # 添加节点
                            for node_name, node_desc, node_category in [
                                (source, item.get('summary', ''), item.get('category', '概念')),
                                (target, item.get('summary', ''), item.get('category', '概念'))
                            ]:
                                if not any(n['name'] == node_name for n in nodes):
                                    # 为节点确定类别
                                    category = node_category
                                    if not category:
                                        if '模型' in node_name:
                                            category = '模型'
                                        elif '数据' in node_name:
                                            category = '数据'
                                        elif '方法' in node_name:
                                            category = '方法'
                                        else:
                                            category = '概念'
                                    
                                    # 确保类别存在
                                    if category not in category_map:
                                        category_map[category] = len(categories)
                                        categories.append({'name': category})
                                    
                                    nodes.append({
                                        'name': node_name,
                                        'category': category_map[category],
                                        'value': item.get('score', 0.5) * 20,  # 使用得分设置价值
                                        'symbolSize': 40 + (item.get('score', 0.5) * 20), # 根据得分调整大小
                                        'draggable': True,
                                        'desc': item.get('summary', '')
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
            
            # 修改所有 LLM 调用
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"回答生成失败: {e}")
            return f"无法生成回答: {str(e)}"
    
    def _generate_final_answer(self, question, context):
        """生成最终综合回答"""
        try:
            # 从知识图谱结果中提取参考文献
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
            
            # 修改提示以包含实际参考文献
            prompt = f"""
            作为水文领域专家，请基于以下子问题的回答，为原始问题提供一个综合全面的回答：
            
            {context}
            
            原始问题: {question}
            
            请给出一个连贯、专业、全面的回答，确保覆盖所有关键信息，并避免重复内容。回答应当结构清晰，语言流畅。
            回答时，不要使用#或*等特殊字符。
            
            请在回答的最后添加以下参考文献列表：
            """
            
            # 添加实际参考文献
            if references:
                prompt += "参考文献：\n"
                for i, ref in enumerate(references[:5], 1):  # 限制为最多5个参考文献
                    prompt += f"{i}. {ref}\n"
            else:
                prompt += "参考文献：暂无可用的参考文献。\n"
            
            # 调用LLM
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048
            )
            
            return response.choices[0].message.content.strip()
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