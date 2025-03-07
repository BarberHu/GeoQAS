# 在 dialogue_manager.py 中

import json
import time
from collections import deque, OrderedDict
import jieba
from django.conf import settings
from xy_neo4j.entity_extract_train.Entity_Mention.integrated_qa_system import IntegratedQASystem

class DialogueManager:
    def __init__(self):
        self.query_stats = deque(maxlen=100)  # 保留最近100次查询数据
        try:
            print("正在初始化问答系统...")
            start_time = time.time()
            
            # 初始化集成问答系统
            NEO4J_CONFIG = {
                "uri": "bolt://localhost:7687",
                "user": "neo4j", 
                "password": "wswy0129"
            }
            
            try:
                # 初始化问答系统
                self.qa_system = IntegratedQASystem(
                    neo4j_config=NEO4J_CONFIG,
                    llm_api_key="sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
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
            # 1. 问题分解阶段 - 强制禁用缓存
            decomp_start_time = time.time()
            print("[DialogueManager] 开始问题分解...")
            
            # 强制禁用缓存，确保每次都执行分解
            if hasattr(self, 'qa_system') and self.qa_system and hasattr(self.qa_system, 'question_decomposer'):
                self.qa_system.question_decomposer.question_cache = {}
            
            sub_questions = self.decompose_question(question)
            decomp_time = time.time() - decomp_start_time
            print(f"[DialogueManager] 问题分解完成，耗时: {decomp_time:.2f}秒")
            if sub_questions:
                print("[DialogueManager] 子问题列表:")
                for i, q in enumerate(sub_questions, 1):
                    print(f"  {i}. {q}")
            
            # 2. 知识图谱查询阶段
            print("\n[DialogueManager] 开始知识图谱查询...")
            kg_start_time = time.time()
            kg_context = self.get_kg_context(question)
            kg_time = time.time() - kg_start_time
            print(f"[DialogueManager] 知识图谱查询完成，耗时: {kg_time:.2f}秒")
            
            # 3. 提取知识图谱节点
            print("[DialogueManager] 开始提取知识图谱节点...")
            kg_nodes = self._extract_kg_nodes_for_vis(kg_context)
            print(f"[DialogueManager] 知识图谱节点统计:")
            print(f"  - 节点数量: {len(kg_nodes.get('nodes', []))}")
            print(f"  - 关系数量: {len(kg_nodes.get('links', []))}")
            
            # 4. 子问题回答阶段
            print("\n[DialogueManager] 开始处理子问题回答...")
            sub_answers = {}
            if sub_questions and isinstance(sub_questions, list) and len(sub_questions) > 0:
                for i, sub_q in enumerate(sub_questions, 1):
                    try:
                        print(f"[DialogueManager] 处理子问题 {i}/{len(sub_questions)}: {sub_q}")
                        sub_ans = self._generate_sub_answer(sub_q)
                        sub_answers[sub_q] = sub_ans
                    except Exception as e:
                        print(f"[DialogueManager] 子问题 {i} 处理失败: {e}")
                        sub_answers[sub_q] = "处理此部分时出错"
            
            # 5. 最终回答生成阶段
            print("\n[DialogueManager] 开始生成最终回答...")
            gen_start_time = time.time()
            final_answer = self.generate_response(question, kg_context)
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
            
            # 组装最终结果
            return {
                "answer": final_answer,
                "sub_questions": sub_questions,
                "sub_answers": sub_answers,
                "kg_context": kg_context,
                "kg_nodes": kg_nodes,
                "time_analysis": time_analysis,
                "thinking_process": self._format_thinking_process(sub_questions, kg_context, sub_answers)
            }
            
        except Exception as e:
            print(f"[DialogueManager] 错误: {str(e)}")
            return {
                "answer": f"抱歉，系统处理您的问题时遇到了错误: {str(e)}",
                "sub_questions": None,
                "sub_answers": None,
                "kg_context": None,
                "kg_nodes": None,
                "time_analysis": {"总思考时间": f"{time.time() - overall_start_time:.2f}秒"},
                "thinking_process": None
            }

    def _generate_sub_answer(self, sub_question: str) -> str:
        """生成子问题的简短回答"""
        try:
            # 如果子问题太短或不是问题，返回空字符串
            if len(sub_question) < 5 or "?" not in sub_question and "？" not in sub_question:
                return ""
                
            # 使用模型生成简短回答
            prompt = f"""请用一句简短的话回答这个问题: {sub_question}
            注意：
            1. 回答必须少于50个字
            2. 直接给出要点，不要有多余的修饰语
            3. 确保回答专业准确"""
            
            # 调用API生成回答
            if self.qa_system:
                return self.qa_system.answer_question(
                    question=prompt,
                    use_decomposition=False,
                    force_refresh=True
                )[:100]  # 截断过长回答
            else:
                # 使用传统方法
                return self.zhipu.get_deepseek_response(prompt)[:100]
                
        except Exception as e:
            print(f"生成子问题简短回答失败: {e}")
            return "无法处理此子问题"

    def _extract_kg_nodes_for_vis(self, kg_context: str) -> dict:
        """从知识图谱上下文中提取节点用于可视化"""
        try:
            nodes = []
            links = []
            categories = []
            category_map = {}
            
            if isinstance(kg_context, str) and kg_context.startswith('['):
                kg_data = json.loads(kg_context)
                
                for item in kg_data:
                    if 'path' in item:
                        # 解析路径 "A → 关系 → B" 格式
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
                                        'draggable': True
                                    })
                            
                            # 添加关系
                            links.append({
                                'source': source,
                                'target': target,
                                'name': relation,
                                'value': relation
                            })
                            
                            # 如果有摘要，添加到节点属性中
                            if 'summary' in item:
                                for node in nodes:
                                    if node['name'] == source or node['name'] == target:
                                        node['desc'] = item['summary']
            
            # 确保返回有效的数据结构
            result = {
                'nodes': nodes if nodes else [],
                'links': links if links else [],
                'categories': categories if categories else []
            }
            print(f"[DialogueManager] 提取的知识图谱数据: {json.dumps(result, ensure_ascii=False)[:200]}...")
            return result
                
        except Exception as e:
            print(f"[DialogueManager] 提取知识图谱节点失败: {e}")
            # 返回有效的空数据结构
            return {
                'nodes': [],
                'links': [],
                'categories': []
            }
        
    def _format_thinking_process(self, sub_questions, kg_context, sub_answers=None):
        """格式化思考过程以便前端展示"""
        thinking = []
        
        # 添加子问题分解
        if sub_questions:
            thinking_item = {
                "title": "问题分解",
                "content": sub_questions
            }
            if sub_answers:  # 如果有子问题答案，添加到结果中
                thinking_item["answers"] = sub_answers
            thinking.append(thinking_item)
        
        # 添加知识图谱检索结果
        if kg_context:
            try:
                # 如果是JSON字符串，尝试解析
                if isinstance(kg_context, str) and kg_context.startswith('['):
                    kg_data = json.loads(kg_context)
                    thinking.append({
                        "title": "知识图谱检索",
                        "content": kg_data
                    })
                else:
                    thinking.append({
                        "title": "知识图谱检索",
                        "content": kg_context if isinstance(kg_context, str) else str(kg_context)
                    })
            except:
                thinking.append({
                    "title": "知识图谱检索",
                    "content": str(kg_context)
                })
                
        return thinking
    
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
        4. 保持黄河流域数据特性
        """
        
        return self.zhipu.get_deepseek_response(prompt)

    def get_kg_context(self, question):
        """获取知识图谱上下文 - 使用集成系统的能力"""
        if self.qa_system and hasattr(self.qa_system, 'text_assembler'):
            try:
                # 使用集成系统的文本组装器
                kg_text = self.qa_system.text_assembler.assemble_text(question)
                return kg_text
            except Exception as e:
                print(f"使用集成系统获取知识图谱上下文失败: {e}")
                # 回退到原有方法
                return self._legacy_get_kg_context(question)
        else:
            # 使用原有方法
            return self._legacy_get_kg_context(question)
    
    def _legacy_get_kg_context(self, question):
        """原始知识图谱查询方法"""
        # 简化的返回，实际可能需要更复杂的实现
        return "使用传统方法查询知识图谱，未找到相关结果。"

    def generate_response(self, question, context):
        """生成最终回答"""
        if self.qa_system:
            try:
                # 使用集成问答系统回答
                return self.qa_system.answer_question(
                    question=question,
                    use_decomposition=len(question) > 20,  # 较长问题使用分解
                    force_refresh=True  # 强制刷新避免缓存
                )
            except Exception as e:
                print(f"使用集成系统生成回答失败: {e}")
                # 回退到原有方法
                return self._legacy_generate_response(question, context)
        else:
            # 使用原有方法
            return self._legacy_generate_response(question, context)
    
    def _legacy_generate_response(self, question, context):
        """原始回答生成方法"""
        prompt = f"""
        基于以下背景知识：
        {context}
        
        请回答问题: {question}
        
        要求：
        1. 回答要准确、专业，采用结构化方式
        2. 说明具体的建模步骤和技术路线
        3. 提供2-3篇参考文献
        """
        
        return self.zhipu.get_deepseek_response(prompt)

# 单例模式工厂类
class DialogueManagerFactory:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DialogueManager()
        return cls._instance