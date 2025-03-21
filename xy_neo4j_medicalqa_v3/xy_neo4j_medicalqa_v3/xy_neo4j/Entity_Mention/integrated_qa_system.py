import json
import concurrent.futures
from typing import List, Dict, Any
from openai import OpenAI
import os
import sys

# 添加当前目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 配置导入部分增加错误捕获
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from config import NEO4J_CONFIG, API_KEYS, LLM_CONFIG
    from llm_client_factory import LLMClientFactory
    print("配置导入成功")
except Exception as e:
    print(f"配置导入失败: {e}")
    # 使用默认配置
    NEO4J_CONFIG = {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "wswy0129"}
    API_KEYS = {"deepseek": "[PROVIDE_YOUR_API_KEY]"}

from .Entity_Order import EntityLinker
from .DeepSeek_mention import DeepSeekMentionRecognizer  # 导入新的实体识别器

class IntegratedQASystem:
    """集成问答系统 - 整合实体识别、实体链接、文本拼接和问答"""
    
    def __init__(self, neo4j_config=None, llm_api_key=None, provider=None):
        """
        初始化集成问答系统
        
        Args:
            neo4j_config: Neo4j数据库配置
            llm_api_key: LLM API密钥
            provider: LLM提供商名称
        """
        # 使用传入的配置或默认配置
        neo4j_config = neo4j_config or NEO4J_CONFIG
        self.provider = provider or LLM_CONFIG["default_provider"]
        llm_api_key = llm_api_key or API_KEYS[self.provider]
        
        # 获取EntityLinker单例实例
        self.entity_linker = EntityLinker.get_instance(neo4j_config)
        
        try:
            # 使用DeepSeek实体识别器替代原来的智谱实体识别器
            self.entity_recognizer = DeepSeekMentionRecognizer(api_key=llm_api_key, provider=self.provider)
            
            # 初始化文本拼接器
            from TextAssembler import TextAssembler
            self.text_assembler = TextAssembler(
                neo4j_config=neo4j_config,
                entity_recognizer=self.entity_recognizer,
                entity_linker=self.entity_linker
            )
            
            # 使用LLM客户端工厂创建LLM客户端
            self.llm_client = LLMClientFactory.create_client(self.provider)
            
            # 初始化问题分解器(optional)
            try:
                from question_decomposition_module import HydrologicalQuestionDecomposer
                self.question_decomposer = HydrologicalQuestionDecomposer(use_api=True)
            except ImportError as e:
                print(f"问题分解器导入失败: {e}")
                self.question_decomposer = None
                
            # 初始化回答缓存
            self.answer_cache = {}
                
        except Exception as e:
            print(f"IntegratedQASystem 初始化失败: {str(e)}")
            raise
    
    def answer_question(self, question: str, use_decomposition: bool = True, force_refresh: bool = False) -> str:
        """回答问题的主方法
        
        Args:
            question: 用户问题
            use_decomposition: 是否使用问题分解
            force_refresh: 是否强制刷新缓存
            
        Returns:
            问题的回答
        """
        # 重置EntityLinker查询状态
        self.entity_linker.reset_query_state()
        
        # 检查缓存
        cache_key = f"{question}_{use_decomposition}"
        if not force_refresh and cache_key in self.answer_cache:
            return self.answer_cache[cache_key]
            
        # 根据问题复杂度决定是否使用分解
        if use_decomposition and self.question_decomposer:
            # 简单检查问题复杂度，避免对简单问题使用分解
            is_complex = self._is_complex_question(question)
            if is_complex:
                answer = self._answer_with_decomposition(question)
            else:
                print(f"问题 '{question}' 被判断为简单问题，不使用分解。")
                answer = self._answer_single_question(question)
        else:
            answer = self._answer_single_question(question)
            
        # 更新缓存
        self.answer_cache[cache_key] = answer
        return answer
    
    def _is_complex_question(self, question: str) -> bool:
        """判断是否为复杂问题
        
        Args:
            question: 用户问题
            
        Returns:
            是否为复杂问题
        """
        # 1. 文本长度检查
        if len(question) < 20:
            return False
            
        # 2. 关键词检查
        complex_indicators = ["如何", "步骤", "流程", "方法", "多个", "不同的", "为什么", "区别", "比较"]
        complexity_score = sum(1 for indicator in complex_indicators if indicator in question)
        
        # 3. 水文专业词汇检查
        hydro_terms = ["SWAT模型", "径流模拟", "参数率定", "子流域", "HRU", "敏感性分析", "气候变化"]
        hydro_score = sum(1 for term in hydro_terms if term in question)
        
        # 综合评分
        return (complexity_score >= 2) or (hydro_score >= 1 and complexity_score >= 1) or ("流域" in question and "数据" in question)
    
    def _answer_single_question(self, question: str) -> str:
        """
        回答单个问题
        
        Args:
            question: 用户问题
            
        Returns:
            问题的回答
        """
        try:
            # 组装文本
            prompt = self.text_assembler.assemble_text(question)
            
            # 调用LLM
            messages = [{"role": "user", "content": prompt}]
            answer = self.llm_client.chat_completion(messages)
            
            # 更新上下文
            self.text_assembler.update_context(question, answer)
            return answer
            
        except Exception as e:
            print(f"回答问题时出错: {e}")
            return f"抱歉，处理您的问题时遇到了技术问题：{str(e)}"
    
    # 修改answer_with_decomposition方法，实现并行处理
    def _answer_with_decomposition(self, question: str) -> str:
        """
        使用问题分解和并行处理回答复杂问题
        
        Args:
            question: 用户问题
            
        Returns:
            整合后的回答
        """
        # 分解问题
        decomposed_questions = self.question_decomposer.decompose_question(question, use_llm=True)
        formatted_questions = self.question_decomposer.format_questions_for_qa(decomposed_questions)
        
        # 减少问题数量，保留最重要的问题
        formatted_questions = formatted_questions[:4]
        
        # 知识状态，用于累积信息
        knowledge_state = {
            "original_question": question,
            "sub_answers": {},
            "kg_contexts": {}
        }
        
        # 并行获取所有子问题的知识图谱上下文
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(formatted_questions))) as executor:
            futures = {}
            for sub_question in formatted_questions:
                future = executor.submit(self.text_assembler.get_comprehensive_kg_context, sub_question)
                futures[future] = sub_question
            
            for future in concurrent.futures.as_completed(futures):
                sub_question = futures[future]
                try:
                    kg_results = future.result()
                    # 将知识图谱结果转换为文本上下文
                    kg_context = self.text_assembler.format_kg_results_for_context(
                        sub_question, 
                        kg_results.get('entity_details', {}).values()
                    )
                    knowledge_state["kg_contexts"][sub_question] = kg_context
                except Exception as e:
                    print(f"获取{sub_question}的知识图谱上下文失败: {e}")
                    knowledge_state["kg_contexts"][sub_question] = "获取知识图谱信息时出错。"
        
        # 顺序处理子问题，整合先前的答案
        for sub_question in formatted_questions:
            # 构建综合提示，包含知识图谱上下文和先前回答
            prompt = self._build_comprehensive_prompt(sub_question, knowledge_state)
            
            try:
                # 调用LLM
                response = self.llm_client.chat.completions.create(
                    model=LLM_CONFIG["default_model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=LLM_CONFIG["temperature"],
                    max_tokens=LLM_CONFIG["max_tokens"]
                )
                
                sub_answer = response.choices[0].message.content
                knowledge_state["sub_answers"][sub_question] = sub_answer
            except Exception as e:
                print(f"处理子问题'{sub_question}'时出错: {e}")
                knowledge_state["sub_answers"][sub_question] = f"回答此问题时遇到技术问题: {str(e)}"
        
        # 构建最终答案提示
        final_prompt = f"""
        基于以下对原始问题"{question}"的分解问答，提供一个整合的完整回答：
        
        {"".join([f'问题：{q}\n回答：{a}\n\n' for q, a in knowledge_state["sub_answers"].items()])}
        
        请提供一个连贯、全面的回答，避免重复信息，并确保覆盖所有关键点。回答应当流畅自然，不要机械地分点列出子问题。
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_CONFIG["default_model"],
                messages=[{"role": "user", "content": final_prompt}],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"]
            )
            
            final_answer = response.choices[0].message.content
            self.text_assembler.update_context(question, final_answer)
            return final_answer
            
        except Exception as e:
            print(f"生成最终回答时出错: {e}")
            # 如果整合失败，返回所有子问题回答的简单组合
            return "综合回答：\n\n" + "\n\n".join([f"关于「{q}」：\n{a}" for q, a in knowledge_state["sub_answers"].items()])
        
    def generate_cypher_query(self, question: str) -> str:
        """生成Cypher查询语句
        
        Args:
            question: 用户问题
            
        Returns:
            Cypher查询语句
        """
        # 组装文本
        prompt = self.text_assembler.get_cypher_query(question)
        
        try:
            # 调用LLM
            response = self.llm_client.chat.completions.create(
                model=LLM_CONFIG["default_model"],
                messages=[{"role": "user", "content": prompt}],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"生成查询语句时出错: {e}")
            return f"生成查询语句失败: {str(e)}"
    
    def execute_query(self, query: str) -> List[Dict]:
        """执行Cypher查询"""
        print(f"\n[Knowledge Graph] 执行查询: {query}")
        try:
            with self.entity_linker.driver.session() as session:
                result = session.run(query)
                records = [dict(record) for record in result]
                print(f"[Knowledge Graph] 查询返回 {len(records)} 条记录")
                return records
                
        except Exception as e:
            print(f"[Knowledge Graph] 查询失败: {e}")
            return []
    
    def query_knowledge_graph(self, query: str) -> List[Dict]:
        """执行知识图谱查询"""
        print(f"\n[Knowledge Graph] 执行查询: {query}")
        try:
            with self.entity_linker.driver.session() as session:
                result = session.run(query)
                records = [dict(record) for record in result]
                print(f"[Knowledge Graph] 查询返回 {len(records)} 条记录")
                return records
                
        except Exception as e:
            print(f"[Knowledge Graph] 查询失败: {e}")
            return []
        
    # 添加新的辅助方法，用于构建综合提示
    def _build_comprehensive_prompt(self, sub_question: str, knowledge_state: Dict[str, Any]) -> str:
        """
        为子问题构建综合提示，整合KG上下文和先前答案
        
        Args:
            sub_question: 当前子问题
            knowledge_state: 当前的知识状态，包含先前回答和KG上下文
            
        Returns:
            构建好的提示
        """
        prompt_parts = []
        
        # 添加系统提示
        prompt_parts.append("你是一个水文领域专家，专注于SWAT模型和径流模拟相关问题的解答。请基于提供的知识信息回答问题。")
        
        # 添加原始问题上下文
        prompt_parts.append(f"用户的原始问题是：{knowledge_state['original_question']}")
        
        # 添加先前回答（如果有）
        prev_answers = []
        for q, a in knowledge_state['sub_answers'].items():
            # 只包含与当前子问题相关的先前答案
            if self._is_related_question(q, sub_question):
                prev_answers.append(f"问题：{q}\n回答：{a}")
        
        if prev_answers:
            prompt_parts.append("以下是相关子问题的回答，可能对当前问题有帮助：\n" + "\n\n".join(prev_answers))
        
        # 添加知识图谱上下文
        kg_context = knowledge_state['kg_contexts'].get(sub_question)
        if kg_context:
            prompt_parts.append(f"相关知识图谱信息：\n{kg_context}")
        
        # 添加当前问题
        prompt_parts.append(f"当前需要回答的问题是：{sub_question}")
        
        # 添加回答要求
        prompt_parts.append("请根据提供的知识图谱信息和上下文，直接回答上述问题。回答应该专业、准确、简洁。如果知识图谱中没有足够信息，请基于专业知识给出合理回答。")
        
        return "\n\n".join(prompt_parts)

    # 添加辅助方法，判断两个问题是否相关
    def _is_related_question(self, question1: str, question2: str) -> bool:
        """
        判断两个问题是否相关
        
        Args:
            question1: 第一个问题
            question2: 第二个问题
            
        Returns:
            两个问题是否相关的布尔值
        """
        # 提取问题中的关键词
        keywords1 = set()
        keywords2 = set()
        
        # 使用实体提取器获取关键词（实体）
        if hasattr(self.entity_recognizer, 'recognize'):
            keywords1 = set(self.entity_recognizer.recognize(question1))
            keywords2 = set(self.entity_recognizer.recognize(question2))
        
        # 计算关键词重叠度
        if keywords1 and keywords2:
            overlap = keywords1.intersection(keywords2)
            if len(overlap) > 0:
                return True
        
        # 如果没有关键词重叠，检查文本相似度
        # 简单实现：检查共同单词（可以用更复杂的相似度算法替换）
        words1 = set(question1.lower().split())
        words2 = set(question2.lower().split())
        common_words = words1.intersection(words2)
        
        # 如果共享单词超过阈值，认为相关
        threshold = 0.3  # 可调整的阈值
        similarity = len(common_words) / max(len(words1), len(words2))
        
        return similarity >= threshold

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
            self.provider = provider
            
            # 更新LLM客户端
            self.llm_client = LLMClientFactory.create_client(provider)
            
            # 更新实体识别器
            self.entity_recognizer = DeepSeekMentionRecognizer(
                api_key=API_KEYS.get(provider),
                provider=provider
            )
            
            return {
                "success": True, 
                "message": f"已切换到 {provider} API",
                "provider": provider
            }
        except Exception as e:
            return {"success": False, "message": f"切换API失败: {str(e)}"}