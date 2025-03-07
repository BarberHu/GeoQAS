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

class IntegratedQASystem:
    """集成问答系统 - 整合实体识别、实体链接、文本拼接和问答"""
    
    def __init__(self, neo4j_config: Dict[str, str], llm_api_key: str):
        """初始化集成问答系统"""
        try:
            # 修改导入语句
            from LLM_mention_final import DirectMentionRecognizer  # 改用正确的文件名
            self.entity_recognizer = DirectMentionRecognizer(api_key=llm_api_key)
            
            # 初始化实体链接器
            from Entity_Order import EntityLinker
            self.entity_linker = EntityLinker(neo4j_config=neo4j_config)
            
            # 初始化文本拼接器
            from TextAssembler import TextAssembler
            self.text_assembler = TextAssembler(
                neo4j_config=neo4j_config,
                entity_recognizer=self.entity_recognizer,
                entity_linker=self.entity_linker
            )
            
            # 初始化LLM客户端
            self.llm_client = OpenAI(
                api_key=llm_api_key,
                base_url="https://api.chatanywhere.tech/v1"
            )
            
            # 初始化问题分解器(optional)
            try:
                from question_decomposition_module import HydrologicalQuestionDecomposer
                self.question_decomposer = HydrologicalQuestionDecomposer(use_api=False)
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
        """回答单个问题
        
        Args:
            question: 用户问题
            
        Returns:
            问题的回答
        """
        # 组装文本
        prompt = self.text_assembler.assemble_text(question)
        
        try:
            # 调用LLM
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # 更新上下文
            self.text_assembler.update_context(question, answer)
            return answer
            
        except Exception as e:
            print(f"回答单个问题时出错: {e}")
            return f"抱歉，处理您的问题时遇到了技术问题：{str(e)}"
    
    def _answer_with_decomposition(self, question: str) -> str:
        """使用问题分解回答复杂问题
        
        Args:
            question: 用户问题
            
        Returns:
            整合后的回答
        """
        # 分解问题 - 使用规则方法更快
        decomposed_questions = self.question_decomposer.decompose_question(question, use_llm=False)
        formatted_questions = self.question_decomposer.format_questions_for_qa(decomposed_questions)
        
        # 减少问题数量，保留3-4个最重要的问题
        formatted_questions = formatted_questions[:4]
        
        # 批量识别所有子问题的实体
        if hasattr(self.entity_recognizer, 'recognize_batch'):
            # 如果实现了批量识别，使用批量识别
            entities_by_question = self.entity_recognizer.recognize_batch(formatted_questions)
        
        # 并行处理子问题
        sub_answers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(formatted_questions))) as executor:
            future_to_question = {}
            
            for sub_question in formatted_questions:
                future = executor.submit(self._answer_single_question, sub_question)
                future_to_question[future] = sub_question
            
            for future in concurrent.futures.as_completed(future_to_question):
                sub_question = future_to_question[future]
                try:
                    sub_answer = future.result()
                    sub_answers.append({"question": sub_question, "answer": sub_answer})
                except Exception as e:
                    print(f"处理子问题 '{sub_question}' 时出错: {e}")
                    sub_answers.append({"question": sub_question, "answer": f"处理时出错: {e}"})
        
        # 按原始问题顺序排序答案
        sub_answers.sort(key=lambda x: formatted_questions.index(x["question"]))
        
        # 整合所有回答
        integration_prompt = f"""
        基于以下对复杂问题"{question}"的分解问答，提供一个整合的完整回答：
        
        {"".join([f'问题：{qa["question"]}\n回答：{qa["answer"]}\n\n' for qa in sub_answers])}
        
        请提供一个连贯、全面的回答，避免重复信息，并确保覆盖所有关键点。不要分点列出子问题。
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": integration_prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            final_answer = response.choices[0].message.content
            self.text_assembler.update_context(question, final_answer)
            return final_answer
            
        except Exception as e:
            print(f"整合回答时出错: {e}")
            # 如果整合失败，返回单独的回答
            return "\n\n".join([f"关于'{qa['question']}'：\n{qa['answer']}" for qa in sub_answers])
    
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
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
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