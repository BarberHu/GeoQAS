import json
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
            # 初始化实体识别器
            from LLM_mention_final import DirectMentionRecognizer
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
            except ImportError:
                self.question_decomposer = None
                
        except Exception as e:
            print(f"IntegratedQASystem 初始化失败: {str(e)}")
            raise
    
    # 其余方法保持不变
    def answer_question(self, question: str, use_decomposition: bool = True) -> str:
        """回答问题的主方法"""
        if use_decomposition and self.question_decomposer:
            return self._answer_with_decomposition(question)
        else:
            return self._answer_single_question(question)
    
    def _answer_single_question(self, question: str) -> str:
        """回答单个问题"""
        prompt = self.text_assembler.assemble_text(question)
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        self.text_assembler.update_context(question, answer)
        return answer
    
    def _answer_with_decomposition(self, question: str) -> str:
        """使用问题分解回答复杂问题"""
        decomposed_questions = self.question_decomposer.decompose_question(question)
        formatted_questions = self.question_decomposer.format_questions_for_qa(decomposed_questions)
        
        sub_answers = []
        for sub_question in formatted_questions:
            sub_answer = self._answer_single_question(sub_question)
            sub_answers.append({"question": sub_question, "answer": sub_answer})
        
        integration_prompt = f"""
        基于以下对复杂问题"{question}"的分解问答，提供一个整合的完整回答：
        
        {"".join([f'问题：{qa["question"]}\n回答：{qa["answer"]}\n\n' for qa in sub_answers])}
        
        请提供一个连贯、全面的回答，避免重复信息，并确保覆盖所有关键点。
        """
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": integration_prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        
        final_answer = response.choices[0].message.content
        self.text_assembler.update_context(question, final_answer)
        return final_answer
    
    def generate_cypher_query(self, question: str) -> str:
        """生成Cypher查询语句"""
        prompt = self.text_assembler.get_cypher_query(question)
        
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def execute_query(self, query: str) -> List[Dict]:
        """执行Cypher查询"""
        with self.entity_linker.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]