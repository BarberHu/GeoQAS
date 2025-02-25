from django.conf import settings
from .get_zhipu_response import GetZhipuResponse
from myneo4j.pyneo_utils import get_all_relation

class DialogueManager:
    def __init__(self):
        self.conversation_history = []
        self.current_focus = None  # 当前对话焦点
        self.zhipu = settings.ZHIPU  # 从 settings 中获取 ZHIPU 实例
        
    def decompose_question(self, question):
        """将复杂问题分解为子问题"""
        # 使用大模型将问题分解为多个子问题
        prompt = f"""
        请将以下地理建模相关问题分解为多个具体的子问题:
        问题: {question}
        按照以下类型进行分解:
        1. 建模目标
        2. 数据需求
        3. 模型选择
        4. 具体步骤
        5. 评价方法
        """
        # 调用智谱AI进行问题分解
        sub_questions = self.zhipu.get_chatglm_response(prompt)
        return sub_questions
        
    def get_kg_context(self, question):
        """从知识图谱获取相关上下文"""
        try:
            # 使用现有的 Neo4j 查询函数
            kg_data = get_all_relation("", "", "")
            context = f"""
            节点信息：{kg_data.get('datas', [])}
            关系信息：{kg_data.get('links', [])}
            """
            return context
        except Exception as e:
            print("获取知识图谱上下文失败:", e)
            return ""
        
    def generate_response(self, question, context):
        """生成回答"""
        prompt = f"""
        基于以下地理建模知识图谱中的信息:
        {context}
        
        请回答问题: {question}
        要求:
        1. 回答要准确、专业
        2. 必须基于提供的知识图谱信息
        3. 如果信息不足，请明确指出
        """
        response = self.zhipu.get_chatglm_response(prompt)
        return response 