import json
from django.conf import settings
from .get_zhipu_response import GetZhipuResponse
from myneo4j.pyneo_utils import get_all_relation
# 在 DialogueManager 中添加性能监控
import time
from collections import deque

class DialogueManager:
    def __init__(self):
        self.query_stats = deque(maxlen=100)  # 保留最近100次查询数据
        try:
            from django.conf import settings
            self.conversation_history = []
            self.current_focus = None  # 当前对话焦点
            if not hasattr(settings, 'ZHIPU'):
                raise Exception("ZHIPU not found in settings")
            self.zhipu = settings.ZHIPU
            print("DialogueManager 初始化成功")
        except Exception as e:
            print("DialogueManager 初始化失败:", e)
            raise
        
        
    def decompose_question(self, question):
        """将复杂问题分解为子问题"""
        try:
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
            sub_questions = self.zhipu.get_chatglm_response(prompt)
            return sub_questions
        except Exception as e:
            print("问题分解失败:", e)
            return "问题分解失败"
        
    # 修改后的 get_kg_context 方法
    def get_kg_context(self, question):
        """优化后的知识图谱查询"""
        start_time = time.time()
        try:
            # 步骤1：实体抽取
            entities = self.extract_key_entities(question)
            
            # 步骤2：构建精准Cypher查询
            cypher = f"""
            MATCH (n)-[r]->(m)
            WHERE n.name IN {entities} OR m.name IN {entities}
            WITH n, r, m
            ORDER BY r.weight DESC
            LIMIT 20  # 限制返回数量
            RETURN 
                n.name as source,
                m.name as target,
                type(r) as relation,
                n.desc as source_desc,
                m.desc as target_desc
            """
            
            # 步骤3：执行查询并过滤
            kg_data = self.execute_cypher(cypher)
            
            # 步骤4：结果压缩
            compressed = []
            for item in kg_data:
                compressed.append({
                    'path': f"{item['source']} → {item['relation']} → {item['target']}",
                    'summary': f"{item['source_desc'][:50]}... | {item['target_desc'][:50]}..."
                })
            duration = (time.time() - start) * 1000  # 毫秒
            self.query_stats.append({
                'entities': len(entities),
                'nodes': len(compressed),
                'time': duration })
            
            return json.dumps(compressed, ensure_ascii=False)
            
        except Exception as e:
            print("优化后的图谱查询失败:", e)
            return ""
        #时间

    def show_performance(self):
        """显示性能指标（答辩时可实时演示）"""
        avg_time = sum(s['time'] for s in self.query_stats)/len(self.query_stats)
        avg_nodes = sum(s['nodes'] for s in self.query_stats)/len(self.query_stats)
        print(f"""
        知识查询性能报告（基于最近{len(self.query_stats)}次查询）：
        - 平均响应时间：{avg_time:.2f}ms
        - 平均返回节点：{avg_nodes:.1f}个
        - Token节省率：{(1 - avg_nodes/230)*100:.1f}% 
        """)

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
        4. 涉及到的建模步骤部分,采用结构化方式展现
        5. 关于建模方案的回答,提供2-3篇参考文献,首先从知识图谱中获取准确的参考文献,其次进行联网搜索,所有提供的参考文献必须真实有效
        """
        response = self.zhipu.get_chatglm_response(prompt)
        return response 