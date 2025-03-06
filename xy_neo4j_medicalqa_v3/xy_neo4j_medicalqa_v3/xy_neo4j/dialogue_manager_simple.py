import json
from django.conf import settings
from .get_zhipu_response import GetDeepseekResponse
from ..myneo4j.pyneo_utils import get_all_relation
import time
from collections import deque, OrderedDict
import jieba
from functools import lru_cache
from py2neo import Graph
import hashlib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from cachetools import LRUCache
from .entity_extract_train.Entity_Mention.integrated_qa_system import IntegratedQASystem

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

class LRUCache(OrderedDict):
    def __init__(self, maxsize=100):
        super().__init__()
        self.maxsize = maxsize
        
    def get(self, key, default=None):
        try:
            value = self[key]
            self.move_to_end(key)
            return value
        except KeyError:
            return default

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.maxsize:
            self.popitem(last=False)

class DialogueManager:
    def __init__(self):
        self.query_stats = deque(maxlen=100)
        try:
            # 初始化集成问答系统
            NEO4J_CONFIG = {
                "uri": "bolt://localhost:7687",
                "user": "neo4j", 
                "password": "wswy0129"
            }
            
            # 初始化问答系统
            self.qa_system = IntegratedQASystem(
                neo4j_config=NEO4J_CONFIG,
                llm_api_key="sk-benW8QASpqo6tXfDsE9Eu6vYxJDhTtHeeeKGSh11wBOqW8SA"
            )
            
            # 初始化其他组件
            if not hasattr(settings, 'ZHIPU'):
                raise Exception("ZHIPU not found in settings")
            self.zhipu = settings.ZHIPU
            jieba.initialize()
            self.execute_cypher = self.execute_query_with_fallback
            self.cache = LRUCache(maxsize=100)
            print("DialogueManager 初始化成功")
            
        except Exception as e:
            print("DialogueManager 初始化失败:", e)
            raise

    def get_response(self, question: str) -> dict:
        """处理用户问题并返回回答"""
        start_time = time.time()
        try:
            # 1. 使用集成问答系统生成答案
            answer = self.qa_system.answer_question(
                question=question,
                use_decomposition=True  # 默认启用问题分解
            )
            
            # 2. 计算思考时间
            thinking_time = time.time() - start_time
            
            # 3. 格式化最终答案
            formatted_answer = f"""
{answer}

-----------------------------------
思考时间：{thinking_time:.2f}秒 | 查询层级：知识图谱+大模型协同
"""
            
            return {
                "answer": formatted_answer,
                "sub_questions": None,  # 如果需要显示问题分解结果，可以从qa_system获取
                "kg_context": "使用知识图谱+大模型协同回答"  # 用于前端展示
            }
            
        except Exception as e:
            print(f"处理问题失败: {str(e)}")
            return {
                "answer": "抱歉，系统处理您的问题时遇到了错误，请稍后重试。",
                "sub_questions": None,
                "kg_context": None
            }

    def decompose_question(self, question):
        """结构化问题分解器（支持多轮对话流程）"""
        try:
            prompt = f"""
            【问题分析任务】
            请将水文建模问题分解为可执行的知识单元，遵循以下规则：
            
            输入问题："{question}"
            
            【分解规则】
            1. 按建模流程的5个阶段分层解析
            2. 每个子问题必须包含"问题焦点"和"预期输出"
            3. 使用水文专业术语（SWAT、DEM预处理等）
            4. 保持黄河流域数据特性
            5. 不要使用#或*等特殊字符
            
            【输出格式】
            问题分解：
            [阶段1] 目标定义
            - 焦点：<明确研究目标>
            - 需求：<时空范围说明>
            - 示例：是否需要对比不同情景模拟？
            
            [阶段2] 数据准备
            - 现有数据：<用户提供的数据清单>
            - 缺失数据：<需要补充的数据类型>
            - 预处理：<数据加工步骤>
            
            [阶段3] 模型适配
            - 候选模型：<推荐2-3个模型>
            - 选择依据：<模型特性对比>
            - 参数建议：<关键参数范围>
            
            [阶段4] 实施路径
            - 步骤流：<编号的步骤列表>
            - 工具链：<推荐软件工具>
            - 检查点：<关键验证节点>
            
            [阶段5] 验证评估
            - 指标：<NSE、R²等>
            - 方法：<交叉验证方式>
            - 基准：<参考数据集>
            """

            example_response = """
            问题分解：
            [阶段1] 目标定义
            - 焦点：建立黄河流域日尺度径流模拟方案
            - 需求：空间范围覆盖干流及汾河、渭河等主要支流，时间跨度2000-2020
            - 示例：需考虑土地利用变化对径流的影响
            
            [阶段2] 数据准备
            - 现有数据：90m SRTM DEM、12站点气象数据
            - 缺失数据：土壤类型图、土地利用演变数据、河道断面测量数据
            - 预处理：DEM填洼处理、气象数据空间插值、数据归一化
            
            [阶段3] 模型适配
            - 候选模型：SWAT(分布式)、HEC-HMS(半分布式)、XAJ(集总式)
            - 选择依据：SWAT适合长期模拟但需要详细土壤数据，HEC-HMS适合洪水过程线
            - 参数建议：CN值取65-78、曼宁系数0.035-0.06
            
            [阶段4] 实施路径
            - 步骤流：
              1. 基于DEM提取河网水系
              2. 划分子流域和HRU
              3. 气象数据空间离散化
              4. 参数敏感性分析
              5. 率定期验证(2000-2010)
              6. 验证期测试(2011-2020)
            - 工具链：ArcSWAT + Python + SWAT-CUP
            - 检查点：Nash系数>0.65、水量平衡误差<15%
            
            [阶段5] 验证评估
            - 指标：NSE=0.72、R²=0.81、RMSE=1.2m³/s
            - 方法：留出法验证、极端降水事件测试
            - 基准：水文年鉴2000-2020径流记录
            
            注：以上分解基于典型水文模拟案例，具体参数和步骤需根据实际情况调整。
            """

            full_prompt = prompt + "\n示例回答：\n" + example_response
            return self.zhipu.get_deepseek_response(full_prompt)
        
        except Exception as e:
            print(f"问题分解失败：{str(e)}")
            return {
                "error": "QE001",
                "message": "问题解析器异常，建议重新表述问题",
                "retry_template": "请用以下格式提问：[研究目标]+[已有数据]+[时间要求]"
            }
        
    def get_kg_context(self, question):
        """分层查询知识图谱"""
        start_time = time.time()
        try:
            # 第一层：精准实体查询
            entities = self.extract_key_entities(question)
            kg_data = self.precise_query(entities)
            
            if not kg_data or len(kg_data) < 5:
                print("精准查询结果不足，进入扩展查询")
                # 第二层：扩展路径查询
                expanded_data = self.expanded_query(entities)
                if expanded_data:  # 添加空值检查
                    kg_data.extend(expanded_data)
            
            if not kg_data or len(kg_data) < 3:
                print("扩展查询结果不足，进入语义查询")
                # 第三层：语义相似查询
                similar_nodes = self.semantic_search(question)
                fallback_data = self.fallback_query(similar_nodes)
                if fallback_data:  # 添加空值检查
                    kg_data.extend(fallback_data)

            # 结果格式化前检查数据
            if not kg_data:
                print("语义查询无结果，使用大模型回答")
                # 第四层：直接使用大模型
                return self.get_llm_response(question)
            
            try:
                # 结果格式化
                compressed = []
                for item in kg_data[:20]:
                    if not isinstance(item, dict):
                        print(f"跳过非字典项: {item}")
                        continue
                        
                    try:
                        compressed.append({
                            'path': f"{item.get('source', '未知')} → {item.get('relation', '关系')} → {item.get('target', '未知')}",
                            'summary': f"{item.get('source_desc', '')} | {item.get('target_desc', '')}"
                        })
                    except Exception as e:
                        print(f"处理项目失败: {item}, 错误: {e}")
                        continue
                
                if not compressed:
                    print("没有有效的知识图谱数据")
                    return self.get_llm_response(question)
                
                return json.dumps(compressed, ensure_ascii=False)
            except Exception as format_error:
                print("结果格式化失败:", format_error)
                return self.get_llm_response(question)

        except Exception as e:
            print("知识图谱查询失败:", e)
            return self.get_llm_response(question)

    def precise_query(self, entities):
        """精准实体查询，包含权重计算"""
        cache_key = f"precise_{','.join(sorted(entities))}"
        if cached := self.cache.get(cache_key):
            return cached

        cypher = f"""
        MATCH path=(n)-[r]->(m)
        WHERE n.name IN {entities} OR m.name IN {entities}
        // 计算权重的多个因素
        WITH n, r, m,
            // 1. 关系频率权重
            CASE WHEN type(r) IN ['使用工具', '应用于', '包含步骤'] THEN 0.3 ELSE 0.1 END +
            // 2. 节点度数权重（归一化）
            1.0 * size((n)--()) / 20 +
            // 3. 属性完整度权重
            CASE 
                WHEN n.desc IS NOT NULL AND m.desc IS NOT NULL THEN 0.2
                WHEN n.desc IS NOT NULL OR m.desc IS NOT NULL THEN 0.1
                ELSE 0
            END +
            // 4. 直接关联权重
            CASE WHEN n.name IN {entities} AND m.name IN {entities} THEN 0.4 ELSE 0 END
        as weight
        
        ORDER BY weight DESC
        LIMIT 15
        RETURN 
            n.name as source,
            m.name as target,
            type(r) as relation,
            n.desc as source_desc,
            m.desc as target_desc,
            weight
        """
        result = self.execute_cypher(cypher)
        self.cache.put(cache_key, result)
        return result

    def expanded_query(self, entities, max_hops=2):
        """扩展路径查询"""
        cypher = f"""
        MATCH path=(n)-[r*1..{max_hops}]->(m)
        WHERE n.name IN {entities} OR m.name IN {entities}
        UNWIND relationships(path) as rel
        RETURN DISTINCT
            startNode(rel).name as source,
            endNode(rel).name as target,
            type(rel) as relation,
            startNode(rel).desc as source_desc,
            endNode(rel).desc as target_desc
        LIMIT 10
        """
        return self.execute_cypher(cypher)

    def semantic_search(self, question):
        """基于语义的相似节点搜索"""
        # 这里可以接入词向量或其他语义相似度计算
        # 暂时使用简单的关键词匹配
        words = jieba.cut(question)
        return [f"'{word}'" for word in words if len(word) > 1]

    def fallback_query(self, similar_nodes):
        """语义回退查询"""
        cypher = f"""
        MATCH (n)-[r]->(m)
        WHERE n.name IN {similar_nodes} OR m.name IN {similar_nodes}
        RETURN 
            n.name as source,
            m.name as target,
            type(r) as relation,
            n.desc as source_desc,
            m.desc as target_desc
        LIMIT 10
        """
        return self.execute_cypher(cypher)

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
        """生成回答，结合知识图谱和大模型的优势"""
        start_time = time.time()
        
        if "注意：" in context:
            # 纯大模型回答的情况
            prompt = f"""
            {context}
            
            基于以上背景，请回答问题: {question}
            要求：
            1. 回答要准确、专业，采用结构化方式
            2. 说明具体的建模步骤和技术路线
            3. 涉及到的建模步骤部分,采用结构化方式展现
            4. 关于建模方案的回答,提供2-3篇参考文献,首先从知识图谱中获取准确的参考文献,其次进行联网搜索,所有提供的参考文献必须真实有效
            5. 提示本回答为大模型回答,请根据问题和背景,给出详细的回答
            6. 告知用户,本回答为大模型回答
            7. 回答时,不要使用#或*等特殊字符    
            8. 对于建模步骤,请生成详细合理的流程图图片
            """
        else:
            # 结合知识图谱和大模型的回答
            prompt = f"""
            基于以下地理建模知识图谱中的信息:
            {context}
            
            请回答问题: {question}
            
            要求:
            1. 首先总结知识图谱中的相关信息，说明其对问题的参考价值
            2. 在此基础上，补充更完整的专业解答：
               - 完善技术路线
               - 补充具体步骤
               - 添加必要的解释
            3. 采用结构化方式展现内容
            4. 提供2-3篇参考文献：
               - 优先使用知识图谱中提到的文献
               - 补充其他相关的重要文献
               - 回答中涉及的参考文献,必须真实有效
            5. 如果知识图谱信息有限，请明确指出哪些是补充的内容
            6. 回答时,不要使用#或*等特殊字符    
            7. 对于建模步骤,请生成详细合理的流程图图片

            """
        
        response = self.zhipu.get_deepseek_response(prompt)
        
        # 计算思考时间
        thinking_time = time.time() - start_time
        
        # 添加思考时间说明
        response_with_time = f"""
{response}

-----------------------------------
思考时间：{thinking_time:.2f}秒 | 查询层级：{'大模型直接回答' if '注意：' in context else '知识图谱+大模型补充'}
"""
        return response_with_time

    def execute_query_with_fallback(self, cypher):
        """使用 get_all_relation 作为备选查询方法"""
        try:
            # 首先尝试使用 get_all_relation
            result = get_all_relation("", "", "")
            if 'datas' in result:
                return result['datas']
            return []
        except Exception as e:
            print("查询执行失败:", e)
            return [] 

    def extract_key_entities(self, text):
        """提取关键实体"""
        try:
            # 使用jieba分词
            words = jieba.lcut(text)
            
            # 过滤停用词和短词
            filtered_words = []
            for word in words:
                if len(word) > 1 and not word.isspace():
                    filtered_words.append(word)
            
            # 转换为Neo4j查询格式
            entities = [f"'{word}'" for word in filtered_words]
            
            # 如果没有提取到实体，返回空列表
            if not entities:
                print(f"未从文本中提取到实体: {text}")
                return []
            
            print(f"提取到的实体: {entities}")
            return entities
        
        except Exception as e:
            print("实体提取失败:", e)
            return [] 

    def get_llm_response(self, question):
        """直接使用大模型生成回答的上下文"""
        prompt = f"""
        请基于你的专业知识，详细回答关于"{question}"的问题。
        要求：
        1. 回答要准确、专业，重点说明技术路线和方法步骤
        2. 采用结构化方式展现内容
        3. 如果是具体建模方案：
           - 说明数据需求
           - 解释技术路线
           - 描述关键步骤
           - 提供评估方法
        4. 建议2-3篇可供参考的重要文献
        5. 回答时,不要使用#或*等特殊字符    
        6. 对于建模步骤,请生成详细合理的流程图图片
        
        """
        context = self.zhipu.get_deepseek_response(prompt)
        return f"""
        注意：图谱中不存在与"{question}"直接相关的信息，以下内容来自大模型的专业知识：
        {context}
        """ 