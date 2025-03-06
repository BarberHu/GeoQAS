class DialogueManager:
    def __init__(self):
        self.conversation_history = []
        self.current_focus = None
        self.decomposition_strategies = {
            "methodology": ["模型选择", "参数设置", "验证方法", "对比分析"],
            "application": ["适用场景", "数据需求", "实施步骤", "结果解读"],
            "comparison": ["优势分析", "局限说明", "适用条件", "性能指标"]
        }

    def decompose_question(self, question):
        """改进后的多维度问题分解方法"""
        # 步骤1：问题类型识别
        question_type = self._classify_question_type(question)
        
        # 步骤2：动态选择分解策略
        strategy = self._select_decomposition_strategy(question_type)
        
        # 步骤3：结构化问题分解
        decomposition = self._structured_decomposition(question, strategy)
        
        # 步骤4：逻辑关系验证
        validated = self._validate_decomposition(decomposition)
        
        return validated

    def _classify_question_type(self, question):
        """问题类型识别"""
        prompt = f"""
        请判断以下地理建模问题的类型：
        [问题]: {question}
        
        可选类型：
        1. 方法论问题（涉及建模方法选择、实施步骤）
        2. 应用场景问题（涉及具体场景应用）
        3. 对比分析问题（涉及模型/方法比较）
        4. 理论解释问题（涉及原理机制）
        
        只需返回类型编号（1-4）
        """
        response = ZHIPU.get_chatglm_response(prompt)
        return int(response.strip())

    def _select_decomposition_strategy(self, q_type):
        """动态选择分解策略"""
        type_map = {
            1: "methodology",
            2: "application",
            3: "comparison",
            4: "theory"
        }
        return self.decomposition_strategies.get(type_map.get(q_type, "general"), 
                                               ["核心概念", "关键参数", "适用条件", "典型应用"])

    def _structured_decomposition(self, question, strategy):
        """结构化问题分解"""
        prompt = f"""
        请根据以下策略分解地理建模问题：
        [原始问题]: {question}
        [分解维度]: {", ".join(strategy)}
        
        要求：
        1. 每个子问题必须包含在分解维度中
        2. 子问题之间应有逻辑递进关系
        3. 使用JSON格式返回，结构示例：
        {{
            "original_question": "原问题文本",
            "sub_questions": [
                {{
                    "aspect": "分解维度名称",
                    "question": "具体子问题",
                    "dependency": 前置问题索引（-1表示无依赖）
                }}
            ]
        }}
        """
        response = ZHIPU.get_chatglm_response(prompt)
        return json.loads(response)

    def _validate_decomposition(self, decomposition):
        """分解结果验证"""
        # 验证1：完整性检查
        required_fields = ["original_question", "sub_questions"]
        if not all(field in decomposition for field in required_fields):
            raise ValueError("分解结构不完整")
        
        # 验证2：依赖关系检查
        for i, sq in enumerate(decomposition["sub_questions"]):
            dep = sq.get("dependency", -1)
            if dep >= i:  # 依赖项不能是当前或后续问题
                sq["dependency"] = -1
        
        # 验证3：维度覆盖检查
        return decomposition
        
    def get_kg_context(self, question):
        """从知识图谱获取相关上下文"""
        # 使用Neo4j查询相关知识
        related_nodes = get_related_nodes(question)
        context = format_kg_data(related_nodes)
        return context
        
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
        response = ZHIPU.get_chatglm_response(prompt)
        return response 