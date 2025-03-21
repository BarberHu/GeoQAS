import json
import os
import glob
from openai import OpenAI
from typing import Dict, Any, List, Tuple, Optional, Set
import hashlib
import re
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from functools import lru_cache
import math
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录

# 添加性能优化配置
class PerformanceConfig:
    # 动态窗口大小配置
    MIN_WINDOW_SIZE = 3000
    MAX_WINDOW_SIZE = 8000
    WINDOW_OVERLAP_RATIO = 0.2  # 窗口重叠比例
    
    # 并行处理配置
    MAX_WORKERS = max(2, multiprocessing.cpu_count() - 1)  # 保留一个核心给系统
    BATCH_SIZE = 5  # API批处理大小
    
    # 缓存配置
    CACHE_SIZE = 1000
    
    # API重试配置
    MAX_RETRIES = 3
    BASE_DELAY = 1  # 基础延迟秒数
    MAX_DELAY = 10  # 最大延迟秒数

class ConfigLoader:
    @staticmethod
    @lru_cache(maxsize=10)  # 添加缓存装饰器
    def load_structure(file_path: str) -> str:
        """加载图谱结构描述文件"""
        full_path = os.path.join(BASE_DIR, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    @lru_cache(maxsize=10)  # 添加缓存装饰器
    def load_examples(file_path: str) -> str:
        """加载示例数据"""
        full_path = os.path.join(BASE_DIR, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        return json.dumps(examples[:2], ensure_ascii=False, indent=2)


class KnowledgeGraphEngine:
    def __init__(self, api_key: str, structure_file: str, example_file: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.system_prompt = self._build_system_prompt(structure_file)
        self.examples = ConfigLoader.load_examples(example_file)
        self.source_cache = {}
        self.result_cache = {}  # 添加结果缓存
        
        # 修正实体类型定义，包含所有16种类型 (添加了文献来源)
        self.required_entity_types = {
            "地理问题", "地理场景", "对象系统", "系统机理", "时空数据", 
            "数据来源", "处理方法", "集成模型", "基础模型", "开发步骤", 
            "评价方法", "评价结果", "模型应用", "应用结果", "总结讨论", "文献来源"
        }
        
        # 完善关系映射表，确保所有实体类型都有对应的关系定义
        self.relation_mapping = {
            "地理问题": ["所属场景", "研究对象", "使用数据", "使用模型", "问题结论"],
            "地理场景": [],  # 终端节点，没有进一步的关系
            "对象系统": ["相关机理"],
            "系统机理": [],  # 终端节点，没有进一步的关系
            "时空数据": ["获取来源", "数据处理"],
            "数据来源": [],  # 终端节点，没有进一步的关系
            "处理方法": [],  # 终端节点，没有进一步的关系
            "集成模型": ["集成依赖", "开发流程", "模型评价", "模拟过程"],
            "基础模型": [],  # 终端节点，没有进一步的关系
            "开发步骤": ["下一步"],
            "评价方法": ["评价结果"],
            "评价结果": [],  # 终端节点，没有进一步的关系
            "模型应用": ["运算结果"],
            "应用结果": ["结果讨论"],
            "总结讨论": [],  # 终端节点，没有进一步的关系
            "文献来源": ["引用于"]  # 文献来源节点，指向各个实体类型
        }
        
        # 添加关系与目标实体类型的映射表，增强验证逻辑
        self.relation_target_types = {
            "所属场景": "地理场景",
            "研究对象": "对象系统",
            "使用数据": "时空数据",
            "使用模型": "集成模型",
            "问题结论": "总结讨论",
            "相关机理": "系统机理",
            "获取来源": "数据来源",
            "数据处理": "处理方法",
            "集成依赖": "基础模型",
            "开发流程": "开发步骤",
            "下一步": "开发步骤",
            "模型评价": "评价方法",
            "模拟过程": "模型应用",
            "评价结果": "评价结果",
            "运算结果": "应用结果",
            "结果讨论": "总结讨论",
            "引用于": "地理问题"  # 引用关系可以指向多种实体类型，这里只列出一种示例
        }
        
        # 最大窗口大小和滑动步长
        self.max_window_size = 6000
        self.window_slide = 3000
        
        # 最大重试次数
        self.max_retries = 3

    def _build_system_prompt(self, structure_file: str) -> str:
        """构建系统提示"""
        structure = ConfigLoader.load_structure(structure_file)
        return f"""
        你是一个地理学知识图谱构建专家，请严格按照以下要求处理输入文档：

        # 结构定义
        {structure}

        # 处理规则
        1. 必须提取文献信息，创建文献来源节点
        2. 所有实体必须与文献来源建立"引用于"关系
        3. 地理问题实体必须唯一，其他实体允许多个
        4. name字段保持精炼，desc字段详细
        5. 严格使用预定义的关系类型，禁止新增类型
        6. 空数组用[]表示，空字符串用""表示
        7. 开发步骤的"下一步"字段必须对应后续步骤的name值
        8. 确保每个实体都与文献来源节点有关联
        """

    def process_document_with_sliding_window(self, text: str) -> Dict:
        """使用滑动窗口处理长文本"""
        # 分割文本为多个窗口
        window_size = 6000
        stride = 3000
        windows = []
        start = 0
        
        while start < len(text):
            end = min(start + window_size, len(text))
            windows.append(text[start:end])
            start += stride
        
        print(f"文本已分割为 {len(windows)} 个窗口进行处理 (窗口大小: {window_size}, 步长: {stride})")
        
        # 处理每个窗口
        windows_data = []
        for i, window_text in enumerate(windows, 1):
            print(f"处理文本块 {i}, 尝试 1/3")
            result = self._process_document_direct(window_text)
            if result:
                windows_data.append(result)
        
        # 合并窗口数据
        if windows_data:
            return self._merge_windows_data(windows_data)
        return {}

    def _merge_windows_data(self, windows_data: List[Dict]) -> Dict:
        """合并多个窗口的知识图谱数据"""
        if not windows_data:
            return {}
        
        # 使用第一个窗口的数据作为基础
        merged_data = windows_data[0].copy()
        
        # 定义需要合并的关系类型
        relation_types = [
            "所属场景", "研究对象", "使用数据", "使用模型", 
            "相关机理", "数据处理", "开发流程", "问题结论"
        ]
        
        # 遍历其他窗口的数据
        for window_data in windows_data[1:]:
            # 合并描述（如果新窗口有更详细的描述）
            if len(window_data.get('desc', '')) > len(merged_data.get('desc', '')):
                merged_data['desc'] = window_data['desc']
            
            # 合并各种关系
            for relation_type in relation_types:
                if relation_type in window_data:
                    self._merge_relation(merged_data, relation_type, window_data[relation_type])
        
        return merged_data

    def _merge_relation(self, target: Dict, relation_type: str, new_entities: List[Dict]) -> None:
        """合并关系中的实体"""
        if relation_type not in target:
            target[relation_type] = []
        
        existing_names = {e.get('name', '') for e in target[relation_type]}
        
        for new_entity in new_entities:
            if new_entity.get('name', '') not in existing_names:
                target[relation_type].append(new_entity)
                existing_names.add(new_entity.get('name', ''))
            else:
                # 更新现有实体的描述（如果新描述更详细）
                for existing in target[relation_type]:
                    if existing.get('name') == new_entity.get('name'):
                        if len(new_entity.get('desc', '')) > len(existing.get('desc', '')):
                            existing['desc'] = new_entity['desc']
                        break

    def _process_document_direct(self, text: str) -> Dict:
        """直接处理单个文本块"""
        try:
            # 提取文献信息，创建文献来源节点
            article_info = self._extract_article_info(text)
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"示例数据：\n{self.examples}"},
                    {"role": "user", "content": f"输入文档：\n{text}"}
                ],
                temperature=0.1,
                max_tokens=8000
            )
            
            content = response.choices[0].message.content
            print(f"API返回内容长度: {len(content)} 字符")
            print(f"开始处理文本块的JSON响应...")
            
            # 提取JSON
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if not json_match:
                json_match = re.search(r'\{[\s\S]*\}', content)
            
            if json_match:
                result = json.loads(json_match.group(1) if "```json" in content else json_match.group(0))
                print("文本块的JSON提取成功")
                
                # 添加文献来源节点
                if "文献来源" not in result:
                    result["文献来源"] = [article_info]
                elif isinstance(result["文献来源"], list) and len(result["文献来源"]) == 0:
                    result["文献来源"] = [article_info]
                
                # 处理引用关系
                self._process_citations(result, article_info)
                
                return result
            
            print("无法从响应中提取JSON")
            return {}
            
        except Exception as e:
            print(f"处理文本块时出错: {str(e)}")
            return {}
            
    def _process_citations(self, data: Dict, article_info: Dict) -> None:
        """处理实体与文献来源的引用关系"""
        # 确保文献来源节点存在引用于关系
        for source in data.get("文献来源", []):
            if "引用于" not in source:
                source["引用于"] = []
                
            # 添加对主实体的引用
            source["引用于"].append({
                "type": "地理问题",
                "name": data.get("name", "")
            })
            
        # 递归处理所有实体，为重要实体添加引用关系
        def add_citations(entity, path=[]):
            if not isinstance(entity, dict) or "type" not in entity or "name" not in entity:
                return
                
            # 为关键实体添加引用关系
            if entity["type"] in ["地理场景", "对象系统", "系统机理", "时空数据", "集成模型"]:
                for source in data.get("文献来源", []):
                    ref_exists = False
                    for ref in source.get("引用于", []):
                        if ref.get("type") == entity["type"] and ref.get("name") == entity["name"]:
                            ref_exists = True
                            break
                            
                    if not ref_exists:
                        if "引用于" not in source:
                            source["引用于"] = []
                        source["引用于"].append({
                            "type": entity["type"],
                            "name": entity["name"]
                        })
            
            # 递归处理其他关系
            for relation in self.relation_mapping.get(entity["type"], []):
                if relation in entity and isinstance(entity[relation], list):
                    for child in entity[relation]:
                        add_citations(child, path + [relation])
        
        # 开始递归处理
        add_citations(data)

    def _validate_basic_structure(self, data: Dict) -> bool:
        """简化的基本结构验证"""
        if not isinstance(data, dict) or "type" not in data:
            return False
        
        if data["type"] != "地理问题":
            return False
        
        # 检查基本字段
        if not all(field in data for field in ["name", "desc", "source_article"]):
            return False
        
        # 检查是否至少有一个关系
        has_relation = False
        for relation in self.relation_mapping["地理问题"]:
            if relation in data and isinstance(data[relation], list) and data[relation]:
                has_relation = True
                break
        
        return has_relation

    @lru_cache(maxsize=PerformanceConfig.CACHE_SIZE)
    def _get_cached_response(self, text_hash: str) -> Optional[Dict]:
        """获取缓存的响应"""
        return self.result_cache.get(text_hash)

    def _cache_response(self, text_hash: str, response: Dict) -> None:
        """缓存响应"""
        self.result_cache[text_hash] = response
        
        # 如果缓存过大，删除最旧的条目
        if len(self.result_cache) > PerformanceConfig.CACHE_SIZE:
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]

    def validate_graph_structure(self, data: Dict) -> Dict[str, Any]:
        """验证生成的知识图谱结构是否符合规定"""
        issues = []
        details = {}

        # 基本结构验证
        if not self._validate_basic_structure(data):
            issues.append("知识图谱缺少基本结构")
            return {"valid": False, "issues": issues, "details": {}}

        # 验证必须包含name和desc字段
        for entity_type in self.required_entity_types:
            details[entity_type] = {"count": 0, "missing_fields": []}
            
        # 验证地理问题
        if data["type"] != "地理问题":
            issues.append(f"根节点类型必须是'地理问题'，当前是'{data['type']}'")
        
        # 验证必须字段 - 不再检查source_article
        if not all(field in data for field in ["name", "desc"]):
            missing = [field for field in ["name", "desc"] if field not in data]
            issues.append(f"地理问题实体缺少必要字段: {', '.join(missing)}")
        
        # 详细验证每个实体和关系
        self._validate_entity_and_relations(data, issues)
        
        # 返回验证结果
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "details": details
        }

    def _validate_entity_and_relations(self, entity: Dict, issues: List[str], path: str = "root") -> None:
        """递归验证实体及其关系"""
        # 1. 验证基本实体属性
        if not isinstance(entity, dict):
            issues.append(f"{path} 不是有效的实体对象")
            return
            
        # 2. 验证type字段
        if "type" not in entity:
            issues.append(f"{path} 缺少type字段")
            return
            
        entity_type = entity["type"]
        if entity_type not in self.required_entity_types:
            issues.append(f"{path} 的实体类型 '{entity_type}' 不在预定义列表中")
            return
            
        # 3. 验证必要字段
        for field in ["name", "desc"]:
            if field not in entity:
                issues.append(f"{path} ({entity_type}) 缺少 {field} 字段")
            elif not entity[field]:  # 字段为空
                issues.append(f"{path} ({entity_type}) 的 {field} 字段为空")
                
        # 4. 验证关系字段
        valid_relations = self.relation_mapping.get(entity_type, [])
        
        for rel in valid_relations:
            if rel in entity:
                if isinstance(entity[rel], list):
                    # 处理列表类型的关系
                    for i, child in enumerate(entity[rel]):
                        child_path = f"{path}.{rel}[{i}]"
                        self._validate_entity_and_relations(child, issues, child_path)
                elif rel == "下一步" and isinstance(entity[rel], str):
                    # 特殊处理"下一步"关系，它只是一个字符串引用
                    if not entity[rel]:
                        issues.append(f"{path} 的 {rel} 关系引用为空字符串")
                else:
                    issues.append(f"{path} 的 {rel} 关系格式错误，应为数组")
                    
        # 5. 验证是否有未定义的关系
        for key in entity.keys():
            if key not in ["type", "name", "desc"] and key not in valid_relations:
                if key != "引用于" or entity_type != "文献来源":  # 特殊处理文献来源的引用关系
                    issues.append(f"{path} 包含未定义的关系或属性: {key}")

    def _extract_article_info(self, text: str) -> Dict[str, Any]:
        """从文本中提取文献信息，创建文献来源节点"""
        # 检查缓存
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()
        if text_hash in self.source_cache:
            return self.source_cache[text_hash]
        
        # 构建提示
        prompt = f"""
        请从以下学术文本中提取文献信息，返回JSON格式的标题和DOI号（如果有）。
        如果没有找到DOI，请尝试提取其他标识符如期刊号。
        只需返回JSON，不要有任何其他文字。

        ```
        {text[:3000]}
        ```

        格式：
        {{
          "title": "文章标题",
          "doi": "DOI号或其他标识符"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            # 提取JSON部分
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                article_info = json.loads(match.group(0))
                if "title" in article_info and "doi" in article_info:
                    # 创建文献来源节点
                    source_node = {
                        "type": "文献来源",
                        "name": article_info["title"],
                        "desc": f"{article_info['title']} ({article_info['doi']})"
                    }
                    # 缓存结果
                    self.source_cache[text_hash] = source_node
                    return source_node
        
        except Exception as e:
            print(f"提取文献信息出错: {str(e)}")
        
        # 如果提取失败，返回默认文献来源节点
        default_source = {
            "type": "文献来源",
            "name": "未知文献",
            "desc": "从文本中未能提取到文献信息"
        }
        self.source_cache[text_hash] = default_source
        return default_source

    def process_document(self, text: str) -> Dict:
        """处理单篇文档，自动选择处理方式"""
        if len(text) > 6000:  # 如果文本较长，使用滑动窗口
            return self.process_document_with_sliding_window(text)
        else:  # 否则使用普通处理
            return self._process_document_direct(text)

    # 新方法：迭代式知识提取
    def iterative_knowledge_extraction(self, text: str, max_iterations: int = 3) -> Dict:
        """通过多轮迭代提高知识提取质量"""
        print(f"开始迭代式知识提取，最大迭代次数: {max_iterations}")
        
        # 第一轮提取
        extracted_data = self.process_document_with_sliding_window(text)
        
        for iteration in range(2, max_iterations + 1):
            print(f"开始第 {iteration}/{max_iterations} 轮迭代提取")
            
            # 验证结果
            validation = self.validate_graph_structure(extracted_data)
            if validation["valid"]:
                print(f"第 {iteration-1} 轮提取的结果已通过验证，无需进一步迭代")
                break
            
            # 获取验证问题
            issues = validation["issues"]
            
            # 构建改进提示
            improvement_prompt = f"""
            我已经从文献中提取了以下知识图谱数据，但存在一些问题需要修复:
            
            问题: {', '.join(issues[:3])}...等{len(issues)}个问题
            
            当前提取的数据:
            ```json
            {json.dumps(extracted_data, ensure_ascii=False, indent=2)}
            ```
            
            请修复上述问题，返回完整修复后的JSON数据。修复内容应该基于以下文本:
            
            ```
            {text[:6000]}
            ```
            
            请严格遵循原有的结构定义，不要删除有效数据，只需修复问题。
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": improvement_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=8000
                )
                
                content = response.choices[0].message.content
                # 提取JSON部分
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if not json_match:
                    json_match = re.search(r'\{[\s\S]*\}', content)
                
                if json_match:
                    improved_data = json.loads(json_match.group(1) if "```json" in content else json_match.group(0))
                    extracted_data = improved_data
                    print(f"第 {iteration} 轮迭代提取完成")
                else:
                    print(f"第 {iteration} 轮迭代未能提取有效JSON，使用上一轮结果")
            except Exception as e:
                print(f"第 {iteration} 轮迭代处理出错: {str(e)}")
                break
        
        return extracted_data

    # 新方法：人机协作接口
    def human_in_the_loop_extraction(self, text: str, expert_feedback_function=None) -> Dict:
        """
        人机协作知识提取流程
        
        Parameters:
        - text: 待处理的文本
        - expert_feedback_function: 专家反馈函数，接收当前提取结果，返回修改建议
        
        Returns:
        - 最终的知识图谱数据
        """
        # 初始自动提取
        extracted_data = self.process_document_with_sliding_window(text)
        
        # 如果没有提供专家反馈函数，直接返回结果
        if expert_feedback_function is None:
            print("未提供专家反馈函数，使用自动提取结果")
            return extracted_data
        
        # 获取专家反馈
        print("正在获取专家反馈...")
        feedback = expert_feedback_function(extracted_data)
        
        if not feedback:
            print("专家未提供反馈，使用自动提取结果")
            return extracted_data
        
        print(f"收到专家反馈: {feedback[:100]}..." if len(feedback) > 100 else feedback)
        
        # 根据专家反馈修改
        try:
            correction_prompt = f"""
            我已经从文献中提取了以下知识图谱数据:
            
            ```json
            {json.dumps(extracted_data, ensure_ascii=False, indent=2)}
            ```
            
            专家提供了以下反馈意见:
            {feedback}
            
            请根据专家反馈修改知识图谱数据，返回完整修复后的JSON。修改应该基于以下文本内容:
            
            ```
            {text[:6000]}
            ```
            
            请严格遵循原有的结构定义，确保所有实体都有正确的关系链接。
            """
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.1,
                max_tokens=8000
            )
            
            content = response.choices[0].message.content
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if not json_match:
                json_match = re.search(r'\{[\s\S]*\}', content)
            
            if json_match:
                corrected_data = json.loads(json_match.group(1) if "```json" in content else json_match.group(0))
                print("已根据专家反馈修改知识图谱")
                return corrected_data
            else:
                print("未能从修正结果中提取JSON，使用原始提取结果")
                return extracted_data
                
        except Exception as e:
            print(f"根据专家反馈修改时出错: {str(e)}")
            return extracted_data

    def batch_process_documents(self, input_dir: str, output_dir: str, max_files: int = -1) -> None:
        """批量处理文档目录"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有.txt文件
        txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
        if max_files > 0:
            txt_files = txt_files[:max_files]
        
        print(f"找到 {len(txt_files)} 个txt文件准备处理")
        
        # 处理进度统计
        successful = 0
        failed = 0
        skipped = 0
        
        # 处理每个文件
        for i, txt_file in enumerate(txt_files):
            file_name = os.path.basename(txt_file)
            output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.json")
            
            # 如果输出文件已存在，跳过处理
            if os.path.exists(output_file):
                print(f"[{i+1}/{len(txt_files)}] 文件 {file_name} 已处理，跳过")
                skipped += 1
                continue
            
            print(f"\n{'='*80}")
            print(f"[{i+1}/{len(txt_files)}] 开始处理文件: {file_name}")
            print(f"{'='*80}")
            
            try:
                # 读取文本文件
                with open(txt_file, 'r', encoding='utf-8', errors='replace') as f:
                    text = f.read()
                
                # 使用迭代式知识提取处理文本
                result = self.iterative_knowledge_extraction(text)
                
                # 保存结果
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                print(f"[{i+1}/{len(txt_files)}] 文件 {file_name} 处理成功，已保存到 {output_file}")
                successful += 1
                
            except Exception as e:
                print(f"[{i+1}/{len(txt_files)}] 处理文件 {file_name} 出错: {str(e)}")
                failed += 1
                
                # 保存错误日志
                error_log_file = os.path.join(output_dir, "error_log.txt")
                with open(error_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{file_name}: {str(e)}\n")
        
        print(f"\n处理总结：")
        print(f"总文件数: {len(txt_files)}")
        print(f"成功处理: {successful} 文件")
        print(f"处理失败: {failed} 文件")
        print(f"已跳过: {skipped} 文件")


# 测试人机协作的简单实现示例
def mock_expert_feedback(data):
    """模拟专家反馈函数"""
    feedback = "请确保地理问题的描述更加准确，并检查是否有研究对象与相关机理的连接缺失。"
    return feedback


def get_api_key(api_key=None):
    """获取API密钥，优先使用参数，其次环境变量，最后使用默认值"""
    if api_key:
        return api_key
    
    # 尝试从环境变量获取
    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    
    # 返回默认值
    return "sk-e38ac2aefd1345538e35919fc794aef5"  # 用户提供的默认密钥

# 使用示例
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='知识图谱构建工具')
    parser.add_argument('--input_dir', type=str, default='./input',
                      help='输入TXT文本数据目录')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='输出JSON数据目录')
    parser.add_argument('--api_key', type=str, default=None,
                      help='Deepseek API密钥')
    parser.add_argument('--max_files', type=int, default=-1,
                      help='处理的最大文件数，-1表示处理所有文件')
    parser.add_argument('--structure_file', type=str, default="图谱结构描述1.txt",
                      help='图谱结构描述文件')
    parser.add_argument('--example_file', type=str, default="一步到位.json",
                      help='示例文件')
    args = parser.parse_args()
    
    try:
        # 获取API密钥
        api_key = get_api_key(args.api_key)
        
        # 初始化知识图谱引擎
        engine = KnowledgeGraphEngine(
            api_key=api_key,
            structure_file=args.structure_file,
            example_file=args.example_file
        )
        
        # 批量处理文档
        engine.batch_process_documents(
            input_dir=args.input_dir, 
            output_dir=args.output_dir,
            max_files=args.max_files
        )
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()
