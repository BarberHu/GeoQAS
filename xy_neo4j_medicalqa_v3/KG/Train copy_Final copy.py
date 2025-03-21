import json
import os
from openai import OpenAI
from typing import Dict, Any
import hashlib

# 添加基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录

class ConfigLoader:
    @staticmethod
    def load_structure(file_path: str) -> str:
        """加载图谱结构描述文件"""
        full_path = os.path.join(BASE_DIR, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
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

    def _build_system_prompt(self, structure_file: str) -> str:
        """构建系统提示"""
        structure = ConfigLoader.load_structure(structure_file)
        return f"""
        你是一个地理学知识图谱构建专家，请严格按照以下要求处理输入文档：

        # 结构定义
        {structure}

        # 处理规则
        1. 必须从文本中提取文献信息，生成source_article字段，格式："标题 (DOI或期刊号)"
        2. 所有实体必须包含source_article字段
        3. 地理问题实体必须唯一，其他实体允许多个
        4. name字段保持精炼，desc字段详细
        5. 严格使用预定义的关系类型，禁止新增类型
        6. 空数组用[]表示，空字符串用""表示
        7. 开发步骤的"下一步"字段必须对应后续步骤的name值
        8. 案例中的 "source_article": "Integrating Smart Grids, Energy Storage, and Renewable Forecasting: A Comprehensive Review and Case Studies (10.1016/j.renene.2024.00235)"也是从文本提取出来的,我并没给案例中的所有实体加上,但请在处理时全部加上

        # 示例参考
        "source_article": "基于SWAT模型的黄河流域水文模拟 (10.1234/abcd5678)"
        """

    def process_document(self, text: str) -> Dict:
        """处理单篇文档"""
        try:
            # 提取文献信息
            article_info = self._extract_article_info(text)
            article_signature = f"{article_info['title']} ({article_info['doi']})"
            
            # 使用 Deepseek API
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"示例数据：\n{self.examples}"},
                    {"role": "user", "content": f"输入文档：\n{text[:8000]}"}
                ],
                stream=False,
                temperature=0.1,
                max_tokens=8000
            )

            content = response.choices[0].message.content
            # 清理JSON字符串
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            try:
                result = json.loads(content)
                return self._validate_and_inject_source(result, article_signature)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}")
                return {}

        except Exception as e:
            print(f"处理失败: {str(e)}")
            return {}

    def _extract_article_info(self, text: str) -> Dict:
        """从文本提取文献元数据"""
        # 简单示例实现，实际应增强解析逻辑
        return {
            "title": text.split("\n")[0].strip()[:50],
            "doi": hashlib.md5(text.encode()).hexdigest()[:10]
        }

    def _validate_and_inject_source(self, data: Dict, source: str) -> Dict:
        """校验并注入文献来源"""
        def process_node(node):
            if isinstance(node, dict):
                # 校验必要字段
                for field in ["type", "name", "desc", "source_article"]:
                    if field not in node:
                        raise ValueError(f"缺少必要字段: {field}")
                
                # 注入文献来源
                node["source_article"] = source
                
                # 递归处理子节点
                for k, v in node.items():
                    if isinstance(v, list):
                        for item in v:
                            process_node(item)
            return node

        return process_node(data)


class DocumentProcessor:
    def __init__(self, engine: KnowledgeGraphEngine):
        self.engine = engine

    def process_files(self, input_dir: str, output_dir: str):
        """批量处理文本文件"""
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            if not filename.endswith(".txt") or not os.path.isfile(input_path):
                continue

            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                result = self.engine.process_document(text)
                
                if result:  # 如果有结果则保存
                    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    print(f"成功处理: {filename}")
                else:
                    print(f"处理失败 {filename}: 无有效结果")
                    
            except Exception as e:
                print(f"处理失败 {filename}: {str(e)}")


if __name__ == "__main__":
    # 配置参数
    API_KEY = "sk-e38ac2aefd1345538e35919fc794aef5"
    STRUCTURE_FILE = os.path.join(BASE_DIR, "图谱结构描述1.txt")
    EXAMPLE_FILE = os.path.join(BASE_DIR, "一步到位.json")
    INPUT_DIR = r"E:\毕业设计\数据库\论文数据\TXT"
    OUTPUT_DIR = r"E:\毕业设计\数据库\论文数据\提取_JSON"

    # 初始化引擎
    engine = KnowledgeGraphEngine(API_KEY, STRUCTURE_FILE, EXAMPLE_FILE)
    processor = DocumentProcessor(engine)

    # 执行处理
    processor.process_files(INPUT_DIR, OUTPUT_DIR)