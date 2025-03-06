"""
基于零样本大模型与动态知识库的通用mention识别系统
核心功能：无需训练数据，动态适配各领域实体识别
"""

import asyncio
import math
from functools import lru_cache
from typing import List, Dict, Tuple
from rapidfuzz import process, fuzz
from transformers import pipeline
from neo4j import GraphDatabase

# --------------------- 配置部分 ---------------------
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "wswy0129"
}

MODEL_CONFIG = {
    "ner_models": [
        {"name": "bert-base-chinese", "type": "bert"},  # 通用中文BERT
        {"name": "BAAI/bge-m3", "type": "text2text"},   # 多语言大模型
        {"name": "facebook/bart-large-mnli", "type": "zero-shot"}  # 零样本分类
    ],
    "confidence_weights": {  # 多维度置信度权重
        "model": 0.4,
        "context": 0.3,
        "kb_popularity": 0.2,
        "domain": 0.1
    }
}

# --------------------- 核心类定义 ---------------------
class DynamicLexicon:
    """动态知识库词典管理器"""
    def __init__(self, neo4j_config: Dict):
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"])
        )
        self.cache = {}  # 实体缓存：{entity_name: {'degree': 10, 'labels': ['地理实体']}}

    def _fetch_related_entities(self, text: str) -> Dict:
        """从知识库查询相关实体"""
        with self.driver.session() as session:
            result = session.run(
                # 修正后的Cypher查询
                """
                MATCH (n) 
                WHERE n.name CONTAINS $text 
                RETURN n.name as name, labels(n) as labels, 
                    COUNT{(n)--()} as degree
                """,  # 使用COUNT{}替代size()
                text=text[:20]
            )
            return {record["name"]: {"labels": record["labels"], "degree": record["degree"]} for record in result}

    def update_lexicon(self, text: str):
        """根据输入文本动态更新词典"""
        # 步骤1：模糊匹配缓存
        cached_matches = process.extract(text, self.cache.keys(), scorer=fuzz.partial_ratio, score_cutoff=70)
        
        # 步骤2：查询知识库补充新实体
        new_entities = self._fetch_related_entities(text)
        
        # 合并更新缓存（LRU策略）
        self.cache.update(new_entities)
        if len(self.cache) > 10000:  # 最大缓存限制
            self.cache = dict(list(self.cache.items())[-5000:])

class EnsembleNER:
    """多模型集成实体识别器"""
    def __init__(self, model_config: Dict):
        self.models = []
        for m in model_config["ner_models"]:
            try:
                pipe = pipeline(
                    task="ner" if m["type"] == "bert" else "text2text-generation",
                    model=m["name"],
                    device_map="cpu"  # 调用cpu
                )
                self.models.append(pipe)
            except Exception as e:
                print(f"模型{m['name']}加载失败: {str(e)}")
    
    def _vote_fusion(self, results: List[List[Dict]]) -> List[Tuple[str, float]]:
        """多模型结果投票融合"""
        entity_scores = {}
        for model_result in results:
            for ent in model_result:
                text = ent["word"].strip()
                score = ent["score"] if "score" in ent else 0.8  # 默认置信度
                if text in entity_scores:
                    entity_scores[text] = max(entity_scores[text], score)
                else:
                    entity_scores[text] = score
        return sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)

    async def async_predict(self, text: str) -> List[Tuple[str, float]]:
        """异步实体识别"""
        results = []
        for model in self.models:
            try:
                if model.task == "ner":
                    res = model(text)
                else:  # 生成式模型处理
                    prompt = f"请从以下文本中提取所有重要实体，用逗号分隔：'{text}'"
                    gen = model(prompt, max_length=50)
                    res = [{"word": e.strip(), "score": 1.0} for e in gen[0]["generated_text"].split(",")]
                results.append(res)
            except Exception as e:
                print(f"模型推理错误: {str(e)}")
        return self._vote_fusion(results)

class MentionRecognizer:
    """mention识别总控制器"""
    def __init__(self, neo4j_config: Dict, model_config: Dict):
        self.lexicon = DynamicLexicon(neo4j_config)
        self.ner = EnsembleNER(model_config)
        self.domain_keywords = self._load_domain_keywords()  # 预加载领域关键词
        
    def _load_domain_keywords(self) -> List[str]:
        """加载领域关键词（示例为地理领域）"""
        return ["流域", "高程", "径流", "DEM", "降水量", "模型"]
    
    def _calculate_confidence(self, entity: str, context: str, kb_info: Dict) -> float:
        """计算综合置信度"""
        # 1. 模型置信度
        model_score = self.ner.last_scores.get(entity, 0.5)
        
        # 2. 上下文连贯性
        context_words = context.split()
        pos = context.find(entity)
        prev_word = context_words[max(0, pos-1)] if pos > 0 else ""
        next_word = context_words[min(len(context_words)-1, pos+1)] if pos < len(context_words)-1 else ""
        context_coherence = 1.0 if any(kw in (prev_word+next_word) for kw in self.domain_keywords) else 0.3
        
        # 3. 知识库热度
        kb_score = math.log(kb_info.get("degree", 1) + 1) / 10  # 归一化
        
        # 4. 领域适配度
        domain_score = 0.7 if any(kw in entity for kw in self.domain_keywords) else 0.3
        
        # 加权综合
        weights = MODEL_CONFIG["confidence_weights"]
        return (
            weights["model"] * model_score +
            weights["context"] * context_coherence +
            weights["kb_popularity"] * kb_score +
            weights["domain"] * domain_score
        )
    
    @lru_cache(maxsize=5000)
    async def recognize(self, text: str) -> List[Dict]:
        """主识别方法"""
        # 步骤1：动态更新词典
        self.lexicon.update_lexicon(text)
        
        # 步骤2：多模型异步预测
        model_entities = await self.ner.async_predict(text)
        
        # 步骤3：知识库模糊匹配
        kb_candidates = process.extract(text, self.lexicon.cache.keys(), 
                                      scorer=fuzz.token_set_ratio, score_cutoff=60)
        
        # 步骤4：结果融合与过滤
        combined = []
        seen = set()
        
        # 处理模型结果
        for ent, score in model_entities:
            if len(ent) < 2:  # 过滤短实体
                continue
            combined.append({
                "entity": ent,
                "source": "model",
                "confidence": self._calculate_confidence(ent, text, self.lexicon.cache.get(ent, {}))
            })
            seen.add(ent)
        
        # 处理知识库匹配结果
        for ent, score, _ in kb_candidates:
            if ent not in seen:
                combined.append({
                    "entity": ent,
                    "source": "kb",
                    "confidence": self._calculate_confidence(ent, text, self.lexicon.cache.get(ent, {}))
                })
                seen.add(ent)
        
        # 动态阈值调整
        threshold = 0.7 - 0.1 * math.exp(-len(text)/15)
        return sorted(
            [item for item in combined if item["confidence"] >= threshold],
            key=lambda x: x["confidence"], 
            reverse=True
        )

# --------------------- 使用示例 ---------------------
async def main():
    # 初始化识别器
    recognizer = MentionRecognizer(NEO4J_CONFIG, MODEL_CONFIG)
    
    # 示例问题
    questions = [
        "我现在有玉龙喀什河流域2005-2010年的降水数据和DEM高程数据，如何进行径流模拟？",
        "松辽流域使用TOPMODEL需要哪些输入参数？",
        "淮河流域洪水预报模型的数据精度要求是什么？"
    ]
    
    # 批量处理
    tasks = [recognizer.recognize(q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    # 输出结果
    for q, ents in zip(questions, results):
        print(f"\n问题：{q}")
        for ent in ents[:3]:  # 显示top3结果
            print(f"- 实体：{ent['entity']} | 来源：{ent['source']} | 置信度：{ent['confidence']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())