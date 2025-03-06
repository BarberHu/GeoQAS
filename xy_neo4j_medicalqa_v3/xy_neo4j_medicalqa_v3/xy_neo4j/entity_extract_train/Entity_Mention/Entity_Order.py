import torch
import json
from typing import List, Dict, Union, Tuple
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz

class EntityLinker:
    """实体链接与排序系统"""
    
    def __init__(self, neo4j_config: Dict, model_path: str = "BAAI/bge-small-zh-v1.5"):
        # 知识库连接
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"])
        )
        
        # 初始化模型
        self.encoder = SentenceTransformer(model_path, device="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.ranking_model = self._init_ranking_model()
        
        # 缓存
        self.cache = {"entity_desc": {}, "entity_names": []}
        self._preload_entities()

    def _preload_entities(self):
        """预加载实体基础信息"""
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN n.name as name, n.desc as desc LIMIT 500")
            for record in result:
                if record["name"]:  # 确保实体名不为空
                    self.cache["entity_desc"][record["name"]] = record.get("desc", "")
            self.cache["entity_names"] = list(self.cache["entity_desc"].keys())

    def _get_candidates(self, mention: str, top_k: int = 50) -> List[str]:
        """获取候选实体"""
        # 如果缓存为空，返回空列表
        if not self.cache["entity_names"]:
            print("警告: 实体缓存为空，请检查Neo4j连接和数据")
            return []
            
        # 阶段1：向量召回
        mention_vec = self.encoder.encode(mention, convert_to_tensor=True)
        entity_vecs = self.encoder.encode(self.cache["entity_names"], convert_to_tensor=True)
        scores = torch.matmul(mention_vec, entity_vecs.T)
        top_indices = torch.topk(scores, k=min(top_k//2, len(self.cache["entity_names"]))).indices.tolist()
        vector_candidates = [self.cache["entity_names"][i] for i in top_indices]

        # 阶段2：模糊匹配
        fuzzy_candidates = process.extract(
            mention, self.cache["entity_names"],
            scorer=fuzz.ratio, limit=top_k//2
        )
        fuzzy_candidates = [c[0] for c in fuzzy_candidates]

        # 合并去重
        return list(set(vector_candidates + fuzzy_candidates))[:top_k]

    def _init_ranking_model(self):
        """初始化轻量级排序模型"""
        class RankingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = AutoModel.from_pretrained("bert-base-chinese")
                self.classifier = torch.nn.Linear(768, 1)
                
            def forward(self, input_ids, attention_mask, token_type_ids=None):
                # 添加token_type_ids参数,但不使用它
                outputs = self.bert(input_ids, attention_mask=attention_mask)
                return self.classifier(outputs.last_hidden_state[:, 0])
        
        model = RankingModel()
        # 加载预训练权重（此处需替换为实际训练路径）
        # model.load_state_dict(torch.load("ranking_model.bin"))
        return model.eval()
    
    def _build_features(self, query: str, candidate: str) -> str:
        """构建特征数据"""
        desc = self.cache["entity_desc"].get(candidate, "")
        # 特征拼接方式1：query + 实体描述
        return f"{query}[SEP]{desc}"
        
        # 特征拼接方式2（可选）：仅实体名
        # return f"{query}[SEP]{candidate}"

    def rank_entities(self, query: str, mention: Union[str, Tuple[str, ...], List[str]], top_k: int = 5) -> Dict[str, List[Dict]]:
        """执行实体排序
        
        Args:
            query: 用户问题
            mention: 单个实体提及(字符串)或多个实体提及(元组或列表)
            top_k: 每个实体返回的top候选数量
            
        Returns:
            Dict[str, List[Dict]]: 以mention为键，排序后实体列表为值的字典
        """
        # 支持处理多个mention (元组或列表形式)
        if isinstance(mention, (tuple, list)):
            results = {}
            for m in mention:
                results[m] = self._rank_single_entity(query, m, top_k)
            return results
        else:
            # 处理单个mention
            return {mention: self._rank_single_entity(query, mention, top_k)}
            
    def _rank_single_entity(self, query: str, mention: str, top_k: int = 5) -> List[Dict]:
        """为单个mention排序候选实体"""
        # 获取候选
        candidates = self._get_candidates(mention)
        if not candidates:
            return []

        # 特征处理
        inputs = [self._build_features(query, cand) for cand in candidates]
        encoded = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        
        # 模型预测
        with torch.no_grad():
            scores = self.ranking_model(**encoded).squeeze().tolist()
            
        # 确保scores是列表，即使只有一个元素
        if not isinstance(scores, list):
            scores = [scores]
        
        # 组合结果
        results = []
        for cand, score in zip(candidates, scores):
            results.append({
                "entity": cand,
                "score": score,
                "desc": self.cache["entity_desc"].get(cand, "")
            })
        
        # 后处理：优先选择包含mention的实体
        return sorted(
            results,
            key=lambda x: (x["score"], mention in x["entity"]),
            reverse=True
        )[:top_k]

# --------------------- 使用示例 ---------------------
if __name__ == "__main__":
    # 配置
    NEO4J_CONFIG = {
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "wswy0129"
    }
    
    linker = EntityLinker(NEO4J_CONFIG)
    
    # 测试案例 - 支持多个mention
    test_cases = [
        ("我现在有淮河流域2015-2020年的气象数据和全国的土壤数据,如何对淮河流域进行径流模拟?", 
         ["淮河流域", "径流模拟", "气象数据", "土壤数据"])
    ]
    
    for query, mentions in test_cases:
        print(f"\n问题: '{query}'")
        results = linker.rank_entities(query, mentions)
        
        # 输出每个mention的排序结果
        for mention, entities in results.items():
            print(f"\n待链接实体: '{mention}'")
            if not entities:
                print(f"  未找到匹配的实体")
                continue
                
            for idx, ent in enumerate(entities, 1):
                print(f"  {idx}. {ent['entity']} (得分: {ent['score']:.2f})")
                print(f"     描述: {ent['desc'][:100]}" + ("..." if len(ent['desc']) > 100 else ""))