import torch
import json
import concurrent.futures
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
        
        # 检测GPU是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"EntityLinker 使用设备: {device}")
        
        # 初始化模型
        self.encoder = SentenceTransformer(model_path, device=device)
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
    
    def _get_candidates_batch(self, mentions: List[str], top_k: int = 50) -> Dict[str, List[str]]:
        """批量获取多个mention的候选实体
        
        Args:
            mentions: 实体提及列表
            top_k: 每个实体返回的候选数量
            
        Returns:
            Dict[str, List[str]]: 以mention为键，候选实体列表为值的字典
        """
        # 如果缓存为空，返回空字典
        if not self.cache["entity_names"]:
            print("警告: 实体缓存为空，请检查Neo4j连接和数据")
            return {mention: [] for mention in mentions}
        
        results = {}
        
        # 批量向量编码 - 充分利用GPU
        mention_vecs = self.encoder.encode(mentions, convert_to_tensor=True)
        entity_vecs = self.encoder.encode(self.cache["entity_names"], convert_to_tensor=True)
        
        # 计算所有mention与所有实体的相似度
        scores = torch.matmul(mention_vecs, entity_vecs.T)
        
        # 对每个mention处理top-k
        for i, mention in enumerate(mentions):
            mention_scores = scores[i]
            top_indices = torch.topk(mention_scores, k=min(top_k//2, len(self.cache["entity_names"]))).indices.tolist()
            vector_candidates = [self.cache["entity_names"][idx] for idx in top_indices]
            
            # 模糊匹配
            fuzzy_candidates = process.extract(
                mention, self.cache["entity_names"],
                scorer=fuzz.ratio, limit=top_k//2
            )
            fuzzy_candidates = [c[0] for c in fuzzy_candidates]
            
            # 合并去重
            results[mention] = list(set(vector_candidates + fuzzy_candidates))[:top_k]
        
        return results

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
            if len(mention) > 3:  # 如果实体数量大于3，使用批量处理
                return self.rank_entities_batch_gpu(query, mention, top_k)
            else:
                results = {}
                for m in mention:
                    results[m] = self._rank_single_entity(query, m, top_k)
                return results
        else:
            # 处理单个mention
            return {mention: self._rank_single_entity(query, mention, top_k)}
            
    def _rank_single_entity(self, query: str, mention: str, top_k: int = 5) -> List[Dict]:
        """为单个mention排序候选实体"""
        print(f"\n[Entity Linking] 处理实体: {mention}")
        try:
            # 获取候选实体
            candidates = self._get_candidates(mention, top_k=top_k*2)
            if not candidates:
                print(f"[Entity Linking] 未找到 '{mention}' 的候选实体")
                return []
            
            # 构建 Cypher 查询
            query = f"""
            MATCH (n)
            WHERE n.name IN {json.dumps(candidates)}
            RETURN n.name as entity, n.description as desc, n.category as category
            """
            print(f"[Entity Linking] 执行查询: {query}")
            
            with self.driver.session() as session:
                result = session.run(query)
                entities = []
                for record in result:
                    entity = {
                        'entity': record['entity'],
                        'desc': record['desc'],
                        'category': record['category'],
                        'score': self._calculate_similarity(mention, record['entity'])
                    }
                    entities.append(entity)
                
                # 按相似度排序
                entities.sort(key=lambda x: x['score'], reverse=True)
                entities = entities[:top_k]
                
                print(f"[Entity Linking] 找到匹配实体数量: {len(entities)}")
                for e in entities[:3]:  # 只显示前3个匹配
                    print(f"  - {e['entity']} (得分: {e['score']:.2f})")
                
                return entities
            
        except Exception as e:
            print(f"[Entity Linking] 实体 '{mention}' 处理失败: {e}")
            return []
    
    def rank_entities_batch(self, query: str, mentions: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
        """并行执行多个实体排序
        
        Args:
            query: 用户问题
            mentions: 实体提及列表
            top_k: 每个实体返回的top候选数量
            
        Returns:
            Dict[str, List[Dict]]: 以mention为键，排序后实体列表为值的字典
        """
        results = {}
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(mentions))) as executor:
            # 提交所有任务
            future_to_mention = {executor.submit(self._rank_single_entity, query, mention, top_k): mention for mention in mentions}
            
            # 处理结果
            for future in concurrent.futures.as_completed(future_to_mention):
                mention = future_to_mention[future]
                try:
                    result = future.result()
                    results[mention] = result
                except Exception as e:
                    print(f"处理实体 '{mention}' 时出错: {e}")
                    results[mention] = []
        
        return results
    
    def rank_entities_batch_gpu(self, query: str, mentions: List[str], top_k: int = 5) -> Dict[str, List[Dict]]:
        """使用GPU加速批量处理实体排序
        
        Args:
            query: 用户问题
            mentions: 实体提及列表
            top_k: 每个实体返回的top候选数量
            
        Returns:
            Dict[str, List[Dict]]: 以mention为键，排序后实体列表为值的字典
        """
        # 获取所有候选实体
        all_candidates_by_mention = self._get_candidates_batch(mentions, top_k=top_k*2)
        
        results = {}
        all_inputs = []
        all_candidates = []
        mention_mapping = []
        
        # 准备批量处理的输入
        for mention in mentions:
            candidates = all_candidates_by_mention.get(mention, [])
            if not candidates:
                results[mention] = []
                continue
                
            for candidate in candidates:
                all_inputs.append(self._build_features(query, candidate))
                all_candidates.append((mention, candidate))
        
        if not all_inputs:
            return {mention: [] for mention in mentions}
        
        # 批量编码
        encoded = self.tokenizer(
            all_inputs, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )
        
        # 批量预测
        with torch.no_grad():
            # 如果有GPU，使用GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for k, v in encoded.items():
                encoded[k] = v.to(device)
                
            self.ranking_model = self.ranking_model.to(device)
            scores = self.ranking_model(**encoded).squeeze().cpu().tolist()
        
        # 确保scores是列表
        if not isinstance(scores, list):
            scores = [scores]
        
        # 整理结果
        mention_results = {}
        for (mention, candidate), score in zip(all_candidates, scores):
            if mention not in mention_results:
                mention_results[mention] = []
                
            mention_results[mention].append({
                "entity": candidate,
                "score": score,
                "desc": self.cache["entity_desc"].get(candidate, "")
            })
        
        # 排序并获取top-k
        for mention, candidates in mention_results.items():
            results[mention] = sorted(
                candidates,
                key=lambda x: (x["score"], mention in x["entity"]),
                reverse=True
            )[:top_k]
        
        return results

    def link_entities(self, mentions: List[str]) -> Dict[str, List[Dict]]:
        """链接实体到知识库"""
        print(f"\n[Entity Linking] 开始实体链接: {mentions}")
        results = {}
        
        try:
            for mention in mentions:
                print(f"\n[Entity Linking] 处理实体: {mention}")
                # 构建 Cypher 查询
                query = f"""
                MATCH (n)
                WHERE n.name =~ '(?i).*{mention}.*' OR n.aliases =~ '(?i).*{mention}.*'
                RETURN n.name as entity, n.description as desc, n.category as category
                """
                print(f"[Entity Linking] 执行查询: {query}")
                
                with self.driver.session() as session:
                    result = session.run(query)
                    entities = []
                    for record in result:
                        entity = {
                            'entity': record['entity'],
                            'desc': record['desc'],
                            'category': record['category'],
                            'score': self._calculate_similarity(mention, record['entity'])
                        }
                        entities.append(entity)
                    
                    # 按相似度排序
                    entities.sort(key=lambda x: x['score'], reverse=True)
                    results[mention] = entities
                    
                    print(f"[Entity Linking] 找到匹配实体数量: {len(entities)}")
                    for e in entities[:3]:  # 只显示前3个匹配
                        print(f"  - {e['entity']} (得分: {e['score']:.2f})")
                    
        except Exception as e:
            print(f"[Entity Linking] 实体链接失败: {e}")
        
        return results

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