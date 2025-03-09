import torch
import json
import concurrent.futures
from typing import List, Dict, Union, Tuple, Any, Optional
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz
import time

class EntityLinker:
    """实体链接与排序系统 - 优化版"""
    
    def __init__(self, neo4j_config: Dict, model_path: str = "BAAI/bge-small-zh-v1.5"):
        # 首先初始化缓存属性
        self.cache = {
            "entity_desc": {},
            "entity_names": [],
            "entity_vectors": None,  # 用于存储预计算的实体向量
            "query_results": {},     # 用于存储查询结果
            "candidate_cache": {}    # 用于存储候选实体
        }
        
        # 知识库连接 - 使用连接池优化
        self.driver = GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"]),
            max_connection_lifetime=3600,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60
        )
        
        # 检测GPU是否可用
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"EntityLinker 使用设备: {device}")
        
        # 初始化模型
        self.encoder = SentenceTransformer(model_path, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.ranking_model = self._init_ranking_model()
        
        # 预加载实体信息
        self._preload_entities()
        
        print(f"EntityLinker初始化完成，缓存包含属性: {list(self.cache.keys())}")

    def _preload_entities(self):
        """预加载实体基础信息并计算向量"""
        print("开始预加载实体信息和计算实体向量...")
        start_time = time.time()
        
        with self.driver.session() as session:
            # 批量加载前500个实体的基本信息
            result = session.run("""
                MATCH (n) 
                WHERE n.name IS NOT NULL
                RETURN n.name as name, n.desc as desc, 
                       labels(n)[0] as category, 
                       n.source_article as reference
                LIMIT 500
            """)
            
            entities = []
            for record in result:
                name = record["name"]
                if name:
                    self.cache["entity_desc"][name] = record.get("desc", "")
                    self.cache["entity_names"].append(name)
                    entities.append({
                        "name": name,
                        "desc": record.get("desc", ""),
                        "category": record.get("category", "Unknown"),
                        "reference": record.get("reference", "")
                    })
        
        # 预计算实体向量（仅在有实体时进行）
        if self.cache["entity_names"]:
            try:
                self.cache["entity_vectors"] = self.encoder.encode(
                    self.cache["entity_names"], 
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
            except Exception as e:
                print(f"预计算实体向量失败: {e}")
                self.cache["entity_vectors"] = None
        
        end_time = time.time()
        print(f"预加载完成，共加载 {len(self.cache['entity_names'])} 个实体，耗时: {end_time - start_time:.2f}秒")

    def query_with_cache(self, cypher_query: str, parameters: Dict) -> List[Dict]:
        """使用缓存执行Cypher查询"""
        # 生成缓存键
        cache_key = f"{cypher_query}_{json.dumps(parameters, sort_keys=True)}"
        
        # 检查缓存
        if cache_key in self.cache["query_results"]:
            return self.cache["query_results"][cache_key]
        
        # 执行查询
        with self.driver.session() as session:
            result = session.run(cypher_query, **parameters)
            data = [dict(record) for record in result]
        
        # 更新缓存
        self.cache["query_results"][cache_key] = data
        return data

    def batch_fetch_entities(self, entity_names: List[str]) -> Dict[str, Dict]:
        """批量获取多个实体的详细信息"""
        if not entity_names:
            return {}
            
        # 过滤已缓存的实体
        uncached_entities = [e for e in entity_names if e not in self.cache["entity_desc"]]
        
        # 如果有未缓存的实体，执行查询
        if uncached_entities:
            with self.driver.session() as session:
                query = """
                MATCH (n)
                WHERE n.name IN $entity_names
                RETURN n.name as name, n.desc as desc, 
                       labels(n)[0] as category, 
                       n.source_article as reference
                """
                
                result = session.run(query, entity_names=uncached_entities)
                
                for record in result:
                    name = record["name"]
                    if name:
                        self.cache["entity_desc"][name] = record.get("desc", "")
                        if name not in self.cache["entity_names"]:
                            self.cache["entity_names"].append(name)
        
        # 收集所有请求的实体信息
        entity_details = {}
        for name in entity_names:
            if name in self.cache["entity_desc"]:
                entity_details[name] = {
                    "name": name,
                    "desc": self.cache["entity_desc"].get(name, ""),
                }
        
        # 批量获取关系
        if entity_details:
            with self.driver.session() as session:
                query = """
                MATCH (n)-[r]-(m)
                WHERE n.name IN $entity_names
                RETURN n.name as source_name, 
                       type(r) as relation_type, 
                       m.name as target_name,
                       startNode(r) = n as is_outgoing
                """
                
                result = session.run(query, entity_names=list(entity_details.keys()))
                
                for record in result:
                    source = record["source_name"]
                    if source in entity_details:
                        if "relations" not in entity_details[source]:
                            entity_details[source]["relations"] = []
                            
                        entity_details[source]["relations"].append({
                            "type": record["relation_type"],
                            "entity": record["target_name"],
                            "direction": "outgoing" if record["is_outgoing"] else "incoming"
                        })
        
        return entity_details

    def _get_candidates_batch_optimized(self, mentions: List[str], top_k: int = 50) -> Dict[str, List[str]]:
        """优化版批量获取候选实体"""
        if not self.cache["entity_names"]:
            print("警告: 实体缓存为空，请检查Neo4j连接和数据")
            return {mention: [] for mention in mentions}
        
        # 检查缓存
        cached_results = {}
        uncached_mentions = []
        
        for mention in mentions:
            cache_key = f"{mention}_{top_k}"
            if cache_key in self.cache["candidate_cache"]:
                cached_results[mention] = self.cache["candidate_cache"][cache_key]
            else:
                uncached_mentions.append(mention)
        
        # 如果所有mention都已缓存，直接返回
        if not uncached_mentions:
            return cached_results
        
        results = cached_results.copy()
        
        # 批量向量编码
        try:
            mention_vecs = self.encoder.encode(uncached_mentions, convert_to_tensor=True)
            
            # 向量检索（使用预计算的实体向量）
            if self.cache["entity_vectors"] is not None:
                scores = torch.matmul(mention_vecs, self.cache["entity_vectors"].T)
                
                # 为每个mention处理top-k
                for i, mention in enumerate(uncached_mentions):
                    mention_scores = scores[i]
                    top_k_half = min(top_k//2, len(self.cache["entity_names"]))
                    top_indices = torch.topk(mention_scores, k=top_k_half).indices.tolist()
                    vector_candidates = [self.cache["entity_names"][idx] for idx in top_indices]
                    
                    # 模糊匹配 - 并行处理
                    fuzzy_candidates = process.extract(
                        mention, self.cache["entity_names"],
                        scorer=fuzz.ratio, limit=top_k//2
                    )
                    fuzzy_candidates = [c[0] for c in fuzzy_candidates]
                    
                    # 合并去重
                    combined = list(set(vector_candidates + fuzzy_candidates))[:top_k]
                    results[mention] = combined
                    
                    # 更新缓存
                    self.cache["candidate_cache"][f"{mention}_{top_k}"] = combined
            else:
                # 回退到单独处理
                for mention in uncached_mentions:
                    results[mention] = self._get_candidates(mention, top_k)
        except Exception as e:
            print(f"批量获取候选实体时出错: {e}")
            # 回退到单独处理
            for mention in uncached_mentions:
                results[mention] = self._get_candidates(mention, top_k)
        
        return results

    def _get_candidates(self, mention: str, top_k: int = 50) -> List[str]:
        """获取候选实体"""
        # 优先使用缓存
        cache_key = f"{mention}_{top_k}"
        if cache_key in self.cache["candidate_cache"]:
            return self.cache["candidate_cache"][cache_key]
            
        # 如果缓存为空，返回空列表
        if not self.cache["entity_names"]:
            print("警告: 实体缓存为空，请检查Neo4j连接和数据")
            return []
        
        try:
            # 阶段1：向量召回
            mention_vec = self.encoder.encode(mention, convert_to_tensor=True)
            
            # 检查是否有预计算的实体向量
            if self.cache["entity_vectors"] is not None:
                entity_vecs = self.cache["entity_vectors"]
            else:
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
            combined = list(set(vector_candidates + fuzzy_candidates))[:top_k]
            
            # 更新缓存
            self.cache["candidate_cache"][cache_key] = combined
            
            return combined
        except Exception as e:
            print(f"获取候选实体时出错: {e}")
            # 回退到简单的模糊匹配
            try:
                fuzzy_candidates = process.extract(
                    mention, self.cache["entity_names"],
                    scorer=fuzz.ratio, limit=top_k
                )
                candidates = [c[0] for c in fuzzy_candidates]
                
                # 更新缓存
                self.cache["candidate_cache"][cache_key] = candidates
                
                return candidates
            except Exception as e2:
                print(f"模糊匹配失败: {e2}")
                return []
    
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
            
            # 构建 Cypher 查询 - 修改查询以确保获取source_article
            query_str = """
            MATCH (n)
            WHERE n.name IN $candidates
            WITH n, 
                 CASE WHEN n.desc IS NOT NULL THEN n.desc
                      ELSE '' END as description,
                 CASE WHEN labels(n)[0] IS NOT NULL THEN labels(n)[0]
                      ELSE 'Unknown' END as category,
                 CASE WHEN n.source_article IS NOT NULL THEN n.source_article
                      ELSE '' END as source
            RETURN n.name as entity, 
                  description as desc, 
                  category,
                  source as reference
            ORDER BY n.name
            """
            print(f"[Entity Linking] 执行查询: {query_str}")
            
            # 使用缓存执行查询
            try:
                result = self.query_with_cache(query_str, {"candidates": candidates})
            except Exception as e:
                print(f"缓存查询失败: {e}")
                # 回退到直接查询
                with self.driver.session() as session:
                    result = session.run(query_str, candidates=candidates)
                    result = [dict(record) for record in result]
            
            entities = []
            for record in result:
                entity = {
                    'entity': record['entity'],
                    'desc': record['desc'],
                    'category': record['category'],
                    'reference': record['reference'],  # 确保包含参考文献
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
        start_time = time.time()
        print(f"[Entity Linking] 开始批量处理 {len(mentions)} 个实体...")
        
        results = {}
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(mentions))) as executor:
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
        
        end_time = time.time()
        print(f"[Entity Linking] 批量处理完成，耗时: {end_time - start_time:.2f}秒")
        
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
        start_time = time.time()
        print(f"[Entity Linking] 开始GPU批量处理 {len(mentions)} 个实体...")
        
        try:
            # 检查GPU可用性
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[Entity Linking] 使用设备: {device}")
            
            # 批量获取候选实体
            try:
                all_candidates = self._get_candidates_batch_optimized(mentions, top_k=top_k*2)
                print(f"[Entity Linking] 成功获取候选实体，平均每个mention {sum(len(c) for c in all_candidates.values())/len(mentions):.1f} 个候选")
            except Exception as e:
                print(f"[Entity Linking] 批量获取候选实体失败: {e}")
                print("[Entity Linking] 降级为单个处理...")
                all_candidates = {}
                for mention in mentions:
                    try:
                        candidates = self._get_candidates(mention, top_k=top_k*2)
                        all_candidates[mention] = candidates
                    except Exception as e2:
                        print(f"[Entity Linking] 获取实体 '{mention}' 候选失败: {e2}")
                        all_candidates[mention] = []
            
            # 批量获取实体信息
            try:
                all_entity_names = set()
                for candidates in all_candidates.values():
                    all_entity_names.update(candidates)
                
                entity_details = self.batch_fetch_entities(list(all_entity_names))
                print(f"[Entity Linking] 成功获取 {len(entity_details)} 个实体详情")
            except Exception as e:
                print(f"[Entity Linking] 批量获取实体详情失败: {e}")
                entity_details = {}
            
            # 处理每个mention
            results = {}
            for mention in mentions:
                try:
                    candidates = all_candidates.get(mention, [])
                    if not candidates:
                        print(f"[Entity Linking] 未找到 '{mention}' 的候选实体")
                        results[mention] = []
                        continue
                    
                    # 处理候选实体
                    mention_results = []
                    for candidate in candidates:
                        try:
                            if candidate in entity_details:
                                entity_info = entity_details[candidate]
                                mention_results.append({
                                    "name": candidate,
                                    "score": self._calculate_similarity(mention, candidate),
                                    "desc": entity_info.get("desc", ""),
                                    "category": entity_info.get("category", "Unknown"),
                                    "reference": entity_info.get("reference", "")
                                })
                        except Exception as e:
                            print(f"[Entity Linking] 处理候选实体 '{candidate}' 失败: {e}")
                            continue
                    
                    # 排序并获取top-k
                    results[mention] = sorted(
                        mention_results,
                        key=lambda x: x["score"],
                        reverse=True
                    )[:top_k]
                    
                except Exception as e:
                    print(f"[Entity Linking] 处理实体 '{mention}' 失败: {e}")
                    results[mention] = []
            
            end_time = time.time()
            print(f"[Entity Linking] 批量处理完成，耗时: {end_time - start_time:.2f}秒")
            return results
            
        except Exception as e:
            print(f"[Entity Linking] 批量处理失败: {e}")
            print("[Entity Linking] 降级为单个处理...")
            
            # 降级为逐个处理
            results = {}
            for mention in mentions:
                try:
                    results[mention] = self._rank_single_entity(query, mention, top_k)
                except Exception as e2:
                    print(f"[Entity Linking] 处理实体 '{mention}' 失败: {e2}")
                    results[mention] = []
            
            end_time = time.time()
            print(f"[Entity Linking] 单个处理完成，耗时: {end_time - start_time:.2f}秒")
            return results

    def extract_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """获取实体的详细信息
        
        Args:
            entity_name: 实体名称
            
        Returns:
            Dict[str, Any]: 实体信息字典
        """
        print(f"[Entity Linking] 获取实体 '{entity_name}' 的详细信息")
        
        # 初始化默认返回值
        entity_info = {
            "name": entity_name,
            "desc": "",
            "category": "Unknown",
            "reference": "",
            "properties": {},
            "relations": []
        }
        
        try:
            # 首先检查缓存
            if entity_name in self.cache["entity_desc"]:
                entity_info["desc"] = self.cache["entity_desc"][entity_name]
                print(f"[Entity Linking] 从缓存获取到实体 '{entity_name}' 的描述")
            
            # 查询Neo4j获取完整信息
            with self.driver.session() as session:
                # 查询节点信息
                node_query = """
                MATCH (n {name: $entity_name})
                RETURN n,
                       labels(n) as labels,
                       properties(n) as properties
                """
                
                try:
                    node_result = session.run(node_query, entity_name=entity_name)
                    record = node_result.single()
                    
                    if record:
                        # 处理节点属性
                        props = record["properties"]
                        entity_info.update({
                            "name": props.get("name", entity_name),
                            "desc": props.get("desc", ""),
                            "reference": props.get("source_article", ""),
                            "properties": props
                        })
                        
                        # 处理节点标签
                        labels = record["labels"]
                        if labels:
                            entity_info["category"] = labels[0]
                        
                        print(f"[Entity Linking] 成功获取实体 '{entity_name}' 的基本信息")
                        
                        # 查询关系信息
                        relations_query = """
                        MATCH (n {name: $entity_name})-[r]-(m)
                        RETURN type(r) as relation_type,
                               m.name as connected_entity,
                               m.desc as connected_desc,
                               labels(m)[0] as connected_category,
                               startNode(r) = n as is_outgoing
                        """
                        
                        relations_result = session.run(relations_query, entity_name=entity_name)
                        
                        # 处理关系
                        for rel_record in relations_result:
                            relation = {
                                "type": rel_record["relation_type"],
                                "entity": rel_record["connected_entity"],
                                "desc": rel_record["connected_desc"] or "",
                                "category": rel_record["connected_category"] or "Unknown",
                                "direction": "outgoing" if rel_record["is_outgoing"] else "incoming"
                            }
                            entity_info["relations"].append(relation)
                        
                        print(f"[Entity Linking] 成功获取实体 '{entity_name}' 的 {len(entity_info['relations'])} 个关系")
                        
                    else:
                        print(f"[Entity Linking] 未找到实体 '{entity_name}'")
                
                except Exception as e:
                    print(f"[Entity Linking] 查询实体 '{entity_name}' 信息时出错: {e}")
            
            # 更新缓存
            self.cache["entity_desc"][entity_name] = entity_info["desc"]
            if entity_name not in self.cache["entity_names"]:
                self.cache["entity_names"].append(entity_name)
            
            return entity_info
            
        except Exception as e:
            print(f"[Entity Linking] 获取实体 '{entity_name}' 信息失败: {e}")
            return entity_info

    def get_related_entities(self, entity_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取与指定实体相关的其他实体"""
        with self.driver.session() as session:
            # 查询出边关系
            outgoing_query = """
            MATCH (n {name: $entity_name})-[r]->(m)
            RETURN n.name as source, 
                   type(r) as relation, 
                   m.name as entity,
                   m.desc as desc,
                   labels(m)[0] as category
            LIMIT $limit
            """
            
            # 查询入边关系
            incoming_query = """
            MATCH (m)-[r]->(n {name: $entity_name})
            RETURN m.name as entity, 
                   type(r) as relation, 
                   n.name as target,
                   m.desc as desc,
                   labels(m)[0] as category
            LIMIT $limit
            """
            
            related = []
            
            # 处理出边关系
            outgoing_result = session.run(outgoing_query, entity_name=entity_name, limit=limit)
            for record in outgoing_result:
                related.append({
                    "entity": record["entity"],
                    "relation": record["relation"],
                    "direction": "outgoing",
                    "desc": record["desc"] or "",
                    "category": record["category"] or "Unknown"
                })
            
            # 处理入边关系
            if len(related) < limit:
                remaining = limit - len(related)
                incoming_result = session.run(incoming_query, entity_name=entity_name, limit=remaining)
                for record in incoming_result:
                    related.append({
                        "entity": record["entity"],
                        "relation": record["relation"],
                        "direction": "incoming",
                        "desc": record["desc"] or "",
                        "category": record["category"] or "Unknown"
                    })
            
            return related
    
    def _calculate_similarity(self, mention: str, entity: str) -> float:
        """计算实体提及与知识库实体的相似度"""
        # 基于编辑距离的简单相似度计算
        from rapidfuzz import fuzz
        score = fuzz.ratio(mention.lower(), entity.lower()) / 100.0
        
        # 考虑部分匹配情况
        if mention.lower() in entity.lower() or entity.lower() in mention.lower():
            score = max(score, 0.7)  # 提升部分匹配得分
            
        return score

    def link_entities(self, mentions: List[str]) -> Dict[str, List[Dict]]:
        """链接实体到知识库"""
        print(f"\n[Entity Linking] 开始实体链接: {mentions}")
        
        # 直接使用批量处理方法
        return self.rank_entities_batch(query="", mentions=mentions, top_k=5)