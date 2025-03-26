from neo4j import GraphDatabase
import json
import os
import logging
from typing import Dict, Any, List, Set, Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j连接配置
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "wswy0129")

# 实体类型与关系的映射
RELATION_MAP = {
    "所属场景": "BELONGS_TO",
    "研究对象": "STUDIES",
    "使用数据": "USES_DATA",
    "使用模型": "USES_MODEL",
    "问题结论": "HAS_CONCLUSION",
    "相关机理": "RELATED_MECHANISM",
    "获取来源": "FROM_SOURCE",
    "数据处理": "PROCESSED_BY",
    "集成依赖": "DEPENDS_ON",
    "开发流程": "DEVELOPMENT_STEP",
    "下一步": "NEXT_STEP",
    "模型评价": "EVALUATED_BY",
    "模拟过程": "SIMULATION_PROCESS",
    "评估结果": "EVALUATION_RESULT",
    "运算结果": "HAS_RESULT",
    "结果讨论": "DISCUSSED_IN"
}

# 构建关系映射字典，实现图谱结构中定义的所有关系和子关系
NESTED_RELATIONS = {
    "地理问题": ["所属场景", "研究对象", "使用数据", "使用模型", "问题结论"],
    "对象系统": ["相关机理"],
    "时空数据": ["获取来源", "数据处理"],
    "集成模型": ["集成依赖", "开发流程", "模型评价", "模拟过程"],
    "开发步骤": ["下一步"],
    "评价方法": ["评估结果"],
    "模型应用": ["运算结果"],
    "应用结果": ["结果讨论"]
}

class Neo4jImporter:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)
        self.node_cache = {}  # 缓存已创建的节点 {(type, name): node_id}
        self.relation_cache = set()  # 缓存已创建的关系，避免重复创建
        self.all_nodes = set()  # 所有创建的节点

    def close(self):
        """关闭Neo4j连接"""
        self.driver.close()

    def create_constraints(self):
        """创建唯一性约束"""
        with self.driver.session() as session:
            # 所有实体类型
            entity_types = [
                "地理问题", "地理场景", "对象系统", "系统机理", 
                "时空数据", "数据来源", "处理方法", 
                "集成模型", "基础模型", "开发步骤", 
                "评价方法", "评价结果", "模型应用", 
                "应用结果", "总结讨论"
            ]
            for label in entity_types:
                try:
                    query = (
                        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) "
                        f"REQUIRE n.name IS UNIQUE"
                    )
                    session.run(query)
                    logger.info(f"为 {label} 创建唯一性约束")
                except Exception as e:
                    logger.error(f"为 {label} 创建约束失败: {str(e)}")

    def import_data(self, file_path: str):
        """导入单个 JSON 文件"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            with self.driver.session() as session:
                # 清除处理状态
                self.relation_cache.clear()
                
                # 处理地理问题及其关系
                session.execute_write(self._process_node, data)
                logger.info(f"成功导入 {file_path}")
                
                # 记录处理的实体和关系数量
                logger.info(f"共创建/更新 {len(self.all_nodes)} 个实体，{len(self.relation_cache)} 个关系")
                
                return True
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误 ({file_path}): {str(e)}")
            return False
        except Exception as e:
            logger.error(f"导入文件 {file_path} 时出错: {str(e)}")
            return False

    def _process_node(self, tx, node_data: Dict[str, Any], parent_node=None, relation_type=None):
        """处理节点及其关系，通用递归函数"""
        if not isinstance(node_data, dict) or "type" not in node_data or "name" not in node_data:
            logger.warning(f"跳过无效数据: {node_data}")
            return None

        # 创建当前节点
        current_node = self._create_node(tx, node_data)
        self.all_nodes.add((node_data["type"], node_data["name"]))
        
        # 如果有父节点，创建关系
        if parent_node and relation_type:
            self._create_relationship(tx, parent_node, current_node, relation_type)
        
        # 处理该节点的所有可能关系
        node_type = node_data["type"]
        potential_relations = NESTED_RELATIONS.get(node_type, [])
        
        for rel in potential_relations:
            if rel in node_data and isinstance(node_data[rel], list):
                # 处理列表类型的关系
                for child_data in node_data[rel]:
                    self._process_node(tx, child_data, current_node, rel)
            elif rel in node_data and isinstance(node_data[rel], str) and rel == "下一步":
                # 特殊处理"下一步"关系，它指向的是name值而非完整对象
                next_step_name = node_data[rel]
                if next_step_name:
                    # 查找或创建下一步节点
                    next_step_node = self._get_or_create_reference_node(tx, "开发步骤", next_step_name)
                    if next_step_node:
                        self._create_relationship(tx, current_node, next_step_node, rel)
            
            # 特殊处理，兼容JSON中使用"评价结果"作为关系键的情况
            elif node_type == "评价方法" and "评价结果" in node_data and "评估结果" not in node_data:
                # 将"评价结果"键下的内容处理为"评估结果"关系
                if isinstance(node_data["评价结果"], list):
                    for child_data in node_data["评价结果"]:
                        self._process_node(tx, child_data, current_node, "评估结果")
                        
        return current_node

    def _get_or_create_reference_node(self, tx, node_type: str, node_name: str):
        """获取或创建引用节点（处理下一步等引用关系）"""
        cache_key = (node_type, node_name)
        
        if cache_key in self.node_cache:
            return self.node_cache[cache_key]
            
        # 创建引用节点
        query = (
            f"MERGE (n:`{node_type}` {{name: $name}}) "
            "RETURN elementId(n) as id, n.name as name"
        )
        result = tx.run(query, parameters={"name": node_name})
        if result.peek():
            record = result.single()
            node_info = {"id": record["id"], "type": node_type, "name": node_name}
            self.node_cache[cache_key] = node_info
            return node_info
        return None

    def _create_node(self, tx, item: Dict[str, Any]) -> Dict[str, str]:
        """创建节点"""
        if not isinstance(item, dict) or "type" not in item or "name" not in item:
            logger.warning(f"跳过无效实体数据: {item}")
            return None
            
        node_type = item["type"]
        node_name = item["name"]
        cache_key = (node_type, node_name)

        if cache_key in self.node_cache:
            return self.node_cache[cache_key]

        # 准备节点属性
        node_properties = {
            "name": node_name,
            "desc": item.get("desc", "")
        }
        
        # 如果是地理问题节点，直接添加source_article属性
        if node_type == "地理问题" and "source_article" in item:
            node_properties["source_article"] = item.get("source_article", "")
        
        # 使用MERGE而非CREATE以支持跨文件融合
        property_keys = ", ".join([f"n.{key} = ${key}" for key in node_properties.keys()])
        query = (
            f"MERGE (n:`{node_type}` {{name: $name}}) "
            f"ON CREATE SET {property_keys} "
            "ON MATCH SET "
            "n.desc = CASE WHEN size(n.desc) < size($desc) THEN $desc ELSE n.desc END "  # 选择更详细的描述
        )
        
        # 为地理问题节点添加source_article属性的更新逻辑
        if node_type == "地理问题" and "source_article" in node_properties:
            query += ", n.source_article = CASE WHEN n.source_article IS NULL OR size(n.source_article) < size($source_article) THEN $source_article ELSE n.source_article END"
        
        query += " RETURN elementId(n) as id, n.name as name"
        
        try:
            result = tx.run(query, parameters=node_properties)
            record = result.single()

            node_info = {"id": record["id"], "type": node_type, "name": node_name}
            self.node_cache[cache_key] = node_info
            return node_info
        except Exception as e:
            logger.error(f"创建节点 {node_type}:{node_name} 失败: {str(e)}")
            return None

    def _create_relationship(self, tx, start: dict, end: dict, rel_type: str):
        """创建关系（避免笛卡尔积）"""
        if not start or not end:
            return False
            
        # 使用关系缓存避免重复创建
        rel_key = (start["id"], RELATION_MAP.get(rel_type, rel_type), end["id"])
        if rel_key in self.relation_cache:
            return True
            
        try:
            cypher_rel = RELATION_MAP.get(rel_type, rel_type)
            # 改进的查询，避免笛卡尔积
            query = (
                "MATCH (a) WHERE elementId(a) = $start_id "
                "MATCH (b) WHERE elementId(b) = $end_id "
                f"MERGE (a)-[r:`{cypher_rel}`]->(b) "
                "RETURN type(r)"
            )
            result = tx.run(query, parameters={
                "start_id": start["id"],
                "end_id": end["id"]
            })
            
            if result.single():
                self.relation_cache.add(rel_key)
                return True
            return False
        except Exception as e:
            logger.error(f"创建关系 {start['type']}:{start['name']} -{rel_type}-> {end['type']}:{end['name']} 失败: {str(e)}")
            return False

    def generate_stats(self):
        """生成导入统计信息"""
        with self.driver.session() as session:
            # 统计节点数量
            node_counts = session.run(
                "MATCH (n) RETURN distinct labels(n) as type, count(*) as count"
            ).data()
            
            # 统计关系数量
            rel_counts = session.run(
                "MATCH ()-[r]->() RETURN type(r) as type, count(*) as count"
            ).data()
            
            return {
                "nodes": {item["type"][0]: item["count"] for item in node_counts},
                "relationships": {item["type"]: item["count"] for item in rel_counts}
            }

    def run_enhanced_knowledge_fusion(self):
        """执行增强的知识融合操作"""
        with self.driver.session() as session:
            logger.info("开始执行增强知识融合...")
            
            try:
                # 1. 基于名称相似度的实体合并 - 分实体类型处理
                entity_types = ["地理问题", "地理场景", "对象系统", "集成模型", "基础模型"]
                for entity_type in entity_types:
                    logger.info(f"处理 {entity_type} 实体的名称相似度融合...")
                    try:
                        # 添加异常处理和类型检查
                        session.run(f"""
                            MATCH (a:`{entity_type}`), (b:`{entity_type}`)
                            WHERE a <> b AND id(a) < id(b)
                            AND apoc.text.levenshteinSimilarity(a.name, b.name) > 0.75
                            WITH a, b ORDER BY apoc.text.levenshteinSimilarity(a.name, b.name) DESC
                            LIMIT 10
                            CALL apoc.refactor.mergeNodes([a,b], {{properties:"combine",mergeRels:true}})
                            YIELD node RETURN node
                        """)
                    except Exception as e:
                        logger.error(f"处理 {entity_type} 实体融合时出错: {str(e)}")
                
                # 2. 基于共享邻居的实体合并
                logger.info("执行基于共享邻居的实体融合...")
                for entity_type in entity_types:
                    for rel_type in RELATION_MAP.values():
                        try:
                            # 修复类型转换问题
                            session.run(f"""
                                MATCH (a:`{entity_type}`)-[r1:`{rel_type}`]->(common)<-[r2:`{rel_type}`]-(b:`{entity_type}`)
                                WHERE a <> b AND id(a) < id(b)
                                WITH a, b, count(common) as commonNeighbors
                                WHERE commonNeighbors >= 2
                                WITH a, b LIMIT 5
                                CALL apoc.refactor.mergeNodes([a,b], {{properties:"combine",mergeRels:true}})
                                YIELD node RETURN node
                            """)
                        except Exception as e:
                            logger.error(f"处理 {entity_type} 和关系 {rel_type} 的共享邻居融合时出错: {str(e)}")
                
                # 3. 处理引用链接（下一步关系）
                logger.info("处理引用链接关系...")
                try:
                    session.run("""
                        MATCH (a:开发步骤)-[r:NEXT_STEP]->(b:开发步骤)
                        OPTIONAL MATCH (c:开发步骤)
                        WHERE c.name = b.name AND c <> b
                        WITH a, b, c WHERE c IS NOT NULL
                        MERGE (a)-[:NEXT_STEP]->(c)
                        DELETE r
                    """)
                except Exception as e:
                    logger.error(f"处理引用链接关系时出错: {str(e)}")
                
                logger.info("增强知识融合完成！")
                
            except Exception as e:
                logger.error(f"增强知识融合操作失败: {str(e)}")
                logger.error(f"错误类型: {type(e).__name__}")
                import traceback
                logger.error(f"错误详情: {traceback.format_exc()}")


def main():
    importer = Neo4jImporter()
    try:
        logger.info("开始创建约束...")
        importer.create_constraints()
        
        data_dir = r"E:\graduate_assay\code_cursor\xy_neo4j_medicalqa_v3\KG\output0320"  # JSON文件存放目录
        
        # 统计处理结果
        total_files = 0
        success_count = 0
        
        # 遍历目录中的所有文件
        for filename in os.listdir(data_dir):
            # 检查文件是否为JSON文件（不区分大小写）
            if filename.lower().endswith('.json'):
                try:
                    # 使用os.path.join处理文件路径
                    filepath = os.path.join(data_dir, filename)
                    logger.info(f"正在处理: {filename}")
                    total_files += 1
                    
                    # 检查文件是否存在和可访问
                    if os.path.exists(filepath) and os.path.isfile(filepath):
                        if importer.import_data(filepath):
                            success_count += 1
                            logger.info(f"成功处理: {filename}")
                    else:
                        logger.error(f"文件不存在或无法访问: {filename}")
                        
                except Exception as e:
                    logger.error(f"处理文件 {filename} 时出错: {str(e)}")
                    continue  # 继续处理下一个文件
        
        # 执行知识融合
        logger.info("开始执行增强知识融合...")
        importer.run_enhanced_knowledge_fusion()
        
        # 生成统计信息
        stats = importer.generate_stats()
        logger.info(f"数据导入统计: 成功导入 {success_count}/{total_files} 个文件")
        logger.info(f"知识图谱统计: {stats}")
        
        logger.info("所有数据导入完成！")
    except Exception as e:
        logger.error(f"导入过程发生错误: {str(e)}")
    finally:
        importer.close()


if __name__ == "__main__":
    main()
