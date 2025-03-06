from neo4j import GraphDatabase
import csv
import os

# 连接到本地的Neo4j数据库，确保Neo4j正在运行
uri = "bolt://localhost:7687"  # 这里是默认的连接URI，调整为你的Neo4j实例URI
username = "neo4j"  # 默认用户名
password = "wswy0129"  # 你的Neo4j密码

# 创建一个Neo4j驱动器实例
driver = GraphDatabase.driver(uri, auth=(username, password))

def get_all_node_names(driver):
    # 创建一个会话来执行Cypher查询
    with driver.session() as session:
        query = "MATCH (n) RETURN n.name AS name"  # 假设节点有一个叫`name`的属性
        result = session.run(query)

        # 返回查询结果
        node_names = [record["name"] for record in result]
        return node_names

# 获取所有节点的名称
node_names = get_all_node_names(driver)

# 指定保存路径
output_path = r"E:\毕业设计\参考\代码\Pycharm用\KG-LM Synergy项目实战\KG-LM Synergy项目实战\项目代码 - cursor\xy_neo4j_medicalqa_v3\xy_neo4j_medicalqa_v3\static\DATA_USE\zero_shot.csv"

try:
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 写入CSV文件
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        # 写入标题行
        writer.writerow(['Entity Name'])
        # 写入数据
        for name in node_names:
            writer.writerow([name])
    print(f"成功保存{len(node_names)}个实体到：{output_path}")

except Exception as e:
    print(f"保存文件时出错：{str(e)}")
    print("请检查：")
    print(f"1. 路径是否存在：{os.path.dirname(output_path)}")
    print("2. 是否有写入权限")
    print("3. 文件是否被其他程序占用")

# 打印所有节点名称（可选）
print("\n所有节点名称:")
for name in node_names:
    print(name)
