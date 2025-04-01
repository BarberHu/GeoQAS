from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher
import re

color = {
    "地理问题": "#1f77b4",    # 深蓝色
    "地理场景": "#2ca02c",    # 深绿色
    "对象系统": "#ff7f0e",    # 橙色
    "系统机理": "#d62728",    # 红色
    "时空数据": "#17becf",    # 青色
    "数据来源": "#8c564b",    # 棕色
    "处理方法": "#e377c2",    # 粉色
    "集成模型": "#9467bd",    # 紫色
    "基础模型": "#bcbd22",    # 黄绿色
    "开发步骤": "#7f7f7f",    # 灰色
    "评价方法": "#ff9896",    # 浅红色
    "评价结果": "#98df8a",    # 浅绿色
    "模型应用": "#ffbb78",    # 浅橙色
    "应用结果": "#aec7e8",    # 浅蓝色
    "总结讨论": "#c49c94",    # 浅棕色
    "other": "#c7c7c7"       # 中灰色
}


def get_node_by_name(g, node_type, name):
    # g=Graph('http://localhost:7474',user='neo4j',password='123456')
    matcher = NodeMatcher(g)
    endnode = matcher.match(node_type, name=name).first()
    print(endnode)
    if endnode != None:
        return endnode
    else:
        return None


def get_str_by_dict(mydict):
    last = ""
    for key in mydict:
        # 处理值为列表的情况
        value = mydict[key]
        if isinstance(value, list):
            # 如果是列表，将其转换为逗号分隔的字符串
            value = ", ".join(str(item) for item in value)
        last = str(key) + ":" + str(value) + "<br>" + last
    return last


def get_all_relation(start, relation, end):
    try:
        # 修改连接方式为 bolt 协议
        g = Graph('bolt://localhost:7687', user='neo4j', password='wswy0129')
        # 测试连接
        test_query = g.run("MATCH (n) RETURN count(n) as count").data()
        print("Neo4j连接成功，节点总数:", test_query[0]['count'])
    except Exception as e:
        print("Neo4j连接失败:", str(e))
        return {"datas": [], "links": [], "legend_data": [], "categories": []}

    datas = []
    links = []
    cache = []
    categories = []
    legend_data = []
    
    valid_relations = [
        "所属场景", "研究对象", "使用数据", "使用模型", "问题结论",
        "相关机理", "获取来源", "数据处理", "集成依赖", "开发流程",
        "下一步", "模型评价", "模拟过程", "评估结果", "运算结果",
        "结果讨论"
    ]

    # 初始化 param
    param = ""
    
    if start != "":
        param = "where n.name='" + start + "'"
    if relation == "":
        mr = "r"
    else:
        if relation in valid_relations:
            mr = "r:" + relation
        else:
            mr = "r"
    if end != "":
        if "where" in param:
            param = param + " and b.name='" + end + "'"
        else:
            param = "where b.name='" + end + "'"

    # 如果没有任何查询条件，返回所有关系
    if not start and not end:
        sql = "MATCH (n)-[r]-(b) RETURN n,r,b"
    else:
        sql = "MATCH (n)-[%s]-(b) %s RETURN n,r,b" % (mr, param)

    print("执行查询:", sql)
    
    try:
        nodes_data_all = g.run(sql).data()
        print("查询结果数量:", len(nodes_data_all))
        
        # 检查第一个结果的格式，用于调试
        if nodes_data_all and len(nodes_data_all) > 0:
            print("第一个结果类型:", type(nodes_data_all[0]))
            print("第一个结果内容:", nodes_data_all[0])
    except Exception as e:
        print("查询执行失败:", str(e))
        return {"datas": [], "links": [], "legend_data": [], "categories": []}

    for nodes_relations in nodes_data_all:
        #print("----")
        # 正确提取节点的标签作为类别
        try:
            # 将标签字符串转换为更清晰的格式
            raw_start_label = str(nodes_relations['n'].labels)
            raw_end_label = str(nodes_relations['b'].labels)
       # 直接从标签列表中提取第一个标签
      # 将集合转换为列表后获取第一个标签
            start_lable = list(nodes_relations['n'].labels)[0] if nodes_relations['n'].labels else "未知类型"
            end_lable = list(nodes_relations['b'].labels)[0] if nodes_relations['b'].labels else "未知类型"
            
           # print(f"节点标签解析: 起始节点={start_lable}, 目标节点={end_lable}")
        except Exception as e:
            print(f"解析标签出错: {e}")
            start_lable = "未知类型"
            end_lable = "未知类型"
        
        try:
            # 确保我们获取的是字典，而不是Node对象
            start = dict(nodes_relations['n'])
            end = dict(nodes_relations['b'])

            
            relation = "relation"
            if "name" not in start or "name" not in end:
                print("节点缺少name属性，跳过")
                continue
                
            # 确保start_name和end_name是字符串类型
            start_name = start.get("name", "")
            end_name = end.get("name", "")
            
            # 如果name是列表，将其转换为字符串
            if isinstance(start_name, list):
                start_name = str(start_name[0]) if start_name else "未命名"
               # print(f"警告: 节点名称是列表类型: {start_name}")
            
            if isinstance(end_name, list):
                end_name = str(end_name[0]) if end_name else "未命名"
              #  print(f"警告: 节点名称是列表类型: {end_name}")
                
            # 确保节点名称是字符串
            start_name = str(start_name)
            end_name = str(end_name)
            
            try:
                relation = str(nodes_relations['r'].keys).split(" ")[4]
            except Exception as e:
                print(f"获取关系类型失败: {e}")
                relation = "关联"  # 默认关系名称
            
            # 处理起始节点
            if start_name not in cache:
                node_data = {
                    "name": start_name,
                    "attr": {
                        # 确保只包含需要的属性，防止包含Neo4j内部属性
                        "name": start_name,
                        "desc": start.get("desc", "")
                    },
                    "color": color.get(start_lable, color["other"]),
                    "des": get_str_by_dict(start),
                    "category": start_lable,
                    "categories": [start_lable]  # 确保categories是列表
                }
                
                # 只为地理问题类型节点添加source_article
                if start_lable == "地理问题" and "source_article" in start:
                    node_data["source_article"] = start["source_article"]
                
                datas.append(node_data)
                cache.append(start_name)
            
            # 处理目标节点
            if end_name not in cache:
                node_data = {
                    "name": end_name,
                    "attr": {
                        # 确保只包含需要的属性，防止包含Neo4j内部属性
                        "name": end_name,
                        "desc": end.get("desc", "")
                    },
                    "color": color.get(end_lable, color["other"]),
                    "des": get_str_by_dict(end),
                    "category": end_lable,
                    "categories": [end_lable]  # 确保categories是列表
                }
                
                # 只为地理问题类型节点添加source_article
                if end_lable == "地理问题" and "source_article" in end:
                    node_data["source_article"] = end["source_article"]
                
                datas.append(node_data)
                cache.append(end_name)

            # 确保 legend_data 和 categories 是一致的
            if start_lable not in legend_data:
                legend_data.append(start_lable)
                categories.append({"name": start_lable})
            if end_lable not in legend_data:
                legend_data.append(end_lable)
                categories.append({"name": end_lable})

            # 添加关系 - 确保使用字符串类型
            cache_relation = f"{start_name}-{end_name}"  # 使用f-string来确保字符串连接
            if cache_relation not in cache:
                links.append({
                    "source": start_name,
                    "target": end_name,
                    "name": relation
                })
                cache.append(cache_relation)
        except Exception as e:
            print(f"处理节点关系时出错: {e}")
            continue

    print("数据处理完成")
    print(f"节点数量: {len(datas)}")
    print(f"关系数量: {len(links)}")
    print(f"分类数量: {len(categories)}")

    return {
        "datas": datas, 
        "links": links, 
        "legend_data": legend_data, 
        "categories": categories
    }
