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
    "地理概念": "#c5b0d5",    # 浅紫色
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
        last = str(key) + ":" + str(mydict[key]) + "<br>" + last
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
        "下一步", "模型评价", "模拟过程", "评价结果", "运算结果",
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
    except Exception as e:
        print("查询执行失败:", str(e))
        return {"datas": [], "links": [], "legend_data": [], "categories": []}

    for nodes_relations in nodes_data_all:
        print("----")
        # 正确提取节点的标签作为类别
        # 去除冒号并确保只获取第一个标签作为类别名称
        try:
            # 将标签字符串转换为更清晰的格式
            raw_start_label = str(nodes_relations['n'].labels)
            raw_end_label = str(nodes_relations['b'].labels)
            
            # 从类似 "frozenset(['地理问题'])" 的格式中提取实际标签名
            start_label_match = re.search(r"'([^']+)'", raw_start_label)
            end_label_match = re.search(r"'([^']+)'", raw_end_label)
            
            start_lable = start_label_match.group(1) if start_label_match else "未知类型"
            end_lable = end_label_match.group(1) if end_label_match else "未知类型"
            
            print(f"节点标签解析: 起始节点={start_lable}, 目标节点={end_lable}")
        except Exception as e:
            print(f"解析标签出错: {e}")
            start_lable = str(nodes_relations['n'].labels).replace(":", "")
            end_lable = str(nodes_relations['b'].labels).replace(":", "")
        
        start = dict(nodes_relations['n'])
        end = dict(nodes_relations['b'])
        relation = "relation"
        if "name" not in start or "name" not in end:
            continue
        start_name = start["name"]
        end_name = end["name"]
        try:
            relation = str(nodes_relations['r'].keys).split(" ")[4]
        except Exception as e:
            print(e)
            continue
        if start_name not in cache:
            if start_lable in color:
                datas.append(
                    {"name": start_name,
                     "attr": start,
                     "color": color[start_lable],
                     "des": get_str_by_dict(start),
                     "category": start_lable})
            else:
                datas.append({"name": start_name,
                              "attr": start,
                              "color": color["other"],
                              "des": get_str_by_dict(start),
                              "category": start_lable})
            cache.append(start_name)
        if end_name not in cache:
            if end_lable in color:
                datas.append({"name": end_name,
                              "attr": end,
                              "color": color[end_lable],
                              "des": get_str_by_dict(end),
                              "category": end_lable})
            else:
                datas.append({"name": end_name,
                              "attr": end,
                              "color": color["other"],
                              "des": get_str_by_dict(end),
                              "category": end_lable})
            cache.append(end_name)

        if start_lable not in legend_data:
            legend_data.append(start_lable)
            categories.append({"name": start_lable})
        if end_lable not in legend_data:
            legend_data.append(end_lable)
            categories.append({"name": end_lable})

        cache_relation = start_name + "-" + end_name
        if cache_relation not in cache:
            links.append(
                {
                    "source": start_name,
                    "target": end_name,
                    "name": relation
                }
            )
            cache.append(cache_relation)

        # 添加参考文献
        if "source_article" in start:
            datas[-1]["source_article"] = start["source_article"]
        if "source_article" in end:
            datas[-1]["target_article"] = end["source_article"]

    print("=====")
    print(datas)
    print(links)

    return {"datas": datas, "links": links, "legend_data": legend_data, "categories": categories}
