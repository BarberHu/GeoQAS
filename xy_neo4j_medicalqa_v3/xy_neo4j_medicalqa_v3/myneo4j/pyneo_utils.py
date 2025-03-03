from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher

color = {
    "地理问题": "#5470c6",    # 蓝色
    "地理场景": "#91cc75",    # 绿色
    "对象系统": "#fac858",    # 黄色
    "系统机理": "#ee6666",    # 红色
    "时空数据": "#73c0de",    # 浅蓝
    "数据来源": "#3ba272",    # 深绿
    "处理方法": "#fc8452",    # 橙色
    "集成模型": "#9a60b4",    # 紫色
    "基础模型": "#ea7ccc",    # 粉色
    "开发步骤": "#5470c6",    # 蓝色
    "评价方法": "#91cc75",    # 绿色
    "评价结果": "#fac858",    # 黄色
    "模型应用": "#ee6666",    # 红色
    "应用结果": "#73c0de",    # 浅蓝
    "总结讨论": "#3ba272",    # 深绿
    "other": "#666666"       # 灰色
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
