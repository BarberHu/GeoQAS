from django.shortcuts import render, HttpResponse
import os
import time
import json
import csv
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from .pyneo_utils import *
from django.views.decorators.csrf import csrf_exempt
import jieba
from .models import MyNode, MyWenda
from django.conf import settings
# Create your views here.
from django.conf import settings
from xy_neo4j.dialogue_manager import DialogueManager


@login_required
def index(request):
    try:
        start = request.GET.get("start", "")
        relation = request.GET.get("relation", "")
        end = request.GET.get("end", "")
        all_datas = get_all_relation(start, relation, end)
        
        # 确保节点数据包含 source_article 属性
        for node in all_datas.get("datas", []):
            if "source_article" not in node:
                node["source_article"] = ""  # 设置默认值
        
        links = json.dumps(all_datas.get("links", []))
        datas = json.dumps(all_datas.get("datas", []))
        categories = json.dumps(all_datas.get("categories", []))
        legend_data = json.dumps(all_datas.get("legend_data", []))

        print("Data to render:", {
            "links": links,
            "datas": datas,
            "categories": categories,
            "legend_data": legend_data
        })
    except Exception as e:
        print("Error in index view:", e)
        links, datas, categories, legend_data = "[]", "[]", "[]", "[]"
    
    return render(request, "index.html", locals())

# 在 views.py 中

@login_required
def wenda(request):
    try:
        print("\n[Views] 接收到问答请求")
        user = request.user
        dialogue_manager = DialogueManager()
        
        if request.method == "GET":
            key = request.GET.get("key", "")
            print(f"[Views] 用户问题: {key}")
            
            # 恢复清除历史记录功能
            if request.GET.get("clean") == "1":
                MyWenda.objects.filter(user=user).delete()
                return render(request, "wenda.html", {"all_wendas": []})
                
            # 获取历史记录
            all_wendas = MyWenda.objects.filter(user=user).order_by("-created_at")[:10]
            
            if not key:
                return render(request, "wenda.html", {"all_wendas": all_wendas})
                
            try:
                print("[Views] 开始调用 DialogueManager 处理问题...")
                response_data = dialogue_manager.get_response(key)
                print("[Views] DialogueManager 处理完成")
                
                # 解包结果
                print("[Views] 开始解析返回结果...")
                answer = response_data.get("answer", "未能获取回答")
                sub_questions = response_data.get("sub_questions")
                sub_answers = response_data.get("sub_answers", {})
                kg_context = response_data.get("kg_context")
                kg_nodes = response_data.get("kg_nodes", {})
                time_analysis = response_data.get("time_analysis", {})
                thinking_process = response_data.get("thinking_process", [])
                
                # 确保 kg_nodes 是有效的 JSON 格式
                if not isinstance(kg_nodes, dict):
                    print("[Views] kg_nodes 不是字典类型，初始化为空字典")
                    kg_nodes = {}

                # 确保必要的键存在
                if 'nodes' not in kg_nodes:
                    kg_nodes['nodes'] = []
                if 'links' not in kg_nodes:
                    kg_nodes['links'] = []
                if 'categories' not in kg_nodes:
                    kg_nodes['categories'] = []
                    
                print(f"[Views] 知识图谱节点数据大小: {len(str(kg_nodes))} 字节")
                
                # 更新思考过程中的子问题答案
                for step in thinking_process:
                    if step.get("title") == "问题分解" and sub_answers:
                        step["answers"] = sub_answers
                
                # 保存对话历史
                print("[Views] 保存对话历史...")
                wenda = MyWenda.objects.create(
                    user=user,
                    question=key,
                    answer=answer,
                    sub_questions=json.dumps(sub_questions, ensure_ascii=False) if sub_questions else "",
                    sub_answers=json.dumps(sub_answers, ensure_ascii=False) if sub_answers else "",
                    kg_context=json.dumps(kg_context, ensure_ascii=False) if isinstance(kg_context, (list, dict)) else str(kg_context),
                    kg_nodes=json.dumps(kg_nodes, ensure_ascii=False) if kg_nodes else "",
                    thinking_process=json.dumps(thinking_process, ensure_ascii=False) if thinking_process else "",
                    time_analysis=json.dumps(time_analysis, ensure_ascii=False) if time_analysis else ""
                )
                
                # 重新获取最新的历史记录
                all_wendas = MyWenda.objects.filter(user=user).order_by("-created_at")[:10]
                
                print("[Views] 准备渲染模板...")
                context = {
                    "all_wendas": all_wendas,
                    "key": key,
                    "answer": answer,
                    "sub_questions": sub_questions,
                    "sub_answers": sub_answers,
                    "kg_context": kg_context,
                    "kg_nodes": kg_nodes,
                    "thinking_process": thinking_process,
                    "time_analysis": time_analysis
                }
                
                print("[Views] 返回响应")
                return render(request, "wenda.html", context)
                
            except ConnectionError:
                return render(request, "wenda.html", {
                    "error": "连接中断，请重试",
                    "all_wendas": all_wendas
                })
                
    except Exception as e:
        print(f"[Views] 视图函数异常: {e}")
        return render(request, "wenda.html", {"error": str(e)})