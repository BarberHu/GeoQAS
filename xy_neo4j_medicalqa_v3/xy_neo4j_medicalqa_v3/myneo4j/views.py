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
        
        # 确保数据不为空
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

@login_required
def wenda(request):
    try:
        user = request.user
        dialogue_manager = DialogueManager()
        
        if request.method == "GET":
            key = request.GET.get("key", "")
            
            # 恢复清除历史记录功能
            if request.GET.get("clean") == "1":
                MyWenda.objects.filter(user=user).delete()
                return render(request, "wenda.html", {"all_wendas": []})
            
            # 获取历史记录
            all_wendas = MyWenda.objects.filter(user=user).order_by("-created_at")[:10]
            
            if not key:
                return render(request, "wenda.html", {"all_wendas": all_wendas})
                
            try:
                # 处理问答逻辑
                sub_questions = dialogue_manager.decompose_question(key)
                kg_context = dialogue_manager.get_kg_context(key)
                answer = dialogue_manager.generate_response(key, kg_context)
                
                # 保存对话历史
                wenda = MyWenda.objects.create(
                    user=user,
                    question=key,
                    sub_questions=sub_questions,
                    answer=answer,
                    kg_context=kg_context
                )
                
                # 重新获取最新的历史记录
                all_wendas = MyWenda.objects.filter(user=user).order_by("-created_at")[:10]
                return render(request, "wenda.html", locals())
                
            except ConnectionError:
                return render(request, "wenda.html", {
                    "error": "连接中断，请重试",
                    "all_wendas": all_wendas  # 使用已获取的历史记录
                })
                
    except Exception as e:
        print("视图函数异常:", e)
        return render(request, "wenda.html", {"error": str(e)})

