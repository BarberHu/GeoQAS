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
from .dialogue_manager import DialogueManager


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
            if not key:
                return render(request, "wenda.html", locals())

            try:
                # 1. 问题分解
                sub_questions = dialogue_manager.decompose_question(key)
                print("问题分解:", sub_questions)
                
                # 2. 获取知识图谱上下文
                kg_context = dialogue_manager.get_kg_context(key)
                print("知识图谱上下文:", kg_context)
                
                # 3. 生成回答
                answer = dialogue_manager.generate_response(key, kg_context)
                print("生成回答:", answer)
                
                # 4. 保存对话历史
                wenda = MyWenda()
                wenda.user = user
                wenda.question = key
                wenda.sub_questions = sub_questions
                wenda.answer = answer
                wenda.kg_context = kg_context
                wenda.save()
                
            except Exception as e:
                print("处理问题失败:", e)
                answer = "抱歉，处理您的问题时出现错误，请稍后再试。"
            
            all_wendas = MyWenda.objects.filter(user=user).order_by("-id")[:10]
            return render(request, "wenda.html", locals())
            
    except Exception as e:
        print("视图函数异常:", e)
        return render(request, "wenda.html", {"error": str(e)})

