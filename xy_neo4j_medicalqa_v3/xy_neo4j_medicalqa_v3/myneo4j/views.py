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
from .models import MyNode, MyWenda, ChatHistory
from django.conf import settings
# Create your views here.
from django.conf import settings
from xy_neo4j.dialogue_manager import DialogueManager
import uuid
from django.http import JsonResponse
from concurrent.futures import ThreadPoolExecutor
from config import LLM_CONFIG

# 创建一个全局的对话管理器实例
dialogue_manager = DialogueManager()

@login_required
def index(request):
    try:
        start = request.GET.get("start", "")
        relation = request.GET.get("relation", "")
        end = request.GET.get("end", "")
        all_datas = get_all_relation(start, relation, end)
        
        # 确保节点数据包含 source_article 属性（只有地理问题类型才有）
        for node in all_datas.get("datas", []):
            # 检查节点是否有标签信息并且是地理问题类型
            categories = node.get("categories", [])
            is_geo_problem = any(cat == "地理问题" for cat in categories)
            
            # 只有地理问题类型才需要source_article
            if not is_geo_problem and "source_article" in node:
                del node["source_article"]  # 删除非地理问题节点的source_article
            elif is_geo_problem and "source_article" not in node:
                node["source_article"] = ""  # 为地理问题类型节点添加默认值
        
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

# 创建线程池
executor = ThreadPoolExecutor(max_workers=5)

@login_required
def wenda(request):
    # 获取或创建会话ID
    session_id = request.session.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['session_id'] = session_id
    
    # 从请求中获取问题
    key = request.GET.get('key', '')
    
    # 获取历史记录
    history = ChatHistory.objects.filter(session_id=session_id).order_by('-timestamp')[:10]
    
    # 获取可用的API提供商
    available_providers = dialogue_manager.get_available_providers()
    
    context = {
        'history': history,
        'providers': available_providers['providers'],
        'current_provider': available_providers['current']
    }
    
    # 如果是AJAX请求，启动异步处理
    if key and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # 返回立即响应，通知前端请求已接收
        return JsonResponse({'status': 'processing'})
    
    elif key:
        # 使用全局对话管理器处理问题
        response = dialogue_manager.get_response(key)
        
        # 创建新的历史记录
        chat_history = ChatHistory(
            session_id=session_id,
            question=key,
            answer=response['answer'],
            has_flowchart='流域' in key or 'SWAT' in key or '模型' in key
        )
        
        # 保存结构化数据
        if 'kg_nodes' in response:
            chat_history.save_kg_nodes(response['kg_nodes'])
            
        if 'thinking_process' in response:
            chat_history.save_thinking_process(response['thinking_process'])
            
        if 'time_analysis' in response:
            chat_history.time_analysis = json.dumps(response['time_analysis'], ensure_ascii=False)
            
        if 'references' in response:
            chat_history.references = json.dumps(response['references'], ensure_ascii=False)
        
        # 保存记录
        chat_history.save()
        
        # 更新上下文
        context.update({
            'key': key,
            'answer': response['answer'],
            'kg_nodes': response.get('kg_nodes'),
            'thinking_process': response.get('thinking_process'),
            'time_analysis': response.get('time_analysis'),
            'references': response.get('references'),
        })
    
    return render(request, 'wenda.html', context)

def clear_history(request):
    """清除历史记录"""
    if request.method == 'POST':
        session_id = request.session.get('session_id')
        if session_id:
            try:
                ChatHistory.objects.filter(session_id=session_id).delete()
                return JsonResponse({'success': True})
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request'})

def export_history(request):
    """导出历史记录"""
    session_id = request.session.get('session_id')
    if not session_id:
        return JsonResponse({'error': 'No session found'}, status=400)
        
    history = ChatHistory.objects.filter(session_id=session_id).order_by('timestamp')
    
    # 构建导出数据
    export_data = []
    for chat in history:
        chat_data = {
            'timestamp': chat.timestamp.isoformat(),
            'question': chat.question,
            'answer': chat.answer
        }
        
        # 添加结构化数据
        if chat.kg_nodes:
            try:
                chat_data['kg_nodes'] = json.loads(chat.kg_nodes)
            except:
                chat_data['kg_nodes'] = None
                
        if chat.thinking_process:
            try:
                chat_data['thinking_process'] = json.loads(chat.thinking_process)
            except:
                chat_data['thinking_process'] = None
        
        export_data.append(chat_data)
    
    # 返回JSON文件
    response = HttpResponse(json.dumps(export_data, ensure_ascii=False, indent=2), 
                           content_type='application/json')
    response['Content-Disposition'] = 'attachment; filename=chat_history.json'
    return response

@csrf_exempt
def switch_api(request):
    """
    切换API提供商
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            provider = data.get('provider')
            
            if not provider:
                return JsonResponse({'success': False, 'message': '未提供API提供商名称'})
            
            # 使用全局对话管理器切换API
            result = dialogue_manager.switch_provider(provider)
            
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'切换API失败: {str(e)}'})
    else:
        return JsonResponse({'success': False, 'message': '仅支持POST请求'})

@csrf_exempt
def get_api_providers(request):
    """
    获取可用的API提供商列表
    """
    try:
        # 使用全局对话管理器获取可用的API提供商
        providers = dialogue_manager.get_available_providers()
        
        return JsonResponse(providers)
    except Exception as e:
        return JsonResponse({'success': False, 'message': f'获取API提供商列表失败: {str(e)}'})