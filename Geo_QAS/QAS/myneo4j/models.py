from django.conf import settings
from django.db import models
import json

class MyWenda(models.Model):
    # 修改此行，使用 settings.AUTH_USER_MODEL 代替直接引用 User
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="用户")
    
    # 其他字段保持不变
    question = models.TextField(verbose_name="问题")
    answer = models.TextField(verbose_name="回答")
    sub_questions = models.TextField(verbose_name="子问题", blank=True, null=True)
    sub_answers = models.TextField(verbose_name="子问题回答", blank=True, null=True)
    kg_context = models.TextField(verbose_name="知识图谱上下文", blank=True, null=True)
    kg_nodes = models.TextField(verbose_name="知识图谱节点", blank=True, null=True)
    thinking_process = models.TextField(verbose_name="思考过程", blank=True, null=True)
    time_analysis = models.TextField(verbose_name="时间分析", blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    
    class Meta:
        verbose_name = "问答记录"
        verbose_name_plural = verbose_name
        ordering = ["-created_at"]
    
class MyNode(models.Model):
    name = models.CharField(max_length=255, verbose_name="节点名称")
    category = models.CharField(max_length=100, verbose_name="类别", blank=True, null=True)
    desc = models.TextField(verbose_name="描述", blank=True, null=True)
    
    class Meta:
        verbose_name = "知识图谱节点"
        verbose_name_plural = verbose_name
        
    def __str__(self):
        return self.name

class ChatHistory(models.Model):
    # 基本信息
    session_id = models.CharField(max_length=64, help_text="会话标识符")
    timestamp = models.DateTimeField(auto_now_add=True, help_text="创建时间")
    
    # 用户问题
    question = models.TextField(help_text="用户问题")
    
    # 回答内容
    answer = models.TextField(help_text="回答文本")
    
    # 结构化数据 - 使用JSON格式存储
    kg_nodes = models.TextField(blank=True, null=True, help_text="知识图谱节点数据(JSON)")
    thinking_process = models.TextField(blank=True, null=True, help_text="思考过程数据(JSON)")
    time_analysis = models.TextField(blank=True, null=True, help_text="时间分析数据(JSON)")
    references = models.TextField(blank=True, null=True, help_text="参考文献(JSON)")
    
    # 是否包含流程图
    has_flowchart = models.BooleanField(default=False, help_text="是否包含流程图")
    
    class Meta:
        ordering = ['-timestamp']
        
    def save_kg_nodes(self, kg_nodes):
        """保存知识图谱节点数据"""
        if isinstance(kg_nodes, dict):
            self.kg_nodes = json.dumps(kg_nodes, ensure_ascii=False)
        elif isinstance(kg_nodes, str):
            self.kg_nodes = kg_nodes
            
    def save_thinking_process(self, thinking_process):
        """保存思考过程数据"""
        if isinstance(thinking_process, list):
            self.thinking_process = json.dumps(thinking_process, ensure_ascii=False)
        elif isinstance(thinking_process, str):
            self.thinking_process = thinking_process
            
    def get_kg_nodes(self):
        """获取知识图谱节点数据"""
        if not self.kg_nodes:
            return None
        try:
            return json.loads(self.kg_nodes)
        except:
            return None
            
    def get_thinking_process(self):
        """获取思考过程数据"""
        if not self.thinking_process:
            return None
        try:
            return json.loads(self.thinking_process)
        except:
            return None