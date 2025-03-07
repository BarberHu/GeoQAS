from django.conf import settings
from django.db import models

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