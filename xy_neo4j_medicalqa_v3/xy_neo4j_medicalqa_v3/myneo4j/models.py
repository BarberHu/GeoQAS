from django.db import models
from accounts.models import UserProfile
# Create your models here.


class MyNode(models.Model):
    name = models.CharField(verbose_name='node的name', blank=True, null=True, default='', max_length=100)
    leixing = models.CharField(verbose_name='类型的中文', blank=True, null=True, default='', max_length=100)

    def __str__(self):
        return str(self.name)

    class Meta:
        ordering = ['-id']
        verbose_name = '节点信息'
        verbose_name_plural = verbose_name


class MyWenda(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    question = models.TextField(verbose_name='问题')
    sub_questions = models.TextField(verbose_name='子问题', null=True, blank=True)
    answer = models.TextField(verbose_name='答案')
    kg_context = models.TextField(verbose_name='知识图谱上下文', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.question)

    class Meta:
        ordering = ['-created_at']
        verbose_name = '问答信息'
        verbose_name_plural = verbose_name