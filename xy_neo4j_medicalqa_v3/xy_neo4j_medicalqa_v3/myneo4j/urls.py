from django.urls import path, re_path  # 保留 re_path 用于正则表达式匹配
from .views import *

urlpatterns = [
    re_path(r"^$", index, name="index"),  # 使用 re_path 替代 url
    re_path(r"^index$", index, name="index"),  # 使用 re_path 替代 url

    re_path(r"^wenda$", wenda, name="wenda"),  # 使用 re_path 替代 url
    path('clear-history', clear_history, name='clear_history'),
    path('export-history', export_history, name='export_history'),
]
