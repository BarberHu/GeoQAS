from django.contrib import admin
from .models import MyWenda

# Register your models here.

# class MyNodeAdmin(admin.ModelAdmin):
#     list_display = ["name", "leixing"]
#     search_fields = ["name"]
#     list_filter = ["leixing"]
#
# admin.site.register(MyNode, MyNodeAdmin)

@admin.register(MyWenda)
class MyWendaAdmin(admin.ModelAdmin):
    list_display = ('user', 'question', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('question', 'answer')
    readonly_fields = ('created_at',)
